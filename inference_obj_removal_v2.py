import json
import cv2
import re
from tqdm import tqdm
import torch
import os
import shutil
import math

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/scratch3/yan204/yxp/Qwen2.5-VL-32B-Instruct"
BSZ = 4 # Reduce batch size for transformers version

# Load model with flash attention for better performance
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

def extract_scores_from_response(text):
    """Extract scores from model output"""
    try:
        # Try to find JSON format response
        import re
        json_pattern = r'\{[^}]*"score"[^}]*\}'
        matches = re.findall(json_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if matches:
            # Try to parse the JSON
            for match in matches:
                try:
                    # Clean up the JSON string
                    cleaned_match = match.replace('\n', '').replace('  ', ' ')
                    json_data = json.loads(cleaned_match)
                    
                    if 'score' in json_data:
                        score = json_data['score']
                        reasoning = json_data.get('reasoning', '')
                        
                        # Handle different score formats
                        if isinstance(score, list) and len(score) >= 2:
                            return score[0], score[1], reasoning
                        elif isinstance(score, (int, float)):
                            return score, None, reasoning
                except:
                    continue
        
        # Fallback: try to extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if len(numbers) >= 2:
            return float(numbers[0]), float(numbers[1]), text
        elif len(numbers) >= 1:
            return float(numbers[0]), None, text
            
        return None, None, text
        
    except Exception as e:
        print(f"Error extracting scores: {e}")
        return None, None, text

# Load preprocessed data
PREPROCESSED_DATA_PATH = "/scratch3/yan204/yxp/Senorita/preprocessed_obj_removal_data.json"
OUTPUT_PATH = "/scratch3/yan204/yxp/Senorita/vie_score_obj_removal_transformers.json"

def load_preprocessed_data():
    """Load preprocessed valid_samples, sc_messages and pq_messages"""
    try:
        with open(PREPROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
            preprocessed = json.load(f)
        valid_samples = preprocessed.get("valid_samples", [])
        sc_messages = preprocessed.get("sc_messages", [])
        pq_messages = preprocessed.get("pq_messages", [])
        frames_dir = preprocessed.get("frames_dir", "")
        return valid_samples, sc_messages, pq_messages, frames_dir
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return [], [], [], ""

def process_batch_with_transformers(messages, max_new_tokens=256):
    """Process a batch of messages using transformers"""
    if not messages:
        return []
    try:
        # Prepare texts
        texts = []
        all_image_inputs = []
        all_video_inputs = []
        
        for msg in messages:
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            
            # 先拿到结果
            result = process_vision_info([msg])
            # 如果返回 None 就手动置成空 tuple
            if result is None:
                image_inputs, video_inputs = [], []
            else:
                image_inputs, video_inputs = result
                # 保护一下，防止单边为 None
                image_inputs = image_inputs or []
                video_inputs = video_inputs or []
            
            all_image_inputs.extend(image_inputs)
            all_video_inputs.extend(video_inputs)
        
        # Process inputs
        inputs = processor(
            text=texts,
            images=all_image_inputs,
            videos=all_video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                temperature=0.1, 
                top_p=0.001
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        return output_texts
    
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return ['error'] * len(messages)

print("Loading preprocessed data...")
valid_samples, sc_messages, pq_messages, frames_dir = load_preprocessed_data()

if len(valid_samples) == 0 or len(sc_messages) == 0 or len(pq_messages) == 0:
    print("No preprocessed data found. Please run preprocess_obj_removal.py first.")
    exit(1)

print(f"Loaded {len(valid_samples)} samples with SC and PQ messages")
print(f"Frames directory: {frames_dir}")

# Process in batches
final_output = []
start_idx = 0

# Resume functionality
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
            final_output = existing.get("results", [])
            start_idx = len(final_output)
            print(f"Resuming from sample index {start_idx}")
    except Exception as e:
        print(f"Error reading existing output file: {e}")

print("Processing samples with VIE scoring using transformers...")

print(len(valid_samples), len(sc_messages), len(pq_messages))

min_len = min(len(valid_samples), len(sc_messages), len(pq_messages))
print(f"Will actually iterate up to index {min_len}")
for i in tqdm(range(start_idx, min_len, BSZ), desc="Processing batches"):
    batch_sc_messages = sc_messages[i:i + BSZ]
    batch_pq_messages = pq_messages[i:i + BSZ]
    batch_samples = valid_samples[i:i + BSZ]

    # Process SC scores
    print(f"Processing SC scores for batch {i//BSZ + 1}")
    sc_responses = process_batch_with_transformers(batch_sc_messages)
    
    # Process PQ scores
    print(f"Processing PQ scores for batch {i//BSZ + 1}")
    pq_responses = process_batch_with_transformers(batch_pq_messages)

    # Process results
    for j, (sample, sc_response, pq_response) in enumerate(zip(batch_samples, sc_responses, pq_responses)):
        # Extract SC scores
        sc_score1, sc_score2, sc_reasoning = extract_scores_from_response(sc_response)
        
        # Extract PQ score
        pq_score, _, pq_reasoning = extract_scores_from_response(pq_response)
        
        # Normalize scores to 0-1 and calculate overall score
        sc_normalized = (sc_score1 / 10.0) if sc_score1 is not None else 0.0
        pq_normalized = (pq_score / 10.0) if pq_score is not None else 0.0
        
        # Calculate overall score: O = sqrt(SC × PQ)
        overall_score = math.sqrt(sc_normalized * pq_normalized) if (sc_normalized > 0 and pq_normalized > 0) else 0.0
        
        # Add analysis results to sample
        sample["sc_response"] = sc_response
        sample["pq_response"] = pq_response
        sample["sc_score"] = sc_score1
        sample["sc_overediting_score"] = sc_score2
        sample["pq_score"] = pq_score
        sample["sc_normalized"] = sc_normalized
        sample["pq_normalized"] = pq_normalized
        sample["overall_score"] = overall_score
        sample["is_high_quality"] = (overall_score > 0.9)
        
        final_output.append(sample)
    
    # Save progress
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
        print(f"Saved progress: {len(final_output)} samples processed")
    except Exception as e:
        print(f"Error writing to output file: {e}")

# Final analysis and statistics
print(f"\nVIE评分测试完成!")
print(f"测试样本总数: {len(final_output)}")

if final_output:
    # 计算筛选率
    high_quality_samples = [s for s in final_output if s.get("overall_score", 0) > 0.9]
    filter_rate = len(high_quality_samples) / len(final_output) * 100
    
    print(f"高质量样本数 (overall_score > 0.9): {len(high_quality_samples)}")
    print(f"筛选率: {filter_rate:.2f}%")
    
    # 分数统计
    sc_scores = [s.get("sc_normalized", 0) for s in final_output if s.get("sc_normalized") is not None]
    pq_scores = [s.get("pq_normalized", 0) for s in final_output if s.get("pq_normalized") is not None]
    overall_scores = [s.get("overall_score", 0) for s in final_output if s.get("overall_score") is not None]
    
    if sc_scores:
        print(f"\n=== SC分数统计 ===")
        print(f"平均值: {sum(sc_scores)/len(sc_scores):.3f}")
        print(f"最小值: {min(sc_scores):.3f}")
        print(f"最大值: {max(sc_scores):.3f}")
    
    if pq_scores:
        print(f"\n=== PQ分数统计 ===")
        print(f"平均值: {sum(pq_scores)/len(pq_scores):.3f}")
        print(f"最小值: {min(pq_scores):.3f}")
        print(f"最大值: {max(pq_scores):.3f}")
    
    if overall_scores:
        print(f"\n=== Overall分数统计 ===")
        print(f"平均值: {sum(overall_scores)/len(overall_scores):.3f}")
        print(f"最小值: {min(overall_scores):.3f}")
        print(f"最大值: {max(overall_scores):.3f}")
    
    # 分数分布统计
    score_ranges = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), 
                    (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 0.9), 
                    (0.9, 0.95), (0.95, 1.0)]
    print(f"\n=== Overall分数分布 ===")
    for low, high in score_ranges:
        count = len([s for s in overall_scores if low <= s < high])
        percentage = count / len(overall_scores) * 100 if overall_scores else 0
        print(f"{low:.2f}-{high:.2f}: {count} 样本 ({percentage:.1f}%)")

# Save high-quality results
hq_output_path = "/scratch3/yan204/yxp/Senorita/hq_obj_removal_filtered_transformers.json"
if final_output:
    hq_samples = [sample for sample in final_output if sample.get("overall_score", 0) > 0.9]
    with open(hq_output_path, "w", encoding="utf-8") as f:
        json.dump({"results": hq_samples}, f, indent=2, ensure_ascii=False)
    print(f"\n高质量样本保存到: {hq_output_path}")

print(f"完整结果保存到: {OUTPUT_PATH}")

# Clean up temporary frames
print("\nCleaning up temporary frames...")
try:
    shutil.rmtree(frames_dir)
    print("Temporary frames cleaned up successfully")
except Exception as e:
    print(f"Error cleaning up temporary frames: {e}")

print("\nVIE评分测试完成!") 