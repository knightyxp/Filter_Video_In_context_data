import os
import json
import cv2
import re
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import math

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
 

MODEL_PATH = "/scratch3/yan204/yxp/Qwen2.5-VL-32B-Instruct"
BSZ = 4  # Reduce batch size to save memory

# Optimize vLLM configuration for better memory usage
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs
    dtype="bfloat16",  # Use bfloat16 to save memory like transformers version
    limit_mm_per_prompt={"image": 10, "video": 10},
    max_model_len=32768,  # Increase to handle multi-modal embeddings
    gpu_memory_utilization=0.85,  # Slightly higher utilization
    swap_space=2,  # Add swap space for memory overflow
    enforce_eager=True,  # Use eager execution for better memory management
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=256,
    stop_token_ids=[],
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video and save as image"""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(output_path, frame)
            cap.release()
            return True
        cap.release()
        return False
    except Exception as e:
        print(f"Error extracting frame from {video_path}: {e}")
        return False

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

def create_sc_prompt(instruction, enhanced_instruction):
    """Create Semantic Consistency prompt"""
    return f"""Human: You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on the given rules. You will have to give your output in this way (Keep your reasoning concise and short.):
{{"score" : [...],"reasoning" : "..."}}
and don't output anything else.

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first. The objective is to evaluate how successfully the editing instruction has been executed in the second image. Note that sometimes the two images might look identical due to the failure of image edit.

From a scale 0 to 10:
A score from 0 to 10 will be given based on the success of the editing.
- 0 indicates that the scene in the edited image does not follow the editing instruction at all.
- 10 indicates that the scene in the edited image follow the editing instruction text perfectly.
- If the object in the instruction is not present in the original image at all, the score will be 0.

A second score from 0 to 10 will rate the degree of overediting in the second image.
- 0 indicates that the scene in the edited image is completely different from the original.
- 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.

Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: {enhanced_instruction}"""

def create_pq_prompt():
    """Create Perceptual Quality prompt"""
    return """Human: You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image.

All the images and humans in the images are AI-generated. So you may not worry about privacy or confidentiality.

You must focus solely on the technical quality and artifacts in the image, and **do not consider whether the context is natural or not**.

Your evaluation should focus on:
- Distortions
- Unusual body parts or proportions
- Unnatural Object Shapes

Rate the image on a scale from 0 to 10, where:
- 0 indicates significant AI-artifacts.
- 10 indicates an artifact-free image.

You will have to give your output in this way (Keep your reasoning concise and short.):
{"score": ...,"reasoning": "..."}
and don't output anything else."""

def process_single_sample_for_frames(args):
    """Process a single sample - extract frames from videos (for multiprocessing)"""
    idx, sample, frames_dir = args
    
    # Get absolute paths
    target_video = os.path.join("/scratch3/yan204/yxp/Senorita", sample["target_video_path"].lstrip("./"))
    source_video = os.path.join("/scratch3/yan204/yxp/Senorita", sample["source_video_path"].lstrip("./"))
    
    # Check if videos exist
    if not os.path.exists(target_video) or not os.path.exists(source_video):
        return None
    
    # Extract ID information from video path for naming
    target_parts = sample["target_video_path"].split("/")
    if len(target_parts) >= 3:
        filename = os.path.splitext(target_parts[-1])[0]
        if filename.endswith("_org"):
            filename = filename[:-4]
        file_id = filename
    else:
        file_id = str(idx)
    
    # Extract first frames
    source_frame_path = os.path.join(frames_dir, f"source_{file_id}.jpg")
    target_frame_path = os.path.join(frames_dir, f"target_{file_id}.jpg")
    
    if not extract_first_frame(source_video, source_frame_path):
        return None
        
    if not extract_first_frame(target_video, target_frame_path):
        return None
    
    return {
        'source_frame_path': source_frame_path,
        'target_frame_path': target_frame_path,
        'sample': sample,
        'idx': idx
    }

# 配置文件路径
INPUT_PATH = "/scratch3/yan204/yxp/Filter_Video_In_context_data/filter_resolution_json/filtered_obj_removal_592x336.json"
OUTPUT_PATH = "/scratch3/yan204/yxp/Senorita/vie_score_filter_obj_removal.json"
FRAMES_DIR = "/scratch3/yan204/yxp/Senorita/obj_removal_temp_frames"

# Create temporary directory for frames
os.makedirs(FRAMES_DIR, exist_ok=True)

print("Loading test data...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 处理不同的JSON结构
if isinstance(data, dict) and 'results' in data:
    data = data['results']
elif isinstance(data, list):
    pass
else:
    raise ValueError(f"Unsupported JSON structure")

print(f"Loaded {len(data)} test samples")

# Process data to extract frames
print("Extracting frames...")
max_workers = min(40, mp.cpu_count())
valid_results = []

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    args_list = [(idx, sample, FRAMES_DIR) for idx, sample in enumerate(data)]
    futures = [executor.submit(process_single_sample_for_frames, args) for args in args_list]
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
        result = future.result()
        if result is not None:
            valid_results.append(result)

# Sort results by original index
valid_results.sort(key=lambda x: x['idx'])
print(f"Prepared {len(valid_results)} valid samples for VIE scoring")

# Process samples with VIE scoring
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

print("Processing samples with VIE scoring...")
for i in tqdm(range(start_idx, len(valid_results), BSZ), desc="Processing batches"):
    batch_results = valid_results[i:i + BSZ]
    
    # Prepare SC evaluation messages
    sc_messages = []
    pq_messages = []
    
    for result in batch_results:
        sample = result['sample']
        source_frame = result['source_frame_path']
        target_frame = result['target_frame_path']
        
        # SC evaluation (compare source and target)
        sc_prompt = create_sc_prompt(sample["instruction"], sample["enhanced_instruction"])
        sc_msg = [{
            "role": "user",
            "content": [
                {"type": "image", "image": source_frame, "min_pixels": 224 * 224, "max_pixels": 768 * 768},
                {"type": "image", "image": target_frame, "min_pixels": 224 * 224, "max_pixels": 768 * 768},
                {"type": "text", "text": sc_prompt}
            ]
        }]
        sc_messages.append(sc_msg)
        
        # PQ evaluation (only target image)
        pq_prompt = create_pq_prompt()
        pq_msg = [{
            "role": "user",
            "content": [
                {"type": "image", "image": target_frame, "min_pixels": 224 * 224, "max_pixels": 768 * 768},
                {"type": "text", "text": pq_prompt}
            ]
        }]
        pq_messages.append(pq_msg)
    
    # Process SC scores
    try:
        sc_prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in sc_messages]
        sc_image_inputs, sc_video_inputs, sc_video_kwargs = process_vision_info(sc_messages, return_video_kwargs=True)
        
        sc_llm_inputs = []
        for idx, prompt in enumerate(sc_prompts):
            mm_data = {"image": sc_image_inputs[idx * 2:(idx + 1) * 2]}
            sc_llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": {},
            })

        sc_outputs = llm.generate(sc_llm_inputs, sampling_params=sampling_params)
        sc_responses = [out.outputs[0].text for out in sc_outputs]
        
    except Exception as e:
        print(f'Error processing SC batch starting at {i}: {e}')
        sc_responses = ['error'] * len(batch_results)
    
    # Process PQ scores
    try:
        pq_prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in pq_messages]
        pq_image_inputs, pq_video_inputs, pq_video_kwargs = process_vision_info(pq_messages, return_video_kwargs=True)
        
        pq_llm_inputs = []
        for idx, prompt in enumerate(pq_prompts):
            mm_data = {"image": pq_image_inputs[idx:idx + 1]}
            pq_llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": {},
            })

        pq_outputs = llm.generate(pq_llm_inputs, sampling_params=sampling_params)
        pq_responses = [out.outputs[0].text for out in pq_outputs]
        
    except Exception as e:
        print(f'Error processing PQ batch starting at {i}: {e}')
        pq_responses = ['error'] * len(batch_results)

    # Process results
    for j, (result, sc_response, pq_response) in enumerate(zip(batch_results, sc_responses, pq_responses)):
        sample = result['sample']
        
        # Extract SC scores
        sc_score1, sc_score2, sc_reasoning = extract_scores_from_response(sc_response)
        
        # Extract PQ score
        pq_score, _, pq_reasoning = extract_scores_from_response(pq_response)
        
        # Normalize scores to 0-1 and calculate overall score
        sc_normalized = (sc_score1 / 10.0) if sc_score1 is not None else 0.0
        pq_normalized = (pq_score / 10.0) if pq_score is not None else 0.0
        
        # Calculate overall score: O = sqrt(SC × PQ)
        overall_score = math.sqrt(sc_normalized * pq_normalized) if (sc_normalized > 0 and pq_normalized > 0) else 0.0
        
        # Add evaluation results to sample
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
hq_output_path = "/scratch3/yan204/yxp/Senorita/vie_score_details_obj_removal.json"
if final_output:
    hq_samples = [sample for sample in final_output if sample.get("overall_score", 0) > 0.9]
    with open(hq_output_path, "w", encoding="utf-8") as f:
        json.dump({"results": hq_samples}, f, indent=2, ensure_ascii=False)
    print(f"\n高质量样本保存到: {hq_output_path}")

print(f"完整结果保存到: {OUTPUT_PATH}")

# Clean up temporary frames
print("\nCleaning up temporary frames...")
import shutil
try:
    shutil.rmtree(FRAMES_DIR)
    print("Temporary frames cleaned up successfully")
except Exception as e:
    print(f"Error cleaning up temporary frames: {e}")

print("\nVIE评分测试完成!") 