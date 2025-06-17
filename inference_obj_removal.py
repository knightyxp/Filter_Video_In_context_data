import json
import cv2
import re
from tqdm import tqdm
import torch
import os
import shutil

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
    max_model_len=8192,
    gpu_memory_utilization=0.85,  # Slightly higher utilization
    swap_space=2,  # Add swap space for memory overflow
    enforce_eager=True,  # Use eager execution for better memory management
    disable_log_stats=True,  # Disable vLLM progress logs
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=256,
    stop_token_ids=[],
    skip_special_tokens=True,
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

def extract_answer(text):
    """Extract answer from model output"""
    # Look for simple yes/no patterns
    text_lower = text.lower().strip()
    
    # Direct yes/no responses
    if text_lower.startswith('yes') or 'yes' in text_lower[:20]:
        return 'yes'
    elif text_lower.startswith('no') or 'no' in text_lower[:20]:
        return 'no'
    
    # Look for positive indicators
    positive_indicators = ['successfully', 'removed', 'absent', 'missing', 'gone', 'deleted', 'indeed', 'correct']
    negative_indicators = ['not', 'no', 'present', 'visible', 'failed', 'unable', 'cannot', 'still there']
    
    pos_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
    neg_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
    
    if pos_count > neg_count:
        return 'yes'
    elif neg_count > pos_count:
        return 'no'
    else:
        return 'uncertain'

# Load preprocessed data
PREPROCESSED_DATA_PATH = "/scratch3/yan204/yxp/Senorita/preprocessed_obj_removal_data.json"
OUTPUT_PATH = "/scratch3/yan204/yxp/Senorita/hq_obj_removal.json"

def load_preprocessed_data():
    """Load preprocessed valid_samples and messages"""
    try:
        with open(PREPROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
            preprocessed = json.load(f)
        valid_samples = preprocessed.get("valid_samples", [])
        messages = preprocessed.get("messages", [])
        frames_dir = preprocessed.get("frames_dir", "")
        return valid_samples, messages, frames_dir
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return [], [], ""

print("Loading preprocessed data...")
valid_samples, messages, frames_dir = load_preprocessed_data()

if len(valid_samples) == 0 or len(messages) == 0:
    print("No preprocessed data found. Please run preprocess_obj_removal.py first.")
    exit(1)

print(f"Loaded {len(valid_samples)} samples and {len(messages)} messages")
print(f"Frames directory: {frames_dir}")

# Continue with the rest of the processing
print(f"Ready to process {len(messages)} samples with vision model...")

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

print("Processing samples with vision model...")
for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing qwen filter data"):
    batch_messages = messages[i:i + BSZ]
    batch_samples = valid_samples[i:i + BSZ]

    try:
        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
        
        llm_inputs = []
        for idx, prompt in enumerate(prompts):
            mm_data = {"image": image_inputs[idx * 2:(idx + 1) * 2]}  # Each sample has 2 images
            
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": {},
            })

        outputs = llm.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=False)
        batch_output_text = [out.outputs[0].text for out in outputs]
        
        
    except Exception as e:
        print(f'Error processing batch starting at {i}: {e}')
        batch_output_text = ['error'] * len(batch_samples)

    # Process results
    for j, (sample, model_output) in enumerate(zip(batch_samples, batch_output_text)):
        answer = extract_answer(model_output)
        
        # Add analysis results to sample
        sample["model_response"] = model_output
        sample["analysis_result"] = answer
        sample["is_high_quality"] = (answer == 'yes')
        
        final_output.append(sample)
    
    # Save progress
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
        #print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
    except Exception as e:
        print(f"Error writing to output file: {e}")

# Final filtering - create high-quality subset
hq_samples = [sample for sample in final_output if sample.get("is_high_quality", False)]

print(f"\nFiltering complete!")
print(f"Total samples processed: {len(final_output)}")
print(f"High-quality samples (answered 'yes'): {len(hq_samples)}")
if len(final_output) > 0:
    print(f"Quality rate: {len(hq_samples)/len(final_output)*100:.2f}%")

# Save high-quality results
hq_output_path = "/projects/D2DCRC/xiangpeng/Senorita/hq_obj_removal_filtered.json"
with open(hq_output_path, "w", encoding="utf-8") as f:
    json.dump({"results": hq_samples}, f, indent=2, ensure_ascii=False)

print(f"High-quality results saved to {hq_output_path}")

# Clean up temporary frames
print("Cleaning up temporary frames...")
try:
    shutil.rmtree(frames_dir)
    print("Temporary frames cleaned up successfully")
except Exception as e:
    print(f"Error cleaning up temporary frames: {e}")

print("Script completed!") 