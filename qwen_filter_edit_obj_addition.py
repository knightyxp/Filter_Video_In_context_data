import os
import json
import cv2
import re
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import glob

# Disable vLLM progress bars
os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"

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
    positive_indicators = ['successfully', 'added', 'present', 'visible', 'appears', 'indeed', 'correct']
    negative_indicators = ['not', 'no', 'absent', 'missing', 'failed', 'unable', 'cannot']
    
    pos_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
    neg_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
    
    if pos_count > neg_count:
        return 'yes'
    elif neg_count > pos_count:
        return 'no'
    else:
        return 'uncertain'

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
    # Example: "./obj_removal_videos_upload/60/828b086b9c7ad78720ffd29c4ab0d883_org.mp4"
    target_parts = sample["target_video_path"].split("/")
    if len(target_parts) >= 3:
        filename = os.path.splitext(target_parts[-1])[0]  # "828b086b9c7ad78720ffd29c4ab0d883_org"
        # Remove "_org" suffix if present
        if filename.endswith("_org"):
            filename = filename[:-4]
        file_id = filename  # "828b086b9c7ad78720ffd29c4ab0d883"
    else:
        # Fallback to index if path format is unexpected
        file_id = str(idx)
    
    # Extract first frames with new naming
    source_frame_path = os.path.join(frames_dir, f"source_{file_id}.jpg")
    target_frame_path = os.path.join(frames_dir, f"target_{file_id}.jpg")
    
    if not extract_first_frame(source_video, source_frame_path):
        return None
        
    if not extract_first_frame(target_video, target_frame_path):
        return None
    
    # Create prompt message
    enhanced_instruction = sample["enhanced_instruction"]
    
    # Extract the main instruction (remove "Add a/an/the" prefix if present)
    instruction_text = enhanced_instruction
    if instruction_text.lower().startswith("add a "):
        object_name = instruction_text[6:]
    elif instruction_text.lower().startswith("add an "):
        object_name = instruction_text[7:]
    elif instruction_text.lower().startswith("add the "):
        object_name = instruction_text[8:]
    elif instruction_text.lower().startswith("add "):
        object_name = instruction_text[4:]
    else:
        object_name = instruction_text
    
    question = f"Looking at these two images, the instruction was '{enhanced_instruction}'. Did the second image successfully add {object_name} to the first image? Please answer with a simple 'yes' or 'no' and provide a brief explanation."
    
    msg = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": source_frame_path,
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {
                "type": "image", 
                "image": target_frame_path,
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {
                "type": "text",
                "text": question
            }
        ]
    }]
    
    return {
        'message': msg,
        'sample': sample,
        'idx': idx
    } 

# Load the filtered data
INPUT_PATH = "/scratch3/yan204/yxp/Filter_Video_In_context_data/filter_resolution_json/filtered_obj_addition_592x336.json"
OUTPUT_PATH = "/scratch3/yan204/yxp/Senorita/hq_obj_addition.json"
FRAMES_DIR = "/scratch3/yan204/yxp/Senorita/obj_addition_temp_frames"

# Create temporary directory for frames
os.makedirs(FRAMES_DIR, exist_ok=True)

print("Loading filtered data...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# Check if frames already exist
def check_existing_frames(frames_dir, num_samples):
    """Check if frames already exist for all samples"""
    if not os.path.exists(frames_dir):
        return False
    
    # Count existing source and target frames
    source_frames = glob.glob(os.path.join(frames_dir, "source_*.jpg"))
    target_frames = glob.glob(os.path.join(frames_dir, "target_*.jpg"))
    
    expected_frames = num_samples * 2  # Each sample has source and target frame
    actual_frames = len(source_frames) + len(target_frames)
    
    print(f"Found {len(source_frames)} source frames and {len(target_frames)} target frames")
    print(f"Expected {expected_frames} frames total, found {actual_frames}")
    
    return actual_frames >= expected_frames

# Process data to extract frames and create prompts
valid_samples = []
messages = []

skip_preprocessing = check_existing_frames(FRAMES_DIR, len(data))

if skip_preprocessing:
    print("Frames already exist, skipping preprocessing...")
    print("Preparing prompts from existing frames...")
    
    # Create prompts using existing frames
    for idx, sample in enumerate(tqdm(data, desc="Creating prompts from existing frames")):
        # Extract ID information from video path for naming (same logic as before)
        target_parts = sample["target_video_path"].split("/")
        if len(target_parts) >= 3:
            filename = os.path.splitext(target_parts[-1])[0]
            if filename.endswith("_org"):
                filename = filename[:-4]
            file_id = filename
        else:
            file_id = str(idx)
        
        # Check if frames exist for this sample
        source_frame_path = os.path.join(FRAMES_DIR, f"source_{file_id}.jpg")
        target_frame_path = os.path.join(FRAMES_DIR, f"target_{file_id}.jpg")
        
        if os.path.exists(source_frame_path) and os.path.exists(target_frame_path):
            # Create prompt message (same logic as before)
            enhanced_instruction = sample["enhanced_instruction"]
            
            instruction_text = enhanced_instruction
            if instruction_text.lower().startswith("add a "):
                object_name = instruction_text[6:]
            elif instruction_text.lower().startswith("add an "):
                object_name = instruction_text[7:]
            elif instruction_text.lower().startswith("add the "):
                object_name = instruction_text[8:]
            elif instruction_text.lower().startswith("add "):
                object_name = instruction_text[4:]
            else:
                object_name = instruction_text
            
            question = f"Looking at these two images, the instruction was '{enhanced_instruction}'. Did the second image successfully add {object_name} to the first image? Please answer with a simple 'yes' or 'no' and provide a brief explanation."
            
            msg = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": source_frame_path,
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28,
                    },
                    {
                        "type": "image", 
                        "image": target_frame_path,
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28,
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }]
            
            messages.append(msg)
            valid_samples.append(sample)
    
    print(f"Prepared {len(messages)} samples from existing frames")
    
else:
    print("Extracting frames and preparing prompts...")
    # Use multiprocessing for frame extraction
    max_workers = min(40, mp.cpu_count())  # Use up to 40 workers or CPU count
    valid_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for multiprocessing
        args_list = [(idx, sample, FRAMES_DIR) for idx, sample in enumerate(data)]
        
        # Submit all tasks
        futures = [executor.submit(process_single_sample_for_frames, args) for args in args_list]
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            result = future.result()
            if result is not None:
                valid_results.append(result)

    # Sort results by original index to maintain order
    valid_results.sort(key=lambda x: x['idx'])

    # Extract messages and samples in original order
    for result in valid_results:
        messages.append(result['message'])
        valid_samples.append(result['sample'])

    print(f"Prepared {len(messages)} valid samples for processing")

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
for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
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

        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
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
hq_output_path = "/projects/D2DCRC/xiangpeng/Senorita/hq_obj_addition_filtered.json"
with open(hq_output_path, "w", encoding="utf-8") as f:
    json.dump({"results": hq_samples}, f, indent=2, ensure_ascii=False)

print(f"High-quality results saved to {hq_output_path}")

# Clean up temporary frames
print("Cleaning up temporary frames...")
import shutil
try:
    shutil.rmtree(FRAMES_DIR)
    print("Temporary frames cleaned up successfully")
except Exception as e:
    print(f"Error cleaning up temporary frames: {e}")

print("Script completed!")

