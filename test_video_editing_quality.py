import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Set environment variables for better GPU memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use only 2 GPUs

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/scratch3/yan204/yxp/Qwen2.5-VL-32B-Instruct"
BSZ = 1  # Process one video pair at a time

# Optimize vLLM configuration for better memory usage
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=2,  # Reduce GPU count to save memory
    dtype="bfloat16",  # Use bfloat16 to save memory like transformers version
    limit_mm_per_prompt={"image": 8, "video": 0},  # Limit to 8 images, no video
    max_model_len=32768,  # Increase context length for multi-modal inputs
    gpu_memory_utilization=0.75,  # Reduce memory utilization
    swap_space=4,  # Increase swap space for memory overflow
    enforce_eager=True,  # Use eager execution for better memory management
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=512,
    stop_token_ids=[],
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

def extract_frames_from_video(video_path, output_dir, num_frames=4):
    """Extract evenly distributed frames from a video"""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            print(f"Video {video_path} has only {total_frames} frames, less than requested {num_frames}")
            num_frames = total_frames
        
        # Calculate frame indices to extract (evenly distributed)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        extracted_frames = []
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(output_dir, f"frame_{i:02d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)
            else:
                print(f"Failed to extract frame {frame_idx} from {video_path}")
        
        cap.release()
        return extracted_frames
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []

def create_video_comparison_prompt():
    """Create a comprehensive prompt for video editing evaluation"""
    prompt = """Please analyze these two sets of video frames. The first 4 frames are from the original video, and the next 4 frames are from the edited video.

Please evaluate the following aspects:

1. **Editing Success**: Was the editing operation successfully applied? What changes can you observe between the original and edited video?

2. **Video Quality**: How is the overall quality of the edited video compared to the original? Consider factors like:
   - Visual clarity and sharpness
   - Color consistency
   - Artifacts or distortions
   - Overall visual fidelity

3. **Motion Analysis**: Analyze the motion characteristics:
   - Is the motion smooth and natural in the edited video?
   - How strong/weak is the motion in both videos?
   - Are there any motion artifacts or unnatural movements?

4. **Editing Region Size**: 
   - What is the approximate size of the edited region relative to the entire frame?
   - Is the editing localized or does it affect a large portion of the video?

Please provide a detailed analysis for each aspect and give an overall assessment of the editing quality.

Format your response as:
**Editing Success**: [Your analysis]
**Video Quality**: [Your analysis] 
**Motion Analysis**: [Your analysis]
**Editing Region Size**: [Your analysis]
**Overall Assessment**: [Your overall evaluation]"""
    
    return prompt

def process_video_pair(org_video_path, edited_video_path, frames_dir):
    """Process a pair of videos and extract frames for comparison"""
    
    # Create subdirectories for frames
    org_frames_dir = os.path.join(frames_dir, "original")
    edited_frames_dir = os.path.join(frames_dir, "edited")
    os.makedirs(org_frames_dir, exist_ok=True)
    os.makedirs(edited_frames_dir, exist_ok=True)
    
    # Extract frames from both videos
    print("Extracting frames from original video...")
    org_frames = extract_frames_from_video(org_video_path, org_frames_dir, num_frames=4)
    
    print("Extracting frames from edited video...")
    edited_frames = extract_frames_from_video(edited_video_path, edited_frames_dir, num_frames=4)
    
    if len(org_frames) != 4 or len(edited_frames) != 4:
        print(f"Failed to extract required frames. Original: {len(org_frames)}, Edited: {len(edited_frames)}")
        return None
    
    # Create message with all frames
    content = []
    
    # Add original video frames
    for i, frame_path in enumerate(org_frames):
        content.append({
            "type": "image",
            "image": frame_path,
            "min_pixels": 224 * 224,
            "max_pixels": 512 * 512,  # Reduce max resolution to save memory
        })
    
    # Add edited video frames
    for i, frame_path in enumerate(edited_frames):
        content.append({
            "type": "image",
            "image": frame_path,
            "min_pixels": 224 * 224,
            "max_pixels": 512 * 512,  # Reduce max resolution to save memory
        })
    
    # Add text prompt
    content.append({
        "type": "text",
        "text": create_video_comparison_prompt()
    })
    
    msg = [{
        "role": "user",
        "content": content
    }]
    
    return {
        'message': msg,
        'org_video_path': org_video_path,
        'edited_video_path': edited_video_path,
        'org_frames': org_frames,
        'edited_frames': edited_frames
    }

def extract_evaluation_results(text):
    """Extract structured evaluation results from model output"""
    results = {
        'editing_success': '',
        'video_quality': '',
        'motion_analysis': '',
        'editing_region_size': '',
        'overall_assessment': '',
        'raw_response': text
    }
    
    # Try to extract structured information
    sections = ['Editing Success', 'Video Quality', 'Motion Analysis', 'Editing Region Size', 'Overall Assessment']
    
    for section in sections:
        pattern = f"\\*\\*{section}\\*\\*:?\\s*(.+?)(?=\\*\\*|$)"
        import re
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            key = section.lower().replace(' ', '_')
            results[key] = content
    
    return results

# Main execution
def main():
    # Set up paths
    ORG_VIDEO = "test_org_video.mp4"
    EDITED_VIDEO = "test_video.mp4"
    FRAMES_DIR = "temp_video_frames"
    OUTPUT_PATH = "video_editing_evaluation.json"
    
    # Check if videos exist
    if not os.path.exists(ORG_VIDEO):
        print(f"Error: Original video {ORG_VIDEO} not found!")
        return
    
    if not os.path.exists(EDITED_VIDEO):
        print(f"Error: Edited video {EDITED_VIDEO} not found!")
        return
    
    # Create temporary directory for frames
    os.makedirs(FRAMES_DIR, exist_ok=True)
    
    print("Processing video pair for evaluation...")
    
    # Process the video pair
    result = process_video_pair(ORG_VIDEO, EDITED_VIDEO, FRAMES_DIR)
    
    if result is None:
        print("Failed to process video pair!")
        return
    
    print("Running video quality evaluation with vision model...")
    
    try:
        # Prepare prompt for the model
        message = result['message']
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        
        # Process vision information
        image_inputs, video_inputs, video_kwargs = process_vision_info([message], return_video_kwargs=True)
        
        # Prepare input for the model
        mm_data = {"image": image_inputs}  # 8 images total (4 original + 4 edited)
        
        llm_input = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": {},
        }
        
        # Generate response
        outputs = llm.generate([llm_input], sampling_params=sampling_params)
        model_output = outputs[0].outputs[0].text
        
        print("Model evaluation completed!")
        print("="*60)
        print("EVALUATION RESULTS:")
        print("="*60)
        print(model_output)
        print("="*60)
        
        # Extract structured results
        evaluation_results = extract_evaluation_results(model_output)
        
        # Save results to JSON
        final_results = {
            'original_video': ORG_VIDEO,
            'edited_video': EDITED_VIDEO,
            'evaluation': evaluation_results,
            'frame_paths': {
                'original_frames': result['org_frames'],
                'edited_frames': result['edited_frames']
            }
        }
        
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to {OUTPUT_PATH}")
        
    except Exception as e:
        print(f'Error during model evaluation: {e}')
        return
    
    # Clean up temporary frames
    print("\nCleaning up temporary frames...")
    import shutil
    try:
        shutil.rmtree(FRAMES_DIR)
        print("Temporary frames cleaned up successfully")
    except Exception as e:
        print(f"Error cleaning up temporary frames: {e}")
    
    print("Video evaluation completed!")

if __name__ == "__main__":
    main() 