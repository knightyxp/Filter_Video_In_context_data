import json
import cv2
import re
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import datetime
import os
import glob

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
    
    # Create prompt message for object removal
    enhanced_instruction = sample["enhanced_instruction"]
    
    # Extract the main instruction (remove "Remove a/an/the" prefix if present)
    instruction_text = enhanced_instruction
    if instruction_text.lower().startswith("remove a "):
        object_name = instruction_text[9:]
    elif instruction_text.lower().startswith("remove an "):
        object_name = instruction_text[10:]
    elif instruction_text.lower().startswith("remove the "):
        object_name = instruction_text[11:]
    elif instruction_text.lower().startswith("remove "):
        object_name = instruction_text[7:]
    else:
        object_name = instruction_text
    
    question = f"Looking at these two images, the instruction was '{enhanced_instruction}'. Did the second image successfully remove {object_name} from the first image? Please answer with a simple 'yes' or 'no' and provide a brief explanation."
    
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
INPUT_PATH = "/scratch3/yan204/yxp/Filter_Video_In_context_data/filter_resolution_json/filtered_obj_removal_592x336.json"
FRAMES_DIR = "/projects/D2DCRC/xiangpeng/Senorita/obj_removal_temp_frames"
# Add paths for saving preprocessed data
PREPROCESSED_DATA_PATH = "/projects/D2DCRC/xiangpeng/Senorita/preprocessed_obj_removal_data.json"

# Create temporary directory for frames
os.makedirs(FRAMES_DIR, exist_ok=True)

print("Loading filtered data...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# Check if preprocessed data already exists
def check_preprocessed_data():
    """Check if preprocessed valid_samples and messages exist"""
    return os.path.exists(PREPROCESSED_DATA_PATH)

def load_preprocessed_data():
    """Load preprocessed valid_samples and messages"""
    try:
        with open(PREPROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
            preprocessed = json.load(f)
        return preprocessed.get("valid_samples", []), preprocessed.get("messages", [])
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return [], []

def save_preprocessed_data(valid_samples, messages):
    """Save preprocessed valid_samples and messages"""
    try:
        preprocessed_data = {
            "valid_samples": valid_samples,
            "messages": messages,
            "timestamp": str(datetime.datetime.now()),
            "total_samples": len(valid_samples),
            "frames_dir": FRAMES_DIR
        }
        with open(PREPROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(preprocessed_data, f, indent=2, ensure_ascii=False)
        print(f"Saved preprocessed data with {len(valid_samples)} samples to {PREPROCESSED_DATA_PATH}")
        print(f"Frames saved to: {FRAMES_DIR}")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")

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

# Check if preprocessed data exists
if check_preprocessed_data():
    print("Found existing preprocessed data, loading...")
    valid_samples, messages = load_preprocessed_data()
    if len(valid_samples) > 0 and len(messages) > 0:
        print(f"Loaded {len(valid_samples)} preprocessed samples and {len(messages)} messages")
    else:
        print("Preprocessed data is empty or corrupted, will reprocess...")
        valid_samples, messages = [], []

# If no valid preprocessed data, process from scratch
if len(valid_samples) == 0 or len(messages) == 0:
    print("Processing data from scratch...")
    
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
                # Create prompt message for object removal
                enhanced_instruction = sample["enhanced_instruction"]
                
                # Extract the main instruction (remove "Remove a/an/the" prefix if present)
                instruction_text = enhanced_instruction
                if instruction_text.lower().startswith("remove a "):
                    object_name = instruction_text[9:]
                elif instruction_text.lower().startswith("remove an "):
                    object_name = instruction_text[10:]
                elif instruction_text.lower().startswith("remove the "):
                    object_name = instruction_text[11:]
                elif instruction_text.lower().startswith("remove "):
                    object_name = instruction_text[7:]
                else:
                    object_name = instruction_text
                
                question = f"Looking at these two images, the instruction was '{enhanced_instruction}'. Did the second image successfully remove {object_name} from the first image? Please answer with a simple 'yes' or 'no' and provide a brief explanation."
                
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
    
    # Save preprocessed data for future use
    save_preprocessed_data(valid_samples, messages)

print(f"Preprocessing complete! Ready to process {len(messages)} samples with vision model...")
print(f"Frames directory: {FRAMES_DIR}")
print(f"Preprocessed data: {PREPROCESSED_DATA_PATH}") 