import json
import cv2
import re
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import datetime
import os

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
    
    # Create SC evaluation message (compare source and target)
    sc_prompt = create_sc_prompt(sample["instruction"], sample["enhanced_instruction"])
    sc_msg = [{
        "role": "user",
        "content": [
            {"type": "image", "image": source_frame_path, "min_pixels": 224 * 224, "max_pixels": 768 * 768},
            {"type": "image", "image": target_frame_path, "min_pixels": 224 * 224, "max_pixels": 768 * 768},
            {"type": "text", "text": sc_prompt}
        ]
    }]
    
    # Create PQ evaluation message (only target image)
    pq_prompt = create_pq_prompt()
    pq_msg = [{
        "role": "user",
        "content": [
            {"type": "image", "image": target_frame_path, "min_pixels": 224 * 224, "max_pixels": 768 * 768},
            {"type": "text", "text": pq_prompt}
        ]
    }]
    
    return {
        'sc_message': sc_msg,
        'pq_message': pq_msg,
        'sample': sample,
        'idx': idx
    } 

# Load the filtered data
INPUT_PATH = "/scratch3/yan204/yxp/Filter_Video_In_context_data/filter_resolution_json/filtered_obj_removal_592x336.json"
FRAMES_DIR = "/scratch3/yan204/yxp/Senorita/obj_removal_temp_frames"
PREPROCESSED_DATA_PATH = "/scratch3/yan204/yxp/Senorita/preprocessed_obj_removal_data.json"

# Create temporary directory for frames
os.makedirs(FRAMES_DIR, exist_ok=True)

print("Loading filtered data...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# Process data to extract frames and create prompts
valid_samples = []
sc_messages = []
pq_messages = []

print("Extracting frames and preparing prompts...")
max_workers = min(40, mp.cpu_count())
valid_results = []

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    args_list = [(idx, sample, FRAMES_DIR) for idx, sample in enumerate(data)]
    futures = [executor.submit(process_single_sample_for_frames, args) for args in args_list]
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
        result = future.result()
        if result is not None:
            valid_results.append(result)

# Sort results by original index to maintain order
valid_results.sort(key=lambda x: x['idx'])

# Extract messages and samples in original order
for result in valid_results:
    sc_messages.append(result['sc_message'])
    pq_messages.append(result['pq_message'])
    valid_samples.append(result['sample'])

print(f"Prepared {len(valid_samples)} valid samples for VIE scoring")

# Save preprocessed data
preprocessed_data = {
    "valid_samples": valid_samples,
    "sc_messages": sc_messages,
    "pq_messages": pq_messages,
    "timestamp": str(datetime.datetime.now()),
    "total_samples": len(valid_samples),
    "frames_dir": FRAMES_DIR
}

with open(PREPROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(preprocessed_data, f, indent=2, ensure_ascii=False)

# Count images in frames directory
print("Counting images in frames directory...")
source_frames = [f for f in os.listdir(FRAMES_DIR) if f.startswith("source_") and f.endswith(".jpg")]
target_frames = [f for f in os.listdir(FRAMES_DIR) if f.startswith("target_") and f.endswith(".jpg")]

print(f"Found {len(source_frames)} source frames and {len(target_frames)} target frames")
print(f"Total images: {len(source_frames) + len(target_frames)}")
print(f"Expected images: {len(valid_samples) * 2}")

print(f"Preprocessing complete! Ready to process {len(valid_samples)} samples with VIE scoring...")
print(f"Frames directory: {FRAMES_DIR}")
print(f"Preprocessed data: {PREPROCESSED_DATA_PATH}") 