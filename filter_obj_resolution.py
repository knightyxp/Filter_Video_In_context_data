import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def is_592x336(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width == 592 and height == 336

def video_diff_ratio(src_path, tgt_path):
    cap1 = cv2.VideoCapture(src_path)
    cap2 = cv2.VideoCapture(tgt_path)
    total_pixels = 0
    diff_pixels = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        diff = cv2.absdiff(frame1, frame2)
        mask = np.any(diff > 30, axis=2)  # 阈值可调
        diff_pixels += np.sum(mask)
        total_pixels += mask.size
    cap1.release()
    cap2.release()
    return diff_pixels / total_pixels if total_pixels else 0

def process_item(item, base_dir):
    src = os.path.join(base_dir, item['source_video_path'])
    tgt = os.path.join(base_dir, item['target_video_path'])
    if not (os.path.exists(src) and os.path.exists(tgt)):
        return None
    if not (is_592x336(src) and is_592x336(tgt)):
        return None
    return item

def filter_by_resolution(json_path, base_dir, out_json):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 替换路径
    for item in data:
        if 'source_video_path' in item:
            item['source_video_path'] = item['source_video_path'].replace(
                './obj_swap_videos/', './obj_swap_upload/')
        if 'target_video_path' in item:
            item['target_video_path'] = item['target_video_path'].replace(
                './obj_swap_videos/', './obj_swap_upload/')
    
    filtered = []
    with ProcessPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(process_item, item, base_dir) for item in data]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(json_path)}"):
            result = f.result()
            if result:
                filtered.append(result)
    
    with open(out_json, 'w') as f:
        json.dump(filtered, f, indent=2)
    print(f"已保存 {len(filtered)} 条到 {out_json}")

if __name__ == '__main__':
    base_dir = '/projects/D2DCRC/xiangpeng/Senorita'

    # 你可以根据实际文件名调整
    # addition_json = os.path.join(base_dir, 'updated_data_obj_addition_videos.json')
    # removal_json  = os.path.join(base_dir, 'updated_data_obj_removal.json')
    swap_json     = os.path.join(base_dir, 'updated_data_obj_swap_videos.json')

    # filter_by_resolution(addition_json, base_dir, os.path.join(base_dir, 'filtered_obj_addition_592x336.json'))
    #filter_by_resolution(removal_json,  base_dir, os.path.join(base_dir, 'filtered_obj_removal_592x336.json'))
    filter_by_resolution(swap_json,     base_dir, os.path.join(base_dir, 'filtered_obj_swap_592x336.json'))