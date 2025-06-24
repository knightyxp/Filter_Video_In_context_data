#!/usr/bin/env python3
"""
随机抽取视频样本用于质量评估
从过滤后的JSON文件中随机抽取100条视频并复制到临时文件夹
"""

import json
import random
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

# 配置参数 - 直接在这里修改
JSON_FILES = [
    "vie_hq_obj_addition_filtered.json"  # VIE评分后的高质量文件
]
OUTPUT_DIR = "./temp_eval_samples_vie"
SAMPLE_SIZE = 100
BASE_PATH = "."
RANDOM_SEED = 42  # 设置为None使用随机种子


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理不同的JSON结构
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unsupported JSON structure in {file_path}")


def get_video_paths(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """从数据中提取视频路径信息"""
    video_info = []
    
    for item in data:
        # 检查是否有必要的视频路径字段
        if 'target_video_path' in item and 'source_video_path' in item:
            video_info.append({
                'target_video_path': item['target_video_path'],
                'source_video_path': item['source_video_path'],
                'instruction': item.get('instruction', ''),
                'enhanced_instruction': item.get('enhanced_instruction', ''),
                'is_high_quality': item.get('is_high_quality', False)
            })
    
    return video_info


def copy_video_files(video_info: List[Dict[str, str]], output_dir: str, base_path: str = ".") -> None:
    """复制视频文件到输出目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建元数据文件
    metadata = []
    
    for i, info in enumerate(video_info):
        target_path = os.path.join(base_path, info['target_video_path'])
        source_path = os.path.join(base_path, info['source_video_path'])
        
        # 生成新的文件名
        target_filename = f"{i:03d}_target_{os.path.basename(target_path)}"
        source_filename = f"{i:03d}_source_{os.path.basename(source_path)}"
        
        target_dest = os.path.join(output_dir, target_filename)
        source_dest = os.path.join(output_dir, source_filename)
        
        # 复制文件
        try:
            if os.path.exists(target_path):
                shutil.copy2(target_path, target_dest)
                print(f"Copied target: {target_filename}")
            else:
                print(f"Warning: Target file not found: {target_path}")
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, source_dest)
                print(f"Copied source: {source_filename}")
            else:
                print(f"Warning: Source file not found: {source_path}")
            
            # 记录元数据
            metadata.append({
                'index': i,
                'target_file': target_filename,
                'source_file': source_filename,
                'instruction': info['instruction'],
                'enhanced_instruction': info['enhanced_instruction'],
                'is_high_quality': info['is_high_quality'],
                'original_target_path': info['target_video_path'],
                'original_source_path': info['source_video_path']
            })
            
        except Exception as e:
            print(f"Error copying files for index {i}: {e}")
    
    # 保存元数据
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nMetadata saved to: {metadata_path}")


def main():
    print("随机抽取视频样本用于质量评估")
    print("=" * 50)
    
    # 设置随机种子
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        print(f"Using random seed: {RANDOM_SEED}")
    
    # 加载所有JSON文件的数据
    all_video_info = []
    
    for json_file in JSON_FILES:
        print(f"Loading data from: {json_file}")
        try:
            data = load_json_data(json_file)
            video_info = get_video_paths(data)
            all_video_info.extend(video_info)
            print(f"  Loaded {len(video_info)} video entries")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    print(f"\nTotal video entries loaded: {len(all_video_info)}")
    
    if len(all_video_info) == 0:
        print("No video entries found!")
        return
    
    # 随机抽取样本
    sample_size = min(SAMPLE_SIZE, len(all_video_info))
    sampled_videos = random.sample(all_video_info, sample_size)
    
    print(f"Randomly sampled {sample_size} videos")
    
    # 复制文件
    print(f"\nCopying files to: {OUTPUT_DIR}")
    copy_video_files(sampled_videos, OUTPUT_DIR, BASE_PATH)
    
    print(f"\nDone! {sample_size} video pairs copied to {OUTPUT_DIR}")


if __name__ == "__main__":
    main() 