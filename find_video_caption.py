import json
import os

def find_video_caption():
    # 目标视频路径
    target_video = "/projects/D2DCRC/xiangpeng/Senorita/obj_removal_videos_upload/189/ffb79099d17fbb6279a23015da9c0cc4_org.mp4"
    
    # JSON文件路径
    json_file = "/projects/D2DCRC/xiangpeng/Senorita/filtered_obj_addition_592x336.json"
    
    # 提取文件名用于匹配
    target_filename = "ffb79099d17fbb6279a23015da9c0cc4_org.mp4"
    
    print(f"Looking for video: {target_video}")
    print(f"Target filename: {target_filename}")
    print(f"Searching in: {json_file}")
    print("="*60)
    
    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} records from JSON file")
        
        # 搜索匹配的视频
        found_records = []
        
        for i, record in enumerate(data):
            # 检查各种可能的路径字段
            paths_to_check = []
            
            # 添加各种可能的路径字段
            for key in record.keys():
                if 'video' in key.lower() or 'path' in key.lower():
                    if isinstance(record[key], str):
                        paths_to_check.append((key, record[key]))
            
            # 检查每个路径是否匹配
            for field_name, path in paths_to_check:
                if target_filename in path or target_video in path:
                    found_records.append((i, field_name, record))
                    break
        
        if found_records:
            print(f"Found {len(found_records)} matching record(s):")
            print("="*60)
            
            for idx, (record_idx, field_name, record) in enumerate(found_records):
                print(f"\n--- Record {idx + 1} (Index: {record_idx}) ---")
                print(f"Matched field: {field_name}")
                print(f"Path: {record[field_name]}")
                
                # 打印所有可能的caption相关字段
                caption_fields = []
                for key, value in record.items():
                    key_lower = key.lower()
                    if any(word in key_lower for word in ['caption', 'description', 'instruction', 'prompt', 'text']):
                        caption_fields.append((key, value))
                
                if caption_fields:
                    print("\nCaption/Description fields:")
                    for key, value in caption_fields:
                        print(f"  {key}: {value}")
                else:
                    print("\nNo caption fields found. All fields in this record:")
                    for key, value in record.items():
                        if isinstance(value, str) and len(value) < 200:  # Only show shorter string fields
                            print(f"  {key}: {value}")
                        elif not isinstance(value, str):
                            print(f"  {key}: {type(value).__name__}")
                        else:
                            print(f"  {key}: {value[:100]}... (truncated)")
                
                print("-" * 40)
        
        else:
            print("No matching records found!")
            print("\nLet me show you some example paths from the file to help debug:")
            
            # 显示前几个记录的路径示例
            for i in range(min(5, len(data))):
                record = data[i]
                print(f"\nRecord {i}:")
                for key, value in record.items():
                    if isinstance(value, str) and ('video' in key.lower() or 'path' in key.lower()):
                        print(f"  {key}: {value}")
            
            print(f"\nSearching for filename pattern in all records...")
            partial_matches = []
            for i, record in enumerate(data):
                for key, value in record.items():
                    if isinstance(value, str) and "ffb79099d17fbb6279a23015da9c0cc4" in value:
                        partial_matches.append((i, key, value))
            
            if partial_matches:
                print(f"Found {len(partial_matches)} records with similar hash:")
                for i, key, value in partial_matches[:5]:  # Show first 5 matches
                    print(f"  Record {i}, {key}: {value}")
    
    except FileNotFoundError:
        print(f"Error: File not found - {json_file}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_video_caption() 