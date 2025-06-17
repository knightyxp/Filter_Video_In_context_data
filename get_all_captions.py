#!/usr/bin/env python3
import json
import sys

def find_all_captions():
    # 目标视频文件名
    target_hash = "ffb79099d17fbb6279a23015da9c0cc4_org.mp4"
    
    # JSON文件路径
    json_file = "/projects/D2DCRC/xiangpeng/Senorita/filtered_obj_addition_592x336.json"
    
    print(f"Looking for video containing: {target_hash}")
    print(f"JSON file: {json_file}")
    print("="*80)
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total records in JSON: {len(data)}")
        
        # Find matching records
        found_records = []
        for i, record in enumerate(data):
            # Check all string fields for the target hash
            for key, value in record.items():
                if isinstance(value, str) and target_hash in value:
                    found_records.append((i, record))
                    break
        
        if found_records:
            print(f"Found {len(found_records)} matching record(s)")
            print("="*80)
            
            for idx, (record_idx, record) in enumerate(found_records):
                print(f"\n--- Record {idx + 1} (Index: {record_idx}) ---")
                
                # Print ALL fields in the record
                print("ALL FIELDS:")
                for key, value in record.items():
                    print(f"  {key}: {value}")
                
                print("-" * 60)
                
                # Also highlight caption-related fields specifically
                caption_fields = []
                for key, value in record.items():
                    key_lower = key.lower()
                    if any(word in key_lower for word in ['caption', 'description', 'instruction', 'prompt', 'text', 'label', 'enhanced']):
                        caption_fields.append((key, value))
                
                if caption_fields:
                    print("CAPTION-RELATED FIELDS:")
                    for key, value in caption_fields:
                        print(f"  {key}: {value}")
                
                print("="*80)
        else:
            print("No matching records found!")
            
            # Try partial matching
            print("Trying partial hash matching...")
            partial_hash = "ffb79099d17fbb6279a23015da9c0cc4"
            
            partial_matches = []
            for i, record in enumerate(data):
                for key, value in record.items():
                    if isinstance(value, str) and partial_hash in value:
                        partial_matches.append((i, key, value, record))
                        break
            
            if partial_matches:
                print(f"Found {len(partial_matches)} records with partial hash match:")
                for i, key, value, record in partial_matches:
                    print(f"\nRecord {i}:")
                    print(f"  Matched field: {key}")
                    print(f"  Matched value: {value}")
                    
                    # Print all caption fields for this record
                    print("  All caption fields:")
                    for field_key, field_value in record.items():
                        field_lower = field_key.lower()
                        if any(word in field_lower for word in ['caption', 'description', 'instruction', 'prompt', 'text', 'label', 'enhanced']):
                            print(f"    {field_key}: {field_value}")
                    print("-" * 40)
    
    except FileNotFoundError:
        print(f"Error: File not found - {json_file}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_all_captions() 