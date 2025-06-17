#!/usr/bin/env python3
import os
import tarfile
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

def extract_tar_gz(tar_path, extract_to=None):
    """Extract a single tar.gz file"""
    try:
        if extract_to is None:
            extract_to = os.path.dirname(tar_path)
        
        print(f"Extracting {os.path.basename(tar_path)}...")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        
        print(f"✓ Finished extracting {os.path.basename(tar_path)}")
        return True
        
    except Exception as e:
        print(f"✗ Error extracting {os.path.basename(tar_path)}: {e}")
        return False

def process_directory(directory_path, max_workers=4):
    """Process all tar.gz files in a directory using multi-threading"""
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist, skipping.")
        return 0, 0
    
    # Find all tar.gz files in the directory
    tar_files = glob.glob(os.path.join(directory_path, "*.tar.gz"))
    
    if not tar_files:
        print(f"No tar.gz files found in {directory_path}")
        return 0, 0
    
    print(f"Found {len(tar_files)} tar.gz files in {directory_path}")
    
    # Extract files in parallel
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all extraction tasks
        future_to_file = {executor.submit(extract_tar_gz, tar_file): tar_file 
                         for tar_file in tar_files}
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            tar_file = future_to_file[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(tar_file)}: {e}")
    
    print(f"Completed processing directory: {directory_path} ({success_count}/{len(tar_files)} successful)")
    return success_count, len(tar_files)

def main():
    """Main function to process all directories"""
    # Change to Senorita directory
    base_dir = "/scratch3/yan204/yxp/Senorita"
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} directory does not exist!")
        return
    
    os.chdir(base_dir)
    print(f"Working in directory: {os.getcwd()}")
    
    # Define all directories to process
    directories = [
        "local_style_transfer_upload",
        "outpainting", 
        "controlable_videos_upload",
        "obj_removal_videos_upload",
        "style_transfer_upload",
        "grounding_upload",
        "obj_removal_videos_upload2",
        "inpainting_upload",
        "obj_swap_upload"
    ]
    
    print(f"Processing {len(directories)} directories...")
    print("=" * 60)
    
    start_time = time.time()
    total_success = 0
    total_files = 0
    
    # Process directories in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all directory processing tasks
        future_to_dir = {executor.submit(process_directory, directory, 4): directory 
                        for directory in directories}
        
        # Process completed directory tasks
        for future in as_completed(future_to_dir):
            directory = future_to_dir[future]
            try:
                success, total = future.result()
                total_success += success
                total_files += total
                print(f"✓ Completed {directory}: {success}/{total} files")
            except Exception as e:
                print(f"✗ Error processing directory {directory}: {e}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("=" * 60)
    print(f"Extraction completed!")
    print(f"Total files processed: {total_success}/{total_files}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    if total_files > 0:
        print(f"Success rate: {(total_success/total_files)*100:.1f}%")

if __name__ == "__main__":
    main() 