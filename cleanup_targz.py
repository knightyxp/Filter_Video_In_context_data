#!/usr/bin/env python3
import os
import glob
from pathlib import Path

def cleanup_targz_files():
    """Delete all tar.gz files in the /Senorita directory"""
    
    # Define the target directory
    senorita_dir = "/scratch3/yan204/yxp/Senorita"
    
    # Check if directory exists
    if not os.path.exists(senorita_dir):
        print(f"Directory {senorita_dir} does not exist!")
        return
    
    # Find all tar.gz files recursively
    pattern = os.path.join(senorita_dir, "**/*.tar.gz")
    tar_gz_files = glob.glob(pattern, recursive=True)
    
    if not tar_gz_files:
        print("No tar.gz files found in /Senorita directory")
        return
    
    print(f"Found {len(tar_gz_files)} tar.gz files to delete:")
    
    # List all files that will be deleted
    for file_path in tar_gz_files:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"  - {file_path} ({file_size:.2f} MB)")
    
    # Ask for confirmation
    response = input(f"\nDo you want to delete these {len(tar_gz_files)} files? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        deleted_count = 0
        total_size_freed = 0
        
        for file_path in tar_gz_files:
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                deleted_count += 1
                total_size_freed += file_size
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        total_size_mb = total_size_freed / (1024 * 1024)
        print(f"\nCleanup completed!")
        print(f"Deleted {deleted_count} files")
        print(f"Freed {total_size_mb:.2f} MB of disk space")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    cleanup_targz_files() 