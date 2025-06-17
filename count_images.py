#!/usr/bin/env python3
import os
import glob
from pathlib import Path
from collections import defaultdict

def count_images(directory_path):
    """Count all image files recursively in the specified directory"""
    
    # Define common image extensions
    image_extensions = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', 
        '.webp', '.svg', '.ico', '.raw', '.cr2', '.nef', '.arw'
    }
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist!")
        return
    
    if not os.path.isdir(directory_path):
        print(f"{directory_path} is not a directory!")
        return
    
    print(f"Scanning directory: {directory_path}")
    print("=" * 50)
    
    # Statistics
    total_images = 0
    extension_counts = defaultdict(int)
    file_sizes = defaultdict(int)
    subdir_counts = defaultdict(int)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Check if it's an image file
            if file_ext in image_extensions:
                total_images += 1
                extension_counts[file_ext] += 1
                
                # Get file size
                try:
                    file_size = os.path.getsize(file_path)
                    file_sizes[file_ext] += file_size
                except OSError:
                    pass
                
                # Count by subdirectory
                rel_path = os.path.relpath(root, directory_path)
                if rel_path == '.':
                    subdir_counts['root'] += 1
                else:
                    subdir_counts[rel_path] += 1
    
    # Display results
    print(f"Total image files found: {total_images}")
    print()
    
    if total_images > 0:
        # Show breakdown by extension
        print("Breakdown by file extension:")
        print("-" * 30)
        for ext, count in sorted(extension_counts.items(), key=lambda x: x[1], reverse=True):
            size_mb = file_sizes[ext] / (1024 * 1024)
            print(f"{ext:8} : {count:6} files ({size_mb:8.2f} MB)")
        
        print()
        
        # Show breakdown by subdirectory (top 10)
        print("Top 10 subdirectories by image count:")
        print("-" * 40)
        sorted_subdirs = sorted(subdir_counts.items(), key=lambda x: x[1], reverse=True)
        for subdir, count in sorted_subdirs[:10]:
            print(f"{subdir:35} : {count:6} images")
        
        if len(sorted_subdirs) > 10:
            print(f"... and {len(sorted_subdirs) - 10} more subdirectories")
        
        print()
        
        # Total size
        total_size = sum(file_sizes.values())
        total_size_mb = total_size / (1024 * 1024)
        total_size_gb = total_size_mb / 1024
        print(f"Total size: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")
        
        # Average file size
        avg_size = total_size / total_images if total_images > 0 else 0
        avg_size_kb = avg_size / 1024
        print(f"Average file size: {avg_size_kb:.2f} KB")
        
    else:
        print("No image files found in the specified directory.")

def main():
    import sys
    
    if len(sys.argv) > 1:
        # Use command line argument
        directory = sys.argv[1]
    else:
        # Ask user for directory
        directory = input("Enter directory path to scan: ").strip()
    
    # Remove trailing slash if present
    directory = directory.rstrip('/')
    
    count_images(directory)

if __name__ == "__main__":
    main() 