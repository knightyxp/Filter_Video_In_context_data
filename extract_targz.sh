#!/bin/bash

cd Senorita

for dir in local_style_transfer_upload outpainting style_transfer_upload; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        cd "$dir"
        for tarfile in *.tar.gz; do
            if [ -f "$tarfile" ]; then
                echo "Extracting $tarfile..."
                tar -xzf "$tarfile"
                echo "Finished extracting $tarfile"
            fi
        done
        cd ..
    else
        echo "Directory $dir does not exist, skipping."
    fi
done

echo "Selected tar.gz files have been extracted!"