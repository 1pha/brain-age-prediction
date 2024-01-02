#!/bin/bash

# Assuming the file name is input.txt
# file="assets/dkt_indices.txt"
read -p "Enter the file name: " file
read -p "Enter devices: " device

export CUDA_VISIBLE_DEVICES=$device

# Check if the file exists
if [ -e "$file" ]; then
    # Read the file line by line and process each line
    while IFS= read -r mask_idx; do
        # Use the line as a variable (you can replace this with your own logic)
        echo "Processing line: $mask_idx"
        python train.py --config-name=train_mask.yaml dataset.mask_idx=$mask_idx
        # Add your logic here using $line as the variable
        
    done < "$file"
else
    echo "File $file not found."
fi
