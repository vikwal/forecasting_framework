#!/bin/bash

# Path to virtual environment
VENV_PATH="frcst"

# Activate virtual environment
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated: $VENV_PATH"
else
    echo "⚠ Warning: Virtual environment not found at $VENV_PATH"
    echo "Please update VENV_PATH in the script or create a virtual environment"
    exit 1
fi

# Directory containing the config files
CONFIG_DIR="configs/wind_100ex50"

# Progress tracking file
PROGRESS_FILE="train_progress_100ex50.txt"

# Counter for GPU assignment
gpu_id=4

# Create progress file if it doesn't exist
touch "$PROGRESS_FILE"

# Counter for tracking
total_configs=0
completed_configs=0
skipped_configs=0

# Iterate through all config files in the directory
for config_file in "$CONFIG_DIR"/config_wind_*.yaml; do
    # Extract the config filename
    config_path="$config_file"
    config_name=$(basename "$config_file")

    ((total_configs++))

    # Check if this config has already been completed
    if grep -Fxq "$config_name" "$PROGRESS_FILE"; then
        echo "[$total_configs] Skipping $config_name (already completed)"
        ((skipped_configs++))
        continue
    fi

    echo "[$total_configs] Starting training for $config_path on GPU $gpu_id"

    # Start the training process and wait for it to complete
    CUDA_VISIBLE_DEVICES=$gpu_id python train_cl.py -m tft -c "$config_path" --save_model

    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        # Mark this config as completed
        echo "$config_name" >> "$PROGRESS_FILE"
        ((completed_configs++))
        echo "✓ Completed training for $config_name"
    else
        echo "✗ Failed training for $config_name (will retry on next run)"
    fi

    echo "---"
done

echo ""
echo "======================================="
echo "All training processes finished!"
echo "Total configs: $total_configs"
echo "Completed in this run: $completed_configs"
echo "Skipped (already done): $skipped_configs"
echo "======================================="
echo "Progress file: $PROGRESS_FILE"
echo "You can monitor the processes with: ps aux | grep train_cl.py"
