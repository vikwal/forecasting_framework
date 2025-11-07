#!/bin/bash
# Define list of 5-digit strings for config names
#configs=("00164" "03362" "03631" "07370")
model="tft"
start_from_gpu=0

# Use array index directly
for i in {1..3}; do
    config_suffix="${configs[$i]:-100}"
    gpu_id=$((i+start_from_gpu))
    screen -dmS ${model}_$i bash -c "
        cd ~/Work/forecasting_framework
        source frcst/bin/activate
        CUDA_VISIBLE_DEVICES=$gpu_id python hpo_cl.py -m $model -c config_wind_${config_suffix} -i $i
    "
    echo "Started screen session ${model}_$i on GPU $gpu_id with config config_wind_${config_suffix}"
    sleep 3
done