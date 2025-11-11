#!/bin/bash
# Define list of 5-digit strings for config names
#configs=("00164" "03362" "03631" "07370")
configs=("03631" "07370")
model="tcn-gru"
start_from_gpu=0

# Use array index directly
for i in {0..1}; do
    config_suffix="${configs[$i]:-100}"
    gpu_id=$((i+start_from_gpu))
    screen -dmS ${model}_$i bash -c "
        cd ~/Work/forecasting_framework
        source frcst/bin/activate
        echo 'Starting HPO for ${model}_$i on GPU $gpu_id...'
        CUDA_VISIBLE_DEVICES=$gpu_id python hpo_cl.py -m $model -c config_wind_${config_suffix} -i $i
        exit_code=\$?
        echo '================================================'
        if [ \$exit_code -eq 0 ]; then
            echo 'Script completed successfully!'
        else
            echo 'Script failed with exit code: '\$exit_code
        fi
        echo 'Session ${model}_$i finished. Press any key to close...'
        read -n 1
        exec bash
    "
    echo "Started screen session ${model}_$i on GPU $gpu_id with config config_wind_${config_suffix}"
    sleep 300
done