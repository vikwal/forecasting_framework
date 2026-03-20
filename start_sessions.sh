#!/bin/bash
# Usage: ./start_sessions.sh <config_dir> [model] [start_from_gpu]
N_GPUS=8  # <-- hier anpassen

config_dir="${1:?Usage: $0 <config_dir> [model] [start_from_gpu]}"
model="${2:-tft}"
start_from_gpu="${3:-0}"
n_gpus=$N_GPUS

# Collect all yaml configs from the directory, sorted
mapfile -t configs < <(ls "$config_dir"/*.yaml 2>/dev/null | sort)

if [ ${#configs[@]} -eq 0 ]; then
    echo "No .yaml configs found in '$config_dir'"
    exit 1
fi

for i in "${!configs[@]}"; do
    config_path="${configs[$i]}"
    config_name="$(basename "${config_path}" .yaml)"
    config_arg="${config_path%.yaml}"
    gpu_id=$(( (i % n_gpus) + start_from_gpu ))
    session="${model}_$((i+1))"

    screen -dmS "$session" bash -c "
        cd ~/Work/forecasting_framework
        source frcst/bin/activate
        echo 'Starting HPO for $session on GPU $gpu_id...'
        CUDA_VISIBLE_DEVICES=$gpu_id python hpo_cl.py -m $model -c $config_arg
        exit_code=\$?
        echo '================================================'
        if [ \$exit_code -eq 0 ]; then
            echo 'Script completed successfully!'
        else
            echo 'Script failed with exit code: '\$exit_code
        fi
        echo 'Session $session finished. Press any key to close...'
        read -n 1
        exec bash
    "
    echo "Started screen session $session on GPU $gpu_id with config $config_name"
    sleep 600
done