#!/bin/bash
# Usage: ./eval_sessions.sh <config_dir> [start_from_gpu]
N_GPUS=8  # <-- hier anpassen

config_dir="${1:?Usage: $0 <config_dir> [start_from_gpu]}"
start_from_gpu="${2:-0}"
n_gpus=$N_GPUS

# Collect all yaml configs from the directory, sorted
mapfile -t configs < <(ls "$config_dir"/*.yaml 2>/dev/null | sort)

if [ ${#configs[@]} -eq 0 ]; then
    echo "No .yaml configs found in '$config_dir'"
    exit 1
fi

echo "Starting evaluation sessions on GPUs $start_from_gpu-$((start_from_gpu + n_gpus - 1))"

for i in "${!configs[@]}"; do
    config_path="${configs[$i]}"
    config_name="$(basename "${config_path}" .yaml)"
    
    # Extract model name from config filename (e.g., config_wind_80cl.yaml -> 80cl)
    model_name=""
    if [[ "$config_name" =~ config_wind_(.+)$ ]]; then
        model_name="${BASH_REMATCH[1]}"
    fi
    
    gpu_id=$(( (i % n_gpus) + start_from_gpu ))
    session="eval_$((i+1))"

    screen -dmS "$session" bash -c "
        cd ~/Work/forecasting_framework
        source frcst/bin/activate
        echo 'Starting evaluation for $session on GPU $gpu_id...'
        echo 'Config: $config_name, Model name filter: $model_name'
        CUDA_VISIBLE_DEVICES=$gpu_id python get_test_results.py --config $config_path -e --model-name $model_name
        exit_code=\$?
        echo '================================================'
        if [ \$exit_code -eq 0 ]; then
            echo 'Evaluation completed successfully!'
        else
            echo 'Evaluation failed with exit code: '\$exit_code
        fi
        echo 'Session $session finished. Press any key to close...'
        read -n 1
        exec bash
    "
    echo "Started screen session $session on GPU $gpu_id with config $config_name"
    sleep 60
done
