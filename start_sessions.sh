#!/bin/bash
# Usage: ./start_sessions.sh <script_type> <config_dir> [model] [start_from_gpu]
# script_type: hpo_cl, hpo_fl, train_cl, train_fl
N_GPUS=8  # <-- hier anpassen

script_type="${1:?Usage: $0 <script_type> <config_dir> [model] [start_from_gpu]}"
config_dir="${2:?Usage: $0 <script_type> <config_dir> [model] [start_from_gpu]}"
model="${3:-tft}"
start_from_gpu="${4:-0}"
n_gpus=$N_GPUS

# Validate script_type
case "$script_type" in
    hpo_cl|hpo_fl|train_cl|train_fl)
        ;;
    *)
        echo "Error: Invalid script_type '$script_type'"
        echo "Must be one of: hpo_cl, hpo_fl, train_cl, train_fl"
        exit 1
        ;;
esac

# Determine Python script and arguments
python_script="${script_type}.py"
base_args="-m $model -c"

# Add --save_model for training scripts
if [[ "$script_type" == train_* ]]; then
    extra_args="--save_model"
else
    extra_args=""
fi

# Collect all yaml configs from the directory, sorted
mapfile -t configs < <(ls "$config_dir"/*.yaml 2>/dev/null | sort)

if [ ${#configs[@]} -eq 0 ]; then
    echo "No .yaml configs found in '$config_dir'"
    exit 1
fi

echo "Starting $script_type sessions with model $model on GPUs $start_from_gpu-$((start_from_gpu + n_gpus - 1))"

for i in "${!configs[@]}"; do
    config_path="${configs[$i]}"
    config_name="$(basename "${config_path}" .yaml)"
    config_arg="${config_path%.yaml}"
    gpu_id=$(( (i % n_gpus) + start_from_gpu ))
    session="${script_type}_${model}_$((i+1))"

    screen -dmS "$session" bash -c "
        cd ~/Work/forecasting_framework
        source frcst/bin/activate
        echo 'Starting $script_type for $session on GPU $gpu_id...'
        CUDA_VISIBLE_DEVICES=$gpu_id python $python_script $base_args $config_arg $extra_args
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
    sleep 60
done