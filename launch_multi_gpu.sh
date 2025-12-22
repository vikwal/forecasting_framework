#!/bin/bash
# Multi-GPU training launcher with caching.
# This script starts multiple HPO processes on different GPUs with shared cached data.
#
# Usage:
#   ./launch_multi_gpu.sh <config.yaml> <model_name> [options]
#
# Example:
#   ./launch_multi_gpu.sh configs/my_config.yaml fnn --gpus 0,1,2,3 --trials-per-gpu 10

set -e  # Exit on any error

# Set TensorFlow environment variables to reduce logging
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

# Default values
GPUS="0"  # GPUs 1-7 (excluding 0 which might be in use)
TRIALS_PER_GPU=1000
FORCE_PREPROCESS=false
CACHE_DIR="data_cache"
USE_CACHE=true  # New: Enable/disable caching

# Virtual environment setup
VENV_PATH="./frcst"  # Path to your virtual environment
PYTHON_CMD="python3"

# Check if virtual environment exists and activate it
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    PYTHON_CMD="python"  # Use python from venv (not python3)
else
    echo "Warning: Virtual environment '$VENV_PATH' not found, using system Python"
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 <config.yaml> <model_name> [options]"
    echo ""
    echo "Required arguments:"
    echo "  config.yaml       Configuration file path"
    echo "  model_name        Model name (fnn, lstm, tft, etc.)"
    echo ""
    echo "Optional arguments:"
    echo "  --gpus LIST       Comma-separated GPU IDs (default: 1,2,3,4,5,6,7)"
    echo "  --trials-per-gpu N   Trials per GPU (default: 10)"
    echo "  --force-preprocess   Force preprocessing even if cache exists"
    echo "  --no-cache        Disable caching (for small datasets)"
    echo "  --cache-dir DIR   Cache directory (default: data_cache)"
    echo "  --venv PATH       Path to virtual environment (default: ./frcst)"
    echo "  --help            Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 configs/wind_config.yaml fnn --gpus 1,2,3 --trials-per-gpu 20"
    echo "  $0 configs/small_config.yaml fnn --no-cache  # For small datasets"
}

# Parse command line arguments
if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

CONFIG_FILE="$1"
MODEL_NAME="$2"
shift 2

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --trials-per-gpu)
            TRIALS_PER_GPU="$2"
            shift 2
            ;;
        --force-preprocess)
            FORCE_PREPROCESS=true
            shift
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required files
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Convert comma-separated GPU list to array
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=================================================="
echo "Multi-GPU HPO Training Launcher"
echo "=================================================="
echo "Config file: $CONFIG_FILE"
echo "Model: $MODEL_NAME"
echo "GPUs: ${GPU_ARRAY[*]} ($NUM_GPUS total)"
echo "Trials per GPU: $TRIALS_PER_GPU"
echo "Cache directory: $CACHE_DIR"
echo "Use caching: $USE_CACHE"
echo "Virtual environment: $VENV_PATH"
echo "Python command: $PYTHON_CMD"
echo "Force preprocessing: $FORCE_PREPROCESS"
echo "=================================================="

# Check if preprocessing is needed
if [ "$USE_CACHE" = false ]; then
    echo "Caching disabled, skipping cache check"
    PREPROCESS_NEEDED="disabled"
elif [ "$FORCE_PREPROCESS" = true ]; then
    echo "Forcing preprocessing due to --force-preprocess flag"
    PREPROCESS_NEEDED=true
else
    # Check if cache exists for this config
    echo "Checking if preprocessing is needed..."
    # We'll let Python handle this check
    PREPROCESS_NEEDED=$($PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from utils.data_cache import DataCache
from utils import tools

try:
    config = tools.load_config('$CONFIG_FILE')
    # Ensure model name is available for cache hash calculation
    config['model']['name'] = '$MODEL_NAME'
    from utils import preprocessing
    features = preprocessing.get_features(config=config)
    cache = DataCache('$CACHE_DIR')
    is_cached, cache_id = cache.is_cached(config, features, '$MODEL_NAME')
    print('false' if is_cached else 'true')
except Exception as e:
    print(f'Error checking cache: {e}', file=sys.stderr)
    print('true')  # Default to preprocessing needed
" 2>/dev/null)
fi

# Step 1: Preprocessing (if needed)
if [ "$PREPROCESS_NEEDED" = "true" ]; then
    echo ""
    echo "Step 1: Running preprocessing and caching data..."
    echo "This will be done once and shared across all GPU processes."

    # Run preprocessing on GPU 0 (or CPU) to create the cache
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" $PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from utils.data_cache import create_or_load_preprocessed_data
from utils import tools, preprocessing

print('Loading configuration...')
config = tools.load_config('$CONFIG_FILE')
# Ensure model name is set for preprocessing
config['model']['name'] = '$MODEL_NAME'
features = preprocessing.get_features(config=config)

print('Creating cached data...')
lazy_fold_loader, cache_id = create_or_load_preprocessed_data(
    config=config,
    features=features,
    model_name='$MODEL_NAME',
    force_reprocess=$([[ "$FORCE_PREPROCESS" == "true" ]] && echo "True" || echo "False"),
    use_cache=True
)

print(f'Cache created with ID: {cache_id}')
print(f'Available folds: {len(lazy_fold_loader)}')
"

    if [ $? -ne 0 ]; then
        echo "Error: Preprocessing failed!"
        exit 1
    fi

    echo "Preprocessing completed successfully!"
elif [ "$PREPROCESS_NEEDED" = "disabled" ]; then
    echo ""
    echo "Step 1: Caching disabled, data will be processed individually by each GPU"
else
    echo "Cache exists, skipping preprocessing."
fi

echo ""
echo "Step 2: Starting multi-GPU training processes..."

# Array to store process IDs
PIDS=()

# Launch training on each GPU
for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    INDEX="gpu${GPU_ID}"

    echo "Starting training on GPU $GPU_ID (index: $INDEX)..."

    # Set environment and start training in background
    # Prepare command arguments
    CMD_ARGS="--model $MODEL_NAME --config $CONFIG_FILE --index $INDEX"

    # Add no-cache flag if caching is disabled
    if [ "$USE_CACHE" = false ]; then
        CMD_ARGS="$CMD_ARGS --no-cache"
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON_CMD hpo_cl.py $CMD_ARGS 2>/dev/null &

    # Store the process ID
    PID=$!
    PIDS+=($PID)

    echo "  GPU $GPU_ID: Process $PID started"

    # Small delay between starts to avoid resource conflicts
    sleep 2
done

echo ""
echo "All training processes started!"
echo "Process IDs: ${PIDS[*]}"
echo ""
echo "Monitoring progress (Ctrl+C to stop monitoring, processes will continue):"

# Function to check if a process is running
is_process_running() {
    kill -0 "$1" 2>/dev/null
}

# Monitor processes
RUNNING_PROCESSES=("${PIDS[@]}")
while [ ${#RUNNING_PROCESSES[@]} -gt 0 ]; do
    sleep 30  # Check every 30 seconds

    NEW_RUNNING=()
    for pid in "${RUNNING_PROCESSES[@]}"; do
        if is_process_running $pid; then
            NEW_RUNNING+=($pid)
        else
            echo "Process $pid completed."
        fi
    done

    RUNNING_PROCESSES=("${NEW_RUNNING[@]}")

    if [ ${#RUNNING_PROCESSES[@]} -gt 0 ]; then
        echo "Still running: ${RUNNING_PROCESSES[*]} ($(date))"

        # Show quick progress summary
        echo "Recent log activity:"
        for i in "${!GPU_ARRAY[@]}"; do
            GPU_ID="${GPU_ARRAY[$i]}"
            INDEX="gpu${GPU_ID}"
            LOG_FILE="logs/hpo_cl_m-${MODEL_NAME}_c-${CONFIG_NAME}_${INDEX}.log"

            if [ -f "$LOG_FILE" ]; then
                LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null | cut -c1-100)
                echo "  GPU $GPU_ID: $LAST_LINE"
            fi
        done
        echo ""
    fi
done

echo ""
echo "=================================================="
echo "All training processes completed!"
echo "Check individual log files for detailed results:"
for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    INDEX="gpu${GPU_ID}"
    echo "  GPU $GPU_ID: logs/hpo_cl_m-${MODEL_NAME}_c-${CONFIG_NAME}_${INDEX}.log"
done
echo "=================================================="