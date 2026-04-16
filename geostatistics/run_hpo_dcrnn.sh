#!/bin/bash
# run_hpo_dcrnn.sh — Start parallel HPO workers for the DCRNN in screen sessions.
#
# Each worker runs hpo_dcrnn.py against the SAME Optuna study (PostgreSQL),
# contributing trials in parallel.  Workers are distributed across available
# GPUs in round-robin order.
#
# Usage
# -----
#   bash geostatistics/run_hpo_dcrnn.sh [OPTIONS]
#
# Options
#   -c CONFIG      Path to YAML config  (default: configs/config_wind_stgcn.yaml)
#   -n N_WORKERS   Number of parallel workers  (default: number of available GPUs)
#   -g GPU_LIST    Comma-separated GPU indices to use  (default: all, e.g. "0,1,2")
#   -s SUFFIX      Study name suffix  (default: none)
#   -h             Show this help
#
# Examples
#   bash geostatistics/run_hpo_dcrnn.sh
#   bash geostatistics/run_hpo_dcrnn.sh -n 4 -g "0,1,2,3" -s run1
#   bash geostatistics/run_hpo_dcrnn.sh -c configs/config_wind_stgcn.yaml -n 2 -g "1,2"

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
VENV_PATH="frcst"
CONFIG="configs/config_wind_stgcn.yaml"
SUFFIX=""
N_WORKERS=""
GPU_LIST=""

# ── Parse args ────────────────────────────────────────────────────────────────
while getopts "c:n:g:s:h" opt; do
    case $opt in
        c) CONFIG="$OPTARG" ;;
        n) N_WORKERS="$OPTARG" ;;
        g) GPU_LIST="$OPTARG" ;;
        s) SUFFIX="$OPTARG" ;;
        h)
            sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: -$OPTARG"; exit 1 ;;
    esac
done

# ── Resolve repo root (script lives in geostatistics/) ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi
source "$VENV_PATH/bin/activate"

# ── Detect available GPUs ─────────────────────────────────────────────────────
if [ -n "$GPU_LIST" ]; then
    IFS=',' read -ra GPUS <<< "$GPU_LIST"
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || echo "0")
fi

N_GPUS=${#GPUS[@]}
if [ -z "$N_WORKERS" ]; then
    N_WORKERS=$N_GPUS
fi

echo "========================================================"
echo "  HPO DCRNN — parallel workers"
echo "  Config   : $CONFIG"
echo "  Study    : same for all workers (PostgreSQL)"
echo "  Workers  : $N_WORKERS"
echo "  GPUs     : ${GPUS[*]}"
echo "  Suffix   : ${SUFFIX:-<none>}"
echo "========================================================"

# ── Build suffix flag ─────────────────────────────────────────────────────────
SUFFIX_FLAG=""
if [ -n "$SUFFIX" ]; then
    SUFFIX_FLAG="--suffix $SUFFIX"
fi

# ── Preprocess / cache step (runs once on GPU 0, synchronously) ──────────────
echo ""
echo "Step 1: Checking / building GNN data cache ..."
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python geostatistics/hpo_dcrnn.py \
    --config "$CONFIG" \
    $SUFFIX_FLAG \
    --preprocess-only

if [ $? -ne 0 ]; then
    echo "ERROR: Cache build step failed. Aborting."
    exit 1
fi
echo "Cache ready."
echo ""

# ── Launch one screen session per worker ──────────────────────────────────────
for ((i=0; i<N_WORKERS; i++)); do
    GPU_IDX=${GPUS[$((i % N_GPUS))]}
    WORKER_SUFFIX="${SUFFIX:+${SUFFIX}_}worker${i}"
    SESSION_NAME="hpo_dcrnn_${WORKER_SUFFIX}"

    CMD="cd '$REPO_ROOT' && source '$VENV_PATH/bin/activate' && \
CUDA_VISIBLE_DEVICES=$GPU_IDX python geostatistics/hpo_dcrnn.py \
    --config '$CONFIG' \
    $SUFFIX_FLAG; \
echo 'Worker $i finished (exit \$?). Press any key to close.'; read -n1"

    echo "  Launching screen '$SESSION_NAME' on GPU $GPU_IDX ..."
    screen -dmS "$SESSION_NAME" bash -c "$CMD"
done

echo ""
echo "All $N_WORKERS workers launched."
echo ""
echo "Monitor:"
echo "  screen -ls                        # list sessions"
echo "  screen -r hpo_dcrnn_${SUFFIX:+${SUFFIX}_}worker0   # attach to worker 0"
echo "  tail -f logs/hpo_dcrnn_*.log      # follow all logs"
echo ""
echo "Stop all:"
echo "  screen -ls | grep hpo_dcrnn | awk '{print \$1}' | xargs -I{} screen -X -S {} quit"
