#!/usr/bin/env bash
# run_hpo_study.sh — Starts HPO screen sessions for 4 model variants.
#
# Usage: ./run_hpo_study.sh [N_PER_EXPERIMENT]
#
#   N=1 →  4 sessions  GPU 0–3  (one per experiment)
#   N=2 →  8 sessions  GPU 0–7  (two per experiment, default)
#   N=3 → 12 sessions  GPU 0–7 then 0–3  (three per experiment)
#
# Sessions are named {prefix}_{i}, e.g. hpo_dcrnn_1, hpo_dcrnn_2.
# Each session logs to logs/{session}.log.
# Multiple workers for the same experiment share one Optuna study
# (SQLite WAL-mode or PostgreSQL if OPTUNA_STORAGE is set).
set -euo pipefail

N=${1:-2}
N_GPUS=8
GEO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # …/geostatistics
PROJECT_DIR="$(dirname "${GEO_DIR}")"                      # …/forecasting_framework
VENV="${PROJECT_DIR}/frcst/bin/activate"

# Each entry: "session_prefix  script  config"
# Scripts are relative to geostatistics/, configs relative to PROJECT_DIR.
EXPERIMENTS=(
    "hpo_dcrnn      geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn_base.yaml"
    "hpo_dcrnn_nwp  geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn.yaml"
    "hpo_mtgnn      geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn.yaml"
    "hpo_wavenet    geostatistics/hpo_wavenet.py  configs/wavenet/config_wind_wavenet.yaml"
)

mkdir -p "${PROJECT_DIR}/logs"

gpu_idx=0
total=0

for exp in "${EXPERIMENTS[@]}"; do
    read -r prefix script config <<< "$exp"

    for i in $(seq 1 "$N"); do
        gpu=$(( gpu_idx % N_GPUS ))
        session="${prefix}_${i}"
        log="${PROJECT_DIR}/logs/${session}.log"

        screen -dmS "$session" bash -c "
            source '${VENV}'
            cd '${PROJECT_DIR}'
            python '${script}' \
                --config '${config}' \
                --gpu ${gpu} \
                --suffix gpu${gpu} \
                2>&1 | tee '${log}'
            exec bash
        "

        echo "Started  screen '${session}'  GPU ${gpu}  log → ${log}"
        gpu_idx=$(( gpu_idx + 1 ))
        total=$(( total + 1 ))
    done
done

echo ""
echo "Started ${total} sessions (${#EXPERIMENTS[@]} experiments × N=${N})."
echo "Overview : screen -ls | grep hpo"
echo "Attach   : screen -r <session-name>"
echo "Logs     : tail -f ${PROJECT_DIR}/logs/<session-name>.log"
