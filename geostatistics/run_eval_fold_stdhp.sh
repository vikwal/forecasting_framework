#!/bin/bash
# Usage: run_eval_fold.sh <fold_idx> <config_path>
set -euo pipefail
cd /home/viktorwalter/Work/forecasting_framework
source frcst/bin/activate
eval "$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE)=' ~/.bashrc)"
FOLD_IDX="$1"
CONFIG="$2"
mkdir -p logs
python geostatistics/evaluate_reference.py -c "$CONFIG" --fold-idx "$FOLD_IDX" \
    > "logs/evaluate_reference_valmode_fold${FOLD_IDX}.log" 2>&1
echo "EXIT_CODE=$? fold=${FOLD_IDX}" >> "logs/evaluate_reference_valmode_fold${FOLD_IDX}.log"
