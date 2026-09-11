#!/usr/bin/env bash
# Ein Retrain-Lauf der raeumlichen 3-fach-CV mit den HPO-besten Hyperparametern.
#
#   run_retrain.sh <ARM> <FOLD> <GPU>
# z. B. run_retrain.sh dcrnn 1 1   -> configs/dcrnn/config_wind_dcrnn_fold1.yaml
#
# Die Hyperparameter kommen ueber --hpo-study auto direkt aus Optuna (bester
# Trial nach gepooltem Val-RMSE); es werden keine Configs materialisiert.
set -euo pipefail

ARM="$1"; FOLD="$2"; GPU="$3"
REPO=/home/viktor/Work/forecasting_framework
CFG="configs/dcrnn/config_wind_${ARM}_fold${FOLD}.yaml"
TAG="${ARM}_fold${FOLD}"
LOG="logs/retrain_${TAG}.log"

cd "$REPO"
[ -f "$CFG" ] || { echo "Config fehlt: $CFG"; exit 1; }
source frcst/bin/activate
eval "$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE|DATA_ROOT)=' ~/.bashrc)"
: "${WEATHER_DB_URL:?WEATHER_DB_URL fehlt}"
: "${ECMWF_WIND_SL_URL:?ECMWF_WIND_SL_URL fehlt}"
: "${OPTUNA_STORAGE:?OPTUNA_STORAGE fehlt}"
: "${DATA_ROOT:?DATA_ROOT fehlt}"

{
  echo "== $(date -Is) START retrain $TAG"
  echo "== host=$(hostname) gpu=$GPU commit=$(git rev-parse --short HEAD)"
  echo "== config=$CFG"
} >> "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" python geostatistics/train_dcrnn.py \
    --config "$CFG" --hpo-study auto --suffix "retrain_fold${FOLD}" >> "$LOG" 2>&1

echo "== $(date -Is) ENDE retrain $TAG exit=$?" >> "$LOG"
