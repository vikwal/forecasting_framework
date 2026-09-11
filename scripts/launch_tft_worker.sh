#!/usr/bin/env bash
# Startet EINEN TFT-BC-HPO-Worker. Gleiche Credential-Behandlung wie
# scripts/launch_dcrnn_worker.sh, siehe dort.
#
#   scripts/launch_tft_worker.sh <REPO> <CONFIG_OHNE_YAML> <GPU> <LOGNAME>
#
# Beispiel:
#   screen -dmS hpo_tft_sp_base_g0 bash scripts/launch_tft_worker.sh \
#       "$PWD" configs/tft_bc/config_wind_tft_sp_base 0 hpo_tft_sp_base_g0
REPO="$1"; CFG="$2"; GPU="$3"; NAME="$4"
cd "$REPO" || { echo "REPO nicht gefunden: $REPO"; exec bash; }
source frcst/bin/activate
eval "$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE|DATA_ROOT)=' ~/.bashrc)"
: "${WEATHER_DB_URL:?WEATHER_DB_URL fehlt}"
: "${ECMWF_WIND_SL_URL:?ECMWF_WIND_SL_URL fehlt}"
: "${OPTUNA_STORAGE:?OPTUNA_STORAGE fehlt}"
: "${DATA_ROOT:?DATA_ROOT fehlt}"
mkdir -p logs/hpo_tft_bc
echo "== $(date -Is) $CFG gpu=$GPU log=$NAME"
python hpo_tft_bc.py -c "$CFG" --gpu "$GPU" --max-cache-gb 1200 >> "logs/hpo_tft_bc/$NAME.log" 2>&1
echo "== Worker beendet, Exit $?"
exec bash
