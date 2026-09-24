#!/usr/bin/env bash
# launch_altab.sh — A/B der Hoehendifferenz-Normierung, Validierungs-Fold 1.
#
# Vier Screens auf den vier A100 von l2:
#   GPU 0  DCRNN grid, /500  Clip 3   (neu)
#   GPU 1  DCRNN IDW,  /500  Clip 3   (neu)
#   GPU 2  DCRNN grid, /3000 Clip 1   (Kontrolle, alte Konvention)
#   GPU 3  DCRNN IDW,  /3000 Clip 1   (Kontrolle)
#
# Die Kontrolle laeuft am selben Commit, weil seit dem 17.08. achtzehn Commits
# den DCRNN-Pfad angefasst haben. Hyperparameter kommen ueber --hpo-study auto
# aus derselben Optuna-Studie wie die veroeffentlichten Laeufe.
# Nichts wird ueberschrieben: eigener Suffix, eigener raw-out-name.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO=$PWD
eval "$(grep -E '^export (OPTUNA_STORAGE|DATA_ROOT|ECMWF_WIND_SL_URL|WEATHER_DB_URL)=' ~/.bashrc)"
: "${OPTUNA_STORAGE:?OPTUNA_STORAGE fehlt}"; : "${DATA_ROOT:?DATA_ROOT fehlt}"
: "${WEATHER_DB_URL:?WEATHER_DB_URL fehlt}"; : "${ECMWF_WIND_SL_URL:?ECMWF_WIND_SL_URL fehlt}"
export OPTUNA_STORAGE DATA_ROOT ECMWF_WIND_SL_URL WEATHER_DB_URL
mkdir -p logs
FOLD=1

start () {
  local ARM=$1 TAG=$2 GPU=$3 NORM=$4 CLIP=$5
  local CFG="configs/dcrnn/config_wind_${ARM}_fold${FOLD}.yaml"
  local SUF="${TAG}_f${FOLD}"
  local STEM="wind_${ARM}_fold${FOLD}_dcrnn_${SUF}"
  local LOG="logs/altab_${ARM}_${TAG}.out"
  local NAME="altab_${ARM}_${TAG}"
  [ -f "$CFG" ] || { echo "Config fehlt: $CFG"; return 1; }
  screen -dmS "$NAME" bash -c "
    cd '$REPO' && source frcst/bin/activate
    export CUDA_VISIBLE_DEVICES=$GPU ALT_DIFF_NORM_M=$NORM ALT_DIFF_CLIP=$CLIP
    echo \"START \$(date) arm=$ARM norm=$NORM clip=$CLIP gpu=$GPU commit=\$(git rev-parse --short HEAD)\" > '$LOG'
    python geostatistics/train_dcrnn.py --config '$CFG' --hpo-study auto --suffix '$SUF' >> '$LOG' 2>&1
    rc=\$?
    if [ \$rc -ne 0 ]; then echo \"EXIT_TRAIN \$rc \$(date)\" >> '$LOG'; exit \$rc; fi
    PKL=\$(ls -t results/${STEM}_*.pkl 2>/dev/null | head -1)
    echo \"PKL=\$PKL\" >> '$LOG'
    if [ -z \"\$PKL\" ]; then echo \"EXIT_NOPKL \$(date)\" >> '$LOG'; exit 1; fi
    python geostatistics/get_test_results_dcrnn.py -m '$STEM' -c '$CFG' \
        --pkl \"\$PKL\" --raw-out-name 'altab_${ARM}_${TAG}_fold${FOLD}' >> '$LOG' 2>&1
    echo \"EXIT \$? \$(date)\" >> '$LOG'"
  echo "gestartet: $NAME  gpu=$GPU  norm=$NORM clip=$CLIP  stem=$STEM"
}

start dcrnn         alt500  0 500  3
start dcrnn_idw_alt alt500  1 500  3
start dcrnn         alt3000 2 3000 1
start dcrnn_idw_alt alt3000 3 3000 1
sleep 3
screen -ls | grep altab || true
