#!/usr/bin/env bash
# launch_notopo.sh — DCRNN grid OHNE die neun Gelaendedeskriptoren (Arm A der
# Analyse "Was sind die statischen Merkmale wert?", Viktor 2026-10-26).
#
# --station-node-features none schaltet nur die neun Topo-Spalten der
# Stationsknoten ab; Breite, Laenge, Hoehe bleiben als Knotenmerkmal, die
# Hoehendifferenz zu den Gitterpunkten bleibt an den Kanten (use_altitude_diff).
#
# Drei Validierungsfolds parallel auf GPU 0..2, der Testlauf (--test-mode,
# 153 Stationen Training, 50 Teststationen) haengt in Screen 1 hinter Fold 1.
# Hyperparameter wie die veroeffentlichten Laeufe (--hpo-study auto).
# Nichts wird ueberschrieben: Suffix notopo, eigene raw-out-names.
#   Val:  data/test_results/notopo_dcrnn_fold{1,2,3}.csv
#   Test: data/test_results/testmode_dcrnn_notopo.csv
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO=$PWD
eval "$(grep -E "^export (OPTUNA_STORAGE|DATA_ROOT|ECMWF_WIND_SL_URL|WEATHER_DB_URL)=" ~/.bashrc)"
: "${OPTUNA_STORAGE:?OPTUNA_STORAGE fehlt}"; : "${DATA_ROOT:?DATA_ROOT fehlt}"
: "${WEATHER_DB_URL:?WEATHER_DB_URL fehlt}"; : "${ECMWF_WIND_SL_URL:?ECMWF_WIND_SL_URL fehlt}"
export OPTUNA_STORAGE DATA_ROOT ECMWF_WIND_SL_URL WEATHER_DB_URL
mkdir -p logs
NF=none

start_fold () {
  local FOLD=$1 GPU=$2 WITH_TEST=$3
  local CFG="configs/dcrnn/config_wind_dcrnn_fold${FOLD}.yaml"
  local SUF="notopo_f${FOLD}"
  local STEM="wind_dcrnn_fold${FOLD}_dcrnn_${SUF}"
  local LOG="logs/notopo_dcrnn_fold${FOLD}.out"
  local NAME="notopo_dcrnn_f${FOLD}"
  [ -f "$CFG" ] || { echo "Config fehlt: $CFG"; return 1; }
  local TESTPART=""
  if [ "$WITH_TEST" = "1" ]; then
    TESTPART="
    TCFG=configs/testmode/full/config_wind_dcrnn_fold1.yaml
    TLOG=logs/notopo_dcrnn_test.out
    echo \"START \$(date) test-mode gpu=$GPU commit=\$(git rev-parse --short HEAD)\" > \$TLOG
    python geostatistics/train_dcrnn.py --config \$TCFG --suffix tnotopo --hpo-study auto --test-mode --station-node-features $NF >> \$TLOG 2>&1
    rc=\$?
    if [ \$rc -ne 0 ]; then echo \"EXIT_TRAIN \$rc \$(date)\" >> \$TLOG; exit \$rc; fi
    python geostatistics/get_test_results_dcrnn.py -m wind_dcrnn_fold1_dcrnn_tnotopo -c \$TCFG --hpo-study auto --test-mode --station-node-features $NF --raw-out-name testmode_dcrnn_notopo >> \$TLOG 2>&1
    echo \"EXIT \$? \$(date)\" >> \$TLOG"
  fi
  screen -dmS "$NAME" bash -c "
    cd '$REPO' && source frcst/bin/activate
    export CUDA_VISIBLE_DEVICES=$GPU
    echo \"START \$(date) fold=$FOLD gpu=$GPU commit=\$(git rev-parse --short HEAD) station_node_features=$NF\" > '$LOG'
    python geostatistics/train_dcrnn.py --config '$CFG' --hpo-study auto --suffix '$SUF' --station-node-features $NF >> '$LOG' 2>&1
    rc=\$?
    if [ \$rc -ne 0 ]; then echo \"EXIT_TRAIN \$rc \$(date)\" >> '$LOG'; exit \$rc; fi
    PKL=\$(ls -t results/${STEM}_*.pkl 2>/dev/null | head -1)
    echo \"PKL=\$PKL\" >> '$LOG'
    if [ -z \"\$PKL\" ]; then echo \"EXIT_NOPKL \$(date)\" >> '$LOG'; exit 1; fi
    python geostatistics/get_test_results_dcrnn.py -m '$STEM' -c '$CFG' --pkl \"\$PKL\" --station-node-features $NF --raw-out-name notopo_dcrnn_fold${FOLD} >> '$LOG' 2>&1
    echo \"EXIT \$? \$(date)\" >> '$LOG'
    $TESTPART"
  echo "gestartet: $NAME gpu=$GPU stem=$STEM test=$WITH_TEST"
}

start_fold 1 0 1
start_fold 2 1 0
start_fold 3 2 0
sleep 3
screen -ls | grep notopo || true
