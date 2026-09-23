#!/usr/bin/env bash
# launch_idwobs.sh dcrnn|mtgnn — IDW + site obs. runs (2026-09-23): the + site obs. arms with the
# prescribed grid weighting (nwp_aggregation=idw_alt) instead of the learned attention.
# Four screens on GPUs 0..3: validation folds 1-3 (train + eval, suffix idwobs) and the
# test-mode run (train on the step1 window, evaluate over the full test year, suffix tidwobs).
# Hyperparameters: --hpo-study auto resolves the hist study (same config file name);
# idw_p / alpha_alt come from the DCRNN D-prime study (trial 109) via the config.
# Paper docs: Graphs_Wind_Speed_Forecasting/docs/handoff.md
set -uo pipefail
ARM=${1:?dcrnn or mtgnn}
cd "$(dirname "$0")/.." || exit 1
eval "$(grep -E '^export (OPTUNA_STORAGE|DATA_ROOT|ECMWF_WIND_SL_URL|WEATHER_DB_URL)=' ~/.bashrc)"
: "${OPTUNA_STORAGE:?OPTUNA_STORAGE fehlt in ~/.bashrc}"; : "${DATA_ROOT:?DATA_ROOT fehlt in ~/.bashrc}"
export OPTUNA_STORAGE DATA_ROOT ECMWF_WIND_SL_URL="${ECMWF_WIND_SL_URL:-}" WEATHER_DB_URL="${WEATHER_DB_URL:-}"
mkdir -p logs
REPO=$PWD
for i in 1 2 3; do
  g=$((i-1)); log=logs/idwobs_${ARM}_fold$i.out
  screen -dmS idwobs_${ARM}_f$i bash -c "cd $REPO && source frcst/bin/activate && export CUDA_VISIBLE_DEVICES=$g && echo START \$(date) > $log && python geostatistics/train_${ARM}.py --config configs/${ARM}/idwobs/config_wind_${ARM}_nwp_hist_fold$i.yaml --suffix idwobs --hpo-study auto >> $log 2>&1 && python geostatistics/get_test_results_${ARM}.py -m wind_${ARM}_nwp_hist_fold${i}_${ARM}_idwobs -c configs/${ARM}/idwobs/config_wind_${ARM}_nwp_hist_fold$i.yaml --hpo-study auto --raw-out-name idwobs_${ARM}_nwp_hist_fold$i >> $log 2>&1; echo EXIT \$? \$(date) >> $log"
done
log=logs/idwobs_${ARM}_test.out
screen -dmS idwobs_${ARM}_test bash -c "cd $REPO && source frcst/bin/activate && export CUDA_VISIBLE_DEVICES=3 && echo START \$(date) > $log && python geostatistics/train_${ARM}.py --config configs/testmode/idwobs_step1/config_wind_${ARM}_nwp_hist_fold1.yaml --suffix tidwobs --hpo-study auto --test-mode >> $log 2>&1 && python geostatistics/get_test_results_${ARM}.py -m wind_${ARM}_nwp_hist_fold1_${ARM}_tidwobs -c configs/testmode/idwobs_full/config_wind_${ARM}_nwp_hist_fold1.yaml --hpo-study auto --test-mode --raw-out-name idwobs_${ARM}_nwp_hist_once >> $log 2>&1; echo EXIT \$? \$(date) >> $log"
sleep 2; screen -ls | grep idwobs
