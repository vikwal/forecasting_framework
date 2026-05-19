#!/usr/bin/env bash
# run_hpo_study.sh — Starts HPO screen sessions across 4 GPUs.
#
# GPU layout:
#
#   GPU 0  hpo_dcrnn           — DCRNN baseline (no NWP nodes)       \
#          hpo_dcrnn_nwp       — DCRNN + explicit NWP nodes (GATv2)  / shared
#   GPU 1  hpo_dcrnn_nwp_hist  — DCRNN + NWP nodes + hist_wind_available
#   GPU 2  hpo_mtgnn           — MTGNN baseline                      \
#          hpo_wavenet         — GraphWaveNet baseline                / shared
#   GPU 3  hpo_mtgnn_nwp       — MTGNN + explicit NWP nodes (GATv2)
#
# Sessions are named {prefix}_{gpu}, e.g. hpo_dcrnn_0, hpo_dcrnn_nwp_0.
# Each session logs to logs/{session}.log.
set -euo pipefail

GEO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # …/geostatistics
PROJECT_DIR="$(dirname "${GEO_DIR}")"                      # …/forecasting_framework
VENV="${PROJECT_DIR}/frcst/bin/activate"

mkdir -p "${PROJECT_DIR}/logs"

# Format: "gpu  session_prefix  script  config"
SESSIONS=(
    #"0  hpo_dcrnn          geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn_base.yaml"
    #"0  hpo_dcrnn_nwp      geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn.yaml"
    #"1  hpo_dcrnn_nwp_hist geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn_nwp_hist.yaml"
    #"2  hpo_mtgnn          geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn.yaml"
    "3  hpo_wavenet        geostatistics/hpo_wavenet.py  configs/wavenet/config_wind_wavenet.yaml"
    "3  hpo_mtgnn          geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn.yaml"
    "4  hpo_mtgnn_nwp      geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn_nwp.yaml"
    "5  hpo_mtgnn_nwp      geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn_nwp.yaml"
)

total=0

for entry in "${SESSIONS[@]}"; do
    read -r gpu prefix script config <<< "$entry"
    session="${prefix}_${gpu}"
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
    total=$(( total + 1 ))
done

echo "Logs     : tail -f ${PROJECT_DIR}/logs/<session-name>.log"
