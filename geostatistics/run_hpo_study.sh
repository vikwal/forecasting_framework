#!/usr/bin/env bash
# run_hpo_study.sh — Starts HPO screen sessions across 8 GPUs.
#
# GPU layout (fixed, N_PER_EXPERIMENT argument not used here):
#
#   GPU 0  hpo_dcrnn           — DCRNN baseline (no NWP nodes)
#   GPU 1  hpo_dcrnn_nwp       — DCRNN + explicit NWP nodes (GATv2)
#   GPU 2  hpo_dcrnn_nwp_hist  — DCRNN + NWP nodes + hist_wind_available
#   GPU 3  hpo_mtgnn           — MTGNN baseline
#   GPU 4  hpo_mtgnn_nwp       — MTGNN + explicit NWP nodes (GATv2)
#   GPU 5  hpo_mtgnn_nwp       — 2nd worker for MTGNN NWP (shares Optuna study)
#   GPU 6  hpo_dcrnn_nwp_hist  — 2nd worker for DCRNN NWP+hist
#   GPU 7  hpo_dcrnn_nwp       — 2nd worker for DCRNN NWP
#
# Sessions are named {prefix}_{gpu}, e.g. hpo_dcrnn_0, hpo_dcrnn_nwp_1.
# Each session logs to logs/{session}.log.
# Multiple workers sharing the same experiment share one Optuna study
# (SQLite WAL-mode or PostgreSQL if OPTUNA_STORAGE is set).
set -euo pipefail

GEO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # …/geostatistics
PROJECT_DIR="$(dirname "${GEO_DIR}")"                      # …/forecasting_framework
VENV="${PROJECT_DIR}/frcst/bin/activate"

mkdir -p "${PROJECT_DIR}/logs"

# Format: "gpu  session_prefix  script  config"
SESSIONS=(
    "0  hpo_dcrnn          geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn_base.yaml"
    "1  hpo_dcrnn_nwp      geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn.yaml"
    "2  hpo_dcrnn_nwp_hist geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn_nwp_hist.yaml"
    "3  hpo_mtgnn          geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn.yaml"
    "4  hpo_mtgnn_nwp      geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn_nwp.yaml"
    "5  hpo_mtgnn_nwp      geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn_nwp.yaml"
    "6  hpo_dcrnn_nwp_hist geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn_nwp_hist.yaml"
    "7  hpo_dcrnn_nwp      geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn.yaml"
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

echo ""
echo "Started ${total} sessions across 8 GPUs."
echo ""
echo "Experiments:"
echo "  GPU 0  hpo_dcrnn           DCRNN baseline"
echo "  GPU 1  hpo_dcrnn_nwp       DCRNN + NWP nodes          [+GPU 7]"
echo "  GPU 2  hpo_dcrnn_nwp_hist  DCRNN + NWP + hist_wind    [+GPU 6]"
echo "  GPU 3  hpo_mtgnn           MTGNN baseline"
echo "  GPU 4  hpo_mtgnn_nwp       MTGNN + NWP nodes          [+GPU 5]"
echo ""
echo "Overview : screen -ls | grep hpo"
echo "Attach   : screen -r <session-name>"
echo "Logs     : tail -f ${PROJECT_DIR}/logs/<session-name>.log"
