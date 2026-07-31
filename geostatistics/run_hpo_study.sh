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

# ── Commit-hash gate ─────────────────────────────────────────────────────
# All workers of a shared Optuna study must run the exact same code. A
# mismatch (e.g. a host missing the topo/spatial-CV branch) writes trials of
# two different model classes into one study with no crash and no warning —
# see docs/topo_features_review_brief.md, blocker 9.1/B4. Set EXPECTED_COMMIT
# (e.g. from a multi-host launcher) to make this host abort on a mismatch;
# without it, the commit is only logged, not enforced.
COMMIT="$(cd "${PROJECT_DIR}" && git rev-parse HEAD)"
DIRTY="$(cd "${PROJECT_DIR}" && git status --porcelain | wc -l | tr -d ' ')"
echo "Commit   : ${COMMIT}$([ "${DIRTY}" -gt 0 ] && echo "  (${DIRTY} dirty file(s))")"
if [[ -n "${EXPECTED_COMMIT:-}" && "${COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    echo "ABORT: HEAD (${COMMIT}) does not match EXPECTED_COMMIT (${EXPECTED_COMMIT})." >&2
    exit 1
fi

# Format: "gpu  session_prefix  script  config"
SESSIONS=(
    #"0  hpo_dcrnn          geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn_base.yaml"
    #"0  hpo_dcrnn_nwp      geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn.yaml"
    #"1  hpo_dcrnn_nwp_hist geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn_nwp_hist.yaml"
    #"2  hpo_mtgnn          geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn.yaml"
    # "3  hpo_wavenet        geostatistics/hpo_wavenet.py  configs/wavenet/config_wind_wavenet.yaml"
    # "3  hpo_mtgnn          geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn.yaml"
    # "4  hpo_mtgnn_nwp      geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn_nwp.yaml"
    # "5  hpo_mtgnn_nwp      geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn_nwp.yaml"
    #"0  hpo_dcrnn_nwp_hist geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn_nwp_hist.yaml"
    "1  hpo_dcrnn_nwp_hist geostatistics/hpo_dcrnn.py    configs/dcrnn/config_wind_dcrnn_nwp_hist.yaml"
    "2  hpo_mtgnn_nwp_hist geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn_nwp_hist.yaml"
    #"3  hpo_mtgnn_nwp_hist geostatistics/hpo_mtgnn.py    configs/mtgnn/config_wind_mtgnn_nwp_hist.yaml"
)

total=0

for entry in "${SESSIONS[@]}"; do
    read -r gpu prefix script config <<< "$entry"
    session="${prefix}_${gpu}"
    log="${PROJECT_DIR}/logs/${session}.log"

    screen -dmS "$session" bash -c "
        source '${VENV}'
        cd '${PROJECT_DIR}'
        { echo \"=== host: \$(hostname)  commit: ${COMMIT}  dirty: ${DIRTY}  config: ${config} ===\"
          python '${script}' \
            --config '${config}' \
            --gpu ${gpu} \
            --suffix gpu${gpu} ; } \
            2>&1 | tee '${log}'
        exec bash
    "

    echo "Started  screen '${session}'  GPU ${gpu}  log → ${log}"
    total=$(( total + 1 ))
done

echo "Logs     : tail -f ${PROJECT_DIR}/logs/<session-name>.log"
