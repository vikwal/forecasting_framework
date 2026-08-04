#!/bin/bash
# Launch TFT-BC HPO workers: one 'base' and one 'hist' worker per GPU.
#
#   ./launch_tft_bc_hpo.sh <n_gpus> [max_cache_gb] [config_stem]
#
# 8 GPUs (l1) -> 16 workers (8 base / 8 hist)
# 4 GPUs (L2) ->  8 workers (4 base / 4 hist)
#
# config_stem selects WHICH pair of configs (and therefore which Optuna studies) the
# workers run; the two variants base/hist are appended to it:
#   wind_tft      (default) -> configs/tft_bc/config_wind_tft_{base,hist}.yaml
#                              = the OLD temporal-CV study (hpo.cv_mode absent)
#   wind_tft_sp             -> configs/tft_bc/config_wind_tft_sp_{base,hist}.yaml
#                              = the spatial-CV study, the one comparable to the
#                                DCRNN/MTGNN/WaveNet benchmarks
# Screen/log names carry the stem, so a spatial and a temporal campaign can coexist
# without colliding.
#
# NOTE on max_cache_gb under cv_mode=spatial: each trial builds THREE cache entries
# (one per spatial fold) instead of one, so the same budget holds a third as many
# (next_n_grid_points, next_n_grid_ecmwf, next_n_stations) combinations before the LRU
# eviction in hpo_tft_bc.py starts re-preprocessing them. Scale the budget accordingly
# (both hosts have multiple TB free) or the workers will spend most of their time in
# preprocessing rather than training.
#
# NOTE: deliberately no -s/--suffix. hpo_tft_bc.py derives the Optuna study name
# from the config name plus the suffix, so a per-worker suffix would split one study
# into several. Only the screen and log names vary per worker.
set -euo pipefail
cd "$(dirname "$0")"

N_GPUS="${1:?usage: $0 <n_gpus> [max_cache_gb] [config_stem]}"
# Each trial samples its own (next_n_grid_points, next_n_grid_ecmwf, next_n_stations)
# and therefore builds its own cache entry (~9 GB). The 150 GB default would keep only
# ~17 of them and evict constantly with 24 workers running; both hosts have multiple TB
# free, so give the cache room instead of re-preprocessing the same combinations.
MAX_CACHE_GB="${2:-400}"
CONFIG_STEM="${3:-wind_tft}"
LOGDIR=logs/hpo_tft_bc
mkdir -p "$LOGDIR"

started=0
for ((g = 0; g < N_GPUS; g++)); do
    for v in base hist; do
        name="hpo_tft_${CONFIG_STEM#wind_tft}_${v}_g${g}"
        name="${name//__/_}"
        if screen -ls 2>/dev/null | grep -q "\.${name}[[:space:]]"; then
            echo "SKIP  $name (already running)"
            continue
        fi
        screen -dmS "$name" bash -c \
            "source frcst/bin/activate && exec python hpo_tft_bc.py \
             -c configs/tft_bc/config_${CONFIG_STEM}_${v} --gpu ${g} \
             --max-cache-gb ${MAX_CACHE_GB} \
             >> ${LOGDIR}/${name}.log 2>&1"
        echo "START $name  (gpu $g, config configs/tft_bc/config_${CONFIG_STEM}_${v}.yaml)"
        started=$((started + 1))
        sleep 1
    done
done

sleep 5
echo
echo "started $started workers; screens now running:"
screen -ls | grep -c "hpo_tft_" || true
