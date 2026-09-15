#!/usr/bin/env bash
# Wertet die Solar-DCRNN-Arme auf dem Validierungsjahr aus und schreibt je Arm
# die Rohvorhersagen, aus denen scripts/eval_solar_arch.py den Vergleich gegen
# den TFT rechnet.
#
#   scripts/run_solar_dcrnn_eval.sh <SUFFIX> [ARM ...]
#
# Ausgaben je Arm:
#   data/raw_preds/solar_dcrnn_<suffix>_<arm>_raw.parquet
#   data/test_results/solar_dcrnn_<suffix>_<arm>.csv
#
# Ohne --test-mode laeuft die Auswertung auf val_start … test_start, also dem
# Validierungsjahr — das Testjahr bleibt zurueckgehalten, bis die Modellwahl
# steht (docs/solar_tft_kampagne.md §3.1).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

SUF="${1:?Suffix fehlt, z.B. v5}"; shift
ARME=("$@"); [ ${#ARME[@]} -eq 0 ] && ARME=(a base nomeas nograph idw_alt nwp_hist)
declare -A GPU_JE_ARM=([a]=0 [idw_alt]=0 [base]=1 [nwp_hist]=1 [nomeas]=2 [nograph]=3)

eval "$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE|DATA_ROOT)=' ~/.bashrc)"
: "${DATA_ROOT:?DATA_ROOT fehlt}"
export WEATHER_DB_URL ECMWF_WIND_SL_URL OPTUNA_STORAGE DATA_ROOT

mkdir -p logs/solar_dcrnn
for arm in "${ARME[@]}"; do
    gpu="${GPU_JE_ARM[$arm]:?unbekannter Arm: $arm}"
    cfg="configs/solar_dcrnn/config_solar_dcrnn_${arm}_fold1.yaml"
    mdl="solar_dcrnn_${arm}_fold1_dcrnn_${SUF}_${arm}"
    log="logs/solar_dcrnn/eval_${arm}_${SUF}.log"
    [ -f "models/${mdl}.pt" ] || { echo "Modell fehlt: models/${mdl}.pt"; exit 1; }
    echo "== $(date -Is) eval $arm gpu=$gpu commit=$(git rev-parse --short HEAD)" > "$log"
    CUDA_VISIBLE_DEVICES="$gpu" setsid nohup frcst/bin/python geostatistics/get_test_results_dcrnn.py \
        -m "$mdl" -c "$cfg" --raw-out-name "solar_dcrnn_${SUF}_${arm}" >> "$log" 2>&1 &
    echo "  $arm → GPU $gpu, Log $log (PID $!)"
    sleep 2
done
