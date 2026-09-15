#!/usr/bin/env bash
# Startet die sechs Solar-DCRNN-Arme auf Fold 1, je einen Prozess pro GPU-Slot.
#
#   scripts/run_solar_dcrnn_arms.sh <SUFFIX> [ARM ...]
#
# Ohne Armliste laufen alle sechs. Die GPU haengt am Armnamen (GPU_JE_ARM),
# nicht an der Aufrufreihenfolge — so landet ein einzeln nachgestarteter Arm
# auf derselben Karte wie im Sammelstart. Auf l2 (4 A100) teilen sich zwei
# Karten je zwei Arme; das DCRNN hat 358k Parameter, Speicher ist nicht die
# Grenze.
#
# Die Laeufe haengen an init (setsid), ueberleben also das Sessionende. Logs
# unter logs/solar_dcrnn/<arm>_<suffix>.log.
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
    log="logs/solar_dcrnn/${arm}_${SUF}.log"
    [ -f "$cfg" ] || { echo "Config fehlt: $cfg"; exit 1; }
    echo "== $(date -Is) $arm gpu=$gpu suffix=${SUF}_${arm} commit=$(git rev-parse --short HEAD)" > "$log"
    CUDA_VISIBLE_DEVICES="$gpu" setsid nohup frcst/bin/python geostatistics/train_dcrnn.py \
        --config "$cfg" --suffix "${SUF}_${arm}" >> "$log" 2>&1 &
    echo "  $arm → GPU $gpu, Log $log (PID $!)"
    sleep 2
done
