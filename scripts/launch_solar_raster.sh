#!/usr/bin/env bash
# Rasterexperiment Solar: 30min / 15min / 10min parallel auf drei GPUs.
#
#   30min  Referenz  — beide Quellen exakt (30 = kgV(10, 15)), keine Umverteilung
#   15min  Variante A — Messung 10 -> 15 min flaechengewichtet umverteilt, ICON nativ
#   10min  Variante C — ICON 15 -> 10 min entlang der Lead-Achse umverteilt, Messung nativ
#
# Alle drei Configs sind identisch bis auf data.freq und die daran gekoppelten
# Schrittzahlen (48 h Horizont in jedem Fall). Station 02712, Training
# 2023-08-01..2024-08-01, Test 2024-08-01..2025-08-01.
#
# Die RMSE der drei Laeufe sind NICHT direkt vergleichbar — verschiedene Raster
# mitteln verschieden stark. Vergleich ueber scripts/compare_solar_raster.py.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
mkdir -p logs/solar_raster results/solar_raster

# GPU 3 ist am staerksten belegt (29 GB) -> 0/1/2 nehmen.
declare -A GPU=( [30min]=0 [15min]=1 [10min]=2 )

pids=()
for freq in 30min 15min 10min; do
    gpu=${GPU[$freq]}
    log="logs/solar_raster/train_${freq}.log"
    echo "starte freq=${freq} auf GPU ${gpu}  ->  ${log}"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. nohup $PY train_cl.py \
        -m tft \
        -c "configs/solar_raster/config_solar_${freq}" \
        -s "raster${freq}" \
        --save_model \
        > "$log" 2>&1 &
    pids+=($!)
    sleep 3
done

echo "PIDs: ${pids[*]}"
echo "Fortschritt:  tail -f logs/solar_raster/train_*.log"

fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "FEHLGESCHLAGEN: pid ${pids[$i]}"
        fail=1
    fi
done

if [ $fail -eq 0 ]; then
    echo "Alle drei Laeufe beendet."
else
    echo "Mindestens ein Lauf ist fehlgeschlagen — Logs pruefen."
fi
exit $fail
