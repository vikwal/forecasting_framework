#!/usr/bin/env bash
# Nachzuegler: Clear-Sky-Index mit korrigierter Kappung fuer dhi.
#
# Der Lauf vom 13.08. nachmittags meldete fuer dhi RMSE 19.7 statt 43.1 (-54 %).
# Das war kein besserer Forecast, sondern eine abgeschnittene Zielgroesse:
# clearsky_index kappt bei k=1.5, aber Diffusstrahlung ist UNTER Wolken maximal —
# 29.5 bis 48.3 % der Tagsamples liegen darueber, der Median bei 1.10 bis 1.46.
# Erkannt daran, dass die NWP-Baseline mitsank (47.77 -> 41.61 W/m²), was bei
# gleichem Raster und Testset unmoeglich ist.
#
# params.clearsky_clip_max: {ghi: 1.5, dhi: 6.0} stellt die Baseline wieder her
# (nachgeprueft: 46.07/50.63 gegen 45.89/50.46 der Referenz).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

echo "$(date '+%H:%M') Nachzuegler wartet auf die Hauptpipeline…" >> "$STATUS"
while pgrep -f "run_solar_pipeline.sh" > /dev/null; do sleep 60; done
while pgrep -f "python train_cl.py" > /dev/null; do sleep 60; done
echo "$(date '+%H:%M') Hauptpipeline durch, starte korrigierte Clear-Sky-Laeufe" >> "$STATUS"

CFG=configs/solar_ablation/config_solar_ab_clearsky2
pids=()
for i in 1 2 3; do
    name="pipe_clearsky2_r${i}"
    gpu=$((i-1))
    echo "$(date '+%H:%M') START  $name (gpu $gpu)" >> "$STATUS"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. setsid nohup $PY train_cl.py \
        -m tft -c "$CFG" -s "$name" > "$OUT/${name}.log" 2>&1 &
    pids+=($!)
    sleep 5
done
for pid in "${pids[@]}"; do wait "$pid"; done

for i in 1 2 3; do
    name="pipe_clearsky2_r${i}"
    if grep -q "Results saved" "$OUT/${name}.log" 2>/dev/null; then
        echo "$(date '+%H:%M') OK     $name" >> "$STATUS"
    else
        echo "$(date '+%H:%M') FEHLER $name — siehe $OUT/${name}.log" >> "$STATUS"
    fi
done

echo "$(date '+%H:%M') Auswertung wird neu erstellt" >> "$STATUS"
PYTHONPATH=. $PY scripts/analyse_solar_pipeline.py > "$OUT/BERICHT.txt" 2>&1
echo "$(date '+%H:%M') fertig — Bericht: $OUT/BERICHT.txt" >> "$STATUS"
