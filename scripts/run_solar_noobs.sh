#!/usr/bin/env bash
# Nachzuegler 3: ohne historische Messungen (observed_features leer).
#
# Trennt, wie viel des Skills aus der Autokorrelation der eigenen Messreihe
# stammt und wie viel aus ICON-D2 selbst. Das Modell sieht nur noch die
# NWP-Felder, die Sonnengeometrie und die statischen Stationsmerkmale.
#
# Nachgeprueft, dass die Historie wirklich fehlt und sonst nichts abweicht:
#   mit  Historie: X['observed'] (1649, 96, 2), known (1649, 192, 17), static (1649, 3)
#   ohne Historie: X['observed'] (1649, 96, 0), known (1649, 192, 17), static (1649, 3)
# Gleiche Stichprobengroesse, gleiche NWP-Baseline (87.55 / 48.17 im Rauchtest).
#
# Name bewusst nicht 'run_solar_pipeline*', sonst faende pgrep -f dieses Skript selbst.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

echo "$(date '+%H:%M') Lauf-ohne-Historie wartet auf die vorherigen Pipelines…" >> "$STATUS"
while pgrep -f "run_solar_pipeline\.sh|run_solar_pipeline2\.sh|run_solar_gridpoints\.sh" > /dev/null; do sleep 60; done
while pgrep -f "python train_cl.py" > /dev/null; do sleep 60; done
echo "$(date '+%H:%M') starte Laeufe ohne historische Messungen" >> "$STATUS"

CFG=configs/solar_ablation/config_solar_ab_noobs
pids=()
for i in 1 2; do
    name="pipe_noobs_r${i}"
    gpu=$((i-1))
    echo "$(date '+%H:%M') START  $name (gpu $gpu)" >> "$STATUS"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. setsid nohup $PY train_cl.py \
        -m tft -c "$CFG" -s "$name" > "$OUT/${name}.log" 2>&1 &
    pids+=($!)
    sleep 5
done
for pid in "${pids[@]}"; do wait "$pid"; done

for i in 1 2; do
    name="pipe_noobs_r${i}"
    if grep -q "Results saved" "$OUT/${name}.log" 2>/dev/null; then
        echo "$(date '+%H:%M') OK     $name" >> "$STATUS"
    else
        echo "$(date '+%H:%M') FEHLER $name — siehe $OUT/${name}.log" >> "$STATUS"
    fi
done

echo "$(date '+%H:%M') Auswertung wird neu erstellt" >> "$STATUS"
PYTHONPATH=. $PY scripts/analyse_solar_pipeline.py > "$OUT/BERICHT.txt" 2>&1
echo "$(date '+%H:%M') ALLES FERTIG — Bericht: $OUT/BERICHT.txt" >> "$STATUS"
