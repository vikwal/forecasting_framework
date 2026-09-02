#!/usr/bin/env bash
# Nachzuegler 2: vier ICON-D2-Gitterpunkte statt einem (Basis residual).
#
# Solar unterstuetzt params.next_n_grid_points genauso wie Wind — die Auswahl in
# solar.select_nearest_sl_points ist geodaetisch, die Spalten heissen <feature>_<rang>
# mit Rang 1 = naechster Punkt. Bisher lief aber jeder Solar-Lauf mit einem Punkt.
#
# Erwartung: wenig Wirkung. Die vier naechsten Punkte liegen bei Station 01358 in
# 0.62 bis 2.43 km, ghi_nwp_1 und ghi_nwp_2 korrelieren mit 0.9984. Anders als bei
# Wind, wo der 12-Punkte-Stencil die Anstroemung ueber Flaeche abbildet, sind die
# Nachbarpunkte hier bei Bewoelkungsstrukturen von mehreren Kilometern redundant.
#
# Zwei Wiederholungen, damit sich das Ergebnis gegen die Lauf-zu-Lauf-Streuung
# halten laesst (bei residual gemessen: 0.08 W/m² bei ghi).
#
# Der Wartename ist bewusst NICHT 'run_solar_pipeline*', sonst wuerde pgrep -f
# dieses Skript selbst finden und ewig warten.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

echo "$(date '+%H:%M') Gitterpunkt-Lauf wartet auf die vorherigen Pipelines…" >> "$STATUS"
while pgrep -f "run_solar_pipeline\.sh|run_solar_pipeline2\.sh" > /dev/null; do sleep 60; done
while pgrep -f "python train_cl.py" > /dev/null; do sleep 60; done
echo "$(date '+%H:%M') starte Gitterpunkt-Laeufe" >> "$STATUS"

CFG=configs/solar_ablation/config_solar_ab_grid4
pids=()
for i in 1 2; do
    name="pipe_grid4_r${i}"
    gpu=$((i-1))
    echo "$(date '+%H:%M') START  $name (gpu $gpu)" >> "$STATUS"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. setsid nohup $PY train_cl.py \
        -m tft -c "$CFG" -s "$name" > "$OUT/${name}.log" 2>&1 &
    pids+=($!)
    sleep 5
done
for pid in "${pids[@]}"; do wait "$pid"; done

for i in 1 2; do
    name="pipe_grid4_r${i}"
    if grep -q "Results saved" "$OUT/${name}.log" 2>/dev/null; then
        echo "$(date '+%H:%M') OK     $name" >> "$STATUS"
    else
        echo "$(date '+%H:%M') FEHLER $name — siehe $OUT/${name}.log" >> "$STATUS"
    fi
done

echo "$(date '+%H:%M') Auswertung wird neu erstellt" >> "$STATUS"
PYTHONPATH=. $PY scripts/analyse_solar_pipeline.py > "$OUT/BERICHT.txt" 2>&1
echo "$(date '+%H:%M') ALLES FERTIG — Bericht: $OUT/BERICHT.txt" >> "$STATUS"
