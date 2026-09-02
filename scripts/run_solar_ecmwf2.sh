#!/usr/bin/env bash
# ECMWF-Stufe 2: zusaetzlich die geometrieabhaengigen Ableitungen ecmwf_dni /
# ecmwf_kt / ecmwf_kd, die bis heute deklariert, aber nie berechnet waren
# (solar_ecmwf.add_ecmwf_geometry_features).
#
# ab_ecmwf2 unterscheidet sich von ab_ecmwf in genau diesen drei Feldern, damit
# der Vergleich einlaufig bleibt:
#   ab_features -> ab_ecmwf   misst den Nutzen der ECMWF-Rohfelder
#   ab_ecmwf    -> ab_ecmwf2  misst den Nutzen der Normierung auf Clear-Sky
#
# Erwartung aus dem Rohdatenvergleich (Station 00183, Tagstunden, Feb..Jul 2025):
#   kt  ICON 0.2531 | ECMWF 0.2950   Fehlerkorrelation 0.60
#   kd  ICON 0.2178 | ECMWF 0.2528   Fehlerkorrelation 0.63
# Dieselbe komplementaere Struktur wie bei den Rohfeldern. Auf der ICON-Seite hat
# kt_nwp in ab_features den ghi-Vorsprung gebracht, deshalb der eigene Lauf.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

echo "$(date '+%H:%M') ECMWF-Stufe-2 eingereiht, 4 Jobs" >> "$STATUS"
while pgrep -f "^frcst/bin/python train_cl.py" > /dev/null; do sleep 60; done

JOBS=(
  "pipe_ecmwf2_r1|configs/solar_ablation/config_solar_ab_ecmwf2"
  "pipe_ecmwf2_r2|configs/solar_ablation/config_solar_ab_ecmwf2"
  "pipe_ecmwf2_r3|configs/solar_ablation/config_solar_ab_ecmwf2"
  "pipe_ecmwf2_r4|configs/solar_ablation/config_solar_ab_ecmwf2"
)

worker() {
    local slot=$1 i
    for (( i=slot; i<${#JOBS[@]}; i+=4 )); do
        IFS='|' read -r name cfg <<< "${JOBS[$i]}"
        echo "$(date '+%H:%M') START  $name (gpu $slot)" >> "$STATUS"
        CUDA_VISIBLE_DEVICES=$slot PYTHONPATH=. $PY train_cl.py \
            -m tft -c "$cfg" -s "$name" > "$OUT/${name}.log" 2>&1
        if grep -q "Results saved" "$OUT/${name}.log" 2>/dev/null; then
            echo "$(date '+%H:%M') OK     $name" >> "$STATUS"
        else
            echo "$(date '+%H:%M') FEHLER $name — siehe $OUT/${name}.log" >> "$STATUS"
        fi
    done
}
for slot in 0 1 2 3; do worker "$slot" & done
wait

echo "$(date '+%H:%M') Auswertung" >> "$STATUS"
PYTHONPATH=. $PY scripts/analyse_solar_pipeline.py > "$OUT/BERICHT.txt" 2>&1
echo "$(date '+%H:%M') FERTIG — $OUT/BERICHT.txt" >> "$STATUS"
