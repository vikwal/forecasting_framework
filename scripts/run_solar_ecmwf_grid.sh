#!/usr/bin/env bash
# Mehrere Gitterpunkte auf ECMWF-Basis — zwei Varianten, je vier Laeufe.
#
#   ab_ecmwf_g4      ab_ecmwf + next_n_grid_points_ecmwf: 4   (vier ECMWF-Punkte)
#   ab_ecmwf_icon4   ab_ecmwf + next_n_grid_points: 4         (vier ICON-Punkte)
#
# Beide unterscheiden sich von ab_ecmwf in genau einer Config-Zeile, der Vergleich
# gegen ab_ecmwf (4 Laeufe, ghi 75.310 / dhi 42.086) bleibt damit einlaeufig.
#
# Warum das interessant ist: das ECMWF-Gitter hat 0.25 Grad Abstand (~28 km), nicht
# die 9 km des nativen HRES. Der naechste Punkt liegt schon 10 km von der Station
# entfernt, vier Punkte spannen eine Box von rund 28 km — eine ganz andere raeumliche
# Skala als ICON-D2 (2.2 km, vier Punkte auf ~3 km). Die Raenge sind zu 0.984..0.998
# korreliert, ihre Mittelwerte unterscheiden sich aber um bis zu 8 W/m2. Es steckt
# also ein raeumliches Signal drin, ueber eine Distanz, auf der sich Bewoelkungs-
# felder real unterscheiden.
#
# ab_ecmwf_icon4 schliesst die letzte Luecke der Ablationsmatrix: grid4 war der
# einzige bisherige Hebel, der nie mit ECMWF kombiniert wurde.
#
# Verifiziert vor dem Start: ghi_nwp_1..4 bzw. ecmwf_ghi_1..4 entstehen, alle
# known_features finden eine Spalte, 70560 Zeilen wie die Referenz (keine Verluste).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

echo "$(date '+%H:%M') ECMWF-Gitterpunkte eingereiht, 8 Jobs" >> "$STATUS"
while pgrep -f "^frcst/bin/python train_cl.py" > /dev/null; do sleep 60; done

JOBS=(
  "pipe_ecmg4_r1|configs/solar_ablation/config_solar_ab_ecmwf_g4"
  "pipe_ecmg4_r2|configs/solar_ablation/config_solar_ab_ecmwf_g4"
  "pipe_ecmg4_r3|configs/solar_ablation/config_solar_ab_ecmwf_g4"
  "pipe_ecmg4_r4|configs/solar_ablation/config_solar_ab_ecmwf_g4"
  "pipe_ecmi4_r1|configs/solar_ablation/config_solar_ab_ecmwf_icon4"
  "pipe_ecmi4_r2|configs/solar_ablation/config_solar_ab_ecmwf_icon4"
  "pipe_ecmi4_r3|configs/solar_ablation/config_solar_ab_ecmwf_icon4"
  "pipe_ecmi4_r4|configs/solar_ablation/config_solar_ab_ecmwf_icon4"
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
