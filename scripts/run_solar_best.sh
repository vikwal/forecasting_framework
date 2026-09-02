#!/usr/bin/env bash
# Absicherung der beiden besten Varianten plus ihre Kombination.
#
#   ab_features   lag bei ghi vorn (78.05), hatte aber nur EINEN Lauf — seine
#                 eigene Streuung war unbekannt. Drei Wiederholungen dazu.
#   ab_featgrid   Zusatzfeatures UND vier Gitterpunkte. Beide Hebel lagen einzeln
#                 rund 0.4 % vor dem einfachen residual, zusammen nie getestet.
#                 Verifiziert: kt_nwp_1..4 entstehen, known_dim 17 -> 48,
#                 keine Fenster gehen verloren (1649 wie die Referenz).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

echo "$(date '+%H:%M') Absicherungslauf gestartet, 5 Jobs" >> "$STATUS"
# Muster am Binaerpfad verankert: sonst trifft pgrep -f auch fremde Shells, deren
# Kommandozeile diesen Text enthaelt (etwa ein Diagnose-Aufruf) — der Waechter
# wartete dadurch auf sich selbst bzw. auf voruebergehende Hilfsprozesse.
while pgrep -f "^frcst/bin/python train_cl.py" > /dev/null; do sleep 60; done

JOBS=(
  "pipe_featgrid_r1|configs/solar_ablation/config_solar_ab_featgrid"
  "pipe_featgrid_r2|configs/solar_ablation/config_solar_ab_featgrid"
  "pipe_feat_r2|configs/solar_ablation/config_solar_ab_features"
  "pipe_feat_r3|configs/solar_ablation/config_solar_ab_features"
  "pipe_feat_r4|configs/solar_ablation/config_solar_ab_features"
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
