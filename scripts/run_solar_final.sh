#!/usr/bin/env bash
# Abschlussbewertung der besten Konfiguration — 12 Monate Training, 12 Monate Test.
#
# Konfiguration (Stand der Ablationen vom 17./18.08.):
#   target_transform  nwp_residual   (groesster Einzelhebel, alles davor lag <1 %)
#   nwp_models        icon-d2 + ecmwf   +4.08 % ghi / +2.56 % dhi, p ~ 1e-5
#   next_n_grid_points 1              4 Punkte brachten mit ECMWF nichts mehr
#   Features          ab_features-Satz (kt_nwp, airmass, dni/dhi_clearsky)
#   ohne Scheduler, ohne ecmwf_dni/kt/kd — beide gemessen wirkungslos
#
# Zwei Varianten, die sich in genau einer Zeile unterscheiden:
#   final_lag    observed_features: [ghi, dhi]   — Messhistorie der letzten 48 h
#   final_nolag  observed_features: []           — rein prognosebasiert
#
# Je 4 Wiederholungen, weil kein Seed gesetzt ist: die Lauf-zu-Lauf-Streuung lag
# in den Ablationen bei ~0.3 RMSE und muss vom Variantenunterschied trennbar sein.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

echo "$(date '+%H:%M') Abschlussbewertung eingereiht, 8 Jobs" >> "$STATUS"
while pgrep -f "^frcst/bin/python train_cl.py" > /dev/null; do sleep 60; done

JOBS=(
  "final_lag_r1|configs/solar_final/config_solar_final_lag"
  "final_lag_r2|configs/solar_final/config_solar_final_lag"
  "final_lag_r3|configs/solar_final/config_solar_final_lag"
  "final_lag_r4|configs/solar_final/config_solar_final_lag"
  "final_nolag_r1|configs/solar_final/config_solar_final_nolag"
  "final_nolag_r2|configs/solar_final/config_solar_final_nolag"
  "final_nolag_r3|configs/solar_final/config_solar_final_nolag"
  "final_nolag_r4|configs/solar_final/config_solar_final_nolag"
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

echo "$(date '+%H:%M') Bilanz" >> "$STATUS"
PYTHONPATH=. $PY scripts/bilanz_solar_final.py > "$OUT/BILANZ.txt" 2>&1
echo "$(date '+%H:%M') FERTIG — $OUT/BILANZ.txt" >> "$STATUS"
