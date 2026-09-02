#!/usr/bin/env bash
# Baseline: alle 85 Stationen mit Strahlungsdaten, rein ZEITLICHER Split.
#
#   Training  2023-08-01 .. 2024-07-27   (faktisch ab 2023-08-08, s. Config)
#   Test      2024-08-01 .. 2025-08-01   dieselben 85 Stationen, voller Jahresgang
#
# Damit misst der Lauf, was das Modell an BEKANNTEN Standorten in einem unbekannten
# Jahr kann. Der Aufschlag fuer unbekannte Standorte kommt im naechsten Schritt
# ueber den stationsdisjunkten Aufbau (configs/solar_final/, scripts/run_solar_final.sh).
#
# Zwei Varianten, Unterschied genau eine Zeile:
#   base_lag    observed_features: [ghi, dhi]   — Messhistorie der letzten 48 h
#   base_nolag  observed_features: []           — rein prognosebasiert
#
# Zwei Stationen fallen unterwegs raus und werden uebersprungen: 04642 (keine
# Trainingsdaten im Fenster) und 04887 (nur 4 Testlaeufe, kein vollstaendiges
# Fenster). Bis zum 18.08.2026 riss letztere den ganzen Lauf mit — siehe den
# Fensterzaehler in preprocessing.py:4018.
#
# Je 2 Wiederholungen: kein Seed gesetzt, und die Lauf-zu-Lauf-Streuung lag in den
# Ablationen bei ~0.3 RMSE. Vier Jobs passen in eine Welle auf vier GPUs, die
# Wiederholungen kosten also keine zusaetzliche Wandzeit.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

echo "$(date '+%H:%M') All-Stations-Baseline eingereiht, 4 Jobs" >> "$STATUS"
while pgrep -f "^frcst/bin/python train_cl.py" > /dev/null; do sleep 60; done

JOBS=(
  "base_lag_r1|configs/solar_baseline/config_solar_base_lag"
  "base_nolag_r1|configs/solar_baseline/config_solar_base_nolag"
  "base_lag_r2|configs/solar_baseline/config_solar_base_lag"
  "base_nolag_r2|configs/solar_baseline/config_solar_base_nolag"
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
PYTHONPATH=. $PY scripts/bilanz_solar_final.py --set baseline > "$OUT/BILANZ_baseline.txt" 2>&1
echo "$(date '+%H:%M') FERTIG — $OUT/BILANZ_baseline.txt" >> "$STATUS"
