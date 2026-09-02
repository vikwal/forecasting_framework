#!/usr/bin/env bash
# Wirkt ein Cosine-Scheduler gegen das Rauschen der Validierungskurve?
#
# ab_ecmwf_cos unterscheidet sich von ab_ecmwf in genau zwei Schluesseln:
#   model.scheduler:       cosine
#   model.scheduler_t_max: 20
#
# Hintergrund: der TFT-Pfad (utils/tools.py, geteilt von Wind und Solar) fuhr bis
# jetzt eine konstante Lernrate — anders als die Graph-Modelle unter geostatistics/,
# die seit jeher scheduler: cosine kennen. Die Val-Kurven schwanken zwischen
# benachbarten Epochen um +-1.5 RMSE, mehr als die Verbesserung, die den "besten"
# Epochenwert ausmacht. Early Stopping waehlt damit das Minimum einer verrauschten
# Reihe.
#
# t_max 20 statt model.epochs (100) ist wesentlich: die Laeufe enden per Early
# Stopping nach 13-16 Epochen. Mit T_max=100 faellt die Lernrate bis dahin um
# ganze 5 % (5.00e-4 -> 4.76e-4), der Scheduler waere wirkungslos. Mit T_max=20
# sind es 79 %.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

echo "$(date '+%H:%M') Cosine-Scheduler eingereiht, 4 Jobs" >> "$STATUS"
while pgrep -f "^frcst/bin/python train_cl.py" > /dev/null; do sleep 60; done

JOBS=(
  "pipe_ecmwfcos_r1|configs/solar_ablation/config_solar_ab_ecmwf_cos"
  "pipe_ecmwfcos_r2|configs/solar_ablation/config_solar_ab_ecmwf_cos"
  "pipe_ecmwfcos_r3|configs/solar_ablation/config_solar_ab_ecmwf_cos"
  "pipe_ecmwfcos_r4|configs/solar_ablation/config_solar_ab_ecmwf_cos"
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
