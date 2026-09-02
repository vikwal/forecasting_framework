#!/usr/bin/env bash
# Zweite Ablationswelle — startet erst, wenn die vier laufenden Solar-Trainings durch sind.
#
#   ab_clearsky      Ziel = ghi/ghi_clearsky (schliesst nwp_residual aus)
#   ab_raster15min   freq 15min, 192 Schritte
#   ab_raster10min   freq 10min, 288 Schritte
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
mkdir -p logs/solar_ablation

echo "warte auf freie GPUs (laufende train_cl.py-Prozesse)…"
while pgrep -f "python train_cl.py" > /dev/null; do sleep 60; done
echo "GPUs frei, starte zweite Welle"

declare -A GPU=( [ab_clearsky]=0 [ab_raster15min]=1 [ab_raster10min]=2 )

pids=()
for variant in ab_clearsky ab_raster15min ab_raster10min; do
    gpu=${GPU[$variant]}
    log="logs/solar_ablation/train_${variant}.log"
    echo "starte ${variant} auf GPU ${gpu}  ->  ${log}"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. setsid nohup $PY train_cl.py \
        -m tft \
        -c "configs/solar_ablation/config_solar_${variant}" \
        -s "$variant" \
        > "$log" 2>&1 &
    pids+=($!)
    sleep 5
done

echo "PIDs: ${pids[*]}"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || { echo "FEHLGESCHLAGEN: pid $pid"; fail=1; }; done
[ $fail -eq 0 ] && echo "Zweite Welle beendet." || echo "Mindestens ein Lauf fehlgeschlagen."
exit $fail
