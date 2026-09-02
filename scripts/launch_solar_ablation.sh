#!/usr/bin/env bash
# Ablationen auf der Basis config_solar_residual (Bias Correction).
#
#   ab_single    nur ghi statt ghi+dhi  -> stoert die Doppelausgabe?
#   ab_features  + kt_nwp, airmass, dni_clearsky, dhi_clearsky
#
# Beide sind bis auf die genannte Aenderung identisch mit dem residual-Referenzlauf,
# also direkt ueber dessen ghi-Zeile vergleichbar.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
mkdir -p logs/solar_ablation

declare -A GPU=( [ab_single]=0 [ab_features]=3 )

pids=()
for variant in ab_single ab_features; do
    gpu=${GPU[$variant]}
    log="logs/solar_ablation/train_${variant}.log"
    echo "starte ${variant} auf GPU ${gpu}  ->  ${log}"
    # setsid: eigene Session, damit ein Abbruch des Wrappers die Laeufe nicht mitreisst
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
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "FEHLGESCHLAGEN: pid $pid"; fail=1; }
done
[ $fail -eq 0 ] && echo "Beide Ablationen beendet." || echo "Mindestens eine fehlgeschlagen."
exit $fail
