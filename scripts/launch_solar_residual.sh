#!/usr/bin/env bash
# Residuum-Experiment: 56 Trainingsstationen, mit und ohne Residuum-Ziel.
#
#   absolut   Kontrolle  — Ziel in W/m² wie bisher
#   residual  Ziel = Messung - ICON-D2-Prognose  (Bias Correction)
#
# Beide Configs sind identisch bis auf params.target_transform. Der Kontrolllauf
# trennt den Effekt des Residuums von dem der 56 Stationen — das Rasterexperiment
# lief auf einer einzigen Station.
#
# RMSE und Skill_NWP sind zwischen beiden Laeufen direkt vergleichbar: im
# Residuumsraum bleiben Differenzen erhalten, und die NWP-Baseline ist dort die
# Nullreihe, deren RMSE exakt dem NWP-Fehler in W/m² entspricht.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
mkdir -p logs/solar_residual results/solar_residual

declare -A GPU=( [absolut]=1 [residual]=2 )

pids=()
for variant in absolut residual; do
    gpu=${GPU[$variant]}
    log="logs/solar_residual/train_${variant}.log"
    echo "starte ${variant} auf GPU ${gpu}  ->  ${log}"
    # setsid: eigene Session, damit ein Abbruch dieses Wrappers die Laeufe nicht
    # mitreisst — beim ersten Anlauf sind so zwei fast fertige Trainings verloren
    # gegangen, kurz vor dem Schreiben der Ergebnisse.
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. setsid nohup $PY train_cl.py \
        -m tft \
        -c "configs/solar_residual/config_solar_${variant}" \
        -s "res${variant}" \
        --save_model \
        > "$log" 2>&1 &
    pids+=($!)
    sleep 5
done

echo "PIDs: ${pids[*]}"
echo "Fortschritt:  tail -f logs/solar_residual/train_*.log"

fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "FEHLGESCHLAGEN: pid $pid"
        fail=1
    fi
done

[ $fail -eq 0 ] && echo "Beide Laeufe beendet." || echo "Mindestens ein Lauf fehlgeschlagen — Logs pruefen."
exit $fail
