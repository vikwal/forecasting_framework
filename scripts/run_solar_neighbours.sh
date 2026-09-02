#!/usr/bin/env bash
# Nachbarstationen statt eigener Historie — Standort ohne eigene Messung.
#
# Alle drei Laeufe mit --save-predictions, weil die Nachbar-Messluecken rund 15 %
# der Fenster kosten (1400 statt 1649 im Rauchtest). Ohne gespeicherte Vorhersagen
# waere der Vergleich auf verschiedenen Stichproben gerechnet; mit ihnen laesst
# sich auf der Schnittmenge auswerten.
#
#   ab_neigh   observed = 4 Nachbarstationen, eigene Historie unterdrueckt
#   ab_noobs   observed leer (Kontrolle: gar keine Messung)
#   residual   Referenz mit eigener Historie liegt bereits als pipe_rast30_pred vor
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

echo "$(date '+%H:%M') Nachbar-Experiment gestartet" >> "$STATUS"
while pgrep -f "python train_cl.py" > /dev/null; do sleep 60; done

declare -a NAME=(pipe_neigh_r1 pipe_neigh_r2 pipe_noobs_pred)
declare -a CFG=(configs/solar_ablation/config_solar_ab_neigh
                configs/solar_ablation/config_solar_ab_neigh
                configs/solar_ablation/config_solar_ab_noobs)

pids=()
for i in 0 1 2; do
    echo "$(date '+%H:%M') START  ${NAME[$i]} (gpu $i)" >> "$STATUS"
    CUDA_VISIBLE_DEVICES=$i PYTHONPATH=. setsid nohup $PY train_cl.py \
        -m tft -c "${CFG[$i]}" -s "${NAME[$i]}" --save-predictions \
        > "$OUT/${NAME[$i]}.log" 2>&1 &
    pids+=($!)
    sleep 5
done
for pid in "${pids[@]}"; do wait "$pid"; done

for n in "${NAME[@]}"; do
    if grep -q "Results saved" "$OUT/${n}.log" 2>/dev/null; then
        echo "$(date '+%H:%M') OK     $n" >> "$STATUS"
    else
        echo "$(date '+%H:%M') FEHLER $n — siehe $OUT/${n}.log" >> "$STATUS"
    fi
done

echo "$(date '+%H:%M') Auswertung" >> "$STATUS"
PYTHONPATH=. $PY scripts/analyse_solar_pipeline.py > "$OUT/BERICHT.txt" 2>&1
PYTHONPATH=. $PY scripts/compare_solar_messnetz.py > "$OUT/BERICHT_MESSNETZ.txt" 2>&1
echo "$(date '+%H:%M') FERTIG — $OUT/BERICHT_MESSNETZ.txt" >> "$STATUS"
