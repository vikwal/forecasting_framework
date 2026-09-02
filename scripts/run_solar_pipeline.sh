#!/usr/bin/env bash
# Nachtlauf: beantwortet die zwei offenen Fragen aus den Ablationen.
#
# BLOCK A — Aufloesung auf GEMEINSAMEM Raster
#   Skill_NWP taugt nicht zum Vergleich ueber Raster hinweg: die NWP-Baseline
#   aendert sich mit der Aufloesung (84.8 bei 30min, 91.0 bei 15min). Deshalb
#   laufen alle drei Raster mit --save-predictions; die Auswertung aggregiert
#   15min/10min zurueck auf 30min und vergleicht auf identischen Samples.
#
# BLOCK B — Streuung zwischen Laeufen
#   Es gibt kein torch.manual_seed im Repo, Initialisierung und Shuffling sind
#   also ohnehin zufaellig. Dieselbe Config mehrfach zu starten misst damit
#   direkt die Lauf-zu-Lauf-Streuung. Ohne die ist jeder Unterschied unter 1 %
#   zwischen den Zielvarianten unbelegt.
#
# Robustheit: jeder Job laeuft in seinem eigenen Prozess; ein Fehlschlag
# stoppt die Warteschlange nicht, sondern wird in status.txt vermerkt.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

R=configs/solar_residual
A=configs/solar_ablation

# Name|Config|Zusatzargumente   — absteigend nach erwarteter Laufzeit sortiert,
# damit die langen Jobs zuerst starten und die Warteschlange gleichmaessig auslaeuft.
JOBS=(
  "pipe_rast10_pred|$A/config_solar_ab_raster10min|--save-predictions"
  "pipe_rast15_pred|$A/config_solar_ab_raster15min|--save-predictions"
  "pipe_rast30_pred|$R/config_solar_residual|--save-predictions"
  "pipe_absolut_r2|$R/config_solar_absolut|"
  "pipe_residual_r2|$R/config_solar_residual|"
  "pipe_clearsky_r2|$A/config_solar_ab_clearsky|"
  "pipe_absolut_r3|$R/config_solar_absolut|"
  "pipe_residual_r3|$R/config_solar_residual|"
  "pipe_clearsky_r3|$A/config_solar_ab_clearsky|"
)

echo "$(date '+%F %H:%M') Pipeline gestartet, ${#JOBS[@]} Jobs" | tee "$STATUS"

echo "$(date '+%H:%M') warte auf laufende Solar-Trainings…" | tee -a "$STATUS"
while pgrep -f "python train_cl.py" > /dev/null; do sleep 60; done
echo "$(date '+%H:%M') GPUs frei" | tee -a "$STATUS"

worker() {                       # $1 = GPU / Slot-Index
    local slot=$1 i
    for (( i=slot; i<${#JOBS[@]}; i+=4 )); do
        IFS='|' read -r name cfg extra <<< "${JOBS[$i]}"
        echo "$(date '+%H:%M') START  $name (gpu $slot)" >> "$STATUS"
        CUDA_VISIBLE_DEVICES=$slot PYTHONPATH=. $PY train_cl.py \
            -m tft -c "$cfg" -s "$name" $extra > "$OUT/${name}.log" 2>&1
        local rc=$?
        if [ $rc -eq 0 ] && grep -q "Results saved" "$OUT/${name}.log"; then
            echo "$(date '+%H:%M') OK     $name" >> "$STATUS"
        else
            echo "$(date '+%H:%M') FEHLER $name (rc=$rc) — siehe $OUT/${name}.log" >> "$STATUS"
        fi
    done
}

for slot in 0 1 2 3; do worker "$slot" & done
wait

echo "$(date '+%H:%M') alle Jobs beendet, starte Auswertung" >> "$STATUS"
PYTHONPATH=. $PY scripts/analyse_solar_pipeline.py > "$OUT/BERICHT.txt" 2>&1
echo "$(date '+%H:%M') Bericht: $OUT/BERICHT.txt" >> "$STATUS"
