#!/usr/bin/env bash
# Bringt ECMWF-HRES als zweite NWP-Quelle zusaetzliche Genauigkeit?
#
# Basis ist ab_features (residual + Zusatzfeatures), das mit 7 Laeufen die am
# besten abgesicherte Variante ist. ab_ecmwf unterscheidet sich davon in genau
# einem Hebel: acht ecmwf_*-Spalten in known_features. Damit ist der Vergleich
# ab_ecmwf gegen ab_features einlaufig.
#
# Motivation aus dem Rohdatenvergleich (4 Stationen, Tagstunden, Feb..Jul 2025):
#   ghi  ICON 117.2 | ECMWF 129.7 | 50/50-Mittel 110.2   Fehlerkorrelation 0.61
#   dhi  ICON  65.9 | ECMWF  71.8 | 50/50-Mittel  63.5   Fehlerkorrelation 0.71
# ECMWF allein ist schlechter, aber die Fehler sind nur maessig korreliert —
# schon das naive Mittel liegt 6 % bzw. 3.6 % vor ICON allein. Zum Vergleich:
# die gesamte Spannweite aller bisherigen Ablationen lag unter 1 %.
#
# Verifiziert vor dem Start: alle 29 known_features finden eine Spalte, die acht
# ecmwf_*-Felder haben 100 % Deckung im Testfenster (ECMWF-Parquets reichen von
# 2023-07 bis 2026-07 und decken Feb..Jul 2025 vollstaendig ab).
#
# NICHT verwendet: ecmwf_dni / ecmwf_kt / ecmwf_kd. Die sind in
# solar_ecmwf.ECMWF_GEOMETRY_DERIVED deklariert, aber nirgends berechnet —
# sie wuerden stillschweigend fehlen.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=frcst/bin/python
OUT=logs/solar_pipeline
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

echo "$(date '+%H:%M') ECMWF-Test gestartet, 4 Jobs" >> "$STATUS"
# Muster am Binaerpfad verankert, sonst trifft pgrep -f auch fremde Shells,
# deren Kommandozeile diesen Text enthaelt.
while pgrep -f "^frcst/bin/python train_cl.py" > /dev/null; do sleep 60; done

JOBS=(
  "pipe_ecmwf_r1|configs/solar_ablation/config_solar_ab_ecmwf"
  "pipe_ecmwf_r2|configs/solar_ablation/config_solar_ab_ecmwf"
  "pipe_ecmwf_r3|configs/solar_ablation/config_solar_ab_ecmwf"
  "pipe_ecmwf_r4|configs/solar_ablation/config_solar_ab_ecmwf"
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
