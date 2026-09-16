#!/usr/bin/env bash
# Holt die Ergebnis-Pickles der auf l1 und ws gerechneten Laeufe nach l2.
# results/ ist gitignored, die Dateien kommen also nicht ueber /sync mit.
#
#   scripts/sammle_solar_lokal.sh <SUFFIX>
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SUF="${1:?Suffix fehlt, z.B. v1}"
cd "$REPO" || exit 1

for h in l1 ws; do
    n=$(ssh "$h" "ls ~/Work/forecasting_framework/results/solar/*_${SUF}_*.pkl 2>/dev/null | wc -l")
    echo "== $h: $n Datei(en)"
    [ "${n:-0}" -eq 0 ] && continue
    rsync -a --info=progress2 \
        "$h:~/Work/forecasting_framework/results/solar/*_${SUF}_*.pkl" \
        results/solar/ 2>&1 | tail -1
done
echo "== l2 gesamt: $(ls results/solar/*_${SUF}_*.pkl 2>/dev/null | wc -l) Datei(en)"
