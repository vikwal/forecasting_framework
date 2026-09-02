#!/usr/bin/env bash
# Ersetzt die globale .hpo_stop durch suffixgenaue Dateien fuer die aktuell
# laufenden Worker. Damit laufen die Alten nach ihrem Trial aus, neue Worker mit
# frischem Suffix sind davon nicht betroffen.
set -e
cd ~/Work/forecasting_framework

SUF=$(ps -eo args --no-headers \
      | grep -oE "^python geostatistics/hpo_(dcrnn|mtgnn|wavenet)\.py .*--suffix [A-Za-z0-9]+" \
      | grep -oE "suffix [A-Za-z0-9]+$" | awk '{print $2}' | sort -u)

if [ -z "$SUF" ]; then
  echo "keine laufenden Worker gefunden"
else
  echo "laufende Suffixe: $(echo $SUF | tr '\n' ' ')"
  for s in $SUF; do
    touch ".hpo_stop_$s"
    echo "  .hpo_stop_$s angelegt"
  done
fi

rm -f .hpo_stop && echo "globale .hpo_stop entfernt"
ls -1 .hpo_stop* 2>/dev/null | sed 's/^/  /' || echo "  (keine Stopp-Dateien)"
