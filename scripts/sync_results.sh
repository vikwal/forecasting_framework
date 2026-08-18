#!/usr/bin/env bash
# Spiegelt Auswertungsergebnisse von l1 und ws nach l2, damit fold_evaluation.ipynb
# alle Arme sieht. Je Host ein Unterverzeichnis statt Ueberschreiben: ecmwf_test_fold0.csv
# und icon_d2_test_fold0.csv heissen auf l1 und l2 gleich, haben aber anderen Inhalt.
#
# Auf l2 ausfuehren:  bash scripts/sync_results.sh
set -uo pipefail
REPO=/home/viktor/Work/forecasting_framework
L1=/home/viktorwalter/Work/forecasting_framework
WS=/home/viktor/Work/forecasting_framework

for sub in test_results raw_preds; do
  for host in l1 ws; do
    src=$([ "$host" = l1 ] && echo "$L1" || echo "$WS")
    dst="$REPO/data/$sub/from_$host"
    mkdir -p "$dst"
    echo "== $sub von $host"
    rsync -a --info=stats1 \
      --include="*.csv" --include="*.parquet" --exclude="*" \
      "$host:$src/data/$sub/" "$dst/" 2>&1 | grep -E "Number of regular files transferred|total size" | sed 's/^/   /'
  done
done

echo
echo "== Bestand auf l2"
for sub in test_results raw_preds; do
  for d in "$REPO/data/$sub" "$REPO/data/$sub/from_l1" "$REPO/data/$sub/from_ws"; do
    n=$(ls "$d"/*.csv "$d"/*.parquet 2>/dev/null | wc -l)
    printf "   %-46s %3s Dateien\n" "${d#$REPO/}" "$n"
  done
done
