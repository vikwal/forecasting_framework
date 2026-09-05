#!/usr/bin/env bash
# Holt die am 2026-09-05 angestossenen Ergebnisse nach l2 (auf l2 ausfuehren), dann Export:
#   bash scripts/collect_testyear_20260905.sh && ./frcst/bin/python scripts/export_paper_metrics.py all
# l1: einmal trainierte HIST-Arme ueber das volle Testjahr (testmode_*_once)
# ws: Seed-Wiederholungen A/D' Validierungsjahr (rep{2,3}_dcrnn[_idw_alt]_fold{1,2,3})
# l2 selbst: MOS Testjahr (testyear_mos_*), TFT Testjahr (testmode_tft_*)
set -uo pipefail
REPO=/home/viktor/Work/forecasting_framework
L1=/home/viktorwalter/Work/forecasting_framework
WS=/home/viktor/Work/forecasting_framework
cd "$REPO" || exit 1
echo "== von l1"
for a in testmode_dcrnn_nwp_hist_once testmode_mtgnn_nwp_hist_once; do
  rsync -a --info=name0 "l1:$L1/data/raw_preds/${a}_raw.parquet" data/raw_preds/ 2>/dev/null && echo "  geholt: $a" || echo "  fehlt auf l1: $a"
  rsync -a "l1:$L1/data/test_results/${a}.csv" data/test_results/ 2>/dev/null || true
done
echo "== von ws"
for r in 2 3; do for arm in dcrnn dcrnn_idw_alt; do for f in 1 2 3; do a="rep${r}_${arm}_fold${f}"
  rsync -a --info=name0 "ws:$WS/data/raw_preds/${a}_raw.parquet" data/raw_preds/ 2>/dev/null && echo "  geholt: $a" || echo "  fehlt auf ws: $a"
done; done; done
echo "== Bestand auf l2 (neue Arme)"
ls -la --time-style=+%F\ %H:%M data/raw_preds/ | grep -E "_once_raw|testyear_mos|testmode_tft|rep[23]_" | awk '{print "  "$6" "$7" "$8}'
