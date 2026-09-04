#!/usr/bin/env bash
# Holt die nach der ECMWF-Reparatur (2026-09-04) neu gerechneten Testjahr-Ergebnisse
# von l1 nach l2 und prueft die Vollstaendigkeit.
#
# Auf l2 ausfuehren:  bash scripts/collect_testmode_rerun.sh
#
# Die beiden s1-Arme werden NICHT neu gerechnet: ihr Trainings- und
# Auswertungsfenster (Aug-Nov 2025) liegt vollstaendig vor dem ECMWF-Defekt.
# Die alten, defekten Ausgaben liegen unter archiv/testmode_ecmwf_defekt_20260904/.
set -uo pipefail
REPO=/home/viktor/Work/forecasting_framework
L1=/home/viktorwalter/Work/forecasting_framework
cd "$REPO" || exit 1

NEU="testmode_dcrnn testmode_dcrnn_idw_alt testmode_mtgnn_nwp
     testmode_dcrnn_nwp_hist_s2 testmode_mtgnn_nwp_hist_s2
     testmode_dcrnn_nwp_hist_s3 testmode_mtgnn_nwp_hist_s3"
ALT="testmode_dcrnn_nwp_hist_s1 testmode_mtgnn_nwp_hist_s1"

fehlt=0
for a in $NEU; do
  if rsync -a --info=name0 "l1:$L1/data/raw_preds/${a}_raw.parquet" data/raw_preds/ 2>/dev/null; then
    rsync -a "l1:$L1/data/test_results/${a}.csv" data/test_results/ 2>/dev/null || true
    echo "  geholt: $a"
  else
    echo "  FEHLT auf l1: ${a}_raw.parquet"; fehlt=$((fehlt+1))
  fi
done

echo
echo "== Bestand der neun Testjahr-Parquets auf l2"
for a in $NEU $ALT; do
  f="data/raw_preds/${a}_raw.parquet"
  if [ -f "$f" ]; then
    printf "  %-34s %s\n" "$a" "$(date -r "$f" +%Y-%m-%d\ %H:%M)"
  else
    printf "  %-34s FEHLT\n" "$a"; fehlt=$((fehlt+1))
  fi
done
for r in icon_d2_test_fold7 ecmwf_test_fold7; do
  f="data/raw_preds/${r}_raw.parquet"
  [ -f "$f" ] && printf "  %-34s %s\n" "$r" "$(date -r "$f" +%Y-%m-%d\ %H:%M)" || { printf "  %-34s FEHLT\n" "$r"; fehlt=$((fehlt+1)); }
done

echo
if [ "$fehlt" -eq 0 ]; then
  echo "Vollstaendig. Naechster Schritt:"
  echo "  ./frcst/bin/python scripts/export_paper_metrics.py test"
else
  echo "$fehlt Datei(en) fehlen — Export noch nicht starten."
fi
