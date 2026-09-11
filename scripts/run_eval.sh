#!/usr/bin/env bash
# Auswertung EINES Retrain-Laufs im Entwicklungsmodus (51 nie gesehene
# Zielstationen je Fold), also genau der Pfad, fuer den N1 gefixt wurde.
#
#   run_eval.sh <REPO> <ARM> <FOLD> <GPU>
#
# Die Architektur wird aus dem Trainings-pkl rekonstruiert (--pkl), nicht ueber
# --hpo-study auto: das pkl ist eingefroren, der Optuna-Bestwert kann sich
# verschieben. KEIN --test-mode.
set -uo pipefail

REPO="$1"; ARM="$2"; FOLD="$3"; GPU="$4"
TAG="${ARM}_fold${FOLD}"
STEM="wind_${ARM}_fold${FOLD}_dcrnn_retrain_fold${FOLD}"
CFG="configs/dcrnn/config_wind_${ARM}_fold${FOLD}.yaml"
LOG="logs/eval_${TAG}.log"

cd "$REPO" || exit 1
source frcst/bin/activate
eval "$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE|DATA_ROOT)=' ~/.bashrc)"

PKL=$(ls -t results/${STEM}_*.pkl 2>/dev/null | head -1)
[ -n "$PKL" ] || { echo "kein pkl fuer $STEM"; exit 1; }

{
  echo "== $(date -Is) START eval $TAG"
  echo "== host=$(hostname) gpu=$GPU commit=$(git rev-parse --short HEAD)"
  echo "== model=$STEM"
  echo "== pkl=$PKL"
} >> "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" python geostatistics/get_test_results_dcrnn.py \
    -m "$STEM" -c "$CFG" --pkl "$PKL" \
    --raw-out-name "retrain_${TAG}" >> "$LOG" 2>&1
rc=$?
echo "== $(date -Is) ENDE eval $TAG exit=$rc" >> "$LOG"
exit $rc
