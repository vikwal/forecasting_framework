#!/usr/bin/env bash
# Startet EINEN MTGNN-HPO-Worker in der laufenden Shell (fuer `screen -dmS`).
# Generisch, im Gegensatz zu launch_mtgnn_r4.sh, das das Suffix hart verdrahtet.
# DB-URLs zur Laufzeit aus ~/.bashrc, damit die Postgres-URL nicht in `ps` steht.
#
#   launch_mtgnn_worker.sh <REPO> <CONFIG> <GPU> <SUFFIX>
REPO="$1"; CFG="$2"; GPU="$3"; SUF="$4"
cd "$REPO" || { echo "REPO nicht gefunden: $REPO"; exec bash; }
source frcst/bin/activate
eval "$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE)=' ~/.bashrc)"
: "${WEATHER_DB_URL:?WEATHER_DB_URL fehlt}"
: "${ECMWF_WIND_SL_URL:?ECMWF_WIND_SL_URL fehlt}"
: "${OPTUNA_STORAGE:?OPTUNA_STORAGE fehlt}"
echo "== $(date -Is) $CFG gpu=$GPU suffix=$SUF commit=$(git rev-parse --short HEAD)"
python geostatistics/hpo_mtgnn.py --config "$CFG" --gpu "$GPU" --suffix "$SUF"
echo "== Worker beendet, Exit $?"
exec bash
