#!/usr/bin/env bash
# Startet EINEN WAVENET-HPO-Worker in der laufenden Shell (fuer `screen -dmS`).
#
# Die DB-URLs werden zur LAUFZEIT aus ~/.bashrc gelesen statt in die
# Kommandozeile geschrieben. Sonst steht die Postgres-URL im Klartext in der
# `ps`-Ausgabe und ist fuer jeden Nutzer des Hosts lesbar. `ssh host '...'` ist
# eine nicht-interaktive Shell, in der ~/.bashrc vor den export-Zeilen abbricht,
# deshalb der explizite grep.
#
#   scripts/launch_wavenet_worker.sh <REPO> <CONFIG> <GPU> <SUFFIX>
#
# Beispiel:
#   screen -dmS hpo_wavenet_nwp_r2 bash scripts/launch_wavenet_worker.sh \
#       "$PWD" configs/wavenet/config_wind_wavenet.yaml 1 r2
REPO="$1"; CFG="$2"; GPU="$3"; SUF="$4"
cd "$REPO" || { echo "REPO nicht gefunden: $REPO"; exec bash; }
source frcst/bin/activate
eval "$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE)=' ~/.bashrc)"
: "${WEATHER_DB_URL:?WEATHER_DB_URL fehlt}"
: "${ECMWF_WIND_SL_URL:?ECMWF_WIND_SL_URL fehlt}"
: "${OPTUNA_STORAGE:?OPTUNA_STORAGE fehlt}"
echo "== $(date -Is) $CFG gpu=$GPU suffix=$SUF commit=$(git rev-parse --short HEAD)"
python geostatistics/hpo_wavenet.py --config "$CFG" --gpu "$GPU" --suffix "$SUF"
echo "== Worker beendet, Exit $?"
exec bash
