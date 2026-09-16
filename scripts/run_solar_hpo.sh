#!/usr/bin/env bash
# Startet die Worker der Solar-TFT-HPO auf den GPUs aller drei Hosts.
#
#   scripts/run_solar_hpo.sh [SUFFIX]
#
# Alle Worker ziehen Trials aus derselben Optuna-Studie in PostgreSQL
# (OPTUNA_STORAGE, auf l1/ws ueber 10.166.32.237) und beenden sich, sobald
# hpo.trials erreicht ist. Parallelbetrieb ist damit unkritisch — anders als
# bei einer SQLite-Datei ueber NFS.
#
# Das Cache-Budget steht je Host, weil der Plattenplatz sich stark
# unterscheidet (l1 hat deutlich weniger frei als l2 und ws) und jede
# Kombination aus next_n_* und optionalen Features einen eigenen Cache-Eintrag
# bekommt.
set -u
REPO_L2="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_L2" || exit 1

SUF="${1:-}"
CFG="configs/solar_tft/config_solar_tft_hpo.yaml"
[ -f "$CFG" ] || { echo "Config fehlt: $CFG"; exit 1; }

#: host:gpu — auf l1 tragen 4-7 derzeit Fremdlast eines anderen Nutzers,
#: die bleiben frei. Die Belegung wandert, vor einem Neustart pruefen.
SLOTS=(lokal:0 lokal:1 lokal:2 lokal:3 l1:0 l1:1 l1:2 l1:3 ws:0 ws:1)

dataroot_fuer() { case "$1" in l1) echo /mnt/nvme1;; *) echo /mnt/lambda1/nvme1;; esac; }
cachegb_fuer()  { case "$1" in l1) echo 80;; ws) echo 300;; *) echo 300;; esac; }
repo_fuer()     { case "$1" in lokal) echo "$REPO_L2";; *) echo '$HOME/Work/forecasting_framework';; esac; }

echo "Studie: $(basename $CFG .yaml)${SUF:+ (Suffix $SUF)} — ${#SLOTS[@]} Worker"
for slot in "${SLOTS[@]}"; do
    host="${slot%%:*}"; gpu="${slot##*:}"
    repo="$(repo_fuer "$host")"; droot="$(dataroot_fuer "$host")"; cgb="$(cachegb_fuer "$host")"
    log="logs/hpo_solar/w_${host}_g${gpu}.log"

    rumpf="cd $repo && mkdir -p logs/hpo_solar && \
eval \"\$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE)=' ~/.bashrc)\" && \
export WEATHER_DB_URL ECMWF_WIND_SL_URL OPTUNA_STORAGE DATA_ROOT=$droot && \
echo \"== \$(date -Is) start gpu=$gpu host=$host commit=\$(git rev-parse --short HEAD)\" >> $log && \
frcst/bin/python hpo_tft_bc.py -c $CFG --gpu $gpu --max-cache-gb $cgb ${SUF:+-s $SUF} >> $log 2>&1; \
echo \"== \$(date -Is) Worker beendet, Exit \$?\" >> $log"

    if [ "$host" = lokal ]; then
        setsid nohup bash -c "$rumpf" > /dev/null 2>&1 &
    else
        setsid nohup ssh "$host" "$rumpf" > /dev/null 2>&1 &
    fi
    echo "  $slot  (Cache-Budget ${cgb} GB, DATA_ROOT $droot)"
    sleep 3
done
echo "Logs: logs/hpo_solar/w_<host>_g<gpu>.log auf dem jeweiligen Host"
echo "Fortschritt: optuna-dashboard (Port 8504) oder scripts/hpo_watch_restart.py"
