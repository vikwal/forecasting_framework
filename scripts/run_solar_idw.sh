#!/usr/bin/env bash
# IDW-Test: drei Varianten der Gitterpunkt-Aggregation, je N Wiederholungen,
# verteilt auf die GPUs aller drei Hosts.
#
#   scripts/run_solar_idw.sh <SUFFIX> [WIEDERHOLUNGEN]
#
# Wiederholungen, weil ein einzelner Lauf die Lauf-zu-Lauf-Streuung (~0.45 W/m²)
# nicht von einem echten Effekt trennen kann.
set -u
REPO_L2="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_L2" || exit 1
SUF="${1:?Suffix fehlt, z.B. v1}"; N="${2:-3}"

VARIANTEN=(nearest idw4 idw9)
SLOTS=(lokal:0 lokal:1 lokal:2 lokal:3 l1:0 l1:1 l1:2 l1:3 ws:0 ws:1)
dataroot_fuer() { case "$1" in l1) echo /mnt/nvme1;; *) echo /mnt/lambda1/nvme1;; esac; }
repo_fuer()     { case "$1" in lokal) echo "$REPO_L2";; *) echo '$HOME/Work/forecasting_framework';; esac; }

JOBS=()
for v in "${VARIANTEN[@]}"; do
    for i in $(seq 1 "$N"); do JOBS+=("$v:$i"); done
done
echo "${#JOBS[@]} Laeufe (${#VARIANTEN[@]} Varianten x $N) auf ${#SLOTS[@]} Slots"

for i in "${!SLOTS[@]}"; do
    slot="${SLOTS[$i]}"; host="${slot%%:*}"; gpu="${slot##*:}"
    meine=()
    for j in "${!JOBS[@]}"; do
        [ $((j % ${#SLOTS[@]})) -eq "$i" ] && meine+=("${JOBS[$j]}")
    done
    [ ${#meine[@]} -eq 0 ] && continue
    repo="$(repo_fuer "$host")"; droot="$(dataroot_fuer "$host")"
    rumpf="cd $repo && mkdir -p logs/solar_idw && \
eval \"\$(grep -E '^export (WEATHER_DB_URL|OPTUNA_STORAGE)=' ~/.bashrc)\" && \
export WEATHER_DB_URL OPTUNA_STORAGE DATA_ROOT=$droot && \
for job in ${meine[*]}; do \
  v=\${job%%:*}; r=\${job##*:}; \
  log=logs/solar_idw/\${v}_r\${r}_${SUF}.log; \
  echo \"== \$(date -Is) \$v Wdh \$r gpu=$gpu\" > \$log; \
  CUDA_VISIBLE_DEVICES=$gpu frcst/bin/python train_cl.py -m tft \
      -c configs/solar_idw/config_solar_idw_\${v}.yaml \
      -s ${SUF}_\${v}_r\${r} --save-predictions >> \$log 2>&1; \
  echo \"== \$(date -Is) fertig, exit \$?\" >> \$log; \
done"
    if [ "$host" = lokal ]; then setsid nohup bash -c "$rumpf" > /dev/null 2>&1 &
    else setsid nohup ssh "$host" "$rumpf" > /dev/null 2>&1 & fi
    echo "  $slot: ${meine[*]}"
    sleep 2
done
