#!/usr/bin/env bash
# Verteilt die Laeufe des Lokal-gegen-Global-Vergleichs auf die GPUs aller
# drei Hosts. Jeder Slot arbeitet seine Jobs nacheinander ab.
#
#   scripts/run_solar_lokal.sh <SUFFIX> [--nur-lokal | --nur-global]
#
# Die Slots stehen in SLOTS als "<host>:<gpu>". 'lokal' heisst: auf l2 direkt,
# sonst ueber ssh. DATA_ROOT unterscheidet sich je Host (auf l1 liegen die
# Daten lokal), deshalb wird es je Slot gesetzt statt vererbt.
#
# Der globale Lauf sieht 62 statt einer Station und dauert entsprechend
# laenger — er startet zuerst und bekommt einen Slot fuer sich.
set -u
REPO_L2="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_L2" || exit 1

SUF="${1:?Suffix fehlt, z.B. v1}"; shift || true
WAS="${1:-alles}"

#: host:gpu — auf l1 tragen 0 und 5-7 Fremdlast, die bleiben frei.
SLOTS=(lokal:0 lokal:1 lokal:2 lokal:3 l1:1 l1:2 l1:3 l1:4 ws:0 ws:1)

repo_fuer() { case "$1" in lokal) echo "$REPO_L2";; *) echo '$HOME/Work/forecasting_framework';; esac; }
dataroot_fuer() { case "$1" in l1) echo /mnt/nvme1;; *) echo /mnt/lambda1/nvme1;; esac; }

# ── Jobliste: global zuerst, dann die lokalen ────────────────────────────
JOBS=()
[ "$WAS" != "--nur-lokal" ] && JOBS+=("configs/solar_lokal/config_solar_global_trans.yaml")
if [ "$WAS" != "--nur-global" ]; then
    for c in configs/solar_lokal/config_solar_lokal_*.yaml; do JOBS+=("$c"); done
fi
echo "${#JOBS[@]} Laeufe auf ${#SLOTS[@]} Slots"

# ── Jobs reihum auf die Slots verteilen ──────────────────────────────────
for i in "${!SLOTS[@]}"; do
    slot="${SLOTS[$i]}"; host="${slot%%:*}"; gpu="${slot##*:}"
    meine=()
    for j in "${!JOBS[@]}"; do
        [ $((j % ${#SLOTS[@]})) -eq "$i" ] && meine+=("${JOBS[$j]}")
    done
    [ ${#meine[@]} -eq 0 ] && continue

    repo="$(repo_fuer "$host")"
    droot="$(dataroot_fuer "$host")"
    # Schleife als eine Zeichenkette, damit sie auch ueber ssh geht.
    liste="${meine[*]}"
    rumpf="cd $repo && mkdir -p logs/solar_lokal && \
eval \"\$(grep -E '^export (WEATHER_DB_URL|OPTUNA_STORAGE)=' ~/.bashrc)\" && \
export WEATHER_DB_URL OPTUNA_STORAGE DATA_ROOT=$droot && \
for cfg in $liste; do \
  name=\$(basename \$cfg .yaml | sed 's/^config_solar_//'); \
  log=logs/solar_lokal/\${name}_${SUF}.log; \
  echo \"== \$(date -Is) \$name gpu=$gpu\" > \$log; \
  CUDA_VISIBLE_DEVICES=$gpu frcst/bin/python train_cl.py -m tft -c \$cfg \
      -s ${SUF}_\$name --save-predictions >> \$log 2>&1; \
  echo \"== \$(date -Is) fertig, exit \$?\" >> \$log; \
done"

    if [ "$host" = lokal ]; then
        setsid nohup bash -c "$rumpf" > /dev/null 2>&1 &
    else
        setsid nohup ssh "$host" "$rumpf" > /dev/null 2>&1 &
    fi
    echo "  $slot: ${#meine[@]} Laeufe"
done
echo "Logs: logs/solar_lokal/<name>_${SUF}.log (auf dem jeweiligen Host)"
