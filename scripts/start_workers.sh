#!/usr/bin/env bash
# start_workers.sh <spec-datei>
#
# Jede Zeile der Spec:  <modell> <config> <gpu> <suffix>
# Beispiel:             mtgnn configs/mtgnn/config_wind_mtgnn_nwp.yaml 1 n1
#
# Prueft vorher: Config existiert, Logname frei, Screen-Name frei, kein Worker
# mit derselben Config/GPU/Suffix-Kombination. Startet nur, wenn alles passt.
set -u
REPO="$HOME/Work/forecasting_framework"
# die launch_*_worker.sh liegen neben diesem Skript in scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC="$1"
COMMIT="${COMMIT:-0}"
cd "$REPO"

# Prozess- und Screen-Liste EINMAL vorab aufnehmen. Wuerde man in der Schleife
# `ps | grep muster` schreiben, findet grep seine eigene Kommandozeile in der
# ps-Ausgabe und jede Pruefung meldet faelschlich "laeuft schon".
PSOUT=$(ps -eo args --no-headers)
SCROUT=$(screen -ls || true)

fail=0
while read -r model cfg gpu suf; do
  [ -z "${model:-}" ] && continue
  case "$model" in \#*) continue;; esac
  stem=$(basename "$cfg" .yaml | sed 's/^config_//')
  log="logs/hpo_${model}_${stem}_${suf}.log"
  name="hpo_${model}_${stem}_${suf}"
  launcher="$SCRIPT_DIR/launch_${model}_worker.sh"
  problem=""
  [ -f "$cfg" ]      || problem="$problem Config-fehlt"
  [ -f "$launcher" ] || problem="$problem Starter-fehlt"
  [ -e "$log" ]      && problem="$problem Log-belegt"
  printf '%s\n' "$SCROUT" | grep -q "[.]${name}[[:space:]]" && problem="$problem Screen-belegt"
  printf '%s\n' "$PSOUT" | grep -qF -- "--config $cfg --gpu $gpu --suffix $suf" \
      && problem="$problem laeuft-schon"
  if [ -n "$problem" ]; then
    echo "FEHLER $name:$problem"
    fail=1
  else
    echo "ok     $name  gpu=$gpu  ->  $log"
  fi
done < "$SPEC"

[ "$fail" -ne 0 ] && { echo "Abbruch, nichts gestartet."; exit 1; }

if [ "$COMMIT" != "1" ]; then
  echo "Trockenlauf. COMMIT=1 setzen."
  exit 0
fi

while read -r model cfg gpu suf; do
  [ -z "${model:-}" ] && continue
  case "$model" in \#*) continue;; esac
  stem=$(basename "$cfg" .yaml | sed 's/^config_//')
  name="hpo_${model}_${stem}_${suf}"
  screen -dmS "$name" bash "$SCRIPT_DIR/launch_${model}_worker.sh" "$REPO" "$cfg" "$gpu" "$suf"
  echo "gestartet: $name"
  sleep 3
done < "$SPEC"
