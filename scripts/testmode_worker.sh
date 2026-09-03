#!/usr/bin/env bash
# testmode_worker.sh <WORKER_ID> <GPU>
# Finale Testauswertung (--test-mode): Training auf files+val_files (153 Stationen),
# Zero-Shot-Eval auf test_files (50).
#   * 3 Arme ohne NWP-Historie: ein Modell, Test 2025-08-01 .. 2026-07-31
#   * 2 HIST-Arme: 3 Expanding-Window-Retrains, je 4 Monate Test
#
#   screen -dmS tm_w1 bash ~/queue_scripts/testmode_worker.sh 1 0
#
# Hostunabhaengig ueber $HOME — l1 hat /home/viktorwalter, l2 und ws /home/viktor.
# Die Aufgabenliste wird in JEDER Runde neu gelesen; Aufgaben duerfen im Betrieb
# ANGEHAENGT werden (nur anhaengen, nie umsortieren — der Cursor ist ein
# globaler Index). Ist die Liste abgearbeitet, wartet der Worker statt zu enden.
# Beenden: touch ~/queue_scripts/.testmode_stop
#
# Schutz gegen Massenabbrueche (am 2026-09-02 hat etwas ausserhalb dieser Queue
# alle Prozesse auf l1 mit SIGTERM beendet): ein fehlgeschlagener Lauf wird ans
# Listenende zurueckgehaengt statt verworfen, danach pausiert der Worker. Drei
# Fehlschlaege in Folge setzen die STOP-Datei und beenden ALLE Worker dieses
# Hosts, damit ein systematischer Fehler nicht die ganze Liste verbrennt.
set -u
WID="${1:?worker id}"; GPU="${2:?gpu}"
REPO="$HOME/Work/forecasting_framework"
QDIR="$HOME/queue_scripts"
TASKS="$QDIR/testmode_tasks.txt"; CURSOR="$QDIR/.testmode_cursor"
LOCK="$QDIR/.testmode_lock"; STATUS="$QDIR/testmode_status.log"; STOP="$QDIR/.testmode_stop"
IDLE_SLEEP=120
FAIL_SLEEP=120
MAX_CONSEC_FAILS=3

cd "$REPO" || exit 1

# .bashrc kehrt auf ws in nicht-interaktiven Shells frueh zurueck (Debian-Standard),
# ein blosses `source` liefert die Exporte dort nicht. Deshalb gezielt nur die
# beiden benoetigten Zeilen auswerten — funktioniert auf allen drei Hosts.
if [[ -z "${OPTUNA_STORAGE:-}" || -z "${WEATHER_DB_URL:-}" ]]; then
  set +u
  eval "$(grep -E '^[[:space:]]*export (OPTUNA_STORAGE|WEATHER_DB_URL)=' "$HOME/.bashrc" 2>/dev/null)" || true
  set -u
fi
for v in OPTUNA_STORAGE WEATHER_DB_URL; do
  if [[ -z "${!v:-}" ]]; then
    echo "$(date '+%F %T') [w$WID gpu$GPU] ABBRUCH: $v ist nicht gesetzt" | tee -a "$STATUS"
    exit 1
  fi
done

source frcst/bin/activate
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p logs/testmode "$QDIR"
[[ -f "$CURSOR" ]] || echo 0 > "$CURSOR"
[[ -f "$LOCK" ]] || : > "$LOCK"
say() { echo "$(date '+%F %T') [$(hostname -s) w$WID gpu$GPU] $*" | tee -a "$STATUS"; }
CONSEC_FAILS=0
say "Worker gestartet"

while :; do
  [[ -f "$STOP" ]] && { say "STOP-Datei — Worker endet"; break; }
  mapfile -t JOBS < <(grep -vE "^[[:space:]]*(#|$)" "$TASKS")
  NJOBS=${#JOBS[@]}
  IDX=$(flock "$LOCK" bash -c "i=\$(cat '$CURSOR'); if (( i < $NJOBS )); then echo \$((i+1)) > '$CURSOR'; fi; echo \$i")
  if (( IDX >= NJOBS )); then
    say "Liste abgearbeitet ($NJOBS) — warte ${IDLE_SLEEP}s auf Nachschub"
    sleep "$IDLE_SLEEP"; continue
  fi
  LINE="${JOBS[$IDX]}"
  read -r ARM MODEL SUB SUF RAW <<< "$LINE"
  CFG="configs/testmode/${SUB}/config_wind_${ARM}_fold1.yaml"
  MNAME="wind_${ARM}_fold1_${MODEL}_${SUF}"
  JOBLOG="logs/testmode/${RAW}.log"
  [[ -f "$CFG" ]] || { say "FEHLT: $CFG (#$IDX)"; continue; }

  say "START #$IDX $ARM/$SUB → $JOBLOG"
  T0=$(date +%s)
  { echo "== $(date -Is) TRAIN $MNAME  cfg=$CFG  host=$(hostname -s) gpu=$GPU"
    python "geostatistics/train_${MODEL}.py" --config "$CFG" --suffix "$SUF" \
      --hpo-study auto --test-mode; } >> "$JOBLOG" 2>&1
  RC_T=$?
  if (( RC_T != 0 )); then
    CONSEC_FAILS=$(( CONSEC_FAILS + 1 ))
    flock "$LOCK" bash -c "printf '%s\n' \"$LINE\" >> '$TASKS'"
    say "FEHLER #$IDX train exit=$RC_T ($ARM/$SUB) — zurueckgehaengt, Fehler in Folge: $CONSEC_FAILS"
    if (( CONSEC_FAILS >= MAX_CONSEC_FAILS )); then
      touch "$STOP"; say "ABBRUCH: $CONSEC_FAILS Fehler in Folge — STOP-Datei gesetzt"; break
    fi
    sleep "$FAIL_SLEEP"; continue
  fi
  T1=$(date +%s)

  { echo "== $(date -Is) EVAL $MNAME"
    python "geostatistics/get_test_results_${MODEL}.py" -m "$MNAME" -c "$CFG" \
      --hpo-study auto --test-mode --raw-out-name "$RAW"; } >> "$JOBLOG" 2>&1
  RC_E=$?
  if (( RC_E != 0 )); then
    CONSEC_FAILS=$(( CONSEC_FAILS + 1 ))
    say "FEHLER #$IDX eval exit=$RC_E ($ARM/$SUB) — Modell liegt vor, Eval nachholen"
  else
    CONSEC_FAILS=0
  fi
  DT=$(( $(date +%s) - T0 )); DTT=$(( T1 - T0 ))
  say "FERTIG #$IDX $ARM/$SUB train=$RC_T eval=$RC_E ${DT}s (train ${DTT}s)"
done
say "Worker $WID beendet"
