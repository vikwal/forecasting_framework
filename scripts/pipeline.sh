#!/usr/bin/env bash
# Arbeitet Schritte NACHEINANDER auf EINER GPU ab, unbeaufsichtigt.
#
#   pipeline.sh <REPO> <GPU> <schritt> [<schritt> ...]
#   schritt = train:<arm>:<fold>  |  eval:<arm>:<fold>
#
# Faellt ein Schritt aus, wird das protokolliert und der naechste trotzdem
# versucht: ein kaputter Lauf soll die uebrigen nicht blockieren. Die
# Schlusspruefung (verify_retrains.py) faengt alles Fehlerhafte ohnehin ab.
set -uo pipefail

REPO="$1"; shift
GPU="$1"; shift
LOG="$REPO/logs/pipeline_gpu${GPU}.log"
log() { echo "$(date -Is) [gpu${GPU}] $*" >> "$LOG"; }

log "Pipeline startet mit $# Schritten: $*"
fails=0
for step in "$@"; do
  action="${step%%:*}"; rest="${step#*:}"
  arm="${rest%%:*}"; fold="${rest##*:}"
  case "$action" in
    train) script=run_retrain.sh ;;
    eval)  script=run_eval.sh ;;
    *) log "UNBEKANNTER Schritt '$step', uebersprungen"; continue ;;
  esac
  log "START $action $arm fold$fold"
  if [ "$action" = "train" ]; then
    bash "$HOME/$script" "$arm" "$fold" "$GPU"
  else
    bash "$HOME/$script" "$REPO" "$arm" "$fold" "$GPU"
  fi
  rc=$?
  log "ENDE  $action $arm fold$fold exit=$rc"
  [ "$rc" -ne 0 ] && fails=$((fails+1))
done
log "Pipeline fertig, $fails Schritt(e) mit Fehler"
