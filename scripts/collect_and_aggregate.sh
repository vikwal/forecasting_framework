#!/usr/bin/env bash
# Sammelt die neun Rohvorhersage-Parquets von ws und l1 nach l2 und rechnet,
# sobald alle da sind, die gefilterte Papiertabelle. Laeuft unbeaufsichtigt.
REPO=/home/viktor/Work/forecasting_framework
DEST=$REPO/data/raw_preds
OUT=$REPO/docs/filtered_table_retrains.txt
LOG=$REPO/logs/collect_aggregate.log
WS=/home/viktor/Work/forecasting_framework/data/raw_preds
L1=/home/viktorwalter/Work/forecasting_framework/data/raw_preds

log(){ echo "$(date -Is) $*" >> "$LOG"; }
mkdir -p "$DEST"
log "Sammler startet"

for i in $(seq 1 400); do
  for f in 1 2 3; do
    for arm in dcrnn dcrnn_base; do
      n="retrain_${arm}_fold${f}_raw.parquet"
      [ -f "$DEST/$n" ] || scp -q -o ConnectTimeout=20 ws:"$WS/$n" "$DEST/" 2>/dev/null && :
    done
    n="retrain_dcrnn_idw_alt_fold${f}_raw.parquet"
    [ -f "$DEST/$n" ] || scp -q -o ConnectTimeout=20 l1:"$L1/$n" "$DEST/" 2>/dev/null && :
  done
  have=$(ls "$DEST"/retrain_*_raw.parquet 2>/dev/null | wc -l)
  log "vorhanden: $have / 9"
  if [ "$have" -ge 9 ]; then
    log "alle neun da, rechne gefilterte Tabelle"
    cd "$REPO" && source frcst/bin/activate
    python /home/viktor/Work/forecasting_framework/misc/filtered_table.py "$REPO" dcrnn,dcrnn_base,dcrnn_idw_alt \
      > "$OUT" 2>&1
    log "fertig -> $OUT"
    exit 0
  fi
  sleep 300
done
log "Sammler-Timeout, nur $have von 9 Parquets"
exit 1
