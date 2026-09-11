#!/usr/bin/env bash
# status.sh -- one-screen training status for l1, l2 and ws.
#
#   bash scripts/status.sh            all hosts (runs remotes over ssh, in parallel)
#   bash scripts/status.sh --local    this host only (what the remotes execute)
#
# Host-agnostic via $HOME. The script is streamed to the remotes over ssh, so it
# does not need to be checked out there. Sessions: tmux first (current
# convention), screen second (legacy launchers still use it).
set -u
REPO="$HOME/Work/forecasting_framework"
QDIR="$HOME/queue_scripts"

report() {
  local host; host=$(hostname)
  echo "################  $host  ($(date '+%F %H:%M'))"

  echo "-- git"
  if [ -d "$REPO/.git" ]; then
    printf "   HEAD %s  |  %s geaenderte Dateien\n" \
      "$(git -C "$REPO" log --oneline -1 | cut -c1-60)" "$(git -C "$REPO" status --porcelain | wc -l)"
  else echo "   (kein Repo unter $REPO)"; fi

  echo "-- gpus"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{printf "   gpu%-2s %3s%%  %6d/%6d MiB\n",$1,$2,$3,$4}' || echo "   (nvidia-smi nicht verfuegbar)"

  echo "-- sessions"
  local t s
  t=$(tmux ls 2>/dev/null | sed 's/^/   tmux   /'); s=$(screen -ls 2>/dev/null | grep -E '^\s+[0-9]+\.' | sed -E 's/^\s+[0-9]+\.//; s/^/   screen /')
  [ -n "$t$s" ] && printf '%s\n' "$t" "$s" | sed '/^$/d' || echo "   (keine)"

  # gpu per pid via nvidia-smi compute-apps (does not rely on CUDA_VISIBLE_DEVICES);
  # also lists other users' jobs, so "which GPU is free for me" is answerable
  declare -A gpu_of; local uuidmap foreign=""
  uuidmap=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)
  while IFS=', ' read -r pid uuid mem; do
    [ -z "$pid" ] && continue
    idx=$(grep "$uuid" <<<"$uuidmap" | cut -d, -f1); gpu_of[$pid]=$idx
    local owner; owner=$(ps -o user= -p "$pid" 2>/dev/null || echo '?')
    [ "$owner" != "$USER" ] && foreign+=$(printf "   gpu%-2s %-14s pid %-8s %6s MiB  %s\n" "$idx" "$owner" "$pid" "$mem" "$(ps -o comm= -p "$pid" 2>/dev/null)")
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  if [ -n "$foreign" ]; then echo "-- fremde gpu-prozesse"; printf '%b' "$foreign"; fi

  echo "-- prozesse (train_/hpo_)"
  local any=0
  while read -r pid rest; do
    any=1
    local et; et=$(ps -o etime= -p "$pid" | tr -d ' ')
    local cfg; cfg=$(grep -oE -- '(-c|--config) [^ ]+' <<<"$rest" | awk '{print $2}' | head -1 | xargs -r basename)
    printf "   pid %-8s gpu %-3s %10s  %s  %s\n" "$pid" "${gpu_of[$pid]:-?}" "$et" "$(awk '{print $2}' <<<"$rest" | xargs -r basename)" "${cfg:-}"
  done < <(pgrep -af 'python[0-9.]* .*(train_|hpo_)' | grep -v pgrep)
  [ $any = 0 ] && echo "   (keine)"

  if [ -f "$QDIR/testmode_tasks.txt" ]; then
    echo "-- queue"
    printf "   cursor %s / %s tasks   stop-file: %s\n" "$(cat "$QDIR/.testmode_cursor" 2>/dev/null || echo -)" \
      "$(grep -cv '^#' "$QDIR/testmode_tasks.txt")" "$([ -e "$QDIR/.testmode_stop" ] && echo JA || echo nein)"
    tail -1 "$QDIR/testmode_status.log" 2>/dev/null | cut -c1-110 | sed 's/^/   /'
  fi

  echo "-- logs (letzte 3 h)"
  local n=0
  while read -r log; do
    n=1; ep=$(grep -aoE 'Epoch +[0-9]+/[0-9]+' "$log" | tail -1)
    printf "   %-46s %s  (%s)\n" "${log#$REPO/}" "${ep:-–}" "$(date -r "$log" '+%H:%M')"
  done < <(find "$REPO/logs" -name '*.log' -mmin -180 2>/dev/null | head -12)
  [ $n = 0 ] && echo "   (keine Aktivitaet)"
  echo
}

if [ "${1:-}" = "--local" ]; then report; exit 0; fi

# all hosts in parallel; remotes get this very script on stdin
tmp=$(mktemp -d)
report > "$tmp/l2" 2>&1 &
for h in l1 ws; do
  timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'bash -s -- --local' < "$0" > "$tmp/$h" 2>&1 \
    || echo "################  $h  NICHT ERREICHBAR" >> "$tmp/$h" &
done
wait
cat "$tmp/l2" "$tmp/l1" "$tmp/ws"; rm -rf "$tmp"
