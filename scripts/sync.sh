#!/usr/bin/env bash
# sync.sh -- keep l1 and ws on the same commit as l2 without disturbing running work.
#
#   bash scripts/sync.sh check          read-only: what a pull would do on l1 and ws
#   bash scripts/sync.sh pull <host>    fast-forward pull on one host (after the check)
#
# The check is what /sync shows before asking. A host is flagged when it has
# local changes, running train_/hpo_ processes, or a queue worker with tasks
# left -- testmode_worker.sh re-reads scripts and configs every round, so a
# pull can change a run that is still going.
set -u
REPO_REL="Work/forecasting_framework"

remote_check() {  # runs ON the remote host
  cd "$HOME/$REPO_REL" 2>/dev/null || { echo "   KEIN REPO"; return; }
  git fetch -q origin 2>/dev/null || { echo "   fetch fehlgeschlagen"; return; }
  local head; head=$(git rev-parse --short HEAD)
  local behind; behind=$(git rev-list --count HEAD..origin/main)
  local ahead;  ahead=$(git rev-list --count origin/main..HEAD)
  local dirty;  dirty=$(git status --porcelain | grep -vc '^??')
  local procs;  procs=$(pgrep -af 'python[0-9.]* .*(train_|hpo_)' | grep -vc pgrep)
  local qleft="-"
  if [ -f "$HOME/queue_scripts/testmode_tasks.txt" ]; then
    local cur tot; cur=$(cat "$HOME/queue_scripts/.testmode_cursor" 2>/dev/null || echo 0)
    tot=$(grep -cv '^#' "$HOME/queue_scripts/testmode_tasks.txt"); qleft=$((tot-cur))
  fi
  printf "   HEAD %s  | %s hinter origin, %s voraus | %s lokal geaendert | %s train/hpo-Prozesse | queue offen: %s\n" \
    "$head" "$behind" "$ahead" "$dirty" "$procs" "$qleft"
  if [ "$behind" -gt 0 ]; then
    echo "   Pull wuerde aendern:"; git diff --name-only HEAD..origin/main | sed 's/^/     /' | head -15
    [ "$(git diff --name-only HEAD..origin/main | wc -l)" -gt 15 ] && echo "     ..."
  fi
  if [ "$dirty" -gt 0 ]; then echo "   lokal geaendert:"; git status --porcelain | grep -v '^??' | sed 's/^/     /' | head -8; fi
  if [ "$procs" -gt 0 ]; then echo "   laufende Prozesse:"; pgrep -af 'python[0-9.]* .*(train_|hpo_)' | grep -v pgrep | cut -c1-100 | sed 's/^/     /'; fi
  local flag=""
  [ "$dirty" -gt 0 ] && flag+=" LOKALE-AENDERUNGEN"
  [ "$procs" -gt 0 ] && flag+=" PROZESSE-LAUFEN"
  [ "$qleft" != "-" ] && [ "$qleft" -gt 0 ] && flag+=" QUEUE-OFFEN"
  [ "$ahead" -gt 0 ] && flag+=" LOKALE-COMMITS"
  [ -n "$flag" ] && echo "   >> NACHFRAGEN:$flag" || { [ "$behind" -gt 0 ] && echo "   >> pull unbedenklich" || echo "   >> aktuell"; }
}

remote_pull() {
  cd "$HOME/$REPO_REL" && git pull --ff-only origin main 2>&1 | tail -2 && echo "   HEAD jetzt $(git rev-parse --short HEAD)"
}

case "${1:-}" in
  --remote-check) remote_check ;;
  --remote-pull)  remote_pull ;;
  check)
    echo "== l2 (hier)"; cd "$HOME/$REPO_REL"
    printf "   HEAD %s | %s lokal geaendert, %s untracked\n" "$(git rev-parse --short HEAD)" \
      "$(git status --porcelain | grep -vc '^??')" "$(git status --porcelain | grep -c '^??')"
    for h in l1 ws; do
      echo "== $h"
      timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'bash -s -- --remote-check' < "$0" 2>&1 || echo "   NICHT ERREICHBAR"
    done ;;
  pull)
    h="${2:?host}"; echo "== pull auf $h"
    timeout 60 ssh -o BatchMode=yes "$h" 'bash -s -- --remote-pull' < "$0" ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
