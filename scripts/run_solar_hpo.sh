#!/usr/bin/env bash
# Startet die Worker der Solar-TFT-HPO auf den GPUs aller drei Hosts.
#
#   scripts/run_solar_hpo.sh [SUFFIX] [SLOT ...]
#
# Ohne Slotliste starten alle Slots aus SLOTS, sonst nur die genannten
# (Format host:gpu) — fuer das Nachstarten einzelner Worker.
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

SUF="${1:-}"; shift || true
# Welcher Arm gefahren wird, steht in CFG. Default ist die Studie mit eigener
# Messhistorie; der Arm ohne sie laeuft ueber
#   CFG=configs/solar_tft_nohist/config_solar_tft_nohist_hpo.yaml scripts/run_solar_hpo.sh
# Die Worker-Zaehlung unten filtert auf den Config-Basename, zwei Arme koennen
# sich also nicht gegenseitig als "laeuft schon" zaehlen.
CFG="${CFG:-configs/solar_tft/config_solar_tft_hpo.yaml}"
[ -f "$CFG" ] || { echo "Config fehlt: $CFG"; exit 1; }

#: host:gpu — auf l1 tragen 4-7 derzeit Fremdlast eines anderen Nutzers,
#: die bleiben frei. Die Belegung wandert, vor einem Neustart pruefen.
#: Jeder Slot doppelt = zwei Trainings auf der GPU (s. maxprogpu_fuer). ws nur
#: einfach. l1:3 ist raus, dort liegt Fremdlast — dafuer l1:4.
SLOTS=(lokal:0 lokal:0 lokal:1 lokal:1 lokal:2 lokal:2 lokal:3 lokal:3
       l1:0 l1:0 l1:1 l1:1 l1:2 l1:2 l1:4 l1:4
       ws:0 ws:1)
[ $# -gt 0 ] && SLOTS=("$@")

dataroot_fuer() { case "$1" in l1) echo /mnt/nvme1;; *) echo /mnt/lambda1/nvme1;; esac; }
# Cache-Verzeichnis: der Default /mnt/nvme2/data_cache existiert auf l2 und l1,
# auf ws nicht — dort scheiterte jeder Trial mit "Permission denied: /mnt/nvme2".
cachedir_fuer() { case "$1" in ws) echo '$HOME/data_cache';; *) echo /mnt/nvme2/data_cache;; esac; }
# 150 GB reichen, seit hpo.optional_features leer ist: die Studie braucht dann
# genau 3 Cache-Eintraege (einen je spatialem Fold) zu je ~16.5 GB, zusammen
# ~50 GB. Mit den drei binaeren Feature-Flags waren es 8 Kombinationen mal 3
# Folds = ~400 GB je Host, und enforce_cache_budget raeumte bei einem Budget von
# 500 GB im Dauerbetrieb Eintraege weg, die ein anderer Worker kurz darauf neu
# bauen musste (60 Evictions allein am 16.09.2026).
# l1 traegt daneben noch ~420 GB Cache der abgeschlossenen Wind-Studien
# (wind_tft_sp_base/_hist) im selben Manifest. Das Budget gilt fuer das ganze
# Manifest, nicht je Studie — mit 150 GB wuerde der erste Solar-Trial dort den
# Wind-Bestand evictieren.
cachegb_fuer()  { case "$1" in l1) echo 600;; *) echo 150;; esac; }
repo_fuer()     { case "$1" in lokal) echo "$REPO_L2";; *) echo '$HOME/Work/forecasting_framework';; esac; }

echo "Studie: $(basename $CFG .yaml)${SUF:+ (Suffix $SUF)} — ${#SLOTS[@]} Worker"
# Mehrfachbelegung: ein Trial belegt mit den aktuellen Suchraumgrenzen 4-15 GB,
# lastet eine GPU aber oft nur zu 15-45 % aus (kleine Modelle). Zwei Trainings je
# GPU passen deshalb auf die A100 (80 GB) und die A6000 (49 GB). Auf ws bleibt es
# bei einem: die RTX 4090 hat 24 GB, und der Host hat nur 125 GB RAM, von denen
# ein Worker samt DataLoader-Kindern rund 30 GB braucht.
maxprogpu_fuer() { case "$1" in ws) echo 1;; *) echo 2;; esac; }

# Zaehlt die laufenden Worker auf host:gpu. Entscheidend ist der Prozessname:
# die DataLoader-Kinder heissen 'pt_data_worker', der Wrapper 'bash' und die
# ssh-Verbindung zu einem Remote-Slot 'ssh' — letztere traegt deren '--gpu N' in
# der eigenen Kommandozeile und wurde vorher auf l2 faelschlich mitgezaehlt.
# Nur 'python' ist ein echter Worker.
laufende_worker() {
    local host="$1" gpu="$2"
    local probe="for p in \$(pgrep -f 'hpo_tft_bc[.]py'); do \
[ \"\$(ps -o comm= -p \$p 2>/dev/null)\" = python ] || continue; \
ps -o args= -p \$p 2>/dev/null | grep -F -- '$(basename "$CFG")' | grep -q -- '--gpu $gpu ' && echo x; \
done | grep -c x"
    if [ "$host" = lokal ]; then bash -c "$probe"; else ssh "$host" "$probe"; fi
}

declare -A lfd   # laufende Nummer je host_gpu, fuer getrennte Logdateien
for slot in "${SLOTS[@]}"; do
    host="${slot%%:*}"; gpu="${slot##*:}"
    n_laeuft="$(laufende_worker "$host" "$gpu")"; n_max="$(maxprogpu_fuer "$host")"
    if [ "${n_laeuft:-0}" -ge "$n_max" ]; then
        echo "  $slot  UEBERSPRUNGEN — dort laufen bereits $n_laeuft von $n_max Workern"
        continue
    fi
    repo="$(repo_fuer "$host")"; droot="$(dataroot_fuer "$host")"; cgb="$(cachegb_fuer "$host")"
    cdir="$(cachedir_fuer "$host")"
    # Eigene Logdatei je Worker, sonst schreiben bei Mehrfachbelegung beide in
    # dieselbe und die Trials lassen sich nicht mehr auseinanderhalten. Die
    # Nummer zaehlt die bereits laufenden mit, damit ein Nachstart einzelner
    # Slots nicht in ein fremdes Log schreibt.
    slot_key="${host}_${gpu}"
    lfd["$slot_key"]=$(( ${lfd["$slot_key"]:-$n_laeuft} + 1 ))
    log="logs/hpo_solar/w_${host}_g${gpu}_${lfd[$slot_key]}.log"

    rumpf="cd $repo && mkdir -p logs/hpo_solar && \
eval \"\$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE)=' ~/.bashrc)\" && \
export WEATHER_DB_URL ECMWF_WIND_SL_URL OPTUNA_STORAGE DATA_ROOT=$droot && \
echo \"== \$(date -Is) start gpu=$gpu host=$host commit=\$(git rev-parse --short HEAD)\" >> $log && \
frcst/bin/python hpo_tft_bc.py -c $CFG --gpu $gpu --max-cache-gb $cgb --cache-dir $cdir ${SUF:+-s $SUF} >> $log 2>&1; \
echo \"== \$(date -Is) Worker beendet, Exit \$?\" >> $log"

    if [ "$host" = lokal ]; then
        setsid nohup bash -c "$rumpf" > /dev/null 2>&1 &
    else
        setsid nohup ssh "$host" "$rumpf" > /dev/null 2>&1 &
    fi
    echo "  $slot  (Cache ${cdir}, Budget ${cgb} GB, DATA_ROOT $droot)"
    sleep 3
done
echo "Logs: logs/hpo_solar/w_<host>_g<gpu>.log auf dem jeweiligen Host"
echo "Fortschritt: optuna-dashboard (Port 8504) oder scripts/hpo_watch_restart.py"
