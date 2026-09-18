#!/usr/bin/env bash
# Faehrt die Kette des Arms ohne Messhistorie unbeaufsichtigt zu Ende:
#
#   1. wartet, bis die Studie ZIEL abgeschlossene Trials hat
#   2. beendet die HPO-Worker auf allen drei Hosts
#   3. trainiert die drei Fold-Modelle und die Schlussmessung (je eine GPU)
#   4. wertet alle vier aus (--eval-split val)
#
# Laeuft mit setsid nohup los und ueberlebt das Sessionende:
#
#   setsid nohup scripts/run_nohist_kette.sh > logs/nohist_kette.log 2>&1 &
#
# Jeder Schritt protokolliert mit Zeitstempel. Schlaegt ein Training fehl, wird
# nur dessen Auswertung uebersprungen — die uebrigen laufen weiter, damit eine
# einzelne Panne nicht die ganze Nacht kostet.
set -u
cd "$(dirname "$0")/.." || exit 1

STUDIE="cl_m-tft-bc_out-96_freq-30min_solar_tft_nohist_hpo"
ARM="solar_tft_nohist"
ZIEL="${ZIEL:-50}"
CACHE="/mnt/nvme2/data_cache"
PY="frcst/bin/python"
LOGDIR="logs/nohist_kette"
mkdir -p "$LOGDIR"

sag() { echo "[$(date '+%F %T')] $*"; }

fertige_trials() {
    $PY -c "
import optuna, os
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
s = optuna.load_study(study_name='$STUDIE', storage=os.environ['OPTUNA_STORAGE'])
print(sum(1 for t in s.trials if t.state.name == 'COMPLETE'))
" 2>/dev/null | tail -1
}

worker_laufen() {
    pgrep -f "hpo_tft_bc.py -c configs/$ARM" > /dev/null && return 0
    for h in l1 ws; do
        ssh "$h" "pgrep -f 'hpo_tft_bc.py -c configs/$ARM' > /dev/null" && return 0
    done
    return 1
}

# ── 1. auf das Trialziel warten ─────────────────────────────────────────────
sag "warte auf $ZIEL abgeschlossene Trials in $STUDIE"
while true; do
    n="$(fertige_trials)"
    if [ "${n:-0}" -ge "$ZIEL" ]; then
        sag "Ziel erreicht: $n abgeschlossene Trials"
        break
    fi
    if ! worker_laufen; then
        sag "keine Worker mehr aktiv (bei ${n:-?} Trials) — mache mit dem Bestand weiter"
        break
    fi
    sleep 300
done

# ── 2. Worker beenden ───────────────────────────────────────────────────────
# Nur Prozesse dieses Arms: das Muster traegt den Config-Pfad, ein Worker einer
# anderen Studie wird nie getroffen. Ein gerade laufender Trial bleibt als
# RUNNING in der Studie stehen; best_trial beruehrt das nicht.
sag "beende die HPO-Worker"
pkill -f "hpo_tft_bc.py -c configs/$ARM"
for h in l1 ws; do
    ssh "$h" "pkill -f 'hpo_tft_bc.py -c configs/$ARM'" 2>/dev/null
done
sleep 20
sag "bester Trial: $($PY -c "
import optuna, os
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
s = optuna.load_study(study_name='$STUDIE', storage=os.environ['OPTUNA_STORAGE'])
print(f'{s.best_trial.number} mit val_rmse {s.best_value:.4f}')
" 2>/dev/null | tail -1)"

# ── 3. Trainings ────────────────────────────────────────────────────────────
# Die vier Laeufe bauen verschiedene Cache-Eintraege (andere Stationsmengen und
# Zeitachsen), gleichzeitiger Start ist deshalb unkritisch — anders als bei
# mehreren Workern derselben Studie mit leerem Cache.
declare -A PIDS
gpu=0
for cfg in fold1 fold2 fold3 testyear; do
    log="$LOGDIR/train_${cfg}.out"
    sag "starte Training $cfg auf GPU $gpu → $log"
    setsid nohup $PY train_cl_tft_bc.py \
        -c "configs/$ARM/config_${ARM}_${cfg}.yaml" \
        --hpo-study "$STUDIE" --gpu "$gpu" --cache-dir "$CACHE" \
        > "$log" 2>&1 &
    PIDS[$cfg]=$!
    gpu=$((gpu + 1))
    sleep 5
done

for cfg in "${!PIDS[@]}"; do
    wait "${PIDS[$cfg]}"
    status=$?
    if [ $status -eq 0 ]; then
        sag "Training $cfg fertig"
    else
        sag "Training $cfg FEHLGESCHLAGEN (Exit $status) — Auswertung entfaellt"
    fi
done

# ── 4. Auswertungen ─────────────────────────────────────────────────────────
# --eval-split val ist Pflicht: das Auswertungsfenster ist [val_start,
# test_start). Ohne das Flag misst die Fold-Auswertung im Testjahr und die
# Schlussmessung in einem leeren Fenster.
gpu=0
for cfg in fold1 fold2 fold3 testyear; do
    modell="models/train_tft_bc_m-tft_c-${ARM}_${cfg}.pt"
    if [ ! -f "$modell" ]; then
        sag "Auswertung $cfg uebersprungen — $modell fehlt"
        gpu=$((gpu + 1)); continue
    fi
    log="$LOGDIR/eval_${cfg}.out"
    sag "starte Auswertung $cfg auf GPU $gpu → $log"
    setsid nohup $PY get_test_results_tft_bc.py \
        -c "configs/$ARM/config_${ARM}_${cfg}.yaml" \
        --hpo-study "$STUDIE" \
        --model-tag "train_tft_bc_m-tft_c-${ARM}_${cfg}" \
        --raw-out-name "tft_${ARM}_${cfg}" --eval-split val \
        --cache-dir "$CACHE" --gpu "$gpu" \
        > "$log" 2>&1 &
    PIDS[$cfg]=$!
    gpu=$((gpu + 1))
    sleep 5
done
for cfg in "${!PIDS[@]}"; do wait "${PIDS[$cfg]}" 2>/dev/null; done

sag "Kette durch. Ergebnisse:"
grep -h "Pooled RMSE" "$LOGDIR"/eval_*.out 2>/dev/null | sed 's/^/    /'
sag "Roh: data/raw_preds/tft_${ARM}_*.parquet, je Station: data/test_results/tft_${ARM}_*.csv"
