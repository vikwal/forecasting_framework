#!/usr/bin/env bash
# GFS test 2026-10-27: all arms x seeds. One queue per (fleet, arm) so the first seed builds the
# data cache and the later seeds reuse it; the single-park replicates run in a fifth queue.
#   bash configs/gfs_test/run_all.sh            (from the repo root, detached with setsid nohup)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=frcst/bin/python
LOG=logs/gfs_test
mkdir -p $LOG

run() {  # gpu config seed
    CUDA_VISIBLE_DEVICES=$1 $PY train_cl.py -c configs/gfs_test/config_gfs_$2 -m tft --seed $3 \
        > $LOG/${2}_s$3.out 2>&1 || echo "FAILED $2 seed $3" >> $LOG/failed.txt
}

queue() {  # gpu config seeds...
    local g=$1 c=$2; shift 2
    for s in "$@"; do run $g $c $s; done
}

queue 0 wind_all_icon_ecmwf 42 43 44 &
queue 1 wind_all_icon_ecmwf_gfs 42 43 44 &
queue 2 pv_all_icon_ecmwf 42 43 44 &
queue 2 pv_all_icon_ecmwf_gfs 42 43 44 &
(
    for c in wind_one_icon_ecmwf wind_one_icon_ecmwf_gfs pv_one_icon_ecmwf pv_one_icon_ecmwf_gfs; do
        queue 0 $c 43 44
    done
) &
wait
echo "gfs_test done: $(date -Is)" >> $LOG/done.txt
