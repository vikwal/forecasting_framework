#!/usr/bin/env bash
# run_siteid_queue.sh GPU TAG MODE... : sequential site-ID-embedding trainings (MODE = fold0..fold4 | global100)
set -u
cd "$(dirname "$0")/.."
GPU=$1; TAG=$2; shift 2
source frcst/bin/activate
mkdir -p logs
QLOG=logs/iaims26_queue_${TAG}.log
for mode in "$@"; do
  if [ "$mode" = "global100" ]; then arg="--global100"; else arg="--fold ${mode#fold}"; fi
  echo "$(date +%F_%T) START siteid_$mode" >> "$QLOG"
  CUDA_VISIBLE_DEVICES=$GPU python iaims26/train_site_embedding.py $arg > logs/iaims26_siteid_${mode}.log 2>&1
  echo "$(date +%F_%T) END siteid_$mode rc=$?" >> "$QLOG"
done
echo "$(date +%F_%T) QUEUE_DONE" >> "$QLOG"
