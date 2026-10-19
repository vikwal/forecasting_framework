#!/usr/bin/env bash
# run_queue.sh GPU TAG CONFIG... : sequential train_cl.py trainings on one GPU
set -u
cd "$(dirname "$0")/.."
GPU=$1; TAG=$2; shift 2
source frcst/bin/activate
mkdir -p logs
QLOG=logs/iaims26_queue_${TAG}.log
for name in "$@"; do
  echo "$(date +%F_%T) START $name" >> "$QLOG"
  CUDA_VISIBLE_DEVICES=$GPU python train_cl.py -m tft -c iaims26/configs/generated/$name --save_model > logs/iaims26_${name}.log 2>&1
  echo "$(date +%F_%T) END $name rc=$?" >> "$QLOG"
done
echo "$(date +%F_%T) QUEUE_DONE" >> "$QLOG"
