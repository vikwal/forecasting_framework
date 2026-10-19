#!/usr/bin/env bash
# run_ftlist.sh GPU TAG STATION... : fine-tuning grid + local baselines for an explicit station list
cd "$(dirname "$0")/.."
source frcst/bin/activate
GPU=$1; TAG=$2; shift 2
echo "$(date +%F_%T) START ftlist $TAG ($*)" >> logs/iaims26_queue_ft.log
CUDA_VISIBLE_DEVICES=$GPU python iaims26/run_finetune_grid.py --local-baseline --stations "$@" > logs/iaims26_ft_${TAG}.log 2>&1
echo "$(date +%F_%T) END ftlist $TAG rc=$?" >> logs/iaims26_queue_ft.log
