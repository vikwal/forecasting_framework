#!/usr/bin/env bash
# run_eval_after.sh GPU K : evaluate fold K (heldout+insample), then its no-static twin once that checkpoint exists
cd "$(dirname "$0")/.."
source frcst/bin/activate
GPU=$1; K=$2; L=logs/iaims26_queue_eval.log
CUDA_VISIBLE_DEVICES=$GPU python iaims26/eval_zeroshot.py --folds $K --splits heldout insample > logs/iaims26_eval_fold$K.log 2>&1
echo "$(date +%F_%T) END eval_fold$K rc=$?" >> $L
until grep -q "END config_wind_iaims26f${K}ns " logs/iaims26_queue_l1g$K.log 2>/dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=$GPU python iaims26/eval_zeroshot.py --config-name config_wind_iaims26f${K}ns --out-tag nsfold$K --splits heldout insample > logs/iaims26_eval_nsfold$K.log 2>&1
echo "$(date +%F_%T) END eval_nsfold$K rc=$?" >> $L
