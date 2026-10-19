#!/usr/bin/env bash
# Launch the 5 zero-shot fold trainings via the unchanged train_cl.py.
# Run from the repo root with the frcst venv active:
#   ./iaims26/run_zeroshot_training.sh            # folds 0-4 on GPUs 0-4
#   ./iaims26/run_zeroshot_training.sh 5 6 7 5 6  # custom GPU per fold
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=("${@:-}")
if [ ${#GPUS[@]} -eq 0 ] || [ -z "${GPUS[0]:-}" ]; then
  GPUS=(0 1 2 3 4)
fi

mkdir -p logs
for k in 0 1 2 3 4; do
  gpu=${GPUS[$k]}
  cfg="iaims26/configs/generated/config_wind_iaims26f${k}"
  log="logs/iaims26_zeroshot_f${k}.log"
  echo "fold ${k} -> GPU ${gpu} (${cfg}) log=${log}"
  CUDA_VISIBLE_DEVICES=${gpu} nohup python train_cl.py -m tft -c "${cfg}" \
    --save_model > "${log}" 2>&1 &
done
echo "5 trainings launched. Monitor with: tail -f logs/iaims26_zeroshot_f*.log"
