"""Orchestrate the fine-tuning grid over all 100 held-out farms.

Farm-major: each farm's data is loaded once and all (budget x variant) cells
run in-process (see finetune.run_farm). Results are per-farm CSVs with
cell-level resume, so the orchestrator can be killed and restarted freely.

Sharding across GPUs/hosts: run one process per GPU, e.g. on l1

  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g nohup python iaims26/run_finetune_grid.py \
        --shard $g/8 > logs/iaims26_ft_$g.log 2>&1 &
  done

(on l2 use --shard 8/12 .. 11/12 with a 12-way split and shard offset).
"""

import argparse
import logging
import time
import traceback

import common
import finetune

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', default='0/1',
                        help='i/N: process stations with index %% N == i')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--stations', nargs='+', default=None,
                        help='explicit station list (overrides folds/shard)')
    parser.add_argument('--local-baseline', action='store_true',
                        help='also train the from-scratch local baseline per farm')
    args = parser.parse_args()

    shard_i, shard_n = (int(x) for x in args.shard.split('/'))

    if args.stations:
        stations = args.stations
    elif args.smoke:
        import gen_configs
        stations = [s for held in gen_configs.SMOKE_HELDOUT for s in held]
    else:
        folds = common.load_folds()
        stations = sorted(s for f in folds['folds'].values() for s in f)

    my_stations = [s for i, s in enumerate(stations) if i % shard_n == shard_i]
    logging.info(f'shard {args.shard}: {len(my_stations)} stations')

    failures = []
    t0 = time.time()
    for i, station in enumerate(my_stations):
        try:
            finetune.run_farm(station, smoke=args.smoke, save_models=args.save_models)
            if args.local_baseline:
                finetune.run_local_baseline(station, smoke=args.smoke)
        except Exception:
            logging.error(f'FAILED station {station}:\n{traceback.format_exc()}')
            failures.append(station)
        done = i + 1
        rate = (time.time() - t0) / done / 60
        logging.info(f'progress: {done}/{len(my_stations)} farms '
                     f'({rate:.1f} min/farm, {len(failures)} failures)')

    if failures:
        logging.error(f'{len(failures)} stations failed: {failures}')
        raise SystemExit(1)
    logging.info('shard complete.')


if __name__ == '__main__':
    main()
