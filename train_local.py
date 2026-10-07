#!/usr/bin/env python3
"""
Local training baseline for FL experiments: every client of an FL config trains its own
model on its own stations only (train_cl.py, one process per GPU slot, in parallel).

    python train_local.py -c configs/parks_v1/config_parks_v1_fl_fedgradient -m tft --gpus 0-7
    python train_local.py -c ... --clients R0 R1 --gpus 0,1 --dry-run

Derived client configs, per-client logs and a manifest go to runs/local/<name>_<timestamp>/;
the merged result (evaluation per station with client_id, same layout as train_fl.py) to
results/<data dir>/local_m-<model>_<name>_<timestamp>.pkl. docs/local_training.md
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from utils import local_training, tools


def main() -> int:
    parser = argparse.ArgumentParser(description="Local training: one model per FL client, in parallel")
    parser.add_argument('-c', '--config', required=True, help='FL config (with fl.clients)')
    parser.add_argument('-m', '--model', default='tft', help='Model (default: tft)')
    parser.add_argument('--clients', nargs='+', default=None, help='Subset of clients (default: all)')
    parser.add_argument('--gpus', default=None, help="GPU ids, e.g. '0-7' or '0,2,3' (default: all)")
    parser.add_argument('--per-gpu', type=int, default=1, help='Concurrent runs per GPU (default: 1)')
    parser.add_argument('--name', default=None,
                        help="Run name (default: FL config name with '_fl_<strategy>' -> '_local')")
    parser.add_argument('--save_model', action='store_true', help='Pass --save_model to train_cl.py')
    parser.add_argument('--dry-run', action='store_true', help='Write the configs, print the commands, do not run')
    args = parser.parse_args()

    cfg_path = args.config if args.config.endswith('.yaml') else f'{args.config}.yaml'
    config = tools.load_config(cfg_path)
    name = args.name or local_name_from(cfg_path, config)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', 'local', f'{name}_{stamp}')
    os.makedirs(run_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(run_dir, 'train_local.log')),
                                  logging.StreamHandler()])

    configs = local_training.client_configs(config, args.clients)
    paths = local_training.write_configs(configs, os.path.join(run_dir, 'configs'), name)
    gpus = local_training.parse_gpus(args.gpus, _n_gpus())
    jobs = [local_training.Job(
                name=cid,
                cmd=[sys.executable, 'train_cl.py', '-c', path, '-m', args.model]
                    + (['--save_model'] if args.save_model else []),
                log=os.path.join(run_dir, 'logs', f'{cid}.out'))
            for cid, path in paths.items()]
    logging.info(f"[local] {len(jobs)} clients from {cfg_path} on GPUs {gpus} "
                 f"x {args.per_gpu} -> {run_dir}")
    for job in jobs:
        logging.info(f"[local]   {job.name}: {len(configs[job.name]['data']['files'])} stations, "
                     f"{' '.join(job.cmd)}")
    if args.dry_run:
        return 0

    local_training.run_jobs(jobs, gpus, per_gpu=args.per_gpu)
    by_client = {j.name: j for j in jobs}
    results = {cid: local_training.result_path_from_log(j.log) for cid, j in by_client.items()}
    failed = sorted(cid for cid, j in by_client.items() if j.returncode != 0 or not results[cid])
    manifest = {'config': cfg_path, 'model': args.model, 'gpus': gpus, 'failed': failed,
                'jobs': {cid: {'gpu': j.gpu, 'returncode': j.returncode, 'log': j.log,
                               'result': results[cid],
                               'runtime_s': (j.end - j.start) if j.end else None}
                         for cid, j in by_client.items()}}
    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    if failed:
        logging.error(f"[local] failed clients: {failed} (logs in {run_dir}/logs)")

    ok = {cid: p for cid, p in results.items() if cid not in failed}
    if ok:
        merged = local_training.merge_results(ok, by_client)
        merged.update({'model': args.model, 'source_config': cfg_path, 'run_dir': run_dir,
                       'failed': failed})
        base_dir = os.path.basename(config['data']['path'])
        out = os.path.join('results', base_dir, f"local_m-{args.model}_{name.removeprefix('config_')}_{stamp}.pkl")
        local_training.save_merged(merged, out)
        logging.info(f"[local] {local_training.summary_line(merged)} -> {out}")
    return 1 if failed else 0


def local_name_from(cfg_path: str, config: dict) -> str:
    strategy = (config.get('fl') or {}).get('strategy')
    return local_training.local_name(cfg_path, strategy)


def _n_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


if __name__ == '__main__':
    sys.exit(main())
