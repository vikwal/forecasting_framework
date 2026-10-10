#!/usr/bin/env python3
"""
Local HPO for FL experiments: one Optuna study per client of an FL config, each client tuned on its
own stations only (hpo_cl.py on the client config of train_local.py, one process per GPU slot).

    python hpo_local.py -c configs/parks_v1/config_parks_v1_curt_v11_fl_fedgradient_static -m tft --gpus 0-7
    python train_local.py -c <same config> -m tft --lookup-hpo     # train with the best trial per client

The client configs get the names train_local.py writes ('<local name>_<client>'), so the study names
of hpo_cl.py and the lookup of train_cl.py (model.lookup_hpo) match. hpo.objective_reduction defaults
to 'best' here (new studies, early stopping restores the best epoch); hpo.trials/kfolds from the config
unless overridden. Configs, logs and a manifest go to runs/hpo_local/<name>_<timestamp>/.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from utils import local_training, tools
from train_local import local_name_from, _n_gpus


def client_hpo_configs(config: dict, clients=None, trials=None, kfolds=None, reduction='best') -> dict:
    """train_local client configs with the HPO overrides."""
    out = local_training.client_configs(config, clients)
    for cfg in out.values():
        hpo = cfg.setdefault('hpo', {})
        hpo.setdefault('objective_reduction', reduction)
        if trials:
            hpo['trials'] = int(trials)
        if kfolds:
            hpo['kfolds'] = int(kfolds)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Local HPO: one Optuna study per FL client")
    parser.add_argument('-c', '--config', required=True, help='FL config (with fl.clients and an hpo section)')
    parser.add_argument('-m', '--model', default='tft', help='Model (default: tft)')
    parser.add_argument('--clients', nargs='+', default=None, help='Subset of clients (default: all)')
    parser.add_argument('--gpus', default=None, help="GPU ids, e.g. '0-7' (default: all)")
    parser.add_argument('--per-gpu', type=int, default=1, help='Concurrent studies per GPU (default: 1)')
    parser.add_argument('--trials', type=int, default=None, help='Override hpo.trials')
    parser.add_argument('--kfolds', type=int, default=None, help='Override hpo.kfolds')
    parser.add_argument('--name', default=None, help='Run name (default: as train_local.py)')
    parser.add_argument('--dry-run', action='store_true', help='Write the configs, print the commands, do not run')
    args = parser.parse_args()

    cfg_path = args.config if args.config.endswith('.yaml') else f'{args.config}.yaml'
    config = tools.load_config(cfg_path)
    if not config.get('hpo'):
        raise ValueError(f"{cfg_path} has no hpo section")
    name = args.name or local_name_from(cfg_path, config)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', 'hpo_local', f'{name}_{stamp}')
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(run_dir, 'hpo_local.log')),
                                  logging.StreamHandler()])

    configs = client_hpo_configs(config, args.clients, args.trials, args.kfolds)
    paths = local_training.write_configs(configs, os.path.join(run_dir, 'configs'), name)
    gpus = local_training.parse_gpus(args.gpus, _n_gpus())
    jobs = [local_training.Job(name=cid, cmd=[sys.executable, 'hpo_cl.py', '-c', path, '-m', args.model],
                               log=os.path.join(run_dir, 'logs', f'{cid}.out'))
            for cid, path in paths.items()]
    logging.info(f"[hpo_local] {len(jobs)} studies from {cfg_path} on GPUs {gpus} x {args.per_gpu} -> {run_dir}")
    for job in jobs:
        logging.info(f"[hpo_local]   {job.name}: {len(configs[job.name]['data']['files'])} stations, "
                     f"{configs[job.name]['hpo'].get('trials')} trials, {' '.join(job.cmd)}")
    if args.dry_run:
        return 0

    local_training.run_jobs(jobs, gpus, per_gpu=args.per_gpu)
    failed = sorted(j.name for j in jobs if j.returncode != 0)
    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump({'config': cfg_path, 'model': args.model, 'gpus': gpus, 'failed': failed,
                   'jobs': {j.name: {'gpu': j.gpu, 'returncode': j.returncode, 'log': j.log,
                                     'runtime_s': (j.end - j.start) if j.end else None} for j in jobs}},
                  f, indent=2)
    if failed:
        logging.error(f"[hpo_local] failed clients: {failed} (logs in {run_dir}/logs)")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
