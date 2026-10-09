"""Local training baseline for FL experiments: every FL client trains its own model on its own
stations only, with the centralized pipeline (train_cl.py), one process per GPU.

Entry point: train_local.py. The client runs are derived from an FL config (fl.clients), so data,
features, split and model are exactly those of the federated run; docs/local_training.md.
"""

import copy
import json
import logging
import os
import pickle
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

# Keys removed from the derived client configs: no foreign stations are trained or evaluated.
DROPPED_DATA_KEYS = ('val_files', 'holdout_files', 'test_files', 'client_files')


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def local_name(fl_config_path: str, strategy: Optional[str] = None) -> str:
    """Name of the local variant: 'config_parks_v1_fl_fedgradient_parkid' -> 'config_parks_v1_local_parkid'."""
    stem = os.path.splitext(os.path.basename(fl_config_path))[0]
    if strategy and f'_fl_{strategy}' in stem:
        return stem.replace(f'_fl_{strategy}', '_local', 1)
    return f'{stem}_local'


def client_configs(config: Dict[str, Any], clients: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    """One centralized config per FL client: data.files = that client's stations,
    held-out/test station lists removed, model.fl False. Everything else unchanged.
    With fl.client_holdout {client: [stations]} those stations become the client's
    data.holdout_files (evaluated, never trained on)."""
    mapping = (config.get('fl') or {}).get('clients')
    if not mapping:
        raise ValueError("the config has no fl.clients mapping")
    if clients:
        unknown = sorted(set(clients) - set(mapping))
        if unknown:
            raise ValueError(f"unknown clients {unknown}; available: {list(mapping)}")
    out = {}
    for cid, stations in mapping.items():
        if clients and cid not in clients:
            continue
        cfg = copy.deepcopy(config)
        cfg['data']['files'] = [str(s) for s in stations]
        for key in DROPPED_DATA_KEYS:
            cfg['data'].pop(key, None)
        cfg['model']['fl'] = False
        # fl.client_holdout: stations of this client that it never trains on (e.g. a new park);
        # the local model is evaluated on them like the global FL model (train_cl holdout_files)
        own_holdout = ((config.get('fl') or {}).get('client_holdout') or {}).get(cid)
        if own_holdout:
            cfg['data']['holdout_files'] = [str(s) for s in own_holdout]
        out[str(cid)] = cfg
    return out


def write_configs(configs: Dict[str, Dict[str, Any]], directory: str, name: str) -> Dict[str, str]:
    """Write the client configs as YAML (paths already resolved, i.e. host-specific: a run
    protocol, not a reusable config). Returns {client: path without .yaml}."""
    os.makedirs(directory, exist_ok=True)
    paths = {}
    for cid, cfg in configs.items():
        stem = os.path.join(directory, f'{name}_{cid}')
        if '.' in stem:
            # train_cl.py cuts the config argument at the first '.'
            raise ValueError(f"config path must not contain '.': {stem}")
        with open(f'{stem}.yaml', 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        paths[cid] = stem
    return paths


# ---------------------------------------------------------------------------
# GPU queue
# ---------------------------------------------------------------------------

def parse_gpus(spec: Optional[str], available: int) -> List[int]:
    """'0-7', '0,2,3', '1-3,6' or None (= all available)."""
    if not spec:
        return list(range(available))
    gpus = []
    for part in str(spec).split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            gpus.extend(range(int(lo), int(hi) + 1))
        elif part:
            gpus.append(int(part))
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"duplicate GPU in {spec!r}")
    return gpus


@dataclass
class Job:
    name: str
    cmd: List[str]
    log: str
    gpu: Optional[int] = None
    returncode: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    env: Dict[str, str] = field(default_factory=dict)


def run_jobs(jobs: List[Job], gpus: Sequence[int], per_gpu: int = 1, poll: float = 2.0,
             cwd: Optional[str] = None) -> List[Job]:
    """Run the jobs as subprocesses, at most ``per_gpu`` per GPU at a time
    (CUDA_VISIBLE_DEVICES=<gpu>); the next job starts as soon as a slot frees up.
    A failing job does not stop the others. Returns the jobs with returncode/start/end/gpu."""
    if not gpus:
        raise ValueError("no GPU slots")
    slots = [g for g in gpus for _ in range(per_gpu)]
    n_parallel = len(slots)
    threads = str(max(1, (os.cpu_count() or n_parallel) // n_parallel))
    queue = list(jobs)
    running = {}                      # Popen -> (job, slot, file handle)
    free = list(slots)
    while queue or running:
        while queue and free:
            job = queue.pop(0)
            gpu = free.pop(0)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
                   'OMP_NUM_THREADS': threads, 'MKL_NUM_THREADS': threads, **job.env}
            os.makedirs(os.path.dirname(os.path.abspath(job.log)), exist_ok=True)
            fh = open(job.log, 'w')
            job.gpu, job.start = gpu, time.time()
            proc = subprocess.Popen(job.cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    cwd=cwd, stdin=subprocess.DEVNULL)
            running[proc] = (job, gpu, fh)
            logging.info(f"[local] start {job.name} on GPU {gpu} (log {job.log})")
        time.sleep(poll)
        for proc in [p for p in running if p.poll() is not None]:
            job, gpu, fh = running.pop(proc)
            fh.close()
            job.returncode, job.end = proc.returncode, time.time()
            free.append(gpu)
            level = logging.INFO if proc.returncode == 0 else logging.ERROR
            logging.log(level, f"[local] end {job.name} rc={proc.returncode} "
                               f"({(job.end - job.start) / 60:.1f} min)")
    return jobs


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

_SAVED = re.compile(r'Results saved to: (\S+\.pkl)')


def result_path_from_log(log: str) -> Optional[str]:
    """Path of the result pickle that train_cl.py reports at the end of its output."""
    if not os.path.exists(log):
        return None
    with open(log, errors='replace') as f:
        hits = _SAVED.findall(f.read())
    return hits[-1] if hits else None


def merge_results(results: Dict[str, str], jobs: Optional[Dict[str, Job]] = None) -> Dict[str, Any]:
    """Merge the per-client train_cl result pickles into one result in the FL layout:
    evaluation rows per station with a client_id column, mean/std rows, per-client
    training summary (epochs, best epoch, runtime)."""
    evals, clients = [], {}
    config = None
    for cid, path in results.items():
        with open(path, 'rb') as f:
            res = pickle.load(f)
        config = config or res.get('config')
        ev = res['evaluation']
        ev = ev[ev['key'].notna()].copy()
        ev['client_id'] = cid
        evals.append(ev)
        hist = res.get('history') or {}
        val = np.asarray(hist.get('val_rmse', []), dtype=float)
        job = (jobs or {}).get(cid)
        clients[cid] = {'result': path, 'n_stations': int(len(ev)),
                        'epochs': int(len(val)),
                        'best_epoch': int(np.argmin(val)) + 1 if len(val) else None,
                        'runtime_s': (job.end - job.start) if job and job.end else None,
                        'gpu': job.gpu if job else None}
    evaluation = pd.concat(evals)
    numeric = evaluation.select_dtypes(include=[np.number])
    evaluation.loc['mean'] = numeric.mean()
    evaluation.loc['std'] = numeric.std()
    return {'strategy': 'local', 'config': config, 'evaluation': evaluation, 'clients': clients}


def summary_line(merged: Dict[str, Any]) -> str:
    ev = merged['evaluation']
    r2 = ev[ev['key'].notna()]['R^2'].astype(float)
    return (f"local: {len(merged['clients'])} clients, {len(r2)} stations, "
            f"R^2 mean {r2.mean():.4f} median {r2.median():.4f}")


def save_merged(merged: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(merged, f)
    with open(path.replace('.pkl', '_clients.json'), 'w') as f:
        json.dump(merged['clients'], f, indent=2)
