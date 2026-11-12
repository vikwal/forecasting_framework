#!/usr/bin/env python3
"""Does more data help on fully synthetic parks? (Viktor 2026-11-11)

Quantile TFT trained from scratch on N synthetic parks with the last M months before August 2024
(training window 2023-07-24..2024-07-31), early stopping and evaluation on 56 *other* synthetic parks, August 2024 to
July 2025 (agreed protocol: ES on the evaluation window). Parks: the 154 client parks are the training pool (nested random
subsets, the same for every seed), the 56 source parks of the synthetic transfer study are the held-out evaluation sites.
Reuses prepared.pkl of pv_synthetic_transfer.py (same windows, scaler and quantiles). The extra test year stays untouched.

    frcst/bin/python scripts/pv_synthetic_scaling.py --seed 42 --gpu 0 --output results/parks_pv_v4_30min/synthetic_scaling_2026-11-11_v1
"""
import argparse
import json
import logging
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import hpo, models  # noqa: E402
import pv_conformal as base  # noqa: E402
from pv_meter_finetune import cat  # noqa: E402
from pv_quantile_finetune import train, snapshot  # noqa: E402
from pv_quantile_metrics import metrics  # noqa: E402

PREPARED = Path('results/parks_pv_v4_30min/synthetic_transfer_2026-11-09_v1/prepared.pkl')
N_PARKS = [5, 20, 80, 154]
MONTHS = [1, 3, 12]
TRAIN_END = pd.Timestamp('2024-08-01', tz='UTC')


def last_months(item, months):
    """Windows of the last M months. prepared.pkl predates index_train; windows are chronological, one per 09 UTC run,
    so the last round(30.4 * M) windows are taken (a few dropped incomplete runs shift the start by days)."""
    if item.get('index_train') is not None:
        keep = np.asarray(pd.DatetimeIndex(item['index_train']) >= TRAIN_END - pd.DateOffset(months=months))
    else:
        keep = np.zeros(len(item['y_train']), bool)
        keep[-int(round(30.4 * months)):] = True
    return {k: v[keep] for k, v in item['X_train'].items()}, item['y_train'][keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{a.gpu}')
    with PREPARED.open('rb') as f:
        bundle = pickle.load(f)
    cfg, pool, held = bundle['cfg'], bundle['target'], bundle['source']
    qs = cfg['model']['tft']['quantiles']
    cfg['params']['random_seed'] = a.seed
    hp = hpo.get_hyperparameters(config=cfg)
    hp['lr'] = 3e-4
    order = sorted(pool)
    perm = [order[i] for i in np.random.default_rng(0).permutation(len(order))]
    out = a.output / f'seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() + ' + dirty'
    (a.output / 'parks.json').write_text(json.dumps(dict(pool_order=perm, held_out=sorted(held)), indent=2))
    rows = pd.read_csv(out / 'metrics.csv').to_dict('records') if (out / 'metrics.csv').exists() else []
    done = {(r['n_parks'], r['months']) for r in rows}
    val = [held[k] for k in sorted(held)]
    for n in N_PARKS:
        for months in MONTHS:
            if (n, months) in done:
                continue
            parts = [last_months(pool[k], months) for k in perm[:n]]
            X = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
            y = np.concatenate([p[1] for p in parts])
            case = out / f'n{n:03d}_m{months:02d}'
            case.mkdir(exist_ok=True)
            torch.manual_seed(a.seed)
            init = models.get_model(config=cfg, hyperparameters=hp)
            state = snapshot(init)
            del init
            model, log, _ = train('local_scaling', [(X, y)], val, state, cfg, hp, device, a.seed, qs, case)
            model.eval()
            per = []
            for k in sorted(held):
                it = held[k]
                pred = base.predict(model, it['X_test'], device)
                per.append(metrics(pred, it['y_test'], qs, np.asarray(it['nwp_raw_test']) > 10))
            m = pd.DataFrame(per).mean().to_dict()
            rows.append(dict(seed=a.seed, n_parks=n, months=months, train_windows=len(y), best_epoch=log['best_epoch'],
                             epochs=log['epochs'], date='2026-11-11', code_pin=pin, **m))
            pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
            logging.info('n=%s months=%s windows=%s pinball_day=%.6f', n, months, len(y), m['pinball_day'])
            del model
            torch.cuda.empty_cache()
    (out / 'COMPLETE').write_text('done\n')


if __name__ == '__main__':
    main()
