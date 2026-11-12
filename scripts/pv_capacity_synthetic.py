#!/usr/bin/env python3
"""Capacity series global vs local on fully synthetic parks (Viktor 2026-11-12, ablation with fixed values, no HPO):
does a shared model gain on per-park models once it is larger? TFT hidden_dim 32, 64, 128, 256 (heads 4, dropout 0.1,
lr 3e-4, rest as the quantile TFT), scratch init, full training year 2023-07-24..2024-07-31, ES and evaluation on
August 2024--July 2025 of the target parks (agreed protocol). Per seed 10 target parks (positions 20..29 of the seed
permutation of the 154 client parks, as in the boundary study) and 90 partner parks (positions 0..19 and 38..107).
  local     one model per target park
  global10  one model pooled over the 10 target parks
  global100 one model pooled over the 10 targets and the 90 partners
Evaluated on each target's test year and, to separate underfitting from generalisation, on its training windows.

    frcst/bin/python scripts/pv_capacity_synthetic.py --seed 42 --gpu 0 --output results/parks_pv_v4_30min/capacity_synthetic_2026-11-12_v1
"""
import argparse
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
from pv_quantile_finetune import train, snapshot  # noqa: E402
from pv_quantile_metrics import metrics  # noqa: E402
from pv_synthetic_scaling import PREPARED  # noqa: E402
from pv_boundary_synthetic import own, pool  # noqa: E402

HIDDEN = [32, 64, 128, 256]


def daylight(item, part, scaler, j):
    g = item[f'X_{part}']['known'][:, -item[f'y_{part}'].shape[1]:, 0] * scaler.scale_[j] + scaler.mean_[j]
    return g > 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--hidden', type=int, nargs='+', default=HIDDEN)
    a = ap.parse_args()
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{a.gpu}')
    with PREPARED.open('rb') as f:
        bundle = pickle.load(f)
    cfg, parks, scaler = bundle['cfg'], bundle['target'], bundle['scaler']
    del bundle
    qs = cfg['model']['tft']['quantiles']
    cfg['params']['random_seed'] = a.seed
    order = sorted(parks)
    perm = [order[i] for i in np.random.default_rng(a.seed).permutation(len(order))]
    targets, partners = perm[20:30], perm[:20] + perm[38:108]
    known = cfg['params']['known_features']
    sx = scaler if hasattr(scaler, 'mean_') else scaler['x']
    j = getattr(sx, '_ff_feature_cols', known).index('gti_nwp')
    out = a.output / f'seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'clients.txt').write_text(f'targets {targets}\npartners {partners}\n')
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() + ' + dirty'
    rows = pd.read_csv(out / 'metrics.csv').to_dict('records') if (out / 'metrics.csv').exists() else []
    done = {r['case'] for r in rows}
    full = {k: own(parks[k], 12, 'year') for k in targets + partners}
    for h in a.hidden:
        hp = hpo.get_hyperparameters(config=cfg)
        hp.update(lr=3e-4, hidden_dim=h)
        torch.manual_seed(a.seed)
        init = models.get_model(config=cfg, hyperparameters=hp)
        state, n_params = snapshot(init), sum(p.numel() for p in init.parameters())
        del init
        plans = [(f'local/{k}', [k], [k]) for k in targets]
        plans += [('global10', targets, targets), ('global100', targets + partners, targets)]
        for name, train_keys, eval_keys in plans:
            case = f'h{h:03d}/{name}'
            if case in done:
                continue
            cdir = out / case
            cdir.mkdir(parents=True, exist_ok=True)
            data = pool([full[k] for k in train_keys])
            model, log, _ = train('local_' + name.split('/')[0], [data], [parks[k] for k in eval_keys], state, cfg, hp,
                                  device, a.seed, qs, cdir)
            model.eval()
            for k in eval_keys:
                it = parks[k]
                for part in ('test', 'train'):
                    pred = base.predict(model, it[f'X_{part}'], device)
                    rows.append(dict(case=case, seed=a.seed, hidden=h, n_params=n_params, arm=name.split('/')[0], park=k,
                                     part=part, n_train_parks=len(train_keys), best_epoch=log['best_epoch'],
                                     epochs=log['epochs'], date='2026-11-12', code_pin=pin,
                                     **metrics(pred, it[f'y_{part}'], qs, daylight(it, part, sx, j))))
            pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
            del model
            torch.cuda.empty_cache()
    (out / 'COMPLETE').write_text('done\n')


if __name__ == '__main__':
    main()
