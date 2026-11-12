#!/usr/bin/env python3
"""Part C of the dependence-boundary study (Viktor 2026-11-11): when does a client still need federated partners?
Client = P of the six meters (P = 1, 2, 3; six fixed clients per P), own data = N days (30, 90) from winter (last N days
before February) or summer/autumn (first N days of the series), or the full history Jul--Jan. Both arms start from the
synthetic-fleet quantile TFT (sim prior), or with --init scratch from random weights:
  local      fine-tuning on the client's own data only
  federated  FedGradient over the client's parks (own data) and the remaining parks (full history)
Early stopping on the client's own parks, February--May 2024 (agreed protocol), evaluation on the client's parks.

    frcst/bin/python scripts/pv_boundary_real.py --seed 42 --gpu 0 --output results/parks_pv_v4_30min/boundary_real_2026-11-11_v1
"""
import argparse
import copy
import itertools
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_quantile_finetune as ft  # noqa: E402
from pv_quantile_coldstart import label_bounds  # noqa: E402
from pv_quantile_metrics import metrics  # noqa: E402
from pv_fl_story_real import window_subset, FEB, SERIES_START  # noqa: E402

BUDGETS = [(30, 'winter'), (30, 'summer'), (90, 'winter'), (90, 'summer'), (None, 'full')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output', required=True)
    ap.add_argument('--sizes', type=int, nargs='+', default=[1, 2, 3], help='client sizes (parks per client) to run')
    ap.add_argument('--init', choices=['sim', 'scratch'], default='sim', help='sim: fleet-pretrained start; scratch: random init')
    a = ap.parse_args()
    out = Path(a.output) / f'seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{a.gpu}')
    cfg = ft.base.load_cfg(ft.base.DATA_CFG)
    cfg['model'] = copy.deepcopy(ft.base.load_cfg(ft.base.Q_CFG)['model'])
    cfg['model']['early_stopping']['monitor'] = 'val_pinball'
    qs = cfg['model']['tft']['quantiles']
    state0, (sx, _) = ft.finetune.load_init(ft.base.Q_MODEL.format(seed=a.seed))
    cfg['scaler_x'] = sx
    features = ft.preprocessing.get_features(config=cfg)
    dfs = ft.preprocessing.get_data(data_dir=cfg['data']['path'], config=cfg, freq=cfg['data']['freq'], features=features,
                                    files_key='files')
    data, bounds = {}, {}
    for key, frame in dfs.items():
        prepared, _ = ft.preprocessing.pipeline(data=frame, config=cfg, known_cols=features['known'], observed_cols=features['observed'],
                                                static_cols=features['static'], target_col=ft.preprocessing.get_target_cols(cfg)[0])
        data[key], bounds[key] = prepared, label_bounds(frame, prepared['index_train'])
    del dfs
    keys = sorted(data)
    park = {k: k.replace('synth_', '').replace('_meter.csv', '') for k in keys}
    cfg['model']['feature_dim'] = ft.tools.get_feature_dim(X=data[keys[0]]['X_train'])
    hp = ft.hpo.get_hyperparameters(config=cfg)
    hp['lr'] = 3e-4
    if a.init == 'scratch':                                                  # same architecture, random weights
        ft.tools.set_seed(a.seed)
        state0 = {k: v.detach().cpu().clone() for k, v in ft.models.get_model(config=cfg, hyperparameters=hp).state_dict().items()}
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() + ' + dirty'
    rng = np.random.default_rng(1)                                                   # same clients for every seed
    clients = [(k,) for k in keys]
    for p in (2, 3):
        combos = list(itertools.combinations(keys, p))
        clients += [combos[i] for i in rng.choice(len(combos), size=6, replace=False)]
    rows = pd.read_csv(out / 'metrics.csv').to_dict('records') if (out / 'metrics.csv').exists() else []
    done = {r['case'] for r in rows}

    def own(k, days, season):
        if season == 'full':
            return ft.cat([data[k]], 'train')
        start, end = (FEB - pd.Timedelta(days=days), FEB) if season == 'winter' else (SERIES_START, SERIES_START + pd.Timedelta(days=days))
        return window_subset(data[k], *bounds[k], start, end)[0]

    for cl in [c for c in clients if len(c) in a.sizes]:
        tag = '+'.join(park[k] for k in cl)
        rest = [k for k in keys if k not in cl]
        for days, season in BUDGETS:
            mine = [own(k, days, season) for k in cl]
            for arm in ('local', 'federated'):
                case = f'P{len(cl)}/{tag}/{season}{days or ""}/{arm}'
                if case in done:
                    continue
                cdir = out / case
                cdir.mkdir(parents=True, exist_ok=True)
                if arm == 'local':
                    X = {f: np.concatenate([m[0][f] for m in mine]) for f in mine[0][0]}
                    train_data, name = [(X, np.concatenate([m[1] for m in mine]))], 'local_client'
                else:
                    train_data, name = mine + [ft.cat([data[k]], 'train') for k in rest], 'fedgradient'
                model, log, _ = ft.train(name, train_data, [data[k] for k in cl], state0, cfg, hp, device, a.seed, qs, cdir)
                model.eval()
                for k in cl:
                    it = data[k]
                    pred = ft.base.predict(model, it['X_test'], device)
                    rows.append(dict(case=case, client=tag, n_parks=len(cl), park=park[k], days=days or -1, season=season,
                                     arm=arm, init=a.init, seed=a.seed, best_epoch=log['best_epoch'], date='2026-11-11', code_pin=pin,
                                     **metrics(pred, it['y_test'], qs, np.asarray(it['nwp_raw_test']) > 10)))
                pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
                del model
                torch.cuda.empty_cache()
    (out / 'COMPLETE').write_text('done\n')


if __name__ == '__main__':
    main()
