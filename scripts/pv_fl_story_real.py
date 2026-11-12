#!/usr/bin/env python3
"""Open experiments of the FL story on the six meters (Viktor 2026-11-11), same protocol as pv_quantile_coldstart.py:
quantile TFT from the synthetic fleet, training labels until January 2024, early stopping and evaluation February--May 2024,
lr 3e-4, FedGradient engine. FedGradient equals central (shown before), so shared runs use FedGradient only.

  E1 newcomer   shared fine-tuning on the five other parks only; early stopping on those five, never on the target
  E3 donors     newcomer with K = 1..4 donors (K = 1 all five, K = 2..4 two fixed random subsets each)
  E2 two-stage  local fine-tuning (own N days or full history, ES on the target) starting from the shared checkpoint
                of the cold-start study (same seed, budget and target)
  E7 coverage   own days from the start of the series (2023-07-24 + N days, summer/autumn) instead of the last N days
                before February; local and shared (with the full donors), N = 30, 90

    frcst/bin/python scripts/pv_fl_story_real.py --seed 42 --gpu 0 --output results/parks_pv_v4_30min/fl_story_real_2026-11-11_v1
"""
import argparse
import copy
import itertools
import json
import logging
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

FW = Path('results/parks_pv_v4_30min')
CS = FW / 'quantile_coldstart_2026-11-09_v1'
FT = FW / 'quantile_ft_2026-11-09_v1'
SERIES_START = pd.Timestamp('2023-07-24', tz='UTC')
FEB = pd.Timestamp('2024-02-01', tz='UTC')


def window_subset(item, lower, upper, start, end):
    """Windows whose whole target horizon lies in [start, end)."""
    lo, hi = pd.to_datetime(lower, utc=True), pd.to_datetime(upper, utc=True)
    keep = np.asarray((lo >= start) & (hi < end))
    if not keep.any():
        raise ValueError(f'no complete windows in {start}..{end}')
    return ({k: v[keep] for k, v in item['X_train'].items()}, item['y_train'][keep]), int(keep.sum())


def load_state(path):
    return torch.load(path, map_location='cpu', weights_only=False)['state_dict']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output', required=True)
    ap.add_argument('--only', nargs='*', default=['E1', 'E3', 'E2', 'E7'])
    a = ap.parse_args()
    out = Path(a.output) / f'seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{a.gpu}')
    cfg = ft.base.load_cfg(ft.base.DATA_CFG)
    cfg['model'] = copy.deepcopy(ft.base.load_cfg(ft.base.Q_CFG)['model'])
    cfg['model']['early_stopping']['monitor'] = 'val_pinball'
    assert cfg['params']['observed_features'] == [] and not cfg['data']['scale_y']
    qs = cfg['model']['tft']['quantiles']
    model_path = ft.base.Q_MODEL.format(seed=a.seed)
    state0, (sx, _) = ft.finetune.load_init(model_path)
    cfg['scaler_x'] = sx
    features = ft.preprocessing.get_features(config=cfg)
    dfs = ft.preprocessing.get_data(data_dir=cfg['data']['path'], config=cfg, freq=cfg['data']['freq'],
                                    features=features, files_key='files')
    data, bounds = {}, {}
    for key, frame in dfs.items():
        if frame.isna().any().any():
            raise ValueError(f'{key}: missing values')
        prepared, _ = ft.preprocessing.pipeline(data=frame, config=cfg, known_cols=features['known'],
                                                observed_cols=features['observed'], static_cols=features['static'],
                                                target_col=ft.preprocessing.get_target_cols(cfg)[0])
        data[key], bounds[key] = prepared, label_bounds(frame, prepared['index_train'])
    del dfs
    keys = sorted(data)
    park = {k: k.replace('synth_', '').replace('_meter.csv', '') for k in keys}
    assert len(keys) == 6
    cfg['model']['feature_dim'] = ft.tools.get_feature_dim(X=data[keys[0]]['X_train'])
    hp = ft.hpo.get_hyperparameters(config=cfg)
    hp['lr'] = 3e-4
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() + ' + dirty'
    full = {k: ft.cat([data[k]], 'train') for k in keys}
    rows, logs = [], []
    done = set()
    if (out / 'metrics.csv').exists():                                    # resume after an interruption
        prev = pd.read_csv(out / 'metrics.csv')
        rows = prev.to_dict('records')
        done = set(prev['case'])

    def run(case, experiment, variant, target, train_data, val_keys, start, **info):
        if case in done:
            return
        cdir = out / case
        cdir.mkdir(parents=True, exist_ok=True)
        name = ('local_' + park[target]) if variant.startswith('local') else 'fedgradient'
        model, log, _ = ft.train(name, train_data, [data[k] for k in val_keys], start, cfg, hp, device, a.seed, qs, cdir)
        model.eval()
        it = data[target]
        pred = ft.base.predict(model, it['X_test'], device)
        rows.append(dict(case=case, experiment=experiment, variant=variant, park=park[target], seed=a.seed,
                         best_epoch=log['best_epoch'], date='2026-11-11', code_pin=pin, **info,
                         **metrics(pred, it['y_test'], qs, np.asarray(it['nwp_raw_test']) > 10)))
        np.savez_compressed(cdir / 'pred.npz', pred=pred, true=it['y_test'], gti=it['nwp_raw_test'], quantiles=qs,
                            index=np.asarray(it['index_test']).astype(str))
        logs.append(dict(case=case, **{k: v for k, v in log.items() if not isinstance(v, (list, dict))}))
        pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
        (out / 'training.json').write_text(json.dumps(logs, indent=2))
        del model
        torch.cuda.empty_cache()

    rng = np.random.default_rng(0)                                           # same donor subsets for every seed
    for target in keys:
        p = park[target]
        donors = [k for k in keys if k != target]
        if 'E1' in a.only:
            run(f'E1/{p}', 'E1_newcomer', 'shared', target, [full[k] for k in donors], donors, state0, donors=5)
        if 'E3' in a.only:
            subsets = [(d,) for d in donors]
            for kk in (2, 3, 4):
                combos = list(itertools.combinations(donors, kk))
                subsets += [combos[i] for i in rng.choice(len(combos), size=2, replace=False)]
            for sub in subsets:
                tag = '+'.join(park[d] for d in sub)
                run(f'E3/{p}/{tag}', 'E3_donors', 'shared', target, [full[d] for d in sub], list(sub), state0,
                    donors=len(sub), donor_set=tag)
        if 'E2' in a.only:
            for days in (7, 14, 30, 60, 90, None):
                if days is None:
                    own, n = full[target], len(data[target]['y_train'])
                    start = load_state(FT / f'seed{a.seed}/fedgradient_best.pt')
                else:
                    own, n = window_subset(data[target], *bounds[target], FEB - pd.Timedelta(days=days), FEB)
                    start = load_state(CS / f'seed{a.seed}/{days:03d}d/{p}/fedgradient_best.pt')
                run(f'E2/{p}/{days or "full"}', 'E2_two_stage', 'local_from_shared', target, [own], [target], start,
                    days=days or -1, own_windows=n)
        if 'E7' in a.only:
            for days in (30, 90):
                own, n = window_subset(data[target], *bounds[target], SERIES_START, SERIES_START + pd.Timedelta(days=days))
                run(f'E7/{p}/{days}/local', 'E7_early_days', 'local', target, [own], [target], state0, days=days, own_windows=n)
                run(f'E7/{p}/{days}/shared', 'E7_early_days', 'shared', target, [own] + [full[k] for k in donors], keys,
                    state0, days=days, own_windows=n)
    (out / 'COMPLETE').write_text('done\n')


if __name__ == '__main__':
    main()
