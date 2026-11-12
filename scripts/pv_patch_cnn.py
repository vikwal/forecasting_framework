#!/usr/bin/env python3
"""ICON-D2 16x16 patches for the six real meters (Viktor 2026-11-12): does spatial NWP context help the day-ahead
quantile TFT? Same data, split and protocol as the dependence-boundary study (train 2023-07-24..2024-01-31, early
stopping and evaluation Feb--May 2024 on the meters, seven quantiles, scratch init, lr 3e-4, no HPO).
Arms (central = FedGradient over the six parks, identical to pooled training; local = one model per park):
  point    TFT on the four point features (reference; samples restricted to runs with a complete patch)
  stats    TFT + patch summaries (mean/std of gti and clct over 3x3, 7x7, 16x16 per step)
  cnn      2D CNN per time step on the patch (gti, clct) -> 16-dim embedding -> TFT
  cnn3     as cnn, first convolution spans three time steps

    frcst/bin/python scripts/pv_patch_cnn.py --prepare --output results/parks_pv_v4_30min/patch_cnn_2026-11-12_v1
    frcst/bin/python scripts/pv_patch_cnn.py --seed 42 --gpu 0 --output results/parks_pv_v4_30min/patch_cnn_2026-11-12_v1
"""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_quantile_finetune as ft  # noqa: E402
from pv_quantile_metrics import metrics  # noqa: E402
import pv_patch_data as pdata  # noqa: E402
from utils.patch_models import PatchTFT, tft_kwargs  # noqa: E402

PATCH_PATH = os.path.join(os.environ['DATA_ROOT'], 'nwp_ready_pv', 'icon-d2_patch16')
ARMS = ['point', 'stats', 'cnn', 'cnn3']
PLANS = [('central', a) for a in ARMS] + [('local', a) for a in ARMS]


def load_cfg():
    cfg = ft.base.load_cfg(ft.base.DATA_CFG)
    cfg['model'] = copy.deepcopy(ft.base.load_cfg(ft.base.Q_CFG)['model'])
    cfg['model']['early_stopping']['monitor'] = 'val_pinball'
    return cfg


def frames(cfg):
    features = ft.preprocessing.get_features(config=cfg)
    dfs = ft.preprocessing.get_data(data_dir=cfg['data']['path'], config=cfg, freq=cfg['data']['freq'],
                                    features=features, files_key='files')
    return features, dfs


def park_id(key):
    return key.replace('synth_', '').replace('.csv', '')


def prepare(out):
    cache = out / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    _, dfs = frames(cfg)
    for key in sorted(dfs):
        pid = park_id(key)
        path = pdata.cache_path(str(cache), pid)
        if os.path.exists(path):
            continue
        p = pdata.load_patch(cfg, pid, PATCH_PATH)
        p.to_parquet(path)
        logging.info('%s: %d runs with a complete patch', pid, p.index.get_level_values('starttime').nunique())


def build_data(cfg, sx, root, arms=('point', 'stats', 'cnn')):
    """Prepared windows per arm and park; patch and summary columns joined to every park frame (runs without a
    complete patch are dropped in all arms). Returns data, keys, base_known, runs lost per park, base_cols."""
    features, dfs = frames(cfg)
    keys = sorted(dfs)
    base_known = list(features['known'])

    # patch and summary columns joined to every park frame; runs without a complete patch are dropped in all arms
    full, lost = {}, {}
    for k in keys:
        p = pd.read_parquet(pdata.cache_path(str(root / 'cache'), park_id(k)))
        p = pd.concat([p, pdata.add_stats(p)], axis=1)
        full[k], lost[park_id(k)] = pdata.attach(dfs[k], p)
    del dfs
    base_cols = [c for c in full[keys[0]].columns if c not in pdata.patch_cols() + pdata.stat_cols() + ['power']]
    train_end = pd.Timestamp(cfg['data']['train_end'], tz='UTC')
    pcols, scols = pdata.patch_cols(), pdata.stat_cols()
    arm_cols = {'point': [], 'stats': scols, 'cnn': pcols, 'cnn3': pcols}
    arm_known = {'point': base_known, 'stats': base_known + scols, 'cnn': base_known + ['pgti', 'pclct'],
                 'cnn3': base_known + ['pgti', 'pclct']}

    data = {}
    for arm in arms:
        sel = {k: full[k][base_cols + arm_cols[arm] + ['power']] for k in keys}
        shared = {f'p{c}': [f'p{c}_{i}' for i in range(pdata.SIZE ** 2)] for c in pdata.CHANNELS} if arm == 'cnn' else {}
        acfg = copy.deepcopy(cfg)
        acfg['scaler_x'] = pdata.extended_scaler(sx, base_cols, list(sel.values()), train_end, shared)
        data[arm] = {}
        for k in keys:
            prepared, _ = ft.preprocessing.pipeline(data=sel[k], config=acfg, known_cols=arm_known[arm],
                                                    observed_cols=features['observed'], static_cols=features['static'],
                                                    target_col=ft.preprocessing.get_target_cols(cfg)[0])
            data[arm][k] = prepared
        logging.info('%s: known dim %d', arm, data[arm][keys[0]]['X_train']['known'].shape[-1])
    if 'cnn' in data:
        data['cnn3'] = data['cnn']
    del full

    return data, keys, base_known, lost, base_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output', required=True)
    ap.add_argument('--prepare', action='store_true', help='only build the patch cache (run once before the seeds)')
    ap.add_argument('--plans', nargs='+', default=[f'{s}/{a}' for s, a in PLANS])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    root = Path(a.output)
    if a.prepare:
        prepare(root)
        return
    out = root / f'seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{a.gpu}')
    cfg = load_cfg()
    qs = cfg['model']['tft']['quantiles']
    _, (sx, _) = ft.finetune.load_init(ft.base.Q_MODEL.format(seed=a.seed))
    data, keys, base_known, lost, base_cols = build_data(cfg, sx, root)
    cfg['model']['feature_dim'] = ft.tools.get_feature_dim(X=data['point'][keys[0]]['X_train'])
    hp = ft.hpo.get_hyperparameters(config=cfg)
    hp['lr'] = 3e-4
    n_point = len(base_known)

    def factory(arm):
        if arm in ('point', 'stats'):
            c = copy.deepcopy(cfg)
            c['model']['feature_dim'] = ft.tools.get_feature_dim(X=data[arm][keys[0]]['X_train'])
            return lambda: ft.models.get_model(config=c, hyperparameters=hp)
        kw = tft_kwargs(cfg, hp)
        return lambda: PatchTFT(n_point, len(pdata.CHANNELS), pdata.SIZE, 16, kw, t_kernel=3 if arm == 'cnn3' else 1)

    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() + ' + dirty'
    (out / 'metadata.json').write_text(json.dumps(dict(
        seed=a.seed, date='2026-11-12', code_pin=pin, lr=3e-4, quantiles=qs, patch_path=PATCH_PATH,
        runs_lost_to_patch_join=lost, base_cols=base_cols, hp={k: str(v) for k, v in hp.items()},
        sizes={k: {s: len(data['point'][k][f'y_{s}']) for s in ('train', 'test')} for k in keys}), indent=2))
    rows = pd.read_csv(out / 'metrics.csv').to_dict('records') if (out / 'metrics.csv').exists() else []
    done = {r['case'] for r in rows}
    for plan in a.plans:
        scope, arm = plan.split('/')
        make = factory(arm)
        ft.tools.set_seed(a.seed)
        state0 = {k: v.detach().cpu().clone() for k, v in make().state_dict().items()}
        n_params = sum(v.numel() for v in state0.values())
        groups = [keys] if scope == 'central' else [[k] for k in keys]
        for grp in groups:
            case = f'{scope}/{arm}' + ('' if scope == 'central' else f'/{park_id(grp[0])}')
            if case in done:
                continue
            cdir = out / case
            cdir.mkdir(parents=True, exist_ok=True)
            d = data[arm]
            name = 'fedgradient' if scope == 'central' else 'local_client'
            model, log, _ = ft.train(name, [ft.cat([d[k]], 'train') for k in grp], [d[k] for k in grp], state0, cfg, hp,
                                     device, a.seed, qs, cdir, factory=make)
            model.eval()
            for k in grp:
                pred = ft.base.predict(model, d[k]['X_test'], device)
                rows.append(dict(case=case, scope=scope, arm=arm, park=park_id(k).replace('_meter', ''), seed=a.seed,
                                 best_epoch=log['best_epoch'], epochs=log['epochs'], n_params=n_params,
                                 date='2026-11-12', code_pin=pin,
                                 **metrics(pred, d[k]['y_test'], qs, np.asarray(d[k]['nwp_raw_test']) > 10)))
                np.savez_compressed(cdir / f'pred_{park_id(k)}.npz', pred=pred, true=d[k]['y_test'],
                                    index=np.asarray(d[k]['index_test']).astype(str))
            pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
            del model
            torch.cuda.empty_cache()
    (out / 'COMPLETE').write_text('done\n')


if __name__ == '__main__':
    main()
