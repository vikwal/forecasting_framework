#!/usr/bin/env python3
"""Dependence boundary with linear quantile regression (Viktor 2026-11-11): parts B (synthetic map) and C (meters) of
pv_boundary_synthetic.py / pv_boundary_real.py with the LR of pv_lr_story.py, plus part D: per weather regime (k-means, k = 3,
on the forecast clear-sky index of the day and its intraday variability, known at issue time) the loss of the local and the
federated arm, so that "federate only the hard regimes" can be evaluated. LR is cheap, so B uses 12 client draws; --partners sets the partner pool (5/20/50/100). Federated = pooled (exact for a linear model).

    frcst/bin/python scripts/pv_boundary_lr.py synthetic --gpu 0 --output results/parks_pv_v4_30min/boundary_lr_2026-11-11_v1
    frcst/bin/python scripts/pv_boundary_lr.py real --gpu 0 --output results/parks_pv_v4_30min/boundary_lr_2026-11-11_v1
"""
import argparse
import copy
import itertools
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_conformal as base  # noqa: E402
from pv_lr_story import Fitter, QS, H, cells, train_cells, test_cells  # noqa: E402
from pv_robustness_infer import train_gti  # noqa: E402
from pv_quantile_coldstart import label_bounds  # noqa: E402
from pv_fl_story_real import window_subset, FEB, SERIES_START  # noqa: E402
from pv_boundary_synthetic import SIZES, PLAN, STARTS  # noqa: E402
from pv_synthetic_scaling import PREPARED  # noqa: E402
from utils import preprocessing  # noqa: E402

FLEET = Path('/mnt/lambda1/nvme1/synthetic/solar/parks_pv_v4_30min')
TWIN = Path('/mnt/lambda1/nvme1/synthetic/solar/trianel_twins_30min')


def regimes(items, clear_file):
    """park -> (windows, horizon) regime labels 0 (dark) .. 2 (clear) from the forecast."""
    feats, cellmaps = [], {}
    for k, it in items.items():
        start = pd.DatetimeIndex(it['index_test']).tz_convert('UTC').tz_localize(None)
        vt = start.values[:, None] + np.arange(H) * np.timedelta64(30, 'm')
        clear = pd.read_parquet(clear_file(k), columns=['ghi_clear_cams'])['ghi_clear_cams']
        clear.index = clear.index.tz_convert('UTC').tz_localize(None)
        gc = clear.reindex(pd.DatetimeIndex(vt.ravel())).values.reshape(vt.shape)
        day = pd.DatetimeIndex(vt.ravel()).normalize().values.reshape(vt.shape)
        c = pd.DataFrame({'gti': np.asarray(it['nwp_raw_test'], float).ravel(), 'clear': gc.ravel(), 'date': day.ravel()})
        c = c[c.clear > 50]
        c['k'] = c.gti / c.clear
        f = c.groupby('date').agg(s=('gti', 'sum'), c=('clear', 'sum'), k_var=('k', 'std'))
        f['k_day'] = f.s / f.c
        f = f[['k_day', 'k_var']].dropna().reset_index()
        f['key'] = k
        feats.append(f)
        cellmaps[k] = day
    F = pd.concat(feats, ignore_index=True)
    X = F[['k_day', 'k_var']].values
    km = KMeans(3, n_init=20, random_state=0).fit((X - X.mean(0)) / X.std(0))
    F['regime'] = km.labels_
    F['regime'] = F.regime.map({r: i for i, r in enumerate(F.groupby('regime').k_day.mean().sort_values().index)})
    out = {}
    for k, day in cellmaps.items():
        m = F[F.key == k].set_index('date').regime
        out[k] = pd.Series(day.ravel()).map(m).fillna(-1).astype(int).values.reshape(day.shape)
    return out


def scores(W, it, c, reg):
    pred = Fitter.predict(W, c)
    y = np.asarray(it['y_test'])
    day = np.asarray(it['nwp_raw_test']) > 10
    e = y[..., None] - pred
    loss = np.maximum(np.asarray(QS) * e, (np.asarray(QS) - 1) * e).mean(-1)
    res = {'pinball_day': float(loss[day].mean())}
    for r in range(3):
        m = day & (reg == r)
        res[f'pinball_r{r}'] = float(loss[m].mean()) if m.any() else np.nan
        res[f'n_r{r}'] = int(m.sum())
    return res


def synthetic(a, fit, pin, n_partners=20):
    with PREPARED.open('rb') as f:
        bundle = pickle.load(f)
    pool_parks, held = bundle['target'], bundle['source']
    del bundle
    order = sorted(pool_parks)
    reg = regimes(pool_parks, lambda k: FLEET / f"{Path(k).name.replace('.csv', '')}.parquet")
    tc = {k: test_cells(pool_parks[k]) for k in order}
    gti = {k: train_gti(pool_parks[k], H) for k in order}
    W_prior, _ = fit.fit([train_cells(held[k]) for k in sorted(held)], steps=4000)   # prior: the 56 source parks

    def own(k, months, start):
        it, n = pool_parks[k], len(pool_parks[k]['y_train'])
        lo = 0 if start == 'year' else STARTS[start]
        hi = n if start == 'year' else min(n, lo + int(round(30.4 * months)))
        X = {f: v[lo:hi] for f, v in it['X_train'].items()}
        return cells(X, it['y_train'][lo:hi], gti[k][lo:hi])

    rows = []
    for draw in range(12):                                                  # 12 independent client/partner draws
        perm = [order[i] for i in np.random.default_rng(100 + draw).permutation(len(order))]
        pos, targets = 20, {}                                                # targets independent of the partner count
        for P in SIZES:
            targets[P], pos = perm[pos:pos + P], pos + P
        partners = (perm[:20] + perm[pos:])[:n_partners]
        partner_cells = [own(k, 12, 'year') for k in partners]
        for P in SIZES:
            target = targets[P]
            for months, start in PLAN:
                mine = [own(k, months, start) for k in target]
                val = [tc[k] for k in target]
                for arm in ('local_scratch', 'federated_scratch', 'local_prior', 'federated_prior'):
                    train = mine if arm.startswith('local') else mine + partner_cells
                    W0 = W_prior if arm.endswith('prior') else None
                    W, _ = fit.fit(train, W0, val=val) if W0 is not None else fit.fit(train)
                    per = pd.DataFrame([scores(W, pool_parks[k], tc[k], reg[k]) for k in target]).mean().to_dict()
                    rows.append(dict(draw=draw, partners=n_partners, n_parks=P, months=months, start=start, arm=arm, date='2026-11-11',
                                     code_pin=pin, **per))
        name = 'synthetic_boundary_lr.csv' if n_partners == 20 else f'synthetic_boundary_lr_p{n_partners:03d}.csv'
        pd.DataFrame(rows).to_csv(a.output / name, index=False)
        print('draw', draw, flush=True)


def real(a, fit, pin):
    cfg = base.load_cfg(base.DATA_CFG)
    _, (sx, _) = base.finetune.load_init(base.Q_MODEL.format(seed=42))
    cfg['scaler_x'] = sx
    features = preprocessing.get_features(config=cfg)
    dfs = preprocessing.get_data(data_dir=cfg['data']['path'], config=cfg, freq=cfg['data']['freq'], features=features,
                                 files_key='files')
    data, bounds = {}, {}
    for key, frame in dfs.items():
        prepared, _ = preprocessing.pipeline(data=frame, config=cfg, known_cols=features['known'], observed_cols=features['observed'],
                                             static_cols=features['static'], target_col=preprocessing.get_target_cols(cfg)[0])
        data[key], bounds[key] = prepared, label_bounds(frame, prepared['index_train'])
    del dfs
    keys = sorted(data)
    park = {k: k.replace('synth_', '').replace('_meter.csv', '') for k in keys}
    reg = regimes(data, lambda k: TWIN / f"synth_{park[k]}.parquet")
    tc = {k: test_cells(data[k]) for k in keys}
    W_sim = np.load(a.output.parent / 'lr_story_2026-11-11_v1' / 'W_sim.npy')

    def own(k, days, season):
        if season == 'full':
            return train_cells(data[k])
        start, end = (FEB - pd.Timedelta(days=days), FEB) if season == 'winter' else (SERIES_START, SERIES_START + pd.Timedelta(days=days))
        (X, y), _ = window_subset(data[k], *bounds[k], start, end)
        lo, hi = pd.to_datetime(bounds[k][0], utc=True), pd.to_datetime(bounds[k][1], utc=True)
        keep = np.asarray((lo >= start) & (hi < end))
        return cells(X, y, train_gti(data[k], H)[keep])

    clients = [(k,) for k in keys] + list(itertools.combinations(keys, 2)) + list(itertools.combinations(keys, 3))
    rows = []
    for cl in clients:
        rest = [k for k in keys if k not in cl]
        for days, season in [(30, 'winter'), (30, 'summer'), (90, 'winter'), (90, 'summer'), (None, 'full')]:
            mine = [own(k, days, season) for k in cl]
            for arm in ('local_scratch', 'federated_scratch', 'local_prior', 'federated_prior'):
                train = mine if arm.startswith('local') else mine + [train_cells(data[k]) for k in rest]
                W = fit.fit(train, W_sim, val=[tc[k] for k in cl])[0] if arm.endswith('prior') else fit.fit(train)[0]
                for k in cl:
                    rows.append(dict(client='+'.join(park[j] for j in cl), n_parks=len(cl), park=park[k], days=days or -1,
                                     season=season, arm=arm, date='2026-11-11', code_pin=pin, **scores(W, data[k], tc[k], reg[k])))
        pd.DataFrame(rows).to_csv(a.output / 'real_boundary_lr.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['real', 'synthetic'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--partners', type=int, default=20, help='synthetic: number of partner parks (full year)')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    fit = Fitter(torch.device(f'cuda:{a.gpu}'))
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() + ' + dirty'
    real(a, fit, pin) if a.mode == 'real' else synthetic(a, fit, pin, a.partners)


if __name__ == '__main__':
    main()
