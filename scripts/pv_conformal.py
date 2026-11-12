#!/usr/bin/env python3
"""Uncertainty on the six meters (Viktor 2026-11-09): does pooling real data pay for the calibration of prediction
intervals? Quantile TFT trained on the synthetic fleet only (config_pv_v4_central_jan_nohist_q7, no power history) gives
zero-shot quantiles; conformalized quantile regression (CQR) widens them with real residuals of the meters.

Calibration: the meters' own data 2023-07-24..2024-01-31 (the training windows, strict split: targets before February); evaluation:
February..May 2024; daytime cells only (gti_nwp > 10 W/m2). Calibration sets per evaluated park:
  raw     no calibration (zero-shot quantiles)
  local   scores of the park itself
  pooled  scores of all six meters (= federated: only the score quantile has to be shared)
  loo     scores of the five other meters (newcomer without own data)
each once for all daytime cells and once per NWP-irradiance bin (Mondrian; fewer scores per bin).
Reference: the point model (config_pv_v4_central_jan_nohist) with symmetric conformal intervals |y - yhat|.

    frcst/bin/python scripts/pv_conformal.py --seeds 42 43 44 --gpu 0
Output: results/parks_pv_v4_30min/conformal/{per_park.csv, summary.csv}"""
import argparse
import copy
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import finetune, hpo, models, preprocessing, tools  # noqa: E402

DATA_CFG = 'configs/pv_parks/config_pv_v4_trianel_real_jan_nohist'
Q_CFG = 'configs/pv_parks/config_pv_v4_central_jan_nohist_q7'
Q_MODEL = 'models/cl_m-tft_out-96_freq-30min_pv_v4_central_jan_nohist_q7_seed{seed}.pt'
P_CFG = 'configs/pv_parks/config_pv_v4_central_jan_nohist'
P_MODEL = 'models/cl_m-tft_out-96_freq-30min_pv_v4_central_jan_nohist.pt'
CAL = ('2023-07-24', '2024-01-31 23:30')                                  # training split of the framework, strict_split: targets end before February
EV = ('2024-02-01', '2024-05-31')
LEVELS = {0.9: (0.05, 0.95), 0.8: (0.1, 0.9), 0.5: (0.25, 0.75)}
BINS = [10, 100, 300, 600, np.inf]                                       # gti_nwp W/m2 (Mondrian classes)


def load_cfg(path):
    cfg = tools.handle_freq(config=tools.load_config(path + '.yaml'))
    cfg['model']['name'], cfg['model']['fl'] = 'tft', False
    return cfg


def windows(dfs, cfg, features, sx, train, test):
    c = copy.deepcopy(cfg)
    c['data'].update(train_start=train[0], train_end=train[1], test_start=test[0], test_end=test[1])
    return {it['key']: it for it in tools.create_data_generator(dfs, c, features, scaler_x=sx, scaler_y=None)}


def model_from(cfg_path, model_path, X, device):
    cfg = load_cfg(cfg_path)
    cfg['model']['feature_dim'] = tools.get_feature_dim(X=X)
    hp = hpo.get_hyperparameters(config=cfg)
    sd, (sx, _) = finetune.load_init(model_path)
    m = models.get_model(config=cfg, hyperparameters=hp)
    m.load_state_dict(sd)
    return m.to(device).eval(), sx, cfg['model']['tft'].get('quantiles') or [0.5]


def predict(model, X, device, bs=256):
    out = []
    keys = ['observed', 'known'] + (['static'] if 'static' in X else [])
    n = len(X['known'])
    with torch.no_grad():
        for i in range(0, n, bs):
            t = [torch.as_tensor(np.asarray(X[k][i:i + bs]), dtype=torch.float32, device=device) for k in keys]
            if 'static' not in X:
                t.append(None)
            out.append(model(*t).cpu().numpy())
    p = np.concatenate(out)
    return p if p.ndim == 3 else p[..., None]                            # (n, horizon, Q)


def cells(it, pred, period):
    """Flat daytime cells of one park: q (cells, Q), y, gti_nwp; only valid times inside the period."""
    idx = pd.DatetimeIndex(it['index_test'])
    h = pred.shape[1]
    valid = idx.values[:, None] + (np.arange(h) * np.timedelta64(30, 'm'))[None, :]
    lo, up = np.datetime64(pd.Timestamp(period[0]).tz_localize(None)), np.datetime64(pd.Timestamp(period[1]).tz_localize(None) + pd.Timedelta(days=1))
    valid = valid.astype('datetime64[ns]')
    nwp = np.asarray(it['nwp_raw_test'], dtype=float)
    y = np.asarray(it['y_test'], dtype=float)
    m = (nwp > 10) & np.isfinite(y) & (valid >= lo) & (valid < up)
    return np.sort(pred[m], axis=-1), y[m], nwp[m]                       # sort: no quantile crossing


def train_cells(it, pred, last_days=None):
    """Calibration cells from the training windows. gti_nwp is not stored for them: recover it from the scaled known
    feature by the linear map that reproduces nwp_raw_test on the test windows of the same park."""
    kf = np.asarray(it['X_test']['known'])[:, -pred.shape[1]:, :]
    raw = np.asarray(it['nwp_raw_test'], dtype=float)
    fits = []
    for j in range(kf.shape[-1]):
        x, z = kf[..., j].ravel(), raw.ravel()
        A = np.c_[x, np.ones_like(x)]
        coef, res, *_ = np.linalg.lstsq(A, z, rcond=None)
        fits.append((float(np.abs(A @ coef - z).max()), j, coef))
    err, j, coef = min(fits, key=lambda t: t[0])
    if err > 1e-2:
        raise ValueError(f"no known feature reproduces gti_nwp (max error {err:.3g} W/m2)")
    nwp = np.asarray(it['X_train']['known'])[:, -pred.shape[1]:, j] * coef[0] + coef[1]
    y = np.asarray(it['y_train'], dtype=float)
    if last_days:                                                          # windows are chronological, one per day
        pred, y, nwp = pred[-last_days:], y[-last_days:], nwp[-last_days:]
    m = (nwp > 10) & np.isfinite(y)
    return np.sort(pred[m], axis=-1), y[m], nwp[m]


def cqr_quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        return np.nan
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(scores)[k - 1])


def interval_metrics(lo, hi, y, alpha):
    cov = (y >= lo) & (y <= hi)
    width = hi - lo
    score = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return dict(coverage=cov.mean(), width=width.mean(), interval_score=score.mean(), n=len(y))


def evaluate(cal, ev, qs, kind, seed):
    """cal/ev: park -> (q, y, nwp). kind 'quantile' (CQR) or 'point' (symmetric)."""
    rows = []
    parks = sorted(ev)
    for level, (ql, qh) in LEVELS.items():
        alpha = 1 - level
        if kind == 'quantile':
            il, ih = qs.index(ql), qs.index(qh)
            score = {p: np.maximum(c[0][:, il] - c[1], c[1] - c[0][:, ih]) for p, c in cal.items()}
        else:
            score = {p: np.abs(c[1] - c[0][:, 0]) for p, c in cal.items()}
        bins = {p: np.digitize(c[2], BINS) for p, c in cal.items()}
        for p in parks:
            q, y, nwp = ev[p]
            b_ev = np.digitize(nwp, BINS)
            if kind == 'quantile':
                base_lo, base_hi = q[:, il], q[:, ih]
            else:
                base_lo = base_hi = q[:, 0]
            sets = {'raw': None, 'local': [p], 'pooled': parks, 'loo': [j for j in parks if j != p]}
            for name, members in sets.items():
                for mondrian in (False, True):
                    if name == 'raw' and (mondrian or kind == 'point'):
                        continue
                    if members is None:
                        c = np.zeros(len(y))
                    elif not mondrian:
                        c = np.full(len(y), cqr_quantile(np.concatenate([score[j] for j in members]), alpha))
                    else:
                        c = np.zeros(len(y))
                        for b in np.unique(b_ev):
                            s = np.concatenate([score[j][bins[j] == b] for j in members])
                            c[b_ev == b] = cqr_quantile(s, alpha)
                    lo, hi = np.clip(base_lo - c, 0, None), base_hi + c
                    rows.append(dict(kind=kind, seed=seed, level=level, calibration=name, mondrian=mondrian, park=p,
                                     n_cal=0 if members is None else int(sum(len(score[j]) for j in members)),
                                     **interval_metrics(lo, hi, y, alpha)))
    return rows


SCARCE_DAYS = [7, 14, 30, 60, 90]


def scarce(own, cal, ev, qs, seed, nd):
    """Park p has only its last nd days: local = those days only, hybrid = those days + the full data of the others."""
    rows = []
    parks = sorted(ev)
    for level, (ql, qh) in LEVELS.items():
        alpha = 1 - level
        il, ih = qs.index(ql), qs.index(qh)
        sc = lambda c: np.maximum(c[0][:, il] - c[1], c[1] - c[0][:, ih])
        for p in parks:
            q, y, nwp = ev[p]
            sets = {'local_scarce': [sc(own[p])], 'hybrid_scarce': [sc(own[p])] + [sc(cal[j]) for j in parks if j != p]}
            for name, ss in sets.items():
                c = cqr_quantile(np.concatenate(ss), alpha)
                lo, hi = np.clip(q[:, il] - c, 0, None), q[:, ih] + c
                rows.append(dict(kind='quantile', seed=seed, level=level, calibration=f'{name}_{nd:02d}d', mondrian=False, park=p,
                                 n_cal=int(sum(len(x) for x in ss)), **interval_metrics(lo, hi, y, alpha)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    device = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    dcfg = load_cfg(DATA_CFG)
    features = preprocessing.get_features(config=dcfg)
    dfs = preprocessing.get_data(data_dir=dcfg['data']['path'], config=dcfg, freq=dcfg['data']['freq'], features=features, files_key='files')
    rows = []
    runs = [('quantile', s, Q_CFG, Q_MODEL.format(seed=s)) for s in a.seeds] + [('point', 42, P_CFG, P_MODEL)]
    for kind, seed, cpath, mpath in runs:
        _, (sx, _) = finetune.load_init(mpath)
        we = windows(dfs, dcfg, features, sx, CAL, EV)                    # X_train = calibration, X_test = evaluation
        k0 = sorted(we)[0]
        model, _, qs = model_from(cpath, mpath, we[k0]['X_test'], device)
        cal = {k.replace('synth_', '').replace('_meter.csv', ''): train_cells(we[k], predict(model, we[k]['X_train'], device)) for k in we}
        ev = {k.replace('synth_', '').replace('_meter.csv', ''): cells(we[k], predict(model, we[k]['X_test'], device), EV) for k in we}
        for p in ev:
            logging.info(f"{kind} seed {seed} {p}: calibration cells {len(cal[p][1])}, evaluation cells {len(ev[p][1])}")
        if kind == 'quantile':                                             # median check against the point model (0.635)
            mi = qs.index(0.5)
            r2 = np.mean([1 - ((c[0][:, mi] - c[1]) ** 2).sum() / ((c[1] - c[1].mean()) ** 2).sum() for c in ev.values()])
            logging.info(f"quantile seed {seed}: median R2 day on the meters {r2:.3f}")
        rows += evaluate(cal, ev, qs, kind, seed)
        if kind == 'quantile':                                             # own data scarce: last N days before February
            pc = {k.replace('synth_', '').replace('_meter.csv', ''): predict(model, we[k]['X_train'], device) for k in we}
            for nd in SCARCE_DAYS:
                own = {k.replace('synth_', '').replace('_meter.csv', ''): train_cells(we[k], pc[k.replace('synth_', '').replace('_meter.csv', '')], nd)
                       for k in we}
                rows += scarce(own, cal, ev, qs, seed, nd)
    df = pd.DataFrame(rows)
    out = os.path.join('results', os.path.basename(dcfg['data']['path']), 'conformal')
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, 'per_park.csv'), index=False)
    s = df.groupby(['kind', 'level', 'calibration', 'mondrian'])[['coverage', 'width', 'interval_score']].mean()
    s['coverage_worst_park'] = df.groupby(['kind', 'level', 'calibration', 'mondrian', 'park']).coverage.mean() \
        .groupby(['kind', 'level', 'calibration', 'mondrian']).apply(lambda x: x.iloc[np.argmax(np.abs(x.values - x.name[1]))])
    s.round(4).to_csv(os.path.join(out, 'summary.csv'))
    print(s.round(4).to_string())


if __name__ == '__main__':
    main()
