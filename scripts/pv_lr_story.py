#!/usr/bin/env python3
"""Linear quantile regression as benchmark for every arm of the FL story (Viktor 2026-11-11).

Model: one linear model per quantile (7 quantiles) on the cells of the forecast windows, inputs = the four known NWP
features at the target step (gti_nwp, gti_clearsky, ecmwf_gti, solar_zenith_cos, scaled like the TFT) plus intercept,
fitted with the pinball loss on daytime cells (gti_nwp > 10 W/m2), prediction 0 at night. Federated training of a linear
model with gradient averaging is identical to pooled training, so "shared" stands for central and federated.

  real       six meters, protocol of the quantile study (training labels until January 2024, ES/evaluation Feb--May 2024):
             zero-shot (pretrained on the 166 synthetic fleet parks, Jul--Jan), local / shared fine-tuning from it (ES),
             newcomer (5 donors, ES on donors), own-days budgets 7..90 and full, two-stage, and from scratch without prior
             (local and shared, fitted to convergence)
  synthetic  scaling: from scratch on N of the 154 client parks with the last M months before August 2024,
             evaluated on the 56 held-out parks Aug 2024--Jul 2025 (same grid as pv_synthetic_scaling.py)

    frcst/bin/python scripts/pv_lr_story.py real --gpu 0 --output results/parks_pv_v4_30min/lr_story_2026-11-11_v1
    frcst/bin/python scripts/pv_lr_story.py synthetic --gpu 1 --output results/parks_pv_v4_30min/lr_story_2026-11-11_v1
"""
import argparse
import copy
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_conformal as base  # noqa: E402
from pv_quantile_coldstart import label_bounds  # noqa: E402
from pv_quantile_metrics import metrics  # noqa: E402
from pv_robustness_infer import train_gti  # noqa: E402
from pv_fl_story_real import window_subset, FEB  # noqa: E402
from pv_synthetic_scaling import PREPARED, N_PARKS, MONTHS, last_months  # noqa: E402
from utils import preprocessing  # noqa: E402

QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
H = 96


def cells(X, y, gti):
    """Flat design (cells, F+1) with intercept, targets and daytime mask from windows (known: history + horizon)."""
    k = np.asarray(X['known'])[:, -H:, :]
    Z = np.concatenate([k.reshape(-1, k.shape[-1]), np.ones((k.shape[0] * H, 1))], axis=1)
    return Z.astype(np.float32), np.asarray(y, dtype=np.float32).ravel(), (np.asarray(gti) > 10).ravel()


def train_cells(it):
    return cells(it['X_train'], it['y_train'], train_gti(it, H))


def test_cells(it):
    return cells(it['X_test'], it['y_test'], it['nwp_raw_test'])


def stack(sets):
    Z = np.concatenate([s[0][s[2]] for s in sets])
    y = np.concatenate([s[1][s[2]] for s in sets])
    return Z, y


class Fitter:
    def __init__(self, device):
        self.dev = device
        self.q = torch.tensor(QS, device=device)

    def loss(self, W, Z, y):
        e = y[:, None] - Z @ W
        return torch.maximum(self.q * e, (self.q - 1) * e).mean()

    def fit(self, train, W0=None, val=None, steps=3000, lr=0.02, every=25, patience=10):
        """Adam on the pinball loss, full batch. Without val: run to `steps`. With val: early stopping on the validation
        pinball (start state is a candidate), as for the TFT."""
        Z, y = (torch.as_tensor(a, device=self.dev) for a in stack(train))
        W = torch.zeros(Z.shape[1], len(QS), device=self.dev) if W0 is None else torch.as_tensor(W0, device=self.dev).clone()
        W.requires_grad_(True)
        opt = torch.optim.Adam([W], lr=lr)
        if val is not None:
            Zv, yv = (torch.as_tensor(a, device=self.dev) for a in stack(val))
            with torch.no_grad():
                best, best_W, best_step = float(self.loss(W, Zv, yv)), W.detach().clone(), 0
            wait = 0
        for s in range(1, steps + 1):
            opt.zero_grad()
            loss = self.loss(W, Z, y)
            loss.backward()
            opt.step()
            if val is not None and s % every == 0:
                with torch.no_grad():
                    v = float(self.loss(W, Zv, yv))
                if v < best - 1e-6:
                    best, best_W, best_step, wait = v, W.detach().clone(), s, 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if val is not None:
            return best_W.cpu().numpy(), best_step
        return W.detach().cpu().numpy(), steps

    @staticmethod
    def predict(W, c):
        Z, _, day = c
        p = Z @ W
        p[~day] = 0.0
        return p.reshape(-1, H, len(QS))


def evaluate(W, it, c):
    pred = Fitter.predict(W, c)
    return metrics(pred, np.asarray(it['y_test']), QS, np.asarray(it['nwp_raw_test']) > 10)


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
    tr = {k: train_cells(data[k]) for k in keys}
    te = {k: test_cells(data[k]) for k in keys}

    def own_cells(k, start, end):
        (X, y), n = window_subset(data[k], *bounds[k], start, end)
        lo = pd.to_datetime(bounds[k][0], utc=True)
        hi = pd.to_datetime(bounds[k][1], utc=True)
        keep = np.asarray((lo >= start) & (hi < end))
        g = train_gti(data[k], H)[keep]
        return cells(X, y, g), n

    # pretraining on the synthetic fleet (166 parks, Jul--Jan), the TFT's training set
    qcfg = base.load_cfg(base.Q_CFG)
    qcfg['data'].update(holdout_files=[], twin_files=[], meter_files=[])
    fdfs = preprocessing.get_data(data_dir=qcfg['data']['path'], config=qcfg, freq=qcfg['data']['freq'], features=features,
                                  files_key='files')
    fleet = [train_cells(it) for it in base.windows(fdfs, qcfg, features, sx, base.CAL, base.EV).values() if len(it['y_train'])]
    del fdfs
    W_sim, _ = fit.fit(fleet, steps=4000)
    rows = []

    def add(experiment, variant, k, W, **info):
        rows.append(dict(experiment=experiment, variant=variant, park=park[k], date='2026-11-11', code_pin=pin, **info,
                         **evaluate(W, data[k], te[k])))

    for k in keys:
        donors = [j for j in keys if j != k]
        add('full', 'zeroshot_sim', k, W_sim)
        W, s = fit.fit([tr[k]], W_sim, val=[te[k]]); add('full', 'local_ft', k, W, step=s)
        W, s = fit.fit([tr[j] for j in keys], W_sim, val=[te[j] for j in keys]); add('full', 'shared_ft', k, W, step=s)
        W, _ = fit.fit([tr[k]]); add('full', 'local_scratch', k, W)
        W, _ = fit.fit([tr[j] for j in keys]); add('full', 'shared_scratch', k, W)
        W_new, s = fit.fit([tr[j] for j in donors], W_sim, val=[te[j] for j in donors]); add('E1', 'newcomer_ft', k, W_new, step=s)
        W, _ = fit.fit([tr[j] for j in donors]); add('E1', 'newcomer_scratch', k, W)
        for days in (7, 14, 30, 60, 90):
            own, n = own_cells(k, FEB - pd.Timedelta(days=days), FEB)
            W, s = fit.fit([own], W_sim, val=[te[k]]); add('coldstart', 'local_ft', k, W, days=days, step=s)
            W_sh, s = fit.fit([own] + [tr[j] for j in donors], W_sim, val=[te[j] for j in keys])
            add('coldstart', 'shared_ft', k, W_sh, days=days, step=s)
            W, s = fit.fit([own], W_sh, val=[te[k]]); add('coldstart', 'two_stage', k, W, days=days, step=s)
            W, _ = fit.fit([own]); add('coldstart', 'local_scratch', k, W, days=days)
            W, _ = fit.fit([own] + [tr[j] for j in donors]); add('coldstart', 'shared_scratch', k, W, days=days)
        pd.DataFrame(rows).to_csv(a.output / 'real_metrics.csv', index=False)
    np.save(a.output / 'W_sim.npy', W_sim)


def synthetic(a, fit, pin):
    with PREPARED.open('rb') as f:
        bundle = pickle.load(f)
    pool, held = bundle['target'], bundle['source']
    perm = [sorted(pool)[i] for i in np.random.default_rng(0).permutation(len(pool))]   # same order as the TFT study
    hc = {k: test_cells(held[k]) for k in sorted(held)}
    rows = []
    for n in N_PARKS:
        for months in MONTHS:
            sets = []
            for k in perm[:n]:
                it = pool[k]
                X, y = last_months(it, months)
                g = train_gti(it, H)[-len(y):]
                sets.append(cells(X, y, g))
            W, _ = fit.fit(sets, steps=4000)
            per = pd.DataFrame([evaluate(W, held[k], hc[k]) for k in sorted(held)]).mean().to_dict()
            rows.append(dict(n_parks=n, months=months, train_cells=int(sum(s[2].sum() for s in sets)), date='2026-11-11',
                             code_pin=pin, **per))
            pd.DataFrame(rows).to_csv(a.output / 'synthetic_scaling_metrics.csv', index=False)
            print(n, months, round(per['pinball_day'], 6), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['real', 'synthetic'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    fit = Fitter(torch.device(f'cuda:{a.gpu}'))
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() + ' + dirty'
    (real if a.mode == 'real' else synthetic)(a, fit, pin)


if __name__ == '__main__':
    main()
