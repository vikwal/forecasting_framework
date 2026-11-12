#!/usr/bin/env python3
"""Zero-shot quantile forecasts for the robustness checks of the FL story (2026-11-10): the synthetic-fleet quantile TFT
(config_pv_v4_central_jan_nohist_q7, seeds 42/43/44, no power history) on the six meters and on their synthetic twins,
for the training period 2023-07-24..2024-01-31 and the evaluation period 2024-02-01..2024-05-31. Inference only.

    frcst/bin/python scripts/pv_robustness_infer.py --gpu 0
Output: results/parks_pv_v4_30min/robustness_2026-11-10_v1/{source}_seed{s}_{period}.npz with, per park, the window
start times (index), targets y, gti_nwp and the raw quantiles pred (windows, horizon, Q)."""
import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import finetune, preprocessing  # noqa: E402
from pv_conformal import CAL, EV, Q_CFG, Q_MODEL, load_cfg, model_from, predict, windows  # noqa: E402

SOURCES = {'meter': 'configs/pv_parks/config_pv_v4_trianel_real_jan_nohist',
           'twin': 'configs/pv_parks/config_pv_v4_trianel_twin_jan_nohist'}


def park_of(key):
    return key.replace('synth_', '').replace('_meter', '').replace('.csv', '').replace('.parquet', '')


def train_gti(it, h):
    """gti_nwp of the training windows from the scaled known feature that reproduces nwp_raw_test (see pv_conformal)."""
    kf = np.asarray(it['X_test']['known'])[:, -h:, :]
    raw = np.asarray(it['nwp_raw_test'], dtype=float)
    best = None
    for j in range(kf.shape[-1]):
        A = np.c_[kf[..., j].ravel(), np.ones(kf[..., j].size)]
        coef = np.linalg.lstsq(A, raw.ravel(), rcond=None)[0]
        err = float(np.abs(A @ coef - raw.ravel()).max())
        best = min(best or (np.inf,), (err, j, coef), key=lambda t: t[0])
    err, j, coef = best
    if err > 1e-2:
        raise ValueError(f'no known feature reproduces gti_nwp (max error {err:.3g})')
    return np.asarray(it['X_train']['known'])[:, -h:, j] * coef[0] + coef[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    import torch
    device = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    out = os.path.join('results', 'parks_pv_v4_30min', 'robustness_2026-11-10_v1')
    os.makedirs(out, exist_ok=True)
    for source, cpath in SOURCES.items():
        dcfg = load_cfg(cpath)
        features = preprocessing.get_features(config=dcfg)
        dfs = preprocessing.get_data(data_dir=dcfg['data']['path'], config=dcfg, freq=dcfg['data']['freq'],
                                     features=features, files_key='files')
        for seed in a.seeds:
            mpath = Q_MODEL.format(seed=seed)
            _, (sx, _) = finetune.load_init(mpath)
            we = windows(dfs, dcfg, features, sx, CAL, EV)
            model, _, qs = model_from(Q_CFG, mpath, we[sorted(we)[0]]['X_test'], device)
            for period in ('cal', 'ev'):
                arrays = {'quantiles': np.asarray(qs)}
                for k, it in we.items():
                    p = park_of(k)
                    X, y = (it['X_train'], it['y_train']) if period == 'cal' else (it['X_test'], it['y_test'])
                    pred = predict(model, X, device)
                    idx = it['index_train'] if period == 'cal' else it['index_test']
                    gti = train_gti(it, pred.shape[1]) if period == 'cal' else np.asarray(it['nwp_raw_test'], dtype=float)
                    arrays.update({f'{p}__pred': pred.astype(np.float32), f'{p}__y': np.asarray(y, dtype=np.float32),
                                   f'{p}__gti': gti.astype(np.float32),
                                   f'{p}__index': np.asarray(idx).astype('datetime64[ns]')})
                    logging.info(f'{source} seed {seed} {period} {p}: {pred.shape}')
                np.savez_compressed(os.path.join(out, f'{source}_seed{seed}_{period}.npz'), **arrays)
    open(os.path.join(out, 'COMPLETE'), 'w').close()


if __name__ == '__main__':
    main()
