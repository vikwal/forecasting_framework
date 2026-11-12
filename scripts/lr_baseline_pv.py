#!/usr/bin/env python3
"""Linear-regression baseline for the PV park configs (Viktor 2026-11-08): one ordinary least-squares model per forecast
step on the same windows the TFT sees (preprocessing.pipeline, global scaler as train_cl), trained on data.files, evaluated
on the test window of data.files and data.holdout_files. Inputs of step h: the known features at that step plus three
summaries of the observed power (last value, mean of the last 24 h, mean of the last 48 h).

    frcst/bin/python scripts/lr_baseline_pv.py -c configs/pv_parks/config_pv_v4_central_split1 [--no-observed]
Result: results/<data dir>/lr_<config>_<timestamp>.pkl in the train_cl layout (evaluation, predictions, config) plus
a CSV of the per-park metrics next to it."""
import argparse
import copy
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import preprocessing, tools  # noqa: E402


def design(X: dict, lead: int, with_observed: bool) -> np.ndarray:
    known, obs = X['known'], X['observed']
    hist = obs.shape[1]
    cols = [known[:, hist + lead, :]]
    if with_observed:
        p = obs[:, :, 0]
        cols += [p[:, -1:], p[:, -48:].mean(axis=1, keepdims=True), p.mean(axis=1, keepdims=True)]
    return np.concatenate(cols, axis=1)


def r2(a, b):
    return float(1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--no-observed', action='store_true', help='known features only (no past power)')
    a = ap.parse_args()
    cfg_path = a.config.split('.yaml')[0]
    config = tools.load_config(f'{cfg_path}.yaml')
    config = tools.handle_freq(config=config)
    config['model']['name'] = 'tft'                                   # window layout of the TFT (known over history + horizon)
    config['model']['fl'] = False
    freq = config['data']['freq']
    features = preprocessing.get_features(config=config)
    data_dir = config['data']['path']
    dfs = preprocessing.get_data(data_dir=data_dir, config=config, freq=freq, features=features, files_key='files')
    holdout = preprocessing.get_data(data_dir=data_dir, config=config, freq=freq, features=features, files_key='holdout_files') \
        if config['data'].get('holdout_files') else {}
    target_cols = preprocessing.get_target_cols(config)
    # global x scaler as in train_cl (TFT branch: no lag features)
    sx = StandardScaler()
    for key, df in tqdm(dfs.items(), desc='scaler'):
        df_train, _ = preprocessing.split_data(data=df, train_frac=config['data']['train_frac'],
                                               train_start=pd.Timestamp(config['data'].get('train_start'), tz='UTC'),
                                               train_end=pd.Timestamp(config['data'].get('train_end'), tz='UTC'),
                                               test_start=pd.Timestamp(config['data']['test_start'], tz='UTC'),
                                               test_end=pd.Timestamp(config['data']['test_end'], tz='UTC'), t_0=0)
        if len(df_train):
            sx.partial_fit(df_train.drop(columns=target_cols, errors='ignore').values)
    gen_cfg = copy.deepcopy(config)
    train_X, train_y, test = [], [], {}
    for d, tag in ((dfs, 'train'), (holdout, 'holdout')):
        for item in tools.create_data_generator(d, gen_cfg, features, scaler_x=sx, scaler_y=None):
            if tag == 'train' and item['X_train']['known'].shape[0]:
                train_X.append(item['X_train']); train_y.append(item['y_train'])
            if item['X_test']['known'].shape[0]:
                test[item['key']] = (tag, item['X_test'], item['y_test'], item['index_test'], item['nwp_raw_test'])
    horizon = train_y[0].shape[1]
    Xtr = {k: np.concatenate([x[k] for x in train_X]) for k in ('known', 'observed')}
    ytr = np.concatenate(train_y)
    print(f'{len(train_X)} training parks, {len(ytr)} windows, {len(test)} evaluated parks, horizon {horizon}', flush=True)
    models = []
    for h in tqdm(range(horizon), desc='fit'):
        Z = design(Xtr, h, not a.no_observed); y = ytr[:, h]
        ok = np.isfinite(Z).all(axis=1) & np.isfinite(y)
        models.append(LinearRegression().fit(Z[ok], y[ok]))
    rows, preds = [], {}
    for key, (tag, X, y, idx, nwp) in test.items():
        P = np.stack([models[h].predict(np.nan_to_num(design(X, h, not a.no_observed))) for h in range(horizon)], axis=1)
        P = np.clip(P, 0.0, None)
        cols = [f't+{i + 1}' for i in range(horizon)]
        preds[(key, target_cols[0])] = {'pred': pd.DataFrame(P, index=idx, columns=cols), 'true': pd.DataFrame(y, index=idx, columns=cols),
                                        'baseline::NWP (gti_nwp)': pd.DataFrame(nwp, index=idx, columns=cols) if nwp is not None else None}
        p, t = P.ravel(), y.ravel(); ok = np.isfinite(p) & np.isfinite(t)
        day = (nwp.ravel() > 10.0) & ok if nwp is not None else ok
        rows.append(dict(key=key, kind=tag, R2=r2(p[ok], t[ok]), RMSE=float(np.sqrt(((p[ok] - t[ok]) ** 2).mean())),
                         MAE=float(np.abs(p[ok] - t[ok]).mean()), r2_day=r2(p[day], t[day]), nrmse_day=float(np.sqrt(((p[day] - t[day]) ** 2).mean())),
                         n_runs=int(len(y))))
    ev = pd.DataFrame(rows)
    print(ev.groupby('kind')[['R2', 'r2_day', 'nrmse_day']].agg(['mean', 'median']).round(4).to_string(), flush=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S'); name = os.path.basename(cfg_path).replace('config_', '')
    out_dir = os.path.join('results', os.path.basename(data_dir)); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"lr{'_noobs' if a.no_observed else ''}_{name}_{stamp}.pkl")
    pickle.dump({'model': 'lr_per_step', 'with_observed': not a.no_observed, 'config': config, 'evaluation': ev, 'predictions': preds},
                open(out, 'wb'))
    ev.to_csv(out.replace('.pkl', '.csv'), index=False)
    print('saved', out)


if __name__ == '__main__':
    main()
