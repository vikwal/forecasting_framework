"""Final models on the HPO date folds (hpo.fold_boundaries) for the central, local and federated
setting, with the best trial of the respective Optuna study, saved weights and time-stamped
predictions per station.

One code path for the three settings: a *unit* is what gets one study and one set of weights —
the 80-park pool (central), one FL client (local) or the federation (FL, global weights). Each
unit has clients {client_id: [stations]}; the central and local units have exactly one client.

Data path mirrors hpo_fl.py (and data_cache.create_or_load_preprocessed_data for the central
case): per client get_data -> StandardScaler fitted on the rows before hpo.scaler_fit_end
(= first fold boundary) -> optional pooling of the client scalers (fl.global_scaler) ->
preprocessing.pipeline per station -> hpo.kfolds_by_dates. The per-station prepared data is
kept with its key so fold predictions carry issue times (the HPO fold cache has neither).
"""
from __future__ import annotations

import copy
import gc
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from utils import federated, hpo, local_training, models, preprocessing, tools

SETTINGS = ('central', 'local', 'fl')


# ----------------------------------------------------------------------------- config / studies
def final_config(config: dict, model_name: str, setting: str, forecast_hours=None) -> dict:
    """Config as the HPO scripts see it, plus the ICON runs of the final training."""
    cfg = copy.deepcopy(config)
    if forecast_hours:
        cfg['data']['forecast_hours'] = [str(h).zfill(2) for h in forecast_hours]
        cfg['data'].pop('train_forecast_hours', None)
    if hpo.fold_boundaries(cfg) and cfg['data'].get('val_files'):
        cfg['data'].pop('val_files')          # as hpo_fl.py: validation = own stations in the fold windows
    cfg = tools.handle_freq(config=cfg)
    cfg['model']['name'] = model_name
    cfg['model']['fl'] = setting == 'fl'
    cfg['model']['verbose'] = 0
    if 'fl' in cfg:
        cfg['fl']['verbose'] = False
    return cfg


def config_stem(cfg_path: str) -> str:
    return os.path.basename(cfg_path).replace('.yaml', '').replace('config_', '', 1)


def study_names(cfg_path: str, config: dict, model_name: str, setting: str, clients=None) -> Dict[str, str]:
    """{unit: study name} as hpo_cl.py / hpo_local.py / hpo_fl.py name their studies."""
    stem = config_stem(cfg_path)
    out_dim, freq = config['model']['output_dim'], config['data']['freq']
    base = f'm-{model_name}_out-{out_dim}_freq-{freq}'
    if setting == 'central':
        return {'central': f'cl_{base}_{stem}'}
    if setting == 'fl':
        return {'fl': f"fl_a-{config['fl']['strategy']}_{base}_{stem}"}
    lname = local_training.local_name(cfg_path, config['fl']['strategy']).replace('config_', '', 1)
    return {cid: f'cl_{base}_{lname}_{cid}' for cid in (clients or config['fl']['clients'])}


def unit_clients(config: dict, setting: str, unit: str) -> Dict[str, List[str]]:
    if setting == 'central':
        return {'central': list(config['data']['files'])}
    if setting == 'local':
        return {unit: list(config['fl']['clients'][unit])}
    return {cid: list(st) for cid, st in config['fl']['clients'].items()}


def best_hyperparameters(config: dict, study_name: str) -> Tuple[dict, Any]:
    study = hpo.load_study(config['hpo']['studies_path'], study_name, required=True)
    hp = hpo.get_hyperparameters(config=config, study=study)
    if config['model'].get('fl'):
        hp['n_rounds'] = config['fl'].get('n_rounds', 10)
        hp['personalization'] = config['fl'].get('personalize', False)
        hp['global_scaler'] = config['fl'].get('global_scaler', False)
        if 'n_local_epochs' in study.best_trial.params:      # read by the client actors from the config
            config['fl']['n_local_epochs'] = int(study.best_trial.params['n_local_epochs'])
    return hp, study


# ----------------------------------------------------------------------------- data
def _fit_client_scalers(dfs: dict, config: dict, features: dict) -> Tuple[StandardScaler, StandardScaler | None]:
    """StandardScaler X (and y for non-power targets) on the rows before hpo.scaler_fit_end (hpo_fl.py)."""
    target_col = config['data']['target_col']
    fit_y = target_col != 'power'
    sx, sy = StandardScaler(), StandardScaler()
    t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
    for key, df in dfs.items():
        df_train, _ = preprocessing.split_data(
            data=df, train_frac=config['data']['train_frac'], test_start=hpo.scaler_fit_end(config),
            test_end=pd.Timestamp(config['data'].get('test_end', '2099-12-31'), tz='UTC'), t_0=t_0)
        if fit_y and target_col in df_train.columns:
            sy.partial_fit(df_train[[target_col]].values)
        sx.partial_fit(df_train.drop(columns=[target_col], errors='ignore').values)
    return sx, (sy if fit_y else None)


def _pool_scalers(scalers: List[StandardScaler]) -> StandardScaler:
    stats = [{'n': s.n_samples_seen_, 'mu': s.mean_, 'var': s.var_} for s in scalers]
    mu, std = federated.aggregate_scalers(stats)
    g = StandardScaler()
    g.mean_, g.scale_, g.var_ = mu, std, std ** 2
    g.n_samples_seen_ = sum(s['n'] for s in stats)
    g.n_features_in_ = len(mu)
    return g


def prepare_clients(config: dict, features: dict, clients: Dict[str, List[str]], pool_scalers: bool,
                    scalers: Dict[str, dict] | None = None) -> dict:
    """{client_id: {'stations': {key: prepared}, 'scaler_x', 'scaler_y'}} with pipeline() output per station.
    `scalers` ({client_id: {'scaler_x', 'scaler_y'}}) applies given scalers instead of fitting them, e.g. a
    trained model's scaler for stations it has never seen."""
    data_dir, freq = config['data']['path'], config['data']['freq']
    out = {}
    for cid, stations in clients.items():
        ccfg = federated.client_data_config(config, stations)
        dfs = preprocessing.get_data(data_dir=data_dir, config=ccfg, freq=freq, features=features,
                                     files_key='client_files')
        if scalers is not None:
            sx, sy = scalers[cid]['scaler_x'], scalers[cid].get('scaler_y')
            logging.info(f'[{cid}] {len(dfs)} stations loaded, given scaler applied')
        else:
            sx, sy = _fit_client_scalers(dfs, config, features)
            logging.info(f'[{cid}] {len(dfs)} stations loaded, scaler fitted before {hpo.scaler_fit_end(config).date()}')
        out[cid] = {'dfs': dfs, 'scaler_x': sx, 'scaler_y': sy}
    if pool_scalers and len(out) > 1 and scalers is None:
        gx = _pool_scalers([c['scaler_x'] for c in out.values()])
        gy = _pool_scalers([c['scaler_y'] for c in out.values()]) if out[next(iter(out))]['scaler_y'] else None
        for c in out.values():
            c['scaler_x'], c['scaler_y'] = gx, gy
        logging.info('client scalers pooled into one global scaler (fl.global_scaler)')
    for cid, c in out.items():
        config['scaler_x'] = c['scaler_x']
        if c['scaler_y'] is not None:
            config['scaler_y'] = c['scaler_y']
        else:
            config.pop('scaler_y', None)
        c['stations'] = {}
        for key, df in c['dfs'].items():
            prepared, _ = preprocessing.pipeline(data=df, config=config, known_cols=features['known'],
                                                 observed_cols=features['observed'], static_cols=features['static'],
                                                 target_col=config['data']['target_col'])
            if prepared is not None and len(prepared.get('y_train', [])) > 0:
                c['stations'][key] = prepared
            else:
                logging.warning(f'[{cid}] station {key}: no training samples after preprocessing, skipped')
        del c['dfs']
        gc.collect()
    config.pop('scaler_x', None)
    config.pop('scaler_y', None)
    return out


def fold_sets(stations: Dict[str, dict], boundaries, horizon_hours: int, fold: int):
    """((X_tr, y_tr), (X_va, y_va)) of date fold `fold` over the given stations (hpo.kfolds_by_dates)."""
    b = [str(x.date()) if hasattr(x, 'date') else str(x) for x in boundaries]
    return hpo.kfolds_by_dates(list(stations.values()), b[fold:fold + 2], horizon_hours)[0]


def window_mask(prepared: dict, boundaries, fold: int) -> np.ndarray:
    t = hpo._sample_times(prepared['index_train'])
    return np.asarray((t >= boundaries[fold]) & (t < boundaries[fold + 1]))


# ----------------------------------------------------------------------------- prediction
def predict(model, X, y, device, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Point predictions and targets of a TFT model for one (X, y) set, as in the training loop."""
    loader = tools.create_pytorch_dataloader(X, y, batch_size=batch_size, shuffle=False, device=device,
                                             num_workers=0, drop_last=False)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                obs, known, static, t = [b.to(device) for b in batch]
            elif len(batch) == 3:
                obs, known, t = [b.to(device) for b in batch]
                static = None
            else:
                raise ValueError(f'unexpected TFT batch of length {len(batch)}')
            preds.append(model(obs, known, static).detach().cpu())
            targets.append(t.detach().cpu())
    p = torch.cat(preds).numpy()
    t = torch.cat(targets).numpy()
    if p.ndim == 3 and t.ndim == 2:          # single quantile
        p = p[..., 0]
    if p.shape != t.shape:
        raise ValueError(f'prediction shape {p.shape} != target shape {t.shape}')
    return p, t


def _frames(p: np.ndarray, t: np.ndarray, times: pd.DatetimeIndex, scaler_y=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if scaler_y is not None:
        p = scaler_y.inverse_transform(p.reshape(-1, 1)).reshape(p.shape)
        t = scaler_y.inverse_transform(t.reshape(-1, 1)).reshape(t.shape)
    cols = [f't+{i + 1}' for i in range(p.shape[1])]
    return (pd.DataFrame(p, index=times, columns=cols), pd.DataFrame(t, index=times, columns=cols))


def metrics(pred: pd.DataFrame, true: pd.DataFrame) -> dict:
    p, t = pred.values, true.values
    err = p - t
    ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum((t - t.mean()) ** 2))
    return {'n': int(t.shape[0]), 'mse': float(np.mean(err ** 2)), 'rmse': float(np.sqrt(np.mean(err ** 2))),
            'mae': float(np.mean(np.abs(err))), 'r2': 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')}


def predict_window(model, clients_prepared: dict, boundaries, fold: int, target: str, device, batch_size: int,
                   mask_fn=window_mask) -> Tuple[dict, pd.DataFrame]:
    """Predictions of every station in the fold window -> ({(station, target): {'pred', 'true'}}, evaluation)."""
    preds, rows = {}, []
    for cid, c in clients_prepared.items():
        for key, prepared in c['stations'].items():
            m = mask_fn(prepared, boundaries, fold)
            if not m.any():
                continue
            times = hpo._sample_times(prepared['index_train'])[m]
            p, t = predict(model, hpo._take(prepared['X_train'], m), prepared['y_train'][m], device, batch_size)
            pred, true = _frames(p, t, times, c['scaler_y'])
            preds[(key, target)] = {'pred': pred, 'true': true}
            rows.append({'key': key, 'client_id': cid, **metrics(pred, true)})
    return preds, pd.DataFrame(rows)


def pooled(evaluation: pd.DataFrame) -> dict:
    n = evaluation['n'].sum()
    mse = float((evaluation['n'] * evaluation['mse']).sum() / n)
    mae = float((evaluation['n'] * evaluation['mae']).sum() / n)
    return {'n': int(n), 'mse': mse, 'rmse': float(np.sqrt(mse)), 'mae': mae}


# ----------------------------------------------------------------------------- training
def train_fold(setting: str, config: dict, hp: dict, clients_prepared: dict, boundaries, fold: int, seed: int,
               device: str, gpu: int | None = None):
    """Train the unit's model on date fold `fold` -> (model on `device`, history, info)."""
    horizon = config['model']['horizon']
    tools.set_seed(int(seed))
    config['params']['random_seed'] = int(seed)
    t0 = time.time()
    if setting in ('central', 'local'):
        stations = {k: v for c in clients_prepared.values() for k, v in c['stations'].items()}
        train, val = fold_sets(stations, boundaries, horizon, fold)
        config['model_name'] = f'final_fold{fold}_s{seed}'
        history, model = tools.training_pipeline(train=train, val=val, hyperparameters=hp, config=config,
                                                 device=device)
        n_train = len(train[1])
        del train, val
        info = {'epochs': len(history['val_rmse']), 'best': int(np.argmin(history['val_rmse'])) + 1,
                'train_samples': n_train, 'history': {k: list(map(float, v)) for k, v in history.items()}}
        state = (model._orig_mod if hasattr(model, '_orig_mod') else model).state_dict()
        state = {k: v.detach().cpu() for k, v in state.items()}
    else:
        partitions = {}
        for cid, c in clients_prepared.items():
            (X_tr, y_tr), (X_va, y_va) = fold_sets(c['stations'], boundaries, horizon, fold)
            partitions[cid] = (X_tr, y_tr, X_va, y_va)
        n_train = sum(len(p[1]) for p in partitions.values())
        config['model_name'] = f'final_fold{fold}_s{seed}'
        if gpu is not None:
            os.environ['FL_GPU_ORDER'] = str(gpu)
        history, clients_weights = federated.run_simulation(partitions=partitions, hyperparameters=hp,
                                                            config=config, global_val_data=None)
        del partitions
        state = {k: v.detach().cpu() for k, v in next(iter(clients_weights.values())).items()}
        ma = history['metrics_aggregated']
        info = {'epochs': int(history.get('last_round', len(ma))), 'best': int(history.get('best_round', 0)),
                'train_samples': n_train, 'history': ma.to_dict(orient='list') if hasattr(ma, 'to_dict') else None,
                'comm_stats': history.get('comm_stats')}
        config['model']['feature_dim'] = tools.get_feature_dim(X=next(iter(clients_prepared.values()))['stations']
                                                               [next(iter(next(iter(clients_prepared.values()))['stations']))]['X_train'])
        model = models.get_model(config=config, hyperparameters=hp)
        model.load_state_dict(state)
        model = model.to(device)
    info['train_min'] = round((time.time() - t0) / 60, 2)
    gc.collect()
    return model, state, info


def save_fold(out_dir: str, model_dir: str, study: str, fold: int, seed: int, state: dict, payload: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    stem = f'{study}_fold{fold}_seed{seed}'
    torch.save(state, os.path.join(model_dir, f'{stem}.pt'))
    path = os.path.join(out_dir, f'{stem}.pkl')
    pd.to_pickle(payload, path)
    return path
