#!/usr/bin/env python3
"""
hpo_trianel_cl.py — Hyperparameter-Optimierung für Trianel Centralized TFT

Führt HPO über alle 3 Expanding-Window-Folds durch.
Pro Trial: Training auf jedem Fold, Mittelwert der Val-Metrik → Optuna.

Alle Fold-Daten werden VOR der HPO-Schleife vorgeladen und als Tensoren
vorbereitet, damit kein Trial für Daten-I/O wartet.

Usage:
  python hpo_trianel_cl.py -c configs/config_trianel_fl.yaml --scenario A
  python hpo_trianel_cl.py --scenario B --trials 50 --gpu 1
  python hpo_trianel_cl.py --scenario A --folds 1 2   # nur zwei Folds nutzen
"""
import argparse
import copy
import gc
import json
import logging
import os

import numpy as np
import optuna
import pandas as pd
import torch
import torch.multiprocessing
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from utils import federated, preprocessing
from utils import hpo as hpo_utils
from utils import tools


# ---------------------------------------------------------------------------
# Fold definitions  (identisch zu trianel_fl / trianel_cl)
# ---------------------------------------------------------------------------

FOLD_CONFIGS = [
    {
        'name':        'fold_1',
        'train_start': '2023-07-24',
        'train_end':   '2023-11-30',
        'val_start':   '2023-12-01',
        'val_end':     '2024-01-31',
    },
    {
        'name':        'fold_2',
        'train_start': '2023-07-24',
        'train_end':   '2024-01-31',
        'val_start':   '2024-02-01',
        'val_end':     '2024-03-31',
    },
    {
        'name':        'fold_3',
        'train_start': '2023-07-24',
        'train_end':   '2024-03-31',
        'val_start':   '2024-04-01',
        'val_end':     '2024-05-31',
    },
]

SCENARIOS = {
    'A': {'use_synthetic': False, 'use_static': False},
    'B': {'use_synthetic': True,  'use_static': True},
    'C': {'use_synthetic': True,  'use_static': False},
    'D': {'use_synthetic': False, 'use_static': True},
}


# ---------------------------------------------------------------------------
# Data helpers  (aus trianel_cl.py übernommen)
# ---------------------------------------------------------------------------

def build_period_config(base_config: dict, fold_config: dict) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg['data']['train_start'] = fold_config['train_start']
    cfg['data']['test_start']  = fold_config['val_start']
    cfg['data']['test_end']    = fold_config['val_end']
    return cfg


def get_features_for_scenario(base_config: dict, scenario: dict) -> dict:
    features = preprocessing.get_features(base_config)
    if not scenario['use_static']:
        features['static'] = []
    return features


def load_all_parks_for_fold(park_ids, period_config, fold_config):
    fold_start    = pd.Timestamp(fold_config['train_start'], tz='UTC')
    freq          = period_config['data'].get('freq', '1H')
    full_features = preprocessing.get_features(period_config)
    all_parks     = {}

    for park_id in tqdm(park_ids, desc=f'  Loading {fold_config["name"]}', unit='park'):
        with ThreadPoolExecutor(max_workers=2) as ex:
            r = ex.submit(preprocessing.preprocess_trianel_tft,
                          park_id, period_config, freq, full_features, fold_start, True)
            s = ex.submit(preprocessing.preprocess_trianel_tft,
                          park_id, period_config, freq, full_features, fold_start, False)
            all_parks[park_id] = {'real': r.result(), 'synth': s.result()}

    return all_parks


def build_clients_data(park_ids, cached_park_data, scenario, base_config):
    static_cols = base_config['params'].get('static_features', [])

    def _apply(df):
        if not scenario['use_static'] and static_cols:
            return df.drop(columns=[c for c in static_cols if c in df.columns], errors='ignore')
        return df

    clients = {}
    for pid in park_ids:
        dfs = {pid: _apply(cached_park_data[pid]['real'])}
        if scenario['use_synthetic'] and cached_park_data[pid]['synth'] is not None:
            dfs[f'{pid}_synth'] = _apply(cached_park_data[pid]['synth'])
        clients[pid] = {'dfs': dfs}
    return clients


def fit_global_scaler(clients_data, period_config):
    target_col = period_config['data']['target_col']
    empty      = []

    for cid, ci in clients_data.items():
        sx, n = StandardScaler(), 0
        for df in ci['dfs'].values():
            df_train, _ = preprocessing.split_data(
                data=df,
                train_frac=period_config['data']['train_frac'],
                test_start=pd.Timestamp(period_config['data']['test_start'], tz='UTC'),
                test_end=pd.Timestamp(period_config['data']['test_end'],   tz='UTC'),
                t_0=0,
            )
            dx = df_train.drop(columns=[target_col], errors='ignore')
            if dx.shape[0] == 0:
                continue
            sx.partial_fit(dx.values)
            n += dx.shape[0]

        if n == 0:
            empty.append(cid)
            continue
        ci['scaler_x'] = sx

    for cid in empty:
        del clients_data[cid]

    stats = [{'n': ci['scaler_x'].n_samples_seen_,
               'mu': ci['scaler_x'].mean_,
               'var': ci['scaler_x'].var_}
              for ci in clients_data.values()]
    mu, std = federated.aggregate_scalers(stats)

    gs = StandardScaler()
    gs.mean_           = mu
    gs.scale_          = std
    gs.var_            = std ** 2
    gs.n_samples_seen_ = sum(s['n'] for s in stats)
    gs.n_features_in_  = len(mu)

    for ci in clients_data.values():
        ci['scaler_x'] = gs
    return gs


def _concat_X(X_list):
    if isinstance(X_list[0], dict):
        return {k: np.concatenate([X[k] for X in X_list], axis=0) for k in X_list[0]}
    return np.concatenate(X_list, axis=0)


def prepare_fold_tensors(clients_data, period_config, features):
    """Konvertiert clients_data → (X_train, y_train, X_val, y_val) als Tensoren."""
    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []

    for cid, ci in clients_data.items():
        gen = tools.create_data_generator(
            dfs=ci['dfs'],
            config=period_config,
            features=features,
            scaler_x=ci['scaler_x'],
        )
        X_tr, y_tr, _, _, test_data, _ = tools.combine_datasets_efficiently(gen)
        X_va, y_va, _, _ = test_data[cid]          # nur real-Daten (key == park_id)

        all_X_train.append(X_tr)
        all_y_train.append(y_tr)
        all_X_val.append(X_va)
        all_y_val.append(y_va)

        del gen
        gc.collect()

    return (
        _concat_X(all_X_train),
        np.concatenate(all_y_train, axis=0),
        _concat_X(all_X_val),
        np.concatenate(all_y_val,   axis=0),
    )


# ---------------------------------------------------------------------------
# Pre-loading aller Folds
# ---------------------------------------------------------------------------

def preload_all_folds(base_config, scenario, selected_folds):
    """
    Lädt alle Folds vorab und gibt eine Liste von (X_train, y_train, X_val, y_val)
    zurück. Wird einmal vor der HPO-Schleife aufgerufen.
    """
    fold_tensors = []

    for fold_config in selected_folds:
        logging.info(f'Pre-loading {fold_config["name"]}  '
                     f'({fold_config["train_start"]} – {fold_config["val_end"]})...')

        period_config = build_period_config(base_config, fold_config)
        park_ids      = list(period_config['fl']['clients'].keys())
        features      = get_features_for_scenario(period_config, scenario)

        # 1. Daten laden (DataFrames)
        cached = load_all_parks_for_fold(park_ids, period_config, fold_config)

        # 2. Clients-Dict aufbauen + Scaler
        clients_data = build_clients_data(park_ids, cached, scenario, base_config)
        fit_global_scaler(clients_data, period_config)

        # 3. Zu Tensoren konvertieren
        logging.info(f'  Converting {fold_config["name"]} to tensors...')
        X_tr, y_tr, X_va, y_va = prepare_fold_tensors(clients_data, period_config, features)

        if isinstance(X_tr, dict):
            n_tr = X_tr[next(iter(X_tr))].shape[0]
            n_va = X_va[next(iter(X_va))].shape[0]
        else:
            n_tr, n_va = X_tr.shape[0], X_va.shape[0]

        logging.info(f'  {fold_config["name"]}: {n_tr} train, {n_va} val sequences')

        fold_tensors.append((X_tr, y_tr, X_va, y_va))

        del cached, clients_data
        gc.collect()

    return fold_tensors, period_config   # period_config vom letzten Fold für Model-Setup


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Trianel CL HPO')
    parser.add_argument('-c', '--config', default='configs/config_trianel_fl.yaml')
    parser.add_argument('--scenario', required=True, choices=list(SCENARIOS.keys()),
                        help='Scenario to optimize (A / B / C / D)')
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3],
                        choices=[1, 2, 3],
                        help='Fold numbers to include in HPO (default: all 3)')
    parser.add_argument('--trials', type=int, default=None,
                        help='Number of trials (overrides config hpo.trials)')
    parser.add_argument('-s', '--suffix', type=str, default='',
                        help='Optional suffix for study name / log file')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU index to use (default: auto)')
    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    scenario      = SCENARIOS[args.scenario]
    scenario_name = args.scenario
    suffix        = f'_{args.suffix}' if args.suffix else ''

    os.makedirs('logs',    exist_ok=True)
    os.makedirs('studies', exist_ok=True)

    log_file = f'logs/hpo_trianel_cl_scenario-{scenario_name}{suffix}.log'
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logging.info('=' * 70)
    logging.info(f'Trianel CL HPO  |  Scenario {scenario_name}  |  '
                 f'use_synthetic={scenario["use_synthetic"]}  '
                 f'use_static={scenario["use_static"]}')
    logging.info('=' * 70)

    # --- Config ---
    base_config = tools.load_config(args.config)
    base_config = tools.handle_freq(base_config)
    base_config['model']['verbose'] = 0
    base_config['model']['name']    = 'tft'

    n_trials = args.trials or base_config['hpo']['trials']

    selected_folds = [FOLD_CONFIGS[i - 1] for i in sorted(set(args.folds))]
    logging.info(f'Folds: {[f["name"] for f in selected_folds]}  |  Trials: {n_trials}')

    # --- Device ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    logging.info(f'Device: {device}')

    # --- Pre-load all fold data (once, before HPO loop) ---
    logging.info('\nPre-loading fold data...')
    fold_tensors, period_config = preload_all_folds(base_config, scenario, selected_folds)
    logging.info(f'All {len(fold_tensors)} folds loaded and converted to tensors.\n')

    # --- Optuna study ---
    study_name = f'trianel_cl_scenario-{scenario_name}{suffix}'
    pruning_config = base_config.get('hpo', {}).get('pruning', {})
    study = hpo_utils.create_or_load_study(
        base_config['hpo']['studies_path'],
        study_name,
        pruning_config=pruning_config,
        config=base_config,
    )

    objectives, is_multi_objective = hpo_utils.get_objectives_from_config(base_config)
    obj_strs = [f'{o["metric"]} ({o["direction"]})' for o in objectives]
    logging.info(f'Study: {study_name}')
    logging.info(f'Mode: {"multi-objective" if is_multi_objective else "single-objective"}')
    logging.info(f'Objectives: {obj_strs}')

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned    = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    logging.info(f'Existing trials: {len(study.trials)} total, '
                 f'{completed} complete, {pruned} pruned')
    logging.info(f'Starting {n_trials - completed} new trials.')

    trial_counter = 0

    while completed < n_trials:
        trial        = study.ask()
        trial_number = len(study.trials) - 1

        hyperparameters = hpo_utils.get_hyperparameters(
            config=base_config,
            hpo=True,
            trial=trial,
        )

        # Skip duplicates
        existing_params = [t.params for t in study.trials]
        if sum(1 for p in existing_params if p == trial.params) > 1:
            logging.warning(f'Trial {trial_number}: duplicate parameters — skipping.')
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            trial_counter += 1
            continue

        logging.info(f'\nTrial {completed + 1}/{n_trials}  —  '
                     f'{json.dumps(hyperparameters)}')

        try:
            fold_metrics = {obj['metric']: [] for obj in objectives}

            for fold_idx, (X_tr, y_tr, X_va, y_va) in enumerate(fold_tensors):

                # Feature dims setzen (kann sich zwischen Szenarien unterscheiden)
                period_config['model']['feature_dim'] = tools.get_feature_dim(X=X_tr)
                period_config['model']['name'] = 'tft'
                period_config['model']['verbose'] = 0

                history, model = tools.training_pipeline(
                    train=(X_tr, y_tr),
                    val=(X_va, y_va),
                    hyperparameters=hyperparameters,
                    config=period_config,
                    device=device,
                )

                # Metrik auslesen
                metric_map = {
                    'loss': 'val_loss', 'val_loss': 'val_loss', 'mse': 'val_loss',
                    'rmse': 'val_rmse', 'mae': 'val_mae',
                    'r2': 'val_r2',     'val_r2': 'val_r2',
                }
                for obj in objectives:
                    key = obj['metric']
                    mapped = metric_map.get(key, key)
                    if key in history and history[key]:
                        fold_metrics[key].append(history[key][-1])
                    elif mapped in history and history[mapped]:
                        fold_metrics[key].append(history[mapped][-1])
                    else:
                        available = [k for k, v in history.items() if v]
                        raise ValueError(
                            f"Metric '{key}' not found in history. "
                            f"Available: {available}"
                        )

                fold_val = fold_metrics[objectives[0]['metric']][-1]
                logging.info(f'  {selected_folds[fold_idx]["name"]}: '
                             f'{objectives[0]["metric"]}={fold_val:.4f}')

                if not is_multi_objective:
                    trial.report(fold_val, step=fold_idx)

                del model, history
                gc.collect()

                if not is_multi_objective and trial.should_prune():
                    raise optuna.TrialPruned()

            # Alle Folds fertig — Mittelwert berechnen
            if is_multi_objective:
                avg_values = [float(np.mean(fold_metrics[obj['metric']])) for obj in objectives]
                summary    = ', '.join(f'{o["metric"]}={v:.4f}'
                                       for o, v in zip(objectives, avg_values))
                study.tell(trial, values=avg_values)
            else:
                avg_values = [float(np.mean(fold_metrics[objectives[0]['metric']]))]
                summary    = f'{objectives[0]["metric"]}={avg_values[0]:.4f}'
                study.tell(trial, avg_values[0])

            logging.info(f'Trial {completed + 1} done  →  avg {summary}')
            completed += 1

        except optuna.TrialPruned:
            logging.info(f'Trial {trial_number} pruned.')
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)

        except KeyboardInterrupt:
            logging.warning(f'Trial {trial_number} interrupted — marking as failed.')
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise

        except Exception as exc:
            logging.error(f'Trial {trial_number} failed: {exc}')
            logging.exception('Traceback:')
            study.tell(trial, state=optuna.trial.TrialState.FAIL)

        trial_counter += 1

    # --- Results ---
    logging.info('\n' + '=' * 70)
    logging.info('HPO COMPLETED')
    logging.info('=' * 70)

    if is_multi_objective:
        best = study.best_trials
        logging.info(f'Pareto front: {len(best)} solutions')
        for i, t in enumerate(best[:5]):
            vals = ', '.join(f'{o["metric"]}={t.values[j]:.4f}'
                             for j, o in enumerate(objectives))
            logging.info(f'  [{i+1}] {vals}  params={json.dumps(t.params)}')
    else:
        logging.info(f'Best trial:  #{study.best_trial.number}')
        logging.info(f'Best {objectives[0]["metric"]}: {study.best_value:.6f}')
        logging.info(f'Best params: {json.dumps(study.best_params, indent=2)}')

    logging.info('=' * 70)


if __name__ == '__main__':
    main()
