#!/usr/bin/env python3
"""
trianel_cl.py — Trianel Centralized & Local TFT Experiment

2×2 Factorial Design (Synthetic Augmentation × Static Features):
  A: real data only, no static features
  B: real + synthetic data, with static features
  C: real + synthetic data, no static features
  D: real data only, with static features

Modes:
  centralized  All parks concatenated → one shared model (default)
  local        One independent model per park, trained in parallel via Ray

3-fold Expanding Window CV (Jul 2023 – May 2024).
Validation is ALWAYS on real SCADA measurements only.

Metrics are written to:
  results/trianel/centralized/metrics_<scenario>.csv
  results/trianel/local/metrics_<scenario>.csv

CSV format: Fold;Scenario;Park;RMSE;MAE;R2

Usage:
  python trianel_cl.py -c configs/config_trianel_fl.yaml
  python trianel_cl.py --mode local --scenarios A D --folds 1 2
  python trianel_cl.py --mode centralized --scenarios B --folds 1 --save_model
"""
import argparse
import copy
import gc
import logging
import math
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import ray
import torch
import torch.multiprocessing
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import federated, preprocessing
from utils import eval as eval_utils
from utils import hpo as hpo_utils
from utils import models as models_utils
from utils import tools


# ---------------------------------------------------------------------------
# Experiment constants  (identical to trianel_fl.py)
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
    #'B': {'use_synthetic': True,  'use_static': True},
    #'C': {'use_synthetic': True,  'use_static': False},
    'D': {'use_synthetic': False, 'use_static': True},
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Trianel CL/Local TFT Experiment')
    parser.add_argument('-c', '--config', default='configs/config_trianel_fl.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--mode', default='centralized',
                        choices=['centralized', 'local'],
                        help='Training mode: centralized (one model for all parks) '
                             'or local (one model per park, no aggregation)')
    parser.add_argument('--scenarios', nargs='+', default=list(SCENARIOS.keys()),
                        choices=list(SCENARIOS.keys()),
                        help='Scenarios to run (default: all)')
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3],
                        choices=[1, 2, 3],
                        help='Fold numbers to run (default: all)')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model weights to disk')
    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity (default: INFO)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
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


def load_all_parks_for_fold(park_ids: list,
                            period_config: dict,
                            fold_config: dict) -> dict:
    """Load all parks once (real + synthetic). Reused across all scenarios."""
    fold_start   = pd.Timestamp(fold_config['train_start'], tz='UTC')
    freq         = period_config['data'].get('freq', '1H')
    full_features = preprocessing.get_features(period_config)
    all_parks_data = {}

    load_real_synth = period_config['params'].get('get_synthetic_real_parks', False)

    for park_id in tqdm(park_ids, desc='Loading parks', unit='park'):
        if load_real_synth:
            with ThreadPoolExecutor(max_workers=2) as executor:
                real_future  = executor.submit(
                    preprocessing.preprocess_trianel_tft,
                    park_id, period_config, freq, full_features, fold_start, True,
                )
                synth_future = executor.submit(
                    preprocessing.preprocess_trianel_tft,
                    park_id, period_config, freq, full_features, fold_start, False,
                )
                real_df  = real_future.result()
                synth_df = synth_future.result()
        else:
            real_df  = preprocessing.preprocess_trianel_tft(
                park_id, period_config, freq, full_features, fold_start, True,
            )
            synth_df = None
        all_parks_data[park_id] = {'real': real_df, 'synth': synth_df}

    return all_parks_data


def load_client_dfs_from_cache(park_id: str,
                               cached_data: dict,
                               scenario: dict,
                               base_config: dict) -> dict:
    static_cols = base_config['params'].get('static_features', [])

    def _apply_scenario(df: pd.DataFrame) -> pd.DataFrame:
        if not scenario['use_static'] and static_cols:
            return df.drop(columns=[c for c in static_cols if c in df.columns], errors='ignore')
        return df

    dfs = {park_id: _apply_scenario(cached_data[park_id]['real'])}
    load_real_synth = base_config['params'].get('get_synthetic_real_parks', False)
    if scenario['use_synthetic'] and load_real_synth and cached_data[park_id].get('synth') is not None:
        dfs[f'{park_id}_synth'] = _apply_scenario(cached_data[park_id]['synth'])
    return {'dfs': dfs}


def fit_and_aggregate_scalers(clients_data: dict, period_config: dict) -> StandardScaler:
    """FedPS: fit local scalers, then aggregate via Verschiebungssatz der Varianz."""
    target_col    = period_config['data']['target_col']
    empty_clients = []

    for client_id, client_info in tqdm(clients_data.items(), desc='Fitting scalers', unit='client'):
        scaler_x = StandardScaler()
        n_train  = 0
        for df in client_info['dfs'].values():
            df_train, _ = preprocessing.split_data(
                data=df,
                train_frac=period_config['data']['train_frac'],
                test_start=pd.Timestamp(period_config['data']['test_start'], tz='UTC'),
                test_end=pd.Timestamp(period_config['data']['test_end'],   tz='UTC'),
                t_0=0,
            )
            df_x = df_train.drop(columns=[target_col], errors='ignore')
            if df_x.shape[0] == 0:
                continue
            scaler_x.partial_fit(df_x.values)
            n_train += df_x.shape[0]

        if n_train == 0:
            logging.warning(f'Client {client_id}: no training data — excluded.')
            empty_clients.append(client_id)
            continue
        client_info['scaler_x'] = scaler_x

    for cid in empty_clients:
        del clients_data[cid]

    if not clients_data:
        raise RuntimeError('No clients with training data.')

    client_stats = [
        {'n': ci['scaler_x'].n_samples_seen_, 'mu': ci['scaler_x'].mean_, 'var': ci['scaler_x'].var_}
        for ci in clients_data.values()
    ]
    mu_global, std_global = federated.aggregate_scalers(client_stats)

    global_scaler                  = StandardScaler()
    global_scaler.mean_            = mu_global
    global_scaler.scale_           = std_global
    global_scaler.var_             = std_global ** 2
    global_scaler.n_samples_seen_  = sum(s['n'] for s in client_stats)
    global_scaler.n_features_in_   = len(mu_global)

    for ci in clients_data.values():
        ci['scaler_x'] = global_scaler

    logging.info(f'FedPS scaler: {global_scaler.n_samples_seen_} train samples, '
                 f'{global_scaler.n_features_in_} features')
    return global_scaler


def _append_metrics_csv(csv_path: str,
                        fold_name: str,
                        scenario_name: str,
                        metrics_per_park: dict) -> None:
    """Append per-park metrics to a semicolon-separated CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as fh:
        if write_header:
            fh.write('Fold;Scenario;Park;RMSE;MAE;R2\n')
        for park_id, m in sorted(metrics_per_park.items()):
            fh.write(
                f'{fold_name};{scenario_name};{park_id};'
                f'{m["RMSE"]:.6f};{m["MAE"]:.6f};{m["R^2"]:.6f}\n'
            )


# ---------------------------------------------------------------------------
# Centralized mode helpers
# ---------------------------------------------------------------------------

def _concat_X(X_list: list):
    """Concatenate a list of X tensors; handles TFT dict format."""
    if isinstance(X_list[0], dict):
        return {k: np.concatenate([X[k] for X in X_list], axis=0) for k in X_list[0]}
    return np.concatenate(X_list, axis=0)


def prepare_cl_data(clients_data: dict, period_config: dict, features: dict) -> tuple:
    """
    Build one combined training tensor from all parks (real + synthetic).
    Val tensors stay per-park (real data only) for fair evaluation.

    Returns:
        X_train_all, y_train_all  — concatenated across all parks/sources
        X_val_all,   y_val_all    — concatenated real val (for early stopping)
        eval_test_data            — {park_id: (X_val, y_val, index_val, scaler_y)}
    """
    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []
    eval_test_data            = {}

    for client_id, client_info in tqdm(clients_data.items(),
                                       desc='Preparing CL data', unit='client'):
        gen = tools.create_data_generator(
            dfs=client_info['dfs'],
            config=period_config,
            features=features,
            scaler_x=client_info['scaler_x'],
        )
        X_train, y_train, _, _, client_test_data, _ = tools.combine_datasets_efficiently(gen)

        all_X_train.append(X_train)
        all_y_train.append(y_train)

        X_val, y_val, index_val, scaler_y_val = client_test_data[client_id]
        all_X_val.append(X_val)
        all_y_val.append(y_val)
        eval_test_data[client_id] = (X_val, y_val, index_val, scaler_y_val)

        del gen
        gc.collect()

    X_train_all = _concat_X(all_X_train)
    y_train_all = np.concatenate(all_y_train, axis=0)
    X_val_all   = _concat_X(all_X_val)
    y_val_all   = np.concatenate(all_y_val,   axis=0)

    return X_train_all, y_train_all, X_val_all, y_val_all, eval_test_data


# ---------------------------------------------------------------------------
# Local mode helpers
# ---------------------------------------------------------------------------

def prepare_local_data(clients_data: dict, period_config: dict, features: dict) -> dict:
    """
    Build per-park train/val tensors for independent local training.

    Returns:
        {park_id: (X_train, y_train, X_val, y_val, scaler_y)}
    """
    per_park = {}
    for client_id, client_info in tqdm(clients_data.items(),
                                       desc='Preparing local data', unit='client'):
        gen = tools.create_data_generator(
            dfs=client_info['dfs'],
            config=period_config,
            features=features,
            scaler_x=client_info['scaler_x'],
        )
        X_train, y_train, _, _, client_test_data, _ = tools.combine_datasets_efficiently(gen)
        X_val, y_val, _idx, scaler_y = client_test_data[client_id]
        per_park[client_id] = (X_train, y_train, X_val, y_val, scaler_y)
        del gen
        gc.collect()
    return per_park


@ray.remote
def _train_local_park(park_id: str,
                       X_train, y_train,
                       X_val, y_val,
                       scaler_y,
                       config: dict,
                       hyperparameters: dict):
    """Train one local model for a single park. Returns (park_id, metrics, state_dict)."""
    import torch, numpy as np
    from utils import tools, eval as eval_utils

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, model = tools.training_pipeline(
        train=(X_train, y_train),
        val=(X_val, y_val),
        hyperparameters=hyperparameters,
        config=config,
        device=device,
    )
    model = model.to(device)
    y_true, y_pred = tools.get_y(X_val, y_val, model, scaler_y=scaler_y, device=device)
    raw = eval_utils.get_metrics(y_pred, y_true)
    metrics = {k: float(v[0]) if isinstance(v, (list, np.ndarray)) else float(v)
               for k, v in raw.items()}
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    return park_id, metrics, state_dict


# ---------------------------------------------------------------------------
# Centralized experiment loop
# ---------------------------------------------------------------------------

def run_centralized_fold_scenario(base_config: dict,
                                  fold_config: dict,
                                  scenario_name: str,
                                  scenario: dict,
                                  args: argparse.Namespace,
                                  cached_park_data: dict) -> dict:
    logging.info('=' * 60)
    logging.info(f'[centralized]  Fold: {fold_config["name"]}  |  Scenario: {scenario_name}')
    logging.info(f'  train: {fold_config["train_start"]} – {fold_config["train_end"]}')
    logging.info(f'  val:   {fold_config["val_start"]}  – {fold_config["val_end"]}')

    period_config = build_period_config(base_config, fold_config)
    features      = get_features_for_scenario(period_config, scenario)
    park_ids      = list(period_config['fl']['clients'].keys())

    clients_data = {}
    for park_id in park_ids:
        clients_data[park_id] = load_client_dfs_from_cache(
            park_id, cached_park_data, scenario, base_config
        )

    logging.info('Computing FedPS global scaler...')
    fit_and_aggregate_scalers(clients_data, period_config)

    logging.info('Preparing combined dataset...')
    X_train, y_train, X_val, y_val, eval_test_data = prepare_cl_data(
        clients_data, period_config, features
    )

    if isinstance(X_train, dict):
        n_train = X_train[next(iter(X_train))].shape[0]
        n_val   = X_val  [next(iter(X_val))  ].shape[0]
    else:
        n_train, n_val = X_train.shape[0], X_val.shape[0]
    logging.info(f'Combined dataset: {n_train} train / {n_val} val sequences from {len(park_ids)} parks')

    period_config['model']['feature_dim'] = tools.get_feature_dim(X=X_train)
    hyperparameters = hpo_utils.get_hyperparameters(period_config, hpo=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Training on {device}...')
    history, model = tools.training_pipeline(
        train=(X_train, y_train),
        val=(X_val, y_val),
        hyperparameters=hyperparameters,
        config=period_config,
        device=device,
    )
    model = model.to(device)

    logging.info('Evaluating per park on real validation data...')
    metrics_per_park = {}
    for park_id in park_ids:
        if park_id not in eval_test_data:
            logging.warning(f'Skipping {park_id}: no validation data')
            continue
        X_val_park, y_val_park, _idx, scaler_y_park = eval_test_data[park_id]
        y_true, y_pred = tools.get_y(X_val_park, y_val_park, model,
                                     scaler_y=scaler_y_park, device=device)
        raw = eval_utils.get_metrics(y_pred, y_true)
        metrics = {k: float(v[0]) if isinstance(v, (list, np.ndarray)) else float(v)
                   for k, v in raw.items()}
        metrics_per_park[park_id] = metrics
        logging.info(f'  {park_id:20s}  RMSE={metrics["RMSE"]:.4f}  '
                     f'MAE={metrics["MAE"]:.4f}  R²={metrics["R^2"]:.4f}')

    if args.save_model:
        model_dir = 'models/trianel'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'cl_tft_{fold_config["name"]}_{scenario_name}.pt')
        torch.save(model.state_dict(), model_path)
        logging.info(f'Model saved    → {model_path}')

    del model
    gc.collect()

    all_rmse = [m['RMSE'] for m in metrics_per_park.values()]
    all_mae  = [m['MAE']  for m in metrics_per_park.values()]
    all_r2   = [m['R^2']  for m in metrics_per_park.values()]
    logging.info(
        f'[{fold_config["name"]} / {scenario_name}]  '
        f'mean RMSE={np.mean(all_rmse):.4f}  '
        f'mean MAE={np.mean(all_mae):.4f}  '
        f'mean R²={np.mean(all_r2):.4f}'
    )

    result = {
        'fold':     fold_config['name'],
        'scenario': scenario_name,
        'metrics':  metrics_per_park,
        'history':  history,
    }

    out_dir = 'results/trianel/centralized'
    os.makedirs(out_dir, exist_ok=True)

    pkl_path = os.path.join(out_dir, f'cl_{fold_config["name"]}_{scenario_name}_results.pkl')
    with open(pkl_path, 'wb') as fh:
        pickle.dump(result, fh)
    logging.info(f'Results saved → {pkl_path}')

    csv_path = os.path.join(out_dir, f'metrics_{scenario_name}.csv')
    _append_metrics_csv(csv_path, fold_config['name'], scenario_name, metrics_per_park)
    logging.info(f'Metrics appended → {csv_path}')

    return result


# ---------------------------------------------------------------------------
# Local experiment loop
# ---------------------------------------------------------------------------

def run_local_fold_scenario(base_config: dict,
                            fold_config: dict,
                            scenario_name: str,
                            scenario: dict,
                            args: argparse.Namespace,
                            cached_park_data: dict) -> dict:
    logging.info('=' * 60)
    logging.info(f'[local]  Fold: {fold_config["name"]}  |  Scenario: {scenario_name}')
    logging.info(f'  train: {fold_config["train_start"]} – {fold_config["train_end"]}')
    logging.info(f'  val:   {fold_config["val_start"]}  – {fold_config["val_end"]}')

    period_config = build_period_config(base_config, fold_config)
    features      = get_features_for_scenario(period_config, scenario)
    park_ids      = list(period_config['fl']['clients'].keys())

    clients_data = {}
    for park_id in park_ids:
        clients_data[park_id] = load_client_dfs_from_cache(
            park_id, cached_park_data, scenario, base_config
        )

    logging.info('Computing FedPS global scaler...')
    fit_and_aggregate_scalers(clients_data, period_config)

    logging.info('Preparing per-park datasets...')
    per_park = prepare_local_data(clients_data, period_config, features)

    first_X_train = next(iter(per_park.values()))[0]
    period_config['model']['feature_dim'] = tools.get_feature_dim(X=first_X_train)
    hyperparameters = hpo_utils.get_hyperparameters(period_config, hpo=False)

    # GPU allocation: same per-GPU packing formula as federated.py
    n_parks   = len(per_park)
    n_gpus    = torch.cuda.device_count()
    if n_gpus > 0:
        actors_per_gpu = math.ceil(n_parks / n_gpus)
        gpu_per_task   = max(0.1, min(1.0, math.floor(1.0 / actors_per_gpu * 10) / 10))
    else:
        gpu_per_task = 0

    ray.init(
        num_cpus=n_parks,
        num_gpus=n_gpus,
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level=logging.WARNING,
    )
    logging.info(
        f'Launching {n_parks} local training tasks '
        f'({n_gpus} GPUs, {gpu_per_task:.1f} GPU/task)...'
    )

    futures = {
        park_id: _train_local_park.options(num_gpus=gpu_per_task).remote(
            park_id, X_train, y_train, X_val, y_val, scaler_y,
            period_config, hyperparameters,
        )
        for park_id, (X_train, y_train, X_val, y_val, scaler_y) in per_park.items()
    }

    model_dir = 'models/trianel'
    if args.save_model:
        os.makedirs(model_dir, exist_ok=True)

    metrics_per_park = {}
    for park_id, future in futures.items():
        returned_id, metrics, state_dict = ray.get(future)
        metrics_per_park[returned_id] = metrics
        logging.info(f'  {returned_id:20s}  RMSE={metrics["RMSE"]:.4f}  '
                     f'MAE={metrics["MAE"]:.4f}  R²={metrics["R^2"]:.4f}')
        if args.save_model:
            model_path = os.path.join(
                model_dir, f'local_tft_{fold_config["name"]}_{scenario_name}_{returned_id}.pt'
            )
            torch.save(state_dict, model_path)
            logging.info(f'  Model saved  → {model_path}')

    all_rmse = [m['RMSE'] for m in metrics_per_park.values()]
    all_mae  = [m['MAE']  for m in metrics_per_park.values()]
    all_r2   = [m['R^2']  for m in metrics_per_park.values()]
    logging.info(
        f'[{fold_config["name"]} / {scenario_name}]  '
        f'mean RMSE={np.mean(all_rmse):.4f}  '
        f'mean MAE={np.mean(all_mae):.4f}  '
        f'mean R²={np.mean(all_r2):.4f}'
    )

    result = {
        'fold':     fold_config['name'],
        'scenario': scenario_name,
        'metrics':  metrics_per_park,
    }

    out_dir = 'results/trianel/local'
    os.makedirs(out_dir, exist_ok=True)

    pkl_path = os.path.join(out_dir, f'local_{fold_config["name"]}_{scenario_name}_results.pkl')
    with open(pkl_path, 'wb') as fh:
        pickle.dump(result, fh)
    logging.info(f'Results saved → {pkl_path}')

    csv_path = os.path.join(out_dir, f'metrics_{scenario_name}.csv')
    _append_metrics_csv(csv_path, fold_config['name'], scenario_name, metrics_per_park)
    logging.info(f'Metrics appended → {csv_path}')

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    os.makedirs('logs', exist_ok=True)
    log_file = f'logs/trianel_{args.mode}.log'
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    base_config = tools.load_config(args.config)
    base_config = tools.handle_freq(base_config)

    selected_folds     = [FOLD_CONFIGS[i - 1] for i in sorted(set(args.folds))]
    selected_scenarios = {k: SCENARIOS[k] for k in args.scenarios}

    run_fold = (run_centralized_fold_scenario if args.mode == 'centralized'
                else run_local_fold_scenario)

    logging.info(f'Trianel {args.mode.upper()} Experiment')
    logging.info(f'Folds:     {[f["name"] for f in selected_folds]}')
    logging.info(f'Scenarios: {list(selected_scenarios.keys())}')

    all_results = []

    for fold_config in selected_folds:
        period_config = build_period_config(base_config, fold_config)
        park_ids      = list(period_config['fl']['clients'].keys())

        logging.info(f'\n{"=" * 70}')
        logging.info(f'[{fold_config["name"]}] Pre-loading {len(park_ids)} parks...')
        cached_park_data = load_all_parks_for_fold(park_ids, period_config, fold_config)
        logging.info(f'Cached {len(cached_park_data)} parks — reused across '
                     f'{len(selected_scenarios)} scenarios')
        logging.info('=' * 70)

        for scenario_name, scenario in selected_scenarios.items():
            result = run_fold(
                base_config=base_config,
                fold_config=fold_config,
                scenario_name=scenario_name,
                scenario=scenario,
                args=args,
                cached_park_data=cached_park_data,
            )
            all_results.append(result)

    logging.info('All folds and scenarios completed.')

    print(f'\n=== Summary ({args.mode}) ===')
    print(f'{"Fold":<10} {"Scenario":<10} {"Mean RMSE":>10} {"Mean MAE":>10} {"Mean R²":>10}')
    print('-' * 54)
    for r in all_results:
        m    = r['metrics']
        rmse = np.mean([v['RMSE'] for v in m.values()]) if m else float('nan')
        mae  = np.mean([v['MAE']  for v in m.values()]) if m else float('nan')
        r2   = np.mean([v['R^2']  for v in m.values()]) if m else float('nan')
        print(f'{r["fold"]:<10} {r["scenario"]:<10} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}')


if __name__ == '__main__':
    main()
