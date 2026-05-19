#!/usr/bin/env python3
"""
trianel_fl.py — Trianel Federated TFT Experiment

2×2 Factorial Design (Synthetic Augmentation × Static Features):
  A: real data only, no static features
  B: real + synthetic data, with static features
  C: real + synthetic data, no static features
  D: real data only, with static features

3-fold Expanding Window CV (Jul 2023 – May 2024).
Validation is ALWAYS on real SCADA measurements only.

Usage:
  python trianel_fl.py -c config_trianel_fl.yaml
  python trianel_fl.py --scenarios A D --folds 1 2
  python trianel_fl.py --scenarios B --folds 1 --save_model
"""
import argparse
import copy
import gc
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
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
# Experiment constants
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
# Pretrained-weight transfer config
# ---------------------------------------------------------------------------

# Substrings matched against state_dict keys to select which layers are
# loaded from the pretrained CL model when --pretrained_context is used.
# Every key containing at least one of these strings will be transferred.
# Freeze / lr behaviour is controlled by fl.pretrained_weights_freeze and
# fl.pretrained_weights_lr in the config.
PRETRAINED_LAYER_PATTERNS = [
    'static_embed', 'static_variable_selection', 'static_context_grn',
    'static_enrichment_grn', 'static_state_h_grn',
]

# Path template for the large CL model.  {fold} is replaced by the fold
# identifier without underscore (fold1 / fold2 / fold3).
PRETRAINED_MODEL_TEMPLATE = (
    'models/trianel/cl_m-tft_out-48_freq-1h_wind_160cl_{fold}.pt'
)


def get_pretrained_keys(state_dict: dict, patterns: list) -> set:
    """Return all state_dict keys whose name contains at least one pattern."""
    return {k for k in state_dict if any(p in k for p in patterns)}


def _is_synth_station(station_id: str) -> bool:
    """Return True for 5-digit numeric IDs (generic synthetic dataset, e.g. '00298')."""
    return station_id.isdigit() and len(station_id) == 5


def _align_dfs_columns(dfs: dict, ref_key: str) -> dict:
    """
    Align all DataFrames in dfs to have exactly the same columns as dfs[ref_key].
    Missing columns are filled with 0.0; extra columns are dropped.

    This is needed because generic synthetic stations (5-digit IDs loaded via
    preprocess_synth_wind_icond2) may lack ECMWF columns that the real trianel
    park DataFrames have — causing StandardScaler.partial_fit to fail with a
    feature-count mismatch.
    """
    if ref_key not in dfs or len(dfs) == 1:
        return dfs
    ref_cols = dfs[ref_key].columns.tolist()
    aligned = {}
    for key, df in dfs.items():
        if key == ref_key:
            aligned[key] = df
            continue
        df = df.copy()
        for col in ref_cols:
            if col not in df.columns:
                df[col] = 0.0
        aligned[key] = df[ref_cols]
    return aligned


def _build_synth_station_config(period_config: dict) -> dict:
    """
    Return a deep-copy of period_config with data paths overridden to point
    at the 160-station fully-synthetic dataset (data.synthetic_* keys).
    Falls back to the original paths if override keys are absent.
    """
    cfg = copy.deepcopy(period_config)
    overrides = {
        'synthetic_data_path':              'path',
        'synth_stations_nwp_path':          'nwp_path',
        'synth_stations_ecmwf_path':        'ecmwf_path',
        'synth_stations_power_curves_path': 'power_curves_path',
    }
    for src_key, dst_key in overrides.items():
        if src_key in period_config['data']:
            cfg['data'][dst_key] = period_config['data'][src_key]
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_metrics_csv(csv_path: str, fold_name: str, scenario_name: str, metrics_per_park: dict) -> None:
    """Append per-park metrics to a semicolon-delimited CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as fh:
        if write_header:
            fh.write('Fold;Scenario;Park;RMSE;MAE;R2\n')
        for park_id, m in sorted(metrics_per_park.items()):
            fh.write(f'{fold_name};{scenario_name};{park_id};'
                     f'{m["RMSE"]:.6f};{m["MAE"]:.6f};{m["R^2"]:.6f}\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Trianel FL TFT Experiment')
    parser.add_argument('-c', '--config', default='configs/config_trianel_fl.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--scenarios', nargs='+', default=list(SCENARIOS.keys()),
                        choices=list(SCENARIOS.keys()),
                        help='Scenarios to run (default: all)')
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3],
                        choices=[1, 2, 3],
                        help='Fold numbers to run (default: all)')
    parser.add_argument('--save_model', action='store_true',
                        help='Save client weights to disk')
    parser.add_argument('--pretrained_context', action='store_true',
                        help='Load personal (non-aggregated) layers from the matching '
                             'centralized model (models/trianel/cl_tft_<fold>_<scenario>.pt) '
                             'and freeze them for the entire FL run')
    parser.add_argument('--gpu', nargs='+', type=int, default=None,
                        help='GPU indices to use (e.g. --gpu 1 2 3). Default: alle GPUs.')
    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity (default: INFO)')
    return parser.parse_args()


def build_period_config(base_config: dict, fold_config: dict) -> dict:
    """Deep-copy base_config and inject fold-specific date boundaries."""
    cfg = copy.deepcopy(base_config)
    cfg['data']['train_start'] = fold_config['train_start']
    cfg['data']['test_start']  = fold_config['val_start']
    cfg['data']['test_end']    = fold_config['val_end']
    return cfg


def get_features_for_scenario(base_config: dict, scenario: dict) -> dict:
    """Return features dict with static stripped for scenarios A and C."""
    features = preprocessing.get_features(base_config)
    if not scenario['use_static']:
        features['static'] = []
    return features


def load_client_dfs(park_id: str,
                    period_config: dict,
                    fold_config: dict,
                    scenario: dict,
                    features: dict) -> dict:
    """
    Load preprocessed DataFrames for one park.

    Returns a dict with key 'dfs':
      - Always contains {park_id: real_df}
      - Also contains {park_id_synth: synth_df} when scenario.use_synthetic=True

    If use_synthetic=True, real and synthetic are loaded in parallel for faster I/O.
    """
    fold_start = pd.Timestamp(fold_config['train_start'], tz='UTC')
    freq = period_config['data'].get('freq', '1H')

    if not scenario['use_synthetic']:
        real_df = preprocessing.preprocess_trianel_tft(
            park_id=park_id,
            config=period_config,
            freq=freq,
            features=features,
            fold_start=fold_start,
            use_real=True,
        )
        dfs = {park_id: real_df}
    else:
        real_df = preprocessing.preprocess_trianel_tft(
            park_id, period_config, freq, features, fold_start, True
        )
        dfs = {park_id: real_df}

        # Trianel-specific "half-synthetic" — only when explicitly enabled
        load_real_synth = period_config['params'].get('get_synthetic_real_parks', False)
        if load_real_synth:
            synth_df = preprocessing.preprocess_trianel_tft(
                park_id, period_config, freq, features, fold_start, False
            )
            dfs[f'{park_id}_synth'] = synth_df

        # Generic 160-station synthetic for this client (5-digit IDs in fl.clients)
        fl_clients = period_config['fl']['clients']
        synth_station_ids = [s for s in fl_clients.get(park_id, []) if _is_synth_station(s)]
        if synth_station_ids:
            synth_cfg = _build_synth_station_config(period_config)
            for station_id in synth_station_ids:
                try:
                    station_df = preprocessing.preprocess_synth_wind_icond2(
                        path=station_id,
                        config=synth_cfg,
                        freq=freq,
                        features=features,
                    )
                    dfs[station_id] = station_df
                except Exception as exc:
                    logging.warning(f'Could not load synth station {station_id}: {exc}')

    # Align all DataFrames to the real park's column set (fills missing ECMWF cols with 0)
    dfs = _align_dfs_columns(dfs, park_id)
    return {'dfs': dfs}


def load_all_parks_for_fold(park_ids: list,
                            period_config: dict,
                            fold_config: dict) -> dict:
    """
    Load all parks for a fold ONCE (real + synthetic data).
    Always loads with ALL features including static so the cache is usable
    by every scenario. load_client_dfs_from_cache() drops static columns
    for scenarios that don't need them.

    Returns:
        {park_id: {'real': real_df, 'synth': synth_df}}
    """
    import time
    fold_start = pd.Timestamp(fold_config['train_start'], tz='UTC')
    freq = period_config['data'].get('freq', '1H')
    # Always include static features in the cache so scenarios B and D can use them.
    full_features = preprocessing.get_features(period_config)
    all_parks_data = {}
    load_times = []

    load_real_synth = period_config['params'].get('get_synthetic_real_parks', False)

    for park_id in tqdm(park_ids, desc='Loading parks (all scenarios)', unit='park'):
        t0 = time.time()

        if load_real_synth:
            with ThreadPoolExecutor(max_workers=2) as executor:
                real_future  = executor.submit(
                    preprocessing.preprocess_trianel_tft,
                    park_id, period_config, freq, full_features, fold_start, True
                )
                synth_future = executor.submit(
                    preprocessing.preprocess_trianel_tft,
                    park_id, period_config, freq, full_features, fold_start, False
                )
                real_df  = real_future.result()
                synth_df = synth_future.result()
        else:
            real_df  = preprocessing.preprocess_trianel_tft(
                park_id, period_config, freq, full_features, fold_start, True
            )
            synth_df = None

        elapsed = time.time() - t0
        load_times.append((park_id, elapsed))
        all_parks_data[park_id] = {'real': real_df, 'synth': synth_df}

    logging.info('\nLoading summary (trianel parks):')
    for park_id, elapsed in load_times:
        logging.info(f'  {park_id}: {elapsed:.2f}s')
    total = sum(t for _, t in load_times)
    avg = total / len(load_times) if load_times else 0
    logging.info(f'  Total: {total:.2f}s, Avg: {avg:.2f}s/park')

    # Load generic synthetic stations (5-digit IDs) found in fl.clients
    fl_clients = period_config['fl']['clients']
    all_synth_ids = sorted({
        s for stations in fl_clients.values() for s in stations
        if _is_synth_station(s)
    })
    if all_synth_ids:
        synth_cfg = _build_synth_station_config(period_config)
        logging.info(f'Loading {len(all_synth_ids)} generic synthetic stations '
                     f'from {synth_cfg["data"]["path"]}...')
        for station_id in tqdm(all_synth_ids, desc='Loading synth stations', unit='station'):
            try:
                synth_df = preprocessing.preprocess_synth_wind_icond2(
                    path=station_id,
                    config=synth_cfg,
                    freq=freq,
                    features=full_features,
                )
                all_parks_data[station_id] = {'real': None, 'synth': synth_df}
            except Exception as exc:
                logging.warning(f'Could not load synthetic station {station_id}: {exc}')
    else:
        logging.info('No generic synthetic station IDs found in fl.clients — skipping.')

    return all_parks_data


def load_client_dfs_from_cache(park_id: str,
                               cached_data: dict,
                               scenario: dict,
                               features: dict,
                               base_config: dict,
                               fl_clients: dict = None) -> dict:
    """
    Prepare client data from pre-loaded cached DataFrames.
    Drops static feature columns when the scenario doesn't use them.

    Args:
        park_id:     Park identifier.
        cached_data: {park_id: {'real': df, 'synth': df}} from load_all_parks_for_fold().
                     Also contains {station_id: {'real': None, 'synth': df}} for generic stations.
        scenario:    Scenario dict ('use_synthetic', 'use_static').
        features:    Scenario-specific features dict (static=[] for scenarios A/C).
        base_config: Full base config (used to identify which columns are static).
        fl_clients:  fl.clients mapping — used to look up which generic station IDs belong
                     to this client. If None, generic synthetic stations are skipped.

    Returns:
        {'dfs': {park_id: real_df, [park_id_synth: synth_df], [station_id: synth_df, ...]}}
    """
    static_cols = base_config['params'].get('static_features', [])

    def _apply_scenario(df: pd.DataFrame) -> pd.DataFrame:
        if not scenario['use_static'] and static_cols:
            return df.drop(columns=[c for c in static_cols if c in df.columns], errors='ignore')
        return df

    real_df  = _apply_scenario(cached_data[park_id]['real'])
    dfs = {park_id: real_df}

    if scenario['use_synthetic']:
        # Trianel-specific "half-synthetic" data — only when explicitly enabled
        load_real_synth = base_config['params'].get('get_synthetic_real_parks', False)
        if load_real_synth and cached_data[park_id].get('synth') is not None:
            dfs[f'{park_id}_synth'] = _apply_scenario(cached_data[park_id]['synth'])

        # Generic 160-station synthetic assigned to this client via fl.clients
        if fl_clients is not None:
            for station_id in fl_clients.get(park_id, []):
                if not _is_synth_station(station_id):
                    continue
                entry = cached_data.get(station_id)
                if entry is None or entry.get('synth') is None:
                    continue
                dfs[station_id] = _apply_scenario(entry['synth'])

    # Align all DataFrames to the real park's column set (fills missing ECMWF cols with 0)
    dfs = _align_dfs_columns(dfs, park_id)
    return {'dfs': dfs}


def fit_and_aggregate_scalers(clients_data: dict, period_config: dict) -> StandardScaler:
    """
    Phase 1: fit a local StandardScaler on each client's training split.
    Phase 2: FedPS aggregation (Verschiebungssatz der Varianz) → global scaler.
    Sets client_info['scaler_x'] = global_scaler for every client in-place.
    Clients whose training split is entirely empty are removed from clients_data.
    """
    target_col = period_config['data']['target_col']
    empty_clients = []

    # Phase 1 — local partial fits
    for client_id, client_info in tqdm(clients_data.items(), desc='Fitting scalers', unit='client'):
        scaler_x = StandardScaler()
        n_train_rows = 0
        for df in client_info['dfs'].values():
            df_train, _ = preprocessing.split_data(
                data=df,
                train_frac=period_config['data']['train_frac'],
                test_start=pd.Timestamp(period_config['data']['test_start'], tz='UTC'),
                test_end=pd.Timestamp(period_config['data']['test_end'], tz='UTC'),
                t_0=0,
            )
            df_train_x = df_train.drop(columns=[target_col], errors='ignore')
            if df_train_x.shape[0] == 0:
                continue
            scaler_x.partial_fit(df_train_x.values)
            n_train_rows += df_train_x.shape[0]

        if n_train_rows == 0:
            logging.warning(f'Client {client_id} has no training data for this fold — excluded.')
            empty_clients.append(client_id)
            continue
        client_info['scaler_x'] = scaler_x

    for cid in empty_clients:
        del clients_data[cid]

    if not clients_data:
        raise RuntimeError('No clients with training data remain after filtering.')

    # Phase 2 — FedPS aggregation
    client_stats_x = [
        {
            'n':   ci['scaler_x'].n_samples_seen_,
            'mu':  ci['scaler_x'].mean_,
            'var': ci['scaler_x'].var_,
        }
        for ci in clients_data.values()
    ]
    mu_global, std_global = federated.aggregate_scalers(client_stats_x)

    global_scaler_x = StandardScaler()
    global_scaler_x.mean_             = mu_global
    global_scaler_x.scale_            = std_global
    global_scaler_x.var_              = std_global ** 2
    global_scaler_x.n_samples_seen_   = sum(s['n'] for s in client_stats_x)
    global_scaler_x.n_features_in_    = len(mu_global)

    for client_info in clients_data.values():
        client_info['scaler_x'] = global_scaler_x

    logging.info(f'FedPS: global scaler applied to {len(clients_data)} clients '
                 f'({global_scaler_x.n_samples_seen_} total train samples, '
                 f'{global_scaler_x.n_features_in_} features)')
    return global_scaler_x


def prepare_partitions(clients_data: dict,
                       period_config: dict,
                       features: dict) -> tuple:
    """
    Build FL partitions per client.

    Train data = real + synthetic (when present) after scaler transform.
    Val data   = real data only (extracted from test_data[park_id]).

    Returns:
        partitions:     {park_id: (X_train, y_train, X_val, y_val)}
        eval_test_data: {park_id: (X_val, y_val, index_val, scaler_y)}
    """
    partitions     = {}
    eval_test_data = {}

    for client_id, client_info in tqdm(clients_data.items(), desc='Preparing partitions', unit='client'):
        gen = tools.create_data_generator(
            dfs=client_info['dfs'],
            config=period_config,
            features=features,
            scaler_x=client_info['scaler_x'],
        )
        X_train, y_train, _, _, client_test_data, _ = tools.combine_datasets_efficiently(gen)

        # Validation is ALWAYS extracted from the real park key only.
        # client_test_data also contains synthetic station keys (e.g. '00298'), but
        # those are never used for evaluation — neither per-round nor in the final eval.
        if client_id not in client_test_data:
            raise KeyError(
                f'Real park key "{client_id}" not found in test_data. '
                f'Available keys: {list(client_test_data.keys())}'
            )
        X_val, y_val, index_val, scaler_y_val = client_test_data[client_id]

        n_sources = len(client_info['dfs'])
        y_train_len = y_train.shape[0] if hasattr(y_train, 'shape') else '?'
        y_val_len   = y_val.shape[0]   if hasattr(y_val,   'shape') else '?'
        logging.info(
            f'  {client_id}: train={y_train_len} samples '
            f'({n_sources} source(s): {list(client_info["dfs"].keys())}), '
            f'val={y_val_len} samples (real only)'
        )

        partitions[client_id]     = (X_train, y_train, X_val, y_val)
        eval_test_data[client_id] = client_test_data[client_id]

        del gen
        gc.collect()

    return partitions, eval_test_data


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_fold_scenario(base_config: dict,
                      fold_config: dict,
                      scenario_name: str,
                      scenario: dict,
                      args: argparse.Namespace,
                      cached_park_data: dict = None) -> dict:
    """Run one fold × scenario combination end-to-end.

    Args:
        cached_park_data: Optional dict of pre-loaded park data from load_all_parks_for_fold().
                          If provided, skips I/O and uses cached data instead.
                          Format: {park_id: {'real': df, 'synth': df}}
    """
    logging.info(f'{"=" * 60}')
    logging.info(f'Fold: {fold_config["name"]}  |  Scenario: {scenario_name}')
    logging.info(f'  train: {fold_config["train_start"]} – {fold_config["train_end"]}')
    logging.info(f'  val:   {fold_config["val_start"]}  – {fold_config["val_end"]}')
    logging.info(f'  use_synthetic={scenario["use_synthetic"]}  '
                 f'use_static={scenario["use_static"]}')

    period_config = build_period_config(base_config, fold_config)
    features      = get_features_for_scenario(period_config, scenario)
    park_ids      = list(period_config['fl']['clients'].keys())

    logging.info(f'Parks: {park_ids}')
    logging.info(f'Features — known: {features["known"]}, '
                 f'observed: {features["observed"]}, '
                 f'static: {features["static"]}')

    # --- 1. Load data ---
    logging.info('Loading client data...')
    clients_data = {}

    if cached_park_data:
        # Use cached data (fast path) — no I/O
        logging.info('Using cached park data (pre-loaded for this fold).')
        for park_id in tqdm(park_ids, desc='Preparing client data', unit='park'):
            try:
                clients_data[park_id] = load_client_dfs_from_cache(
                    park_id, cached_park_data, scenario, features, base_config,
                    fl_clients=period_config['fl']['clients'],
                )
            except Exception as exc:
                logging.error(f'Failed to prepare cached data for {park_id}: {exc}')
                raise
    else:
        # Load from disk (slow path) — full I/O
        logging.info('Loading park data from disk (no cache provided).')
        for park_id in tqdm(park_ids, desc='Loading parks', unit='park'):
            try:
                clients_data[park_id] = load_client_dfs(
                    park_id, period_config, fold_config, scenario, features
                )
            except Exception as exc:
                logging.error(f'Failed to load data for {park_id}: {exc}')
                raise

    # --- 2. FedPS scalers ---
    logging.info('Computing FedPS global scaler...')
    fit_and_aggregate_scalers(clients_data, period_config)

    # --- 3. Prepare partitions ---
    logging.info('Preparing FL partitions...')
    partitions, eval_test_data = prepare_partitions(clients_data, period_config, features)

    # Set feature_dim in config (required by get_model)
    first_X_train = next(iter(partitions.values()))[0]
    period_config['model']['feature_dim'] = tools.get_feature_dim(X=first_X_train)
    logging.info(f'Feature dims: {period_config["model"]["feature_dim"]}')

    # --- 4. Hyperparameters ---
    hyperparameters = hpo_utils.get_hyperparameters(period_config, hpo=False)
    logging.info(f'Hyperparameters: {hyperparameters}')

    # --- 5. FL training ---
    initial_pretrained_weights = None
    if args.pretrained_context:
        fold_tag = fold_config['name'].replace('_', '')   # 'fold_1' → 'fold1'
        pt_path  = PRETRAINED_MODEL_TEMPLATE.format(fold=fold_tag)
        if os.path.exists(pt_path):
            pretrained_sd    = torch.load(pt_path, map_location='cpu')
            pretrained_keys  = get_pretrained_keys(pretrained_sd, PRETRAINED_LAYER_PATTERNS)
            initial_pretrained_weights = {k: v for k, v in pretrained_sd.items()
                                          if k in pretrained_keys}
            freeze = period_config['fl'].get('pretrained_weights_freeze', True)
            pt_lr  = period_config['fl'].get('pretrained_weights_lr', 'default * 0.1')
            logging.info(
                f'pretrained_context: loaded {len(initial_pretrained_weights)} tensors '
                f'from {pt_path}  |  freeze={freeze}  pretrained_lr={pt_lr}'
            )
        else:
            logging.warning(
                f'--pretrained_context: no model at {pt_path} — running without pretrained weights'
            )

    logging.info('Starting FL simulation...')
    history, clients_weights = federated.run_simulation(
        partitions=partitions,
        config=period_config,
        hyperparameters=hyperparameters,
        initial_pretrained_weights=initial_pretrained_weights,
    )

    # --- 6. Evaluate on real val data ---
    logging.info('Evaluating on real validation data...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics_per_park = {}

    for park_id in park_ids:
        if park_id not in eval_test_data or park_id not in clients_weights:
            logging.warning(f'Skipping evaluation for {park_id} (missing data or weights)')
            continue

        X_val, y_val, _index_val, scaler_y_val = eval_test_data[park_id]

        model = models_utils.get_model(config=period_config, hyperparameters=hyperparameters)
        model.load_state_dict(clients_weights[park_id])
        model = model.to(device)

        y_true, y_pred = tools.get_y(X_val, y_val, model, scaler_y=scaler_y_val, device=device)
        raw = eval_utils.get_metrics(y_pred, y_true)
        # get_metrics returns single-element lists; extract scalars for clean aggregation
        metrics = {k: float(v[0]) if isinstance(v, (list, np.ndarray)) else float(v)
                   for k, v in raw.items()}
        metrics_per_park[park_id] = metrics
        logging.info(f'  {park_id:20s}  RMSE={metrics["RMSE"]:.4f}  '
                     f'MAE={metrics["MAE"]:.4f}  R²={metrics["R^2"]:.4f}')

        del model
        gc.collect()

    # --- 7. Aggregate summary ---
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
        'history':  history if period_config['fl'].get('save_history', False) else None,
    }

    # --- 8. Persist results ---
    out_dir = period_config['eval']['results_path']
    os.makedirs(out_dir, exist_ok=True)

    result_file = os.path.join(out_dir, f'{fold_config["name"]}_{scenario_name}_results.pkl')
    with open(result_file, 'wb') as fh:
        pickle.dump(result, fh)
    logging.info(f'Results saved → {result_file}')

    csv_path = os.path.join('results', 'trianel', 'federated', f'metrics_{scenario_name}.csv')
    _append_metrics_csv(csv_path, fold_config['name'], scenario_name, metrics_per_park)
    logging.info(f'Metrics CSV   → {csv_path}')

    if args.save_model:
        weights_file = os.path.join(out_dir, f'{fold_config["name"]}_{scenario_name}_weights.pkl')
        with open(weights_file, 'wb') as fh:
            pickle.dump(clients_weights, fh)
        logging.info(f'Weights saved  → {weights_file}')

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # GPU initialization is delegated to federated.run_simulation() for proper Ray management.
    # Initializing GPU here would reserve CUDA_VISIBLE_DEVICES before Ray.init(),
    # preventing Ray from distributing GPUs to Actors.
    # Only initialize GPU after FL training for server-side evaluation (see line ~328).

    base_config = tools.load_config(args.config)
    base_config = tools.handle_freq(base_config)

    selected_folds     = [FOLD_CONFIGS[i - 1] for i in sorted(set(args.folds))]
    selected_scenarios = {k: SCENARIOS[k] for k in args.scenarios}

    logging.info(f'Trianel FL Experiment')
    logging.info(f'Folds:     {[f["name"] for f in selected_folds]}')
    logging.info(f'Scenarios: {list(selected_scenarios.keys())}')

    all_results = []
    for fold_config in selected_folds:
        # Pre-load all parks for this fold (real + synthetic) ONCE
        period_config = build_period_config(base_config, fold_config)
        park_ids = list(period_config['fl']['clients'].keys())

        logging.info(f"\n{'='*70}")
        logging.info(f"[{fold_config['name']}] Pre-loading all {len(park_ids)} parks (real + synthetic)...")
        cached_park_data = load_all_parks_for_fold(park_ids, period_config, fold_config)
        logging.info(f"✓ Cached {len(cached_park_data)} parks — will be reused across {len(selected_scenarios)} scenarios")
        logging.info(f"{'='*70}\n")

        # Now iterate scenarios using cached data (no more I/O!)
        for scenario_name, scenario in selected_scenarios.items():
            result = run_fold_scenario(
                base_config=base_config,
                fold_config=fold_config,
                scenario_name=scenario_name,
                scenario=scenario,
                args=args,
                cached_park_data=cached_park_data,  # ← Pass cached data
            )
            all_results.append(result)

    logging.info('All folds and scenarios completed.')

    # Print compact summary table
    print('\n=== Summary ===')
    print(f'{"Fold":<10} {"Scenario":<10} {"Mean RMSE":>10} {"Mean MAE":>10} {"Mean R²":>10}')
    print('-' * 52)
    for r in all_results:
        m = r['metrics']
        rmse = np.mean([v['RMSE'] for v in m.values()]) if m else float('nan')
        mae  = np.mean([v['MAE']  for v in m.values()]) if m else float('nan')
        r2   = np.mean([v['R^2']  for v in m.values()]) if m else float('nan')
        print(f'{r["fold"]:<10} {r["scenario"]:<10} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}')


if __name__ == '__main__':
    main()
