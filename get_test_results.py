#!/usr/bin/env python3
"""
Evaluate trained TFT models on test data and save metrics to CSV.

This script iterates over trained TFT models and:
1. Loads the configuration and hyperparameters
2. Prepares the data
3. Loads the trained model
4. Makes predictions on test data
5. Calculates metrics: R², RMSE, MAE
6. Saves results to CSV

Supports five modes:
- Default: Process station-specific models → test_results_all.csv
- -g/--global: Process global models → test_results_global.csv
- -e/--evallocal: Evaluate global models on local datasets → test_results_<model>.csv
- --evaldaily: Evaluate global models on local datasets → daily_results_<model>.csv
  (CSV with forecast date rows, station ID columns, R² values per forecast run)
- --localperformance: Evaluate local models on local datasets → daily_results_local.csv
  (CSV with forecast date rows, station ID columns, R² values per forecast run)
"""

import os
import sys
import warnings
import yaml

warnings.filterwarnings('ignore', message='X does not have valid feature names', category=UserWarning)
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import gc
import pickle
import logging
from datetime import datetime
import argparse

# Import project utilities
from utils.models import get_model
from utils import preprocessing, tools, hpo

# Setup logging
def setup_logging():
    """Setup logging to file and console."""
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'test_results_{timestamp}.log'

    # Create logger
    logger = logging.getLogger('test_results')
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()


def load_scaler_for_model(model_path, models_dir=None):
    """Try to load a saved scaler for the given model. Returns (scaler_x, scaler_y)."""
    if models_dir is None:
        models_dir = Path('models')
    model_stem = Path(model_path).stem
    scaler_path = Path(models_dir) / 'scaler' / f'scaler_{model_stem}.pkl'
    scaler_y_path = Path(models_dir) / 'scaler' / f'scaler_y_{model_stem}.pkl'

    scaler_x = None
    if scaler_path.exists():
        try:
            with open(scaler_path, 'rb') as f:
                scaler_x = pickle.load(f)
            logger.info(f"Loaded scaler from: {scaler_path}")
        except Exception as e:
            logger.warning(f"Could not load scaler from {scaler_path}: {e}")
    else:
        logger.debug(f"No saved scaler found at {scaler_path}, will fit from data")

    scaler_y = None
    if scaler_y_path.exists():
        try:
            with open(scaler_y_path, 'rb') as f:
                scaler_y = pickle.load(f)
            logger.info(f"Loaded scaler_y from: {scaler_y_path}")
        except Exception as e:
            logger.warning(f"Could not load scaler_y from {scaler_y_path}: {e}")
    else:
        logger.warning(f"scaler_y not found at {scaler_y_path} — will fall back to fitting from data.")

    return scaler_x, scaler_y


def get_config_path_from_model_name(model_name):
    """Extract config file path from model name."""
    parts = model_name.replace('.pt', '').split('_')

    if 'wind' not in parts:
        return None, False

    wind_idx = parts.index('wind')
    identifier = '_'.join(parts[wind_idx:])

    # Handle nostatic case
    is_nostatic = 'nostatic' in identifier
    if is_nostatic:
        identifier = identifier.replace('_nostatic', '')

    numeric_id = parts[-1] if not is_nostatic else parts[-2]

    if numeric_id.isdigit() and len(numeric_id) == 5:
        # Numeric ID -> subdirectory
        if os.path.exists(f'configs/wind_100ex50/config_{identifier}.yaml'):
            config_path = Path(f'configs/wind_100ex50/config_{identifier}.yaml')
        else:
            config_path = Path(f'configs/wind_50/config_{identifier}.yaml')
    else:
        # Named config -> root configs directory
        config_path = Path(f'configs/config_{identifier}.yaml')

    return config_path, is_nostatic


def load_config(config_path, is_nostatic=False):
    """Load config from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if is_nostatic:
        config['params']['static_features'] = []

    return config


def extract_station_id(model_name):
    """Extract station ID from model name."""
    parts = model_name.replace('.pt', '').split('_')
    station_id = parts[-1]
    station_id = station_id.replace('nostatic', '').strip('_')
    return station_id


def extract_model_name(model_name):
    """Extract model identifier from model name for global models."""
    parts = model_name.replace('.pt', '').split('_')
    if 'wind' not in parts:
        return None

    wind_idx = parts.index('wind')
    identifier = '_'.join(parts[wind_idx:])
    return identifier


def calculate_metrics(y_true, y_pred):
    """Calculate R², RMSE, and MAE."""
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate metrics
    r2 = r2_score(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)

    return r2, rmse, mae


def compute_skill_nwp(dfs, test_data, y_test_np, rmse_model, config_data, target_col):
    """
    Compute Skill_NWP = 1 - RMSE_model / RMSE_NWP.

    RMSE_NWP is the direct RMSE between the nearest NWP grid point's wind_speed_h10
    column and the measured wind_speed column over the test period.
    No windowing or aggregation across forecast runs — timestamps are deduplicated
    so each observed value is counted once.

    Column priority for NWP baseline:
      1. '_1' suffix (geodesic_next method, nearest point by construction)
      2. Inner quad labels '_NW', '_NE', '_SW', '_SE' (relative_position method)
      3. First available wind_speed_h10_* column

    Returns float or None (None when target_col != 'wind_speed' or data is missing).
    """
    if target_col != 'wind_speed':
        return None

    test_start_ts = pd.Timestamp(config_data['test_start'], tz='UTC')
    test_end_ts   = pd.Timestamp(config_data['test_end'],   tz='UTC')

    y_nwp_all  = []
    y_true_all = []

    for key in test_data:
        if key not in dfs:
            continue
        df_raw = dfs[key]

        # --- select NWP baseline column ---
        # The nearest grid point label is stored in df.attrs during preprocessing.
        # For geodesic_next: label='1' → column 'wind_speed_h10_1'.
        # For relative_position: label=compass direction of geodesically nearest point (e.g. 'SW').
        if 'wind_speed' not in df_raw.columns:
            continue

        nearest_label = df_raw.attrs.get('nwp_nearest_label')
        if nearest_label is None:
            logger.warning(f"nwp_nearest_label not set in df.attrs for {key} — skipping Skill_NWP")
            continue

        nwp_col = f'wind_speed_h10_{nearest_label}'
        if nwp_col not in df_raw.columns:
            logger.warning(f"Expected NWP baseline column '{nwp_col}' not found for {key} — skipping")
            continue

        # --- flatten MultiIndex to unique timestamps (no averaging) ---
        if isinstance(df_raw.index, pd.MultiIndex):
            df_flat = (df_raw[[nwp_col, 'wind_speed']]
                       .reset_index()
                       .drop_duplicates('timestamp')
                       .set_index('timestamp'))
        else:
            df_flat = df_raw[[nwp_col, 'wind_speed']]

        df_test = df_flat.loc[test_start_ts:test_end_ts]
        if df_test.empty:
            continue

        mask = df_test[nwp_col].notna() & df_test['wind_speed'].notna()
        y_nwp_all.append(df_test.loc[mask, nwp_col].values)
        y_true_all.append(df_test.loc[mask, 'wind_speed'].values)

    if not y_nwp_all:
        return None

    y_nwp  = np.concatenate(y_nwp_all)
    y_true = np.concatenate(y_true_all)
    rmse_nwp = float(np.sqrt(mean_squared_error(y_true, y_nwp)))
    if rmse_nwp == 0:
        return None
    return 1.0 - rmse_model / rmse_nwp


def evaluate_model(model_path, models_dir, identifier_key='station_id', identifier_value=None, config=None, scaler_x=None, scaler_y=None):
    """
    Evaluate a single model on test data.

    Args:
        config:   Optional pre-loaded config dict. If None, derived from model filename.
        scaler_x: Optional pre-fitted StandardScaler for features. If None, fitted internally.
        scaler_y: Optional pre-fitted StandardScaler for target. Required when target_col != 'power'.

    Returns:
        Tuple of (identifier_value, metrics_dict)
    """
    model_name = model_path.name
    logger.info("="*80)
    logger.info(f"Evaluating: {model_name}")
    logger.info("="*80)

    # Extract identifier
    if identifier_value is None:
        identifier_value = extract_station_id(model_name) if identifier_key == 'station_id' else extract_model_name(model_name)

    logger.info(f"{identifier_key}: {identifier_value}")

    # Get config (from argument or derived from model name)
    if config is None:
        config_path, is_nostatic = get_config_path_from_model_name(model_name)
        if config_path is None or not config_path.exists():
            logger.warning(f"Config not found for {model_name}, skipping")
            return identifier_value, None

        logger.info(f"Config: {config_path}")

        try:
            config = load_config(config_path, is_nostatic)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return identifier_value, None

    config['model']['name'] = 'tft'

    # Extract hyperparameters from model name
    parts = model_name.replace('.pt', '').split('_')
    output_dim = int([p for p in parts if 'out-' in p][0].replace('out-', ''))
    freq_part = [p for p in parts if 'freq-' in p][0]
    freq = freq_part.replace('freq-', '')
    freq_idx = parts.index(freq_part)
    study_name_suffix = '_'.join(parts[freq_idx+1:])
    study_name = f'cl_m-{config["model"]["name"]}_out-{output_dim}_freq-{freq}_{study_name_suffix}'

    # Load hyperparameters
    study = None
    try:
        if config['model']['lookup_hpo']:
            study = hpo.load_study(config['hpo']['studies_path'], study_name)
            logger.info(f"Study loaded: {study.study_name}")
    except Exception as e:
        logger.warning(f"Could not load study: {e}")

    hyperparameters = hpo.get_hyperparameters(config=config, study=study)
    features = preprocessing.get_features(config=config)

    # Load data
    try:
        data_dir = config['data']['path']
        dfs = preprocessing.get_data(
            data_dir=data_dir,
            config=config,
            freq=config['data']['freq'],
            features=features
        )
        logger.info(f"Loaded {len(dfs)} dataframes")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return identifier_value, None

    # Fit global scaler (or use provided one)
    target_col = config['data']['target_col']
    try:
        if scaler_x is not None:
            global_scaler_x = scaler_x
        else:
            global_scaler_x = StandardScaler()
            test_start = pd.Timestamp(config['data']['test_start'], tz='UTC')
            test_end = pd.Timestamp(config['data']['test_end'], tz='UTC')

            if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
                test_end = test_end.replace(hour=23, minute=0, second=0)

            for key, df in dfs.items():
                df_temp = df.copy()
                if target_col in df_temp.columns:
                    df_temp.drop(target_col, axis=1, inplace=True)

                t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
                df_train, _ = preprocessing.split_data(
                    data=df_temp,
                    train_frac=config['data']['train_frac'],
                    test_start=test_start,
                    test_end=test_end,
                    t_0=t_0
                )
                global_scaler_x.partial_fit(df_train)
                del df_temp, df_train
                gc.collect()

        # Fit scaler_y on target column if needed (when target is not pre-normalised)
        global_scaler_y = scaler_y
        if global_scaler_y is None and target_col != 'power':
            logger.warning("No saved scaler_y found — fitting global scaler_y from training data as fallback.")
            global_scaler_y = StandardScaler()
            test_start = pd.Timestamp(config['data']['test_start'], tz='UTC')
            test_end = pd.Timestamp(config['data']['test_end'], tz='UTC')
            if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
                test_end = test_end.replace(hour=23, minute=0, second=0)
            t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
            for key, df in dfs.items():
                if target_col not in df.columns:
                    continue
                df_train, _ = preprocessing.split_data(
                    data=df[[target_col]],
                    train_frac=config['data']['train_frac'],
                    test_start=test_start,
                    test_end=test_end,
                    t_0=t_0
                )
                global_scaler_y.partial_fit(df_train)
                gc.collect()
    except Exception as e:
        logger.error(f"Error fitting scaler: {e}")
        return identifier_value, None

    # Prepare data
    try:
        data_generator = tools.create_data_generator(dfs, config, features, scaler_x=global_scaler_x,
                                                     scaler_y=global_scaler_y if target_col != 'power' else None)
        X_train, y_train, X_test, y_test, test_data = tools.combine_datasets_efficiently(data_generator)

        # Extract feature dimensions
        feature_dims = {}
        if isinstance(X_train, dict):
            feature_dims['observed_dim'] = X_train['observed'].shape[-1]
            feature_dims['known_dim'] = X_train['known'].shape[-1] if 'known' in X_train else 0
            feature_dims['static_dim'] = X_train['static'].shape[-1] if 'static' in X_train else 0
        else:
            feature_dims['observed_dim'] = X_train.shape[-1]
            feature_dims['known_dim'] = 0
            feature_dims['static_dim'] = 0

        config['model']['feature_dim'] = feature_dims
        gc.collect()
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return identifier_value, None

    # Load model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove '_orig_mod.' prefix
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                cleaned_state_dict[key.replace('_orig_mod.', '')] = value
            else:
                cleaned_state_dict[key] = value

        hyperparameters['model_type'] = 'tft'
        hyperparameters.update(feature_dims)

        model = get_model(config, hyperparameters)
        model.load_state_dict(cleaned_state_dict)
        model.eval()
        logger.info("Model loaded")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return identifier_value, None

    # Run inference
    try:
        test_observed = X_test['observed'] if isinstance(X_test, dict) else X_test
        test_known = X_test.get('known', None) if isinstance(X_test, dict) else None
        test_static = X_test.get('static', None) if isinstance(X_test, dict) else None

        if not isinstance(test_observed, torch.Tensor):
            test_observed = torch.FloatTensor(test_observed)
        if test_known is not None and not isinstance(test_known, torch.Tensor):
            test_known = torch.FloatTensor(test_known)
        if test_static is not None and not isinstance(test_static, torch.Tensor):
            test_static = torch.FloatTensor(test_static)

        with torch.no_grad():
            if test_static is not None:
                predictions = model(test_observed, test_known, test_static)
            else:
                predictions = model(test_observed, test_known)

        predictions_np = predictions.cpu().numpy()
        y_test_np = y_test if isinstance(y_test, np.ndarray) else y_test.cpu().numpy()

        logger.info("Predictions complete")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return identifier_value, None

    # Calculate metrics
    try:
        # Reshape if needed (to match train_cl.py)
        predictions_np = predictions_np.reshape(-1, y_test_np.shape[-1])

        # Inverse-transform if target was scaled (e.g. wind_speed)
        if global_scaler_y is not None:
            predictions_np = global_scaler_y.inverse_transform(predictions_np)
            y_test_np = global_scaler_y.inverse_transform(y_test_np.reshape(-1, predictions_np.shape[-1]))

        # Clip negative values to 0 (only meaningful for power)
        if target_col == 'power':
            predictions_np[predictions_np < 0] = 0

        r2, rmse, mae = calculate_metrics(y_test_np, predictions_np)

        try:
            skill_nwp = compute_skill_nwp(dfs, test_data, y_test_np, rmse, config['data'], target_col)
        except Exception as e_nwp:
            logger.warning(f"Could not compute Skill_NWP: {e_nwp}")
            skill_nwp = None

        metrics = {
            identifier_key: identifier_value,
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'skill_nwp': float(skill_nwp) if skill_nwp is not None else None
        }

        skill_nwp_str = f"{skill_nwp:.4f}" if skill_nwp is not None else "N/A"
        logger.info(f"Metrics - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, Skill_NWP: {skill_nwp_str}")
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return identifier_value, None

    # Clean up
    del model, checkpoint, X_train, y_train, X_test, y_test, dfs
    gc.collect()

    return identifier_value, metrics


def evaluate_global_model_daily_r2(model_path, models_dir, local_configs_dirs):
    """
    Evaluate a global model on multiple local datasets with per-forecast R² values.

    Returns:
        DataFrame with forecast dates as rows, station IDs as columns, R² values in cells
    """
    model_name = model_path.name
    global_model_id = extract_model_name(model_name)

    logger.info("="*80)
    logger.info(f"Evaluating GLOBAL model {global_model_id} for DAILY R² on LOCAL datasets")
    logger.info("="*80)

    # Get config path
    config_path, is_nostatic = get_config_path_from_model_name(model_name)
    if config_path is None or not config_path.exists():
        logger.error(f"Config not found for global model {model_name}")
        return None

    logger.info(f"Global model config: {config_path}")

    # Load global config
    try:
        global_config = load_config(config_path, is_nostatic)
    except Exception as e:
        logger.error(f"Error loading global config: {e}")
        return None

    global_config['model']['name'] = 'tft'

    # Load hyperparameters
    parts = model_name.replace('.pt', '').split('_')
    output_dim = int([p for p in parts if 'out-' in p][0].replace('out-', ''))
    freq_part = [p for p in parts if 'freq-' in p][0]
    freq = freq_part.replace('freq-', '')
    freq_idx = parts.index(freq_part)
    study_name_suffix = '_'.join(parts[freq_idx+1:])
    study_name = f'cl_m-{global_config["model"]["name"]}_out-{output_dim}_freq-{freq}_{study_name_suffix}'

    study = None
    try:
        if global_config['model']['lookup_hpo']:
            study = hpo.load_study(global_config['hpo']['studies_path'], study_name)
            logger.info(f"Study loaded: {study.study_name}")
    except Exception as e:
        logger.warning(f"Could not load study: {e}")

    hyperparameters = hpo.get_hyperparameters(config=global_config, study=study)
    features = preprocessing.get_features(config=global_config)

    # Find local station configs
    local_config_files = []
    for config_dir in local_configs_dirs:
        if config_dir.exists():
            local_config_files.extend(list(config_dir.glob('config_wind_*.yaml')))
    local_config_files = sorted(local_config_files)
    logger.info(f"Found {len(local_config_files)} local station configs")

    # Load global model
    try:
        logger.info("Loading global model...")
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                cleaned_state_dict[key.replace('_orig_mod.', '')] = value
            else:
                cleaned_state_dict[key] = value

        logger.info("Global model loaded")
    except Exception as e:
        logger.error(f"Error loading global model: {e}")
        return None

    # Try to load saved scaler, fall back to fitting
    global_scaler_x, global_scaler_y = load_scaler_for_model(model_path)
    target_col = global_config['data']['target_col']

    if global_scaler_x is None:
        logger.info("="*80)
        logger.info("Fitting GLOBAL scaler on all stations' training data...")
        logger.info("="*80)

        global_scaler_x = StandardScaler()
        test_start = pd.Timestamp(global_config['data']['test_start'], tz='UTC')
        test_end = pd.Timestamp(global_config['data']['test_end'], tz='UTC')

        if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
            test_end = test_end.replace(hour=23, minute=0, second=0)

        for i, local_config_path in enumerate(local_config_files):
            station_id = local_config_path.stem.replace('config_wind_', '')

            try:
                with open(local_config_path, 'r') as f:
                    local_config = yaml.safe_load(f)

                data_dir = local_config['data']['path']
                dfs = preprocessing.get_data(
                    data_dir=data_dir,
                    config=local_config,
                    freq=global_config['data']['freq'],
                    features=features
                )

                if len(dfs) == 0:
                    continue

                for key, df in dfs.items():
                    df_temp = df.copy()
                    if target_col in df_temp.columns:
                        df_temp.drop(target_col, axis=1, inplace=True)

                    t_0 = 0 if global_config['eval']['eval_on_all_test_data'] else global_config['eval']['t_0']
                    df_train, _ = preprocessing.split_data(
                        data=df_temp,
                        train_frac=global_config['data']['train_frac'],
                        test_start=test_start,
                        test_end=test_end,
                        t_0=t_0
                    )
                    global_scaler_x.partial_fit(df_train)
                    del df_temp, df_train

                del dfs
                gc.collect()

                if (i + 1) % 10 == 0:
                    logger.info(f"  Fitted scaler on {i+1}/{len(local_config_files)} stations")
            except Exception as e:
                logger.warning(f"Could not fit scaler on station {station_id}: {e}")
                continue

        logger.info(f"Global scaler fitted on {len(local_config_files)} stations")
        logger.info("="*80)

    if global_scaler_y is None and target_col != 'power':
        logger.warning("No saved scaler_y — fitting global scaler_y from training data as fallback.")
        global_scaler_y = StandardScaler()
        test_start = pd.Timestamp(global_config['data']['test_start'], tz='UTC')
        test_end = pd.Timestamp(global_config['data']['test_end'], tz='UTC')
        if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
            test_end = test_end.replace(hour=23, minute=0, second=0)
        t_0 = 0 if global_config['eval']['eval_on_all_test_data'] else global_config['eval']['t_0']
        for local_config_path in local_config_files:
            try:
                with open(local_config_path, 'r') as f:
                    local_config = yaml.safe_load(f)
                dfs = preprocessing.get_data(data_dir=local_config['data']['path'], config=local_config,
                                             freq=global_config['data']['freq'], features=features)
                for key, df in dfs.items():
                    if target_col not in df.columns:
                        continue
                    df_train, _ = preprocessing.split_data(data=df[[target_col]],
                                                           train_frac=global_config['data']['train_frac'],
                                                           test_start=test_start, test_end=test_end, t_0=t_0)
                    global_scaler_y.partial_fit(df_train)
                gc.collect()
            except Exception:
                continue

    # Dictionary to collect R² values per forecast date and station
    # Structure: {forecast_date: {station_id: r2_value}}
    daily_r2_data = {}

    # Evaluate on each local dataset
    for i, local_config_path in enumerate(local_config_files):
        station_id = local_config_path.stem.replace('config_wind_', '')
        logger.info(f"\n[{i+1}/{len(local_config_files)}] Evaluating on station {station_id}")

        try:
            with open(local_config_path, 'r') as f:
                local_config = yaml.safe_load(f)

            data_dir = local_config['data']['path']
            dfs = preprocessing.get_data(
                data_dir=data_dir,
                config=local_config,
                freq=global_config['data']['freq'],
                features=features
            )

            if len(dfs) == 0:
                logger.warning(f"No data found for station {station_id}")
                continue

            # Prepare data with GLOBAL scaler
            data_generator = tools.create_data_generator(dfs, global_config, features, scaler_x=global_scaler_x,
                                                         scaler_y=global_scaler_y if target_col != 'power' else None)
            X_train, y_train, X_test, y_test, test_data = tools.combine_datasets_efficiently(data_generator)

            # Extract feature dimensions
            feature_dims = {}
            if isinstance(X_train, dict):
                feature_dims['observed_dim'] = X_train['observed'].shape[-1]
                feature_dims['known_dim'] = X_train['known'].shape[-1] if 'known' in X_train else 0
                feature_dims['static_dim'] = X_train['static'].shape[-1] if 'static' in X_train else 0
            else:
                feature_dims['observed_dim'] = X_train.shape[-1]
                feature_dims['known_dim'] = 0
                feature_dims['static_dim'] = 0

            # Create station-specific config
            import copy
            station_config = copy.deepcopy(global_config)
            station_config['model']['feature_dim'] = feature_dims

            # Create model and load weights
            hyperparameters_copy = hyperparameters.copy()
            hyperparameters_copy['model_type'] = 'tft'
            hyperparameters_copy.update(feature_dims)

            model = get_model(station_config, hyperparameters_copy)
            model.load_state_dict(cleaned_state_dict)
            model.eval()

            # Run inference
            test_observed = X_test['observed'] if isinstance(X_test, dict) else X_test
            test_known = X_test.get('known', None) if isinstance(X_test, dict) else None
            test_static = X_test.get('static', None) if isinstance(X_test, dict) else None

            if not isinstance(test_observed, torch.Tensor):
                test_observed = torch.FloatTensor(test_observed)
            if test_known is not None and not isinstance(test_known, torch.Tensor):
                test_known = torch.FloatTensor(test_known)
            if test_static is not None and not isinstance(test_static, torch.Tensor):
                test_static = torch.FloatTensor(test_static)

            with torch.no_grad():
                if test_static is not None:
                    predictions = model(test_observed, test_known, test_static)
                else:
                    predictions = model(test_observed, test_known)

            predictions_np = predictions.cpu().numpy()
            y_test_np = y_test if isinstance(y_test, np.ndarray) else y_test.cpu().numpy()

            # Reshape if needed
            predictions_np = predictions_np.reshape(-1, y_test_np.shape[-1])

            # Inverse-transform if target was scaled
            if global_scaler_y is not None:
                predictions_np = global_scaler_y.inverse_transform(predictions_np)
                y_test_np = global_scaler_y.inverse_transform(y_test_np.reshape(-1, predictions_np.shape[-1]))

            # Clip negative values to 0 (only meaningful for power)
            if target_col == 'power':
                predictions_np[predictions_np < 0] = 0

            # Extract forecast dates from test_data
            # test_data is a dict with key -> (X_test, y_test, index_test, scaler_y)
            forecast_dates = None
            for key, data_tuple in test_data.items():
                index_test = data_tuple[2]
                if forecast_dates is None:
                    forecast_dates = index_test
                else:
                    forecast_dates = np.concatenate([forecast_dates, index_test])

            # Calculate R² for each forecast run
            num_forecasts = predictions_np.shape[0]
            for j in range(num_forecasts):
                y_true_single = y_test_np[j, :]  # Shape: (48,)
                y_pred_single = predictions_np[j, :]  # Shape: (48,)

                # Calculate R² for this single forecast
                r2_single = r2_score(y_true_single, y_pred_single)

                # Get the forecast date
                forecast_date = forecast_dates[j]

                # Store in dictionary
                if forecast_date not in daily_r2_data:
                    daily_r2_data[forecast_date] = {}
                daily_r2_data[forecast_date][station_id] = r2_single

            logger.info(f"Collected {num_forecasts} forecast runs for station {station_id}")

            # Clean up
            del model, X_train, y_train, X_test, y_test, dfs
            gc.collect()

        except Exception as e:
            logger.error(f"Error evaluating station {station_id}: {e}")
            continue

    # Convert to DataFrame
    logger.info("="*80)
    logger.info("Converting results to DataFrame...")
    logger.info("="*80)

    if not daily_r2_data:
        logger.warning("No data collected!")
        return None

    # Create DataFrame from nested dictionary
    df_result = pd.DataFrame.from_dict(daily_r2_data, orient='index')
    df_result.index.name = 'forecast_date'
    df_result = df_result.sort_index()

    logger.info(f"Result DataFrame shape: {df_result.shape}")
    logger.info(f"  Rows (forecast dates): {len(df_result)}")
    logger.info(f"  Columns (stations): {len(df_result.columns)}")

    return df_result


def evaluate_local_models_daily_r2(models_dir, local_configs_dirs):
    """
    Evaluate local station-specific models with per-forecast R² values.

    Returns:
        DataFrame with forecast dates as rows, station IDs as columns, R² values in cells
    """
    logger.info("="*80)
    logger.info("Evaluating LOCAL station-specific models for DAILY R²")
    logger.info("="*80)

    # Find local station configs
    local_config_files = []
    for config_dir in local_configs_dirs:
        if config_dir.exists():
            local_config_files.extend(list(config_dir.glob('config_wind_*.yaml')))
    local_config_files = sorted(local_config_files)
    logger.info(f"Found {len(local_config_files)} local station configs")

    # Dictionary to collect R² values per forecast date and station
    # Structure: {forecast_date: {station_id: r2_value}}
    daily_r2_data = {}

    # Evaluate each station's local model
    for i, local_config_path in enumerate(local_config_files):
        station_id = local_config_path.stem.replace('config_wind_', '')
        logger.info(f"\n[{i+1}/{len(local_config_files)}] Evaluating station {station_id}")

        # Find the station-specific model
        model_files = list(models_dir.glob(f'*_{station_id}.pt'))
        if len(model_files) == 0:
            logger.warning(f"No model found for station {station_id}, skipping")
            continue

        # Use the first matching model (should be only one)
        model_path = model_files[0]
        logger.info(f"  Model: {model_path.name}")

        try:
            # Load config
            with open(local_config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Check if nostatic model
            is_nostatic = 'nostatic' in model_path.name
            if is_nostatic:
                config['params']['static_features'] = []

            config['model']['name'] = 'tft'

            # Get hyperparameters and features
            features = preprocessing.get_features(config=config)

            # Load data
            data_dir = config['data']['path']
            dfs = preprocessing.get_data(
                data_dir=data_dir,
                config=config,
                freq=config['data']['freq'],
                features=features
            )

            if len(dfs) == 0:
                logger.warning(f"No data found for station {station_id}")
                continue

            # Try to load saved scaler, fall back to fitting
            global_scaler_x, global_scaler_y = load_scaler_for_model(model_path)
            target_col = config['data']['target_col']

            if global_scaler_x is None:
                global_scaler_x = StandardScaler()
                test_start = pd.Timestamp(config['data']['test_start'], tz='UTC')
                test_end = pd.Timestamp(config['data']['test_end'], tz='UTC')

                if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
                    test_end = test_end.replace(hour=23, minute=0, second=0)

                for key, df in dfs.items():
                    df_temp = df.copy()
                    if target_col in df_temp.columns:
                        df_temp.drop(target_col, axis=1, inplace=True)

                    t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
                    df_train, _ = preprocessing.split_data(
                        data=df_temp,
                        train_frac=config['data']['train_frac'],
                        test_start=test_start,
                        test_end=test_end,
                        t_0=t_0
                    )
                    global_scaler_x.partial_fit(df_train)
                    del df_temp, df_train

            if global_scaler_y is None and target_col != 'power':
                global_scaler_y = StandardScaler()
                test_start = pd.Timestamp(config['data']['test_start'], tz='UTC')
                test_end = pd.Timestamp(config['data']['test_end'], tz='UTC')
                if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
                    test_end = test_end.replace(hour=23, minute=0, second=0)
                t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
                for key, df in dfs.items():
                    if target_col not in df.columns:
                        continue
                    df_train, _ = preprocessing.split_data(data=df[[target_col]],
                                                           train_frac=config['data']['train_frac'],
                                                           test_start=test_start, test_end=test_end, t_0=t_0)
                    global_scaler_y.partial_fit(df_train)

            # Prepare data
            data_generator = tools.create_data_generator(dfs, config, features, scaler_x=global_scaler_x,
                                                         scaler_y=global_scaler_y if target_col != 'power' else None)
            X_train, y_train, X_test, y_test, test_data = tools.combine_datasets_efficiently(data_generator)

            # Extract feature dimensions
            feature_dims = {}
            if isinstance(X_train, dict):
                feature_dims['observed_dim'] = X_train['observed'].shape[-1]
                feature_dims['known_dim'] = X_train['known'].shape[-1] if 'known' in X_train else 0
                feature_dims['static_dim'] = X_train['static'].shape[-1] if 'static' in X_train else 0
            else:
                feature_dims['observed_dim'] = X_train.shape[-1]
                feature_dims['known_dim'] = 0
                feature_dims['static_dim'] = 0

            config['model']['feature_dim'] = feature_dims

            # Extract forecast dates
            forecast_dates = None
            for key, data_tuple in test_data.items():
                index_test = data_tuple[2]
                if forecast_dates is None:
                    forecast_dates = index_test
                else:
                    forecast_dates = np.concatenate([forecast_dates, index_test])

            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    cleaned_state_dict[key.replace('_orig_mod.', '')] = value
                else:
                    cleaned_state_dict[key] = value

            # Get hyperparameters from model filename
            model_name = model_path.name
            parts = model_name.replace('.pt', '').split('_')
            output_dim = int([p for p in parts if 'out-' in p][0].replace('out-', ''))
            freq_part = [p for p in parts if 'freq-' in p][0]
            freq = freq_part.replace('freq-', '')
            freq_idx = parts.index(freq_part)
            study_name_suffix = '_'.join(parts[freq_idx+1:])
            study_name = f'cl_m-{config["model"]["name"]}_out-{output_dim}_freq-{freq}_{study_name_suffix}'

            study = None
            try:
                if config['model']['lookup_hpo']:
                    study = hpo.load_study(config['hpo']['studies_path'], study_name)
            except:
                pass

            hyperparameters = hpo.get_hyperparameters(config=config, study=study)
            hyperparameters['model_type'] = 'tft'
            hyperparameters.update(feature_dims)

            model = get_model(config, hyperparameters)
            model.load_state_dict(cleaned_state_dict)
            model.eval()

            # Run inference
            test_observed = X_test['observed'] if isinstance(X_test, dict) else X_test
            test_known = X_test.get('known', None) if isinstance(X_test, dict) else None
            test_static = X_test.get('static', None) if isinstance(X_test, dict) else None

            if not isinstance(test_observed, torch.Tensor):
                test_observed = torch.FloatTensor(test_observed)
            if test_known is not None and not isinstance(test_known, torch.Tensor):
                test_known = torch.FloatTensor(test_known)
            if test_static is not None and not isinstance(test_static, torch.Tensor):
                test_static = torch.FloatTensor(test_static)

            with torch.no_grad():
                if test_static is not None:
                    predictions = model(test_observed, test_known, test_static)
                else:
                    predictions = model(test_observed, test_known)

            predictions_np = predictions.cpu().numpy()
            y_test_np = y_test if isinstance(y_test, np.ndarray) else y_test.cpu().numpy()

            # Reshape
            predictions_np = predictions_np.reshape(-1, y_test_np.shape[-1])

            # Inverse-transform if target was scaled
            if global_scaler_y is not None:
                predictions_np = global_scaler_y.inverse_transform(predictions_np)
                y_test_np = global_scaler_y.inverse_transform(y_test_np.reshape(-1, predictions_np.shape[-1]))

            if target_col == 'power':
                predictions_np[predictions_np < 0] = 0

            # Calculate R² for each forecast run
            num_forecasts = predictions_np.shape[0]
            for j in range(num_forecasts):
                y_true_single = y_test_np[j, :]
                y_pred_single = predictions_np[j, :]

                # Calculate R² for this single forecast
                r2_single = r2_score(y_true_single, y_pred_single)

                # Get the forecast date
                forecast_date = forecast_dates[j]

                # Store in dictionary
                if forecast_date not in daily_r2_data:
                    daily_r2_data[forecast_date] = {}
                daily_r2_data[forecast_date][station_id] = r2_single

            logger.info(f"  Collected {num_forecasts} forecast runs")

            # Clean up
            del model, X_train, y_train, X_test, y_test, dfs
            gc.collect()

        except Exception as e:
            logger.error(f"Error evaluating station {station_id}: {e}")
            continue

    # Convert to DataFrame
    logger.info("="*80)
    logger.info("Converting results to DataFrame...")
    logger.info("="*80)

    if not daily_r2_data:
        logger.warning("No data collected!")
        return None

    # Create DataFrame from nested dictionary
    df_result = pd.DataFrame.from_dict(daily_r2_data, orient='index')
    df_result.index.name = 'forecast_date'
    df_result = df_result.sort_index()

    logger.info(f"Result DataFrame shape: {df_result.shape}")
    logger.info(f"  Rows (forecast dates): {len(df_result)}")
    logger.info(f"  Columns (stations): {len(df_result.columns)}")

    return df_result


def evaluate_global_model_on_local_datasets(model_path, models_dir, local_configs_dirs):
    """
    Evaluate a global model on multiple local datasets.

    Yields:
        Tuples of (station_id, metrics) for each local dataset
    """
    model_name = model_path.name
    global_model_id = extract_model_name(model_name)

    logger.info("="*80)
    logger.info(f"Evaluating GLOBAL model {global_model_id} on LOCAL datasets")
    logger.info("="*80)

    # Get config path
    config_path, is_nostatic = get_config_path_from_model_name(model_name)
    if config_path is None or not config_path.exists():
        logger.error(f"Config not found for global model {model_name}")
        return

    logger.info(f"Global model config: {config_path}")

    # Load global config
    try:
        global_config = load_config(config_path, is_nostatic)
    except Exception as e:
        logger.error(f"Error loading global config: {e}")
        return

    global_config['model']['name'] = 'tft'

    # Load hyperparameters
    parts = model_name.replace('.pt', '').split('_')
    output_dim = int([p for p in parts if 'out-' in p][0].replace('out-', ''))
    freq_part = [p for p in parts if 'freq-' in p][0]
    freq = freq_part.replace('freq-', '')
    freq_idx = parts.index(freq_part)
    study_name_suffix = '_'.join(parts[freq_idx+1:])
    study_name = f'cl_m-{global_config["model"]["name"]}_out-{output_dim}_freq-{freq}_{study_name_suffix}'

    study = None
    try:
        if global_config['model']['lookup_hpo']:
            study = hpo.load_study(global_config['hpo']['studies_path'], study_name)
            logger.info(f"Study loaded: {study.study_name}")
    except Exception as e:
        logger.warning(f"Could not load study: {e}")

    hyperparameters = hpo.get_hyperparameters(config=global_config, study=study)
    features = preprocessing.get_features(config=global_config)

    # Find local station configs
    local_config_files = []
    for config_dir in local_configs_dirs:
        if config_dir.exists():
            local_config_files.extend(list(config_dir.glob('config_wind_*.yaml')))
    local_config_files = sorted(local_config_files)
    logger.info(f"Found {len(local_config_files)} local station configs")

    # Load global model
    try:
        logger.info("Loading global model...")
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                cleaned_state_dict[key.replace('_orig_mod.', '')] = value
            else:
                cleaned_state_dict[key] = value

        logger.info("Global model loaded")
    except Exception as e:
        logger.error(f"Error loading global model: {e}")
        return

    # Try to load saved scaler, fall back to fitting
    global_scaler_x, global_scaler_y = load_scaler_for_model(model_path)
    target_col = global_config['data']['target_col']

    if global_scaler_x is None:
        logger.info("="*80)
        logger.info("Fitting GLOBAL scaler on all stations' training data...")
        logger.info("="*80)

        global_scaler_x = StandardScaler()
        test_start = pd.Timestamp(global_config['data']['test_start'], tz='UTC')
        test_end = pd.Timestamp(global_config['data']['test_end'], tz='UTC')

        if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
            test_end = test_end.replace(hour=23, minute=0, second=0)

        for i, local_config_path in enumerate(local_config_files):
            station_id = local_config_path.stem.replace('config_wind_', '')

            try:
                with open(local_config_path, 'r') as f:
                    local_config = yaml.safe_load(f)

                data_dir = local_config['data']['path']
                dfs = preprocessing.get_data(
                    data_dir=data_dir,
                    config=local_config,
                    freq=global_config['data']['freq'],
                    features=features
                )

                if len(dfs) == 0:
                    continue

                for key, df in dfs.items():
                    df_temp = df.copy()
                    if target_col in df_temp.columns:
                        df_temp.drop(target_col, axis=1, inplace=True)

                    t_0 = 0 if global_config['eval']['eval_on_all_test_data'] else global_config['eval']['t_0']
                    df_train, _ = preprocessing.split_data(
                        data=df_temp,
                        train_frac=global_config['data']['train_frac'],
                        test_start=test_start,
                        test_end=test_end,
                        t_0=t_0
                    )
                    global_scaler_x.partial_fit(df_train)
                    del df_temp, df_train

                del dfs
                gc.collect()

                if (i + 1) % 10 == 0:
                    logger.info(f"  Fitted scaler on {i+1}/{len(local_config_files)} stations")
            except Exception as e:
                logger.warning(f"Could not fit scaler on station {station_id}: {e}")
                continue

        logger.info(f"Global scaler fitted on {len(local_config_files)} stations")
        logger.info("="*80)

    if global_scaler_y is None and target_col != 'power':
        logger.warning("No saved scaler_y — fitting global scaler_y from training data as fallback.")
        global_scaler_y = StandardScaler()
        test_start = pd.Timestamp(global_config['data']['test_start'], tz='UTC')
        test_end = pd.Timestamp(global_config['data']['test_end'], tz='UTC')
        if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
            test_end = test_end.replace(hour=23, minute=0, second=0)
        t_0 = 0 if global_config['eval']['eval_on_all_test_data'] else global_config['eval']['t_0']
        for local_config_path in local_config_files:
            try:
                with open(local_config_path, 'r') as f:
                    local_config = yaml.safe_load(f)
                dfs = preprocessing.get_data(data_dir=local_config['data']['path'], config=local_config,
                                             freq=global_config['data']['freq'], features=features)
                for key, df in dfs.items():
                    if target_col not in df.columns:
                        continue
                    df_train, _ = preprocessing.split_data(data=df[[target_col]],
                                                           train_frac=global_config['data']['train_frac'],
                                                           test_start=test_start, test_end=test_end, t_0=t_0)
                    global_scaler_y.partial_fit(df_train)
                gc.collect()
            except Exception:
                continue

    # Evaluate on each local dataset
    for i, local_config_path in enumerate(local_config_files):
        station_id = local_config_path.stem.replace('config_wind_', '')
        logger.info(f"\n[{i+1}/{len(local_config_files)}] Evaluating on station {station_id}")

        try:
            with open(local_config_path, 'r') as f:
                local_config = yaml.safe_load(f)

            data_dir = local_config['data']['path']
            dfs = preprocessing.get_data(
                data_dir=data_dir,
                config=local_config,
                freq=global_config['data']['freq'],
                features=features
            )

            if len(dfs) == 0:
                logger.warning(f"No data found for station {station_id}")
                continue

            # Prepare data with GLOBAL scaler
            data_generator = tools.create_data_generator(dfs, global_config, features, scaler_x=global_scaler_x,
                                                         scaler_y=global_scaler_y if target_col != 'power' else None)
            X_train, y_train, X_test, y_test, test_data = tools.combine_datasets_efficiently(data_generator)

            # Extract feature dimensions
            feature_dims = {}
            if isinstance(X_train, dict):
                feature_dims['observed_dim'] = X_train['observed'].shape[-1]
                feature_dims['known_dim'] = X_train['known'].shape[-1] if 'known' in X_train else 0
                feature_dims['static_dim'] = X_train['static'].shape[-1] if 'static' in X_train else 0
            else:
                feature_dims['observed_dim'] = X_train.shape[-1]
                feature_dims['known_dim'] = 0
                feature_dims['static_dim'] = 0

            # Create station-specific config
            import copy
            station_config = copy.deepcopy(global_config)
            station_config['model']['feature_dim'] = feature_dims

            # Create model and load weights
            hyperparameters_copy = hyperparameters.copy()
            hyperparameters_copy['model_type'] = 'tft'
            hyperparameters_copy.update(feature_dims)

            model = get_model(station_config, hyperparameters_copy)
            model.load_state_dict(cleaned_state_dict)
            model.eval()

            # Run inference
            test_observed = X_test['observed'] if isinstance(X_test, dict) else X_test
            test_known = X_test.get('known', None) if isinstance(X_test, dict) else None
            test_static = X_test.get('static', None) if isinstance(X_test, dict) else None

            if not isinstance(test_observed, torch.Tensor):
                test_observed = torch.FloatTensor(test_observed)
            if test_known is not None and not isinstance(test_known, torch.Tensor):
                test_known = torch.FloatTensor(test_known)
            if test_static is not None and not isinstance(test_static, torch.Tensor):
                test_static = torch.FloatTensor(test_static)

            with torch.no_grad():
                if test_static is not None:
                    predictions = model(test_observed, test_known, test_static)
                else:
                    predictions = model(test_observed, test_known)

            predictions_np = predictions.cpu().numpy()
            y_test_np = y_test if isinstance(y_test, np.ndarray) else y_test.cpu().numpy()

            # Reshape if needed (to match train_cl.py)
            predictions_np = predictions_np.reshape(-1, y_test_np.shape[-1])

            # Inverse-transform if target was scaled
            if global_scaler_y is not None:
                predictions_np = global_scaler_y.inverse_transform(predictions_np)
                y_test_np = global_scaler_y.inverse_transform(y_test_np.reshape(-1, predictions_np.shape[-1]))

            # Clip negative values to 0 (only meaningful for power)
            if target_col == 'power':
                predictions_np[predictions_np < 0] = 0

            # Calculate metrics
            r2, rmse, mae = calculate_metrics(y_test_np, predictions_np)

            try:
                skill_nwp = compute_skill_nwp(dfs, test_data, y_test_np, rmse, global_config['data'], target_col)
            except Exception as e_nwp:
                logger.warning(f"Could not compute Skill_NWP: {e_nwp}")
                skill_nwp = None

            metrics = {
                'station_id': station_id,
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'skill_nwp': float(skill_nwp) if skill_nwp is not None else None
            }

            skill_nwp_str = f"{skill_nwp:.4f}" if skill_nwp is not None else "N/A"
            logger.info(f"Metrics - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, Skill_NWP: {skill_nwp_str}")

            yield (station_id, metrics)

            # Clean up
            del model, X_train, y_train, X_test, y_test, dfs
            gc.collect()

        except Exception as e:
            logger.error(f"Error evaluating station {station_id}: {e}")
            continue


def evaluate_model_chronos(config, chronos_pipeline, station_id, scaler_x=None):
    """
    Evaluate a Chronos-2 pipeline on test data for a single station.

    config['data']['files'] must already be set to [station_id].
    Returns (station_id, metrics_dict) or (station_id, None) on failure.
    """
    logger.info("="*80)
    logger.info(f"Evaluating Chronos-2 | station: {station_id}")
    logger.info("="*80)

    try:
        features = preprocessing.get_features(config=config)
        data_dir = config['data']['path']
        dfs = preprocessing.get_data(
            data_dir=data_dir,
            config=config,
            freq=config['data']['freq'],
            features=features,
        )
        logger.info(f"Loaded {len(dfs)} dataframe(s)")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return station_id, None

    try:
        if scaler_x is not None:
            global_scaler_x = scaler_x
        else:
            global_scaler_x = StandardScaler()
            target_col = config['data']['target_col']
            test_start = pd.Timestamp(config['data']['test_start'], tz='UTC')
            test_end = pd.Timestamp(config['data']['test_end'], tz='UTC')
            if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
                test_end = test_end.replace(hour=23, minute=0, second=0)

            t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
            for key, df in dfs.items():
                df_temp = df.copy()
                if target_col in df_temp.columns:
                    df_temp.drop(target_col, axis=1, inplace=True)
                df_train, _ = preprocessing.split_data(
                    data=df_temp,
                    train_frac=config['data']['train_frac'],
                    test_start=test_start,
                    test_end=test_end,
                    t_0=t_0,
                )
                global_scaler_x.partial_fit(df_train)
                del df_temp, df_train
                gc.collect()
    except Exception as e:
        logger.error(f"Error fitting scaler: {e}")
        return station_id, None

    try:
        data_generator = tools.create_data_generator(dfs, config, features, scaler_x=global_scaler_x)
        X_train, y_train, X_test, y_test, _ = tools.combine_datasets_efficiently(data_generator)
        gc.collect()
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return station_id, None

    try:
        known_cols = config['params'].get('known_features', [])
        scaler_y = None
        if isinstance(X_test, dict):
            scaler_y = X_test.get('scalers', {}).get('y', None) if 'scalers' in X_test else None
        y_true, y_pred = tools.get_y_chronos2(
            X_test=X_test,
            y_test=y_test,
            chronos_pipeline=chronos_pipeline,
            config=config,
            known_cols=known_cols,
            scaler_y=scaler_y,
        )
        logger.info("Chronos-2 inference complete")
    except Exception as e:
        logger.error(f"Error during Chronos-2 inference: {e}")
        return station_id, None

    try:
        r2, rmse, mae = calculate_metrics(y_true, y_pred)
        metrics = {
            'station_id': station_id,
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
        }
        logger.info(f"Metrics — R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return station_id, None

    del X_train, y_train, X_test, y_test, dfs
    gc.collect()
    return station_id, metrics


def main():
    """Main function to evaluate all models."""
    parser = argparse.ArgumentParser(description='Evaluate TFT models on test data')
    parser.add_argument('-g', '--global', dest='global_models', action='store_true',
                        help='Evaluate global models instead of station-specific models')
    parser.add_argument('-e', '--evallocal', dest='eval_local', action='store_true',
                        help='Evaluate global models on local datasets')
    parser.add_argument('--evaldaily', dest='eval_daily', action='store_true',
                        help='Evaluate global models on local datasets with daily R² per station (CSV format)')
    parser.add_argument('--localperformance', dest='local_performance', action='store_true',
                        help='Evaluate local station-specific models with daily R² per station (CSV format)')
    parser.add_argument('--model', dest='model', default='tft',
                        help='Model type to evaluate: "tft" (default) or "chronos"')
    parser.add_argument('--config', dest='config_path', default=None,
                        help='Path to config YAML (required for --model chronos)')
    parser.add_argument('--zero-shot', dest='zero_shot', action='store_true',
                        help='For Chronos: load from HuggingFace instead of local fine-tuned model')
    parser.add_argument('--model-name', dest='model_name', default=None,
                        help='Filter: only evaluate the global model whose filename contains this string')
    args = parser.parse_args()

    # Setup paths
    models_dir = Path('models')
    local_configs_dirs = [Path('configs/wind_50'), Path('configs/wind_100ex50')]
    output_dir = Path('data/test_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── CHRONOS MODE ──────────────────────────────────────────────────────────
    if args.model == 'chronos':
        if args.config_path is None:
            logger.error("--config is required when --model chronos is used")
            return

        from chronos import BaseChronosPipeline

        config = load_config(Path(args.config_path))
        config['model']['name'] = 'chronos'
        chronos_cfg = config['model']['chronos']
        repo_id = chronos_cfg.get('repo_id', 'amazon/chronos-2')
        device_map = chronos_cfg.get('device_map', 'cuda')
        station_ids = config['data']['files']

        output_file = output_dir / 'test_results_chronos.csv'

        # Pre-load a single zero-shot pipeline (reused for all stations)
        zeroshot_pipeline = None
        if args.zero_shot:
            logger.info(f"Loading Chronos-2 zero-shot from {repo_id}")
            zeroshot_pipeline = BaseChronosPipeline.from_pretrained(repo_id, device_map=device_map)

        processed_ids: set = set()
        if output_file.exists():
            try:
                existing_df = pd.read_csv(output_file)
                processed_ids = set(existing_df['station_id'].unique())
                logger.info(f"Already processed {len(processed_ids)} stations")
            except Exception as e:
                logger.warning(f"Could not read existing CSV: {e}")

        for station_id in station_ids:
            if station_id in processed_ids:
                logger.info(f"Skipping {station_id} – already processed")
                continue

            # Determine which pipeline to use for this station
            local_model_dir = models_dir / f'chronos_{station_id}'
            if args.zero_shot:
                pipeline_to_use = zeroshot_pipeline
            elif local_model_dir.exists():
                logger.info(f"Loading fine-tuned Chronos-2 from {local_model_dir}")
                pipeline_to_use = BaseChronosPipeline.from_pretrained(
                    str(local_model_dir), device_map=device_map
                )
            else:
                logger.warning(
                    f"No fine-tuned model found at {local_model_dir}. "
                    f"Falling back to zero-shot from {repo_id}"
                )
                if zeroshot_pipeline is None:
                    zeroshot_pipeline = BaseChronosPipeline.from_pretrained(repo_id, device_map=device_map)
                pipeline_to_use = zeroshot_pipeline

            station_config = {**config, 'data': {**config['data'], 'files': [station_id]}}

            chronos_scaler_x, _ = load_scaler_for_model(local_model_dir.name) if local_model_dir.exists() else (None, None)
            station_id_val, metrics = evaluate_model_chronos(station_config, pipeline_to_use, station_id, scaler_x=chronos_scaler_x)

            if metrics:
                new_df = pd.DataFrame([metrics])
                if not output_file.exists():
                    new_df.to_csv(output_file, index=False, mode='w')
                else:
                    new_df.to_csv(output_file, index=False, mode='a', header=False)
                processed_ids.add(station_id_val)

            # Release fine-tuned pipeline from memory
            if not args.zero_shot and local_model_dir.exists():
                del pipeline_to_use
                gc.collect()

        logger.info("="*80)
        logger.info("CHRONOS EVALUATION COMPLETE")
        logger.info("="*80)
        if output_file.exists():
            try:
                final_df = pd.read_csv(output_file)
                logger.info(f"Results saved to {output_file}")
                logger.info(f"  Stations: {len(final_df)}")
                logger.info(f"  Mean R²:   {final_df['r2'].mean():.4f}")
                logger.info(f"  Mean RMSE: {final_df['rmse'].mean():.4f}")
                logger.info(f"  Mean MAE:  {final_df['mae'].mean():.4f}")
            except Exception as e:
                logger.error(f"Could not read final CSV: {e}")
        return
    # ── END CHRONOS MODE ──────────────────────────────────────────────────────

    # ── EXPLICIT CONFIG MODE (TFT with --config) ──────────────────────────────
    if args.config_path is not None and args.model != 'chronos':
        import copy
        config = load_config(Path(args.config_path))
        config['model']['name'] = 'tft'
        station_ids = config['data']['files']
        logger.info(f"Config: {args.config_path}")
        logger.info(f"Stations from config: {len(station_ids)}")

        # Find global models, optionally filtered by --model-name
        all_model_files = sorted([f for f in models_dir.glob('*.pt') if f.is_file()])
        global_model_files = []
        for f in all_model_files:
            model_id = extract_model_name(f.name)
            if model_id:
                parts = model_id.split('_')
                last_part = parts[-1] if parts[-1] != 'nostatic' else parts[-2]
                if not (last_part.isdigit() and len(last_part) == 5):
                    if args.model_name is None or args.model_name in f.name:
                        global_model_files.append(f)

        logger.info(f"Found {len(global_model_files)} global model(s) to evaluate")

        for model_path in global_model_files:
            global_model_id = extract_model_name(model_path.name)

            if args.eval_local:
                # Per-station evaluation: fit global scaler once, then one row per station
                output_file = output_dir / f'test_results_{global_model_id}.csv'
                logger.info(f"Per-station evaluation → {output_file}")

                processed_stations = set()
                if output_file.exists():
                    try:
                        existing_df = pd.read_csv(output_file)
                        processed_stations = set(existing_df['station_id'].unique())
                        logger.info(f"Already processed {len(processed_stations)} stations")
                    except Exception as e:
                        logger.warning(f"Could not read existing CSV: {e}")

                # Try to load saved scaler, fall back to fitting from data
                features = preprocessing.get_features(config=config)
                global_scaler_x, global_scaler_y = load_scaler_for_model(model_path)

                if global_scaler_x is None:
                    logger.info("Fitting global scaler on all stations from config...")
                    global_scaler_x = StandardScaler()
                    target_col = config['data']['target_col']
                    test_start = pd.Timestamp(config['data']['test_start'], tz='UTC')
                    test_end = pd.Timestamp(config['data']['test_end'], tz='UTC')
                    if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
                        test_end = test_end.replace(hour=23, minute=0, second=0)
                    t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']

                    for j, sid in enumerate(station_ids):
                        try:
                            station_config = copy.deepcopy(config)
                            station_config['data']['files'] = [sid]
                            dfs = preprocessing.get_data(
                                data_dir=config['data']['path'],
                                config=station_config,
                                freq=config['data']['freq'],
                                features=features
                            )
                            for key, df in dfs.items():
                                df_temp = df.copy()
                                if target_col in df_temp.columns:
                                    df_temp.drop(target_col, axis=1, inplace=True)
                                df_train, _ = preprocessing.split_data(
                                    data=df_temp,
                                    train_frac=config['data']['train_frac'],
                                    test_start=test_start,
                                    test_end=test_end,
                                    t_0=t_0
                                )
                                global_scaler_x.partial_fit(df_train)
                                del df_temp, df_train
                            del dfs
                            gc.collect()
                            if (j + 1) % 10 == 0:
                                logger.info(f"  Fitted scaler on {j+1}/{len(station_ids)} stations")
                        except Exception as e:
                            logger.warning(f"Could not fit scaler on station {sid}: {e}")

                    logger.info("Global scaler fitted.")

                # Evaluate each station with the shared scaler
                for sid in station_ids:
                    if sid in processed_stations:
                        logger.info(f"Skipping {sid} – already processed")
                        continue
                    station_config = copy.deepcopy(config)
                    station_config['data']['files'] = [sid]
                    _, metrics = evaluate_model(
                        model_path, models_dir,
                        identifier_key='station_id',
                        identifier_value=sid,
                        config=station_config,
                        scaler_x=global_scaler_x,
                        scaler_y=global_scaler_y
                    )
                    if metrics:
                        new_df = pd.DataFrame([metrics])
                        if not output_file.exists():
                            new_df.to_csv(output_file, index=False, mode='w')
                        else:
                            new_df.to_csv(output_file, index=False, mode='a', header=False)
                        processed_stations.add(sid)

            else:
                # Aggregate evaluation: single row for all stations combined
                output_file = output_dir / f'test_results_{global_model_id}_agg.csv'
                logger.info(f"Aggregate evaluation → {output_file}")

                if output_file.exists():
                    logger.info("Output file already exists, skipping.")
                    continue

                _, metrics = evaluate_model(
                    model_path, models_dir,
                    identifier_key='model',
                    identifier_value=global_model_id,
                    config=config
                )
                if metrics:
                    pd.DataFrame([metrics]).to_csv(output_file, index=False)
                    logger.info(f"Saved aggregate result to {output_file}")

        logger.info("="*80)
        logger.info("CONFIG-BASED EVALUATION COMPLETE")
        logger.info("="*80)
        return
    # ── END EXPLICIT CONFIG MODE ───────────────────────────────────────────────

    # MODE 0: Evaluate local station-specific models - DAILY R² CSV
    if args.local_performance:
        logger.info("="*80)
        logger.info("MODE: Evaluating LOCAL station-specific models - DAILY R² CSV")
        logger.info("="*80)

        output_file = output_dir / 'daily_results_local.csv'

        # Skip if already exists
        if output_file.exists():
            logger.info(f"Output file already exists, skipping: {output_file}")
        else:
            try:
                df_daily_r2 = evaluate_local_models_daily_r2(
                    models_dir, local_configs_dirs
                )

                if df_daily_r2 is not None:
                    df_daily_r2.to_csv(output_file)
                    logger.info(f"Saved results to: {output_file}")
                    logger.info(f"  Shape: {df_daily_r2.shape}")
                    logger.info(f"  Mean R² across all forecasts/stations: {df_daily_r2.mean().mean():.4f}")
                else:
                    logger.warning("No results collected")

            except Exception as e:
                logger.error(f"Error processing local models: {e}")

        logger.info("\n" + "="*80)
        logger.info("LOCALPERFORMANCE MODE COMPLETE")
        logger.info("="*80)
        return

    # MODE 1: Evaluate global models on local datasets - DAILY R² CSV
    if args.eval_daily:
        logger.info("="*80)
        logger.info("MODE: Evaluating GLOBAL models on LOCAL datasets - DAILY R² CSV")
        logger.info("="*80)

        # Find global models
        all_model_files = sorted([f for f in models_dir.glob('*.pt') if f.is_file()])
        global_model_files = []
        for f in all_model_files:
            model_id = extract_model_name(f.name)
            if model_id:
                parts = model_id.split('_')
                last_part = parts[-1] if parts[-1] != 'nostatic' else parts[-2]
                if not (last_part.isdigit() and len(last_part) == 5):
                    if args.model_name is None or args.model_name in f.name:
                        global_model_files.append(f)

        logger.info(f"Found {len(global_model_files)} global models")

        # Evaluate each global model
        for i, model_path in enumerate(global_model_files):
            global_model_id = extract_model_name(model_path.name)
            output_file = output_dir / f'daily_results_{global_model_id}.csv'

            logger.info(f"\n[{i+1}/{len(global_model_files)}] Processing: {global_model_id}")
            logger.info(f"Output: {output_file}")

            # Skip if already exists
            if output_file.exists():
                logger.info(f"Output file already exists, skipping: {output_file}")
                continue

            # Determine which station configs to use based on model name
            # CRITICAL: The scaler should be fitted on the same stations as during training!
            if 'wind_100' in global_model_id:
                # 100-park model: use ALL 100 stations
                model_local_configs_dirs = [Path('configs/wind_50'), Path('configs/wind_100ex50')]
                logger.info(f"Model '{global_model_id}': Using 100 stations for scaler fitting")
            elif 'wind_50' in global_model_id:
                # 50-park model: use ONLY 50 stations (wind_50)
                model_local_configs_dirs = [Path('configs/wind_50')]
                logger.info(f"Model '{global_model_id}': Using 50 stations for scaler fitting")
            else:
                # Default: use all
                model_local_configs_dirs = local_configs_dirs
                logger.warning(f"Model '{global_model_id}': Could not determine station count from name, using default (all 100)")

            try:
                df_daily_r2 = evaluate_global_model_daily_r2(
                    model_path, models_dir, model_local_configs_dirs
                )

                if df_daily_r2 is not None:
                    df_daily_r2.to_csv(output_file)
                    logger.info(f"Saved results to: {output_file}")
                    logger.info(f"  Shape: {df_daily_r2.shape}")
                    logger.info(f"  Mean R² across all forecasts/stations: {df_daily_r2.mean().mean():.4f}")
                else:
                    logger.warning(f"No results for {global_model_id}")

            except Exception as e:
                logger.error(f"Error processing {global_model_id}: {e}")
                continue

        logger.info("\n" + "="*80)
        logger.info("EVALDAILY MODE COMPLETE")
        logger.info("="*80)
        return

    # MODE 1: Evaluate global models on local datasets
    if args.eval_local:
        logger.info("="*80)
        logger.info("MODE: Evaluating GLOBAL models on LOCAL datasets")
        logger.info("="*80)

        # Find global models
        all_model_files = sorted([f for f in models_dir.glob('*.pt') if f.is_file()])
        global_model_files = []
        for f in all_model_files:
            model_id = extract_model_name(f.name)
            if model_id:
                parts = model_id.split('_')
                last_part = parts[-1] if parts[-1] != 'nostatic' else parts[-2]
                if not (last_part.isdigit() and len(last_part) == 5):
                    if args.model_name is None or args.model_name in f.name:
                        global_model_files.append(f)

        logger.info(f"Found {len(global_model_files)} global models")

        # Evaluate each global model
        for i, model_path in enumerate(global_model_files):
            global_model_id = extract_model_name(model_path.name)
            output_file = output_dir / f'test_results_{global_model_id}.csv'

            logger.info(f"\n[{i+1}/{len(global_model_files)}] Processing: {global_model_id}")
            logger.info(f"Output: {output_file}")

            # Check already processed
            processed_stations = set()
            if output_file.exists():
                try:
                    existing_df = pd.read_csv(output_file)
                    processed_stations = set(existing_df['station_id'].unique())
                    logger.info(f"Already processed {len(processed_stations)} stations")
                except Exception as e:
                    logger.warning(f"Could not read existing CSV: {e}")

            try:
                all_station_results = evaluate_global_model_on_local_datasets(
                    model_path, models_dir, local_configs_dirs
                )

                new_count = 0
                for station_id, metrics in all_station_results:
                    if station_id in processed_stations:
                        logger.info(f"Skipping station {station_id} - already processed")
                        continue

                    if metrics:
                        new_df = pd.DataFrame([metrics])

                        if not output_file.exists():
                            new_df.to_csv(output_file, index=False, mode='w')
                        else:
                            new_df.to_csv(output_file, index=False, mode='a', header=False)

                        processed_stations.add(station_id)
                        new_count += 1

                logger.info(f"Completed {global_model_id}: processed {new_count} new stations")
            except Exception as e:
                logger.error(f"Error processing {global_model_id}: {e}")
                continue

        logger.info("\n" + "="*80)
        logger.info("EVALLOCAL MODE COMPLETE")
        logger.info("="*80)
        return

    # MODE 2: Global or station-specific models
    if args.global_models:
        output_file = output_dir / 'test_results_global.csv'
        identifier_key = 'model'
        logger.info("Evaluating GLOBAL models")
    else:
        output_file = output_dir / 'test_results_all.csv'
        identifier_key = 'station_id'
        logger.info("Evaluating STATION-SPECIFIC models")

    # Find models
    all_model_files = sorted([f for f in models_dir.glob('*.pt') if f.is_file()])

    if args.global_models:
        model_files = []
        for f in all_model_files:
            model_id = extract_model_name(f.name)
            if model_id:
                parts = model_id.split('_')
                last_part = parts[-1] if parts[-1] != 'nostatic' else parts[-2]
                if not (last_part.isdigit() and len(last_part) == 5):
                    if args.model_name is None or args.model_name in f.name:
                        model_files.append(f)
        logger.info(f"Found {len(model_files)} global models")
    else:
        model_files = []
        for f in all_model_files:
            station_id = extract_station_id(f.name)
            if station_id.isdigit() and len(station_id) == 5:
                model_files.append(f)
        logger.info(f"Found {len(model_files)} station-specific models")

    # Check already processed
    processed_identifiers = set()
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
            processed_identifiers = set(existing_df[identifier_key].unique())
            logger.info(f"Already processed {len(processed_identifiers)} {identifier_key}s")
        except Exception as e:
            logger.warning(f"Could not read existing CSV: {e}")

    # Evaluate each model
    total_models = len(model_files)
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for i, model_path in enumerate(model_files):
        if args.global_models:
            temp_identifier = extract_model_name(model_path.name)
        else:
            temp_identifier = extract_station_id(model_path.name)

        if temp_identifier in processed_identifiers:
            logger.info(f"[{i+1}/{total_models}] Skipping {model_path.name} - already processed")
            skipped_count += 1
            continue

        logger.info(f"[{i+1}/{total_models}]")
        try:
            saved_scaler_x, saved_scaler_y = load_scaler_for_model(model_path)
            identifier_value, metrics = evaluate_model(
                model_path,
                models_dir,
                identifier_key=identifier_key,
                identifier_value=temp_identifier,
                scaler_x=saved_scaler_x,
                scaler_y=saved_scaler_y
            )

            if metrics:
                new_df = pd.DataFrame([metrics])

                if not output_file.exists():
                    new_df.to_csv(output_file, index=False, mode='w')
                else:
                    new_df.to_csv(output_file, index=False, mode='a', header=False)

                processed_identifiers.add(identifier_value)
                processed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            logger.error(f"Fatal error processing {model_path.name}: {e}")
            failed_count += 1
            continue

    # Summary
    logger.info("="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total models: {total_models}")
    logger.info(f"Successfully evaluated: {processed_count}")
    logger.info(f"Skipped (already done): {skipped_count}")
    logger.info(f"Failed: {failed_count}")

    if output_file.exists():
        try:
            final_df = pd.read_csv(output_file)
            logger.info(f"\nResults in {output_file}:")
            logger.info(f"  Total records: {len(final_df)}")
            logger.info(f"  Mean R²: {final_df['r2'].mean():.4f}")
            logger.info(f"  Mean RMSE: {final_df['rmse'].mean():.4f}")
            logger.info(f"  Mean MAE: {final_df['mae'].mean():.4f}")
        except Exception as e:
            logger.error(f"Could not read final CSV: {e}")


if __name__ == '__main__':
    main()
