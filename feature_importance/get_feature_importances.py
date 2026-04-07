#!/usr/bin/env python3
"""
Extract feature importances from trained TFT models and save to CSV.

This script iterates over trained TFT models in the models/ directory and:
1. Loads the configuration and hyperparameters
2. Prepares the data
3. Loads the trained model
4. Runs inference to extract attention weights
5. Computes average variable selection weights
6. Saves results to CSV with format: station_id,feature_type,feature_name,avg_weight
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import gc
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
    log_file = logs_dir / f'feature_importances_{timestamp}.log'

    # Create logger
    logger = logging.getLogger('feature_importances')
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


def get_config_path_from_model_name(model_name):
    """
    Extract config file path from model name.

    Example model names:
    - cl_m-tft_out-48_freq-1h_wind_50.pt -> config_wind_50.yaml
    - cl_m-tft_out-48_freq-1h_wind_00164.pt -> wind_100ex50/config_wind_00164.yaml
    - cl_m-tft_out-48_freq-1h_wind_50_nostatic.pt -> config_wind_50.yaml (special case)
    """
    # Extract the identifier part (e.g., 'wind_50', 'wind_00164', 'wind_50_nostatic')
    parts = model_name.replace('.pt', '').split('_')

    # Find the 'wind' part and everything after
    if 'wind' not in parts:
        return None, False

    wind_idx = parts.index('wind')
    identifier = '_'.join(parts[wind_idx:])

    # Handle nostatic case: use base config (e.g., wind_50 from wind_50_nostatic)
    is_nostatic = 'nostatic' in identifier
    if is_nostatic:
        # Remove '_nostatic' suffix
        identifier = identifier.replace('_nostatic', '')

    # Determine config path
    # Short names like 'wind_50', 'wind_100' are in configs/
    # Numeric IDs like 'wind_00164' are in configs/wind_100ex50/
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
    """
    Load config from YAML file.
    If is_nostatic is True, set static_features to empty list.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if is_nostatic:
        config['params']['static_features'] = []
        print('ℹ️  Nostatic model detected: static_features set to []')

    return config


def extract_station_id(model_name):
    """Extract station ID from model name."""
    parts = model_name.replace('.pt', '').split('_')
    # Station ID is the last part (either numeric like 00164 or name like 50)
    station_id = parts[-1]
    # Remove 'nostatic' if present
    station_id = station_id.replace('nostatic', '').strip('_')
    return station_id


def extract_model_name(model_name):
    """Extract model identifier from model name for global models."""
    parts = model_name.replace('.pt', '').split('_')
    # Find the 'wind' part and everything after
    if 'wind' not in parts:
        return None

    wind_idx = parts.index('wind')
    identifier = '_'.join(parts[wind_idx:])
    return identifier


def process_model(model_path, models_dir, identifier_key='station_id', identifier_value=None):
    """
    Process a single model and extract feature importances.

    Args:
        model_path: Path to the model file
        models_dir: Directory containing models
        identifier_key: Name of the identifier column ('station_id' or 'model')
        identifier_value: Value for the identifier (e.g., '00164' or 'wind_50')

    Returns:
        Tuple of (identifier_value, List of dictionaries) with structure:
        {identifier_key: str, 'feature_type': str, 'feature_name': str, 'avg_weight': float}
    """
    model_name = model_path.name
    logger.info("="*80)
    logger.info(f"Processing: {model_name}")
    logger.info("="*80)

    # Use provided identifier or extract from model name
    if identifier_value is None:
        identifier_value = extract_station_id(model_name) if identifier_key == 'station_id' else extract_model_name(model_name)

    logger.info(f"{identifier_key}: {identifier_value}")

    # Get config path
    config_path, is_nostatic = get_config_path_from_model_name(model_name)
    if config_path is None or not config_path.exists():
        logger.warning(f"Config not found for {model_name}, skipping")
        return identifier_value, []

    logger.info(f"Config: {config_path}")

    # Load configuration
    try:
        config = load_config(config_path, is_nostatic)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return identifier_value, []

    # Set model name in config
    config['model']['name'] = 'tft'

    # Extract hyperparameters from model name for Optuna study lookup
    parts = model_name.replace('.pt', '').split('_')
    output_dim = int([p for p in parts if 'out-' in p][0].replace('out-', ''))
    freq_part = [p for p in parts if 'freq-' in p][0]
    freq = freq_part.replace('freq-', '')

    # Build study name suffix (everything after freq)
    freq_idx = parts.index(freq_part)
    study_name_suffix = '_'.join(parts[freq_idx+1:])
    study_name = f'cl_m-{config["model"]["name"]}_out-{output_dim}_freq-{freq}_{study_name_suffix}'

    # Load hyperparameters from Optuna study
    study = None
    try:
        if config['model']['lookup_hpo']:
            study = hpo.load_study(config['hpo']['studies_path'], study_name)
            logger.info(f"Study loaded: {study.study_name}")
    except Exception as e:
        logger.warning(f"Could not load study: {e}")

    hyperparameters = hpo.get_hyperparameters(config=config, study=study)

    # Get features
    features = preprocessing.get_features(config=config)
    logger.info(f"Features: {features}")

    # Load data
    try:
        data_dir = config['data']['path']
        freq = config['data']['freq']

        dfs = preprocessing.get_data(
            data_dir=data_dir,
            config=config,
            freq=freq,
            features=features
        )
        logger.info(f"Loaded {len(dfs)} dataframes")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return identifier_value, []

    # Fit global scaler
    try:
        logger.info("Fitting global scaler...")
        global_scaler_x = StandardScaler()

        target_col = config['data']['target_col']
        test_start = pd.Timestamp(config['data']['test_start'], tz='UTC')
        test_end = pd.Timestamp(config['data']['test_end'], tz='UTC')

        # If test_end is at 00:00, extend to include the whole day
        if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
            test_end = test_end.replace(hour=23, minute=0, second=0)

        for key, df in dfs.items():
            df_temp = df.copy()

            # Drop target column
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

        logger.info("Global scaler fitted")
    except Exception as e:
        logger.error(f"Error fitting scaler: {e}")
        return identifier_value, []

    # Prepare data for TFT
    try:
        logger.info("Preparing data for TFT...")
        data_generator = tools.create_data_generator(dfs, config, features, scaler_x=global_scaler_x)
        X_train, y_train, X_test, y_test, test_data, _ = tools.combine_datasets_efficiently(data_generator)

        # Extract feature dimensions from prepared data
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
        logger.info(f"Data prepared. Feature dims: {feature_dims}")

        gc.collect()
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return identifier_value, []

    # Load model
    try:
        logger.info("Loading model...")
        checkpoint = torch.load(model_path, map_location='cpu')

        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove '_orig_mod.' prefix from keys (added by torch.compile)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                cleaned_state_dict[key.replace('_orig_mod.', '')] = value
            else:
                cleaned_state_dict[key] = value

        # Create model instance with correct dimensions
        hyperparameters['model_type'] = 'tft'
        hyperparameters.update(feature_dims)

        model = get_model(config, hyperparameters)
        model.load_state_dict(cleaned_state_dict)
        model.eval()

        logger.info("Model loaded")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return identifier_value, []

    # Run inference and extract attention weights
    try:
        logger.info("Running inference...")

        # Extract test data tensors
        test_observed = X_test['observed'] if isinstance(X_test, dict) else X_test
        test_known = X_test.get('known', None) if isinstance(X_test, dict) else None
        test_static = X_test.get('static', None) if isinstance(X_test, dict) else None

        # Convert to torch tensors if needed
        if not isinstance(test_observed, torch.Tensor):
            test_observed = torch.FloatTensor(test_observed)
        if test_known is not None and not isinstance(test_known, torch.Tensor):
            test_known = torch.FloatTensor(test_known)
        if test_static is not None and not isinstance(test_static, torch.Tensor):
            test_static = torch.FloatTensor(test_static)

        # Forward pass with attention weight extraction
        with torch.no_grad():
            if test_static is not None:
                predictions, attention_dict = model(
                    test_observed,
                    test_known,
                    test_static,
                    return_attention_weights=True
                )
            else:
                predictions, attention_dict = model(
                    test_observed,
                    test_known,
                    return_attention_weights=True
                )

        logger.info("Inference complete")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return identifier_value, []

    # Extract variable selection weights
    results = []

    # Static features
    if 'static_weights' in attention_dict and attention_dict['static_weights'] is not None:
        static_weights = attention_dict['static_weights'].cpu().numpy()
        avg_static_weights = static_weights.mean(axis=(0, 1)) if static_weights.ndim > 2 else static_weights.mean(axis=0)
        static_features = config['params']['static_features']

        for feature, weight in zip(static_features, avg_static_weights):
            results.append({
                identifier_key: identifier_value,
                'feature_type': 'static',
                'feature_name': feature,
                'avg_weight': float(weight)
            })
        logger.info(f"Extracted {len(static_features)} static feature importances")

    # Past features (observed + known)
    if 'past_weights' in attention_dict:
        past_weights = attention_dict['past_weights'].cpu().numpy()
        avg_past_weights = past_weights.mean(axis=(0, 1))
        past_features = config['params']['observed_features'] + config['params']['known_features']

        for feature, weight in zip(past_features, avg_past_weights):
            results.append({
                identifier_key: identifier_value,
                'feature_type': 'past',
                'feature_name': feature,
                'avg_weight': float(weight)
            })
        logger.info(f"Extracted {len(past_features)} past feature importances")

    # Future features (known only)
    if 'future_weights' in attention_dict:
        future_weights = attention_dict['future_weights'].cpu().numpy()
        avg_future_weights = future_weights.mean(axis=(0, 1))
        future_features = config['params']['known_features']

        for feature, weight in zip(future_features, avg_future_weights):
            results.append({
                identifier_key: identifier_value,
                'feature_type': 'future',
                'feature_name': feature,
                'avg_weight': float(weight)
            })
        logger.info(f"Extracted {len(future_features)} future feature importances")

    # Clean up
    del model, checkpoint, X_train, y_train, X_test, y_test, dfs
    gc.collect()
    logger.info(f"Completed processing {model_name}")
    return identifier_value, results


def process_global_model_on_local_datasets(model_path, models_dir, local_configs_dirs):
    """
    Process a global model on multiple local datasets.

    Args:
        model_path: Path to the global model file
        models_dir: Directory containing models
        local_configs_dirs: List of directories containing local station configs

    Yields:
        Tuples of (station_id, results) for each local dataset as they are processed
    """
    model_name = model_path.name
    global_model_id = extract_model_name(model_name)

    logger.info("="*80)
    logger.info(f"Processing GLOBAL model {global_model_id} on LOCAL datasets")
    logger.info("="*80)

    # Get config path for the global model
    config_path, is_nostatic = get_config_path_from_model_name(model_name)
    if config_path is None or not config_path.exists():
        logger.error(f"Config not found for global model {model_name}")
        return []

    logger.info(f"Global model config: {config_path}")

    # Load global model configuration
    try:
        global_config = load_config(config_path, is_nostatic)
    except Exception as e:
        logger.error(f"Error loading global config: {e}")
        return []

    # Set model name
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

    # Get features from global config
    features = preprocessing.get_features(config=global_config)
    logger.info(f"Features: {features}")

    # Find all local station configs from both directories
    local_config_files = []
    for config_dir in local_configs_dirs:
        if config_dir.exists():
            local_config_files.extend(list(config_dir.glob('config_wind_*.yaml')))
    local_config_files = sorted(local_config_files)
    logger.info(f"Found {len(local_config_files)} local station configs")

    # Load the global model once (we'll use it for all datasets)
    try:
        logger.info("Loading global model...")
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

        logger.info("Global model loaded")
    except Exception as e:
        logger.error(f"Error loading global model: {e}")
        return

    # ===================================================================
    # FIT GLOBAL SCALER on ALL stations' training data (consistent with training)
    # ===================================================================
    logger.info("="*80)
    logger.info("Fitting GLOBAL scaler on all stations' training data...")
    logger.info("="*80)

    global_scaler_x = StandardScaler()
    target_col = global_config['data']['target_col']
    test_start = pd.Timestamp(global_config['data']['test_start'], tz='UTC')
    test_end = pd.Timestamp(global_config['data']['test_end'], tz='UTC')

    if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
        test_end = test_end.replace(hour=23, minute=0, second=0)

    # Iterate through all local configs to fit global scaler
    for i, local_config_path in enumerate(local_config_files):
        station_id = local_config_path.stem.replace('config_wind_', '')

        try:
            # Load local config
            with open(local_config_path, 'r') as f:
                local_config = yaml.safe_load(f)

            data_dir = local_config['data']['path']

            # Load data
            dfs = preprocessing.get_data(
                data_dir=data_dir,
                config=local_config,
                freq=global_config['data']['freq'],
                features=features
            )

            if len(dfs) == 0:
                continue

            # Fit scaler on training data only
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

    # Process each local dataset
    for i, local_config_path in enumerate(local_config_files):
        # Extract station ID from config filename
        station_id = local_config_path.stem.replace('config_wind_', '')

        logger.info(f"\n[{i+1}/{len(local_config_files)}] Processing station {station_id}")

        try:
            # Load local config to get the specific data file
            with open(local_config_path, 'r') as f:
                local_config = yaml.safe_load(f)

            # Use global model's features but local data
            data_dir = local_config['data']['path']

            # Load local station data
            dfs = preprocessing.get_data(
                data_dir=data_dir,
                config=local_config,
                freq=global_config['data']['freq'],
                features=features
            )

            if len(dfs) == 0:
                logger.warning(f"No data found for station {station_id}")
                continue

            logger.info(f"Loaded {len(dfs)} dataframes for station {station_id}")

            # Prepare data using the GLOBAL scaler (already fitted on all stations)
            data_generator = tools.create_data_generator(dfs, global_config, features, scaler_x=global_scaler_x)
            X_train, y_train, X_test, y_test, test_data, _ = tools.combine_datasets_efficiently(data_generator)

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

            # Create a DEEP COPY of global_config for this station to avoid shared state
            import copy
            station_config = copy.deepcopy(global_config)
            station_config['model']['feature_dim'] = feature_dims

            # Create model instance and load weights
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

            # DEBUG: Print static feature values for this station
            if test_static is not None:
                logger.info(f"DEBUG - Station {station_id} static values (first sample):")
                static_features = station_config['params']['static_features']
                static_vals = test_static[0] if test_static.ndim > 1 else test_static
                for feat, val in zip(static_features, static_vals):
                    logger.info(f"  {feat}: {val}")

            if not isinstance(test_observed, torch.Tensor):
                test_observed = torch.FloatTensor(test_observed)
            if test_known is not None and not isinstance(test_known, torch.Tensor):
                test_known = torch.FloatTensor(test_known)
            if test_static is not None and not isinstance(test_static, torch.Tensor):
                test_static = torch.FloatTensor(test_static)

            with torch.no_grad():
                if test_static is not None:
                    predictions, attention_dict = model(
                        test_observed, test_known, test_static,
                        return_attention_weights=True
                    )
                else:
                    predictions, attention_dict = model(
                        test_observed, test_known,
                        return_attention_weights=True
                    )

            # Extract variable selection weights
            results = []

            # Static features
            if 'static_weights' in attention_dict and attention_dict['static_weights'] is not None:
                static_weights = attention_dict['static_weights'].cpu().numpy()
                avg_static_weights = static_weights.mean(axis=(0, 1)) if static_weights.ndim > 2 else static_weights.mean(axis=0)
                static_features = station_config['params']['static_features']

                # DEBUG: Print static weights
                logger.info(f"DEBUG - Station {station_id} static weights:")
                logger.info(f"  Shape: {static_weights.shape}")
                logger.info(f"  Avg weights: {avg_static_weights}")

                for feature, weight in zip(static_features, avg_static_weights):
                    results.append({
                        'station_id': station_id,
                        'feature_type': 'static',
                        'feature_name': feature,
                        'avg_weight': float(weight)
                    })

            # Past features
            if 'past_weights' in attention_dict:
                past_weights = attention_dict['past_weights'].cpu().numpy()
                avg_past_weights = past_weights.mean(axis=(0, 1))
                past_features = station_config['params']['observed_features'] + station_config['params']['known_features']

                for feature, weight in zip(past_features, avg_past_weights):
                    results.append({
                        'station_id': station_id,
                        'feature_type': 'past',
                        'feature_name': feature,
                        'avg_weight': float(weight)
                    })

            # Future features
            if 'future_weights' in attention_dict:
                future_weights = attention_dict['future_weights'].cpu().numpy()
                avg_future_weights = future_weights.mean(axis=(0, 1))
                future_features = station_config['params']['known_features']

                for feature, weight in zip(future_features, avg_future_weights):
                    results.append({
                        'station_id': station_id,
                        'feature_type': 'future',
                        'feature_name': feature,
                        'avg_weight': float(weight)
                    })

            logger.info(f"Extracted {len(results)} feature importances for station {station_id}")

            # Yield results immediately for incremental CSV writing
            yield (station_id, results)

            # Clean up
            del model, X_train, y_train, X_test, y_test, dfs
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing station {station_id}: {e}")
            continue


def main():
    """Main function to process all models."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract feature importances from TFT models')
    parser.add_argument('-g', '--global', dest='global_models', action='store_true',
                        help='Process global models (wind_50, wind_100, etc.) instead of station-specific models')
    parser.add_argument('-e', '--evallocal', dest='eval_local', action='store_true',
                        help='Evaluate global models on local datasets (creates separate CSV for each global model)')
    args = parser.parse_args()

    # Setup paths (relative to current working directory, not script location)
    models_dir = Path('models')
    local_configs_dirs = [Path('configs/wind_50'), Path('configs/wind_100ex50')]

    # MODE 1: Evaluate global models on local datasets
    if args.eval_local:
        logger.info("="*80)
        logger.info("MODE: Evaluating GLOBAL models on LOCAL datasets")
        logger.info("="*80)
        logger.info(f"Scanning config directories: {[str(d) for d in local_configs_dirs]}")

        # Find all global models
        all_model_files = sorted([f for f in models_dir.glob('*.pt') if f.is_file()])
        logger.info(f"Total model files found: {len(all_model_files)}")

        global_model_files = []
        for f in all_model_files:
            model_id = extract_model_name(f.name)
            if model_id:
                parts = model_id.split('_')
                last_part = parts[-1] if parts[-1] != 'nostatic' else parts[-2]
                is_global = not (last_part.isdigit() and len(last_part) == 5)

                # Debug logging
                if is_global or last_part in ['50', '100']:
                    logger.info(f"  Checking {f.name}: model_id={model_id}, last_part={last_part}, len={len(last_part)}, is_global={is_global}")

                if is_global:
                    global_model_files.append(f)

        logger.info(f"Found {len(global_model_files)} global models to evaluate")

        # Process each global model
        for i, model_path in enumerate(global_model_files):
            global_model_id = extract_model_name(model_path.name)
            output_file = f'feature_importances_{global_model_id}.csv'

            logger.info(f"\n[{i+1}/{len(global_model_files)}] Processing global model: {global_model_id}")
            logger.info(f"Output: {output_file}")

            # Check which stations have already been processed for this model
            processed_stations = set()
            if Path(output_file).exists():
                try:
                    existing_df = pd.read_csv(output_file)
                    processed_stations = set(existing_df['station_id'].unique())
                    logger.info(f"Already processed {len(processed_stations)} stations for {global_model_id}")
                except Exception as e:
                    logger.warning(f"Could not read existing CSV: {e}")

            try:
                # Process global model on all local datasets
                all_station_results = process_global_model_on_local_datasets(
                    model_path, models_dir, local_configs_dirs
                )

                # Save results incrementally
                new_count = 0
                for station_id, results in all_station_results:
                    if station_id in processed_stations:
                        logger.info(f"Skipping station {station_id} - already processed")
                        continue

                    if results:
                        new_df = pd.DataFrame(results)

                        # Append to CSV
                        if not Path(output_file).exists():
                            new_df.to_csv(output_file, index=False, mode='w')
                            logger.info(f"Created {output_file}")
                        else:
                            new_df.to_csv(output_file, index=False, mode='a', header=False)

                        logger.info(f"Saved {len(results)} records for station {station_id}")
                        processed_stations.add(station_id)
                        new_count += 1

                logger.info(f"Completed {global_model_id}: processed {new_count} new stations")

                # Summary for this model
                if Path(output_file).exists():
                    final_df = pd.read_csv(output_file)
                    logger.info(f"{output_file}: {len(final_df)} total records, {final_df['station_id'].nunique()} stations")

            except Exception as e:
                logger.error(f"Error processing global model {global_model_id}: {e}")
                continue
            break

        logger.info("\n" + "="*80)
        logger.info("EVALLOCAL MODE COMPLETE")
        logger.info("="*80)
        return

    # MODE 2: Global models or Station-specific models
    if args.global_models:
        output_file = 'feature_importances_global.csv'
        identifier_key = 'model'
        logger.info("Processing GLOBAL models")
    else:
        output_file = 'feature_importances_all.csv'
        identifier_key = 'station_id'
        logger.info("Processing STATION-SPECIFIC models")

    # Find all TFT models
    all_model_files = sorted([f for f in models_dir.glob('*.pt') if f.is_file()])

    # Filter models based on global flag
    if args.global_models:
        # Global models: NOT numeric 5-digit IDs
        model_files = []
        for f in all_model_files:
            model_id = extract_model_name(f.name)
            if model_id:
                parts = model_id.split('_')
                last_part = parts[-1] if parts[-1] != 'nostatic' else parts[-2]
                if not (last_part.isdigit() and len(last_part) == 5):
                    model_files.append(f)
        logger.info(f"Found {len(model_files)} global model files")
    else:
        # Station-specific models: numeric 5-digit IDs
        model_files = []
        for f in all_model_files:
            station_id = extract_station_id(f.name)
            if station_id.isdigit() and len(station_id) == 5:
                model_files.append(f)
        logger.info(f"Found {len(model_files)} station-specific model files")

    # Check which identifiers have already been processed
    processed_identifiers = set()
    if Path(output_file).exists():
        try:
            existing_df = pd.read_csv(output_file)
            processed_identifiers = set(existing_df[identifier_key].unique())
            logger.info(f"Found existing results file with {len(processed_identifiers)} {identifier_key}s already processed")
            logger.info(f"Already processed: {sorted(processed_identifiers)}")
        except Exception as e:
            logger.warning(f"Could not read existing CSV: {e}")

    # Process each model
    total_models = len(model_files)
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for i, model_path in enumerate(model_files):
        # Extract identifier
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
            identifier_value, results = process_model(
                model_path,
                models_dir,
                identifier_key=identifier_key,
                identifier_value=temp_identifier
            )

            if results:
                # Convert to DataFrame
                new_df = pd.DataFrame(results)

                # Append to CSV
                if not Path(output_file).exists():
                    new_df.to_csv(output_file, index=False, mode='w')
                    logger.info(f"Created {output_file}")
                else:
                    new_df.to_csv(output_file, index=False, mode='a', header=False)

                logger.info(f"Saved {len(results)} records for {identifier_key} {identifier_value} to {output_file}")
                processed_identifiers.add(identifier_value)
                processed_count += 1
            else:
                logger.warning(f"No results for {model_path.name}")
                failed_count += 1

        except Exception as e:
            logger.error(f"Fatal error processing {model_path.name}: {e}")
            failed_count += 1
            continue

    # Final summary
    logger.info("="*80)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total models found: {total_models}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Skipped (already done): {skipped_count}")
    logger.info(f"Failed: {failed_count}")

    if Path(output_file).exists():
        try:
            final_df = pd.read_csv(output_file)
            logger.info(f"\nFinal results in {output_file}:")
            logger.info(f"  Total records: {len(final_df)}")
            logger.info(f"  Unique {identifier_key}s: {final_df[identifier_key].nunique()}")
            logger.info(f"  Feature types: {final_df['feature_type'].unique().tolist()}")
            logger.info(f"\nPreview:")
            logger.info(f"\n{final_df.head(10).to_string()}")
        except Exception as e:
            logger.error(f"Could not read final CSV: {e}")
    else:
        logger.warning("No results file created")


if __name__ == '__main__':
    main()
