#!/usr/bin/env python3
"""
Federated Learning Training Script
Adapted to match train_cl.py architecture with PyTorch
"""

import os
import copy
import json
import argparse
import pandas as pd
import numpy as np
import logging
import pickle
import gc
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.multiprocessing

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import preprocessing, tools, hpo, eval, federated, models


def main() -> None:
    logger = logging.getLogger(__name__)

    # Argument parser
    parser = argparse.ArgumentParser(description="Federated Learning Training")
    parser.add_argument('-m', '--model', type=str, default='tft', help='Select Model (default: tft)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    parser.add_argument('-s', '--suffix', type=str, default='', help='Define suffix for study name (default: empty)')
    parser.add_argument('--save_model', action='store_true', default=False, help='Save trained model to models directory (default: False)')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    suffix = ''
    if args.suffix:
        suffix = f'_{args.suffix}'
    if '.yaml' in args.config:
        args.config = args.config.split('.')[0]
    if '/' in args.config:
        config_name = args.config.split('/')[-1]
    else:
        config_name = args.config

    # FL-specific log naming
    study_name_suffix = '_'.join(config_name.split('_')[1:])
    if args.suffix:
        study_name_suffix += f'_{args.suffix}'
    log_file = f'logs/train_fl_m-{args.model}_{study_name_suffix}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # GPU initialization
    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("Using CPU")

    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load config
    config = tools.load_config(f'{args.config}.yaml')
    freq = config['data']['freq']
    params = config['params']
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']

    logging.info(f'Federated Model: PyTorch {args.model.upper()}, Output dim: {output_dim}, Frequency: {freq}, '
                f'Lookback: {lookback}, Horizon: {horizon}, Step size: {config["model"]["step_size"]}')

    # Set model name and FL flag for preprocessing pipeline
    config['model']['name'] = args.model
    config['model']['fl'] = True  # Disable torch.compile in FL mode

    # Extract retrain settings with defaults
    retrain_interval = config['data'].get('retrain_interval', 1)
    eval_interval = config['data'].get('eval_interval', retrain_interval)

    # Validate eval_interval configuration
    if retrain_interval > 1 and eval_interval != retrain_interval:
        logging.warning(
            f"Invalid configuration: retrain_interval={retrain_interval} and eval_interval={eval_interval}. "
            f"eval_interval can only differ from retrain_interval when retrain_interval=1. "
            f"Setting eval_interval to {retrain_interval}."
        )
        eval_interval = retrain_interval
    elif retrain_interval == 1 and eval_interval < 1:
        logging.warning(f"Invalid eval_interval={eval_interval}. Setting to 1.")
        eval_interval = 1

    logging.info(f"Retrain interval: {retrain_interval}, Eval interval: {eval_interval}")

    # Get features
    data_dir = config['data']['path']
    features = preprocessing.get_features(config=config)

    # Setup paths
    base_dir = os.path.basename(data_dir)
    target_dir = os.path.join('results', base_dir)
    os.makedirs(target_dir, exist_ok=True)

    fl_strategy = config['fl'].get('strategy', 'fedavg')
    study_name = f'fl_a-{fl_strategy}_m-{args.model}_out-{output_dim}_freq-{freq}_{study_name_suffix}'

    logging.info(f'Start Federated Learning for Study: {study_name}')
    logging.info(f'Config: {json.dumps(params, indent=2)}')

    # Calculate test periods for evaluation
    test_start = pd.Timestamp(config['data']['test_start'], tz='UTC')
    test_end = pd.Timestamp(config['data']['test_end'], tz='UTC')

    # If test_end is at 00:00, extend to include the whole day
    if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
        test_end = test_end.replace(hour=23, minute=0, second=0)

    if eval_interval > 1:
        # Custom date-based splitting logic
        all_starttimes = set()

        # Load initial data to get dates (use first client as sample)
        first_client_id = list(config['fl']['clients'].keys())[0]
        first_client_files = config['fl']['clients'][first_client_id]
        temp_config = copy.deepcopy(config)
        temp_config['data']['files'] = first_client_files
        temp_dfs = preprocessing.get_data(
            data_dir=data_dir,
            config=temp_config,
            freq=freq,
            features=features
        )

        for df in temp_dfs.values():
            if isinstance(df.index, pd.MultiIndex):
                dates = pd.to_datetime(df.index.get_level_values('starttime')).normalize().unique()
                all_starttimes.update(dates)
            else:
                dates = pd.to_datetime(df.index).normalize().unique()
                all_starttimes.update(dates)

        del temp_dfs
        gc.collect()

        sorted_dates = sorted(list(all_starttimes))
        if sorted_dates and sorted_dates[0].tzinfo:
            ts_test_start = test_start.tz_convert(sorted_dates[0].tzinfo)
            ts_test_end = test_end.tz_convert(sorted_dates[0].tzinfo)
        else:
            ts_test_start = test_start.tz_localize(None)
            ts_test_end = test_end.tz_localize(None)

        test_dates = [d for d in sorted_dates if d >= ts_test_start and d <= ts_test_end]

        if not test_dates:
            logging.warning("No data found within test range! Fallback to standard calculation.")
            test_periods = tools.calculate_retrain_periods(test_start, test_end, eval_interval, freq)
        else:
            chunks = np.array_split(test_dates, eval_interval)
            test_periods = []
            for chunk in chunks:
                if len(chunk) > 0:
                    p_start = chunk[0]
                    p_end = chunk[-1]
                    p_end = p_end + pd.Timedelta(hours=23, minutes=59, seconds=59)
                    test_periods.append((p_start, p_end))

        if retrain_interval > 1:
            logging.info(f"Test periods (Date-based, with retraining): {len(test_periods)}")
        else:
            logging.info(f"Test periods (Date-based, evaluation only): {len(test_periods)}")
        for i, (start, end) in enumerate(test_periods):
            logging.info(f"  Period {i+1}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    else:
        test_periods = [(test_start, test_end)]
        logging.info(f"Single test period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")

    # Storage for multiple retraining results
    all_evaluations = []
    all_histories = []

    # Get hyperparameters using hpo.get_hyperparameters
    study = None
    if config['model'].get('lookup_hpo', False):
        logging.info(f"Looking up hyperparameters for study: {study_name}")
        study = None #hpo.load_study(config['hpo']['studies_path'], study_name)
    hyperparameters = hpo.get_hyperparameters(config=config, study=study)

    # Add FL-specific hyperparameters
    hyperparameters['n_rounds'] = config['fl'].get('n_rounds', 10)
    hyperparameters['personalization'] = config['fl'].get('personalize', False)
    hyperparameters['global_scaler'] = config['fl'].get('global_scaler', False)

    logging.info(f"Hyperparameters: {json.dumps(hyperparameters, indent=2)}")

    # Determine if we need to train in this loop or just evaluate
    train_once_eval_multiple = (retrain_interval == 1 and eval_interval > 1)
    trained_clients_weights = None  # Store weights when training once

    # --- LOAD TURBINE ASSIGNMENTS TO ENSURE CONSISTENT SEEDS ---
    # Load station_turbine_assignments.csv to map station_ids to seeds
    # This ensures the same turbines are used for the same stations in both CL and FL
    station_assignments_path = 'data/station_turbine_assignments.csv'
    if os.path.exists(station_assignments_path):
        station_assignments = pd.read_csv(station_assignments_path, dtype={'station_id': str})
        # Create a mapping: station_id -> seed_offset
        # The seed offset is determined by the order in config['data']['files']
        station_to_seed = {}
        base_seed = config['params'].get('random_seed', 42)

        for idx, file_id in enumerate(config['data']['files']):
            station_to_seed[file_id] = base_seed + idx

        logging.debug(f"Loaded turbine assignments for {len(station_to_seed)} stations from config")
    else:
        logging.warning(f"Station turbine assignments file not found: {station_assignments_path}")
        logging.warning("Turbine assignments may not match centralized training!")
        station_to_seed = None

    # --- LOAD AND PREPARE CLIENT DATA WITH CLIENT-SPECIFIC SCALERS ---
    clients_data = {}

    for client_id, station_ids in config['fl']['clients'].items():
        logging.info(f'Loading data for client: {client_id} with stations: {station_ids}')

        # Create client-specific config
        client_config = copy.deepcopy(config)
        client_config['data']['files'] = station_ids

        # Set the random seeds for this client's stations to match CL training
        # We need to pass the seed to each preprocessing call
        if station_to_seed:
            for station_id in station_ids:
                if station_id in station_to_seed:
                    # Store the correct seed for this station
                    # This will be used during preprocessing.get_data
                    logging.debug(f"Station {station_id} will use seed: {station_to_seed[station_id]}")
                else:
                    logging.warning(f"Station {station_id} not found in station_to_seed mapping")

        # Load client data
        client_dfs = preprocessing.get_data(
            data_dir=data_dir,
            config=client_config,
            freq=freq,
            features=features
        )

        clients_data[client_id] = {
            'dfs': client_dfs,
            'station_ids': station_ids,
            'scaler_x': None  # Will be fitted per period
        }

    logging.info(f"Loaded data for {len(clients_data)} clients.")

    # Optional separate validation/test stations (val_files), analogous to train_cl.py.
    # When present: training sequences stay per-client (from fl.clients / data.files),
    # and the final evaluation uses the global FL model (weights before fine-tuning)
    # applied to these unseen stations — allowing out-of-sample generalisation testing.
    val_dfs = None
    if config['data'].get('val_files'):
        logging.info("val_files found — loading separate validation/test stations for FL evaluation.")
        val_config = copy.deepcopy(config)
        val_config['data']['files'] = config['data']['val_files']
        val_dfs = preprocessing.get_data(
            data_dir=data_dir,
            config=val_config,
            freq=freq,
            features=features
        )
        logging.info(f"Loaded {len(val_dfs)} val stations.")

    # Loop over test periods (training and/or evaluation)
    for period_idx, (current_test_start, current_test_end) in enumerate(test_periods):
        logging.info(f"\n{'='*80}")
        if train_once_eval_multiple:
            if period_idx == 0:
                logging.info(f"Training cycle 1/1 (will evaluate on {len(test_periods)} periods)")
            else:
                logging.info(f"Evaluation cycle {period_idx + 1}/{len(test_periods)}")
        else:
            logging.info(f"Training cycle {period_idx + 1}/{len(test_periods)}")
        logging.info(f"Test period: {current_test_start.strftime('%Y-%m-%d')} to {current_test_end.strftime('%Y-%m-%d')}")
        logging.info(f"{'='*80}\n")

        # Create period-specific config
        period_config = copy.deepcopy(config)

        # For train_once_eval_multiple: use original test_start for training
        if train_once_eval_multiple:
            training_test_start = test_start.strftime('%Y-%m-%d %H:%M')
            eval_test_start = current_test_start.strftime('%Y-%m-%d %H:%M')
            eval_test_end = current_test_end.strftime('%Y-%m-%d %H:%M')
        else:
            training_test_start = current_test_start.strftime('%Y-%m-%d %H:%M')
            eval_test_start = current_test_start.strftime('%Y-%m-%d %H:%M')
            eval_test_end = current_test_end.strftime('%Y-%m-%d %H:%M')

        # Set test boundaries
        period_config['data']['test_start'] = training_test_start
        period_config['data']['test_end'] = eval_test_end

        # Determine if we need to train in this iteration
        should_train = (period_idx == 0) if train_once_eval_multiple else True

        # --- PREPARE CLIENT PARTITIONS ---
        partitions = {}
        test_data = {}  # Store test data for evaluation

        # Phase 1: Fit local scalers for all clients
        if should_train:
            target_col = period_config['data']['target_col']
            fit_scaler_y = target_col != 'power'  # power is pre-normalised to [0, 1]

            for client_id, client_info in clients_data.items():
                logging.info(f"Preparing data for client: {client_id}")
                logging.debug(f"Fitting scaler for client {client_id}...")
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()

                for key, df in tqdm(client_info['dfs'].items(), desc=f"Fitting Scaler for {client_id}"):
                    df_temp = df.copy()

                    # For non-TFT models, create lag features before fitting scaler
                    if period_config['model']['name'] not in ['tft', 'tcn-tft', 'stemgnn']:
                        for col in features['observed']:
                            all_observed_cols = [new_col for new_col in df_temp.columns if new_col == col or new_col.startswith(col + '_lag_')]
                            for new_col in all_observed_cols:
                                df_temp = preprocessing.lag_features(
                                    data=df_temp,
                                    lookback=period_config['model']['lookback'],
                                    horizon=period_config['model']['horizon'],
                                    lag_in_col=period_config['data']['lag_in_col'],
                                    target_col=new_col
                                )
                                if new_col != target_col and new_col not in features['known']:
                                    df_temp.drop(new_col, axis=1, inplace=True, errors='ignore')

                    t_0 = 0 if period_config['eval']['eval_on_all_test_data'] else period_config['eval']['t_0']
                    df_train, _ = preprocessing.split_data(
                        data=df_temp,
                        train_frac=period_config['data']['train_frac'],
                        test_start=pd.Timestamp(period_config['data']['test_start'], tz='UTC'),
                        test_end=pd.Timestamp(period_config['data']['test_end'], tz='UTC'),
                        t_0=t_0
                    )

                    # Fit Y scaler on target before dropping it
                    if fit_scaler_y and target_col in df_train.columns:
                        scaler_y.partial_fit(df_train[[target_col]].values)

                    # Fit X scaler on all features except target
                    df_train_x = df_train.drop(columns=[target_col], errors='ignore')
                    scaler_x.partial_fit(df_train_x.values)

                    del df_temp, df_train, df_train_x
                    gc.collect()

                client_info['scaler_x'] = scaler_x
                client_info['scaler_y'] = scaler_y if fit_scaler_y else None
                logging.debug(f"Scaler fitted for client {client_id}.")

            # Phase 2: Aggregate scalers if global_scaler is enabled
            if period_config['fl'].get('global_scaler', False):
                logging.info("Aggregating local scalers into a global scaler...")

                # Aggregate scaler_x
                client_stats_x = []
                for client_id, client_info in clients_data.items():
                    local_scaler = client_info['scaler_x']
                    client_stats_x.append({
                        'n': local_scaler.n_samples_seen_,
                        'mu': local_scaler.mean_,
                        'var': local_scaler.var_
                    })

                mu_global, std_global = federated.aggregate_scalers(client_stats_x)

                global_scaler_x = StandardScaler()
                global_scaler_x.mean_ = mu_global
                global_scaler_x.scale_ = std_global
                global_scaler_x.var_ = std_global ** 2
                global_scaler_x.n_samples_seen_ = sum(s['n'] for s in client_stats_x)
                global_scaler_x.n_features_in_ = len(mu_global)

                for client_id, client_info in clients_data.items():
                    client_info['scaler_x'] = global_scaler_x

                # Aggregate scaler_y (1-dimensional target)
                if fit_scaler_y:
                    client_stats_y = []
                    for client_id, client_info in clients_data.items():
                        local_sy = client_info['scaler_y']
                        if local_sy is not None and local_sy.n_samples_seen_ is not None:
                            client_stats_y.append({
                                'n': local_sy.n_samples_seen_,
                                'mu': local_sy.mean_,
                                'var': local_sy.var_
                            })

                    if client_stats_y:
                        mu_y, std_y = federated.aggregate_scalers(client_stats_y)
                        global_scaler_y = StandardScaler()
                        global_scaler_y.mean_ = mu_y
                        global_scaler_y.scale_ = std_y
                        global_scaler_y.var_ = std_y ** 2
                        global_scaler_y.n_samples_seen_ = sum(s['n'] for s in client_stats_y)
                        global_scaler_y.n_features_in_ = len(mu_y)

                        for client_id, client_info in clients_data.items():
                            client_info['scaler_y'] = global_scaler_y

                logging.info(f"Global scalers (X and Y) applied to all {len(clients_data)} clients.")

        # Phase 3: Process data with the assigned scaler
        for client_id, client_info in clients_data.items():
            logging.info(f"Processing data for client: {client_id}")

            # Use memory-efficient data processing
            logging.debug(f"Processing data for client {client_id}...")
            data_generator = tools.create_data_generator(
                client_info['dfs'],
                period_config,
                features,
                scaler_x=client_info['scaler_x'],
                scaler_y=client_info.get('scaler_y')
            )
            X_train, y_train, X_test, y_test, client_test_data = tools.combine_datasets_efficiently(data_generator)

            # Store partition for federated training
            # Use ALL training data for FL training, test data for both validation AND evaluation
            X_train_fl = X_train
            y_train_fl = y_train

            # Use test data for both FL validation and final evaluation
            X_val_fl = X_test
            y_val_fl = y_test

            partitions[client_id] = (X_train_fl, y_train_fl, X_val_fl, y_val_fl)

            # Store test data for evaluation (same as validation data)
            test_data[client_id] = client_test_data

            logging.debug(f"Client {client_id} data splits:")
            logging.debug(f"  Train: {len(y_train_fl)} samples (before test_start)")
            logging.debug(f"  Val+Test: {len(y_val_fl)} samples (test_start to test_end)")

            del X_train, y_train, X_test, y_test, data_generator
            gc.collect()

        # Generate test data from val_files (uses global scaler, fitted on clients' data only).
        # Val stations are held out from FL training entirely — their data is only used here.
        val_test_data = None
        if val_dfs is not None:
            # All clients share the same scaler when global_scaler=True.
            # If global_scaler=False, we use the first client's scaler as best approximation
            # (a warning is logged to make this explicit).
            representative_scaler = list(clients_data.values())[0]['scaler_x']
            if not period_config['fl'].get('global_scaler', False):
                logging.warning(
                    "val_files is set but global_scaler=False: val stations will be scaled "
                    "with the first client's local scaler. Consider enabling global_scaler=True "
                    "for consistent scaling across all stations (especially with static features)."
                )
            val_generator = tools.create_data_generator(
                val_dfs, period_config, features,
                scaler_x=representative_scaler
            )
            _, _, _, _, val_test_data = tools.combine_datasets_efficiently(val_generator)
            logging.info(f"Generated test data for {len(val_test_data)} val stations.")

        # Log data shapes for first client
        first_client = list(partitions.keys())[0]
        X_sample = partitions[first_client][0]

        if isinstance(X_sample, dict):
            if 'known' in X_sample:
                logging.info(f'Data shape: X_train known: {X_sample["known"].shape}')
            if 'static' in X_sample:
                logging.info(f'Data shape: X_train static: {X_sample["static"].shape}')
            logging.info(f'Data shape: X_train observed: {X_sample["observed"].shape}')
        else:
            logging.info(f'Data shape: X_train: {X_sample.shape}')

        # Set feature_dim for model creation
        period_config['model']['feature_dim'] = tools.get_feature_dim(X=X_sample)
        logging.debug(f"Feature dim set to: {period_config['model']['feature_dim']}")

        # Device selection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # --- FEDERATED TRAINING ---
        if should_train:

            history, clients_weights = federated.run_simulation(
                partitions=partitions,
                hyperparameters=hyperparameters,
                config=period_config
            )

            # Store weights if training once
            if train_once_eval_multiple:
                trained_clients_weights = clients_weights
                logging.info(f"Federated model trained and will be reused for all {len(test_periods)} evaluation periods.")

            logging.info("Federated training completed.")
            # Log final round metrics
            if 'metrics_aggregated' in history and not history['metrics_aggregated'].empty:
                final_metrics = history['metrics_aggregated'].iloc[-1]
                logging.info(f"Final Round Metrics: {final_metrics.to_dict()}")
        else:
            # Evaluation only: use previously trained weights
            logging.info("Evaluation only (using previously trained model)...")
            clients_weights = trained_clients_weights
            history = None

        # Save the global FL weights (last aggregation round) before any per-client fine-tuning.
        # These are used to evaluate val stations, which have no client-specific weights.
        global_fl_weights = copy.deepcopy(list(clients_weights.values())[0]) if clients_weights else None

        # --- OPTIONAL FINE-TUNING ---
        # Fine-tune the global model on each client's local data with early stopping
        if period_config['fl'].get('fine_tune', False) and clients_weights:
            logging.info("Starting client-specific fine-tuning with early stopping...")

            # Parallelize fine-tuning with Ray
            import ray

            # Check if Ray is already initialized (from FL training)
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            # Calculate GPU allocation for fine-tuning
            total_num_gpus = torch.cuda.device_count()
            n_clients = len(clients_data)
            gpu_per_actor = min(1, total_num_gpus / n_clients) if total_num_gpus > 0 else 0

            logging.info(f"Fine-tuning: {n_clients} clients in parallel, {total_num_gpus} GPUs available, {gpu_per_actor:.2f} GPU per client")

            # Create Ray remote function for fine-tuning
            @ray.remote(num_gpus=gpu_per_actor)
            def fine_tune_client(client_id, X_train, y_train, X_val, y_val,
                                global_weights, config, hyperparameters):
                """Fine-tune a single client in parallel."""
                import torch
                from utils import models, tools
                import copy
                import logging

                logging.info(f"Fine-tuning for client: {client_id}")

                # Device setup
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Create model with global weights
                model = models.get_model(config=config, hyperparameters=hyperparameters)
                model.load_state_dict(global_weights)
                model = model.to(device)

                # Fine-tune with model.early_stopping config
                fine_tune_config = copy.deepcopy(config)

                # Use model.early_stopping for fine-tuning
                if 'early_stopping' in config.get('model', {}):
                    fine_tune_config['model']['early_stopping'] = config['model']['early_stopping']
                    logging.debug(f"Fine-tuning client {client_id} with model.early_stopping config")
                else:
                    # Fallback: enable early stopping with defaults
                    fine_tune_config['model']['early_stopping'] = {
                        'enabled': True,
                        'patience': 5,
                        'min_delta': 0.0001,
                        'monitor': 'val_rmse',
                        'mode': 'min',
                        'restore_best_weights': True
                    }
                    logging.debug(f"Fine-tuning client {client_id} with default early stopping")

                # Use fine_tune_epochs from config or default
                fine_tune_hyperparams = copy.deepcopy(hyperparameters)
                fine_tune_hyperparams['epochs'] = config['fl'].get('fine_tune_epochs', 50)

                logging.debug(f"Fine-tuning client {client_id} with {fine_tune_hyperparams['epochs']} epochs max")

                # Fine-tune model
                _, fine_tuned_model = tools.training_pipeline(
                    train=(X_train, y_train),
                    val=(X_val, y_val),
                    hyperparameters=fine_tune_hyperparams,
                    config=fine_tune_config,
                    device=device
                )

                # Return state_dict (must be on CPU for serialization)
                return {
                    'client_id': client_id,
                    'weights': {k: v.cpu() for k, v in fine_tuned_model.state_dict().items()}
                }

            # Launch fine-tuning jobs in parallel
            fine_tune_jobs = []
            for client_id, client_info in clients_data.items():
                # Get client's training data
                X_train_client, y_train_client, X_val_client, y_val_client = partitions[client_id]

                # Launch Ray job
                job = fine_tune_client.remote(
                    client_id=client_id,
                    X_train=X_train_client,
                    y_train=y_train_client,
                    X_val=X_val_client,
                    y_val=y_val_client,
                    global_weights=clients_weights[client_id],
                    config=period_config,
                    hyperparameters=hyperparameters
                )
                fine_tune_jobs.append(job)

            # Wait for all fine-tuning jobs to complete
            logging.info(f"Waiting for {len(fine_tune_jobs)} fine-tuning jobs to complete...")
            fine_tune_results = ray.get(fine_tune_jobs)

            # Collect fine-tuned weights
            fine_tuned_weights = {}
            for result in fine_tune_results:
                client_id = result['client_id']
                fine_tuned_weights[client_id] = result['weights']
                logging.info(f"Fine-tuning completed for client: {client_id}")

            # Replace global weights with fine-tuned weights
            clients_weights = fine_tuned_weights
            logging.info("Fine-tuning completed for all clients.")

            gc.collect()

        # --- GENERATE PREDICTIONS AND EVALUATE PER PARK ---
        logging.info('Start evaluation pipeline...')

        # Evaluate each park individually
        eval_results = []
        for client_id, client_test_data in test_data.items():
            logging.info(f'Evaluating client: {client_id}')

            # Create model with client-specific weights (either FL or fine-tuned)
            model = models.get_model(config=period_config, hyperparameters=hyperparameters)
            model.load_state_dict(clients_weights[client_id])
            model = model.to(device)

            # Evaluate each park in this client
            for park_key, (X_test_park, y_test_park, index_test_park, scaler_y_park) in client_test_data.items():
                logging.debug(f'Evaluating park: {park_key} for client: {client_id}')

                # Get park data
                park_df = clients_data[client_id]['dfs'][park_key]

                # Determine test_start for evaluation
                if train_once_eval_multiple:
                    test_start_str = eval_test_start
                else:
                    test_start_str = period_config['data']['test_start']

                t_0 = 0 if period_config['eval']['eval_on_all_test_data'] else period_config['eval']['t_0']

                # Run evaluation pipeline for this park
                park_eval = eval.evaluation_pipeline(
                    data=park_df,
                    model=model,
                    model_name=f'{args.model.upper()}',
                    X_test=X_test_park,
                    y_test=y_test_park,
                    scaler_y=scaler_y_park,
                    output_dim=output_dim,
                    horizon=horizon,
                    index_test=index_test_park,
                    test_start=test_start_str,
                    t_0=t_0,
                    park_id=park_key,
                    synth_dir=None,
                    get_physical_persistence=False,
                    target_col=period_config['data']['target_col'],
                    evaluate_on_all_test_data=period_config['eval']['eval_on_all_test_data'],
                    device=device
                )
                park_eval['key'] = park_key
                park_eval['client_id'] = client_id
                eval_results.append(park_eval)

            del model
            gc.collect()

        # --- EVALUATE VAL STATIONS (out-of-sample generalisation) ---
        # Uses the global FL model (weights before fine-tuning), since val stations
        # were never part of any client's training data.
        if val_test_data is not None and global_fl_weights is not None:
            logging.info("Evaluating val stations with global FL model (pre-fine-tune weights)...")
            val_model = models.get_model(config=period_config, hyperparameters=hyperparameters)
            val_model.load_state_dict(global_fl_weights)
            val_model = val_model.to(device)

            t_0 = 0 if period_config['eval']['eval_on_all_test_data'] else period_config['eval']['t_0']
            test_start_str = eval_test_start if train_once_eval_multiple else period_config['data']['test_start']

            for park_key, (X_test_park, y_test_park, index_test_park, scaler_y_park) in val_test_data.items():
                park_df = val_dfs[park_key]
                park_eval = eval.evaluation_pipeline(
                    data=park_df,
                    model=val_model,
                    model_name=f'{args.model.upper()}',
                    X_test=X_test_park,
                    y_test=y_test_park,
                    scaler_y=scaler_y_park,
                    output_dim=output_dim,
                    horizon=horizon,
                    index_test=index_test_park,
                    test_start=test_start_str,
                    t_0=t_0,
                    park_id=park_key,
                    synth_dir=None,
                    get_physical_persistence=False,
                    target_col=period_config['data']['target_col'],
                    evaluate_on_all_test_data=period_config['eval']['eval_on_all_test_data'],
                    device=device
                )
                park_eval['key'] = park_key
                park_eval['client_id'] = 'val'
                eval_results.append(park_eval)

            del val_model
            gc.collect()
            logging.info(f"Val station evaluation completed: {len(val_test_data)} stations.")

        # Combine all evaluations for this period
        if eval_results:
            period_evaluation = pd.concat(eval_results, ignore_index=False)
        else:
            period_evaluation = pd.DataFrame()

        logging.info(f"Per-park evaluation completed: {len(eval_results)} parks evaluated")

        if not period_evaluation.empty:
            # Add aggregated statistics
            period_evaluation.loc['mean'] = period_evaluation.mean(numeric_only=True)
            period_evaluation.loc['std'] = period_evaluation.std(numeric_only=True)
            logging.info(f"\n{period_evaluation.to_string()}")

            # Log metrics grouped by client_id
            numeric_cols = period_evaluation.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude mean/std rows for grouping
            eval_without_agg = period_evaluation.drop(['mean', 'std'], errors='ignore')
            if 'client_id' in eval_without_agg.columns:
                client_grouped = eval_without_agg.groupby('client_id')[numeric_cols].mean()
                logging.info(f"\nMetrics grouped by client_id (mean):\n{client_grouped.to_string()}")

        period_evaluation['output_dim'] = output_dim
        period_evaluation['freq'] = freq
        period_evaluation['strategy'] = fl_strategy
        period_evaluation['personalization'] = config['fl'].get('personalize', False)

        if period_config['eval']['eval_on_all_test_data']:
            period_evaluation['t_0'] = None
        else:
            period_evaluation['t_0'] = period_config['eval']['t_0']

        # Store results for this period
        all_evaluations.append(period_evaluation)
        if history is not None:
            all_histories.append(history)

        if train_once_eval_multiple:
            if period_idx == 0:
                logging.info(f"Completed training cycle 1/1")
            else:
                logging.info(f"Completed evaluation cycle {period_idx + 1}/{len(test_periods)}")
        else:
            logging.info(f"Completed training cycle {period_idx + 1}/{len(test_periods)}")

    # --- AGGREGATE RESULTS ACROSS ALL RETRAINING PERIODS ---
    logging.info(f"\n{'='*80}")

    if len(all_evaluations) > 1:
        # Calculate mean evaluation across all retrainings/evaluations
        first_eval = all_evaluations[0]
        numeric_cols = first_eval.select_dtypes(include=[np.number]).columns
        non_numeric_cols = first_eval.select_dtypes(exclude=[np.number]).columns

        evaluation = first_eval.copy()

        # Average numeric columns row-wise across all evaluations
        for col in numeric_cols:
            col_data = pd.concat([eval_df[col] for eval_df in all_evaluations], axis=1)
            evaluation[col] = col_data.mean(axis=1)

        logging.info(f"Aggregated evaluation:")
        logging.info(evaluation)

        # Build results with individual results
        results = {
            'hyperparameters': hyperparameters,
            'config': config,
            'history': all_histories if all_histories else None,
            'evaluation': evaluation,
            'individual_evaluations': all_evaluations,
            'test_dates': test_periods,
            'clients_weights': clients_weights
        }
    else:
        # Single training run
        evaluation = all_evaluations[0]
        history_result = all_histories[0] if all_histories else None
        results = {
            'hyperparameters': hyperparameters,
            'config': config,
            'history': history_result,
            'evaluation': evaluation,
            'test_dates': test_periods,
            'clients_weights': clients_weights
        }

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path_to_pkl = os.path.join('results', base_dir, f'{study_name}_{timestamp}.pkl')

    with open(path_to_pkl, 'wb') as f:
        pickle.dump(results, f)

    logging.info(f"\nFederated training completed! Results saved to: {path_to_pkl}")

    # Save model if requested
    if args.save_model:
        os.makedirs(os.path.join('models', 'scaler'), exist_ok=True)

        if config['fl'].get('personalize', False):
            # Personalized: save one model and scaler per client
            for client_id, weights in clients_weights.items():
                model_name = f'{study_name}_id-{client_id}.pt'
                model_path = os.path.join('models', model_name)
                torch.save(weights, model_path)
                logging.info(f"Client {client_id} model saved to: {model_path}")

                scaler_name = f'scaler_{study_name}_id-{client_id}.pkl'
                scaler_path = os.path.join('models', 'scaler', scaler_name)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(clients_data[client_id]['scaler_x'], f)
                logging.info(f"Client {client_id} scaler saved to: {scaler_path}")

                if clients_data[client_id].get('scaler_y') is not None:
                    scaler_y_path = scaler_path.replace('scaler_', 'scaler_y_')
                    with open(scaler_y_path, 'wb') as f:
                        pickle.dump(clients_data[client_id]['scaler_y'], f)
                    logging.info(f"Client {client_id} scaler_y saved to: {scaler_y_path}")
        else:
            # Global model: all clients share the same weights, save once
            first_client_id = list(clients_weights.keys())[0]
            model_name = f'{study_name}_global.pt'
            model_path = os.path.join('models', model_name)
            torch.save(clients_weights[first_client_id], model_path)
            logging.info(f"Global model saved to: {model_path}")

            scaler_name = f'scaler_{study_name}_global.pkl'
            scaler_path = os.path.join('models', 'scaler', scaler_name)
            with open(scaler_path, 'wb') as f:
                pickle.dump(clients_data[first_client_id]['scaler_x'], f)
            logging.info(f"Global scaler saved to: {scaler_path}")

            if clients_data[first_client_id].get('scaler_y') is not None:
                scaler_y_path = scaler_path.replace('scaler_', 'scaler_y_')
                with open(scaler_y_path, 'wb') as f:
                    pickle.dump(clients_data[first_client_id]['scaler_y'], f)
                logging.info(f"Global scaler_y saved to: {scaler_y_path}")

    # Log final metrics
    if results['history'] is not None:
        histories = results['history'] if isinstance(results['history'], list) else [results['history']]

        for i, hist in enumerate(histories):
            if 'metrics_aggregated' in hist and not hist['metrics_aggregated'].empty:
                logging.info(f"\nFL Cycle {i+1} - Aggregated Metrics:")
                logging.info(hist['metrics_aggregated'].to_string())
    else:
        logging.info("\nNo training history (evaluation-only mode)")

    # Cleanup
    del clients_data, partitions, test_data
    gc.collect()


if __name__ == '__main__':
    main()