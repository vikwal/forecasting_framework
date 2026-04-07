# Federated learning hyperparameter optimization for PyTorch models

import os
import copy
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import gc

import torch
import optuna

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

# Reduce CUDA memory fragmentation from many concurrent actors with varying tensor sizes
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from sklearn.preprocessing import StandardScaler
from utils import preprocessing, tools, hpo, federated, models

optuna.logging.set_verbosity(optuna.logging.INFO)


def main() -> None:
    # Argument parser
    parser = argparse.ArgumentParser(description="Federated Learning Hyperparameter Optimization")
    parser.add_argument('-m', '--model', type=str, default='tft', help='Select Model (default: tft)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    parser.add_argument('-s', '--suffix', type=str, default='', help='Define suffix')
    parser.add_argument('--gpu', type=int, default=None, help='GPU to use (default: auto-select)')
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
    
    # Remove 'config_' prefix if present
    if config_name.startswith('config_'):
        config_name = config_name[7:]  # Remove 'config_' (7 characters)
    log_file = f'logs/hpo_fl_m-{args.model}_c-{config_name}{suffix}.log'

    # Configure logging
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ],
        force=True
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"NEW HPO FL SESSION - Model: {args.model}, Config: {args.config}")
    logger.info("=" * 80)

    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)

    # Load config
    config = tools.load_config(f'{args.config}.yaml')

    # Override verbose settings for HPO
    config['model']['verbose'] = 0
    config['fl']['verbose'] = False

    # Extract FL strategy for HPO (used in study name and log file)
    fl_strategy = config['fl']['strategy']

    # Update log file name to include strategy (now that config is loaded)
    log_file_new = f'logs/hpo_fl_a-{fl_strategy}_m-{args.model}_c-{config_name}{suffix}.log'
    if log_file_new != log_file:
        # Replace the file handler with the updated filename
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        for handler in logging.getLogger().handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logging.getLogger().removeHandler(handler)
                handler.close()
        new_handler = logging.FileHandler(log_file_new, mode='a')
        new_handler.setFormatter(formatter)
        logging.getLogger().addHandler(new_handler)

    freq = config['data']['freq']
    params = config['params']
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    config['model']['fl'] = True  # Usually False for HPO to reduce variance
    test_start = pd.Timestamp(config.get('data').get('test_start', 0))

    logging.info(
        f'HPO for Federated Model: {args.model}, Output dim: {output_dim}, Frequency: {freq}, '
        f'Lookback: {lookback}, Horizon: {horizon}, Step size: {config["model"]["step_size"]}'
    )

    config['model']['name'] = args.model

    # Get features
    data_dir = config['data']['path']
    features = preprocessing.get_features(config=config)

    # Setup paths
    base_dir = os.path.basename(data_dir)
    target_dir = os.path.join('results', base_dir)
    os.makedirs(target_dir, exist_ok=True)

    study_name_suffix = config_name
    if args.suffix:
        study_name_suffix += f'_{args.suffix}'
    study_name = f'fl_a-{fl_strategy}_m-{args.model}_out-{output_dim}_freq-{freq}_{study_name_suffix}'
    logging.info(f'Starting HPO for Federated Study: {study_name}')

    # Create study with config (supports both single and multi-objective)
    pruning_config = config.get('hpo', {}).get('pruning', {})
    study = hpo.create_or_load_study(
        config['hpo']['studies_path'],
        study_name,
        pruning_config=pruning_config,
        config=config  # Pass config for flexible objective detection
    )

    # Get objectives from config
    objectives, is_multi_objective = hpo.get_objectives_from_config(config)
    logging.info(f"HPO Mode: {'Multi-Objective' if is_multi_objective else 'Single-Objective'}")

    # Format objectives list for logging
    obj_strs = [f"{obj['metric']} ({obj['direction']})" for obj in objectives]
    logging.info(f"Objectives: {obj_strs}")

    # --- LOAD TURBINE ASSIGNMENTS TO ENSURE CONSISTENT SEEDS ---
    station_assignments_path = 'data/station_turbine_assignments.csv'
    if os.path.exists(station_assignments_path):
        station_to_seed = {}
        base_seed = config['params'].get('random_seed', 42)
        for idx, file_id in enumerate(config['data']['files']):
            station_to_seed[file_id] = base_seed + idx
        logging.debug(f"Loaded turbine assignments for {len(station_to_seed)} stations from config")
    else:
        logging.warning(f"Station turbine assignments file not found: {station_assignments_path}")
        station_to_seed = None

    # --- LOAD AND PREPARE CLIENT DATA WITH CLIENT-SPECIFIC SCALERS ---
    logging.info("Loading and preparing client data...")
    clients_data = {}

    for client_id, station_ids in tqdm(config['fl']['clients'].items(), desc="Loading clients", unit="client"):
        logging.debug(f'Loading data for client: {client_id} with stations: {station_ids}')

        # Create client-specific config
        client_config = copy.deepcopy(config)
        client_config['data']['files'] = station_ids

        # Log seed assignments for debugging
        if station_to_seed:
            for station_id in station_ids:
                if station_id in station_to_seed:
                    logging.debug(f"Station {station_id} will use seed: {station_to_seed[station_id]}")

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
            'scaler_x': None
        }

    logging.info(f"Loaded data for {len(clients_data)} clients.")

    # --- FIT SCALERS PER CLIENT ---
    logging.info("Fitting scalers per client...")
    target_col = config['data']['target_col']
    fit_scaler_y = target_col != 'power'  # power is pre-normalised to [0, 1]

    for client_id, client_info in clients_data.items():
        logging.debug(f"Fitting scaler for client {client_id}...")
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        for key, df in tqdm(client_info['dfs'].items(), desc=f"Fitting Scaler for {client_id}"):
            df_temp = df.copy()

            # For non-TFT models, create lag features before fitting scaler
            if config['model']['name'] not in ['tft', 'stemgnn']:
                for col in features['observed']:
                    all_observed_cols = [new_col for new_col in df_temp.columns if new_col == col or new_col.startswith(col + '_lag_')]
                    for new_col in all_observed_cols:
                        df_temp = preprocessing.lag_features(
                            data=df_temp,
                            lookback=config['model']['lookback'],
                            horizon=config['model']['horizon'],
                            lag_in_col=config['data']['lag_in_col'],
                            target_col=new_col
                        )
                        if new_col != target_col and new_col not in features['known']:
                            df_temp.drop(new_col, axis=1, inplace=True, errors='ignore')

            t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
            df_train, _ = preprocessing.split_data(
                data=df_temp,
                train_frac=config['data']['train_frac'],
                test_start=pd.Timestamp(config['data']['test_start'], tz='UTC'),
                test_end=pd.Timestamp(config['data'].get('test_end', '2099-12-31'), tz='UTC'),
                t_0=t_0
            )

            # Fit Y scaler on target before dropping it
            if fit_scaler_y and target_col in df_train.columns:
                scaler_y.partial_fit(df_train[[target_col]].values)

            df_train_x = df_train.drop(columns=[target_col], errors='ignore')
            scaler_x.partial_fit(df_train_x.values)

            del df_temp, df_train, df_train_x
            gc.collect()

        client_info['scaler_x'] = scaler_x
        client_info['scaler_y'] = scaler_y if fit_scaler_y else None
        logging.debug(f"Scaler fitted for client {client_id}.")

    # --- AGGREGATE SCALERS IF global_scaler IS ENABLED ---
    if config['fl'].get('global_scaler', False):
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

        # Aggregate scaler_y
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

    # --- PROCESS DATA, APPLY min_train_date, AND CREATE K-FOLD PARTITIONS ---
    # Mirrors hpo_cl.py / data_cache.py: data before min_train_date is always included
    # in every fold's training set (as a mandatory "min_block"), so all folds have
    # comparable training size and fold 0 is not disadvantaged vs. later folds.
    n_splits = config['hpo']['kfolds']
    min_train_date = config['hpo'].get('min_train_date', None)
    logging.info(
        f"Processing data and creating {n_splits}-fold partitions "
        f"(min_train_date={min_train_date}) for {len(clients_data)} clients..."
    )
    client_ids_ordered = []
    client_kfolds = {}  # {client_id: [((X_tr, y_tr), (X_val, y_val)), ...]}

    for client_id, client_info in clients_data.items():
        logging.debug(f"Processing data for client: {client_id}")

        # Set scalers into config so preprocessing.pipeline() picks them up
        # (same pattern as tools.create_data_generator)
        config['scaler_x'] = client_info['scaler_x']
        if client_info.get('scaler_y'):
            config['scaler_y'] = client_info['scaler_y']
        else:
            config.pop('scaler_y', None)

        # Run preprocessing.pipeline() per station to obtain prepared_data including
        # index_train, which kfolds_with_per_file_min_train_len needs for date-based splitting
        client_prepared = []
        for key, df in client_info['dfs'].items():
            prepared_data, _ = preprocessing.pipeline(
                data=df,
                config=config,
                known_cols=features['known'],
                observed_cols=features['observed'],
                static_cols=features['static'],
                target_col=config['data']['target_col']
            )
            if prepared_data is not None and len(prepared_data.get('y_train', [])) > 0:
                client_prepared.append(prepared_data)

        if not client_prepared:
            logging.warning(f"Client {client_id}: no training data after preprocessing — skipping.")
            continue

        # Build k-folds with min_train_date — identical approach to hpo_cl.py / data_cache.py
        fold_data = hpo.kfolds_with_per_file_min_train_len(
            prepared_datasets=client_prepared,
            n_splits=n_splits,
            val_split=config['hpo']['val_split'],
            min_train_date=min_train_date
        )
        client_kfolds[client_id] = fold_data
        client_ids_ordered.append(client_id)

        del client_prepared
        gc.collect()

    # Clean up scaler keys added to config
    config.pop('scaler_x', None)
    config.pop('scaler_y', None)

    # Restructure into kfolds_partitions[fold_idx] = [(X_tr, y_tr, X_val, y_val), ...] per client
    kfolds_partitions = []
    for fold_idx in range(n_splits):
        fold_clients = []
        for client_id in client_ids_ordered:
            (X_tr, y_tr), (X_val, y_val) = client_kfolds[client_id][fold_idx]
            fold_clients.append((X_tr, y_tr, X_val, y_val))
        kfolds_partitions.append(fold_clients)

    logging.info(f"Created {len(kfolds_partitions)} folds, each with {len(client_ids_ordered)} client partitions")

    # Log fold sizes for first client
    for fold_idx, fold_partitions in enumerate(kfolds_partitions):
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = fold_partitions[0]
        logging.info(
            f"  Fold {fold_idx}: {len(y_train_fold)} train, {len(y_val_fold)} val samples"
            f" (client {client_ids_ordered[0]})"
        )

    del client_kfolds
    gc.collect()

    # --- LOAD VAL SET FROM val_files (optional) ---
    # Val stations are loaded for the full TRAINING period (before test_start).
    # Split into n_splits+1 equal chunks; fold k uses chunk k+1, consistent with
    # the approach in data_cache.py (_replace_val_with_val_files).
    # Scaler comes from client 0 (= global scaler when global_scaler=True).
    val_files_chunks = None  # list of (X_val_chunk, y_val_chunk) per fold
    if config['data'].get('val_files'):
        logging.info("val_files found — loading val stations for training period (k-fold aligned).")
        first_client_id_for_scaler = client_ids_ordered[0]
        val_scaler_x = clients_data[first_client_id_for_scaler]['scaler_x']
        val_scaler_y = clients_data[first_client_id_for_scaler].get('scaler_y')

        val_dfs = preprocessing.get_data(
            data_dir=data_dir,
            config=config,
            freq=freq,
            features=features,
            files_key='val_files'
        )
        val_generator = tools.create_data_generator(
            val_dfs, config, features,
            scaler_x=val_scaler_x,
            scaler_y=val_scaler_y
        )
        X_val_full, y_val_full, _, _, _, _ = tools.combine_datasets_efficiently(val_generator)

        n_val = len(y_val_full)
        if n_val > 0:
            n_chunks = n_splits + 1
            chunk_size = n_val // n_chunks
            val_files_chunks = []
            for k in range(n_splits):
                start = (k + 1) * chunk_size
                end = (k + 2) * chunk_size if k < n_splits - 1 else n_val
                if isinstance(X_val_full, dict):
                    X_chunk = {key: arr[start:end] for key, arr in X_val_full.items()}
                else:
                    X_chunk = X_val_full[start:end]
                val_files_chunks.append((X_chunk, y_val_full[start:end]))
            logging.info(
                f"Val sequences split into {n_splits} chunks (~{chunk_size} samples each) "
                f"from {len(val_dfs)} val stations."
            )
        else:
            logging.warning("val_files produced no training-period samples — falling back to k-fold val.")

    # --- GET FEATURE DIM ---
    X_sample = kfolds_partitions[0][0][0]  # fold 0, first client, X_train
    feature_dim = tools.get_feature_dim(X_sample)
    config['model']['feature_dim'] = feature_dim

    if isinstance(X_sample, dict):
        if 'known' in X_sample:
            logging.info(f'Data shape: X_train known: {X_sample["known"].shape}')
        if 'static' in X_sample:
            logging.info(f'Data shape: X_train static: {X_sample["static"].shape}')
        if 'observed' in X_sample:
            logging.info(f'Data shape: X_train observed: {X_sample["observed"].shape}')
    else:
        logging.info(f'Data shape: X_train: {X_sample.shape}')

    # --- HPO LOOP ---
    len_trials = len(study.trials)
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    logging.info(f'Starting Federated HPO with {config["hpo"]["trials"] - completed_trials} new trials.')
    logging.info(f'Previous trials: {len_trials} total, {completed_trials} completed, {pruned_trials} pruned.')

    # Get personalization setting from HPO FL config or FL config
    personalize = config['fl'].get('personalize', False)

    trial_counter = 0
    while completed_trials < config['hpo']['trials']:
        trial = study.ask()
        trial_number = len_trials + trial_counter

        hyperparameters = hpo.get_hyperparameters(
            config=config,
            hpo=True,
            trial=trial
        )

        # Add FL-specific hyperparameters
        hyperparameters['personalization'] = personalize

        # Check for duplicate parameters
        existing_params = [t.params for t in study.trials]
        current_params = trial.params
        param_count = sum(1 for params in existing_params if params == current_params)

        if param_count > 1:
            logging.warning(
                f"Trial number {trial_number}: Duplicate parameters detected "
                f"(found {param_count} times), marking as failed..."
            )
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue

        logging.info(f"Trial {completed_trials+1}/{config['hpo']['trials']}: {json.dumps(hyperparameters)}")

        try:
            accuracies = []

            for fold_idx, fold_partitions in enumerate(kfolds_partitions):
                # Convert fold partitions list to dict with client IDs for run_simulation.
                # Keep the k-fold val splits on the clients (useful for local early stopping).
                # When val_files is set and personalize=False, global evaluation happens once
                # server-side via global_val_data instead of redundantly on every client.
                partitions_for_fold = {}
                for i, client_id in enumerate(client_ids_ordered):
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold = fold_partitions[i]
                    partitions_for_fold[client_id] = (X_train_fold, y_train_fold, X_val_fold, y_val_fold)

                global_val_data = None
                if val_files_chunks is not None and not personalize:
                    global_val_data = val_files_chunks[fold_idx]

                # Config model name for callbacks
                config['model_name'] = (
                    f'hpo_fl_m-{args.model}_out-{output_dim}_freq-{freq}_'
                    f'trial-{trial_number}_fold-{fold_idx}'
                )

                # Suppress FL/Ray logs during simulation to keep HPO output clean
                root_logger = logging.getLogger()
                original_level = root_logger.level
                root_logger.setLevel(logging.WARNING)
                try:
                    history, _ = federated.run_simulation(
                        partitions=partitions_for_fold,
                        hyperparameters=hyperparameters,
                        config=config,
                        global_val_data=global_val_data
                    )
                finally:
                    root_logger.setLevel(original_level)

                # Extract metrics from the last round's aggregated metrics
                metrics_df = history['metrics_aggregated']

                # Collect metrics for all objectives
                fold_metrics = {}
                metric_map = {
                    'loss': 'val_loss',
                    'val_loss': 'val_loss',
                    'mse': 'val_loss',
                    'rmse': 'val_rmse',
                    'mae': 'val_mae',
                    'r2': 'val_r^2',
                    'val_r2': 'val_r^2',
                    'val_rmse': 'val_rmse',
                    'val_mae': 'val_mae',
                    'val_r^2': 'val_r^2',
                }

                for obj in objectives:
                    metric_key = obj['metric']

                    # Try the metric directly
                    if metric_key in metrics_df.columns:
                        fold_metrics[metric_key] = metrics_df[metric_key].iloc[-1]
                    elif metric_map.get(metric_key) in metrics_df.columns:
                        mapped_key = metric_map.get(metric_key)
                        fold_metrics[metric_key] = metrics_df[mapped_key].iloc[-1]
                    else:
                        available_metrics = list(metrics_df.columns)
                        raise ValueError(
                            f"Could not find metric '{metric_key}' in aggregated metrics. "
                            f"Available: {available_metrics}"
                        )

                # Store metrics for this fold
                if not accuracies:
                    # Initialize dict for each objective
                    accuracies = {obj['metric']: [] for obj in objectives}

                for obj in objectives:
                    accuracies[obj['metric']].append(fold_metrics[obj['metric']])

                # Report intermediate value for pruning (only for single-objective)
                if not is_multi_objective:
                    pruning_value = fold_metrics[objectives[0]['metric']]
                    trial.report(pruning_value, step=fold_idx)

                # Log fold result with key metrics (val_rmse + val_r^2)
                rmse_val = metrics_df['val_rmse'].iloc[-1] if 'val_rmse' in metrics_df.columns else None
                r2_val = metrics_df['val_r^2'].iloc[-1] if 'val_r^2' in metrics_df.columns else None
                fold_log_parts = [f'Fold {fold_idx + 1}/{len(kfolds_partitions)}']
                if rmse_val is not None:
                    fold_log_parts.append(f'val_rmse: {rmse_val:.4f}')
                if r2_val is not None:
                    fold_log_parts.append(f'val_r^2: {r2_val:.4f}')
                logging.info(', '.join(fold_log_parts))

                # Clean up
                del history
                gc.collect()

                # Check for pruning (only for single-objective)
                if not is_multi_objective and trial.should_prune():
                    raise optuna.TrialPruned()

            else:
                # All folds completed - calculate averages and report
                if is_multi_objective:
                    # Multi-objective: calculate average for each objective
                    average_values = []
                    metrics_summary = []
                    for obj in objectives:
                        metric_name = obj['metric']
                        avg_value = np.mean(accuracies[metric_name])
                        average_values.append(avg_value)
                        metrics_summary.append(f"{metric_name}: {avg_value:.4f}")

                    logging.info(f'Fold averages: { {k: [round(v, 4) for v in vals] for k, vals in accuracies.items()} }')
                    study.tell(trial, values=average_values)
                    logging.info(
                        f'Trial number {trial_number+1} completed with: {", ".join(metrics_summary)}'
                    )
                else:
                    # Single objective: simple average
                    metric_name = objectives[0]['metric']
                    fold_values = accuracies[metric_name]
                    average_accuracy = np.mean(fold_values)

                    logging.info(f'Fold values: {[round(v, 4) for v in fold_values]}')
                    study.tell(trial, average_accuracy)
                    logging.info(
                        f'Trial number {trial_number+1} completed with average '
                        f'{metric_name}: {average_accuracy:.4f}'
                    )

                logging.info(f'Progress: {completed_trials+1}/{config["hpo"]["trials"]} successful trials completed.')
                completed_trials += 1

        except optuna.TrialPruned:
            logging.info(f'Trial number {trial_number+1} was pruned by Optuna')
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)

        except KeyboardInterrupt:
            logging.warning(f'Trial number {trial_number+1} interrupted by user. Marking as failed.')
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise

        except Exception as e:
            logging.error(f'Trial number {trial_number+1} failed with error: {str(e)}. Marking as failed.')
            logging.exception("Full traceback:")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise  # Re-raise to see the full error; remove this for production to continue with next trial

        trial_counter += 1

    # --- FINAL REPORT ---
    logging.info("=" * 80)
    logging.info("HPO FL COMPLETED")
    logging.info("=" * 80)

    if is_multi_objective:
        # Multi-objective: show Pareto front
        best_trials = study.best_trials
        logging.info(f"Pareto Front: {len(best_trials)} non-dominated solutions")
        logging.info("-" * 80)

        for i, trial in enumerate(best_trials[:10]):  # Show top 10
            values_str = ", ".join([
                f"{obj['metric']}: {trial.values[j]:.6f}"
                for j, obj in enumerate(objectives)
            ])
            logging.info(f"  Solution {i+1}: {values_str}")
            if i == 0:  # Show params for first solution
                logging.info(f"  Params: {json.dumps(trial.params, indent=2)}")

        if len(best_trials) > 10:
            logging.info(f"  ... and {len(best_trials) - 10} more solutions")

        logging.info("-" * 80)
        logging.info("Note: Multiple optimal solutions found. Using first solution as default.")
        default_trial = best_trials[0]
        values_str = ", ".join([
            f"{obj['metric']}: {default_trial.values[j]:.6f}"
            for j, obj in enumerate(objectives)
        ])
        logging.info(f"Default: {values_str}")
        logging.info(f"Params: {json.dumps(default_trial.params, indent=2)}")
    else:
        # Single objective: simple best trial
        logging.info(f"Best trial: {study.best_trial.number}")
        logging.info(f"Best {objectives[0]['metric']}: {study.best_value:.6f}")
        logging.info(f"Best params: {json.dumps(study.best_params, indent=2)}")

    logging.info("=" * 80)

    # Cleanup
    del clients_data, kfolds_partitions
    gc.collect()

if __name__ == '__main__':
    main()