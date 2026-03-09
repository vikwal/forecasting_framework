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

from sklearn.preprocessing import StandardScaler
from utils import preprocessing, tools, hpo, federated, models

optuna.logging.set_verbosity(optuna.logging.INFO)


def main() -> None:
    # Argument parser
    parser = argparse.ArgumentParser(description="Federated Learning Hyperparameter Optimization")
    parser.add_argument('-m', '--model', type=str, default='tft', help='Select Model (default: tft)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    parser.add_argument('-i', '--index', type=str, default='', help='Define index')
    parser.add_argument('--gpu', type=int, default=None, help='GPU to use (default: auto-select)')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    index = ''
    if args.index:
        index = f'_{args.index}'
    if '.yaml' in args.config:
        args.config = args.config.split('.')[0]
    if '/' in args.config:
        config_name = args.config.split('/')[-1]
        if len(config_name.split('_')) == 3:
            config_name = '_'.join(config_name.split('_')[1:])
    else:
        config_name = args.config
    log_file = f'logs/hpo_fl_m-{args.model}_c-{config_name}{index}.log'

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
    log_file_new = f'logs/hpo_fl_a-{fl_strategy}_m-{args.model}_c-{config_name}{index}.log'
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
    config['model']['fl'] = True
    config['model']['shuffle'] = False  # Usually False for HPO to reduce variance
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
    suffix = ''
    if config['fl'].get('personalize', False):
        suffix += '_pers'
    study_name = f'fl_a-{fl_strategy}_m-{args.model}_out-{output_dim}_freq-{freq}_{study_name_suffix}{suffix}'
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

    for client_id, station_ids in config['fl']['clients'].items():
        logging.info(f'Loading data for client: {client_id} with stations: {station_ids}')

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
    for client_id, client_info in clients_data.items():
        logging.debug(f"Fitting scaler for client {client_id}...")
        scaler_x = StandardScaler()

        for key, df in tqdm(client_info['dfs'].items(), desc=f"Fitting Scaler for {client_id}"):
            df_temp = df.copy()
            target_col = config['data']['target_col']

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

            # Drop target column
            if target_col in df_temp.columns:
                df_temp.drop(target_col, axis=1, inplace=True)

            t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
            df_train, _ = preprocessing.split_data(
                data=df_temp,
                train_frac=config['data']['train_frac'],
                test_start=pd.Timestamp(config['data']['test_start'], tz='UTC'),
                test_end=pd.Timestamp(config['data'].get('test_end', '2099-12-31'), tz='UTC'),
                t_0=t_0
            )
            scaler_x.partial_fit(df_train.values)

            del df_temp, df_train
            gc.collect()

        client_info['scaler_x'] = scaler_x
        logging.debug(f"Scaler fitted for client {client_id}.")

    # --- AGGREGATE SCALERS IF global_scaler IS ENABLED ---
    if config['fl'].get('global_scaler', False):
        logging.info("Aggregating local scalers into a global scaler...")
        client_stats = []
        for client_id, client_info in clients_data.items():
            local_scaler = client_info['scaler_x']
            client_stats.append({
                'n': local_scaler.n_samples_seen_,
                'mu': local_scaler.mean_,
                'var': local_scaler.var_
            })

        mu_global, std_global = federated.aggregate_scalers(client_stats)

        global_scaler = StandardScaler()
        global_scaler.mean_ = mu_global
        global_scaler.scale_ = std_global
        global_scaler.var_ = std_global ** 2
        global_scaler.n_samples_seen_ = sum(s['n'] for s in client_stats)
        global_scaler.n_features_in_ = len(mu_global)

        for client_id, client_info in clients_data.items():
            client_info['scaler_x'] = global_scaler

        logging.info(f"Global scaler applied to all {len(clients_data)} clients.")

    # --- PROCESS DATA WITH SCALER AND CREATE TRAINING PARTITIONS ---
    logging.info("Processing data and creating training partitions...")
    client_partitions = {}  # {client_id: (X_train, y_train)}
    client_ids_ordered = []

    for client_id, client_info in clients_data.items():
        logging.info(f"Processing data for client: {client_id}")

        data_generator = tools.create_data_generator(
            client_info['dfs'],
            config,
            features,
            scaler_x=client_info['scaler_x']
        )
        X_train, y_train, X_test, y_test, _ = tools.combine_datasets_efficiently(data_generator)

        # For HPO we only need training data
        client_partitions[client_id] = (X_train, y_train)
        client_ids_ordered.append(client_id)

        logging.info(f"  Client {client_id}: {len(y_train)} train samples")

        del X_test, y_test, data_generator
        gc.collect()

    # --- GET FEATURE DIM ---
    first_client_id = client_ids_ordered[0]
    X_sample = client_partitions[first_client_id][0]
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

    # --- CREATE K-FOLD PARTITIONS ---
    n_splits = config['hpo']['kfolds']
    logging.info(f"Creating {n_splits}-fold partitions for {len(client_partitions)} clients...")

    # Prepare partitions as list of (X_train, y_train) tuples for get_kfolds_partitions
    partitions_list = [client_partitions[cid] for cid in client_ids_ordered]
    kfolds_partitions = federated.get_kfolds_partitions(n_splits=n_splits, partitions=partitions_list)

    logging.info(f"Created {len(kfolds_partitions)} folds, each with {len(client_ids_ordered)} client partitions")

    # Log fold sizes for first client
    for fold_idx, fold_partitions in enumerate(kfolds_partitions):
        first_part = fold_partitions[0]  # First client's partition
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = first_part
        train_len = len(y_train_fold)
        val_len = len(y_val_fold)
        logging.info(f"  Fold {fold_idx}: {train_len} train, {val_len} val samples (client {client_ids_ordered[0]})")

    # Free original partitions - we now have them in k-fold form
    del client_partitions, partitions_list
    gc.collect()

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
                # Convert fold partitions list to dict with client IDs for run_simulation
                partitions_for_fold = {}
                for i, client_id in enumerate(client_ids_ordered):
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold = fold_partitions[i]
                    partitions_for_fold[client_id] = (X_train_fold, y_train_fold, X_val_fold, y_val_fold)

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
                        config=config
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