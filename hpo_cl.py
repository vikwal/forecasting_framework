# Hyperparameter optimization for PyTorch models trained on all parks data at once

import os
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

from utils import preprocessing, tools, hpo, data_cache

optuna.logging.set_verbosity(optuna.logging.INFO)

def main() -> None:
    # Argument parser
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument('-m', '--model', type=str, default='tft', help='Select Model (default: tft)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    parser.add_argument('-i', '--index', type=str, default='', help='Define index')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching for small datasets')
    parser.add_argument('--gpu', type=int, default=None, help='GPU to use (default: auto-select)')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    index = ''
    if args.index:
        index = f'_{args.index}'
    if '/' in args.config:
        config_name = args.config.split('/')[-1]
    else:
        config_name = args.config
    log_file = f'logs/hpo_cl_m-{args.model}_c-{config_name}{index}.log'

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
    logger.info(f"NEW HPO SESSION - Model: {args.model}, Config: {args.config}")
    logger.info("=" * 80)

    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)

    # Load config
    if '.yaml' in args.config:
        args.config = args.config.split('.')[0]
    config = tools.load_config(f'{args.config}.yaml')

    # Override verbose setting
    config['model']['verbose'] = 0

    freq = config['data']['freq']
    params = config['params']
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    config['model']['fl'] = False
    test_end = config.get('data').get('test_end', None)
    test_start = config['data']['test_start']

    logging.info(
        f'HPO for Model: {args.model}, Output dim: {output_dim}, Frequency: {freq}, '
        f'Lookback: {lookback}, Horizon: {horizon}, Step size: {config["model"]["step_size"]}, '
        f'Test start: {test_start}, Test end: {test_end}'
    )

    config['model']['name'] = args.model
    test_start = pd.Timestamp(config.get('data').get('test_start', 0))

    # Get features
    data_dir = config['data']['path']
    features = preprocessing.get_features(config=config)

    # Setup paths
    base_dir = os.path.basename(data_dir)
    target_dir = os.path.join('results', base_dir)
    os.makedirs(target_dir, exist_ok=True)
    study_name_suffix = '_'.join(args.config.split('_')[1:])
    study_name = f'cl_m-{args.model}_out-{output_dim}_freq-{freq}_{study_name_suffix}'
    config['model']['name'] = args.model

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

    # Format objectives list for logging (can't use nested f-strings with backslashes)
    obj_strs = [f"{obj['metric']} ({obj['direction']})" for obj in objectives]
    logging.info(f"Objectives: {obj_strs}")

    # Load preprocessed data with caching
    use_cache = not args.no_cache
    if use_cache:
        logging.info("Creating or loading preprocessed data with caching...")
    else:
        logging.info("Processing data without caching (small dataset mode)...")

    lazy_fold_loader, cache_id = data_cache.create_or_load_preprocessed_data(
        config=config,
        features=features,
        model_name=args.model,
        force_reprocess=False,
        use_cache=use_cache
    )

    if use_cache:
        logging.info(f"Using cached data with ID: {cache_id}")
    else:
        logging.info("Using data directly from memory (no cache)")
    logging.info(f"Available folds: {len(lazy_fold_loader)}")

    # Log fold information
    for i in range(min(3, len(lazy_fold_loader))):
        fold_info = lazy_fold_loader.get_fold_info(i)
        logging.info(f"Fold {i}: {fold_info['train_samples']} train, {fold_info['val_samples']} val samples")

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    logging.info(f"Using device: {device}")

    # Run HPO
    len_trials = len(study.trials)
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    logging.info(f'Starting HPO with {config["hpo"]["trials"] - completed_trials} new trials.')
    logging.info(f'Previous trials: {len_trials} total, {completed_trials} completed, {pruned_trials} pruned.')

    trial_counter = 0
    while completed_trials < config['hpo']['trials']:
        trial = study.ask()
        trial_number = len_trials + trial_counter

        hyperparameters = hpo.get_hyperparameters(
            config=config,
            hpo=True,
            trial=trial
        )

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

        logging.info(f"Complete trial number {completed_trials+1}: {json.dumps(hyperparameters)}")

        try:
            accuracies = []
            for fold_idx in range(len(lazy_fold_loader)):
                # Load fold on demand
                fold = lazy_fold_loader[fold_idx]
                train, val = fold

                # Debug: Log fold data info
                X_train, y_train = train
                if val and val[0] is not None:
                    X_val, y_val = val
                    logging.debug(f"Fold {fold_idx}: Train samples: {len(y_train)}, Val samples: {len(y_val)}")
                else:
                    logging.warning(f"Fold {fold_idx}: No validation data! Train samples: {len(y_train)}")

                # Train with PyTorch
                config['model_name'] = (
                    f'hpo_cl_m-{args.model}_out-{output_dim}_freq-{freq}'
                    f'trial-{trial_number}_fold-{fold_idx}'
                )

                history, model = tools.training_pipeline(
                    train=train,
                    val=val,
                    hyperparameters=hyperparameters,
                    config=config,
                    device=device
                )

                # Metric mapping for compatibility
                metric_map = {
                    'loss': 'val_loss',
                    'val_loss': 'val_loss',
                    'mse': 'val_loss',
                    'rmse': 'val_rmse',
                    'mae': 'val_mae',
                    'r2': 'val_r2',
                    'val_r2': 'val_r2'
                }

                # Collect metrics for all objectives
                fold_metrics = {}
                for obj in objectives:
                    metric_key = obj['metric']

                    # Try to get the metric (with mapping support)
                    if metric_key in history and len(history[metric_key]) > 0:
                        fold_metrics[metric_key] = history[metric_key][-1]
                    elif metric_map.get(metric_key) in history and len(history[metric_map.get(metric_key, '')]) > 0:
                        mapped_key = metric_map.get(metric_key)
                        fold_metrics[metric_key] = history[mapped_key][-1]
                    else:
                        # Better error message showing which metrics are available and non-empty
                        available_metrics = [k for k in history.keys() if len(history[k]) > 0]
                        empty_metrics = [k for k in history.keys() if len(history[k]) == 0]

                        error_msg = f"Could not find metric '{metric_key}' with values. "
                        if available_metrics:
                            error_msg += f"Available (non-empty): {available_metrics}. "
                        if empty_metrics:
                            error_msg += f"Empty: {empty_metrics}."

                        raise ValueError(error_msg)

                # Store metrics for this fold
                if not accuracies:
                    # Initialize dict for each objective
                    accuracies = {obj['metric']: [] for obj in objectives}

                for obj in objectives:
                    accuracies[obj['metric']].append(fold_metrics[obj['metric']])

                # Report intermediate value for pruning (only for single-objective)
                # Note: Pruning is not supported in multi-objective optimization
                if not is_multi_objective:
                    pruning_value = fold_metrics[objectives[0]['metric']]
                    trial.report(pruning_value, step=fold_idx)

                # Log all objective values
                metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in fold_metrics.items()])
                logging.info(
                    f'Processed fold {fold_idx + 1}/{len(lazy_fold_loader)}, {metrics_str}'
                )

                # Clean up model
                del model, history
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

                    logging.info(f'Fold averages: {accuracies}')
                    study.tell(trial, values=average_values)
                    logging.info(
                        f'Trial number {trial_number+1} completed with: {", ".join(metrics_summary)}'
                    )
                else:
                    # Single objective: simple average
                    metric_name = objectives[0]['metric']
                    fold_values = accuracies[metric_name]
                    average_accuracy = np.mean(fold_values)

                    logging.info(f'Accuracies for the folds: {fold_values}')
                    study.tell(trial, average_accuracy)
                    logging.info(
                        f'Trial number {trial_number+1} completed with average '
                        f'{metric_name}: {average_accuracy:.4f}'
                    )

                completed_trials += 1
                logging.info(f'Progress: {completed_trials}/{config["hpo"]["trials"]} successful trials completed.')

        except optuna.TrialPruned:
            logging.info(f'Trial number {trial_number+1} was pruned by Optuna')
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            trial_counter += 1

        except KeyboardInterrupt:
            logging.warning(f'Trial number {trial_number+1} interrupted by user. Marking as failed.')
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise

        except Exception as e:
            logging.error(f'Trial number {trial_number+1} failed with error: {str(e)}. Marking as failed.')
            logging.exception("Full traceback:")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            trial_counter += 1
            raise  # Continue with next trial instead of crashing

        trial_counter += 1

    logging.info("=" * 80)
    logging.info("HPO COMPLETED")
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

        # User can manually select from Pareto front or use first trial as default
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

if __name__ == '__main__':
    main()
