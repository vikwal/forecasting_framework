# Hyperparameter optimization for model which is trained on all parks data at once

# Suppress TensorFlow logging for cleaner HPO logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import gc

# Import TensorFlow and disable verbose logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# Suppress additional TensorFlow info messages
tf.autograph.set_verbosity(0)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

import optuna

from utils import preprocessing, tools, hpo, data_cache

optuna.logging.set_verbosity(optuna.logging.INFO)

def main() -> None:
    tools.initialize_gpu()
    # argument parser
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Tensorflow/Keras")
    parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    parser.add_argument('-i', '--index', type=str, default='', help='Define index')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching for small datasets')
    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    index = ''
    if args.index:
        index = f'_{args.index}'
    log_file = f'logs/hpo_cl_m-{args.model}_c-{args.config}{index}.log'

    # Configure logging with append mode
    # Clear any existing handlers first
    logging.getLogger().handlers.clear()

    logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
        logging.FileHandler(log_file, mode='a'),  # Append mode instead of overwrite
        logging.StreamHandler()
        ],
        force=True)  # Force reconfiguration of logging

    # Add session separator to distinguish different HPO runs
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"NEW HPO SESSION STARTED - Model: {args.model}, Config: {args.config}")
    logger.info("=" * 80)
    # create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)
    # read config
    if '.yaml' in args.config:
        args.config = args.config.split('.')[0]
    config = tools.load_config(f'configs/{args.config}.yaml')

    # Override verbose setting to suppress training progress bars in HPO logs
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
    logging.info(f'HPO for Model: {args.model}, Output dim: {output_dim}, Frequency: {freq}, Lookback: {lookback}, Horizon: {horizon}, Step size: {config["model"]["step_size"]}, Test start: {test_start}, Test end: {test_end}')
    config['model']['name'] = args.model
    test_start = pd.Timestamp(config.get('data').get('test_start', 0))
    # get observed, known and static features
    data_dir = config['data']['path']
    features = preprocessing.get_features(config=config)
    # get the right dataset name
    base_dir = os.path.basename(data_dir)
    target_dir = os.path.join('results', base_dir)
    os.makedirs(target_dir, exist_ok=True)
    study_name_suffix = '_'.join(args.config.split('_')[1:3])
    study_name = f'cl_m-{args.model}_out-{output_dim}_freq-{freq}_{study_name_suffix}'
    config['model']['name'] = args.model

    # Create study for hyperparameter optimization
    pruning_config = config.get('hpo', {}).get('pruning', {})
    study = hpo.create_or_load_study(config['hpo']['studies_path'], study_name,
                                    direction='minimize', pruning_config=pruning_config)

    # Create or load preprocessed data with caching (MEMORY OPTIMIZED)
    use_cache = not args.no_cache
    if use_cache:
        logging.info("Creating or loading preprocessed data with caching...")
    else:
        logging.info("Processing data without caching (small dataset mode)...")

    lazy_fold_loader, cache_id = data_cache.create_or_load_preprocessed_data(
        config=config,
        features=features,
        model_name=args.model,
        force_reprocess=False,  # Set to True to force reprocessing
        use_cache=use_cache
    )

    if use_cache:
        logging.info(f"Using cached data with ID: {cache_id}")
    else:
        logging.info("Using data directly from memory (no cache)")
    logging.info(f"Available folds: {len(lazy_fold_loader)}")

    # Log fold information
    for i in range(min(3, len(lazy_fold_loader))):  # Show info for first 3 folds
        fold_info = lazy_fold_loader.get_fold_info(i)
        logging.info(f"Fold {i}: {fold_info['train_samples']} train, {fold_info['val_samples']} val samples")

    # Run hyperparameter optimization
    len_trials = len(study.trials)
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    logging.info(f'Starting HPO with {config["hpo"]["trials"] - completed_trials} new trials.')
    logging.info(f'Previous trials: {len_trials} total, {completed_trials} completed, {pruned_trials} pruned.')

    trial_counter = 0
    while completed_trials < config['hpo']['trials']:
        trial = study.ask()
        trial_number = len_trials + trial_counter
        hyperparameters = hpo.get_hyperparameters(config=config,
                                                  hpo=True,
                                                  trial=trial)
        # Check for duplicate parameters from other trials (including running ones)
        existing_params = [t.params for t in study.trials]
        current_params = trial.params
        # Count how many trials have these exact parameters
        param_count = sum(1 for params in existing_params if params == current_params)
        if param_count > 1:  # More than just the current trial
            logging.warning(f"Trial number {trial_number}: Duplicate parameters detected (found {param_count} times), marking as failed...")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue  # Kein trial_counter += 1, damit Nummerierung stimmt
        logging.info(f"Complete trial number {completed_trials+1}: {json.dumps(hyperparameters)}")
        try:
            accuracies = []
            for fold_idx in range(len(lazy_fold_loader)):
                # Load fold on demand (lazy loading)
                fold = lazy_fold_loader[fold_idx]
                train, val = fold
                # config model name relevant for callbacks
                config['model_name'] = f'hpo_cl_m-{args.model}_out-{output_dim}_freq-{freq}_trial-{trial_number}_fold-{fold_idx}'
                history, _ = tools.training_pipeline(train=train,
                                                     val=val,
                                                     hyperparameters=hyperparameters,
                                                     config=config)
                fold_accuracy = history.history[config['hpo']['metric']][-1]
                accuracies.append(fold_accuracy)
                # Report intermediate value für MedianPruner
                trial.report(fold_accuracy, step=fold_idx)
                logging.info(f'Processed fold {fold_idx + 1}/{len(lazy_fold_loader)}, {config["hpo"]["metric"]}: {fold_accuracy:.4f}')
                if trial.should_prune():
                    raise optuna.TrialPruned()
            else:
                # Dieser Block wird NUR ausgeführt, wenn die Fold-Schleife NICHT durch break verlassen wurde
                logging.info(f'Accuracies for the folds: {accuracies}')
                average_accuracy = sum(accuracies) / len(accuracies)
                study.tell(trial, average_accuracy)
                completed_trials += 1
                logging.info(f'Trial number {trial_number+1} completed with average {config["hpo"]["metric"]}: {average_accuracy:.4f}')
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
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise

        trial_counter += 1

if __name__ == '__main__':
    main()