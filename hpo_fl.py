# Federated learning hyperparameter optimization

import os
import json
import optuna
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from utils import tools, preprocessing, federated, hpo

optuna.logging.set_verbosity(optuna.logging.INFO)


def main() -> None:
    logger = logging.getLogger(__name__)
    # argument parser
    parser = argparse.ArgumentParser(description="Federated Learning HPO with Tensorflow/Keras")
    parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    log_file = f'logs/hpo_fl_m-{args.model}.log'
    logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
        ])
    # create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)
    # read config
    if '.yaml' in args.config:
        args.config = args.config.split('.')[0]
    config = tools.load_config(f'configs/{args.config}.yaml')
    freq = config['data']['freq']
    params = config['params']
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    logging.info(f'HPO for Federated Model: {args.model}, Output dim: {output_dim}, Frequency: {freq}, Lookback: {lookback}, Horizon: {horizon}')
    config['model']['name'] = args.model
    config['model']['shuffle'] = False  # Usually False for HPO to reduce variance
    config['model']['fl'] = True
    test_start = pd.Timestamp(config.get('data').get('test_start', 0))

    # get observed, known and static features
    features = preprocessing.get_features(config=config)

    # get the right dataset name
    base_dir = os.path.basename(config['data']['path'])
    target_dir = os.path.join('results', base_dir)
    os.makedirs(target_dir, exist_ok=True)

    # Create study name
    suffix = ''
    if config['fl']['personalize']:
        suffix += '_pers'
    study_name = f'fl_a-{config["fl"]["strategy"]}_m-{args.model}_out-{output_dim}_freq-{freq}{suffix}'
    logging.info(f'Starting HPO for Federated Study: {study_name}')

    # Create study for hyperparameter optimization
    study = hpo.create_or_load_study(config['hpo']['studies_path'], study_name, direction='minimize')

    # load and prepare training and test data using federated data structure
    clients_data = federated.load_federated_data(config=config,
                                               freq=freq,
                                               features=features,
                                               target_col=config['data']['target_col'])
    # load and prepare training and test data using federated data structure
    clients_data = federated.load_federated_data(config=config,
                                               freq=freq,
                                               features=features,
                                               target_col=config['data']['target_col'])

    # Prepare k-folds for cross-validation (federated version)
    client_kfolds = {}

    # Process each client's data and create k-folds
    for client_id, client_files in clients_data.items():
        logging.info(f'Creating k-folds for client: {client_id}')

        X_train_client = None
        y_train_client = None

        # Process all files for this client and combine them
        for file_key, df in client_files.items():
            logging.info(f'Preprocessing {file_key} for client {client_id} (HPO).')
            prepared_data, _ = preprocessing.pipeline(data=df,
                                                config=config,
                                                known_cols=features['known'],
                                                observed_cols=features['observed'],
                                                static_cols=features['static'],
                                                test_start=test_start,
                                                target_col=config['data']['target_col'])

            # Combine data within client if multiple files
            if X_train_client is None:
                X_train_client = prepared_data['X_train']
                y_train_client = prepared_data['y_train']
            else:
                X_train_client = tools.concatenate_data(old=X_train_client, new=prepared_data['X_train'])
                y_train_client = np.concatenate((y_train_client, prepared_data['y_train']))

        # Create k-folds for this client
        client_folds = hpo.kfolds(X=X_train_client,
                                 y=y_train_client,
                                 n_splits=config['hpo']['kfolds'],
                                 val_split=config['hpo']['val_split'])
        client_kfolds[client_id] = client_folds

    # Create federated k-folds by combining folds across clients
    k_partitions = []
    for fold_idx in range(config['hpo']['kfolds']):
        partitions_for_fold = {}
        for client_id, client_folds in client_kfolds.items():
            if fold_idx < len(client_folds):
                train_data, val_data = client_folds[fold_idx]
                X_train, y_train = train_data
                X_val, y_val = val_data
                partitions_for_fold[client_id] = (X_train, y_train, X_val, y_val)
        k_partitions.append(partitions_for_fold)

    # Log sample data shapes for debugging
    if k_partitions:
        first_fold = k_partitions[0]
        first_client_id = list(first_fold.keys())[0]
        X_sample = first_fold[first_client_id][0]
        y_sample = first_fold[first_client_id][1]

        if isinstance(X_sample, dict):
            if 'known_input' in X_sample:
                logging.info(f'Sample HPO data shape: X_train known: {X_sample["known_input"].shape}')
            if 'static_input' in X_sample:
                logging.info(f'Sample HPO data shape: X_train static: {X_sample["static_input"].shape}')
            if 'observed_input' in X_sample:
                logging.info(f'Sample HPO data shape: X_train observed: {X_sample["observed_input"].shape}')
            logging.info(f'Sample HPO data shape: y_train: {y_sample.shape}')
        else:
            logging.info(f'Sample HPO data shape: X_train: {X_sample.shape}, y_train: {y_sample.shape}')

    feature_dim = tools.get_feature_dim(X_sample)
    config['model']['feature_dim'] = feature_dim

    # Run hyperparameter optimization
    len_trials = len(study.trials)
    logging.info(f'Starting Federated HPO with {config["hpo"]["trials"] - len_trials} new trials.')

    for i in tqdm(range(len_trials, config['hpo']['trials']), desc="Federated HPO Progress"):
        combinations = [trial.params for trial in study.trials]
        trial = study.ask()
        hyperparameters = hpo.get_hyperparameters(config=config,
                                                  hpo=True,
                                                  trial=trial)
        hyperparameters['personalization'] = config['fl']['personalize']

        check_params = hyperparameters.copy()
        if check_params in combinations:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue

        logger.info(f"Federated Trial {i+1}: {json.dumps(hyperparameters)}")
        accuracies = []

        # Evaluate on each k-fold
        for fold_idx, partitions_for_fold in enumerate(k_partitions):
            # config model name relevant for callbacks
            config['model_name'] = f'hpo_fl_m-{args.model}_out-{output_dim}_freq-{freq}_trial-{i}_fold-{fold_idx}'

            history, _ = federated.run_simulation(partitions=partitions_for_fold,
                                                 hyperparameters=hyperparameters,
                                                 config=config)

            # Extract metric from the last round
            metric_name = f'val_{config["hpo"]["metric"]}'
            if metric_name in history['metrics_aggregated'].columns:
                final_metric = history['metrics_aggregated'][metric_name].iloc[-1]
            else:
                # Fallback to training metric if validation metric not available
                metric_name = f'train_{config["hpo"]["metric"]}'
                final_metric = history['metrics_aggregated'][metric_name].iloc[-1]

            accuracies.append(final_metric)
            logging.info(f'Processed fold {fold_idx + 1}/{len(k_partitions)}.')

        logging.info(f'Accuracies for the folds: {accuracies}')
        average_accuracy = sum(accuracies) / len(accuracies)
        study.tell(trial, average_accuracy)
        logging.info(f'Federated Trial {i+1} completed with average {config["hpo"]["metric"]}: {average_accuracy:.6f}')

if __name__ == '__main__':
    main()