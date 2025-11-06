# Hyperparameter optimization for model which is trained on all parks data at once

import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

import optuna

from utils import preprocessing, tools, hpo

optuna.logging.set_verbosity(optuna.logging.INFO)

def main() -> None:
    logger = logging.getLogger(__name__)
    # argument parser
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Tensorflow/Keras")
    parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    parser.add_argument('-i', '--index', type=str, default='', help='Define index')
    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    index = ''
    if args.index:
        index = f'_{args.index}'
    log_file = f'logs/hpo_cl_m-{args.model}{index}.log'
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
    config['model']['fl'] = False
    logging.info(f'HPO for Model: {args.model}, Output dim: {output_dim}, Frequency: {freq}, Lookback: {lookback}, Horizon: {horizon}, Step size: {config["model"]["step_size"]}')
    config['model']['name'] = args.model
    test_start = pd.Timestamp(config.get('data').get('test_start', 0))
    # get observed, known and static features
    data_dir = config['data']['path']
    features = preprocessing.get_features(config=config)
    # get the right dataset name
    base_dir = os.path.basename(data_dir)
    target_dir = os.path.join('results', base_dir)
    os.makedirs(target_dir, exist_ok=True)
    study_name_suffix = ''
    if len(config['data']['files']) == 1:
        study_name_suffix = f'_{config["data"]["files"][0]}'
    study_name = f'cl_m-{args.model}_out-{output_dim}_freq-{freq}{study_name_suffix}'
    config['model']['name'] = args.model

    # Create study for hyperparameter optimization
    study = hpo.create_or_load_study(config['hpo']['studies_path'], study_name, direction='minimize')

    # load and prepare training and test data
    dfs = preprocessing.get_data(data_dir=data_dir,
                                 config=config,
                                 freq=freq,
                                 features=features)

    # Prepare k-folds for cross-validation
    kfolds = []
    X_train_all = None
    y_train_all = None

    for key, df in tqdm(dfs.items(), desc="Preprocessing data"):
        logging.debug(f'Preprocessing {key} for HPO.')
        prepared_data, dfs[key] = preprocessing.pipeline(data=df,
                                            config=config,
                                            known_cols=features['known'],
                                            observed_cols=features['observed'],
                                            static_cols=features['static'],
                                            test_start=test_start,
                                            target_col=config['data']['target_col'])

        # Combine all training data for k-fold creation
        if X_train_all is None:
            X_train_all = prepared_data['X_train']
            y_train_all = prepared_data['y_train']
        else:
            X_train_all = tools.concatenate_data(old=X_train_all, new=prepared_data['X_train'])
            y_train_all = np.concatenate((y_train_all, prepared_data['y_train']))

    # Create k-folds from combined training data
    combined_kfolds = hpo.kfolds(X=X_train_all,
                                y=y_train_all,
                                n_splits=config['hpo']['kfolds'],
                                val_split=config['hpo']['val_split'])

    # Run hyperparameter optimization
    len_trials = len(study.trials)
    logging.info(f'Starting HPO with {config["hpo"]["trials"] - len_trials} new trials.')

    for i in tqdm(range(len_trials, config['hpo']['trials']), desc="HPO Progress"):
        combinations = [trial.params for trial in study.trials]
        trial = study.ask()
        hyperparameters = hpo.get_hyperparameters(config=config,
                                                  hpo=True,
                                                  trial=trial)
        check_params = hyperparameters.copy()
        if check_params in combinations:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue

        logger.info(f"Trial {i}: {json.dumps(hyperparameters)}")
        accuracies = []

        for fold_idx, fold in enumerate(combined_kfolds):
            train, val = fold
            # config model name relevant for callbacks
            config['model_name'] = f'hpo_cl_m-{args.model}_out-{output_dim}_freq-{freq}_trial-{i}_fold-{fold_idx}'

            history, _ = tools.training_pipeline(train=train,
                                                 val=val,
                                                 hyperparameters=hyperparameters,
                                                 config=config)
            accuracies.append(history.history[config['hpo']['metric']][-1])
            logging.info(f'Processed fold {fold_idx + 1}/{len(combined_kfolds)}.')

        logging.info(f'Accuracies for the folds: {accuracies}')
        average_accuracy = sum(accuracies) / len(accuracies)
        study.tell(trial, average_accuracy)
        logging.info(f'Trial {i+1} completed with average {config["hpo"]["metric"]}: {average_accuracy:.6f}')

if __name__ == '__main__':
    main()