# Hyperparameter optimization for model which is trained on all parks data at once

import os
import yaml
import json
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import logging

import optuna

import models
import preprocessing
import utils
import utils_eval
import pickle

optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    utils.initialize_gpu(2)
    logger = logging.getLogger(__name__)
    # argument parser
    parser = argparse.ArgumentParser(description="Simulation with Tensorflow/Keras")
    parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
    parser.add_argument('--kfolds', '-k', action='store_true', help='Boolean for Kfolds (default: False)')
    parser.add_argument('-d', '--data', type=str, help='Select dataset')
    args = parser.parse_args()
    # create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)
    # read config
    config = utils.load_config('config.yaml')
    freq = config['data']['freq']
    output_dim, lag_dim, horizon = utils.handle_freq(freq=freq,
                                                     output_dim=config['model']['output_dim'],
                                                     lag_dim=config['data']['lag_dim'],
                                                     horizon=config['data']['horizon'])
    output_dim = 48
    config['model']['output_dim'] = output_dim
    config['data']['horizon'] = horizon
    config['data']['lag_dim'] = lag_dim
    config['model']['shuffle'] = True

    conf_name = f'all_d-{args.data}_m-{args.model}_out-{output_dim}_freq-{freq}'
    config['model_name'] = conf_name
    study = utils.create_or_load_study('studies/', conf_name, direction='minimize')
    # load and prepare training and test data
    dfs = preprocessing.get_data(data=args.data,
                                data_dir=config['data']['path'],
                                freq=freq)
    X_train = None
    for key, df in tqdm(dfs.items()):
        df = utils.impute_index(data=df)
        df = preprocessing.lag_features(df=df,
                                            lag_dim=lag_dim,
                                            horizon=horizon,
                                            lag_in_col=config['data']['lag_in_col'])
        windows = preprocessing.prepare_data(data=df,
                                            output_dim=output_dim,
                                            train_frac=config['data']['train_frac'],
                                            scale_y=config['data']['scale_y'])
        if X_train is None:
            X_train, y_train = windows['X_train'], windows['y_train']
        else:
            X_train = np.concatenate((X_train, windows['X_train']))
            y_train = np.concatenate((y_train, windows['y_train']))
        scaler = windows['scaler']
        scaler_x, scaler_y = scaler[0], scaler[1]
    print(f'X_train.shape: {X_train.shape}', f'y_train.shape: {y_train.shape}')

    len_trials = len(study.trials)
    n_samples = X_train.shape[0]
    shuffled_indices = np.random.permutation(n_samples)
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    for i in tqdm(range(len_trials, config['hpo']['trials'])):
        combinations = [trial.params for trial in study.trials]
        trial = study.ask()
        hyperparameters = utils.get_hyperparameters(model_name=args.model,
                                                    config=config,
                                                    hpo=True,
                                                    trial=trial)
        check_params = hyperparameters.copy()
        check_params.pop('model_name')
        if check_params in combinations:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        logger.info(json.dumps(hyperparameters))
        if args.kfolds:
            accuracies = []
            kfolds = utils.kfolds(X=X_train,
                                    y=y_train,
                                    n_splits=config['hpo']['kfolds'])
            for fold in kfolds:
                train, val = fold
                history, _ = utils.training_pipeline(train=train,
                                                     val=val,
                                                     hyperparameters=hyperparameters,
                                                     config=config)
                accuracies.append(history.history[config['hpo']['metric']][-1])
                logging.info(f'Processed {len(accuracies)} folds.')
            average_accuracy = sum(accuracies) / len(accuracies)
            study.tell(trial, average_accuracy)
        else:
            val_split = config['hpo']['val_split']
            X_train, y_train = X_train[:int(len(X_train)*val_split)], y_train[:int(len(y_train)*val_split)]
            train = X_train, y_train
            val = X_train[int(len(X_train)*val_split):], y_train[int(len(y_train)*val_split):]
            history, _ = utils.training_pipeline(train=train,
                                              val=val,
                                              hyperparameters=hyperparameters,
                                              config=config)
            accuracy = history.history[config['hpo']['metric']][-1]
            study.tell(trial, accuracy)

if __name__ == '__main__':
    main()