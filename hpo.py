# Hyperparameter Optimization Study

import os
import yaml
import json
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging

import optuna

from utils import utils, preprocessing

optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    logger = logging.getLogger(__name__)
    # argument parser
    parser = argparse.ArgumentParser(description="Flower Simulation with Tensorflow/Keras")
    parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
    parser.add_argument('--kfolds', '-k', action='store_true', help='Boolean for Kfolds (default: False)')
    parser.add_argument('-d', '--data', type=str, help='Select pv/wind (default: pv)')
    args = parser.parse_args()
    # create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)
    # read config
    config = utils.load_config('config.yaml')
    # decide on pv or wind
    if args.data == 'pv':
        data_path = config['data']['pv_path']
        file = '1B_Trina.csv'
    else:
        data_path = config['data']['wind_path']
        file = ''
    # load and prepare training and test data
    path = os.path.join(data_path, file)
    rel_features = config['data']['rel_features']
    rel_features.append(config['data']['target_col'])
    df = preprocessing.preprocess_1b_trina(path=path,
                                           timestamp_col=config['data']['timestamp_col'],
                                           freq=config['data']['freq'],
                                           rel_features=rel_features)
    df.drop(df[df.isnull().any(axis=1)].index, inplace=True)
    # create temporal features
    df = preprocessing.lag_features(df=df,
                                    target_col=config['data']['target_col'],
                                    lag_dim=config['data']['lag_dim'],
                                    horizon=config['data']['horizon'],
                                    lag_in_col=config['data']['lag_in_col'])
    windows = preprocessing.prepare_data(data=df,
                                         train_end=config['data']['train_end'],
                                         test_start=config['data']['test_start'],
                                         output_dim=config['model']['output_dim'],
                                         target_col=config['data']['target_col'],
                                         scale_y=config['data']['scale_y'],
                                         return_index=config['data']['return_index'])
    X_train, y_train, X_test, y_test = windows[0], windows[1], windows[2], windows[3]
    if config['data']['return_index']:
        index_train, index_test = windows[4], windows[5]
    scaler_y = None
    if config['data']['scale_y']:
        scaler_y = windows[6]
    study = utils.create_or_load_study('studies/', f'{args.data}_{args.model}', direction='minimize')
    len_trials = len(study.trials)
    for i in range(len_trials, config['hpo']['trials']):
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
                history = utils.training_pipeline(train=train,
                                                    val=val,
                                                    hyperparameters=hyperparameters,
                                                    config=config)
                accuracies.append(history.history[config['model']['metrics']][-1])
            average_accuracy = sum(accuracies) / len(accuracies)
            study.tell(trial, average_accuracy)
        else:
            windows = preprocessing.prepare_data(data=df,
                                        train_end=config['data']['train_val_end'],
                                        test_start=config['data']['val_start'],
                                        output_dim=config['model']['output_dim'],
                                        target_col=config['data']['target_col'],
                                        scale_y=config['data']['scale_y'],
                                        return_index=config['data']['return_index'])
            X_train, y_train, X_val, y_val = windows[0], windows[1], windows[2], windows[3]
            train = X_train, y_train
            val = X_val, y_val
            history = utils.training_pipeline(train=train,
                                                val=val,
                                                hyperparameters=hyperparameters,
                                                config=config)
            accuracy = history.history[config['model']['metrics']][-1]
            study.tell(trial, accuracy)

if __name__ == '__main__':
    main()