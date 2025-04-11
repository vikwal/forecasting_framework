# Train model on all parks data at once

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
    config['model']['output_dim'] = output_dim
    config['data']['horizon'] = horizon
    config['data']['lag_dim'] = lag_dim

    conf_name = f'all_d-{args.data}_m-{args.model}_out-{output_dim}_freq-{freq}'
    config['model_name'] = conf_name
    path_to_pkl = os.path.join('results', args.data, f'{conf_name}.pkl')
    if os.path.exists(path_to_pkl):
        with open(path_to_pkl, 'rb') as f:
            results = pickle.load(f)
        test_data = results['test_data']
        model = results['model']
    else:
        # load and prepare training and test data
        dfs = preprocessing.get_data(data=args.data,
                                    data_dir=config['data']['path'],
                                    freq=freq)
        results = {}
        test_data = {}
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
            test_data[key] = windows['X_test'], windows['y_test'], windows['index_test']
            scaler = windows['scaler']
            scaler_x, scaler_y = scaler[0], scaler[1]
            #X_train, y_train = X_train[:len(X_train)*0.75], y_train[:len(y_train)*0.75]
            #X_val, y_val = X_train[len(X_train)*0.75:], y_train[len(y_train)*0.75:]
        print(f'X_train.shape: {X_train.shape}', f'y_train.shape: {y_train.shape}')
        study = None#utils.load_study('studies/', f'{args.data}_{args.model}')
        hyperparameters = utils.get_hyperparameters(model_name=args.model,
                                                    config=config,
                                                    study=study)
        # set hyperparameters manually
        hyperparameters['batch_size'] = 16
        hyperparameters['epochs'] = 20
        hyperparameters['filters'] = 64
        hyperparameters['kernel_size'] = 2
        hyperparameters['n_layers'] = 2
        hyperparameters['n_cnn_layers'] = 2
        hyperparameters['n_rnn_layers'] = 2
        hyperparameters['lr'] = 0.0004
        hyperparameters['units'] = 64

        # save hyperparameters in results
        results['hyperparameters'] = hyperparameters

        train = X_train, y_train
        history, model = utils.training_pipeline(train=train,
                                                 hyperparameters=hyperparameters,
                                                 config=config)
        # save progress
        results['history'] = history
        results['model'] = model
        results['test_data'] = test_data

        with open(path_to_pkl, 'wb') as f:
            pickle.dump(results, f)

    # evaluate the global model on the specific test datasets
    evaluation = pd.DataFrame()
    for key, (X_test, y_test, index_test) in tqdm(test_data.items()):
        # evaluate model
        y_true, y_pred = utils.get_y(X_test=X_test,
                                     y_test=y_test,
                                     output_dim=output_dim,
                                     model=model)
        df_pred = utils.y_to_df(y=y_pred,
                                output_dim=output_dim,
                                horizon=horizon,
                                index_test=index_test,
                                t_0=None if config['eval']['eval_on_all_test_data'] else config['eval']['t_0'])
        df_true = utils.y_to_df(y=y_true,
                                output_dim=output_dim,
                                horizon=horizon,
                                index_test=index_test,
                                t_0=None if config['eval']['eval_on_all_test_data'] else config['eval']['t_0'])
        entry = utils_eval.get_metrics(y_pred=df_pred.values,
                                         y_true=df_true.values)
        entry['key'] = [key]
        new_evaluation = pd.DataFrame(entry)
        if evaluation.empty:
            evaluation = new_evaluation
        else:
            evaluation = pd.concat([evaluation, new_evaluation], axis=0)
    # save evaluation
    evaluation['output_dim'] = output_dim
    evaluation['freq'] = freq
    if config['eval']['eval_on_all_test_data']:
        evaluation['t_0'] = None
    else:
        evaluation['t_0'] = config['eval']['t_0']
    evaluation['Models'] = args.model
    results['evaluation'] = evaluation
    # save results
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()