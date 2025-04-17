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

from utils import utils, eval, preprocessing
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
    config['model']['output_dim'] = 48
    output_dim, lookback, horizon = utils.handle_freq(freq=freq,
                                                     output_dim=config['model']['output_dim'],
                                                     lookback=config['model']['lookback'],
                                                     horizon=config['model']['horizon'])
    config['model']['output_dim'] = output_dim
    config['model']['horizon'] = horizon
    config['model']['lookback'] = lookback
    config['model']['shuffle'] = True
    # get observed, known and static features
    known, observed, static = preprocessing.get_features(data='pvod')

    study_name = f'all_d-{args.data}_m-{args.model}_out-{output_dim}_freq-{freq}'
    conf_name = study_name
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
            logging.info(f'Preprocessing pipeline started.')
            prepared_data, df = preprocessing.pipeline(data=df,
                                                model=args.model,
                                                config=config,
                                                known_cols=known,
                                                observed_cols=observed,
                                                static_cols=static)
            X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
            results[key] = {}  # Initialize the key in the results dictionary
            index_test = prepared_data['index_test']
            scalers = prepared_data['scalers']
            scaler_y = scalers['y']
            if X_train is None:
                X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
                X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
            else:
                X_train = utils.concatenate_data(old=X_train, new=prepared_data['X_train'])
                X_test = utils.concatenate_data(old=X_test, new=prepared_data['X_test'])
                y_train = np.concatenate((y_train, prepared_data['y_train']))
                y_test = np.concatenate((y_test, prepared_data['y_test']))
            test_data[key] = prepared_data['X_test'], prepared_data['y_test'], prepared_data['index_test']
            scaler = prepared_data['scalers']
            scaler_y = scalers['y']
            #X_train, y_train = X_train[:len(X_train)*0.75], y_train[:len(y_train)*0.75]
            #X_val, y_val = X_train[len(X_train)*0.75:], y_train[len(y_train)*0.75:]
        study = utils.load_study('studies/', study_name)
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

        hyperparameters['n_heads'] = 2
        hyperparameters['lookback'] = lookback
        hyperparameters['horizon'] = horizon
        hyperparameters['hidden_dim'] = 60
        hyperparameters['dropout'] = 0.1

        # save hyperparameters in results
        results['hyperparameters'] = hyperparameters

        train = X_train, y_train
        test = X_test, y_test #X_val, y_val
        config['model_name'] = f'all_m-{args.model}_out-{output_dim}_freq-{freq}'
        logging.info(f'Training pipeline started.')
        history, model = utils.training_pipeline(train=train,
                                                 val=test,
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
    logging.info(f'Evaluation pipeline started.')
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
        entry = eval.get_metrics(y_pred=df_pred.values,
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