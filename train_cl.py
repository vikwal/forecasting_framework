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
import pickle
import optuna

from utils import tools, eval, preprocessing, hpo

optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    logger = logging.getLogger(__name__)
    # argument parser
    parser = argparse.ArgumentParser(description="Simulation with Tensorflow/Keras")
    parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
    parser.add_argument('--kfolds', '-k', action='store_true', help='Boolean for Kfolds (default: False)')
    parser.add_argument('-d', '--data', type=str, help='Select dataset')
    parser.add_argument('-g', '--gpu', type=int, help='Select gpu')
    args = parser.parse_args()
    tools.initialize_gpu(use_gpu=args.gpu)
    # create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)
    # read config
    config = tools.load_config('config.yaml')
    freq = config['data']['freq']
    config['model']['output_dim'] = 48
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    config['model']['name'] = args.model
    config['model']['shuffle'] = True
    # get observed, known and static features
    known, observed, static = preprocessing.get_features(data='pvod')

    study_name = f'cl_d-{args.data}_m-{args.model}_out-{output_dim}_freq-{freq}'
    config['model']['name'] = args.model
    path_to_pkl = os.path.join('results', args.data, f'{study_name}.pkl')
    # load and prepare training and test data
    dfs = preprocessing.get_data(data=args.data,
                                data_dir=config['data']['path'],
                                freq=freq)
    results = {}
    test_data = {}
    X_train = None
    for key, df in dfs.items():
        logging.info(f'Preprocessing {key}.')
        prepared_data, dfs[key] = preprocessing.pipeline(data=df,
                                            config=config,
                                            known_cols=known,
                                            observed_cols=observed,
                                            static_cols=static)
        X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
        index_test = prepared_data['index_test']
        scalers = prepared_data['scalers']
        scaler_y = scalers['y']
        if X_train is None:
            X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
            X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
        else:
            X_train = tools.concatenate_data(old=X_train, new=prepared_data['X_train'])
            X_test = tools.concatenate_data(old=X_test, new=prepared_data['X_test'])
            y_train = np.concatenate((y_train, prepared_data['y_train']))
            y_test = np.concatenate((y_test, prepared_data['y_test']))
        test_data[key] = prepared_data['X_test'], prepared_data['y_test'], prepared_data['index_test'], prepared_data['scalers']['y']
    if os.path.exists(path_to_pkl) and not config['model']['force_retrain']:
        with open(path_to_pkl, 'rb') as f:
            results = pickle.load(f)
        model = results['model']
    else:
        study = hpo.load_study(config['hpo']['studies_path'], study_name)
        hyperparameters = hpo.get_hyperparameters(config=config,
                                                    study=study)
        logger.info(json.dumps(hyperparameters))
        # save hyperparameters in results
        results['hyperparameters'] = hyperparameters

        train = X_train, y_train
        test = X_test, y_test #X_val, y_val
        # config model name relevant for callbacks
        config['model_name'] = f'cl_m-{args.model}_out-{output_dim}_freq-{freq}'
        logging.info(f'Training pipeline started.')
        history, model = tools.training_pipeline(train=train,
                                                 val=test,
                                                 hyperparameters=hyperparameters,
                                                 config=config)
        # save progress
        results['history'] = history
        #results['model'] = model
        with open(path_to_pkl, 'wb') as f:
            pickle.dump(results, f)

    logging.info('Start evaluation pipeline.')
    # evaluate the global model on the specific test datasets
    evaluation = pd.DataFrame()
    for key, df in dfs.items():
        X_test, y_test, index_test, scaler_y = test_data[key]
        new_evaluation = eval.evaluation_pipeline(data=df,
                                                model=model,
                                                model_name=args.model,
                                                X_test=X_test,
                                                y_test=y_test,
                                                scaler_y=scaler_y,
                                                output_dim=output_dim,
                                                horizon=horizon,
                                                index_test=index_test,
                                                t_0=config['eval']['t_0'],
                                                evaluate_on_all_test_data=config['eval']['eval_on_all_test_data'])
        new_evaluation['output_dim'] = output_dim
        new_evaluation['freq'] = freq
        new_evaluation['key'] = key
        if config['eval']['eval_on_all_test_data']:
            new_evaluation['t_0'] = None
        else:
            new_evaluation['t_0'] = config['eval']['t_0']
        if evaluation.empty:
            evaluation = new_evaluation
        else:
            evaluation = pd.concat([evaluation, new_evaluation], axis=0)
    # save evaluation
    results['evaluation'] = evaluation
    results['config'] = config
    # save results
    with open(f'results/{args.data}/{study_name}.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()