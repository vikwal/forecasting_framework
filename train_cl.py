# Train model on all parks data at once

import os
import json
import argparse
import pandas as pd
import numpy as np
import logging
import pickle
import optuna
import gc
from datetime import datetime
from tqdm import tqdm

from utils import tools, eval, preprocessing, hpo

optuna.logging.set_verbosity(optuna.logging.INFO)

def main() -> None:
    logger = logging.getLogger(__name__)
    # argument parser
    parser = argparse.ArgumentParser(description="Simulation with Tensorflow/Keras")
    parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    parser.add_argument('-i', '--index', type=str, default='', help='Define index')
    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    index = ''
    if args.index:
        index = f'_{args.index}'
    log_file = f'logs/train_cl_m-{args.model}{index}.log'
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
    synth_dir = config['data']['synth_path']
    params = config['params']
    #config['model']['output_dim'] = 1
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    logging.info(f'Model: {args.model}, Output dim: {output_dim}, Frequency: {freq}, Lookback: {lookback}, Horizon: {horizon}, Step size: {config["model"]["step_size"]}')
    config['model']['name'] = args.model
    test_start = pd.Timestamp(config.get('data').get('test_start', 0))
    #config['model']['shuffle'] = True
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
    # load and prepare training and test data - MEMORY OPTIMIZED
    dfs = preprocessing.get_data(data_dir=data_dir,
                                 config=config,
                                 freq=freq,
                                 features=features)

    # Use memory-efficient data processing
    logging.debug("Starting memory-efficient data processing...")
    data_generator = tools.create_data_generator(dfs, config, features)
    X_train, y_train, X_test, y_test, test_data = tools.combine_datasets_efficiently(data_generator)

    # Store evaluation DataFrames before cleaning up memory
    eval_dfs = dfs.copy()

    # Explizit dfs löschen um Speicher freizugeben
    del dfs
    gc.collect()
    logging.debug("Data processing completed and memory cleaned up.")

    # Initialize results and paths
    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    path_to_pkl = os.path.join('results', base_dir, f'{study_name}_{timestamp}.pkl')
    # log shapes
    if isinstance(X_train, dict):
        if 'known' in X_train:
            logging.info(f'Data shape: X_train known: {X_train["known"].shape},  X_test known: {X_test["known"].shape}')
        else:
            logging.warning('No known data provided.')
        if 'static' in X_train:
            logging.info(f'Data shape: X_train static: {X_train["static"].shape}, X_test static: {X_test["static"].shape}')
        else:
            logging.warning('No static data provided.')
        logging.info(f'Data shape: X_train observed: {X_train["observed"].shape}, X_test observed: {X_test["observed"].shape}')
        logging.info(f'Data shape: y_train: {y_train.shape}, y_test: {y_test.shape}')
    else:
        logging.info(f'Data shape: X_train: {X_train.shape}, X_test: {X_test.shape}')
        logging.info(f'Data shape: y_train: {y_train.shape}, y_test: {y_test.shape}')

    # load study and get best hyperparameters
    if os.path.exists(path_to_pkl) and not config['model']['force_retrain']:
        with open(path_to_pkl, 'rb') as f:
            results = pickle.load(f)
        model = results['model']
    else:
        study = None
        if config['model']['lookup_hpo']:
            study = hpo.load_study(config['hpo']['studies_path'], study_name)
        hyperparameters = hpo.get_hyperparameters(config=config,
                                                    study=study)
        # save hyperparameters in results
        results['hyperparameters'] = hyperparameters
        logging.info(json.dumps(hyperparameters))
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
        results['history'] = history.history
        #results['model'] = model
        with open(path_to_pkl, 'wb') as f:
            pickle.dump(results, f)

    logging.info('Start evaluation pipeline.')
    # evaluate the global model on the specific test datasets
    evaluation = pd.DataFrame()

    for key in test_data.keys():
        if key not in eval_dfs:
            logging.warning(f"Key {key} not found in evaluation data, skipping...")
            continue

        park_id = key.split('.csv')[0][-5:]
        X_test, y_test, index_test, scaler_y = test_data[key]
        df = eval_dfs[key]

        new_evaluation = eval.evaluation_pipeline(data=df,
                                                model=model,
                                                model_name=args.model,
                                                X_test=X_test,
                                                y_test=y_test,
                                                scaler_y=scaler_y,
                                                output_dim=output_dim,
                                                horizon=horizon,
                                                index_test=index_test,
                                                test_start=test_start,
                                                park_id=park_id,
                                                synth_dir=synth_dir,
                                                target_col=config['data']['target_col'],
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
    # include last row as column wise average
    evaluation.loc['mean'] = evaluation.mean(numeric_only=True)
    print(evaluation)
    results['evaluation'] = evaluation
    results['config'] = config
    # save results
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()