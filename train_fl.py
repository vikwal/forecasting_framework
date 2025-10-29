# Federated learning simulation

import os
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

from utils import tools, eval, preprocessing, federated, hpo, models

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['RAY_DEDUP_LOGS'] = '0'


def main() -> None:
    logger = logging.getLogger(__name__)
    tools.initialize_gpu()
    # argument parser
    parser = argparse.ArgumentParser(description="Federated Learning Simulation with Tensorflow/Keras")
    parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    args = parser.parse_args()
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
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    logging.info(f'Federated Model: {args.model}, Output dim: {output_dim}, Frequency: {freq}, Lookback: {lookback}, Horizon: {horizon}')
    config['model']['name'] = args.model
    config['model']['shuffle'] = True
    config['model']['fl'] = True
    test_start = pd.Timestamp(config.get('data').get('test_start', 0))

    # get observed, known and static features
    features = preprocessing.get_features(config=config)

    # get the right dataset name
    base_dir = os.path.basename(config['data']['path'])
    target_dir = os.path.join('results', base_dir)
    os.makedirs(target_dir, exist_ok=True)

    study_name = f'fl_a-{config["fl"]["strategy"]}_m-{args.model}_out-{output_dim}_freq-{freq}'
    suffix = ''
    if config['fl']['personalize']:
        suffix += '_pers'
    study_name += suffix
    logging.info(f'Start Federated Simulation for Study: {study_name}')
    config['model']['name'] = args.model
    # load and prepare training and test data using federated data structure
    clients_data = federated.load_federated_data(config=config,
                                                freq=freq,
                                                features=features,
                                                target_col=config['data']['target_col'])

    results = {}
    partitions = {}
    test_data = {}
    client_ids = []
    dfs = {}  # For storing processed dataframes for evaluation

    # Process each client's data
    for client_id, client_files in clients_data.items():
        logging.info(f'Processing data for client: {client_id}')
        client_ids.append(client_id)

        X_train_client = None
        X_test_client = None
        y_train_client = None
        y_test_client = None

        # Process all files for this client
        for file_key, df in client_files.items():
            logging.info(f'Preprocessing {file_key} for client {client_id}.')
            prepared_data, processed_df = preprocessing.pipeline(data=df,
                                                config=config,
                                                known_cols=features['known'],
                                                observed_cols=features['observed'],
                                                static_cols=features['static'],
                                                test_start=test_start,
                                                target_col=config['data']['target_col'])

            # Combine data within client if multiple files
            if X_train_client is None:
                X_train_client = prepared_data['X_train']
                X_test_client = prepared_data['X_test']
                y_train_client = prepared_data['y_train']
                y_test_client = prepared_data['y_test']
            else:
                X_train_client = tools.concatenate_data(old=X_train_client, new=prepared_data['X_train'])
                X_test_client = tools.concatenate_data(old=X_test_client, new=prepared_data['X_test'])
                y_train_client = np.concatenate((y_train_client, prepared_data['y_train']))
                y_test_client = np.concatenate((y_test_client, prepared_data['y_test']))

            # Store test data and processed dataframes for evaluation
            test_data_key = f'{client_id}_{file_key}'
            test_data[test_data_key] = (prepared_data['X_test'], prepared_data['y_test'],
                                      prepared_data['index_test'], prepared_data['scalers']['y'])
            dfs[test_data_key] = processed_df

        # Store client partition
        partitions[client_id] = (X_train_client, y_train_client, X_test_client, y_test_client)

    # Log data shapes for first client (for debugging)
    first_client = list(partitions.keys())[0]
    X_sample = partitions[first_client][0]

    feature_dim = tools.get_feature_dim(X_sample)
    config['model']['feature_dim'] = feature_dim

    # get hyperparameters, load from study if exists
    study = hpo.load_study(config['hpo']['studies_path'], study_name)
    hyperparameters = hpo.get_hyperparameters(config=config,
                                              study=study)
    hyperparameters['personalization'] = config['fl']['personalize']
    results['hyperparameters'] = hyperparameters
    logger.info(json.dumps(hyperparameters))

    # check if trained model exists
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    path_to_pkl = os.path.join('results', base_dir, f'{study_name}_{timestamp}.pkl')
    if os.path.exists(path_to_pkl) and not config['fl']['force_retrain']:
        logging.info(f'Model already exists. Skip federated training and start evaluation.')
        with open(path_to_pkl, 'rb') as f:
            results = pickle.load(f)
        clients_weights = results['clients_weights']
    else:
        logging.info(f'Start Federated Forecasts Simulation.')
        history, clients_weights = federated.run_simulation(partitions=partitions,
                                                  hyperparameters=hyperparameters,
                                                  config=config)
        # save progress
        results['history'] = history
        results['clients_weights'] = clients_weights
        with open(path_to_pkl, 'wb') as f:
            pickle.dump(results, f)

    logging.info(f'\nEvaluation pipeline started.')
    evaluation = pd.DataFrame()

    for test_data_key, df in dfs.items():
        # Extract client_id and file info from test_data_key
        client_id = test_data_key.split('_')[0]
        file_key = '_'.join(test_data_key.split('_')[1:])
        park_id = test_data_key.split('.csv')[0][-5:] if '.csv' in test_data_key else test_data_key[-5:]

        X_test, y_test, index_test, scaler_y = test_data[test_data_key]
        model = models.get_model(config=config, hyperparameters=hyperparameters)
        model.set_weights(clients_weights[client_id])

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
        new_evaluation['key'] = file_key
        new_evaluation['client_id'] = client_id
        if config['eval']['eval_on_all_test_data']:
            new_evaluation['t_0'] = None
        else:
            new_evaluation['t_0'] = config['eval']['t_0']
        if evaluation.empty:
            evaluation = new_evaluation
        else:
            evaluation = pd.concat([evaluation, new_evaluation], axis=0)

    # save evaluation
    evaluation['strategy'] = config['fl']['strategy']
    evaluation['personalization'] = config['fl']['personalize']
    print(evaluation)
    results['evaluation'] = evaluation
    results['config'] = config
    # save results
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()