# Federated learning simulation

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import logging

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
    parser.add_argument('-d', '--data', type=str, help='Select dataset')
    args = parser.parse_args()
    # create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs(os.path.join('results', args.data), exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)
    # read config
    config = tools.load_config('config.yaml')
    freq = config['data']['freq']
    config['model']['output_dim'] = 48 # delete when
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    config['model']['shuffle'] = True
    config['model']['fl'] = True
    # get the right dataset name
    dataset_name = args.data
    if '/' in args.data:
        dataset_name = dataset_name.replace('/', '_')
    target_dir = os.path.join('results', dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    study_name = f'fl_a-{config["fl"]["strategy"]}_d-{dataset_name}_m-{args.model}_out-{output_dim}'
    suffix = ''
    if config['fl']['personalize']: suffix+='_pers'
    study_name+=suffix
    logging.info(f'Start Federated Simulation for Study: {study_name}')
    config['model']['name'] = args.model
    # load and prepare training and test data
    known, observed, static = preprocessing.get_features(dataset_name=args.data)
    data = preprocessing.get_data(dataset_name=args.data,
                                 data_dir=config['data']['path'],
                                 freq=freq)
    dict_depth = lambda d: 1 + max(map(dict_depth, d.values())) if isinstance(d, dict) and d else 0
    data_depth = dict_depth(data)
    results = {}
    partitions = {}
    test_data = {}
    client_ids = []
    # case when each file is a client in args.data
    if data_depth == 1:
        dfs = data
        for key, df in data.items():
            logging.info(f'Preprocessing pipeline for {key} started.')
            prepared_data, dfs[key] = preprocessing.pipeline(data=df,
                                                config=config,
                                                known_cols=known,
                                                observed_cols=observed,
                                                static_cols=static)
            X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
            X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
            #X_train, y_train, X_val, y_val = tools.split_val(X=X_train, y=y_train, val_split=config['data']['val_frac'])
            #partitions.append((X_train, y_train, X_test, y_test))
            partitions[key] = (X_train, y_train, X_test, y_test)
            test_data[key] = prepared_data['X_test'], prepared_data['y_test'], prepared_data['index_test'], prepared_data['scalers']['y']
            client_ids.append(key)
    # case when each client has multiple files in args.data
    elif data_depth == 2:
        dfs = {}
        X_train = None
        for client_id, files in data.items():
            for key, df in files.items():
                logging.info(f'Preprocessing pipeline for client: {client_id}, file: {key} started.')
                prepared_data, _ = preprocessing.pipeline(data=df,
                                                    config=config,
                                                    known_cols=known,
                                                    observed_cols=observed,
                                                    static_cols=static)
                if X_train is None:
                    X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
                    X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
                else:
                    X_train = tools.concatenate_data(old=X_train, new=prepared_data['X_train'])
                    X_test = tools.concatenate_data(old=X_test, new=prepared_data['X_test'])
                    y_train = np.concatenate((y_train, prepared_data['y_train']))
                    y_test = np.concatenate((y_test, prepared_data['y_test']))
                test_data_key = f'{client_id}_{key}'
                test_data[test_data_key] = prepared_data['X_test'], prepared_data['y_test'], prepared_data['index_test'], prepared_data['scalers']['y']
                dfs[test_data_key] = df
            partitions[client_id] = (X_train, y_train, X_test, y_test)
            client_ids.append(client_id)
    # case when data depth is wrong
    else:
        raise ValueError(f'Dictionary depth of data is {data_depth}, expected 1 or 2. Please check.')
    feature_dim = tools.get_feature_dim(X_train)
    config['model']['feature_dim'] = feature_dim
    # get hyperparameters, load from study if exists
    study = hpo.load_study(config['hpo']['studies_path'], study_name)
    hyperparameters = hpo.get_hyperparameters(config=config,
                                              study=study)
    hyperparameters['personalization'] = config['fl']['personalize']
    results['hyperparameters'] = hyperparameters
    logger.info(json.dumps(hyperparameters))
    # check if trained model exists
    path_to_pkl = os.path.join('results', dataset_name, f'{study_name}.pkl')
    if os.path.exists(path_to_pkl) and not config['fl']['force_retrain']:
        logging.info(f'Model already exists. Skip federated training and start evaluation.')
        with open(path_to_pkl, 'rb') as f:
            results = pickle.load(f)
        model = results['model']
    else:
        #logger.info(json.dumps(hyperparameters))
        logging.info(f'Start Federated Forecasts Simulation.')
        history, clients_weights = federated.run_simulation(partitions=partitions,
                                                  hyperparameters=hyperparameters,
                                                  config=config)
        # save progress
        #results['model'] = model.get_weights()
        results['history'] = history
        with open(path_to_pkl, 'wb') as f:
            pickle.dump(results, f)
    logging.info(f'\nEvaluation pipeline started.')
    evaluation = pd.DataFrame()
    for client_id, (key, df) in zip(client_ids, dfs.items()):
        X_test, y_test, index_test, scaler_y = test_data[key]
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
    evaluation['strategy'] = config['fl']['strategy']
    evaluation['personalization'] = config['fl']['personalize']
    results['evaluation'] = evaluation
    results['config'] = config
    # save results
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()