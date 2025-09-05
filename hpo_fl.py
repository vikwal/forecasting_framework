# Federated learning simulation

import os
import json
import optuna
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from utils import tools, preprocessing, federated, hpo


optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    logger = logging.getLogger(__name__)
    # argument parser
    parser = argparse.ArgumentParser(description="Federated Learning Simulation with Tensorflow/Keras")
    parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
    parser.add_argument('-d', '--data', type=str, help='Select dataset')
    args = parser.parse_args()
    # create directories
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
    config['model']['shuffle'] = False
    config['model']['fl'] = True
    config['model']['name'] = args.model
    # get observed, known and static features
    suffix = ''
    if config['hpo']['fl']['personalize']: suffix+='_pers'
    study_name = f'fl_a-{config["hpo"]["fl"]["strategy"]}_d-{args.data}_m-{args.model}_out-{output_dim}_freq-{freq}'
    study_name+=suffix
    study = hpo.create_or_load_study(config['hpo']['studies_path'], study_name, direction='minimize')
    # load and prepare training and test data
    known, observed, static = preprocessing.get_features(dataset_name=args.data)
    rel_features = known + observed
    raw_data_dict = preprocessing.get_data(dataset_name=args.data,
                                 data_dir=config['data']['path'],
                                 freq=freq,
                                 rel_features=rel_features)
        # restructure the dictionary of raw dataframes when nested folders
    if '/' in args.data:
        data = defaultdict(dict)
        for key, df in raw_data_dict.items():
            client, filename = key.split('_', 1)  # 'hka', 'pvpark_1.csv'
            park_id = filename.replace('.csv', '')  # 'pvpark_1'
            data[client][filename] = df
    else:
        data = raw_data_dict
    dict_depth = lambda d: 1 + max(map(dict_depth, d.values())) if isinstance(d, dict) and d else 0
    data_depth = dict_depth(data)
    partitions = []
    if data_depth == 1:
        dfs = data
        for key, df in data.items():
            logging.info(f'Preprocessing pipeline for {key} started.')
            prepared_data, dfs[key] = preprocessing.pipeline(data=df,
                                                config=config,
                                                known_cols=known,
                                                observed_cols=observed,
                                                static_cols=static,
                                                target_col=config['data']['target_col'])
            X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
            X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
            #X_train, y_train, X_val, y_val = tools.split_val(X=X_train, y=y_train, val_split=config['data']['val_frac'])
            partitions.append((X_train, y_train, X_test, y_test))
        k_partitions = federated.get_kfolds_partitions(n_splits=config['hpo']['kfolds'], partitions=partitions)
    elif data_depth == 2:
        for client_id, files in data.items():
            series_folds = []
            for key, df in files.items():
                logging.info(f'Preprocessing pipeline for client: {client_id}, file: {key} started.')
                prepared_data, _ = preprocessing.pipeline(data=df,
                                                        config=config,
                                                        known_cols=known,
                                                        observed_cols=observed,
                                                        static_cols=static,
                                                        target_col=config['data']['target_col'])
                folds = hpo.kfolds(X=prepared_data['X_train'],
                                    y=prepared_data['y_train'],
                                    n_splits=config['hpo']['kfolds'],
                                    val_split=config['data']['val_frac'])
                series_folds.append(folds)
            client_kfolds = hpo.combine_kfolds(n_splits=config['hpo']['kfolds'],
                                        kfolds_per_series=series_folds)
            partitions.append(client_kfolds)
        k_partitions = list(zip(*partitions))  # korrekt zusammensetzen
        dfs = data
    # case when data depth is wrong
    else:
        raise ValueError(f'Dictionary depth of data is {data_depth}, expected 1 or 2. Please check.')
    # get feature_dim from the data
    feature_dim = tools.get_feature_dim(X_train)
    config['model']['feature_dim'] = feature_dim
    # get hyperparameters, load from study if exists
    len_trials = len(study.trials)
    metric = f'val_{config["hpo"]["metric"]}'
    for i in tqdm(range(len_trials, config['hpo']['trials'])):
        combinations = [trial.params for trial in study.trials]
        trial = study.ask()
        hyperparameters = hpo.get_hyperparameters(config=config,
                                                    hpo=True,
                                                    trial=trial)
        hyperparameters['personlization'] = config['hpo']['fl']['personalize']
        config['fl']['personalize'] = config['hpo']['fl']['personalize']
        config['fl']['strategy'] = config['hpo']['fl']['strategy']
        check_params = hyperparameters.copy()
        if check_params in combinations:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        logger.info(json.dumps(hyperparameters))
        accuracies = []
        for partitions in k_partitions:
            history, _ = federated.run_simulation(partitions=partitions,
                                                  config=config,
                                                  hyperparameters=hyperparameters,
                                                  client_ids=dfs.keys())
            accuracies.append(history['metrics_aggregated'][metric].iloc[-1])
            logging.info(f'Processed {len(accuracies)} folds.')
        average_accuracy = sum(accuracies) / len(accuracies)
        study.tell(trial, average_accuracy)

if __name__ == '__main__':
    main()