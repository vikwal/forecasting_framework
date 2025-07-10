# Train model on all parks sequentially

import os
import pickle
import json
import argparse
import pandas as pd
from tqdm import tqdm
import optuna
import logging

from utils import tools, preprocessing, hpo

logging.basicConfig(level=logging.INFO,
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
    os.makedirs(os.path.join('results', args.data), exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)
    # read config and initialize variables
    config = tools.load_config('config.yaml')
    freq = config['data']['freq']
    config['model']['output_dim'] = 48
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    config['model']['name'] = args.model
    config['model']['shuffle'] = False
    study_name = f'd-{args.data}_m-{args.model}_out-{output_dim}_freq-{freq}'
    study = hpo.create_or_load_study(config['hpo']['studies_path'], study_name, direction='minimize')
    # get observed, known and static features
    known, observed, static = preprocessing.get_features(dataset_name=args.data)
    # load and prepare training and test data
    dfs = preprocessing.get_data(dataset_name=args.data,
                                 data_dir=config['data']['path'],
                                 freq=freq)
    len_trials = len(study.trials)
    metric = f'eval_{config["hpo"]["metric"]}'
    for i in tqdm(range(len_trials, config['hpo']['trials'])):
        combinations = [trial.params for trial in study.trials]
        trial = study.ask()
        hyperparameters = hpo.get_hyperparameters(config=config,
                                                    hpo=True,
                                                    trial=trial)
        check_params = hyperparameters.copy()
        if check_params in combinations:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        logger.info(json.dumps(hyperparameters))
    for key, data in tqdm(dfs.items()):
        logging.info(f'Preprocessing pipeline for {key} started.')
        prepared_data, df = preprocessing.pipeline(data=data,
                                               config=config,
                                               known_cols=known,
                                               observed_cols=observed,
                                               static_cols=static)
        X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
        X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
        index_test = prepared_data['index_test']
        scalers = prepared_data['scalers']
        scaler_y = scalers['y']
        hyperparameters = hpo.get_hyperparameters(config=config,
                                                    study=study)

        #logger.info(json.dumps(hyperparameters))
        train = X_train, y_train
        test = X_test, y_test #X_val, y_val
        id = key.split('.')[0][-2:]
        config['model_name'] = f'm-{args.model}_id-{id}_out-{output_dim}_freq-{freq}'
        logging.info(f'Training pipeline for {key} started.')
        _, model = tools.training_pipeline(train=train,
                                                 val=test,
                                                 hyperparameters=hyperparameters,
                                                 config=config)
        logging.info(f'Evaluation pipeline for {key} started.')


if __name__ == '__main__':
    main()