# Hyperparameter optimization for model which is trained on all parks data at once

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import logging

import optuna

from utils import preprocessing, tools, hpo

optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    #utils.initialize_gpu(2)
    logger = logging.getLogger(__name__)
    # argument parser
    parser = argparse.ArgumentParser(description="Simulation with Tensorflow/Keras")
    parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
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
    config['model']['name'] = args.model
    config['model']['shuffle'] = True
    config['model']['fl'] = False
    # get observed, known and static features
    known, observed, static = preprocessing.get_features(data=args.data)

    study_name = f'cl_d-{args.data}_m-{args.model}_out-{output_dim}_freq-{freq}'#_incrfltrs-{config["hpo"]["cnn"]["increase_filters"]}'
    config['model_name'] = args.model
    study = hpo.create_or_load_study(config['hpo']['studies_path'], study_name, direction='minimize')
    # load and prepare training and test data
    dfs = preprocessing.get_data(data=args.data,
                                data_dir=config['data']['path'],
                                freq=freq)
    kfolds = []
    for key, df in tqdm(dfs.items()):
        prepared_data, df = preprocessing.pipeline(data=df,
                                            config=config,
                                            known_cols=known,
                                            observed_cols=observed,
                                            static_cols=static)
        scalers = prepared_data['scalers']
        scaler_y = scalers['y']
        new_kfolds = hpo.kfolds(X=prepared_data['X_train'],
                                  y=prepared_data['y_train'],
                                  n_splits=config['hpo']['kfolds'],
                                  val_split=config['hpo']['val_split'])
        kfolds.append(new_kfolds)

    combined_kfolds = hpo.combine_kfolds(n_splits=config['hpo']['kfolds'],
                                         kfolds_per_series=kfolds)

    len_trials = len(study.trials)
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
        accuracies = []
        for fold in combined_kfolds:
            train, val = fold
            history, _ = tools.training_pipeline(train=train,
                                                 val=val,
                                                 hyperparameters=hyperparameters,
                                                 config=config)
            accuracies.append(history.history[config['hpo']['metric']][-1])
            logging.info(f'Processed {len(accuracies)} folds.')
        average_accuracy = sum(accuracies) / len(accuracies)
        study.tell(trial, average_accuracy)

if __name__ == '__main__':
    main()