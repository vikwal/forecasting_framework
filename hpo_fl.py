# Federated learning simulation

import os
import json
import optuna
import logging
import argparse
from tqdm import tqdm

from utils import tools, preprocessing, federated


optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)
    # read config
    config = tools.load_config('config.yaml')
    freq = config['data']['freq']
    config['model']['output_dim'] = 1 # delete when
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    config['model']['shuffle'] = False
    config['model']['type'] = 'fl'
    study_name = f'fl_d-{args.data}_m-{args.model}_out-{output_dim}_freq-{freq}'
    study = tools.create_or_load_study('studies/', study_name, direction='minimize')
    config['model']['name'] = args.model
    # get observed, known and static features
    known, observed, static = preprocessing.get_features(data=args.data)
    # load and prepare training and test data
    dfs = preprocessing.get_data(data=args.data,
                                 data_dir=config['data']['path'],
                                 freq=freq)
    partitions = []
    for key, df in dfs.items():
        logging.info(f'Preprocessing pipeline for {key} started.')
        prepared_data, dfs[key] = preprocessing.pipeline(data=df,
                                            config=config,
                                            known_cols=known,
                                            observed_cols=observed,
                                            static_cols=static)
        X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
        X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
        #X_train, y_train, X_val, y_val = tools.split_val(X=X_train, y=y_train, val_split=config['data']['val_frac'])
        partitions.append((X_train, y_train, X_test, y_test))
    kfolds_partitions = federated.get_kfolds_partitions(n_splits=config['hpo']['kfolds'], partitions=partitions)
    # get feature_dim from the data
    feature_dim = tools.get_feature_dim(X_train)
    config['model']['feature_dim'] = feature_dim
    # get hyperparameters, load from study if exists
    len_trials = len(study.trials)
    metric = f'eval_{config["hpo"]["metric"]}'
    for i in tqdm(range(len_trials, config['hpo']['trials'])):
        combinations = [trial.params for trial in study.trials]
        trial = study.ask()
        hyperparameters = tools.get_hyperparameters(config=config,
                                                    hpo=True,
                                                    trial=trial)
        check_params = hyperparameters.copy()
        if check_params in combinations:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        logger.info(json.dumps(hyperparameters))
        accuracies = []
        for partitions in kfolds_partitions:
            history, _ = federated.run_simulation(partitions=partitions,
                                                  config=config,
                                                  hyperparameters=hyperparameters)
            accuracies.append(history['metrics_aggregated'][metric].iloc[-1])
            logging.info(f'Processed {len(accuracies)} folds.')
        average_accuracy = sum(accuracies) / len(accuracies)
        study.tell(trial, average_accuracy)

if __name__ == '__main__':
    main()