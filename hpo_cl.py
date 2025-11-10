# Hyperparameter optimization for model which is trained on all parks data at once

import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

import optuna

from utils import preprocessing, tools, hpo

optuna.logging.set_verbosity(optuna.logging.INFO)

def main() -> None:
    tools.initialize_gpu()
    logger = logging.getLogger(__name__)
    # argument parser
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Tensorflow/Keras")
    parser.add_argument('-m', '--model', type=str, default='fnn', help='Select Model (default: fnn)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    parser.add_argument('-i', '--index', type=str, default='', help='Define index')
    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    index = ''
    if args.index:
        index = f'_{args.index}'
    log_file = f'logs/hpo_cl_m-{args.model}{index}.log'
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
    params = config['params']
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    config['model']['fl'] = False
    test_end = config.get('data').get('test_end', None)
    test_start = config['data']['test_start']
    logging.info(f'HPO for Model: {args.model}, Output dim: {output_dim}, Frequency: {freq}, Lookback: {lookback}, Horizon: {horizon}, Step size: {config["model"]["step_size"]}, Test start: {test_start}, Test end: {test_end}')
    config['model']['name'] = args.model
    test_start = pd.Timestamp(config.get('data').get('test_start', 0))
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

    # Create study for hyperparameter optimization
    pruning_config = config.get('hpo', {}).get('pruning', {})
    study = hpo.create_or_load_study(config['hpo']['studies_path'], study_name,
                                    direction='minimize', pruning_config=pruning_config)

    # load and prepare training and test data
    dfs = preprocessing.get_data(data_dir=data_dir,
                                 config=config,
                                 freq=freq,
                                 features=features)

    # Prepare k-folds for cross-validation with per-file minimum training length
    kfolds = []
    prepared_datasets = []

    for key, df in tqdm(dfs.items(), desc="Preprocessing data"):
        logging.debug(f'Preprocessing {key} for HPO.')
        prepared_data, dfs[key] = preprocessing.pipeline(data=df,
                                            config=config,
                                            known_cols=features['known'],
                                            observed_cols=features['observed'],
                                            static_cols=features['static'],
                                            target_col=config['data']['target_col'])
        prepared_datasets.append(prepared_data)

    # Create k-folds with per-file minimum training length consideration
    min_train_len = config['hpo'].get('min_train_len', None)
    step_size = config['model'].get('step_size', 1)
    combined_kfolds = hpo.kfolds_with_per_file_min_train_len(
        prepared_datasets=prepared_datasets,
        n_splits=config['hpo']['kfolds'],
        val_split=config['hpo']['val_split'],
        min_train_len=min_train_len,
        step_size=step_size
    )

    # Run hyperparameter optimization
    len_trials = len(study.trials)
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logging.info(f'Starting HPO with {config["hpo"]["trials"] - completed_trials} new trials.')
    logging.info(f'Previous trials: {len_trials} total, {completed_trials} completed successfully.')

    trial_counter = 0
    while completed_trials < config['hpo']['trials']:
        trial = study.ask()
        trial_number = len_trials + trial_counter
        hyperparameters = hpo.get_hyperparameters(config=config,
                                                  hpo=True,
                                                  trial=trial)
        # Check for duplicate parameters from other trials (including running ones)
        existing_params = [t.params for t in study.trials]
        current_params = trial.params
        # Count how many trials have these exact parameters
        param_count = sum(1 for params in existing_params if params == current_params)
        if param_count > 1:  # More than just the current trial
            logging.info(f"Trial {trial_number}: Duplicate parameters detected (found {param_count} times), marking as failed...")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue  # Kein trial_counter += 1, damit Nummerierung stimmt

        logger.info(f"Trial {trial_number}: {json.dumps(hyperparameters)}")
        try:
            accuracies = []
            for fold_idx, fold in enumerate(combined_kfolds):
                train, val = fold
                # config model name relevant for callbacks
                config['model_name'] = f'hpo_cl_m-{args.model}_out-{output_dim}_freq-{freq}_trial-{trial_number}_fold-{fold_idx}'
                history, _ = tools.training_pipeline(train=train,
                                                     val=val,
                                                     hyperparameters=hyperparameters,
                                                     config=config)
                fold_accuracy = history.history[config['hpo']['metric']][-1]
                accuracies.append(fold_accuracy)
                # Report intermediate value für MedianPruner
                trial.report(fold_accuracy, step=fold_idx)
                logging.info(f'Processed fold {fold_idx + 1}/{len(combined_kfolds)}, {config["hpo"]["metric"]}: {fold_accuracy:.4f}')
                if trial.should_prune():
                    raise optuna.TrialPruned()
            else:
                # Dieser Block wird NUR ausgeführt, wenn die Fold-Schleife NICHT durch break verlassen wurde
                logging.info(f'Accuracies for the folds: {accuracies}')
                average_accuracy = sum(accuracies) / len(accuracies)
                study.tell(trial, average_accuracy)
                completed_trials += 1
                logging.info(f'Trial {trial_number+1} completed with average {config["hpo"]["metric"]}: {average_accuracy:.4f}')
                logging.info(f'Progress: {completed_trials}/{config["hpo"]["trials"]} successful trials completed.')

        except optuna.TrialPruned:
            logging.info(f'Trial {trial_number} was pruned by Optuna')
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            trial_counter += 1

        except KeyboardInterrupt:
            logging.warning(f'Trial {trial_number+1} interrupted by user. Marking as failed.')
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise

        except Exception as e:
            logging.error(f'Trial {trial_number+1} failed with error: {str(e)}. Marking as failed.')
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise

        trial_counter += 1

if __name__ == '__main__':
    main()