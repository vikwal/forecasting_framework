# Train model on all parks sequentially

import os
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import logging

from utils import tools, eval, preprocessing

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
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)
    # read config and initialize variables
    config = tools.load_config('config.yaml')
    freq = config['data']['freq']
    config['model']['output_dim'] = 1
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    config['model']['name'] = args.model
    config['model']['shuffle'] = False
    study_name = f'all_d-{args.data}_m-{args.model}_out-{output_dim}_freq-{freq}'
    # get observed, known and static features
    known, observed, static = preprocessing.get_features(data=args.data)
    # load and prepare training and test data
    dfs = preprocessing.get_data(data=args.data,
                                 data_dir=config['data']['path'],
                                 freq=freq)
    results = {}
    # create lag features
    evaluation = pd.DataFrame()
    for key, df in tqdm(dfs.items()):
        logging.info(f'Preprocessing pipeline for {key} started.')
        prepared_data, _ = preprocessing.pipeline(data=df,
                                               config=config,
                                               known_cols=known,
                                               observed_cols=observed,
                                               static_cols=static)
        X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
        X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
        results[key] = {}  # Initialize the key in the results dictionary
        index_test = prepared_data['index_test']
        scalers = prepared_data['scalers']
        scaler_y = scalers['y']
        study = tools.load_study('studies/', study_name)
        hyperparameters = tools.get_hyperparameters(config=config,
                                                    study=study)

        #logger.info(json.dumps(hyperparameters))
        train = X_train, y_train
        test = X_test, y_test #X_val, y_val
        id = key.split('.')[0][-2:]
        config['model_name'] = f'm-{args.model}_id-{id}_out-{output_dim}_freq-{freq}'
        logging.info(f'Training pipeline for {key} started.')
        history, model = tools.training_pipeline(train=train,
                                                 val=test,
                                                 hyperparameters=hyperparameters,
                                                 config=config)
        # save history and model
        results[key]['history'] = history
        #results[key]['model'] = model
        logging.info(f'Evaluation pipeline for {key} started.')
        if config['eval']['retrain_interval'] != 0:
            new_evaluation = eval.evaluate_retrain(config=config,
                                                   data=df,
                                                   cols = (known, observed, static),
                                                   index_test=index_test,
                                                   scaler_y=scaler_y,
                                                   hyperparameters=hyperparameters,
                                                   model=model)
            new_evaluation['retrain_interval'] = config['eval']['retrain_interval']
        else:
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
            new_evaluation['retrain_interval'] = None
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
    results['hyperparameters'] = hyperparameters
    results['evaluation'] = evaluation
    conf_name = f'm-{args.model}_out-{output_dim}_freq-{freq}'
    results['conf_name'] = conf_name
    # save results
    with open(f'results/{args.data}/d-{args.data}_{conf_name}.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()