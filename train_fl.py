# Federated learning simulation

import os
import pickle
import argparse
import pandas as pd
from tensorflow import keras
from tqdm import tqdm
import logging

from utils import tools
import optuna

from utils import eval, preprocessing, federated


optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['RAY_DEDUP_LOGS'] = '0'


def main() -> None:
    logger = logging.getLogger(__name__)
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
    config['model']['output_dim'] = 48 # delete when
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    config['model']['shuffle'] = False
    study_name = f'fl_d-{args.data}_m-{args.model}_out-{output_dim}_freq-{freq}'
    config['model']['name'] = args.model
    # get observed, known and static features
    known, observed, static = preprocessing.get_features(data=args.data)
    # load and prepare training and test data
    dfs = preprocessing.get_data(data=args.data,
                                 data_dir=config['data']['path'],
                                 freq=freq)
    results = {}
    partitions = []
    test_data = {}
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
        test_data[key] = prepared_data['X_test'], prepared_data['y_test'], prepared_data['index_test'], prepared_data['scalers']['y']
    feature_dim = tools.get_feature_dim(X_train)
    config['model']['feature_dim'] = feature_dim
    # get hyperparameters, load from study if exists
    study = tools.load_study('studies/', study_name)
    hyperparameters = tools.get_hyperparameters(config=config,
                                                study=study)
    # check if trained model exists
    path_to_pkl = os.path.join('results', args.data, f'{study_name}.pkl')
    if os.path.exists(path_to_pkl):
        logging.info(f'Model already exists. Skip federated training and start evaluation.')
        with open(path_to_pkl, 'rb') as f:
            results = pickle.load(f)
        model = results['model']
    else:
        #logger.info(json.dumps(hyperparameters))
        logging.info(f'Start Federated Forecasts Simulation.')
        history, model = federated.run_simulation(partitions=partitions,
                                                  hyperparameters=hyperparameters,
                                                  config=config)
        # save progress
        #results['model'] = model
        results['history'] = history
        with open(path_to_pkl, 'wb') as f:
            pickle.dump(results, f)
    logging.info(f'\nEvaluation pipeline started.')
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
    results['hyperparameters'] = hyperparameters
    results['evaluation'] = evaluation
    # save results
    with open(f'results/{args.data}/{study_name}.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()