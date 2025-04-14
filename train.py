# Train model on all parks sequentially

import os
import yaml
import argparse
import pandas as pd
from tensorflow import keras
from tqdm import tqdm
import logging

import optuna

import preprocessing
import utils
import utils_eval
import pickle

optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level=logging.DEBUG,
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
    # read config
    config = utils.load_config('config.yaml')
    freq = config['data']['freq']
    output_dim, lag_dim, horizon = utils.handle_freq(freq=freq,
                                                     output_dim=config['model']['output_dim'],
                                                     lag_dim=config['data']['lag_dim'],
                                                     horizon=config['data']['horizon'])
    output_dim = 1 # dont forget to delete
    config['model']['output_dim'] = output_dim
    config['data']['horizon'] = horizon
    config['data']['lag_dim'] = lag_dim
    config['model']['shuffle'] = False
    # load and prepare training and test data
    dfs = preprocessing.get_data(data=args.data,
                                 data_dir=config['data']['path'],
                                 freq=freq)
    results = {}
    # create lag features
    evaluation = pd.DataFrame()
    for key, df in tqdm(dfs.items()):
        df = utils.impute_index(data=df)
        df = preprocessing.lag_features(df=df,
                                              lag_dim=lag_dim,
                                              horizon=horizon,
                                              lag_in_col=config['data']['lag_in_col'])
        windows = preprocessing.prepare_data(data=df,
                                            output_dim=output_dim,
                                            train_frac=config['data']['train_frac'],
                                            scale_y=config['data']['scale_y'])
        X_train, y_train, X_test, y_test = windows['X_train'], windows['y_train'], windows['X_test'], windows['y_test']
        #X_train, y_train = X_train[:len(X_train)*0.75], y_train[:len(y_train)*0.75]
        #X_val, y_val = X_train[len(X_train)*0.75:], y_train[len(y_train)*0.75:]
        results[key] = {}  # Initialize the key in the results dictionary
        #results[key]['test_data'] = (X_test, y_test)
        index_train, index_test = windows['index_train'], windows['index_test']
        scaler = windows['scaler']
        scaler_x = scaler[0]
        scaler_y = scaler[1]
        study = None#utils.load_study('studies/', f'{args.data}_{args.model}')
        hyperparameters = utils.get_hyperparameters(model_name=args.model,
                                                    config=config,
                                                    study=study)
        # set hyperparameters manually
        hyperparameters['batch_size'] = 16
        hyperparameters['epochs'] = 20
        hyperparameters['filters'] = 64
        hyperparameters['kernel_size'] = 2
        hyperparameters['n_layers'] = 2
        hyperparameters['n_cnn_layers'] = 2
        hyperparameters['n_rnn_layers'] = 2
        hyperparameters['lr'] = 0.0004
        hyperparameters['units'] = 64

        #logger.info(json.dumps(hyperparameters))
        train = X_train, y_train
        test = X_test, y_test #X_val, y_val
        id = key.split('.')[0][-2:]
        config['model_name'] = f'm-{args.model}_id-{id}_out-{output_dim}_freq-{freq}'
        history, model = utils.training_pipeline(train=train,
                                                 val=test,
                                                 hyperparameters=hyperparameters,
                                                 config=config)
        # save history and model
        results[key]['history'] = history
        results[key]['model'] = model

        new_evaluation = utils_eval.evaluation_pipeline(data=df,
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
    conf_name = f'm-{args.model}_out-{output_dim}_freq-{freq}'
    results['conf_name'] = conf_name
    # save results
    with open(f'results/{args.data}/d-{args.data}_{conf_name}.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()