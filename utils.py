import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from tensorflow import keras
import logging
import optuna

import preprocessing
import models
import utils_eval



def load_config(path: str):
    with open('config.yaml','r') as file_object:
        config = yaml.load(file_object,Loader=yaml.SafeLoader)
    return config

def impute_index(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    freq = df.index[1] - df.index[0]
    start = df.index[0]
    end = df.index[-1]
    date_range = pd.date_range(start=start, end=end, freq=freq)
    if len(df) == len(date_range):
        return df
    df = df.reindex(date_range)
    if freq == pd.Timedelta('1h'):
        shift = 24
    elif freq == pd.Timedelta('15min'):
        shift = 96
    while df.isna().any().any():
        df = df.fillna(df.shift(shift))
    return df

def get_y(X_test: np.ndarray,
          y_test: np.ndarray,
          output_dim: int,
          model: keras.Model,
          scaler_y: StandardScaler = None) -> Tuple[np.ndarray, np.ndarray]:
    if scaler_y:
        y_pred = scaler_y.inverse_transform(model.predict(X_test))
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, output_dim))
    else:
        y_pred = model.predict(X_test)
        y_true = y_test#.reshape(-1, output_dim)
    y_pred[y_pred < 0] = 0
    return y_true, y_pred

def y_to_df(y: np.ndarray,
            output_dim: int,
            horizon: int,
            index_test: np.ndarray,
            t_0=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = [f't+{i+1}' for i in range(output_dim)]
    df = pd.DataFrame(data=y, columns=cols, index=index_test)
    if output_dim == 1:
        new_columns_dict = {}
        base_col_series = df.iloc[:, 0]
        for i in range(2, horizon + 1):
            shift_amount = -(i - 1)
            new_columns_dict[f't+{i}'] = base_col_series.shift(shift_amount)
        if new_columns_dict: # Nur konkatenieren, wenn neue Spalten erstellt wurden
            new_cols_df = pd.DataFrame(new_columns_dict, index=df.index)
            df = pd.concat([df, new_cols_df], axis=1)
        df.dropna(inplace=True)
    if t_0:
        df = df.loc[(df.index.time == pd.to_datetime(f'{t_0}:00:00').time())]
    return df

def create_or_load_study(path, study_name, direction):
    storage = f'sqlite:///{path}{study_name}.db'
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction=direction,
        load_if_exists=True
    )
    return study

def load_study(studies_dir: str,
               study_name: str):
    path = os.path.join(studies_dir, study_name)
    storage = f'sqlite:///{path}.db'
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
    except:
        study = None
    return study

def kfolds(X: np.ndarray,
           y: np.ndarray,
           n_splits: int) -> List:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    kfolds = []
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        kfolds.append(((X_train, y_train), (X_val, y_val)))
    return kfolds

def get_hyperparameters(model_name: str,
                        config: dict,
                        hpo=False,
                        trial=None,
                        study=None) -> dict:
    hyperparameters = {}
    hyperparameters['model_name'] = model_name
    if model_name == 'xgb':
        hyperparameters['objective'] = config['model']['xgb']['objective']
        hyperparameters['eval_metric'] = config['model']['xgb']['eval_metric']
        booster = config['model']['xgb']['booster']
        eta = config['model']['xgb']['eta']
        max_depth = config['model']['xgb']['max_depth']
        num_parallel_tree = config['model']['xgb']['num_parallel_tree']
        subsample = config['model']['xgb']['subsample']
        tree_method = config['model']['xgb']['tree_method']
        num_local_round = config['model']['xgb']['num_local_round']
        if hpo:
            #hyperparameters['booster'] = trial.suggest_categorical('booster', booster)
            hyperparameters['eta'] = trial.suggest_float('eta', eta[0], eta[1])
            hyperparameters['max_depth'] = trial.suggest_int('max_depth', max_depth[0], max_depth[1])
            hyperparameters['num_parallel_tree'] = trial.suggest_int('num_parallel_tree', num_parallel_tree[0], num_parallel_tree[1])
            hyperparameters['subsample'] = trial.suggest_float('subsample', subsample[0], subsample[1])
            hyperparameters['tree_method'] = trial.suggest_categorical('tree_method', tree_method)
            hyperparameters['num_local_round'] = trial.suggest_int('num_local_round', num_local_round[0], num_local_round[1])
            return hyperparameters
        hyperparameters['booster'] = booster#[0]
        hyperparameters['eta'] = eta[0]
        hyperparameters['max_depth'] = max_depth[0]
        hyperparameters['num_parallel_tree'] = num_parallel_tree[0]
        hyperparameters['subsample'] = subsample[0]
        hyperparameters['tree_method'] = tree_method[0]
        hyperparameters['num_local_round'] = num_local_round[0]
        return hyperparameters
    batch_size = config['hpo']['batch_size']
    epochs = config['hpo']['epochs']
    n_layers = config['hpo']['n_layers']
    learning_rate = config['hpo']['learning_rate']
    filters = config['hpo']['cnn']['filters']
    kernel_size = config['hpo']['cnn']['kernel_size']
    rnn_units = config['hpo']['rnn']['units']
    fnn_units = config['hpo']['fnn']['units']
    if hpo:
        hyperparameters['batch_size'] = trial.suggest_int('batch_size', batch_size[0], batch_size[1])
        hyperparameters['epochs'] = trial.suggest_int('epochs', epochs[0], epochs[1])
        hyperparameters['n_layers'] = trial.suggest_int('n_layers', n_layers[0], n_layers[1])
        hyperparameters['lr'] = trial.suggest_float('lr', learning_rate[0], learning_rate[1], log=True)
        if model_name == 'cnn' or model_name == 'tcn':
            hyperparameters['filters'] = trial.suggest_int('filters', filters[0], filters[1])
            hyperparameters['kernel_size'] = trial.suggest_int('kernel_size', kernel_size[0], kernel_size[1])
        elif model_name == 'lstm' or model_name == 'bilstm':
            hyperparameters['units'] = trial.suggest_int('units', rnn_units[0], rnn_units[1])
        else:
            hyperparameters['units'] = trial.suggest_int('units', fnn_units[0], fnn_units[1])
    else:
        if study:
            trial = study.best_trial
            for key, value in trial.params.items():
                hyperparameters[key] =  value
        else:
            hyperparameters['batch_size'] = batch_size[0]
            hyperparameters['epochs'] = epochs[0]
            hyperparameters['n_layers'] = n_layers[0]
            hyperparameters['lr'] = learning_rate[0]
            if model_name == 'cnn' or model_name == 'tcn':
                hyperparameters['filters'] = filters[0]
                hyperparameters['kernel_size'] = kernel_size[0]
            elif model_name == 'lstm' or model_name == 'bilstm':
                hyperparameters['units'] = rnn_units
            else:
                hyperparameters['units'] = fnn_units[0]
    return hyperparameters

def load_hyperparams(study_name: str,
                     config: dict):
    studies_dir = config['hpo']['studies_dir']
    study = load_study(studies_dir=studies_dir,
                       study_name=study_name)
    if study:
        return study.best_trial.params
    return None

def training_pipeline(train: Tuple[np.ndarray, np.ndarray],
                      hyperparameters: dict,
                      config: dict,
                      val: Tuple[np.ndarray, np.ndarray] = None):
    X_train, y_train = train
    n_features = X_train.shape[2]
    model = models.get_model(config=config,
                             n_features=n_features,
                             output_dim=config['model']['output_dim'],
                             hyperparameters=hyperparameters)
    if config['model']['callbacks']:
        callbacks = [keras.callbacks.ModelCheckpoint(f'models/{config["model_name"]}.keras', save_best_only=True)]
    logging.info(f'Training: {config["model_name"]} model')
    history = model.fit(
        x = X_train,
        y = y_train,
        batch_size = hyperparameters['batch_size'],
        epochs = hyperparameters['epochs'],
        verbose = config['model']['verbose'],
        validation_data = val,
        callbacks = callbacks if config['model']['callbacks'] else None,
        shuffle = True
    )
    return history, model

def handle_freq(freq: str,
                lag_dim: int,
                horizon: int,
                output_dim: int) -> Tuple[int, int]:
    '''
    Handle frequency and output dimension.
    '''
    if freq == '15min':
        if not output_dim == 1:
            output_dim = output_dim * 4
        horizon = horizon * 4
        lag_dim = lag_dim * 4
    if not output_dim == 1:
        horizon = output_dim
    return output_dim, lag_dim, horizon