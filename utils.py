import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from tensorflow import keras
import optuna

import preprocessing
import models

def persistence_2daysago(y:pd.Series,
                         horizon: int,
                         from_date=None) -> pd.Series:
    '''
    Persistence model which takes the realized scenario 2 days before.
    '''
    freq = (y.index[1] - y.index[0]).total_seconds() / 60
    if freq == 60:
        shifted = y.shift(horizon)
    elif freq == 15:
        shifted = y.shift(horizon*4)
        
    y_pers = pd.Series(data=shifted, index=y.index)
    if from_date:
        y_pers = y_pers[from_date:]
    return y_pers

def persistence_today(data:pd.DataFrame,
                      horizon: int):
    '''
    Persistence model which takes the last 12 hours and forecasts the next 12 hours.
    For 48 hours horizon this results in 2 identical days.
    '''
    y_pers = None
    return y_pers

def lin_reg(data: pd.DataFrame,
            train_end: str,
            test_start: str,
            target_col: str):
    '''
    Persistence model which uses linear regression.
    '''
    df = data.copy()
    df.dropna(inplace=True)
    y_train = df[:train_end][target_col].values
    df.drop([target_col], axis=1, inplace=True)
    X_train = df[:train_end].values
    X_test = df[test_start:].values
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pers = model.predict(X_test)
    return y_pers

def benchmark_models(data: pd.DataFrame,
                     target_col: str,
                     horizon: int,
                     train_end: str,
                     test_start: str,
                     output_dim: int,
                     index_test: np.ndarray,
                     t_0=None):
    '''
    Pipeline for applying different benchmark models.
    '''
    results = {}
    # 2 days ago persistence
    y_pers = persistence_2daysago(y=data[target_col],
                                  horizon=horizon,
                                  from_date=test_start)
    y_pers = preprocessing.make_windows(data=y_pers,
                                        output_dim=output_dim)
    df_pers = y_to_df(y=y_pers,
                      output_dim=output_dim,
                      horizon=horizon,
                      index_test=index_test,
                      t_0=t_0)
    results['2daysago'] = df_pers
    # linear regression persistence
    y_pers = lin_reg(data=data,
                     train_end=train_end,
                     test_start=test_start,
                     target_col=target_col)
    y_pers = preprocessing.make_windows(data=y_pers,
                                        output_dim=output_dim)
    
    df_pers = y_to_df(y=y_pers,
                      output_dim=output_dim,
                      horizon=horizon,
                      index_test=index_test,
                      t_0=t_0)
    results['LinearRegression'] = df_pers
    return results

def evaluate_models(pred: pd.DataFrame,
                    true: pd.DataFrame,
                    persistence: list,
                    persistence_model: str) -> pd.DataFrame:
    evaluation = get_metrics(y_pred=pred.values,
                             y_true=true.values)
    evaluation['Models'] = ['Main'] 
    for model, y_pred in persistence.items():
        evaluation['Models'].append(model)
        metrics = get_metrics(y_pred=y_pred.values,
                              y_true=true.values)
        for metric, value in metrics.items():
            evaluation[metric].append(value[0])
    results = pd.DataFrame(data=evaluation)
    results.set_index('Models', inplace=True)
    # skill factor
    results['Skill'] = 0.0
    for model in evaluation['Models']:
        results.loc[model, 'Skill'] = 1 - results.loc[model].RMSE / results.loc[persistence_model].RMSE
    return results

def get_metrics(y_pred: np.ndarray,
                y_true: np.ndarray) -> dict:
    error = y_pred - y_true
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(np.square(error).mean())
    mae = np.abs(error).mean()
    metrics = {'R^2': [r2],
               'RMSE': [rmse],
               'MAE': [mae]}
    return metrics

def get_y(X_test: np.ndarray,
          y_test: np.ndarray,
          output_dim: int,
          scaler_y: StandardScaler,
          model: keras.Model) -> Tuple[np.ndarray, np.ndarray]:
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
        for i in range(2, horizon + 1):
            df[f't+{i}'] = df.iloc[:, 0].shift(-i)
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

def load_study(path, study_name):
    storage = f'sqlite:///{path}{study_name}.db'
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


def training_pipeline(train: Tuple[np.ndarray, np.ndarray],
                      val: Tuple[np.ndarray, np.ndarray],
                      hyperparameters: dict,
                      config: dict):
    X_train, y_train = train
    n_features = X_train.shape[2]
    model = models.get_model(config=config,
                             n_features=n_features,
                             output_dim=config['model']['output_dim'],
                             hyperparameters=hyperparameters)
    history = model.fit(
        x = X_train,
        y = y_train,
        batch_size = hyperparameters['batch_size'],
        epochs = hyperparameters['epochs'],
        verbose = config['model']['verbose'],
        validation_data = val,
        #callbacks = callbacks,
        shuffle = False
    )
    return history