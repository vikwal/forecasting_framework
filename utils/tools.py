import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
import optuna

from . import models

def load_config(path: str):
    with open(path,'r') as file_object:
        config = yaml.load(file_object,Loader=yaml.SafeLoader)
    return config

def get_y(X_test: Any, # can be dict for tft or numpy array
          y_test: np.ndarray,
          model: keras.Model,
          scaler_y: StandardScaler = None) -> Tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test).reshape(-1, y_test.shape[-1])
    if scaler_y:
        y_pred = scaler_y.inverse_transform(y_pred)
        y_true = scaler_y.inverse_transform(y_test)
    else:
        y_true = y_test#.reshape(-1, output_dim)
    if len(y_pred.shape) == 3: # if seq2seq output
        y_pred = y_pred[:,:,-1] # take last output from seq
    y_pred[y_pred < 0] = 0
    return y_true, y_pred

def y_to_df(y: np.ndarray,
            output_dim: int,
            horizon: int,
            index_test: np.ndarray,
            t_0=None) -> pd.DataFrame:
    col_shape = y.shape[-1]
    cols = [f't+{i+1}' for i in range(col_shape)]
    df = pd.DataFrame(data=y, columns=cols, index=index_test)
    # should be solved simpler in future e.g. via prior making windows if neccesary
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



def get_feature_dim(X: Any):
    if type(X) == np.ndarray:
        feature_dim = X.shape[2]
    # relevant for tft
    elif (len(X) <= 3):
        feature_dim = {}
        feature_dim['observed_dim'] = X['observed_input'].shape[-1]
        feature_dim['known_dim'] = X['known_input'].shape[-1]
        feature_dim['static_dim'] = X['static_input'].shape[-1] if 'static_input' in X else 0
    return feature_dim


def training_pipeline(train: Tuple[np.ndarray, np.ndarray],
                      hyperparameters: dict,
                      config: dict,
                      val: Tuple[np.ndarray, np.ndarray] = None):
    X_train, y_train = train
    if val: X_val, y_val = val
    config['model']['feature_dim'] = get_feature_dim(X=X_train)
    model = models.get_model(config=config,
                             hyperparameters=hyperparameters)
    if config['model']['callbacks']:
        callbacks = [keras.callbacks.ModelCheckpoint(f'models/{config["model"]["name"]}.keras', save_best_only=True)]
    history = model.fit(
        x = X_train,
        y = y_train,
        batch_size = hyperparameters['batch_size'],
        epochs = hyperparameters['epochs'],
        verbose = config['model']['verbose'],
        validation_data = (X_val, y_val) if val else None,
        callbacks = callbacks if config['model']['callbacks'] else None,
        shuffle = config['model']['shuffle']
    )
    return history, model

def handle_freq(config: Dict[str, Any]) -> Tuple[int, int, int]:
    '''
    Adjust config output_dim, horizon and lookback in dependency of time series resolution (freq).
    '''
    freq = config['data']['freq']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    output_dim = config['model']['output_dim']
    if freq == '15min':
        if not output_dim == 1:
            output_dim = output_dim * 4
        horizon = horizon * 4
        lookback = lookback * 4
    if not output_dim == 1:
        horizon = output_dim
    config['data']['freq'] = freq
    config['model']['lookback'] = lookback
    config['model']['horizon'] = horizon
    config['model']['output_dim'] = output_dim
    return config

def concatenate_data(old, new):
    if type(old) == np.ndarray:
        return np.concatenate((old, new))
    elif type(old) == dict:
        result = {}
        for key, value in old.items():
            result[key] = np.concatenate((value, new[key]))
        return result

def initialize_gpu(use_gpu=None):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if use_gpu:
            tf.config.experimental.set_visible_devices(gpus[use_gpu], 'GPU')
        #else:
        #    strategy = tf.distribute.MirroredStrategy()
    else:
        print("No Physical GPUs found.")

