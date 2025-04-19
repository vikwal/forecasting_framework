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
          output_dim: int,
          model: keras.Model,
          scaler_y: StandardScaler = None) -> Tuple[np.ndarray, np.ndarray]:
    if (type(X_test) == dict):
        if ('static_input' in X_test.keys()):
            if (len(X_test['static_input']) == 0): del X_test['static_input']
    y_pred = model.predict(X_test).reshape(-1, y_test.shape[-1])
    if scaler_y:
        y_pred = scaler_y.inverse_transform(y_pred)
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, output_dim))
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
        os.remove(f'{path}.db')
        study = None
    return study

def split_val(X: Any,
              y: np.ndarray,
              val_split):
    if val_split == 0:
        return X, y, None, None
    val_index = int(len(y)*(1-val_split))
    # case for tft
    if type(X) == dict:
        X_train, X_val = {}, {}
        for key, value in X.items():
            if len(value) == 0:
                continue
            X_train[key] = value[:val_index]
            X_val[key] = value[val_index:]
    else:
        X_train = X[:val_index]
        X_val = X[val_index:]
    y_train = y[:val_index]
    y_val = y[val_index:]
    return X_train, y_train, X_val, y_val

def kfolds(X: Any,
           y: np.ndarray,
           n_splits: int,
           val_split: float = None) -> List:
    kfolds = []
    if n_splits == 1: # if not kfolds
        X_train, y_train, X_val, y_val = split_val(X=X, y=y, val_split=val_split)
        kfolds.append(((X_train, y_train), (X_val, y_val)))
        return kfolds
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, val_index in tscv.split(y):
        if type(X) == dict:
            X_train, X_val = {}, {}
            for key, value in X.items():
                if len(value) == 0:
                    continue
                X_train[key] = value[train_index]
                X_val[key] = value[val_index]
        else:
            X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        kfolds.append(((X_train, y_train), (X_val, y_val)))
    return kfolds



def get_hyperparameters(config: dict,
                        hpo=False,
                        trial=None,
                        study=None) -> dict:
    hyperparameters = {}
    model_name = config['model']['name']
    batch_size = config['hpo']['batch_size']
    epochs = config['hpo']['epochs']
    n_layers = config['hpo']['n_layers']
    n_cnn_layers = config['hpo']['n_cnn_layers']
    n_rnn_layers = config['hpo']['n_rnn_layers']
    learning_rate = config['hpo']['learning_rate']
    filters = config['hpo']['cnn']['filters']
    kernel_size = config['hpo']['cnn']['kernel_size']
    rnn_units = config['hpo']['rnn']['units']
    fnn_units = config['hpo']['fnn']['units']
    n_heads = config['hpo']['tft']['n_heads']
    hidden_dim = config['hpo']['tft']['hidden_dim']
    dropout = config['hpo']['tft']['dropout']
    lookback = config['model']['lookback']#config['hpo']['tft']['lookback']
    horizon = config['model']['horizon']
    is_cnn_type = 'cnn' in model_name or 'tcn' in model_name
    is_rnn_type = 'lstm' in model_name or 'gru' in model_name
    is_fnn_type = model_name == 'fnn'
    is_tft_type = model_name == 'tft'
    if hpo:
        hyperparameters['batch_size'] = trial.suggest_int('batch_size', batch_size[0], batch_size[1])
        hyperparameters['epochs'] = trial.suggest_int('epochs', epochs[0], epochs[1])
        hyperparameters['lr'] = trial.suggest_float('lr', learning_rate[0], learning_rate[1], log=True)
        if is_cnn_type:
            hyperparameters['filters'] = trial.suggest_int('filters', filters[0], filters[1])
            hyperparameters['kernel_size'] = trial.suggest_int('kernel_size', kernel_size[0], kernel_size[1])
            hyperparameters['n_cnn_layers'] = trial.suggest_int('n_cnn_layers', n_cnn_layers[0], n_cnn_layers[1])
        if is_rnn_type:
            hyperparameters['units'] = trial.suggest_int('units', rnn_units[0], rnn_units[1])
            hyperparameters['n_rnn_layers'] = trial.suggest_int('n_rnn_layers', n_rnn_layers[0], n_rnn_layers[1])
        if is_fnn_type:
            hyperparameters['n_layers'] = trial.suggest_int('n_layers', n_layers[0], n_layers[1])
            hyperparameters['units'] = trial.suggest_int('units', fnn_units[0], fnn_units[1])
        if is_tft_type:
            hyperparameters['horizon'] = horizon
            hyperparameters['lookback'] = lookback#trial.suggest_categorical('lookback', lookback)
            hyperparameters['n_heads'] = trial.suggest_int('n_heads', n_heads[0], n_heads[1])
            hyperparameters['hidden_dim'] = trial.suggest_int('hidden_dim', hidden_dim[0], hidden_dim[1])
            hyperparameters['dropout'] = trial.suggest_float('dropout', dropout[0], dropout[1])
    else:
        if study and study.best_trial:
            hyperparameters.update(study.best_trial)
        else:
            hyperparameters['batch_size'] = config['model']['batch_size']
            hyperparameters['epochs'] = config['model']['epochs']
            hyperparameters['lr'] = config['model']['lr']
            if is_cnn_type:
                hyperparameters['filters'] = config['model']['cnn']['filters']
                hyperparameters['kernel_size'] = config['model']['cnn']['kernel_size']
                hyperparameters['n_cnn_layers'] = config['model']['cnn']['n_cnn_layers']
            if is_rnn_type:
                hyperparameters['units'] = config['model']['rnn']['units']
                hyperparameters['n_rnn_layers'] = config['model']['rnn']['n_rnn_layers']
            if is_fnn_type:
                hyperparameters['n_layers'] = config['model']['fnn']['n_layers']
                hyperparameters['units'] = config['model']['fnn']['units']
            if is_tft_type:
                hyperparameters['horizon'] = config['model']['tft']['horizon']
                hyperparameters['lookback'] = config['model']['tft']['lookback']
                hyperparameters['n_heads'] = config['model']['tft']['n_heads']
                hyperparameters['hidden_dim'] = config['model']['tft']['hidden_dim']
                hyperparameters['dropout'] = config['model']['tft']['dropout']
    return hyperparameters

def load_hyperparams(study_name: str,
                     config: dict):
    studies_dir = config['hpo']['studies_dir']
    study = load_study(studies_dir=studies_dir,
                       study_name=study_name)
    if study:
        return study.best_trial.params
    return None

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
        obs_new = np.concatenate((old['observed_input'], new['observed_input']))
        known_new = np.concatenate((old['known_input'], new['known_input']))
        static_new = np.concatenate((old['static_input'], new['static_input']))
        new = {
            'observed_input': obs_new,
            'known_input': known_new,
            'static_input': static_new}
        return new

def initialize_gpu(use_gpu: int = 0):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[use_gpu], 'GPU')
    else:
        print("No Physical GPUs found.")

def concatenate_processed_data(data_parts: List) -> Any:
    """Konkateniert eine Liste von verarbeiteten Daten (numpy arrays oder dicts)."""
    # Filtere None-Werte heraus, die durch Fehler/leere Splits entstanden sein könnten
    valid_parts = [p for p in data_parts if p is not None]
    if not valid_parts:
        return None

    first_item = valid_parts[0]

    if isinstance(first_item, np.ndarray):
        # Prüfe auf leere Arrays, bevor konkateniert wird
        valid_parts = [p for p in valid_parts if p.shape[0] > 0]
        return np.concatenate(valid_parts, axis=0)

    elif isinstance(first_item, dict):
         # Stelle sicher, dass alle Teile Dictionaries sind
         if not all(isinstance(p, dict) for p in valid_parts):
             raise TypeError("Mische Typen beim Konkatenieren von Dictionaries")

         combined_dict = {}
         # Nehme an, alle dicts haben dieselben Keys wie das erste
         keys = first_item.keys()
         for key in keys:
             arrays_to_concat = []
             for d in valid_parts:
                arrays_to_concat.append(d[key])

             if arrays_to_concat: # Nur konkatenieren, wenn es etwas gibt
                 # Sicherstellen, dass alle Teile für diesen Key numpy arrays sind
                 if not all(isinstance(arr, np.ndarray) for arr in arrays_to_concat):
                     raise TypeError(f"Nicht-Numpy-Array gefunden für Key '{key}' beim Konkatenieren innerhalb des Dictionaries.")
                 combined_dict[key] = np.concatenate(arrays_to_concat, axis=0)
             else:
                 # Fallback: Leeres Array oder None, je nach Anforderung
                 # Hier verwenden wir None, um anzuzeigen, dass keine Daten für diesen Key vorhanden waren
                 combined_dict[key] = None
         return combined_dict

def combine_kfolds(n_splits: int,
                   kfolds_per_series: list):
    combined_kfolds = []
    for i in range(n_splits): # Iteriere über die Split-Indizes (0 bis k-1)
        xtr_parts = []
        ytr_parts = []
        xval_parts = []
        yval_parts = []

        # Sammle die Daten des i-ten Splits von JEDER Serie
        for series_folds in kfolds_per_series:
            (xtr, ytr), (xval, yval) = series_folds[i]

            xtr_parts.append(xtr)
            ytr_parts.append(ytr)
            xval_parts.append(xval)
            yval_parts.append(yval)

        # Konkateniere die gesammelten Teile für diesen Split-Index i
        # Konkateniere y (sollten immer numpy arrays sein)
        y_train_combined = np.concatenate([p for p in ytr_parts if p is not None and p.shape[0]>0], axis=0)
        y_val_combined = np.concatenate([p for p in yval_parts if p is not None and p.shape[0]>0], axis=0)

        # Konkateniere X mithilfe der Helper-Funktion
        X_train_combined = concatenate_processed_data(xtr_parts)
        X_val_combined = concatenate_processed_data(xval_parts)

        # Füge den kombinierten Split zur finalen Liste hinzu
        combined_kfolds.append(((X_train_combined, y_train_combined), (X_val_combined, y_val_combined)))
    return combined_kfolds