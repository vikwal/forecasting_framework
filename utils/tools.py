import yaml
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler
import os
import logging
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

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
            index: np.ndarray,
            t_0=None) -> pd.DataFrame:
    index_for_df = index
    if index.shape[-1] == 3:
        index_for_df = index[:,0]
        if output_dim == 1:
            y = y.reshape(-1, horizon)
            index_for_df = np.array(list(set(index_for_df)))
            index_for_df.sort()
    col_shape = y.shape[-1]
    cols = [f't+{i+1}' for i in range(col_shape)]
    df = pd.DataFrame(data=y, columns=cols, index=index_for_df)
    # should be solved simpler in future e.g. via prior making windows if neccesary
    if output_dim == 1 and not index.shape[-1] == 3:
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
        df = df.loc[(df.index.time == pd.to_datetime(f'{t_0}:00').time())]
    return df



def get_feature_dim(X: Any):
    if type(X) == np.ndarray:
        # Nach Transposition ist die Form (samples, time, features), also feature_dim = X.shape[2]
        feature_dim = X.shape[2]
    # relevant for tft
    elif (len(X) <= 3):
        feature_dim = {}
        feature_dim['observed_dim'] = X['observed'].shape[-1]
        feature_dim['known_dim'] = X['known'].shape[-1]
        feature_dim['static_dim'] = X['static'].shape[-1] if 'static' in X else 0
    return feature_dim


def create_tf_dataset_from_arrays(X, y, batch_size, shuffle=True, buffer_size=5000):
    """
    Erstellt ein optimiertes tf.data.Dataset mit memory-effizienten Optionen.
    ALTERNATIVE: Nutze from_tensor_slices für bessere Performance, aber mit reduzierter Buffer-Size
    """
    logging.debug(f"Creating TensorFlow dataset with batch_size={batch_size}, shuffle={shuffle}")

    # OPTION 1: Verwende from_tensor_slices (einfacher, aber etwas mehr RAM)
    if isinstance(X, dict):
        # TFT case
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        # Standard case
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

    # Optimierte Pipeline
    if shuffle:
        # Reduzierte Buffer-Size für weniger RAM-Verbrauch
        actual_buffer_size = min(len(y), buffer_size)
        dataset = dataset.shuffle(
            buffer_size=actual_buffer_size,
            reshuffle_each_iteration=True
        )
        logging.debug(f"Applied shuffling with buffer_size={actual_buffer_size}")

    # Batch und Prefetch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    logging.debug(f"Dataset created successfully")
    return dataset

def training_pipeline(train: Tuple[np.ndarray, np.ndarray],
                      hyperparameters: dict,
                      config: dict,
                      val: Tuple[np.ndarray, np.ndarray] = None):
    X_train, y_train = train
    if val: X_val, y_val = val
    config['model']['feature_dim'] = get_feature_dim(X=X_train)
    parallelize = config['model'].get('parallelize', False)

    if parallelize:
        available_gpus = tf.config.list_physical_devices('GPU')
        n_gpus = len(available_gpus)

        if n_gpus > 1:
            logging.info(f"Found {n_gpus} GPUs for parallel training")
            original_batch_size = hyperparameters.get('batch_size')

            if original_batch_size % n_gpus != 0:
                adjusted_batch_size = original_batch_size
                while adjusted_batch_size % n_gpus != 0:
                    adjusted_batch_size += 1

                hyperparameters['batch_size'] = adjusted_batch_size
                logging.warning(f"Adjusted batch size from {original_batch_size} to {adjusted_batch_size} to be divisible by {n_gpus} GPUs")
                logging.debug(f"Each GPU will process {adjusted_batch_size // n_gpus} samples per batch")
            else:
                logging.debug(f"Batch size {original_batch_size} is divisible by {n_gpus} GPUs")
                logging.debug(f"Each GPU will process {original_batch_size // n_gpus} samples per batch")

            # Create strategy
            strategy = tf.distribute.MirroredStrategy()
            logging.debug(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
        else:
            logging.debug("Only 1 GPU available, using single GPU training")
            strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.get_strategy()  # DefaultStrategy (no distribution)
        logging.debug("Using default strategy (single device)")

    with strategy.scope():
        model = models.get_model(config=config, hyperparameters=hyperparameters)
        batch_size = hyperparameters['batch_size']

        # MEMORY OPTIMIZED: Use generator-based dataset creation
        logging.debug("Creating memory-efficient training dataset...")
        train_dataset = create_tf_dataset_from_arrays(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=config['model']['shuffle'],
            buffer_size=5000  # Reduziert von 10000 für weniger RAM
        )

        val_dataset = None
        if val:
            logging.debug("Creating validation dataset...")
            val_dataset = create_tf_dataset_from_arrays(
                X_val, y_val,
                batch_size=batch_size,
                shuffle=False  # Validation nicht shuffeln
            )

        # Explizit freigeben nach Dataset-Erstellung
        logging.debug("Cleaning up arrays after dataset creation...")
        del X_train, y_train
        if val:
            del X_val, y_val
        gc.collect()

    if config['model']['callbacks']:
        callbacks = [keras.callbacks.ModelCheckpoint(f'models/{config["model"]["name"]}.keras', save_best_only=True)]

    history = model.fit(
        x=train_dataset,
        epochs=hyperparameters['epochs'],
        verbose=config['model']['verbose'],
        validation_data=val_dataset,
        callbacks=callbacks if config['model']['callbacks'] else None
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

def create_data_generator(dfs, config, features):
    """
    Generator-basierte Datenverarbeitung für memory-effizientes Training.
    Lädt und verarbeitet Dateien einzeln, um RAM-Verbrauch zu minimieren.
    """
    # Import hier um zirkuläre Imports zu vermeiden
    from . import preprocessing

    for key, df in dfs.items():
        logging.debug(f'Processing {key} in generator.')
        prepared_data, _ = preprocessing.pipeline(
            data=df,
            config=config,
            known_cols=features['known'],
            observed_cols=features['observed'],
            static_cols=features['static'],
            target_col=config['data']['target_col']
        )

        yield {
            'key': key,
            'X_train': prepared_data['X_train'],
            'y_train': prepared_data['y_train'],
            'X_test': prepared_data['X_test'],
            'y_test': prepared_data['y_test'],
            'index_test': prepared_data['index_test'],
            'scalers': prepared_data['scalers']
        }

        # Explizit freigeben
        del prepared_data
        del df
        import gc
        gc.collect()

def combine_datasets_efficiently(data_generator):
    """
    Kombiniert Datasets memory-effizient durch schrittweise Akkumulation.
    Vermeidet RAM-Verdopplung durch temporäre Arrays.
    """
    X_train, y_train = None, None
    X_test, y_test = None, None
    test_data = {}
    total_samples = 0

    for data_dict in data_generator:
        key = data_dict['key']
        logging.debug(f'Combining data from {key}')

        # Für Test-Daten: Speichere separat (weniger problematisch wegen kleinerer Größe)
        test_data[key] = (
            data_dict['X_test'],
            data_dict['y_test'],
            data_dict['index_test'],
            data_dict['scalers']['y']
        )

        # Kombiniere auch Test-Daten für globale Auswertung
        if X_test is None:
            X_test = data_dict['X_test']
            y_test = data_dict['y_test']
        else:
            # Memory-effizientere Konkatenation für Test-Daten
            if isinstance(X_test, dict):
                # TFT case
                new_X_test = {}
                for feature_key in X_test.keys():
                    old_len = X_test[feature_key].shape[0]
                    new_len = data_dict['X_test'][feature_key].shape[0]
                    combined_shape = (old_len + new_len,) + X_test[feature_key].shape[1:]

                    combined_array = np.empty(combined_shape, dtype=X_test[feature_key].dtype)
                    combined_array[:old_len] = X_test[feature_key]
                    combined_array[old_len:] = data_dict['X_test'][feature_key]

                    new_X_test[feature_key] = combined_array

                del X_test
                X_test = new_X_test
            else:
                # Standard numpy case
                old_len = X_test.shape[0]
                new_len = data_dict['X_test'].shape[0]
                combined_shape = (old_len + new_len,) + X_test.shape[1:]

                combined_X_test = np.empty(combined_shape, dtype=X_test.dtype)
                combined_X_test[:old_len] = X_test
                combined_X_test[old_len:] = data_dict['X_test']

                del X_test
                X_test = combined_X_test

            # Gleiches für y_test
            old_len = y_test.shape[0]
            new_len = data_dict['y_test'].shape[0]
            combined_shape = (old_len + new_len,) + y_test.shape[1:]

            combined_y_test = np.empty(combined_shape, dtype=y_test.dtype)
            combined_y_test[:old_len] = y_test
            combined_y_test[old_len:] = data_dict['y_test']

            del y_test
            y_test = combined_y_test

        # Für Training-Daten: Akkumuliere effizient
        if X_train is None:
            X_train = data_dict['X_train']
            y_train = data_dict['y_train']
        else:
            # Memory-effizientere Konkatenation
            if isinstance(X_train, dict):
                # TFT case
                new_X_train = {}
                for feature_key in X_train.keys():
                    # Erstelle neues Array direkt in korrekter Größe
                    old_len = X_train[feature_key].shape[0]
                    new_len = data_dict['X_train'][feature_key].shape[0]
                    combined_shape = (old_len + new_len,) + X_train[feature_key].shape[1:]

                    combined_array = np.empty(combined_shape, dtype=X_train[feature_key].dtype)
                    combined_array[:old_len] = X_train[feature_key]
                    combined_array[old_len:] = data_dict['X_train'][feature_key]

                    new_X_train[feature_key] = combined_array

                # Explizit alte Arrays freigeben
                del X_train
                X_train = new_X_train
            else:
                # Standard numpy case
                old_len = X_train.shape[0]
                new_len = data_dict['X_train'].shape[0]
                combined_shape = (old_len + new_len,) + X_train.shape[1:]

                combined_X = np.empty(combined_shape, dtype=X_train.dtype)
                combined_X[:old_len] = X_train
                combined_X[old_len:] = data_dict['X_train']

                del X_train  # Explizit freigeben
                X_train = combined_X

            # Gleiches für y_train
            old_len = y_train.shape[0]
            new_len = data_dict['y_train'].shape[0]
            combined_shape = (old_len + new_len,) + y_train.shape[1:]

            combined_y = np.empty(combined_shape, dtype=y_train.dtype)
            combined_y[:old_len] = y_train
            combined_y[old_len:] = data_dict['y_train']

            del y_train  # Explizit freigeben
            y_train = combined_y

        total_samples += len(data_dict['y_train'])
        logging.debug(f'Combined data now has {total_samples} samples')

        # Explizit freigeben
        del data_dict
        import gc
        gc.collect()

    return X_train, y_train, X_test, y_test, test_data

def initialize_gpu(use_gpu=None):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if use_gpu is not None:
            if isinstance(use_gpu, list):
                selected_gpus = [gpus[i] for i in use_gpu if i < len(gpus)]
            else:
                selected_gpus = [gpus[use_gpu]] if use_gpu < len(gpus) else gpus
            tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')
            print(f"Using GPUs: {[gpu.name for gpu in selected_gpus]}")
        else:
            print(f"Using all available GPUs: {[gpu.name for gpu in gpus]}")
    else:
        print("No Physical GPUs found.")

