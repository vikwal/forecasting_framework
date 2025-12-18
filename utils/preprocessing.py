
import os
import random
import warnings
import re
import gc
import math
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from geopy.distance import geodesic
import metpy.calc as mpcalc
from metpy.units import units
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import logging
from typing import List, Tuple, Dict, Any
from sklearn.decomposition import PCA

from . import meteo


def _get_data_from_config_files(config: dict,
                                freq: str,
                                features: dict = None,
                                target_col: str = 'power') -> Dict[str, pd.DataFrame]:
    """
    Load data based on parks specified in config.

    Args:
        config: Configuration dictionary containing 'data' section with 'path' and 'parks'
        freq: Frequency for resampling
        rel_features: Relevant features to include
        target_col: Target column name

    Returns:
        Dictionary mapping client names to DataFrames
    """
    data_config = config.get('data', {})
    base_path = data_config.get('path', '')
    files = data_config.get('files', [])

    if not files:
        raise ValueError("Files list is empty in config")

    if not base_path:
        raise ValueError("Path is not specified in config")

    dfs = {}

    # Determine the data type based on config structure or path patterns
    # Check if this is wind data (based on config or path)
    is_wind_data = ('wind' in base_path.lower() or
                   'turbines' in config.get('params', {}))

    is_pv_data = ('pv' in base_path.lower() or
                  'solar' in base_path.lower())
    for file in files:
        # Try different possible file patterns
        possible_paths = [
            os.path.join(base_path, f"{file}.csv"),
            os.path.join(base_path, f"synth_{file}.csv"),
            os.path.join(base_path, file)  # in case file already contains .csv
        ]

        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break

        if not file_path:
            logging.warning(f"File not found for file {file} in any expected format, skipping")
            continue

        try:
            if is_wind_data:
                # Check NWP model type for wind data
                if 'open-meteo' in config.get('data', {}).get('nwp_path', ''):
                    df = preprocess_synth_wind_openmeteo(path=file_path,
                                                       config=config,
                                                       freq=freq,
                                                       features=features.copy() if features else None,
                                                       target_col=target_col)
                elif 'icon-d2' in config.get('data', {}).get('nwp_path', '') or 'icon_d2' in config.get('data', {}).get('nwp_path', ''):
                    df = preprocess_synth_wind_icond2(path=file_path,
                                                        config=config,
                                                        freq=freq,
                                                        features=features.copy() if features else None)
                else:
                    raise ValueError('NWP model source is not known for wind data')

            elif is_pv_data:
                df = preprocess_synth_pv(path=file_path,
                                       freq=freq,
                                       features=features)
            else:
                # Generic CSV loading - try to infer preprocessing method
                logging.warning(f"Unknown data type for {file_path}, using generic CSV loading")
                df = pd.read_csv(file_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    if freq:
                        df = df.resample(freq).mean()
            key = os.path.basename(file_path)
            dfs[key] = df
            logging.debug(f"Successfully loaded data for file: {key}")
        except Exception as e:
            logging.error(f"Error loading data for file {file}: {str(e)}")
            continue
    if not dfs:
        raise ValueError("No valid data files were loaded from the specified files")
    return dfs


def process_large_file_chunked(file_path: str,
                             preprocess_func: callable,
                             preprocess_kwargs: dict,
                             chunk_size: int = 50000) -> pd.DataFrame:
    """
    Verarbeitet große Dateien in Chunks um RAM-Verbrauch zu reduzieren.
    """
    logging.info(f"Processing large file {file_path} in chunks of {chunk_size}")

    # Lese Datei-Info ohne alles zu laden
    try:
        # Bestimme Dateigröße
        file_size = os.path.getsize(file_path)
        logging.debug(f"File size: {file_size / (1024**3):.2f} GB")

        # Für sehr große Dateien verwende chunked processing
        if file_size > 100 * 1024 * 1024:  # > 100MB
            logging.debug(f"Using chunked processing for large file")

            # Lese in Chunks
            chunk_list = []
            for chunk in pd.read_csv(file_path, sep=';', chunksize=chunk_size):
                # Verarbeite jeden Chunk separat
                preprocess_kwargs['path'] = file_path  # Update path for each chunk
                processed_chunk = preprocess_func(**preprocess_kwargs, data_chunk=chunk)
                chunk_list.append(processed_chunk)

                # Explizit freigeben
                del chunk, processed_chunk
                gc.collect()

            # Kombiniere Chunks
            result = pd.concat(chunk_list, ignore_index=True)
            del chunk_list
            gc.collect()

            return result
        else:
            # Standard processing für kleinere Dateien
            preprocess_kwargs['path'] = file_path
            return preprocess_func(**preprocess_kwargs)

    except Exception as e:
        logging.warning(f"Chunked processing failed: {e}, falling back to standard processing")
        preprocess_kwargs['path'] = file_path
        return preprocess_func(**preprocess_kwargs)

def get_data(data_dir: str,
             freq: str,
             config: dict,
             features: dict = None,
             target_col: str = 'power') -> Dict[str, pd.DataFrame]:

    # Check if parks are specified in config - if so, use config-based loading
    if 'files' in config.get('data', {}):
        return _get_data_from_config_files(config, freq, features, target_col)

    # Otherwise, use the existing dataset_name-based loading
    elif 'wind' in data_dir:
        logging.info(f'Getting the data from: {data_dir}')

        # Determine preprocessing function based on NWP model
        if 'open-meteo' in config['data']['nwp_path']:
            preprocess_func = preprocess_synth_wind_openmeteo
            preprocess_kwargs = {'path': None, 'config': config, 'freq': freq, 'features': features.copy(), 'target_col': target_col}
        elif 'icon-d2' in config['data']['nwp_path']:
            preprocess_func = preprocess_synth_wind_icond2
            preprocess_kwargs = {'path': None, 'config': config, 'freq': freq, 'features': features.copy()}
        else:
            raise ValueError('NWP model source is not known')

        return _process_files(data_dir, preprocess_func, preprocess_kwargs, file_filter=lambda f: f.endswith('.csv') and 'synth' in f)

    elif 'pv' in data_dir or 'solar' in data_dir:
        logging.info(f'Getting the data from: {data_dir}')

        preprocess_kwargs = {'path': None, 'freq': freq, 'features': features.copy()}
        return _process_files(data_dir, preprocess_synth_pv, preprocess_kwargs, file_filter=lambda f: f.endswith('.csv') and 'pvpark' in f)

    else:
        raise ValueError(f'Unknown data dir: {data_dir}. Please check the data directory or add a new preprocessing function.')

def split_data(data: pd.DataFrame,
               train_frac: float = 0.75,
               train_start: pd.Timestamp = None,
               test_start: pd.Timestamp = None,
               test_end: pd.Timestamp = None,
               t_0: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train and test sets.
    """
    df = data.copy()
    index = data.index

    # Split data into train and test sets
    if test_start:
        train_end = test_start - pd.Timedelta(hours=0.25)
    else:
        train_periods = int(len(df) * train_frac)-1
        train_end = index[train_periods].normalize() + pd.Timedelta(hours=(t_0-0.25))
        train_end = pd.Timestamp(train_end)
        test_start = train_end + pd.Timedelta(hours=0.25)
        test_start = pd.Timestamp(test_start)

    # handle MultiIndex if needed
    if type(data.index) == pd.core.indexes.multi.MultiIndex:
        # Ensure timestamps are timezone-aware (UTC) for comparison
        ts_test_start = pd.to_datetime(test_start).tz_convert('UTC') if pd.to_datetime(test_start).tzinfo else pd.to_datetime(test_start).tz_localize('UTC')
        if test_end:
            ts_test_end = pd.to_datetime(test_end).tz_convert('UTC') if pd.to_datetime(test_end).tzinfo else pd.to_datetime(test_end).tz_localize('UTC')
        else:
            ts_test_end = data.index.get_level_values('starttime').max()
        ts_train_end = pd.to_datetime(train_end).tz_convert('UTC') if pd.to_datetime(train_end).tzinfo else pd.to_datetime(train_end).tz_localize('UTC')
        if train_start is not None and not pd.isna(train_start):
            ts_train_start = pd.to_datetime(train_start).tz_convert('UTC') if pd.to_datetime(train_start).tzinfo else pd.to_datetime(train_start).tz_localize('UTC')
        else:
            ts_train_start = data.index.get_level_values('starttime').min()

        df_train = df[(df.index.get_level_values('starttime') < ts_test_start) &
                      (df.index.get_level_values('starttime') >= ts_train_start)]
        df_test = df[(df.index.get_level_values('starttime') > ts_train_end) &
                     (df.index.get_level_values('starttime') <= ts_test_end)]

    else:
        df_train = df[train_start:train_end]
        df_test = df[test_start:test_end]

    #print(df_train.index)
    #print(df_test.index)
    #logging.info(f'TRAIN START: {df_train.index.min()}')
    #logging.info(f'TRAIN END: {df_train.index.max()}')
    #logging.info(f'TEST START: {df_test.index.min()}')
    #logging.info(f'TEST END: {df_test.index.max()}')

    return df_train, df_test


def pipeline(data: pd.DataFrame,
             config: Dict[str, Any],
             known_cols: List[str] = None,
             observed_cols: List[str] = None,
             static_cols: List[str] = None,
             target_col: str = 'power') -> Tuple[Dict, pd.DataFrame]:
    df = data.copy()

    # MEMORY CLEANUP: Delete original data after copying
    del data
    gc.collect()

    df = knn_imputer(data=df, n_neighbors=config['data']['n_neighbors'])
    t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']

    if config['model']['name'] == 'tft':
        #logging.debug(f'Features ready for prepare_data(): {df.columns.to_list()}')
        prepared_data = prepare_data_for_tft(data=df,
                                             history_length=config['model']['lookback'], # e.g., 72 (for 3 days history)
                                             future_horizon=config['model']['horizon'], # e.g., 24 (for 24 hours forecast)
                                             step_size=config['model']['step_size'],
                                             known_future_cols=known_cols,
                                             observed_past_cols=observed_cols,
                                             static_cols=static_cols,
                                             train_frac=config['data']['train_frac'],
                                             train_start=pd.Timestamp(config['data'].get('train_start', None)),
                                             test_start=pd.Timestamp(config['data'].get('test_start', None)),
                                             test_end=pd.Timestamp(config['data'].get('test_end', None)),
                                             t_0=t_0,
                                             scaler_x=config.get('scaler_x', None))
    elif config['model']['name'] == 'stemgnn':
        n_nodes = config['params'].get('next_n_grid_points', 12)
        prepared_data = prepare_data_for_stemgnn(data=df,
                                                 n_nodes=n_nodes,
                                                 history_length=config['model']['lookback'],
                                                 future_horizon=config['model']['horizon'],
                                                 step_size=config['model']['step_size'],
                                                 train_frac=config['data']['train_frac'],
                                                 target_col=target_col,
                                                 train_start=pd.Timestamp(config['data'].get('train_start', None)),
                                                 test_start=pd.Timestamp(config['data'].get('test_start', None)),
                                                 test_end=pd.Timestamp(config['data'].get('test_end', None)),
                                                 t_0=t_0,
                                                 scale_target=config['data'].get('scale_y', False),
                                                 scaler_x=config.get('scaler_x', None))
    else:
        # create lag features for observed columns
        for col in observed_cols:
            all_known_cols = [new_col for new_col in df.columns if col in new_col]
            all_observed_cols = [new_col for new_col in df.columns if col in new_col]
            for new_col in all_observed_cols:
                df = lag_features(data=df,
                                lookback=config['model']['lookback'],
                                horizon=config['model']['horizon'],
                                lag_in_col=config['data']['lag_in_col'],
                                target_col=new_col)
                if new_col != target_col and new_col not in all_known_cols:
                    df.drop(new_col, axis=1, inplace=True, errors='ignore')
                    # MEMORY CLEANUP: Garbage collect after dropping columns
                    gc.collect()

        #logging.debug(f'Features ready for prepare_data(): {df.columns.to_list()}')
        prepared_data = prepare_data(data=df,
                                     output_dim=config['model']['output_dim'],
                                     step_size=config['model']['step_size'],
                                     train_frac=config['data']['train_frac'],
                                     scale_y=config['data']['scale_y'],
                                     t_0=t_0,
                                     train_start=pd.Timestamp(config['data'].get('train_start', None), tz='UTC'),
                                     test_start=pd.Timestamp(config['data'].get('test_start', None), tz='UTC'),
                                     test_end=pd.Timestamp(config['data'].get('test_end', None), tz='UTC'),
                                     target_col=target_col,
                                     scaler_x=config.get('scaler_x', None))

    # MEMORY CLEANUP: Force garbage collection after processing
    gc.collect()
    return prepared_data, df


def _process_files(target_dir: str, preprocess_func, preprocess_kwargs: dict, file_filter):
    """Helper function to process files in directory structure with optional client subdirectories."""
    client_dirs = [name for name in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, name))]
    dfs = defaultdict(dict)

    if not client_dirs:
        # Process files directly in target directory
        client_files = [f for f in os.listdir(target_dir) if file_filter(f)]
        for file in client_files:
            path = os.path.join(target_dir, file)
            preprocess_kwargs['path'] = path
            df = preprocess_func(**preprocess_kwargs)
            dfs[file] = df
    else:
        # Process files in client subdirectories
        for client in client_dirs:
            client_files = [f for f in os.listdir(os.path.join(target_dir, client)) if file_filter(f)]
            for file in client_files:
                path = os.path.join(target_dir, client, file)
                preprocess_kwargs['path'] = path
                df = preprocess_func(**preprocess_kwargs)
                dfs[f'{client}_{file}'] = df
    return dfs

def knn_imputer(data: pd.DataFrame,
               n_neighbors: int = 5):
    # To help KNNImputer estimating the temporal saisonalities we add encoded temporal features.
    index = data.index
    if type(index) == pd.core.indexes.multi.MultiIndex:
        index = data.index.get_level_values('timestamp')
    data['hour_sin'] = np.sin(2 * np.pi * index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * index.hour / 24)
    data['month_sin'] = np.sin(2 * np.pi * index.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * index.month / 12)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data)
    df = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(df_scaled)), columns=data.columns, index=data.index)
    df.drop(['hour_sin', 'hour_cos', 'month_sin', 'month_cos'], axis=1, inplace=True)
    #df[df < 0.01] = 0
    return df


def drop_days(frame: pd.DataFrame,
              target_col: str,
              threshold=12,
              verbose=True):
    grouped_by_day = frame.groupby(pd.Grouper(freq='D'))
    days_to_drop = grouped_by_day[target_col].apply(lambda x: x.isnull().sum() > threshold)
    days_to_keep = days_to_drop[~days_to_drop].index
    n_dropped = len(days_to_drop) - len(days_to_keep)
    days_dropped = days_to_drop[days_to_drop == True].index.to_list()
    if verbose:
        print('Number of dropped days:', n_dropped)
        print('Days dropped:')
        for day in days_dropped:
            print(day)
    frame = frame[frame.index.normalize().isin(days_to_keep)]
    return frame

def lag_features(data: pd.DataFrame,
                 lookback: int,
                 horizon: int,
                 target_col='power',
                 lag_in_col=False) -> pd.DataFrame:
    df = data.copy()
    if type(data.index) == pd.core.indexes.multi.MultiIndex:
        target_vec = df.groupby(df.index.get_level_values('timestamp')).mean()[target_col].to_frame()
    else:
        target_vec = df[[target_col]].copy()
    if lag_in_col:
        lags = list(range(horizon, lookback + 1))
    else:
        lags = list(range(horizon, lookback + 1, horizon))
    for lag in lags:
        target_vec[f"{target_col}_lag_{lag}"] = target_vec[target_col].shift(lag)
    target_vec.drop(target_col, axis=1, inplace=True)
    if type(data.index) == pd.core.indexes.multi.MultiIndex:
        df.reset_index(inplace=True)
        df = df.merge(
            target_vec,
            how="left",
            left_on="timestamp",
            right_index=True
        )
        df.set_index(["starttime", "forecasttime", "timestamp"], inplace=True)
    else:
        df = df.merge(
            target_vec,
            how="left",
            left_index=True,
            right_index=True
        )
    return df

def make_windows_efficient(data: np.ndarray,
                           seq_len: int,
                           step_size: int = 1,
                           indices: pd.DatetimeIndex = None,
                           use_memmap: bool = False) -> np.ndarray:
    """
    Memory-effiziente Windowing-Funktion mit optionaler Memory-Mapping.
    """
    n_windows = (data.shape[0] - seq_len) // step_size + 1

    if use_memmap and n_windows > 1000:  # Nur bei vielen Fenstern
        import tempfile
        # Erstelle temporäre Memory-Mapped Datei
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        # Erstelle Memory-Mapped Array
        windows_shape = (n_windows, seq_len) + data.shape[1:]
        windows = np.memmap(temp_file.name, dtype=data.dtype, mode='w+', shape=windows_shape)

        # Fülle Memory-Mapped Array
        for i, start_idx in enumerate(range(0, data.shape[0] - seq_len + 1, step_size)):
            windows[i] = data[start_idx:start_idx + seq_len]
    else:
        # Standard-Implementierung mit Stride-Tricks für bessere Performance
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            # Moderne NumPy sliding window (verfügbar ab NumPy 1.20.0)
            windows = sliding_window_view(data, window_shape=seq_len, axis=0)[::step_size]
        except ImportError:
            # Fallback auf stride_tricks
            from numpy.lib.stride_tricks import as_strided
            windows_shape = (n_windows, seq_len) + data.shape[1:]
            strides = (data.strides[0] * step_size, data.strides[0]) + data.strides[1:]
            windows = as_strided(data, shape=windows_shape, strides=strides)

    # Indizierung wie vorher
    if hasattr(data, 'index') and indices is not None:
        index = data.index
        freq = data.index[1] - data.index[0]
        shifted_index = index
        shifted_index = shifted_index[:shifted_index.shape[0]-seq_len+1:step_size]
        mask = np.array([True if i in indices else False for i in shifted_index])
        windows = windows[mask]

    return windows

def make_windows(data: np.ndarray,
                 seq_len: int,
                 step_size: int = 1,
                 indices: pd.DatetimeIndex = None) -> np.ndarray:
    """
    OPTIMIZED: Memory-effiziente Version der ursprünglichen Funktion.
    """
    # Verwende die effiziente Implementierung
    return make_windows_efficient(data, seq_len, step_size, indices, use_memmap=False)

def apply_scaling(df_train,
                  df_test,
                  scaler_type=StandardScaler,
                  fit=True):
    if callable(scaler_type):
        scaler = scaler_type()
    else:
        scaler = scaler_type

    if len(df_train.shape) == 1:
        df_train = df_train.values.reshape(-1, 1)
    if len(df_test.shape) == 1:
        df_test = df_test.values.reshape(-1, 1)

    if fit:
        scaled_train = scaler.fit_transform(df_train.values if hasattr(df_train, 'values') else df_train)
    else:
        scaled_train = scaler.transform(df_train.values if hasattr(df_train, 'values') else df_train)

    if len(df_test) == 0:
        scaled_test = df_test
    else:
        scaled_test = scaler.transform(df_test.values if hasattr(df_test, 'values') else df_test)
    return scaled_train, scaled_test, scaler

def prepare_data(data: pd.DataFrame,
                 output_dim: int,
                 step_size: int = 1,
                 train_frac: float = 0.75,
                 scale_x: bool = True,
                 scale_y: bool = False,
                 target_col: str = 'power',
                 t_0: int = 0,
                 train_start: pd.Timestamp = None,
                 test_start: pd.Timestamp = None,
                 test_end: pd.Timestamp = None,
                 seq2seq: bool = False,
                 scaler_x: StandardScaler = None):
    df = data.copy()
    df.dropna(inplace=True)
    index = data.index
    # MultiIndex handling
    if type(data.index) == pd.core.indexes.multi.MultiIndex:
        index = data.index.get_level_values('starttime')
        df = df.groupby(level='starttime').filter(
            lambda g: g.index.get_level_values('forecasttime').nunique() == 48
        )
    target = df[[target_col]]
    df.drop(target_col, axis=1, inplace=True)

    # Use helper function for splitting
    df_train, df_test = split_data(df, train_frac, train_start, test_start, test_end, t_0)
    target_train, target_test = split_data(target, train_frac, train_start, test_start, test_end, t_0)

    #logging.info(f"Training data range: {df_train.index.min()} to {df_train.index.max()} ({len(df_train)} rows)")
    #logging.info(f"Test data range:     {df_test.index.min()} to {df_test.index.max()} ({len(df_test)} rows)")
    scalers = {}
    scalers['x'] = None
    scalers['y'] = None
    if scale_x:
        if scaler_x:
            # Use provided global scaler
            X_train, X_test, scaler_x = apply_scaling(df_train,
                                                      df_test,
                                                      scaler_type=scaler_x,
                                                      fit=False)
            # Override the fitted scaler with the global one to be sure
            scaler_x = scaler_x
        else:
            # Fit new scaler
            X_train, X_test, scaler_x = apply_scaling(df_train,
                                                      df_test,
                                                      StandardScaler,
                                                      fit=True)
        scalers['x'] = scaler_x
    else:
        X_train = df_train.values
        X_test = df_test.values
    if scale_y:
        Y_train, Y_test, scaler_y = apply_scaling(target_train,
                                                  target_test,
                                                  StandardScaler)
        scalers['y'] = scaler_y
    else:
        Y_train = target_train
        Y_test = target_test
    # Keep indices as DatetimeIndex/MultiIndex instead of converting to numpy array
    # This is important for min_train_date filtering in HPO
    index_positions = range(0, X_train.shape[0]-output_dim+1, step_size)
    index_train = df_train.index[list(index_positions)]

    index_positions_test = range(0, X_test.shape[0]-output_dim+1, step_size)
    index_test = df_test.index[list(index_positions_test)]


    X_train = make_windows(data=X_train, seq_len=output_dim, step_size=step_size)
    X_test = make_windows(data=X_test, seq_len=output_dim, step_size=step_size)

    y_train = make_windows(data=Y_train, seq_len=output_dim, step_size=step_size).reshape(-1, output_dim)
    y_test = make_windows(data=Y_test, seq_len=output_dim, step_size=step_size).reshape(-1, output_dim)

    # FIXED: Transpose X arrays from (samples, features, time) to (samples, time, features) for Keras
    X_train = X_train.transpose(0, 2, 1)  # (samples, features, time) -> (samples, time, features)
    X_test = X_test.transpose(0, 2, 1)    # (samples, features, time) -> (samples, time, features)

    if seq2seq:
        y_train = y_train.reshape(-1, output_dim, 1)
        y_test = y_test.reshape(-1, output_dim, 1)
    results = {}
    results['X_train'], results['y_train'] = X_train, y_train
    results['index_train'], results['index_test'] = index_train, index_test
    results['X_test'], results['y_test'] = X_test, y_test
    results['scalers'] = scalers
    return results

def preprocess_synth_pv(path: str,
                        freq: str = '1H',
                        features: dict = None) -> pd.DataFrame:
    timestamp_col = 'timestamp'
    target_col = 'power'
    # Extract station_id from the path
    # We use os.path.basename to get the filename, then split to get the ID
    # Assumes filename format like 'synth_ID.csv' or 'ID.csv'
    filename = os.path.basename(path)
    # Remove extension
    filename_no_ext = os.path.splitext(filename)[0]
    # If it starts with 'synth_', remove it
    if filename_no_ext.startswith('synth_'):
        station_id = filename_no_ext.split('_')[-1]
    else:
        station_id = filename_no_ext
    parameter_file = path.replace(os.path.basename(path), 'pv_parameter.csv')
    metadata = pd.read_csv(parameter_file, sep=';')
    installed_capacity = metadata.loc[metadata.park_id == int(station_id)]['installed_capacity'].values[0]
    df = pd.read_csv(path, sep=';')
    df.drop('park_id', axis=1, inplace=True) # remove if not needed
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df.set_index(timestamp_col, inplace=True)
    df[target_col] = df[target_col] / installed_capacity # in Watts
    df = df.resample(freq, closed='left', label='left', origin='start').mean()
    rel_features = features['known'] + features['observed']
    if rel_features:
        return df[rel_features]
    return df

def get_saturated_vapor_pressure(temperature: pd.Series,
                                 model: str = 'improved_magnus') -> pd.Series:
    if temperature.mean() > 100:
        temperature = temperature - 273.15
    def huang(temp):
        return np.where(
            temp > 0,
            np.exp(34.494 - (4924.99 / (temp + 237.1))) / (temp + 105) ** 1.57,
            np.exp(43.494 - (6545.8 / (temp + 278))) / (temp + 868) ** 2
        )
    def improved_magnus(temp):
        return np.where(
            temp > 0,
            610.94 * np.exp((17.625 * temp) / (temp + 243.04)),
            611.21 * np.exp((22.587 * temp) / (temp + 273.86))
        )
    model_functions = {
        'huang': huang,
        'improved_magnus': improved_magnus,
    }
    if model not in model_functions:
        raise ValueError(f"Unknown model: {model}")
    return model_functions[model](temperature)

def get_density(temperature: pd.Series,
                pressure: pd.Series,
                relhum: pd.Series,
                sat_vap_ps: pd.Series) -> pd.Series:
    R_dry = 287.05  # Specific gas constant dry air (J/(kg·K))
    R_w = 461.5  # Specific gas constaint water vapor (J/(kg·K))
    # check if relative humidity is in the range between 0 and 1
    if relhum.max() > 1:
        relhum /= 100
    if temperature.mean() < 100:
        temperature += 273.15 # from celsius to kelvin
    if pressure.mean() < 5e3:
        pressure *= 100
    p_w = relhum * sat_vap_ps
    p_g = pressure - p_w
    rho_g = p_g / (R_dry * temperature)
    rho_w = p_w / (R_w * temperature)
    rho = rho_g + rho_w
    return rho

def circle_cap_area(r: float, y: float) -> float:
    """Fläche oberhalb einer Horizontalen im Kreis mit Radius r.
    Kreis ist im Ursprung (0,0) zentriert. y ist die Höhe der Linie."""
    if y <= -r:   # Linie ganz unten → ganze Kreisfläche
        return np.pi * r**2
    if y >= r:    # Linie ganz oben → keine Fläche
        return 0.0
    return r**2 * np.arccos(y/r) - y * np.sqrt(r**2 - y**2)

def get_A_weights(turbines: pd.DataFrame, layers: list) -> np.ndarray:
    """
    Calculates the farm-averaged rotor area weights for given height layers.
    Layers: list of (lower_bound, upper_bound, level_name)
    """
    A_total = 0
    total_A_slices = np.zeros(len(layers))
    for turbine in turbines.itertuples():
        r = turbine.diameter / 2
        height = turbine.hub_height
        turbine_area = np.pi * r**2
        A_total += turbine_area
        for i, (down, top, _) in enumerate(layers):
            y_top = top - height if top <= height + r else r
            y_down = down - height if down >= height - r else -r
            if y_down < r and y_top > -r:
                y_top = min(y_top, r)
                y_down = max(y_down, -r)
                A_slice = circle_cap_area(r, y_down) - circle_cap_area(r, y_top)
                total_A_slices[i] += A_slice
    if A_total > 0:
        return total_A_slices / A_total
    else:
        logging.warning("Total rotor area is zero, cannot compute weights.")
        return total_A_slices

def _process_csv_file(args):
    """
    Process a single CSV file for Icon-D2 data.
    This function is designed to be called in parallel.
    """
    csv_file, distance, grid_lat, grid_lon, csv_path, config, i, turbines = args

    try:
        # Load CSV file
        df_grid = pd.read_csv(csv_path)

        # Convert timestamp
        df_grid['starttime'] = pd.to_datetime(df_grid['starttime'], utc=True)
        df_grid['timestamp'] = df_grid['starttime'] + pd.to_timedelta(df_grid['forecasttime'], unit='h')

        # Calculate wind speed from u and v components
        df_grid['wind_speed'] = np.sqrt(df_grid['u_wind']**2 + df_grid['v_wind']**2)

        # Calculate height as middle between top and bottom level
        df_grid['height'] = ((df_grid['toplevel'] + df_grid['bottomlevel']) / 2).round().astype(int)

        # Calculate relative humidity from specific humidity
        specific_humidity = df_grid['qs'].values * units.dimensionless
        temp = df_grid['temperature'].values * units.kelvin
        press = df_grid['pressure'].values * units.pascal

        rel_humidity = mpcalc.relative_humidity_from_specific_humidity(
            press, temp, specific_humidity
        ).magnitude
        df_grid['relative_humidity'] = rel_humidity * 100  # Convert to percentage

        # Get density if required
        if config['params'].get('get_density', False):
            # Calculate saturated vapor pressure
            sat_vap_ps = get_saturated_vapor_pressure(temperature=df_grid['temperature'],
                                                     model='huang')
            # Calculate density using your method
            df_grid['density'] = get_density(temperature=df_grid['temperature'],
                                           pressure=df_grid['pressure'],
                                           relhum=df_grid['relative_humidity'],
                                           sat_vap_ps=sat_vap_ps)

        # Select relevant weather variables for pivoting
        weather_vars = ['wind_speed', 'temperature', 'pressure', 'qs', 'relative_humidity']
        if 'density' in df_grid.columns:
            weather_vars.append('density')

        # Pivot by height levels
        df_pivoted_list = []
        for var in weather_vars:
            if var in df_grid.columns:
                pivot_df = df_grid.pivot_table(
                    index=['timestamp', 'starttime', 'forecasttime'],
                    columns='height',
                    values=var,
                    aggfunc='mean'
                ).reset_index()
                # Rename columns with height suffix
                new_cols = ['timestamp', 'starttime', 'forecasttime']
                for col in pivot_df.columns[3:]:  # Skip timestamp columns
                    new_cols.append(f"{var}_h{col}")
                pivot_df.columns = new_cols
                df_pivoted_list.append(pivot_df)

        # Merge all pivoted variables for this grid point
        df_grid_final = df_pivoted_list[0]
        for df_pivot in df_pivoted_list[1:]:
            df_grid_final = df_grid_final.merge(
                df_pivot,
                on=['timestamp', 'starttime', 'forecasttime'],
                how='outer'
            )

        # Calculate rotor equivalent wind speed if requested
        if config['params'].get('aggregate_nwp_layers') == 'weighted_mean':
            # Identify available wind speed columns and heights
            wind_cols = [col for col in df_grid_final.columns if col.startswith('wind_speed_h')]
            if wind_cols:
                # Extract heights from column names
                heights = sorted([int(col.split('_h')[1]) for col in wind_cols])

                # Construct layers for get_A_weights
                if len(heights) > 1:
                    midpoints = [(heights[j] + heights[j + 1]) / 2 for j in range(len(heights) - 1)]
                    layers = []
                    layers.append((0.0, midpoints[0], heights[0]))
                    for j in range(len(midpoints) - 1):
                        layers.append((midpoints[j], midpoints[j+1], heights[j+1]))
                    layers.append((midpoints[-1], np.inf, heights[-1]))

                    area_weights = get_A_weights(turbines, layers)
                    weighted_speed_cubed_series = pd.Series(0.0, index=df_grid_final.index, dtype=float)

                    for j, (lower_bound, upper_bound, level_name) in enumerate(layers):
                        wind_col = f'wind_speed_h{level_name}'
                        if wind_col in df_grid_final.columns and area_weights[j] > 0:
                            weighted_speed_cubed_series += area_weights[j] * (df_grid_final[wind_col] ** 3)

                    df_grid_final['wind_speed_rotor_eq'] = (weighted_speed_cubed_series) ** (1/3)

            # Identify available density columns and heights
            density_cols = [col for col in df_grid_final.columns if col.startswith('density_h')]
            if density_cols:
                # Extract heights from column names
                heights = sorted([int(col.split('_h')[1]) for col in density_cols])

                # Construct layers for get_A_weights
                if len(heights) > 1:
                    midpoints = [(heights[j] + heights[j + 1]) / 2 for j in range(len(heights) - 1)]
                    layers = []
                    layers.append((0.0, midpoints[0], heights[0]))
                    for j in range(len(midpoints) - 1):
                        layers.append((midpoints[j], midpoints[j+1], heights[j+1]))
                    layers.append((midpoints[-1], np.inf, heights[-1]))

                    area_weights = get_A_weights(turbines, layers)
                    weighted_density_series = pd.Series(0.0, index=df_grid_final.index, dtype=float)

                    for j, (lower_bound, upper_bound, level_name) in enumerate(layers):
                        density_col = f'density_h{level_name}'
                        if density_col in df_grid_final.columns and area_weights[j] > 0:
                            weighted_density_series += area_weights[j] * df_grid_final[density_col]

                    df_grid_final['density_rotor_eq'] = weighted_density_series

        # Add distance suffix to all feature columns (except timestamp columns)
        feature_cols = [col for col in df_grid_final.columns if col not in ['timestamp', 'starttime', 'forecasttime']]
        rename_dict = {col: f"{col}_{i}" for col in feature_cols}
        df_grid_final.rename(columns=rename_dict, inplace=True)

        return df_grid_final, i, csv_file

    except Exception as e:
        logging.error(f"Error processing CSV file {csv_file}: {str(e)}")
        return None, i, csv_file


def _process_forecast_hour(args):
    """
    Process a single forecast hour for Icon-D2 data.
    This function is designed to be called in parallel.
    """
    forecast_hour, config, station_id, station_lat, station_lon, next_n_grid_points, turbines = args

    try:
        icon_d2_base_path = f"{config['data']['nwp_path']}/ML/{forecast_hour}/{station_id}"

        if not os.path.exists(icon_d2_base_path):
            logging.warning(f"Icon-D2 data path not found: {icon_d2_base_path}")
            return None, forecast_hour

        # Get all CSV files for this station
        csv_files = [f for f in os.listdir(icon_d2_base_path) if f.endswith('_ML.csv')]

        # Extract coordinates from filenames and calculate distances
        file_distances = []
        for csv_file in csv_files:
            # Extract coordinates from filename: lat_lon_lat_lon_ML.csv
            parts = csv_file.replace('_ML.csv', '').split('_')
            if len(parts) >= 4:
                try:
                    grid_lat = float(f"{parts[0]}.{parts[1]}")
                    grid_lon = float(f"{parts[2]}.{parts[3]}")

                    # Calculate distance to station
                    distance = geodesic((station_lat, station_lon), (grid_lat, grid_lon)).kilometers
                    # distance = math.sqrt((grid_lat - station_lat) ** 2 + (grid_lon - station_lon) ** 2)
                    file_distances.append((csv_file, distance, grid_lat, grid_lon))
                except (ValueError, IndexError):
                    continue

        # Sort by distance and take the nearest n points
        file_distances.sort(key=lambda x: x[1])
        nearest_files = file_distances[:next_n_grid_points]

        if not nearest_files:
            logging.warning(f"No valid files found for forecast hour {forecast_hour}")
            return None, forecast_hour

        # Prepare arguments for parallel CSV processing
        csv_args = []
        for i, (csv_file, distance, grid_lat, grid_lon) in enumerate(nearest_files, 1):
            csv_path = os.path.join(icon_d2_base_path, csv_file)
            csv_args.append((csv_file, distance, grid_lat, grid_lon, csv_path, config, i, turbines))

        # Process CSV files in parallel using ThreadPoolExecutor (I/O bound)
        dfs_list = []
        with ThreadPoolExecutor(max_workers=min(len(csv_args), 4)) as executor:
            future_to_csv = {executor.submit(_process_csv_file, arg): arg for arg in csv_args}

            for future in as_completed(future_to_csv):
                result = future.result()
                if result[0] is not None:
                    dfs_list.append((result[0], result[1]))  # (dataframe, index)

        # Sort by index to maintain order
        dfs_list.sort(key=lambda x: x[1])
        dfs_list = [df for df, _ in dfs_list]

        # Merge all grid point dataframes for this forecast hour
        if dfs_list:
            df_forecast_hour = dfs_list[0]
            for df_grid in dfs_list[1:]:
                df_forecast_hour = df_forecast_hour.merge(
                    df_grid,
                    on=['timestamp', 'starttime', 'forecasttime'],
                    how='outer'
                )

            # --- Aggregation Logic ---
            if config['params'].get('aggregate_grid_points', False):
                # 1. Calculate weights (Inverse Distance Weighting)
                # distances are in nearest_files: [(csv_file, distance, lat, lon), ...]
                # The dfs_list is sorted by index (1 to N), which corresponds to nearest_files sorted by distance.

                # Extract distances in order
                distances = [nf[1] for nf in nearest_files]

                # Handle zero distance (if station is exactly on a grid point)
                weights = []
                if any(d == 0 for d in distances):
                    for d in distances:
                        weights.append(1.0 if d == 0 else 0.0)
                else:
                    weights = [1.0 / d for d in distances]

                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                # 2. Identify base features and aggregate
                # Columns are like 'wind_speed_h120_1', 'wind_speed_h120_2', etc.
                # We want to produce 'wind_speed_h120' = w1*..._1 + w2*..._2 + ...

                # Get all columns except timestamps
                all_cols = df_forecast_hour.columns
                timestamp_cols = ['timestamp', 'starttime', 'forecasttime']
                feature_cols = [c for c in all_cols if c not in timestamp_cols]

                # Find unique base feature names (remove suffix _\d+)
                # We assume suffix is always _<index> where index corresponds to the grid point rank (1-based)
                base_features = set()
                for col in feature_cols:
                    # Match suffix _<digit> at the end
                    match = re.search(r'_(\d+)$', col)
                    if match:
                        base_name = col[:match.start()]
                        base_features.add(base_name)

                # Compute weighted sums
                for base_feat in base_features:
                    weighted_sum = 0
                    found_any = False
                    for i, weight in enumerate(weights):
                        # Grid point index is i+1
                        col_name = f"{base_feat}_{i+1}"
                        if col_name in df_forecast_hour.columns:
                            weighted_sum += df_forecast_hour[col_name] * weight
                            found_any = True

                    if found_any:
                        # Assign to new column without suffix
                        df_forecast_hour[base_feat] = weighted_sum

                # 3. Keep only aggregated columns and timestamps
                cols_to_keep = timestamp_cols + list(base_features)
                # Filter to keep only columns that actually exist
                cols_to_keep = [c for c in cols_to_keep if c in df_forecast_hour.columns]
                df_forecast_hour = df_forecast_hour[cols_to_keep]

            return df_forecast_hour, forecast_hour
        else:
            return None, forecast_hour

    except Exception as e:
        logging.error(f"Error processing forecast hour {forecast_hour}: {str(e)}")
        return None, forecast_hour


def preprocess_synth_wind_icond2(path: str,
                          config: dict,
                          freq: str = '1H',
                          features: dict = None) -> pd.DataFrame:
    """
    Preprocess synthetic wind data from Icon-D2 model.

    Args:
        path: Path to synthetic wind data (not used, station_id extracted from config)
        config: Configuration dictionary
        freq: Frequency for resampling
        features: Dictionary with 'known' and 'observed' features

    Returns:
        Preprocessed DataFrame
    """

    # Extract station_id from the path
    # We use os.path.basename to get the filename, then split to get the ID
    # Assumes filename format like 'synth_ID.csv' or 'ID.csv'
    filename = os.path.basename(path)
    # Remove extension
    filename_no_ext = os.path.splitext(filename)[0]
    # If it starts with 'synth_', remove it
    if filename_no_ext.startswith('synth_'):
        station_id = filename_no_ext.split('_')[-1]
    else:
        station_id = filename_no_ext

    # Load synthetic wind data and turbine parameters (analog zu openmeteo)
    synth_path = os.path.join(config['data']['path'], f'synth_{station_id}.csv')
    wind_parameter_path = os.path.join(config['data']['path'], 'wind_parameter.csv')
    turbine_parameter_path = os.path.join(config['data']['path'], 'turbine_parameter.csv')
    power_curves_path = os.path.join(config['data']['power_curves_path'], 'turbine_power.csv')

    # Load synthetic wind data and parameters
    df_synth = pd.read_csv(synth_path, sep=';')
    df_synth['timestamp'] = pd.to_datetime(df_synth['timestamp'], utc=True)
    df_synth.set_index('timestamp', inplace=True)

    wind_parameter = pd.read_csv(wind_parameter_path, sep=';', dtype={'park_id': str})
    turbine_parameter = pd.read_csv(turbine_parameter_path, sep=';', dtype={'park_id': str})
    power_curves = pd.read_csv(power_curves_path, sep=';', decimal=',')
    altitude = wind_parameter.loc[wind_parameter.park_id == station_id]['altitude'].values[0]
    # Handle commissioning date and park age
    commissioning_date = wind_parameter.loc[wind_parameter.park_id == station_id]['commissioning_date'].values[0]
    if 'random_seeds' not in config['params']:
        random.seed(config['params']['random_seed'])
    else:
        iterator = config['params'].get('iterator', 0)
        config['params']['iterator'] = iterator
        random_seed = config['params']['random_seeds'][iterator]
        config['params']['random_seed'] = random_seed
        random.seed(random_seed)
        config['params']['iterator'] = iterator + 1

    if commissioning_date != '-':
        park_age_years = (pd.Timestamp.now() - pd.to_datetime(commissioning_date)).days // 365
    else:
        park_age_years = 0

    # Turbine handling (analog zu openmeteo)
    static_features = features.get('static', [])
    static_data = {}
    heights = []

    if 'turbines' in config['params']:
        turbines_list = config['params']['turbines']
        if 'turbines_per_park' in config['params']:
            turbines_list = random.sample(turbines_list, config['params']['turbines_per_park'])
            config['params']['random_seed'] += 1
        power_curves = power_curves[turbines_list]
        installed_capacity = power_curves.sum(axis=1).max() * 1000 # in Watts
        turbines = turbine_parameter.loc[turbine_parameter.turbine_name.isin(turbines_list)]
        turbine_indices = turbines['turbine']
        heights = turbines['hub_height'].values
        power_cols = [f'power_{i}' for i in turbine_indices]
        wind_cols = [f'wind_speed_{i}' for i in turbine_indices]
        df_synth['power'] = df_synth[power_cols].sum(axis=1)

        # Keep only selected turbine columns + power + generic wind_speed
        keep_cols = power_cols + wind_cols + ['power']
        if 'wind_speed' in df_synth.columns:
            keep_cols.append('wind_speed')
        df_synth = df_synth[keep_cols]
    else:
        df_synth['power'] = 0
        for col in df_synth.columns:
            if 'power' in col:
                df_synth['power'] += df_synth[col]
        installed_capacity = df_synth['power'].max()
        heights = turbine_parameter['hub_height'].values
        turbines = turbine_parameter

    # Static features handling
    if static_features:
        static_data['park_id'] = station_id
        static_data['park_age'] = park_age_years
        static_data['installed_capacity'] = installed_capacity
        static_data['altitude'] = altitude
        for index, turbine in turbines.iterrows():
            turbine_name = turbine['turbine_name']
            turbine_id = turbine['turbine']
            static_data[f'hub_height'] = turbine['hub_height']
            static_data[f'rotor_diameter'] = turbine['diameter']
            static_data[f'rated_power'] = power_curves[turbine_name].max() * 1000
            static_data[f'cut_in'] = turbine['cut_in']
            static_data[f'cut_out'] = turbine['cut_out']
            static_data[f'rated_wind_speed'] = turbine['rated']


    # Normalize power
    df_synth['power'] = df_synth['power'] / installed_capacity
    df_synth = df_synth.resample('1H', closed='left', label='left', origin='start').mean()

    # Load station coordinates
    stations_path = config['data']['stations_master']
    stations_df = pd.read_csv(stations_path, sep=',', dtype={'station_id': str})
    station_info = stations_df[stations_df['station_id'] == station_id]

    if station_info.empty:
        raise ValueError(f"Station {station_id} not found in stations list")

    station_lat = station_info['latitude'].iloc[0]
    station_lon = station_info['longitude'].iloc[0]

    # Parameters from config
    next_n_grid_points = config['params'].get('next_n_grid_points', 12)
    forecast_hours = ['06', '09', '12', '15']  # Available forecast hours
    rel_features = list(set(features['known'] + features['observed'])) if features else []

    # Prepare arguments for parallel forecast hour processing
    forecast_args = []
    for forecast_hour in forecast_hours:
        forecast_args.append((forecast_hour, config, station_id, station_lat, station_lon, next_n_grid_points, turbines))

    # Process forecast hours in parallel using ProcessPoolExecutor (CPU bound)
    all_forecast_dfs = []
    max_workers = min(len(forecast_hours), mp.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_hour = {executor.submit(_process_forecast_hour, arg): arg[0] for arg in forecast_args}

        for future in as_completed(future_to_hour):
            hour = future_to_hour[future]
            result = future.result()
            if result[0] is not None:
                all_forecast_dfs.append((result[0], hour))

    # Sort by forecast hour to maintain consistent order
    hour_order = {'06': 0, '09': 1, '12': 2, '15': 3}
    all_forecast_dfs.sort(key=lambda x: hour_order.get(x[1], 99))
    all_forecast_dfs = [df for df, _ in all_forecast_dfs]

    if not all_forecast_dfs:
        raise ValueError("No data could be loaded from any forecast hour")

    # Combine all forecast hours - keep all columns including starttime, forecasttime
    df_final = pd.concat(all_forecast_dfs, ignore_index=False)

    # 1. Filter out forecasttime=0 (analysis values)
    if 'forecasttime' in df_final.columns:
        logging.debug(f"Filtering forecasttime=0: Before: {len(df_final)} rows")
        df_final = df_final[df_final['forecasttime'] != 0].copy()
        logging.debug(f"After filtering forecasttime=0: {len(df_final)} rows")

    # 2. Filter incomplete forecasts (must have exactly 48 steps)
    if 'starttime' in df_final.columns:
        logging.debug("Filtering incomplete forecasts (requiring 48 steps per starttime)...")
        # Count rows per starttime
        counts = df_final.groupby('starttime').size()
        valid_starttimes = counts[counts == 48].index

        # Filter
        df_final = df_final[df_final['starttime'].isin(valid_starttimes)].copy()
        logging.debug(f"After filtering incomplete forecasts: {len(df_final)} rows ({len(valid_starttimes)} unique starttimes)")

    # 3. Merge with synthetic wind power data
    # We use reset_index() before merging to ensure clean merge
    if isinstance(df_final.index, pd.MultiIndex):
        df_final.reset_index(inplace=True, drop=True) # drop=True because index was likely RangeIndex or garbage

    # Merge on timestamp
    # df_synth has timestamp as index
    df_merged = df_final.merge(df_synth, left_on='timestamp', right_index=True, how='left')

    # 4. Set MultiIndex strictly: ['starttime', 'forecasttime', 'timestamp']
    if 'starttime' in df_merged.columns and 'forecasttime' in df_merged.columns and 'timestamp' in df_merged.columns:
        df_merged.set_index(['starttime', 'forecasttime', 'timestamp'], inplace=True)
        # Sort index to ensure order: starttime -> forecasttime
        df_merged.sort_index(inplace=True)
        logging.debug(f"Set and sorted MultiIndex: {df_merged.index.names}")
    else:
        # Should not happen given the requirements
        logging.error("CRITICAL: Could not set MultiIndex - missing columns!")
        df_merged.set_index('timestamp', inplace=True)

    # Filter relevant features if specified
    rel_features = list(set(features['known'] + features['observed'])) if features else []

    # Handle turbine-specific features
    new_rel_features = []

    # Expand generic turbine features to specific turbine columns ONLY if in rel_features
    if 'wind_speed_t' in rel_features:
        rel_features.remove('wind_speed_t')
        for index, turbine in turbines.iterrows():
            turbine_id = turbine['turbine']
            turbine_col = f'wind_speed_{turbine_id}'
            if turbine_col in df_merged.columns:
                new_rel_features.append(turbine_col)

    if 'density_t' in rel_features:
        rel_features.remove('density_t')
        for index, turbine in turbines.iterrows():
            turbine_id = turbine['turbine']
            turbine_col = f'density_{turbine_id}'
            if turbine_col in df_merged.columns:
                new_rel_features.append(turbine_col)

    # Drop unwanted columns
    cols_to_drop = []

    # Always drop power_t* columns (we only keep the sum 'power')
    for col in df_merged.columns:
        if col.startswith('power_t'):
            cols_to_drop.append(col)

    # Drop wind_speed_t* columns if not explicitly requested in rel_features
    if 'wind_speed_t' not in features.get('known', []) and 'wind_speed_t' not in features.get('observed', []):
        for col in df_merged.columns:
            if col.startswith('wind_speed_t'):
                cols_to_drop.append(col)

    # Always drop generic wind_speed (without suffix)
    if 'wind_speed' in df_merged.columns and 'wind_speed' not in rel_features:
        cols_to_drop.append('wind_speed')

    # Drop the unwanted columns
    if cols_to_drop:
        df_merged.drop(columns=cols_to_drop, inplace=True)

    # Add Icon-D2 features
    if rel_features:
        # Create list of all possible feature combinations with suffixes
        available_cols = []

        # Create regex patterns for strict matching
        # Matches exactly the feature name, optionally followed by _<digits> (for grid points)
        patterns = [re.compile(f"^{re.escape(feat)}(_\d+)?$") for feat in rel_features]

        for col in df_merged.columns:
            if col in ['power']:  # Keep power and other important columns
                available_cols.append(col)
                continue

            # Check if column matches any of the requested features strictly
            for pattern in patterns:
                if pattern.match(col):
                    if col not in available_cols:
                        available_cols.append(col)
                    break

        # Add turbine features and target
        available_cols.extend(new_rel_features)
        if config['data']['target_col'] not in available_cols:
            available_cols.append(config['data']['target_col'])
        available_cols.extend(static_features)

        # Add static features
        for static_feature in static_features:
            if any(static_feature in key for key in static_data.keys()):
                matching_key = next(key for key in static_data.keys() if static_feature in key)
                df_merged[static_feature] = static_data[matching_key]

        # Remove duplicates and filter
        available_cols = list(dict.fromkeys(available_cols))
        available_cols = [col for col in available_cols if col in df_merged.columns]
        df_merged = df_merged[available_cols]

    # # Resample if needed
    # if freq.upper() != '1H':
    #     # Handle MultiIndex resampling
    #     if isinstance(df_merged.index, pd.MultiIndex):
    #          df_merged = df_merged.resample(freq, level='timestamp', closed='left', label='left', origin='start').mean()
    #     else:
    #          df_merged = df_merged.resample(freq, closed='left', label='left', origin='start').mean()

    # --- PCA Analysis (Test) ---
    if config['data']['use_pca']:
        # Select features for PCA: only known_features
        original_known_features = features.get('known', [])

        # Create regex patterns for known features, including grid point suffixes
        known_feature_patterns = [re.compile(f"^{re.escape(feat)}(_\d+)?$") for feat in original_known_features]

        pca_known_cols = []
        for col in df_merged.columns:
            for pattern in known_feature_patterns:
                if pattern.match(col):
                    if pd.api.types.is_numeric_dtype(df_merged[col]):
                        pca_known_cols.append(col)
                    break

        if len(pca_known_cols) > 0:
            logging.debug(f'Station {station_id}: {len(pca_known_cols)} known features for PCA')

            # Extract known features for PCA
            x_pca_known = df_merged[pca_known_cols].values

            # Standardize
            scaler_pca_known = StandardScaler()
            x_pca_known_scaled = scaler_pca_known.fit_transform(x_pca_known)

            # PCA
            n_components = min(config['data']['pca_components'], len(pca_known_cols))
            pca_known = PCA(n_components=n_components)
            principal_components_known = pca_known.fit_transform(x_pca_known_scaled)

            logging.debug(f"Known features PCA - Explained variance ratio (first {n_components} components): {pca_known.explained_variance_ratio_}")
            logging.debug(f"Known features PCA - Cumulative explained variance: {np.sum(pca_known.explained_variance_ratio_):.4f}")

            # Create new column names for principal components
            pca_component_names = [f'known_pca_{i}' for i in range(n_components)]

            # Create a DataFrame for principal components
            df_pca_known = pd.DataFrame(data=principal_components_known,
                                        columns=pca_component_names,
                                        index=df_merged.index)

            # Drop original known features from df_merged
            df_merged.drop(columns=pca_known_cols, inplace=True)

            # Add new principal components to df_merged
            df_merged = pd.concat([df_merged, df_pca_known], axis=1)
        else:
             logging.debug(f'Station {station_id}: No suitable known features for PCA found.')
    # ---------------------------

    # Final cleanup
    df_merged.dropna(inplace=True)

    # Ensure index is sorted (should already be, but safe is safe)
    df_merged.sort_index(inplace=True)

    # We removed the manual timestamp filtering because we now strictly filter for complete 48h forecasts per starttime
    # This is more robust and fits the MultiIndex structure better.

    return df_merged


def get_density_at_height(h2: float,
                          rho1: pd.Series,
                          t1: pd.Series) -> pd.Series:
    R = 8.31451
    M_air = 0.028949 # dry air
    M_h20 = 0.018015 # water
    g = 9.81
    t1 = t1 + 273.15
    h1 = 2 # because of temperature measured at 2 m
    temp_gradient = 0.00649
    M = M_air # molar mass of air (including water vapor) is less than that of dry air
    delta_h = h2 - h1
    rho2 = rho1 * ( 1 - (temp_gradient * delta_h) / t1 ) ** ( (M * g) / (temp_gradient * R) - 1)
    return rho2


def preprocess_synth_wind_openmeteo(path: str,
                                    config: dict,
                                    freq: str = '1H',
                                    target_col: str = 'power',
                                    features: dict = None) -> pd.DataFrame:
    timestamp_col = 'timestamp'
    park_id = path.split('_')[-1].split('.')[0]
    wind_parameter_path = path.replace(os.path.basename(path), 'wind_parameter.csv')
    turbine_parameter_path = path.replace(os.path.basename(path), 'turbine_parameter.csv')
    power_curves_path = os.path.join(config['data']['power_curves_path'], 'turbine_power.csv')
    nwp_base_path = config['data']['nwp_path']
    nwp_models = config['params']['openmeteo']['nwp_models']
    rel_features = list(set(features['known'] + features['observed']))
    static_features = features.get('static', [])
    static_data = {}
    wind_parameter = pd.read_csv(wind_parameter_path, sep=';', dtype={'park_id': str})
    turbine_parameter = pd.read_csv(turbine_parameter_path, sep=';', dtype={'park_id': str})
    power_curves = pd.read_csv(power_curves_path, sep=';', decimal=',')
    df = pd.read_csv(path, sep=';')
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df.set_index(timestamp_col, inplace=True)
    commissioning_date = wind_parameter.loc[wind_parameter.park_id == park_id]['commissioning_date'].values[0]
    altitude = wind_parameter.loc[wind_parameter.park_id == park_id]['altitude'].values[0]
    if 'random_seeds' not in config['params']:
        random.seed(config['params']['random_seed'])
    else:
        iterator = config['params'].get('iterator', 0)
        config['params']['iterator'] = iterator
        random_seed = config['params']['random_seeds'][iterator]
        config['params']['random_seed'] = random_seed
        random.seed(random_seed)
        config['params']['iterator'] = iterator + 1
    if commissioning_date != '-':
        park_age_years = (pd.Timestamp.now() - pd.to_datetime(commissioning_date)).days // 365
    else:
        park_age_years = 0
    heights = []
    if 'turbines' in config['params']:
        turbines_list = config['params']['turbines']
        if 'turbines_per_park' in config['params']:
            turbines_list = random.sample(turbines_list, config['params']['turbines_per_park'])
            config['params']['random_seed'] += 1
        power_curves = power_curves[turbines_list]
        installed_capacity = power_curves.sum(axis=1).max() * 1000 # in Watts
        turbines = turbine_parameter.loc[turbine_parameter.turbine_name.isin(turbines_list)]
        turbine_indices = turbines['turbine']
        heights = turbines['hub_height'].values
        power_cols = [f'power_{i}' for i in turbine_indices]
        df['power'] = df[power_cols].sum(axis=1)
    else:
        df['power'] = 0
        for col in df.columns:
            if 'power' in col: df['power'] += df[col]
        installed_capacity = df['power'].max()
        heights = turbine_parameter['hub_height'].values
        turbines = turbine_parameter
    if static_features:
        static_data['park_id'] = park_id
        static_data['park_age'] = park_age_years
        static_data['installed_capacity'] = installed_capacity
        static_data['altitude'] = altitude
        for index, turbine in turbines.iterrows():
            turbine_name = turbine['turbine_name']
            turbine_id = turbine['turbine']
            if 'wind_speed_hub' in features['observed']:
                rel_features.append(f'wind_speed_t{turbine_id}')
                # drop wind_speed_hub from features['observed']
                rel_features.remove('wind_speed_hub')
            static_data[f'hub_height'] = turbine['hub_height']
            static_data[f'rotor_diameter'] = turbine['diameter']
            static_data[f'rated_power'] = power_curves[turbine_name].max() * 1000
            static_data[f'cut_in'] = turbine['cut_in']
            static_data[f'cut_out'] = turbine['cut_out']
            static_data[f'rated_wind_speed'] = turbine['rated']
    df['power'] = df['power'] / installed_capacity # in Watts
    df = df.resample(freq, closed='left', label='left', origin='start').mean()

    # Expand generic turbine features to specific turbine columns
    if 'wind_speed_t' in rel_features:
        rel_features.remove('wind_speed_t')
        for index, turbine in turbines.iterrows():
            turbine_id = turbine['turbine']
            turbine_col = f'wind_speed_{turbine_id}'  # turbine_id is already 't1', 't2', etc.
            #logging.debug(f"Looking for turbine column: {turbine_col}")
            if turbine_col in df.columns:
                rel_features.append(turbine_col)
            else:
                logging.warning(f"Turbine column {turbine_col} not found in dataframe")

    if 'density_t' in rel_features:
        rel_features.remove('density_t')
        for index, turbine in turbines.iterrows():
            turbine_id = turbine['turbine']
            turbine_col = f'density_{turbine_id}'  # turbine_id is already 't1', 't2', etc.
            if turbine_col in df.columns:
                rel_features.append(turbine_col)
            else:
                logging.warning(f"Turbine column {turbine_col} not found in dataframe")

    # get nwp data if nwp is mentioned in the known features
    df_nwp = pd.DataFrame()
    new_rel_features = []
    exclude_features = ['power', 'wind_speed']  # Only exclude generic features that shouldn't be model-specific
    basis_features = rel_features.copy()  # Use updated rel_features after turbine expansion
    height_levels = [10, 80, 120, 180]
    nwp_heights = [10, 77.745, 126.858, 183.592]
    for model in nwp_models:
        nwp_path = os.path.join(nwp_base_path, model, f'{model}_{park_id}.csv')
        nwp = pd.read_csv(nwp_path, sep=',')
        nwp.rename(columns={'date': 'timestamp'}, inplace=True)
        nwp['timestamp'] = pd.to_datetime(nwp['timestamp'], utc=True)
        # rename nwp data
        for col in nwp.columns:
            if col not in ['timestamp']:
                nwp.rename(columns={col: f'{col}_{model}'}, inplace=True)
        nwp.set_index('timestamp', inplace=True)
        # get density if required
        if config['params']['get_density']:
            nwp[f'sat_vap_ps_{model}'] = get_saturated_vapor_pressure(temperature=nwp[f'temperature_2m_{model}'],
                                                                      model='huang')
            nwp[f'density_2m_{model}'] = get_density(temperature=nwp[f'temperature_2m_{model}'],
                                            pressure=nwp[f'surface_pressure_{model}'],
                                            relhum=nwp[f'relative_humidity_2m_{model}'],
                                            sat_vap_ps=nwp[f'sat_vap_ps_{model}'])
            for h in height_levels:
                nwp[f'density_{h}m_{model}'] = get_density_at_height(rho1=nwp[f'density_2m_{model}'],
                                                                    t1=nwp[f'temperature_2m_{model}'],
                                                                    h2=int(h))
        nwp_features = []
        for col in basis_features:
            if col not in exclude_features:
                # Check if this is a turbine-specific measurement (like wind_speed_t1, wind_speed_t2, etc.)
                if col.startswith('wind_speed_t') and col[len('wind_speed_t'):].isdigit():
                    # This is a turbine measurement, don't add model suffix
                    if col in df.columns:  # Check if it exists in the main dataframe
                        new_rel_features.append(col)
                    continue
                elif col.startswith('density_t') and col[len('density_t'):].isdigit():
                    # This is a turbine measurement, don't add model suffix
                    if col in df.columns:  # Check if it exists in the main dataframe
                        new_rel_features.append(col)
                    continue
                else:
                    # This is an NWP feature, add model suffix
                    model_specific_col = f'{col}_{model}'
                    if model_specific_col in nwp.columns:
                        nwp_features.append(model_specific_col)
                        new_rel_features.append(model_specific_col)
                    else:
                        logging.warning(f"Column {model_specific_col} not found in NWP data for model {model}")
            else:
                if col not in new_rel_features:  # Avoid duplicates
                    new_rel_features.append(col)
        wind_speed_cols = [f'wind_speed_{h}m_{model}' for h in height_levels]
        if config['params']['aggregate_nwp_layers'] == 'weighted_mean' or \
            config['params']['aggregate_nwp_layers'] == 'mean':
            nwp_features.extend(wind_speed_cols)
        nwp_features_unique = list(dict.fromkeys(nwp_features))
        nwp_features_final = [f for f in nwp_features_unique if f in nwp.columns]
        nwp = nwp[nwp_features_final].copy()
        rel_features.extend(nwp_features)
        # aggregate multilevels
        if config['params']['aggregate_nwp_layers'] == 'weighted_mean':
            midpoints = [(nwp_heights[i] + nwp_heights[i + 1]) / 2 for i in range(len(nwp_heights) - 1)]
            layers = []
            layers.append((0.0, midpoints[0], height_levels[0]))
            for i in range(len(midpoints) - 1):
                layers.append((midpoints[i], midpoints[i+1], height_levels[i+1]))
            layers.append((midpoints[-1], np.inf, height_levels[-1]))

            area_weights = get_A_weights(turbines, layers)
            weighted_speed_cubed_series = pd.Series(0.0, index=nwp.index, dtype=float)

            for i, (lower_bound, upper_bound, level_name) in enumerate(layers):
                wind_col = f'wind_speed_{level_name}m_{model}'
                if wind_col in nwp.columns and area_weights[i] > 0:
                    weighted_speed_cubed_series += area_weights[i] * (nwp[wind_col] ** 3)

            nwp[f'wind_speed_rotor_eq_{model}'] = (weighted_speed_cubed_series) ** (1/3)
            new_rel_features.append(f'wind_speed_rotor_eq_{model}')
            wind_speed_cols_to_remove = [col for col in nwp.columns if col.startswith('wind_speed_') and col.endswith(f'_{model}') and 'rotor_eq' not in col and col not in rel_features]
            nwp.drop(columns=wind_speed_cols_to_remove, inplace=True)
        elif config['params']['aggregate_nwp_layers'] == 'mean':
            mean_wind_speed = nwp[wind_speed_cols[1:]].mean(axis=1)
            mean_col_name = f'wind_speed_mean_height_{model}'
            nwp[mean_col_name] = mean_wind_speed
            new_rel_features.append(mean_col_name)
            wind_speed_cols_to_remove = [col for col in nwp.columns if col.startswith('wind_speed_') and col.endswith(f'_{model}') and 'mean_height' not in col and col not in rel_features]
            nwp.drop(columns=wind_speed_cols_to_remove, inplace=True)
        elif config['params']['aggregate_nwp_layers'] == 'not':
            pass
        else:
            raise ValueError(f"Unknown option for 'aggregate_nwp_layers': {config['params']['aggregate_nwp_layers']}. Please choose from 'weighted_mean', 'mean', or 'not'.")
        if df_nwp.empty:
            df_nwp = nwp.copy()
        else:
            df_nwp = pd.merge(df_nwp, nwp, left_index=True, right_index=True)
    # merge nwp data and wind park data
    df = df_nwp.merge(df, left_index=True, right_index=True, how='left')
    new_rel_features.append(target_col)
    new_rel_features.extend(static_features)
    for static_feature in static_features:
        if any(static_feature in key for key in static_data.keys()):
            matching_key = next(key for key in static_data.keys() if static_feature in key)
            df[static_feature] = static_data[matching_key]
        else:
            logging.warning(f"Static feature '{static_feature}' not found in static data keys.")
    new_rel_features = list(set(new_rel_features))
    df = df[new_rel_features]
    df.reset_index(inplace=True)
    df.sort_values(['timestamp'], ascending=True, inplace=True)
    df.set_index(['timestamp'], inplace=True)
    df.dropna(inplace=True)
    first_timestamp = df.index[0]
    # time series start at hour divisible by 3
    first_hour = first_timestamp.hour
    if first_hour % 3 != 0:
        next_valid_hour = ((first_hour // 3) + 1) * 3
        # Handle hour overflow (e.g., if next_valid_hour = 24, it becomes 0 of next day)
        if next_valid_hour >= 24:
            # Move to next day at hour 0
            start_timestamp = first_timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(days=1)
        else:
            # Same day, but at the next valid hour
            start_timestamp = first_timestamp.replace(hour=next_valid_hour, minute=0, second=0, microsecond=0)
        logging.debug(f"Adjusting start time from {first_timestamp} (hour {first_hour}) to {start_timestamp} (hour {start_timestamp.hour}) - hour divisible by 3")
        df = df[df.index >= start_timestamp]
    else:
        logging.debug(f"Time series already starts at valid hour {first_hour} (divisible by 3)")
    return df


def get_features(config: dict = None) -> Dict[str, list]:
    if 'known_features' in config['params'] and 'observed_features' in config['params']:
        known = config['params']['known_features']
        observed = config['params']['observed_features']
        static = config['params'].get('static_features', [])
        features_dict = {
            "known": known,
            "observed": observed,
            "static": static
        }
    else:
        features_dict = {
            "known": None,
            "observed": None,
            "static": None
        }
    return features_dict

def prepare_data_for_tft(data: pd.DataFrame,
                         history_length: int, # e.g., 72 (for 3 days history)
                         future_horizon: int, # e.g., 24 (for 24 hours forecast)
                         known_future_cols: list,
                         observed_past_cols: list,
                         static_cols: list,
                         step_size: int = 1, # Step size for windowing (e.g., 3 to take every 3rd window)
                         train_frac: float = 0.75,
                         target_col: str = 'power',
                         train_start: pd.Timestamp = None,
                         test_start: pd.Timestamp = None,
                         test_end: pd.Timestamp = None,
                         t_0: int = 0,
                         scale_target: bool = False,
                         scaler_x: StandardScaler = None): # New flag to control lag feature
    """
    Prepares data for a Temporal Fusion Transformer, creating a lagged target input.
    Args:
        data (pd.DataFrame): DataFrame with a time index and all features.
        target_col (str): Name of the target column.
        history_length (int): Number of past time steps for input history.
        future_horizon (int): Number of future time steps for prediction horizon.
        static_cols (list): List of static column names.
        known_future_cols (list): List of known future input column names.
        observed_past_cols (list): List of observed past input column names.
        train_end_idx: Index or date where training data ends.
        test_start_idx: Index or date where test data begins.
        scale_target (bool): Whether to scale the actual target variable (y).
        return_index (bool): Whether to return timestamps corresponding to the forecast start.
    Returns:
        tuple: Contains the prepared data arrays for training and testing,
               optionally scalers and indices.
               Format: (X_static_train, X_known_train, X_observed_train, y_train,
                        X_static_test, X_known_test, X_observed_test, y_test,
                        ...) plus optionally scaler_dict, train_indices, test_indices
    """
    # Prepare feature columns for TFT
    known_future_cols, observed_past_cols = prepare_features_for_tft(
        cols=data.columns.tolist(),
        known_future_cols=known_future_cols,
        observed_past_cols=observed_past_cols
    )
    #logging.debug(f"Known future columns for TFT: {known_future_cols}")
    #logging.debug(f"Observed past columns for TFT: {observed_past_cols}")
    # add target to observed_past_cols if not already included
    #if target_col not in observed_past_cols:
    #    observed_past_cols.append(target_col)
    df = data.copy()

    # --- Data Splitting ---
    # For NWP data with forecast runs, we need to extend test_start backwards
    # to include historical forecasts required for creating the first test windows
    is_nwp_data = isinstance(df.index, pd.MultiIndex) and 'starttime' in df.index.names

    if is_nwp_data and test_start is not None:
        # Extend test_start by step_size to include required historical forecasts
        # Example: If test_start='2025-08-01' and step_size=48h,
        # we need forecasts from '2025-07-30' to build windows for predictions starting '2025-08-01'
        original_test_start = test_start
        test_start_adjusted = pd.Timestamp(test_start) - pd.Timedelta(hours=step_size)
        logging.debug(f"NWP data detected: Extending test data start from {original_test_start} "
                    f"to {test_start_adjusted} (step_size={step_size}h) to include required historical forecasts")
        test_start = test_start_adjusted

    # Use helper function for splitting which handles MultiIndex correctly
    train_df, test_df = split_data(df, train_frac, train_start, test_start, test_end, t_0)

    #logging.info(f"Training data range: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
    #logging.info(f"Test data range:     {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")
    scalers = {}
    # Static
    X_static_train = get_static_features(data=train_df,
                                         static_cols=static_cols)
    X_static_test = get_static_features(data=test_df,
                                        static_cols=static_cols)

    # Scale Static Features if scaler_x is provided and static features exist
    # CRITICAL: Static features must be scaled with the same scaler as dynamic features!
    if scaler_x and X_static_train.size > 0 and static_cols:
        try:
            # Check if scaler knows about static features
            if hasattr(scaler_x, 'feature_names_in_'):
                scaler_features = list(scaler_x.feature_names_in_)
                # Find which static features are in the scaler
                static_features_in_scaler = [col for col in static_cols if col in scaler_features]

                if static_features_in_scaler:
                    # Create a dummy dataframe with just the static features for transformation
                    # We need to create a row with all features the scaler expects
                    dummy_row_train = pd.DataFrame(0.0, index=[0], columns=scaler_features)
                    dummy_row_test = pd.DataFrame(0.0, index=[0], columns=scaler_features)

                    # Fill in the static feature values
                    for i, col in enumerate(static_cols):
                        if col in static_features_in_scaler:
                            dummy_row_train[col] = X_static_train[i]
                            dummy_row_test[col] = X_static_test[i]

                    # Transform
                    scaled_train = scaler_x.transform(dummy_row_train)
                    scaled_test = scaler_x.transform(dummy_row_test)

                    # Extract only the static feature columns
                    static_indices = [scaler_features.index(col) for col in static_features_in_scaler]
                    X_static_train = scaled_train[0, static_indices]
                    X_static_test = scaled_test[0, static_indices]

                    logging.debug(f"Static features scaled using global scaler_x: {static_features_in_scaler}")
                    logging.debug(f"Scaled static train: shape={X_static_train.shape}, values={X_static_train}")
                    logging.debug(f"Scaled static test: shape={X_static_test.shape}, values={X_static_test}")

                    # Ensure correct shape: should be (n_static_features,)
                    assert X_static_train.shape == (len(static_features_in_scaler),), \
                        f"X_static_train has wrong shape: {X_static_train.shape}, expected ({len(static_features_in_scaler)},)"
                    assert X_static_test.shape == (len(static_features_in_scaler),), \
                        f"X_static_test has wrong shape: {X_static_test.shape}, expected ({len(static_features_in_scaler)},)"

                else:
                    logging.warning(f"Static features {static_cols} not found in scaler_x feature names. "
                                  f"Static features will NOT be scaled! This may cause poor model performance.")
            else:
                logging.warning("Scaler has no feature_names_in_. Cannot verify if static features are included. "
                              "Static features will NOT be scaled! This may cause poor model performance.")
        except Exception as e:
            logging.error(f"Error scaling static features: {e}. "
                        f"Static features will NOT be scaled! This may cause poor model performance.")

    # Scale Known Future Features
    known_train_data, known_test_data = None, None

    # Scale Observed Past Features (now includes lagged target if enabled)
    observed_train_data, observed_test_data = None, None

    if scaler_x:
        # GLOBAL SCALING STRATEGY
        # We must scale the *entire* feature set because scaler_x was fitted on all features.
        # We cannot scale subsets (known/observed) individually with the global scaler.

        # Identify all feature columns (everything except target)
        # Note: train_df/test_df might contain target_col
        feature_cols = [c for c in train_df.columns if c != target_col]

        # Create copies to avoid side effects
        train_df_scaled = train_df.copy()
        test_df_scaled = test_df.copy()

        # Transform all features
        # We use the DataFrame directly so sklearn can match feature names
        train_df_scaled[feature_cols] = scaler_x.transform(train_df[feature_cols])
        test_df_scaled[feature_cols] = scaler_x.transform(test_df[feature_cols])

        # Now extract the specific columns from the scaled dataframes
        if known_future_cols:
            known_train_data = train_df_scaled[known_future_cols].values
            known_test_data = test_df_scaled[known_future_cols].values
            scalers['x_known'] = scaler_x

        if observed_past_cols:
            observed_train_data = train_df_scaled[observed_past_cols].values
            observed_test_data = test_df_scaled[observed_past_cols].values
            scalers['x_observed'] = scaler_x

    else:
        # LOCAL SCALING STRATEGY (Per Group)
        if known_future_cols:
            known_train_data, known_test_data, scalers['x_known'] = apply_scaling(train_df[known_future_cols].values,
                                                                                test_df[known_future_cols].values,
                                                                                StandardScaler)
        if observed_past_cols:
            observed_train_data, observed_test_data, scalers['x_observed'] = apply_scaling(train_df[observed_past_cols].values,
                                                                                        test_df[observed_past_cols].values,
                                                                                        StandardScaler)
    # Scale Target Variable (y) Separately
    target_train_raw = train_df[[target_col]].values
    target_test_raw = test_df[[target_col]].values
    if scale_target:
        target_train_scaled, target_test_scaled, target_scaler = apply_scaling(target_train_raw,
                                                                               target_test_raw,
                                                                               StandardScaler)
        scalers['y'] = target_scaler
        logging.info(f"Target variable '{target_col}' scaled separately.")
    else:
        target_train_scaled = target_train_raw
        target_test_scaled = target_test_raw
        scalers['y'] = None

    # Apply windowing to training and test sets using the prepared (scaled) data
    X_known_train, X_observed_train, y_train, train_indices = create_tft_sequences(
        known_train_data,
        observed_train_data,
        target_train_scaled, # Pass the separately handled target data
        train_df.index,
        history_length,
        future_horizon,
        step_size
    )
    X_known_test, X_observed_test, y_test, test_indices = create_tft_sequences(
        known_test_data,
        observed_test_data,
        target_test_scaled, # Pass the separately handled target data
        test_df.index,
        history_length,
        future_horizon,
        step_size
    )

    num_train_samples = X_observed_train.shape[0] # z.B. 24918
    num_test_samples = X_observed_test.shape[0]

    if X_static_train.size > 0 and num_train_samples > 0:
        X_static_train = np.tile(X_static_train, (num_train_samples, 1))

    if X_static_test.size > 0 and num_test_samples > 0:
        X_static_test = np.tile(X_static_test, (num_test_samples, 1))

    #logging.debug("Shapes of generated arrays (Train):")
    # Expected shapes:
    # X_static: (num_samples, n_static_features)
    # X_known:  (num_samples, history_len + future_horizon, n_known_features)
    # X_observed:(num_samples, history_len, n_observed_input_features)
    # y:        (num_samples, future_horizon)
    #logging.debug(f"X_static_train:   {X_static_train.shape}")
    #logging.debug(f"X_known_train:    {X_known_train.shape}")
    #logging.debug(f"X_observed_train: {X_observed_train.shape}")
    #logging.debug(f"y_train:          {y_train.shape}")

    # --- Return Results ---
    results = {}
    results['y_train'], results['y_test'] = y_train, y_test

    results['scalers'] = scalers
    results['index_train'], results['index_test'] = train_indices, test_indices

    X_train: Dict[str, np.ndarray] = {
            'observed': X_observed_train,
            'known': X_known_train,
            'static': X_static_train, # Hinzugefügt (kann None/leer sein)
    }
    X_test: Dict[str, np.ndarray] = {
        'observed': X_observed_test,
        'known': X_known_test,
        'static': X_static_test # Static Testdaten hinzufügen
    }
    results['X_train'], results['X_test'] = X_train, X_test
    static_is_zero = X_train['static'].shape[-1] == 0
    if static_is_zero:
        del X_train['static']
        del X_test['static']
    return results

def prepare_features_for_tft(cols: list,
                             known_future_cols: list,
                             observed_past_cols: list):
    new_known_cols = []
    new_observed_cols = []

    # Check for PCA columns (if PCA was applied in preprocessing)
    pca_cols = [c for c in cols if 'known_pca_' in c]
    if pca_cols:
        new_known_cols.extend(pca_cols)

    # Handle specific feature mappings for Icon-D2 data
    for known_feat in known_future_cols:
        if known_feat == 'wind_speed_nwp':
            # Map wind_speed_nwp to all wind_speed_h* columns
            matching_cols = [col for col in cols if 'wind_speed_h' in col]
            new_known_cols.extend(matching_cols)
        else:
            # Standard substring matching
            matching_cols = [col for col in cols if known_feat in col]
            new_known_cols.extend(matching_cols)

    for observed_feat in observed_past_cols:
        matching_cols = [col for col in cols if observed_feat in col]
        new_observed_cols.extend(matching_cols)

    # Add special NWP aggregated features if present
    for col in cols:
        if 'wind_speed_rotor_eq' in col and col not in new_known_cols:
            new_known_cols.append(col)
        if 'wind_speed_mean_height' in col and col not in new_known_cols:
            new_known_cols.append(col)

    # Remove duplicates while preserving order
    new_known_cols = list(dict.fromkeys(new_known_cols))
    new_observed_cols = list(dict.fromkeys(new_observed_cols))

    return new_known_cols, new_observed_cols

def get_static_features(data: pd.DataFrame,
                        static_cols: list):
    if not static_cols:
        logging.debug('No static data provided.')
        return np.array([])
    static_features = []
    for col in static_cols:
        static_features.append(data[col].values[0])
    return np.array(static_features)

def create_tft_sequences(known_data: np.ndarray,
                         observed_data: np.ndarray,
                         target_data: np.ndarray,
                         indices: np.ndarray,
                         history_len: int,
                         future_len: int,
                         step_size: int = 1):
        """
        Creates sequences for TFT inputs and outputs.

        For NWP data with MultiIndex ['starttime', 'forecasttime', 'timestamp']:
        - Each unique starttime represents one forecast run with future_len timesteps
        - For each current forecast, we combine:
          * Past data from a forecast that occurred step_size hours earlier
          * Future data from the current forecast

        For regular time series data:
        - Uses standard linear windowing
        """
        X_known_list, X_observed_list, y_list, index_positions = [], [], [], []

        # Detect if this is NWP data with forecast runs
        is_nwp_data = isinstance(indices, pd.MultiIndex) and 'starttime' in indices.names

        if is_nwp_data:
            # NWP-aware windowing: one window per forecast run
            logging.debug("Using NWP-aware windowing for MultiIndex data with starttime")

            # PRE-EXTRACT index levels once (expensive operation)
            starttimes_all = indices.get_level_values('starttime')
            timestamps_all = indices.get_level_values('timestamp')

            # Get unique starttimes and sort
            unique_starttimes = starttimes_all.unique().sort_values()
            n_forecasts = len(unique_starttimes)
            logging.debug(f"Found {n_forecasts} unique forecast runs (starttimes)")

            # Pre-compute starttime to indices mapping for O(1) lookup
            starttime_to_indices = {}
            for st in unique_starttimes:
                starttime_to_indices[st] = np.where(starttimes_all == st)[0]

            # Pre-compute timestamp to indices mapping for observed features
            if observed_data is not None:
                timestamp_to_first_idx = {}
                for ts in timestamps_all.unique():
                    # Store first occurrence of each timestamp
                    timestamp_to_first_idx[ts] = np.where(timestamps_all == ts)[0][0]

            # Process each forecast run
            for current_start in unique_starttimes:
                # Calculate the starttime we need for past data
                previous_start = current_start - pd.Timedelta(hours=step_size)

                # Quick check if previous forecast exists
                if previous_start not in starttime_to_indices:
                    continue

                # Get indices using pre-computed mapping (O(1) instead of O(n))
                current_indices = starttime_to_indices[current_start]
                previous_indices = starttime_to_indices[previous_start]

                # Validate data availability
                if len(current_indices) == 0 or len(previous_indices) == 0:
                    continue

                # Known features: combine past (from previous run) + future (from current run)
                if known_data is not None:
                    # Verify we have the expected number of timesteps
                    if len(previous_indices) != future_len or len(current_indices) != future_len:
                        logging.warning(f"Unexpected known data length at {current_start}: "
                                      f"past={len(previous_indices)}, future={len(current_indices)}, expected={future_len}")
                        continue

                    # Direct indexing without intermediate copies
                    known_window = np.vstack([known_data[previous_indices], known_data[current_indices]])
                    X_known_list.append(known_window)

                # Observed features: select by timestamp (power is measured, not forecasted)
                if observed_data is not None:
                    # Get the timestamp range we need
                    forecast_start_time = timestamps_all[current_indices[0]]  # First timestamp of current forecast
                    observed_end_time = forecast_start_time
                    observed_start_time = observed_end_time - pd.Timedelta(hours=history_len)

                    # Find timestamps in range using pre-computed mapping
                    # Generate hourly timestamps for the range
                    expected_timestamps = pd.date_range(
                        start=observed_start_time,
                        end=observed_end_time,
                        freq='1H',
                        inclusive='left'  # Exclude end
                    )

                    if len(expected_timestamps) != history_len:
                        logging.warning(f"Unexpected timestamp range at {current_start}: "
                                      f"got {len(expected_timestamps)}, expected {history_len}")
                        if known_data is not None and len(X_known_list) > 0:
                            X_known_list.pop()
                        continue

                    # Collect observed data for these timestamps
                    obs_data_list = []
                    missing_timestamps = []
                    for ts in expected_timestamps:
                        if ts in timestamp_to_first_idx:
                            idx = timestamp_to_first_idx[ts]
                            obs_data_list.append(observed_data[idx])
                        else:
                            missing_timestamps.append(ts)

                    if len(obs_data_list) == history_len:
                        observed_window = np.array(obs_data_list)
                        X_observed_list.append(observed_window)
                    else:
                        logging.warning(f"Missing {len(missing_timestamps)} timestamps at {current_start}: "
                                      f"found {len(obs_data_list)}, expected {history_len}")
                        # Remove the known window we just added
                        if known_data is not None and len(X_known_list) > 0:
                            X_known_list.pop()
                        continue

                # Target: from current forecast
                if len(current_indices) != future_len:
                    logging.warning(f"Unexpected target length at {current_start}: {len(current_indices)}, expected {future_len}")
                    # Remove windows we just added
                    if known_data is not None and len(X_known_list) > 0:
                        X_known_list.pop()
                    if observed_data is not None and len(X_observed_list) > 0:
                        X_observed_list.pop()
                    continue

                target_window = target_data[current_indices].flatten()
                y_list.append(target_window)

                # Use current starttime as the index
                index_positions.append(current_start)

            # Convert index_positions to Index type
            if len(index_positions) > 0:
                index_list = pd.Index(index_positions)
            else:
                index_list = pd.Index([])

            logging.debug(f"Created {len(y_list)} windows from NWP data")

        else:
            # Original linear windowing for non-NWP data
            logging.debug("Using standard linear windowing for non-MultiIndex data")

            total_len = history_len + future_len
            if observed_data is not None:
                len_data = len(observed_data)
            if known_data is not None:
                len_data = len(known_data)

            for i in range(0, len_data - total_len + 1, step_size):
                if observed_data is not None:
                    observed_past_window = observed_data[i : i + history_len]
                    X_observed_list.append(observed_past_window)
                if known_data is not None:
                    known_future_window = known_data[i : i + total_len]
                    X_known_list.append(known_future_window)
                target_window = target_data[i + history_len : i + total_len].flatten()
                y_list.append(target_window)
                index_positions.append(i + history_len)

            # Preserve DatetimeIndex/MultiIndex type
            index_list = indices[index_positions] if len(index_positions) > 0 else indices[[]]

        return (np.array(X_known_list),
                np.array(X_observed_list),
                np.array(y_list),
                index_list)


def prepare_data_for_stemgnn(data: pd.DataFrame,
                             n_nodes: int,
                             history_length: int,
                             future_horizon: int,
                             step_size: int = 1,
                             train_frac: float = 0.75,
                             target_col: str = 'power',
                             test_start: pd.Timestamp = None,
                             test_end: pd.Timestamp = None,
                             t_0: int = 0,
                             scale_target: bool = False,
                             scaler_x: StandardScaler = None):
    """
    Prepares data for StemGNN.
    Ensures columns are ordered by Node: [Node1_Feat1, Node1_Feat2, ..., Node2_Feat1, ...]
    """
    df = data.copy()

    # Identify Node Columns
    # Assuming columns have suffix _1, _2, ..., _n_nodes
    # Or they are just features if n_nodes=1?
    # In preprocess_synth_wind_icond2, we have _1, _2...

    # We need to find all features that have node suffixes
    # and group them.

    # 1. Identify base features
    # We look for columns ending with _1, _2, etc.
    # But wait, some columns might not have suffixes (e.g. power, timestamp).
    # Power is usually the target and might be single (for the park) or per turbine?
    # Usually 'power' is the target for the whole park.
    # StemGNN expects (Batch, Time, Nodes, Features).
    # If we are forecasting 'power' for the park, is 'power' one of the nodes?
    # Or is 'power' a separate target?
    # The user's StemGNN implementation (and my port) outputs (Batch, Output_Dim).
    # If Output_Dim = Horizon, it forecasts a single time series (or flattened).
    # If we want to forecast power, we usually treat it as a separate task or include it.

    # However, for the GNN input, we need the grid data.
    # Let's assume we want to use the grid points as nodes.
    # And maybe 'power' is just another feature or target?
    # If the model predicts 'power', it comes from the final Dense layer.

    # Reordering logic:
    # Find all columns matching pattern `_{i}` where i in 1..n_nodes

    node_cols = []
    for i in range(1, n_nodes + 1):
        # Find columns ending with _{i}
        # Be careful not to match _{i}0 or similar.
        # Regex: `_i$`
        suffix = f'_{i}'
        cols_for_node = [c for c in df.columns if c.endswith(suffix)]
        # Sort them to ensure consistent feature order per node
        cols_for_node.sort()
        node_cols.extend(cols_for_node)

    # Check if we found columns
    if not node_cols:
        logging.warning("No node-specific columns found (ending in _1, _2, ...). Using all columns as is.")
        ordered_cols = df.columns.tolist()
    else:
        # Add non-node columns (like power?)
        other_cols = [c for c in df.columns if c not in node_cols]
        # If we want to include them, where do they go?
        # StemGNN expects homogeneous nodes.
        # If 'power' is the target, maybe it's not part of the GNN input grid?
        # Or maybe we treat 'power' as a separate input?
        # For now, let's assume the GNN input is ONLY the grid data.
        # And 'power' is the target y.
        # But wait, if we want to use past 'power' as input?
        # We can append it. But it breaks the (Nodes, Features) structure if it's just 1 column.
        # Unless we repeat it for all nodes?

        # Let's stick to the grid points for the GNN part.
        ordered_cols = node_cols

    # Filter df to ordered columns for X
    # BUT if we have a global scaler, we must scale BEFORE reordering/filtering
    # because the scaler expects the original columns.

    scalers = {} # Initialize scalers dictionary here
    already_scaled_x = False

    if scaler_x:
        # We need to scale the features in df that match scaler's features.
        # scaler_x was fitted on df_train (from train_cl.py), which had target dropped.
        # df here has target.

        # Identify feature columns (all except target)
        feature_cols = [c for c in df.columns if c != target_col]

        # Check if scaler feature names match
        if hasattr(scaler_x, 'feature_names_in_'):
            # Sklearn 1.0+ stores feature names
            scaler_features = scaler_x.feature_names_in_
            # Ensure we have these columns
            missing = [c for c in scaler_features if c not in df.columns]
            if missing:
                logging.warning(f"Scaler expects features {missing} which are missing in data.")

            # Select features in correct order for scaler
            X_to_scale = df[scaler_features]

            # Transform
            X_scaled_values = scaler_x.transform(X_to_scale)

            # Create DataFrame with scaled values and original feature names
            X_scaled_df = pd.DataFrame(X_scaled_values, columns=scaler_features, index=df.index)

            # Now we can select the ordered node columns from this scaled dataframe
            # ordered_cols must be a subset of scaler_features
            # (assuming node columns are features)

            # Check if ordered_cols are in X_scaled_df
            valid_ordered_cols = [c for c in ordered_cols if c in X_scaled_df.columns]
            if len(valid_ordered_cols) < len(ordered_cols):
                logging.warning("Some node columns were not found in the scaled data.")

            X_df = X_scaled_df[valid_ordered_cols]

            # We also need y_df
            y_df = df[[target_col]]

            # We have scaled X, so we don't need to scale X again later.
            # We set a flag or handle it.
            already_scaled_x = True
            scalers['x'] = scaler_x

        else:
            # Fallback if no feature names (older sklearn or not fitted with DF)
            # Assume df (minus target) matches scaler input
            logging.warning("Scaler has no feature_names_in_. Assuming all non-target columns match scaler input.")
            X_to_scale = df.drop(columns=[target_col], errors='ignore')
            X_scaled_values = scaler_x.transform(X_to_scale)
            X_scaled_df = pd.DataFrame(X_scaled_values, columns=X_to_scale.columns, index=df.index)
            X_df = X_scaled_df[ordered_cols]
            y_df = df[[target_col]]
            already_scaled_x = True
            scalers['x'] = scaler_x

    else:
        # Standard path: Filter first, then scale later
        X_df = df[ordered_cols]
        y_df = df[[target_col]]
        already_scaled_x = False

    # Split
    if test_start:
        train_end = test_start - pd.Timedelta(hours=0.25)
    else:
        train_periods = int(len(df) * train_frac)
        train_end = df.index[train_periods].normalize() + pd.Timedelta(hours=(t_0-0.25))
        test_start = train_end + pd.Timedelta(hours=0.25)

    train_end = pd.Timestamp(train_end, tzinfo=data.index.tz)
    test_start = pd.Timestamp(test_start, tzinfo=data.index.tz)
    test_end = pd.Timestamp(test_end, tzinfo=data.index.tz)

    X_train_df = X_df[:train_end]
    X_test_df = X_df[test_start:test_end]
    y_train_df = y_df[:train_end]
    y_test_df = y_df[test_start:test_end]
    # Scaling (if not already done)
    if not already_scaled_x:
        scaler_x = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X_train_df)
        X_test_scaled = scaler_x.transform(X_test_df)
        scalers['x'] = scaler_x
    else:
        # Already scaled
        X_train_scaled = X_train_df.values
        X_test_scaled = X_test_df.values

    scaler_y = StandardScaler()
    if scale_target:
        y_train_scaled = scaler_y.fit_transform(y_train_df)
        y_test_scaled = scaler_y.transform(y_test_df)
    else:
        y_train_scaled = y_train_df.values
        y_test_scaled = y_test_df.values

    # Windowing
    # We use make_windows (efficient)
    # X: (Batch, Time, Flat)
    # y: (Batch, Horizon)

    # We need to align X and y.
    # make_windows takes data and returns windows.
    # We need X windows and y windows (targets).

    def create_windows(X, y, indices, seq_len, horizon, step=1):
        X_wins = []
        y_wins = []
        idx_wins = []
        # Total length needed: seq_len + horizon
        # We take X[i : i+seq_len] and y[i+seq_len : i+seq_len+horizon]
        # But y is usually a single step or sequence.
        # If horizon > 1, y is sequence.

        start_idx = 0
        max_idx = len(X) - seq_len - horizon + 1

        for i in range(start_idx, max_idx, step):
            X_wins.append(X[i : i+seq_len])
            y_wins.append(y[i+seq_len : i+seq_len+horizon].flatten())
            idx_wins.append(indices[i+seq_len]) # Timestamp at forecast start (t+1)

        return np.array(X_wins), np.array(y_wins), np.array(idx_wins)

    X_train, y_train, index_train = create_windows(X_train_scaled, y_train_scaled, X_train_df.index, history_length, future_horizon, step_size)
    X_test, y_test, index_test = create_windows(X_test_scaled, y_test_scaled, X_test_df.index, history_length, future_horizon, step_size)

    results = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'index_train': index_train,
        'index_test': index_test,
        'scalers': {'x': scaler_x, 'y': scaler_y if scale_target else None}
    }

    return results