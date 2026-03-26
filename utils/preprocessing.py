
import os
import random
import warnings
import re
import copy
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
from tqdm import tqdm
from sklearn.decomposition import PCA

from . import meteo


def _get_data_from_config_files(config: dict,
                                freq: str,
                                features: dict = None,
                                target_col: str = 'power',
                                files_key: str = 'files') -> Dict[str, pd.DataFrame]:
    """
    Load data based on parks specified in config.

    Args:
        config: Configuration dictionary containing 'data' section with 'path' and 'parks'
        freq: Frequency for resampling
        rel_features: Relevant features to include
        target_col: Target column name
        files_key: Key in config['data'] to read the file list from.
                   Use 'files' (default) for training data or 'val_files' for separate
                   validation/test data sources.

    Returns:
        Dictionary mapping client names to DataFrames
    """
    data_config = config.get('data', {})
    base_path = data_config.get('path', '')
    files = data_config.get(files_key, [])

    if not files:
        raise ValueError(f"Files list is empty in config (key: '{files_key}')")

    if not base_path:
        raise ValueError("Path is not specified in config")

    dfs = {}

    # Determine the data type based on config structure or path patterns
    # Check if this is wind data (based on config or path)
    is_wind_data = ('wind' in base_path.lower() or
                   'turbines' in config.get('params', {}))

    # Load station-turbine assignments CSV for deterministic turbine mapping
    assignments_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'data', 'station_turbine_assignments_160.csv')
    station_turbine_map = {}
    if os.path.exists(assignments_path):
        try:
            assignments_df = pd.read_csv(assignments_path, dtype={'station_id': str})
            station_turbine_map = dict(zip(assignments_df['station_id'], assignments_df['turbine_type']))
            #logging.info(f"Loaded {len(station_turbine_map)} station-turbine assignments from {assignments_path}")
        except Exception as e:
            logging.warning(f"Could not load station_turbine_assignments.csv: {e}. Falling back to random seed.")
    else:
        logging.warning(f"station_turbine_assignments.csv not found at {assignments_path}. Falling back to random seed.")

    # Store original seed (used as fallback if CSV lookup fails)
    original_seed = config['params'].get('random_seed', 42)

    # Process each file
    for file_idx, file in tqdm(enumerate(files), total=len(files), desc=f"Loading stations [{files_key}]", unit="station", disable=len(files) <= 1):
        file_path = os.path.join(base_path, f"synth_{file}.csv")
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            continue

        try:
            # Create a copy of config for this specific file
            file_config = copy.deepcopy(config)

            # Look up turbine type from station_turbine_assignments.csv
            station_id = str(file).zfill(5)  # Ensure consistent zero-padded format
            if station_id in station_turbine_map:
                file_config['params']['station_turbine_type'] = station_turbine_map[station_id]
                logging.debug(f"Processing {file}: assigned turbine '{station_turbine_map[station_id]}' from CSV")
            else:
                # Fallback to seed-based random assignment
                file_config['params']['random_seed'] = original_seed + file_idx
                logging.warning(f"Station {station_id} not found in assignments CSV. "
                                f"Falling back to seed {file_config['params']['random_seed']}")

            # Determine preprocessing based on data type
            is_wind_data = 'wind' in base_path.lower()
            is_pv_data = 'pv' in base_path.lower() or 'solar' in base_path.lower()

            if is_wind_data:
                # Wind data preprocessing
                if 'open-meteo' in config['data']['nwp_path']:
                    df = preprocess_synth_wind_openmeteo(path=file_path,
                                                        config=file_config,
                                                        freq=freq,
                                                        features=features.copy() if features else None,
                                                        target_col=target_col)
                elif 'icon-d2' in config['data']['nwp_path']:
                    df = preprocess_synth_wind_icond2(path=file_path,
                                                     config=file_config,
                                                     freq=freq,
                                                     features=features.copy() if features else None)
                else:
                    raise ValueError('NWP model source is not known for wind data')

            elif is_pv_data:
                df = preprocess_synth_pv(path=file_path,
                                       freq=freq,
                                       features=features)
            else:
                # Generic CSV loading
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

    # Restore original seed in config
    config['params']['random_seed'] = original_seed

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
             target_col: str = 'power',
             files_key: str = 'files') -> Dict[str, pd.DataFrame]:
    """Load station data.

    Args:
        files_key: Which key in config['data'] holds the file list.
                   'files' (default) for training stations, 'val_files' for a
                   separate validation/test set whose scaler is fitted on 'files'.
    """
    # Check if parks are specified in config - if so, use config-based loading
    if files_key in config.get('data', {}):
        return _get_data_from_config_files(config, freq, features, target_col, files_key=files_key)

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
    with open('/tmp/ecmwf_debug.txt', 'a') as _dbg:
        _ecmwf_after = [c for c in df.columns if c.startswith('ecmwf_')]
        _dbg.write(f'after_knn: {list(df.columns)} ecmwf={_ecmwf_after}\n')
    t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']

    if config['model']['name'] in ('tft', 'tcn-tft'):
        #logging.debug(f'Features ready for prepare_data(): {df.columns.to_list()}')
        scale_target = config['data']['scale_y'] if target_col == 'power' else True
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
                                             target_col=target_col,
                                             scale_target=scale_target,
                                             scaler_x=config.get('scaler_x', None),
                                             scaler_y=config.get('scaler_y', None))
    elif config['model']['name'] == 'chronos':
        prepared_data = prepare_data_for_chronos2(
            data=df,
            history_length=config['model']['lookback'],
            future_horizon=config['model']['horizon'],
            step_size=config['model']['step_size'],
            known_future_cols=known_cols,
            observed_past_cols=observed_cols,
            static_cols=static_cols,
            train_frac=config['data']['train_frac'],
            train_start=pd.Timestamp(config['data'].get('train_start', None)),
            test_start=pd.Timestamp(config['data'].get('test_start', None)),
            test_end=pd.Timestamp(config['data'].get('test_end', None)),
            t_0=t_0,
            target_col=target_col,
            scaler_x=config.get('scaler_x', None),
        )
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
            all_known_cols = [new_col for new_col in df.columns if new_col == col or new_col.startswith(col + '_lag_')]
            all_observed_cols = [new_col for new_col in df.columns if new_col == col or new_col.startswith(col + '_lag_')]
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
        # 'power' is pre-normalized to [0,1] via installed_capacity, so scale_y can stay False.
        # All other targets (e.g. 'wind_speed') are raw physical quantities and need scaling.
        scale_y = config['data']['scale_y'] if target_col == 'power' else True
        prepared_data = prepare_data(data=df,
                                     output_dim=config['model']['output_dim'],
                                     step_size=config['model']['step_size'],
                                     train_frac=config['data']['train_frac'],
                                     scale_y=scale_y,
                                     t_0=t_0,
                                     train_start=pd.Timestamp(config['data'].get('train_start', None), tz='UTC'),
                                     test_start=pd.Timestamp(config['data'].get('test_start', None), tz='UTC'),
                                     test_end=pd.Timestamp(config['data'].get('test_end', None), tz='UTC'),
                                     target_col=target_col,
                                     scaler_x=config.get('scaler_x', None),
                                     scaler_y=config.get('scaler_y', None))

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
                 scaler_x: StandardScaler = None,
                 scaler_y: StandardScaler = None):
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
        Y_train, Y_test, scaler_y = apply_scaling(
            target_train, target_test,
            scaler_type=scaler_y if scaler_y is not None else StandardScaler,
            fit=(scaler_y is None)
        )
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
        df_grid.drop_duplicates(subset=['starttime', 'forecasttime', 'toplevel', 'bottomlevel'], keep='last', inplace=True)

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
        # u_wind / v_wind are included so callers can request raw components (e.g. u_wind_h10)
        weather_vars = ['wind_speed', 'u_wind', 'v_wind', 'temperature', 'pressure', 'qs', 'relative_humidity']
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


def _select_grid_points_relative_position(file_distances, station_lat, station_lon, next_n_grid_points):
    """
    Select NWP grid points by their consistent relative spatial position around the station.

    Unlike the geodesic_next method (which simply takes the N nearest points), this method
    identifies the enclosing quad using np.searchsorted on sorted 1D lat/lon grid arrays and
    assigns each returned point a fixed directional label (e.g. "NW", "SE"). This ensures that
    the model always sees the same feature channel for the same compass direction, regardless of
    where the station sits within the quad.

    Convention:
        lat_idx  — index of the first grid latitude >= station_lat  (first point *north* of station)
        lon_idx  — index of the first grid longitude >= station_lon (first point *east* of station)

    Inner 2×2 quad (used for both n=4 and n=12):
        NW: grid_lats[lat_idx],     grid_lons[lon_idx - 1]
        NE: grid_lats[lat_idx],     grid_lons[lon_idx]
        SW: grid_lats[lat_idx - 1], grid_lons[lon_idx - 1]
        SE: grid_lats[lat_idx - 1], grid_lons[lon_idx]

    Args:
        file_distances: list of (csv_file, distance_km, grid_lat, grid_lon) for ALL available files
        station_lat: station latitude
        station_lon: station longitude
        next_n_grid_points: must be 1, 4, or 12

    Returns:
        labeled_points: list of (csv_file, distance_km, grid_lat, grid_lon, label)
        station_rel_lat: float in [0,1] or None for n=1
        station_rel_lon: float in [0,1] or None for n=1

    Raises:
        ValueError: if next_n_grid_points not in {1, 4, 12}
    """
    if next_n_grid_points == 1:
        # Fall back to geodesic_next: single nearest point, no spatial labeling
        file_distances_sorted = sorted(file_distances, key=lambda x: x[1])
        csv_file, distance, grid_lat, grid_lon = file_distances_sorted[0]
        return [(csv_file, distance, grid_lat, grid_lon, "1")], None, None

    if next_n_grid_points not in (4, 12):
        raise ValueError(
            f"relative_position method only supports next_n_grid_points in {{1, 4, 12}}, "
            f"got {next_n_grid_points}."
        )

    # Build sorted unique 1D lat/lon arrays from all available grid points.
    # Use plain Python lists so that indexing returns Python floats (same type as in
    # coord_to_file), avoiding any numpy.float64 vs float mismatch in dict lookups.
    grid_lats = sorted(set(fd[2] for fd in file_distances))
    grid_lons = sorted(set(fd[3] for fd in file_distances))

    # Find lat_idx: first index where grid_lats[lat_idx] >= station_lat  (first point north)
    # Find lon_idx: first index where grid_lons[lon_idx] >= station_lon  (first point east)
    # np.searchsorted accepts Python lists directly.
    lat_idx = int(np.searchsorted(grid_lats, station_lat, side='left'))
    lon_idx = int(np.searchsorted(grid_lons, station_lon, side='left'))

    if next_n_grid_points == 4:
        # Need lat_idx - 1 >= 0  and  lat_idx <= len - 1
        lat_idx = int(np.clip(lat_idx, 1, len(grid_lats) - 1))
        lon_idx = int(np.clip(lon_idx, 1, len(grid_lons) - 1))

        # Inner 2×2 enclosing quad
        point_definitions = [
            ( 0, -1, "NW"),  ( 0,  0, "NE"),
            (-1, -1, "SW"),  (-1,  0, "SE"),
        ]

    else:  # next_n_grid_points == 12
        # 4×4 grid minus the 4 corners.  Offsets span [-2, +1] in lat and [-2, +1] in lon,
        # so we need lat_idx - 2 >= 0  and  lat_idx + 1 <= len - 1  →  clip to [2, len - 2]
        lat_idx = int(np.clip(lat_idx, 2, len(grid_lats) - 2))
        lon_idx = int(np.clip(lon_idx, 2, len(grid_lons) - 2))

        # Layout (station sits inside the inner NW/NE/SW/SE quad):
        #       NNW  NNE
        # WNW  NW   NE   ENE
        # WSW  SW   SE   ESE
        #       SSW  SSE
        point_definitions = [
            # Inner 4 (the enclosing quad)
            ( 0, -1, "NW"),  ( 0,  0, "NE"),
            (-1, -1, "SW"),  (-1,  0, "SE"),
            # North row
            ( 1, -1, "NNW"), ( 1,  0, "NNE"),
            # South row
            (-2, -1, "SSW"), (-2,  0, "SSE"),
            # West column
            ( 0, -2, "WNW"), (-1, -2, "WSW"),
            # East column
            ( 0,  1, "ENE"), (-1,  1, "ESE"),
        ]

    # Build a lookup dict (grid_lat, grid_lon) -> (csv_file, distance) for fast O(1) access
    coord_to_file = {
        (fd[2], fd[3]): (fd[0], fd[1]) for fd in file_distances
    }

    # Resolve each defined point to the corresponding CSV file.
    # Primary: exact coordinate lookup.
    # Fallback: nearest available file by geodesic distance (handles gaps in rotated-grid data,
    # where not every lat × lon combination from grid_lats × grid_lons has a file).
    # Deduplication: each file is assigned to at most one label.
    used_files: set = set()
    labeled_points = []
    for lat_offset, lon_offset, label in point_definitions:
        # Apply offset relative to the enclosing-quad corner (lat_idx, lon_idx)
        target_lat_i = int(np.clip(lat_idx + lat_offset, 0, len(grid_lats) - 1))
        target_lon_i = int(np.clip(lon_idx + lon_offset, 0, len(grid_lons) - 1))

        target_lat = grid_lats[target_lat_i]  # Python float — exact match with coord_to_file keys
        target_lon = grid_lons[target_lon_i]

        if (target_lat, target_lon) in coord_to_file:
            csv_file, distance = coord_to_file[(target_lat, target_lon)]
            if csv_file not in used_files:
                labeled_points.append((csv_file, distance, target_lat, target_lon, label))
                used_files.add(csv_file)
                continue

        # Fallback: nearest unused file by geodesic distance.
        # Needed when the rotated ICON-D2 grid does not cover every lat×lon intersection.
        # Single-pass minimum scan (O(M)) avoids sorting the full file list per label.
        best_fd = None
        best_dist = float('inf')
        for fd in file_distances:
            if fd[0] not in used_files:
                d = geodesic((fd[2], fd[3]), (target_lat, target_lon)).kilometers
                if d < best_dist:
                    best_dist = d
                    best_fd = fd
        if best_fd is not None:
            logging.debug(
                f"relative_position: exact point ({target_lat:.4f}, {target_lon:.4f}) "
                f"for '{label}' not available; using nearest ({best_fd[2]:.4f}, {best_fd[3]:.4f}) "
                f"at {best_dist:.4f} km."
            )
            labeled_points.append((best_fd[0], best_fd[1], best_fd[2], best_fd[3], label))
            used_files.add(best_fd[0])
        else:
            logging.warning(f"relative_position: no unique file left for label '{label}' — skipping.")

    # Compute relative position of the station within the inner 2×2 quad.
    # SW corner:  (grid_lats[lat_idx - 1], grid_lons[lon_idx - 1])
    # NW corner:  (grid_lats[lat_idx],     grid_lons[lon_idx - 1])   (same lon as SW)
    # SE corner:  (grid_lats[lat_idx - 1], grid_lons[lon_idx])        (same lat as SW)
    sw_lat = grid_lats[lat_idx - 1]
    nw_lat = grid_lats[lat_idx]        # == SW lat + one grid step north
    sw_lon = grid_lons[lon_idx - 1]
    se_lon = grid_lons[lon_idx]        # == SW lon + one grid step east

    # station_rel_lat: 0 = southern edge of quad, 1 = northern edge
    station_rel_lat = float((station_lat - sw_lat) / (nw_lat - sw_lat))
    # station_rel_lon: 0 = western edge of quad, 1 = eastern edge
    station_rel_lon = float((station_lon - sw_lon) / (se_lon - sw_lon))

    return labeled_points, station_rel_lat, station_rel_lon


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

        # Parse coordinates from filenames. Geodesic distance computation is deferred to
        # after grid point selection so that it is only called for the N selected files,
        # not for every file in the directory (which can be 50+ for a single station).
        file_coords = []  # (csv_file, grid_lat, grid_lon)
        for csv_file in csv_files:
            parts = csv_file.replace('_ML.csv', '').split('_')
            if len(parts) >= 4:
                try:
                    grid_lat = float(f"{parts[0]}.{parts[1]}")
                    grid_lon = float(f"{parts[2]}.{parts[3]}")
                    file_coords.append((csv_file, grid_lat, grid_lon))
                except (ValueError, IndexError):
                    continue

        # --- Grid point selection: branch on configured method ---
        get_method = config['params'].get('get_next_grid_points_method', 'geodesic_next')
        station_rel_lat = None
        station_rel_lon = None

        if get_method == 'geodesic_next':
            # ---- existing behavior: rank by geodesic distance, completely unchanged ----
            # Geodesic is computed for all files because the ranking over all of them is needed.
            file_distances = [
                (csv_file, geodesic((station_lat, station_lon), (lat, lon)).kilometers, lat, lon)
                for csv_file, lat, lon in file_coords
            ]
            file_distances.sort(key=lambda x: x[1])
            nearest_files = file_distances[:next_n_grid_points]

            if not nearest_files:
                logging.warning(f"No valid files found for forecast hour {forecast_hour}")
                return None, forecast_hour

            # The nearest point always gets rank i=1, so its column suffix is '1'.
            nearest_label = '1'

            # Prepare arguments for parallel CSV processing
            # i is the 1-based distance rank; used as column suffix (e.g. wind_speed_h78_1)
            csv_args = []
            for i, (csv_file, distance, grid_lat, grid_lon) in enumerate(nearest_files, 1):
                csv_path = os.path.join(icon_d2_base_path, csv_file)
                csv_args.append((csv_file, distance, grid_lat, grid_lon, csv_path, config, i, turbines))

        elif get_method == 'relative_position':
            # ---- new behavior: assign consistent compass-direction labels (NW, NE, …) ----
            # Pass dummy distances (0.0) — spatial selection uses coordinates only.
            file_distances = [(csv_file, 0.0, lat, lon) for csv_file, lat, lon in file_coords]
            labeled_points, station_rel_lat, station_rel_lon = _select_grid_points_relative_position(
                file_distances, station_lat, station_lon, next_n_grid_points
            )

            if not labeled_points:
                logging.warning(f"No valid files found for forecast hour {forecast_hour}")
                return None, forecast_hour

            # Now compute geodesic only for the N selected files (not all M files in directory).
            labeled_points = [
                (f, geodesic((station_lat, station_lon), (lat, lon)).kilometers, lat, lon, label)
                for f, _, lat, lon, label in labeled_points
            ]

            # The geodesically nearest labeled point determines the NWP baseline column.
            nearest_label = str(min(labeled_points, key=lambda x: x[1])[4])

            # nearest_files keeps the plain 4-tuple format expected by the IDW aggregation
            # code below (distances are at index [1])
            nearest_files = [(f, d, lat, lon) for f, d, lat, lon, _ in labeled_points]

            # label is used as column suffix (e.g. wind_speed_h78_NW) instead of an int rank
            csv_args = []
            for csv_file, distance, grid_lat, grid_lon, label in labeled_points:
                csv_path = os.path.join(icon_d2_base_path, csv_file)
                csv_args.append((csv_file, distance, grid_lat, grid_lon, csv_path, config, label, turbines))

        else:
            raise ValueError(
                f"Unknown get_next_grid_points_method: '{get_method}'. "
                f"Must be 'geodesic_next' or 'relative_position'."
            )

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

            # Return station_rel_lat/lon (relative_position only) and the nearest NWP label
            # so the caller can store them as metadata on the final DataFrame.
            return df_forecast_hour, forecast_hour, station_rel_lat, station_rel_lon, nearest_label
        else:
            return None, forecast_hour

    except Exception as e:
        logging.error(f"Error processing forecast hour {forecast_hour}: {str(e)}")
        return None, forecast_hour


def extrapolate_to_hub_height(df: pd.DataFrame, source_col_prefix: str, hub_height: float) -> pd.Series:
    """
    Extrapoliert Windgeschwindigkeit von Messhöhe auf Nabenhöhe mittels Power Law.

    Alpha (Scherungsexponent) wird per Zeitschritt aus der Quellspalte (als untere Referenz)
    und wind_speed_h127 (als obere Referenz) geschätzt:
        alpha = ln(v127 / v_source) / ln(127 / h_source)

    Spalten ohne _h<N>-Suffix (z.B. 'wind_speed') werden als h_source=10m angenommen.

    Extrapolation:
        v_hub = v_source * (hub_height / h_source) ** alpha

    Args:
        df:                DataFrame mit NWP- und/oder Messspalten.
        source_col_prefix: Spaltenname ohne Gitterpunkt-Suffix (z.B. 'wind_speed_h10' oder 'wind_speed').
                           Quellhöhe wird aus _h<N> extrahiert; ohne Suffix wird h_source=10m angenommen.
        hub_height:        Zielhöhe in Metern (Nabenhöhe).

    Returns:
        pd.Series mit extrapolierten Werten bei hub_height, benannt 'wind_speed_hub'.

    Raises:
        ValueError: Wenn Quellspalte oder wind_speed_h127* nicht gefunden.
    """
    # Quellhöhe aus Spaltenname extrahieren; Fallback 10m für Spalten ohne _h<N>
    h_match = re.search(r'_h(\d+)', source_col_prefix)
    if h_match:
        h_source = float(h_match.group(1))
    else:
        h_source = 10.0
        logging.info(
            f"extrapolate: Keine Höhe in Spaltenname '{source_col_prefix}' gefunden — "
            f"h_source=10m wird angenommen."
        )

    # Quellspalte finden: exakter Match oder Gitterpunkt-Suffix _<N> (z.B. wind_speed_h10_1)
    # Kein beliebiges startswith, um z.B. 'wind_speed' nicht auf 'wind_speed_h10_1' zu matchen
    source_pattern = re.compile(f"^{re.escape(source_col_prefix)}(_\\d+)?$")
    source_candidates = [c for c in df.columns if source_pattern.match(c)]
    if not source_candidates:
        raise ValueError(
            f"Keine Spalte für '{source_col_prefix}' im DataFrame gefunden. "
            f"Verfügbare Spalten: {list(df.columns)}"
        )
    source_col = next((c for c in source_candidates if c == source_col_prefix),  # exakter Match bevorzugt
                      next((c for c in source_candidates if c.endswith('_1')), source_candidates[0]))

    # Obere Referenzspalte für Alpha-Schätzung: wind_speed_h127 (nächster Gitterpunkt)
    v127_candidates = [c for c in df.columns if c.startswith('wind_speed_h127')]
    if not v127_candidates:
        raise ValueError(
            "Alpha-Schätzung erfordert Spalten 'wind_speed_h127*' im DataFrame. "
            "Sicherstellen, dass diese Höhe in den NWP-Daten vorhanden ist."
        )
    v127_col = next((c for c in v127_candidates if c.endswith('_1')), v127_candidates[0])

    v_source = df[source_col].clip(lower=0.01)
    v127 = df[v127_col].clip(lower=0.01)

    # Per-Zeitschritt Scherungsexponent; physikalisch sinnvoll auf [0, 1] clippen
    # Sonderfall: h_source == 127m → kein Höhenunterschied, alpha=0, v_hub=v_source
    if h_source == 127.0:
        logging.warning("extrapolate: h_source == 127m == h_ref_high → alpha=0, keine Extrapolation.")
        alpha = pd.Series(0.0, index=df.index)
    else:
        alpha = (np.log(v127 / v_source) / np.log(127.0 / h_source)).clip(0.0, 1.0)

    v_hub = df[source_col] * (hub_height / h_source) ** alpha
    v_hub.name = 'wind_speed_hub_extrap'

    logging.info(
        f"Extrapoliere '{source_col}' (h={h_source:.0f}m) → Nabenhöhe {hub_height:.1f}m "
        f"via Power Law. Mean alpha={alpha.mean():.3f}, alpha-Referenz: '{source_col}' und '{v127_col}'."
    )
    return v_hub


def _fetch_ecmwf_data(station_lat: float,
                      station_lon: float,
                      next_n_grid_points: int,
                      db_url: str,
                      raw_columns: list,
                      starttime_min: pd.Timestamp = None,
                      starttime_max: pd.Timestamp = None) -> pd.DataFrame:
    """
    Fetch ECMWF wind data from PostGIS database for the N nearest grid points.

    Credentials are read from the ECMWF_WIND_SL_URL environment variable (never stored in configs).

    Args:
        station_lat: Station latitude
        station_lon: Station longitude
        next_n_grid_points: Number of nearest ECMWF grid points to fetch
        db_url: SQLAlchemy connection URL (e.g. postgresql://user:pass@host/db)
        raw_columns: List of raw DB column names to fetch (e.g. ['u_wind_10m', 'v_wind_10m'])
        starttime_min: Optional lower bound for ECMWF starttime (inclusive)
        starttime_max: Optional upper bound for ECMWF starttime (inclusive)

    Returns:
        DataFrame with columns: starttime (UTC, tz-aware), forecasttime (int), <raw_columns>, rank (int)
    """
    try:
        from sqlalchemy import create_engine
    except ImportError:
        raise ImportError(
            "sqlalchemy is required for ECMWF data fetching. "
            "Install with: pip install sqlalchemy psycopg2-binary"
        )

    date_filter = ""
    if starttime_min is not None and starttime_max is not None:
        # Format as ISO strings; values are internal pd.Timestamps, not user input
        ts_min = starttime_min.strftime('%Y-%m-%d %H:%M:%S%z')
        ts_max = starttime_max.strftime('%Y-%m-%d %H:%M:%S%z')
        date_filter = f"AND e.starttime BETWEEN '{ts_min}' AND '{ts_max}'"

    # Denormalize column names back to DB convention (u_wind10m → u_wind_10m)
    # raw_columns use the normalized form; DB stores them with underscore before the numeric suffix.
    db_col_names = [re.sub(r'(\D)(\d+m)', r'\1_\2', col) for col in raw_columns]
    select_cols = ", ".join(f"e.{col}" for col in db_col_names)
    query = f"""
    WITH nearest_points AS (
        SELECT geom,
               ROW_NUMBER() OVER (
                   ORDER BY geom <-> ST_SetSRID(ST_MakePoint({station_lon}, {station_lat}), 4326)
               ) AS rank
        FROM ecmwf_grid_points
        ORDER BY geom <-> ST_SetSRID(ST_MakePoint({station_lon}, {station_lat}), 4326)
        LIMIT {next_n_grid_points}
    )
    SELECT e.starttime, e.forecasttime, {select_cols}, n.rank
    FROM ecmwf_wind_sl e
    JOIN nearest_points n ON e.geom = n.geom
    WHERE 1=1 {date_filter}
    ORDER BY n.rank, e.starttime, e.forecasttime
    """

    engine = create_engine(db_url)
    try:
        df = pd.read_sql(query, engine)
    finally:
        engine.dispose()

    df['starttime'] = pd.to_datetime(df['starttime'], utc=True)
    df['forecasttime'] = df['forecasttime'].astype(int)
    df['rank'] = df['rank'].astype(int)
    # Normalize column names: remove underscore before numeric+m suffix (e.g. u_wind_10m → u_wind10m)
    # so naming is consistent with ICON-D2 conventions used in ecmwf_features / known_features config.
    df.columns = [re.sub(r'_(\d+m)', r'\1', c) for c in df.columns]
    # Integer key YYYYMMDDHH — avoids datetime64[ns] vs datetime64[us] merge mismatches
    # that occur when pandas 2.x reads timestamps from a DB (µs precision) vs CSV (ns precision).
    df['_st_key'] = (
        df['starttime'].dt.year * 1000000
        + df['starttime'].dt.month * 10000
        + df['starttime'].dt.day * 100
        + df['starttime'].dt.hour
    ).astype(np.int64)
    return df


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
    if 'station_turbine_type' not in config['params']:
        # Fallback: use random seed if no CSV-based assignment
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
        if 'station_turbine_type' in config['params']:
            # Deterministic assignment from station_turbine_assignments.csv
            assigned_turbine = config['params']['station_turbine_type']
            if assigned_turbine in turbines_list:
                turbines_list = [assigned_turbine]
                logging.debug(f"Using assigned turbine '{assigned_turbine}' for station {station_id}")
            else:
                logging.warning(f"Assigned turbine '{assigned_turbine}' not in turbines list. "
                                f"Falling back to random selection.")
                turbines_list = random.sample(turbines_list, config['params']['turbines_per_park'])
                config['params']['random_seed'] += 1
        elif 'turbines_per_park' in config['params']:
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

    if static_features:
        static_data['latitude'] = station_lat
        static_data['longitude'] = station_lon

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
    # Relative-position scalars are the same for every forecast hour; capture from first result
    _station_rel_lat = None
    _nearest_label = None
    _station_rel_lon = None

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_hour = {executor.submit(_process_forecast_hour, arg): arg[0] for arg in forecast_args}

        for future in as_completed(future_to_hour):
            hour = future_to_hour[future]
            result = future.result()
            if result[0] is not None:
                all_forecast_dfs.append((result[0], hour))
                # Capture relative-position scalars (index 2/3) and nearest NWP label (index 4).
                # Take the first non-None value across forecast hours — they should be identical
                # for a given station since its position relative to the grid doesn't change.
                if len(result) > 2 and result[2] is not None and _station_rel_lat is None:
                    _station_rel_lat = result[2]
                    _station_rel_lon = result[3]
                if len(result) > 4 and result[4] is not None and _nearest_label is None:
                    _nearest_label = result[4]

    # If relative_position method returned relative scalars, store them as static features
    if _station_rel_lat is not None:
        static_data['station_rel_lat'] = _station_rel_lat
        static_data['station_rel_lon'] = _station_rel_lon

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

    # 3b. Load neighbor station data for '_next' features if next_n_stations > 0
    _next_n_stations = config['params'].get('next_n_stations') or 0
    _next_features_requested = [f for f in (features.get('known', []) + features.get('observed', []))
                                 if f.endswith('_next')] if features else []

    if _next_n_stations > 0 and _next_features_requested:
        _base_next_features = [f[:-5] for f in _next_features_requested]  # strip '_next'
        _station_coords = (station_lat, station_lon)
        _other_stations = wind_parameter[wind_parameter['park_id'] != station_id].copy()
        _other_stations['_distance_km'] = _other_stations.apply(
            lambda row: geodesic(_station_coords, (row['latitude'], row['longitude'])).km, axis=1
        )
        _nearest = _other_stations.nsmallest(_next_n_stations, '_distance_km')

        for _rank, (_, _neighbor_row) in enumerate(_nearest.iterrows(), start=1):
            _neighbor_id = str(_neighbor_row['park_id'])
            _neighbor_synth_path = os.path.join(config['data']['path'], f'synth_{_neighbor_id}.csv')
            if not os.path.exists(_neighbor_synth_path):
                logging.warning(f"next_n_stations: synth file for neighbor {_neighbor_id} not found, skipping.")
                continue

            _df_neighbor = pd.read_csv(_neighbor_synth_path, sep=';')
            _df_neighbor['timestamp'] = pd.to_datetime(_df_neighbor['timestamp'], utc=True)
            _df_neighbor.set_index('timestamp', inplace=True)

            for _base_feat in _base_next_features:
                _col_name = f'{_base_feat}_next_{_rank}'
                if _base_feat in _df_neighbor.columns:
                    df_merged = df_merged.merge(
                        _df_neighbor[[_base_feat]].rename(columns={_base_feat: _col_name}),
                        left_on='timestamp', right_index=True, how='left'
                    )
                else:
                    logging.warning(f"next_n_stations: feature '{_base_feat}' not in neighbor {_neighbor_id}, skipping.")

        logging.debug(f"Station {station_id}: loaded '{_next_features_requested}' from {_next_n_stations} nearest neighbors")

    # 3c. Merge ECMWF features if requested
    _nwp_models = config['params'].get('nwp_models', ['icon-d2'])
    _ecmwf_features_cfg = config['params'].get('ecmwf_features', [])

    # Derive which ECMWF output features are requested from known/observed features (ecmwf_* entries)
    _ecmwf_known_feats = [
        f for f in (features.get('known', []) + features.get('observed', []))
        if f.startswith('ecmwf_')
    ] if features else []

    if 'ecmwf' in _nwp_models and _ecmwf_features_cfg and _ecmwf_known_feats:
        _db_url = os.environ.get('ECMWF_WIND_SL_URL')
        if not _db_url:
            logging.warning(
                f"Station {station_id}: ECMWF_WIND_SL_URL not set — skipping ECMWF data. "
                "Export the variable before running (e.g. export ECMWF_WIND_SL_URL=postgresql://...)."
            )
        else:
            # Determine date range from the already-loaded ICON-D2 data
            _icon_starttimes = pd.to_datetime(df_merged['starttime'], utc=True)
            _ecmwf_ts_min = _icon_starttimes.min().normalize()
            _ecmwf_ts_max_data = _icon_starttimes.max().normalize() + pd.Timedelta(hours=13)
            _test_end = config['data'].get('test_end')
            if _test_end:
                # +12h so the noon ECMWF run on test_end is included
                _ecmwf_ts_max = min(
                    _ecmwf_ts_max_data,
                    pd.Timestamp(_test_end, tz='UTC') + pd.Timedelta(hours=12)
                )
            else:
                _ecmwf_ts_max = _ecmwf_ts_max_data

            try:
                _next_n_grid_ecmwf = config['params'].get('next_n_grid_ecmwf', next_n_grid_points)
                _df_ecmwf = _fetch_ecmwf_data(
                    station_lat=station_lat,
                    station_lon=station_lon,
                    next_n_grid_points=_next_n_grid_ecmwf,
                    db_url=_db_url,
                    raw_columns=_ecmwf_features_cfg,
                    starttime_min=_ecmwf_ts_min,
                    starttime_max=_ecmwf_ts_max,
                )
            except Exception as _ecmwf_exc:
                logging.error(f"Station {station_id}: ECMWF fetch failed — {_ecmwf_exc}")
                _df_ecmwf = pd.DataFrame()

            if _df_ecmwf.empty:
                logging.warning(f"Station {station_id}: ECMWF query returned no data.")
            else:
                # For each ICON-D2 row, compute the corresponding ECMWF (starttime, forecasttime):
                #   ECMWF runs at 00:00 and 12:00 UTC.
                #   Use the most recent ECMWF run <= icon starttime.
                #   valid_time = icon_starttime + icon_forecasttime  →  ecmwf_forecasttime = valid_time - ecmwf_starttime
                _icon_st = pd.to_datetime(df_merged['starttime'], utc=True)
                _ecmwf_run_hour = np.where(_icon_st.dt.hour >= 12, 12, 0)
                _ecmwf_starttime = _icon_st.dt.normalize() + pd.to_timedelta(_ecmwf_run_hour, unit='h')
                # Ensure UTC timezone is preserved after arithmetic (guards against tz stripping)
                if _ecmwf_starttime.dt.tz is None:
                    _ecmwf_starttime = _ecmwf_starttime.dt.tz_localize('UTC')
                _valid_time = _icon_st + pd.to_timedelta(df_merged['forecasttime'], unit='h')
                _ecmwf_forecasttime = (
                    (_valid_time - _ecmwf_starttime).dt.total_seconds() / 3600
                ).astype(int)

                df_merged = df_merged.copy()
                # Use integer YYYYMMDDHH key instead of timestamp to avoid dtype mismatches
                # between datetime64[ns, UTC] (CSV) and datetime64[us, UTC] (DB via SQLAlchemy)
                df_merged['_ecmwf_st_key'] = (
                    _ecmwf_starttime.dt.year * 1000000
                    + _ecmwf_starttime.dt.month * 10000
                    + _ecmwf_starttime.dt.day * 100
                    + _ecmwf_starttime.dt.hour
                ).astype(np.int64)
                df_merged['_ecmwf_ft'] = _ecmwf_forecasttime

                # Precompute test_end boundary for warning filter (rows beyond test_end are expected to have no match)
                _test_end_ts = (
                    pd.Timestamp(_test_end, tz='UTC') if _test_end else None
                )

                for _rank in sorted(_df_ecmwf['rank'].unique()):
                    _df_rank = _df_ecmwf[_df_ecmwf['rank'] == _rank][
                        ['_st_key', 'forecasttime'] + _ecmwf_features_cfg
                    ].copy()
                    _df_rank = _df_rank.rename(columns={
                        '_st_key': '_ecmwf_st_key',
                        'forecasttime': '_ecmwf_ft',
                    })

                    # Compute/rename output features based on known_features entries starting with 'ecmwf_'
                    for _out_feat in _ecmwf_known_feats:
                        _raw_name = _out_feat[len('ecmwf_'):]  # strip 'ecmwf_' prefix
                        if _out_feat == 'ecmwf_wind_speed_h10':
                            if 'u_wind10m' in _df_rank.columns and 'v_wind10m' in _df_rank.columns:
                                _df_rank[f'ecmwf_wind_speed_h10_{_rank}'] = np.sqrt(
                                    _df_rank['u_wind10m'] ** 2 + _df_rank['v_wind10m'] ** 2
                                )
                            else:
                                logging.warning(
                                    f"Station {station_id}: 'ecmwf_wind_speed_h10' requires "
                                    "'u_wind10m' and 'v_wind10m' in ecmwf_features — skipping."
                                )
                        elif _raw_name in _df_rank.columns:
                            _df_rank[f'{_out_feat}_{_rank}'] = _df_rank[_raw_name]
                        else:
                            logging.warning(
                                f"Station {station_id}: ECMWF column '{_raw_name}' not fetched "
                                f"(not in ecmwf_features) — skipping '{_out_feat}'."
                            )

                    _ecmwf_out_cols = [
                        c for c in _df_rank.columns
                        if c not in ('_ecmwf_st_key', '_ecmwf_ft') and c not in _ecmwf_features_cfg
                    ]
                    _df_rank = _df_rank[['_ecmwf_st_key', '_ecmwf_ft'] + _ecmwf_out_cols].drop_duplicates(
                        subset=['_ecmwf_st_key', '_ecmwf_ft']
                    )

                    df_merged = df_merged.merge(
                        _df_rank, on=['_ecmwf_st_key', '_ecmwf_ft'], how='left'
                    )
                    _missing_mask = df_merged[[c for c in _ecmwf_out_cols if c in df_merged.columns]].isna().any(axis=1)
                    if _test_end_ts is not None:
                        # Only warn for rows within test_end — rows beyond it have no ECMWF data by design
                        _missing_mask = _missing_mask & (df_merged['starttime'] <= _test_end_ts)
                    _missing = _missing_mask.sum()
                    if _missing > 0:
                        logging.warning(
                            f"Station {station_id}: {_missing} rows within test_end have no ECMWF match for rank {_rank}."
                        )

                df_merged.drop(columns=['_ecmwf_st_key', '_ecmwf_ft'], inplace=True)

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

    # Replace wind_speed_t with wind_speed_hub for consistent naming across turbines
    if 'wind_speed_t' in rel_features:
        rel_features.remove('wind_speed_t')
        # Add wind_speed_hub as the standardized feature name
        # The actual column will be renamed from wind_speed_t* to wind_speed_hub later
        new_rel_features.append('wind_speed_hub')


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
    # IMPORTANT: Don't drop if wind_speed_hub is requested, as we need to rename wind_speed_t* to wind_speed_hub
    if ('wind_speed_t' not in features.get('known', []) and 'wind_speed_t' not in features.get('observed', []) and
        'wind_speed_hub' not in features.get('known', []) and 'wind_speed_hub' not in features.get('observed', [])):
        for col in df_merged.columns:
            if col.startswith('wind_speed_t'):
                cols_to_drop.append(col)


    # Always drop generic wind_speed (without suffix), unless it's the target column
    # or used as extrapolation source (must survive until after extrapolation)
    if ('wind_speed' in df_merged.columns and 'wind_speed' not in rel_features
            and config['data'].get('target_col') != 'wind_speed'
            and config['params'].get('extrapolate') != 'wind_speed'):
        cols_to_drop.append('wind_speed')

    # Drop the unwanted columns
    if cols_to_drop:
        df_merged.drop(columns=cols_to_drop, inplace=True)

    # Rename wind_speed_t* columns to wind_speed_hub for consistent feature names across turbines
    # This is important for federated learning where different stations may have different turbine types
    # Check if either wind_speed_t or wind_speed_hub is requested (wind_speed_hub is the new standardized name)
    if ('wind_speed_t' in features.get('known', []) or 'wind_speed_t' in features.get('observed', []) or
        'wind_speed_hub' in features.get('known', []) or 'wind_speed_hub' in features.get('observed', [])):
        wind_speed_t_cols = [col for col in df_merged.columns if col.startswith('wind_speed_t') and col != 'wind_speed_t']
        if wind_speed_t_cols:
            # There should only be one wind_speed_t* column per station (one turbine type)
            # Rename it to wind_speed_hub so all stations have the same column name
            for col in wind_speed_t_cols:
                df_merged.rename(columns={col: 'wind_speed_hub'}, inplace=True)

    # Extrapolate wind speed to hub height (Power Law)
    extrapolate_col = config['params'].get('extrapolate')
    if extrapolate_col:
        hub_height_val = float(np.mean(heights)) if len(heights) > 0 else None
        if hub_height_val is None:
            logging.warning("extrapolate: Nabenhöhe nicht verfügbar — Extrapolation übersprungen.")
        else:
            df_merged['wind_speed_hub_extrap'] = extrapolate_to_hub_height(
                df=df_merged,
                source_col_prefix=extrapolate_col,
                hub_height=hub_height_val
            )

    # Add Icon-D2 features
    if rel_features:
        # Create list of all possible feature combinations with suffixes
        available_cols = []

        # Create regex patterns for strict matching
        # Matches exactly the feature name, optionally followed by _<digits> (for grid points)
        patterns = [re.compile(f"^{re.escape(feat)}(_[A-Z0-9]+)?$") for feat in rel_features]

        for col in df_merged.columns:

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

        # Keep wind_speed_h10* for NWP baseline evaluation when target is wind_speed
        if config['data'].get('target_col') == 'wind_speed':
            for col in df_merged.columns:
                if col.startswith('wind_speed_h10') and col not in available_cols:
                    available_cols.append(col)

        # Auto-include wind_speed_hub_extrap when extrapolation was applied
        if config['params'].get('extrapolate') and 'wind_speed_hub_extrap' in df_merged.columns:
            if 'wind_speed_hub_extrap' not in available_cols:
                available_cols.append('wind_speed_hub_extrap')

        # Remove duplicates and filter
        available_cols = list(dict.fromkeys(available_cols))
        available_cols = [col for col in available_cols if col in df_merged.columns]
        df_merged = df_merged[available_cols]

    # Sin/cos encode all wind direction columns (cyclical feature, 0-359 degrees)
    wind_dir_cols = [col for col in df_merged.columns if 'wind_direction' in col]
    if wind_dir_cols:
        for col in wind_dir_cols:
            rad = np.deg2rad(df_merged[col])
            df_merged[f'{col}_sin'] = np.sin(rad)
            df_merged[f'{col}_cos'] = np.cos(rad)
        df_merged.drop(columns=wind_dir_cols, inplace=True)
        logging.debug(f"Sin/cos encoded wind direction columns: {wind_dir_cols}")

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
        known_feature_patterns = [re.compile(f"^{re.escape(feat)}(_[A-Z0-9]+)?$") for feat in original_known_features]

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

    # Store the nearest NWP grid point label as DataFrame metadata so that
    # compute_skill_nwp can select the correct baseline column without guessing.
    # For geodesic_next: always '1'. For relative_position: geodesically nearest label (e.g. 'SW').
    if _nearest_label is not None:
        df_merged.attrs['nwp_nearest_label'] = _nearest_label

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
    if 'station_turbine_type' not in config['params']:
        # Fallback: use random seed if no CSV-based assignment
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
        if 'station_turbine_type' in config['params']:
            # Deterministic assignment from station_turbine_assignments.csv
            assigned_turbine = config['params']['station_turbine_type']
            if assigned_turbine in turbines_list:
                turbines_list = [assigned_turbine]
                logging.debug(f"Using assigned turbine '{assigned_turbine}' for station {park_id}")
            else:
                logging.warning(f"Assigned turbine '{assigned_turbine}' not in turbines list. "
                                f"Falling back to random selection.")
                turbines_list = random.sample(turbines_list, config['params']['turbines_per_park'])
                config['params']['random_seed'] += 1
        elif 'turbines_per_park' in config['params']:
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
            if 'wind_speed_rotor_eq' in rel_features:
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
                         scaler_x: StandardScaler = None,
                         scaler_y: StandardScaler = None):
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
            # Use feature_cols (the columns used to fit the scaler) to find static feature indices
            # feature_cols is defined later at line 1815, so we compute it here too
            scaler_feature_cols = [c for c in train_df.columns if c != target_col]
            static_features_in_scaler = [col for col in static_cols if col in scaler_feature_cols]

            if static_features_in_scaler:
                # Create a dummy row with zeros for all features
                dummy_row_train = np.zeros((1, len(scaler_feature_cols)))
                dummy_row_test = np.zeros((1, len(scaler_feature_cols)))

                # Fill in the static feature values at the correct positions
                for i, col in enumerate(static_cols):
                    if col in static_features_in_scaler:
                        col_idx = scaler_feature_cols.index(col)
                        dummy_row_train[0, col_idx] = X_static_train[i]
                        dummy_row_test[0, col_idx] = X_static_test[i]

                # Transform
                scaled_train = scaler_x.transform(dummy_row_train)
                scaled_test = scaler_x.transform(dummy_row_test)

                # Extract only the static feature columns
                static_indices = [scaler_feature_cols.index(col) for col in static_features_in_scaler]
                X_static_train = scaled_train[0, static_indices]
                X_static_test = scaled_test[0, static_indices]

                logging.debug(f"Static features scaled using global scaler_x: {static_features_in_scaler}")
            else:
                logging.warning(f"Static features {static_cols} not found in feature columns. "
                              f"Static features will NOT be scaled!")
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
        train_df_scaled[feature_cols] = scaler_x.transform(train_df[feature_cols].values)
        test_df_scaled[feature_cols] = scaler_x.transform(test_df[feature_cols].values)

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
        target_train_scaled, target_test_scaled, target_scaler = apply_scaling(
            target_train_raw, target_test_raw,
            scaler_type=scaler_y if scaler_y is not None else StandardScaler,
            fit=(scaler_y is None)
        )
        scalers['y'] = target_scaler
        logging.debug(f"Target variable '{target_col}' scaled ({'global' if scaler_y is not None else 'per-station'} scaler).")
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

    has_observed = X_observed_train.ndim == 3 and X_observed_train.shape[-1] > 0
    num_train_samples = X_observed_train.shape[0] if has_observed else X_known_train.shape[0]
    num_test_samples = X_observed_test.shape[0] if has_observed else X_known_test.shape[0]

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

            # Number of past forecast runs needed to cover history_len.
            # E.g. history_len=96, future_len=48 → 2 past runs needed.
            n_past_runs = math.ceil(history_len / future_len)

            # Process each forecast run
            for current_start in unique_starttimes:
                # Get current run indices (needed for target + observed timestamp anchor)
                current_indices = starttime_to_indices.get(current_start)
                if current_indices is None or len(current_indices) == 0:
                    continue

                # Known features: stack n_past_runs previous runs (oldest first) + current run.
                # Resulting shape: (n_past_runs * future_len + future_len) = history_len + future_len
                if known_data is not None:
                    past_chunks = []
                    valid = True
                    for k in range(n_past_runs, 0, -1):  # oldest run first
                        past_start = current_start - pd.Timedelta(hours=k * step_size)
                        past_indices = starttime_to_indices.get(past_start)
                        if past_indices is None or len(past_indices) != future_len:
                            valid = False
                            break
                        past_chunks.append(known_data[past_indices])

                    if not valid:
                        continue

                    if len(current_indices) != future_len:
                        logging.warning(f"Unexpected current run length at {current_start}: "
                                       f"{len(current_indices)}, expected {future_len}")
                        continue

                    known_window = np.vstack(past_chunks + [known_data[current_indices]])
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

        n_samples = len(y_list)
        X_observed_arr = (np.zeros((n_samples, history_len, 0))
                          if not X_observed_list
                          else np.array(X_observed_list))
        return (np.array(X_known_list),
                X_observed_arr,
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


def prepare_data_for_chronos2(data: pd.DataFrame,
                               history_length: int,
                               future_horizon: int,
                               known_future_cols: list,
                               observed_past_cols: list,
                               static_cols: list,
                               step_size: int = 1,
                               train_frac: float = 0.75,
                               target_col: str = 'power',
                               train_start: pd.Timestamp = None,
                               test_start: pd.Timestamp = None,
                               test_end: pd.Timestamp = None,
                               t_0: int = 0,
                               scaler_x=None):
    """
    Prepare data for Chronos-2.

    Reuses the TFT windowing logic (same observed/known/static structure).
    The target column is NOT scaled because Chronos-2 applies its own robust
    normalisation internally.  NWP features (known_future_cols) are scaled via
    scaler_x so that covariate magnitudes are comparable.

    Returns the same results dict as prepare_data_for_tft(), so caching and
    get_y_chronos2() can reuse it without modification.
    """
    return prepare_data_for_tft(
        data=data,
        history_length=history_length,
        future_horizon=future_horizon,
        known_future_cols=known_future_cols,
        observed_past_cols=observed_past_cols,
        static_cols=static_cols,
        step_size=step_size,
        train_frac=train_frac,
        target_col=target_col,
        train_start=train_start,
        test_start=test_start,
        test_end=test_end,
        t_0=t_0,
        scale_target=False,
        scaler_x=scaler_x,
    )

    return results