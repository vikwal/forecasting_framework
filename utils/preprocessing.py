import os
import random
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import logging
from typing import List, Tuple, Dict, Any

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
                    df = preprocess_synth_wind(path=file_path,
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
            preprocess_func = preprocess_synth_wind
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

def pipeline(data: pd.DataFrame,
             config: Dict[str, Any],
             known_cols: List[str] = None,
             observed_cols: List[str] = None,
             static_cols: List[str] = None,
             target_col: str = 'power') -> Tuple[Dict, pd.DataFrame]:
    df = data.copy()
    df = knn_imputer(data=df, n_neighbors=config['data']['n_neighbors'])
    t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
    if config['model']['name'] == 'tft':
        logging.debug(f'Features ready for prepare_data(): {df.columns.to_list()}')
        prepared_data = prepare_data_for_tft(data=df,
                                             history_length=config['model']['lookback'], # e.g., 72 (for 3 days history)
                                             future_horizon=config['model']['horizon'], # e.g., 24 (for 24 hours forecast)
                                             step_size=config['model']['step_size'],
                                             known_future_cols=known_cols,
                                             observed_past_cols=observed_cols,
                                             static_cols=static_cols,
                                             train_frac=config['data']['train_frac'],
                                             test_start=pd.Timestamp(config['data']['test_start']),
                                             test_end=pd.Timestamp(config['data']['test_end']),
                                             t_0=t_0)
    else:
        # create new known cols
        for col in known_cols:
            all_known_cols = [new_col for new_col in df.columns if col in new_col]
        for col in observed_cols:
            all_observed_cols = [new_col for new_col in df.columns if col in new_col]
            for new_col in all_observed_cols:
                df = lag_features(data=df,
                                lookback=config['model']['lookback'],
                                horizon=config['model']['horizon'],
                                lag_in_col=config['data']['lag_in_col'],
                                target_col=new_col)
                if new_col != target_col and new_col not in all_known_cols: df.drop(new_col, axis=1, inplace=True, errors='ignore')
        logging.debug(f'Features ready for prepare_data(): {df.columns.to_list()}')
        prepared_data = prepare_data(data=df,
                                     output_dim=config['model']['output_dim'],
                                     step_size=config['model']['step_size'],
                                     train_frac=config['data']['train_frac'],
                                     scale_y=config['data']['scale_y'],
                                     t_0=t_0,
                                     test_start=pd.Timestamp(config['data']['test_start'], tz='UTC'),
                                     test_end=pd.Timestamp(config['data']['test_end'], tz='UTC'),
                                     target_col=target_col)
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

def make_windows(data: np.ndarray,
                 seq_len: int,
                 step_size: int = 1,
                 indices: pd.DatetimeIndex = None) -> np.ndarray:
    windows = np.array([data[i:i + seq_len] for i in range(0,
                                                           data.shape[0]-seq_len+1,
                                                           step_size)])
    if 'index' in dir(data) and indices is not None:
        index = data.index
        freq = data.index[1] - data.index[0]
        shifted_index = index # - freq # laut Gemini ein Bug
        shifted_index = shifted_index[:shifted_index.shape[0]-seq_len+1]
        mask = np.array([True if i in indices else False for i in shifted_index])
        windows = windows[mask]
    return windows

def apply_scaling(df_train,
                  df_test,
                  scaler_type=StandardScaler):
    scaler = scaler_type()
    if len(df_train.shape) == 1:
        df_train = df_train.values.reshape(-1, 1)
    if len(df_test.shape) == 1:
        df_test = df_test.values.reshape(-1, 1)
    scaled_train = scaler.fit_transform(df_train)
    if len(df_test) == 0:
        scaled_test = df_test
    else:
        scaled_test = scaler.transform(df_test)
    return scaled_train, scaled_test, scaler

def prepare_data(data: pd.DataFrame,
                 output_dim: int,
                 step_size: int = 1,
                 train_frac: float = 0.75,
                 scale_x: bool = True,
                 scale_y: bool = False,
                 target_col: str = 'power',
                 t_0: int = 0,
                 test_start: pd.Timestamp = None,
                 test_end: pd.Timestamp = None,
                 seq2seq: bool = False):
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
    # Split data into train and test sets with a 75/25 ratio, ensuring full days
    if test_start:
        train_end = test_start - pd.Timedelta(hours=0.25)
    else:
        train_periods = int(len(df) * train_frac)-1
        train_end = index[train_periods].normalize() + pd.Timedelta(hours=(t_0-0.25))
        train_end = pd.Timestamp(train_end)
        test_start = train_end + pd.Timedelta(hours=0.25)
        test_start = pd.Timestamp(test_start)
    # handel MultiIndex if needed
    if type(data.index) == pd.core.indexes.multi.MultiIndex:
        df_train = df[df.index.get_level_values('starttime') < pd.Timestamp(test_start, tz='UTC')]
        df_test = df[df.index.get_level_values('starttime') > pd.Timestamp(train_end, tz='UTC')]
        target_train = target[target.index.get_level_values('starttime') < pd.Timestamp(test_start, tz='UTC')]
        target_test = target[target.index.get_level_values('starttime') > pd.Timestamp(train_end, tz='UTC')]
    else:
        df_train = df[:train_end]
        df_test = df[test_start:test_end]
        target_train = target[:train_end]
        target_test = target[test_start:test_end]

    #logging.info(f"Training data range: {df_train.index.min()} to {df_train.index.max()} ({len(df_train)} rows)")
    #logging.info(f"Test data range:     {df_test.index.min()} to {df_test.index.max()} ({len(df_test)} rows)")
    scalers = {}
    scalers['x'] = None
    scalers['y'] = None
    if scale_x:
        X_train, X_test, scaler_x = apply_scaling(df_train,
                                                  df_test,
                                                  StandardScaler)
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
    index_train = np.array([df_train.index[i] for i in range(0,
                                                             X_train.shape[0]-output_dim+1,
                                                             step_size)])
    index_test = np.array([df_test.index[i] for i in range(0,
                                                           X_test.shape[0]-output_dim+1,
                                                           step_size)])

    X_train = make_windows(data=X_train, seq_len=output_dim, step_size=step_size)
    X_test = make_windows(data=X_test, seq_len=output_dim, step_size=step_size)
    y_train = make_windows(data=Y_train, seq_len=output_dim, step_size=step_size).reshape(-1, output_dim)
    y_test = make_windows(data=Y_test, seq_len=output_dim, step_size=step_size).reshape(-1, output_dim)
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
    park_id = path.split('_')[-1].split('.')[0]
    parameter_file = path.replace(os.path.basename(path), 'pv_parameter.csv')
    metadata = pd.read_csv(parameter_file, sep=';')
    installed_capacity = metadata.loc[metadata.park_id == int(park_id)]['installed_capacity'].values[0]
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

def preprocess_synth_wind(path: str,
                          config: dict,
                          freq: str = '1H',
                          features: dict = None) -> pd.DataFrame:

    timestamp_col = 'timestamp'
    park_id = path.split('_')[-1].split('.')[0]
    wind_parameter_path = path.replace(os.path.basename(path), 'wind_parameter.csv')
    turbine_parameter_path = path.replace(os.path.basename(path), 'turbine_parameter.csv')
    power_curves_path = os.path.join(config['data']['power_curves_path'], 'turbine_power.csv')
    nwp_path = os.path.join(config['data']['nwp_path'], f'ML_Station_{park_id}.csv')
    rel_features = list(set(features['known'] + features['observed']))
    if 'turbines' in config['params']:
        turbines_list = config['params']['turbines']
    else:
        turbines_list = None
    wind_parameter = pd.read_csv(wind_parameter_path, sep=';', dtype={'park_id': str})
    turbine_parameter = pd.read_csv(turbine_parameter_path, sep=';', dtype={'park_id': str})
    power_curves = pd.read_csv(power_curves_path, sep=';', decimal=',')
    df = pd.read_csv(path, sep=';')
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df.set_index(timestamp_col, inplace=True)
    heights = []
    if turbines_list:
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
    df['power'] = df['power'] / installed_capacity # in Watts
    df = df.resample(freq, closed='left', label='left', origin='start').mean()
    # get nwp data if nwp is mentioned in the known features
    nwp = pd.read_csv(nwp_path, sep=',')
    nwp['starttime'] = pd.to_datetime(nwp['starttime'], utc=True)
    nwp['timestamp'] = nwp['starttime'] + pd.to_timedelta(nwp['forecasttime'], unit='h')
    # rename nwp data
    for col in nwp.columns:
        if col not in ['timestamp', 'starttime', 'forecasttime', 'toplevel', 'bottomlevel']:
            nwp.rename(columns={col: f'{col}_nwp'}, inplace=True)
    nwp['wind_speed_nwp'] = np.sqrt(nwp['u_wind_nwp']**2 + nwp['v_wind_nwp']**2)
    nwp.set_index('timestamp', inplace=True)
    # get density if required
    if config['params']['get_density']:
        nwp['sat_vap_ps'] = get_saturated_vapor_pressure(temperature=nwp['temperature_nwp'],
                                                         model='huang')
        nwp['density_nwp'] = get_density(temperature=nwp['temperature_nwp'],
                                         pressure=nwp['pressure_nwp'],
                                         relhum=nwp['relhum_nwp'],
                                         sat_vap_ps=nwp['sat_vap_ps'])
    nwp_features = rel_features.copy()
    for col in rel_features:
        if 'nwp' not in col:
            nwp_features.remove(col)
    nwp_features += ['forecasttime', 'toplevel', 'bottomlevel']
    nwp = nwp[nwp_features].copy()
    # aggregate multilevels
    if config['params']['aggregate_nwp_layers'] == 'pivot':
        mapping = {20.0: 10,
                   55.212: 37.606,
                   100.277: 77.745,
                   153.438: 126.858,
                   213.746: 183.592,
                   280.598: 247.172}
        #features_not_to_pivot = [col in nwp.columns if '_nwp' in col]
        features_to_pivot = config['params']['features_to_pivot']
        for col in features_to_pivot:
            rel_features.remove(col)
        features_not_to_pivot = [col for col in nwp.columns if col not in features_to_pivot]
        weights = get_A_weights(turbines=turbines, nwp=nwp)
        # pivot features
        nwp_pivot = nwp[features_to_pivot + ['forecasttime', 'toplevel']].copy()
        nwp_pivot.reset_index(inplace=True)
        nwp_pivot['level'] = nwp_pivot['toplevel'].map(mapping)
        nwp_pivot = nwp_pivot.pivot(index=['timestamp', 'forecasttime'],
                                    columns=['level'],
                                    values=features_to_pivot)
        nwp_pivot.reset_index(inplace=True)
        column_names = []
        for col in nwp_pivot.columns:
            if 'nwp' in col[0]:
                new_col_name = f'{col[0]}_{col[1]}'
                column_names.append(new_col_name)
                rel_features.append(new_col_name)
            else:
                column_names.append(col[0])
        nwp_pivot.columns = column_names
        nwp_pivot.set_index('timestamp', inplace=True)
        # get weighted averages of features not to pivot
        layers_idx = nwp.groupby(['toplevel','bottomlevel']).size().index
        layers = pd.DataFrame({
            'toplevel': [t for (t, b) in layers_idx],
            'bottomlevel': [b for (t, b) in layers_idx],
            'weight': weights
        })
        nwp = nwp.reset_index()
        nwp = nwp.merge(layers, on=['toplevel','bottomlevel'], how='left', validate='m:1')
        weighted_cols = []
        for col in features_not_to_pivot:
            if '_nwp' in col:
                nwp[f'{col}_w'] = nwp[col] * nwp['weight']
                weighted_cols.append(f'{col}_w')
        cols_to_group = weighted_cols + ['weight']
        agg = (
            nwp.groupby(['timestamp','forecasttime'], sort=False)[
                cols_to_group
            ].sum()
            .reset_index()
        )
        wsum = agg['weight']
        agg[weighted_cols] = (
            agg[weighted_cols].div(wsum, axis=0)
        )
        nwp = (
            agg.drop(columns='weight')
            .rename(columns=lambda c: c.replace('_w',''))
            .set_index('timestamp')
        )
        # merge pivoted and aggregated features
        nwp.reset_index(inplace=True)
        nwp_pivot.reset_index(inplace=True)
        nwp = nwp.merge(nwp_pivot, on=['timestamp', 'forecasttime'], how='left')
        nwp.set_index('timestamp', inplace=True)
    elif config['params']['aggregate_nwp_layers'] == 'weighted_mean':
        weights = get_A_weights(turbines=turbines, nwp=nwp)
        layers_idx = nwp.groupby(['toplevel','bottomlevel']).size().index
        layers = pd.DataFrame({
            'toplevel': [t for (t, b) in layers_idx],
            'bottomlevel': [b for (t, b) in layers_idx],
            'weight': weights
        })
        nwp = nwp.reset_index()  # timestamp zurück als Spalte
        nwp = nwp.merge(layers, on=['toplevel','bottomlevel'], how='left', validate='m:1')
        weighted_cols = []
        for col in nwp.columns:
            if '_nwp' in col and col != 'wind_speed_nwp':
                nwp[f'{col}_w'] = nwp[col] * nwp['weight']
                weighted_cols.append(f'{col}_w')
        nwp['wind_speed_nwp_w']  = nwp['weight'] * (nwp['wind_speed_nwp']**3)
        cols_to_group = weighted_cols + ['wind_speed_nwp_w','weight']
        agg = (
            nwp.groupby(['timestamp','forecasttime'], sort=False)[
                cols_to_group
            ].sum()
            .reset_index()
        )
        wsum = agg['weight']
        agg[weighted_cols] = (
            agg[weighted_cols].div(wsum, axis=0)
        )
        agg['wind_speed_nwp_w'] = (agg['wind_speed_nwp_w']/wsum)**(1/3)
        nwp = (
            agg.drop(columns='weight')
            .rename(columns=lambda c: c.replace('_w',''))
            .set_index('timestamp')
        )
    elif config['params']['aggregate_nwp_layers'] == 'mean':
        logging.warning('Mean aggregation of NWP layers not implemented yet')
        #nwp = nwp.groupby(['level']).mean()
        #nwp.drop(['level'], axis=1, inplace=True)
    elif config['params']['aggregate_nwp_layers'] == 'not':
        rel_features += ['level']
    else:
        raise ValueError(f"Unknown option for 'aggregate_nwp_layers': {config['params']['aggregate_nwp_layers']}. Please choose from 'to_pivot', 'weighted_mean', 'mean', or 'not'.")
    # merge nwp data and wind park data
    df = nwp.merge(df, left_index=True, right_index=True, how='left')
    rel_features.append('forecasttime')
    df = df[rel_features]
    df.reset_index(inplace=True)
    df['starttime'] = df['timestamp'] - pd.to_timedelta(df['forecasttime'], unit='h')
    df.sort_values(['starttime', 'forecasttime'], ascending=True, inplace=True)
    # drop all rows with forecasttime = 0
    df.drop(df[df.forecasttime == 0].index, inplace=True)
    df.set_index(['starttime', 'forecasttime', 'timestamp'], inplace=True)
    #df.drop('timestamp', axis=1, inplace=True)
    # drop forecast runs with < 48 forecasted hours
    df.dropna(inplace=True)
    df = df.groupby(level='starttime').filter(
        lambda g: g.index.get_level_values('forecasttime').nunique() == 48
    )
    return df

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
    random.seed(config['params']['random_seed'])
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
            logging.debug(f"Looking for turbine column: {turbine_col}")
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
                        logging.debug(f"Column {model_specific_col} not found in NWP data for model {model}")
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
                         test_start: pd.Timestamp = None,
                         test_end: pd.Timestamp = None,
                         t_0: int = 0,
                         scale_target: bool = False): # New flag to control lag feature
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
    logging.debug(f"Known future columns for TFT: {known_future_cols}")
    logging.debug(f"Observed past columns for TFT: {observed_past_cols}")
    # add target to observed_past_cols if not already included
    #if target_col not in observed_past_cols:
    #    observed_past_cols.append(target_col)
    df = data.copy()
    # --- Data Splitting ---
    if test_start:
        train_end = test_start - pd.Timedelta(hours=0.25)
    else:
        train_periods = int(len(df) * train_frac)
        train_end = df.index[train_periods].normalize() + pd.Timedelta(hours=(t_0-0.25))
        test_start = train_end + pd.Timedelta(hours=0.25)

    train_end = pd.Timestamp(train_end, tzinfo=data.index.tz)
    test_start = pd.Timestamp(test_start, tzinfo=data.index.tz)
    test_end = pd.Timestamp(test_end, tzinfo=data.index.tz)
    train_df = df[:train_end]
    test_df = df[test_start:test_end]

    #logging.info(f"Training data range: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
    #logging.info(f"Test data range:     {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")
    scalers = {}
    # Static
    X_static_train = get_static_features(data=train_df,
                                         static_cols=static_cols)
    X_static_test = get_static_features(data=test_df,
                                        static_cols=static_cols)
    # Scale Known Future Features
    known_train_data, known_test_data = None, None
    if known_future_cols:
        known_train_data, known_test_data, scalers['x_known'] = apply_scaling(train_df[known_future_cols].values,
                                                                            test_df[known_future_cols].values,
                                                                            StandardScaler)
    # Scale Observed Past Features (now includes lagged target if enabled)
    observed_train_data, observed_test_data = None, None
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

    logging.debug("Shapes of generated arrays (Train):")
    # Expected shapes:
    # X_static: (num_samples, n_static_features)
    # X_known:  (num_samples, history_len + future_horizon, n_known_features)
    # X_observed:(num_samples, history_len, n_observed_input_features)
    # y:        (num_samples, future_horizon)
    logging.debug(f"X_static_train:   {X_static_train.shape}")
    logging.debug(f"X_known_train:    {X_known_train.shape}")
    logging.debug(f"X_observed_train: {X_observed_train.shape}")
    logging.debug(f"y_train:          {y_train.shape}")

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
    new_known_cols = [
        col for col in cols
        if any(known in col for known in known_future_cols)
    ]
    new_observed_cols = [
        col for col in cols
        if any(observed in col for observed in observed_past_cols)
    ]
    for col in cols:
        if 'wind_speed_rotor_eq' in col and col not in new_known_cols:
            new_known_cols.append(col)
        if 'wind_speed_mean_height' in col and col not in new_known_cols:
            new_known_cols.append(col)
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
        """Creates sequences for TFT inputs and outputs."""
        X_known_list, X_observed_list, y_list, index_list = [], [], [], []
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
            target_window = target_data[i + history_len : i + total_len].flatten() # Ensure shape (future_horizon,)
            timestamp = indices[i + history_len]
            y_list.append(target_window)
            index_list.append(timestamp)
        return (np.array(X_known_list),
                np.array(X_observed_list),
                np.array(y_list),
                np.array(index_list))