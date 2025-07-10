import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import logging
from typing import List, Tuple, Dict, Any

from . import meteo

def pipeline(data: pd.DataFrame,
             config: Dict[str, Any],
             known_cols: List[str] = None,
             observed_cols: List[str] = None,
             static_cols: List[str] = None,
             df_static: pd.DataFrame = pd.DataFrame(),
             test_start: pd.Timestamp = None) -> Tuple[Dict, pd.DataFrame]:
    df = data.copy()
    df = knn_imputer(data=df, n_neighbors=config['data']['n_neighbors'])
    t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
    if config['model']['name'] == 'tft':
        df = lag_features(data=df,
                          lookback=config['model']['lookback'],
                          horizon=config['model']['horizon'],
                          lag_in_col=config['data']['lag_in_col'])
        prepared_data = prepare_data_for_tft(data=df,
                                             static_data=df_static,
                                             history_length=config['model']['lookback'], # e.g., 72 (for 3 days history)
                                             future_horizon=config['model']['horizon'], # e.g., 24 (for 24 hours forecast)
                                             known_future_cols=known_cols,
                                             observed_past_cols=observed_cols,
                                             train_frac=config['data']['train_frac'],
                                             test_start=pd.Timestamp(config['data']['test_start']),
                                             t_0=t_0)
    else:
        for col in observed_cols:
            df = lag_features(data=df,
                            lookback=config['model']['lookback'],
                            horizon=config['model']['horizon'],
                            lag_in_col=config['data']['lag_in_col'],
                            target_col=col)
            if col != 'power': df.drop(col, axis=1, inplace=True)
        prepared_data = prepare_data(data=df,
                                     output_dim=config['model']['output_dim'],
                                     train_frac=config['data']['train_frac'],
                                     scale_y=config['data']['scale_y'],
                                     t_0=t_0,
                                     test_start=test_start)
    return prepared_data, df


def get_data(data_dir: str,
             dataset_name: str,
             freq: str,
             rel_features: list = None) -> Dict[str, pd.DataFrame]:
    if dataset_name == '1b_trina':
        file_name = '1B_trina.csv'
        path = os.path.join(data_dir, file_name)
        df = preprocess_1b_trina(path=path,
                                 freq=freq)
        return {file_name: df}
    elif dataset_name == 'pvod':
        target_dir = 'PVODdatasets_v1'
        files = os.listdir(os.path.join(data_dir, target_dir))
        files = [f for f in files if f.endswith('.csv') and 'metadata' not in f]
        files = sorted(files)
        dfs = {}
        for file in files:
            path = os.path.join(data_dir, target_dir, file)
            df = preprocess_pvod(path=path,
                                 freq=freq,
                                 rel_features=rel_features)
            dfs[file] = df
        return dfs
    elif 'synth_pv' in dataset_name:
        target_dir = os.path.join(data_dir, dataset_name)
        logging.info(f'Getting the data from: {target_dir}')
        client_dirs = [name for name in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, name))]
        dfs = defaultdict(dict)
        if not client_dirs:
            client = dataset_name.split('/')[1]
            client_files = os.listdir(target_dir)
            client_files = [f for f in client_files if f.endswith('.csv') and 'pvpark' in f]
            for file in client_files:
                path = os.path.join(target_dir, file)
                df = preprocess_synth_pv(path=path,
                                         freq=freq)
                dfs[file] = df
        else:
            for client in client_dirs:
                client_files = os.listdir(os.path.join(target_dir, client))
                client_files = [f for f in client_files if f.endswith('.csv') and 'pvpark' in f]
                for file in client_files:
                    path = os.path.join(target_dir, client, file)
                    df = preprocess_synth_pv(path=path,
                                            freq=freq)
                    dfs[client][file] = df
        return dfs
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}. Please check the dataset name or add a new preprocessing function.')

def knn_imputer(data: pd.DataFrame,
               n_neighbors: int = 5):
    # To help KNNImputer estimating the temporal saisonalities we add encoded temporal features.
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
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
    if lag_in_col:
        for lag in range(horizon, horizon+lookback+1):
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        return df
    lookback = max(horizon, lookback)
    lags = [horizon]
    lags.extend(range(horizon+24, lookback+1, 24))
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df

def make_windows(data: np.ndarray,
                 seq_len: int) -> np.ndarray:
    return np.array([data[i:i + seq_len] for i in range(data.shape[0]-seq_len+1)])

def apply_scaling(df_train,
                  df_test,
                  scaler_type=StandardScaler):
    """Fits scaler on train data and transforms train and test data."""
    scaler = scaler_type()
    if len(df_train.shape) == 1:
        df_train = df_train.values.reshape(-1, 1)
    if len(df_test.shape) == 1:
        df_test = df_test.values.reshape(-1, 1)
    # Important: Fit scaler ONLY on training data!
    scaled_train = scaler.fit_transform(df_train)
    if len(df_test) == 0:
        scaled_test = df_test
    else:
        scaled_test = scaler.transform(df_test) # Use fitted scaler to transform test data
    return scaled_train, scaled_test, scaler

def prepare_data(data: pd.DataFrame,
                 output_dim: int,
                 train_frac: float = 0.75,
                 scale_x: bool = True,
                 scale_y: bool = False,
                 target_col: str = 'power',
                 t_0: int = 0,
                 test_start: pd.Timestamp = None,
                 seq2seq: bool = False):
    df = data.copy()
    df.dropna(inplace=True)
    target = df[[target_col]]
    df.drop(target_col, axis=1, inplace=True)
    # Split data into train and test sets with a 75/25 ratio, ensuring full days
    if test_start:
        train_end = test_start - pd.Timedelta(hours=0.25)
    else:
        train_periods = int(len(df) * train_frac)
        train_end = df.index[train_periods].normalize() + pd.Timedelta(hours=(t_0-0.25))
        test_start = train_end + pd.Timedelta(hours=0.25)
    scalers = {}
    scalers['x'] = None
    scalers['y'] = None
    if scale_x:
        X_train, X_test, scaler_x = apply_scaling(df[:train_end],
                                                  df[test_start:],
                                                  StandardScaler)
        scalers['x'] = scaler_x
    else:
        X_train = df[:train_end].values
        X_test = df[test_start:].values
    if scale_y:
        Y_train, Y_test, scaler_y = apply_scaling(target[:train_end],
                                                  target[test_start:],
                                                  StandardScaler)
        scalers['y'] = scaler_y
    else:
        Y_train = target[:train_end]
        Y_test = target[test_start:]
    index_train = np.array([df[:train_end].index[i] for i in range(X_train.shape[0]-output_dim+1)])
    index_test = np.array([df[test_start:].index[i] for i in range(X_test.shape[0]-output_dim+1)])
    X_train = make_windows(data=X_train, seq_len=output_dim)
    X_test = make_windows(data=X_test, seq_len=output_dim)
    y_train = make_windows(data=Y_train, seq_len=output_dim).reshape(-1, output_dim)
    y_test = make_windows(data=Y_test, seq_len=output_dim).reshape(-1, output_dim)
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
                        rel_features: list = None) -> pd.DataFrame:
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
    if rel_features:
        return df[rel_features]
    return df

def preprocess_pvod(path: str,
                    freq=None,
                    rel_features=None) -> pd.DataFrame:
    timestamp_col = 'date_time'
    target_col = 'power'
    station_id = path.split('/')[-1].split('.')[0]
    df = pd.read_csv(path, delimiter=',')
    metadata_file = path.replace(os.path.basename(path).split('.')[0], 'metadata')
    metadata = pd.read_csv(metadata_file)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    df['azimuth'] = 180
    df['tilt'] = int(metadata.loc[metadata.Station_ID == station_id]['Array_Tilt'].str[-3:-1].iloc[0])
    df['latitude'] = float(metadata.loc[metadata.Station_ID == station_id]['Latitude'].iloc[0])
    df['longitude'] = float(metadata.loc[metadata.Station_ID == station_id]['Longitude'].iloc[0])
    df['nwp_gti'] = meteo.get_total_irradiance(ghi=df['nwp_globalirrad'],
                                            pressure=df['nwp_pressure'],
                                            temperature=df['nwp_temperature'],
                                            latitude=df['latitude'],
                                            longitude=df['longitude'],
                                            surface_tilt=df['tilt'],
                                            surface_azimuth=df['azimuth'],
                                            time_index=df.index,
                                            dni=df['nwp_directirrad'])
    df['lmd_gti'] = meteo.get_total_irradiance(ghi=df['lmd_totalirrad'],
                                            pressure=df['lmd_pressure'],
                                            temperature=df['lmd_temperature'],
                                            latitude=df['latitude'],
                                            longitude=df['longitude'],
                                            surface_tilt=df['tilt'],
                                            surface_azimuth=df['azimuth'],
                                            time_index=df.index,
                                            dhi=df['lmd_diffuseirrad'])
    df['lmd_gti'] = df['lmd_gti'].fillna(0)
    # normalize time series by installed capacity
    df[target_col] = df[target_col] / (metadata.loc[metadata.Station_ID == station_id]['Capacity'].values[0] / 1000)
    df.rename(columns={target_col: 'power'}, inplace=True)
    if freq:
        df = df.resample(freq).mean().copy()
    if rel_features:
        return df[rel_features]
    return df

def preprocess_1b_trina(path: str,
                        freq: str) -> pd.DataFrame:
    rel_features = ['Global_Horizontal_Radiation',
                    'Diffuse_Horizontal_Radiation',
                    'Weather_Temperature_Celsius',
                    'Weather_Relative_Humidity',
                    'Active_Power']
    timestamp_col = 'timestamp'
    target_col = rel_features[-1]
    df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    df = df.resample(freq).mean().copy()
    df.rename(columns={target_col: 'power'}, inplace=True)
    #if len(pd.date_range(df.index[0], df.index[-1], freq=freq)) == len(df):
    #    print('Data has no missing timesteps in index.')
    #else:
    #    print('Data has missing timestamps in index.')
    #df = df.groupby(df.index.time).ffill()
    df = df['2014-01-01':'2018-12-31'].copy() # 1B Trina specific
    if rel_features:
        return df[rel_features]
    return df


def get_features(dataset_name: str) -> Tuple[List, List]:
    known, observed, static = None, None, None
    if 'synth_pv' in dataset_name:
        known = ['global_horizontal_irradiance',
                 'diffuse_horizontal_irradiance',
                 #'air_temperature',
                 #'wind_speed',
                 'direct_normal_irradiance']
        observed = ['power']
        static = []
    elif dataset_name == 'pvod':
        known = [
                #'nwp_globalirrad',
                #'nwp_directirrad',
                'nwp_gti',
                'nwp_temperature',
                'nwp_humidity',
                'nwp_windspeed']
        observed = [
                    #'lmd_totalirrad',
                    #'lmd_diffuseirrad',
                    'lmd_gti',
                    'lmd_temperature',
                    #'lmd_pressure',
                    #'lmd_winddirection',
                    'lmd_windspeed',
                    'power']
        static = [
                'Station_ID'
                'Array_Tilt']
    return known, observed, static


def prepare_data_for_tft(data: pd.DataFrame,
                         static_data: pd.DataFrame,
                         history_length: int, # e.g., 72 (for 3 days history)
                         future_horizon: int, # e.g., 24 (for 24 hours forecast)
                         known_future_cols: list,
                         observed_past_cols: list,
                         train_frac: float = 0.75,
                         target_col: str = 'power',
                         test_start: pd.Timestamp = None,
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
    # add target to observed_past_cols if not already included
    if target_col not in observed_past_cols:
        observed_past_cols.append(target_col)
    df = data.copy()
    # --- Data Splitting ---
    if test_start:
        train_end = test_start - pd.Timedelta(hours=0.25)
    else:
        train_periods = int(len(df) * train_frac)
        train_end = df.index[train_periods].normalize() + pd.Timedelta(hours=(t_0-0.25))
        test_start = train_end + pd.Timedelta(hours=0.25)

    train_df = df[:train_end]
    test_df = df[test_start:]

    print(f"Training data range: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
    print(f"Test data range:     {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")
    scalers = {}
    # Static
    X_static_train = get_static_features(data=data,
                                         static_data=static_data)
    X_static_test = get_static_features(data=data,
                                        static_data=static_data)
    # Scale Known Future Features
    known_train_data, known_test_data, scalers['x_known'] = apply_scaling(train_df[known_future_cols].values,
                                                                          test_df[known_future_cols].values,
                                                                          StandardScaler)
    # Scale Observed Past Features (now includes lagged target if enabled)
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
        future_horizon
    )
    X_known_test, X_observed_test, y_test, test_indices = create_tft_sequences(
        known_test_data,
        observed_test_data,
        target_test_scaled, # Pass the separately handled target data
        test_df.index,
        history_length,
        future_horizon
    )
    print("\nShapes of generated arrays (Train):")
    # Expected shapes:
    # X_static: (num_samples, n_static_features)
    # X_known:  (num_samples, history_len + future_horizon, n_known_features)
    # X_observed:(num_samples, history_len, n_observed_input_features)
    # y:        (num_samples, future_horizon)
    print(f"X_static_train:   {X_static_train.shape}")
    print(f"X_known_train:    {X_known_train.shape}")
    print(f"X_observed_train: {X_observed_train.shape}")
    print(f"y_train:          {y_train.shape}")

    # --- Return Results ---
    results = {}
    results['y_train'], results['y_test'] = y_train, y_test

    results['scalers'] = scalers
    results['index_train'], results['index_test'] = train_indices, test_indices

    X_train: Dict[str, np.ndarray] = {
            'observed_input': X_observed_train,
            'known_input': X_known_train,
            'static_input': X_static_train, # Hinzugefügt (kann None/leer sein)
    }
    X_test: Dict[str, np.ndarray] = {
        'observed_input': X_observed_test,
        'known_input': X_known_test,
        'static_input': X_static_test # Static Testdaten hinzufügen
    }
    results['X_train'], results['X_test'] = X_train, X_test
    static_is_zero = X_train['static_input'].shape[-1] == 0
    if static_is_zero:
        del X_train['static_input']
        del X_test['static_input']
    return results

def get_static_features(data: pd.DataFrame,
                        static_data: pd.DataFrame):
    if static_data.empty:
        print('No static data provided.')
        return np.array([])
    id = data[data.columns[0]].iloc[0] # Assuming the first column is the ID
    # Extract static features for the given ID
    static_features = static_data.loc[id]
    num_static_features = len(static_features.columns)
    static_features = static_features.values.reshape(num_static_features,)
    return static_features

def create_tft_sequences(known_data, observed_data, target_data,
                         indices, history_len, future_len):
        """Creates sequences for TFT inputs and outputs."""
        X_known_list, X_observed_list, y_list, index_list = [], [], [], []
        # Total window length needed for one sample
        total_len = history_len + future_len
        # Iterate through possible start points for the history window
        # End point ensures there's enough data for history AND future horizon
        for i in range(len(observed_data) - total_len + 1):
            # Observed past window: indices [i, i + history_len - 1]
            observed_past_window = observed_data[i : i + history_len]
            # Known future window: indices [i, i + history_len + future_horizon - 1]
            known_future_window = known_data[i : i + total_len]
            # Static features: Take the value(s) at the end of the history period
            # Shape should be (num_static_features,)
            # Target (y) window: indices [i + history_len, i + total_len - 1]
            # Extracted from the separate (potentially scaled) target data
            target_window = target_data[i + history_len : i + total_len].flatten() # Ensure shape (future_horizon,)
            # Timestamp: Corresponds to the first prediction step
            timestamp = indices[i + history_len]
            # Append the created windows/values to the lists
            X_known_list.append(known_future_window)
            X_observed_list.append(observed_past_window)
            y_list.append(target_window)
            index_list.append(timestamp)
        # Convert lists to numpy arrays
        return (np.array(X_known_list),
                np.array(X_observed_list),
                np.array(y_list),
                np.array(index_list))