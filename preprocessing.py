import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import List

# generic functions (dataset independent)

def get_data(data: str,
             data_dir: str,
             freq: str) -> List[pd.DataFrame]:
    if data == '1b_trina':
        file_name = '1B_trina.csv'
        path = os.path.join(data_dir, file_name)
        df = preprocess_1b_trina(path=path,
                                 freq=freq)
        return {file_name: df}
    elif data == 'pvod':
        dir_name = 'PVODdatasets_v1'
        files = os.listdir(os.path.join(data_dir, dir_name))
        files = [f for f in files if f.endswith('.csv') and 'metadata' not in f]
        files = sorted(files)
        dfs = {}
        for file in files:
            path = os.path.join(data_dir, dir_name, file)
            df = preprocess_pvod(path=path,
                                 freq=freq)
            dfs[file] = df
        return dfs

def impute(data: pd.DataFrame,
           imputer: str) -> pd.DataFrame:
    df = data
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

def lag_features(df: pd.DataFrame,
                 lag_dim: int,
                 horizon: int,
                 target_col='power',
                 lag_in_col=False) -> pd.DataFrame:
    if lag_in_col:
        for lag in range(horizon, horizon+lag_dim+1):
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        return df
    lag_dim = max(horizon, lag_dim)
    lags = [horizon]
    lags.extend(range(horizon+24, lag_dim+1, 24))
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    df.dropna(inplace=True)
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
    scaled_test = scaler.transform(df_test) # Use fitted scaler to transform test data
    return scaled_train, scaled_test, scaler

def prepare_data(data: pd.DataFrame,
                 output_dim: int,
                 train_frac=0.75,
                 scale_x=True,
                 scale_y=False,
                 target_col='power'):
    df = data.copy()
    df.dropna(inplace=True)
    target = df[[target_col]]
    df.drop(target_col, axis=1, inplace=True)
    # Split data into train and test sets with a 75/25 ratio, ensuring full days
    train_periods = int(len(df) * train_frac)
    train_end = df.index[train_periods - 1].normalize()
    test_start = train_end + pd.Timedelta(days=1)
    scaler = [None, None]
    if scale_x:
        X_train, X_test, scaler_x = apply_scaling(df[:train_end],
                                                  df[test_start:],
                                                  StandardScaler)
        scaler[0] = scaler_x
    else:
        X_train = df[:train_end].values
        X_test = df[test_start:].values
    if scale_y:
        Y_train, Y_test, scaler_y = apply_scaling(target[:train_end],
                                                  target[test_start:],
                                                  StandardScaler)
        scaler[1] = scaler_y
    else:

        Y_train = target[:train_end]
        Y_test = target[test_start:]
    index_train = np.array([df[:train_end].index[i] for i in range(X_train.shape[0]-output_dim+1)])
    index_test = np.array([df[test_start:].index[i] for i in range(X_test.shape[0]-output_dim+1)])
    X_train = make_windows(data=X_train, seq_len=output_dim)
    X_test = make_windows(data=X_test, seq_len=output_dim)
    y_train = make_windows(data=Y_train, seq_len=output_dim).reshape(-1, output_dim)
    y_test = make_windows(data=Y_test, seq_len=output_dim).reshape(-1, output_dim)
    windows = {}
    windows['X_train'] = X_train
    windows['y_train'] = y_train
    windows['X_test'] = X_test
    windows['y_test'] = y_test
    windows['scaler'] = scaler
    windows['index_train'] = index_train
    windows['index_test'] = index_test
    return windows

def preprocess_pvod(path: str,
                    freq=None) -> pd.DataFrame:
    rel_features = ['nwp_globalirrad',
                    'nwp_directirrad',
                    'nwp_temperature',
                    'nwp_humidity',
                    'nwp_windspeed',
                    'power']
    timestamp_col = 'date_time'
    target_col = rel_features[-1]
    station_id = path.split('/')[-1].split('.')[0]
    df = pd.read_csv(path, delimiter=',')
    metadata_file = path.replace(os.path.basename(path).split('.')[0], 'metadata')
    metadata = pd.read_csv(metadata_file)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df[target_col] = df[target_col] / (metadata.loc[metadata.Station_ID == station_id]['Capacity'].values[0] / 1000)
    df.rename(columns={target_col: 'power'}, inplace=True)
    df.set_index(timestamp_col, inplace=True)
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

def germansolarfarm(path: str,
                    timestamp_col: str,
                    rel_features=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    if rel_features:
        return df[rel_features]
    return df

def europewindfarm(path: str,
                   timestamp_col: str,
                   rel_features=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop('ForecastingTime', axis=1, inplace=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    if rel_features:
        return df[rel_features]
    return df
