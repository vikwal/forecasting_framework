import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import impute

# generic functions (dataset independent)

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
                 target_col: str,
                 lag_dim: int,
                 horizon: int,
                 lag_in_col=False) -> pd.DataFrame:
    if lag_in_col:
        for lag in range(horizon, horizon+lag_dim+1):
            df[f'lag_{lag}h'] = df[target_col].shift(lag)
        return df
    lag_dim = max(horizon, lag_dim)
    lags = [horizon]
    lags.extend(range(horizon+24, lag_dim+1, 24))
    for lag in lags:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)
    return df

def make_windows(data: np.ndarray,
                 output_dim: int) -> np.ndarray:
    return np.array([data[i:i + output_dim] for i in range(data.shape[0]-output_dim+1)])

def prepare_data(data: pd.DataFrame, 
                 train_end: str, 
                 test_start: str, 
                 output_dim: int, 
                 target_col: str,
                 scale_x=True,
                 scale_y=False,
                 return_index=False):
    df = data.copy()
    df.dropna(inplace=True)
    target = df[[target_col]]
    df.drop(target_col, axis=1, inplace=True)
    rawX_train = df[:train_end].values
    rawX_test = df[test_start:].values
    rawY_train = target[:train_end]
    rawY_test = target[test_start:]
    if scale_x:
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(rawX_train)
        X_test = scaler_x.transform(rawX_test)
    else:
        X_train = rawX_train
        X_test = rawX_test
    if scale_y:
        scaler_y = StandardScaler()
        Y_train = scaler_y.fit_transform(rawY_train)
        Y_test = scaler_y.transform(rawY_test)
    else:
        Y_train = rawY_train
        Y_test = rawY_test
    index_train = np.array([df[:train_end].index[i] for i in range(X_train.shape[0]-output_dim+1)])
    index_test = np.array([df[test_start:].index[i] for i in range(X_test.shape[0]-output_dim+1)])
    X_train = make_windows(data=X_train, output_dim=output_dim)
    X_test = make_windows(data=X_test, output_dim=output_dim)
    y_train = make_windows(data=Y_train, output_dim=output_dim).reshape(-1, output_dim)
    y_test = make_windows(data=Y_test, output_dim=output_dim).reshape(-1, output_dim)
    if return_index & scale_y:
        return X_train, y_train, X_test, y_test, index_train, index_test, scaler_y
    elif return_index & (not scale_y):
        return X_train, y_train, X_test, y_test, index_train, index_test
    elif (not return_index) & scale_y:
        return X_train, y_train, X_test, y_test, scaler_y
    return X_train, y_train, X_test, y_test  # [ X_train.shape = (batch_size, n_features, output_dim) ]

# dataset specific functions

def preprocess_1b_trina(path: str, 
                        timestamp_col: str, 
                        freq: str,
                        rel_features=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(['Active_Energy_Delivered_Received', 'Current_Phase_Average'], axis=1, inplace=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    df = df.resample(freq).mean().copy()
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
