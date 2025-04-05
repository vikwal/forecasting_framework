import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

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
                 seq_len: int) -> np.ndarray:
    return np.array([data[i:i + seq_len] for i in range(data.shape[0]-seq_len+1)])

def apply_scaling(df_train,
                  df_test,
                  cols=None,
                  scaler_type=StandardScaler):
    """Fits scaler on train data and transforms train and test data."""
    scaler = scaler_type()
    if cols is None:
        cols = df_train.columns.tolist()
    # Important: Fit scaler ONLY on training data!
    scaled_train = scaler.fit_transform(df_train[cols])
    scaled_test = scaler.transform(df_test[cols]) # Use fitted scaler to transform test data
    return scaled_train, scaled_test, scaler

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
    X_train = make_windows(data=X_train, seq_len=output_dim)
    X_test = make_windows(data=X_test, seq_len=output_dim)
    y_train = make_windows(data=Y_train, seq_len=output_dim).reshape(-1, output_dim)
    y_test = make_windows(data=Y_test, seq_len=output_dim).reshape(-1, output_dim)
    if return_index & scale_y:
        return X_train, y_train, X_test, y_test, index_train, index_test, scaler_y
    elif return_index & (not scale_y):
        return X_train, y_train, X_test, y_test, index_train, index_test
    elif (not return_index) & scale_y:
        return X_train, y_train, X_test, y_test, scaler_y
    return X_train, y_train, X_test, y_test  # [ X_train.shape = (batch_size, n_features, output_dim) ]

def prepare_data_tft(data: pd.DataFrame,
                    static_data: pd.DataFrame,
                    target_col: str,
                    history_length: int, # e.g., 72 (for 3 days history)
                    future_horizon: int, # e.g., 24 (for 24 hours forecast)
                    known_future_cols: list,
                    observed_past_cols: list, # Should NOT contain target_col anymore
                    train_end_idx, # Can be a date string or int index
                    test_start_idx, # Can be a date string or int index
                    scale_target=True, # Scaler for the actual target variable (y)
                    return_index=True): # New flag to control lag feature
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
                                    **Should typically NOT contain target_col if
                                    add_lagged_target_input=True**.
        train_end_idx: Index or date where training data ends.
        test_start_idx: Index or date where test data begins.
        scale_known_future (bool): Whether to scale known future features.
        scale_observed_past (bool): Whether to scale observed past features (incl. lagged target).
        scale_target (bool): Whether to scale the actual target variable (y).
        scale_static (bool): Whether to scale static features.
        return_index (bool): Whether to return timestamps corresponding to the forecast start.
    Returns:
        tuple: Contains the prepared data arrays for training and testing,
               optionally scalers and indices.
               Format: (X_static_train, X_known_train, X_observed_train, y_train,
                        X_static_test, X_known_test, X_observed_test, y_test,
                        ...) plus optionally scaler_dict, train_indices, test_indices
    """
    df = data.copy()
    # --- Data Splitting ---
    try:
        train_df = df.loc[:train_end_idx]
        test_df = df.loc[test_start_idx:]
    except KeyError:
        # Handle integer-based indexing if loc fails
        train_end_pos = df.index.get_loc(train_end_idx)
        test_start_pos = df.index.get_loc(test_start_idx)
        train_df = df.iloc[:train_end_pos + 1]
        test_df = df.iloc[test_start_pos:]

    print(f"Training data range: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
    print(f"Test data range:     {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")
    scalers = {}
    # Static
    X_static_train = get_static_features(data=data,
                                         static_data=static_data)
    X_static_test = get_static_features(data=data,
                                        static_data=static_data)
    # Scale Known Future Features
    known_train_data, known_test_data, scalers['known_future'] = apply_scaling(
        train_df, test_df, known_future_cols, StandardScaler
    )
    # Scale Observed Past Features (now includes lagged target if enabled)
    observed_train_data, observed_test_data, scalers['observed_past'] = apply_scaling(
        train_df, test_df, observed_past_cols, StandardScaler
    )
    # Scale Target Variable (y) Separately
    target_train_raw = train_df[[target_col]].values
    target_test_raw = test_df[[target_col]].values
    if scale_target:
        target_scaler = StandardScaler()
        target_train_scaled = target_scaler.fit_transform(target_train_raw)
        target_test_scaled = target_scaler.transform(target_test_raw)
        scalers['target'] = target_scaler
        print(f"Target variable '{target_col}' scaled separately.")
    else:
        target_train_scaled = target_train_raw
        target_test_scaled = target_test_raw
        scalers['target'] = None

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
    result = [X_static_train, X_known_train, X_observed_train, y_train,
              X_static_test, X_known_test, X_observed_test, y_test]

    # Append scalers (important for inverse transforming predictions)
    result.append(scalers) # Always return scalers dict
    if return_index:
        result.append(train_indices)
        result.append(test_indices)
    return tuple(result)

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
