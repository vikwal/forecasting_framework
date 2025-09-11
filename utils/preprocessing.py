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
             test_start: pd.Timestamp = None,
             target_col: str = 'power') -> Tuple[Dict, pd.DataFrame]:
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
        if config['model']['create_lag']:
            for col in observed_cols:
                df = lag_features(data=df,
                                lookback=config['model']['lookback'],
                                horizon=config['model']['horizon'],
                                lag_in_col=config['data']['lag_in_col'],
                                target_col=col)
                if col != target_col and col not in known_cols: df.drop(col, axis=1, inplace=True, errors='ignore')
        prepared_data = prepare_data(data=df,
                                     output_dim=config['model']['output_dim'],
                                     step_size=config['model']['step_size'],
                                     train_frac=config['data']['train_frac'],
                                     scale_y=config['data']['scale_y'],
                                     t_0=t_0,
                                     test_start=test_start,
                                     target_col=target_col)
    return prepared_data, df


def get_data(data_dir: str,
             dataset_name: str,
             freq: str,
             config: dict,
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
    elif 'reninja_n_dwd' in dataset_name:
        files = os.listdir(os.path.join(data_dir, dataset_name))
        files = [f for f in files if f.endswith('.csv') and 'metadata' not in f]
        files = sorted(files)
        dfs = {}
        for file in files:
            path = os.path.join(data_dir, dataset_name, file)
            df = preprocess_reninja_n_dwd(path=path,
                                          rel_features=rel_features)
            dfs[file] = df
        return dfs
    elif 'synth_wind' in dataset_name:
        rel_features_copy = rel_features.copy()
        target_dir = os.path.join(data_dir, dataset_name)
        logging.info(f'Getting the data from: {target_dir}')
        client_dirs = [name for name in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, name))]
        dfs = defaultdict(dict)
        if not client_dirs:
            client = dataset_name.split('/')[1]
            client_files = os.listdir(target_dir)
            client_files = [f for f in client_files if f.endswith('.csv') and 'synth' in f]
            for file in client_files:
                path = os.path.join(target_dir, file)
                df = preprocess_synth_wind(path=path,
                                           config=config,
                                           freq=freq,
                                           rel_features=rel_features_copy)
                key = file
                dfs[key] = df
        else:
            for client in client_dirs:
                client_files = os.listdir(os.path.join(target_dir, client))
                client_files = [f for f in client_files if f.endswith('.csv') and 'synth' in f]
                for file in client_files:
                    path = os.path.join(target_dir, client, file)
                    df = preprocess_synth_wind(path=path,
                                               config=config,
                                               freq=freq,
                                               rel_features=rel_features_copy)
                    key = f'{client}_{file}'
                    dfs[key] = df
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
                key = file
                dfs[key] = df
        else:
            for client in client_dirs:
                client_files = os.listdir(os.path.join(target_dir, client))
                client_files = [f for f in client_files if f.endswith('.csv') and 'pvpark' in f]
                for file in client_files:
                    path = os.path.join(target_dir, client, file)
                    df = preprocess_synth_pv(path=path,
                                            freq=freq)
                    key = f'{client}_{file}'
                    dfs[key] = df
        return dfs
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}. Please check the dataset name or add a new preprocessing function.')

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
        shifted_index = index - freq
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
        df_test = df[test_start:]
        target_train = target[:train_end]
        target_test = target[test_start:]
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

def preprocess_synth_wind(path: str,
                          config: dict,
                          freq: str = '1H',
                          rel_features: list = None) -> pd.DataFrame:
    def circle_cap_area(r: float, y: float) -> float:
        """Fläche oberhalb einer Horizontalen im Kreis mit Radius r.
        Kreis ist im Ursprung (0,0) zentriert. y ist die Höhe der Linie."""
        if y <= -r:   # Linie ganz unten → ganze Kreisfläche
            return np.pi * r**2
        if y >= r:    # Linie ganz oben → keine Fläche
            return 0.0
        return r**2 * np.arccos(y/r) - y * np.sqrt(r**2 - y**2)

    def get_A_weights(turbines: pd.DataFrame, nwp: pd.DataFrame) -> np.ndarray:
        A_total = 0
        A_all_turbines = []
        for turbine in turbines.itertuples():
            r = turbine.diameter / 2
            height = turbine.hub_height
            rotor_top = height + r
            rotor_bottom = height - r
            A_total += np.pi * r**2
            A = []
            for top, down in nwp.groupby(['toplevel','bottomlevel']).size().index:
                y_top = top - height
                y_down = down - height
                A_slice = circle_cap_area(r, y_down) - circle_cap_area(r, y_top)
                A.append(A_slice)
            A_all_turbines.append(A)
        A_all_turbines = np.array(A_all_turbines)
        return A_all_turbines.sum(axis=0) / A_total

    def get_saturated_vapor_pressure(temperature: pd.Series,
                                     model: str = 'improved_magnus') -> pd.Series:
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
            relhum = relhum / 100
        if temperature.mean() < 100:
            temperature = temperature + 273.15 # from celsius to kelvin
        p_w = relhum * sat_vap_ps
        p_g = pressure - p_w
        rho_g = p_g / (R_dry * temperature)
        rho_w = p_w / (R_w * temperature)
        rho = rho_g + rho_w
        return rho

    timestamp_col = 'timestamp'
    park_id = path.split('_')[-1].split('.')[0]
    wind_parameter_path = path.replace(os.path.basename(path), 'wind_parameter.csv')
    turbine_parameter_path = path.replace(os.path.basename(path), 'turbine_parameter.csv')
    power_curves_path = os.path.join(config['data']['power_curves_path'], 'turbine_power.csv')
    nwp_path = os.path.join(config['data']['nwp_path'], f'ML_Station_{park_id}.csv')
    rel_features = list(set(rel_features))
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
        mapping = {20.0: 1,
                   55.212: 2,
                   100.277: 3,
                   153.438: 4,
                   213.746: 5,
                   280.598: 6}
        #features_not_to_pivot = [col in nwp.columns if '_nwp' in col]
        features_to_pivot = config['params']['features_to_pivot']
        for col in features_to_pivot:
            rel_features.remove(col)
        features_not_to_pivot = [col for col in nwp.columns if col not in features_to_pivot]
        weights = get_A_weights(turbines=turbines, nwp=nwp)
        # pivot features
        nwp_pivot = nwp[features_to_pivot + ['forecasttime', 'toplevel']].copy()
        nwp_pivot.reset_index(inplace=True)
        nwp_pivot['toplevel'] = nwp_pivot['toplevel'].map(mapping)
        nwp_pivot = nwp_pivot.pivot(index=['timestamp', 'forecasttime'],
                                    columns=['toplevel'],
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
        for top, down in nwp.groupby(['toplevel', 'bottomlevel']).size().index:
            if down <= np.mean(heights) <= top:
                nwp = nwp[(nwp['toplevel'] == top) & (nwp['bottomlevel'] == down)]
                break
        nwp.drop(['toplevel', 'bottomlevel'], axis=1, inplace=True)
    elif config['params']['aggregate_nwp_layers'] == 'not':
        rel_features += ['toplevel', 'bottomlevel']
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

def preprocess_reninja_n_dwd(path: str,
                             rel_features: list = None) -> pd.DataFrame:
    timestamp_col = 'timestamp'
    target_col = 'wind_speed'
    df = pd.read_csv(path, sep=';', decimal='.')
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df.set_index(timestamp_col, inplace=True)
    #df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    #df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    #df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    #df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
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


def get_features(dataset_name: str,
                 config: dict = None) -> Tuple[List, List]:
    if 'known_features' in config['params'] and 'observed_features' in config['params']:
        known = config['params']['known_features']
        observed = config['params']['observed_features']
        static = config['params'].get('static_features', [])
        return known, observed, static
    else:
        known, observed, static = None, None, None
    if 'synth_wind' in dataset_name:
        known = ['wind_speed_nwp', 'temperature_nwp', 'pressure_nwp', 'relhum_nwp']
        observed = ['power']
        static = []
    elif 'synth_pv' in dataset_name:
        known = ['global_horizontal_irradiance',
                 'diffuse_horizontal_irradiance',
                 #'air_temperature',
                 #'wind_speed',
                 'direct_normal_irradiance']
        observed = ['power']
        static = []
    elif dataset_name == 'reninja_n_dwd':
        known = ['wind_speed_nwp',
                 'temperature_2m',
                 'density',
                 'relative_humidity',
                 'std_v_wind',
                 'pressure',
                 'wind_direction',
                 'w_vert',
                 'saturated_vapor_pressure'
                 ]
        observed = ['wind_speed']
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