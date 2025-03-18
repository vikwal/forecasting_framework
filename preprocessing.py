import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def drop_days(frame, target_col, threshold=12, verbose=1):

    grouped_by_day = frame.groupby(pd.Grouper(freq='D'))
    days_to_drop = grouped_by_day[target_col].apply(lambda x: x.isnull().sum() > threshold)
    days_to_keep = days_to_drop[~days_to_drop].index

    n_dropped = len(days_to_drop) - len(days_to_keep)
    days_dropped = days_to_drop[days_to_drop == True].index.to_list()
    if verbose == 1:
        print('Number of dropped days:', n_dropped)
        print('Days dropped:')
        for day in days_dropped:
            print(day)

    frame = frame[frame.index.normalize().isin(days_to_keep)]

    return frame


def make_windows(df, train_end, test_start, output_dim, target_col='TARGET'):
    df.dropna(inplace=True)
    target = df[[target_col]]
    df.drop(target_col, axis=1, inplace=True)
    rawX_train = df[:train_end].values
    rawX_test = df[test_start:].values
    rawY_train = target[:train_end]
    rawY_test = target[test_start:]
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(rawX_train)
    X_test = scaler_x.transform(rawX_test)
    #scaler_y = StandardScaler()
    Y_train = rawY_train #scaler_y.fit_transform(rawY_train)
    Y_test = rawY_test #scaler_y.transform(rawY_test)
    X_train = np.array([X_train[i:i + output_dim] for i in range(X_train.shape[0]-output_dim+1)])
    X_test = np.array([X_test[i:i + output_dim] for i in range(X_test.shape[0]-output_dim+1)])
    y_train = np.array([Y_train[i:i + output_dim] for i in range(Y_train.shape[0]-output_dim+1)]).reshape(-1, output_dim)
    y_test = np.array([Y_test[i:i + output_dim] for i in range(Y_test.shape[0]-output_dim+1)]).reshape(-1, output_dim)
    return X_train, y_train, X_test, y_test  # [ X_train.shape = (batch_size, n_features, output_dim) ]


def preprocess_1b_trina(df, datetime_col, target_col, freq):
    df.drop(['Active_Energy_Delivered_Received', 'Current_Phase_Average'], axis=1, inplace=True)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    df = df.resample(freq).mean().copy()
    if len(pd.date_range(df.index[0], df.index[-1], freq=freq)) == len(df):
        print('Data has no missing timesteps in index.')
    else:
        print('Data has missing timestamps in index.')
    df = df.groupby(df.index.time).ffill()
    df = df['2014-01-01':'2018-12-31'].copy() # 1B Trina specific
    return df


def germansolarfarm(df, datetime_col, target_col):
    
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    
    return df


def europewindfarm(df, datetime_col, target_col):
    
    df.drop('ForecastingTime', axis=1, inplace=True)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    
    return df


def baywa(path, datetime_col, target_col):
    df = pd.read_csv(path, sep=';', decimal='.')

    # handle time col
    df.dropna(inplace=True)
    df[datetime_col] = pd.to_datetime(df[datetime_col], format='%d.%m.%y %H:%M')
    start = df[datetime_col].iloc[0].tz_localize('Europe/Berlin')
    end = df[datetime_col].iloc[-1].tz_localize('Europe/Berlin')
    df[datetime_col] = pd.date_range(start, end, freq='h')

    # handle target col
    df[target_col] = df[target_col].str.replace(',00$', '', regex=True)
    df[target_col] = df[target_col].str.replace('.', '')
    df[target_col] = df[target_col].astype('int')

    df.rename(columns={datetime_col: 'DATETIME', target_col: 'TARGET'}, inplace=True)
    
    df.set_index('DATETIME', inplace=True)

    return df


def vattenfall(path, target_col):
    df = pd.read_csv(path, sep=';', decimal=',')

    df['DELIVERY_DATE'] = pd.to_datetime(df['DELIVERY_DATE'], format='%d.%m.%y', dayfirst=True)
    df['DATETIME'] = pd.to_datetime(df['DELIVERY_DATE'].dt.strftime('%Y-%m-%d') + ' ' + df['DELIVERY_HOUR'])
    df['DATETIME'] = df['DATETIME'].dt.tz_localize('Europe/Berlin', ambiguous='infer')

    df.drop(['DELIVERY_DATE', 'DELIVERY_HOUR'], axis=1, inplace=True)
    df.set_index('DATETIME', inplace=True)

    rename_mask = {'DE_WIND_PRODUCTION_kW': 'TARGET',
                   'DE_WIND_AVAILABLE_CAPACITY': 'AVAIL_CAPA',
                   'DE_WIND_CURTAILMENT': 'CURTAIL'}
    
    df.rename(columns=rename_mask, inplace=True)

    df.drop(['AVAIL_CAPA', 'CURTAIL'], axis=1, inplace=True)

    return df


def trianel(path, target_col):
    
    df = pd.read_csv(path, sep=';', decimal=',')

    df.rename(columns={'Unnamed: 0': 'DATETIME', target_col: 'TARGET'}, inplace=True)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%d.%m.%Y %H:%M')
    df.set_index('DATETIME', inplace=True)
    
    return df
