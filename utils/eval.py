import os
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from . import tools
from . import preprocessing

def persistence(y: pd.Series,
                horizon: int,
                from_date=None) -> pd.Series:
    '''
    Persistence model which takes the realized scenario from past.
    '''
    shifted = y.shift(horizon)
    y_pers = pd.Series(data=shifted, index=y.index)
    if type(y.index) == pd.core.indexes.multi.MultiIndex:
        y_pers = y_pers.reset_index().groupby('timestamp').mean().iloc[:,-1]
    if from_date:
        y_pers = y_pers[from_date:]
    return y_pers


def lin_reg(data: pd.DataFrame,
            train_end: str,
            test_start: str,
            target_col: str):
    '''
    Persistence model which uses linear regression.
    '''
    df = data.copy()
    df.dropna(inplace=True)
    y_train = df[:train_end][target_col].values
    df.drop([target_col], axis=1, inplace=True)
    X_train = df[:train_end].values
    X_test = df[test_start:].values
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pers = model.predict(X_test)
    return y_pers

def get_synth_wind(synth_dir: str,
                   park_id: str,
                   from_date: str = None,
                   to_date: str = None,
                   params: dict = None):
    file_path = os.path.join(synth_dir, f'synth_{park_id}.csv')
    df = pd.read_csv(file_path, sep=';')
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    turbines = params['turbines']
    turbine_params_path = os.path.join(synth_dir, 'turbine_parameter.csv')
    turbine_params = pd.read_csv(turbine_params_path, sep=';', dtype={'park_id': str})
    y_synth = pd.Series(data=np.zeros(len(df)), index=df.index)
    installed_capacity = 0
    for turbine in turbines:
        turbine_row = turbine_params.loc[turbine_params.turbine_name == turbine]
        installed_capacity += turbine_row['rated'].iloc[0]
        turbine_id = turbine_row['turbine'].iloc[0]
        y_synth += df[f'power_{turbine_id}']
    y_synth /= (installed_capacity * 1000 * 1000) # MW -> W
    if from_date and to_date:
        y_synth = y_synth[from_date:to_date]
    return y_synth

def get_synth_pv(synth_dir: str,
                   park_id: str,
                   from_date: str = None):
    file_path = os.path.join(synth_dir, f'synth_{park_id}.csv')
    df = pd.read_csv(file_path, sep=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    y_synth = df['power']
    if from_date:
        y_synth = y_synth[from_date:]
    return y_synth

def benchmark_models(data: pd.DataFrame,
                     horizon: int,
                     train_end: str,
                     test_start: str,
                     output_dim: int,
                     index_test: np.ndarray,
                     target_col='power',
                     t_0=None):
    '''
    Pipeline for applying different benchmark models.
    '''
    results = {}
    # Persistence
    y_pers = persistence(y=data[target_col],
                        horizon=horizon,
                        from_date=test_start)
    y_pers = preprocessing.make_windows(data=y_pers,
                                        seq_len=output_dim)
    df_pers = tools.y_to_df(y=y_pers,
                      output_dim=output_dim,
                      horizon=horizon,
                      index=index_test,
                      t_0=t_0)
    results['Persistence'] = df_pers
    # linear regression persistence
    y_pers = lin_reg(data=data,
                     train_end=train_end,
                     test_start=test_start,
                     target_col=target_col)
    y_pers = preprocessing.make_windows(data=y_pers,
                                        seq_len=output_dim)
    df_pers = tools.y_to_df(y=y_pers,
                      output_dim=output_dim,
                      horizon=horizon,
                      index=index_test,
                      t_0=t_0)
    results['LinearRegression'] = df_pers
    return results

def evaluate_models(pred: pd.DataFrame,
                    true: pd.DataFrame,
                    persistence: dict,
                    main_model_name='Main',
                    drop_except_main=False) -> pd.DataFrame:
    evaluation = get_metrics(y_pred=pred.values,
                             y_true=true.values)
    evaluation['Models'] = [main_model_name]
    for model, y_pred in persistence.items():
        evaluation['Models'].append(model)
        metrics = get_metrics(y_pred=y_pred.values,
                              y_true=true.values)
        for metric, value in metrics.items():
            evaluation[metric].append(value[0])
    results = pd.DataFrame(data=evaluation)
    results.set_index('Models', inplace=True)
    # skill factor
    results['Skill'] = 0.0
    #RMSE_p = results.loc['Persistence'].RMSE
    for model in evaluation['Models']:
        results.loc[model, 'Skill'] = 1 - results.loc[model].RMSE / results.loc['Persistence'].RMSE
    # drop all models except main model
    if drop_except_main:
        results = results.loc[[main_model_name]]
    #results['RMSE_p'] = RMSE_p
    return results

def get_metrics(y_pred: np.ndarray,
                y_true: np.ndarray) -> dict:
    error = y_pred - y_true
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(np.square(error).mean())
    mae = np.abs(error).mean()
    metrics = {'R^2': [r2],
               'RMSE': [rmse],
               'MAE': [mae]}
    return metrics

def evaluation_pipeline(data: pd.DataFrame,
                        model: keras.Model,
                        model_name: str,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        scaler_y: StandardScaler,
                        output_dim: int,
                        horizon: int,
                        index_test: np.ndarray,
                        test_start: str,
                        t_0: int,
                        park_id: str = None,
                        synth_dir: str = None,
                        get_physical_persistence: bool = False,
                        timestamp_col: str ='timestamp',
                        target_col: str ='power',
                        evaluate_on_all_test_data: bool = True) -> pd.DataFrame:
    y_true, y_pred = tools.get_y(X_test=X_test,
                                 y_test=y_test,
                                 scaler_y=scaler_y,
                                 model=model)
    df_pred = tools.y_to_df(y=y_pred,
                            output_dim=output_dim,
                            horizon=horizon,
                            index=index_test,
                            t_0=None if evaluate_on_all_test_data else t_0)
    df_true = tools.y_to_df(y=y_true,
                            output_dim=output_dim,
                            horizon=horizon,
                            index=index_test,
                            t_0=None if evaluate_on_all_test_data else t_0)

    test_indices = None
    if len(index_test.shape) != 1:
        test_indices = index_test[:,0]
        if output_dim == 1:
            test_indices = np.array(list(set(index_test[:,2])))
            test_indices.sort()
            y_pers = y_pers[test_indices]
            y_pers_raw = pd.DataFrame(index_test, columns=list(data.index.names))
            y_pers = y_pers_raw.merge(y_pers.to_frame(), how='left', on=timestamp_col)[target_col].values
            test_indices = None
    else:
        test_indices = index_test

    pers = {}
    test_start_ts = pd.Timestamp(test_start, tz=data.index.tz)
    y_pers = persistence(y=data[target_col],
                        horizon=horizon,
                        from_date=test_start_ts)
    # get persistence (most recent value)
    y_pers = preprocessing.make_windows(data=y_pers,
                                        seq_len=y_pred.shape[-1],
                                        step_size=1,
                                        indices=test_indices)
    df_pers = tools.y_to_df(y=y_pers,
                            output_dim=output_dim,
                            horizon=horizon,
                            index=index_test,
                            t_0=None if evaluate_on_all_test_data else t_0)
    pers['Persistence'] = df_pers
    # get physical persistence
    if get_physical_persistence:
        if 'wind' in synth_dir:
            y_synth = get_synth_wind(synth_dir=synth_dir,
                                    park_id=park_id,
                                    from_date=index_test[0],
                                    to_date=data.index[-1])
        elif 'solar' in synth_dir:
            y_synth = get_synth_pv(synth_dir=synth_dir,
                                    park_id=park_id,
                                    from_date=index_test[0],
                                    to_date=data.index[-1])
        y_synth = preprocessing.make_windows(data=y_synth,
                                             seq_len=y_pred.shape[-1],
                                             step_size=1, # has to be 1 filtering goes over t_0 in df_synth
                                             indices=test_indices)
        df_synth = tools.y_to_df(y=y_pers,
                                output_dim=output_dim,
                                horizon=horizon,
                                index=index_test,
                                t_0=None if evaluate_on_all_test_data else t_0)
        pers['Synth'] = df_synth
    evaluation = evaluate_models(pred=df_pred,
                                 true=df_true,
                                 persistence=pers,
                                 main_model_name=model_name,
                                 drop_except_main=True)
    return evaluation

def evaluate_retrain(config,
                     data,
                     index_test,
                     model,
                     hyperparameters,
                     cols,
                     scaler_y=None,
                     target_col='power'):
    t_0 = None if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
    retrain_interval = config['eval']['retrain_interval']
    output_dim = config['model']['output_dim']
    horizon = config['model']['horizon']
    freq = config['data']['freq']
    known, observed, static = cols
    if freq == '15min':
        retrain_interval /= 4
    full_days = len(index_test) // retrain_interval - 1
    y_true, y_pred, y_pers, index = None, None, None, None
    for day in range(full_days):
        from_index = day * retrain_interval
        if output_dim == 1:
            # here horizon is determined via the index
            adj_horizon = horizon
            to_index = day * retrain_interval + horizon
        else:
            # here horizon is already fixed by the output_dim
            adj_horizon = retrain_interval
            to_index = retrain_interval * (1 + day)
        index_day = index_test[from_index:to_index]
        prepared_data, df = preprocessing.pipeline(data=data,
                                                   config=config,
                                                   known_cols=known,
                                                   observed_cols=observed,
                                                   static_cols=static,
                                                   test_start=index_day[0])
        X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
        X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
        X_test, y_test = X_test[:adj_horizon], y_test[:adj_horizon]
        y_true_new, y_pred_new = tools.get_y(X_test=X_test,
                                             y_test=y_test,
                                             model=model,
                                             scaler_y=scaler_y)
        y_pers_raw = persistence(y=df[target_col],
                                 horizon=horizon,
                                 from_date=str(index_test[0].date()))
        y_pers_new = preprocessing.make_windows(y_pers_raw, y_pred_new.shape[-1], step_size=1)
        y_pers_new = y_pers_new[from_index:to_index]
        if y_pred is None:
            y_true, y_pred, y_pers, index = y_true_new, y_pred_new, y_pers_new, index_day
        else:
            y_true = np.concatenate((y_true, y_true_new))
            y_pred = np.concatenate((y_pred, y_pred_new))
            y_pers = np.concatenate((y_pers, y_pers_new))
            index = np.concatenate((index, index_day))
        model.fit(
            x = X_train,
            y = y_train,
            batch_size = hyperparameters['batch_size'],
            epochs = 2,#hyperparameters['epochs'],
            verbose = config['model']['verbose'],
            shuffle = config['model']['shuffle']
        )
    df_pred = tools.y_to_df(y_pred, output_dim, horizon, index, t_0)
    df_pers = tools.y_to_df(y_pers, output_dim, horizon, index, t_0)
    df_true = tools.y_to_df(y_true, output_dim, horizon, index, t_0)
    pers = {}
    pers['Persistence'] = df_pers
    evaluation = evaluate_models(pred=df_pred,
                                 true=df_true,
                                 persistence=pers,
                                 main_model_name=config['model']['name'],
                                 drop_except_main=True)
    return evaluation