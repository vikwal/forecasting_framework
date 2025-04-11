import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import utils
import preprocessing

def persistence(y: pd.Series,
                horizon: int,
                from_date=None) -> pd.Series:
    '''
    Persistence model which takes the realized scenario from past.
    '''
    shifted = y.shift(horizon)
    y_pers = pd.Series(data=shifted, index=y.index)
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
    df_pers = utils.y_to_df(y=y_pers,
                      output_dim=output_dim,
                      horizon=horizon,
                      index_test=index_test,
                      t_0=t_0)
    results['Persistence'] = df_pers
    # linear regression persistence
    y_pers = lin_reg(data=data,
                     train_end=train_end,
                     test_start=test_start,
                     target_col=target_col)
    y_pers = preprocessing.make_windows(data=y_pers,
                                        seq_len=output_dim)

    df_pers = utils.y_to_df(y=y_pers,
                      output_dim=output_dim,
                      horizon=horizon,
                      index_test=index_test,
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
                        t_0: int,
                        target_col='power',
                        evaluate_on_all_test_data=True) -> pd.DataFrame:
    y_true, y_pred = utils.get_y(X_test=X_test,
                           y_test=y_test,
                           output_dim=output_dim,
                           scaler_y=scaler_y,
                           model=model)
    if len(y_pred.shape) == 3:
        if y_pred.shape[1] == 1 & y_pred.shape[2] == 1:
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[2])
    df_pred = utils.y_to_df(y=y_pred,
                 output_dim=output_dim,
                 horizon=horizon,
                 index_test=index_test,
                 t_0=None if evaluate_on_all_test_data else t_0)
    df_true = utils.y_to_df(y=y_true,
                      output_dim=output_dim,
                      horizon=horizon,
                      index_test=index_test,
                      t_0=None if evaluate_on_all_test_data else t_0)
    y_pers = persistence(y=data[target_col],
                        horizon=horizon,
                        from_date=str(index_test[0].date()))
    y_pers = preprocessing.make_windows(data=y_pers,
                                        seq_len=output_dim)
    df_pers = utils.y_to_df(y=y_pers,
                      output_dim=output_dim,
                      horizon=horizon,
                      index_test=index_test,
                      t_0=None if evaluate_on_all_test_data else t_0)
    pers = {}
    pers['Persistence'] = df_pers
    evaluation = evaluate_models(pred=df_pred,
                                 true=df_true,
                                 persistence=pers,
                                 main_model_name=model_name,
                                 drop_except_main=True)
    return evaluation