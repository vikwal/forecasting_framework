import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from datetime import datetime, timedelta
from tensorflow import keras
from typing import Tuple, Union, Optional

from . import eval


def plot_training(history,
                  validation=False,
                  save_name=None):
    loss = history.history['loss']
    if validation:
        val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'blue', label='Training loss')
    if validation:
        plt.plot(epochs, val_loss, 'green', label='Validation loss')
    plt.title('Loss plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save_name:
        plt.savefig(save_name)
    plt.legend()
    plt.show()

def random_date(start_date: str,
                end_date: str) -> str:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days_between = (end - start).days
    random_days = random.randint(0, days_between)
    random_datetime = start + timedelta(days=random_days)
    return random_datetime.strftime("%Y-%m-%d")

def plot_forecast(y_pred: pd.DataFrame,
                  y_true: pd.DataFrame,
                  date: str,
                  horizon: int,
                  figsize: Tuple[int, int],
                  print_metric=False,
                  grid=True):
    fig = plt.figure(figsize=figsize)
    if print_metric:
        metrics = eval.get_metrics(y_pred=y_pred,
                                    y_true=y_true)
        for key, value in metrics.items():
            print(f"{key}: {value[0]}")
    x_values = np.arange(1, len(y_pred) + 1, 1)
    positions = np.arange(1, horizon + 1)
    step = max(1, horizon // 10)
    xticks_positions = positions[step-1::step]
    xticks_labels = [f't+{i}' for i in range(1, horizon + 1)][step-1::step]
    plt.plot(x_values, y_pred, label='Predicted')
    plt.plot(x_values, y_true, label='True')
    plt.title(f'Forecast for {date}')
    plt.xlabel('Forecasted hour')
    plt.ylabel('Normalized power')
    plt.xticks(xticks_positions, xticks_labels)
    plt.legend()
    plt.grid(grid, linestyle='--', alpha=0.6)
    plt.show()

def plot_error(pred: pd.DataFrame,
               true: pd.DataFrame,
               t_0: Optional[Union[str, int]],
               horizon: int,
               figsize: Tuple[int, int],
               grid=True):
    fig = plt.figure(figsize=figsize)
    error = pred - true
    if t_0 is not None:
        t_parsed = pd.to_datetime(str(t_0)).time()
        mask = error.index.time == t_parsed
        error = error.loc[mask]
    error = error.mean(axis=0).values
    x_values = np.arange(1, horizon + 1, 1)
    plt.plot(x_values, error)
    positions = np.arange(1, horizon + 1)
    step = max(1, horizon // 10)
    xticks_positions = positions[step-1::step]
    xticks_labels = [f't+{i}' for i in range(1, horizon + 1)][step-1::step]

    xlabel = 'Forecasted hour'
    title = 'Mean error in forecasted hour'
    if t_0: title = f'{title} ({t_0} UTC runs)'

    plt.xticks(xticks_positions, xticks_labels)
    plt.title(title)
    plt.ylabel('Error (Pred - True)')
    plt.xlabel(xlabel)
    plt.grid(grid, linestyle='--', alpha=0.6)
    plt.show()

def plot_boxplots(pred: pd.DataFrame,
                  true: pd.DataFrame,
                  horizon: int,
                  figsize: Tuple[int, int],
                  t_0: str = None,
                  showfliers: bool = False,
                  grid: bool = True):
    common_idx = pred.index.intersection(true.index)
    common_cols = pred.columns.intersection(true.columns)
    pred = pred.loc[common_idx, common_cols]
    true = true.loc[common_idx, common_cols]

    error = pred - true

    if t_0 is not None:
        t_parsed = pd.to_datetime(str(t_0)).time()
        mask = error.index.time == t_parsed
        error = error.loc[mask]

    data_to_plot = []
    for i in range(1, horizon + 1):
        col = f't+{i}'
        if col in error.columns:
            data_to_plot.append(error[col].dropna().values)
        else:
            data_to_plot.append(np.array([]))  # leerer Platz, falls Kolonne fehlt

    fig = plt.figure(figsize=figsize)
    positions = np.arange(1, horizon + 1)
    bp = plt.boxplot(data_to_plot, positions=positions, showfliers=showfliers)

    for median in bp['medians']:
        median.set(color='black')

    step = max(1, horizon // 10)
    xticks_positions = positions[step-1::step]
    xticks_labels = [f't+{i}' for i in range(1, horizon + 1)][step-1::step]

    xlabel = 'Forecasted hour'
    title = 'Error in forecasted hour'
    if t_0: title = f'{title} ({t_0} UTC runs)'

    plt.xticks(xticks_positions, xticks_labels)
    plt.title(title)
    plt.ylabel('Error (Pred - True)')
    plt.xlabel(xlabel)
    plt.grid(grid, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_error_distribution(pred: pd.DataFrame,
                            true: pd.DataFrame,
                            figsize: Tuple[int, int],
                            bins=100,
                            t_0=None,
                            grid=True):
    errors = pred - true
    title = "Error distribution"
    if t_0:
        errors = errors[(errors.index.time == pd.to_datetime(f'{t_0}:00').time())]
        title = f'{title} at {t_0} UTC'
    errors = errors.values.flatten()
    plt.figure(figsize=figsize)
    sns.histplot(data=errors, stat='density', bins=bins)
    plt.xlabel("Error (Pred - True)")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(grid, linestyle='--', alpha=0.6)
    plt.show()