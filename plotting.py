import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from datetime import datetime, timedelta
from tensorflow import keras

import utils
import utils_eval


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

def plot_forecast(pred: pd.DataFrame,
                  true: pd.DataFrame,
                  date: str,
                  horizon: int,
                  figsize: Tuple[int, int],
                  t_0=None,
                  print_metric=False,
                  grid=True):
    fig = plt.figure(figsize=figsize)
    if t_0:
        time_index = pd.to_datetime(f'{date} {t_0}:00:00')
        y_pred = pred.loc[time_index].values
        y_true = true.loc[time_index].values
    else:
        y_pred = pred.loc[date].values
        y_true = true.loc[date].values
    if print_metric:
        metrics = utils_eval.get_metrics(y_pred=y_pred,
                                    y_true=y_true)
        for key, value in metrics.items():
            print(f"{key}: {value[0]}")
    x_values = np.arange(1, len(y_pred) + 1, 1)
    #x_ticks = [(t_0 + i) % 24 for i in range(horizon)]
    plt.plot(x_values, y_pred, label='Predicted')
    plt.plot(x_values, y_true, label='True')
    # if horizon > 35:
    #     plt.axvline(x=13, color='r', linestyle='--')
    #     plt.axvline(x=36, color='r', linestyle='--')
    #     plt.axvspan(13, 36, color='gray', alpha=0.2)
    #xticks_positions = x_values[::2]
    # xticks_labels = [x_ticks[i] for i in range(len(x_ticks)) if i % 2 == 0]
    # if len(xticks_positions) > len(xticks_labels):
    #     xticks_positions = xticks_positions[:len(xticks_labels)]
    # elif len(xticks_labels) > len(xticks_positions):
    #     xticks_labels = xticks_labels[:len(xticks_positions)]
    #plt.xticks(xticks_positions, xticks_labels)
    x_values = np.arange(1, horizon + 1, 1)
    if isinstance(t_0, int):  # Assuming hourly resolution
        x_ticks = [(t_0 + i) % 24 for i in range(horizon)]
        xticks_positions = x_values[::6]  # Every third position
        xticks_labels = [x_ticks[i] for i in range(len(x_ticks)) if i % 6 == 0]  # Every third label
    else:  # Assuming finer resolution, e.g., 15 minutes
        time_resolution = pd.to_timedelta(t_0).seconds // 60  # Convert to minutes
        x_ticks = [(time_resolution * i) % (24 * 60) for i in range(horizon)]
        xticks_positions = x_values[::24]  # Every 3 hours (12 * 15 minutes = 180 minutes)
        xticks_labels = [f"{x_ticks[i] // 60:02}:{x_ticks[i] % 60:02}" for i in range(len(x_ticks)) if i % 24 == 0]
    plt.xticks(xticks_positions, xticks_labels)
    plt.title(f'Forecast for {date}')
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(grid, linestyle='--', alpha=0.6)
    plt.show()

def plot_error(pred: pd.DataFrame,
               true: pd.DataFrame,
               t_0: int,
               horizon: int,
               figsize: Tuple[int, int],
               grid=True):
    fig = plt.figure(figsize=figsize)
    error = pred - true
    t_0_errors = error[(error.index.time == pd.to_datetime(f'{t_0}:00:00').time())]
    t_0_errors = t_0_errors.mean(axis=0).values
    x_values = np.arange(1, len(t_0_errors) + 1, 1)
    #x_ticks = [(t_0 + i) % 24 for i in range(horizon)]
    plt.plot(x_values, t_0_errors)
    # if horizon > 35:
    #     plt.axvline(x=13, color='r', linestyle='--')
    #     plt.axvline(x=36, color='r', linestyle='--')
    #     plt.axvspan(13, 36, color='gray', alpha=0.2)
    #xticks_positions = x_values[::2]  # Jedes zweite Element der Positionen
    #xticks_labels = [x_ticks[i] for i in range(len(x_ticks)) if i % 2 == 0]  # Jedes zweite Label
    #plt.xticks(xticks_positions, xticks_labels)
    x_values = np.arange(1, horizon + 1, 1)
    if isinstance(t_0, int):  # Assuming hourly resolution
        x_ticks = [(t_0 + i) % 24 for i in range(horizon)]
        xticks_positions = x_values[::6]  # Every third position
        xticks_labels = [x_ticks[i] for i in range(len(x_ticks)) if i % 6 == 0]  # Every third label
    else:  # Assuming finer resolution, e.g., 15 minutes
        time_resolution = pd.to_timedelta(t_0).seconds // 60  # Convert to minutes
        x_ticks = [(time_resolution * i) % (24 * 60) for i in range(horizon)]
        xticks_positions = x_values[::24]  # Every 3 hours (12 * 15 minutes = 180 minutes)
        xticks_labels = [f"{x_ticks[i] // 60:02}:{x_ticks[i] % 60:02}" for i in range(len(x_ticks)) if i % 24 == 0]
    plt.xticks(xticks_positions, xticks_labels)
    plt.title('Mean error in forecasted hour')
    plt.ylabel('Error')
    plt.xlabel('Time')
    plt.grid(grid, linestyle='--', alpha=0.6)
    plt.show()

def plot_boxplots(pred: pd.DataFrame,
                 true: pd.DataFrame,
                 t_0: int,
                 horizon: int,
                 figsize: Tuple[int, int],
                 showfliers=False,
                 grid=True):
    fig = plt.figure(figsize=figsize)
    error = pred - true
    t_0_errors = error[(error.index.time == pd.to_datetime(f'{t_0}:00:00').time())]
    data_to_plot = []
    for i in range(1, horizon + 1):
        column_name = f't+{i}'
        if column_name in t_0_errors.columns:
            data_to_plot.append(t_0_errors[column_name].values)
        else:
            data_to_plot.append([])
    boxplot_return = plt.boxplot(data_to_plot, positions=np.arange(1, horizon + 1), showfliers=showfliers)
    for median in boxplot_return['medians']:
        median.set(color='black')
    x_values = np.arange(1, horizon + 1, 1)
    if isinstance(t_0, int):  # Assuming hourly resolution
        x_ticks = [(t_0 + i) % 24 for i in range(horizon)]
        xticks_positions = x_values[::6]  # Every third position
        xticks_labels = [x_ticks[i] for i in range(len(x_ticks)) if i % 6 == 0]  # Every third label
    else:  # Assuming finer resolution, e.g., 15 minutes
        time_resolution = pd.to_timedelta(t_0).seconds // 60  # Convert to minutes
        x_ticks = [(time_resolution * i) % (24 * 60) for i in range(horizon)]
        xticks_positions = x_values[::24]  # Every 3 hours (12 * 15 minutes = 180 minutes)
        xticks_labels = [f"{x_ticks[i] // 60:02}:{x_ticks[i] % 60:02}" for i in range(len(x_ticks)) if i % 24 == 0]
    plt.xticks(xticks_positions, xticks_labels)
    plt.title('Error in forecasted hour')
    plt.ylabel('Error')
    plt.xlabel('Time')
    plt.grid(grid, linestyle='--', alpha=0.6)
    plt.show()

def plot_error_distribution(pred: pd.DataFrame,
                            true: pd.DataFrame,
                            figsize: Tuple[int, int],
                            bins=100,
                            t_0=None,
                            grid=True):
    errors = pred - true
    if t_0:
        errors = errors[(errors.index.time == pd.to_datetime(f'{t_0}:00:00').time())]
    errors = errors.values.flatten()
    plt.figure(figsize=figsize)
    sns.histplot(data=errors, stat='probability', bins=bins)
    plt.xlabel("Error")
    plt.ylabel("Probability")
    plt.title("Error distribution")
    plt.grid(grid, linestyle='--', alpha=0.6)
    plt.show()