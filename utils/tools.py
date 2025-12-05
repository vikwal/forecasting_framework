"""
utilities for data loading, training, and model management.
"""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler
import os
import logging
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from . import models



def load_config(path: str):
    """Load YAML configuration file - framework agnostic"""
    with open(path, 'r') as file_object:
        config = yaml.load(file_object, Loader=yaml.SafeLoader)
    return config


def get_y(X_test: Any,
          y_test: np.ndarray,
          model: nn.Module,
          scaler_y: StandardScaler = None,
          device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions from PyTorch model"""
    model.eval()

    with torch.no_grad():
        if isinstance(X_test, dict):
            # TFT case
            X_test_tensors = {
                'observed': torch.FloatTensor(X_test['observed']).to(device),
                'known': torch.FloatTensor(X_test['known']).to(device),
            }
            if 'static' in X_test:
                X_test_tensors['static'] = torch.FloatTensor(X_test['static']).to(device)
                y_pred = model(X_test_tensors['observed'], X_test_tensors['known'], X_test_tensors['static'])
            else:
                y_pred = model(X_test_tensors['observed'], X_test_tensors['known'], None)
        else:
            # Standard case
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_pred = model(X_test_tensor)

        y_pred = y_pred.cpu().numpy()

    # Reshape if needed
    y_pred = y_pred.reshape(-1, y_test.shape[-1])

    if scaler_y:
        y_pred = scaler_y.inverse_transform(y_pred)
        y_true = scaler_y.inverse_transform(y_test)
    else:
        y_true = y_test

    if len(y_pred.shape) == 3:  # if seq2seq output
        y_pred = y_pred[:, :, -1]  # take last output from seq

    y_pred[y_pred < 0] = 0
    return y_true, y_pred


def y_to_df(y: np.ndarray,
            output_dim: int,
            horizon: int,
            index: np.ndarray,
            t_0=None) -> pd.DataFrame:
    """Convert predictions to DataFrame - framework agnostic"""
    index_for_df = index
    if index.shape[-1] == 3:
        index_for_df = index[:, 0]
        if output_dim == 1:
            y = y.reshape(-1, horizon)
            index_for_df = np.array(list(set(index_for_df)))
            index_for_df.sort()
    col_shape = y.shape[-1]
    cols = [f't+{i+1}' for i in range(col_shape)]
    df = pd.DataFrame(data=y, columns=cols, index=index_for_df)

    if output_dim == 1 and not index.shape[-1] == 3:
        new_columns_dict = {}
        base_col_series = df.iloc[:, 0]
        for i in range(2, horizon + 1):
            shift_amount = -(i - 1)
            new_columns_dict[f't+{i}'] = base_col_series.shift(shift_amount)
        if new_columns_dict:
            new_cols_df = pd.DataFrame(new_columns_dict, index=df.index)
            df = pd.concat([df, new_cols_df], axis=1)
        df.dropna(inplace=True)

    if t_0:
        df = df.loc[(df.index.time == pd.to_datetime(f'{t_0}:00').time())]

    return df


def get_feature_dim(X: Any):
    """Get feature dimensions - framework agnostic"""
    if type(X) == np.ndarray:
        feature_dim = X.shape[2]
    elif (len(X) <= 3):  # TFT case
        feature_dim = {}
        feature_dim['observed_dim'] = X['observed'].shape[-1]
        feature_dim['known_dim'] = X['known'].shape[-1]
        feature_dim['static_dim'] = X['static'].shape[-1] if 'static' in X else 0
    return feature_dim


def create_pytorch_dataloader(X, y, batch_size, shuffle=True, num_workers=0, device='cpu', drop_last=True):
    """
    Create PyTorch DataLoader from numpy arrays.
    More efficient than TensorFlow's tf.data.Dataset.

    Args:
        num_workers: Number of worker processes for data loading.
                    'auto' = min(8, cpu_count // 2) for optimal performance
                    int = specific number of workers
                    0 = single-threaded (not recommended for GPU training)
        drop_last: Whether to drop the last incomplete batch.
                   Should be True for training, False for validation/testing.
    """
    # Auto-detect optimal num_workers
    if num_workers == 'auto':
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # Use at most 8 workers, or half of CPU cores (leave resources for main process)
        num_workers = max(1, cpu_count // 2)
        logging.debug(f"Auto-detected num_workers={num_workers} (CPU count: {cpu_count})")

    logging.debug(f"Creating PyTorch DataLoader with batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}, drop_last={drop_last}")

    if isinstance(X, dict):
        # TFT case - use zero-copy conversion from numpy
        tensors = [
            torch.from_numpy(X['observed']).float(),
            torch.from_numpy(X['known']).float(),
        ]
        if 'static' in X:
            tensors.append(torch.from_numpy(X['static']).float())
        tensors.append(torch.from_numpy(y).float())

        dataset = TensorDataset(*tensors)
    else:
        # Standard case - use zero-copy conversion from numpy
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float()
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        drop_last=drop_last,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
    )

    logging.debug(f"DataLoader created successfully with {len(dataset)} samples, {len(dataloader)} batches")
    return dataloader


def training_pipeline(train: Tuple[np.ndarray, np.ndarray],
                      hyperparameters: dict,
                      config: dict,
                      val: Tuple[np.ndarray, np.ndarray] = None,
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    PyTorch training pipeline.
    Returns: history (dict), model (nn.Module)
    """
    X_train, y_train = train
    X_val, y_val = val if val else (None, None)

    config['model']['feature_dim'] = get_feature_dim(X=X_train)

    # Create model
    model = models.get_model(config=config, hyperparameters=hyperparameters)
    model = model.to(device)
    model = torch.compile(model)

    logging.debug(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    batch_size = hyperparameters['batch_size']

    # Create DataLoaders
    train_loader = create_pytorch_dataloader(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=config['model']['shuffle'],
        device=device,
        num_workers=config['model']['num_workers'],
        drop_last=True  # Drop incomplete batches for training
    )

    val_loader = None
    if val:
        val_loader = create_pytorch_dataloader(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False,
            device=device,
            num_workers=config['model']['num_workers'],
            drop_last=False  # Keep all validation samples, even incomplete batches
        )
        logging.debug(f"Validation loader created with {len(val_loader.dataset)} samples")
    else:
        logging.warning("No validation data provided - validation metrics will not be available!")

    # Free memory after DataLoader creation
    del X_train, y_train
    if val:
        del X_val, y_val
    gc.collect()

    # Setup optimizer and loss
    lr = hyperparameters.get('learning_rate', hyperparameters.get('lr', 0.001))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    epochs = hyperparameters.get('epochs', 200)
    history = {
        'train_loss': [], 'val_loss': [],
        'train_rmse': [], 'val_rmse': [],
        'train_mae': [], 'val_mae': [],
        'train_r2': [], 'val_r2': []
    }

    # Early Stopping setup
    early_stopping_config = config['model'].get('early_stopping', {})
    use_early_stopping = early_stopping_config.get('enabled', False)

    if use_early_stopping and val_loader is not None:
        patience = early_stopping_config.get('patience', 10)
        min_delta = early_stopping_config.get('min_delta', 0.0001)
        monitor_metric = early_stopping_config.get('monitor', 'val_loss')
        mode = early_stopping_config.get('mode', 'min')  # 'min' or 'max'
        restore_best_weights = early_stopping_config.get('restore_best_weights', True)

        logging.debug(f"Early Stopping enabled: patience={patience}, min_delta={min_delta}, monitor={monitor_metric}, mode={mode}")

        best_metric_value = float('inf') if mode == 'min' else float('-inf')
        epochs_without_improvement = 0
        best_model_state = None
    else:
        use_early_stopping = False
        if early_stopping_config.get('enabled', False) and val_loader is None:
            logging.warning("Early stopping enabled but no validation data provided. Disabling early stopping.")

    # Determine if model is TFT based on model name, not batch structure
    is_tft = config['model']['name'] == 'tft'

    for epoch in range(epochs):
        # Training
        model.train()
        train_preds_list = []
        train_targets_list = []

        for batch in train_loader:
            if is_tft:
                # TFT case - expects (observed, known, [static], targets)
                if len(batch) == 4:
                    obs, known, static, targets = [b.to(device) for b in batch]
                elif len(batch) == 3:
                    obs, known, targets = [b.to(device) for b in batch]
                    static = None
                else:
                    raise ValueError(f"Unexpected batch size for TFT: {len(batch)}")

                predictions = model(obs, known, static)
            else:
                # Standard case - expects (inputs, targets)
                if len(batch) != 2:
                    raise ValueError(f"Unexpected batch size for non-TFT model: {len(batch)}. Expected 2 (inputs, targets)")
                inputs, targets = [b.to(device) for b in batch]
                predictions = model(inputs)

            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if 'clipnorm' in config['model'].get('tft', {}):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['model']['tft']['clipnorm'])

            optimizer.step()

            # Move to CPU immediately to free GPU memory
            train_preds_list.append(predictions.detach().cpu())
            train_targets_list.append(targets.detach().cpu())

        # Calculate training metrics (tensors already on CPU)
        train_preds = torch.cat(train_preds_list, dim=0).numpy()
        train_targets = torch.cat(train_targets_list, dim=0).numpy()

        train_mse = np.mean((train_preds - train_targets) ** 2)
        train_rmse = np.sqrt(train_mse)
        train_mae = np.mean(np.abs(train_preds - train_targets))

        ss_res = np.sum((train_targets - train_preds) ** 2)
        ss_tot = np.sum((train_targets - np.mean(train_targets)) ** 2)
        train_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        history['train_loss'].append(train_mse)
        history['train_rmse'].append(train_rmse)
        history['train_mae'].append(train_mae)
        history['train_r2'].append(train_r2)

        # Validation
        if val_loader:
            model.eval()
            val_preds_list = []
            val_targets_list = []

            with torch.no_grad():
                for batch in val_loader:
                    if is_tft:
                        # TFT case - expects (observed, known, [static], targets)
                        if len(batch) == 4:
                            obs, known, static, targets = [b.to(device) for b in batch]
                        elif len(batch) == 3:
                            obs, known, targets = [b.to(device) for b in batch]
                            static = None
                        else:
                            raise ValueError(f"Unexpected batch size for TFT: {len(batch)}")
                        predictions = model(obs, known, static)
                    else:
                        # Standard case - expects (inputs, targets)
                        if len(batch) != 2:
                            raise ValueError(f"Unexpected batch size for non-TFT model: {len(batch)}. Expected 2 (inputs, targets)")
                        inputs, targets = [b.to(device) for b in batch]
                        predictions = model(inputs)

                    val_preds_list.append(predictions)
                    val_targets_list.append(targets)

            val_preds = torch.cat(val_preds_list, dim=0).cpu().numpy()
            val_targets = torch.cat(val_targets_list, dim=0).cpu().numpy()

            val_mse = np.mean((val_preds - val_targets) ** 2)
            val_rmse = np.sqrt(val_mse)
            val_mae = np.mean(np.abs(val_preds - val_targets))

            ss_res = np.sum((val_targets - val_preds) ** 2)
            ss_tot = np.sum((val_targets - np.mean(val_targets)) ** 2)
            val_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            history['val_loss'].append(val_mse)
            history['val_rmse'].append(val_rmse)
            history['val_mae'].append(val_mae)
            history['val_r2'].append(val_r2)

            # Early Stopping check
            if use_early_stopping:
                # Get current metric value
                current_metrics = {
                    'val_loss': val_mse,
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_r2': val_r2
                }
                current_value = current_metrics.get(monitor_metric, val_mse)

                # Check for improvement
                if mode == 'min':
                    improved = current_value < (best_metric_value - min_delta)
                else:  # mode == 'max'
                    improved = current_value > (best_metric_value + min_delta)

                if improved:
                    best_metric_value = current_value
                    epochs_without_improvement = 0
                    if restore_best_weights:
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    epochs_without_improvement += 1
                # Stop if patience exceeded
                if epochs_without_improvement >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    logging.info(f"Best {monitor_metric}: {best_metric_value:.6f}")
                    if restore_best_weights and best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        logging.debug("Restored best model weights")
                    break

            if config['model'].get('verbose', 1) > 0:
                logging.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train: MSE={train_mse:.6f}, RMSE={train_rmse:.6f}, R²={train_r2:.4f} | "
                    f"Val: MSE={val_mse:.6f}, RMSE={val_rmse:.6f}, R²={val_r2:.4f}"
                )
        else:
            if config['model'].get('verbose', 1) > 0:
                logging.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train: MSE={train_mse:.6f}, RMSE={train_rmse:.6f}, R²={train_r2:.4f}"
                )

    return history, model


def handle_freq(config: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust config based on time series frequency - framework agnostic"""
    freq = config['data']['freq']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    output_dim = config['model']['output_dim']

    if freq == '15min':
        if not output_dim == 1:
            output_dim = output_dim * 4
        horizon = horizon * 4
        lookback = lookback * 4

    if not output_dim == 1:
        horizon = output_dim

    config['data']['freq'] = freq
    config['model']['lookback'] = lookback
    config['model']['horizon'] = horizon
    config['model']['output_dim'] = output_dim

    return config


def concatenate_data(old, new):
    """Concatenate data - framework agnostic"""
    if type(old) == np.ndarray:
        return np.concatenate((old, new))
    elif type(old) == dict:
        result = {}
        for key, value in old.items():
            result[key] = np.concatenate((value, new[key]))
        return result


def create_data_generator(dfs, config, features, scaler_x=None):
    """
    Generator for memory-efficient data processing - framework agnostic.
    Reuses preprocessing.pipeline from TensorFlow version.
    """
    from . import preprocessing

    if scaler_x:
        config['scaler_x'] = scaler_x

    for key, df in dfs.items():
        logging.debug(f'Processing {key} in generator.')
        prepared_data, _ = preprocessing.pipeline(
            data=df,
            config=config,
            known_cols=features['known'],
            observed_cols=features['observed'],
            static_cols=features['static'],
            target_col=config['data']['target_col']
        )

        yield {
            'key': key,
            'X_train': prepared_data['X_train'],
            'y_train': prepared_data['y_train'],
            'X_test': prepared_data['X_test'],
            'y_test': prepared_data['y_test'],
            'index_test': prepared_data['index_test'],
            'scalers': prepared_data['scalers']
        }

        del prepared_data
        del df
        gc.collect()


def combine_datasets_efficiently(data_generator):
    """
    Combine datasets efficiently - framework agnostic.
    Identical to TensorFlow version.
    """
    X_train, y_train = None, None
    X_test, y_test = None, None
    test_data = {}
    total_samples = 0

    for data_dict in data_generator:
        key = data_dict['key']
        logging.debug(f'Combining data from {key}')

        test_data[key] = (
            data_dict['X_test'],
            data_dict['y_test'],
            data_dict['index_test'],
            data_dict['scalers']['y']
        )

        # Combine test data
        if X_test is None:
            X_test = data_dict['X_test']
            y_test = data_dict['y_test']
        else:
            if isinstance(X_test, dict):
                # TFT case
                new_X_test = {}
                for feature_key in X_test.keys():
                    old_len = X_test[feature_key].shape[0]
                    new_len = data_dict['X_test'][feature_key].shape[0]
                    combined_shape = (old_len + new_len,) + X_test[feature_key].shape[1:]

                    combined_array = np.empty(combined_shape, dtype=X_test[feature_key].dtype)
                    combined_array[:old_len] = X_test[feature_key]
                    combined_array[old_len:] = data_dict['X_test'][feature_key]

                    new_X_test[feature_key] = combined_array

                del X_test
                X_test = new_X_test
            else:
                old_len = X_test.shape[0]
                new_len = data_dict['X_test'].shape[0]
                combined_shape = (old_len + new_len,) + X_test.shape[1:]

                combined_X_test = np.empty(combined_shape, dtype=X_test.dtype)
                combined_X_test[:old_len] = X_test
                combined_X_test[old_len:] = data_dict['X_test']

                del X_test
                X_test = combined_X_test

            old_len = y_test.shape[0]
            new_len = data_dict['y_test'].shape[0]
            combined_shape = (old_len + new_len,) + y_test.shape[1:]

            combined_y_test = np.empty(combined_shape, dtype=y_test.dtype)
            combined_y_test[:old_len] = y_test
            combined_y_test[old_len:] = data_dict['y_test']

            del y_test
            y_test = combined_y_test

        # Combine training data
        if X_train is None:
            X_train = data_dict['X_train']
            y_train = data_dict['y_train']
        else:
            if isinstance(X_train, dict):
                new_X_train = {}
                for feature_key in X_train.keys():
                    old_len = X_train[feature_key].shape[0]
                    new_len = data_dict['X_train'][feature_key].shape[0]
                    combined_shape = (old_len + new_len,) + X_train[feature_key].shape[1:]

                    combined_array = np.empty(combined_shape, dtype=X_train[feature_key].dtype)
                    combined_array[:old_len] = X_train[feature_key]
                    combined_array[old_len:] = data_dict['X_train'][feature_key]

                    new_X_train[feature_key] = combined_array

                del X_train
                X_train = new_X_train
            else:
                old_len = X_train.shape[0]
                new_len = data_dict['X_train'].shape[0]
                combined_shape = (old_len + new_len,) + X_train.shape[1:]

                combined_X = np.empty(combined_shape, dtype=X_train.dtype)
                combined_X[:old_len] = X_train
                combined_X[old_len:] = data_dict['X_train']

                del X_train
                X_train = combined_X

            old_len = y_train.shape[0]
            new_len = data_dict['y_train'].shape[0]
            combined_shape = (old_len + new_len,) + y_train.shape[1:]

            combined_y = np.empty(combined_shape, dtype=y_train.dtype)
            combined_y[:old_len] = y_train
            combined_y[old_len:] = data_dict['y_train']

            del y_train
            y_train = combined_y

        total_samples += len(data_dict['y_train'])
        logging.debug(f'Combined data now has {total_samples} samples')

        del data_dict
        gc.collect()

    return X_train, y_train, X_test, y_test, test_data

def calculate_retrain_periods(test_start: pd.Timestamp,
                               test_end: pd.Timestamp,
                               retrain_interval: int,
                               freq: str) -> list:
    """
    Calculate test periods for retraining.

    Args:
        test_start: Start of the entire test period
        test_end: End of the entire test period
        retrain_interval: Number of splits/retraining cycles
        freq: Frequency of the time series (e.g. '1h')

    Returns:
        List of (start, end) tuples for each retrain cycle
    """
    # Generate complete date range
    full_range = pd.date_range(start=test_start, end=test_end, freq=freq)
    total_periods = len(full_range)

    # Calculate period size for each interval
    period_size = total_periods // retrain_interval
    remainder = total_periods % retrain_interval

    periods = []
    start_idx = 0

    for i in range(retrain_interval):
        # Distribute remainder across LAST periods (not first)
        # This helps align with natural day boundaries
        extra = 1 if i >= (retrain_interval - remainder) else 0
        current_size = period_size + extra

        # Calculate end index for this period
        end_idx = start_idx + current_size - 1

        # Ensure we don't exceed the range
        if end_idx >= total_periods:
            end_idx = total_periods - 1

        period_start = full_range[start_idx]
        period_end = full_range[end_idx]

        periods.append((period_start, period_end))

        # Next period starts right after this one ends
        start_idx = end_idx + 1

    return periods

def initialize_gpu(use_gpu=None):
    """
    Initialize GPU for PyTorch.

    Args:
        use_gpu: None (use all), int (single GPU), or list of ints (multiple GPUs)
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s)")

        if use_gpu is not None:
            if isinstance(use_gpu, list):
                # Set visible devices
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, use_gpu))
                print(f"Using GPUs: {use_gpu}")
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu)
                print(f"Using GPU: {use_gpu}")
        else:
            print(f"Using all {num_gpus} GPU(s)")

        # Set default device
        torch.cuda.set_device(0)
    else:
        print("No GPUs found. Using CPU.")
