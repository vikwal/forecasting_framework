"""
utilities for data loading, training, and model management.
"""

import copy
import math
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
import logging
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from . import models


def _pinball_loss(predictions: torch.Tensor, targets: torch.Tensor, quantiles: list) -> torch.Tensor:
    """Pinball (quantile) loss averaged over all quantiles.

    Args:
        predictions: (batch, horizon, num_quantiles) or (batch, horizon) when num_quantiles=1
        targets:     (batch, horizon)
        quantiles:   list of floats, e.g. [0.1, 0.5, 0.9]
    """
    losses = []
    for i, q in enumerate(quantiles):
        # predictions may be 2D (num_quantiles=1, squeezed) or 3D (multiple quantiles)
        if predictions.dim() == 2:
            pred_q = predictions
        else:
            pred_q = predictions[..., i]
        errors = targets - pred_q
        losses.append(torch.max((q - 1) * errors, q * errors))
    return torch.mean(torch.stack(losses, dim=-1))



def _env_var_constructor(loader, node):
    var_name = loader.construct_scalar(node)
    value = os.environ.get(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not set (required by YAML !ENV tag)")
    return value

_env_loader = yaml.SafeLoader
yaml.add_constructor('!ENV', _env_var_constructor, Loader=_env_loader)


def load_config(path: str):
    """Load YAML configuration file - framework agnostic"""
    with open(path, 'r') as file_object:
        config = yaml.load(file_object, Loader=_env_loader)
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


def get_y_chronos2(X_test: dict,
                   y_test: np.ndarray,
                   chronos_pipeline,
                   config: dict,
                   known_cols: list,
                   scaler_y=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from a Chronos-2 pipeline.

    X_test must have the structure produced by prepare_data_for_chronos2():
        'observed': (n_windows, lookback, n_observed)  – unscaled power history
        'known':    (n_windows, lookback+horizon, n_known) – scaled NWP features

    Returns (y_true, y_pred) as numpy arrays of shape (n_windows, horizon).
    """
    chronos_cfg = config['model']['chronos']
    horizon = config['model']['horizon']
    lookback = config['model']['lookback']

    observed = X_test['observed']              # (n, lookback, 1)
    known_full = X_test.get('known', None)     # (n, lookback+horizon, n_known)
    n_windows = observed.shape[0]

    # Slice NWP into past (lookback) and future (horizon) parts
    known_past   = known_full[:, :lookback, :] if known_full is not None else None
    known_future = known_full[:, lookback:,  :] if known_full is not None else None

    # Build long-format DataFrames vectorized (avoids O(n*t*k) Python loops)
    base_ts = pd.date_range('2020-01-01', periods=lookback + horizon, freq='1h')

    # --- context_df ---
    item_ids_ctx = np.repeat(np.arange(n_windows).astype(str), lookback)
    ts_ctx = np.tile(base_ts[:lookback], n_windows)
    context_df = pd.DataFrame({
        'item_id': item_ids_ctx,
        'timestamp': ts_ctx,
        'target': observed[:, :, 0].ravel(),
    })
    if known_past is not None:
        for k, col in enumerate(known_cols):
            context_df[col] = known_past[:, :, k].ravel()

    # --- future_df ---
    future_df = None
    if known_future is not None:
        item_ids_fut = np.repeat(np.arange(n_windows).astype(str), horizon)
        ts_fut = np.tile(base_ts[lookback:], n_windows)
        future_df = pd.DataFrame({
            'item_id': item_ids_fut,
            'timestamp': ts_fut,
        })
        for k, col in enumerate(known_cols):
            future_df[col] = known_future[:, :, k].ravel()

    quantile_levels = chronos_cfg.get('quantile_levels', [0.5])

    pred_df = chronos_pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=horizon,
        quantile_levels=quantile_levels,
        id_column='item_id',
        timestamp_column='timestamp',
        target='target',
        batch_size=config['model'].get('batch_size', 256),
        context_length=lookback,
        cross_learning=chronos_cfg.get('cross_learning', False),
    )

    # Identify the quantile column (Chronos-2 names vary across versions)
    target_q = quantile_levels[0]
    candidate_cols = [f'q{target_q}', str(target_q), f'target_q{target_q}', 'mean']
    median_col = next((c for c in candidate_cols if c in pred_df.columns), None)
    if median_col is None:
        non_meta = [c for c in pred_df.columns if c not in ('item_id', 'timestamp')]
        median_col = non_meta[-1]

    predictions = []
    for i in range(n_windows):
        item_preds = (
            pred_df[pred_df['item_id'] == str(i)]
            .sort_values('timestamp')[median_col]
            .values
        )
        predictions.append(item_preds[:horizon])

    y_pred = np.clip(np.array(predictions), 0, None)  # (n_windows, horizon)

    y_true = y_test
    if scaler_y:
        y_pred = scaler_y.inverse_transform(y_pred)
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, horizon))

    return y_true, y_pred


def build_chronos_fit_inputs(X_dict: dict, y: np.ndarray, known_cols: list) -> list:
    """
    Convert X_dict/y arrays to the Chronos-2 fit() input format.

    Each entry is a dict with:
      'target'            : 1-D array of shape (lookback + horizon,)
      'past_covariates'   : {col: array(lookback+horizon)} — NWP full window
      'future_covariates' : {col: None} — signals known-future to the model
    """
    observed = X_dict['observed']           # (n, lookback, n_obs) — n_obs may be 0
    known_full = X_dict.get('known', None)  # (n, lookback+horizon, n_known)
    has_observed = observed.shape[2] > 0
    inputs = []
    for i in range(observed.shape[0]):
        if has_observed:
            target = np.concatenate([observed[i, :, 0], y[i, :]])
        else:
            target = y[i, :]
        entry = {
            'target': target,
        }
        if known_full is not None and len(known_cols) > 0:
            lookback = observed.shape[1]
            cov_start = 0 if has_observed else lookback
            entry['past_covariates'] = {
                col: known_full[i, cov_start:, k].astype(np.float32)
                for k, col in enumerate(known_cols)
            }
            entry['future_covariates'] = {col: None for col in known_cols}
        inputs.append(entry)
    return inputs


class ChronosEarlyStoppingCallback:
    """
    Wraps a HuggingFace TrainerCallback that computes val (and train) RMSE/R²
    after every epoch and optionally stops training early.

    Usage in train_cl.py
    --------------------
    cb = ChronosEarlyStoppingCallback(
        pipeline=chronos_pipeline,
        X_train=X_tr, y_train=y_tr,
        X_val=X_val,  y_val=y_val,
        config=period_config,
        known_cols=known_cols_fit,
        early_stopping_cfg=period_config['model'].get('early_stopping', {}),
        steps_per_epoch=steps_per_epoch,
    )
    chronos_pipeline.fit(..., callbacks=[cb.callback])
    history = cb.history
    """

    # Max windows used for train-metric inference (keeps it fast)
    MAX_TRAIN_SAMPLES = 2000

    def __init__(self, pipeline, X_train, y_train, X_val, y_val,
                 config, known_cols, early_stopping_cfg, steps_per_epoch):
        self.pipeline = pipeline
        self.X_val = X_val
        self.y_val = y_val
        self.config = config
        self.known_cols = known_cols
        self.steps_per_epoch = steps_per_epoch

        # Subsample training data for metric inference
        n_tr = X_train['observed'].shape[0]
        idx = np.random.choice(n_tr, min(n_tr, self.MAX_TRAIN_SAMPLES), replace=False)
        self.X_train_sub = {k: v[idx] for k, v in X_train.items()}
        self.y_train_sub = y_train[idx]

        # Early stopping settings
        es = early_stopping_cfg
        self.use_early_stopping = es.get('enabled', False)
        self.patience = es.get('patience', 10)
        self.min_delta = es.get('min_delta', 0.0001)
        self.monitor = es.get('monitor', 'val_rmse')
        self.mode = es.get('mode', 'min')
        self.restore_best_weights = es.get('restore_best_weights', True)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_rmse': [], 'val_rmse': [],
            'train_r2':   [], 'val_r2':   [],
        }
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.best_state_dict = None

        self.callback = self._build_hf_callback()

    def _infer(self, model, X, y):
        """Run predict_df with a temporarily swapped model; return (y_true, y_pred)."""
        orig = self.pipeline.model
        self.pipeline.model = model
        try:
            y_true, y_pred = get_y_chronos2(
                X_test=X, y_test=y,
                chronos_pipeline=self.pipeline,
                config=self.config,
                known_cols=self.known_cols,
            )
        finally:
            self.pipeline.model = orig
        return y_true, y_pred

    def _compute_metrics(self, y_true, y_pred):
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        r2   = float(r2_score(y_true.flatten(), y_pred.flatten()))
        mse  = float(np.mean((y_pred - y_true) ** 2))
        return rmse, r2, mse

    def _build_hf_callback(self):
        from transformers import TrainerCallback

        outer = self  # closure reference

        class _Callback(TrainerCallback):
            def on_log(self, args, state, control, model, logs=None, **kwargs):
                if logs is None or 'loss' not in logs:
                    return
                # Fire only at epoch boundaries
                current_step = state.global_step
                if current_step % outer.steps_per_epoch != 0:
                    return

                train_loss = logs['loss']
                epoch = round(current_step / outer.steps_per_epoch)

                # Val metrics
                y_true_v, y_pred_v = outer._infer(model, outer.X_val, outer.y_val)
                val_rmse, val_r2, val_loss = outer._compute_metrics(y_true_v, y_pred_v)

                # Train metrics (subsample)
                y_true_t, y_pred_t = outer._infer(model, outer.X_train_sub, outer.y_train_sub)
                train_rmse, train_r2, _ = outer._compute_metrics(y_true_t, y_pred_t)

                outer.history['train_loss'].append(train_loss)
                outer.history['val_loss'].append(val_loss)
                outer.history['train_rmse'].append(train_rmse)
                outer.history['val_rmse'].append(val_rmse)
                outer.history['train_r2'].append(train_r2)
                outer.history['val_r2'].append(val_r2)

                logging.info(
                    f"Epoch {epoch} | "
                    f"train_loss={train_loss:.4f}  train_rmse={train_rmse:.4f}  train_r²={train_r2:.4f} | "
                    f"val_loss={val_loss:.4f}  val_rmse={val_rmse:.4f}  val_r²={val_r2:.4f}"
                )

                if not outer.use_early_stopping:
                    return

                current = val_rmse if outer.monitor == 'val_rmse' else val_r2
                improved = (
                    current < outer.best_metric - outer.min_delta
                    if outer.mode == 'min'
                    else current > outer.best_metric + outer.min_delta
                )
                if improved:
                    outer.best_metric = current
                    outer.patience_counter = 0
                    if outer.restore_best_weights:
                        outer.best_state_dict = copy.deepcopy(model.state_dict())
                else:
                    outer.patience_counter += 1
                    logging.debug(
                        f"No improvement ({outer.monitor}). "
                        f"Patience {outer.patience_counter}/{outer.patience}"
                    )
                    if outer.patience_counter >= outer.patience:
                        logging.info(f"Early stopping triggered at epoch {epoch}.")
                        control.should_training_stop = True

            def on_train_end(self, args, state, control, model, **kwargs):
                if outer.restore_best_weights and outer.best_state_dict is not None:
                    model.load_state_dict(outer.best_state_dict)
                    logging.info("Chronos-2: restored best model weights.")

        return _Callback()


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

    # Compile model if not in federated mode (torch.compile causes state_dict key mismatch in FL)
    use_compile = config['model'].get('use_compile', True)
    if use_compile and not config.get('model', {}).get('fl', False):
        model = torch.compile(model)
        logging.debug("Model compiled with torch.compile")
    else:
        logging.debug("Skipping torch.compile (disabled or in FL mode)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: trainable={trainable_params:,}, total={total_params:,}")

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
    quantiles = config['model'].get('tft', {}).get('quantiles', None)
    if quantiles:
        criterion = lambda pred, tgt: _pinball_loss(pred, tgt, quantiles)
        median_idx = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))
    else:
        criterion = nn.MSELoss()
        median_idx = None

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
    is_tft = config['model']['name'] in ('tft', 'tcn-tft')

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

        # For multi-quantile TFT output use median quantile for point metrics
        if median_idx is not None and train_preds.ndim == 3:
            train_preds_point = train_preds[..., median_idx]
        else:
            train_preds_point = train_preds
        train_mse = np.mean((train_preds_point - train_targets) ** 2)
        train_rmse = np.sqrt(train_mse)
        train_mae = np.mean(np.abs(train_preds_point - train_targets))

        ss_res = np.sum((train_targets - train_preds_point) ** 2)
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

            # For multi-quantile TFT output use median quantile for point metrics
            if median_idx is not None and val_preds.ndim == 3:
                val_preds_point = val_preds[..., median_idx]
            else:
                val_preds_point = val_preds
            val_mse = np.mean((val_preds_point - val_targets) ** 2)
            val_rmse = np.sqrt(val_mse)
            val_mae = np.mean(np.abs(val_preds_point - val_targets))

            ss_res = np.sum((val_targets - val_preds_point) ** 2)
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


def create_data_generator(dfs, config, features, scaler_x=None, scaler_y=None):
    """
    Generator for memory-efficient data processing - framework agnostic.
    Reuses preprocessing.pipeline from TensorFlow version.
    """
    from . import preprocessing

    if scaler_x:
        config['scaler_x'] = scaler_x
    if scaler_y:
        config['scaler_y'] = scaler_y

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
            'scalers': prepared_data['scalers'],
            'nwp_raw_test': prepared_data.get('nwp_raw_test', None)
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
    nwp_raw_test = None
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

        # Combine NWP raw test data (for Skill_NWP calculation)
        nwp_raw_new = data_dict.get('nwp_raw_test', None)
        if nwp_raw_new is not None:
            if nwp_raw_test is None:
                nwp_raw_test = nwp_raw_new
            else:
                old_len = nwp_raw_test.shape[0]
                new_len = nwp_raw_new.shape[0]
                combined_shape = (old_len + new_len,) + nwp_raw_test.shape[1:]
                combined_nwp = np.empty(combined_shape, dtype=nwp_raw_test.dtype)
                combined_nwp[:old_len] = nwp_raw_test
                combined_nwp[old_len:] = nwp_raw_new
                del nwp_raw_test
                nwp_raw_test = combined_nwp

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

    return X_train, y_train, X_test, y_test, test_data, nwp_raw_test

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
