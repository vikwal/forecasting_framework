import ray
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from . import tools, models, preprocessing


def aggregate_scalers(client_stats):
    n_total = sum(s['n'] for s in client_stats)
    mu_global = sum(s['n'] * s['mu'] for s in client_stats) / n_total
    var_global = sum(s['n'] * (s['var'] + s['mu']**2) for s in client_stats) / n_total - mu_global**2
    var_global = np.maximum(var_global, 0)
    return mu_global, np.sqrt(var_global)


def load_federated_data(config, freq, features, target_col='power'):
    """
    Load data for federated learning based on client mapping in config.
    """
    if 'clients' not in config.get('fl', {}):
        raise ValueError("No 'clients' mapping found in config['fl']. Please define client-to-files mapping.")

    clients_data = {}
    clients_mapping = config['fl']['clients']
    base_path = config['data']['path']

    for client_id, file_list in clients_mapping.items():
        logging.info(f'Loading data for client: {client_id}')

        # Create temporary config for this client
        client_config = config.copy()
        client_config['data'] = config['data'].copy()
        client_config['data']['files'] = file_list

        # Load data for this client using existing get_data function
        client_files = preprocessing.get_data(data_dir=base_path,
                                            config=client_config,
                                            freq=freq,
                                            features=features)

        clients_data[client_id] = client_files

    return clients_data


def state_dict_to_numpy_list(state_dict):
    """Convert PyTorch state_dict to list of numpy arrays for aggregation"""
    return [v.cpu().numpy() for v in state_dict.values()]


def numpy_list_to_state_dict(numpy_list, reference_state_dict):
    """Convert list of numpy arrays back to state_dict format"""
    result = {}
    for (k, ref_v), np_v in zip(reference_state_dict.items(), numpy_list):
        tensor = torch.from_numpy(np_v)
        # Ensure same dtype and device as reference
        if ref_v.dtype != tensor.dtype:
            tensor = tensor.to(dtype=ref_v.dtype)
        result[k] = tensor.to(ref_v.device)
    return result


class ServerOptimizer:
    """
    Manages state and update logic for server-side optimizers like FedAvgM and FedAdam.
    Works with PyTorch state_dicts.
    """
    def __init__(self,
                 strategy: str,
                 hyperparameters: Dict[str, Any],
                 initial_weights: Dict[str, torch.Tensor]):
        self.strategy = strategy.lower()
        self.hyperparameters = hyperparameters
        self.step_count = 0

        # Get server optimizer hyperparameters (with defaults)
        self.server_lr = hyperparameters.get('server_lr', 1.0)
        self.beta_1 = hyperparameters.get('beta_1', 0.9)
        self.beta_2 = hyperparameters.get('beta_2', 0.999)
        self.tau = hyperparameters.get('tau', 1e-8)

        # Initialize state variables as numpy arrays for easier computation
        self.initial_weights_list = state_dict_to_numpy_list(initial_weights)
        self.server_state_v = [np.zeros_like(w) for w in self.initial_weights_list]

        if self.strategy in ['fedadam', 'fedyogi']:
            self.server_state_m = [np.zeros_like(w) for w in self.initial_weights_list]

        logging.info(f"[ServerOptimizer] Initialized for strategy '{self.strategy}' with server_lr={self.server_lr}, "
                     f"beta_1={self.beta_1}, beta_2={self.beta_2}, tau={self.tau}")

    def step(self,
             old_global_weights: Dict[str, torch.Tensor],
             aggregated_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform server optimization step.

        Args:
            old_global_weights: Global weights before the update
            aggregated_weights: Weights aggregated through FedAvg

        Returns:
            dict: Final global weights after server optimization
        """
        self.step_count += 1

        # Convert to numpy lists for computation
        old_weights_list = state_dict_to_numpy_list(old_global_weights)
        agg_weights_list = state_dict_to_numpy_list(aggregated_weights)

        # Calculate average update delta
        delta_t = [agg - old for agg, old in zip(agg_weights_list, old_weights_list)]

        if self.strategy == 'fedavgm':
            # FedAvgM Update Rule
            self.server_state_v[:] = [self.beta_1 * v + dt for v, dt in zip(self.server_state_v, delta_t)]
            final_weights_list = [old + self.server_lr * v for old, v in zip(old_weights_list, self.server_state_v)]
            logging.debug(f"[ServerOptimizer FedAvgM] Applied momentum update. Step: {self.step_count}")

        elif self.strategy == 'fedadam':
            # FedAdam Update Rule
            self.server_state_m[:] = [self.beta_1 * m + (1 - self.beta_1) * dt
                                      for m, dt in zip(self.server_state_m, delta_t)]
            self.server_state_v[:] = [self.beta_2 * v + (1 - self.beta_2) * np.square(dt)
                                      for v, dt in zip(self.server_state_v, delta_t)]

            # Bias correction
            beta_1_power = self.beta_1 ** self.step_count
            beta_2_power = self.beta_2 ** self.step_count
            m_hat = [m / (1 - beta_1_power) for m in self.server_state_m]
            v_hat = [v / (1 - beta_2_power) for v in self.server_state_v]

            final_weights_list = [old + self.server_lr * m_h / (np.sqrt(v_h) + self.tau)
                             for old, m_h, v_h in zip(old_weights_list, m_hat, v_hat)]
            logging.debug(f"[ServerOptimizer FedAdam] Applied Adam update. Step: {self.step_count}")

        elif self.strategy == 'fedyogi':
            logging.warning(f"FedYogi strategy not fully implemented yet.")
            final_weights_list = agg_weights_list

        elif self.strategy == 'fedadagrad':
            logging.warning(f"FedAdagrad strategy not fully implemented yet.")
            final_weights_list = agg_weights_list
        else:
            logging.error(f"Unknown server optimization strategy: {self.strategy}")
            final_weights_list = agg_weights_list

        # Convert back to state_dict
        return numpy_list_to_state_dict(final_weights_list, old_global_weights)


@ray.remote
class ClientActor:
    def __init__(self,
                 client_id: int,
                 X_train: Any,
                 y_train: Any,
                 X_val: Any,
                 y_val: Any,
                 config: Dict[str, Any],
                 hyperparameters: Dict[str, Any]):
        self.client_id = client_id
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.config = config
        self.hyperparameters = hyperparameters

        # Initialize GPU
        tools.initialize_gpu()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model within the actor
        self.model = models.get_model(config=self.config, hyperparameters=self.hyperparameters)
        self.model = self.model.to(self.device)

        self.logger = logging.getLogger(f"ClientActor_{self.client_id}")
        self.logger.setLevel(logging.INFO)
        logging.info(f"[ClientActor {self.client_id}] Initialized PyTorch model on {self.device}.")

    def train(self, global_weights: Dict[str, torch.Tensor]):
        """Performs model training for a communication round using PyTorch."""
        from . import tools
        import copy
        from torch.utils.data import TensorDataset, DataLoader

        verbose = self.config['fl'].get('verbose', False)
        personalize = self.config['fl'].get('personalize', False)

        # Load global weights (handle personalization if needed)
        if personalize:
            model_name = self.config['model']['name']
            # Split GLOBAL weights to get only shared parts
            shared_global, _ = self._split_weights_by_layer_name(global_weights, model_name)
            # Split LOCAL weights to get only personal parts
            _, personal_local = self._split_weights_by_layer_name(self.model.state_dict(), model_name)
            # Merge: shared from global + personal from local
            merged_weights = {**shared_global, **personal_local}
            self.model.load_state_dict(merged_weights, strict=False)
            logging.debug(f"[ClientActor {self.client_id}] Loaded {len(shared_global)} shared weights from global, {len(personal_local)} personal weights from local")
        else:
            self.model.load_state_dict(global_weights)

        # Determine number of local epochs
        fl_early_stopping = self.config.get('fl', {}).get('early_stopping', {})
        if fl_early_stopping.get('enabled', False):
            n_epochs = self.config.get('fl', {}).get('n_local_epochs', 1)
            patience = fl_early_stopping.get('patience', 10)
            min_delta = fl_early_stopping.get('min_delta', 0.0001)
            mode = fl_early_stopping.get('mode', 'min')
            restore_best = fl_early_stopping.get('restore_best_weights', True)
        else:
            n_epochs = self.config.get('fl', {}).get('n_local_epochs', 10)
            patience = None  # No early stopping

        # Create DataLoaders
        batch_size = self.hyperparameters['batch_size']
        is_tft = self.config['model']['name'] in ('tft', 'tcn-tft')

        if isinstance(self.X_train, dict):
            # TFT case
            tensors = [torch.from_numpy(self.X_train['observed']).float(),
                       torch.from_numpy(self.X_train['known']).float()]
            if 'static' in self.X_train:
                tensors.append(torch.from_numpy(self.X_train['static']).float())
            tensors.append(torch.from_numpy(self.y_train).float())
            train_dataset = TensorDataset(*tensors)
        else:
            train_dataset = TensorDataset(
                torch.from_numpy(self.X_train).float(),
                torch.from_numpy(self.y_train).float()
            )

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=self.config['model'].get('shuffle', True),
                                  drop_last=True, pin_memory=True)

        val_loader = None
        if self.X_val is not None and self.y_val is not None:
            if isinstance(self.X_val, dict):
                val_tensors = [torch.from_numpy(self.X_val['observed']).float(),
                               torch.from_numpy(self.X_val['known']).float()]
                if 'static' in self.X_val:
                    val_tensors.append(torch.from_numpy(self.X_val['static']).float())
                val_tensors.append(torch.from_numpy(self.y_val).float())
                val_dataset = TensorDataset(*val_tensors)
            else:
                val_dataset = TensorDataset(
                    torch.from_numpy(self.X_val).float(),
                    torch.from_numpy(self.y_val).float()
                )
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, drop_last=False, pin_memory=True)

        # Setup optimizer and loss — train the EXISTING self.model
        lr = self.hyperparameters.get('learning_rate', self.hyperparameters.get('lr', 0.001))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        quantiles = self.config['model'].get('tft', {}).get('quantiles', None)
        if quantiles:
            criterion = lambda pred, tgt: tools._pinball_loss(pred, tgt, quantiles)
            median_idx = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))
        else:
            criterion = nn.MSELoss()
            median_idx = None

        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': [],
                   'train_mae': [], 'val_mae': [], 'train_r2': [], 'val_r2': []}

        best_val = float('inf') if not patience or mode == 'min' else float('-inf')
        epochs_no_improve = 0
        best_state = None

        for epoch in range(n_epochs):
            # --- Training ---
            self.model.train()
            train_preds, train_targets = [], []

            for batch in train_loader:
                if is_tft:
                    if len(batch) == 4:
                        obs, known, static, targets = [b.to(self.device) for b in batch]
                        predictions = self.model(obs, known, static)
                    elif len(batch) == 3:
                        obs, known, targets = [b.to(self.device) for b in batch]
                        predictions = self.model(obs, known)
                else:
                    inputs, targets = [b.to(self.device) for b in batch]
                    predictions = self.model(inputs)

                loss = criterion(predictions, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_preds.append(predictions.detach().cpu().numpy())
                train_targets.append(targets.detach().cpu().numpy())

            # Compute train metrics
            train_preds = np.concatenate(train_preds)
            train_targets = np.concatenate(train_targets)
            if median_idx is not None and train_preds.ndim == 3:
                train_preds_point = train_preds[..., median_idx]
            else:
                train_preds_point = train_preds
            train_mse = np.mean((train_targets - train_preds_point) ** 2)
            train_mae = np.mean(np.abs(train_targets - train_preds_point))
            ss_res = np.sum((train_targets - train_preds_point) ** 2)
            ss_tot = np.sum((train_targets - np.mean(train_targets)) ** 2)
            train_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            history['train_loss'].append(train_mse)
            history['train_rmse'].append(np.sqrt(train_mse))
            history['train_mae'].append(train_mae)
            history['train_r2'].append(train_r2)

            # --- Validation ---
            if val_loader is not None:
                self.model.eval()
                val_preds, val_targets_list = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        if is_tft:
                            if len(batch) == 4:
                                obs, known, static, targets = [b.to(self.device) for b in batch]
                                predictions = self.model(obs, known, static)
                            elif len(batch) == 3:
                                obs, known, targets = [b.to(self.device) for b in batch]
                                predictions = self.model(obs, known)
                        else:
                            inputs, targets = [b.to(self.device) for b in batch]
                            predictions = self.model(inputs)

                        val_preds.append(predictions.cpu().numpy())
                        val_targets_list.append(targets.cpu().numpy())

                val_preds = np.concatenate(val_preds)
                val_targets_arr = np.concatenate(val_targets_list)
                if median_idx is not None and val_preds.ndim == 3:
                    val_preds_point = val_preds[..., median_idx]
                else:
                    val_preds_point = val_preds
                val_mse = np.mean((val_targets_arr - val_preds_point) ** 2)
                val_mae = np.mean(np.abs(val_targets_arr - val_preds_point))
                ss_res = np.sum((val_targets_arr - val_preds_point) ** 2)
                ss_tot = np.sum((val_targets_arr - np.mean(val_targets_arr)) ** 2)
                val_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                history['val_loss'].append(val_mse)
                history['val_rmse'].append(np.sqrt(val_mse))
                history['val_mae'].append(val_mae)
                history['val_r2'].append(val_r2)

                # Early stopping check
                if patience:
                    current_val = val_mse if mode == 'min' else val_r2
                    improved = (current_val < best_val - min_delta) if mode == 'min' else (current_val > best_val + min_delta)
                    if improved:
                        best_val = current_val
                        epochs_no_improve = 0
                        if restore_best:
                            best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            if restore_best and best_state:
                                self.model.load_state_dict(best_state)
                            break

        if verbose:
            logging.info(f"[ClientActor {self.client_id}] Training completed ({epoch+1} epochs).")

        # Extract final metrics
        metrics = {}
        for key, values in history.items():
            if len(values) > 0:
                metrics[key] = values[-1]

        results = {
            'client_id': self.client_id,
            'weights': self.model.state_dict(),
            'n_samples': len(self.y_train),
            'metrics': metrics,
            'history': history
        }
        return results

    def evaluate(self, global_weights: Dict[str, torch.Tensor]):
        """Performs local evaluation at the end of a communication round."""
        personalize = self.config['fl'].get('personalize', False)

        if personalize:
            model_name = self.config['model']['name']
            # Split GLOBAL weights to get only shared parts
            shared_global, _ = self._split_weights_by_layer_name(global_weights, model_name)
            # Split LOCAL weights to get only personal parts
            _, personal_local = self._split_weights_by_layer_name(self.model.state_dict(), model_name)
            # Merge: shared from global + personal from local
            merged_weights = {**shared_global, **personal_local}
            self.model.load_state_dict(merged_weights, strict=False)
        else:
            self.model.load_state_dict(global_weights)

        if self.X_val is not None and self.y_val is not None:
            self.model.eval()
            metrics = {}

            with torch.no_grad():
                # Prepare input
                if isinstance(self.X_val, dict):
                    # TFT case
                    X_val_tensors = {k: torch.from_numpy(v).float().to(self.device) for k, v in self.X_val.items()}
                    y_val_tensor = torch.from_numpy(self.y_val).float().to(self.device)

                    if 'static' in X_val_tensors:
                        predictions = self.model(X_val_tensors['observed'], X_val_tensors['known'], X_val_tensors['static'])
                    else:
                        predictions = self.model(X_val_tensors['observed'], X_val_tensors['known'])
                else:
                    X_val_tensor = torch.from_numpy(self.X_val).float().to(self.device)
                    y_val_tensor = torch.from_numpy(self.y_val).float().to(self.device)
                    predictions = self.model(X_val_tensor)

                # Calculate metrics
                mse = nn.MSELoss()(predictions, y_val_tensor).item()
                mae = nn.L1Loss()(predictions, y_val_tensor).item()
                rmse = np.sqrt(mse)

                pred_np = predictions.cpu().numpy()
                target_np = y_val_tensor.cpu().numpy()
                ss_res = np.sum((target_np - pred_np) ** 2)
                ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                metrics = {
                    'loss': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r^2': r2
                }
        else:
            logging.warning(f"[ClientActor {self.client_id}] No validation data available for evaluation.")
            metrics = None

        logging.info(f"[ClientActor {self.client_id}] Local evaluation completed.")
        results = {
            'client_id': self.client_id,
            'n_samples': len(self.y_val) if self.y_val is not None else 0,
            'metrics': metrics,
            'weights': self.model.state_dict()
        }
        return results

    def _split_weights_by_layer_name(self, state_dict, model_name):
        """
        Split weights into shared and personal based on model architecture.
        Uses get_shared_keys() as single source of truth for the split logic.

        Returns:
            shared_weights: Weights to be aggregated globally
            personal_weights: Weights kept locally per client
        """
        shared_keys = get_shared_keys(state_dict, model_name)
        shared_weights = {k: v for k, v in state_dict.items() if k in shared_keys}
        personal_weights = {k: v for k, v in state_dict.items() if k not in shared_keys}
        return shared_weights, personal_weights


# Shared layer patterns per model — single source of truth for personalization split.
# Edit here to change which layers are shared vs personal.
SHARED_LAYER_PATTERNS = {
    'tft': ['lstm_encoder', 'lstm_decoder', 'lstm_gate', 'lstm_ln'],
    # tcn-tft: TCN feature extractors + projection are shared globally;
    # VSN, LSTM, attention and all TFT core layers remain local.
    'tcn-tft': ['tcn_observed', 'tcn_known', 'tcn_proj'],
}


def get_shared_keys(state_dict: Dict[str, torch.Tensor], model_name: str) -> set:
    """
    Identify which keys in state_dict are shared (vs personal) for a given model.

    Args:
        state_dict: Model state dictionary
        model_name: Name of the model ('tcn-gru', 'tft', etc.)

    Returns:
        Set of keys that should be shared across clients
    """
    shared_keys = set()
    model_name_lower = model_name.lower()

    for key in state_dict.keys():
        is_shared = False

        if model_name_lower == 'tcn-gru':
            if 'conv_stack' in key:
                is_shared = True
            elif 'rnn' not in key and 'attention' not in key and 'output_fc' not in key:
                is_shared = True

        elif model_name_lower == 'tft':
            if any(s in key for s in SHARED_LAYER_PATTERNS['tft']):
                is_shared = True

        elif model_name_lower == 'tcn-tft':
            if any(s in key for s in SHARED_LAYER_PATTERNS['tcn-tft']):
                is_shared = True
        else:
            # Unknown model - aggregate all weights
            is_shared = True

        if is_shared:
            shared_keys.add(key)

    return shared_keys


def aggregate_weights(client_results: List[Dict[str, Any]], config: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
    """
    Aggregate weights from multiple clients using FedAvg (weighted average).
    Works with PyTorch state_dicts.

    Args:
        client_results: List of dictionaries containing client weights and metadata
        config: Configuration dictionary (optional, needed for personalization)

    Returns:
        Aggregated weights dictionary
    """
    total_samples = sum(client['n_samples'] for client in client_results)

    # Check if personalization is enabled
    personalize = False
    model_name = None
    if config is not None:
        personalize = config.get('fl', {}).get('personalize', False)
        if personalize:
            model_name = config['model']['name']

    # Initialize aggregated weights
    first_weights = client_results[0]['weights']

    if personalize and model_name:
        # Only aggregate SHARED weights when personalization is enabled
        shared_keys = get_shared_keys(first_weights, model_name)

        # Only initialize aggregation for shared keys
        weights_agg = {k: torch.zeros_like(v) for k, v in first_weights.items() if k in shared_keys}

        logging.debug(f"[Server] Aggregating {len(weights_agg)} shared weights (out of {len(first_weights)} total)")
    else:
        # No personalization - aggregate all weights
        weights_agg = {k: torch.zeros_like(v) for k, v in first_weights.items()}

    for client in client_results:
        client_weights = client['weights']
        num_samples = client['n_samples']
        weight_factor = num_samples / total_samples

        for key in weights_agg.keys():
            weights_agg[key] += client_weights[key] * weight_factor

    return weights_agg



def aggregate_metrics(client_results: List[Dict[str, Any]]):
    """Aggregate metrics from multiple clients (weighted by sample count)."""
    metrics_sum = {}
    metrics_total_samples = {}

    for client in client_results:
        num_samples = client['n_samples']
        metrics = client.get('metrics')

        if metrics is None:
            continue

        for metric, value in metrics.items():
            # Aggregate ALL metrics (both training and validation)
            if metric not in metrics_sum:
                metrics_sum[metric] = 0.0
                metrics_total_samples[metric] = 0.0
            metrics_sum[metric] += value * num_samples
            metrics_total_samples[metric] += num_samples

    metrics_agg = {}
    for metric, total_value in metrics_sum.items():
        metrics_agg[metric] = total_value / metrics_total_samples[metric]

    return metrics_agg if metrics_agg else None


def get_train_history(client_results: List[Dict[str, Any]]):
    """Extract training history from client results."""
    histories = {}
    for client in client_results:
        client_id = client['client_id']
        history = client.get('history')
        histories[client_id] = history
    return histories


def get_clients_weights(client_results: List[Dict[str, Any]]):
    """Extract weights of each client from the results."""
    clients_weights = {}
    for client in client_results:
        client_id = client['client_id']
        clients_weights[client_id] = client['weights']
    return clients_weights


def get_kfolds_partitions(n_splits, partitions):
    """
    Create k-fold partitions for time series cross-validation.
    Compatible with both numpy arrays and dictionaries (for TFT).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    raw_partitions = []

    for i, part in enumerate(partitions):
        X_train_orig = part[0]
        y_train = part[1]
        folds = []
        data_for_splitting = y_train

        for fold_idx, (train_index, val_index) in enumerate(tscv.split(data_for_splitting)):
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            if isinstance(X_train_orig, np.ndarray):
                X_train_fold = X_train_orig[train_index]
                X_val_fold = X_train_orig[val_index]
            elif isinstance(X_train_orig, dict):
                # Case for TFT
                X_train_fold = {}
                X_val_fold = {}
                for key, value_array in X_train_orig.items():
                    X_train_fold[key] = value_array[train_index]
                    X_val_fold[key] = value_array[val_index]

            folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
        raw_partitions.append(folds)

    kfolds_partitions = []
    for split_idx in range(n_splits):
        partition_for_this_fold = []
        for client_folds in raw_partitions:
            partition_for_this_fold.append(client_folds[split_idx])
        kfolds_partitions.append(partition_for_this_fold)

    return kfolds_partitions


def run_simulation(partitions: Any,
                   config: Dict[str, Any],
                   hyperparameters: Dict[str, Any],
                   client_ids = None):
    """
    Perform FL simulation using Ray and PyTorch.

    Args:
        partitions: Dictionary or list of client partitions
        config: Configuration dictionary
        hyperparameters: Training hyperparameters
        client_ids: Optional list of client IDs

    Returns:
        history: Training history
        clients_weights: Final weights for each client
    """

    if isinstance(partitions, list):
        if client_ids:
            if len(partitions) != len(client_ids):
                raise ValueError("Length of partitions and client_ids must match.")
            partitions = {client_id: part for client_id, part in zip(client_ids, partitions)}
        else:
            partitions = {i: part for i, part in enumerate(partitions)}

    n_clients = len(partitions)
    n_rounds = hyperparameters.get('n_rounds', config['fl'].get('n_rounds', 10))
    save_history = config['fl'].get('save_history', False)
    strategy = config['fl'].get('strategy', 'fedavg').lower()

    # GPU configuration
    total_num_gpus = torch.cuda.device_count()
    gpu_per_actor = min(1, total_num_gpus/n_clients) if total_num_gpus > 0 else 0

    logging.info(f"FL Simulation: {n_clients} clients, {n_rounds} rounds, {total_num_gpus} GPUs, strategy={strategy}")

    # Initialize Ray
    ray.init(
        num_gpus=total_num_gpus,
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level=logging.WARNING,
        log_to_driver=True
    )

    # Create client actors
    client_actors = []
    for key, value in partitions.items():
        X_train, y_train, X_val, y_val = value
        actor = ClientActor.options(num_gpus=gpu_per_actor).remote(
            client_id=key,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
            hyperparameters=hyperparameters
        )
        client_actors.append(actor)

    logging.info("Start FL simulation using Ray and PyTorch.")
    start_time = time.time()

    # Initialize global model
    logging.debug('[Server] Global model is initialized.')
    global_model = models.get_model(config=config, hyperparameters=hyperparameters)
    global_weights = global_model.state_dict()

    # Initialize server optimizer if needed
    server_optimizer = None
    if strategy != 'fedavg':
        # When personalization is enabled, only initialize with shared weights
        personalize = config.get('fl', {}).get('personalize', False)

        if personalize:
            model_name = config['model']['name']
            shared_keys = get_shared_keys(global_weights, model_name)
            shared_weights = {k: v for k, v in global_weights.items() if k in shared_keys}
            server_optimizer = ServerOptimizer(strategy, hyperparameters, shared_weights)
            logging.info(f"[Server] Initialized ServerOptimizer with {len(shared_weights)} shared weights (personalization enabled)")
        else:
            server_optimizer = ServerOptimizer(strategy, hyperparameters, global_weights)
            logging.info(f"[Server] Initialized ServerOptimizer with {len(global_weights)} weights")

    # Log shared layer patterns for personalization
    personalize = config.get('fl', {}).get('personalize', False)
    if personalize:
        model_name_log = config['model']['name'].lower()
        patterns = SHARED_LAYER_PATTERNS.get(model_name_log, 'all (no personalization logic defined)')
        logging.info(f"[Server] Shared layer patterns for '{model_name_log}': {patterns}")

    history = {}
    all_rounds_metrics_data = []

    # Global early stopping setup
    global_es_config = config['fl'].get('global_early_stopping', {})
    global_es_enabled = global_es_config.get('enabled', False)
    global_es_patience = global_es_config.get('patience', 10)
    global_es_min_delta = global_es_config.get('min_delta', 0.0001)
    global_es_monitor = global_es_config.get('monitor', 'val_rmse')
    global_es_mode = global_es_config.get('mode', 'min')
    best_global_val = float('inf') if global_es_mode == 'min' else float('-inf')
    global_es_counter = 0
    best_global_weights = None
    best_clients_weights = None
    best_round = 0

    if global_es_enabled:
        logging.info(f"[Global Early Stopping] Enabled — monitor={global_es_monitor}, patience={global_es_patience}, min_delta={global_es_min_delta}, mode={global_es_mode}")

    for round_num in range(1, n_rounds+1):
        history[round_num] = {}
        logging.info(f"--- Round {round_num}/{n_rounds} ---")
        round_start_time = time.time()

        # Client training
        logging.debug("[Server] Start train jobs on clients.")
        results_refs = [actor.train.remote(global_weights) for actor in client_actors]
        client_results = ray.get(results_refs)
        logging.debug(f"[Server] All clients trained. Start aggregation.")

        # Aggregate weights and metrics
        aggregated_weights = aggregate_weights(client_results, config)
        train_metrics_agg = aggregate_metrics(client_results)

        # Apply server-side optimization
        if server_optimizer:
            # When personalization is enabled, aggregated_weights only contains shared weights
            # We need to extract only shared weights from global_weights for proper shape matching
            personalize = config.get('fl', {}).get('personalize', False)

            if personalize:
                # Extract only the shared weights from global_weights that match aggregated_weights
                shared_global_weights = {k: v for k, v in global_weights.items() if k in aggregated_weights}
                new_shared_weights = server_optimizer.step(shared_global_weights, aggregated_weights)
                new_global_weights = new_shared_weights
            else:
                new_global_weights = server_optimizer.step(global_weights, aggregated_weights)
        else:
            new_global_weights = aggregated_weights

        # Save client histories
        if save_history:
            history[round_num]['history'] = get_train_history(client_results)

        logging.debug('[Server] Training metrics aggregated.')

        # Update global model
        if new_global_weights:
            personalize = config.get('fl', {}).get('personalize', False)

            if personalize:
                # When personalization is enabled, aggregated weights only contain shared weights
                # We need to merge them with existing personal weights from the global model
                current_global_state = global_model.state_dict()

                # Update only the keys present in new_global_weights (shared weights)
                for key, value in new_global_weights.items():
                    current_global_state[key] = value

                # Load the merged state dict
                global_model.load_state_dict(current_global_state)
                global_weights = current_global_state

                logging.debug(f"[Server] Updated {len(new_global_weights)} shared weights in global model")
            else:
                # No personalization - load all weights normally
                global_model.load_state_dict(new_global_weights)
                global_weights = new_global_weights
        else:
            logging.warning("[Server] Nothing to aggregate.")

        # Local evaluation
        logging.debug(f'[Server] Start local evaluation.')
        results_refs = [actor.evaluate.remote(global_weights) for actor in client_actors]
        client_results = ray.get(results_refs)

        # Aggregate evaluation metrics
        val_metrics_agg = aggregate_metrics(client_results)
        logging.debug(f'[Server] Evaluation metrics aggregated.')

        # Global early stopping check
        if global_es_enabled and val_metrics_agg is not None:
            monitor_key = global_es_monitor[len('val_'):] if global_es_monitor.startswith('val_') else global_es_monitor
            current_val = val_metrics_agg.get(monitor_key)
            if current_val is not None:
                improved = (current_val < best_global_val - global_es_min_delta) if global_es_mode == 'min' \
                           else (current_val > best_global_val + global_es_min_delta)
                if improved:
                    best_global_val = current_val
                    global_es_counter = 0
                    best_round = round_num
                    best_global_weights = {k: v.clone() for k, v in global_weights.items()}
                    best_clients_weights = get_clients_weights(client_results)
                    logging.info(f"[Global Early Stopping] Round {round_num}: {global_es_monitor}={current_val:.6f} improved. New best.")
                else:
                    global_es_counter += 1
                    logging.info(f"[Global Early Stopping] Round {round_num}: {global_es_monitor}={current_val:.6f} no improvement. Counter: {global_es_counter}/{global_es_patience}")
                    if global_es_counter >= global_es_patience:
                        logging.info(f"[Global Early Stopping] Early stopping triggered after round {round_num}. Best round: {best_round}, {global_es_monitor}={best_global_val:.6f}")
                        # Store round data before breaking
                        round_data = {'round': round_num}
                        if train_metrics_agg:
                            for key, value in train_metrics_agg.items():
                                round_data[key if key.startswith('train_') else f'train_{key}'] = value
                        if val_metrics_agg:
                            for key, value in val_metrics_agg.items():
                                round_data[f'val_{key}'] = value
                        all_rounds_metrics_data.append(round_data)
                        break

        # Collect round data
        round_data = {'round': round_num}
        if train_metrics_agg:
            for key, value in train_metrics_agg.items():
                # Don't add 'train_' prefix if it already exists
                if key.startswith('train_'):
                    round_data[key] = value
                else:
                    round_data[f'train_{key}'] = value
        if val_metrics_agg:
            for key, value in val_metrics_agg.items():
                round_data[f'val_{key}'] = value
        all_rounds_metrics_data.append(round_data)

        round_end_time = time.time()
        logging.info(f"Round {round_num} terminated in {round_end_time - round_start_time:.2f} seconds.")

    end_time = time.time()
    logging.info(f"--- Simulation terminated in {end_time - start_time:.2f} seconds ---")
    ray.shutdown()

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_rounds_metrics_data)
    metrics_df = metrics_df.set_index('round')
    history['metrics_aggregated'] = metrics_df

    # Restore best weights if global early stopping was used
    if global_es_enabled and best_global_weights is not None:
        logging.info(f"[Global ES] Restoring best global weights from round {best_round} ({global_es_monitor}={best_global_val:.6f})")
        clients_weights = best_clients_weights
    else:
        clients_weights = get_clients_weights(client_results)

    return history, clients_weights