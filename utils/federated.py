import ray
import math
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
        logging.debug(f'Loading data for client: {client_id}')

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

    def step_from_gradients(self,
                            global_weights: Dict[str, torch.Tensor],
                            aggregated_gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """FedSGD: apply server optimizer using raw gradients.
        Converts to pseudo weight-delta (delta = -grad) then delegates to step().
        Sign: pseudo = old - grad → delta_t = pseudo - old = -grad
        Adam: m_t += (1-β₁)·(-grad), final = old + lr·m_hat/... = old - lr·grad/... ✓
        """
        pseudo_aggregated = {
            k: global_weights[k].cpu() - g.cpu()
            for k, g in aggregated_gradients.items()
        }
        return self.step(global_weights, pseudo_aggregated)


def _log_gpu_mem(tag: str, device, client_id):
    """DEBUG HELPER — log GPU memory stats at key points. Remove when done debugging.

    Uses a dedicated 'gpumem' logger set to INFO so it is not suppressed by
    callers that temporarily raise the root logger to WARNING (e.g. hpo_fl.py).
    """
    import os
    _gpumem_logger = logging.getLogger('gpumem')
    if _gpumem_logger.level == logging.NOTSET or _gpumem_logger.level > logging.INFO:
        _gpumem_logger.setLevel(logging.INFO)
        if not _gpumem_logger.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            _gpumem_logger.addHandler(_h)
        _gpumem_logger.propagate = False

    pid = os.getpid()
    if torch.cuda.is_available():
        try:
            if device is not None and hasattr(device, 'type') and device.type == 'cuda':
                idx = device.index if device.index is not None else torch.cuda.current_device()
            else:
                idx = torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(idx) / 1024**3
            reserved = torch.cuda.memory_reserved(idx) / 1024**3
            total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
            free = total - reserved
            _gpumem_logger.info(
                f"[GPUMEM] [{tag}] client={client_id} pid={pid} "
                f"gpu={idx} alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB free_est={free:.2f}GiB total={total:.1f}GiB"
            )
        except Exception as e:
            _gpumem_logger.info(f"[GPUMEM] [{tag}] client={client_id} pid={pid} — error querying GPU: {e}")
    else:
        _gpumem_logger.info(f"[GPUMEM] [{tag}] client={client_id} pid={pid} — no CUDA available")


@ray.remote
class ClientActor:
    def __init__(self,
                 client_id: int,
                 X_train: Any,
                 y_train: Any,
                 X_val: Any,
                 y_val: Any,
                 config: Dict[str, Any],
                 hyperparameters: Dict[str, Any],
                 initial_personal_weights: Dict[str, Any] = None,
                 initial_pretrained_weights: Dict[str, Any] = None):
        self.client_id = client_id
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.config = config
        self.hyperparameters = hyperparameters

        # Must be set before the first CUDA memory allocation (allocator reads it once).
        import os
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

        # In Ray Actors, GPU is already allocated via Ray resource scheduling.
        # Do NOT call tools.initialize_gpu() as it would override Ray's CUDA_VISIBLE_DEVICES.
        # Ray sets CUDA_VISIBLE_DEVICES automatically for the Actor's GPU slice.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model stays on CPU at init; moved to GPU lazily in train()/evaluate()
        # and returned to CPU afterwards so idle actors don't occupy GPU memory.
        self.model = models.get_model(config=self.config, hyperparameters=self.hyperparameters)
        # Names of params loaded from a pretrained model (used for per-group lr in train()).
        self._pretrained_param_names: set = set()

        if initial_pretrained_weights is not None:
            freeze_pt = config['fl'].get('pretrained_weights_freeze', True)
            sd = self.model.state_dict()
            matched_pt = {k: v for k, v in initial_pretrained_weights.items() if k in sd}
            if matched_pt:
                sd.update(matched_pt)
                self.model.load_state_dict(sd)
                self._pretrained_param_names = set(matched_pt.keys())
            if freeze_pt:
                for name, param in self.model.named_parameters():
                    if name in self._pretrained_param_names:
                        param.requires_grad_(False)
            n_frozen = sum(1 for n, p in self.model.named_parameters()
                           if n in self._pretrained_param_names and not p.requires_grad)
            pt_lr_val = config['fl'].get('pretrained_weights_lr', 'default*0.1')
            logging.info(
                f"[ClientActor {client_id}] pretrained_weights: loaded {len(matched_pt)} tensors, "
                f"{'frozen' if freeze_pt else f'trainable at pretrained_lr={pt_lr_val}'} "
                f"({n_frozen} params frozen)"
            )

        if initial_personal_weights is not None:
            model_name = config['model']['name']
            shared_keys = get_shared_keys(self.model.state_dict(), model_name, config=config)
            sd = self.model.state_dict()
            matched = {k: v for k, v in initial_personal_weights.items()
                       if k in sd and k not in shared_keys}
            if matched:
                sd.update(matched)
                self.model.load_state_dict(sd)
            # Freeze personal parameters so the optimizer skips them every round.
            # load_state_dict copies data in-place, so requires_grad stays False.
            for name, param in self.model.named_parameters():
                if name not in shared_keys:
                    param.requires_grad_(False)
            n_frozen = sum(1 for _, p in self.model.named_parameters() if not p.requires_grad)
            logging.info(f"[ClientActor {client_id}] pretrained_context: loaded {len(matched)} "
                         f"personal tensors, froze {n_frozen} parameters")

        self.logger = logging.getLogger(f"ClientActor_{self.client_id}")
        self.logger.setLevel(logging.INFO)
        # _log_gpu_mem("init", self.device, self.client_id)
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')
        cuda_count = torch.cuda.device_count()
        logging.info(f"[ClientActor {self.client_id}] device={self.device}  CUDA_VISIBLE_DEVICES={cuda_visible}  cuda_device_count={cuda_count}")

    def train(self, global_weights: Dict[str, torch.Tensor]):
        """Performs model training for a communication round using PyTorch."""
        from . import tools
        import copy
        import os
        from torch.utils.data import TensorDataset, DataLoader

        _train_start = time.time()
        _phys_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')
        logging.info(f"[TIMING] client={self.client_id} pid={os.getpid()} phys_gpu={_phys_gpu} train_START")

        # _log_gpu_mem("train_before_to_device", self.device, self.client_id)
        self.model = self.model.to(self.device)
        # _log_gpu_mem("train_after_to_device", self.device, self.client_id)

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
        freeze_pt     = self.config['fl'].get('pretrained_weights_freeze', True)
        pretrained_lr = self.config['fl'].get('pretrained_weights_lr', lr * 0.1)

        if self._pretrained_param_names and not freeze_pt:
            # Two param groups: pretrained layers at lower lr, everything else at main lr.
            pretrained_params = [p for n, p in self.model.named_parameters()
                                 if p.requires_grad and n in self._pretrained_param_names]
            other_params      = [p for n, p in self.model.named_parameters()
                                 if p.requires_grad and n not in self._pretrained_param_names]
            param_groups = []
            if other_params:
                param_groups.append({'params': other_params,      'lr': lr})
            if pretrained_params:
                param_groups.append({'params': pretrained_params, 'lr': pretrained_lr})
            optimizer = torch.optim.Adam(param_groups)
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=lr)
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

            # if epoch == 0:
            #     _log_gpu_mem("train_epoch0_start", self.device, self.client_id)

            for batch_idx, batch in enumerate(train_loader):
                if is_tft:
                    if len(batch) == 4:
                        obs, known, static, targets = [b.to(self.device) for b in batch]
                        # if epoch == 0 and batch_idx == 0:
                        #     _log_gpu_mem("train_batch0_before_forward", self.device, self.client_id)
                        #     logging.info(f"[GPUMEM] [train_batch0_shapes] client={self.client_id} obs={tuple(obs.shape)} known={tuple(known.shape)} static={tuple(static.shape)}")
                        predictions = self.model(obs, known, static)
                    elif len(batch) == 3:
                        obs, known, targets = [b.to(self.device) for b in batch]
                        # if epoch == 0 and batch_idx == 0:
                        #     _log_gpu_mem("train_batch0_before_forward", self.device, self.client_id)
                        #     logging.info(f"[GPUMEM] [train_batch0_shapes] client={self.client_id} obs={tuple(obs.shape)} known={tuple(known.shape)}")
                        predictions = self.model(obs, known)
                else:
                    inputs, targets = [b.to(self.device) for b in batch]
                    # if epoch == 0 and batch_idx == 0:
                    #     _log_gpu_mem("train_batch0_before_forward", self.device, self.client_id)
                    #     logging.info(f"[GPUMEM] [train_batch0_shapes] client={self.client_id} inputs={tuple(inputs.shape)}")
                    predictions = self.model(inputs)

                # if epoch == 0 and batch_idx == 0:
                #     _log_gpu_mem("train_batch0_after_forward", self.device, self.client_id)

                loss = criterion(predictions, targets)

                # FedProx proximal term: (μ/2) * ||w_local - w_global||²
                _mu = self.config.get('fl', {}).get('fedprox', {}).get('mu', 0.0)
                if _mu > 0.0:
                    _prox = torch.tensor(0.0, device=self.device)
                    _personalize = self.config.get('fl', {}).get('personalize', False)
                    if _personalize:
                        _model_name = self.config['model']['name']
                        _eligible = set(self._split_weights_by_layer_name(global_weights, _model_name)[0].keys())
                    else:
                        _eligible = set(global_weights.keys())
                    for _name, _param in self.model.named_parameters():
                        if _param.requires_grad and _name in _eligible:
                            _prox = _prox + ((_param - global_weights[_name].to(self.device)) ** 2).sum()
                    loss = loss + (_mu / 2.0) * _prox

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_preds.append(predictions.detach().cpu().numpy())
                train_targets.append(targets.detach().cpu().numpy())

            # if epoch == 0:
            #     _log_gpu_mem("train_epoch0_end", self.device, self.client_id)

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

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

        # _log_gpu_mem("train_before_cpu_offload", self.device, self.client_id)
        results = {
            'client_id': self.client_id,
            'weights': self.model.state_dict(),
            'n_samples': len(self.y_train),
            'metrics': metrics,
            'history': history
        }

        self.model = self.model.to('cpu')
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        # _log_gpu_mem("train_after_cpu_offload", self.device, self.client_id)

        _train_dur = time.time() - _train_start
        logging.info(f"[TIMING] client={self.client_id} pid={os.getpid()} phys_gpu={_phys_gpu} train_END dur={_train_dur:.1f}s")

        return results

    def evaluate(self, global_weights: Dict[str, torch.Tensor]):
        """Performs local evaluation at the end of a communication round."""
        # _log_gpu_mem("eval_before_to_device", self.device, self.client_id)
        self.model = self.model.to(self.device)
        # _log_gpu_mem("eval_after_to_device", self.device, self.client_id)

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
                    # _log_gpu_mem("eval_before_data_to_device", self.device, self.client_id)
                    X_val_tensors = {k: torch.from_numpy(v).float().to(self.device) for k, v in self.X_val.items()}
                    y_val_tensor = torch.from_numpy(self.y_val).float().to(self.device)
                    # _log_gpu_mem("eval_after_data_to_device", self.device, self.client_id)

                    if 'static' in X_val_tensors:
                        predictions = self.model(X_val_tensors['observed'], X_val_tensors['known'], X_val_tensors['static'])
                    else:
                        predictions = self.model(X_val_tensors['observed'], X_val_tensors['known'])
                else:
                    # _log_gpu_mem("eval_before_data_to_device", self.device, self.client_id)
                    X_val_tensor = torch.from_numpy(self.X_val).float().to(self.device)
                    y_val_tensor = torch.from_numpy(self.y_val).float().to(self.device)
                    # _log_gpu_mem("eval_after_data_to_device", self.device, self.client_id)
                    predictions = self.model(X_val_tensor)
                # _log_gpu_mem("eval_after_forward", self.device, self.client_id)

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
        # _log_gpu_mem("eval_before_cpu_offload", self.device, self.client_id)
        results = {
            'client_id': self.client_id,
            'n_samples': len(self.y_val) if self.y_val is not None else 0,
            'metrics': metrics,
            'weights': self.model.state_dict()
        }

        self.model = self.model.to('cpu')
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        # _log_gpu_mem("eval_after_cpu_offload", self.device, self.client_id)

        return results

    def compute_gradient(self, global_weights: Dict[str, torch.Tensor]) -> Dict:
        """FedSGD: sample one mini-batch and return raw gradients (no optimizer.step)."""
        from . import tools
        from torch.utils.data import TensorDataset, DataLoader

        self.model = self.model.to(self.device)
        personalize = self.config['fl'].get('personalize', False)

        # Load weights (same personalization logic as train())
        if personalize:
            model_name = self.config['model']['name']
            shared_global, _ = self._split_weights_by_layer_name(global_weights, model_name)
            _, personal_local = self._split_weights_by_layer_name(self.model.state_dict(), model_name)
            self.model.load_state_dict({**shared_global, **personal_local}, strict=False)
        else:
            self.model.load_state_dict(global_weights)

        # Build DataLoader (same as train() L320–340)
        batch_size = self.hyperparameters['batch_size']
        is_tft = self.config['model']['name'] in ('tft', 'tcn-tft')
        if isinstance(self.X_train, dict):
            tensors = [torch.from_numpy(self.X_train['observed']).float(),
                       torch.from_numpy(self.X_train['known']).float()]
            if 'static' in self.X_train:
                tensors.append(torch.from_numpy(self.X_train['static']).float())
            tensors.append(torch.from_numpy(self.y_train).float())
        else:
            tensors = [torch.from_numpy(self.X_train).float(),
                       torch.from_numpy(self.y_train).float()]
        train_loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                                  shuffle=self.config['model'].get('shuffle', True),
                                  drop_last=True)

        # Loss (same as train() L379–384)
        quantiles = self.config['model'].get('tft', {}).get('quantiles', None)
        if quantiles:
            criterion = lambda pred, tgt: tools._pinball_loss(pred, tgt, quantiles)
        else:
            criterion = nn.MSELoss()

        # One batch: forward + backward, NO optimizer.step()
        self.model.train()
        self.model.zero_grad()
        batch = next(iter(train_loader))

        if is_tft:
            if len(batch) == 4:
                obs, known, static, targets = [b.to(self.device) for b in batch]
                predictions = self.model(obs, known, static)
            else:
                obs, known, targets = [b.to(self.device) for b in batch]
                predictions = self.model(obs, known)
        else:
            inputs, targets = [b.to(self.device) for b in batch]
            predictions = self.model(inputs)

        loss = criterion(predictions, targets)
        loss.backward()

        # Optional gradient clipping
        grad_clip = self.config.get('fl', {}).get('fedsgd', {}).get('gradient_clip', None)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        # Personalization: apply local gradient step for personal layers,
        # then return only shared layer gradients to the server.
        # Personal layers are updated in-place on self.model (persistent across rounds).
        if personalize:
            shared_keys = set(self._split_weights_by_layer_name(
                global_weights, self.config['model']['name'])[0].keys())
            _plr = self.config.get('fl', {}).get('fedsgd', {}).get('personal_lr')
            personal_lr = _plr if _plr is not None else self.hyperparameters.get('learning_rate', 0.001)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name not in shared_keys:
                        param.data.sub_(personal_lr * param.grad)
        else:
            shared_keys = None

        # Extract gradients to send to server (shared only when personalize=True)
        gradients = {
            name: (param.grad.detach().cpu().clone() if param.grad is not None
                   else torch.zeros_like(param.data.cpu()))
            for name, param in self.model.named_parameters()
            if shared_keys is None or name in shared_keys
        }

        self.model.cpu()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return {
            'client_id': self.client_id,
            'gradients': gradients,
            'n_samples': targets.shape[0],
            'metrics': {'train_loss': loss.item()},
            'history': {'train_loss': [loss.item()]}
        }

    def _split_weights_by_layer_name(self, state_dict, model_name):
        """
        Split weights into shared and personal based on model architecture.
        Uses get_shared_keys() as single source of truth for the split logic.

        Returns:
            shared_weights: Weights to be aggregated globally
            personal_weights: Weights kept locally per client
        """
        shared_keys = get_shared_keys(state_dict, model_name, config=self.config)
        shared_weights = {k: v for k, v in state_dict.items() if k in shared_keys}
        personal_weights = {k: v for k, v in state_dict.items() if k not in shared_keys}
        return shared_weights, personal_weights


# Shared layer patterns per model — single source of truth for personalization split.
# Edit here to change which layers are shared vs personal.
SHARED_LAYER_PATTERNS = {
    'tft': [#'static_embed', 'static_variable_selection', 'static_context_grn',
            #'static_enrichment_grn', 'static_state_h_grn',
            #'observed_embed',
            #'known_embed',
            #'vsn_past',
            #'vsn_future',
            'lstm_encoder',
            'lstm_decoder',
            'lstm_gate',
            'lstm_ln',
            'enrichment_grn',
            'attn_gate',
            'attn_ln',
            'positionwise_grn',
            #'output_gate',
            #'output_ln',
            #'output_layer',
            ],
    # tcn-tft: TCN feature extractors + projection are shared globally;
    # VSN, LSTM, attention and all TFT core layers remain local.
    'tcn-tft': ['tcn_observed', 'tcn_known', 'tcn_proj'],
}


def get_shared_keys(state_dict: Dict[str, torch.Tensor], model_name: str,
                    config: Dict[str, Any] = None) -> set:
    """
    Identify which keys in state_dict are shared (vs personal) for a given model.

    Pattern priority:
      1. fl.shared_layer_patterns.<model_name> from config (if present)
      2. Hardcoded SHARED_LAYER_PATTERNS fallback (with tcn-gru negative-match logic)

    Args:
        state_dict: Model state dictionary
        model_name: Name of the model ('tcn-gru', 'tft', etc.)
        config: Full experiment config dict (optional).  If supplied and
                fl.shared_layer_patterns contains an entry for model_name,
                that list of substrings is used instead of the hardcoded dict.

    Returns:
        Set of keys that should be shared across clients
    """
    model_name_lower = model_name.lower()

    # --- 1. Config-driven patterns (highest priority) ---
    if config is not None:
        cfg_patterns = config.get('fl', {}).get('shared_layer_patterns', {})
        if model_name_lower in cfg_patterns:
            patterns = cfg_patterns[model_name_lower]
            return {k for k in state_dict if any(p in k for p in patterns)}

    # --- 2. Hardcoded fallback ---
    shared_keys = set()
    for key in state_dict.keys():
        is_shared = False

        if model_name_lower == 'tcn-gru':
            if 'conv_stack' in key:
                is_shared = True
            elif 'rnn' not in key and 'attention' not in key and 'output_fc' not in key:
                is_shared = True

        elif model_name_lower in SHARED_LAYER_PATTERNS:
            if any(s in key for s in SHARED_LAYER_PATTERNS[model_name_lower]):
                is_shared = True
        else:
            # Unknown model — aggregate all weights
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
        shared_keys = get_shared_keys(first_weights, model_name, config=config)

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



def aggregate_gradients(
    client_results: List[Dict[str, Any]],
    config: Dict[str, Any] = None
) -> Dict[str, torch.Tensor]:
    """Weighted or uniform average of raw gradients (FedSGD)."""
    weighting = 'uniform'
    if config:
        weighting = config.get('fl', {}).get('fedsgd', {}).get('gradient_weighting', 'uniform')

    n_clients = len(client_results)
    total_samples = sum(r['n_samples'] for r in client_results)
    aggregated = {}

    for key in client_results[0]['gradients'].keys():
        if weighting == 'weighted':
            aggregated[key] = sum(
                r['gradients'][key] * (r['n_samples'] / total_samples)
                for r in client_results
            )
        else:  # uniform
            aggregated[key] = sum(r['gradients'][key] for r in client_results) / n_clients

    return aggregated


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


def evaluate_model_globally(model, X_val, y_val, config, hyperparameters):
    """
    Evaluate a model once on a global validation set (server-side, no Ray).
    Returns a metrics dict with the same keys as ClientActor.evaluate().
    """
    from torch.utils.data import TensorDataset, DataLoader

    device = next(model.parameters()).device
    batch_size = hyperparameters.get('batch_size', 128)
    is_tft = config['model']['name'] in ('tft', 'tcn-tft')
    quantiles = config['model'].get('tft', {}).get('quantiles', None)
    median_idx = None
    if quantiles:
        median_idx = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))

    if isinstance(X_val, dict):
        val_tensors = [torch.from_numpy(X_val['observed']).float(),
                       torch.from_numpy(X_val['known']).float()]
        if 'static' in X_val:
            val_tensors.append(torch.from_numpy(X_val['static']).float())
        val_tensors.append(torch.from_numpy(y_val).float())
        val_dataset = TensorDataset(*val_tensors)
    else:
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float()
        )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    val_preds, val_targets_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            if is_tft:
                if len(batch) == 4:
                    obs, known, static, targets = [b.to(device) for b in batch]
                    predictions = model(obs, known, static)
                elif len(batch) == 3:
                    obs, known, targets = [b.to(device) for b in batch]
                    predictions = model(obs, known)
            else:
                inputs, targets = [b.to(device) for b in batch]
                predictions = model(inputs)
            val_preds.append(predictions.cpu().numpy())
            val_targets_list.append(targets.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets_list)
    if median_idx is not None and val_preds.ndim == 3:
        val_preds = val_preds[..., median_idx]

    mse = float(np.mean((val_targets - val_preds) ** 2))
    mae = float(np.mean(np.abs(val_targets - val_preds)))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((val_targets - val_preds) ** 2)
    ss_tot = np.sum((val_targets - np.mean(val_targets)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    return {'loss': mse, 'mae': mae, 'rmse': rmse, 'r^2': r2}


def run_simulation(partitions: Any,
                   config: Dict[str, Any],
                   hyperparameters: Dict[str, Any],
                   client_ids = None,
                   global_val_data = None,
                   initial_personal_weights: Dict[str, Any] = None,
                   initial_pretrained_weights: Dict[str, Any] = None):
    """
    Perform FL simulation using Ray and PyTorch.

    Args:
        partitions: Dictionary or list of client partitions
        config: Configuration dictionary
        hyperparameters: Training hyperparameters
        client_ids: Optional list of client IDs
        global_val_data: Optional (X_val, y_val) tuple for server-side global evaluation.
            When provided and personalize=False, the global model is evaluated once
            server-side instead of running evaluate() on every client actor.

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
    # gpu_per_actor is derived from actors-per-GPU (ceiling), then floored to 1 decimal.
    # This ensures per-GPU packing works: ceil(n_clients/n_gpus) actors each need
    # floor(1/actors_per_gpu * 10)/10 GPU fractions → sum per GPU ≤ 1.0.
    total_num_gpus = torch.cuda.device_count()
    if total_num_gpus > 0:
        _actors_per_gpu = math.ceil(n_clients / total_num_gpus)
        gpu_per_actor = max(0.1, min(1.0, math.floor(1.0 / _actors_per_gpu * 10) / 10))
    else:
        gpu_per_actor = 0
    max_concurrent = config['fl'].get('max_concurrent_actors', n_clients)

    # Validate max_concurrent_actors
    if max_concurrent <= 0:
        logging.warning(
            f"max_concurrent_actors={max_concurrent} is invalid (must be > 0). "
            f"Setting to n_clients={n_clients}."
        )
        max_concurrent = n_clients
    elif max_concurrent > n_clients:
        logging.warning(
            f"max_concurrent_actors={max_concurrent} > n_clients={n_clients}. "
            f"Capping to {n_clients}."
        )
        max_concurrent = n_clients

    # Check if CUDA_VISIBLE_DEVICES was already set (would break Ray GPU distribution)
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        logging.warning(
            f"[Ray-GPU] CUDA_VISIBLE_DEVICES is already set to '{os.environ.get('CUDA_VISIBLE_DEVICES')}'. "
            f"This will prevent Ray from properly distributing GPUs to Actors. "
            f"Clearing it to allow Ray GPU management."
        )
        del os.environ['CUDA_VISIBLE_DEVICES']

    logging.info(
        f"FL Simulation: {n_clients} clients, {n_rounds} rounds, {total_num_gpus} GPUs "
        f"(gpu_per_actor={gpu_per_actor}), strategy={strategy}, max_concurrent_actors={max_concurrent}"
    )

    # Initialize Ray — cap the CPU worker pool to avoid spawning idle workers
    # proportional to the full core count. Actors only need one slot each.
    ray.init(
        num_cpus=max(n_clients, max_concurrent),
        num_gpus=total_num_gpus,
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level=logging.WARNING,
        log_to_driver=True,
        runtime_env={"env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}}
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
            hyperparameters=hyperparameters,
            initial_personal_weights=initial_personal_weights,
            initial_pretrained_weights=initial_pretrained_weights,
        )
        client_actors.append(actor)

    logging.info("Start FL simulation using Ray and PyTorch.")
    start_time = time.time()

    # Initialize global model — on GPU if available so server-side evaluation runs on GPU
    logging.debug('[Server] Global model is initialized.')
    server_device = torch.device('cuda:0' if total_num_gpus > 0 else 'cpu')
    global_model = models.get_model(config=config, hyperparameters=hyperparameters)
    # _log_gpu_mem("server_before_global_model_to_device", server_device, "server")
    global_model = global_model.to(server_device)
    # _log_gpu_mem("server_after_global_model_to_device", server_device, "server")
    global_weights = global_model.state_dict()

    # Initialize server optimizer if needed
    server_optimizer = None
    if strategy != 'fedavg':
        # When personalization is enabled, only initialize with shared weights
        personalize = config.get('fl', {}).get('personalize', False)

        # FedSGD delegates to fedadam/fedavgm server optimizer internally
        _opt_strategy = strategy
        if strategy == 'fedsgd':
            _sgd_sub = config['fl'].get('fedsgd', {}).get('server_optimizer', 'adam')
            _opt_strategy = 'fedadam' if _sgd_sub == 'adam' else 'fedavgm'

        if personalize:
            model_name = config['model']['name']
            shared_keys = get_shared_keys(global_weights, model_name, config=config)
            shared_weights = {k: v for k, v in global_weights.items() if k in shared_keys}
            server_optimizer = ServerOptimizer(_opt_strategy, hyperparameters, shared_weights)
            logging.info(f"[Server] Initialized ServerOptimizer({_opt_strategy}) with {len(shared_weights)} shared weights (personalization enabled)")
        else:
            server_optimizer = ServerOptimizer(_opt_strategy, hyperparameters, global_weights)
            logging.info(f"[Server] Initialized ServerOptimizer({_opt_strategy}) with {len(global_weights)} weights")

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
        logging.debug(f"--- Round {round_num}/{n_rounds} ---")
        round_start_time = time.time()

        # Client training — processed in batches to limit concurrent GPU usage
        # global_weights must be on CPU before passing to workers: if the server model
        # lives on GPU, Ray serialises CUDA tensors and every worker re-allocates them
        # on GPU during deserialisation, causing a spike proportional to n_concurrent.
        global_weights_cpu = {k: v.cpu() for k, v in global_weights.items()}
        client_results = []

        if strategy == 'fedsgd':
            # FedSGD: clients return raw gradients from a single mini-batch
            logging.debug(f"[Server] Start FedSGD gradient jobs on clients (batch_size={max_concurrent}).")
            for batch_start in range(0, n_clients, max_concurrent):
                batch_actors = client_actors[batch_start:batch_start + max_concurrent]
                batch_results = ray.get([a.compute_gradient.remote(global_weights_cpu)
                                         for a in batch_actors])
                client_results.extend(batch_results)
            logging.debug(f"[Server] All gradients collected. Aggregating.")
            aggregated_gradients = aggregate_gradients(client_results, config)
            train_metrics_agg = aggregate_metrics(client_results)
            personalize = config.get('fl', {}).get('personalize', False)
            if personalize:
                shared_global_weights = {k: v for k, v in global_weights.items() if k in aggregated_gradients}
                new_global_weights = server_optimizer.step_from_gradients(shared_global_weights, aggregated_gradients)
            else:
                new_global_weights = server_optimizer.step_from_gradients(global_weights, aggregated_gradients)
        else:
            # FedAvg / FedAdam / FedAvgM: clients return updated weights after local training
            logging.debug(f"[Server] Start train jobs on clients (batch_size={max_concurrent}).")
            for batch_start in range(0, n_clients, max_concurrent):
                batch_actors = client_actors[batch_start:batch_start + max_concurrent]
                batch_results = ray.get([a.train.remote(global_weights_cpu) for a in batch_actors])
                client_results.extend(batch_results)
            logging.debug(f"[Server] All clients trained. Start aggregation.")
            aggregated_weights = aggregate_weights(client_results, config)
            train_metrics_agg = aggregate_metrics(client_results)
            if server_optimizer:
                personalize = config.get('fl', {}).get('personalize', False)
                if personalize:
                    shared_global_weights = {k: v for k, v in global_weights.items() if k in aggregated_weights}
                    new_global_weights = server_optimizer.step(shared_global_weights, aggregated_weights)
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

        # Evaluation: server-side (once) when global_val_data is provided and no personalization,
        # otherwise distributed across all client actors.
        personalize_eval = config.get('fl', {}).get('personalize', False)
        if global_val_data is not None and not personalize_eval:
            logging.debug(f'[Server] Global evaluation on server (single pass, device={server_device}).')
            X_val_g, y_val_g = global_val_data
            val_metrics_agg = evaluate_model_globally(global_model, X_val_g, y_val_g, config, hyperparameters)
            logging.debug('[Server] Global evaluation completed.')
        else:
            logging.debug(f'[Server] Start local evaluation.')
            results_refs = [actor.evaluate.remote(global_weights_cpu) for actor in client_actors]
            client_results = ray.get(results_refs)
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
        # Build a compact per-round summary line at INFO level
        _parts = [f"Round {round_num:>3}/{n_rounds}  ({round_end_time - round_start_time:.1f}s)"]
        if train_metrics_agg:
            _rmse = train_metrics_agg.get('rmse', train_metrics_agg.get('train_rmse'))
            if _rmse is not None:
                _parts.append(f"train_rmse={_rmse:.4f}")
        if val_metrics_agg:
            _vrmse = val_metrics_agg.get('rmse', val_metrics_agg.get('val_rmse'))
            if _vrmse is not None:
                _parts.append(f"val_rmse={_vrmse:.4f}")
        logging.info("  ".join(_parts))

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