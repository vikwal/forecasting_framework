"""FedGradient: federated training by per-batch gradient aggregation.

Clients send raw mini-batch gradients instead of model weights; the server averages them
and takes the optimizer step on the global model after every batch (docs/fedgradient.md).

  * One round = one epoch: every client permutes its training windows (seeded with
    ``seed * 1000 + round``) and walks through them in batches without replacement.
    Each server step consumes exactly one batch gradient from every client that still
    has batches left; exhausted clients drop out of the mean, the round ends when all
    clients are through.
  * The server optimizer is a plain ``torch.optim`` optimizer on the global model
    (``fl.fedgradient.server_optimizer``: adam | adamw | sgd | sgd_momentum); its state is
    part of the checkpoint.
  * Categorical embeddings (TFT ``static_embed[i]`` = ``nn.Embedding``, e.g. a park id) are
    aggregated row by row: a row is averaged only over the clients that own that category,
    rows without a contributing client are not touched by the optimizer at all
    (no momentum drift, no weight decay).

Everything here is plain PyTorch; ``utils.federated`` wires it to the Ray actors.
"""

import os
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

SERVER_OPTIMIZERS = ('adam', 'adamw', 'sgd', 'sgd_momentum')

# Hyperparameter keys of the strategy (flat, prefixed so they never collide with the
# client/CL keys 'lr', 'weight_decay', 'clipnorm').
HP_KEYS = ('server_optimizer', 'server_lr', 'server_beta_1', 'server_beta_2', 'server_eps',
           'server_momentum', 'server_weight_decay', 'client_clipnorm', 'gradient_weighting')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _required(cfg: dict, key: str):
    if cfg.get(key) is None:
        raise ValueError(f"fl.fedgradient.{key} is required for server_optimizer "
                         f"'{cfg.get('server_optimizer')}'")
    return cfg[key]


def fedgradient_hyperparameters(config: dict) -> Dict[str, Any]:
    """Strategy hyperparameters from ``fl.fedgradient``. Values live in the config, the
    code has no hidden defaults apart from ``weight_decay`` (0) and ``weighting``.

    ``clipnorm``: client-side gradient clipping before sending; missing key falls back to
    ``model.tft.clipnorm`` (the CL setting), ``null``/0 disables clipping.
    """
    cfg = (config.get('fl') or {}).get('fedgradient')
    if not cfg:
        raise ValueError("fl.strategy 'fedgradient' needs an fl.fedgradient section")
    name = str(_required(cfg, 'server_optimizer')).lower()
    if name not in SERVER_OPTIMIZERS:
        raise ValueError(f"fl.fedgradient.server_optimizer must be one of {SERVER_OPTIMIZERS}, got {name!r}")
    weighting = str(cfg.get('weighting', 'batch_size')).lower()
    if weighting not in ('batch_size', 'uniform'):
        raise ValueError(f"fl.fedgradient.weighting must be 'batch_size' or 'uniform', got {weighting!r}")
    if 'clipnorm' in cfg:
        clip = cfg['clipnorm']
    else:
        clip = ((config.get('model') or {}).get('tft') or {}).get('clipnorm')
    hp = {'server_optimizer': name,
          'server_lr': float(_required(cfg, 'server_lr')),
          'server_weight_decay': float(cfg.get('weight_decay') or 0.0),
          'client_clipnorm': float(clip) if clip else None,
          'gradient_weighting': weighting}
    if name in ('adam', 'adamw'):
        hp['server_beta_1'] = float(_required(cfg, 'beta_1'))
        hp['server_beta_2'] = float(_required(cfg, 'beta_2'))
        hp['server_eps'] = float(_required(cfg, 'eps'))
    if name == 'sgd_momentum':
        hp['server_momentum'] = float(_required(cfg, 'momentum'))
    return hp


def build_server_optimizer(params, hp: Dict[str, Any]) -> torch.optim.Optimizer:
    """torch.optim optimizer for the global model from the strategy hyperparameters."""
    name = hp['server_optimizer']
    lr = hp['server_lr']
    wd = hp.get('server_weight_decay', 0.0) or 0.0
    if name in ('adam', 'adamw'):
        cls = torch.optim.Adam if name == 'adam' else torch.optim.AdamW
        return cls(params, lr=lr, betas=(hp['server_beta_1'], hp['server_beta_2']),
                   eps=hp['server_eps'], weight_decay=wd)
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    if name == 'sgd_momentum':
        return torch.optim.SGD(params, lr=lr, momentum=hp['server_momentum'], weight_decay=wd)
    raise ValueError(f"unknown server optimizer {name!r}")


def round_seed(seed: int, round_num: int) -> int:
    return int(seed) * 1000 + int(round_num)


# ---------------------------------------------------------------------------
# Row-wise (per-category) aggregation of embedding tables
# ---------------------------------------------------------------------------

def embedding_row_params(model: nn.Module) -> Dict[str, int]:
    """Parameter name -> static input column of each categorical embedding table
    (TFT: ``static_embed[i]`` is an ``nn.Embedding`` when static feature i is categorical)."""
    out = {}
    embeds = getattr(model, 'static_embed', None)
    if embeds is None:
        return out
    for i, module in enumerate(embeds):
        if isinstance(module, nn.Embedding):
            out[f'static_embed.{i}.weight'] = i
    return out


def owned_rows(X_train: Any, row_params: Dict[str, int],
               n_rows: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """Boolean row mask per embedding parameter: the categories present in a client's
    training data (codes arrive as floats from the dataframe pipeline)."""
    out = {}
    if not row_params:
        return out
    if not isinstance(X_train, dict) or 'static' not in X_train:
        raise ValueError("row-wise aggregation needs the static inputs (X_train['static'])")
    static = np.asarray(X_train['static'])
    for name, col in row_params.items():
        codes = np.unique(np.rint(static[:, col]).astype(np.int64))
        if codes.size and (codes.min() < 0 or codes.max() >= n_rows[name]):
            raise ValueError(f"{name}: category code out of range [0, {n_rows[name]})")
        mask = torch.zeros(n_rows[name], dtype=torch.bool)
        mask[torch.from_numpy(codes)] = True
        out[name] = mask
    return out


def rowwise_average(tensors: Sequence[torch.Tensor], weights: Sequence[float],
                    masks: Sequence[torch.Tensor],
                    fallback: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean of each row over the tensors whose mask owns that row.

    Rows owned by nobody get ``fallback`` (or 0). Returns (average, has_contribution)."""
    ref = tensors[0].float()
    num = torch.zeros_like(ref)
    den = torch.zeros(ref.shape[0], dtype=ref.dtype)
    for t, w, m in zip(tensors, weights, masks):
        wm = m.to(ref.dtype) * float(w)
        num += t.float() * wm.view(-1, *([1] * (ref.dim() - 1)))
        den += wm
    has = den > 0
    out = fallback.float().clone() if fallback is not None else torch.zeros_like(ref)
    out[has] = num[has] / den[has].view(-1, *([1] * (ref.dim() - 1)))
    return out.to(tensors[0].dtype), has


def aggregate_gradients(results: List[Dict[str, Any]],
                        row_owners: Optional[Dict[Any, Dict[str, torch.Tensor]]] = None,
                        weighting: str = 'batch_size'
                        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Average the batch gradients of the participating clients.

    Ordinary parameters: mean over all participants (weighted by batch size or uniform).
    Embedding tables listed in ``row_owners``: per row only over the participants owning it.
    Returns (gradients, row_masks) where row_masks marks the rows with a contribution.
    """
    weights = [float(r['n_samples']) if weighting == 'batch_size' else 1.0 for r in results]
    total = sum(weights)
    first_owners = (row_owners or {}).get(results[0]['client_id'], {})
    grads, row_masks = {}, {}
    for key in results[0]['gradients']:
        if key in first_owners:
            grads[key], row_masks[key] = rowwise_average(
                [r['gradients'][key] for r in results], weights,
                [row_owners[r['client_id']][key] for r in results])
        else:
            grads[key] = sum(r['gradients'][key].float() * (w / total)
                             for r, w in zip(results, weights))
    return grads, row_masks


def _is_row_state(value: Any, param: torch.Tensor) -> bool:
    return torch.is_tensor(value) and value.dim() > 0 and value.shape == param.shape


def server_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                grads: Dict[str, torch.Tensor],
                row_masks: Optional[Dict[str, torch.Tensor]] = None) -> None:
    """Write the aggregated gradients into ``.grad`` and take one optimizer step.

    Rows of an embedding table without a contribution keep their value and their
    optimizer state (momentum, second moment), so neither momentum nor weight decay
    moves categories nobody trained in this step."""
    params = dict(model.named_parameters())
    for name, p in params.items():
        g = grads.get(name)
        p.grad = None if (g is None or not p.requires_grad) else g.to(device=p.device, dtype=p.dtype)

    frozen = {}
    for name, has in (row_masks or {}).items():
        if bool(has.all()):
            continue
        p = params[name]
        keep = (~has).to(p.device)
        state = optimizer.state.get(p, {})
        snap = {k: v[keep].clone() for k, v in state.items() if _is_row_state(v, p)}
        frozen[name] = (keep, p.data[keep].clone(), snap)

    optimizer.step()

    for name, (keep, rows, snap) in frozen.items():
        p = params[name]
        p.data[keep] = rows
        for k, v in optimizer.state[p].items():
            if _is_row_state(v, p):
                v[keep] = snap[k] if k in snap else 0
    optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Batches, forward pass, client
# ---------------------------------------------------------------------------

def epoch_batches(n: int, batch_size: int, seed: int,
                  drop_last: bool = True, shuffle: bool = True) -> List[torch.Tensor]:
    """Index batches of one epoch: a seeded permutation, consumed without replacement."""
    if shuffle:
        order = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
    else:
        order = torch.arange(n)
    n_full = n // batch_size
    batches = [order[i * batch_size:(i + 1) * batch_size] for i in range(n_full)]
    if not drop_last and n % batch_size:
        batches.append(order[n_full * batch_size:])
    return batches


def data_tensors(X: Any, y: np.ndarray, device) -> List[torch.Tensor]:
    """[observed, known, (static), y] for TFT dicts, [X, y] otherwise — float32 on device."""
    if isinstance(X, dict):
        arrays = [X['observed'], X['known']] + ([X['static']] if 'static' in X else []) + [y]
    else:
        arrays = [X, y]
    return [torch.as_tensor(np.asarray(a), dtype=torch.float32).to(device) for a in arrays]


def forward(model: nn.Module, batch: Sequence[torch.Tensor], is_tft: bool) -> torch.Tensor:
    inputs = batch[:-1]
    if is_tft:
        return model(*inputs)
    return model(inputs[0])


def load_params(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    """Copy parameter values in place (keeps the model and its optimizer-free state on device)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in weights:
                p.copy_(weights[name].to(p.device, non_blocking=True))


class GradientClient:
    """Client side of FedGradient: holds the training data on the device, serves one batch
    gradient per server step and accumulates the round's training metrics."""

    def __init__(self, model: nn.Module, X_train: Any, y_train: np.ndarray,
                 batch_size: int, criterion: Callable, device,
                 is_tft: bool = True, clipnorm: Optional[float] = None,
                 median_idx: Optional[int] = None, shuffle: bool = True,
                 client_id: Any = None, dropout_seed_offset: int = 0):
        self.client_id = client_id
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tensors = data_tensors(X_train, y_train, self.device)
        self.n = int(self.tensors[-1].shape[0])
        self.batch_size = int(batch_size)
        self.criterion = criterion
        self.is_tft = is_tft
        self.clipnorm = clipnorm
        self.median_idx = median_idx
        self.shuffle = shuffle
        self.dropout_seed_offset = int(dropout_seed_offset)
        self.trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.batches: List[torch.Tensor] = []
        self.cursor = 0
        self._reset_stats()

    def _reset_stats(self):
        self._sse = 0.0
        self._sae = 0.0
        self._n_el = 0
        self._loss_sum = 0.0
        self._n_samples = 0

    def prepare_round(self, seed: int, weights: Optional[Dict[str, torch.Tensor]] = None) -> int:
        """Load the global state, build this round's batch schedule; returns #batches."""
        if weights is not None:
            self.model.load_state_dict({k: v for k, v in weights.items()}, strict=True)
        self.batches = [b.to(self.device) for b in
                        epoch_batches(self.n, self.batch_size, seed, drop_last=True, shuffle=self.shuffle)]
        self.cursor = 0
        torch.manual_seed(int(seed) + self.dropout_seed_offset)   # dropout masks
        self._reset_stats()
        return len(self.batches)

    @property
    def remaining(self) -> int:
        return len(self.batches) - self.cursor

    def step(self, weights: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Forward/backward on the next batch with the given global parameters, clip,
        return the gradients (CPU). No optimizer step on the client."""
        if self.remaining <= 0:
            raise RuntimeError(f"client {self.client_id}: no batches left in this round")
        if weights is not None:
            load_params(self.model, weights)
        idx = self.batches[self.cursor]
        self.cursor += 1
        batch = [t[idx] for t in self.tensors]
        targets = batch[-1]

        self.model.train()
        self.model.zero_grad(set_to_none=True)
        pred = forward(self.model, batch, self.is_tft)
        loss = self.criterion(pred, targets)
        loss.backward()
        if self.clipnorm:
            torch.nn.utils.clip_grad_norm_([p for _, p in self.trainable], self.clipnorm)
        grads = {n: (p.grad.detach().to('cpu', copy=True) if p.grad is not None
                     else torch.zeros_like(p, device='cpu'))
                 for n, p in self.trainable}

        with torch.no_grad():
            point = pred[..., self.median_idx] if (self.median_idx is not None and pred.dim() == 3) else pred
            err = (point.reshape(targets.shape) - targets).float()
            self._sse += float((err ** 2).sum())
            self._sae += float(err.abs().sum())
            self._n_el += err.numel()
        bs = int(targets.shape[0])
        self._loss_sum += float(loss.detach()) * bs
        self._n_samples += bs
        return {'client_id': self.client_id, 'gradients': grads, 'n_samples': bs,
                'remaining': self.remaining}

    def round_metrics(self) -> Dict[str, Any]:
        n_el = max(self._n_el, 1)
        mse = self._sse / n_el
        metrics = {'train_loss': mse, 'train_rmse': float(np.sqrt(mse)),
                   'train_mae': self._sae / n_el,
                   'train_objective': self._loss_sum / max(self._n_samples, 1)}
        return {'client_id': self.client_id, 'n_samples': self._n_samples, 'metrics': metrics}


# ---------------------------------------------------------------------------
# Server: one round
# ---------------------------------------------------------------------------

def tensor_bytes(tensors: Dict[str, torch.Tensor]) -> int:
    return int(sum(t.element_size() * t.numel() for t in tensors.values()))


class CommStats:
    """Communication accounting per client (bytes up/down) and server sync steps."""

    def __init__(self, client_ids: Sequence[Any]):
        self.sync_steps = 0
        self.upload = {str(c): 0 for c in client_ids}
        self.download = {str(c): 0 for c in client_ids}
        self.eval_download = {str(c): 0 for c in client_ids}

    def add_step(self, upload: Dict[Any, int], download: Dict[Any, int]) -> None:
        self.sync_steps += 1
        for c, b in upload.items():
            self.upload[str(c)] += int(b)
        for c, b in download.items():
            self.download[str(c)] += int(b)

    def add_eval(self, client_ids: Sequence[Any], n_bytes: int) -> None:
        for c in client_ids:
            self.eval_download[str(c)] += int(n_bytes)

    def to_dict(self) -> Dict[str, Any]:
        up, down, ev = sum(self.upload.values()), sum(self.download.values()), sum(self.eval_download.values())
        return {'sync_steps': self.sync_steps,
                'upload_bytes': dict(self.upload), 'download_bytes': dict(self.download),
                'eval_download_bytes': dict(self.eval_download),
                'total_upload_bytes': up, 'total_download_bytes': down,
                'total_eval_download_bytes': ev, 'total_bytes': up + down + ev}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CommStats':
        obj = cls(list(d['upload_bytes'].keys()))
        obj.sync_steps = int(d['sync_steps'])
        obj.upload.update(d['upload_bytes'])
        obj.download.update(d['download_bytes'])
        obj.eval_download.update(d.get('eval_download_bytes', {}))
        return obj


def params_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().to('cpu', copy=True) for n, p in model.named_parameters()}


def run_round(dispatch: Callable[[str, List[int], Any], List[Dict[str, Any]]],
              client_ids: Sequence[Any], global_model: nn.Module,
              optimizer: torch.optim.Optimizer, seed: int,
              row_owners: Optional[Dict[Any, Dict[str, torch.Tensor]]] = None,
              weighting: str = 'batch_size',
              comm: Optional[CommStats] = None) -> Tuple[List[Dict[str, Any]], int]:
    """One FedGradient round (= one epoch over every client's training windows).

    ``dispatch(method, client_indices, arg)`` calls ``method`` ('prepare_round' with
    (seed, state_dict), 'step' with the parameter dict, 'round_metrics' without argument)
    on the given clients and returns their results in order.
    Returns (per-client round metrics, number of server steps).
    """
    state = {k: v.detach().to('cpu', copy=True) for k, v in global_model.state_dict().items()}
    n_batches = dispatch('prepare_round', list(range(len(client_ids))), (seed, state))
    active = [i for i, nb in enumerate(n_batches) if nb > 0]
    # The round-start state equals the first step's parameters, so its broadcast is
    # accounted as the first step's download (buffers never change in FedGradient).
    steps = 0
    while active:
        weights = params_cpu(global_model)
        results = dispatch('step', active, weights)
        grads, masks = aggregate_gradients(results, row_owners, weighting)
        server_step(global_model, optimizer, grads, masks)
        steps += 1
        if comm is not None:
            w_bytes = tensor_bytes(weights)
            comm.add_step(upload={client_ids[i]: tensor_bytes(r['gradients']) for i, r in zip(active, results)},
                          download={client_ids[i]: w_bytes for i in active})
        active = [i for i, r in zip(active, results) if r['remaining'] > 0]
    metrics = dispatch('round_metrics', list(range(len(client_ids))), None)
    return metrics, steps


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    """Atomic torch.save (write to a temp file, then rename)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location='cpu', weights_only=False)
