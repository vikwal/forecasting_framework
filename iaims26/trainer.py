"""Slim training loop for fine-tuning / custom models.

Mirrors the semantics of utils/tools.py:training_pipeline (Adam, MSE,
clip_grad_norm_, early stopping on val_rmse with restore-best-weights) but
accepts a pre-built model and optimizer, which training_pipeline cannot do
(it constructs and torch.compiles the model internally). Validated against
training_pipeline by iaims26 smoke tests.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import tools


def train_model(model,
                X_train, y_train,
                X_val, y_val,
                batch_size: int,
                lr: float,
                device: str = 'cuda',
                epochs: int = 100,
                patience: int = 10,
                min_delta: float = 0.0001,
                clipnorm: float = None,
                shuffle: bool = True,
                num_workers: int = 0,
                verbose: bool = False):
    """Train `model` (possibly partially frozen) on TFT-style dict inputs.

    Returns (history, model, epochs_run). Early stopping monitors val_rmse
    and restores the best weights, like the repo's training_pipeline.
    """
    model = model.to(device)
    train_loader = tools.create_pytorch_dataloader(
        X_train, y_train, batch_size=batch_size, shuffle=shuffle,
        device=device, num_workers=num_workers, drop_last=True)
    # drop_last=True on tiny few-shot sets can drop everything -> fall back
    if len(train_loader) == 0:
        train_loader = tools.create_pytorch_dataloader(
            X_train, y_train, batch_size=batch_size, shuffle=shuffle,
            device=device, num_workers=num_workers, drop_last=False)
    val_loader = tools.create_pytorch_dataloader(
        X_val, y_val, batch_size=batch_size, shuffle=False,
        device=device, num_workers=num_workers, drop_last=False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError('no trainable parameters — check the freeze variant')
    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': [],
               'train_r2': [], 'val_r2': []}
    best_val = float('inf')
    best_state = None
    epochs_without_improvement = 0
    epochs_run = 0

    def _forward(batch):
        if len(batch) == 4:
            obs, known, static, targets = [b.to(device) for b in batch]
        elif len(batch) == 3:
            obs, known, targets = [b.to(device) for b in batch]
            static = None
        else:
            raise ValueError(f'unexpected batch structure of size {len(batch)}')
        return model(obs, known, static), targets

    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        preds_list, targets_list = [], []
        for batch in train_loader:
            predictions, targets = _forward(batch)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            if clipnorm is not None:
                torch.nn.utils.clip_grad_norm_(trainable, clipnorm)
            optimizer.step()
            preds_list.append(predictions.detach().cpu())
            targets_list.append(targets.detach().cpu())

        train_preds = torch.cat(preds_list).numpy()
        train_targets = torch.cat(targets_list).numpy()
        train_mse = float(np.mean((train_preds - train_targets) ** 2))
        ss_tot = np.sum((train_targets - train_targets.mean()) ** 2)
        train_r2 = 1 - np.sum((train_targets - train_preds) ** 2) / ss_tot if ss_tot else 0.0
        history['train_loss'].append(train_mse)
        history['train_rmse'].append(float(np.sqrt(train_mse)))
        history['train_r2'].append(float(train_r2))

        model.eval()
        preds_list, targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                predictions, targets = _forward(batch)
                preds_list.append(predictions.cpu())
                targets_list.append(targets.cpu())
        val_preds = torch.cat(preds_list).numpy()
        val_targets = torch.cat(targets_list).numpy()
        val_mse = float(np.mean((val_preds - val_targets) ** 2))
        val_rmse = float(np.sqrt(val_mse))
        ss_tot = np.sum((val_targets - val_targets.mean()) ** 2)
        val_r2 = 1 - np.sum((val_targets - val_preds) ** 2) / ss_tot if ss_tot else 0.0
        history['val_loss'].append(val_mse)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(float(val_r2))

        if verbose:
            logging.info(f'Epoch {epoch + 1}/{epochs} - train_rmse={np.sqrt(train_mse):.5f} '
                         f'val_rmse={val_rmse:.5f} val_r2={val_r2:.4f}')

        if val_rmse < best_val - min_delta:
            best_val = val_rmse
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, model, epochs_run
