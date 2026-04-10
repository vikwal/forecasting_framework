"""
Loss functions for the STGNN, with optional exponential horizon weighting.

Horizon weighting: assign higher weight to near-term forecast steps
(step 1 is most important, step 48 least), using exponential decay.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _horizon_weights(horizon: int, decay: float, device: torch.device) -> Tensor:
    """
    Compute normalised exponential decay weights over the forecast horizon.

    w_t = decay^t,  then normalised so sum == horizon.

    Parameters
    ----------
    horizon : number of forecast steps
    decay :   decay factor per step (e.g. 0.95 means step t has weight 0.95^t)
    device :  target device

    Returns
    -------
    (horizon,) float tensor of weights summing to ``horizon``
    """
    t = torch.arange(horizon, dtype=torch.float32, device=device)
    w = decay ** t
    w = w / w.sum() * horizon  # normalise
    return w


class WeightedMSELoss(nn.Module):
    """
    MSE loss with optional exponential horizon weighting.

    Parameters
    ----------
    forecast_horizon :      number of forecast steps (H)
    weight_by_horizon :     if True, apply exponential decay weights
    horizon_decay :         decay factor per step (only used if weight_by_horizon)
    """

    def __init__(
        self,
        forecast_horizon: int = 48,
        weight_by_horizon: bool = True,
        horizon_decay: float = 0.95,
    ) -> None:
        super().__init__()
        self.H = forecast_horizon
        self.weighted = weight_by_horizon
        self.decay = horizon_decay
        self._cached_weights: Tensor | None = None

    def _get_weights(self, device: torch.device) -> Tensor:
        if self._cached_weights is None or self._cached_weights.device != device:
            self._cached_weights = _horizon_weights(self.H, self.decay, device)
        return self._cached_weights

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Parameters
        ----------
        preds :   (N_target, H)
        targets : (N_target, H)

        Returns
        -------
        Scalar loss.
        """
        err = (preds - targets) ** 2   # (N, H)
        if self.weighted:
            w = self._get_weights(preds.device)  # (H,)
            err = err * w.unsqueeze(0)
        return err.mean()


class WeightedMAELoss(nn.Module):
    """MAE loss with optional exponential horizon weighting."""

    def __init__(
        self,
        forecast_horizon: int = 48,
        weight_by_horizon: bool = True,
        horizon_decay: float = 0.95,
    ) -> None:
        super().__init__()
        self.H = forecast_horizon
        self.weighted = weight_by_horizon
        self.decay = horizon_decay
        self._cached_weights: Tensor | None = None

    def _get_weights(self, device: torch.device) -> Tensor:
        if self._cached_weights is None or self._cached_weights.device != device:
            self._cached_weights = _horizon_weights(self.H, self.decay, device)
        return self._cached_weights

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        err = (preds - targets).abs()
        if self.weighted:
            w = self._get_weights(preds.device)
            err = err * w.unsqueeze(0)
        return err.mean()


class WeightedHuberLoss(nn.Module):
    """Huber (smooth L1) loss with optional horizon weighting."""

    def __init__(
        self,
        forecast_horizon: int = 48,
        weight_by_horizon: bool = True,
        horizon_decay: float = 0.95,
        delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.H = forecast_horizon
        self.weighted = weight_by_horizon
        self.decay = horizon_decay
        self.delta = delta
        self._cached_weights: Tensor | None = None

    def _get_weights(self, device: torch.device) -> Tensor:
        if self._cached_weights is None or self._cached_weights.device != device:
            self._cached_weights = _horizon_weights(self.H, self.decay, device)
        return self._cached_weights

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        err = F.huber_loss(preds, targets, reduction="none", delta=self.delta)
        if self.weighted:
            w = self._get_weights(preds.device)
            err = err * w.unsqueeze(0)
        return err.mean()


def build_loss(
    loss_fn: str,
    forecast_horizon: int,
    weight_by_horizon: bool,
    horizon_decay: float,
) -> nn.Module:
    """
    Factory for loss functions.

    Parameters
    ----------
    loss_fn :           "mse", "mae", or "huber"
    forecast_horizon :  H
    weight_by_horizon : enable horizon weighting
    horizon_decay :     decay factor per step

    Returns
    -------
    Loss module.
    """
    kwargs = dict(
        forecast_horizon=forecast_horizon,
        weight_by_horizon=weight_by_horizon,
        horizon_decay=horizon_decay,
    )
    if loss_fn == "mse":
        return WeightedMSELoss(**kwargs)
    elif loss_fn == "mae":
        return WeightedMAELoss(**kwargs)
    elif loss_fn == "huber":
        return WeightedHuberLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn!r}. Choose from 'mse', 'mae', 'huber'.")
