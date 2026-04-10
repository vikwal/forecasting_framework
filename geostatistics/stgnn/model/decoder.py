"""
Decoder: maps the station temporal sequence to per-step wind speed forecasts.

The processor outputs (N_s, T, latent_dim) where T = history_length +
forecast_horizon.  The decoder:

  1. Selects target station nodes (those masked during training).
  2. Slices the *forecast* portion of the temporal sequence:
       h_fore = h_target[:, T_hist : T_hist + T_fore, :]   # (N_target, T_fore, d)
  3. Applies a single shared linear layer *per timestep* to produce a scalar
     wind-speed prediction:
       pred_t = Linear(d → 1) applied to h_fore[:, t, :]    # for each t

This gives each forecast step direct access to its own temporal latent
representation — which encodes the NWP state at that specific hour — rather
than reconstructing all 48 steps from a single summary vector.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ..config import ModelConfig


class Decoder(nn.Module):
    """
    Per-step linear decoder for wind speed forecasting.

    Parameters
    ----------
    config : ModelConfig
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.T_hist = config.history_length
        self.T_fore = config.forecast_horizon
        d = config.latent_dim

        # Optional: a small MLP before the final projection gives the decoder
        # a bit of non-linearity while staying lightweight.
        # Using a single Linear here keeps the decoder's inductive bias minimal
        # and lets the processor carry the representational load.
        self.proj = nn.Linear(d, 1)

    def forward(
        self,
        station_embeddings: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        station_embeddings : (N_stations, T, latent_dim)
            Full temporal sequence from the processor for all station nodes.
        target_mask :        (N_stations,) boolean
            True for target (unobserved) stations to decode.

        Returns
        -------
        predictions : (N_target, forecast_horizon)
            Wind speed predictions for each target station and each forecast
            step.  Shape matches the ground_truth tensor in SampleBatch.
        """
        # Select target stations
        h_target = station_embeddings[target_mask]      # (N_target, T, d)

        # Slice the forecast horizon portion
        h_fore = h_target[
            :, self.T_hist : self.T_hist + self.T_fore, :
        ]                                               # (N_target, T_fore, d)

        # Per-step linear projection: shared Linear applied to each step
        pred = self.proj(h_fore).squeeze(-1)            # (N_target, T_fore)

        return pred
