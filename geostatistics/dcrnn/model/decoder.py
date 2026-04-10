"""
Seq2Seq Autoregressive Decoder with NWP node attention.

NWP messages for all T_fore steps are pre-computed in one vectorised
GATv2 call (forward_sequence), then the autoregressive loop indexes into them.

Per forecast step t ∈ [0, T_fore):

  1. nwp_msg = nwp_msgs[t]  (pre-computed before the loop)

  2. DCGRU step:
       input_t = cat( y_{t-1}, nwp_msg_t, static )   → (N_s, 1 + nwp_out_dim + S)
       H[l]    = DCGRUCell_l( input_t, H[l] )

  3. Output projection:
       ŷ_t = Linear( H[-1] )               → (N_s,)
       → collect target stations → (N_target,)

Teacher forcing:
  During training, y_{t-1} for target stations is replaced by the
  ground-truth value with probability `teacher_forcing_ratio`.
  Neighbour stations always use their true last value (no forcing needed
  since their measurements are observed).
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch import Tensor

from .dcgru_cell import DCGRUCell
from .nwp_attention import NWPAttentionLayer


class DCGRUDecoder(nn.Module):
    """
    Multi-layer autoregressive DCGRU decoder with NWP attention.

    Parameters
    ----------
    nwp_out_dim      : output dimension of NWPAttentionLayer
    static_dim       : S — time-invariant station features
    hidden_dim       : DCGRU hidden dimension (must match encoder)
    forecast_horizon : T_fore
    num_layers       : stacked DCGRU cells (must match encoder)
    K                : diffusion hops
    dropout          : dropout in cells
    icond2_dim       : I2
    ecmwf_dim        : E2
    nwp_heads        : attention heads
    edge_dim         : edge_attr dimension
    """

    def __init__(
        self,
        nwp_out_dim: int,
        static_dim: int,
        hidden_dim: int,
        forecast_horizon: int,
        num_layers: int = 2,
        K: int = 2,
        dropout: float = 0.0,
        icond2_dim: int = 3,
        ecmwf_dim: int = 3,
        nwp_heads: int = 4,
        edge_dim: int = 3,
    ) -> None:
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.num_layers = num_layers

        self.nwp_attn = NWPAttentionLayer(
            icond2_dim=icond2_dim,
            ecmwf_dim=ecmwf_dim,
            station_dim=hidden_dim,
            nwp_out_dim=nwp_out_dim,
            heads=nwp_heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        gru_input_dim = 1 + nwp_out_dim + static_dim   # scalar y + nwp_msg + static
        self.cells = nn.ModuleList([
            DCGRUCell(
                gru_input_dim if i == 0 else hidden_dim,
                hidden_dim, K, dropout,
            )
            for i in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        H_init: list[Tensor],               # from encoder, len=num_layers
        icond2_fore: Tensor,                # (T_fore, N_i, I2)
        ecmwf_fore: Tensor,                 # (T_fore, N_e, E2)
        static: Tensor,                     # (N_s, S)
        s2s_edge_index: Tensor,
        s2s_edge_weight: Tensor,
        i2s_edge_index: Tensor,
        i2s_edge_attr: Tensor,
        e2s_edge_index: Tensor,
        e2s_edge_attr: Tensor,
        y_last_hist: Tensor,                # (N_s,) — last history value
        target_mask: Tensor,                # (N_s,) bool
        teacher_forcing_targets: Tensor | None = None,  # (N_target, T_fore)
        teacher_forcing_ratio: float = 0.5,
    ) -> Tensor:
        """
        Returns
        -------
        preds : (N_target, T_fore)
        """
        T_fore = self.forecast_horizon
        device = icond2_fore.device
        use_tf = (
            teacher_forcing_targets is not None
            and self.training
        )

        N_s = static.size(0)
        H = [h.clone() for h in H_init]
        y_prev = y_last_hist.clone()          # (N_s,)
        preds_list: list[Tensor] = []

        # Pre-compute all forecast NWP messages in one vectorised GATv2 call
        nwp_msgs = self.nwp_attn.forward_sequence(
            icond2_fore, ecmwf_fore, N_s,
            i2s_edge_index, i2s_edge_attr,
            e2s_edge_index, e2s_edge_attr,
        )                                     # (T_fore, N_s, nwp_out_dim)

        for t in range(T_fore):
            nwp_msg = nwp_msgs[t]            # (N_s, nwp_out_dim)

            # --- Build decoder input ---
            input_t = torch.cat(
                [y_prev.unsqueeze(-1), nwp_msg, static], dim=-1
            )                                # (N_s, 1 + nwp_out_dim + S)

            # --- DCGRU step ---
            x_t = input_t
            for l, cell in enumerate(self.cells):
                H[l] = cell(x_t, H[l], s2s_edge_index, s2s_edge_weight)
                x_t  = H[l]

            # --- Output projection ---
            y_hat = self.out_proj(H[-1]).squeeze(-1)   # (N_s,)
            preds_list.append(y_hat[target_mask])       # (N_target,)

            # --- Update y_prev for next step ---
            y_next = y_hat.detach().clone()
            if use_tf and random.random() < teacher_forcing_ratio:
                y_next[target_mask] = teacher_forcing_targets[:, t]
            y_prev = y_next

        return torch.stack(preds_list, dim=1)   # (N_target, T_fore)
