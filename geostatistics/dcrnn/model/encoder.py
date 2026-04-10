"""
Seq2Seq Encoder with NWP node attention.

NWP messages are pre-computed for all T_hist steps at once using the
block-diagonal trick in NWPAttentionLayer.forward_sequence(), then the
DCGRU cells loop over timesteps using the pre-computed embeddings.

Per timestep t ∈ [0, T_hist):

  2. DCGRU step:
       input_t = cat( meas_t, nwp_msgs[t], static )  → (N_s, M + nwp_out_dim + S)
       H[l]    = DCGRUCell_l( input_t, H[l] )         (over station s2s graph)

Output: H_list = [H_0, …, H_{L-1}]  — final hidden states, one per DCGRU layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .dcgru_cell import DCGRUCell
from .nwp_attention import NWPAttentionLayer


class DCGRUEncoder(nn.Module):
    """
    Multi-layer DCGRU encoder with per-step NWP attention.

    Parameters
    ----------
    meas_dim        : M — measurement features per step
    nwp_out_dim     : output dimension of NWPAttentionLayer
    static_dim      : S — time-invariant station features
    hidden_dim      : DCGRU hidden dimension
    num_layers      : stacked DCGRU cells
    K               : diffusion hops in DCGRUCell
    dropout         : dropout in DCGRU cells
    icond2_dim      : I2 — raw ICON-D2 features per step
    ecmwf_dim       : E2 — raw ECMWF features per step
    nwp_heads       : attention heads in NWPAttentionLayer
    edge_dim        : edge_attr dimension from graph builder
    """

    def __init__(
        self,
        meas_dim: int,
        nwp_out_dim: int,
        static_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        K: int = 2,
        dropout: float = 0.0,
        icond2_dim: int = 3,
        ecmwf_dim: int = 3,
        nwp_heads: int = 4,
        edge_dim: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
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

        gru_input_dim = meas_dim + nwp_out_dim + static_dim
        self.cells = nn.ModuleList([
            DCGRUCell(
                gru_input_dim if i == 0 else hidden_dim,
                hidden_dim, K, dropout,
            )
            for i in range(num_layers)
        ])

    def forward(
        self,
        meas_seq: Tensor,       # (T_hist, N_s, M)
        icond2_seq: Tensor,     # (T_hist, N_i, I2)
        ecmwf_seq: Tensor,      # (T_hist, N_e, E2)
        static: Tensor,         # (N_s, S)
        s2s_edge_index: Tensor, # (2, E_s2s)
        s2s_edge_weight: Tensor,# (E_s2s,)
        i2s_edge_index: Tensor, # (2, E_i2s)
        i2s_edge_attr: Tensor,  # (E_i2s, edge_dim)
        e2s_edge_index: Tensor, # (2, E_e2s)
        e2s_edge_attr: Tensor,  # (E_e2s, edge_dim)
    ) -> list[Tensor]:
        """
        Returns
        -------
        H_list : [H_0, …, H_{L-1}], each (N_s, hidden_dim).
        """
        T_hist = meas_seq.size(0)
        N_s    = meas_seq.size(1)
        device = meas_seq.device

        H = [
            torch.zeros(N_s, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

        # Pre-compute all NWP messages in one vectorised GATv2 call
        nwp_msgs = self.nwp_attn.forward_sequence(
            icond2_seq, ecmwf_seq, N_s,
            i2s_edge_index, i2s_edge_attr,
            e2s_edge_index, e2s_edge_attr,
        )                                            # (T_hist, N_s, nwp_out_dim)

        for t in range(T_hist):
            x_t = torch.cat([meas_seq[t], nwp_msgs[t], static], dim=-1)
            for l, cell in enumerate(self.cells):
                H[l] = cell(x_t, H[l], s2s_edge_index, s2s_edge_weight)
                x_t  = H[l]

        return H
