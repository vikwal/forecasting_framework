"""
Seq2Seq Encoder with NWP node attention.

NWP messages are computed per timestep using the current last-layer hidden
state H[-1] as the GATv2 attention query, so the NWP attention is conditioned
on what the station has learned so far.

Per timestep t ∈ [0, T_hist):

  1. NWP attention step:
       nwp_msg_t = GATv2(icond2_t, ecmwf_t → station | H[-1])   (N_s, nwp_out_dim)
       (H[-1] = zeros at t=0, then updated hidden state for t>0)

  2. DCGRU step:
       input_t = cat( meas_t, nwp_msg_t, static )  → (N_s, M + nwp_out_dim + S)
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
        nwp_nodes: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.nwp_nodes   = nwp_nodes
        self.nwp_out_dim = nwp_out_dim

        if nwp_nodes and nwp_out_dim > 0:
            self.nwp_attn = NWPAttentionLayer(
                icond2_dim=icond2_dim,
                ecmwf_dim=ecmwf_dim,
                station_dim=hidden_dim,
                nwp_out_dim=nwp_out_dim,
                heads=nwp_heads,
                edge_dim=edge_dim,
                dropout=dropout,
            )

        if nwp_nodes:
            gru_input_dim = meas_dim + nwp_out_dim + static_dim
        else:
            # meas_dim already includes NWP (M + I2 + E2); no GATv2 output
            gru_input_dim = meas_dim + static_dim

        self.cells = nn.ModuleList([
            DCGRUCell(
                gru_input_dim if i == 0 else hidden_dim,
                hidden_dim, K, dropout,
            )
            for i in range(num_layers)
        ])

    def forward(
        self,
        meas_seq: Tensor,           # (T_hist, N_s, M)
        icond2_seq: Tensor,         # (T_hist, N_i, I2)
        ecmwf_seq: Tensor,          # (T_hist, N_e, E2)
        static: Tensor,             # (N_s, S)
        s2s_edge_index: Tensor,     # (2, E_s2s)
        s2s_edge_weight: Tensor,    # (E_s2s,)  — static fallback
        i2s_edge_index: Tensor,     # (2, E_i2s)
        i2s_edge_attr: Tensor,      # (E_i2s, edge_dim)
        e2s_edge_index: Tensor,     # (2, E_e2s)
        e2s_edge_attr: Tensor,      # (E_e2s, edge_dim)
        wind_dir_seq: Tensor | None = None,   # (T_hist, N_s) degrees, met. convention
        s2s_dist_norm: Tensor | None = None,  # (E_s2s,) — normalised distance
        s2s_bearing: Tensor | None = None,    # (E_s2s,) — azimuth src→dst in radians
    ) -> list[Tensor]:
        """
        Returns
        -------
        H_list : [H_0, …, H_{L-1}], each (N_s, hidden_dim).
        """
        T_hist = meas_seq.size(0)
        N_s    = meas_seq.size(1)
        device = meas_seq.device

        use_dir = wind_dir_seq is not None

        H = [
            torch.zeros(N_s, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

        for t in range(T_hist):
            if self.nwp_nodes and self.nwp_out_dim > 0:
                # GATv2 attention conditioned on current last-layer hidden state
                nwp_msg_t = self.nwp_attn.forward(
                    icond2_seq[t], ecmwf_seq[t], H[-1],
                    i2s_edge_index, i2s_edge_attr,
                    e2s_edge_index, e2s_edge_attr,
                )                                    # (N_s, nwp_out_dim)
                x_t = torch.cat([meas_seq[t], nwp_msg_t, static], dim=-1)
            else:
                # nwp_nodes=False: NWP already in meas_seq (M+I2+E2 channels)
                # nwp_nodes=True, nwp_out_dim=0: nwp_injection bypass, no NWP
                x_t = torch.cat([meas_seq[t], static], dim=-1)

            ew_t = (
                DCGRUCell.directional_edge_weight(
                    wind_dir_seq[t], s2s_edge_index[0], s2s_dist_norm, s2s_bearing
                )
                if use_dir else s2s_edge_weight
            )

            for l, cell in enumerate(self.cells):
                H[l] = cell(x_t, H[l], s2s_edge_index, ew_t)
                x_t  = H[l]

        return H
