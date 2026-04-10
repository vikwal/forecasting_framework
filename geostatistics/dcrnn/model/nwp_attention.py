"""
NWPAttentionLayer — aggregates NWP grid-point features into station embeddings
via bipartite GATv2 message passing.

Two calling modes
-----------------
forward()          : single timestep  (N_i/e, F) → (N_s, nwp_out_dim)
                     Used in sequential loops where H changes each step.

forward_sequence() : all T timesteps at once using the block-diagonal trick.
                     Treats T as a batch dimension, expands edge_index to
                     cover T independent time-slices in one GATv2 call.
                     Mathematically identical to calling forward() T times,
                     but eliminates 96× Python/kernel-dispatch overhead.

Block-diagonal trick (forward_sequence)
-----------------------------------------
For T timesteps and N_i source nodes with E edges to N_s destination nodes:

  i2_flat   = icond2_seq.reshape(T*N_i, I2)       # stack all time-slices
  h_query   = zeros(T*N_s, station_dim)            # zero query → time-invariant
  i2s_exp   = expand_hetero_edge_index(i2s_ei, N_i, N_s, T)   # (2, T*E)
  i2s_ea_r  = i2s_ea.repeat(T, 1)                 # (T*E, edge_dim)
  msg_flat  = gat_i2s((i2_flat, h_query), i2s_exp, i2s_ea_r)  # (T*N_s, d)
  nwp_seq   = msg_flat.reshape(T, N_s, d)

Using zeros as query makes attention time-invariant (NWP source features
drive the scores entirely), which is valid because the NWP-to-station
mapping is determined by geography, not the current hidden state.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv


def _expand_hetero_edge_index(
    ei: Tensor, N_src: int, N_dst: int, T: int
) -> Tensor:
    """Expand bipartite edge_index (2, E) to cover T independent time-slices."""
    t = torch.arange(T, device=ei.device)
    src = ei[0].unsqueeze(0) + (t * N_src).view(-1, 1)   # (T, E)
    dst = ei[1].unsqueeze(0) + (t * N_dst).view(-1, 1)   # (T, E)
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # (2, T*E)


class NWPAttentionLayer(nn.Module):
    """
    Bipartite GATv2 message passing: NWP nodes → station nodes.

    Parameters
    ----------
    icond2_dim  : raw ICON-D2 features per step (I2)
    ecmwf_dim   : raw ECMWF features per step (E2)
    station_dim : station hidden dim used as attention query (= hidden_dim)
    nwp_out_dim : output dimension; must be divisible by heads
    heads       : number of GATv2 attention heads
    edge_dim    : edge_attr columns from HeterogeneousGraphBuilder
    dropout     : GATv2 attention dropout
    """

    def __init__(
        self,
        icond2_dim: int,
        ecmwf_dim: int,
        station_dim: int,
        nwp_out_dim: int,
        heads: int = 4,
        edge_dim: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert nwp_out_dim % heads == 0
        out_per_head = nwp_out_dim // heads
        self.station_dim = station_dim
        self.nwp_out_dim = nwp_out_dim

        self.gat_i2s = GATv2Conv(
            in_channels=(icond2_dim, station_dim),
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            add_self_loops=False,
            dropout=dropout,
        )
        self.gat_e2s = GATv2Conv(
            in_channels=(ecmwf_dim, station_dim),
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            add_self_loops=False,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(nwp_out_dim)

    # ------------------------------------------------------------------
    # Single-step forward (kept for flexibility / debugging)
    # ------------------------------------------------------------------

    def forward(
        self,
        icond2_t: Tensor,        # (N_i, I2)
        ecmwf_t: Tensor,         # (N_e, E2)
        h_station: Tensor,       # (N_s, station_dim)
        i2s_edge_index: Tensor,
        i2s_edge_attr: Tensor,
        e2s_edge_index: Tensor,
        e2s_edge_attr: Tensor,
    ) -> Tensor:                 # (N_s, nwp_out_dim)
        msg_i = self.gat_i2s((icond2_t, h_station), i2s_edge_index, i2s_edge_attr)
        msg_e = self.gat_e2s((ecmwf_t,  h_station), e2s_edge_index, e2s_edge_attr)
        return self.norm(msg_i + msg_e)

    # ------------------------------------------------------------------
    # Vectorised forward over all T timesteps (block-diagonal trick)
    # ------------------------------------------------------------------

    def forward_sequence(
        self,
        icond2_seq: Tensor,      # (T, N_i, I2)
        ecmwf_seq: Tensor,       # (T, N_e, E2)
        N_s: int,
        i2s_edge_index: Tensor,  # (2, E_i2s)
        i2s_edge_attr: Tensor,   # (E_i2s, edge_dim)
        e2s_edge_index: Tensor,  # (2, E_e2s)
        e2s_edge_attr: Tensor,   # (E_e2s, edge_dim)
    ) -> Tensor:                 # (T, N_s, nwp_out_dim)
        T, N_i, _ = icond2_seq.shape
        N_e = ecmwf_seq.size(1)
        device = icond2_seq.device

        # Zero query: attention driven by NWP source features (time-invariant)
        h_q = torch.zeros(T * N_s, self.station_dim, device=device)

        i2_flat = icond2_seq.reshape(T * N_i, -1)
        e2_flat = ecmwf_seq.reshape(T * N_e, -1)

        i2s_ei_exp = _expand_hetero_edge_index(i2s_edge_index, N_i, N_s, T)
        e2s_ei_exp = _expand_hetero_edge_index(e2s_edge_index, N_e, N_s, T)
        i2s_ea_exp = i2s_edge_attr.repeat(T, 1)
        e2s_ea_exp = e2s_edge_attr.repeat(T, 1)

        msg_i = self.gat_i2s((i2_flat, h_q), i2s_ei_exp, i2s_ea_exp)  # (T*N_s, d)
        msg_e = self.gat_e2s((e2_flat, h_q), e2s_ei_exp, e2s_ea_exp)  # (T*N_s, d)

        nwp_flat = self.norm(msg_i + msg_e)                            # (T*N_s, d)
        return nwp_flat.reshape(T, N_s, self.nwp_out_dim)             # (T, N_s, d)