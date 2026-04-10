"""
Processor: heterogeneous Spatio-Temporal (ST) blocks using GATv2Conv.

Each ST-block applies:
  1. Temporal 1-D convolution per node type  (acausal, "same" padding)
  2. Spatial GATv2 message passing           (time-synchronous)
  3. Residual connection + LayerNorm

Time-synchronous spatial step
------------------------------
Node features have shape (N, T, d).  To run GATv2Conv spatially across nodes
for *each timestep independently*, we reshape to (N*T, d) and expand the
edge_index so that edges at timestep t connect nodes in index range
[t*N_src, (t+1)*N_src) → [t*N_dst, (t+1)*N_dst).  This is equivalent to
a block-diagonal adjacency matrix over the T time-slices.

Because geometric edge attributes (distance, bearing, altitude diff) are
time-invariant, we simply repeat the (E, edge_dim) tensor T times.

Message-passing order within each ST-block
-------------------------------------------
  1. icond2  → station  (NWP informs stations)
  2. ecmwf   → station  (NWP informs stations)
  3. station → station  (spatial diffusion among stations)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv

from ..config import ModelConfig


# ---------------------------------------------------------------------------
# Edge-index expansion helpers
# ---------------------------------------------------------------------------

def _expand_homo_edge_index(ei: Tensor, N: int, T: int) -> Tensor:
    """
    Expand a homogeneous edge_index (2, E) to (2, T*E) so that at each
    timestep t the edges connect nodes in the same time-slice.

    Node i at timestep t maps to global index t*N + i.
    """
    E = ei.shape[1]
    offsets = torch.arange(T, device=ei.device).view(-1, 1) * N   # (T, 1)
    src = ei[0].unsqueeze(0) + offsets    # (T, E)
    dst = ei[1].unsqueeze(0) + offsets    # (T, E)
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # (2, T*E)


def _expand_hetero_edge_index(
    ei: Tensor, N_src: int, N_dst: int, T: int
) -> Tensor:
    """
    Expand a bipartite edge_index (2, E) where src ∈ [0, N_src) and
    dst ∈ [0, N_dst) to cover T independent time-slices.
    """
    E = ei.shape[1]
    t = torch.arange(T, device=ei.device)
    src = ei[0].unsqueeze(0) + (t * N_src).view(-1, 1)   # (T, E)
    dst = ei[1].unsqueeze(0) + (t * N_dst).view(-1, 1)   # (T, E)
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # (2, T*E)


# ---------------------------------------------------------------------------
# Temporal convolution block (used inside ST-block)
# ---------------------------------------------------------------------------

class _TemporalConvBlock(nn.Module):
    """
    Acausal Conv1d → SiLU applied per-node.

    Acausal (same-padding) is intentional here: after the causal encoder has
    done the initial feature extraction, the processor is allowed to attend to
    both past and future NWP timesteps (measurements are already masked).
    """

    def __init__(self, d: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(d, d, kernel_size, padding=pad)
        self.act  = nn.SiLU()

    def forward(self, h: Tensor) -> Tensor:
        # h: (N, T, d)
        x = h.permute(0, 2, 1)            # (N, d, T)
        x = self.act(self.conv(x))         # (N, d, T)
        return x.permute(0, 2, 1)          # (N, T, d)


# ---------------------------------------------------------------------------
# One ST-block
# ---------------------------------------------------------------------------

class STBlock(nn.Module):
    """
    Spatio-Temporal Block:
      step 1 — temporal Conv1d per node type (residual)
      step 2 — spatial GATv2 over the time-expanded graph (residual + norm)

    Parameters
    ----------
    latent_dim    : node embedding dimension (= heads * out_per_head)
    edge_dim      : raw edge-attribute dimension fed to GATv2
    heads         : number of GATv2 attention heads
    kernel_size   : Conv1d kernel size for the temporal step
    dropout       : attention + projection dropout in GATv2
    """

    def __init__(
        self,
        latent_dim: int,
        edge_dim: int,
        heads: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert latent_dim % heads == 0, (
            f"latent_dim ({latent_dim}) must be divisible by heads ({heads})"
        )
        d = latent_dim
        out_per_head = d // heads

        # --- Temporal convolutions (one per node type that exists in graph) ---
        self.temp_conv_s = _TemporalConvBlock(d, kernel_size)
        self.temp_conv_i = _TemporalConvBlock(d, kernel_size)
        self.temp_conv_e = _TemporalConvBlock(d, kernel_size)

        # --- Spatial GATv2 layers ---
        # NWP → station: bipartite (in_channels as tuple, no self-loops)
        self.gat_i2s = GATv2Conv(
            in_channels=(d, d),
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            add_self_loops=False,
            dropout=dropout,
        )
        self.gat_e2s = GATv2Conv(
            in_channels=(d, d),
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            add_self_loops=False,
            dropout=dropout,
        )
        # station → station: homogeneous (self-loops included for isolated nodes)
        self.gat_s2s = GATv2Conv(
            in_channels=d,
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            add_self_loops=True,
            dropout=dropout,
        )

        # --- Layer norms (applied to station embeddings after spatial step) ---
        self.norm_s = nn.LayerNorm(d)

        self.drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------

    def forward(
        self,
        node_emb: dict[str, Tensor],
        edge_index_dict: dict[tuple, Tensor],
        edge_attr_dict: dict[tuple, Tensor],
    ) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        node_emb :       {"station": (N_s, T, d), "icond2": (N_i, T, d),
                          "ecmwf": (N_e, T, d)}
        edge_index_dict : {edge_type: (2, E)}  raw edge indices
        edge_attr_dict  : {edge_type: (E, edge_dim)} raw geometric features

        Returns
        -------
        Updated node_emb dict (same shapes as input).
        """
        i2s_key = ("icond2", "informs", "station")
        e2s_key = ("ecmwf",  "informs", "station")
        s2s_key = ("station", "near",   "station")

        h_s = node_emb["station"]   # (N_s, T, d)
        h_i = node_emb["icond2"]    # (N_i, T, d)
        h_e = node_emb["ecmwf"]     # (N_e, T, d)
        N_s, T, d = h_s.shape
        N_i = h_i.shape[0]
        N_e = h_e.shape[0]

        # ── Step 1: temporal convolution (residual per node type) ──────────
        h_s = self.temp_conv_s(h_s) + h_s   # (N_s, T, d)
        h_i = self.temp_conv_i(h_i) + h_i   # (N_i, T, d)
        h_e = self.temp_conv_e(h_e) + h_e   # (N_e, T, d)

        # ── Step 2: spatial GATv2 (time-synchronous) ───────────────────────
        # Flatten time into the node dimension
        h_s_flat = h_s.reshape(N_s * T, d)
        h_i_flat = h_i.reshape(N_i * T, d)
        h_e_flat = h_e.reshape(N_e * T, d)

        # Expand edge indices for all T timesteps
        i2s_ei_exp = _expand_hetero_edge_index(
            edge_index_dict[i2s_key], N_i, N_s, T
        )
        e2s_ei_exp = _expand_hetero_edge_index(
            edge_index_dict[e2s_key], N_e, N_s, T
        )
        s2s_ei_exp = _expand_homo_edge_index(
            edge_index_dict[s2s_key], N_s, T
        )

        # Repeat geometric edge attributes for each timestep
        i2s_ea_exp = edge_attr_dict[i2s_key].repeat(T, 1)   # (T*E_i2s, edge_dim)
        e2s_ea_exp = edge_attr_dict[e2s_key].repeat(T, 1)
        s2s_ea_exp = edge_attr_dict[s2s_key].repeat(T, 1)

        # NWP → station (sequential, each updating h_s_new)
        h_s_new = self.gat_i2s(
            (h_i_flat, h_s_flat), i2s_ei_exp, i2s_ea_exp
        )                                                    # (N_s*T, d)
        h_s_new = self.gat_e2s(
            (h_e_flat, h_s_new), e2s_ei_exp, e2s_ea_exp
        )                                                    # (N_s*T, d)
        # station → station
        h_s_new = self.gat_s2s(h_s_new, s2s_ei_exp, s2s_ea_exp)  # (N_s*T, d)

        # Reshape, residual, norm
        h_s_new = h_s_new.reshape(N_s, T, d)
        h_s = self.norm_s(self.drop(h_s_new) + h_s)         # (N_s, T, d)

        return {"station": h_s, "icond2": h_i, "ecmwf": h_e}


# ---------------------------------------------------------------------------
# Full processor stack
# ---------------------------------------------------------------------------

class HeteroProcessor(nn.Module):
    """
    Stack of ``num_message_passing_layers`` ST-blocks.

    The ``heads`` field in ModelConfig (default 4) controls GATv2 attention
    heads.  ``latent_dim`` must be divisible by ``heads``.

    Parameters
    ----------
    config : ModelConfig
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        edge_dim = config.edge_input_dim()
        heads    = getattr(config, "processor_heads", 4)

        self.blocks = nn.ModuleList([
            STBlock(
                latent_dim=config.latent_dim,
                edge_dim=edge_dim,
                heads=heads,
                kernel_size=config.cnn_kernel_size,
                dropout=config.dropout,
            )
            for _ in range(config.num_message_passing_layers)
        ])

    def forward(
        self,
        node_emb: dict[str, Tensor],
        edge_index_dict: dict[tuple, Tensor],
        edge_attr_dict: dict[tuple, Tensor],
    ) -> dict[str, Tensor]:
        """
        Run all ST-blocks sequentially.

        Parameters
        ----------
        node_emb       : {"station": (N_s, T, d), ...}
        edge_index_dict : {edge_type: (2, E)} — raw indices from HeteroData
        edge_attr_dict  : {edge_type: (E, edge_dim)} — raw geometric attrs

        Returns
        -------
        Final node embedding dict; same shapes as input.
        Edge embeddings are not returned (GATv2 handles attention internally).
        """
        for block in self.blocks:
            node_emb = block(node_emb, edge_index_dict, edge_attr_dict)
        return node_emb
