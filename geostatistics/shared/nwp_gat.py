"""
HomoNWPAttentionLayer — bipartite GATv2 aggregation of NWP grid-point features
into station embeddings, designed for homogeneous samplers (MTGNN / WaveNet).

Unlike NWPAttentionLayer (DCRNN), the bipartite graph here is *regular*: each
station has exactly k nearest grid points, so edge_index is built on-the-fly
without a HeterogeneousGraphBuilder.

Calling convention
------------------
    forward_sequence(nwp_seq, N_s, k) -> (T, N_s, nwp_out_dim)

    nwp_seq : (T, N_s*k, nwp_feat_dim)
              Ordering: [t, n*k+j, :] = features of j-th grid point of station n.
              This matches the layout produced by HomoSampler with aggregate_nwp=False:
              for station n, channels [M + j*I2 : M + (j+1)*I2] in batch.x.

Block-diagonal trick (identical to NWPAttentionLayer.forward_sequence)
-----------------------------------------------------------------------
  Flattens T time-steps into one GATv2 call by expanding edge_index to cover
  T independent bipartite graphs stacked block-diagonally.  Each block has
  N_s*k source nodes (grid points) and N_s destination nodes (stations).

  Time-invariant attention: Zero-Query h_q = zeros(T*N_s, nwp_out_dim) means
  attention scores are driven entirely by NWP source features (geography).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv

from geostatistics.dcrnn.model.nwp_attention import (
    _expand_hetero_edge_index, _idw_weights, validate_idw_p, validate_alpha_alt,
)

# Altitude normaliser of the HOMOGENEOUS sampler's edge_attr[:, 3]
# (homo_sampler._init_grid_knn: (grid_alt - station_alt) / 500, clipped to +-3).
# NOT the 3000 m literal of the heterogeneous graph builder used by DCRNN.
_HOMO_ALT_NORM_M = 500.0


class HomoNWPAttentionLayer(nn.Module):
    """
    Bipartite GATv2: k NWP grid-point nodes → station nodes.

    Parameters
    ----------
    nwp_feat_dim : raw NWP features per grid point (I2)
    nwp_out_dim  : output dimension; must be divisible by heads
    heads        : number of GATv2 attention heads
    edge_dim     : edge-attribute dimension (None = no edge attrs)
    dropout      : GATv2 attention dropout
    """

    def __init__(
        self,
        nwp_feat_dim: int,
        nwp_out_dim: int,
        heads: int = 4,
        edge_dim: int | None = None,
        dropout: float = 0.0,
        aggregation: str = "attention",   # "attention" | "idw" | "idw_alt" (2026-09-23, IDW + site obs.)
        idw_p: float = 2.0,
        alpha_alt: float = 10.0,
        max_dist_km: float = 0.0,          # idw_alt only: physical-km normaliser of edge_attr[:, 0]
    ) -> None:
        super().__init__()
        if aggregation not in ("attention", "idw", "idw_alt"):
            raise ValueError(f"aggregation must be 'attention', 'idw' or 'idw_alt', got {aggregation!r}")
        self.aggregation  = aggregation
        self.nwp_feat_dim = nwp_feat_dim
        self.nwp_out_dim  = nwp_out_dim
        self.idw_p        = float(idw_p)
        self.alpha_alt    = float(alpha_alt)
        self.max_dist_km  = float(max_dist_km)
        if aggregation == "attention":
            if nwp_out_dim % heads != 0:
                raise ValueError(
                    f"nwp_out_dim ({nwp_out_dim}) must be divisible by heads ({heads})"
                )
            out_per_head = nwp_out_dim // heads
            # dst_dim = nwp_out_dim used only for Zero-Query (actual values are 0)
            self.gat = GATv2Conv(
                in_channels=(nwp_feat_dim, nwp_out_dim),
                out_channels=out_per_head,
                heads=heads,
                concat=True,
                edge_dim=edge_dim,
                add_self_loops=False,
                dropout=dropout,
            )
        else:
            # idw / idw_alt: mirrors NWPAttentionLayer (DCRNN path). Only the
            # learnable source projection survives, the weights are prescribed
            # from the edge geometry (distance, and height for idw_alt).
            validate_idw_p(self.idw_p)
            if aggregation == "idw_alt":
                validate_alpha_alt(self.alpha_alt)
                if self.max_dist_km <= 0.0:
                    raise ValueError(
                        "aggregation='idw_alt' needs max_dist_km > 0 (physical-km normaliser "
                        "of the sampler's distance column, HomoSampler.nwp_max_dist_km / "
                        f"ecmwf_max_dist_km), got {max_dist_km!r}."
                    )
            self.lin = nn.Linear(nwp_feat_dim, nwp_out_dim, bias=True)
        self.norm = nn.LayerNorm(nwp_out_dim)

    def _idw_sequence(
        self,
        nwp_seq: Tensor,               # (T, N_s*k, nwp_feat_dim)
        N_s: int,
        k: int,
        edge_attr: Tensor,             # (N_s*k, 4): dist_norm, sin, cos, alt_diff/500
    ) -> Tensor:                       # (T, N_s, nwp_out_dim)
        if edge_attr is None:
            raise ValueError("aggregation in {'idw', 'idw_alt'} needs edge_attr (distance column)")
        T = nwp_seq.size(0)
        E = N_s * k
        device = nwp_seq.device
        dst = torch.repeat_interleave(torch.arange(N_s, device=device), k)   # (E,)
        if self.aggregation == "idw_alt":
            d_km    = edge_attr[:, 0].clamp(min=0.0) * self.max_dist_km
            dalt_km = (edge_attr[:, 3] * _HOMO_ALT_NORM_M / 1000.0).abs()
            dist    = torch.sqrt(d_km.pow(2) + (self.alpha_alt * dalt_km).pow(2))
        else:
            dist = edge_attr[:, 0].clamp(min=0.0)
        w = _idw_weights(dist, dst, N_s, self.idw_p)                          # (E,), sums to 1 per station
        msg = self.lin(nwp_seq) * w.view(1, E, 1)                             # (T, E, d)
        # regular bipartite layout: edges n*k .. n*k+k-1 all point at station n
        out = msg.view(T, N_s, k, self.nwp_out_dim).sum(dim=2)               # (T, N_s, d)
        return self.norm(out)

    def forward_sequence(
        self,
        nwp_seq: Tensor,               # (T, N_s*k, nwp_feat_dim)
        N_s: int,
        k: int,
        edge_attr: Tensor | None = None,  # (N_s*k, edge_dim) optional
    ) -> Tensor:                       # (T, N_s, nwp_out_dim)
        if self.aggregation != "attention":
            return self._idw_sequence(nwp_seq, N_s, k, edge_attr)
        T      = nwp_seq.size(0)
        device = nwp_seq.device

        # Regular bipartite edge_index: j-th grid point of station n → station n
        src = torch.arange(N_s * k, device=device)
        dst = torch.repeat_interleave(torch.arange(N_s, device=device), k)
        ei  = torch.stack([src, dst])                                   # (2, N_s*k)

        # Block-diagonal expansion for T timesteps → (2, T * N_s * k)
        ei_exp = _expand_hetero_edge_index(ei, N_s * k, N_s, T)

        # Flatten time: (T * N_s * k, nwp_feat_dim)
        nwp_flat = nwp_seq.reshape(T * N_s * k, self.nwp_feat_dim)

        # Zero-Query: attention driven purely by NWP features
        h_q = torch.zeros(T * N_s, self.nwp_out_dim, device=device)   # (T*N_s, d)

        # Expand edge attributes if provided
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            ea_exp = edge_attr.repeat(T, 1)                             # (T*N_s*k, e)
        else:
            ea_exp = None

        msg = self.gat((nwp_flat, h_q), ei_exp, ea_exp)                # (T*N_s, d)
        return self.norm(msg).reshape(T, N_s, self.nwp_out_dim)        # (T, N_s, d)
