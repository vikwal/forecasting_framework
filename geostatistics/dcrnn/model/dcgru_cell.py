"""
DCGRUCell — Diffusion Convolutional GRU cell (DCRNN, Li et al. 2018).

The standard GRU's linear transforms  W*X + U*H  are replaced by
K-hop diffusion convolutions on the station graph:

    r  = σ( DC([X, H]) )
    z  = σ( DC([X, H]) )
    c  = tanh( DC([X, r ⊙ H]) )
    H' = z ⊙ H + (1-z) ⊙ c

where DC(·) is a learnable K-hop random-walk diffusion convolution.

Diffusion convolution (DiffConv)
---------------------------------
The random-walk transition matrix is P = D_out^{-1} W, where W are the
graph edge weights and D_out is the WEIGHTED out-degree matrix
(paper Section 2.2: D_O = diag(W·1)).  For K hops:

    DC(X) = Σ_{k=0}^{K}  θ_k · P^k X

Each hop is computed iteratively via message passing so no dense
matrix-matrix products are needed.  The weight matrices θ_k are
independent Linear layers (no shared weights across hops).

Edge weights
------------
Call ``DCGRUCell.edge_weight_from_attr(edge_attr)`` to derive scalar
edge weights from the graph-builder's edge_attr tensor.  The first
column is the normalised geodesic distance ∈ [0,1].  Weights are
computed as a Gaussian kernel:
  w_ij = exp(-d_ij² / σ²)
which gives higher weight to closer neighbours.  DiffConv then applies
row-normalisation internally (P = D_O^{-1} W).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing


# ---------------------------------------------------------------------------
# Diffusion convolution primitive
# ---------------------------------------------------------------------------

class DiffConv(MessagePassing):
    """
    K-hop random-walk diffusion convolution.

    DC(X) = Σ_{k=0}^{K}  Linear_k(P^k X)

    where P = D^{-1} W  (row-normalised).

    Parameters
    ----------
    in_channels  : input feature dimension
    out_channels : output feature dimension
    K            : number of diffusion hops (K=0 → pure self-loop / MLP)
    bias         : learnable bias on the final output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 2,
        bias: bool = True,
    ) -> None:
        """
        K-hop random-walk diffusion convolution.

        DC(X) = Σ_{k=0}^{K}  Linear_k(P^k X)

        Note on K convention
        --------------------
        Here K is the MAXIMUM HOP INDEX, giving K+1 terms total (k=0..K).
        Paper Eq. 2 uses K as the NUMBER OF TERMS (k=0..K-1).
        → Our default K=2 (3 terms) ≡ paper's K=3.
        """
        super().__init__(aggr="add")
        self.K = K
        # One Linear per hop (k=0 is self-transform, k=1..K are diffusion hops)
        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(K + 1)
        ])
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in self.lins:
            nn.init.xavier_uniform_(lin.weight)

    def forward(
        self,
        x: Tensor,           # (N, in_channels)
        edge_index: Tensor,  # (2, E)
        edge_weight: Tensor, # (E,)  unnormalised; row-normalised here
    ) -> Tensor:
        N = x.size(0)

        # Row-normalise: P_ij = w_ij / Σ_j w_ij  (weighted out-degree of source)
        # Paper Section 2.2: D_O = diag(W·1) — sum of outgoing edge *weights*, not edge count
        row = edge_index[0]
        out_deg = torch.zeros(N, dtype=x.dtype, device=x.device)
        out_deg.scatter_add_(0, row, edge_weight)
        out_deg = out_deg.clamp(min=1e-8)
        norm_w = edge_weight / out_deg[row]   # (E,)

        # k=0: self-transform
        out = self.lins[0](x)
        x_k = x
        for k in range(1, self.K + 1):
            x_k = self.propagate(edge_index, x=x_k, edge_weight=norm_w)  # (N, in)
            out = out + self.lins[k](x_k)

        if self.bias_param is not None:
            out = out + self.bias_param
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.unsqueeze(-1) * x_j


# ---------------------------------------------------------------------------
# Bidirectional diffusion convolution primitive
# ---------------------------------------------------------------------------

class BiDirDiffConv(nn.Module):
    """
    Bidirectional K-hop diffusion convolution (DCRNN, Li et al. 2018).

    DC_bidir(X) = Σ_{k=0}^{K}  (Linear_fwd_k(P_fwd^k X) + Linear_bwd_k(P_bwd^k X))

    P_fwd = D_O^{-1} W  (forward:  original edge direction)
    P_bwd = D_I^{-1} W^T (backward: reversed edge direction)

    Separate weight matrices per direction allow the model to distinguish
    upstream from downstream influence — as in the original DCRNN paper.
    Only used for station-to-station (s2s) edges; NWP→station edges are
    handled by GATv2Conv in NWPAttentionLayer and remain unidirectional.

    Parameters
    ----------
    in_channels  : input feature dimension
    out_channels : output feature dimension
    K            : number of diffusion hops
    bias         : learnable bias on the final output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 2,
        bias: bool = True,
    ) -> None:
        """
        Bidirectional K-hop diffusion convolution.

        Note on K convention
        --------------------
        K is the MAXIMUM HOP INDEX (K+1 terms, k=0..K) — same as DiffConv.
        Paper Eq. 2 uses K as the number of terms (k=0..K-1).
        → Our default K=2 ≡ paper's K=3.
        """
        super().__init__()
        self.fwd = DiffConv(in_channels, out_channels, K, bias=False)
        self.bwd = DiffConv(in_channels, out_channels, K, bias=False)
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        edge_index_rev = edge_index.flip(0)   # swap src ↔ dst for backward pass
        out = self.fwd(x, edge_index, edge_weight) \
            + self.bwd(x, edge_index_rev, edge_weight)
        if self.bias_param is not None:
            out = out + self.bias_param
        return out


# ---------------------------------------------------------------------------
# DCGRU cell
# ---------------------------------------------------------------------------

class DCGRUCell(nn.Module):
    """
    Single DCGRU cell.

    Replaces W*X + U*H with BiDirDiffConv([X, H]) for each gate.
    The candidate gate uses BiDirDiffConv([X, r ⊙ H]) (element-wise reset).
    Bidirectional diffusion uses separate forward/backward weights per gate,
    matching the DCRNN paper (Li et al. 2018, Eq. 2).

    Parameters
    ----------
    input_dim  : dimension of input features at each step
    hidden_dim : GRU hidden state dimension
    K          : diffusion hops
    dropout    : applied to input before the gates
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        K: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop = nn.Dropout(dropout)

        # [X ‖ H] concatenation: input_dim + hidden_dim → hidden_dim
        xh_dim = input_dim + hidden_dim
        self.dc_reset  = BiDirDiffConv(xh_dim, hidden_dim, K)
        self.dc_update = BiDirDiffConv(xh_dim, hidden_dim, K)
        self.dc_cand   = BiDirDiffConv(xh_dim, hidden_dim, K)

    def forward(
        self,
        x: Tensor,           # (N, input_dim)
        h: Tensor,           # (N, hidden_dim)
        edge_index: Tensor,  # (2, E)
        edge_weight: Tensor, # (E,)
    ) -> Tensor:
        """Returns updated hidden state (N, hidden_dim)."""
        x = self.drop(x)
        xh = torch.cat([x, h], dim=-1)              # (N, input_dim+hidden_dim)

        r = torch.sigmoid(self.dc_reset(xh,  edge_index, edge_weight))
        z = torch.sigmoid(self.dc_update(xh, edge_index, edge_weight))

        xrh = torch.cat([x, r * h], dim=-1)         # (N, input_dim+hidden_dim)
        c   = torch.tanh(self.dc_cand(xrh, edge_index, edge_weight))

        h_new = z * h + (1.0 - z) * c               # (N, hidden_dim)
        return h_new

    @staticmethod
    def edge_weight_from_attr(edge_attr: Tensor, sigma: float = 0.2) -> Tensor:
        """
        Gaussian kernel over normalised distance: exp(-d²/σ²).

        Paper Appendix E.1:
            W_ij = exp(-dist(v_i,v_j)² / σ²)  if dist ≤ κ, else 0

        The κ-threshold is not applied here; the graph builder already
        sparsifies via k-nearest-neighbours.

        Parameters
        ----------
        edge_attr : (E, F) tensor — column 0 is normalised geodesic distance ∈ [0, 1]
        sigma     : scale parameter of the Gaussian kernel (default: 0.2)

        Returns
        -------
        (E,) float32 tensor of positive edge weights
        """
        d = edge_attr[:, 0]
        return torch.exp(-(d ** 2) / (sigma ** 2))

    @staticmethod
    def directional_edge_weight(
        flow_dir_rad: Tensor,   # (N_s,) wind flow direction in radians (geographic, North=0 CW)
        src_indices: Tensor,    # (E,)   source node index per edge
        dist_norm: Tensor,      # (E,)   normalised geodesic distance ∈ [0,1]
        bearing_rad: Tensor,    # (E,)   azimuth src→dst in radians
    ) -> Tensor:
        """
        Wind-conditioned edge weights: exp(−d) × clamp(cos(flow_dir − bearing), 0).

        flow_dir_rad is the direction the wind is blowing TOWARDS in radians
        (geographic convention: North=0, clockwise positive).
        Edges whose bearing aligns with the wind flow receive higher weight;
        opposing edges get weight 0.
        """
        alignment = torch.cos(flow_dir_rad[src_indices] - bearing_rad).clamp(min=0.0)
        return torch.exp(-dist_norm) * alignment
