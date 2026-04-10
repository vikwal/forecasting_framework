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
graph edge weights and D_out is the out-degree matrix.  For K hops:

    DC(X) = Σ_{k=0}^{K}  θ_k · P^k X

Each hop is computed iteratively via message passing so no dense
matrix-matrix products are needed.  The weight matrices θ_k are
independent Linear layers (no shared weights across hops).

Edge weights
------------
Call ``DCGRUCell.edge_weight_from_attr(edge_attr)`` to derive scalar
edge weights from the graph-builder's edge_attr tensor.  The first
column is the normalised geodesic distance ∈ [0,1], so we use
  w_ij = exp(−d_ij)
which gives higher weight to closer neighbours.  DiffConv then applies
row-normalisation internally (P = D^{-1} W).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


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

        # Row-normalise: P_ij = w_ij / Σ_j w_ij  (out-degree of source)
        row = edge_index[0]
        out_deg = degree(row, num_nodes=N, dtype=x.dtype).clamp(min=1.0)
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
# DCGRU cell
# ---------------------------------------------------------------------------

class DCGRUCell(nn.Module):
    """
    Single DCGRU cell.

    Replaces W*X + U*H with DiffConv([X, H]) for each gate.
    The candidate gate uses DiffConv([X, r ⊙ H]) (element-wise reset).

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
        self.dc_reset  = DiffConv(xh_dim, hidden_dim, K)
        self.dc_update = DiffConv(xh_dim, hidden_dim, K)
        self.dc_cand   = DiffConv(xh_dim, hidden_dim, K)

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
    def edge_weight_from_attr(edge_attr: Tensor) -> Tensor:
        """
        Derive scalar edge weights from the graph-builder edge_attr.

        Convention: column 0 of edge_attr is the normalised geodesic
        distance ∈ [0, 1].  Returns exp(−d) ∈ (0, 1].

        Parameters
        ----------
        edge_attr : (E, F) tensor from HeteroData

        Returns
        -------
        (E,) float32 tensor of positive edge weights
        """
        return torch.exp(-edge_attr[:, 0])
