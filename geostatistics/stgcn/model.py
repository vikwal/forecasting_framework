"""
geostatistics/stgcn/model.py — Inductive STGCN for spatial forecasting.

Based on "Spatio-Temporal Graph Convolutional Networks: A Deep Learning
Framework for Traffic Forecasting" (Yu et al., 2018), adapted for the
inductive setting.

Key differences from the original paper
-----------------------------------------
  • Inductive adaptive adjacency: MLP(static) → node embeddings replaces the
    fixed graph Laplacian / ChebConv of the transductive original.
  • Gated temporal convolutions (tanh ⊙ sigmoid) with causal padding replace
    the original GLU channel-split, preserving the same causal semantics.
  • No NWP injection — NWP is aggregated to station level by HomoSampler.

Architecture
------------
  Input projection → n_blocks × STConvBlock → output MLP (skip sum)

  STConvBlock (T-G-T sandwich):
    TemporalGate 1 : gated causal dilated Conv1d (tanh ⊙ sigmoid)
    GraphConv      : A_adp @ h @ W  (row-normalised soft adjacency, 1-hop)
    TemporalGate 2 : gated causal dilated Conv1d (tanh ⊙ sigmoid)
    BatchNorm + skip connection

  Adaptive adjacency (inductive):
    emb = MLP(static)
    E1  = tanh(α W1 emb),  E2 = tanh(α W2 emb)
    A   = softmax(ReLU(E1 · E2ᵀ))           — asymmetric, row-normalised

  Receptive field (kernel_size=3, n_blocks=5, dilations [1,2,4,8,16]):
    RF = 1 + Σ 2*(kernel_size-1)*d_i = 1 + 2*2*(1+2+4+8+16) = 125 ≥ T_total=96
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Gated temporal convolution
# ---------------------------------------------------------------------------

class TemporalGate(nn.Module):
    """Gated causal dilated Conv1d: output = tanh(h1) ⊙ sigmoid(h2).

    Input  : (B*N, C_in, T)
    Output : (B*N, C_out, T)  — same T via left causal padding
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            c_in, 2 * c_out,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self._causal_pad = pad
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x : (BN, C_in, T)
        x = F.pad(x, (self._causal_pad, 0))
        gates = self.conv(x)                                        # (BN, 2*C_out, T)
        C_h = gates.shape[1] // 2
        h = torch.tanh(gates[:, :C_h, :]) * torch.sigmoid(gates[:, C_h:, :])
        return self.dropout(h)                                      # (BN, C_out, T)


# ---------------------------------------------------------------------------
# 1-hop graph convolution
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    """Single-hop graph conv: h_out = A @ h_in @ W.

    x     : (N, BT, C_in)
    a_adp : (N, N) normalised adjacency
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.lin = nn.Linear(c_in, c_out)

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        h = torch.einsum("nm,mbc->nbc", a, x)   # (N, BT, C_in)
        return self.lin(h)                        # (N, BT, C_out)


# ---------------------------------------------------------------------------
# STGCN Block (T-G-T sandwich)
# ---------------------------------------------------------------------------

class STConvBlock(nn.Module):
    """Spatio-temporal conv block: TemporalGate → GraphConv → TemporalGate."""

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t1   = TemporalGate(c_in,     c_hidden, kernel_size, dilation, dropout)
        self.gcn  = GraphConv(c_hidden, c_hidden)
        self.t2   = TemporalGate(c_hidden, c_hidden, kernel_size, dilation, dropout)
        self.bn   = nn.BatchNorm1d(c_hidden)
        self.skip_proj = nn.Linear(c_hidden, c_hidden)
        self.res_proj  = nn.Linear(c_in, c_hidden) if c_in != c_hidden else nn.Identity()

    def forward(
        self,
        x: Tensor,
        a_adp: Tensor,
        skip_acc: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        # x : (B, N, T, C_in)
        B, N, T, C_in = x.shape

        res = self.res_proj(x)    # (B, N, T, C_h)

        # Temporal gate 1: (B, N, T, C_in) → (BN, C_in, T) → (BN, C_h, T) → (B, N, T, C_h)
        h = x.permute(0, 1, 3, 2).reshape(B * N, C_in, T)
        h = self.t1(h)                                              # (BN, C_h, T)
        C_h = h.shape[1]
        h = h.reshape(B, N, C_h, T).permute(0, 1, 3, 2)           # (B, N, T, C_h)

        # Graph conv: (B, N, T, C_h) → (N, B*T, C_h) → (B, N, T, C_h)
        h_s = h.permute(1, 0, 2, 3).reshape(N, B * T, C_h)
        h_s = self.gcn(h_s, a_adp)                                 # (N, BT, C_h)
        h   = h_s.reshape(N, B, T, C_h).permute(1, 0, 2, 3)       # (B, N, T, C_h)

        # Temporal gate 2: (B, N, T, C_h) → (BN, C_h, T) → (B, N, T, C_h)
        h = h.permute(0, 1, 3, 2).reshape(B * N, C_h, T)
        h = self.t2(h)                                              # (BN, C_h, T)
        h = h.reshape(B, N, C_h, T).permute(0, 1, 3, 2)           # (B, N, T, C_h)

        # Residual + BN
        h = h + res
        h = self.bn(h.reshape(B * N * T, C_h)).reshape(B, N, T, C_h)

        # Skip connection
        skip     = self.skip_proj(h)
        skip_acc = skip if skip_acc is None else skip_acc + skip

        return h, skip_acc


# ---------------------------------------------------------------------------
# STGCN Model
# ---------------------------------------------------------------------------

class STGCNModel(nn.Module):
    """
    Inductive STGCN for spatio-temporal forecasting.

    Parameters
    ----------
    in_channels     : M + I2 (measurement + NWP features per time step)
    static_dim      : dimension of static node features (6 for HomoSampler)
    hidden          : number of hidden channels
    n_blocks        : number of STConvBlocks (dilations [1,2,4,8,…];
                      ≥5 required for T_total=96 with kernel_size=3)
    emb_dim         : node embedding dimension for adaptive adjacency
    graph_alpha     : temperature scaling in E1, E2 = tanh(α · W · emb)
    kernel_size     : temporal convolution kernel size (default 3)
    dropout         : dropout probability
    history_length  : H
    forecast_horizon: F_h
    """

    def __init__(
        self,
        in_channels: int,
        static_dim: int,
        hidden: int,
        n_blocks: int = 5,
        emb_dim: int = 64,
        graph_alpha: float = 3.0,
        kernel_size: int = 3,
        dropout: float = 0.1,
        history_length: int = 48,
        forecast_horizon: int = 48,
    ) -> None:
        super().__init__()
        self.H     = history_length
        self.Fh    = forecast_horizon
        self.alpha = graph_alpha

        # Inductive embedding MLP: static → node embedding
        self.emb_mlp = nn.Sequential(
            nn.Linear(static_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # Asymmetric projections for adaptive adjacency
        self.adp_W1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.adp_W2 = nn.Linear(emb_dim, emb_dim, bias=False)

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden)

        # T-G-T blocks with exponential dilations.
        # RF = 1 + Σ_i 2*(kernel_size-1)*2^i  (two temporal gates per block)
        # n_blocks=5, kernel_size=3 → RF = 1 + 2*2*(1+2+4+8+16) = 125 ≥ T_total=96
        dilations = [2 ** i for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            STConvBlock(
                c_in=hidden, c_hidden=hidden,
                kernel_size=kernel_size,
                dilation=dilations[i],
                dropout=dropout,
            )
            for i in range(n_blocks)
        ])

        # Output MLP from accumulated skip connections
        self.out_fc1 = nn.Linear(hidden, hidden)
        self.out_fc2 = nn.Linear(hidden, forecast_horizon)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Adaptive adjacency
    # ------------------------------------------------------------------

    def _build_adjacency(self, static: Tensor) -> Tensor:
        """Inductive adaptive adjacency.

        A_adp = softmax(ReLU(E1 · E2ᵀ))  where
        E1 = tanh(α · W1 · emb),  E2 = tanh(α · W2 · emb)

        static : (N, S)
        returns: (N, N) row-normalised soft adjacency
        """
        emb = self.emb_mlp(static)
        E1  = torch.tanh(self.alpha * self.adp_W1(emb))
        E2  = torch.tanh(self.alpha * self.adp_W2(emb))
        A   = F.relu(E1 @ E2.T)
        return F.softmax(A, dim=1)                                   # (N, N)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        static: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        x           : (B, N, T_total, M+I2)
        static      : (B, N, 6) or (N, 6)
        target_mask : (B*N,) or (N,) bool — True = target station

        Returns
        -------
        (N_target_total, F_h)
        """
        B, N = x.shape[:2]

        static_single = static if static.dim() == 2 else static[0]

        a_adp = self._build_adjacency(static_single)   # (N, N)

        h = self.input_proj(x)                         # (B, N, T_total, hidden)

        skip_acc = None
        for block in self.blocks:
            h, skip_acc = block(h, a_adp, skip_acc)

        # Output from skip sum: use last time step
        out = skip_acc[:, :, -1, :]                    # (B, N, hidden)
        out = self.dropout(F.relu(self.out_fc1(out)))
        out = self.out_fc2(out)                        # (B, N, F_h)

        out_flat = out.reshape(B * N, self.Fh)
        if target_mask.shape[0] == N:
            mask_flat = target_mask.repeat(B)
        else:
            mask_flat = target_mask

        return out_flat[mask_flat]                     # (N_target, F_h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
