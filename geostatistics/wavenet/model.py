"""
geostatistics/wavenet/model.py — Inductive Graph WaveNet for spatial forecasting.

Based on "Adaptive Graph Convolutional Recurrent Network for Traffic
Forecasting" (Wu et al., 2019, "Graph WaveNet"), adapted for the inductive
setting.

Key differences from the original paper
-----------------------------------------
  • **Inductive self-adaptive adjacency**: node-ID embeddings are replaced by
    MLP(static_features) → E.  Adaptive adjacency:
      A_adp = softmax(ReLU(E1 · E2ᵀ))
    where E1 = tanh(α · W1 · E),  E2 = tanh(α · W2 · E).
    This allows the model to generalise to unseen stations at inference time.
  • **Fixed adjacency**: not used here (no pre-built graph required) — the
    self-adaptive adjacency fully replaces it, consistent with the ablation
    "noAdp" → "adp-only" in the paper.
  • **Input / output**: same convention as MTGNNModel — (B, N, T_total, M+I2)
    in, (N_target, F_h) out.

Architecture
------------
  Input norm + linear → stacked GWNBlocks → output MLP (skip connections)

  GWNBlock:
    Residual TCN branch: gated causal dilated Conv1d (tanh ⊙ sigmoid)
    Graph branch:        K-hop power-series diffusion over A_adp (paper Eq. 7)
    Output: relu(gated_tcn + gcn)  +  skip connection

  Diffusion convolution (MultiHopDiffusion):
    Z = Σ_{k=0}^K  A_adp^k · X · W_k   (K=2 paper default)
    (No forward/backward split — adaptive adj is already asymmetric)

  Output:
    Sum of skip connections from all blocks
    → ReLU → Linear(hidden → hidden) → ReLU → Linear(hidden → F_h)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geostatistics.shared.nwp_gat import HomoNWPAttentionLayer


# ---------------------------------------------------------------------------
# Multi-Hop Diffusion Graph Convolution
# ---------------------------------------------------------------------------

class MultiHopDiffusion(nn.Module):
    """K-hop power-series diffusion: Z = Σ_{k=0}^K A^k · X · W_k  (paper Eq. 7).

    k=0 term retains ego-features (identity path); k≥1 terms aggregate
    neighbourhood information at increasing hop distances.
    """

    def __init__(self, c_in: int, c_out: int, K: int = 2) -> None:
        super().__init__()
        self.K    = K
        self.lins = nn.ModuleList([nn.Linear(c_in, c_out) for _ in range(K + 1)])

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        # x : (N, BT, C_in),  a : (N, N) normalised adjacency
        h   = x
        out = self.lins[0](x)                              # k=0: identity
        for k in range(1, self.K + 1):
            h   = torch.einsum("nm,mbc->nbc", a, h)       # A^k · X
            out = out + self.lins[k](h)
        return out                                         # (N, BT, C_out)


# ---------------------------------------------------------------------------
# Graph WaveNet Block
# ---------------------------------------------------------------------------

class GWNBlock(nn.Module):
    """Gated TCN + diffusion GCN with residual and skip."""

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        kernel_size: int = 2,
        dilation: int = 1,
        dropout: float = 0.1,
        K: int = 2,
    ) -> None:
        super().__init__()
        # Gated temporal convolution: outputs 2 * c_hidden for gate split
        pad = (kernel_size - 1) * dilation
        self.tcn = nn.Conv1d(
            c_in, 2 * c_hidden,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,   # manual causal padding
        )
        self._causal_pad = pad

        # Graph convolution (K-hop power-series, adaptive adj only)
        self.gcn = MultiHopDiffusion(c_hidden, c_hidden, K=K)

        self.ln      = nn.LayerNorm(c_hidden)
        self.dropout = nn.Dropout(dropout)
        self.skip_conv = nn.Linear(c_hidden, c_hidden)
        # 1×1 residual projection when c_in != c_hidden
        self.res_proj = nn.Linear(c_in, c_hidden) if c_in != c_hidden else nn.Identity()

    def forward(
        self,
        x: Tensor,
        a_adp: Tensor,
        skip_acc: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        # x : (B, N, T, C_in)
        B, N, T, C_in = x.shape
        C_h = self.tcn.out_channels // 2

        # Residual path
        res = self.res_proj(x)  # (B, N, T, C_h)

        # Gated TCN: permute to (BN, C_in, T), causal-pad, convolve
        h = x.permute(0, 1, 3, 2).reshape(B * N, C_in, T)
        h = F.pad(h, (self._causal_pad, 0))
        gates = self.tcn(h)                                       # (BN, 2*C_h, T)
        h = torch.tanh(gates[:, :C_h, :]) * torch.sigmoid(gates[:, C_h:, :])  # (BN, C_h, T)
        h = self.dropout(h)
        h = h.reshape(B, N, C_h, T).permute(0, 1, 3, 2)          # (B, N, T, C_h)

        # Spatial GCN: reshape to (N, B*T, C_h)
        h_s = h.permute(1, 0, 2, 3).reshape(N, B * T, C_h)
        h_s = self.gcn(h_s, a_adp)
        h   = h_s.reshape(N, B, T, C_h).permute(1, 0, 2, 3)      # (B, N, T, C_h)

        # Residual + LN
        h = self.ln(h + res)

        # Skip
        skip = self.skip_conv(h)
        skip_acc = skip if skip_acc is None else skip_acc + skip

        return h, skip_acc


# ---------------------------------------------------------------------------
# Graph WaveNet Model
# ---------------------------------------------------------------------------

class GraphWaveNetModel(nn.Module):
    """
    Inductive Graph WaveNet for spatio-temporal forecasting.

    Parameters
    ----------
    in_channels     : M + I2 (measurement + NWP features per time step)
    static_dim      : dimension of static node features (6 for HomoSampler)
    hidden          : number of hidden channels
    n_blocks        : number of GWNBlocks (dilations cycle through [1,2,4,8,16,32];
                      ≥12 required for T_total=96 with kernel_size=2)
    K_hop     : hops in MultiHopDiffusion (paper default K=2)
    emb_dim         : node embedding dimension for adaptive adjacency
    graph_alpha     : temperature scaling in E1, E2 = tanh(α · W · emb)
    kernel_size     : TCN kernel size (default 2)
    dropout         : dropout probability
    history_length  : H
    forecast_horizon: F_h
    """

    def __init__(
        self,
        in_channels: int,
        static_dim: int,
        hidden: int,
        n_blocks: int = 12,
        K_hop: int = 2,
        emb_dim: int = 64,
        graph_alpha: float = 3.0,
        kernel_size: int = 2,
        dropout: float = 0.1,
        history_length: int = 48,
        forecast_horizon: int = 48,
        nwp_nodes: bool = False,
        nwp_feat_dim: int = 0,
        k_nwp: int = 4,
        nwp_out_dim: int = 32,
        nwp_heads: int = 4,
        M: int = 0,
    ) -> None:
        super().__init__()
        self.H         = history_length
        self.Fh        = forecast_horizon
        self.alpha     = graph_alpha
        self.nwp_nodes = nwp_nodes
        self.M         = M
        self.k_nwp     = k_nwp
        self.nwp_feat_dim = nwp_feat_dim

        if nwp_nodes:
            self.nwp_attn = HomoNWPAttentionLayer(
                nwp_feat_dim=nwp_feat_dim,
                nwp_out_dim=nwp_out_dim,
                heads=nwp_heads,
            )
            self.nwp_i2_channels = k_nwp * nwp_feat_dim  # slice boundary in x
        proj_in = in_channels

        # Inductive embedding: static → E
        self.emb_mlp = nn.Sequential(
            nn.Linear(static_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # Asymmetric projections for adaptive adjacency
        self.adp_W1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.adp_W2 = nn.Linear(emb_dim, emb_dim, bias=False)

        # Edge-feature bias: [dist_norm, sin_bearing, cos_bearing, alt_diff_norm] → scalar
        self.edge_fc = nn.Linear(4, 1, bias=True)

        # Input projection
        self.input_proj = nn.Linear(proj_in, hidden)

        # Dilation cycle [1,2,4,8,16,32]: 2 full cycles (12 blocks) give
        # RF = 1 + (kernel_size-1)*sum(dilations) = 1 + 126 = 127 ≥ T_total=96.
        # A shorter cycle [1,2,4,8,16] would require 17+ blocks for the same coverage.
        dilation_cycle = [1, 2, 4, 8, 16, 32]
        dilations = [dilation_cycle[i % len(dilation_cycle)] for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            GWNBlock(
                c_in=hidden, c_hidden=hidden,
                kernel_size=kernel_size,
                dilation=dilations[i],
                dropout=dropout,
                K=K_hop,
            )
            for i in range(n_blocks)
        ])

        # Output MLP from skip sum
        self.out_fc1 = nn.Linear(hidden, hidden)
        self.out_fc2 = nn.Linear(hidden, forecast_horizon)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Adaptive adjacency
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_edge_features(static: Tensor) -> Tensor:
        """Directed pairwise edge features from static node features.

        Recovers lat/lon from sin/cos encoding (columns 0-3), uses normalised
        altitude (column 4).  Returns 4-dim feature vector per directed edge:
          [dist_norm, sin(bearing i→j), cos(bearing i→j), alt_diff_norm]

        static  : (N, 6+)
        returns : (N, N, 4)  — row=source, col=destination
        """
        lat = torch.atan2(static[:, 0], static[:, 1])
        lon = torch.atan2(static[:, 2], static[:, 3])
        alt = static[:, 4]

        lat_i = lat.unsqueeze(1)
        lat_j = lat.unsqueeze(0)
        lon_i = lon.unsqueeze(1)
        lon_j = lon.unsqueeze(0)
        dlat  = lat_j - lat_i
        dlon  = lon_j - lon_i

        a = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat_i) * torch.cos(lat_j) * torch.sin(dlon / 2) ** 2)
        dist_km   = 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())
        dist_norm = dist_km / dist_km.max().clamp(min=1e-8)

        y       = torch.sin(dlon) * torch.cos(lat_j)
        x       = (torch.cos(lat_i) * torch.sin(lat_j)
                   - torch.sin(lat_i) * torch.cos(lat_j) * torch.cos(dlon))
        bearing = torch.atan2(y, x)

        alt_diff = (alt.unsqueeze(0) - alt.unsqueeze(1)).clamp(-3.0, 3.0) / 3.0

        return torch.stack(
            [dist_norm, torch.sin(bearing), torch.cos(bearing), alt_diff], dim=-1
        )

    def _build_adjacency(self, static: Tensor) -> Tensor:
        """Inductive adaptive adjacency with pairwise edge features.

        A_adp = softmax(ReLU(E1 · E2ᵀ + edge_bias))  where
        E1 = tanh(α · W1 · emb),  E2 = tanh(α · W2 · emb)

        static : (N, S)
        returns: (N, N) row-normalised soft adjacency
        """
        emb = self.emb_mlp(static)                             # (N, d_emb)
        E1  = torch.tanh(self.alpha * self.adp_W1(emb))
        E2  = torch.tanh(self.alpha * self.adp_W2(emb))

        ef        = self._pairwise_edge_features(static)       # (N, N, 4)
        edge_bias = self.edge_fc(ef).squeeze(-1)               # (N, N)

        A = F.relu(E1 @ E2.T + edge_bias)
        return F.softmax(A, dim=1)                             # (N, N)

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
        B, N, T = x.shape[:3]

        if static.dim() == 2:
            static_single = static
        else:
            static_single = static[0]

        # Adaptive adjacency (built once per forward)
        a_adp = self._build_adjacency(static_single)   # (N, N)

        # NWP aggregation via GATv2 when nwp_nodes=True (B=1 guaranteed by HomoSampler)
        if self.nwp_nodes:
            i2_end = self.M + self.nwp_i2_channels
            meas   = x[..., :self.M]         # (B, N, T, M)
            nwp_i2 = x[..., self.M:i2_end]  # (B, N, T, k*I2)
            ecmwf  = x[..., i2_end:]         # (B, N, T, k_e*E2) — empty when k_ecmwf=0
            # Reshape ICON-D2 to (T, N*k, I2) for HomoNWPAttentionLayer
            nwp_t = nwp_i2.reshape(B, N, T, self.k_nwp, self.nwp_feat_dim)  # (B,N,T,k,I2)
            nwp_t = nwp_t.permute(0, 2, 1, 3, 4)                            # (B,T,N,k,I2)
            nwp_t = nwp_t.reshape(B, T, N * self.k_nwp, self.nwp_feat_dim)  # (B,T,N*k,I2)
            nwp_agg = self.nwp_attn.forward_sequence(nwp_t[0], N, self.k_nwp)  # (T,N,d)
            nwp_agg = nwp_agg.permute(1, 0, 2).unsqueeze(0)                 # (1,N,T,d)
            x = torch.cat([meas, nwp_agg, ecmwf], dim=-1)                   # (B,N,T,M+d+E)

        # Input projection
        h = self.input_proj(x)                         # (B, N, T_total, hidden)

        # Stacked blocks, accumulate skip
        skip_acc = None
        for block in self.blocks:
            h, skip_acc = block(h, a_adp, skip_acc)

        # Output from skip sum: use last time step
        out = skip_acc[:, :, -1, :]                    # (B, N, hidden)
        out = self.dropout(F.relu(self.out_fc1(out)))
        out = self.out_fc2(out)                        # (B, N, F_h)

        # Select target stations
        out_flat = out.reshape(B * N, self.Fh)
        if target_mask.shape[0] == N:
            mask_flat = target_mask.repeat(B)
        else:
            mask_flat = target_mask

        return out_flat[mask_flat]                     # (N_target, F_h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
