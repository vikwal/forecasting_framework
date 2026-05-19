"""
geostatistics/mtgnn/model.py — Inductive MTGNN for spatial wind/solar forecasting.

Based on "Connecting the Dots: Multivariate Time Series Forecasting with Graph
Neural Networks" (Wu et al., 2020), adapted for the inductive setting where
node identities are unknown at training time.

Key differences from the original paper
-----------------------------------------
  • **Inductive graph learning**: instead of transductive learnable node-ID
    embeddings, static station features (lat, lon, alt, type) are mapped by a
    small MLP to node embeddings.  The graph-learning formula is otherwise
    identical: A = ReLU(tanh(α (M1 M2ᵀ − M2 M1ᵀ))).
  • **Input**: (B, N, T_total, M+I2) where the first H steps contain real
    measurements (target stations zeroed by IGNNK) and the last F_h steps
    carry only NWP (measurements padded with 0).
  • **Output**: (B, N_target, F_h) — forecast for masked stations only.
  • **Curriculum learning**: controlled externally via the `cl_steps` argument
    to `forward()`.  The trainer increments cl_steps from 1 to F_h.

Architecture
------------
  Input projection → stacked MTGNNLayers → output MLP

  MTGNNLayer:
    tc_filter: DilatedInception (kernel sizes 2, 3, 6, 7, concat → c_hidden channels)
    tc_gate:   DilatedInception (kernel sizes 2, 3, 6, 7, concat → c_hidden channels)
      → gating tanh(tc_filter) ⊙ sigmoid(tc_gate)  — all scales in both paths
    mixhop_fwd: MixHopProp with a_hat_fwd   (inflow, A)
    mixhop_bwd: MixHopProp with a_hat_bwd   (outflow, Aᵀ separately normalised)
      → outputs summed → skip connection

  Graph learning (once per forward pass):
    static (N, S) → MLP → E ∈ ℝ^(N, d_emb)
    M1 = tanh(α · W1·E),  M2 = tanh(α · W2·E)  [separate projections]
    A_raw = ReLU(tanh(α · (M1 M2ᵀ − M2 M1ᵀ) + edge_bias))
    → diagonal = 0 → TopK per row (if topk_graph set) → +I (self-loop)
    → a_hat_fwd: row-normalise  (paper Eq. 8)
    → a_hat_bwd: A_raw^T row-normalised independently  (DCRNN/GWN convention)

  MixHop propagation (K hops):
    H^(0) = H_in
    H^(k) = β · H_in + (1−β) · Â · H^(k−1)
    out = Linear(concat[H^(0), H^(1), …, H^(K)])
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geostatistics.shared.nwp_gat import HomoNWPAttentionLayer


# ---------------------------------------------------------------------------
# Dilated Inception Block
# ---------------------------------------------------------------------------

class DilatedInception(nn.Module):
    """Four parallel causal dilated convolutions with kernel sizes [2, 3, 6, 7].

    Input  : (B*N, C_in, T)                — time as last dim (Conv1d convention)
    Output : (B*N, 4 * c_per_branch, T)    — branches concatenated (paper Eq. 12)

    Each branch uses padding (kernel_size - 1, 0) on the left so the output
    length equals the input length (strictly causal, no future leakage).
    Channel layout: [k=2: 0..B-1, k=3: B..2B-1, k=6: 2B..3B-1, k=7: 3B..4B-1]
    where B = c_per_branch.
    """

    KERNEL_SIZES = [2, 3, 6, 7]

    def __init__(self, c_in: int, c_per_branch: int, dilation: int = 1) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                c_in, c_per_branch,
                kernel_size=k,
                dilation=dilation,
                padding=0,   # manual causal padding below
            )
            for k in self.KERNEL_SIZES
        ])

    def forward(self, x: Tensor) -> Tensor:
        # x : (BN, C_in, T)
        branches: list[Tensor] = []
        for conv, k in zip(self.convs, self.KERNEL_SIZES):
            pad = (k - 1) * conv.dilation[0]
            branches.append(conv(F.pad(x, (pad, 0))))   # (BN, c_per_branch, T)
        return torch.cat(branches, dim=1)                # (BN, 4*c_per_branch, T)


# ---------------------------------------------------------------------------
# Mix-Hop Graph Propagation
# ---------------------------------------------------------------------------

class MixHopProp(nn.Module):
    """K-hop residual graph diffusion; concat all hops then project.

    H^(k) = β · H_in + (1−β) · Â · H^(k−1)
    out    = Linear(concat[H^(0) … H^(K)])

    A_hat is passed in (already row-normalised).
    """

    def __init__(self, c_in: int, K: int = 2, beta: float = 0.05) -> None:
        super().__init__()
        self.K    = K
        self.beta = beta
        self.proj = nn.Linear((K + 1) * c_in, c_in)

    def forward(self, x: Tensor, a_hat: Tensor) -> Tensor:
        # x     : (N, BT, C)  — reshape before calling
        # a_hat : (N, N) normalised adjacency
        hops = [x]
        h = x
        for _ in range(self.K):
            h = self.beta * x + (1 - self.beta) * torch.einsum("nm,mbc->nbc", a_hat, h)
            hops.append(h)
        return self.proj(torch.cat(hops, dim=-1))   # (N, BT, C)


# ---------------------------------------------------------------------------
# MTGNN Layer
# ---------------------------------------------------------------------------

class MTGNNLayer(nn.Module):
    """One MTGNN layer: temporal gating → spatial bi-directional mix-hop → skip."""

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        K_hop: int = 2,
        beta: float = 0.05,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        # Two independent inceptions (paper Fig. 5a): filter and gate paths are
        # separate so each sees all four kernel sizes, with independent weights.
        # c_per_branch = c_hidden//4 → concat 4 branches → c_hidden channels each.
        self.tc_filter  = DilatedInception(c_in, c_hidden // 4, dilation=dilation)
        self.tc_gate    = DilatedInception(c_in, c_hidden // 4, dilation=dilation)
        # Bi-directional mix-hop (paper Fig. 4a): inflow (A) and outflow (Aᵀ),
        # separate weight matrices, outputs summed.
        self.mixhop_fwd = MixHopProp(c_hidden, K=K_hop, beta=beta)
        self.mixhop_bwd = MixHopProp(c_hidden, K=K_hop, beta=beta)
        self.skip_proj  = nn.Linear(c_hidden, c_hidden)
        self.ln         = nn.LayerNorm(c_hidden)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        a_hat_fwd: Tensor,
        a_hat_bwd: Tensor,
        skip_acc: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        # x : (B, N, T, C_in)
        B, N, T, C_in = x.shape
        C_h = self.c_hidden

        # Temporal: two independent inceptions → element-wise gate (paper Fig. 5a)
        h_orig = x.permute(0, 1, 3, 2).reshape(B * N, C_in, T)    # (BN, C_in, T)
        filt   = torch.tanh(self.tc_filter(h_orig))                 # (BN, C_h, T)
        gate   = torch.sigmoid(self.tc_gate(h_orig))                # (BN, C_h, T)
        h      = self.dropout(filt * gate)                          # (BN, C_h, T)

        # Back to (B, N, T, C_h)
        h = h.reshape(B, N, C_h, T).permute(0, 1, 3, 2)

        # Spatial: bi-directional mix-hop, outputs summed (paper Fig. 4a)
        h_s = h.permute(1, 0, 2, 3).reshape(N, B * T, C_h)         # (N, BT, C_h)
        h_s = (self.mixhop_fwd(h_s, a_hat_fwd)
             + self.mixhop_bwd(h_s, a_hat_bwd))                     # (N, BT, C_h)
        h   = h_s.reshape(N, B, T, C_h).permute(1, 0, 2, 3)        # (B, N, T, C_h)

        # LN per (B, N, T) position over feature dim
        h = self.ln(h)

        # Skip
        skip     = self.skip_proj(h)
        skip_acc = skip if skip_acc is None else skip_acc + skip

        return h, skip_acc


# ---------------------------------------------------------------------------
# MTGNN Model
# ---------------------------------------------------------------------------

class MTGNNModel(nn.Module):
    """
    Inductive MTGNN for spatio-temporal forecasting.

    Parameters
    ----------
    in_channels     : M + I2 (measurement + NWP features per time step)
    static_dim      : dimension of static node features (6 for HomoSampler)
    hidden          : number of hidden channels per layer
    n_layers        : number of MTGNNLayers (≥5 required for T_total=96; see receptive_field())
    K_hop           : mix-hop order
    beta            : mix-hop residual weight
    emb_dim         : node embedding dimension for inductive graph learning
    graph_alpha     : scaling in graph learning tanh(α · M1 M2ᵀ)
    dropout         : dropout probability
    history_length  : H (number of history time steps)
    forecast_horizon: F_h (number of forecast time steps to produce)
    cl_steps        : curriculum steps (1..F_h); model uses last cl_steps output
                      steps.  Passed at forward() time, not at construction.
    topk_graph      : if set, keep only top-k neighbours per row in A before
                      adding self-loop (paper Eqs. 4-6).  None = dense graph.
    """

    def __init__(
        self,
        in_channels: int,
        static_dim: int,
        hidden: int,
        n_layers: int = 5,
        K_hop: int = 2,
        beta: float = 0.05,
        emb_dim: int = 64,
        graph_alpha: float = 3.0,
        dropout: float = 0.1,
        history_length: int = 48,
        forecast_horizon: int = 48,
        nwp_nodes: bool = False,
        nwp_feat_dim: int = 0,
        k_nwp: int = 4,
        nwp_out_dim: int = 32,
        nwp_heads: int = 4,
        M: int = 0,
        topk_graph: int | None = None,
    ) -> None:
        super().__init__()
        self.H            = history_length
        self.Fh           = forecast_horizon
        self.alpha        = graph_alpha
        self.nwp_nodes    = nwp_nodes
        self.M            = M
        self.k_nwp        = k_nwp
        self.nwp_feat_dim = nwp_feat_dim
        self.topk_graph   = topk_graph
        T_total = history_length + forecast_horizon

        if nwp_nodes:
            self.nwp_attn = HomoNWPAttentionLayer(
                nwp_feat_dim=nwp_feat_dim,
                nwp_out_dim=nwp_out_dim,
                heads=nwp_heads,
            )
            self.nwp_i2_channels = k_nwp * nwp_feat_dim  # slice boundary in x
        # in_channels is always proj_in; the training script must compute it as
        # M + nwp_out_dim + ecmwf_channels when nwp_nodes=True
        proj_in = in_channels

        # Inductive node embedding MLP: static → embedding
        self.emb_mlp = nn.Sequential(
            nn.Linear(static_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # Separate projections for M1 / M2 so the antisymmetric graph-learning
        # formula M1@M2^T - M2@M1^T is non-zero (shared embedding would cancel).
        self.adp_W1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.adp_W2 = nn.Linear(emb_dim, emb_dim, bias=False)

        # Edge-feature bias: [dist_norm, sin_bearing, cos_bearing, alt_diff_norm] → scalar
        self.edge_fc = nn.Linear(4, 1, bias=True)

        # Input projection
        self.input_proj = nn.Linear(proj_in, hidden)

        # Dilations grow as 2^i (no cycling) so the receptive field scales with
        # n_layers without repeating small dilations.
        # RF = 1 + max_kernel_size * sum(2^i, i=0..n_layers-1) = 1 + 6*(2^n_layers - 1)
        # n_layers=5 → RF=187, n_layers=4 → RF=91 (insufficient for T_total=96)
        dilations = [2 ** i for i in range(n_layers)]
        self.layers = nn.ModuleList([
            MTGNNLayer(hidden, hidden, K_hop=K_hop, beta=beta,
                       dilation=dilations[i], dropout=dropout)
            for i in range(n_layers)
        ])

        # Output head: skip_acc (hidden) → linear → F_h predictions
        self.out_fc1 = nn.Linear(hidden, hidden)
        self.out_fc2 = nn.Linear(hidden, forecast_horizon)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Graph construction (inductive)
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_edge_features(static: Tensor) -> Tensor:
        """Compute directed pairwise edge features from static node features.

        Recovers lat/lon from sin/cos encoding (columns 0-3), uses normalised
        altitude (column 4).  Returns 4-dim feature vector per directed edge:
          [dist_norm, sin(bearing i→j), cos(bearing i→j), alt_diff_norm]

        static  : (N, 6+)
        returns : (N, N, 4)  — row=source, col=destination
        """
        lat = torch.atan2(static[:, 0], static[:, 1])   # (N,)  radians
        lon = torch.atan2(static[:, 2], static[:, 3])   # (N,)
        alt = static[:, 4]                               # (N,)  normalised

        lat_i = lat.unsqueeze(1)   # (N, 1) source
        lat_j = lat.unsqueeze(0)   # (1, N) destination
        lon_i = lon.unsqueeze(1)
        lon_j = lon.unsqueeze(0)
        dlat  = lat_j - lat_i      # (N, N)
        dlon  = lon_j - lon_i

        # Haversine distance
        a = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat_i) * torch.cos(lat_j) * torch.sin(dlon / 2) ** 2)
        dist_km   = 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())
        dist_norm = dist_km / dist_km.max().clamp(min=1e-8)

        # Bearing i → j
        y       = torch.sin(dlon) * torch.cos(lat_j)
        x       = (torch.cos(lat_i) * torch.sin(lat_j)
                   - torch.sin(lat_i) * torch.cos(lat_j) * torch.cos(dlon))
        bearing = torch.atan2(y, x)                      # (N, N)

        # Altitude difference (dst − src), clipped via normalised scale
        alt_diff = (alt.unsqueeze(0) - alt.unsqueeze(1)).clamp(-3.0, 3.0) / 3.0

        return torch.stack(
            [dist_norm, torch.sin(bearing), torch.cos(bearing), alt_diff], dim=-1
        )   # (N, N, 4)

    def _build_adjacency(self, static: Tensor) -> tuple[Tensor, Tensor]:
        """Build forward and backward row-normalised adjacencies.

        Operation order is load-bearing:
          1. A_raw  from graph learning (ReLU·tanh·antisymmetric + edge_bias)
          2. diagonal → 0   (TopK selects only off-diagonal neighbours)
          3. TopK sparsification per row   (paper Eqs. 4-6, if topk_graph is set)
          4. self-loop +I   (paper Eq. 8: Â = D̂⁻¹(A + I))
          5. forward  row-normalise
          6. backward: A_raw^T separately row-normalised (DCRNN/GWN convention —
             NOT the transpose of the already-normalised forward matrix)

        static : (N, S)
        returns: a_hat_fwd (N, N) row-stochastic,
                 a_hat_bwd (N, N) row-stochastic
        """
        N  = static.shape[0]
        E  = self.emb_mlp(static)                          # (N, d_emb)
        M1 = torch.tanh(self.alpha * self.adp_W1(E))
        M2 = torch.tanh(self.alpha * self.adp_W2(E))

        ef        = self._pairwise_edge_features(static)   # (N, N, 4)
        edge_bias = self.edge_fc(ef).squeeze(-1)           # (N, N)

        A_raw = F.relu(torch.tanh(self.alpha * (M1 @ M2.T - M2 @ M1.T) + edge_bias))

        eye   = torch.eye(N, device=A_raw.device)
        A_raw = A_raw * (1 - eye)                          # zero diagonal before TopK

        if self.topk_graph is not None and self.topk_graph < N:
            topk_vals, _ = A_raw.topk(self.topk_graph, dim=1)
            threshold    = topk_vals[:, -1:]               # (N, 1)
            A_raw        = A_raw * (A_raw >= threshold).float()

        A_raw = A_raw + eye                                # self-loop (paper Eq. 8)

        # Forward: row-normalised
        row_sum_fwd = A_raw.sum(dim=1, keepdim=True).clamp(min=1e-8)
        a_hat_fwd   = A_raw / row_sum_fwd                 # (N, N) row-stochastic

        # Backward: A_raw^T separately row-normalised
        A_raw_T     = A_raw.T.contiguous()
        row_sum_bwd = A_raw_T.sum(dim=1, keepdim=True).clamp(min=1e-8)
        a_hat_bwd   = A_raw_T / row_sum_bwd               # (N, N) row-stochastic

        return a_hat_fwd, a_hat_bwd

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        static: Tensor,
        target_mask: Tensor,
        cl_steps: int | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x           : (B, N, T_total, M+I2) — measurements + NWP, history+forecast
        static      : (B, N, 6) or (N, 6) — static node features
        target_mask : (B*N,) or (N,) bool — True = target station
        cl_steps    : curriculum step count (1..F_h); full F_h when None

        Returns
        -------
        (N_target_total, F_h_cl)  — predictions for masked stations,
                                    F_h_cl = cl_steps or forecast_horizon
        """
        if cl_steps is None:
            cl_steps = self.Fh

        B, N, T = x.shape[:3]

        # Handle both batched (B, N, S) and single-sample (N, S) static
        if static.dim() == 2:
            static_single = static           # (N, S) — same graph for all samples
        else:
            static_single = static[0]        # (N, S) — take first (identical across batch)

        # Build forward + backward adjacency once per forward pass
        a_hat_fwd, a_hat_bwd = self._build_adjacency(static_single)   # (N, N) each

        # NWP aggregation via GATv2 when nwp_nodes=True (B=1 guaranteed by HomoSampler)
        if self.nwp_nodes:
            i2_end = self.M + self.nwp_i2_channels
            meas   = x[..., :self.M]      # (B, N, T, M)
            nwp_i2 = x[..., self.M:i2_end]  # (B, N, T, k*I2)
            ecmwf  = x[..., i2_end:]      # (B, N, T, k_e*E2) — empty when k_ecmwf=0
            # Reshape ICON-D2 to (T, N*k, I2) for HomoNWPAttentionLayer
            nwp_t = nwp_i2.reshape(B, N, T, self.k_nwp, self.nwp_feat_dim)  # (B,N,T,k,I2)
            nwp_t = nwp_t.permute(0, 2, 1, 3, 4)                            # (B,T,N,k,I2)
            nwp_t = nwp_t.reshape(B, T, N * self.k_nwp, self.nwp_feat_dim)  # (B,T,N*k,I2)
            nwp_agg = self.nwp_attn.forward_sequence(nwp_t[0], N, self.k_nwp)  # (T,N,d)
            nwp_agg = nwp_agg.permute(1, 0, 2).unsqueeze(0)                 # (1,N,T,d)
            # ECMWF channels passed through directly (already IDW-concat from sampler)
            x = torch.cat([meas, nwp_agg, ecmwf], dim=-1)                   # (B,N,T,M+d+E)

        # Input projection: (B, N, T, in) → (B, N, T, hidden)
        h = self.input_proj(x)    # (B, N, T, hidden)

        # Stacked layers
        skip_acc = None
        for layer in self.layers:
            h, skip_acc = layer(h, a_hat_fwd, a_hat_bwd, skip_acc)

        # Output: map last skip position (T_total-1) → F_h predictions, clip to cl_steps
        out = self.out_fc2(
            self.dropout(F.relu(self.out_fc1(skip_acc[:, :, -1, :])))
        )                                                 # (B, N, F_h)
        out = out[:, :, :cl_steps]                        # (B, N, cl_steps)

        # Flatten batch × node, then select target stations
        out_flat = out.reshape(B * N, cl_steps)           # (B*N, cl_steps)

        # target_mask may be (N,) repeated B times or (B*N,)
        if target_mask.shape[0] == N:
            mask_flat = target_mask.repeat(B)
        else:
            mask_flat = target_mask

        # Pad to F_h (zeros for curriculum steps not yet trained)
        if cl_steps < self.Fh:
            pad = torch.zeros(
                out_flat.shape[0], self.Fh - cl_steps,
                device=out_flat.device, dtype=out_flat.dtype,
            )
            out_flat = torch.cat([out_flat, pad], dim=1)

        return out_flat[mask_flat]                         # (N_target, F_h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
