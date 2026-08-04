"""
geostatistics/wavenet/model.py — Inductive Graph WaveNet for spatial forecasting.

Based on "Graph WaveNet for Deep Spatial-Temporal Graph Modeling" (Wu et al.,
2019, arXiv:1906.00121), adapted for the inductive setting.

Key differences from the original paper
-----------------------------------------
  • **Inductive self-adaptive adjacency**: node-ID embeddings are replaced by
    MLP(static_features) → E.  Adaptive adjacency:
      A_adp = softmax(ReLU(E1 · E2ᵀ))
    where E1 = tanh(α · W1 · E),  E2 = tanh(α · W2 · E).
    This allows the model to generalise to unseen stations at inference time.
    The original E1, E2 are randomly initialised per-node parameters (paper
    Sec. 3.2), which cannot serve stations unseen during training.  The same
    authors sanction the substitution one year later for the equivalent module
    in MTGNN (Wu et al., 2020, Sec. 4.2: "we can also set E1 = E2 = Z, where Z
    is a static node feature matrix").
  • **Predefined adjacency** (``predefined_adj``): off by default for backward
    compatibility with existing checkpoints.  When on, a thresholded Gaussian
    distance kernel is added as a second diffusion branch, giving the paper's
    full Eq. 6 instead of the Eq. 7 adaptive-only fallback — see
    ``_predefined_adjacency``.
  • **Edge-feature bias**: ``edge_fc`` adds a geometry-derived scalar bias
    inside the adaptive adjacency.  Not part of the original paper; it is one
    way of injecting the physical graph when only Eq. 7 is available.
  • **Input / output**: same convention as MTGNNModel — (B, N, T_total, M+I2)
    in, (N_target, F_h) out.

Architecture
------------
  Input norm + linear → stacked GWNBlocks → output MLP (skip connections)

  GWNBlock:
    Residual TCN branch: gated causal dilated Conv1d (tanh ⊙ sigmoid)
    Graph branch:        K-hop power-series diffusion, see below
    Output: relu(gated_tcn + gcn)  +  skip connection

  Diffusion convolution (MultiHopDiffusion), per branch:
    Z = Σ_{k=0}^K  A^k · X · W_k   (K=2 paper default)
    predefined_adj=False → A = A_adp only                    (paper Eq. 7)
    predefined_adj=True  → sum over A_adp and P              (paper Eq. 6)
    (No forward/backward split: A_adp is already asymmetric, and the geodesic
     station graph is symmetric, so P_f and P_b would coincide.)

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
from geostatistics.homo_sampler import NWP_EDGE_DIM


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
        predefined_adj: bool = False,
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

        # Graph convolution over the self-adaptive adjacency (paper Eq. 7 term)
        self.gcn = MultiHopDiffusion(c_hidden, c_hidden, K=K)
        # Second branch over the predefined distance adjacency, giving the full
        # paper Eq. 6 sum. Only created when enabled, so a state_dict from a
        # predefined_adj=False model stays byte-compatible.
        self.gcn_pre = MultiHopDiffusion(c_hidden, c_hidden, K=K) if predefined_adj else None

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
        a_pre: Tensor | None = None,
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
        h_g = self.gcn(h_s, a_adp)
        if self.gcn_pre is not None and a_pre is not None:
            h_g = h_g + self.gcn_pre(h_s, a_pre)                 # paper Eq. 6 sum
        h   = h_g.reshape(N, B, T, C_h).permute(1, 0, 2, 3)      # (B, N, T, C_h)

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
    predefined_adj  : add the predefined distance adjacency as a second
                      diffusion branch (paper Eq. 6). Default False keeps the
                      Eq. 7 adaptive-only behaviour and the existing state_dict
                      layout, so old checkpoints stay loadable.
    adj_sigma       : sigma of the Gaussian distance kernel, on normalised
                      distances in [0, 1] (matches DCRNN's edge_weight_sigma)
    adj_threshold   : kernel weights below this are set to zero before row
                      normalisation — the "thresholded" part of the kernel
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
        k_ecmwf: int = 0,
        ecmwf_feat_dim: int = 0,
        ecmwf_out_dim: int = 32,
        M: int = 0,
        predefined_adj: bool = False,
        adj_sigma: float = 0.2,
        adj_threshold: float = 0.1,
        topo_dim: int = 0,
        broadcast_topo: bool = False,
    ) -> None:
        super().__init__()
        self.H         = history_length
        self.Fh        = forecast_horizon
        self.alpha     = graph_alpha
        self.nwp_nodes = nwp_nodes
        self.M         = M
        self.k_nwp     = k_nwp
        self.nwp_feat_dim = nwp_feat_dim
        self.k_ecmwf      = k_ecmwf
        self.ecmwf_feat_dim = ecmwf_feat_dim
        self.ecmwf_attn   = None
        self.predefined_adj = predefined_adj
        self.adj_sigma      = adj_sigma
        self.adj_threshold  = adj_threshold
        self.topo_dim       = topo_dim
        self.broadcast_topo = broadcast_topo and topo_dim > 0

        if nwp_nodes:
            self.nwp_attn = HomoNWPAttentionLayer(
                nwp_feat_dim=nwp_feat_dim,
                nwp_out_dim=nwp_out_dim,
                heads=nwp_heads,
                edge_dim=NWP_EDGE_DIM,
            )
            self.nwp_i2_channels = k_nwp * nwp_feat_dim  # slice boundary in x
            # ECMWF als zweiter Knotentyp, eigene Attention und eigene Kanten —
            # sonst haette nwp_nodes hier eine andere Bedeutung als beim DCRNN.
            if k_ecmwf > 0 and ecmwf_feat_dim > 0:
                self.ecmwf_attn = HomoNWPAttentionLayer(
                    nwp_feat_dim=ecmwf_feat_dim,
                    nwp_out_dim=ecmwf_out_dim,
                    heads=nwp_heads,
                    edge_dim=NWP_EDGE_DIM,
                )
        # Broadcast widening is added here rather than by the caller so the two
        # can never disagree.
        proj_in = in_channels + (topo_dim if self.broadcast_topo else 0)

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
                predefined_adj=predefined_adj,
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
    def _pairwise_geo(static: Tensor) -> tuple[Tensor, Tensor]:
        """Pairwise normalised geodesic distance and bearing from static features.

        Recovers lat/lon from the sin/cos encoding in columns 0-3 (see
        homo_sampler._build_static).  Both returns are (N, N) with row=source,
        col=destination.  Shared by the adaptive-adjacency edge bias and the
        predefined distance adjacency so the geometry is computed once and the
        two stay consistent.
        """
        lat = torch.atan2(static[:, 0], static[:, 1])
        lon = torch.atan2(static[:, 2], static[:, 3])

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

        return dist_norm, bearing

    @classmethod
    def _pairwise_edge_features(cls, static: Tensor) -> Tensor:
        """Directed pairwise edge features from static node features.

        Adds normalised altitude difference (column 4) to the shared geometry.
        Returns 4-dim feature vector per directed edge:
          [dist_norm, sin(bearing i→j), cos(bearing i→j), alt_diff_norm]

        static  : (N, 6+)
        returns : (N, N, 4)  — row=source, col=destination
        """
        dist_norm, bearing = cls._pairwise_geo(static)
        alt = static[:, 4]
        alt_diff = (alt.unsqueeze(0) - alt.unsqueeze(1)).clamp(-3.0, 3.0) / 3.0

        return torch.stack(
            [dist_norm, torch.sin(bearing), torch.cos(bearing), alt_diff], dim=-1
        )

    @classmethod
    def _predefined_adjacency(
        cls,
        static: Tensor,
        sigma: float,
        threshold: float,
        target_mask: Tensor | None = None,
    ) -> Tensor:
        """Row-normalised predefined adjacency from a thresholded Gaussian kernel.

        Paper Eq. 6 combines predefined spatial dependencies with the
        self-adaptive matrix; Eq. 7 (adaptive only) is explicitly the fallback
        "when the graph structure is unavailable".  Station coordinates *are*
        available here, so the predefined term is built as in the paper's
        experimental setup (Sec. 4): W_ij = exp(-(d_ij/sigma)^2), zeroed below
        ``threshold``, then row-normalised into a transition matrix.

        The forward/backward split of Eq. 6 is omitted on purpose: it exists for
        directed road networks, whereas a geodesic station graph is symmetric,
        so P_f = A/rowsum(A) and P_b = A^T/rowsum(A^T) are identical.

        No self-loops are added — MultiHopDiffusion's k=0 term already carries
        the ego features (P^0 = I), and d_ii = 0 puts weight 1 on the kernel
        diagonal anyway, so no row can end up all-zero.
        """
        dist_norm, _ = cls._pairwise_geo(static)
        w = torch.exp(-((dist_norm / sigma) ** 2))
        w = w * (w >= threshold).to(w.dtype)
        if target_mask is not None and bool(target_mask.any()):
            # Same reasoning as _mask_target_pairs, but this matrix is weights
            # rather than logits: zero before row-normalising, keep the diagonal.
            tm = target_mask.to(w.device).view(-1)
            tt = tm.view(-1, 1) & tm.view(1, -1)
            tt = tt & ~torch.eye(w.shape[0], dtype=torch.bool, device=w.device)
            w = w * (~tt).to(w.dtype)
        return w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def _mask_target_pairs(a: Tensor, target_mask: Tensor | None) -> Tensor:
        """Set target → target entries to -inf, keeping every self-loop.

        The adaptive adjacency is dense over the batch's nodes, so a validation
        batch — 51 targets among ~141 nodes — would let each target draw a large
        share of its attention from other targets, while a training batch with
        1-10 targets barely can. Those co-targets are absent when a single new
        site is served at inference. Masking before the softmax redistributes the
        weight onto real neighbours instead of merely zeroing it afterwards.

        The diagonal is exempt: unlike MTGNN there is no separate ``+I`` step, so
        a masked diagonal would strip the target's self-loop.
        """
        if target_mask is None or not bool(target_mask.any()):
            return a
        tm = target_mask.to(a.device).view(-1)
        tt = tm.view(-1, 1) & tm.view(1, -1)
        tt = tt & ~torch.eye(a.shape[0], dtype=torch.bool, device=a.device)
        return a.masked_fill(tt, float("-inf"))

    def _build_adjacency(
        self, static: Tensor, target_mask: Tensor | None = None,
    ) -> Tensor:
        """Inductive adaptive adjacency with pairwise edge features.

        A_adp = softmax(ReLU(E1 · E2ᵀ + edge_bias))  where
        E1 = tanh(α · W1 · emb),  E2 = tanh(α · W2 · emb)

        static      : (N, S)
        target_mask : (N,) bool — True = target station. None disables masking.
        returns: (N, N) row-normalised soft adjacency
        """
        emb = self.emb_mlp(static)                             # (N, d_emb)
        E1  = torch.tanh(self.alpha * self.adp_W1(emb))
        E2  = torch.tanh(self.alpha * self.adp_W2(emb))

        ef        = self._pairwise_edge_features(static)       # (N, N, 4)
        edge_bias = self.edge_fc(ef).squeeze(-1)               # (N, N)

        A = F.relu(E1 @ E2.T + edge_bias)
        A = self._mask_target_pairs(A, target_mask)
        return F.softmax(A, dim=1)                             # (N, N)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        static: Tensor,
        target_mask: Tensor,
        nwp_edge_attr: Tensor | None = None,
        ecmwf_edge_attr: Tensor | None = None,
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

        # Adaptive adjacency (built once per forward). The node set is identical
        # across the batch, so the first N entries of target_mask describe it
        # whichever layout the caller used.
        tm_single = target_mask if target_mask.shape[0] == N else target_mask[:N]
        a_adp = self._build_adjacency(static_single, tm_single)   # (N, N)
        a_pre = (
            self._predefined_adjacency(
                static_single, self.adj_sigma, self.adj_threshold, tm_single)
            if self.predefined_adj else None
        )                                              # (N, N) or None

        # NWP aggregation via GATv2 when nwp_nodes=True (B=1 guaranteed by HomoSampler)
        if self.nwp_nodes:
            if nwp_edge_attr is None:
                raise ValueError(
                    "nwp_nodes=True braucht nwp_edge_attr (Distanz/Azimut/Hoehe je "
                    "NWP->Station-Kante). Ohne diese ist die Zero-Query-Attention "
                    "permutationsaequivariant ueber die k Gitterpunkte."
                )
            i2_end = self.M + self.nwp_i2_channels
            meas   = x[..., :self.M]         # (B, N, T, M)
            nwp_i2 = x[..., self.M:i2_end]  # (B, N, T, k*I2)
            ecmwf  = x[..., i2_end:]         # (B, N, T, k_e*E2) — empty when k_ecmwf=0
            # Reshape ICON-D2 to (T, N*k, I2) for HomoNWPAttentionLayer
            nwp_t = nwp_i2.reshape(B, N, T, self.k_nwp, self.nwp_feat_dim)  # (B,N,T,k,I2)
            nwp_t = nwp_t.permute(0, 2, 1, 3, 4)                            # (B,T,N,k,I2)
            nwp_t = nwp_t.reshape(B, T, N * self.k_nwp, self.nwp_feat_dim)  # (B,T,N*k,I2)
            nwp_agg = self.nwp_attn.forward_sequence(
                nwp_t[0], N, self.k_nwp, edge_attr=nwp_edge_attr)           # (T,N,d)
            nwp_agg = nwp_agg.permute(1, 0, 2).unsqueeze(0)                 # (1,N,T,d)
            if self.ecmwf_attn is not None:
                if ecmwf_edge_attr is None:
                    raise ValueError("k_ecmwf>0 braucht ecmwf_edge_attr")
                e_t = ecmwf.reshape(B, N, T, self.k_ecmwf, self.ecmwf_feat_dim)
                e_t = e_t.permute(0, 2, 1, 3, 4).reshape(
                    B, T, N * self.k_ecmwf, self.ecmwf_feat_dim)
                ecmwf = self.ecmwf_attn.forward_sequence(
                    e_t[0], N, self.k_ecmwf, edge_attr=ecmwf_edge_attr)
                ecmwf = ecmwf.permute(1, 0, 2).unsqueeze(0)                 # (1,N,T,d_e)
            x = torch.cat([meas, nwp_agg, ecmwf], dim=-1)                   # (B,N,T,M+d+d_e)

        # Topographic node features as extra input channels, constant along T.
        # Appended after the NWP re-assembly above so its slice boundaries stay
        # valid. Paper Sec. 3.1 defines the input as X ∈ R^{N×D×S} with arbitrary
        # D, and Sec. 3.2 notes the graph convolution "supports multi-dimensional
        # inputs", so extra channels use the interface as specified.
        if self.broadcast_topo:
            topo = static_single[:, 6:6 + self.topo_dim]                     # (N, topo_dim)
            topo = topo.view(1, N, 1, self.topo_dim).expand(B, N, T, self.topo_dim)
            x = torch.cat([x, topo], dim=-1)

        # Input projection
        h = self.input_proj(x)                         # (B, N, T_total, hidden)

        # Stacked blocks, accumulate skip
        skip_acc = None
        for block in self.blocks:
            h, skip_acc = block(h, a_adp, skip_acc, a_pre)

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
