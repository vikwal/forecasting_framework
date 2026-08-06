"""
NWPAttentionLayer — aggregates NWP grid-point features into station embeddings
via bipartite message passing from ICON-D2 / ECMWF nodes to station nodes.

Two aggregation modes (``aggregation`` constructor arg)
---------------------------------------------------------
"attention" (default) : learned bipartite GATv2 attention, conditioned on the
                         station query (see forward()). This is variant A.

"idw"                  : R6(a) ablation — fixed inverse-distance weights
                         replace the *learned attention scores*, and nothing
                         else. This is variant D; see the dedicated docstring
                         below for the exact design.

Two calling modes
-----------------
forward()          : single timestep  (N_i/e, F) → (N_s, nwp_out_dim)
                     Used in sequential loops where H changes each step.
                     Both aggregation modes are wired here — this is the
                     only entry point the DCRNN encoder/decoder call
                     (dcrnn/model/encoder.py:130, dcrnn/model/decoder.py:144).

forward_sequence() : all T timesteps at once using the block-diagonal trick.
                     Treats T as a batch dimension, expands edge_index to
                     cover T independent time-slices in one GATv2 call.
                     Mathematically identical to calling forward() T times,
                     but eliminates 96× Python/kernel-dispatch overhead.
                     Used exclusively by MTGNN/WaveNet via
                     geostatistics/shared/nwp_gat.py — the DCRNN path never
                     calls this method and it stays "attention"-only;
                     "idw" is not wired into it.

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

------------------------------------------------------------------------
Design of aggregation="idw" (docs/review_round2_findings.md R6(a))
------------------------------------------------------------------------
Against DCRNN GRID's full RMSE improvement over raw ICON-D2, the review found
that the *entire* measurable graph benefit sits in BASE -> NOGRAPH, i.e. in
the learned bipartite grid attention itself (p = 1.2e-5), while the station
graph and neighbour measurements together are not statistically distinguishable
from zero (Wilcoxon p = 0.143). "Why not just interpolate?" is therefore the
central question for the NWP-attention design, and this ablation is the
answer: replace the *learned attention weights* with fixed, distance-based
weights and hold everything else fixed.

    out_i = LayerNorm( sum_j  w_ij * (W x_j + b) ),   w_ij = d_ij^-p / sum_k d_ik^-p

* The linear source projection W (+ bias b) stays learnable. This isolates
  what is being ablated to the *weighting* (attention) — a variant that also
  dropped the projection would ablate the feature transform at the same
  time and the comparison would no longer isolate one thing.
* Edge attributes beyond distance (bearing, altitude diff, …) do NOT enter
  the weight computation. Only the distance column of edge_attr is read.
  This is a deliberate simplification, not an oversight: it is the honest
  counterpart of "just interpolate" (plain IDW does not know about wind
  direction either), and it deliberately also ablates the ability to
  privilege the upwind grid point. The natural extension — folding bearing
  in as an additional fixed channel (e.g. down-weighting downwind points by
  a fixed factor derived from cos(bearing)) — is possible without touching
  the encoder/decoder call sites, but is out of scope here and left for
  whoever revisits this: see ``_idw_weights`` below for where it would go.
* The weights do not depend on h_station (the GATv2 query in attention
  mode) — geography alone determines them, so the aggregation is
  time-invariant in the same sense forward_sequence()'s zero-query trick is
  for MTGNN/WaveNet. This is NOT implemented as a block-diagonal shortcut
  here (forward_sequence stays attention-only, see above); the module docstring
  calls it out explicitly so the property is not silently optimised into a
  cached "computed once" shortcut that would look identical for a fixed
  graph but is a different design decision (e.g. it would stop tracking a
  case where edge_attr changed between timesteps, which does not happen
  today but is not an invariant this class should quietly assume).
* Output dimension, LayerNorm, and all downstream processing are identical
  to attention mode: checkpoints for A and D differ only in the NWP
  attention layer's own parameters.
* Column 0 of edge_attr is the distance. This is not re-derived here; it is
  the established convention of this codebase — see
  ``dcgru_cell.py:DCGRUCell.edge_weight_from_attr``: "column 0 is the
  normalised geodesic distance ∈ [0,1]" — produced by
  ``stgnn/utils/spatial.py:edge_features()`` as
  ``geodesic_km(src, dst) / max_dist_km`` whenever ``use_distance`` is set
  (the default). For the icond2->station / ecmwf->station edges specifically
  (``graph_builder.py:_build_nwp_to_station_edges``), ``max_dist_km`` is a
  single scalar — the maximum station-to-nearest-grid-point distance over
  the *entire* edge type — so every edge of that type shares the same
  normalisation constant. IDW weights are a ratio of distances into the same
  softmax-style per-destination sum, so that shared constant cancels exactly:
  computing w_ij from the *normalised* column gives bit-identical weights to
  computing it from physical km. No unit conversion is needed or performed.
  A distance of exactly 0 (station coincides with a grid point) is guarded
  by clamping to ``idw_eps`` before the ``**-p``, so it dominates the
  aggregate (correctly — it is a coincident point) without producing inf/NaN.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv

_IDW_EPS = 1e-6      # floor on the (normalised, unitless) distance before **-p
_DIST_COL = 0        # edge_attr column holding normalised distance (see module docstring)


def _expand_hetero_edge_index(
    ei: Tensor, N_src: int, N_dst: int, T: int
) -> Tensor:
    """Expand bipartite edge_index (2, E) to cover T independent time-slices."""
    t = torch.arange(T, device=ei.device)
    src = ei[0].unsqueeze(0) + (t * N_src).view(-1, 1)   # (T, E)
    dst = ei[1].unsqueeze(0) + (t * N_dst).view(-1, 1)   # (T, E)
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # (2, T*E)


def _idw_weights(
    edge_index: Tensor,       # (2, E) — row 0 = source (NWP), row 1 = dest (station)
    edge_attr: Tensor,        # (E, edge_dim)
    N_dst: int,
    p: float,
    eps: float = _IDW_EPS,
    dist_col: int = _DIST_COL,
) -> Tensor:
    """
    Per-destination inverse-distance weights, normalised to sum to 1 over
    every destination node's incoming edges: w_ij = d_ij^-p / sum_k d_ik^-p.

    Pure function of (edge_index, edge_attr) — no learnable state, no
    dependence on node features or any query — so it is independently
    testable and, by construction, identical whether called once per
    timestep or cached (the weights are the same every call for a fixed
    graph; only the source features x_j vary per timestep, see the module
    docstring's note on why this is not turned into a forward_sequence-style
    shortcut here).

    Only ``edge_attr[:, dist_col]`` is read — bearing/altitude/other edge
    columns are deliberately ignored, see the class docstring.

    Returns
    -------
    (E,) tensor. Destinations with zero incoming edges contribute no rows
    (nothing to normalise); scatter-adding these weights against per-edge
    messages naturally leaves such destinations at zero — see
    ``NWPAttentionLayer._idw_aggregate``.
    """
    if edge_index.numel() == 0:
        return torch.zeros(0, dtype=edge_attr.dtype, device=edge_attr.device)
    dist = edge_attr[:, dist_col].clamp(min=eps)
    w = dist.pow(-p)
    dst = edge_index[1]
    denom = torch.zeros(N_dst, dtype=w.dtype, device=w.device)
    denom.scatter_add_(0, dst, w)
    denom = denom.clamp(min=eps)          # defensive only: w > 0 always, so
    return w / denom[dst]                  # denom > 0 wherever dst has an edge


class NWPAttentionLayer(nn.Module):
    """
    Bipartite message passing: NWP nodes → station nodes.

    Parameters
    ----------
    icond2_dim  : raw ICON-D2 features per step (I2)
    ecmwf_dim   : raw ECMWF features per step (E2)
    station_dim : station hidden dim used as attention query (= hidden_dim)
    nwp_out_dim : output dimension; must be divisible by heads
    heads       : number of GATv2 attention heads (aggregation="attention" only)
    edge_dim    : edge_attr columns from HeterogeneousGraphBuilder
    dropout     : GATv2 attention dropout (aggregation="attention" only)
    aggregation : "attention" (default, variant A) | "idw" (variant D, R6(a))
    idw_p       : inverse-distance power (aggregation="idw" only), default 2.0
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
        aggregation: str = "attention",
        idw_p: float = 2.0,
    ) -> None:
        super().__init__()
        if aggregation not in ("attention", "idw"):
            raise ValueError(f"Unknown aggregation: {aggregation!r} (expected 'attention' or 'idw')")
        if aggregation == "idw" and edge_dim < 1:
            raise ValueError(
                f"aggregation='idw' needs edge_dim >= 1 (column {_DIST_COL} must hold the "
                f"distance feature), got edge_dim={edge_dim}."
            )
        assert nwp_out_dim % heads == 0
        out_per_head = nwp_out_dim // heads
        self.station_dim  = station_dim
        self.nwp_out_dim  = nwp_out_dim
        self.ecmwf_dim    = ecmwf_dim
        self.aggregation  = aggregation
        self.idw_p        = float(idw_p)

        if aggregation == "attention":
            self.gat_i2s = GATv2Conv(
                in_channels=(icond2_dim, station_dim),
                out_channels=out_per_head,
                heads=heads,
                concat=True,
                edge_dim=edge_dim,
                add_self_loops=False,
                dropout=dropout,
            )
            if ecmwf_dim > 0:
                self.gat_e2s = GATv2Conv(
                    in_channels=(ecmwf_dim, station_dim),
                    out_channels=out_per_head,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                    dropout=dropout,
                )
        else:
            # idw: only the learnable source projection W (+ bias) survives —
            # see the class docstring for why this, and only this, stays learnable.
            self.lin_i2s = nn.Linear(icond2_dim, nwp_out_dim, bias=True)
            if ecmwf_dim > 0:
                self.lin_e2s = nn.Linear(ecmwf_dim, nwp_out_dim, bias=True)

        self.norm = nn.LayerNorm(nwp_out_dim)

    # ------------------------------------------------------------------
    # idw aggregation helper
    # ------------------------------------------------------------------

    def _idw_aggregate(
        self,
        x_src: Tensor,            # (N_src, F_src)
        edge_index: Tensor,       # (2, E)
        edge_attr: Tensor,        # (E, edge_dim)
        lin: nn.Linear,
        N_dst: int,
    ) -> Tensor:                  # (N_dst, nwp_out_dim)
        out = torch.zeros(N_dst, self.nwp_out_dim, dtype=x_src.dtype, device=x_src.device)
        if edge_index.numel() == 0:
            return out
        w = _idw_weights(edge_index, edge_attr, N_dst, self.idw_p)   # (E,)
        src, dst = edge_index[0], edge_index[1]
        msg = lin(x_src)[src] * w.unsqueeze(-1)                       # (E, nwp_out_dim)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.nwp_out_dim), msg)
        return out

    # ------------------------------------------------------------------
    # Single-step forward (kept for flexibility / debugging)
    # ------------------------------------------------------------------

    def forward(
        self,
        icond2_t: Tensor,        # (N_i, I2)
        ecmwf_t: Tensor,         # (N_e, E2)
        h_station: Tensor,       # (N_s, station_dim) — attention query; unused in idw mode
        i2s_edge_index: Tensor,
        i2s_edge_attr: Tensor,
        e2s_edge_index: Tensor,
        e2s_edge_attr: Tensor,
    ) -> Tensor:                 # (N_s, nwp_out_dim)
        N_s = h_station.size(0)
        if self.aggregation == "idw":
            msg_i = self._idw_aggregate(icond2_t, i2s_edge_index, i2s_edge_attr, self.lin_i2s, N_s)
            if self.ecmwf_dim == 0:
                return self.norm(msg_i)
            msg_e = self._idw_aggregate(ecmwf_t, e2s_edge_index, e2s_edge_attr, self.lin_e2s, N_s)
            return self.norm(msg_i + msg_e)

        msg_i = self.gat_i2s((icond2_t, h_station), i2s_edge_index, i2s_edge_attr)
        if self.ecmwf_dim == 0:
            return self.norm(msg_i)
        msg_e = self.gat_e2s((ecmwf_t,  h_station), e2s_edge_index, e2s_edge_attr)
        return self.norm(msg_i + msg_e)

    # ------------------------------------------------------------------
    # Vectorised forward over all T timesteps (block-diagonal trick)
    #
    # Attention-only — the DCRNN path (the only caller of aggregation="idw")
    # never calls this method; see the module docstring.
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
        if self.aggregation != "attention":
            raise NotImplementedError(
                "forward_sequence() is attention-only (used exclusively by the "
                "MTGNN/WaveNet path via geostatistics/shared/nwp_gat.py). The DCRNN "
                "path — the only caller of aggregation='idw' — always uses forward()."
            )
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
        if self.ecmwf_dim == 0:
            return self.norm(msg_i).reshape(T, N_s, self.nwp_out_dim)

        msg_e = self.gat_e2s((e2_flat, h_q), e2s_ei_exp, e2s_ea_exp)  # (T*N_s, d)
        nwp_flat = self.norm(msg_i + msg_e)                            # (T*N_s, d)
        return nwp_flat.reshape(T, N_s, self.nwp_out_dim)             # (T, N_s, d)
