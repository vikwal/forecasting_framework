"""
NWPAttentionLayer — aggregates NWP grid-point features into station embeddings
via bipartite message passing from ICON-D2 / ECMWF nodes to station nodes.

Three aggregation modes (``aggregation`` constructor arg)
---------------------------------------------------------
"attention" (default) : learned bipartite GATv2 attention, conditioned on the
                         station query (see forward()). This is variant A.

"idw"                  : R6(a) ablation — fixed inverse-distance weights
                         replace the *learned attention scores*, and nothing
                         else. This is variant D; see the dedicated docstring
                         below for the exact design.

"idw_alt"              : D plus a height correction (great-circle distance
                          with a height offset, matching the construction in
                          ertz2025postprocessing). This is variant D'; see
                          "Design of aggregation='idw_alt'" below.

Two calling modes
-----------------
forward()          : single timestep  (N_i/e, F) → (N_s, nwp_out_dim)
                     Used in sequential loops where H changes each step.
                     All three aggregation modes are wired here — this is the
                     only entry point the DCRNN encoder/decoder call
                     (dcrnn/model/encoder.py:130, dcrnn/model/decoder.py:144).

forward_sequence() : all T timesteps at once using the block-diagonal trick.
                     Treats T as a batch dimension, expands edge_index to
                     cover T independent time-slices in one GATv2 call.
                     Mathematically identical to calling forward() T times,
                     but eliminates 96× Python/kernel-dispatch overhead.
                     Currently has NO caller anywhere in the tree: the
                     DCRNN encoder/decoder always use forward(), and
                     MTGNN/WaveNet do not use this class at all — they use the
                     separate HomoNWPAttentionLayer in
                     geostatistics/shared/nwp_gat.py, which carries its own
                     copy of the block-diagonal trick. It stays
                     "attention"-only and raises NotImplementedError under
                     "idw"/"idw_alt"; wiring either in here would be dead code
                     today.

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
  the weight computation of plain "idw". Only the distance column of
  edge_attr is read. This is a deliberate simplification, not an oversight:
  it is the honest counterpart of "just interpolate" (plain IDW does not know
  about wind direction either), and it deliberately also ablates the ability
  to privilege the upwind grid point. Bearing as a fixed extra channel is
  still not implemented (out of scope); height IS now covered by "idw_alt"
  below, so the remaining gap is direction only.
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
  to attention mode: checkpoints for A, D and D' differ only in the NWP
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
  normalisation constant.
* A distance of exactly 0 (station coincides with a grid point) is guarded by
  the min-distance renormalisation in ``_idw_weights`` below — see its
  docstring for the mechanism and why it removes the old float32-overflow
  cap on ``idw_p``.

------------------------------------------------------------------------
Min-distance renormalisation of the IDW weights (both "idw" and "idw_alt")
------------------------------------------------------------------------
The weights are computed as

    w_ij = (d_ij / d_i,min)^-p / sum_k (d_ik / d_i,min)^-p        (d_i,min = min_k d_ik)

instead of the unnormalised ``d_ij^-p / sum_k d_ik^-p``. The two are
algebraically identical — d_i,min cancels between numerator and
denominator — but the renormalised form is unit-independent (any constant
rescaling of every distance into one destination's edge group, e.g. the
graph-wide normalisation constant, cancels the same way, so "idw" can feed it
the *normalised* [0,1] column directly) and numerically strictly better
behaved: every ratio d_ij/d_i,min is >= 1 by construction (exactly 1 for the
nearest point), so ``ratio ** -p`` lies in (0, 1] no matter how large p is —
there is no longer any edge whose weight can overflow to inf in float32, and
the nearest-point edge always contributes exactly 1.0 to the per-destination
sum, so that sum can never be 0 either. This removes the float32-overflow cap
(``_IDW_P_MAX`` in the previous revision of this module, ~6.42 for
eps=1e-6): with the pre-renormalisation formula, a distance sitting on the
epsilon floor raised to a large enough power did overflow to inf and turn the
per-station normalisation into inf/inf = NaN; that failure mode does not
exist here, so p is no longer upper-bounded (p > 0 is still required — p <= 0
does not down-weight distance at all or inverts it, see ``validate_idw_p``).
Measured against the real fold graph (153 stations, k=4, p=2, 918 icond2 /
612 ecmwf edges): the renormalised weights deviate from the
pre-renormalisation weights by at most 1.79e-7 (icond2) / 1.19e-7 (ecmwf) —
float32 rounding, not a behavioural change (verify_idw.py §9). The resulting
DCRNN forward pass shifts correspondingly little: on a full model forward
(349505 params, real fixture batch) the pre- vs post-renormalisation
prediction differs by at most 4.8e-7 (mean 9.7e-8) against a prediction
std of 0.30 — i.e. ~1.6e-6 relative, indistinguishable from float32 noise.
Every parameter tensor is unaffected (renormalisation only changes the
forward computation, not initialisation), so this shift is the full story:
it is not building up across layers into something larger.

------------------------------------------------------------------------
Design of aggregation="idw_alt" (variant D', height-corrected IDW)
------------------------------------------------------------------------
Plain "idw" answers "why not just interpolate horizontally" — but
height-aware interpolation is the actual standard in operational
meteorological post-processing, so a referee could reasonably ask why the
ablation ladder skips it. "idw_alt" adds exactly that rung, using the SAME
construction as the closest prior work, ertz2025postprocessing (ASCMO 2025,
bias-correction at German SYNOP stations): great-circle distance with a
height offset,

    d3d_ij = sqrt( d_ij^2 + (alpha * Delta_alt_ij)^2 )     (d_ij, Delta_alt_ij in km)
    w_ij   = d3d_ij^-p / sum_k d3d_ik^-p                    (min-renormalised, see above)

d3d_ij then replaces d_ij everywhere "idw" would have used the raw distance;
everything else — learnable W, LayerNorm placement, no bearing in the
message, time-invariance, symmetry between ICON-D2 and ECMWF — is identical
to plain "idw".

Recovering physical kilometres for BOTH terms — the trap
----------------------------------------------------------
Column 0 (distance) is normalised by a *graph-dependent* scalar
(max_dist_km, different per config/fold and not otherwise recoverable at
forward() time — see graph_builder.py:_build_nwp_to_station_edges). Column
``altitude_diff_col()`` (usually 3, but see DCRNNConfig.altitude_diff_col()
for why it is derived rather than hardcoded) is normalised by a *fixed
literal*, ``_ALT_COL_NORM_M = 3000.0`` metres, hardcoded in
stgnn/utils/spatial.py:edge_features() — NOT graph-dependent.

Plain "idw" never needed to know either constant: it only ever computes a
RATIO of distances within one destination's edge group, and both distances
carry the *same* graph-wide normalisation constant, so it cancels exactly
regardless of what the constant's value actually is (see "idw" docstring
above; the min-distance renormalisation makes this doubly explicit, since
"idw" now literally never leaves normalised-[0,1] units).

"idw_alt" breaks that cancellation: sqrt(d^2 + (alpha*dalt)^2) is NOT
homogeneous under two *independently, differently* normalised inputs — you
cannot recover the correct combined quantity by combining two numbers on two
different [0,1]-ish scales and calling the result "distance". Both terms
MUST be converted to a shared physical unit (km) before the sqrt. The
altitude term converts trivially (fixed 3000 m constant, always available).
The distance term does not: max_dist_km has to be recovered from the graph
that was actually built for this run. HeterogeneousGraphBuilder.build()
persists it as ``data[edge_type].max_dist_km`` for exactly this reason (see
graph_builder.py); DCRNNConfig.attach_nwp_geometry(base_graph) reads it onto
icond2_max_dist_km / ecmwf_max_dist_km, and every driver script
(train_dcrnn.py, get_test_results_dcrnn.py, hpo_dcrnn.py) must call it before
constructing DCRNN(config) — DCRNN.__init__ hard-fails under "idw_alt" if
this was skipped, rather than silently running with max_dist_km=0 (which
would silently degenerate d3d to just the height term).

Back-calculation against the real graph (data_cache/gnns/*/derived.pkl, 153
stations, 1071 icond2 / 553 ecmwf grid points): reconstructing the altitude
column from the real station/grid altitudes matches the stored edge_attr to
within 8.4e-5 m (float32 rounding) for both edge types, and 0 of 918 icond2 /
612 ecmwf edges hit the +-3000 m clip (max observed |Delta_alt| = 1605 m /
2215 m). The implied distance normaliser (edge_attr[:,0] against
independently recomputed geodesic km) is exactly constant across each edge
type (3.4945 km for icond2, 29.2554 km for ecmwf), confirming it is the
single graph-wide scalar the code assumes.

Choice of alpha_alt (default 10.0)
-------------------------------------
alpha_alt trades off the vertical against the horizontal term; too large and
the correction stops being a *correction* and dominates the geometry outright
(everything ends up ordered by height alone, defeating the point of keeping
a distance-based baseline at all). Calibrated against the measured
distributions on the real fold graph (same source as above), NOT a rule of
thumb:

  icond2 (k=4, 918 edges): Delta_alt median 14.0 m, q75 32.8 m, max 1605 m;
    horizontal distance median 2145 m, q75 2690 m, max 3495 m.
    Delta_alt / distance ratio: median 0.79%, q75 1.74%, q95 6.62%, max 76%.
  ecmwf (k=4, 612 edges): Delta_alt median 35.9 m, q75 104.0 m, max 2215 m;
    horizontal distance median 17519 m, q75 21540 m, max 29255 m.
    Delta_alt / distance ratio: median 0.24%, q75 0.67%, q95 2.17%, max 16%.

At alpha_alt=10: the vertical term is ~6.5% of the horizontal median (icond2)
/ ~2.0% (ecmwf) — clearly subordinate in the typical case — while the
vertical term exceeds the horizontal one for 2.5% of icond2 edges and 0.2%
of ecmwf edges: exactly the steep-terrain minority (e.g. Alpine stations)
where a height correction is supposed to matter, without the correction
taking over the flat-terrain majority. At alpha_alt=50 that fraction grows to
22%/6% — no longer a correction, closer to a height-only ranking — which is
why 10, not a much larger value, was chosen despite the median criterion
alone tolerating far more headroom (median would not be dominated even at
alpha_alt=100).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter

_IDW_EPS = 1e-6      # floor on distance (any unit) before the ratio/power law
_DIST_COL = 0         # edge_attr column holding normalised distance (see module docstring)
_ALT_COL_NORM_M = 3000.0  # MUST match stgnn/utils/spatial.py:edge_features()'s altitude_diff
                           # normalisation literal ("rough normalisation: +-3000 m range -> +-1").
                           # Not derived/imported from there because edge_features() has no public
                           # constant to import; if that literal ever changes, this one has to
                           # change with it — deliberately loud about that coupling here.


def validate_idw_p(p: float) -> float:
    """Range-check the inverse-distance power. Shared by DCRNNConfig.from_yaml,
    geostatistics/ablations/guard.py and NWPAttentionLayer.__init__ so the same
    bound holds whichever entry point a run comes through.

    Only p > 0 is enforced. The float32-overflow upper bound that used to live
    here (_IDW_P_MAX, ~6.42) is gone: it was a consequence of raising a
    distance that could be arbitrarily close to 0 (on the graph-wide [0,1]
    normalised scale) to a large negative power. The min-distance
    renormalisation in _idw_weights() replaced that with a ratio that is
    always >= 1, so `ratio ** -p` is bounded in (0, 1] for every finite p > 0
    — no overflow is possible any more, at any p. See the module docstring's
    "Min-distance renormalisation" section for the full argument."""
    p = float(p)
    if not math.isfinite(p) or p <= 0.0:
        raise ValueError(
            f"idw_p must be a finite number > 0, got {p!r}. p <= 0 does not "
            f"down-weight distance at all (p == 0 is a plain unweighted mean, "
            f"p < 0 *inverts* the weighting and favours the most distant grid "
            f"point) — neither is the IDW baseline this ablation is defined as."
        )
    return p


def validate_alpha_alt(alpha: float) -> float:
    """Range-check idw_alt's vertical/horizontal trade-off. Shared the same
    way validate_idw_p is."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(
            f"alpha_alt must be a finite number >= 0, got {alpha!r}. "
            f"alpha_alt=0 degenerates idw_alt to plain idw (height ignored); "
            f"negative values have no defined meaning under a Euclidean d3d."
        )
    return alpha


def _expand_hetero_edge_index(
    ei: Tensor, N_src: int, N_dst: int, T: int
) -> Tensor:
    """Expand bipartite edge_index (2, E) to cover T independent time-slices."""
    t = torch.arange(T, device=ei.device)
    src = ei[0].unsqueeze(0) + (t * N_src).view(-1, 1)   # (T, E)
    dst = ei[1].unsqueeze(0) + (t * N_dst).view(-1, 1)   # (T, E)
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # (2, T*E)


def _idw_alt_distance_km(
    edge_attr: Tensor,
    alt_col: int,
    max_dist_km: float,
    alpha: float,
    dist_col: int = _DIST_COL,
) -> Tensor:
    """
    d3d_ij = sqrt(d_ij^2 + (alpha * |Delta_alt_ij|)^2), both terms converted
    to physical kilometres first — see the module docstring's "Recovering
    physical kilometres" section for why this conversion is unavoidable here
    (unlike plain "idw", the two normalisation constants do not cancel).

    edge_attr[:, dist_col] is normalised distance in [0, 1); multiplying by
    max_dist_km (recovered via DCRNNConfig.attach_nwp_geometry, see there)
    recovers physical km.

    edge_attr[:, alt_col] is clip((dst_alt - src_alt) / _ALT_COL_NORM_M, -1, 1);
    multiplying back by _ALT_COL_NORM_M and converting m -> km recovers the
    (signed) physical height difference. Only its magnitude matters here since
    it is squared.
    """
    d_km = edge_attr[:, dist_col].clamp(min=0.0) * max_dist_km
    dalt_km = (edge_attr[:, alt_col] * _ALT_COL_NORM_M / 1000.0).abs()
    return torch.sqrt(d_km.pow(2) + (alpha * dalt_km).pow(2))


def _idw_weights(
    dist: Tensor,              # (E,) nonnegative "distance" in any consistent unit/scale
    dst: Tensor,                # (E,) destination index per edge
    N_dst: int,
    p: float,
    eps: float = _IDW_EPS,
) -> Tensor:
    """
    Per-destination inverse-distance weights, renormalised to the
    per-destination MINIMUM distance before the power law:

        w_ij = (d_ij / d_i,min)^-p / sum_k (d_ik / d_i,min)^-p

    See the module docstring's "Min-distance renormalisation" section for why
    this is algebraically identical to the un-renormalised form but strictly
    better numerically (bounded in (0, 1], no float32 overflow at any p > 0,
    unit-independent — ``dist`` may be the raw normalised distance column
    (plain "idw") or a physical d3d in km (``idw_alt``), the result is the
    same either way up to float32 rounding).

    A distance of exactly 0 (a coincident node) is floored to ``eps`` before
    anything else, so the coincident edge's ratio is exactly 1 and it
    receives the full weight (correctly — it dominates the aggregate).

    Returns
    -------
    (E,) tensor. Destinations with zero incoming edges contribute no rows;
    scatter-adding these weights against per-edge messages naturally leaves
    such destinations at zero — see ``NWPAttentionLayer._idw_aggregate``.
    """
    if dist.numel() == 0:
        return torch.zeros(0, dtype=dist.dtype, device=dist.device)
    dist = dist.clamp(min=eps)
    d_min = scatter(dist, dst, dim=0, dim_size=N_dst, reduce="min")
    ratio = dist / d_min[dst]                # >= 1 for every edge, == 1 for the nearest
    w_unnorm = ratio.pow(-p)                  # in (0, 1] — no overflow possible
    denom = torch.zeros(N_dst, dtype=w_unnorm.dtype, device=w_unnorm.device)
    denom.scatter_add_(0, dst, w_unnorm)
    # denom >= 1 always: the nearest-point edge (ratio == 1) contributes
    # exactly 1.0 to whichever destination's sum it belongs to, and denom is
    # only ever indexed (via denom[dst] below) at destinations that have at
    # least one edge — so no epsilon clamp is needed here, unlike the
    # pre-renormalisation version of this function.
    return w_unnorm / denom[dst]


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
    aggregation : "attention" (default, variant A) | "idw" (variant D) | "idw_alt" (variant D')
    idw_p       : inverse-distance power (aggregation in {"idw","idw_alt"} only), default 2.0
    alpha_alt   : vertical/horizontal trade-off (aggregation="idw_alt" only), default 10.0
    alt_col     : edge_attr column holding altitude_diff (aggregation="idw_alt" only);
                  see DCRNNConfig.altitude_diff_col()
    icond2_max_dist_km : physical-km normaliser for icond2 distance column (aggregation="idw_alt" only)
    ecmwf_max_dist_km  : physical-km normaliser for ecmwf distance column (aggregation="idw_alt" only)
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
        alpha_alt: float = 10.0,
        alt_col: int | None = None,
        icond2_max_dist_km: float = 0.0,
        ecmwf_max_dist_km: float = 0.0,
    ) -> None:
        super().__init__()
        if aggregation not in ("attention", "idw", "idw_alt"):
            raise ValueError(
                f"Unknown aggregation: {aggregation!r} (expected 'attention', 'idw' or 'idw_alt')"
            )
        if aggregation in ("idw", "idw_alt"):
            if edge_dim < 1:
                raise ValueError(
                    f"aggregation={aggregation!r} needs edge_dim >= 1 (column {_DIST_COL} must "
                    f"hold the distance feature), got edge_dim={edge_dim}."
                )
            idw_p = validate_idw_p(idw_p)
        if aggregation == "idw_alt":
            if alt_col is None or alt_col < 0 or alt_col >= edge_dim:
                raise ValueError(
                    f"aggregation='idw_alt' needs a valid alt_col in [0, edge_dim) "
                    f"(the altitude_diff column), got alt_col={alt_col!r} for edge_dim={edge_dim}. "
                    f"Use DCRNNConfig.altitude_diff_col() to derive it."
                )
            if icond2_max_dist_km <= 0.0:
                raise ValueError(
                    "aggregation='idw_alt' needs icond2_max_dist_km > 0 (physical-km normaliser "
                    "for the icond2 distance column, recovered via "
                    "DCRNNConfig.attach_nwp_geometry(base_graph)), got "
                    f"{icond2_max_dist_km!r}."
                )
            if ecmwf_dim > 0 and ecmwf_max_dist_km <= 0.0:
                raise ValueError(
                    "aggregation='idw_alt' with ecmwf_dim > 0 needs ecmwf_max_dist_km > 0 "
                    "(physical-km normaliser for the ecmwf distance column, recovered via "
                    f"DCRNNConfig.attach_nwp_geometry(base_graph)), got {ecmwf_max_dist_km!r}."
                )
            alpha_alt = validate_alpha_alt(alpha_alt)
        assert nwp_out_dim % heads == 0
        out_per_head = nwp_out_dim // heads
        self.station_dim  = station_dim
        self.nwp_out_dim  = nwp_out_dim
        self.ecmwf_dim    = ecmwf_dim
        self.aggregation  = aggregation
        self.idw_p        = float(idw_p)
        self.alpha_alt    = float(alpha_alt)
        self.alt_col      = alt_col
        self.icond2_max_dist_km = float(icond2_max_dist_km)
        self.ecmwf_max_dist_km  = float(ecmwf_max_dist_km)

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
            # idw / idw_alt: only the learnable source projection W (+ bias)
            # survives — see the class docstring for why this, and only this,
            # stays learnable.
            self.lin_i2s = nn.Linear(icond2_dim, nwp_out_dim, bias=True)
            if ecmwf_dim > 0:
                self.lin_e2s = nn.Linear(ecmwf_dim, nwp_out_dim, bias=True)

        self.norm = nn.LayerNorm(nwp_out_dim)

    # ------------------------------------------------------------------
    # idw / idw_alt aggregation helper
    # ------------------------------------------------------------------

    def _idw_aggregate(
        self,
        x_src: Tensor,            # (N_src, F_src)
        edge_index: Tensor,       # (2, E)
        edge_attr: Tensor,        # (E, edge_dim)
        lin: nn.Linear,
        N_dst: int,
        max_dist_km: float,       # only read under aggregation="idw_alt"
    ) -> Tensor:                  # (N_dst, nwp_out_dim)
        out = torch.zeros(N_dst, self.nwp_out_dim, dtype=x_src.dtype, device=x_src.device)
        if edge_index.numel() == 0:
            return out
        dst = edge_index[1]
        if self.aggregation == "idw_alt":
            dist = _idw_alt_distance_km(
                edge_attr, self.alt_col, max_dist_km, self.alpha_alt,
            )
        else:
            dist = edge_attr[:, _DIST_COL].clamp(min=0.0)
        w = _idw_weights(dist, dst, N_dst, self.idw_p)                 # (E,)
        src = edge_index[0]
        msg = lin(x_src)[src] * w.unsqueeze(-1)                        # (E, nwp_out_dim)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.nwp_out_dim), msg)
        return out

    # ------------------------------------------------------------------
    # Single-step forward (kept for flexibility / debugging)
    # ------------------------------------------------------------------

    def forward(
        self,
        icond2_t: Tensor,        # (N_i, I2)
        ecmwf_t: Tensor,         # (N_e, E2)
        h_station: Tensor,       # (N_s, station_dim) — attention query; unused in idw/idw_alt mode
        i2s_edge_index: Tensor,
        i2s_edge_attr: Tensor,
        e2s_edge_index: Tensor,
        e2s_edge_attr: Tensor,
    ) -> Tensor:                 # (N_s, nwp_out_dim)
        N_s = h_station.size(0)
        if self.aggregation in ("idw", "idw_alt"):
            msg_i = self._idw_aggregate(
                icond2_t, i2s_edge_index, i2s_edge_attr, self.lin_i2s, N_s,
                self.icond2_max_dist_km,
            )
            if self.ecmwf_dim == 0:
                return self.norm(msg_i)
            msg_e = self._idw_aggregate(
                ecmwf_t, e2s_edge_index, e2s_edge_attr, self.lin_e2s, N_s,
                self.ecmwf_max_dist_km,
            )
            return self.norm(msg_i + msg_e)

        msg_i = self.gat_i2s((icond2_t, h_station), i2s_edge_index, i2s_edge_attr)
        if self.ecmwf_dim == 0:
            return self.norm(msg_i)
        msg_e = self.gat_e2s((ecmwf_t,  h_station), e2s_edge_index, e2s_edge_attr)
        return self.norm(msg_i + msg_e)

    # ------------------------------------------------------------------
    # Vectorised forward over all T timesteps (block-diagonal trick)
    #
    # Attention-only — the DCRNN path (the only caller of aggregation in
    # {"idw", "idw_alt"}) never calls this method; see the module docstring.
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
                "path — the only caller of aggregation in {'idw', 'idw_alt'} — always "
                "uses forward()."
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
