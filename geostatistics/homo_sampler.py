"""
geostatistics/homo_sampler.py — Lightweight homogeneous-graph sampler.

Shared by MTGNN and GraphWaveNet baseline models.  Unlike the heterogeneous
TrainingSampler this sampler:

  • Returns plain tensor HomoBatch (no HeteroData / torch_geometric dependency)
  • Aggregates NWP grid points to station level (IDW-mean or raw concatenation)
  • Supports both ICON-D2 (run-indexed) and ECMWF (time-indexed) NWP sources

Shape conventions
-----------------
  meas_scaled          : (T, N, M)
  grid_icond2_scaled   : (R, n_leads, N_grid_i2, I2)
  grid_ecmwf_scaled    : (T, N_grid_e2, E2)  — optional, None if unused
  x in HomoBatch       : (N_sub, T_total, in_channels)   T_total = H + F_h
  static in HomoBatch  : (N_sub, 6)
  target_mask          : (N_sub,) bool  — True = target / IGNNK-masked station
  ground_truth         : (N_target, F_h)

aggregate_nwp parameter
------------------------
  True  (default) — IDW-weighted mean of k nearest NWP grid points per station.
                    ICON-D2 contributes I2 channels, ECMWF E2 channels.
                    in_channels = M + I2 + E2
  False           — all k grid points concatenated as individual features.
                    ICON-D2 contributes k_nwp * I2, ECMWF k_ecmwf * E2 channels.
                    in_channels = M + k_nwp*I2 + k_ecmwf*E2

Static features (6 columns per station)
----------------------------------------
  0  sin(lat_rad)
  1  cos(lat_rad)
  2  sin(lon_rad)
  3  cos(lon_rad)
  4  altitude normalised by train-station mean/std
  5  type indicator : 0 = target (masked), 1 = neighbour

Design notes
------------
  • n_leads == H == F_h is assumed (standard ICON-D2 run-based setup).
  • Validation iterates deterministically: all train stations = neighbours,
    all val stations = targets (full LOO setup).
  • max_neighbor_stations caps neighbours during training only.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor


# ---------------------------------------------------------------------------
# Batch dataclass
# ---------------------------------------------------------------------------

@dataclass
class HomoBatch:
    x:            Tensor  # (N_sub, T_total, M+I2)
    static:       Tensor  # (N_sub, 6)
    target_mask:  Tensor  # (N_sub,) bool — True = target station
    ground_truth: Tensor  # (N_target, F_h)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class HomoSampler:
    """
    Lightweight sampler for homogeneous station-only graph models (MTGNN, WaveNet).

    Parameters
    ----------
    meas_scaled          : (T, N, M) — scaled station measurements
    grid_icond2_scaled   : (R, n_leads, N_grid, I2) — scaled ICON-D2 NWP runs
    train_run_pairs      : list[(r_curr, r_hist, t_run_abs)] — training pairs
    val_run_pairs        : list[(r_curr, r_hist, t_run_abs)] — validation pairs
    train_station_indices: list[int] — indices into dim-1 of meas_scaled
    val_station_indices  : list[int] — indices into dim-1 of meas_scaled
    lats, lons, alts     : (N,) station coordinates / elevation in metres
    icond2_coords        : (N_grid, 2) — (lat, lon) of ICON-D2 grid points
    history_length       : H   (must equal n_leads in grid_icond2_scaled)
    forecast_horizon     : F_h (must equal n_leads in grid_icond2_scaled)
    target_feat_idx      : column index of the target variable in M
    k_nwp                : number of nearest NWP grid points per station
    min_target_stations  : minimum IGNNK target count per training sample
    max_target_stations  : maximum IGNNK target count per training sample
    max_neighbor_stations: cap on neighbour count during training
    """

    def __init__(
        self,
        *,
        meas_scaled: np.ndarray,
        grid_icond2_scaled: np.ndarray,
        train_run_pairs: list[tuple[int, int, int]],
        val_run_pairs: list[tuple[int, int, int]],
        train_station_indices: list[int],
        val_station_indices: list[int],
        lats: np.ndarray,
        lons: np.ndarray,
        alts: np.ndarray,
        icond2_coords: np.ndarray,
        history_length: int,
        forecast_horizon: int,
        target_feat_idx: int = 0,
        k_nwp: int = 4,
        min_target_stations: int = 1,
        max_target_stations: int = 10,
        max_neighbor_stations: int = 60,
        next_n_neighbors: int | None = None,
        hist_wind_available: bool = False,
        # ECMWF (optional)
        grid_ecmwf_scaled: np.ndarray | None = None,  # (T, N_ecmwf_grid, E2)
        ecmwf_coords: np.ndarray | None = None,        # (N_ecmwf_grid, 2)
        k_ecmwf: int = 0,
        aggregate_nwp: bool = True,
    ) -> None:
        self.meas      = meas_scaled            # (T, N, M)
        self.nwp_runs  = grid_icond2_scaled     # (R, n_leads, N_grid_i2, I2)
        self.ecmwf_nwp = grid_ecmwf_scaled      # (T, N_grid_e2, E2) or None
        self.train_pairs = train_run_pairs
        self.val_pairs   = val_run_pairs
        self.train_idx   = list(train_station_indices)
        self.val_idx     = list(val_station_indices)
        self.H           = history_length
        self.Fh          = forecast_horizon
        self.target_feat_idx = target_feat_idx
        self.k_nwp       = k_nwp
        self.k_ecmwf     = k_ecmwf
        self.aggregate_nwp = aggregate_nwp
        self.min_tgt = min_target_stations
        self.max_tgt = max_target_stations
        self.max_nbr = max_neighbor_stations
        self.next_n_neighbors = next_n_neighbors
        self.hist_wind_available = hist_wind_available

        self.I2 = grid_icond2_scaled.shape[3]
        self.E2 = grid_ecmwf_scaled.shape[2] if grid_ecmwf_scaled is not None else 0

        self._init_nwp_knn(lats, lons, icond2_coords)
        if grid_ecmwf_scaled is not None and k_ecmwf > 0 and ecmwf_coords is not None:
            self._init_ecmwf_knn(lats, lons, ecmwf_coords)
        else:
            self._ecmwf_knn_idx = np.empty((len(lats), 0), dtype=np.int32)
            self._ecmwf_knn_w   = np.empty((len(lats), 0), dtype=np.float32)
        self._init_static(lats, lons, alts)

    @property
    def in_channels(self) -> int:
        """Total feature channels per station per time step in HomoBatch.x."""
        M = self.meas.shape[2]
        if self.aggregate_nwp:
            return M + self.I2 + self.E2
        else:
            k_i2 = self._nwp_knn_idx.shape[1]
            k_e2 = self._ecmwf_knn_idx.shape[1]
            return M + k_i2 * self.I2 + k_e2 * self.E2

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_nwp_knn(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        icond2_coords: np.ndarray,
    ) -> None:
        """Precompute k nearest ICON-D2 grid-point indices and inv-dist weights."""
        k = min(self.k_nwp, len(icond2_coords))
        tree = cKDTree(icond2_coords)
        station_ll = np.stack([lats, lons], axis=1)
        dists, idxs = tree.query(station_ll, k=k)
        if k == 1:
            dists = dists[:, np.newaxis]
            idxs  = idxs[:, np.newaxis]

        eps = 1e-6
        raw_w   = 1.0 / np.maximum(dists, eps)
        weights = raw_w / raw_w.sum(axis=1, keepdims=True)

        self._nwp_knn_idx = idxs.astype(np.int32)    # (N, k_nwp)
        self._nwp_knn_w   = weights.astype(np.float32)

    def _init_ecmwf_knn(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        ecmwf_coords: np.ndarray,
    ) -> None:
        """Precompute k nearest ECMWF grid-point indices and inv-dist weights."""
        k = min(self.k_ecmwf, len(ecmwf_coords))
        tree = cKDTree(ecmwf_coords)
        station_ll = np.stack([lats, lons], axis=1)
        dists, idxs = tree.query(station_ll, k=k)
        if k == 1:
            dists = dists[:, np.newaxis]
            idxs  = idxs[:, np.newaxis]

        eps = 1e-6
        raw_w   = 1.0 / np.maximum(dists, eps)
        weights = raw_w / raw_w.sum(axis=1, keepdims=True)

        self._ecmwf_knn_idx = idxs.astype(np.int32)    # (N, k_ecmwf)
        self._ecmwf_knn_w   = weights.astype(np.float32)

    def _init_static(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        alts: np.ndarray,
    ) -> None:
        """Precompute sin/cos-encoded lat/lon and train-normalised altitude."""
        lat_r = np.deg2rad(lats)
        lon_r = np.deg2rad(lons)
        self._sin_lat = np.sin(lat_r).astype(np.float32)
        self._cos_lat = np.cos(lat_r).astype(np.float32)
        self._sin_lon = np.sin(lon_r).astype(np.float32)
        self._cos_lon = np.cos(lon_r).astype(np.float32)

        train_alts = alts[self.train_idx]
        alt_mean   = float(train_alts.mean())
        alt_std    = max(float(train_alts.std()), 1.0)
        self._alt_norm = ((alts - alt_mean) / alt_std).astype(np.float32)

        # Station-to-station coordinates (radians) for next_n_neighbors selection
        self._station_coords_rad = np.stack([lat_r, lon_r], axis=1).astype(np.float64)

    # ------------------------------------------------------------------
    # NWP aggregation
    # ------------------------------------------------------------------

    def _aggregate_nwp(
        self,
        r_hist: int,
        r_curr: int,
        sub_indices: list[int],
    ) -> np.ndarray:
        """ICON-D2 NWP for a station subset.

        aggregate_nwp=True  → IDW-weighted mean  → (N_sub, T_total, I2)
        aggregate_nwp=False → concat k points    → (N_sub, T_total, k*I2)
        """
        idxs = self._nwp_knn_idx[sub_indices]   # (N_sub, k)
        wgts = self._nwp_knn_w[sub_indices]     # (N_sub, k)

        def _process(run: np.ndarray) -> np.ndarray:
            # run : (n_leads, N_grid, I2)
            sub = run[:, idxs, :]                        # (n_leads, N_sub, k, I2)
            if self.aggregate_nwp:
                w   = wgts[np.newaxis, :, :, np.newaxis]  # (1, N_sub, k, 1)
                agg = (sub * w).sum(axis=2)               # (n_leads, N_sub, I2)
                return agg.transpose(1, 0, 2).astype(np.float32)  # (N_sub, n_leads, I2)
            else:
                n_leads, N_sub, k, I2 = sub.shape
                return sub.transpose(1, 0, 2, 3).reshape(N_sub, n_leads, k * I2).astype(np.float32)

        hist = _process(self.nwp_runs[r_hist])   # (N_sub, n_leads, *)
        curr = _process(self.nwp_runs[r_curr])
        return np.concatenate([hist, curr], axis=1)   # (N_sub, T_total, *)

    def _aggregate_ecmwf(
        self,
        t_hist_abs: int,
        t_end_abs: int,
        sub_indices: list[int],
    ) -> np.ndarray | None:
        """ECMWF NWP for a station subset over the given time window.

        aggregate_nwp=True  → IDW-weighted mean  → (N_sub, T_total, E2)
        aggregate_nwp=False → concat k points    → (N_sub, T_total, k*E2)
        Returns None if ECMWF is not configured.
        """
        if self.ecmwf_nwp is None or self._ecmwf_knn_idx.shape[1] == 0:
            return None

        idxs = self._ecmwf_knn_idx[sub_indices]   # (N_sub, k)
        wgts = self._ecmwf_knn_w[sub_indices]     # (N_sub, k)

        # (T_total, N_sub, k, E2)
        window = self.ecmwf_nwp[t_hist_abs:t_end_abs]   # (T_total, N_grid_e2, E2)
        sub    = window[:, idxs, :]                      # (T_total, N_sub, k, E2)

        if self.aggregate_nwp:
            w   = wgts[np.newaxis, :, :, np.newaxis]        # (1, N_sub, k, 1)
            agg = (sub * w).sum(axis=2)                      # (T_total, N_sub, E2)
            return agg.transpose(1, 0, 2).astype(np.float32)  # (N_sub, T_total, E2)
        else:
            T_total, N_sub, k, E2 = sub.shape
            return sub.transpose(1, 0, 2, 3).reshape(N_sub, T_total, k * E2).astype(np.float32)

    # ------------------------------------------------------------------
    # Static features
    # ------------------------------------------------------------------

    def _build_static(
        self,
        sub_indices: list[int],
        target_mask_np: np.ndarray,
    ) -> np.ndarray:
        """Build (N_sub, 6) static feature matrix.

        Columns: sin_lat, cos_lat, sin_lon, cos_lon, alt_norm, type_indicator
        type_indicator: 0 = target (IGNNK masked), 1 = neighbour
        """
        idx = np.array(sub_indices)
        return np.stack([
            self._sin_lat[idx],
            self._cos_lat[idx],
            self._sin_lon[idx],
            self._cos_lon[idx],
            self._alt_norm[idx],
            (~target_mask_np).astype(np.float32),
        ], axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Batch assembly
    # ------------------------------------------------------------------

    def _make_batch(
        self,
        r_curr: int,
        r_hist: int,
        t_run_abs: int,
        sub_indices: list[int],
        target_mask_np: np.ndarray,
    ) -> HomoBatch:
        """Assemble one HomoBatch.

        Parameters
        ----------
        sub_indices    : global station indices (into dim-1 of self.meas)
        target_mask_np : (N_sub,) bool — True = target / IGNNK-zeroed station
        """
        N_sub  = len(sub_indices)
        M      = self.meas.shape[2]
        T_total = self.H + self.Fh
        t_hist_abs = t_run_abs - self.H

        # Measurements — history window, target stations zeroed (IGNNK)
        meas_hist = self.meas[t_hist_abs:t_run_abs, :, :][:, sub_indices, :].copy()
        if not self.hist_wind_available:
            meas_hist[:, target_mask_np, :] = 0.0

        # Pad forecast horizon with zeros → (N_sub, T_total, M)
        meas_full = np.zeros((N_sub, T_total, M), dtype=np.float32)
        meas_full[:, :self.H, :] = meas_hist.transpose(1, 0, 2)

        # ICON-D2 aggregated at station level → (N_sub, T_total, I2 or k*I2)
        nwp_agg = self._aggregate_nwp(r_hist, r_curr, sub_indices)

        # ECMWF aggregated at station level → (N_sub, T_total, E2 or k*E2) or None
        ecmwf_agg = self._aggregate_ecmwf(t_hist_abs, t_run_abs + self.Fh, sub_indices)

        parts = [meas_full, nwp_agg]
        if ecmwf_agg is not None:
            parts.append(ecmwf_agg)
        x = np.concatenate(parts, axis=2)   # (N_sub, T_total, in_channels)

        # Static features
        static = self._build_static(sub_indices, target_mask_np)  # (N_sub, 6)

        # Ground truth — target stations, forecast horizon
        gt_raw = self.meas[t_run_abs:t_run_abs + self.Fh, :, self.target_feat_idx]
        gt_sub = gt_raw[:, sub_indices]                     # (F_h, N_sub)
        gt_target = gt_sub[:, target_mask_np].T             # (N_target, F_h)

        return HomoBatch(
            x            = torch.from_numpy(x),
            static       = torch.from_numpy(static),
            target_mask  = torch.from_numpy(target_mask_np),
            ground_truth = torch.from_numpy(gt_target.copy().astype(np.float32)),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample_train(self) -> HomoBatch:
        """Draw one random training batch (IGNNK masking on a random run pair)."""
        r_curr, r_hist, t_run_abs = random.choice(self.train_pairs)
        N_train = len(self.train_idx)

        n_target = random.randint(
            min(self.min_tgt, N_train),
            min(self.max_tgt, N_train),
        )
        target_local = sorted(random.sample(range(N_train), n_target))
        target_set   = set(target_local)
        nbr_local    = [i for i in range(N_train) if i not in target_set]
        if self.next_n_neighbors is not None and self.next_n_neighbors < len(nbr_local):
            # Deterministic: pick spatially nearest candidates to the target group
            tgt_global  = [self.train_idx[i] for i in target_local]
            cand_global = [self.train_idx[i] for i in nbr_local]
            tgt_coords  = self._station_coords_rad[tgt_global]   # (n_tgt, 2)
            cand_coords = self._station_coords_rad[cand_global]  # (n_cand, 2)
            tgt_tree    = cKDTree(tgt_coords)
            min_dists, _ = tgt_tree.query(cand_coords, k=1)      # min dist to any target
            order       = np.argsort(min_dists)
            keep        = {cand_global[i] for i in order[:self.next_n_neighbors]}
            nbr_local   = [i for i in nbr_local if self.train_idx[i] in keep]
        elif self.max_nbr < len(nbr_local):
            nbr_local = random.sample(nbr_local, self.max_nbr)

        all_local = sorted(nbr_local + target_local)
        sub_indices    = [self.train_idx[i] for i in all_local]
        target_mask_np = np.array([i in target_set for i in all_local], dtype=bool)

        return self._make_batch(r_curr, r_hist, t_run_abs, sub_indices, target_mask_np)

    def iter_val(self):
        """Iterate over all validation run pairs deterministically.

        For each pair: all train stations act as neighbours, all val stations
        are targets (full LOO evaluation).  Yields HomoBatch.
        """
        N_nbr = len(self.train_idx)
        all_global = self.train_idx + self.val_idx
        target_mask_np = np.zeros(len(all_global), dtype=bool)
        target_mask_np[N_nbr:] = True

        for r_curr, r_hist, t_run_abs in self.val_pairs:
            yield self._make_batch(r_curr, r_hist, t_run_abs, all_global, target_mask_np)

    def iter_val_meta(self):
        """Like iter_val() but also yields (r_curr, t_run_abs) for each pair."""
        N_nbr = len(self.train_idx)
        all_global = self.train_idx + self.val_idx
        target_mask_np = np.zeros(len(all_global), dtype=bool)
        target_mask_np[N_nbr:] = True

        for r_curr, r_hist, t_run_abs in self.val_pairs:
            yield (
                self._make_batch(r_curr, r_hist, t_run_abs, all_global, target_mask_np),
                r_curr,
                t_run_abs,
            )


# ---------------------------------------------------------------------------
# Standalone evaluation helper (shared by MTGNN and WaveNet train scripts)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_homo_model(
    model: torch.nn.Module,
    sampler: HomoSampler,
    device: torch.device,
    val_ids: list[str],
    meas_raw: np.ndarray,           # (T, N, M) — physical units, unscaled
    grid_icond2_runs: np.ndarray,   # (R, n_leads, N_grid, I2) — physical units
    target_scale: float,
    target_mean: float,
    target_feat_idx: int,
    nwp_ws_feat_idx: int = 0,       # index of wind_speed feature in I2 dim
    max_pairs: int | None = None,   # cap number of val run pairs (None = all)
) -> "pd.DataFrame":
    """Per-station evaluation on validation run pairs.

    Single-pass design (no LOO): all train stations serve as context,
    all val stations are predicted simultaneously — one forward pass per
    run pair.  This is the appropriate evaluation mode for the training
    script (fast, no combinatorial LOO overhead).

    Returns a DataFrame with columns:
        station_id, mae, rmse, r2, skill, skill_nwp, n_samples

    skill     = 1 − RMSE_model / RMSE_persistence
    skill_nwp = 1 − RMSE_model / RMSE_{nearest ICON-D2 grid point}
    """
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model.eval()
    N_val = len(val_ids)
    F_h   = sampler.Fh

    # Nearest ICON-D2 grid point index per val station (into N_grid axis)
    val_nearest = sampler._nwp_knn_idx[sampler.val_idx, 0]   # (N_val,)

    preds_acc: list[list] = [[] for _ in range(N_val)]
    gts_acc:   list[list] = [[] for _ in range(N_val)]
    nwp_acc:   list[list] = [[] for _ in range(N_val)]
    pers_acc:  list[list] = [[] for _ in range(N_val)]

    pairs_done = 0
    for batch, r_curr, t_run_abs in sampler.iter_val_meta():
        if max_pairs is not None and pairs_done >= max_pairs:
            break
        pairs_done += 1
        x           = batch.x.unsqueeze(0).to(device)
        static      = batch.static.to(device)
        target_mask = batch.target_mask.to(device)
        if target_mask.sum() == 0:
            continue

        pred = model(x, static, target_mask)   # (N_val, F_h) scaled

        pred_phys = pred.cpu().numpy() * target_scale + target_mean    # (N_val, F_h)
        gt_phys   = batch.ground_truth.numpy() * target_scale + target_mean  # (N_val, F_h)

        # NWP baseline: nearest ICON-D2 grid point in physical units
        # grid_icond2_runs[r_curr] has shape (n_leads, N_grid, I2)
        nwp_raw = grid_icond2_runs[r_curr, :F_h, :, nwp_ws_feat_idx]  # (F_h, N_grid)
        nwp_fc  = nwp_raw[:, val_nearest].T.astype(np.float32)          # (N_val, F_h)

        # Persistence baseline: last actual measurement before forecast start
        pers_vals = meas_raw[t_run_abs - 1, sampler.val_idx, target_feat_idx]  # (N_val,)
        pers_fc   = np.repeat(pers_vals[:, np.newaxis], F_h, axis=1).astype(np.float32)

        for i in range(N_val):
            preds_acc[i].append(pred_phys[i])
            gts_acc[i].append(gt_phys[i])
            nwp_acc[i].append(nwp_fc[i])
            pers_acc[i].append(pers_fc[i])

    rows = []
    for i, sid in enumerate(val_ids):
        if not preds_acc[i]:
            continue

        p  = np.concatenate(preds_acc[i])   # (n_pairs * F_h,)
        g  = np.concatenate(gts_acc[i])
        n  = np.concatenate(nwp_acc[i])
        ps = np.concatenate(pers_acc[i])

        valid = ~(np.isnan(p) | np.isnan(g))
        if valid.sum() < 2:
            continue
        p_v, g_v = p[valid], g[valid]

        rmse = float(math.sqrt(mean_squared_error(g_v, p_v)))
        mae  = float(mean_absolute_error(g_v, p_v))
        r2   = float(r2_score(g_v, p_v))

        valid_p = ~(np.isnan(ps) | np.isnan(g))
        if valid_p.sum() >= 2:
            rmse_pers = float(math.sqrt(mean_squared_error(g[valid_p], ps[valid_p])))
            skill     = (1.0 - rmse / rmse_pers) if rmse_pers > 0 else float("nan")
        else:
            skill = float("nan")

        valid_n = ~(np.isnan(n) | np.isnan(g))
        if valid_n.sum() >= 2:
            rmse_nwp  = float(math.sqrt(mean_squared_error(g[valid_n], n[valid_n])))
            skill_nwp = (1.0 - rmse / rmse_nwp) if rmse_nwp > 0 else float("nan")
        else:
            skill_nwp = float("nan")

        rows.append({
            "station_id": sid,
            "mae":        mae,
            "rmse":       rmse,
            "r2":         r2,
            "skill":      skill,
            "skill_nwp":  skill_nwp,
            "n_samples":  int(valid.sum()),
        })

    return pd.DataFrame(rows)
