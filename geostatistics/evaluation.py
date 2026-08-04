"""
geostatistics/evaluation.py — Single-pass evaluation for STGNN2 / DCRNN.

Imported by:
  - get_test_results_stgnn2.py  (standalone evaluation script)
  - train_stgnn2.py             (optional post-training eval via --eval)
  - train_dcrnn.py              (same)

Evaluation design
-----------------
  observer=train,  target=val  (zero-shot: train context, all val stations as targets)

All metrics are computed in physical units (inverse-transformed).
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from geostatistics.stgnn.training.sampler import TrainingSampler
from geostatistics.stgnn.utils.normalization import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature index helper
# ---------------------------------------------------------------------------

def find_ws_feat_idx(features: list[str]) -> int | None:
    """Return index of wind_speed_10m in features, or None if not found.

    Must match exactly — apply_dir_encoding reorders columns (non-consumed
    features first) so wind_speed_38m ends up at index 0 in dir_in_deg mode,
    which would be wrong for the NWP baseline.
    """
    for i, f in enumerate(features):
        if f == "wind_speed_10m":
            return i
    # Fallback: first feature starting with wind_speed (logs a warning at call site)
    for i, f in enumerate(features):
        if f.startswith("wind_speed"):
            return i
    return None


# ---------------------------------------------------------------------------
# Build a HeteroData eval batch for an arbitrary observer / target split
# ---------------------------------------------------------------------------

def build_eval_batch(
    sampler: TrainingSampler,
    r_curr: int,
    r_hist: int,
    t_run_abs: int,
    station_meas_scaled: np.ndarray,        # (T, N_all, M)
    station_nearest_grid: np.ndarray,       # (N_all,)
    grid_icond2_runs_scaled: np.ndarray,    # (R, 48, N_grid, I2)
    station_ecmwf_nwp_scaled: np.ndarray,  # (T, N_all, E2)
    station_static: np.ndarray,             # (N_all, S-1)  without type indicator
    ecmwf_nwp_scaled: np.ndarray,          # (T, N_ecmwf, E2)
    icond2_static: np.ndarray,
    ecmwf_static: np.ndarray,
    target_global: list[int],
    observer_global: list[int],
    fold_train_indices: list[int],
    target_feat_idx: int,
    H_hist: int,
    H_fore: int,
    interpol_meas: np.ndarray | None = None,  # (T, N_all) Kriging lag, pre-scaled
    hist_wind_available: bool = False,
    neighbour_meas_available: bool = True,   # ablation B/C: False → no station has measurements
    station_k_nearest_grid: np.ndarray | None = None,  # (N_all, k) — k nearest for nwp_nodes=False
    station_k_nearest_ecmwf: np.ndarray | None = None, # (N_all, k_e) — k nearest ECMWF, nwp_nodes=False
) -> tuple:
    """
    Build a HeteroData evaluation batch for the given station split.

    Returns
    -------
    data        : HeteroData (not yet on GPU)
    target_mask : (N_all,) bool tensor
    gt_scaled   : (N_target, H_fore) numpy array — scaled ground truth
    """
    all_global = observer_global + target_global
    N_obs = len(observer_global)
    N_all = len(all_global)

    target_mask = torch.zeros(N_all, dtype=torch.bool)
    target_mask[N_obs:] = True

    t_hist_abs = t_run_abs - H_hist

    if station_k_nearest_grid is not None:
        # k nearest: (N_all, k) → features (N_all, 48, k*I2) matching training
        k_idx   = station_k_nearest_grid[all_global]             # (N_all, k)
        i2_hist = grid_icond2_runs_scaled[r_hist, :, k_idx, :].transpose(0, 2, 1, 3).reshape(N_all, 48, -1)
        i2_curr = grid_icond2_runs_scaled[r_curr, :, k_idx, :].transpose(0, 2, 1, 3).reshape(N_all, 48, -1)
    else:
        nearest = station_nearest_grid[all_global]
        i2_hist = grid_icond2_runs_scaled[r_hist, :, nearest, :]    # (N_all, 48, I2)
        i2_curr = grid_icond2_runs_scaled[r_curr, :, nearest, :]    # (N_all, 48, I2)
    i2_full = np.concatenate([i2_hist, i2_curr], axis=1)        # (N_all, 96, [k*]I2)

    i2_grid_full = np.concatenate([
        grid_icond2_runs_scaled[r_hist],
        grid_icond2_runs_scaled[r_curr],
    ], axis=0)                                                   # (96, N_grid, I2)

    e2_grid_full = ecmwf_nwp_scaled[t_hist_abs:t_run_abs + H_fore]   # (96, N_ecmwf, E2)
    if station_k_nearest_ecmwf is not None:
        # k naechste ECMWF-Punkte konkateniert, spiegelbildlich zu ICON-D2 oben
        ke_idx  = station_k_nearest_ecmwf[all_global]            # (N_all, k_e)
        e2_full = e2_grid_full[:, ke_idx, :].transpose(1, 0, 2, 3).reshape(
            N_all, e2_grid_full.shape[0], -1)                    # (N_all, 96, k_e*E2)
    else:
        e2_full = station_ecmwf_nwp_scaled[t_hist_abs:t_run_abs + H_fore, :, :][:, all_global, :]
        e2_full = e2_full.transpose(1, 0, 2)                     # (N_all, 96, E2)

    meas_hist = station_meas_scaled[t_hist_abs:t_run_abs, :, :][:, all_global, :].copy()
    # Order matters: ablation B subsumes the IGNNK zeroing and must come first,
    # otherwise only the target stations would lose their measurements. Same rule
    # as in TrainingSampler.sample_train / sample_val.
    if not neighbour_meas_available:
        meas_hist[:, :, :] = 0.0                # ablation B/C: nobody has measurements
    elif not hist_wind_available:
        meas_hist[:, N_obs:, :] = 0.0           # IGNNK masking (variant A)

    if interpol_meas is not None:
        rk_slice = interpol_meas[t_hist_abs:t_run_abs, :][:, all_global, np.newaxis]
        meas_hist = np.concatenate([meas_hist, rk_slice], axis=2)

    gt_raw = station_meas_scaled[t_run_abs:t_run_abs + H_fore, :, target_feat_idx]
    gt_scaled = gt_raw[:, all_global][:, N_obs:].T.copy()      # (N_target, H_fore)

    stat_sub  = station_static[all_global, :]
    type_ind  = (~target_mask).float().unsqueeze(1).numpy()
    stat_full = np.concatenate([stat_sub, type_ind], axis=1)   # (N_all, S)

    data = sampler._make_data(
        all_global=all_global,
        meas_hist=meas_hist,
        i2_full=i2_full,
        e2_full=e2_full,
        stat_full=stat_full,
        icond2_nwp=i2_grid_full,
        ecmwf_nwp=e2_grid_full,
        icond2_static=icond2_static,
        ecmwf_static=ecmwf_static,
        fold_train_indices=fold_train_indices,
        target_global=target_global,
    )
    return data, target_mask, gt_scaled


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    sampler: TrainingSampler,
    device: torch.device,
    meas_raw: np.ndarray,                    # (T, N_all, M) — physical units
    meas_scaled: np.ndarray,                 # (T, N_all, M) — scaled
    station_nearest_grid: np.ndarray,        # (N_all,)
    grid_icond2_runs_raw: np.ndarray,        # (R, 48, N_grid, I2) — physical
    grid_icond2_runs_scaled: np.ndarray,
    station_ecmwf_nwp_scaled: np.ndarray,
    station_static: np.ndarray,
    ecmwf_nwp_scaled: np.ndarray,
    icond2_static: np.ndarray,
    ecmwf_static: np.ndarray,
    meas_scaler: StandardScaler,
    target_feat_idx: int,
    ws_feat_idx_i2: int | None,
    H_hist: int,
    H_fore: int,
    train_station_indices: list[int],
    val_station_indices: list[int],
    all_ids: list[str],
    test_run_pairs: list[tuple[int, int, int]],
    interpol_meas: np.ndarray | None = None,  # (T, N_all) Kriging lag, pre-scaled
    hist_wind_available: bool = False,
    neighbour_meas_available: bool = True,   # ablation B/C: False → no station has measurements
    timestamps: "pd.DatetimeIndex | None" = None,
    station_k_nearest_grid: np.ndarray | None = None,  # (N_all, k) — k nearest for nwp_nodes=False
    station_k_nearest_ecmwf: np.ndarray | None = None, # (N_all, k_e) — k nearest ECMWF, nwp_nodes=False
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Single-pass evaluation over all test run pairs.

    All train stations serve as context; all val stations are predicted simultaneously.
    Returns (station_df, raw_df):
      station_df — per-station aggregate metrics: station_id, mae, rmse, r2, skill, skill_nwp, n_samples
      raw_df     — per-prediction rows: station_id, run_time, valid_time, horizon, pred, gt, nwp_ref, pers_ref
                   (run_time / valid_time are NaT when timestamps=None)
    """
    preds_acc: dict[int, list[np.ndarray]] = defaultdict(list)
    gt_acc:    dict[int, list[np.ndarray]] = defaultdict(list)
    nwp_acc:   dict[int, list[np.ndarray]] = defaultdict(list)
    pers_acc:  dict[int, list[np.ndarray]] = defaultdict(list)
    raw_records: list[dict] = []

    mean_ws = float(meas_scaler.mean_[target_feat_idx])
    std_ws  = float(meas_scaler.std_[target_feat_idx] + meas_scaler.eps)

    def _to_phys(arr: np.ndarray) -> np.ndarray:
        return arr * std_ws + mean_ws

    common = dict(
        sampler=sampler,
        station_meas_scaled=meas_scaled,
        station_nearest_grid=station_nearest_grid,
        station_k_nearest_grid=station_k_nearest_grid,
        station_k_nearest_ecmwf=station_k_nearest_ecmwf,
        grid_icond2_runs_scaled=grid_icond2_runs_scaled,
        station_ecmwf_nwp_scaled=station_ecmwf_nwp_scaled,
        station_static=station_static,
        ecmwf_nwp_scaled=ecmwf_nwp_scaled,
        icond2_static=icond2_static,
        ecmwf_static=ecmwf_static,
        target_feat_idx=target_feat_idx,
        H_hist=H_hist,
        H_fore=H_fore,
        interpol_meas=interpol_meas,
        hist_wind_available=hist_wind_available,
        neighbour_meas_available=neighbour_meas_available,
    )

    def _nwp_ref(gidx: int, r_curr: int) -> np.ndarray:
        if ws_feat_idx_i2 is None:
            return np.full(H_fore, np.nan, dtype=np.float32)
        return grid_icond2_runs_raw[
            r_curr, :H_fore, station_nearest_grid[gidx], ws_feat_idx_i2
        ]

    def _pers_ref(gidx: int, t_run_abs: int) -> np.ndarray:
        val = float(meas_raw[t_run_abs - 1, gidx, target_feat_idx])
        return np.full(H_fore, val, dtype=np.float32)

    # Observer (context) selection must MATCH training: the model was trained
    # seeing only the next_n_neighbors nearest train stations per target, not all
    # train stations. Using all of them at eval changes the station graph topology
    # and degrades models that rely on the neighbour context (esp. nwp_nodes=true
    # + hist_wind_available=false). Shared with sample_val so the two cannot drift.
    observer_global = sampler.select_val_neighbours(
        val_station_indices, train_station_indices,
    )
    logger.info(
        "Observer context: %d / %d train stations (next_n_neighbors=%s)",
        len(observer_global), len(train_station_indices), sampler.tc.next_n_neighbors,
    )

    model.eval()
    with torch.no_grad():
        for step, (r_curr, r_hist, t_run_abs) in enumerate(test_run_pairs):
            if step % 10 == 0:
                logger.info("  Pair %d / %d", step + 1, len(test_run_pairs))

            if not val_station_indices:
                continue

            data_a, mask_a, _ = build_eval_batch(
                **common,
                r_curr=r_curr, r_hist=r_hist, t_run_abs=t_run_abs,
                target_global=val_station_indices,
                observer_global=observer_global,
                fold_train_indices=train_station_indices,
            )
            preds_a = _to_phys(
                model(data_a.to(device), mask_a.to(device)).cpu().numpy()
            )  # (N_val, H_fore)
            gt_a = meas_raw[
                t_run_abs:t_run_abs + H_fore, :, target_feat_idx
            ][:, val_station_indices].T  # (N_val, H_fore)

            run_ts = timestamps[t_run_abs - 1] if timestamps is not None else None
            for i, gidx in enumerate(val_station_indices):
                nwp_h  = _nwp_ref(gidx, r_curr)
                pers_h = _pers_ref(gidx, t_run_abs)
                preds_acc[gidx].append(preds_a[i])
                gt_acc[gidx].append(gt_a[i])
                nwp_acc[gidx].append(nwp_h)
                pers_acc[gidx].append(pers_h)
                sid = all_ids[gidx]
                for h in range(H_fore):
                    raw_records.append({
                        "station_id": sid,
                        "run_time":   run_ts,
                        "valid_time": (run_ts + pd.Timedelta(hours=h + 1)) if run_ts is not None else None,
                        "horizon":    h + 1,
                        "pred":       float(preds_a[i, h]),
                        "gt":         float(gt_a[i, h]),
                        "nwp_ref":    float(nwp_h[h]),
                        "pers_ref":   float(pers_h[h]),
                    })

    logger.info("Computing per-station metrics …")
    records = []

    for gidx in val_station_indices:
        p_all  = np.concatenate(preds_acc[gidx])
        g_all  = np.concatenate(gt_acc[gidx])
        n_all  = np.concatenate(nwp_acc[gidx])
        ps_all = np.concatenate(pers_acc[gidx])

        valid = ~(np.isnan(p_all) | np.isnan(g_all))
        if valid.sum() < 2:
            logger.warning(
                "Station %s: too few valid samples (%d), skipping",
                all_ids[gidx], int(valid.sum()),
            )
            continue

        p_v, g_v = p_all[valid], g_all[valid]
        r2   = float(r2_score(g_v, p_v))
        rmse = float(math.sqrt(mean_squared_error(g_v, p_v)))
        mae  = float(mean_absolute_error(g_v, p_v))

        valid_pers = ~(np.isnan(ps_all) | np.isnan(g_all))
        if valid_pers.sum() >= 2:
            rmse_pers = float(math.sqrt(mean_squared_error(g_all[valid_pers], ps_all[valid_pers])))
            skill     = (1.0 - rmse / rmse_pers) if rmse_pers > 0 else float("nan")
        else:
            skill = float("nan")

        valid_nwp = ~(np.isnan(n_all) | np.isnan(g_all))
        if valid_nwp.sum() >= 2:
            rmse_nwp  = float(math.sqrt(mean_squared_error(g_all[valid_nwp], n_all[valid_nwp])))
            skill_nwp = (1.0 - rmse / rmse_nwp) if rmse_nwp > 0 else float("nan")
        else:
            skill_nwp = float("nan")

        records.append({
            "station_id": all_ids[gidx],
            "mae":        mae,
            "rmse":       rmse,
            "r2":         r2,
            "skill":      skill,
            "skill_nwp":  skill_nwp,
            "n_samples":  int(valid.sum()),
        })

    return pd.DataFrame(records), pd.DataFrame(raw_records)
