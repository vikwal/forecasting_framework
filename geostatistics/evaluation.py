"""
geostatistics/evaluation.py — Shared 4-pass LOO evaluation for STGNN2 / DCRNN.

Imported by:
  - get_test_results_stgnn2.py  (standalone evaluation script)
  - train_stgnn2.py             (optional post-training eval via --eval)
  - train_dcrnn.py              (same)

Four-pass design
----------------
  Pass A  observer=train,           target=val     (zero-shot, train context)
  Pass B  observer=train∖{s},       target={s}     (in-sample,  LOO train)
  Pass C  observer=val+train∖{s},   target={s}     (in-sample,  LOO full)
  Pass D  observer=train+val∖{s},   target={s}     (zero-shot,  LOO full)

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
    """Return index of the wind-speed feature, or None if not found."""
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
    target_feat_idx: int,
    H_hist: int,
    H_fore: int,
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

    nearest = station_nearest_grid[all_global]
    i2_hist = grid_icond2_runs_scaled[r_hist, :, nearest, :]    # (N_all, 48, I2)
    i2_curr = grid_icond2_runs_scaled[r_curr, :, nearest, :]    # (N_all, 48, I2)
    i2_full = np.concatenate([i2_hist, i2_curr], axis=1)        # (N_all, 96, I2)

    i2_grid_full = np.concatenate([
        grid_icond2_runs_scaled[r_hist],
        grid_icond2_runs_scaled[r_curr],
    ], axis=0)                                                   # (96, N_grid, I2)

    e2_full = station_ecmwf_nwp_scaled[t_hist_abs:t_run_abs + H_fore, :, :][:, all_global, :]
    e2_full = e2_full.transpose(1, 0, 2)                        # (N_all, 96, E2)
    e2_grid_full = ecmwf_nwp_scaled[t_hist_abs:t_run_abs + H_fore]   # (96, N_ecmwf, E2)

    meas_hist = station_meas_scaled[t_hist_abs:t_run_abs, :, :][:, all_global, :].copy()
    meas_hist[:, N_obs:, :] = 0.0

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
) -> pd.DataFrame:
    """
    Four-pass LOO evaluation over all test run pairs.

    Returns a per-station metrics DataFrame with columns:
      station_id, r2, rmse, mae, skill_nwp, sample, pass
    """
    preds_acc: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    gt_acc:    dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    nwp_acc:   dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)

    mean_ws = float(meas_scaler.mean_[target_feat_idx])
    std_ws  = float(meas_scaler.std_[target_feat_idx] + meas_scaler.eps)

    def _to_phys(arr: np.ndarray) -> np.ndarray:
        return arr * std_ws + mean_ws

    common = dict(
        sampler=sampler,
        station_meas_scaled=meas_scaled,
        station_nearest_grid=station_nearest_grid,
        grid_icond2_runs_scaled=grid_icond2_runs_scaled,
        station_ecmwf_nwp_scaled=station_ecmwf_nwp_scaled,
        station_static=station_static,
        ecmwf_nwp_scaled=ecmwf_nwp_scaled,
        icond2_static=icond2_static,
        ecmwf_static=ecmwf_static,
        target_feat_idx=target_feat_idx,
        H_hist=H_hist,
        H_fore=H_fore,
    )

    def _nwp_ref(gidx: int, r_curr: int) -> np.ndarray:
        if ws_feat_idx_i2 is None:
            return np.full(H_fore, np.nan, dtype=np.float32)
        return grid_icond2_runs_raw[
            r_curr, :H_fore, station_nearest_grid[gidx], ws_feat_idx_i2
        ]

    def _run_loo(
        pool: list[int],
        target_gidx: int,
        pass_label: str,
        r_curr_: int,
        r_hist_: int,
        t_run_abs_: int,
    ) -> None:
        observer = [g for g in pool if g != target_gidx]
        data, mask, _ = build_eval_batch(
            **common,
            r_curr=r_curr_, r_hist=r_hist_, t_run_abs=t_run_abs_,
            target_global=[target_gidx],
            observer_global=observer,
        )
        preds = model(data.to(device), mask.to(device))   # (1, H_fore)
        preds_phys = _to_phys(preds.cpu().numpy()[0])      # (H_fore,)
        gt_phys = meas_raw[
            t_run_abs_:t_run_abs_ + H_fore, target_gidx, target_feat_idx
        ]
        key = (target_gidx, pass_label)
        preds_acc[key].append(preds_phys)
        gt_acc[key].append(gt_phys)
        nwp_acc[key].append(_nwp_ref(target_gidx, r_curr_))

    model.eval()
    with torch.no_grad():
        for step, (r_curr, r_hist, t_run_abs) in enumerate(test_run_pairs):
            if step % 10 == 0:
                logger.info("  Pair %d / %d", step + 1, len(test_run_pairs))

            # Pass A: all train observed → all val as targets
            if val_station_indices:
                data_a, mask_a, _ = build_eval_batch(
                    **common,
                    r_curr=r_curr, r_hist=r_hist, t_run_abs=t_run_abs,
                    target_global=val_station_indices,
                    observer_global=train_station_indices,
                )
                preds_a = _to_phys(
                    model(data_a.to(device), mask_a.to(device)).cpu().numpy()
                )  # (N_val, H_fore)
                gt_a = meas_raw[
                    t_run_abs:t_run_abs + H_fore, :, target_feat_idx
                ][:, val_station_indices].T  # (N_val, H_fore)
                for i, gidx in enumerate(val_station_indices):
                    preds_acc[(gidx, "A")].append(preds_a[i])
                    gt_acc[(gidx, "A")].append(gt_a[i])
                    nwp_acc[(gidx, "A")].append(_nwp_ref(gidx, r_curr))

            # Pass B: LOO over train, pool = train only
            for gidx in train_station_indices:
                _run_loo(train_station_indices, gidx, "B", r_curr, r_hist, t_run_abs)

            # Pass C: LOO over train, pool = train + val
            full_pool_train = train_station_indices + val_station_indices
            for gidx in train_station_indices:
                _run_loo(full_pool_train, gidx, "C", r_curr, r_hist, t_run_abs)

            # Pass D: LOO over val, pool = train + val
            full_pool_val = train_station_indices + val_station_indices
            for gidx in val_station_indices:
                _run_loo(full_pool_val, gidx, "D", r_curr, r_hist, t_run_abs)

    logger.info("Computing per-station metrics …")
    train_set = set(train_station_indices)
    records = []

    for (gidx, pass_label) in sorted(preds_acc.keys()):
        p_all = np.concatenate(preds_acc[(gidx, pass_label)])
        g_all = np.concatenate(gt_acc[(gidx, pass_label)])
        n_all = np.concatenate(nwp_acc[(gidx, pass_label)])

        valid = ~(np.isnan(p_all) | np.isnan(g_all))
        if valid.sum() < 2:
            logger.warning(
                "Station %s pass %s: too few valid samples (%d), skipping",
                all_ids[gidx], pass_label, int(valid.sum()),
            )
            continue

        p_v, g_v = p_all[valid], g_all[valid]
        r2   = float(r2_score(g_v, p_v))
        rmse = float(math.sqrt(mean_squared_error(g_v, p_v)))
        mae  = float(mean_absolute_error(g_v, p_v))

        valid_nwp = ~(np.isnan(n_all) | np.isnan(g_all))
        if valid_nwp.sum() >= 2:
            rmse_nwp  = float(math.sqrt(mean_squared_error(g_all[valid_nwp], n_all[valid_nwp])))
            skill_nwp = (1.0 - rmse / rmse_nwp) if rmse_nwp > 0 else float("nan")
        else:
            skill_nwp = float("nan")

        records.append({
            "station_id": all_ids[gidx],
            "r2":         r2,
            "rmse":       rmse,
            "mae":        mae,
            "skill_nwp":  skill_nwp,
            "sample":     "in" if gidx in train_set else "out",
            "pass":       pass_label,
        })

    return pd.DataFrame(records)
