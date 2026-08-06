#!/usr/bin/env python3
"""
evaluate_reference.py — Evaluate ICON-D2 and ECMWF NWP baselines on test data.

Uses the exact same val-pair loop as get_test_results_mtgnn.py so results are
directly comparable to trained model outputs.

Outputs:
  data/raw_preds/icon_d2_fold{N}_raw.parquet
  data/test_results/icon_d2_fold{N}.csv
  data/raw_preds/ecmwf_fold{N}_raw.parquet   (only if ecmwf_path is configured)
  data/test_results/ecmwf_fold{N}.csv        (only if ecmwf_path is configured)

Usage:
  cd /home/viktor/Work/forecasting_framework

  # Fold 0 — use any config that covers this fold's time window.
  # NWP config preferred so ECMWF gets loaded too.
  python geostatistics/evaluate_reference.py \\
      -c configs/mtgnn/config_wind_mtgnn_nwp_fold1.yaml --fold-idx 0

  # BASE config (no ECMWF path) → produces only icon_d2 outputs:
  python geostatistics/evaluate_reference.py \\
      -c configs/mtgnn/config_wind_mtgnn_fold1.yaml --fold-idx 0
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from geostatistics.train_stgnn2 import (
    load_yaml,
    load_station_measurements,
    load_station_metadata,
    load_icond2_ml_runs,
    load_ecmwf_parquet_at_stations_and_grid,
    load_interpol_imputation,
    apply_interpol_imputation,
    load_knn_imputation,
    apply_knn_imputation,
)
from geostatistics.train_dcrnn import encode_circular_measurements, apply_dir_encoding
from geostatistics.stgnn.utils.normalization import StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_reference")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _station_metrics(
    preds_acc: list[list],
    gts_acc:   list[list],
    nwp_acc:   list[list],
    pers_acc:  list[list],
    val_ids:   list[str],
) -> pd.DataFrame:
    rows = []
    for i, sid in enumerate(val_ids):
        if not preds_acc[i]:
            continue
        p  = np.concatenate(preds_acc[i])
        g  = np.concatenate(gts_acc[i])
        n  = np.concatenate(nwp_acc[i])
        ps = np.concatenate(pers_acc[i])

        ok = ~(np.isnan(p) | np.isnan(g))
        if ok.sum() < 2:
            continue
        p_v, g_v = p[ok], g[ok]

        rmse = float(math.sqrt(mean_squared_error(g_v, p_v)))
        mae  = float(mean_absolute_error(g_v, p_v))
        r2   = float(r2_score(g_v, p_v))

        ok_p = ~(np.isnan(ps) | np.isnan(g))
        pers_rmse = (float(math.sqrt(mean_squared_error(g[ok_p], ps[ok_p])))
                     if ok_p.sum() >= 2 else np.nan)

        ok_n = ~(np.isnan(n) | np.isnan(g))
        nwp_rmse  = (float(math.sqrt(mean_squared_error(g[ok_n], n[ok_n])))
                     if ok_n.sum() >= 2 else np.nan)

        skill     = (1.0 - rmse / pers_rmse) if (not np.isnan(pers_rmse) and pers_rmse > 0) else np.nan
        skill_nwp = (1.0 - rmse / nwp_rmse)  if (not np.isnan(nwp_rmse)  and nwp_rmse  > 0) else np.nan

        rows.append(dict(
            station_id=sid, mae=mae, rmse=rmse, r2=r2,
            skill=skill, skill_nwp=skill_nwp, n_samples=int(ok.sum()),
        ))
    return pd.DataFrame(rows)


def _save(station_df: pd.DataFrame, raw_df: pd.DataFrame, stem: str) -> None:
    out_dir = Path("data/test_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    station_df.to_csv(csv_path, index=False)
    logger.info("CSV     → %s  (%d stations)", csv_path, len(station_df))

    raw_dir  = Path("data/raw_preds")
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq_path  = raw_dir / f"{stem}_raw.parquet"
    raw_df.to_parquet(pq_path, index=False)
    logger.info("Parquet → %s  (%d rows)", pq_path, len(raw_df))


# ---------------------------------------------------------------------------
# Core evaluation loop (no model — pure NWP)
# ---------------------------------------------------------------------------

def evaluate_nwp_baselines(
    val_run_pairs:     list[tuple[int, int, int]],
    val_indices:       np.ndarray,          # indices into all_ids (train+val)
    val_ids:           list[str],
    meas_raw:          np.ndarray,          # (T, N, M) physical
    grid_icond2_runs:  np.ndarray,          # (R, n_leads, N_grid_i2, I2) physical
    station_ecmwf_nwp: np.ndarray | None,   # (T, N_all, E2) physical — already at each
                                             # station's own geodesically-nearest grid point
    target_feat_idx:   int,
    nwp_ws_feat_idx:   int,
    ecmwf_ws_feat_idx: int,
    nearest_i2:        np.ndarray,          # (N_val,) nearest ICON-D2 grid idx per val station
                                             # (geodesic — from load_icond2_ml_runs itself)
    timestamps:        pd.DatetimeIndex,
    F_h:               int,
):
    N_val     = len(val_ids)
    has_ecmwf = station_ecmwf_nwp is not None

    i2_preds = [[] for _ in range(N_val)]
    i2_gts   = [[] for _ in range(N_val)]
    i2_pers  = [[] for _ in range(N_val)]
    i2_recs  = []

    e2_preds = [[] for _ in range(N_val)]
    e2_gts   = [[] for _ in range(N_val)]
    e2_nwp   = [[] for _ in range(N_val)]   # ICON-D2 as reference for ECMWF skill_nwp
    e2_pers  = [[] for _ in range(N_val)]
    e2_recs  = []

    for r_curr, _r_hist, t_run_abs in val_run_pairs:
        run_ts = timestamps[t_run_abs - 1]

        # Ground truth: (N_val, F_h)
        gt_phys = meas_raw[t_run_abs : t_run_abs + F_h, val_indices, target_feat_idx].T

        # ICON-D2 nearest grid point forecast: (N_val, F_h)
        nwp_slice = grid_icond2_runs[r_curr, :F_h, :, nwp_ws_feat_idx]   # (F_h, N_grid_i2)
        i2_fc     = nwp_slice[:, nearest_i2].T.astype(np.float32)         # (N_val, F_h)

        # Persistence baseline: last observed value before run time
        pers_vals = meas_raw[t_run_abs - 1, val_indices, target_feat_idx]  # (N_val,)
        pers_fc   = np.repeat(pers_vals[:, None], F_h, axis=1).astype(np.float32)

        # ECMWF at each station's own (geodesically) nearest grid point — already
        # resolved by load_ecmwf_parquet_at_stations_and_grid; index directly via
        # val_indices exactly like meas_raw, no re-derived nearest-neighbour here.
        if has_ecmwf and t_run_abs + F_h <= station_ecmwf_nwp.shape[0]:
            e2_slice = station_ecmwf_nwp[t_run_abs : t_run_abs + F_h, val_indices, ecmwf_ws_feat_idx]  # (F_h, N_val)
            e2_fc    = e2_slice.T.astype(np.float32)         # (N_val, F_h)
        else:
            e2_fc = None

        for i in range(N_val):
            i2_preds[i].append(i2_fc[i])
            i2_gts[i].append(gt_phys[i])
            i2_pers[i].append(pers_fc[i])

            sid = val_ids[i]
            for h in range(F_h):
                i2_recs.append({
                    "station_id": sid,
                    "run_time":   run_ts,
                    "valid_time": run_ts + pd.Timedelta(hours=h + 1),
                    "horizon":    h + 1,
                    "pred":       float(i2_fc[i, h]),
                    "gt":         float(gt_phys[i, h]),
                    "nwp_ref":    float(i2_fc[i, h]),   # ICON-D2 is its own reference
                    "pers_ref":   float(pers_fc[i, h]),
                })

            if e2_fc is not None:
                e2_preds[i].append(e2_fc[i])
                e2_gts[i].append(gt_phys[i])
                e2_nwp[i].append(i2_fc[i])
                e2_pers[i].append(pers_fc[i])
                for h in range(F_h):
                    e2_recs.append({
                        "station_id": sid,
                        "run_time":   run_ts,
                        "valid_time": run_ts + pd.Timedelta(hours=h + 1),
                        "horizon":    h + 1,
                        "pred":       float(e2_fc[i, h]),
                        "gt":         float(gt_phys[i, h]),
                        "nwp_ref":    float(i2_fc[i, h]),
                        "pers_ref":   float(pers_fc[i, h]),
                    })

    # ICON-D2: skill_nwp is NaN (ICON-D2 is its own reference → ratio = 1 → skill = 0)
    i2_nwp = [[np.full(F_h, np.nan)] * len(p) for p in i2_preds]  # dummy
    i2_station = _station_metrics(i2_preds, i2_gts, i2_nwp, i2_pers, val_ids)

    if e2_recs:
        e2_station = _station_metrics(e2_preds, e2_gts, e2_nwp, e2_pers, val_ids)
        e2_raw     = pd.DataFrame(e2_recs)
    else:
        e2_station = None
        e2_raw     = None

    return i2_station, pd.DataFrame(i2_recs), e2_station, e2_raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ICON-D2 and ECMWF NWP baselines on the test fold."
    )
    parser.add_argument("-c", "--config",    required=True,
                        help="Path to YAML config (any fold config; NWP config preferred for ECMWF)")
    parser.add_argument("--fold-idx",        type=int, required=True,
                        help="Notebook fold index (0/1/2) used in output filenames")
    parser.add_argument("--test-mode",       action="store_true",
                        help="Use test_files as val set (same as in get_test_results_mtgnn.py)")
    parser.add_argument("--ecmwf-features",  default=None,
                        help="Comma-separated ECMWF features override (default: from config or wind_speed_10m)")
    args = parser.parse_args()

    cfg      = load_yaml(args.config)
    data_cfg = cfg["data"]
    mcfg     = cfg.get("mtgnn", cfg.get("dcrnn", {}))

    # ── Feature / dimension config ──────────────────────────────────────────
    icond2_features  = mcfg.get("icond2_features") or ["wind_speed_10m"]
    i2_mode          = mcfg.get("icond2_feature_mode", "absolute")
    e2_mode          = mcfg.get("ecmwf_feature_mode", "absolute")
    measurement_cols = list(mcfg.get("measurement_features") or ["wind_speed", "wind_direction"])
    target_col       = mcfg.get("target_col", "wind_speed")
    run_hours        = tuple(mcfg.get("icond2_run_hours", [6, 9, 12, 15]))
    next_n_icond2    = mcfg.get("next_n_icond2", 4)
    n_workers        = mcfg.get("n_workers", 8)
    nwp_path         = data_cfg.get("nwp_path")
    data_path        = data_cfg["path"]
    H   = mcfg.get("history_length",   48)
    F_h = mcfg.get("forecast_horizon", 48)
    freq   = data_cfg.get("freq", "1h")
    freq_h = {"1h": 1.0, "1H": 1.0, "30min": 0.5, "30T": 0.5}.get(freq, 1.0)

    ecmwf_features_load = (
        args.ecmwf_features.split(",") if args.ecmwf_features
        else (mcfg.get("ecmwf_features") or ["wind_speed_10m"])
    )

    # ── Station IDs ─────────────────────────────────────────────────────────
    if args.test_mode:
        train_ids = [str(s) for s in data_cfg["files"]] + [str(s) for s in data_cfg["val_files"]]
        val_ids   = [str(s) for s in data_cfg["test_files"]]
    else:
        train_ids = [str(s) for s in data_cfg["files"]]
        val_ids   = [str(s) for s in data_cfg["val_files"]]
    all_ids = train_ids + val_ids
    N_train, N_val = len(train_ids), len(val_ids)
    val_indices = np.arange(N_train, N_train + N_val)
    logger.info("Train: %d  Val: %d", N_train, N_val)

    # ── Measurements ─────────────────────────────────────────────────────────
    test_end   = data_cfg.get("test_end")
    run_cutoff = pd.Timestamp(test_end, tz="UTC") if test_end else None

    logger.info("Loading station measurements …")
    meas_raw, timestamps = load_station_measurements(
        data_path, all_ids, cols=measurement_cols, freq=freq
    )
    if run_cutoff is not None:
        cut_idx    = int(np.searchsorted(timestamps, run_cutoff + pd.Timedelta(days=2), side="right"))
        meas_raw   = meas_raw[:cut_idx]
        timestamps = timestamps[:cut_idx]
    T = len(timestamps)
    logger.info("T=%d  (%s … %s)", T, timestamps[0], timestamps[-1])

    # ── Imputation ──────────────────────────────────────────────────────────
    interpol_path = data_cfg.get("interpol_path")
    if interpol_path:
        rk_pred  = load_interpol_imputation(interpol_path, all_ids, timestamps)
        meas_raw = apply_interpol_imputation(meas_raw, rk_pred, measurement_cols, target_col)

    knnimputer_path = data_cfg.get("knnimputer_path")
    if knnimputer_path:
        for col in measurement_cols:
            feat_idx = measurement_cols.index(col)
            if not np.isnan(meas_raw[:, :, feat_idx]).any():
                continue
            knn_arr  = load_knn_imputation(knnimputer_path, col, all_ids, timestamps, freq=freq)
            meas_raw = apply_knn_imputation(meas_raw, knn_arr, measurement_cols, col)

    _meas_nan_any = np.isnan(meas_raw).any(axis=(1, 2))
    meas_raw, measurement_cols = encode_circular_measurements(meas_raw, measurement_cols)

    # ── Temporal split ───────────────────────────────────────────────────────
    # Mirrors get_test_results_dcrnn.py:284-306 — without a val_start/eval_cutoff
    # boundary, a dev-mode run (val_ids = val_files) was always scored over the
    # held-out test period, because test_start was the only boundary this
    # script knew about (bug: dev-mode CSVs covered 2025-08-01..2026-04-01
    # instead of the val window, while the matching model raw_preds covered
    # 2024-08-01..2025-08-02).
    test_start = data_cfg.get("test_start")
    val_start  = data_cfg.get("val_start")
    boundary   = test_start if (args.test_mode or not val_start) else val_start
    if boundary:
        split_t = int(np.searchsorted(timestamps, pd.Timestamp(boundary, tz="UTC"), side="left"))
    else:
        split_t = int(T * (1 - data_cfg.get("val_frac", 0.2)))
    split_time = timestamps[split_t]

    # Dev-mode upper bound: everything from test_start on is held-out test
    # material and must not leak into a --test-mode-less (dev/val) eval run.
    if val_start and test_start and not args.test_mode:
        _vc = int(np.searchsorted(timestamps, pd.Timestamp(test_start, tz="UTC"), side="left"))
        eval_cutoff = timestamps[_vc] if _vc < T else None
    else:
        eval_cutoff = None

    logger.info(
        "═══ EVAL WINDOW: %s … %s   mode=%s   stations=%d ═══",
        split_time,
        eval_cutoff if eval_cutoff is not None else "end",
        "TEST" if args.test_mode else "DEV",
        N_val,
    )

    # ── Station metadata ─────────────────────────────────────────────────────
    meta_path = data_cfg.get("stations_master")
    lats, lons, alts = load_station_metadata(data_path, all_ids, meta_path=meta_path)
    station_coords   = np.stack([lats, lons], axis=1)

    # ── ICON-D2 ─────────────────────────────────────────────────────────────
    logger.info("Loading ICON-D2 runs …")
    run_times, icond2_coords, grid_icond2_runs, station_nearest_i2_all = load_icond2_ml_runs(
        nwp_path=nwp_path, station_ids=all_ids, station_coords=station_coords,
        features=icond2_features, run_hours=run_hours, next_n_grid=next_n_icond2,
        n_workers=n_workers, cutoff=run_cutoff,
    )
    if i2_mode == "dir_in_deg":
        grid_icond2_runs, icond2_features = apply_dir_encoding(grid_icond2_runs, icond2_features)
    I2      = len(icond2_features)
    n_leads = grid_icond2_runs.shape[1]
    R       = len(run_times)
    logger.info("ICON-D2: R=%d  N_grid=%d  I2=%d  n_leads=%d", R, len(icond2_coords), I2, n_leads)

    nwp_ws_feat_idx = next(
        (i for i, f in enumerate(icond2_features) if f == "wind_speed_10m"),
        next((i for i, f in enumerate(icond2_features) if "wind_speed" in f), 0),
    )
    logger.info("ICON-D2 wind_speed feature idx: %d (%s)", nwp_ws_feat_idx, icond2_features[nwp_ws_feat_idx])

    # Nearest ICON-D2 grid point per val station — reuse load_icond2_ml_runs' OWN
    # geodesic assignment (pyproj Geod, same as evaluation.py's model path via
    # station_nearest_grid), instead of recomputing it with a plain cKDTree on raw
    # (lat, lon) degrees. A Euclidean degree distance does not shrink longitude by
    # cos(latitude), so at ~54°N it can rank a grid point that is genuinely farther
    # away (geodesic) as "nearer". Found via station 05142 (fold 0): the old
    # cKDTree path picked a point 1519 m away over the true 1500 m nearest
    # neighbour, moving that station's ICON-D2 RMSE by +0.775 and the fold's
    # station-mean by +0.015 — this is what caused the D4 cross-check mismatch.
    nearest_i2 = station_nearest_i2_all[N_train:]   # (N_val,) — same order as val_ids

    # ── ECMWF (optional) ─────────────────────────────────────────────────────
    ecmwf_path        = data_cfg.get("ecmwf_path")
    station_ecmwf_nwp = None   # (T, N_all, E2) physical
    ecmwf_ws_feat_idx = 0

    if ecmwf_path and os.path.exists(ecmwf_path):
        logger.info("Loading ECMWF from %s …", ecmwf_path)
        station_ecmwf_nwp, _, _, _ = load_ecmwf_parquet_at_stations_and_grid(
            parquet_path=ecmwf_path,
            station_lats=lats,
            station_lons=lons,
            features=ecmwf_features_load,
            timestamps=timestamps,
            next_n_grid_per_station=1,
        )
        # station_ecmwf_nwp is already resolved to each station's own nearest grid
        # point via pyproj geodesic distance INSIDE the loader (station_nearest
        # list, train_stgnn2.py). Use it directly — do not re-derive a "nearest"
        # index from the deduplicated grid array with a Euclidean lat/lon cKDTree;
        # that is exactly the bug just fixed for ICON-D2 above.
        if e2_mode == "dir_in_deg":
            station_ecmwf_nwp, ecmwf_features_load = apply_dir_encoding(station_ecmwf_nwp, ecmwf_features_load)
        E2 = station_ecmwf_nwp.shape[2]
        ecmwf_ws_feat_idx = next(
            (i for i, f in enumerate(ecmwf_features_load) if f == "wind_speed_10m"),
            next((i for i, f in enumerate(ecmwf_features_load) if "wind_speed" in f), 0),
        )
        logger.info("ECMWF: E2=%d  ws idx=%d (%s)", E2, ecmwf_ws_feat_idx, ecmwf_features_load[ecmwf_ws_feat_idx])
    elif ecmwf_path:
        logger.warning("ecmwf_path %s not found — ECMWF skipped", ecmwf_path)
    else:
        logger.info("No ecmwf_path in config — producing ICON-D2 only")

    # ── Val run pairs ────────────────────────────────────────────────────────
    logger.info("Building val run pairs …")
    ts_lookup = pd.Series(np.arange(T), index=timestamps)
    val_run_pairs: list[tuple[int, int, int]] = []

    for r_curr in range(R):
        t_run = run_times[r_curr]
        if t_run < split_time:
            continue
        if eval_cutoff is not None and t_run >= eval_cutoff:
            continue
        if t_run not in ts_lookup.index:
            continue
        # t_run_abs zeigt auf den ERSTEN PROGNOSESCHRITT (t_run + 1h), nicht auf
        # die Laufzeit: ICON-D2 liefert Leads 1..48, gueltig t_run+1 .. t_run+48.
        # Alle Mess-, Ziel- und ECMWF-Slices haengen an diesem Index und sind damit
        # zeitgleich mit der NWP-Vorhersage (Bias-Correction-Setup).
        t_run_abs = int(ts_lookup[t_run]) + 1
        if t_run_abs < H or t_run_abs + F_h > T:
            continue
        t_hist_target = t_run - pd.Timedelta(hours=H * freq_h)
        diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
        r_hist  = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600:
            continue
        if _meas_nan_any[t_run_abs - H : t_run_abs + F_h].any():
            continue
        val_run_pairs.append((r_curr, r_hist, t_run_abs))

    logger.info(
        "═══ EVAL WINDOW CONFIRMED: %s … %s   mode=%s   stations=%d   run_pairs=%d ═══",
        split_time,
        eval_cutoff if eval_cutoff is not None else "end",
        "TEST" if args.test_mode else "DEV",
        N_val,
        len(val_run_pairs),
    )
    if not val_run_pairs:
        logger.error("No val run pairs — check val_start / test_start / test_end in config")
        sys.exit(1)

    target_feat_idx = measurement_cols.index(target_col)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    logger.info("Evaluating NWP baselines …")
    i2_station, i2_raw, e2_station, e2_raw = evaluate_nwp_baselines(
        val_run_pairs      = val_run_pairs,
        val_indices        = val_indices,
        val_ids            = val_ids,
        meas_raw           = meas_raw,
        grid_icond2_runs   = grid_icond2_runs,
        station_ecmwf_nwp  = station_ecmwf_nwp,
        target_feat_idx    = target_feat_idx,
        nwp_ws_feat_idx    = nwp_ws_feat_idx,
        ecmwf_ws_feat_idx  = ecmwf_ws_feat_idx,
        nearest_i2         = nearest_i2,
        timestamps         = timestamps,
        F_h                = F_h,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    n = args.fold_idx
    # Test-Modus → eigener Stem (icon_d2_test_fold{n}), damit die Val-Referenz
    # nicht überschrieben wird und der Notebook-Split-Loader sie als split=test erkennt.
    sfx = "_test" if args.test_mode else ""
    _save(i2_station, i2_raw, stem=f"icon_d2{sfx}_fold{n}")

    if e2_station is not None and not e2_station.empty:
        _save(e2_station, e2_raw, stem=f"ecmwf{sfx}_fold{n}")
    else:
        logger.info("No ECMWF data → ecmwf%s_fold%d not written", sfx, n)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("─── ICON-D2 summary (fold %d) ───────────────────────────────", n)
    for col in ["rmse", "mae", "r2", "skill"]:
        if col in i2_station.columns:
            logger.info("  %-12s  mean=%.4f  std=%.4f",
                        col, i2_station[col].mean(), i2_station[col].std())

    if e2_station is not None and not e2_station.empty:
        logger.info("─── ECMWF summary (fold %d) ─────────────────────────────────", n)
        for col in ["rmse", "mae", "r2", "skill", "skill_nwp"]:
            if col in e2_station.columns:
                logger.info("  %-12s  mean=%.4f  std=%.4f",
                            col, e2_station[col].mean(), e2_station[col].std())


if __name__ == "__main__":
    main()
