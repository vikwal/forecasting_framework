#!/usr/bin/env python3
"""
get_test_results_mtgnn.py — Evaluate trained MTGNN models on test data.

Pendant to get_test_results_dcrnn.py for the homogeneous-graph MTGNN architecture.
Loads a trained checkpoint, runs inference on the test/val period, and writes:
  • data/test_results/test_results_{model_stem}.csv  — per-station aggregate metrics
  • data/raw_preds/{model_stem}_raw.parquet          — per-prediction raw data

Usage
-----
    python geostatistics/get_test_results_mtgnn.py \\
        -m wind_mtgnn_fold0 \\
        -c configs/config_wind_mtgnn_fold0.yaml \\
        --hpo-study auto
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

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
from geostatistics.homo_sampler import HomoSampler, evaluate_homo_model
from geostatistics.mtgnn import MTGNNModel
from geostatistics.stgnn.utils.normalization import StandardScaler


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(model_name: str) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"eval_mtgnn_{model_name}.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("eval_mtgnn")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


logger = logging.getLogger("eval_mtgnn")


# ---------------------------------------------------------------------------
# Model file resolution
# ---------------------------------------------------------------------------

def resolve_model_file(model_name: str, models_dir: Path) -> Path:
    if model_name.endswith(".pt") and Path(model_name).exists():
        return Path(model_name)
    candidates = [f for f in sorted(models_dir.glob("*.pt")) if model_name in f.stem]
    if len(candidates) == 0:
        logger.error("No model file found in %s whose filename contains %r.", models_dir, model_name)
        sys.exit(1)
    if len(candidates) > 1:
        logger.error(
            "Model name %r is ambiguous — %d files match:\n  %s",
            model_name, len(candidates), "\n  ".join(f.name for f in candidates)
        )
        sys.exit(1)
    return candidates[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MTGNN model on test data")
    parser.add_argument("-m", "--model-name", required=True, help="Substring of the model filename or path to .pt")
    parser.add_argument("-c", "--config",     required=True, help="Path to the YAML configuration file")
    parser.add_argument("--hpo-study",  default=None, help="Load best params from Optuna (auto or path)")
    parser.add_argument("--models-dir", default="models", help="Directory to search for models")
    parser.add_argument("--out-suffix", default="", help="Suffix for output files")
    parser.add_argument("--raw-out-name", default=None,
                        help="Override parquet filename stem (without _raw.parquet); "
                             "e.g. mtgnn_wind_mtgnn_fold0 → data/raw_preds/mtgnn_wind_mtgnn_fold0_raw.parquet")
    parser.add_argument(
        "--test-mode", action="store_true",
        help=(
            "Final-evaluation mode: train_ids = files + val_files, "
            "val_ids = test_files. Must match --test-mode used in train_mtgnn.py."
        ),
    )
    args = parser.parse_args()

    # ── Resolve model file ──────────────────────────────────────────────────
    model_path = resolve_model_file(args.model_name, Path(args.models_dir))
    model_stem = model_path.stem
    _setup_logging(model_stem)
    logger.info("=== MTGNN Evaluation ===")
    logger.info("Model: %s", model_path.name)

    # ── Load configuration ──────────────────────────────────────────────────
    cfg      = load_yaml(args.config)
    data_cfg = cfg["data"]
    mcfg     = cfg.get("mtgnn", {})

    # ── HPO overrides ───────────────────────────────────────────────────────
    if args.hpo_study:
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is not installed")
        config_stem = Path(args.config).stem.replace("config_", "")
        hpo_stem    = re.sub(r'_fold\d+$', '', config_stem)
        freq        = data_cfg.get("freq", "1h")
        F_h_tmp     = mcfg.get("forecast_horizon", 48)
        study_name  = f"cl_m-mtgnn_out-{F_h_tmp}_freq-{freq}_{hpo_stem}"

        storage_url = os.environ.get("OPTUNA_STORAGE")
        if storage_url:
            storage = storage_url
            logger.info("Loading HPO overrides from PostgreSQL (OPTUNA_STORAGE) …")
        else:
            db_path = args.hpo_study if args.hpo_study != "auto" else f"studies/hpo_mtgnn_{hpo_stem}.db"
            storage = f"sqlite:///{db_path}"
            logger.info("Loading HPO overrides from SQLite: %s", db_path)
        study = optuna.load_study(study_name=study_name, storage=storage)
        logger.info("Overriding with best params from trial #%d", study.best_trial.number)
        mcfg.update(study.best_params)

    # ── Feature / dimension config ──────────────────────────────────────────
    icond2_features  = mcfg.get("icond2_features") or []
    i2_mode          = mcfg.get("icond2_feature_mode", "absolute")
    e2_mode          = mcfg.get("ecmwf_feature_mode",  "absolute")
    measurement_cols = mcfg.get("measurement_features") or []
    target_col       = mcfg.get("target_col", "wind_speed")
    if target_col not in measurement_cols:
        raise ValueError(f"target_col '{target_col}' must be in measurement_features")

    run_hours     = tuple(mcfg.get("icond2_run_hours", [6, 9, 12, 15]))
    next_n_icond2 = mcfg.get("next_n_icond2", 4)
    n_workers     = mcfg.get("n_workers", 8)
    nwp_path      = data_cfg.get("nwp_path")
    data_path     = data_cfg["path"]
    use_case      = data_cfg.get("use_case", "wind")

    H   = mcfg.get("history_length",   48)
    F_h = mcfg.get("forecast_horizon", 48)

    freq   = data_cfg.get("freq", "1h")
    _freq_h_map = {"1h": 1.0, "1H": 1.0, "30min": 0.5, "30T": 0.5, "15min": 0.25, "15T": 0.25}
    freq_h = _freq_h_map.get(freq, 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Station IDs ─────────────────────────────────────────────────────────
    if args.test_mode:
        if not data_cfg.get("test_files"):
            raise ValueError("--test-mode requires 'test_files' in data config.")
        train_ids = [str(s) for s in data_cfg["files"]] + [str(s) for s in data_cfg["val_files"]]
        val_ids   = [str(s) for s in data_cfg["test_files"]]
        logger.info("Mode: FINAL EVAL — train: %d (files+val_files)  test: %d (test_files)",
                    len(train_ids), len(val_ids))
    else:
        train_ids = [str(s) for s in data_cfg["files"]]
        val_ids   = [str(s) for s in data_cfg["val_files"]]
        logger.info("Mode: DEVELOPMENT — train: %d (files)  val: %d (val_files)",
                    len(train_ids), len(val_ids))
    all_ids = train_ids + val_ids
    N_train, N_val = len(train_ids), len(val_ids)

    # ── Station measurements ─────────────────────────────────────────────────
    test_end   = data_cfg.get("test_end")
    run_cutoff = pd.Timestamp(test_end, tz="UTC") if test_end else None

    logger.info("Loading station measurements …")
    meas_raw, timestamps = load_station_measurements(data_path, all_ids, cols=measurement_cols, freq=freq)

    if run_cutoff is not None:
        meas_cutoff = run_cutoff + pd.Timedelta(days=2)
        cut_idx     = int(np.searchsorted(timestamps, meas_cutoff, side="right"))
        meas_raw    = meas_raw[:cut_idx]
        timestamps  = timestamps[:cut_idx]
    T = len(timestamps)
    logger.info("Timestamps: %d  (%s … %s)", T, timestamps[0], timestamps[-1])

    # ── Imputation ──────────────────────────────────────────────────────────
    interpol_path = data_cfg.get("interpol_path")
    if interpol_path:
        rk_pred = load_interpol_imputation(interpol_path, all_ids, timestamps)
        meas_raw = apply_interpol_imputation(meas_raw, rk_pred, measurement_cols, target_col)

    knnimputer_path = data_cfg.get("knnimputer_path")
    if knnimputer_path:
        for col in measurement_cols:
            if int(np.isnan(meas_raw[:, :, measurement_cols.index(col)]).sum()) == 0:
                continue
            knn_arr = load_knn_imputation(knnimputer_path, col, all_ids, timestamps, freq=freq)
            meas_raw = apply_knn_imputation(meas_raw, knn_arr, measurement_cols, col)

    _meas_nan_any = np.isnan(meas_raw).any(axis=(1, 2))
    meas_raw, measurement_cols = encode_circular_measurements(meas_raw, measurement_cols)

    # ── Temporal split ───────────────────────────────────────────────────────
    test_start = data_cfg.get("test_start")
    if test_start:
        split_t = int(np.searchsorted(timestamps, pd.Timestamp(test_start, tz="UTC"), side="left"))
    else:
        split_t = int(T * (1 - data_cfg.get("val_frac", 0.2)))
    split_time = timestamps[split_t]
    logger.info("Test period starts at %s", split_time)

    # ── Station metadata ─────────────────────────────────────────────────────
    meta_path = data_cfg.get("stations_master")
    lats, lons, alts = load_station_metadata(data_path, all_ids, meta_path=meta_path)
    station_coords   = np.stack([lats, lons], axis=1)

    # ── ICON-D2 runs ─────────────────────────────────────────────────────────
    if use_case == "solar":
        from geostatistics.solar_preprocessing import load_solar_sl_runs
        run_times, icond2_coords, grid_icond2_runs, _ = load_solar_sl_runs(
            nwp_path=nwp_path, station_ids=all_ids, station_coords=station_coords,
            features=icond2_features, run_hours=run_hours, next_n_grid=next_n_icond2,
            n_workers=n_workers, cutoff=run_cutoff, freq_h=freq_h,
        )
    else:
        run_times, icond2_coords, grid_icond2_runs, _ = load_icond2_ml_runs(
            nwp_path=nwp_path, station_ids=all_ids, station_coords=station_coords,
            features=icond2_features, run_hours=run_hours, next_n_grid=next_n_icond2,
            n_workers=n_workers, cutoff=run_cutoff,
        )
    R       = len(run_times)
    if i2_mode == "dir_in_deg":
        grid_icond2_runs, icond2_features = apply_dir_encoding(grid_icond2_runs, icond2_features)
    I2      = len(icond2_features)
    n_leads = grid_icond2_runs.shape[1]
    logger.info("ICON-D2 grid nodes: %d  runs: %d  I2: %d", len(icond2_coords), R, I2)

    # ── ECMWF (optional) ─────────────────────────────────────────────────────
    next_n_ecmwf   = mcfg.get("next_n_ecmwf", 0)
    ecmwf_features = mcfg.get("ecmwf_features") or []
    nwp_nodes      = mcfg.get("nwp_nodes", False)
    aggregate_nwp  = False if nwp_nodes else mcfg.get("aggregate_nwp", True)
    grid_ecmwf_scaled: np.ndarray | None = None
    ecmwf_coords:      np.ndarray | None = None

    if next_n_ecmwf > 0 and ecmwf_features:
        ecmwf_path = data_cfg.get("ecmwf_path")
        if ecmwf_path and os.path.exists(ecmwf_path):
            logger.info("Loading ECMWF NWP (%d features, k=%d) …", len(ecmwf_features), next_n_ecmwf)
            _, ecmwf_coords, grid_ecmwf_runs, _ = load_ecmwf_parquet_at_stations_and_grid(
                parquet_path=ecmwf_path, station_lats=lats, station_lons=lons,
                features=ecmwf_features, timestamps=timestamps,
                next_n_grid_per_station=next_n_ecmwf,
            )
            if e2_mode == "dir_in_deg":
                grid_ecmwf_runs, ecmwf_features = apply_dir_encoding(grid_ecmwf_runs, ecmwf_features)
            E2 = grid_ecmwf_runs.shape[2]
            e2_scaler = StandardScaler()
            e2_scaler.fit(grid_ecmwf_runs[:split_t].reshape(-1, E2))
            grid_ecmwf_scaled = e2_scaler.transform(
                grid_ecmwf_runs.reshape(-1, E2)
            ).reshape(T, len(ecmwf_coords), E2)
        else:
            logger.warning("next_n_ecmwf=%d but ecmwf_path not set or missing — ECMWF disabled", next_n_ecmwf)
            next_n_ecmwf = 0

    # ── Scalers ──────────────────────────────────────────────────────────────
    M_meas = len(measurement_cols)
    meas_scaler = StandardScaler()
    meas_scaler.fit(meas_raw[:split_t, :N_train].reshape(-1, M_meas))
    meas_scaled = meas_scaler.transform(meas_raw.reshape(-1, M_meas)).reshape(T, len(all_ids), M_meas)

    train_r_mask = run_times < split_time
    i2_scaler    = StandardScaler()
    i2_scaler.fit(grid_icond2_runs[train_r_mask].reshape(-1, I2))
    grid_icond2_scaled = i2_scaler.transform(
        grid_icond2_runs.reshape(-1, I2)
    ).reshape(R, n_leads, len(icond2_coords), I2)

    target_feat_idx = measurement_cols.index(target_col)
    target_scale    = float(meas_scaler.std_[target_feat_idx] + meas_scaler.eps)
    target_mean     = float(meas_scaler.mean_[target_feat_idx])

    nwp_ws_feat_idx = next(
        (i for i, f in enumerate(icond2_features) if f == "wind_speed_10m"),
        next((i for i, f in enumerate(icond2_features) if "wind_speed" in f), 0)
    )

    # ── Run pairs (test / val window only) ───────────────────────────────────
    logger.info("Identifying test run pairs …")
    ts_lookup      = pd.Series(np.arange(T), index=timestamps)
    val_run_pairs: list[tuple[int, int, int]] = []

    for r_curr in range(R):
        t_run = run_times[r_curr]
        if t_run < split_time:
            continue
        if t_run not in ts_lookup.index:
            continue
        t_run_abs = int(ts_lookup[t_run])
        if t_run_abs < H or t_run_abs + F_h > T:
            continue

        t_hist_target = t_run - pd.Timedelta(hours=H * freq_h)
        diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
        r_hist  = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600:
            continue
        if _meas_nan_any[t_run_abs - H:t_run_abs + F_h].any():
            continue
        val_run_pairs.append((r_curr, r_hist, t_run_abs))

    if not val_run_pairs:
        logger.error("No test run pairs found!")
        sys.exit(1)
    logger.info("Test run pairs: %d", len(val_run_pairs))

    # ── HomoSampler ─────────────────────────────────────────────────────────
    sampler = HomoSampler(
        meas_scaled           = meas_scaled,
        grid_icond2_scaled    = grid_icond2_scaled,
        train_run_pairs       = [],
        val_run_pairs         = val_run_pairs,
        train_station_indices = list(range(N_train)),
        val_station_indices   = list(range(N_train, N_train + N_val)),
        lats                  = lats,
        lons                  = lons,
        alts                  = alts,
        icond2_coords         = icond2_coords,
        history_length        = H,
        forecast_horizon      = F_h,
        target_feat_idx       = target_feat_idx,
        k_nwp                 = mcfg.get("next_n_icond2", next_n_icond2),
        min_target_stations   = mcfg.get("min_target_stations", 1),
        max_target_stations   = mcfg.get("max_target_stations", 10),
        max_neighbor_stations = mcfg.get("max_neighbor_stations", 60),
        next_n_neighbors      = mcfg.get("next_n_neighbors", None),
        hist_wind_available   = mcfg.get("hist_wind_available", False),
        grid_ecmwf_scaled     = grid_ecmwf_scaled,
        ecmwf_coords          = ecmwf_coords,
        k_ecmwf               = next_n_ecmwf,
        aggregate_nwp         = aggregate_nwp,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    M_meas_only    = M_meas
    nwp_out_dim    = mcfg.get("nwp_out_dim", 32) if nwp_nodes else 0
    nwp_heads      = mcfg.get("nwp_heads", 4)    if nwp_nodes else 4
    ecmwf_channels = (
        (sampler.in_channels - M_meas_only - mcfg.get("next_n_icond2", next_n_icond2) * I2)
        if nwp_nodes else 0
    )
    in_channels_model = (M_meas_only + nwp_out_dim + ecmwf_channels) if nwp_nodes else sampler.in_channels

    model = MTGNNModel(
        in_channels      = in_channels_model,
        static_dim       = 6,
        hidden           = mcfg.get("hidden", 64),
        n_layers         = mcfg.get("n_layers", 4),
        K_hop            = mcfg.get("K_hop", 2),
        beta             = mcfg.get("beta", 0.05),
        emb_dim          = mcfg.get("emb_dim", 64),
        graph_alpha      = mcfg.get("graph_alpha", 3.0),
        dropout          = mcfg.get("dropout", 0.1),
        history_length   = H,
        forecast_horizon = F_h,
        nwp_nodes        = nwp_nodes,
        nwp_feat_dim     = I2,
        k_nwp            = mcfg.get("next_n_icond2", next_n_icond2),
        nwp_out_dim      = nwp_out_dim,
        nwp_heads        = nwp_heads,
        M                = M_meas_only,
        topk_graph       = mcfg.get("topk_graph", None),
    )

    logger.info("Loading weights …")
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        sd = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(device)

    # ── Evaluation ───────────────────────────────────────────────────────────
    logger.info("Starting inference …")
    results_df, raw_df = evaluate_homo_model(
        model           = model,
        sampler         = sampler,
        device          = device,
        val_ids         = val_ids,
        meas_raw        = meas_raw,
        grid_icond2_runs= grid_icond2_runs,
        target_scale    = target_scale,
        target_mean     = target_mean,
        target_feat_idx = target_feat_idx,
        nwp_ws_feat_idx = nwp_ws_feat_idx,
        timestamps      = timestamps,
    )

    # ── Save per-station CSV ──────────────────────────────────────────────────
    out_dir  = Path("data/test_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"test_results_{model_stem}{args.out_suffix}.csv"
    out_path = out_dir / out_name
    results_df.to_csv(out_path, index=False)
    logger.info("Results saved → %s", out_path)

    # ── Save raw predictions parquet ──────────────────────────────────────────
    if not raw_df.empty:
        raw_dir  = Path("data/raw_preds")
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_stem = args.raw_out_name if args.raw_out_name else f"{model_stem}{args.out_suffix}"
        raw_path = raw_dir / f"{raw_stem}_raw.parquet"
        raw_df.to_parquet(raw_path, index=False)
        logger.info("Raw predictions saved → %s", raw_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    if not results_df.empty:
        cols = ["mae", "rmse", "r2", "skill", "skill_nwp"]
        tbl  = results_df.set_index("station_id")[cols]
        mean_row = tbl.mean().to_frame().T
        mean_row.index = ["MEAN"]
        tbl = pd.concat([tbl, mean_row])
        logger.info("Per-station evaluation:\n%s", tbl.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
