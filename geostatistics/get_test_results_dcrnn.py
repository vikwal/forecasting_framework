#!/usr/bin/env python3
"""
get_test_results_dcrnn.py — Evaluate trained DCRNN models on test data.

This script is a specialized version of get_test_results_stgnn2.py that
properly handles DCRNN-specific features like:
  - icond2_feature_mode / ecmwf_feature_mode (resolve_feature_mode)
  - HPO best params loading from Optuna (mirroring train_dcrnn.py)
  - Loading configuration from training results .pkl files

Usage
-----
    python geostatistics/get_test_results_dcrnn.py \
        -m wind_stgcn_dcrnn_v1 \
        -c configs/config_wind_stgcn.yaml \
        --hpo-study auto
"""
from __future__ import annotations

import argparse
import re
import logging
import os
import pickle
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

# Make repo root importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent.parent))

from geostatistics.train_stgnn2 import (
    load_yaml,
    load_station_measurements,
    load_station_metadata,
    load_icond2_ml_runs,
    load_ecmwf_parquet_at_stations_and_grid,
    load_nwp_elevations,
    load_interpol_imputation,
    apply_interpol_imputation,
    load_knn_imputation,
    apply_knn_imputation,
)
from geostatistics.train_dcrnn import resolve_feature_mode
from geostatistics.dcrnn import DCRNNConfig, DCRNN
from geostatistics.stgnn import HeterogeneousGraphBuilder
from geostatistics.stgnn.training.sampler import TrainingSampler
from geostatistics.stgnn.utils.normalization import StandardScaler
from geostatistics.evaluation import evaluate, find_ws_feat_idx


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(model_name: str) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"eval_dcrnn_{model_name}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("eval_dcrnn")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


logger = logging.getLogger("eval_dcrnn")


# ---------------------------------------------------------------------------
# Model file resolution
# ---------------------------------------------------------------------------

def resolve_model_file(model_name: str, models_dir: Path) -> Path:
    """Find a unique .pt file in *models_dir* whose stem contains *model_name*."""
    if model_name.endswith(".pt") and Path(model_name).exists():
        return Path(model_name)
    
    candidates = [f for f in sorted(models_dir.glob("*.pt")) if model_name in f.stem]
    if len(candidates) == 0:
        logger.error(
            "No model file found in %s whose filename contains %r.",
            models_dir, model_name
        )
        sys.exit(1)
    if len(candidates) > 1:
        logger.error(
            "Model name %r is ambiguous — %d files match:\n  %s",
            model_name, len(candidates), "\n  ".join(f.name for f in candidates)
        )
        sys.exit(1)
    return candidates[0]


def _load_state_dict(checkpoint: dict | object) -> dict:
    """Extract and clean the model state_dict from a checkpoint."""
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        sd = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        sd = checkpoint
    else:
        raise ValueError("Unrecognised checkpoint format")
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DCRNN model on test data")
    parser.add_argument("-m", "--model-name", required=True, help="Substring of the model filename or path to .pt")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--pkl", help="Optional: Path to the training results .pkl to load the exact config used")
    parser.add_argument("--hpo-study", default=None, help="Load best params from Optuna (auto or path)")
    parser.add_argument("--models-dir", default="models", help="Directory to search for models")
    parser.add_argument("--out-suffix", default="", help="Suffix for output CSV")
    parser.add_argument(
        "--test-mode", action="store_true",
        help=(
            "Final-evaluation mode: train_ids = files + val_files (153), "
            "val_ids = test_files (50). Must match the --test-mode used in train_dcrnn.py."
        ),
    )
    args = parser.parse_args()

    # ── Resolve model file ──────────────────────────────────────────────
    model_path = resolve_model_file(args.model_name, Path(args.models_dir))
    model_stem = model_path.stem
    _setup_logging(model_stem)
    logger.info("=== DCRNN Evaluation ===")
    logger.info("Model: %s", model_path.name)

    # ── Load configuration ──────────────────────────────────────────────
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    
    # If .pkl is provided, load config from there
    dcrnn_cfg = {}
    if args.pkl:
        logger.info("Loading config from training PKL: %s", args.pkl)
        with open(args.pkl, "rb") as f:
            pkl_data = pickle.load(f)
        dcrnn_cfg = pkl_data.get("config", {})
    else:
        dcrnn_cfg = cfg.get("dcrnn", {})

    # Optional: HPO overrides (mirroring train_dcrnn.py)
    if args.hpo_study:
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna not installed")
        
        config_stem = Path(args.config).stem.replace("config_", "")
        hpo_stem = re.sub(r'_fold\d+$', '', config_stem)
        freq = data_cfg.get("freq", "1h")
        H_fore_tmp = dcrnn_cfg.get("forecast_horizon", 48)
        study_name = f"cl_m-dcrnn_out-{H_fore_tmp}_freq-{freq}_{hpo_stem}"

        storage_url = os.environ.get("OPTUNA_STORAGE")
        if storage_url:
            storage = storage_url
            logger.info("Loading HPO overrides from PostgreSQL (OPTUNA_STORAGE) …")
        else:
            db_path = args.hpo_study if args.hpo_study != "auto" else f"studies/hpo_dcrnn_{hpo_stem}.db"
            storage = f"sqlite:///{db_path}"
            logger.info("Loading HPO overrides from SQLite: %s", db_path)
        study = optuna.load_study(study_name=study_name, storage=storage)
        logger.info("Overriding with best params from trial #%d", study.best_trial.number)
        dcrnn_cfg.update(study.best_params)
        if "nwp_heads" in dcrnn_cfg and "nwp_out_per_head" in dcrnn_cfg:
            dcrnn_cfg["nwp_out_dim"] = dcrnn_cfg.pop("nwp_out_per_head") * dcrnn_cfg["nwp_heads"]

    # ── Feature config (including resolve_feature_mode) ──────────────────
    icond2_features_all = dcrnn_cfg.get("icond2_features") or []
    ecmwf_features_all  = dcrnn_cfg.get("ecmwf_features")  or []
    i2_mode = dcrnn_cfg.get("icond2_feature_mode", "both")
    e2_mode = dcrnn_cfg.get("ecmwf_feature_mode",  "both")
    icond2_features = resolve_feature_mode(icond2_features_all, i2_mode)
    ecmwf_features  = resolve_feature_mode(ecmwf_features_all,  e2_mode)
    logger.info("Features: ICON-D2 mode=%s (%d), ECMWF mode=%s (%d)", 
                i2_mode, len(icond2_features), e2_mode, len(ecmwf_features))

    measurement_cols = dcrnn_cfg.get("measurement_features")
    target_col       = dcrnn_cfg.get("target_col")
    target_feat_idx  = measurement_cols.index(target_col)
    
    H_hist = dcrnn_cfg.get("history_length", 48)
    H_fore = dcrnn_cfg.get("forecast_horizon", 48)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Station IDs ──────────────────────────────────────────────────────
    if args.test_mode:
        if not data_cfg.get("test_files"):
            raise ValueError("--test-mode requires 'test_files' to be set in the data config.")
        train_ids = [str(s) for s in data_cfg["files"]] + [str(s) for s in data_cfg["val_files"]]
        val_ids   = [str(s) for s in data_cfg["test_files"]]
        logger.info("Mode: FINAL EVAL — train: %d (files+val_files)  test (zero-shot): %d (test_files)",
                    len(train_ids), len(val_ids))
    else:
        train_ids = [str(s) for s in data_cfg["files"]]
        val_ids   = [str(s) for s in data_cfg["val_files"]]
        logger.info("Mode: DEVELOPMENT — train: %d (files)  val (zero-shot): %d (val_files)",
                    len(train_ids), len(val_ids))
    all_ids   = train_ids + val_ids
    N_train   = len(train_ids)
    N_val     = len(val_ids)
    train_station_indices = list(range(N_train))
    val_station_indices   = list(range(N_train, N_train + N_val))

    # ── Load data ────────────────────────────────────────────────────────
    data_path = data_cfg["path"]
    nwp_path  = data_cfg.get("nwp_path")
    
    logger.info("Loading station measurements …")
    meas_raw, timestamps = load_station_measurements(data_path, all_ids, cols=measurement_cols)
    T = len(timestamps)

    # Imputation (if paths present)
    interpol_path = data_cfg.get("interpol_path")
    if interpol_path:
        rk_pred = load_interpol_imputation(interpol_path, all_ids, timestamps)
        meas_raw = apply_interpol_imputation(meas_raw, rk_pred, measurement_cols, target_col)
    
    knnimputer_path = data_cfg.get("knnimputer_path")
    if knnimputer_path and "wind_direction" in measurement_cols:
        knn_wd = load_knn_imputation(knnimputer_path, "wind_direction", all_ids, timestamps)
        meas_raw = apply_knn_imputation(meas_raw, knn_wd, measurement_cols, "wind_direction")

    # Temporal split
    test_start = data_cfg.get("test_start")
    test_end   = data_cfg.get("test_end")
    if test_start:
        ts_cutoff = pd.Timestamp(test_start, tz="UTC")
        split_t = int(np.searchsorted(timestamps, ts_cutoff, side="left"))
    else:
        split_t = int(T * (1 - data_cfg.get("val_frac", 0.2)))
    split_time = timestamps[split_t]
    run_cutoff = pd.Timestamp(test_end, tz="UTC") if test_end else None
    logger.info("Test period starts at %s", split_time)

    # Metadata
    lats, lons, alts = load_station_metadata(data_path, all_ids, meta_path=data_cfg.get("stations_master"))
    station_coords = np.stack([lats, lons], axis=1)

    # NWP Skill baseline index
    ws_feat_idx_i2 = find_ws_feat_idx(icond2_features)

    # ICON-D2 ML runs
    run_hours = tuple(dcrnn_cfg.get("icond2_run_hours", [6, 9, 12, 15]))
    logger.info("Loading ICON-D2 runs …")
    run_times, icond2_coords, grid_icond2_runs_raw, station_nearest_grid = \
        load_icond2_ml_runs(
            nwp_path=nwp_path, station_ids=all_ids, station_coords=station_coords,
            features=icond2_features, run_hours=run_hours, 
            next_n_grid=dcrnn_cfg.get("next_n_icond2", 4), cutoff=run_cutoff
        )
    R = len(run_times)
    N_igrid = len(icond2_coords)

    # ECMWF NWP
    ecmwf_parquet_file = os.path.join(data_cfg.get("ecmwf_path", "/mnt/lambda1/nvme1/ecmwf/parquet"), "ecmwf_wind_sl_full.parquet")
    E2 = len(ecmwf_features)
    if os.path.exists(ecmwf_parquet_file):
        station_ecmwf_nwp, ecmwf_coords, ecmwf_nwp, ecmwf_alts = \
            load_ecmwf_parquet_at_stations_and_grid(
                parquet_path=ecmwf_parquet_file, station_lats=lats, station_lons=lons,
                features=ecmwf_features, timestamps=timestamps,
                next_n_grid_per_station=dcrnn_cfg.get("next_n_ecmwf", 4)
            )
    else:
        logger.warning("ECMWF parquet not found — using zeros")
        station_ecmwf_nwp = np.zeros((T, len(all_ids), E2), dtype=np.float32)
        ecmwf_coords = np.zeros((1, 2), dtype=np.float32)
        ecmwf_nwp = np.zeros((T, 1, E2), dtype=np.float32)
        ecmwf_alts = np.zeros(1, dtype=np.float32)

    # NWP Altitudes
    if dcrnn_cfg.get("use_altitude_diff", False):
        weather_db_url = os.environ.get("WEATHER_DB_URL")
        ecmwf_db_url   = os.environ.get("ECMWF_WIND_SL_URL")
        if weather_db_url and ecmwf_db_url:
            icond2_alts, ecmwf_alts = load_nwp_elevations(
                weather_db_url=weather_db_url, ecmwf_db_url=ecmwf_db_url,
                icond2_coords=icond2_coords, ecmwf_coords=ecmwf_coords
            )
            icond2_alts += 10.0
            ecmwf_alts += 10.0
        else:
            icond2_alts = np.zeros(N_igrid, dtype=np.float32)
    else:
        icond2_alts = np.zeros(N_igrid, dtype=np.float32)

    # ── Scalers ─────────────────────────────────────────────────────────
    logger.info("Fitting scalers …")
    M_meas = len(measurement_cols)
    meas_scaler = StandardScaler()
    meas_scaler.fit(meas_raw[:split_t, :N_train].reshape(-1, M_meas))
    meas_scaled = meas_scaler.transform(meas_raw.reshape(-1, M_meas)).reshape(T, len(all_ids), M_meas)

    train_r_mask = run_times < split_time
    i2_scaler = StandardScaler()
    i2_scaler.fit(grid_icond2_runs_raw[train_r_mask].reshape(-1, len(icond2_features)))
    grid_icond2_runs_scaled = i2_scaler.transform(grid_icond2_runs_raw.reshape(-1, len(icond2_features))).reshape(R, 48, N_igrid, len(icond2_features))

    e2_scaler = StandardScaler()
    e2_scaler.fit(station_ecmwf_nwp[:split_t, :N_train].reshape(-1, E2))
    station_ecmwf_scaled = e2_scaler.transform(station_ecmwf_nwp.reshape(-1, E2)).reshape(T, len(all_ids), E2)
    ecmwf_nwp_scaled = e2_scaler.transform(ecmwf_nwp.reshape(-1, E2)).reshape(T, len(ecmwf_coords), E2)

    stat_scaler = StandardScaler()
    raw_static  = np.stack([lats, lons, alts], axis=1).astype(np.float32)
    stat_scaler.fit(raw_static[:N_train])
    station_static_scaled = stat_scaler.transform(raw_static)

    icond2_static_scaled = StandardScaler().fit_transform(np.concatenate([icond2_coords, icond2_alts[:, None]], axis=1)).astype(np.float32)
    ecmwf_static_scaled = StandardScaler().fit_transform(np.concatenate([ecmwf_coords, ecmwf_alts[:, None]], axis=1)).astype(np.float32)

    # ── Test run pairs ───────────────────────────────────────────────────
    logger.info("Identifying test run pairs …")
    ts_lookup = pd.Series(np.arange(T), index=timestamps)
    _meas_nan_any = np.isnan(meas_raw[:, :, 0]).any(axis=1)
    
    test_run_pairs: list[tuple[int, int, int]] = []
    for r_curr in range(R):
        t_run = run_times[r_curr]
        if t_run < split_time: continue
        if t_run not in ts_lookup.index: continue
        t_run_abs = int(ts_lookup[t_run])
        if t_run_abs < H_hist or t_run_abs + H_fore > T: continue
        
        t_hist_target = t_run - pd.Timedelta(hours=H_hist)
        diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
        r_hist  = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600: continue
        if _meas_nan_any[t_run_abs - H_hist : t_run_abs + H_fore].any(): continue
        
        test_run_pairs.append((r_curr, r_hist, t_run_abs))

    if not test_run_pairs:
        logger.error("No test run pairs found!")
        sys.exit(1)
    logger.info("Test run pairs: %d", len(test_run_pairs))

    # ── Model & Graph ────────────────────────────────────────────────────
    model_cfg = DCRNNConfig.from_yaml(
        dcrnn_cfg, icond2_features=icond2_features, ecmwf_features=ecmwf_features,
        measurement_features=measurement_cols, target_col=target_col,
        n_train=N_train, n_val=N_val, checkpoint_path=str(model_path)
    )
    builder = HeterogeneousGraphBuilder(model_cfg.graph)
    base_graph = builder.build(
        station_coords=station_coords, station_altitudes=alts,
        icond2_grid_coords=icond2_coords, ecmwf_grid_coords=ecmwf_coords,
        icond2_altitudes=icond2_alts, ecmwf_altitudes=ecmwf_alts
    )
    sampler = TrainingSampler(model_cfg, builder, base_graph, target_feat_idx=target_feat_idx, station_coords=station_coords)
    model = DCRNN(model_cfg)
    
    logger.info("Loading weights …")
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(_load_state_dict(ckpt))
    model = model.to(device)

    # ── Evaluation ───────────────────────────────────────────────────────
    logger.info("Starting inference …")
    results_df = evaluate(
        model=model, sampler=sampler, device=device,
        meas_raw=meas_raw, meas_scaled=meas_scaled,
        station_nearest_grid=station_nearest_grid,
        grid_icond2_runs_raw=grid_icond2_runs_raw, grid_icond2_runs_scaled=grid_icond2_runs_scaled,
        station_ecmwf_nwp_scaled=station_ecmwf_scaled, station_static=station_static_scaled,
        ecmwf_nwp_scaled=ecmwf_nwp_scaled, icond2_static=icond2_static_scaled, ecmwf_static=ecmwf_static_scaled,
        meas_scaler=meas_scaler, target_feat_idx=target_feat_idx, ws_feat_idx_i2=ws_feat_idx_i2,
        H_hist=H_hist, H_fore=H_fore,
        train_station_indices=train_station_indices, val_station_indices=val_station_indices,
        all_ids=all_ids, test_run_pairs=test_run_pairs
    )

    # ── Save results ─────────────────────────────────────────────────────
    out_dir = Path("data/test_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"test_results_{model_stem}{args.out_suffix}.csv"
    out_path = out_dir / out_name
    results_df.to_csv(out_path, index=False)
    logger.info("Results saved → %s", out_path)

    # Summary
    if not results_df.empty:
        for p in ["A", "B", "C", "D"]:
            sub = results_df[results_df["pass"] == p]
            if sub.empty: continue
            logger.info("  Pass %s: R²=%.3f, RMSE=%.3f, Skill=%.3f", 
                        p, sub["r2"].mean(), sub["rmse"].mean(), sub["skill_nwp"].mean(skipna=True))

if __name__ == "__main__":
    main()
