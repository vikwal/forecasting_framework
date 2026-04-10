#!/usr/bin/env python3
"""
get_test_results_stgnn2.py — Evaluate trained STGNN2 / DCRNN models on test data.

Supports both architectures (detected automatically from the model filename).
Runs inference over all test-period run pairs, accumulates per-station
predictions in physical units (m/s), and writes a summary CSV.

Output: data/test_results/test_results_{model-name}.csv
Columns: station_id, r2, rmse, mae, skill_nwp, sample

Inference strategy  (2×2 ablation design)
------------------------------------------
For each test run pair four forward passes are made:

  Pass A  observer=train              target=val
          → zero-shot, train-only context

  Pass B  observer=train_half1        target=train_half2  (+swap)
          → in-sample, train-only context

  Pass C  observer=val+train_half1    target=train_half2  (+swap)
          → in-sample, full (train+val) context

  Pass D  observer=train+val_half1    target=val_half2    (+swap)
          → zero-shot, full (train+val) context

B/C/D use a deterministic 50/50 split of the target group; two sub-passes
per pair ensure every station in the target group is evaluated exactly once.
The CSV records the pass (A–D) in an additional 'pass' column.

Skill_NWP = 1 − (RMSE_model / RMSE_nwp)
  RMSE_nwp: ICON-D2 nearest-grid-point forecast for the same 48-step window.

Usage
-----
    python get_test_results_stgnn2.py -m wind_stgcn_stgnn2_v1 \
        -c configs/config_wind_stgcn.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make repo root importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent.parent))

from geostatistics.train_stgnn2 import (
    load_yaml,
    load_station_measurements,
    load_station_metadata,
    load_icond2_ml_runs,
    load_ecmwf_parquet_at_stations_and_grid,
    load_nwp_elevations,
)
from geostatistics.stgnn import HeterogeneousGraphBuilder, ModelConfig, STGNN
from geostatistics.stgnn.training.sampler import TrainingSampler
from geostatistics.stgnn.utils.normalization import StandardScaler
from geostatistics.dcrnn import DCRNNConfig, DCRNN
from geostatistics.evaluation import build_eval_batch, evaluate, find_ws_feat_idx


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(model_name: str) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"eval_stgnn2_{model_name}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("eval_stgnn2")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.FileHandler(log_path))
        logger.addHandler(logging.StreamHandler())
    for h in logger.handlers:
        h.setFormatter(fmt)
    return logger


logger = logging.getLogger("eval_stgnn2")


# ---------------------------------------------------------------------------
# Model file resolution
# ---------------------------------------------------------------------------

def resolve_model_file(model_name: str, models_dir: Path) -> Path:
    """
    Find a unique .pt file in *models_dir* whose stem contains *model_name*.
    Exits with an informative error if 0 or >1 candidates are found.
    """
    candidates = [f for f in sorted(models_dir.glob("*.pt")) if model_name in f.stem]
    if len(candidates) == 0:
        logger.error(
            "No model file found in %s whose filename contains %r.  "
            "Available .pt files: %s",
            models_dir,
            model_name,
            [f.name for f in models_dir.glob("*.pt")],
        )
        sys.exit(1)
    if len(candidates) > 1:
        logger.error(
            "Model name %r is ambiguous — %d files match:\n  %s\n"
            "Please provide a more precise name.",
            model_name,
            len(candidates),
            "\n  ".join(f.name for f in candidates),
        )
        sys.exit(1)
    return candidates[0]


def detect_architecture(model_path: Path) -> str:
    """Return 'stgnn2' or 'dcrnn' based on the model filename."""
    stem = model_path.stem.lower()
    if "dcrnn" in stem:
        return "dcrnn"
    if "stgnn" in stem:
        return "stgnn2"
    raise ValueError(
        f"Cannot detect architecture from filename {model_path.name!r}. "
        "Expected 'stgnn2' (or 'stgnn') or 'dcrnn' in the name."
    )


# ---------------------------------------------------------------------------
# State-dict loading (strips torch.compile prefix)
# ---------------------------------------------------------------------------

def _load_state_dict(checkpoint: dict | object) -> dict:
    """Extract and clean the model state_dict from a checkpoint."""
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        sd = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        # Assume the checkpoint IS the state_dict
        sd = checkpoint
    else:
        raise ValueError("Unrecognised checkpoint format")

    # Strip torch.compile prefix if present
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate STGNN2 / DCRNN models on test data"
    )
    parser.add_argument(
        "-m", "--model-name", required=True,
        help="Substring of the model filename (e.g. 'stgnn2_v1' for "
             "'wind_stgcn_stgnn2_v1.pt').  Must match exactly one file.",
    )
    parser.add_argument(
        "-c", "--config", default="configs/config_wind_stgcn.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--models-dir", default="models",
        help="Directory to search for model checkpoints (default: models/).",
    )
    parser.add_argument(
        "--evaldaily", action="store_true",
        help="[NOT YET IMPLEMENTED] Save additional per-forecast-day R² CSV.",
    )
    args = parser.parse_args()

    # ── Set up logging ──────────────────────────────────────────────────
    _setup_logging(args.model_name)
    logger.info("=== STGNN2 / DCRNN Evaluation ===")

    if args.evaldaily:
        logger.warning("--evaldaily was set but is not yet implemented; ignored.")
        # TODO: implement per-forecast-day R² aggregation and wide CSV output

    # ── Resolve model file + architecture ──────────────────────────────
    models_dir = Path(args.models_dir)
    model_path = resolve_model_file(args.model_name, models_dir)
    arch = detect_architecture(model_path)
    logger.info("Model: %s  (architecture: %s)", model_path.name, arch)

    # ── Load config ─────────────────────────────────────────────────────
    cfg      = load_yaml(args.config)
    data_cfg = cfg["data"]
    arch_cfg = cfg.get(arch, cfg.get("stgnn2", {}))   # "stgnn2" or "dcrnn" section
    logger.info("Config: %s  (section: %s)", args.config, arch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Station IDs ──────────────────────────────────────────────────────
    train_ids = [str(s) for s in data_cfg["files"]]
    val_ids   = [str(s) for s in data_cfg["val_files"]]
    all_ids   = train_ids + val_ids
    N_train   = len(train_ids)
    N_val     = len(val_ids)
    train_station_indices = list(range(N_train))
    val_station_indices   = list(range(N_train, N_train + N_val))
    logger.info("Stations — train: %d  val (zero-shot): %d", N_train, N_val)

    # ── Feature config ───────────────────────────────────────────────────
    icond2_features  = arch_cfg.get("icond2_features")
    ecmwf_features   = arch_cfg.get("ecmwf_features")
    measurement_cols = arch_cfg.get("measurement_features")
    target_col       = arch_cfg.get("target_col")
    if target_col not in measurement_cols:
        raise ValueError(f"target_col '{target_col}' not in measurement_features")
    target_feat_idx = measurement_cols.index(target_col)

    H_hist = arch_cfg.get("history_length", 48)
    H_fore = arch_cfg.get("forecast_horizon", 48)

    run_hours     = tuple(arch_cfg.get("icond2_run_hours", [6, 9, 12, 15]))
    next_n_icond2 = arch_cfg.get("next_n_icond2", 4)
    next_n_ecmwf  = arch_cfg.get("next_n_ecmwf", 4)
    n_workers     = arch_cfg.get("n_workers", 8)
    nwp_path      = data_cfg.get("nwp_path")
    data_path     = data_cfg["path"]

    # NWP feature index for Skill_NWP baseline
    ws_feat_idx_i2 = find_ws_feat_idx(icond2_features)
    if ws_feat_idx_i2 is None:
        logger.warning(
            "No 'wind_speed_*' feature found in icond2_features=%s — "
            "Skill_NWP will be NaN for all stations.",
            icond2_features,
        )
    else:
        logger.info("NWP Skill baseline: icond2_features[%d] = %s",
                    ws_feat_idx_i2, icond2_features[ws_feat_idx_i2])

    # ── Load station measurements ────────────────────────────────────────
    logger.info("Loading station measurements …")
    meas_raw, timestamps = load_station_measurements(
        data_path, all_ids, cols=measurement_cols
    )
    T = len(timestamps)
    logger.info("Timestamps: %d  (%s … %s)", T, timestamps[0], timestamps[-1])

    # Temporal split
    test_start = data_cfg.get("test_start")
    test_end   = data_cfg.get("test_end")
    if test_start:
        ts_cutoff = pd.Timestamp(test_start, tz="UTC")
        split_t   = int(np.searchsorted(timestamps, ts_cutoff, side="left"))
    else:
        split_t = int(T * (1 - data_cfg.get("val_frac", 0.2)))
    split_time = timestamps[split_t]
    run_cutoff = pd.Timestamp(test_end, tz="UTC") if test_end else None
    logger.info("Temporal split — train ends at %s  (index %d)", split_time.date(), split_t)

    # ── Station metadata ────────────────────────────────────────────────
    lats, lons, alts = load_station_metadata(data_path, all_ids)
    station_coords = np.stack([lats, lons], axis=1)

    # ── ICON-D2 ML runs ─────────────────────────────────────────────────
    logger.info("Loading ICON-D2 ML runs (hours %s) …", list(run_hours))
    run_times, icond2_coords, grid_icond2_runs_raw, station_nearest_grid = \
        load_icond2_ml_runs(
            nwp_path=nwp_path,
            station_ids=all_ids,
            station_coords=station_coords,
            features=icond2_features,
            run_hours=run_hours,
            next_n_grid=next_n_icond2,
            n_workers=n_workers,
            cutoff=run_cutoff,
        )
    R = len(run_times)
    I2 = len(icond2_features)
    N_igrid = len(icond2_coords)
    logger.info("ICON-D2 grid nodes: %d  runs: %d", N_igrid, R)

    # ── ECMWF NWP ───────────────────────────────────────────────────────
    ecmwf_parquet_file = os.path.join(
        data_cfg.get("ecmwf_path", "/mnt/lambda1/nvme1/ecmwf/parquet"),
        "ecmwf_wind_sl_full.parquet",
    )
    E2 = len(ecmwf_features)

    if os.path.exists(ecmwf_parquet_file):
        station_ecmwf_nwp, ecmwf_coords, ecmwf_nwp, ecmwf_alts = \
            load_ecmwf_parquet_at_stations_and_grid(
                parquet_path=ecmwf_parquet_file,
                station_lats=lats, station_lons=lons,
                features=ecmwf_features, timestamps=timestamps,
                next_n_grid_per_station=next_n_ecmwf,
            )
        logger.info("ECMWF grid nodes: %d", len(ecmwf_coords))
    else:
        logger.warning("ECMWF parquet not found at %s — using zeros", ecmwf_parquet_file)
        station_ecmwf_nwp = np.zeros((T, len(all_ids), E2), dtype=np.float32)
        ec_lats = np.arange(47.5, 55.0, 0.5)
        ec_lons = np.arange(6.0,  15.5, 0.5)
        eg, lg  = np.meshgrid(ec_lats, ec_lons)
        ecmwf_coords = np.stack([eg.ravel(), lg.ravel()], axis=1).astype(np.float32)
        ecmwf_nwp    = np.zeros((T, len(ecmwf_coords), E2), dtype=np.float32)
        ecmwf_alts   = np.zeros(len(ecmwf_coords), dtype=np.float32)

    # ── NWP altitudes ────────────────────────────────────────────────────
    if arch_cfg.get("use_altitude_diff", False):
        weather_db_url = os.environ.get("WEATHER_DB_URL")
        ecmwf_db_url   = os.environ.get("ECMWF_WIND_SL_URL")
        if weather_db_url and ecmwf_db_url:
            logger.info("Loading NWP elevations from DB …")
            icond2_alts, ecmwf_alts = load_nwp_elevations(
                weather_db_url=weather_db_url,
                ecmwf_db_url=ecmwf_db_url,
                icond2_coords=icond2_coords,
                ecmwf_coords=ecmwf_coords,
            )
            icond2_alts += 10.0
            ecmwf_alts  += 10.0
        else:
            logger.warning("DB URL(s) missing — NWP altitudes set to 0 m")
            icond2_alts = np.zeros(N_igrid, dtype=np.float32)
    else:
        icond2_alts = np.zeros(N_igrid, dtype=np.float32)

    # ── Scalers — fit on training data only ─────────────────────────────
    logger.info("Fitting scalers on training data …")
    M_meas = len(measurement_cols)

    meas_scaler = StandardScaler()
    meas_scaler.fit(meas_raw[:split_t, :N_train].reshape(-1, M_meas))
    meas_scaled = meas_scaler.transform(
        meas_raw.reshape(-1, M_meas)
    ).reshape(T, len(all_ids), M_meas)

    train_r_mask = run_times < split_time
    i2_scaler = StandardScaler()
    i2_scaler.fit(grid_icond2_runs_raw[train_r_mask].reshape(-1, I2))
    grid_icond2_runs_scaled = i2_scaler.transform(
        grid_icond2_runs_raw.reshape(-1, I2)
    ).reshape(R, 48, N_igrid, I2)

    e2_scaler = StandardScaler()
    e2_scaler.fit(station_ecmwf_nwp[:split_t, :N_train].reshape(-1, E2))
    station_ecmwf_nwp_scaled = e2_scaler.transform(
        station_ecmwf_nwp.reshape(-1, E2)
    ).reshape(T, len(all_ids), E2)
    ecmwf_nwp_scaled = e2_scaler.transform(
        ecmwf_nwp.reshape(-1, E2)
    ).reshape(T, len(ecmwf_coords), E2)

    stat_scaler = StandardScaler()
    raw_static  = np.stack([lats, lons, alts], axis=1).astype(np.float32)
    stat_scaler.fit(raw_static[:N_train])
    station_static_scaled = stat_scaler.transform(raw_static)   # (N_all, 3)

    icond2_static_scaled = StandardScaler().fit_transform(
        np.concatenate([icond2_coords, icond2_alts[:, None]], axis=1)
    ).astype(np.float32)
    ecmwf_static_scaled = StandardScaler().fit_transform(
        np.concatenate([ecmwf_coords, ecmwf_alts[:, None]], axis=1)
    ).astype(np.float32)

    # ── Test run pairs ───────────────────────────────────────────────────
    logger.info("Identifying test run pairs (t_run >= %s) …", split_time.date())
    ts_lookup = pd.Series(np.arange(T), index=timestamps)
    _meas_nan_any = np.isnan(meas_raw[:, :, 0]).any(axis=1)

    test_run_pairs: list[tuple[int, int, int]] = []
    skipped = 0

    for r_curr in range(R):
        t_run = run_times[r_curr]
        if t_run < split_time:            # only test period
            continue
        if t_run not in ts_lookup.index:
            skipped += 1; continue
        t_run_abs = int(ts_lookup[t_run])
        if t_run_abs < H_hist or t_run_abs + H_fore > T:
            skipped += 1; continue

        t_hist_target = t_run - pd.Timedelta(hours=H_hist)
        diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
        r_hist  = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600:
            skipped += 1; continue
        if _meas_nan_any[t_run_abs - H_hist : t_run_abs + H_fore].any():
            skipped += 1; continue

        test_run_pairs.append((r_curr, r_hist, t_run_abs))

    logger.info("Test run pairs: %d  (skipped: %d)", len(test_run_pairs), skipped)
    if not test_run_pairs:
        logger.error("No test run pairs found — check test_start / test_end in config")
        sys.exit(1)

    # ── Model config + graph ─────────────────────────────────────────────
    if arch == "stgnn2":
        model_cfg = ModelConfig.from_yaml(
            arch_cfg,
            icond2_features=icond2_features,
            ecmwf_features=ecmwf_features,
            measurement_features=measurement_cols,
            n_train=N_train,
            n_val=N_val,
            checkpoint_path=str(model_path),
        )
        builder    = HeterogeneousGraphBuilder(model_cfg.graph)
        base_graph = builder.build(
            station_coords=station_coords,
            station_altitudes=alts,
            icond2_grid_coords=icond2_coords,
            ecmwf_grid_coords=ecmwf_coords,
            icond2_altitudes=icond2_alts,
            ecmwf_altitudes=ecmwf_alts,
        )
        sampler = TrainingSampler(
            model_cfg, builder, base_graph,
            target_feat_idx=target_feat_idx,
            station_coords=station_coords,
        )
        model = STGNN(model_cfg)

    else:  # dcrnn
        dcrnn_cfg = DCRNNConfig.from_yaml(
            arch_cfg,
            icond2_features=icond2_features,
            ecmwf_features=ecmwf_features,
            measurement_features=measurement_cols,
            target_col=target_col,
            n_train=N_train,
            n_val=N_val,
            checkpoint_path=str(model_path),
        )
        builder    = HeterogeneousGraphBuilder(dcrnn_cfg.graph)
        base_graph = builder.build(
            station_coords=station_coords,
            station_altitudes=alts,
            icond2_grid_coords=icond2_coords,
            ecmwf_grid_coords=ecmwf_coords,
            icond2_altitudes=icond2_alts,
            ecmwf_altitudes=ecmwf_alts,
        )
        sampler = TrainingSampler(
            dcrnn_cfg, builder, base_graph,
            target_feat_idx=target_feat_idx,
            station_coords=station_coords,
        )
        model = DCRNN(dcrnn_cfg)

    logger.info(
        "Graph — s2s edges: %d  i2s: %d  e2s: %d",
        base_graph["station", "near", "station"].edge_index.shape[1] // 2,
        base_graph["icond2", "informs", "station"].edge_index.shape[1],
        base_graph["ecmwf",  "informs", "station"].edge_index.shape[1],
    )

    # ── Load model weights ───────────────────────────────────────────────
    logger.info("Loading checkpoint: %s", model_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = _load_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    logger.info("Parameters: %s", f"{model.count_parameters():,}")

    # ── Run evaluation ───────────────────────────────────────────────────
    logger.info("Starting inference over %d test pairs …", len(test_run_pairs))
    results_df = evaluate(
        model=model,
        sampler=sampler,
        device=device,
        meas_raw=meas_raw,
        meas_scaled=meas_scaled,
        station_nearest_grid=station_nearest_grid,
        grid_icond2_runs_raw=grid_icond2_runs_raw,
        grid_icond2_runs_scaled=grid_icond2_runs_scaled,
        station_ecmwf_nwp_scaled=station_ecmwf_nwp_scaled,
        station_static=station_static_scaled,
        ecmwf_nwp_scaled=ecmwf_nwp_scaled,
        icond2_static=icond2_static_scaled,
        ecmwf_static=ecmwf_static_scaled,
        meas_scaler=meas_scaler,
        target_feat_idx=target_feat_idx,
        ws_feat_idx_i2=ws_feat_idx_i2,
        H_hist=H_hist,
        H_fore=H_fore,
        train_station_indices=train_station_indices,
        val_station_indices=val_station_indices,
        all_ids=all_ids,
        test_run_pairs=test_run_pairs,
    )

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_dir = Path("data/test_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"test_results_{args.model_name}.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Results saved → %s  (%d stations)", out_path, len(results_df))

    # Summary grouped by pass
    if not results_df.empty:
        for pass_label in ["A", "B", "C", "D"]:
            sub = results_df[results_df["pass"] == pass_label]
            if sub.empty:
                continue
            sample_label = sub["sample"].iloc[0]
            logger.info(
                "  [Pass %s / %s] n=%d  R²=%.3f  RMSE=%.3f m/s  "
                "MAE=%.3f m/s  Skill=%.3f",
                pass_label, sample_label, len(sub),
                sub["r2"].mean(), sub["rmse"].mean(),
                sub["mae"].mean(), sub["skill_nwp"].mean(skipna=True),
            )


if __name__ == "__main__":
    main()