"""
train_dcrnn.py — Training script for the DCRNN Seq2Seq model.

Reuses all data loading from train_stgnn2.py (ICON-D2, ECMWF, station
measurements, scaler fitting, graph building, run-pair construction).
Only the model, config, sampler wiring, and trainer differ.

DCRNN vs STGNN key differences
--------------------------------
  Model    : Seq2Seq DCGRUCell instead of ST-blocks with GATv2Conv
  Decoder  : autoregressive loop with NWP injection at each step
  TF ratio : teacher_forcing_ratio linearly decayed over training
  Config   : ``dcrnn`` section in the YAML (not ``stgnn2``)

Add a ``dcrnn`` section to your config YAML, e.g.::

    dcrnn:
      icond2_features: ['u_10m', 'v_10m', 'wind_speed_10m']
      ecmwf_features:  ['u_wind10m', 'v_wind10m', 'wind_speed_10m']
      measurement_features: ['wind_speed', 'wind_direction']
      target_col: 'wind_speed'
      n_workers: 16

      history_length: 48
      forecast_horizon: 48
      temporal_encoding: gru    # sampler output format

      hidden: 128
      num_layers: 2
      diffusion_K: 2
      dropout: 0.1
      teacher_forcing_ratio: 0.5  # starting ratio (decays to 0)

      station_connectivity: delaunay
      next_n_icond2: 4
      next_n_ecmwf: 4
      use_altitude_diff: true
      neighbor_radius_km: 500

      min_target_stations: 1
      max_target_stations: 10
      max_neighbor_stations: 60
      loss_fn: mse
      loss_weights_by_horizon: true
      horizon_decay: 0.95
      lr: 3.0e-4
      weight_decay: 1.0e-5
      scheduler: cosine
      batch_size: 8
      max_epochs: 200
      patience: 15
      gradient_clip: 1.0
      icond2_run_hours: [6, 9, 12, 15]

Usage
-----
    python geostatistics/train_dcrnn.py \\
        --config configs/config_wind_stgcn.yaml \\
        --suffix v1
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Reuse ALL data-loading functions from train_stgnn2 ──────────────────────
from geostatistics.train_stgnn2 import (
    load_yaml,
    load_station_measurements,
    load_station_metadata,
    load_icond2_ml_runs,
    load_ecmwf_at_stations_and_grid,
    load_ecmwf_parquet_at_stations_and_grid,
    load_nwp_elevations,
)

# ── DCRNN-specific imports ───────────────────────────────────────────────────
from geostatistics.dcrnn import DCRNNConfig, DCRNN
from geostatistics.dcrnn.training import DCRNNTrainer

# ── Shared graph and sampling infrastructure ─────────────────────────────────
from geostatistics.stgnn import HeterogeneousGraphBuilder
from geostatistics.stgnn.training.sampler import TrainingSampler
from geostatistics.stgnn.utils.normalization import StandardScaler
from geostatistics.evaluation import evaluate as run_evaluation, find_ws_feat_idx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_dcrnn")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DCRNN Seq2Seq model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument(
        "--eval", action="store_true",
        help="Run 4-pass LOO evaluation on val run pairs after training and save results in the pkl.",
    )
    args = parser.parse_args()

    cfg      = load_yaml(args.config)
    data_cfg = cfg["data"]
    dcrnn_cfg = cfg.get("dcrnn", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config_stem = Path(args.config).stem.replace("config_", "")
    model_name  = (
        f"{config_stem}_dcrnn_{args.suffix}.pt"
        if args.suffix else
        f"{config_stem}_dcrnn.pt"
    )
    model_path = Path("models") / model_name

    # Set up dynamic log file
    log_stem = Path(args.config).stem
    log_name = f"train_dcrnn_{log_stem}_{args.suffix}.log" if args.suffix else f"train_dcrnn_{log_stem}.log"
    log_path = Path("logs") / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    logging.getLogger("geostatistics").addHandler(fh)

    # ------------------------------------------------------------------
    # Station IDs
    # ------------------------------------------------------------------
    train_ids = [str(s) for s in data_cfg["files"]]
    val_ids   = [str(s) for s in data_cfg["val_files"]]
    all_ids   = train_ids + val_ids
    N_train   = len(train_ids)
    N_val     = len(val_ids)
    logger.info("Stations — train: %d  val: %d", N_train, N_val)

    # ------------------------------------------------------------------
    # Feature config
    # ------------------------------------------------------------------
    icond2_features  = dcrnn_cfg.get("icond2_features")
    ecmwf_features   = dcrnn_cfg.get("ecmwf_features")
    measurement_cols = dcrnn_cfg.get("measurement_features")
    target_col       = dcrnn_cfg.get("target_col")
    if target_col not in measurement_cols:
        raise ValueError(f"target_col '{target_col}' must be in measurement_features")

    run_hours     = tuple(dcrnn_cfg.get("icond2_run_hours", [6, 9, 12, 15]))
    next_n_icond2 = dcrnn_cfg.get("next_n_icond2", 4)
    n_workers     = dcrnn_cfg.get("n_workers", 8)
    nwp_path      = data_cfg.get("nwp_path")
    data_path     = data_cfg["path"]

    H   = dcrnn_cfg.get("history_length", 48)
    F_h = dcrnn_cfg.get("forecast_horizon", 48)

    # ------------------------------------------------------------------
    # Station measurements  (T, N_all, M)  — identical to stgnn2
    # ------------------------------------------------------------------
    logger.info("Loading station measurements …")
    meas_raw, timestamps = load_station_measurements(
        data_path, all_ids, cols=measurement_cols
    )
    T = len(timestamps)
    logger.info("Timestamps: %d  (%s … %s)", T, timestamps[0], timestamps[-1])

    nan_counts = np.isnan(meas_raw[:, :, 0]).sum(axis=0)
    bad = [(all_ids[i], int(nan_counts[i])) for i in np.where(nan_counts > 0)[0]]
    if bad:
        logger.warning(
            "Measurement NaN audit: %d stations have gaps: %s", len(bad), bad
        )
    _meas_nan_any = np.isnan(meas_raw[:, :, 0]).any(axis=1)

    test_start = data_cfg.get("test_start")
    test_end   = data_cfg.get("test_end")
    if test_start:
        ts_cutoff = pd.Timestamp(test_start, tz="UTC")
        split_t   = int(np.searchsorted(timestamps, ts_cutoff, side="left"))
    else:
        split_t = int(T * (1 - data_cfg.get("val_frac", 0.2)))
    split_time = timestamps[split_t]
    run_cutoff = pd.Timestamp(test_end, tz="UTC") if test_end else None
    logger.info("Temporal split — train_T: %d  val_T: %d  (split at %s)",
                split_t, T - split_t, split_time.date())

    # ------------------------------------------------------------------
    # Station metadata
    # ------------------------------------------------------------------
    lats, lons, alts = load_station_metadata(data_path, all_ids)
    station_coords = np.stack([lats, lons], axis=1)

    # ------------------------------------------------------------------
    # ICON-D2 ML runs  →  (R, 48, N_grid, I2)
    # ------------------------------------------------------------------
    logger.info("Loading ICON-D2 ML runs (hours %s) …", list(run_hours))
    run_times, icond2_coords, grid_icond2_runs, station_nearest_grid = \
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

    # ------------------------------------------------------------------
    # ECMWF NWP
    # ------------------------------------------------------------------
    db_url = os.environ.get("ECMWF_WIND_SL_URL")

    ecmwf_parquet_dir = data_cfg.get("ecmwf_path", "/mnt/lambda1/nvme1/ecmwf/parquet")
    ecmwf_parquet_file = os.path.join(ecmwf_parquet_dir, "ecmwf_wind_sl_full.parquet")

    if os.path.exists(ecmwf_parquet_file):
        station_ecmwf_nwp, ecmwf_coords, ecmwf_nwp, ecmwf_alts = \
            load_ecmwf_parquet_at_stations_and_grid(
                parquet_path=ecmwf_parquet_file,
                station_lats=lats, station_lons=lons,
                features=ecmwf_features, timestamps=timestamps,
                next_n_grid_per_station=dcrnn_cfg.get("next_n_ecmwf", 4),
            )
        logger.info("ECMWF grid nodes: %d", len(ecmwf_coords))
    else:
        logger.warning(f"Parquet file {ecmwf_parquet_file} not found. Using fallbacks...")
        # if db_url:
        #     logger.info("Loading ECMWF NWP from database …")
        #     station_ecmwf_nwp, ecmwf_coords, ecmwf_nwp, ecmwf_alts = \
        #         load_ecmwf_at_stations_and_grid(
        #             db_url=db_url,
        #             station_lats=lats, station_lons=lons,
        #             features=ecmwf_features, timestamps=timestamps,
        #             next_n_grid_per_station=dcrnn_cfg.get("next_n_ecmwf", 4),
        #         )
        #     logger.info("ECMWF grid nodes: %d", len(ecmwf_coords))
        # else:
        logger.warning("ECMWF_WIND_SL_URL not set — using zero ECMWF features")
        station_ecmwf_nwp = np.zeros(
            (T, len(all_ids), len(ecmwf_features)), dtype=np.float32
        )
        ec_lats = np.arange(47.5, 55.0, 0.5)
        ec_lons = np.arange(6.0, 15.5, 0.5)
        eg, lg  = np.meshgrid(ec_lats, ec_lons)
        ecmwf_coords = np.stack([eg.ravel(), lg.ravel()], axis=1).astype(np.float32)
        ecmwf_nwp    = np.zeros((T, len(ecmwf_coords), len(ecmwf_features)), dtype=np.float32)
        ecmwf_alts   = np.zeros(len(ecmwf_coords), dtype=np.float32)

    # ------------------------------------------------------------------
    # NWP elevations
    # ------------------------------------------------------------------
    weather_db_url = os.environ.get("WEATHER_DB_URL")
    if dcrnn_cfg.get("use_altitude_diff", False):
        if weather_db_url and db_url:
            logger.info("Loading NWP elevations …")
            icond2_alts, ecmwf_alts = load_nwp_elevations(
                weather_db_url=weather_db_url,
                ecmwf_db_url=db_url,
                icond2_coords=icond2_coords,
                ecmwf_coords=ecmwf_coords,
            )
            # Offset grid elevations by +10.0m because grid coords are ground-level and NWP features are 10m wind
            icond2_alts += 10.0
            ecmwf_alts += 10.0
        else:
            logger.warning("DB URL missing — NWP altitudes set to 0 m")
            icond2_alts = np.zeros(N_igrid, dtype=np.float32)
    else:
        icond2_alts = np.zeros(N_igrid, dtype=np.float32)

    # ------------------------------------------------------------------
    # Scalers
    # ------------------------------------------------------------------
    logger.info("Fitting scalers …")
    M_meas = len(measurement_cols)

    meas_scaler = StandardScaler()
    meas_scaler.fit(meas_raw[:split_t, :N_train].reshape(-1, M_meas))
    meas_scaled = meas_scaler.transform(
        meas_raw.reshape(-1, M_meas)
    ).reshape(T, len(all_ids), M_meas)

    train_r_mask = run_times < split_time
    i2_scaler = StandardScaler()
    i2_scaler.fit(grid_icond2_runs[train_r_mask].reshape(-1, I2))
    grid_icond2_runs_scaled = i2_scaler.transform(
        grid_icond2_runs.reshape(-1, I2)
    ).reshape(R, 48, N_igrid, I2)

    E2 = len(ecmwf_features)
    e2_scaler = StandardScaler()
    e2_scaler.fit(station_ecmwf_nwp[:split_t, :N_train].reshape(-1, E2))
    station_ecmwf_scaled = e2_scaler.transform(
        station_ecmwf_nwp.reshape(-1, E2)
    ).reshape(T, len(all_ids), E2)
    ecmwf_nwp_scaled = e2_scaler.transform(
        ecmwf_nwp.reshape(-1, E2)
    ).reshape(T, len(ecmwf_coords), E2)

    stat_scaler = StandardScaler()
    raw_static  = np.stack([lats, lons, alts], axis=1)
    stat_scaler.fit(raw_static[:N_train])
    station_static_scaled = stat_scaler.transform(raw_static)

    icond2_static_scaled = StandardScaler().fit_transform(
        np.concatenate([icond2_coords, icond2_alts[:, None]], axis=1)
    ).astype(np.float32)
    ecmwf_static_scaled = StandardScaler().fit_transform(
        np.concatenate([ecmwf_coords, ecmwf_alts[:, None]], axis=1)
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Run pairs
    # ------------------------------------------------------------------
    ts_lookup = pd.Series(np.arange(T), index=timestamps)

    train_run_pairs: list[tuple[int, int, int]] = []
    val_run_pairs:   list[tuple[int, int, int]] = []
    skipped = 0

    for r_curr in range(R):
        t_run = run_times[r_curr]
        if t_run not in ts_lookup.index:
            skipped += 1; continue
        t_run_abs = int(ts_lookup[t_run])
        if t_run_abs < H or t_run_abs + F_h > T:
            skipped += 1; continue

        t_hist_target = t_run - pd.Timedelta(hours=H)
        diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
        r_hist  = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600:
            skipped += 1; continue
        if _meas_nan_any[t_run_abs - H : t_run_abs + F_h].any():
            skipped += 1; continue

        pair = (r_curr, r_hist, t_run_abs)
        (train_run_pairs if t_run < split_time else val_run_pairs).append(pair)

    logger.info(
        "Run pairs — train: %d  val: %d  skipped: %d",
        len(train_run_pairs), len(val_run_pairs), skipped,
    )

    # ------------------------------------------------------------------
    # DCRNN config
    # ------------------------------------------------------------------
    model_cfg = DCRNNConfig.from_yaml(
        dcrnn_cfg,
        icond2_features=icond2_features,
        ecmwf_features=ecmwf_features,
        measurement_features=measurement_cols,
        target_col=target_col,
        n_train=N_train,
        n_val=N_val,
        checkpoint_path=str(model_path),
    )

    # ------------------------------------------------------------------
    # Graph  (shared builder and sampler from stgnn)
    # ------------------------------------------------------------------
    builder    = HeterogeneousGraphBuilder(model_cfg.graph)
    base_graph = builder.build(
        station_coords=station_coords,
        station_altitudes=alts,
        icond2_grid_coords=icond2_coords,
        ecmwf_grid_coords=ecmwf_coords,
        icond2_altitudes=icond2_alts,
        ecmwf_altitudes=ecmwf_alts,
    )
    logger.info(
        "Graph — s2s: %d  i2s: %d  e2s: %d",
        base_graph["station", "near", "station"].edge_index.shape[1] // 2,
        base_graph["icond2", "informs", "station"].edge_index.shape[1],
        base_graph["ecmwf",  "informs", "station"].edge_index.shape[1],
    )

    # ------------------------------------------------------------------
    # Sampler  — reused unchanged from stgnn
    # The sampler reads model_cfg.temporal_encoding to decide output shape.
    # DCRNNConfig exposes .training, .history_length, .forecast_horizon,
    # .temporal_encoding — exactly what TrainingSampler needs.
    # ------------------------------------------------------------------
    target_feat_idx = model_cfg.target_feat_idx
    sampler = TrainingSampler(
        model_cfg, builder, base_graph,
        target_feat_idx=target_feat_idx,
        station_coords=station_coords,
    )

    # ------------------------------------------------------------------
    # Model + Trainer
    # ------------------------------------------------------------------
    model = DCRNN(model_cfg)
    logger.info("DCRNN parameters: %s", f"{model.count_parameters():,}")

    trainer = DCRNNTrainer(
        model=model,
        sampler=sampler,
        config=model_cfg,
        device=device,
        teacher_forcing_start=dcrnn_cfg.get("teacher_forcing_ratio", 1.0),
        teacher_forcing_end=0.0,
    )

    train_station_indices = list(range(N_train))
    val_station_indices   = list(range(N_train, N_train + N_val))

    logger.info("Starting training …")
    logger.info("Hyperparameters:\n%s", yaml.dump(dcrnn_cfg, default_flow_style=False))
    fit_result = trainer.fit(
        station_meas=meas_scaled,
        station_nearest_grid=station_nearest_grid,
        grid_icond2_runs=grid_icond2_runs_scaled,
        station_ecmwf_nwp=station_ecmwf_scaled,
        station_static=station_static_scaled,
        ecmwf_nwp=ecmwf_nwp_scaled,
        icond2_static=icond2_static_scaled,
        ecmwf_static=ecmwf_static_scaled,
        train_run_pairs=train_run_pairs,
        val_run_pairs=val_run_pairs,
        train_station_indices=train_station_indices,
        val_station_indices=val_station_indices,
    )

    logger.info("Training complete. Best model: %s", model_path)

    # ------------------------------------------------------------------
    # Optional post-training evaluation
    # ------------------------------------------------------------------
    eval_df: pd.DataFrame | None = None
    if args.eval:
        logger.info("Running 4-pass LOO evaluation on val run pairs …")
        # Load best checkpoint
        ckpt = torch.load(model_path, map_location=device)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(sd)
        ws_feat_idx_i2 = find_ws_feat_idx(icond2_features)
        eval_df = run_evaluation(
            model=model,
            sampler=sampler,
            device=device,
            meas_raw=meas_raw,
            meas_scaled=meas_scaled,
            station_nearest_grid=station_nearest_grid,
            grid_icond2_runs_raw=grid_icond2_runs,
            grid_icond2_runs_scaled=grid_icond2_runs_scaled,
            station_ecmwf_nwp_scaled=station_ecmwf_scaled,
            station_static=station_static_scaled,
            ecmwf_nwp_scaled=ecmwf_nwp_scaled,
            icond2_static=icond2_static_scaled,
            ecmwf_static=ecmwf_static_scaled,
            meas_scaler=meas_scaler,
            target_feat_idx=target_feat_idx,
            ws_feat_idx_i2=ws_feat_idx_i2,
            H_hist=H,
            H_fore=F_h,
            train_station_indices=train_station_indices,
            val_station_indices=val_station_indices,
            all_ids=all_ids,
            test_run_pairs=val_run_pairs,
        )
        logger.info("Evaluation done — %d rows", len(eval_df))

    # ------------------------------------------------------------------
    # Save training pkl
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_stem = Path(model_name).stem   # e.g. "wind_stgcn_dcrnn_v1"
    pkl_path = Path("results") / f"{pkl_stem}_{timestamp}.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    pkl_data = {
        "model_name":    pkl_stem,
        "architecture":  "dcrnn",
        "config":        dcrnn_cfg,
        "train_ids":     train_ids,
        "val_ids":       val_ids,
        "n_train":       N_train,
        "n_val":         N_val,
        "history":       fit_result["history"],
        "best_val_loss": fit_result["best_val_loss"],
        "stopped_epoch": fit_result["stopped_epoch"],
        "evaluation":    eval_df,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)
    logger.info("Training results saved → %s", pkl_path)


if __name__ == "__main__":
    main()
