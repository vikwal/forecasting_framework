"""
train_wavenet.py — Training script for the Graph WaveNet baseline model.

Identical data loading pipeline to train_mtgnn.py; only the model class and
config section differ (``wavenet`` instead of ``mtgnn``).  No curriculum
learning — Graph WaveNet outputs the full forecast horizon directly.

Usage
-----
    python geostatistics/train_wavenet.py \\
        --config configs/config_wind_wavenet.yaml \\
        --suffix v1
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]

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
from geostatistics.homo_sampler import HomoSampler, evaluate_homo_model
from geostatistics.wavenet import GraphWaveNetModel
from geostatistics.stgnn.utils.normalization import StandardScaler
from geostatistics.train_dcrnn import encode_circular_measurements, apply_dir_encoding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_wavenet")


# ---------------------------------------------------------------------------
# Loss helpers  (identical to train_mtgnn.py)
# ---------------------------------------------------------------------------

def _horizon_weights(F_h: int, decay: float, device: torch.device) -> torch.Tensor:
    w = torch.tensor([decay ** i for i in range(F_h)], dtype=torch.float32, device=device)
    return w / w.sum()


def _masked_loss(pred: torch.Tensor, gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - gt) ** 2
    return (diff2 * w.unsqueeze(0)).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _metrics(
    preds: torch.Tensor,
    gts: torch.Tensor,
    target_scale: float,
    target_mean: float,
) -> tuple[float, float]:
    """RMSE and R² in physical units."""
    p      = preds * target_scale + target_mean
    g      = gts   * target_scale + target_mean
    err    = p - g
    rmse   = float(err.pow(2).mean().sqrt())
    ss_res = float(err.pow(2).sum())
    ss_tot = float((g - g.mean()).pow(2).sum())
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    return rmse, r2


def _train_epoch(
    model: GraphWaveNetModel,
    sampler: HomoSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
    w: torch.Tensor,
    target_scale: float,
    target_mean: float,
    grad_accum: int = 1,
    max_grad_norm: float = 5.0,
) -> tuple[float, float, float]:
    """Returns (weighted_loss, RMSE_phys, R²_phys)."""
    model.train()
    total_loss = 0.0
    all_preds: list[torch.Tensor] = []
    all_gts:   list[torch.Tensor] = []
    optimizer.zero_grad()

    for step_i in range(steps):
        batch = sampler.sample_train()
        x           = batch.x.unsqueeze(0).to(device)
        static      = batch.static.to(device)
        target_mask = batch.target_mask.to(device)
        gt          = batch.ground_truth.to(device)

        if target_mask.sum() == 0:
            continue

        pred = model(x, static, target_mask)
        loss = _masked_loss(pred, gt, w) / grad_accum
        loss.backward()
        total_loss += loss.item() * grad_accum
        all_preds.append(pred.detach().cpu())
        all_gts.append(gt.cpu())

        if (step_i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

    if steps % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    preds_cat = torch.cat(all_preds, dim=0)
    gts_cat   = torch.cat(all_gts,   dim=0)
    rmse, r2  = _metrics(preds_cat, gts_cat, target_scale, target_mean)
    return total_loss / max(steps, 1), rmse, r2


@torch.no_grad()
def _val_epoch(
    model: GraphWaveNetModel,
    sampler: HomoSampler,
    device: torch.device,
    w: torch.Tensor,
    target_scale: float = 1.0,
    target_mean: float = 0.0,
) -> tuple[float, float, float]:
    """Returns (weighted_loss, RMSE_phys, R²_phys)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[torch.Tensor] = []
    all_gts:   list[torch.Tensor] = []
    count = 0
    for batch in sampler.iter_val():
        x           = batch.x.unsqueeze(0).to(device)
        static      = batch.static.to(device)
        target_mask = batch.target_mask.to(device)
        gt          = batch.ground_truth.to(device)
        if target_mask.sum() == 0:
            continue
        pred = model(x, static, target_mask)
        total_loss += _masked_loss(pred, gt, w).item()
        all_preds.append(pred.cpu())
        all_gts.append(gt.cpu())
        count += 1

    preds_cat = torch.cat(all_preds, dim=0)
    gts_cat   = torch.cat(all_gts,   dim=0)
    rmse, r2  = _metrics(preds_cat, gts_cat, target_scale, target_mean)
    return total_loss / max(count, 1), rmse, r2




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Graph WaveNet baseline model")
    parser.add_argument("--config",    required=True)
    parser.add_argument("--suffix",    default="")
    parser.add_argument("--test-mode", action="store_true",
                        help="Final training mode: train on files+val_files, val on test_files")
    parser.add_argument("--eval", action="store_true",
                        help="Run per-station LOO evaluation on val run pairs after training")
    args = parser.parse_args()

    cfg      = load_yaml(args.config)
    data_cfg = cfg["data"]
    mcfg     = cfg.get("wavenet", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config_stem = Path(args.config).stem.replace("config_", "")
    model_name  = f"{config_stem}_wavenet_{args.suffix}.pt" if args.suffix else f"{config_stem}_wavenet.pt"
    model_path  = Path("models") / model_name

    log_stem = Path(args.config).stem
    log_name = f"train_wavenet_{log_stem}_{args.suffix}.log" if args.suffix else f"train_wavenet_{log_stem}.log"
    log_path = Path("logs") / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)

    # ------------------------------------------------------------------
    # Station IDs
    # ------------------------------------------------------------------
    if args.test_mode:
        if not data_cfg.get("test_files"):
            raise ValueError("--test-mode requires 'test_files' in data config.")
        train_ids = [str(s) for s in data_cfg["files"]] + [str(s) for s in data_cfg["val_files"]]
        val_ids   = [str(s) for s in data_cfg["test_files"]]
    else:
        train_ids = [str(s) for s in data_cfg["files"]]
        val_ids   = [str(s) for s in data_cfg["val_files"]]
    all_ids = train_ids + val_ids
    N_train, N_val = len(train_ids), len(val_ids)
    logger.info("Stations — train: %d  val: %d", N_train, N_val)

    # ------------------------------------------------------------------
    # Feature config
    # ------------------------------------------------------------------
    use_case         = data_cfg.get("use_case", "wind")
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

    H   = mcfg.get("history_length", 48)
    F_h = mcfg.get("forecast_horizon", 48)

    freq   = data_cfg.get("freq", "1h")
    _freq_h_map = {"1h": 1.0, "1H": 1.0, "30min": 0.5, "30T": 0.5, "15min": 0.25, "15T": 0.25}
    freq_h = _freq_h_map.get(freq, 1.0)

    # ------------------------------------------------------------------
    # Station measurements
    # ------------------------------------------------------------------
    test_end   = data_cfg.get("test_end")
    run_cutoff = pd.Timestamp(test_end, tz="UTC") if test_end else None

    logger.info("Loading station measurements …")
    meas_raw, timestamps = load_station_measurements(
        data_path, all_ids, cols=measurement_cols, freq=freq,
    )

    if run_cutoff is not None:
        meas_cutoff = run_cutoff + pd.Timedelta(days=2)
        cut_idx     = int(np.searchsorted(timestamps, meas_cutoff, side="right"))
        meas_raw    = meas_raw[:cut_idx]
        timestamps  = timestamps[:cut_idx]
    T = len(timestamps)
    logger.info("Timestamps: %d  (%s … %s)", T, timestamps[0], timestamps[-1])

    # ------------------------------------------------------------------
    # Imputation
    # ------------------------------------------------------------------
    interpol_path = data_cfg.get("interpol_path")
    if interpol_path:
        rk_pred  = load_interpol_imputation(interpol_path, all_ids, timestamps)
        meas_raw = apply_interpol_imputation(meas_raw, rk_pred, measurement_cols, target_col)

    knnimputer_path = data_cfg.get("knnimputer_path")
    if knnimputer_path:
        for col in measurement_cols:
            if int(np.isnan(meas_raw[:, :, measurement_cols.index(col)]).sum()) == 0:
                continue
            knn_arr  = load_knn_imputation(knnimputer_path, col, all_ids, timestamps, freq=freq)
            meas_raw = apply_knn_imputation(meas_raw, knn_arr, measurement_cols, col)

    handle_nans = mcfg.get("handle_nans", "break")
    nan_counts  = np.isnan(meas_raw).any(axis=2).sum(axis=0)
    bad_ids     = [all_ids[i] for i in np.where(nan_counts > 0)[0]]
    if bad_ids:
        msg = f"{len(bad_ids)} station(s) still have NaN after imputation: {bad_ids}"
        if handle_nans == "break":
            raise ValueError(msg)
        logger.warning(msg)

    _meas_nan_any = np.isnan(meas_raw).any(axis=(1, 2))
    meas_raw, measurement_cols = encode_circular_measurements(meas_raw, measurement_cols)

    test_start = data_cfg.get("test_start")
    if test_start:
        split_t = int(np.searchsorted(timestamps, pd.Timestamp(test_start, tz="UTC"), side="left"))
    else:
        split_t = int(T * (1 - data_cfg.get("val_frac", 0.2)))
    split_time = timestamps[split_t]
    logger.info("Temporal split at %s  (train: %d  val: %d steps)", split_time.date(), split_t, T - split_t)

    # ------------------------------------------------------------------
    # Station metadata
    # ------------------------------------------------------------------
    meta_path = data_cfg.get("stations_master")
    lats, lons, alts = load_station_metadata(data_path, all_ids, meta_path=meta_path)
    station_coords   = np.stack([lats, lons], axis=1)

    # ------------------------------------------------------------------
    # ICON-D2 runs
    # ------------------------------------------------------------------
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
    logger.info("ICON-D2 grid nodes: %d  runs: %d  leads/run: %d  I2: %d", len(icond2_coords), R, n_leads, I2)

    # ------------------------------------------------------------------
    # Scalers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # ECMWF (optional)
    # ------------------------------------------------------------------
    next_n_ecmwf        = mcfg.get("next_n_ecmwf", 0)
    ecmwf_features = mcfg.get("ecmwf_features") or []
    nwp_nodes      = mcfg.get("nwp_nodes", False)
    # When nwp_nodes=True, HomoSampler must concatenate k grid points (not aggregate)
    aggregate_nwp  = False if nwp_nodes else mcfg.get("aggregate_nwp", True)
    if nwp_nodes:
        logger.info("nwp_nodes=True — forcing aggregate_nwp=False (k×I2 channels)")
    grid_ecmwf_scaled: np.ndarray | None = None
    ecmwf_coords:      np.ndarray | None = None

    if next_n_ecmwf > 0 and ecmwf_features:
        ecmwf_path = data_cfg.get("ecmwf_path")
        if ecmwf_path and os.path.exists(ecmwf_path):
            logger.info("Loading ECMWF NWP (%d features, k=%d) …", len(ecmwf_features), next_n_ecmwf)
            _, ecmwf_coords, grid_ecmwf_runs, _ = load_ecmwf_parquet_at_stations_and_grid(
                parquet_path=ecmwf_path,
                station_lats=lats,
                station_lons=lons,
                features=ecmwf_features,
                timestamps=timestamps,
                next_n_grid_per_station=next_n_ecmwf,
            )
            if e2_mode == "dir_in_deg":
                grid_ecmwf_runs, ecmwf_features = apply_dir_encoding(grid_ecmwf_runs, ecmwf_features)
            E2 = grid_ecmwf_runs.shape[2]
            logger.info("ECMWF grid nodes: %d  features: %d", len(ecmwf_coords), E2)
            e2_scaler = StandardScaler()
            e2_scaler.fit(grid_ecmwf_runs[:split_t].reshape(-1, E2))
            grid_ecmwf_scaled = e2_scaler.transform(
                grid_ecmwf_runs.reshape(-1, E2)
            ).reshape(T, len(ecmwf_coords), E2)
        else:
            logger.warning("next_n_ecmwf=%d but ecmwf_path not set or missing — ECMWF disabled", next_n_ecmwf)
            next_n_ecmwf = 0

    agg_mode = "IDW-mean" if aggregate_nwp else "concat"
    logger.info("NWP aggregation: %s  (aggregate_nwp=%s)", agg_mode, aggregate_nwp)

    # Index of wind_speed_10m in ICON-D2 for Skill_NWP baseline.
    # Must use the exact feature name — not the first "wind_speed" match, because
    # apply_dir_encoding reorders columns (non-consumed features first) and
    # wind_speed_38m would end up at index 0 in dir_in_deg mode.
    nwp_ws_feat_idx = next(
        (i for i, f in enumerate(icond2_features) if f == "wind_speed_10m"), None
    )
    if nwp_ws_feat_idx is None:
        nwp_ws_feat_idx = next(
            (i for i, f in enumerate(icond2_features) if "wind_speed" in f), 0
        )
        logger.warning(
            "wind_speed_10m not found in ICON-D2 features %s — "
            "Skill_NWP baseline uses '%s' (index %d) instead.",
            icond2_features, icond2_features[nwp_ws_feat_idx], nwp_ws_feat_idx,
        )

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

        t_hist_target = t_run - pd.Timedelta(hours=H * freq_h)
        diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
        r_hist  = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600:
            skipped += 1; continue
        if _meas_nan_any[t_run_abs - H:t_run_abs + F_h].any():
            skipped += 1; continue

        pair = (r_curr, r_hist, t_run_abs)
        (train_run_pairs if t_run < split_time else val_run_pairs).append(pair)

    logger.info("Run pairs — train: %d  val: %d  skipped: %d",
                len(train_run_pairs), len(val_run_pairs), skipped)

    # ------------------------------------------------------------------
    # HomoSampler
    # ------------------------------------------------------------------
    target_feat_idx = measurement_cols.index(target_col)
    sampler = HomoSampler(
        meas_scaled           = meas_scaled,
        grid_icond2_scaled    = grid_icond2_scaled,
        train_run_pairs       = train_run_pairs,
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
        grid_ecmwf_scaled     = grid_ecmwf_scaled,
        ecmwf_coords          = ecmwf_coords,
        k_ecmwf               = next_n_ecmwf,
        aggregate_nwp         = aggregate_nwp,
    )
    logger.info("in_channels: %d  (aggregate_nwp=%s)", sampler.in_channels, aggregate_nwp)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    M_meas_only    = len(measurement_cols)
    nwp_out_dim    = mcfg.get("nwp_out_dim", 32) if nwp_nodes else 0
    nwp_heads      = mcfg.get("nwp_heads", 4)     if nwp_nodes else 4
    # When nwp_nodes=True, sampler disaggregates ECMWF too (aggregate_nwp=False);
    # ECMWF channels pass through input_proj directly alongside GATv2 output.
    ecmwf_channels    = (sampler.in_channels - M_meas_only - mcfg.get("next_n_icond2", next_n_icond2) * I2) if nwp_nodes else 0
    in_channels_model = (M_meas_only + nwp_out_dim + ecmwf_channels) if nwp_nodes else sampler.in_channels
    logger.info(
        "nwp_nodes=%s  input_proj_in=%d  (M=%d nwp_out=%d ecmwf=%d  aggregate_nwp=%s)",
        nwp_nodes, in_channels_model, M_meas_only, nwp_out_dim, ecmwf_channels, aggregate_nwp,
    )

    model = GraphWaveNetModel(
        in_channels      = in_channels_model,
        static_dim       = 6,
        hidden           = mcfg.get("hidden", 64),
        n_blocks         = mcfg.get("n_blocks", 8),
        K_hop      = mcfg.get("K_hop", 2),
        emb_dim          = mcfg.get("emb_dim", 64),
        graph_alpha      = mcfg.get("graph_alpha", 3.0),
        kernel_size      = mcfg.get("kernel_size", 2),
        dropout          = mcfg.get("dropout", 0.1),
        history_length   = H,
        forecast_horizon = F_h,
        nwp_nodes        = nwp_nodes,
        nwp_feat_dim     = I2,
        k_nwp            = mcfg.get("next_n_icond2", next_n_icond2),
        nwp_out_dim      = nwp_out_dim,
        nwp_heads        = nwp_heads,
        M                = M_meas_only,
    ).to(device)
    logger.info("GraphWaveNet parameters: %s", f"{model.count_parameters():,}")
    logger.info("Hyperparameters:\n%s", yaml.dump(mcfg, default_flow_style=False))

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = mcfg.get("lr", 3e-4),
        weight_decay = mcfg.get("weight_decay", 1e-5),
    )
    max_epochs     = mcfg.get("max_epochs", 200)
    scheduler_type = mcfg.get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5,
        )

    patience        = mcfg.get("patience", 15)
    grad_accum      = mcfg.get("grad_accum", 4)
    steps_per_epoch = len(train_run_pairs)
    max_grad_norm   = mcfg.get("gradient_clip", 5.0)
    horizon_decay   = mcfg.get("horizon_decay", 0.95)

    w = _horizon_weights(F_h, horizon_decay, device)

    # Physical-unit scaling for RMSE / MAE display
    target_scale = float(meas_scaler.std_[target_feat_idx] + meas_scaler.eps)
    target_mean  = float(meas_scaler.mean_[target_feat_idx])

    tb_dir = Path("runs") / Path(model_name).stem
    writer = SummaryWriter(log_dir=str(tb_dir)) if SummaryWriter is not None else None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val    = float("inf")
    no_improve  = 0
    history: list[dict] = []
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        train_loss, train_rmse, train_r2 = _train_epoch(
            model, sampler, optimizer, device,
            steps=steps_per_epoch, w=w,
            target_scale=target_scale,
            target_mean=target_mean,
            grad_accum=grad_accum, max_grad_norm=max_grad_norm,
        )
        val_loss, val_rmse, val_r2 = _val_epoch(
            model, sampler, device, w, target_scale, target_mean,
        )

        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_rmse)

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss, "train_rmse": train_rmse, "train_r2": train_r2,
            "val_loss":   val_loss,   "val_rmse":   val_rmse,   "val_r2":   val_r2,
            "lr":         optimizer.param_groups[0]["lr"],
        })
        logger.info(
            "Epoch %3d/%d  train loss=%.4f RMSE=%.4f R²=%.4f | val loss=%.4f RMSE=%.4f R²=%.4f | lr=%.2e",
            epoch, max_epochs,
            train_loss, train_rmse, train_r2,
            val_loss, val_rmse, val_r2,
            optimizer.param_groups[0]["lr"],
        )
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/rmse", train_rmse, epoch)
            writer.add_scalar("train/r2",   train_r2,   epoch)
            writer.add_scalar("val/loss",   val_loss,   epoch)
            writer.add_scalar("val/rmse",   val_rmse,   epoch)
            writer.add_scalar("val/r2",     val_r2,     epoch)

        if val_rmse < best_val:
            best_val = val_rmse
            no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.debug("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    if writer is not None:
        writer.close()
    logger.debug("Training complete. Best val RMSE: %.4f  Model: %s", best_val, model_path)

    # ------------------------------------------------------------------
    # Optional post-training evaluation (--eval)
    # ------------------------------------------------------------------
    eval_df: pd.DataFrame | None = None
    if args.eval:
        logger.info("Running per-station evaluation on val run pairs …")
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt)
        eval_df = evaluate_homo_model(
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
        )
        cols = ["mae", "rmse", "r2", "skill", "skill_nwp"]
        tbl  = eval_df.set_index("station_id")[cols]
        mean_row = tbl.mean().to_frame().T
        mean_row.index = ["MEAN"]
        tbl = pd.concat([tbl, mean_row])
        logger.info("Per-station evaluation:\n%s", tbl.to_string(float_format="%.4f"))

    # ------------------------------------------------------------------
    # Save pkl
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_path  = Path("results") / f"{Path(model_name).stem}_{timestamp}.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "model_name":    Path(model_name).stem,
                "model_path":    str(model_path),
                "architecture":  "wavenet",
                "mode":          "test" if args.test_mode else "dev",
                "config":        mcfg,
                "train_ids":     train_ids,
                "val_ids":       val_ids,
                "n_train":       N_train,
                "n_val":         N_val,
                "history":       history,
                "best_val_rmse": best_val,
                "stopped_epoch": history[-1]["epoch"],
                "evaluation":    eval_df,
            },
            f,
        )
    logger.info("Results saved → %s", pkl_path)


if __name__ == "__main__":
    main()
