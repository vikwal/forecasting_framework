"""
hpo_mtgnn.py — Hyperparameter optimisation for the MTGNN baseline model.

Mirrors the design of hpo_dcrnn.py:
  - Optuna study stored in SQLite (studies/) or PostgreSQL (OPTUNA_STORAGE env)
  - HP bounds defined in the config YAML under mtgnn.hpo.params
  - Data loaded once outside the trial loop for efficiency
  - Single objective: minimise mean val RMSE (m/s) across CV folds

Usage
-----
    python geostatistics/hpo_mtgnn.py \
        --config configs/mtgnn/config_wind_mtgnn.yaml [--suffix v1] [--gpu 1]

Config
------
Add an ``hpo`` block under the ``mtgnn`` section in your YAML:

    mtgnn:
      hpo:
        trials: 50
        studies_path: studies/
        max_epochs_per_trial: 100
        patience_per_trial: 10
        steps_per_epoch_per_trial: 100
        n_val_stations: 40
        n_folds: 3
        min_train_date: '2023-01-01'
        pruner: median             # or 'none'
        pruner_n_startup_trials: 5
        pruner_n_warmup_steps: 0
        params:
          hidden:
            type: categorical
            choices: [32, 64, 128]
          n_layers:
            type: int
            low: 3
            high: 6
          K_hop:
            type: int
            low: 1
            high: 3
          lr:
            type: float
            low: 1.0e-5
            high: 1.0e-3
            log: true
          weight_decay:
            type: float
            low: 1.0e-6
            high: 1.0e-3
            log: true
          dropout:
            type: float
            low: 0.0
            high: 0.4
          horizon_decay:
            type: float
            low: 0.80
            high: 1.00
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from optuna.samplers import TPESampler

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
from geostatistics.train_mtgnn import (
    _train_epoch,
    _val_epoch,
    _horizon_weights,
)
from geostatistics.homo_sampler import HomoSampler
from geostatistics.mtgnn import MTGNNModel
from geostatistics.stgnn.utils.normalization import StandardScaler
from geostatistics.train_dcrnn import encode_circular_measurements, apply_dir_encoding
from utils.data_cache import GNNCache


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HP sampling
# ---------------------------------------------------------------------------

def _suggest(trial: optuna.Trial, name: str, spec: dict):
    ptype = spec["type"]
    if ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if ptype == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if ptype == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    raise ValueError(f"Unknown HPO param type {ptype!r} for '{name}'")


def sample_hyperparameters(trial: optuna.Trial, hpo_params: dict) -> dict:
    return {name: _suggest(trial, name, spec) for name, spec in hpo_params.items()}


# ---------------------------------------------------------------------------
# Run pair construction
# ---------------------------------------------------------------------------

def _build_all_run_pairs(
    run_times: pd.DatetimeIndex,
    timestamps: pd.DatetimeIndex,
    meas_nan_any: np.ndarray,
    test_start: pd.Timestamp,
    H: int,
    F_h: int,
    freq_h: float = 1.0,
) -> list[tuple[int, int, int]]:
    T = len(timestamps)
    ts_lookup = pd.Series(np.arange(T), index=timestamps)
    pairs: list[tuple[int, int, int]] = []
    skipped = 0

    for r_curr in range(len(run_times)):
        t_run = run_times[r_curr]
        if t_run >= test_start:
            continue
        if t_run not in ts_lookup.index:
            skipped += 1
            continue
        t_run_abs = int(ts_lookup[t_run])
        if t_run_abs < H or t_run_abs + F_h > T:
            skipped += 1
            continue
        t_hist_target = t_run - pd.Timedelta(hours=H * freq_h)
        diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
        r_hist = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600:
            skipped += 1
            continue
        if meas_nan_any[t_run_abs - H: t_run_abs + F_h].any():
            skipped += 1
            continue
        pairs.append((r_curr, r_hist, t_run_abs))

    logger.info("All pre-test run pairs: %d  skipped: %d", len(pairs), skipped)
    return pairs


# ---------------------------------------------------------------------------
# Inline training loop (with curriculum learning)
# ---------------------------------------------------------------------------

def _fit_model(
    model: MTGNNModel,
    sampler: HomoSampler,
    device: torch.device,
    target_scale: float,
    target_mean: float,
    F_h: int,
    max_epochs: int,
    patience: int,
    steps_per_epoch: int,
    lr: float,
    weight_decay: float,
    horizon_decay: float,
    scheduler_type: str,
    grad_accum: int,
    max_grad_norm: float,
    cl_period: int,
    ckpt_path: Path,
) -> tuple[float, int]:
    """Train MTGNN with curriculum learning; return (best_val_rmse, stopped_epoch)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5,
        )

    w = _horizon_weights(F_h, horizon_decay, device)
    best_val   = math.inf
    no_improve = 0
    prev_cl    = 0
    stopped    = max_epochs

    for epoch in range(1, max_epochs + 1):
        cl_steps = min((epoch - 1) // cl_period + 1, F_h)

        if cl_steps > prev_cl:
            no_improve = 0
            best_val   = math.inf
            prev_cl    = cl_steps

        _train_epoch(
            model, sampler, optimizer, device,
            steps=steps_per_epoch, cl_steps=cl_steps, w=w,
            target_scale=target_scale, target_mean=target_mean,
            grad_accum=grad_accum, max_grad_norm=max_grad_norm,
        )
        _, val_rmse, _ = _val_epoch(
            model, sampler, device, w, target_scale, target_mean, cl_steps=cl_steps,
        )

        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_rmse)

        if val_rmse < best_val:
            best_val   = val_rmse
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                stopped = epoch
                break

    return best_val, stopped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HPO for MTGNN")
    parser.add_argument("--config",  required=True, help="Path to YAML config")
    parser.add_argument("--suffix",  default="",    help="Log-file suffix")
    parser.add_argument("--gpu",     type=int, default=None, help="GPU index")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Load data then exit (no trials)")
    args = parser.parse_args()

    config_path = Path(args.config)
    config_stem = config_path.stem.replace("config_", "")
    suffix      = f"_{args.suffix}" if args.suffix else ""

    cfg      = load_yaml(args.config)
    data_cfg = cfg["data"]
    mcfg     = cfg.get("mtgnn", {})
    hpo_cfg  = mcfg.get("hpo", {})

    if not hpo_cfg:
        print("ERROR: No 'hpo' block found under 'mtgnn' in config.")
        sys.exit(1)

    freq     = data_cfg.get("freq", "1h")
    _fmap    = {"1h": 1.0, "1H": 1.0, "30min": 0.5, "30T": 0.5, "15min": 0.25, "15T": 0.25}
    freq_h   = _fmap.get(freq, 1.0)
    F_h_base = mcfg.get("forecast_horizon", 48)

    study_name = f"cl_m-mtgnn_out-{F_h_base}_freq-{freq}_{config_stem}"

    _setup_logging(Path("logs") / f"hpo_mtgnn_{config_stem}{suffix}.log")
    logger.info("=" * 70)
    logger.info("HPO MTGNN — config: %s  study: %s", args.config, study_name)
    logger.info("=" * 70)

    hpo_params         = hpo_cfg.get("params", {})
    n_trials           = hpo_cfg.get("trials", 30)
    studies_path       = hpo_cfg.get("studies_path", "studies/")
    max_epochs         = hpo_cfg.get("max_epochs_per_trial", 100)
    patience           = hpo_cfg.get("patience_per_trial", 10)
    n_val_stations     = hpo_cfg.get("n_val_stations", None)
    n_folds            = hpo_cfg.get("n_folds", 3)
    min_train_date     = hpo_cfg.get("min_train_date")
    pruner_type        = hpo_cfg.get("pruner", "median")
    pruner_n_startup   = hpo_cfg.get("pruner_n_startup_trials", 5)
    pruner_n_warmup    = hpo_cfg.get("pruner_n_warmup_steps", 0)

    if pruner_type == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_n_startup,
            n_warmup_steps=pruner_n_warmup,
        )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if args.gpu is not None and torch.cuda.is_available():
        device_str = f"cuda:{args.gpu}"
    device = torch.device(device_str)
    logger.info("Device: %s", device)

    os.makedirs(studies_path, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ── Feature config ────────────────────────────────────────────────────────
    icond2_features  = mcfg.get("icond2_features") or []
    i2_mode          = mcfg.get("icond2_feature_mode", "absolute")
    e2_mode          = mcfg.get("ecmwf_feature_mode",  "absolute")
    measurement_cols = mcfg.get("measurement_features") or []
    target_col       = mcfg.get("target_col", "wind_speed")
    if target_col not in measurement_cols:
        raise ValueError(f"target_col '{target_col}' not in measurement_features")

    run_hours     = tuple(mcfg.get("icond2_run_hours", [6, 9, 12, 15]))
    next_n_icond2 = mcfg.get("next_n_icond2", 4)
    n_workers     = mcfg.get("n_workers", 8)
    nwp_path      = data_cfg.get("nwp_path")
    data_path     = data_cfg["path"]
    H             = mcfg.get("history_length", 48)
    F_h           = mcfg.get("forecast_horizon", 48)
    use_case      = data_cfg.get("use_case", "wind")

    # ── Station IDs ───────────────────────────────────────────────────────────
    train_ids = [str(s) for s in data_cfg["files"]]
    val_ids   = [str(s) for s in data_cfg["val_files"]]
    all_ids   = train_ids + val_ids
    N_train   = len(train_ids)
    N_val     = len(val_ids)
    logger.info("Stations — train: %d  val: %d", N_train, N_val)

    test_start = data_cfg.get("test_start")
    test_end   = data_cfg.get("test_end")
    run_cutoff = pd.Timestamp(test_end, tz="UTC") if test_end else None
    test_start_dt = pd.Timestamp(test_start, tz="UTC")

    # ── Static config settings (must be known before cache key) ───────────────
    next_n_ecmwf   = mcfg.get("next_n_ecmwf", 0)
    ecmwf_features = mcfg.get("ecmwf_features") or []
    nwp_nodes      = mcfg.get("nwp_nodes", False)
    aggregate_nwp  = False if nwp_nodes else mcfg.get("aggregate_nwp", True)
    if nwp_nodes:
        logger.info("nwp_nodes=True — forcing aggregate_nwp=False")

    # Max-Bound-Trick: when next_n_icond2 is an HPO param, load data once with the max
    # bound so all trials share the pre-loaded arrays — no reload per trial.
    _k_nwp_hpo = hpo_params.get("next_n_icond2")
    if _k_nwp_hpo:
        next_n_icond2 = max(next_n_icond2, _k_nwp_hpo.get("high", next_n_icond2))
        logger.info("next_n_icond2 in HPO — loading %d grid points (max bound)", next_n_icond2)
    _next_n_ecmwf_hpo = hpo_params.get("next_n_ecmwf")
    if _next_n_ecmwf_hpo:
        next_n_ecmwf = max(next_n_ecmwf, _next_n_ecmwf_hpo.get("high", next_n_ecmwf))
        logger.info("next_n_ecmwf in HPO — loading %d ECMWF grid points (max bound)", next_n_ecmwf)

    # ── Cache setup ───────────────────────────────────────────────────────────
    cache_cfg = cfg.get("cache", {})
    use_cache = cache_cfg.get("use_cache", True)
    cache_dir = cache_cfg.get("cache_dir", "data_cache/gnns")
    gnn_cache = GNNCache(cache_dir) if use_cache else None
    _key_cfg  = {"data": cfg["data"], "dcrnn": {
        "icond2_features":    list(icond2_features),
        "ecmwf_features":     list(ecmwf_features),
        "measurement_features": list(measurement_cols),
        "next_n_icond2":      next_n_icond2,
        "next_n_ecmwf":       next_n_ecmwf,
        "icond2_run_hours":   list(run_hours),
        "use_altitude_diff":  False,
    }}
    cache_key = GNNCache.make_key(_key_cfg) if use_cache else None
    if use_cache:
        logger.info("GNNCache key: %s  dir: %s", cache_key, cache_dir)

    M_meas = len(measurement_cols)
    I2     = len(icond2_features)

    if use_cache and gnn_cache.exists(cache_key):
        # ── Fast path: load from cache ────────────────────────────────────────
        logger.info("GNNCache HIT — loading arrays (grid_icond2_runs: RAM, rest: mmap) …")
        _arrays = gnn_cache.load_arrays(
            cache_key,
            names=["grid_icond2_runs", "meas_raw", "grid_ecmwf_raw"],
            mmap=True,
            mmap_overrides={"grid_icond2_runs": False},
        )
        derived = gnn_cache.load_derived(cache_key)

        grid_icond2_runs = _arrays["grid_icond2_runs"]
        meas_raw         = _arrays["meas_raw"]
        _ecmwf_arr       = _arrays["grid_ecmwf_raw"]
        grid_ecmwf_raw   = _ecmwf_arr if _ecmwf_arr.shape[1] > 0 else None

        timestamps    = derived["timestamps"]
        run_times     = derived["run_times"]
        lats          = derived["lats"]
        lons          = derived["lons"]
        alts          = derived["alts"]
        icond2_coords = derived["icond2_coords"]
        _ec           = derived.get("ecmwf_coords", np.empty((0, 2), dtype=np.float32))
        ecmwf_coords  = _ec if _ec.shape[0] > 0 else None

        T       = len(timestamps)
        R       = len(run_times)
        n_leads = grid_icond2_runs.shape[1]

        _meas_nan_any = np.isnan(meas_raw).any(axis=(1, 2))
        all_run_pairs = _build_all_run_pairs(
            run_times, timestamps, _meas_nan_any, test_start_dt, H, F_h, freq_h,
        )
        logger.info(
            "Loaded from cache — T=%d  R=%d  N_igrid=%d  all_pairs=%d",
            T, R, len(icond2_coords), len(all_run_pairs),
        )
    else:
        # ── Slow path: load raw data, then write cache ────────────────────────
        logger.info("GNNCache MISS — loading raw data …")

        logger.info("Loading station measurements …")
        meas_raw, timestamps = load_station_measurements(
            data_path, all_ids, cols=measurement_cols, freq=freq,
        )

        if run_cutoff is not None:
            meas_cutoff = run_cutoff + pd.Timedelta(days=2)
            cut_idx = int(np.searchsorted(timestamps, meas_cutoff, side="right"))
            meas_raw   = meas_raw[:cut_idx]
            timestamps = timestamps[:cut_idx]
        T = len(timestamps)
        logger.info("Timestamps: %d  (%s … %s)", T, timestamps[0], timestamps[-1])

        rk_pred = None
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

        _meas_nan_any = np.isnan(meas_raw).any(axis=(1, 2))

        meta_path = data_cfg.get("stations_master")
        lats, lons, alts = load_station_metadata(data_path, all_ids, meta_path=meta_path)

        logger.info("Loading ICON-D2 runs (next_n_grid=%d) …", next_n_icond2)
        if use_case == "solar":
            from geostatistics.solar_preprocessing import load_solar_sl_runs
            run_times, icond2_coords, grid_icond2_runs, _ = load_solar_sl_runs(
                nwp_path=nwp_path, station_ids=all_ids,
                station_coords=np.stack([lats, lons], axis=1),
                features=icond2_features, run_hours=run_hours,
                next_n_grid=next_n_icond2, n_workers=n_workers,
                cutoff=run_cutoff, freq_h=freq_h,
            )
        else:
            run_times, icond2_coords, grid_icond2_runs, _ = load_icond2_ml_runs(
                nwp_path=nwp_path, station_ids=all_ids,
                station_coords=np.stack([lats, lons], axis=1),
                features=icond2_features, run_hours=run_hours,
                next_n_grid=next_n_icond2, n_workers=n_workers,
                cutoff=run_cutoff,
            )
        R       = len(run_times)
        n_leads = grid_icond2_runs.shape[1]
        logger.info("ICON-D2: %d grid nodes  %d runs  %d leads", len(icond2_coords), R, n_leads)

        grid_ecmwf_raw: np.ndarray | None = None
        ecmwf_coords:   np.ndarray | None = None

        if next_n_ecmwf > 0 and ecmwf_features:
            ecmwf_path = data_cfg.get("ecmwf_path")
            if ecmwf_path and os.path.exists(ecmwf_path):
                logger.info("Loading ECMWF NWP (%d feat, k=%d) …", len(ecmwf_features), next_n_ecmwf)
                _, ecmwf_coords, grid_ecmwf_raw, _ = load_ecmwf_parquet_at_stations_and_grid(
                    parquet_path=ecmwf_path, station_lats=lats, station_lons=lons,
                    features=ecmwf_features, timestamps=timestamps,
                    next_n_grid_per_station=next_n_ecmwf,
                )
                logger.info("ECMWF: %d grid nodes", len(ecmwf_coords))
            else:
                logger.warning("next_n_ecmwf=%d but ecmwf_path missing — ECMWF disabled", next_n_ecmwf)
                next_n_ecmwf = 0

        all_run_pairs = _build_all_run_pairs(
            run_times, timestamps, _meas_nan_any, test_start_dt, H, F_h, freq_h,
        )

        if use_cache:
            logger.info("GNNCache — writing cache (key=%s) …", cache_key)
            gnn_cache.save(
                cache_key,
                arrays={
                    "grid_icond2_runs": np.array(grid_icond2_runs),
                    "meas_raw":         np.array(meas_raw),
                    "grid_ecmwf_raw":   (np.array(grid_ecmwf_raw) if grid_ecmwf_raw is not None
                                         else np.empty((T, 0, 0), dtype=np.float32)),
                },
                derived={
                    "timestamps":    timestamps,
                    "run_times":     run_times,
                    "lats":          lats,
                    "lons":          lons,
                    "alts":          alts,
                    "icond2_coords": icond2_coords,
                    "ecmwf_coords":  (ecmwf_coords if ecmwf_coords is not None
                                      else np.empty((0, 2), dtype=np.float32)),
                    "all_run_pairs": all_run_pairs,
                    "meas_nan_any":  _meas_nan_any,
                },
            )
            logger.info("GNNCache — cache written.")

    # ── Circular / dir_in_deg encoding (applied once, after cache) ───────────
    meas_raw, measurement_cols = encode_circular_measurements(meas_raw, measurement_cols)
    M_meas = len(measurement_cols)

    _icond2_hpo_mode = "icond2_feature_mode" in hpo_params
    _ecmwf_hpo_mode  = "ecmwf_feature_mode"  in hpo_params

    if not _icond2_hpo_mode and i2_mode == "dir_in_deg":
        grid_icond2_runs, icond2_features = apply_dir_encoding(grid_icond2_runs, icond2_features)
        I2 = len(icond2_features)
    if not _ecmwf_hpo_mode and e2_mode == "dir_in_deg" and grid_ecmwf_raw is not None:
        grid_ecmwf_raw, ecmwf_features = apply_dir_encoding(grid_ecmwf_raw, ecmwf_features)

    if _icond2_hpo_mode:
        _i2_enc, _i2_feat_enc = apply_dir_encoding(np.array(grid_icond2_runs), list(icond2_features))
        _I2_enc = len(_i2_feat_enc)
    else:
        _i2_enc, _i2_feat_enc, _I2_enc = grid_icond2_runs, icond2_features, I2
    if _ecmwf_hpo_mode and grid_ecmwf_raw is not None:
        _ecmwf_enc, _ = apply_dir_encoding(np.array(grid_ecmwf_raw), list(ecmwf_features))
    else:
        _ecmwf_enc = grid_ecmwf_raw

    if args.preprocess_only:
        logger.info("--preprocess-only done. Exiting.")
        return

    # ── HPO val stations ──────────────────────────────────────────────────────
    n_hpo_val = min(n_val_stations, N_val) if n_val_stations else N_val
    hpo_val_station_indices = list(range(N_train, N_train + n_hpo_val))
    train_station_indices   = list(range(N_train))
    logger.info("HPO val stations: %d of %d", n_hpo_val, N_val)

    target_feat_idx = measurement_cols.index(target_col)

    # ── CV fold boundaries (expanding window) ─────────────────────────────────
    if min_train_date:
        min_train_dt = pd.Timestamp(min_train_date, tz="UTC")
    else:
        min_train_dt = timestamps[0]

    foldable_secs = (test_start_dt - min_train_dt).total_seconds()
    seg_secs      = foldable_secs / (n_folds + 1)
    fold_splits: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for k in range(n_folds):
        val_start = min_train_dt + pd.Timedelta(seconds=seg_secs * (k + 1))
        val_end   = min_train_dt + pd.Timedelta(seconds=seg_secs * (k + 2))
        fold_splits.append((val_start, val_end))
        logger.info(
            "Fold %d — train: [start, %s)  val: [%s, %s)",
            k, val_start.date(), val_start.date(), val_end.date(),
        )

    def _fold_pairs(val_start: pd.Timestamp, val_end: pd.Timestamp):
        tr, va = [], []
        for r_curr, r_hist, t_run_abs in all_run_pairs:
            t = timestamps[t_run_abs]
            if   t < val_start: tr.append((r_curr, r_hist, t_run_abs))
            elif t < val_end:   va.append((r_curr, r_hist, t_run_abs))
        return tr, va

    # ── Objective ─────────────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        sampled = sample_hyperparameters(trial, hpo_params)
        logger.info("Trial %d — hyperparameters: %s", trial.number, sampled)

        trial_cfg = copy.deepcopy(mcfg)
        trial_cfg.update(sampled)

        _trial_i2_mode  = trial_cfg.get("icond2_feature_mode", i2_mode)
        _trial_e2_mode  = trial_cfg.get("ecmwf_feature_mode",  e2_mode)
        _t_icond2 = _i2_enc    if (_icond2_hpo_mode and _trial_i2_mode == "dir_in_deg") else grid_icond2_runs
        _t_I2     = _I2_enc    if (_icond2_hpo_mode and _trial_i2_mode == "dir_in_deg") else I2
        _t_ecmwf  = _ecmwf_enc if (_ecmwf_hpo_mode  and _trial_e2_mode == "dir_in_deg") else grid_ecmwf_raw

        hidden        = int(trial_cfg.get("hidden", 64))
        n_layers      = int(trial_cfg.get("n_layers", 4))
        K_hop         = int(trial_cfg.get("K_hop", 2))
        beta          = float(trial_cfg.get("beta", 0.05))
        emb_dim       = int(trial_cfg.get("emb_dim", 64))
        graph_alpha   = float(trial_cfg.get("graph_alpha", 3.0))
        dropout       = float(trial_cfg.get("dropout", 0.1))
        lr            = float(trial_cfg.get("lr", 3e-4))
        weight_decay  = float(trial_cfg.get("weight_decay", 1e-5))
        horizon_decay = float(trial_cfg.get("horizon_decay", 0.95))
        grad_accum    = int(trial_cfg.get("grad_accum", 1))
        max_grad_norm = float(trial_cfg.get("gradient_clip", 5.0))
        scheduler_type = str(trial_cfg.get("scheduler", "cosine"))
        cl_period     = int(trial_cfg.get("cl_period", 3))
        next_n_icond2_trial   = int(trial_cfg.get("next_n_icond2", next_n_icond2))
        next_n_ecmwf_trial    = int(trial_cfg.get("next_n_ecmwf",  next_n_ecmwf))
        next_n_nbr_trial = trial_cfg.get("next_n_neighbors", mcfg.get("next_n_neighbors", None))
        if next_n_nbr_trial is not None:
            next_n_nbr_trial = int(next_n_nbr_trial)
        topk_graph_trial = trial_cfg.get("topk_graph", mcfg.get("topk_graph", None))
        if topk_graph_trial is not None:
            topk_graph_trial = int(topk_graph_trial)

        fold_rmses: list[float] = []

        for fold_idx, (val_start, val_end) in enumerate(fold_splits):
            tr_pairs, va_pairs = _fold_pairs(val_start, val_end)
            if not tr_pairs or not va_pairs:
                logger.warning(
                    "Trial %d fold %d: empty split (train=%d val=%d), skipping",
                    trial.number, fold_idx, len(tr_pairs), len(va_pairs),
                )
                continue

            # Per-fold scalers: fit only on fold-train window + train stations
            fold_t = int(np.searchsorted(timestamps, val_start, side="left"))
            fold_r = run_times < val_start

            meas_scaler = StandardScaler()
            meas_scaler.fit(meas_raw[:fold_t, :N_train].reshape(-1, M_meas))
            meas_scaled = meas_scaler.transform(
                meas_raw.reshape(-1, M_meas)
            ).reshape(T, len(all_ids), M_meas)

            i2_scaler = StandardScaler()
            i2_scaler.fit(_t_icond2[fold_r].reshape(-1, _t_I2))
            grid_icond2_scaled = i2_scaler.transform(
                _t_icond2.reshape(-1, _t_I2)
            ).reshape(R, n_leads, len(icond2_coords), _t_I2)

            grid_ecmwf_scaled: np.ndarray | None = None
            if next_n_ecmwf > 0 and _t_ecmwf is not None:
                E2 = _t_ecmwf.shape[2]
                e2_scaler = StandardScaler()
                e2_scaler.fit(_t_ecmwf[:fold_t].reshape(-1, E2))
                grid_ecmwf_scaled = e2_scaler.transform(
                    _t_ecmwf.reshape(-1, E2)
                ).reshape(T, len(ecmwf_coords), E2)

            fold_target_scale = float(meas_scaler.std_[target_feat_idx] + meas_scaler.eps)
            fold_target_mean  = float(meas_scaler.mean_[target_feat_idx])

            sampler = HomoSampler(
                meas_scaled           = meas_scaled,
                grid_icond2_scaled    = grid_icond2_scaled,
                train_run_pairs       = tr_pairs,
                val_run_pairs         = va_pairs,
                train_station_indices = train_station_indices,
                val_station_indices   = hpo_val_station_indices,
                lats                  = lats,
                lons                  = lons,
                alts                  = alts,
                icond2_coords         = icond2_coords,
                history_length        = H,
                forecast_horizon      = F_h,
                target_feat_idx       = target_feat_idx,
                k_nwp                 = next_n_icond2_trial,
                min_target_stations   = mcfg.get("min_target_stations", 1),
                max_target_stations   = mcfg.get("max_target_stations", 10),
                max_neighbor_stations = mcfg.get("max_neighbor_stations", 60),
                next_n_neighbors      = next_n_nbr_trial,
                grid_ecmwf_scaled     = grid_ecmwf_scaled,
                ecmwf_coords          = ecmwf_coords,
                next_n_ecmwf               = next_n_ecmwf_trial,
                aggregate_nwp         = aggregate_nwp,
            )

            M_meas_only     = len(measurement_cols)
            nwp_out_dim     = int(trial_cfg.get("nwp_out_dim", mcfg.get("nwp_out_dim", 32))) if nwp_nodes else 0
            nwp_heads_v     = int(trial_cfg.get("nwp_heads",   mcfg.get("nwp_heads", 4)))    if nwp_nodes else 4
            ecmwf_channels  = (sampler.in_channels - M_meas_only - next_n_icond2_trial * _t_I2) if nwp_nodes else 0
            in_ch_model     = (M_meas_only + nwp_out_dim + ecmwf_channels) if nwp_nodes else sampler.in_channels

            model = MTGNNModel(
                in_channels      = in_ch_model,
                static_dim       = 6,
                hidden           = hidden,
                n_layers         = n_layers,
                K_hop            = K_hop,
                beta             = beta,
                emb_dim          = emb_dim,
                graph_alpha      = graph_alpha,
                dropout          = dropout,
                history_length   = H,
                forecast_horizon = F_h,
                nwp_nodes        = nwp_nodes,
                nwp_feat_dim     = _t_I2,
                k_nwp            = next_n_icond2_trial,
                nwp_out_dim      = nwp_out_dim,
                nwp_heads        = nwp_heads_v,
                M                = M_meas_only,
                topk_graph       = topk_graph_trial,
            ).to(device)

            ckpt = Path(f"models/_hpo_mtgnn_trial{trial.number}_fold{fold_idx}.pt")

            best_rmse, stopped = _fit_model(
                model          = model,
                sampler        = sampler,
                device         = device,
                target_scale   = fold_target_scale,
                target_mean    = fold_target_mean,
                F_h            = F_h,
                max_epochs     = max_epochs,
                patience       = patience,
                steps_per_epoch= len(tr_pairs),
                lr             = lr,
                weight_decay   = weight_decay,
                horizon_decay  = horizon_decay,
                scheduler_type = scheduler_type,
                grad_accum     = grad_accum,
                max_grad_norm  = max_grad_norm,
                cl_period      = cl_period,
                ckpt_path      = ckpt,
            )

            if ckpt.exists():
                ckpt.unlink()

            fold_rmses.append(best_rmse)
            logger.info(
                "Trial %d fold %d — best_val_rmse=%.4f  stopped=%d",
                trial.number, fold_idx, best_rmse, stopped,
            )

            trial.report(float(np.mean(fold_rmses)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_rmses:
            raise ValueError("All folds were empty — check min_train_date / data range.")

        mean_rmse = float(np.mean(fold_rmses))
        logger.info(
            "Trial %d done — fold RMSEs: %s  mean=%.4f",
            trial.number,
            [f"{v:.4f}" for v in fold_rmses],
            mean_rmse,
        )
        return mean_rmse

    # ── Create / resume study ─────────────────────────────────────────────────
    storage_url = os.environ.get("OPTUNA_STORAGE")
    if storage_url:
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=60,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600},
        )
        logger.info("Optuna storage: PostgreSQL (OPTUNA_STORAGE)")
    else:
        db_path = Path(studies_path) / f"hpo_mtgnn_{config_stem}{suffix}.db"
        storage = f"sqlite:///{db_path}"
        logger.info("Optuna storage: SQLite → %s", db_path)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=TPESampler(),
        pruner=pruner,
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(n_trials - completed, 0)
    logger.info(
        "Study loaded — %d completed, %d remaining of %d total",
        completed, remaining, n_trials,
    )

    while remaining > 0:
        try:
            study.optimize(objective, n_trials=remaining, catch=(Exception,))
            break
        except Exception as exc:
            logger.warning("study.optimize interrupted (%s) — reconnecting …", exc)
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            remaining = max(n_trials - completed, 0)
            if remaining == 0:
                break

    # ── Results ───────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("HPO COMPLETE")
    logger.info("Best trial:       #%d", study.best_trial.number)
    logger.info("Best val RMSE:    %.6f m/s", study.best_value)
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info("  %-30s %s", k, v)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
