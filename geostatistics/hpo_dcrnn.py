"""
hpo_dcrnn.py — Hyperparameter optimisation for the DCRNN Seq2Seq model.

Mirrors the design of hpo_cl.py:
  - Optuna study stored in SQLite (studies/) or PostgreSQL (OPTUNA_STORAGE env)
  - HP bounds defined in the config YAML under dcrnn.hpo.params
  - Data loaded once outside the trial loop for efficiency
  - Single objective: minimise val_loss

Usage
-----
    python hpo_dcrnn.py --config configs/config_wind_stgcn.yaml [--suffix v1] [--gpu 1]

Config
------
Add an ``hpo`` block under the ``dcrnn`` section in your YAML:

    dcrnn:
      hpo:
        trials: 50
        studies_path: studies/
        max_epochs_per_trial: 60
        patience_per_trial: 8
        params:
          hidden:
            type: categorical
            choices: [32, 64, 128]
          lr:
            type: float
            low: 1.0e-5
            high: 1.0e-3
            log: true
          ...
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from utils.data_cache import GNNCache
from optuna.samplers import TPESampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]

from geostatistics.train_dcrnn import resolve_feature_mode, feature_indices
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
from geostatistics.dcrnn import DCRNNConfig, DCRNN
from geostatistics.dcrnn.training import DCRNNTrainer
from geostatistics.stgnn import HeterogeneousGraphBuilder
from geostatistics.stgnn.training.sampler import TrainingSampler
from geostatistics.stgnn.utils.normalization import StandardScaler

import pandas as pd
import pyproj

_GEOD = pyproj.Geod(ellps="WGS84")

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
# HP sampling from config bounds
# ---------------------------------------------------------------------------

def _suggest(trial: optuna.Trial, name: str, spec: dict):
    """Sample one hyperparameter according to its type spec."""
    ptype = spec["type"]
    if ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if ptype == "int":
        return trial.suggest_int(name, spec["low"], spec["high"],
                                  step=spec.get("step", 1))
    if ptype == "float":
        return trial.suggest_float(name, spec["low"], spec["high"],
                                    log=spec.get("log", False))
    raise ValueError(f"Unknown HPO param type {ptype!r} for '{name}'")


def sample_hyperparameters(trial: optuna.Trial, hpo_params: dict) -> dict:
    """
    Sample all hyperparameters from their config specs.

    Special handling for nwp_heads / nwp_out_per_head:
      - ``nwp_heads`` is sampled first.
      - ``nwp_out_per_head`` is sampled independently.
      - ``nwp_out_dim = nwp_heads * nwp_out_per_head`` is derived (not sampled
        directly), mirroring the hpo_cl.py approach of using ``step=n_heads``
        to guarantee divisibility without trial pruning.
    """
    sampled: dict = {}
    for name, spec in hpo_params.items():
        sampled[name] = _suggest(trial, name, spec)

    # Derive nwp_out_dim from nwp_heads × nwp_out_per_head (if both present)
    if "nwp_heads" in sampled and "nwp_out_per_head" in sampled:
        sampled["nwp_out_dim"] = sampled.pop("nwp_out_per_head") * sampled["nwp_heads"]
    elif "nwp_out_per_head" in sampled:
        sampled.pop("nwp_out_per_head")   # can't derive without nwp_heads

    return sampled


# ---------------------------------------------------------------------------
# Build run pairs (identical logic to train_dcrnn.py)
# ---------------------------------------------------------------------------

def _build_run_pairs(
    run_times: pd.DatetimeIndex,
    timestamps: pd.DatetimeIndex,
    meas_nan_any: np.ndarray,
    split_time: pd.Timestamp,
    H: int,
    F_h: int,
):
    T = len(timestamps)
    ts_lookup = pd.Series(np.arange(T), index=timestamps)
    train_pairs, val_pairs = [], []
    skipped = 0

    for r_curr in range(len(run_times)):
        t_run = run_times[r_curr]
        if t_run not in ts_lookup.index:
            skipped += 1
            continue
        t_run_abs = int(ts_lookup[t_run])
        if t_run_abs < H or t_run_abs + F_h > T:
            skipped += 1
            continue
        t_hist_target = t_run - pd.Timedelta(hours=H)
        diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
        r_hist = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600:
            skipped += 1
            continue
        if meas_nan_any[t_run_abs - H : t_run_abs + F_h].any():
            skipped += 1
            continue
        pair = (r_curr, r_hist, t_run_abs)
        if t_run < split_time:
            train_pairs.append(pair)
        else:
            val_pairs.append(pair)

    logger.info("Run pairs — train: %d  val: %d  skipped: %d",
                len(train_pairs), len(val_pairs), skipped)
    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HPO for DCRNN")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--suffix", default="", help="Log-file suffix (no effect on study name)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (default: auto)")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Build cache then exit (used by run_hpo_dcrnn.sh before launching workers)")
    args = parser.parse_args()

    config_path = Path(args.config)
    config_stem = config_path.stem.replace("config_", "")
    suffix = f"_{args.suffix}" if args.suffix else ""

    cfg      = load_yaml(args.config)
    data_cfg = cfg["data"]
    dcrnn_cfg = cfg.get("dcrnn", {})

    freq       = data_cfg.get("freq", "1h")
    H_fore_tmp = dcrnn_cfg.get("forecast_horizon", 48)
    # Mirror hpo_cl.py naming: cl_m-{model}_out-{output_dim}_freq-{freq}_{config}{suffix}
    study_name = f"cl_m-dcrnn_out-{H_fore_tmp}_freq-{freq}_{config_stem}"

    _setup_logging(Path("logs") / f"hpo_dcrnn_{config_stem}{suffix}.log")
    logger.info("=" * 70)
    logger.info("HPO DCRNN — config: %s  study: %s", args.config, study_name)
    logger.info("=" * 70)

    hpo_cfg  = dcrnn_cfg.get("hpo", {})

    if not hpo_cfg:
        logger.error("No 'hpo' block found under 'dcrnn' in config. Aborting.")
        sys.exit(1)

    hpo_params      = hpo_cfg.get("params", {})
    n_trials        = hpo_cfg.get("trials", 30)
    studies_path    = hpo_cfg.get("studies_path", "studies/")
    max_epochs      = hpo_cfg.get("max_epochs_per_trial", 60)
    patience        = hpo_cfg.get("patience_per_trial", 8)
    use_tensorboard = hpo_cfg.get("tensorboard_monitoring", False)

    # Device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if args.gpu is not None and torch.cuda.is_available():
        device_str = f"cuda:{args.gpu}"
    device = torch.device(device_str)
    logger.info("Device: %s", device)

    os.makedirs(studies_path, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ── Feature config ───────────────────────────────────────────────────────
    # Always load the FULL feature set so per-trial subsets can be sliced cheaply.
    icond2_features_all = dcrnn_cfg["icond2_features"]   # full list from config
    ecmwf_features_all  = dcrnn_cfg["ecmwf_features"]
    icond2_features = icond2_features_all                # used for data loading
    ecmwf_features  = ecmwf_features_all
    measurement_cols = dcrnn_cfg["measurement_features"]
    target_col       = dcrnn_cfg["target_col"]
    H   = dcrnn_cfg.get("history_length", 48)
    F_h = dcrnn_cfg.get("forecast_horizon", 48)
    run_hours     = tuple(dcrnn_cfg.get("icond2_run_hours", [6, 9, 12, 15]))
    next_n_icond2 = dcrnn_cfg.get("next_n_icond2", 4)
    next_n_ecmwf  = dcrnn_cfg.get("next_n_ecmwf", 4)
    n_workers     = dcrnn_cfg.get("n_workers", 8)
    nwp_path      = data_cfg.get("nwp_path")
    data_path     = data_cfg["path"]

    if target_col not in measurement_cols:
        raise ValueError(f"target_col '{target_col}' not in measurement_features")

    # ── Station IDs ──────────────────────────────────────────────────────────
    train_ids = [str(s) for s in data_cfg["files"]]
    val_ids   = [str(s) for s in data_cfg["val_files"]]
    all_ids   = train_ids + val_ids
    N_train   = len(train_ids)
    N_val     = len(val_ids)
    logger.info("Stations — train: %d  val: %d", N_train, N_val)

    # ── Cache setup ───────────────────────────────────────────────────────────
    cache_cfg  = cfg.get("cache", {})
    use_cache  = cache_cfg.get("use_cache", True)
    cache_dir  = cache_cfg.get("cache_dir", "data_cache/gnns")
    gnn_cache  = GNNCache(cache_dir) if use_cache else None
    cache_key  = GNNCache.make_key(cfg) if use_cache else None
    if use_cache:
        logger.info("GNNCache key: %s  dir: %s", cache_key, cache_dir)

    test_start = data_cfg.get("test_start")
    test_end   = data_cfg.get("test_end")
    run_cutoff = pd.Timestamp(test_end, tz="UTC") if test_end else None

    if use_cache and gnn_cache.exists(cache_key):
        # ── Fast path: load from cache ────────────────────────────────────────
        # Arrays are mmap'd — all worker processes share the OS page-cache.
        # Raw (unscaled) tensors are stored so that each fold can fit its own
        # scaler on the correct training window without data leakage.
        # grid_icond2_runs is loaded fully into RAM (mmap=False) because
        # StandardScaler.transform() always copies it internally — mmap would
        # only add page-fault overhead without any sharing benefit.
        # All other arrays stay memory-mapped (shared OS page-cache across workers).
        logger.info("GNNCache HIT — loading arrays (grid_icond2_runs: RAM, rest: mmap) …")
        arrays, derived = gnn_cache.load(
            cache_key,
            mmap=True,
            mmap_overrides={"grid_icond2_runs": False},
        )

        grid_icond2_runs     = arrays["grid_icond2_runs"]
        meas_raw             = arrays["meas_raw"]
        station_ecmwf_nwp    = arrays["station_ecmwf_nwp"]
        ecmwf_nwp            = arrays["ecmwf_nwp"]

        timestamps           = derived["timestamps"]
        run_times            = derived["run_times"]
        lats                 = derived["lats"]
        lons                 = derived["lons"]
        alts                 = derived["alts"]
        station_coords       = derived["station_coords"]
        icond2_coords        = derived["icond2_coords"]
        ecmwf_coords         = derived["ecmwf_coords"]
        icond2_alts          = derived["icond2_alts"]
        ecmwf_alts           = derived["ecmwf_alts"]
        station_nearest_grid = derived["station_nearest_grid"]
        station_static_scaled   = derived["station_static_scaled"]
        icond2_static_scaled    = derived["icond2_static_scaled"]
        ecmwf_static_scaled     = derived["ecmwf_static_scaled"]
        split_t              = derived["split_t"]
        split_time           = derived["split_time"]
        all_run_pairs        = derived["all_run_pairs"]
        _meas_nan_any        = derived["meas_nan_any"]

        T       = len(timestamps)
        R       = len(run_times)
        N_igrid = len(icond2_coords)
        I2      = len(icond2_features_all)
        E2      = len(ecmwf_features_all)
        M_meas  = len(measurement_cols)

        # Recompute meas_nan_any and all_run_pairs from loaded meas_raw.
        # The cached values may have been built with an older check (wind_speed
        # only); re-deriving here ensures ALL measurement features are checked,
        # so runs with NaN wind_direction in the window are also excluded.
        _meas_nan_any = np.isnan(meas_raw).any(axis=(1, 2))
        all_run_pairs, _ = _build_run_pairs(
            run_times, timestamps, _meas_nan_any, split_time, H, F_h
        )
        logger.info(
            "Loaded from cache — T=%d  R=%d  N_igrid=%d  all_pairs=%d",
            T, R, N_igrid, len(all_run_pairs),
        )
    else:
        # ── Slow path: load raw data, (optionally save to cache) ──────────────
        # Scalers are NOT applied here — they are fit per-fold inside objective()
        # to avoid data leakage across fold boundaries.
        logger.info("GNNCache MISS — loading raw data …")

        logger.info("Loading station measurements …")
        meas_raw, timestamps = load_station_measurements(data_path, all_ids, cols=measurement_cols)
        T = len(timestamps)

        interpol_path = data_cfg.get("interpol_path")
        if interpol_path:
            logger.info("Loading interpolation (rk_pred) for imputation from %s …", interpol_path)
            rk_pred = load_interpol_imputation(interpol_path, all_ids, timestamps)
            nan_before = int(np.isnan(meas_raw[:, :, measurement_cols.index(target_col)]).sum())
            meas_raw = apply_interpol_imputation(meas_raw, rk_pred, measurement_cols, target_col)
            nan_after = int(np.isnan(meas_raw[:, :, measurement_cols.index(target_col)]).sum())
            logger.info("Imputation: %d NaN → %d NaN in '%s'", nan_before, nan_after, target_col)

        knnimputer_path = data_cfg.get("knnimputer_path")
        if knnimputer_path and "wind_direction" in measurement_cols:
            knn_wd = load_knn_imputation(knnimputer_path, "wind_direction", all_ids, timestamps)
            nan_before = int(np.isnan(meas_raw[:, :, measurement_cols.index("wind_direction")]).sum())
            meas_raw = apply_knn_imputation(meas_raw, knn_wd, measurement_cols, "wind_direction")
            nan_after = int(np.isnan(meas_raw[:, :, measurement_cols.index("wind_direction")]).sum())
            logger.info("KNN imputation: %d NaN → %d NaN in 'wind_direction'", nan_before, nan_after)

        _meas_nan_any = np.isnan(meas_raw).any(axis=(1, 2))

        if test_start:
            ts_cutoff = pd.Timestamp(test_start, tz="UTC")
            split_t   = int(np.searchsorted(timestamps, ts_cutoff, side="left"))
        else:
            split_t = int(T * (1 - data_cfg.get("val_frac", 0.2)))
        split_time = timestamps[split_t]
        logger.info("Temporal split at %s (index %d)", split_time.date(), split_t)

        meta_path = data_cfg.get("wind_parameter_path")
        lats, lons, alts = load_station_metadata(data_path, all_ids, meta_path=meta_path)
        station_coords = np.stack([lats, lons], axis=1)

        logger.info("Loading ICON-D2 ML runs …")
        run_times, icond2_coords, grid_icond2_runs, station_nearest_grid = load_icond2_ml_runs(
            nwp_path=nwp_path,
            station_ids=all_ids,
            station_coords=station_coords,
            features=icond2_features,
            run_hours=run_hours,
            next_n_grid=next_n_icond2,
            n_workers=n_workers,
            cutoff=run_cutoff,
        )
        R  = len(run_times)
        I2 = len(icond2_features)
        N_igrid = len(icond2_coords)
        logger.info("ICON-D2: %d grid nodes, %d runs", N_igrid, R)

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
        else:
            logger.warning("ECMWF parquet not found — using zeros")
            station_ecmwf_nwp = np.zeros((T, len(all_ids), E2), dtype=np.float32)
            ec_lats = np.arange(47.5, 55.0, 0.5)
            ec_lons = np.arange(6.0, 15.5, 0.5)
            eg, lg  = np.meshgrid(ec_lats, ec_lons)
            ecmwf_coords = np.stack([eg.ravel(), lg.ravel()], axis=1).astype(np.float32)
            ecmwf_nwp    = np.zeros((T, len(ecmwf_coords), E2), dtype=np.float32)
            ecmwf_alts   = np.zeros(len(ecmwf_coords), dtype=np.float32)

        # Altitudes
        if dcrnn_cfg.get("use_altitude_diff", False):
            weather_db_url = os.environ.get("WEATHER_DB_URL")
            ecmwf_db_url   = os.environ.get("ECMWF_WIND_SL_URL")
            if weather_db_url and ecmwf_db_url:
                icond2_alts, ecmwf_alts = load_nwp_elevations(
                    weather_db_url=weather_db_url, ecmwf_db_url=ecmwf_db_url,
                    icond2_coords=icond2_coords, ecmwf_coords=ecmwf_coords,
                )
                icond2_alts += 10.0
                ecmwf_alts  += 10.0
            else:
                logger.warning("DB URLs not set — NWP altitudes = 0")
                icond2_alts = np.zeros(N_igrid, dtype=np.float32)
        else:
            icond2_alts = np.zeros(N_igrid, dtype=np.float32)

        M_meas = len(measurement_cols)
        I2     = len(icond2_features_all)
        E2     = len(ecmwf_features_all)

        # Static node features: coordinates + altitude are time- and fold-independent,
        # so fitting on all nodes is not leakage — these go into the cache scaled.
        stat_scaler = StandardScaler()
        raw_static  = np.stack([lats, lons, alts], axis=1)
        stat_scaler.fit(raw_static[:N_train])
        station_static_scaled = stat_scaler.transform(raw_static)

        # icond2 / ecmwf static features are purely geometric (coordinates + altitude).
        # Fitting on all grid nodes is safe: there is no temporal component and the
        # full grid is always observed regardless of which stations are in train/val.
        icond2_static_scaled = StandardScaler().fit_transform(
            np.concatenate([icond2_coords, icond2_alts[:, None]], axis=1)
        ).astype(np.float32)
        ecmwf_static_scaled = StandardScaler().fit_transform(
            np.concatenate([ecmwf_coords, ecmwf_alts[:, None]], axis=1)
        ).astype(np.float32)

        # ── All pre-test run pairs ─────────────────────────────────────────────
        all_run_pairs, _ = _build_run_pairs(
            run_times, timestamps, _meas_nan_any, split_time, H, F_h
        )

        # ── Write cache ───────────────────────────────────────────────────────
        # Store RAW (unscaled) tensors so each fold can fit its own scaler on
        # [:fold_val_start, :N_train] without leaking future fold-val data.
        if use_cache:
            logger.info("GNNCache — writing cache (key=%s) …", cache_key)
            gnn_cache.save(
                cache_key,
                arrays={
                    "grid_icond2_runs":   np.array(grid_icond2_runs),
                    "meas_raw":           np.array(meas_raw),
                    "station_ecmwf_nwp":  np.array(station_ecmwf_nwp),
                    "ecmwf_nwp":          np.array(ecmwf_nwp),
                },
                derived={
                    "timestamps":            timestamps,
                    "run_times":             run_times,
                    "lats":                  lats,
                    "lons":                  lons,
                    "alts":                  alts,
                    "station_coords":        station_coords,
                    "icond2_coords":         icond2_coords,
                    "ecmwf_coords":          ecmwf_coords,
                    "icond2_alts":           icond2_alts,
                    "ecmwf_alts":            ecmwf_alts,
                    "station_nearest_grid":  station_nearest_grid,
                    "station_static_scaled": station_static_scaled,
                    "icond2_static_scaled":  icond2_static_scaled,
                    "ecmwf_static_scaled":   ecmwf_static_scaled,
                    "split_t":               split_t,
                    "split_time":            split_time,
                    "all_run_pairs":         all_run_pairs,
                    "meas_nan_any":          _meas_nan_any,
                },
            )
            logger.info("GNNCache — cache written.")

    logger.info("Total pre-test run pairs available for CV: %d", len(all_run_pairs))

    # ── Exit early if --preprocess-only ───────────────────────────────────────
    if args.preprocess_only:
        logger.info("--preprocess-only done. Exiting.")
        return

    # ── Graph (fixed topology, built once) ───────────────────────────────────
    target_feat_idx   = measurement_cols.index(target_col)
    train_station_indices = list(range(N_train))

    # HPO val stations: first n_val_stations from val_files
    n_val_stations = hpo_cfg.get("n_val_stations", 40)
    hpo_val_station_indices = list(range(N_train, N_train + min(n_val_stations, N_val)))
    logger.info(
        "HPO val stations: first %d of %d val stations", len(hpo_val_station_indices), N_val
    )

    # Build graph using a base config (topology never changes across trials)
    _base_cfg = copy.deepcopy(dcrnn_cfg)
    _base_cfg.setdefault("hidden", 64)
    _base_cfg.setdefault("num_layers", 2)
    _base_cfg.setdefault("diffusion_K", 4)
    _base_cfg.setdefault("nwp_out_dim", 64)
    _base_cfg.setdefault("nwp_heads", 4)
    _base_cfg.setdefault("dropout", 0.3)
    _base_cfg.setdefault("teacher_forcing_ratio", 0.3)
    _base_cfg.setdefault("lr", 3e-4)
    _base_cfg.setdefault("weight_decay", 1e-3)
    _base_cfg.setdefault("batch_size", 8)
    _base_cfg.setdefault("horizon_decay", 0.95)
    _base_cfg.setdefault("scheduler", "cosine")

    _base_model_cfg = DCRNNConfig.from_yaml(
        _base_cfg,
        icond2_features=icond2_features,
        ecmwf_features=ecmwf_features,
        measurement_features=measurement_cols,
        target_col=target_col,
        n_train=N_train,
        n_val=N_val,
        checkpoint_path="models/_hpo_dcrnn_tmp.pt",
    )

    builder    = HeterogeneousGraphBuilder(_base_model_cfg.graph)
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

    # ── Time-based CV fold boundaries (expanding window, same as hpo_cl.py) ──
    #
    # min_train_date is the END of the fixed training block that is always
    # included in training — identical semantics to hpo_cl.py / apply_min_train_len.
    # Only the period [min_train_date, test_start] is sliced into n_folds segments
    # for validation.  This guarantees a minimum training size in every fold.
    #
    # Layout (n_folds=3):
    #
    #   data_start ──────── min_train_date ────────────────── test_start
    #   │                   │              │              │              │
    #   │   fixed block     │   seg 0      │   seg 1      │   seg 2      │
    #   │  (always train)   │              │              │              │
    #
    #   Fold 0:  train = [data_start, min_train_date + seg_0)
    #            val   = [min_train_date + seg_0, min_train_date + seg_1)
    #   Fold 1:  train = [data_start, min_train_date + seg_1)
    #            val   = [min_train_date + seg_1, min_train_date + seg_2)
    #   Fold 2:  train = [data_start, min_train_date + seg_2)
    #            val   = [min_train_date + seg_2, test_start)

    n_folds        = hpo_cfg.get("n_folds", 3)
    min_train_date = hpo_cfg.get("min_train_date")
    if min_train_date:
        min_train_dt = pd.Timestamp(min_train_date, tz="UTC")
    else:
        min_train_dt = timestamps[0]

    test_start_dt = pd.Timestamp(data_cfg["test_start"], tz="UTC")
    # Mirror TimeSeriesSplit(n_splits=n_folds):
    # The foldable range [min_train_dt, test_start] is divided into n_folds+1
    # equal chunks.  Chunk 0 always goes to training; chunks 1..n_folds are the
    # validation windows.  This is why the user sets min_train_date earlier
    # than the first desired val window — the first chunk acts as extra training.
    foldable_secs = (test_start_dt - min_train_dt).total_seconds()
    seg_secs      = foldable_secs / (n_folds + 1)

    fold_splits: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for k in range(n_folds):
        val_start = min_train_dt + pd.Timedelta(seconds=seg_secs * (k + 1))
        val_end   = min_train_dt + pd.Timedelta(seconds=seg_secs * (k + 2))
        fold_splits.append((val_start, val_end))
        logger.info(
            "Fold %d — train: [data_start, %s)  val: [%s, %s)",
            k, val_start.date(), val_start.date(), val_end.date(),
        )

    def _fold_pairs(
        val_start: pd.Timestamp, val_end: pd.Timestamp
    ) -> tuple[list, list]:
        """
        Split all_run_pairs into train / val for one fold.

        Train: all pairs with t_run < val_start  (fixed block + expanding part)
        Val:   pairs with val_start <= t_run < val_end
        """
        tr, va = [], []
        for r_curr, r_hist, t_run_abs in all_run_pairs:
            t = timestamps[t_run_abs]
            if t < val_start:
                tr.append((r_curr, r_hist, t_run_abs))
            elif t < val_end:
                va.append((r_curr, r_hist, t_run_abs))
        return tr, va

    # ── Objective ─────────────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        sampled = sample_hyperparameters(trial, hpo_params)
        logger.info("Trial %d — params: %s", trial.number, sampled)

        trial_cfg = copy.deepcopy(dcrnn_cfg)
        trial_cfg.update(sampled)
        trial_cfg["max_epochs"] = max_epochs
        trial_cfg["patience"]   = patience

        # ── Feature mode per trial ────────────────────────────────────────────
        i2_mode = sampled.pop("icond2_feature_mode", dcrnn_cfg.get("icond2_feature_mode", "both"))
        e2_mode = sampled.pop("ecmwf_feature_mode",  dcrnn_cfg.get("ecmwf_feature_mode",  "both"))
        trial_i2_features = resolve_feature_mode(icond2_features_all, i2_mode)
        trial_e2_features = resolve_feature_mode(ecmwf_features_all,  e2_mode)
        i2_idx = feature_indices(icond2_features_all, trial_i2_features)
        e2_idx = feature_indices(ecmwf_features_all,  trial_e2_features)
        trial_I2 = len(trial_i2_features)
        trial_E2 = len(trial_e2_features)
        # Slice the pre-loaded arrays to the trial feature subset
        trial_grid_icond2 = grid_icond2_runs[:, :, :, i2_idx]   # (R, 48, N_igrid, trial_I2)
        trial_ecmwf_nwp   = ecmwf_nwp[:, :, e2_idx]              # (T, N_ecmwf, trial_E2)
        trial_sta_ecmwf   = station_ecmwf_nwp[:, :, e2_idx]      # (T, N_all, trial_E2)

        model_cfg = DCRNNConfig.from_yaml(
            trial_cfg,
            icond2_features=trial_i2_features,
            ecmwf_features=trial_e2_features,
            measurement_features=measurement_cols,
            target_col=target_col,
            n_train=N_train,
            n_val=len(hpo_val_station_indices),
            checkpoint_path="models/_hpo_dcrnn_tmp.pt",
        )

        sampler_obj = TrainingSampler(
            model_cfg, builder, base_graph,
            target_feat_idx=target_feat_idx,
            station_coords=station_coords,
        )

        fold_losses: list[float] = []
        for fold_idx, (val_start, val_end) in enumerate(fold_splits):
            train_pairs_fold, val_pairs_fold = _fold_pairs(val_start, val_end)
            if not train_pairs_fold or not val_pairs_fold:
                logger.warning(
                    "Trial %d fold %d: empty split (train=%d val=%d), skipping",
                    trial.number, fold_idx, len(train_pairs_fold), len(val_pairs_fold),
                )
                continue

            # ── Per-fold scalers (fit only on fold-train window) ──────────────
            # val_start marks the fold's training cutoff — fitting on data up to
            # this point prevents any leakage from the fold-val period.
            fold_train_t = int(np.searchsorted(timestamps, val_start, side="left"))
            fold_train_r = run_times < val_start

            fold_meas_scaler = StandardScaler()
            fold_meas_scaler.fit(meas_raw[:fold_train_t, :N_train].reshape(-1, M_meas))
            meas_scaled = fold_meas_scaler.transform(
                meas_raw.reshape(-1, M_meas)
            ).reshape(T, len(all_ids), M_meas)

            fold_i2_scaler = StandardScaler()
            fold_i2_scaler.fit(trial_grid_icond2[fold_train_r].reshape(-1, trial_I2))
            grid_icond2_runs_scaled = fold_i2_scaler.transform(
                trial_grid_icond2.reshape(-1, trial_I2)
            ).reshape(R, 48, N_igrid, trial_I2)

            fold_e2_scaler = StandardScaler()
            fold_e2_scaler.fit(trial_sta_ecmwf[:fold_train_t, :N_train].reshape(-1, trial_E2))
            station_ecmwf_scaled = fold_e2_scaler.transform(
                trial_sta_ecmwf.reshape(-1, trial_E2)
            ).reshape(T, len(all_ids), trial_E2)
            ecmwf_nwp_scaled = fold_e2_scaler.transform(
                trial_ecmwf_nwp.reshape(-1, trial_E2)
            ).reshape(T, len(ecmwf_coords), trial_E2)

            fold_data = dict(
                station_meas=meas_scaled,
                station_nearest_grid=station_nearest_grid,
                grid_icond2_runs=grid_icond2_runs_scaled,
                station_ecmwf_nwp=station_ecmwf_scaled,
                station_static=station_static_scaled,
                ecmwf_nwp=ecmwf_nwp_scaled,
                icond2_static=icond2_static_scaled,
                ecmwf_static=ecmwf_static_scaled,
                train_station_indices=train_station_indices,
                val_station_indices=hpo_val_station_indices,
            )

            ckpt = Path(f"models/_hpo_dcrnn_trial{trial.number}_fold{fold_idx}.pt")
            trial_cfg["checkpoint_path"] = str(ckpt)

            tb_dir = Path("runs") / f"hpo_dcrnn_{config_stem}" / f"trial_{trial.number:04d}_fold_{fold_idx}"
            writer = SummaryWriter(log_dir=str(tb_dir)) if (use_tensorboard and SummaryWriter is not None) else None

            model   = DCRNN(model_cfg)
            trainer = DCRNNTrainer(
                model=model,
                sampler=sampler_obj,
                config=model_cfg,
                device=device,
                teacher_forcing_start=float(trial_cfg.get("teacher_forcing_ratio", 0.5)),
                teacher_forcing_end=0.0,
                writer=writer,
            )
            # Override checkpoint path in trainer
            trainer._ckpt_path = ckpt

            result = trainer.fit(
                **fold_data,
                train_run_pairs=train_pairs_fold,
                val_run_pairs=val_pairs_fold,
                verbose=False,
            )
            if writer is not None:
                writer.close()

            fold_loss = result["best_val_loss"]
            fold_losses.append(fold_loss)
            logger.info(
                "Trial %d fold %d — best_val_loss=%.4f  stopped=%d",
                trial.number, fold_idx, fold_loss, result["stopped_epoch"],
            )

            if ckpt.exists():
                ckpt.unlink()

            # Intermediate pruning after each fold (single-objective)
            trial.report(float(np.mean(fold_losses)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_losses:
            raise ValueError("All folds were empty — check min_train_date / data range.")

        mean_loss = float(np.mean(fold_losses))
        logger.info(
            "Trial %d done — fold losses: %s  mean=%.4f",
            trial.number,
            [f"{v:.4f}" for v in fold_losses],
            mean_loss,
        )
        return mean_loss

    # ── Create / resume study ─────────────────────────────────────────────────
    storage_url = os.environ.get("OPTUNA_STORAGE")
    if storage_url:
        storage = storage_url
        logger.info("Using Optuna storage: PostgreSQL (OPTUNA_STORAGE)")
    else:
        db_path = Path(studies_path) / f"hpo_dcrnn_{config_stem}{suffix}.db"
        storage = f"sqlite:///{db_path}"
        logger.info("Using SQLite storage: %s", db_path)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=42),
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(n_trials - completed, 0)
    logger.info(
        "Study loaded — %d completed, %d remaining out of %d total trials",
        completed, remaining, n_trials,
    )

    if remaining == 0:
        logger.info("All trials already completed.")
    else:
        study.optimize(
            objective,
            n_trials=remaining,
            catch=(Exception,),
        )

    # ── Results ───────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("HPO COMPLETE")
    logger.info("Best trial:     #%d", study.best_trial.number)
    logger.info("Best val_loss:  %.6f", study.best_value)
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info("  %-30s %s", k, v)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
