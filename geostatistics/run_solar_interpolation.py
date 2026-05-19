#!/usr/bin/env python3
"""
run_solar_interpolation.py — Spatial GHI interpolation pipeline (IDW / OK / RK).

Analogous to run_spatial_interpolation.py but designed for solar irradiance.
Uses ICON-D2 SL (Surface Level) parquet files as NWP covariate for
Regression Kriging.

Usage (run from forecasting_framework/)
----------------------------------------
    python geostatistics/run_solar_interpolation.py \\
        --config configs/config_solar_interpol.yaml [--suffix v1]

Config structure
----------------
    data:
      path:            '/mnt/nvme1/synthetic/raw/solar'
      nwp_path:        '/mnt/nvme1/icon-d2/parquet'
      stations_master: 'data/stations_master.csv'
      knn_cache_path:  '/mnt/nvme1/synthetic/knnimputer/solar'
      target_col:      'ghi'
      extra_cols:      ['dhi']      # optional non-target cols to KNN-impute
      run_hours:       [6]

    interpolation:
      get_theta_opt:       false
      k_neighbors:         30
      idw_power:           2.0
      variogram_model:     gaussian
      n_variogram_lags:    30
      max_variogram_dist:  1000
      variogram_detrend:   true
      rk_nan_threshold:    0.5
      knn_impute_neighbors: 10
      rk_features:
        - altitude
        - nwp_ghi
      anisotropy:
        enabled: false
        angle: 0
        ratio: 1.0

    output:
      path:        'results/geostatistics'
      target_path: '/mnt/nvme1/synthetic/interpol/solar'
      prefix:      'solar_interp'
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.interpolation import (
    compute_anisotropic_distance_matrix,
    compute_distance_matrix,
    compute_empirical_semivariance,
    compute_metrics,
    fit_global_variogram,
    run_loo_cv,
)

logger = logging.getLogger(__name__)

_DERIVED_SOLAR = {"ghi_nwp", "dhi_nwp", "bhi_nwp"}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# NWP loading
# ---------------------------------------------------------------------------

def load_solar_nwp_feature(
    nwp_path: str,
    station_ids: list[str],
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    timestamps: pd.DatetimeIndex,
    feature: str = "ghi_nwp",
    run_hours: tuple[int, ...] = (6,),
) -> np.ndarray:
    """Load a solar NWP feature from ICON-D2 SL parquets, aligned to *timestamps*.

    For each (station, hour), picks the forecast with the shortest positive
    lead time from the configured *run_hours*.

    Returns
    -------
    (T, N) float64 array — NaN where no data available.
    """
    from geostatistics.solar_preprocessing import (
        _select_nearest_sl_files,
        _load_solar_sl_parquet,
    )

    T = len(timestamps)
    N = len(station_ids)
    result = np.full((T, N), np.nan, dtype=np.float64)

    if timestamps.tz is None:
        ts_idx = pd.DatetimeIndex(timestamps).tz_localize("UTC")
    else:
        ts_idx = timestamps

    sl_base = Path(nwp_path) / "SL"
    leads = np.arange(1, 49, dtype=int)  # lead_1h = 1..48

    for j, sid in enumerate(tqdm(station_ids, desc=f"Loading NWP '{feature}'", unit="station")):
        s_lat = float(station_lats[j])
        s_lon = float(station_lons[j])

        rows_valid_time = []
        rows_lead = []
        rows_value = []

        for rh in run_hours:
            sid_dir = sl_base / f"{rh:02d}" / sid
            if not sid_dir.exists():
                continue
            nearest = _select_nearest_sl_files(sid_dir, s_lat, s_lon, k=1)
            if not nearest:
                continue
            fpath = nearest[0][0]
            try:
                run_times, arr = _load_solar_sl_parquet(fpath, [feature])  # (R, 48, 1)
            except Exception as exc:
                logger.debug("Station %s rh=%02d feature='%s': %s", sid, rh, feature, exc)
                continue

            R_local = len(run_times)
            for ri in range(R_local):
                rt = run_times[ri]
                for li in range(48):
                    val = float(arr[ri, li, 0])
                    if np.isnan(val):
                        continue
                    lead_1h = leads[li]
                    valid_time = rt + pd.Timedelta(hours=int(lead_1h))
                    rows_valid_time.append(valid_time)
                    rows_lead.append(lead_1h)
                    rows_value.append(val)

        if not rows_valid_time:
            continue

        df = pd.DataFrame({
            "valid_time": rows_valid_time,
            "lead_1h": rows_lead,
            "value": rows_value,
        })
        # Shortest lead wins when multiple runs cover the same valid_time
        df = df.sort_values(["valid_time", "lead_1h"])
        df = df.drop_duplicates(subset=["valid_time"], keep="first")
        df = df.set_index("valid_time")
        result[:, j] = df["value"].reindex(ts_idx).values

    n_valid = int(np.sum(~np.isnan(result)))
    logger.info(
        "NWP '%s' loaded: %.1f%% values available",
        feature, 100.0 * n_valid / (T * N),
    )
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config: dict) -> tuple:
    """Load solar station data; return a (T, N) pivot and coordinate arrays.

    Returns
    -------
    pivot          : DataFrame (T×N) of *target_col*; index=timestamp, cols=station_id.
    raw_pivot      : DataFrame (T×N) of raw *target_col* before KNN imputation.
    lats           : (N,) latitude array.
    lons           : (N,) longitude array.
    alts           : (N,) altitude array in metres.
    station_ids    : List[str]
    rk_feature_names   : list of active RK covariate names (or None)
    rk_static_features : dict name → array
    rk_dynamic_features: dict name → (T, N) array
    """
    data_cfg   = config["data"]
    interp_cfg = config.get("interpolation", {})

    data_path   = data_cfg["path"]
    target_col  = data_cfg.get("target_col", "ghi")
    extra_cols  = [str(c) for c in data_cfg.get("extra_cols", [])]
    run_hours   = tuple(int(h) for h in data_cfg.get("run_hours", [6]))
    knn_k       = int(interp_cfg.get("knn_impute_neighbors", 10))
    cache_dir   = data_cfg.get("knn_cache_path", "/mnt/nvme1/synthetic/knnimputer/solar")
    os.makedirs(cache_dir, exist_ok=True)

    # --- station file discovery ---
    station_file_map: dict[str, str] = {}
    for fname in sorted(os.listdir(data_path)):
        if not fname.startswith("Station_"):
            continue
        if not (fname.endswith(".parquet") or fname.endswith(".csv")):
            continue
        sid = os.path.splitext(fname)[0].replace("Station_", "")
        fpath = os.path.join(data_path, fname)
        chosen = station_file_map.get(sid)
        if chosen is None or (fname.endswith(".parquet") and chosen.endswith(".csv")):
            station_file_map[sid] = fpath

    files_list = [str(s) for s in data_cfg.get("files", [])]
    if files_list:
        station_ids = [s for s in files_list if s in station_file_map]
        missing = [s for s in files_list if s not in station_file_map]
        if missing:
            logger.warning("%d stations in config 'files' not found on disk: %s%s",
                           len(missing), missing[:5], " …" if len(missing) > 5 else "")
        logger.info("Using %d/%d stations from config 'files' list", len(station_ids), len(files_list))
    else:
        station_ids = sorted(station_file_map.keys())
        exclude = [str(s) for s in data_cfg.get("exclude_stations", [])]
        if exclude:
            station_ids = [s for s in station_ids if s not in exclude]
        logger.info("Discovered %d stations in %s", len(station_ids), data_path)

    # --- metadata ---
    meta_path = data_cfg.get("stations_master", "data/stations_master.csv")
    if not os.path.isabs(meta_path):
        meta_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), meta_path
        )
    meta = pd.read_csv(meta_path, dtype={"station_id": str})
    meta["station_id"] = meta["station_id"].astype(str)
    meta = meta.set_index("station_id")

    missing_meta = [s for s in station_ids if s not in meta.index]
    if missing_meta:
        logger.warning("%d stations not in stations_master — skipping: %s ...",
                       len(missing_meta), missing_meta[:5])
        station_ids = [s for s in station_ids if s in meta.index]

    lats = meta.loc[station_ids, "latitude"].values.astype(np.float64)
    lons = meta.loc[station_ids, "longitude"].values.astype(np.float64)
    alts = meta.loc[station_ids, "station_height"].values.astype(np.float64)

    # --- per-station time series ---
    cols_to_load = [target_col] + extra_cols
    all_dfs = []
    for sid in tqdm(station_ids, desc="Reading station files", unit="station"):
        fpath = station_file_map[sid]
        if fpath.endswith(".parquet"):
            df = pd.read_parquet(fpath)
            if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        else:
            df = pd.read_csv(fpath, parse_dates=["timestamp"])
        df["station_id"] = sid
        available = [c for c in ["timestamp", "station_id"] + cols_to_load if c in df.columns]
        all_dfs.append(df[available])

    logger.info("Concatenating %d DataFrames …", len(all_dfs))
    combined = pd.concat(all_dfs, ignore_index=True)

    if combined["timestamp"].dt.tz is None:
        combined["timestamp"] = combined["timestamp"].dt.tz_localize("UTC")

    test_start = data_cfg.get("test_start")
    test_end   = data_cfg.get("test_end")
    if test_start:
        combined = combined[combined["timestamp"] >= pd.Timestamp(test_start, tz="UTC")]
    if test_end:
        combined = combined[combined["timestamp"] <= pd.Timestamp(test_end, tz="UTC")]

    # raw (before KNN) for export
    raw_pivot = combined.pivot_table(
        index="timestamp", columns="station_id", values=target_col, aggfunc="mean"
    ).reindex(columns=station_ids)
    raw_pivot.index = pd.DatetimeIndex(raw_pivot.index)
    raw_pivot = raw_pivot.sort_index()
    raw_pivot = raw_pivot.resample("1h", closed="left", label="left").mean()

    # --- KNN imputation of target column ---
    _t_start = (test_start or "start").replace(":", "").replace("-", "").replace(" ", "")
    _t_end   = (test_end   or "end"  ).replace(":", "").replace("-", "").replace(" ", "")
    _sid_hash = hashlib.md5(",".join(sorted(station_ids)).encode()).hexdigest()[:8]

    def _knn_impute_col(col_name: str) -> pd.DataFrame:
        cache_fname = f"{col_name}_knn{knn_k}_{_t_start}_{_t_end}_{_sid_hash}.parquet"
        cache_path  = os.path.join(cache_dir, cache_fname)

        if os.path.isfile(cache_path):
            logger.info("KNN cache hit for '%s': %s", col_name, cache_path)
            piv = pd.read_parquet(cache_path)
            piv.index = pd.DatetimeIndex(piv.index)
            return piv

        logger.info("Building KNN-imputed pivot for '%s' …", col_name)
        piv_raw = combined.pivot_table(
            index="timestamp", columns="station_id", values=col_name, aggfunc="mean"
        ).reindex(columns=station_ids)
        n_missing = int(piv_raw.isna().sum().sum())
        if n_missing > 0:
            from sklearn.impute import KNNImputer
            logger.info("KNN imputation '%s' (k=%d): %d missing (%.2f%%) — fitting …",
                        col_name, knn_k, n_missing, 100.0 * n_missing / piv_raw.size)
            imputed = KNNImputer(n_neighbors=knn_k).fit_transform(piv_raw.values)
            piv_raw = pd.DataFrame(imputed, index=piv_raw.index, columns=piv_raw.columns)
            logger.info("KNN imputation '%s' done.", col_name)
        else:
            logger.info("No missing '%s' values — KNN imputation skipped.", col_name)
        piv_raw.to_parquet(cache_path)
        logger.info("KNN-imputed '%s' cached → %s", col_name, cache_path)
        return piv_raw

    target_knn_pivot = _knn_impute_col(target_col)

    # Write imputed target back into combined
    imputed_long = target_knn_pivot.reset_index().melt(
        id_vars="timestamp", var_name="station_id", value_name=f"{target_col}_imputed"
    )
    combined = combined.merge(imputed_long, on=["timestamp", "station_id"], how="left")
    combined[target_col] = combined[f"{target_col}_imputed"]
    combined = combined.drop(columns=[f"{target_col}_imputed"])

    # KNN-impute extra columns too (for full dataset consistency)
    for ec in extra_cols:
        if ec not in combined.columns:
            continue
        ec_knn = _knn_impute_col(ec)
        ec_long = ec_knn.reset_index().melt(
            id_vars="timestamp", var_name="station_id", value_name=f"{ec}_imputed"
        )
        combined = combined.merge(ec_long, on=["timestamp", "station_id"], how="left")
        combined[ec] = combined[f"{ec}_imputed"]
        combined = combined.drop(columns=[f"{ec}_imputed"])

    # Build main (T, N) hourly pivot
    pivot = combined.pivot_table(
        index="timestamp", columns="station_id", values=target_col, aggfunc="mean"
    ).reindex(columns=station_ids)
    pivot.index = pd.DatetimeIndex(pivot.index)
    pivot = pivot.sort_index()
    pivot = pivot.resample("1h", closed="left", label="left").mean()

    # Fill remaining hourly NaN from KNN parquet
    remaining_nan = int(pivot.isna().sum().sum())
    if remaining_nan > 0:
        tgt_hourly = target_knn_pivot.resample("1h", closed="left", label="left").mean()
        tgt_hourly = tgt_hourly.reindex(index=pivot.index, columns=pivot.columns)
        pivot = pivot.fillna(tgt_hourly)
        after = int(pivot.isna().sum().sum())
        logger.info("Filled %d hourly NaN from KNN (%d remaining).",
                    remaining_nan - after, after)

    # --- RK feature matrices ---
    rk_feature_names_cfg = interp_cfg.get("rk_features") or []
    nwp_path = data_cfg.get("nwp_path")
    rk_nan_threshold = float(interp_cfg.get("rk_nan_threshold", 0.5))

    rk_static_features: dict  = {}
    rk_dynamic_features: dict = {}

    if rk_feature_names_cfg:
        rk_static_features["altitude"] = alts

        nwp_features_requested = [f for f in rk_feature_names_cfg
                                   if f in _DERIVED_SOLAR or f in ("alb_rad", "clct", "t_2m")]
        station_features_requested = [
            f for f in rk_feature_names_cfg
            if f not in ("altitude",) and f not in nwp_features_requested
        ]

        # Load NWP features from SL parquets
        for nwp_feat in nwp_features_requested:
            if not nwp_path:
                logger.warning("NWP feature '%s' requested but data.nwp_path not set — skipping.",
                               nwp_feat)
                continue
            mat = load_solar_nwp_feature(
                nwp_path=nwp_path,
                station_ids=station_ids,
                station_lats=lats,
                station_lons=lons,
                timestamps=pivot.index,
                feature=nwp_feat,
                run_hours=run_hours,
            )
            rk_dynamic_features[nwp_feat] = mat
            logger.info("Loaded RK feature '%s' as (T, N) matrix.", nwp_feat)

        # Load non-NWP dynamic features from station files
        for fname in station_features_requested:
            if fname in combined.columns:
                piv = combined.pivot_table(
                    index="timestamp", columns="station_id", values=fname, aggfunc="mean"
                ).reindex(columns=station_ids)
                piv.index = pd.DatetimeIndex(piv.index)
                piv = piv.sort_index().resample("1h", closed="left", label="left").mean()
                piv = piv.reindex(pivot.index)
                rk_dynamic_features[fname] = piv.values.astype(np.float64)
                logger.info("Loaded RK feature '%s' as (T, N) matrix.", fname)
            else:
                logger.warning("RK feature '%s' not in station files — skipping.", fname)

        # Keep only features that were loaded
        rk_feature_names = [
            f for f in rk_feature_names_cfg
            if f in rk_static_features or f in rk_dynamic_features
        ]

        # Drop timestamps where >threshold% of stations have NaN in any dynamic feature
        if rk_dynamic_features:
            valid_mask = np.ones(len(pivot), dtype=bool)
            for fname, arr in rk_dynamic_features.items():
                nan_frac = np.mean(np.isnan(arr), axis=1)
                valid_mask &= nan_frac <= rk_nan_threshold
                n_over = int((nan_frac > rk_nan_threshold).sum())
                if n_over:
                    logger.info("  Feature '%s': %d timestamps have >%.0f%% NaN → dropped.",
                                fname, n_over, 100 * rk_nan_threshold)
            n_dropped = int((~valid_mask).sum())
            if n_dropped:
                logger.info("Dropping %d timestamps with excessive NaN in RK features.", n_dropped)
                pivot = pivot.iloc[valid_mask]
                rk_dynamic_features = {k: v[valid_mask] for k, v in rk_dynamic_features.items()}

        logger.info("Final RK feature list: %s", rk_feature_names)
    else:
        rk_feature_names = None

    raw_pivot = raw_pivot.reindex(index=pivot.index, columns=station_ids)
    logger.info("Loaded data: %d timestamps × %d stations", len(pivot), len(station_ids))

    return (
        pivot,
        raw_pivot,
        lats,
        lons,
        alts,
        station_ids,
        rk_feature_names,
        rk_static_features,
        rk_dynamic_features,
    )


# ---------------------------------------------------------------------------
# Optimal parameter loading from Optuna study
# ---------------------------------------------------------------------------

def load_best_params_from_study(config: dict) -> dict | None:
    import optuna
    hpo_cfg = config.get("hpo", {})
    study_name = hpo_cfg.get("study_name")
    if not study_name:
        return None

    storage_url = os.environ.get("OPTUNA_STORAGE")
    if storage_url:
        storage = storage_url
    else:
        studies_dir = Path(__file__).parent.parent / "studies"
        sqlite_path = studies_dir / f"{study_name}.db"
        if not sqlite_path.exists():
            logger.warning("No SQLite study at %s — using config params.", sqlite_path)
            return None
        storage = f"sqlite:///{sqlite_path}"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            return None
        best = study.best_trial
        logger.info("Loaded best trial #%d from study '%s'  RK RMSE=%.4f",
                    best.number, study_name, best.value)
        return best.params
    except Exception as exc:
        logger.warning("Failed to load study '%s': %s", study_name, exc)
        return None


def apply_hpo_params_to_config(config: dict, hpo_params: dict) -> None:
    interp_cfg = config.setdefault("interpolation", {})
    aniso_cfg  = interp_cfg.setdefault("anisotropy", {})
    mapping = {
        "k_neighbors":        ("interpolation", "k_neighbors"),
        "idw_power":          ("interpolation", "idw_power"),
        "variogram_model":    ("interpolation", "variogram_model"),
        "n_variogram_lags":   ("interpolation", "n_variogram_lags"),
        "variogram_detrend":  ("interpolation", "variogram_detrend"),
        "anisotropy_enabled": ("anisotropy",    "enabled"),
        "anisotropy_angle":   ("anisotropy",    "angle"),
        "anisotropy_ratio":   ("anisotropy",    "ratio"),
    }
    for param, (section, key) in mapping.items():
        if param in hpo_params:
            target = aniso_cfg if section == "anisotropy" else interp_cfg
            target[key] = hpo_params[param]

    rk_features = list(interp_cfg.get("rk_features") or [])
    for nwp_feat in _DERIVED_SOLAR | {"alb_rad", "clct", "t_2m"}:
        toggle_key = f"use_{nwp_feat}"
        if toggle_key in hpo_params:
            if hpo_params[toggle_key] and nwp_feat not in rk_features:
                rk_features.append(nwp_feat)
            elif not hpo_params[toggle_key]:
                rk_features = [f for f in rk_features if f != nwp_feat]
    interp_cfg["rk_features"] = rk_features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spatial solar GHI interpolation: IDW / OK / RK with LOO-CV"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("-s", "--suffix", default="")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=log_level,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    config = load_config(args.config)
    interp_cfg = config.get("interpolation", {})
    out_cfg    = config.get("output", {})

    get_theta_opt = bool(interp_cfg.get("get_theta_opt", False))
    if get_theta_opt:
        logger.info("get_theta_opt=true — loading best params from Optuna study …")
        best = load_best_params_from_study(config)
        if best:
            apply_hpo_params_to_config(config, best)
            interp_cfg = config["interpolation"]

    k                  = int(interp_cfg.get("k_neighbors", 30))
    idw_power          = float(interp_cfg.get("idw_power", 2.0))
    n_lags             = int(interp_cfg.get("n_variogram_lags", 30))
    max_dist           = interp_cfg.get("max_variogram_dist")
    variogram_model    = interp_cfg.get("variogram_model", "gaussian")
    variogram_segments = int(interp_cfg.get("variogram_segments", 1))
    variogram_detrend  = bool(interp_cfg.get("variogram_detrend", True))
    aniso_cfg          = interp_cfg.get("anisotropy", {})
    output_dir         = out_cfg.get("path", "results/geostatistics")
    config_stem        = os.path.splitext(os.path.basename(args.config))[0].removeprefix("config_")
    prefix             = config_stem + (f"_{args.suffix}" if args.suffix else "")

    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{prefix}.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    logger.info("=== Solar interpolation — prefix: %s ===", prefix)

    # 1. Load data
    pivot, raw_pivot, lats, lons, alts, station_ids, \
        rk_feature_names, rk_static_features, rk_dynamic_features = load_data(config)
    values_matrix = pivot.values
    timestamps    = pivot.index
    T             = len(timestamps)

    # 2. Distance matrix
    logger.info("Computing %d×%d geodesic distance matrix …", len(station_ids), len(station_ids))
    dist_matrix = compute_distance_matrix(lats, lons)

    # 3. Anisotropic distance matrix (optional)
    aniso_dist_matrix = None
    if aniso_cfg.get("enabled", False):
        angle = float(aniso_cfg["angle"])
        ratio = float(aniso_cfg["ratio"])
        logger.info("Computing anisotropic distance matrix (angle=%.1f°, ratio=%.3f) …", angle, ratio)
        aniso_dist_matrix = compute_anisotropic_distance_matrix(lats, lons, angle, ratio)

    # 4. Variogram fitting
    if variogram_segments == 0:
        n_segs = T
        seg_slices = [slice(t, t + 1) for t in range(T)]
        segment_indices = np.arange(T, dtype=int)
    elif variogram_segments == 1:
        n_segs = 1
        seg_slices = [slice(0, T)]
        segment_indices = np.zeros(T, dtype=int)
    else:
        n_segs = variogram_segments
        boundaries = np.array_split(np.arange(T), n_segs)
        seg_slices = [slice(int(b[0]), int(b[-1]) + 1) for b in boundaries]
        segment_indices = np.zeros(T, dtype=int)
        for i, b in enumerate(boundaries):
            segment_indices[b] = i

    variogram_params_list = []
    for seg_i, sl in enumerate(seg_slices):
        lags_emp, sv_emp = compute_empirical_semivariance(
            values_matrix[sl], dist_matrix, n_lags=n_lags,
            max_dist=max_dist, detrend=variogram_detrend,
        )
        vp = fit_global_variogram(lags_emp, sv_emp, model=variogram_model)
        variogram_params_list.append(vp)
        if n_segs <= 5 or seg_i == 0 or seg_i == n_segs - 1:
            logger.info("  Segment %d/%d: %s", seg_i + 1, n_segs, vp)

    # 5. LOO-CV
    logger.info("Running LOO-CV (%d stations × %d timestamps) …", len(station_ids), T)
    predictions = run_loo_cv(
        values_matrix=values_matrix,
        timestamps=timestamps,
        lats=lats,
        lons=lons,
        alts=alts,
        station_ids=station_ids,
        dist_matrix=dist_matrix,
        variogram_params_list=variogram_params_list,
        segment_indices=segment_indices,
        k=k,
        idw_power=idw_power,
        rk_feature_names=rk_feature_names or None,
        rk_static_features=rk_static_features or None,
        rk_dynamic_features=rk_dynamic_features or None,
        aniso_dist_matrix=aniso_dist_matrix,
    )

    # Attach raw observations
    raw_obs_long = raw_pivot.reset_index().melt(
        id_vars="timestamp", var_name="station_id", value_name=f"{config['data'].get('target_col', 'ghi')}_raw"
    )
    predictions = predictions.merge(raw_obs_long, on=["timestamp", "station_id"], how="left")

    # 6. Save combined predictions
    pred_path = os.path.join(output_dir, f"{prefix}_loo_predictions.csv")
    predictions.to_csv(pred_path, index=False)
    logger.info("Saved LOO predictions → %s", pred_path)

    # 7. Save per-station parquet (used by train_dcrnn.py for imputation)
    target_path = out_cfg.get("target_path", os.path.join(output_dir, f"{prefix}_per_station"))
    os.makedirs(target_path, exist_ok=True)
    for sid in station_ids:
        sid_df = predictions[predictions["station_id"] == sid].copy()
        sid_df = sid_df.sort_values("timestamp").reset_index(drop=True)
        sid_df.to_parquet(os.path.join(target_path, f"Station_{sid}.parquet"), index=False)
    logger.info("Saved %d per-station Parquet files → %s", len(station_ids), target_path)

    # 8. Metrics
    per_station_df, summary_df = compute_metrics(predictions)
    per_station_df.to_csv(os.path.join(output_dir, f"{prefix}_results_per_station.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, f"{prefix}_results_summary.csv"), index=False)
    logger.info("Saved metrics.")

    logger.info("=== Done ===")
    print("\nGlobal summary (mean over stations):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
