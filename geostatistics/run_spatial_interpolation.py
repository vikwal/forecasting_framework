#!/usr/bin/env python3
"""Entry point for the spatial wind-speed interpolation pipeline.

Runs IDW, Ordinary Kriging, and Regression Kriging in a leave-one-out
cross-validation across all stations defined in the config, then writes
results and diagnostic scatter plots to the configured output directory.

Usage (run from forecasting_framework/):
    python geostatistics/run_spatial_interpolation.py --config configs/config_spatial_interpolation.yaml
"""

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Allow imports from the project root (utils/, etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.interpolation import (
    compute_anisotropic_distance_matrix,
    compute_distance_matrix,
    compute_empirical_semivariance,
    compute_metrics,
    fit_global_variogram,
    run_loo_cv,
    wind_to_uv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & data loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config from *path*."""
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_coords_from_filename(fname: str):
    """Parse grid point coordinates from ICON-D2 Parquet filename.

    Expected format: ``<lat_int>_<lat_frac>_<lon_int>_<lon_frac>_ML.parquet``
    Example: ``52_15_10_45_ML.parquet`` → (52.15, 10.45)

    Returns (lat, lon) as floats, or None if parsing fails.
    """
    stem = fname.replace("_ML.parquet", "").replace("_ML.csv", "")
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    try:
        lat = float(f"{parts[0]}.{parts[1]}")
        lon = float(f"{parts[2]}.{parts[3]}")
        return lat, lon
    except (ValueError, IndexError):
        return None


def load_nwp_wind_speed(
    nwp_path: str,
    station_ids: list,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    timestamps: pd.DatetimeIndex,
    hub_height: float = 10.0,
    forecast_hours: tuple = ("06", "09", "12", "15"),
) -> np.ndarray:
    """Load ICON-D2 NWP wind speed per station from Parquet files, aligned to *timestamps*.

    File structure::

        <nwp_path>/ML/<forecast_hour>/<station_id>/<lat_int>_<lat_frac>_<lon_int>_<lon_frac>_ML.parquet

    For each station:
    - Parses grid point coordinates from all Parquet filenames in the folder.
    - Selects the nearest grid point via geodesic distance.
    - Loads that Parquet across all forecast hours.
    - Selects the height level closest to *hub_height* metres.
    - When multiple forecast runs cover the same timestamp, keeps the one
      with the smallest forecasttime (= most recent / shortest lead time).

    Returns
    -------
    Tuple of three ndarrays, each of shape (T, N):
      - wind speed (m/s) at hub_height
      - u_wind from the selected height level (same row as wind speed)
      - v_wind from the selected height level (same row as wind speed)
    """
    from geopy.distance import geodesic

    T = len(timestamps)
    N = len(station_ids)
    result   = np.full((T, N), np.nan, dtype=np.float64)
    result_u = np.full((T, N), np.nan, dtype=np.float64)
    result_v = np.full((T, N), np.nan, dtype=np.float64)

    if timestamps.tz is None:
        ts_idx = pd.DatetimeIndex(timestamps).tz_localize("UTC")
    else:
        ts_idx = timestamps

    for j, sid in tqdm(enumerate(station_ids), total=N, desc="Loading ICON-D2 NWP", unit="station"):
        station_coord = (station_lats[j], station_lons[j])

        # --- find nearest grid point ---
        grid_candidates = []  # list of (dist, fname)

        for fh in forecast_hours:
            folder = os.path.join(nwp_path, "ML", fh, sid)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.endswith("_ML.parquet"):
                    continue
                coords = _parse_coords_from_filename(fname)
                if coords is None:
                    continue
                dist = geodesic(station_coord, coords).km
                grid_candidates.append((dist, fname))
            break  # filenames are the same across forecast hours

        if not grid_candidates:
            logger.warning("Station %s: no NWP grid point found — nwp_wind_speed will be NaN.", sid)
            continue

        nearest_fname = min(grid_candidates, key=lambda x: x[0])[1]
        logger.debug("Station %s: nearest grid point %s", sid, nearest_fname)

        # --- load that grid point Parquet across all forecast hours ---
        frames = []
        for fh in forecast_hours:
            fpath = os.path.join(nwp_path, "ML", fh, sid, nearest_fname)
            if not os.path.isfile(fpath):
                continue
            try:
                df_nwp = pd.read_parquet(fpath)
                frames.append(df_nwp)
            except Exception as exc:
                logger.debug("Skipping %s: %s", fpath, exc)

        if not frames:
            logger.warning("Station %s: could not load NWP Parquet — nwp_wind_speed will be NaN.", sid)
            continue

        data = pd.concat(frames, ignore_index=True)
        data = data[data["forecasttime"] > 0].copy()

        data["wind_speed_nwp"] = np.sqrt(data["u_wind"] ** 2 + data["v_wind"] ** 2)
        data["height"] = (data["toplevel"] + data["bottomlevel"]) / 2.0
        data["timestamp"] = pd.to_datetime(data["starttime"], utc=True) + pd.to_timedelta(
            data["forecasttime"], unit="h"
        )
        data["height_diff"] = (data["height"] - hub_height).abs()

        # For each timestamp: level closest to hub_height, then most recent forecast
        data = data.sort_values(["timestamp", "height_diff", "forecasttime"])
        data = data.drop_duplicates(subset=["timestamp"], keep="first")

        ts_data = data.set_index("timestamp")
        result[:, j]   = ts_data["wind_speed_nwp"].reindex(ts_idx).values
        # u_wind / v_wind from the same (best) height level — these are the 10m components
        # when hub_height=10 since toplevel=20/bottomlevel=0 gives height=10 → height_diff=0
        result_u[:, j] = ts_data["u_wind"].reindex(ts_idx).values
        result_v[:, j] = ts_data["v_wind"].reindex(ts_idx).values

    n_valid = np.sum(~np.isnan(result))
    logger.info(
        "NWP wind speed loaded: %.1f%% of values available.",
        100.0 * n_valid / (T * N),
    )
    return result, result_u, result_v


def load_ecmwf_wind_speed(
    parquet_path: str,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    """Load ECMWF wind speed at 10 m per station from a Parquet file, aligned to *timestamps*.

    Finds the nearest ECMWF grid point per station via geodesic distance.
    Computes wind_speed = sqrt(u_wind10m² + v_wind10m²).
    When multiple forecast runs cover the same valid_time, keeps the one
    with the smallest forecasttime (= most recent / shortest lead time).

    Returns
    -------
    Tuple of three ndarrays, each of shape (T, N):
      - wind speed (m/s) at 10 m
      - u-component at 10 m (``u_wind10m``)
      - v-component at 10 m (``v_wind10m``)
    """
    from geopy.distance import geodesic

    T = len(timestamps)
    N = len(station_lats)
    result   = np.full((T, N), np.nan, dtype=np.float64)
    result_u = np.full((T, N), np.nan, dtype=np.float64)
    result_v = np.full((T, N), np.nan, dtype=np.float64)

    if timestamps.tz is None:
        ts_idx = pd.DatetimeIndex(timestamps).tz_localize("UTC")
    else:
        ts_idx = timestamps

    logger.info("Loading ECMWF parquet: %s", parquet_path)
    df_full = pd.read_parquet(parquet_path)
    df_full = df_full[df_full["forecasttime"] > 0].copy()
    df_full["wind_speed_ecmwf"] = np.sqrt(df_full["u_wind10m"] ** 2 + df_full["v_wind10m"] ** 2)

    # Filter to relevant time range
    ts_min = ts_idx.min()
    ts_max = ts_idx.max()
    df_full = df_full[(df_full["valid_time"] >= ts_min) & (df_full["valid_time"] <= ts_max)]

    unique_grids = df_full[["grid_lat", "grid_lon"]].drop_duplicates().values

    for j in tqdm(range(N), desc="Loading ECMWF", unit="station"):
        lat, lon = station_lats[j], station_lons[j]

        # Find nearest grid point
        dists = [geodesic((lat, lon), (g[0], g[1])).km for g in unique_grids]
        nearest_idx = int(np.argmin(dists))
        g_lat, g_lon = unique_grids[nearest_idx]

        mask = (
            np.isclose(df_full["grid_lat"].values, g_lat, atol=1e-5) &
            np.isclose(df_full["grid_lon"].values, g_lon, atol=1e-5)
        )
        data = df_full[mask].copy()

        if data.empty:
            continue

        # Keep most recent forecast per valid_time (smallest forecasttime)
        data = data.sort_values(["valid_time", "forecasttime"])
        data = data.drop_duplicates(subset=["valid_time"], keep="first")

        ts_data = data.set_index("valid_time")
        result[:, j]   = ts_data["wind_speed_ecmwf"].reindex(ts_idx).values
        result_u[:, j] = ts_data["u_wind10m"].reindex(ts_idx).values
        result_v[:, j] = ts_data["v_wind10m"].reindex(ts_idx).values

    n_valid = np.sum(~np.isnan(result))
    logger.info(
        "ECMWF wind speed loaded: %.1f%% of values available.",
        100.0 * n_valid / (T * N),
    )
    return result, result_u, result_v


def load_data(config: dict) -> tuple:
    """Load station CSVs and metadata; return a (T, N) pivot and coordinate arrays.

    Reads:
      - ``{data.path}/wind_parameter.csv``  → lat, lon, altitude per station
      - ``{data.path}/synth_{station_id}.csv`` → per-station wind-speed time series

    Applies optional ``test_start`` / ``test_end`` filters from the config.
    If ``interpolation.interpolate_uv`` is true, also builds u/v component
    matrices from the wind_direction column (meteorological convention).

    Returns:
        pivot:       DataFrame (T×N) of wind_speed; index=timestamp, cols=station_id.
        lats:        (N,) latitude array.
        lons:        (N,) longitude array.
        alts:        (N,) altitude array in metres.
        station_ids: List[str] of station IDs in column order.
        u_matrix:    (T, N) zonal component array, or None if not requested.
        v_matrix:    (T, N) meridional component array, or None if not requested.
    """
    data_path = config["data"]["path"]
    interp_cfg = config.get("interpolation", {})
    do_uv = interp_cfg.get("interpolate_uv", False)
    # nwp_components / ecmwf_components: 'absolute' | 'components' | 'both'
    # 'absolute'   — only wind speed scalar (default, backward-compatible)
    # 'components' — only u/v components, no scalar
    # 'both'       — wind speed scalar + u/v components
    # False / true kept for backward compat (False → 'absolute', True → 'both')
    def _parse_components_mode(val) -> str:
        if val is False or val is None or val == "absolute":
            return "absolute"
        if val is True or val == "both":
            return "both"
        if val == "components":
            return "components"
        return "absolute"

    nwp_components_mode   = _parse_components_mode(interp_cfg.get("nwp_components",   "absolute"))
    ecmwf_components_mode = _parse_components_mode(interp_cfg.get("ecmwf_components", "absolute"))

    # --- auto-discover station IDs from Station_*.csv files ---
    station_files = sorted(
        f for f in os.listdir(data_path) if f.startswith("Station_") and f.endswith(".csv")
    )
    station_ids = [os.path.splitext(f)[0].replace("Station_", "") for f in station_files]
    exclude = [str(s) for s in config["data"].get("exclude_stations", [])]
    if exclude:
        station_ids = [s for s in station_ids if s not in exclude]
        logger.info("Excluding %d stations: %s", len(exclude), exclude)
    logger.info("Discovered %d stations in %s", len(station_ids), data_path)

    # --- metadata from stations_master.csv ---
    meta_path = config["data"].get("stations_master", "data/stations_master.csv")
    if not os.path.isabs(meta_path):
        meta_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), meta_path
        )
    meta = pd.read_csv(meta_path, dtype={"station_id": str})
    meta["station_id"] = meta["station_id"].astype(str)
    meta = meta.set_index("station_id")

    missing_meta = [s for s in station_ids if s not in meta.index]
    if missing_meta:
        logger.warning(
            "%d station IDs not found in stations_master — they will be skipped: %s ...",
            len(missing_meta), missing_meta[:5],
        )
        station_ids = [s for s in station_ids if s in meta.index]

    lats = meta.loc[station_ids, "latitude"].values.astype(np.float64)
    lons = meta.loc[station_ids, "longitude"].values.astype(np.float64)
    alts = meta.loc[station_ids, "station_height"].values.astype(np.float64)

    # --- per-station time series ---
    cols_to_load = ["station_id", "timestamp", "wind_speed"]
    # Dynamic RK features (all except 'altitude' which comes from metadata,
    # and 'nwp_wind_speed' / 'ecmwf_wind_speed' which are loaded separately)
    rk_feature_names_cfg = interp_cfg.get("rk_features") or []
    nwp_wind_requested   = "nwp_wind_speed"  in rk_feature_names_cfg
    ecmwf_wind_requested = "ecmwf_wind_speed" in rk_feature_names_cfg
    direction_in_rk      = "wind_direction"   in rk_feature_names_cfg
    # Load wind_direction if needed for u/v interpolation OR as RK feature
    need_direction = do_uv or direction_in_rk
    if need_direction:
        cols_to_load.append("wind_direction")
    dynamic_rk_cols = [
        f for f in rk_feature_names_cfg
        if f not in ("altitude", "nwp_wind_speed", "ecmwf_wind_speed", "wind_direction")
    ]
    cols_to_load.extend(dynamic_rk_cols)

    all_dfs = []
    for sid in tqdm(station_ids, desc="Reading station CSVs", unit="station"):
        fpath = os.path.join(data_path, f"Station_{sid}.csv")
        df = pd.read_csv(fpath, parse_dates=["timestamp"])
        df["station_id"] = sid
        available = [c for c in cols_to_load if c in df.columns]
        all_dfs.append(df[available])

    logger.info("Concatenating %d station DataFrames ...", len(all_dfs))
    combined = pd.concat(all_dfs, ignore_index=True)

    # Optional time filter — strip tz from timestamps in data if needed
    if combined["timestamp"].dt.tz is None:
        combined["timestamp"] = combined["timestamp"].dt.tz_localize("UTC")

    test_start = config["data"].get("test_start")
    test_end = config["data"].get("test_end")
    if test_start:
        combined = combined[combined["timestamp"] >= pd.Timestamp(test_start, tz="UTC")]
    if test_end:
        combined = combined[combined["timestamp"] <= pd.Timestamp(test_end, tz="UTC")]

    # --- KNN imputation of wind_speed on 10-min resolution (Timestamps × Stations) ---
    # Result is cached as Parquet so subsequent runs skip the expensive fit_transform.
    # Cache key encodes the station set, time bounds and k so a changed config
    # triggers a fresh computation.
    knn_k = int(interp_cfg.get("knn_impute_neighbors", 10))
    cache_dir = config["data"].get(
        "knn_cache_path", "/mnt/lambda1/nvme1/synthetic/knnimputer/wind"
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Build a short but stable cache key
    _t_start = (test_start or "start").replace(":", "").replace("-", "").replace(" ", "")
    _t_end   = (test_end   or "end"  ).replace(":", "").replace("-", "").replace(" ", "")
    import hashlib
    _sid_hash = hashlib.md5(",".join(sorted(station_ids)).encode()).hexdigest()[:8]
    cache_fname = f"wind_speed_knn{knn_k}_{_t_start}_{_t_end}_{_sid_hash}.parquet"
    cache_path  = os.path.join(cache_dir, cache_fname)

    if os.path.isfile(cache_path):
        logger.info("Loading KNN-imputed wind_speed from cache: %s", cache_path)
        ws_pivot_raw = pd.read_parquet(cache_path)
        ws_pivot_raw.index = pd.DatetimeIndex(ws_pivot_raw.index)
    else:
        logger.info("Building wind_speed pivot (%d stations) for KNN imputation ...", len(station_ids))
        ws_pivot_raw = combined.pivot_table(
            index="timestamp", columns="station_id", values="wind_speed", aggfunc="mean"
        ).reindex(columns=station_ids)
        n_missing_before = int(ws_pivot_raw.isna().sum().sum())
        if n_missing_before > 0:
            from sklearn.impute import KNNImputer
            logger.info(
                "KNN imputation (k=%d): %d missing values (%.2f%%) — fitting ...",
                knn_k, n_missing_before, 100.0 * n_missing_before / ws_pivot_raw.size,
            )
            imputer = KNNImputer(n_neighbors=knn_k)
            ws_imputed = imputer.fit_transform(ws_pivot_raw.values)
            ws_pivot_raw = pd.DataFrame(
                ws_imputed, index=ws_pivot_raw.index, columns=ws_pivot_raw.columns
            )
            logger.info("KNN imputation done.")
        else:
            logger.info("No missing wind_speed values — KNN imputation skipped.")
        ws_pivot_raw.to_parquet(cache_path)
        logger.info("KNN-imputed wind_speed cached → %s", cache_path)

    # Write imputed wind_speed back into combined
    ws_imputed_long = ws_pivot_raw.reset_index().melt(
        id_vars="timestamp", var_name="station_id", value_name="wind_speed_imputed"
    )
    combined = combined.merge(ws_imputed_long, on=["timestamp", "station_id"], how="left")
    combined["wind_speed"] = combined["wind_speed_imputed"]
    combined = combined.drop(columns=["wind_speed_imputed"])

    # --- KNN imputation of wind_direction via sin/cos transformation ---
    # Converts to (sin, cos) before imputation to handle circularity correctly
    # (e.g. 359° and 1° are correctly treated as neighbours), then back-transforms.
    if need_direction and "wind_direction" in combined.columns:
        cache_fname_dir = f"wind_direction_knn{knn_k}_{_t_start}_{_t_end}_{_sid_hash}.parquet"
        cache_path_dir  = os.path.join(cache_dir, cache_fname_dir)

        if os.path.isfile(cache_path_dir):
            logger.info("Loading KNN-imputed wind_direction from cache: %s", cache_path_dir)
            dir_pivot_raw = pd.read_parquet(cache_path_dir)
            dir_pivot_raw.index = pd.DatetimeIndex(dir_pivot_raw.index)
        else:
            logger.info("Building wind_direction pivot for KNN imputation (sin/cos) ...")
            dir_pivot_raw = combined.pivot_table(
                index="timestamp", columns="station_id", values="wind_direction", aggfunc="mean"
            ).reindex(columns=station_ids)
            n_missing_dir = int(dir_pivot_raw.isna().sum().sum())
            if n_missing_dir > 0:
                from sklearn.impute import KNNImputer
                logger.info(
                    "KNN imputation wind_direction (k=%d): %d missing values (%.2f%%) — fitting ...",
                    knn_k, n_missing_dir, 100.0 * n_missing_dir / dir_pivot_raw.size,
                )
                rad = np.deg2rad(dir_pivot_raw.values)
                sin_vals = np.sin(rad)   # (T, N)
                cos_vals = np.cos(rad)   # (T, N)
                # Stack sin and cos side-by-side: (T, 2N) — imputer sees both components
                combined_sc = np.concatenate([sin_vals, cos_vals], axis=1)
                imputer_dir = KNNImputer(n_neighbors=knn_k)
                imputed_sc  = imputer_dir.fit_transform(combined_sc)
                sin_imp = imputed_sc[:, :len(station_ids)]
                cos_imp = imputed_sc[:, len(station_ids):]
                # Back-transform: atan2 → degrees → [0, 360)
                dir_imp = np.rad2deg(np.arctan2(sin_imp, cos_imp)) % 360
                dir_pivot_raw = pd.DataFrame(
                    dir_imp, index=dir_pivot_raw.index, columns=dir_pivot_raw.columns
                )
                logger.info("KNN imputation wind_direction done.")
            else:
                logger.info("No missing wind_direction values — KNN imputation skipped.")
            dir_pivot_raw.to_parquet(cache_path_dir)
            logger.info("KNN-imputed wind_direction cached → %s", cache_path_dir)

        # Write imputed wind_direction back into combined
        dir_imputed_long = dir_pivot_raw.reset_index().melt(
            id_vars="timestamp", var_name="station_id", value_name="wind_direction_imputed"
        )
        combined = combined.merge(dir_imputed_long, on=["timestamp", "station_id"], how="left")
        combined["wind_direction"] = combined["wind_direction_imputed"]
        combined = combined.drop(columns=["wind_direction_imputed"])

    # Build (T, N) pivot for wind_speed — resample to hourly to match NWP/ECMWF resolution.
    # closed='left', label='left' (pandas default): timestamps 00:00..00:50 → hour 00:00,
    # 01:00..01:50 → hour 01:00, etc.
    pivot = combined.pivot_table(
        index="timestamp", columns="station_id", values="wind_speed", aggfunc="mean"
    )
    pivot = pivot.reindex(columns=station_ids)
    pivot.index = pd.DatetimeIndex(pivot.index)
    pivot = pivot.sort_index()
    pivot = pivot.resample("1h", closed="left", label="left").mean()

    # Fill any remaining NaN in the hourly pivot from the KNN-imputed wind_speed.
    # Stations whose raw CSV starts later than the common time range (e.g. a station
    # with no data before 2023-07-31) produce NaN in the pivot because the KNN
    # LEFT-join only fills rows that already exist in combined.  The KNN parquet,
    # however, was built by imputing the full (T×N) matrix and therefore contains
    # valid values for those gaps — resample it directly to hourly and fill.
    remaining_nan = int(pivot.isna().sum().sum())
    if remaining_nan > 0:
        ws_hourly = ws_pivot_raw.resample("1h", closed="left", label="left").mean()
        ws_hourly = ws_hourly.reindex(index=pivot.index, columns=pivot.columns)
        before = remaining_nan
        pivot = pivot.fillna(ws_hourly)
        after = int(pivot.isna().sum().sum())
        logger.info(
            "Filled %d hourly NaN in wind_speed pivot from KNN-imputed data "
            "(%d remaining after fill).", before - after, after
        )

    u_matrix = None
    v_matrix = None

    if do_uv and "wind_direction" in combined.columns:
        pivot_dir = combined.pivot_table(
            index="timestamp", columns="station_id", values="wind_direction", aggfunc="mean"
        )
        pivot_dir = pivot_dir.reindex(columns=station_ids)
        pivot_dir.index = pd.DatetimeIndex(pivot_dir.index)
        pivot_dir = pivot_dir.sort_index()
        pivot_dir = pivot_dir.resample("1h", closed="left", label="left").mean()

        u_arr, v_arr = wind_to_uv(pivot.values, pivot_dir.values)
        u_matrix = u_arr
        v_matrix = v_arr
        logger.info("Computed u/v wind component matrices.")
    elif do_uv:
        logger.warning("interpolate_uv=true but 'wind_direction' column not found in CSVs.")

    # --- RK feature matrices ---
    # 'altitude' is static (from metadata); everything else is loaded from CSVs.
    rk_feature_names = interp_cfg.get("rk_features")  # None if not set
    rk_static_features: dict = {}
    rk_dynamic_features: dict = {}

    if rk_feature_names:
        rk_static_features["altitude"] = alts   # always available as static
        dynamic_names = [
            f for f in rk_feature_names
            if f not in ("altitude", "nwp_wind_speed", "ecmwf_wind_speed")
        ]
        for fname in dynamic_names:
            if fname in combined.columns:
                piv = combined.pivot_table(
                    index="timestamp", columns="station_id", values=fname, aggfunc="mean"
                )
                piv = piv.reindex(columns=station_ids)
                piv.index = pd.DatetimeIndex(piv.index)
                piv = piv.sort_index()
                piv = piv.resample("1h", closed="left", label="left").mean()
                # Align to pivot index after resampling
                piv = piv.reindex(pivot.index)
                rk_dynamic_features[fname] = piv.values.astype(np.float64)
                logger.info("Loaded RK feature '%s' as (T, N) matrix.", fname)
            else:
                logger.warning("RK feature '%s' not found in CSVs — skipping.", fname)

        # Load NWP wind speed from raw ICON-D2 CSVs if requested
        if nwp_wind_requested:
            nwp_path = config["data"].get("nwp_path")
            if not nwp_path:
                logger.warning("nwp_wind_speed requested but data.nwp_path not set in config — skipping.")
            else:
                hub_height = float(interp_cfg.get("nwp_hub_height", 10.0))
                logger.info("=== Loading ICON-D2 NWP wind speed (hub_height=%.0f m) ===", hub_height)
                nwp_matrix, nwp_u_mat, nwp_v_mat = load_nwp_wind_speed(
                    nwp_path=nwp_path,
                    station_ids=station_ids,
                    station_lats=lats,
                    station_lons=lons,
                    timestamps=pivot.index,
                    hub_height=hub_height,
                )
                if nwp_components_mode in ("absolute", "both"):
                    rk_dynamic_features["nwp_wind_speed"] = nwp_matrix
                    logger.info("Loaded RK feature 'nwp_wind_speed' as (T, N) matrix.")
                if nwp_components_mode in ("components", "both"):
                    if np.all(np.isnan(nwp_u_mat)):
                        logger.warning(
                            "nwp_components=%s requested but u_10m/v_10m columns not found "
                            "in NWP parquet — component features skipped.", nwp_components_mode
                        )
                    else:
                        rk_dynamic_features["nwp_u_wind"] = nwp_u_mat
                        rk_dynamic_features["nwp_v_wind"] = nwp_v_mat
                        logger.info("Loaded NWP u/v components (nwp_u_wind, nwp_v_wind).")

        # Load ECMWF wind speed from Parquet file if requested
        if ecmwf_wind_requested:
            ecmwf_path = config["data"].get("ecmwf_path")
            if not ecmwf_path:
                logger.warning("ecmwf_wind_speed requested but data.ecmwf_path not set — skipping.")
            else:
                ecmwf_parquet = os.path.join(ecmwf_path, "ecmwf_wind_sl_full.parquet")
                if not os.path.isfile(ecmwf_parquet):
                    logger.warning("ECMWF parquet not found at %s — skipping.", ecmwf_parquet)
                else:
                    logger.info("=== Loading ECMWF wind speed from parquet ===")
                    ecmwf_matrix, ecmwf_u_mat, ecmwf_v_mat = load_ecmwf_wind_speed(
                        parquet_path=ecmwf_parquet,
                        station_lats=lats,
                        station_lons=lons,
                        timestamps=pivot.index,
                    )
                    if ecmwf_components_mode in ("absolute", "both"):
                        rk_dynamic_features["ecmwf_wind_speed"] = ecmwf_matrix
                        logger.info("Loaded RK feature 'ecmwf_wind_speed' as (T, N) matrix.")
                    if ecmwf_components_mode in ("components", "both"):
                        rk_dynamic_features["ecmwf_u_wind"] = ecmwf_u_mat
                        rk_dynamic_features["ecmwf_v_wind"] = ecmwf_v_mat
                        logger.info("Loaded ECMWF u/v components (ecmwf_u_wind, ecmwf_v_wind).")

        # Keep only features that were actually loaded; append dynamically-added
        # component features (nwp_u_wind etc.) that are not in the config list.
        rk_feature_names = [
            f for f in rk_feature_names
            if f in rk_static_features or f in rk_dynamic_features
        ]
        for extra in ("nwp_u_wind", "nwp_v_wind", "ecmwf_u_wind", "ecmwf_v_wind"):
            if extra in rk_dynamic_features and extra not in rk_feature_names:
                rk_feature_names.append(extra)
        logger.info("Final RK feature list: %s", rk_feature_names)

        # Diagnostics: NaN coverage per dynamic feature
        for fname, arr in rk_dynamic_features.items():
            nan_pct = 100.0 * np.sum(np.isnan(arr)) / arr.size
            logger.info("  Feature '%s': %.1f%% NaN", fname, nan_pct)

        # Drop timestamps where MORE than rk_nan_threshold of stations have NaN in any
        # dynamic feature.  Default 0.5 (50%) keeps timestamps where only a few
        # stations are missing, while still dropping timestamps where a whole
        # data source (e.g. ECMWF) is unavailable (all stations NaN).
        rk_nan_threshold = float(interp_cfg.get("rk_nan_threshold", 0.5))
        if rk_dynamic_features:
            valid_mask = np.ones(len(pivot), dtype=bool)
            for fname, arr in rk_dynamic_features.items():
                nan_frac_per_ts = np.mean(np.isnan(arr), axis=1)   # (T,) fraction of stations with NaN
                valid_mask &= nan_frac_per_ts <= rk_nan_threshold
                n_over = int((nan_frac_per_ts > rk_nan_threshold).sum())
                if n_over:
                    logger.info(
                        "  Feature '%s': %d timestamps have >%.0f%% stations with NaN → will be dropped.",
                        fname, n_over, 100 * rk_nan_threshold,
                    )
            n_dropped = int((~valid_mask).sum())
            if n_dropped > 0:
                logger.info(
                    "Dropping %d timestamps with NaN in dynamic RK features (%.1f%%).",
                    n_dropped, 100.0 * n_dropped / len(pivot),
                )
                pivot = pivot.iloc[valid_mask]
                if u_matrix is not None:
                    u_matrix = u_matrix[valid_mask]
                    v_matrix = v_matrix[valid_mask]
                rk_dynamic_features = {k: v[valid_mask] for k, v in rk_dynamic_features.items()}

    logger.info(
        "Loaded data: %d timestamps × %d stations", len(pivot), len(station_ids)
    )
    return pivot, lats, lons, alts, station_ids, u_matrix, v_matrix, rk_feature_names, rk_static_features, rk_dynamic_features


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_variogram_plot(
    lags_emp: np.ndarray,
    sv_emp: np.ndarray,
    variogram_params: dict,
    output_dir: str,
    prefix: str,
) -> None:
    """Save a plot of the empirical semivariance and the fitted variogram model."""
    from utils.interpolation import _get_variogram_fn

    model_name = variogram_params.get("model", "spherical")
    model_fn = _get_variogram_fn(model_name)

    h_fit = np.linspace(0, lags_emp[-1] * 1.05, 300)
    sv_fit = model_fn(
        h_fit,
        variogram_params["nugget"],
        variogram_params["psill"],
        variogram_params["range"],
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(lags_emp, sv_emp, color="steelblue", zorder=3, label="Empirisch")
    ax.plot(h_fit, sv_fit, color="tomato", linewidth=1.8, label=f"{model_name.capitalize()} (Fit)")

    p = variogram_params
    ax.axhline(p["nugget"], color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(p["nugget"] + p["psill"], color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(p["range"], color="gray", linewidth=0.8, linestyle="--")
    ax.text(
        p["range"] + lags_emp[-1] * 0.01,
        sv_emp.min() * 0.95,
        f"range={p['range']:.1f} km",
        fontsize=8,
        color="gray",
    )
    param_str = f"nugget={p['nugget']:.3f}  psill={p['psill']:.3f}  range={p['range']:.1f} km"
    ax.set_title(f"Globales Variogramm ({model_name}) — {param_str}", fontsize=9)
    ax.set_xlabel("Lag-Distanz (km)")
    ax.set_ylabel("Semivarianz (m²/s²)")
    ax.legend(fontsize=9)
    plt.tight_layout()

    fname = os.path.join(output_dir, f"{prefix}_variogram.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info("Saved variogram plot → %s", fname)


def save_scatter_plot(df: pd.DataFrame, method: str, output_dir: str, prefix: str) -> None:
    """Save a scatter plot of observed vs predicted for *method*.

    For scalar wind-speed methods ('idw', 'ok', 'rk') the observed column is
    'wind_speed_observed'.  For component methods ('u_idw', 'v_ok', …) the
    observed column is inferred from the method prefix.
    """
    pred_col = f"{method}_pred"

    # Determine the matching observed column
    comp = method.split("_")[0]  # 'idw'/'ok'/'rk' → comp=='idw'; 'u_idw' → comp=='u'
    if comp in ("u", "v"):
        obs_col = f"{comp}_observed"
        xlabel = f"Observed {comp} component (m/s)"
        ylabel = f"{method.upper()} predicted (m/s)"
    else:
        obs_col = "wind_speed_observed"
        xlabel = "Observed wind speed (m/s)"
        ylabel = f"{method.upper()} predicted (m/s)"

    plot_df = df.dropna(subset=[obs_col, pred_col])
    obs = plot_df[obs_col]
    pred = plot_df[pred_col]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs, pred, alpha=0.15, s=4, color="steelblue", rasterized=True)
    lim = [min(obs.min(), pred.min()) - 0.5, max(obs.max(), pred.max()) + 0.5]
    ax.plot(lim, lim, "r--", linewidth=1.0, label="1:1 line")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{method.upper()} — Observed vs. Predicted (LOO-CV)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    fname = os.path.join(output_dir, f"{prefix}_scatter_{method}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info("Saved scatter plot → %s", fname)


# ---------------------------------------------------------------------------
# Optimal parameter loading from Optuna study
# ---------------------------------------------------------------------------

def load_best_params_from_study(config: dict) -> dict | None:
    """Load the best trial parameters from the Optuna study defined in config.hpo.

    Returns a dict of param_name → value, or None if no study / no completed
    trials were found (caller should fall back to interpolation config values).
    """
    import optuna
    from pathlib import Path

    hpo_cfg = config.get("hpo", {})
    study_name = hpo_cfg.get("study_name")
    if not study_name:
        logger.warning("get_theta_opt=true but no hpo.study_name in config — using config params.")
        return None

    storage_url = os.environ.get("OPTUNA_STORAGE")
    if storage_url:
        storage = storage_url
        storage_label = "PostgreSQL (OPTUNA_STORAGE)"
    else:
        studies_dir = Path(__file__).parent.parent / "studies"
        sqlite_path = studies_dir / f"{study_name}.db"
        if not sqlite_path.exists():
            logger.warning(
                "get_theta_opt=true but no SQLite study found at %s — using config params.", sqlite_path
            )
            return None
        storage = f"sqlite:///{sqlite_path}"
        storage_label = str(sqlite_path)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            logger.warning(
                "Study '%s' found in %s but has no completed trials — using config params.",
                study_name, storage_label,
            )
            return None
        best = study.best_trial
        logger.info(
            "Loaded best trial #%d from study '%s' (%s)  RK RMSE=%.4f",
            best.number, study_name, storage_label, best.value,
        )
        for k_, v_ in best.params.items():
            logger.info("  %-30s %s", k_, v_)
        return best.params
    except Exception as exc:
        logger.warning("Failed to load Optuna study '%s': %s — using config params.", study_name, exc)
        return None


def apply_hpo_params_to_config(config: dict, hpo_params: dict) -> None:
    """Write best HPO parameters back into config['interpolation'] in-place.

    Only keys that are recognised interpolation parameters are applied;
    HPO-only flags (use_direction, use_temperature_2m, …) are handled
    separately by adjusting rk_features.
    """
    interp_cfg = config.setdefault("interpolation", {})
    aniso_cfg  = interp_cfg.setdefault("anisotropy", {})

    mapping = {
        "k_neighbors":       ("interpolation", "k_neighbors"),
        "idw_power":         ("interpolation", "idw_power"),
        "variogram_model":   ("interpolation", "variogram_model"),
        "n_variogram_lags":  ("interpolation", "n_variogram_lags"),
        "variogram_detrend": ("interpolation", "variogram_detrend"),
        "anisotropy_enabled":("anisotropy",    "enabled"),
        "anisotropy_angle":  ("anisotropy",    "angle"),
        "anisotropy_ratio":  ("anisotropy",    "ratio"),
        "nwp_components":    ("interpolation", "nwp_components"),
        "ecmwf_components":  ("interpolation", "ecmwf_components"),
        "interpolate_uv":    ("interpolation", "interpolate_uv"),
    }

    for param, (section, key) in mapping.items():
        if param in hpo_params:
            target = aniso_cfg if section == "anisotropy" else interp_cfg
            target[key] = hpo_params[param]

    # Feature-inclusion flags: rebuild rk_features accordingly
    rk_features = list(interp_cfg.get("rk_features") or [])
    if "use_temperature_2m" in hpo_params:
        if hpo_params["use_temperature_2m"] and "temperature_2m" not in rk_features:
            rk_features.append("temperature_2m")
        elif not hpo_params["use_temperature_2m"]:
            rk_features = [f for f in rk_features if f != "temperature_2m"]
    if "use_direction" in hpo_params:
        if hpo_params["use_direction"] and "wind_direction" not in rk_features:
            rk_features.append("wind_direction")
        elif not hpo_params["use_direction"]:
            rk_features = [f for f in rk_features if f != "wind_direction"]
    interp_cfg["rk_features"] = rk_features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spatial wind-speed interpolation: IDW / OK / RK with LOO-CV"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("-s", "--suffix", default="", help="Suffix appended to all output filenames, e.g. '_nwp' or '_exp'")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)
    log_fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=log_level, format=log_fmt, datefmt="%H:%M:%S")

    config = load_config(args.config)
    interp_cfg = config.get("interpolation", {})
    out_cfg = config.get("output", {})

    # Optionally override interpolation params with best HPO trial
    get_theta_opt = bool(interp_cfg.get("get_theta_opt", False))
    if get_theta_opt:
        logger.info("get_theta_opt=true — loading best parameters from Optuna study ...")
        best_params = load_best_params_from_study(config)
        if best_params:
            apply_hpo_params_to_config(config, best_params)
            interp_cfg = config["interpolation"]   # re-bind after in-place update
            logger.info("Interpolation config updated with best HPO parameters.")
        else:
            logger.info("Falling back to interpolation config parameters.")
    else:
        logger.info("get_theta_opt=false — using interpolation config parameters.")

    k = int(interp_cfg.get("k_neighbors", 8))
    idw_power = float(interp_cfg.get("idw_power", 2.0))
    n_lags = int(interp_cfg.get("n_variogram_lags", 20))
    max_dist = interp_cfg.get("max_variogram_dist")      # None → auto
    variogram_model = interp_cfg.get("variogram_model", "spherical")
    variogram_segments = int(interp_cfg.get("variogram_segments", 1))
    variogram_detrend = bool(interp_cfg.get("variogram_detrend", False))
    aniso_cfg = interp_cfg.get("anisotropy", {})
    output_dir = out_cfg.get("path", "data/geostatistics")
    config_stem = os.path.splitext(os.path.basename(args.config))[0]  # e.g. "config_spatial_interpolation"
    config_stem = config_stem.removeprefix("config_")                  # e.g. "spatial_interpolation"
    prefix = config_stem + (f"_{args.suffix}" if args.suffix else "")

    os.makedirs(output_dir, exist_ok=True)

    # Add file handler now that we know prefix and output_dir
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{prefix}.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    logger.info("=== Output prefix: %s  |  dir: %s ===", prefix, output_dir)
    logger.info("=== Log file: %s ===", log_path)

    # 1. Load data
    logger.info("=== Loading data ===")
    pivot, lats, lons, alts, station_ids, u_matrix, v_matrix, \
        rk_feature_names, rk_static_features, rk_dynamic_features = load_data(config)
    values_matrix = pivot.values  # (T, N)
    timestamps = pivot.index
    T = len(timestamps)

    # 2. Distance matrix (geodesic) — used for k-NN selection and IDW
    logger.info("=== Computing %d×%d geodesic distance matrix ===", len(station_ids), len(station_ids))
    dist_matrix = compute_distance_matrix(lats, lons)

    # 3. Anisotropic distance matrix (optional) — used for kriging system only
    aniso_dist_matrix = None
    if aniso_cfg.get("enabled", False):
        angle = float(aniso_cfg["angle"])
        ratio = float(aniso_cfg["ratio"])
        logger.info(
            "=== Computing anisotropic distance matrix (angle=%.1f°, ratio=%.3f) ===",
            angle, ratio,
        )
        aniso_dist_matrix = compute_anisotropic_distance_matrix(lats, lons, angle, ratio)

    # 4. Variogram fitting — one per segment
    #    variogram_segments=1  → one global variogram (current behaviour)
    #    variogram_segments=0  → one per timestamp (T segments)
    #    variogram_segments=N  → N equal time-blocks
    if variogram_segments == 0:
        n_segs = T
        seg_slices = [slice(t, t + 1) for t in range(T)]
        segment_indices = np.arange(T, dtype=int)
        logger.info("=== Per-timestamp mode: fitting %d variograms ===", T)
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
        logger.info("=== Fitting %d segment variograms ===", n_segs)

    variogram_params_list = []
    flat_variogram_timestamps = []
    for seg_i, sl in enumerate(seg_slices):
        seg_vals = values_matrix[sl]
        lags_emp, sv_emp = compute_empirical_semivariance(
            seg_vals, dist_matrix, n_lags=n_lags, max_dist=max_dist,
            detrend=variogram_detrend,
        )
        vp = fit_global_variogram(lags_emp, sv_emp, model=variogram_model)
        variogram_params_list.append(vp)
        if n_segs <= 12 or seg_i == 0 or seg_i == n_segs - 1:
            logger.info("  Segment %d/%d: %s", seg_i + 1, n_segs, vp)
        # Collect timestamps where variogram is flat (nugget + psill ≈ 0)
        if vp["nugget"] + vp["psill"] < 1e-6:
            flat_variogram_timestamps.append(str(timestamps[sl.start]))

    if flat_variogram_timestamps:
        flat_df = pd.DataFrame({"timestamp": flat_variogram_timestamps})
        flat_path = os.path.join(output_dir, f"{prefix}_flat_variogram_timestamps.csv")
        flat_df.to_csv(flat_path, index=False)
        logger.info(
            "%d timestamps with flat variogram (sill≈0) saved → %s",
            len(flat_variogram_timestamps), flat_path,
        )

    # Save variogram plot for the first (or only) segment
    lags_emp0, sv_emp0 = compute_empirical_semivariance(
        values_matrix[seg_slices[0]], dist_matrix, n_lags=n_lags, max_dist=max_dist,
        detrend=variogram_detrend,
    )
    save_variogram_plot(lags_emp0, sv_emp0, variogram_params_list[0], output_dir, prefix)

    # 5. LOO-CV
    logger.info("=== Running LOO-CV (%d stations × %d timestamps) ===", len(station_ids), T)
    if u_matrix is not None:
        logger.info("u/v component interpolation enabled.")
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
        u_matrix=u_matrix,
        v_matrix=v_matrix,
        rk_feature_names=rk_feature_names or None,
        rk_static_features=rk_static_features or None,
        rk_dynamic_features=rk_dynamic_features or None,
        aniso_dist_matrix=aniso_dist_matrix,
    )

    # 5. Save raw predictions (combined)
    pred_path = os.path.join(output_dir, f"{prefix}_loo_predictions.csv")
    predictions.to_csv(pred_path, index=False)
    logger.info("Saved LOO predictions → %s", pred_path)

    # 5b. Save per-station timeseries CSVs to target_path
    target_path = out_cfg.get("target_path", os.path.join(output_dir, f"{prefix}_per_station"))
    os.makedirs(target_path, exist_ok=True)
    for sid in station_ids:
        sid_df = predictions[predictions["station_id"] == sid].copy()
        sid_df = sid_df.sort_values("timestamp").reset_index(drop=True)
        sid_path = os.path.join(target_path, f"Station_{sid}.csv")
        sid_df.to_csv(sid_path, index=False)
    logger.info("Saved %d per-station timeseries CSVs → %s", len(station_ids), target_path)

    # 6. Metrics
    logger.info("=== Computing metrics ===")
    per_station_df, summary_df = compute_metrics(predictions)

    ps_path = os.path.join(output_dir, f"{prefix}_results_per_station.csv")
    per_station_df.to_csv(ps_path, index=False)
    logger.info("Saved per-station metrics → %s", ps_path)

    sm_path = os.path.join(output_dir, f"{prefix}_results_summary.csv")
    summary_df.to_csv(sm_path, index=False)
    logger.info("Saved summary metrics → %s", sm_path)

    # 7. Scatter plots for scalar wind speed
    logger.info("=== Saving scatter plots ===")
    for method in ["idw", "ok", "rk"]:
        save_scatter_plot(predictions, method, output_dir, prefix)

    # Scatter plots for u/v and derived vector speed
    if u_matrix is not None:
        for comp in ("u", "v"):
            for method in ("idw", "ok", "rk"):
                col = f"{comp}_{method}_pred"
                if col in predictions.columns:
                    save_scatter_plot(predictions, f"{comp}_{method}", output_dir, prefix)

    logger.info("=== Done ===")
    print("\nGlobal summary (mean over stations):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
