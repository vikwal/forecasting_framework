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
    """Parse grid point coordinates from ICON-D2 CSV filename.

    Expected format: ``<lat_int>_<lat_frac>_<lon_int>_<lon_frac>_ML.csv``
    Example: ``52_15_10_45_ML.csv`` → (52.15, 10.45)

    Returns (lat, lon) as floats, or None if parsing fails.
    """
    stem = fname.replace("_ML.csv", "")
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
    """Load ICON-D2 NWP wind speed per station, aligned to *timestamps*.

    File structure::

        <nwp_path>/ML/<forecast_hour>/<station_id>/<lat_int>_<lat_frac>_<lon_int>_<lon_frac>_ML.csv

    For each station:
    - Parses grid point coordinates from all CSV filenames in the folder.
    - Selects the nearest grid point via geodesic distance.
    - Loads that single CSV across all forecast hours.
    - Selects the height level closest to *hub_height* metres.
    - When multiple forecast runs cover the same timestamp, keeps the one
      with the smallest forecasttime (= most recent / shortest lead time).

    Returns
    -------
    ndarray of shape (T, N), NaN where no NWP data is available.
    """
    from geopy.distance import geodesic

    T = len(timestamps)
    N = len(station_ids)
    result = np.full((T, N), np.nan, dtype=np.float64)

    if timestamps.tz is None:
        ts_idx = pd.DatetimeIndex(timestamps).tz_localize("UTC")
    else:
        ts_idx = timestamps

    for j, sid in tqdm(enumerate(station_ids), total=N, desc="Loading ICON-D2 NWP", unit="station"):
        station_coord = (station_lats[j], station_lons[j])

        # --- find nearest grid point (same across all forecast hours) ---
        nearest_fname = None
        min_dist = np.inf

        # Use first available forecast hour to discover grid point filenames
        for fh in forecast_hours:
            folder = os.path.join(nwp_path, "ML", fh, sid)
            if not os.path.isdir(folder):
                continue
            candidates = [f for f in os.listdir(folder) if f.endswith("_ML.csv")]
            for fname in candidates:
                coords = _parse_coords_from_filename(fname)
                if coords is None:
                    continue
                dist = geodesic(station_coord, coords).km
                if dist < min_dist:
                    min_dist = dist
                    nearest_fname = fname
            break  # grid point filenames are the same for all forecast hours

        if nearest_fname is None:
            logger.warning("Station %s: no NWP grid point found — nwp_wind_speed will be NaN.", sid)
            continue

        logger.debug("Station %s: nearest grid point %s (%.2f km)", sid, nearest_fname, min_dist)

        # --- load that grid point CSV across all forecast hours ---
        frames = []
        for fh in forecast_hours:
            fpath = os.path.join(nwp_path, "ML", fh, sid, nearest_fname)
            if not os.path.isfile(fpath):
                continue
            try:
                df_nwp = pd.read_csv(fpath, parse_dates=["starttime"])
                frames.append(df_nwp)
            except Exception as exc:
                logger.debug("Skipping %s: %s", fpath, exc)

        if not frames:
            logger.warning("Station %s: could not load NWP CSV — nwp_wind_speed will be NaN.", sid)
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

        nwp_series = data.set_index("timestamp")["wind_speed_nwp"]
        result[:, j] = nwp_series.reindex(ts_idx).values

    n_valid = np.sum(~np.isnan(result))
    logger.info(
        "NWP wind speed loaded: %.1f%% of values available.",
        100.0 * n_valid / (T * N),
    )
    return result


def load_ecmwf_wind_speed(
    station_ids: list,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    timestamps: pd.DatetimeIndex,
    db_url: str,
) -> np.ndarray:
    """Load ECMWF wind speed at 10 m per station from PostGIS database, aligned to *timestamps*.

    Uses the nearest ECMWF grid point (rank=1) via PostGIS KNN operator.
    Computes wind_speed = sqrt(u_wind10m² + v_wind10m²).
    When multiple forecast runs cover the same timestamp, keeps the one
    with the smallest forecasttime (= most recent / shortest lead time).

    Returns
    -------
    ndarray of shape (T, N), NaN where no ECMWF data is available.
    """
    try:
        from sqlalchemy import create_engine
    except ImportError:
        raise ImportError("sqlalchemy required: pip install sqlalchemy psycopg2-binary")

    T = len(timestamps)
    N = len(station_ids)
    result = np.full((T, N), np.nan, dtype=np.float64)

    if timestamps.tz is None:
        ts_idx = pd.DatetimeIndex(timestamps).tz_localize("UTC")
    else:
        ts_idx = timestamps

    ts_min = ts_idx.min().strftime("%Y-%m-%d %H:%M:%S%z")
    ts_max = ts_idx.max().strftime("%Y-%m-%d %H:%M:%S%z")

    for j, sid in tqdm(enumerate(station_ids), total=N, desc="Loading ECMWF", unit="station"):
        lat, lon = station_lats[j], station_lons[j]

        query = f"""
        WITH nearest AS (
            SELECT geom,
                   ROW_NUMBER() OVER (
                       ORDER BY geom <-> ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)
                   ) AS rank
            FROM ecmwf_grid_points
            ORDER BY geom <-> ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)
            LIMIT 1
        )
        SELECT e.starttime, e.forecasttime, e.u_wind_10m, e.v_wind_10m
        FROM ecmwf_wind_sl e
        JOIN nearest n ON e.geom = n.geom
        WHERE e.forecasttime > 0
          AND e.starttime BETWEEN '{ts_min}' AND '{ts_max}'
        ORDER BY e.starttime, e.forecasttime
        """

        engine = create_engine(db_url)
        try:
            data = pd.read_sql(query, engine)
        except Exception as exc:
            logger.warning("Station %s: ECMWF query failed (%s) — skipping.", sid, exc)
            continue
        finally:
            engine.dispose()

        if data.empty:
            logger.warning("Station %s: no ECMWF data returned.", sid)
            continue

        data["starttime"] = pd.to_datetime(data["starttime"], utc=True)
        data["timestamp"] = data["starttime"] + pd.to_timedelta(data["forecasttime"], unit="h")
        data["wind_speed_ecmwf"] = np.sqrt(data["u_wind_10m"] ** 2 + data["v_wind_10m"] ** 2)

        # Keep most recent forecast per timestamp (smallest forecasttime)
        data = data.sort_values(["timestamp", "forecasttime"])
        data = data.drop_duplicates(subset=["timestamp"], keep="first")

        ecmwf_series = data.set_index("timestamp")["wind_speed_ecmwf"]
        result[:, j] = ecmwf_series.reindex(ts_idx).values

    n_valid = np.sum(~np.isnan(result))
    logger.info(
        "ECMWF wind speed loaded: %.1f%% of values available.",
        100.0 * n_valid / (T * N),
    )
    return result


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
    station_ids = [str(s) for s in config["data"]["files"]]
    interp_cfg = config.get("interpolation", {})
    do_uv = interp_cfg.get("interpolate_uv", False)

    # --- metadata ---
    meta_path = os.path.join(data_path, "wind_parameter.csv")
    meta = pd.read_csv(meta_path, sep=";", dtype={"park_id": str})
    meta["park_id"] = meta["park_id"].astype(str)
    meta = meta.set_index("park_id")

    missing_meta = [s for s in station_ids if s not in meta.index]
    if missing_meta:
        raise ValueError(
            f"{len(missing_meta)} station IDs not found in wind_parameter.csv: "
            f"{missing_meta[:5]} ..."
        )

    lats = meta.loc[station_ids, "latitude"].values.astype(np.float64)
    lons = meta.loc[station_ids, "longitude"].values.astype(np.float64)
    alts = meta.loc[station_ids, "altitude"].values.astype(np.float64)

    # --- per-station time series ---
    cols_to_load = ["station_id", "timestamp", "wind_speed"]
    if do_uv:
        cols_to_load.append("wind_direction")
    # Dynamic RK features (all except 'altitude' which comes from metadata,
    # and 'nwp_wind_speed' which is computed from wind_speed_t* columns)
    rk_feature_names_cfg = interp_cfg.get("rk_features") or []
    nwp_wind_requested = "nwp_wind_speed" in rk_feature_names_cfg
    ecmwf_wind_requested = "ecmwf_wind_speed" in rk_feature_names_cfg
    dynamic_rk_cols = [
        f for f in rk_feature_names_cfg
        if f not in ("altitude", "nwp_wind_speed", "ecmwf_wind_speed")
    ]
    cols_to_load.extend(dynamic_rk_cols)

    all_dfs = []
    for sid in station_ids:
        fpath = os.path.join(data_path, f"synth_{sid}.csv")
        df = pd.read_csv(fpath, sep=";", parse_dates=["timestamp"])
        df["station_id"] = sid
        available = [c for c in cols_to_load if c in df.columns]
        all_dfs.append(df[available])

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

    # Build (T, N) pivot for wind_speed — preserve station order from config
    pivot = combined.pivot_table(
        index="timestamp", columns="station_id", values="wind_speed", aggfunc="first"
    )
    pivot = pivot.reindex(columns=station_ids)
    pivot.index = pd.DatetimeIndex(pivot.index)
    pivot = pivot.sort_index()

    u_matrix = None
    v_matrix = None

    if do_uv and "wind_direction" in combined.columns:
        pivot_dir = combined.pivot_table(
            index="timestamp", columns="station_id", values="wind_direction", aggfunc="first"
        )
        pivot_dir = pivot_dir.reindex(columns=station_ids)
        pivot_dir.index = pd.DatetimeIndex(pivot_dir.index)
        pivot_dir = pivot_dir.sort_index()

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
        dynamic_names = [f for f in rk_feature_names if f not in ("altitude", "nwp_wind_speed", "ecmwf_wind_speed")]
        for fname in dynamic_names:
            if fname in combined.columns:
                piv = combined.pivot_table(
                    index="timestamp", columns="station_id", values=fname, aggfunc="first"
                )
                piv = piv.reindex(columns=station_ids)
                piv.index = pd.DatetimeIndex(piv.index)
                piv = piv.sort_index()
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
                nwp_matrix = load_nwp_wind_speed(
                    nwp_path=nwp_path,
                    station_ids=station_ids,
                    station_lats=lats,
                    station_lons=lons,
                    timestamps=pivot.index,
                    hub_height=hub_height,
                )
                rk_dynamic_features["nwp_wind_speed"] = nwp_matrix
                logger.info("Loaded RK feature 'nwp_wind_speed' as (T, N) matrix.")

        # Load ECMWF wind speed from PostGIS database if requested
        if ecmwf_wind_requested:
            db_url = os.environ.get("ECMWF_WIND_SL_URL")
            if not db_url:
                logger.warning("ecmwf_wind_speed requested but ECMWF_WIND_SL_URL not set — skipping.")
            else:
                logger.info("=== Loading ECMWF wind speed from database ===")
                ecmwf_matrix = load_ecmwf_wind_speed(
                    station_ids=station_ids,
                    station_lats=lats,
                    station_lons=lons,
                    timestamps=pivot.index,
                    db_url=db_url,
                )
                rk_dynamic_features["ecmwf_wind_speed"] = ecmwf_matrix
                logger.info("Loaded RK feature 'ecmwf_wind_speed' as (T, N) matrix.")

        # Keep only features that were actually loaded
        rk_feature_names = [
            f for f in rk_feature_names
            if f in rk_static_features or f in rk_dynamic_features
        ]
        logger.info("Final RK feature list: %s", rk_feature_names)

        # Diagnostics: NaN coverage per dynamic feature
        for fname, arr in rk_dynamic_features.items():
            nan_pct = 100.0 * np.sum(np.isnan(arr)) / arr.size
            logger.info("  Feature '%s': %.1f%% NaN", fname, nan_pct)

        # Drop timestamps where any dynamic feature has NaN in any station
        if rk_dynamic_features:
            valid_mask = np.ones(len(pivot), dtype=bool)
            for fname, arr in rk_dynamic_features.items():
                valid_mask &= ~np.any(np.isnan(arr), axis=1)
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

    # 5. Save raw predictions
    pred_path = os.path.join(output_dir, f"{prefix}_loo_predictions.csv")
    predictions.to_csv(pred_path, index=False)
    logger.info("Saved LOO predictions → %s", pred_path)

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
