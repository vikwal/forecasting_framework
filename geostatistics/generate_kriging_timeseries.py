#!/usr/bin/env python3
"""Generate kriging interpolated wind speed timeseries for each station.

Unlike the LOO cross-validation script, this generates a complete interpolated
timeseries for each target station using ALL other stations (excluding the target).

Usage:
    python geostatistics/generate_kriging_timeseries.py --config configs/config_spatial_interpolation.yaml
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.interpolation import (
    compute_anisotropic_distance_matrix,
    compute_distance_matrix,
    compute_empirical_semivariance,
    fit_global_variogram,
)

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load YAML config from *path*."""
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_coords_from_filename(fname: str):
    """Parse grid point coordinates from ICON-D2 CSV filename."""
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
    """Load ICON-D2 NWP wind speed per station."""
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

        nearest_fname = None
        min_dist = np.inf

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
            break

        if nearest_fname is None:
            logger.warning("Station %s: no NWP grid point found — nwp_wind_speed will be NaN.", sid)
            continue

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

        data = data.sort_values(["timestamp", "height_diff", "forecasttime"])
        data = data.drop_duplicates(subset=["timestamp"], keep="first")

        nwp_series = data.set_index("timestamp")["wind_speed_nwp"]
        result[:, j] = nwp_series.reindex(ts_idx).values

    return result


def load_ecmwf_wind_speed(
    station_ids: list,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    timestamps: pd.DatetimeIndex,
    db_url: str,
) -> np.ndarray:
    """Load ECMWF wind speed at 10 m per station from PostGIS database."""
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

        data = data.sort_values(["timestamp", "forecasttime"])
        data = data.drop_duplicates(subset=["timestamp"], keep="first")

        ecmwf_series = data.set_index("timestamp")["wind_speed_ecmwf"]
        result[:, j] = ecmwf_series.reindex(ts_idx).values

    return result


def load_temperature_2m(
    station_ids: list,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    timestamps: pd.DatetimeIndex,
    db_url: str,
) -> np.ndarray:
    """Load temperature_2m per station from PostGIS database."""
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

    for j, sid in tqdm(enumerate(station_ids), total=N, desc="Loading temperature_2m", unit="station"):
        lat, lon = station_lats[j], station_lons[j]

        query = f"""
        WITH nearest AS (
            SELECT geom
            FROM ecmwf_grid_points
            ORDER BY geom <-> ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)
            LIMIT 1
        )
        SELECT e.starttime, e.forecasttime, e.temperature_2m
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
            logger.warning("Station %s: temperature_2m query failed (%s) — skipping.", sid, exc)
            continue
        finally:
            engine.dispose()

        if data.empty:
            logger.warning("Station %s: no temperature_2m data returned.", sid)
            continue

        data["starttime"] = pd.to_datetime(data["starttime"], utc=True)
        data["timestamp"] = data["starttime"] + pd.to_timedelta(data["forecasttime"], unit="h")

        data = data.sort_values(["timestamp", "forecasttime"])
        data = data.drop_duplicates(subset=["timestamp"], keep="first")

        temp_series = data.set_index("timestamp")["temperature_2m"]
        result[:, j] = temp_series.reindex(ts_idx).values

    return result


def load_data(config: dict):
    """Load station data and regression kriging features."""
    data_cfg = config["data"]
    interp_cfg = config["interpolation"]

    data_path = data_cfg["path"]
    nwp_path = data_cfg.get("nwp_path")
    station_ids = [str(s).zfill(5) for s in data_cfg["files"]]

    test_start = data_cfg.get("test_start")
    test_end = data_cfg.get("test_end")

    logger.info(f"Loading data for {len(station_ids)} stations from {data_path}")

    # Load station metadata
    meta_path = "/mnt/nvme1/synthetic/wind/wind_hourly_age_20251103/wind_parameter.csv"
    meta_df = pd.read_csv(meta_path, sep=";")
    meta_df["park_id"] = meta_df["park_id"].astype(str).str.zfill(5)
    meta_df = meta_df[meta_df["park_id"].isin(station_ids)].set_index("park_id")

    lats = np.array([meta_df.loc[sid, "latitude"] for sid in station_ids])
    lons = np.array([meta_df.loc[sid, "longitude"] for sid in station_ids])
    alts = np.array([meta_df.loc[sid, "altitude"] for sid in station_ids])
    hub_height = 10.0

    # Load wind speed data
    frames = []
    for sid in station_ids:
        fpath = os.path.join(data_path, f"synth_{sid}.csv")
        df = pd.read_csv(fpath, sep=";", parse_dates=["timestamp"])
        df["station_id"] = sid
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if test_start:
        combined = combined[combined["timestamp"] >= test_start]
    if test_end:
        combined = combined[combined["timestamp"] <= test_end]

    pivot = combined.pivot(index="timestamp", columns="station_id", values="wind_speed")
    pivot = pivot.dropna(how="all")

    # Also pivot temperature_2m if needed
    temp_pivot = None
    if "temperature_2m" in interp_cfg.get("rk_features", []):
        temp_pivot = combined.pivot(index="timestamp", columns="station_id", values="temperature_2m")
        temp_pivot = temp_pivot.reindex(pivot.index)

    logger.info(f"Loaded {len(pivot)} timestamps from {pivot.index.min()} to {pivot.index.max()}")

    # Load regression kriging features
    rk_feature_names = interp_cfg.get("rk_features", [])
    rk_static_features = {}
    rk_dynamic_features = {}

    if rk_feature_names:
        logger.info("=== Loading regression kriging features ===")

        # Static features
        if "altitude" in rk_feature_names:
            rk_static_features["altitude"] = alts
            logger.info("Loaded RK feature 'altitude' as (N,) array.")

        # Dynamic features from station data
        if "temperature_2m" in rk_feature_names and temp_pivot is not None:
            rk_dynamic_features["temperature_2m"] = temp_pivot.values
            logger.info("Loaded RK feature 'temperature_2m' from station data as (T, N) matrix.")

        # Dynamic features
        nwp_wind_requested = "nwp_wind_speed" in rk_feature_names
        ecmwf_wind_requested = "ecmwf_wind_speed" in rk_feature_names
        temp_requested = "temperature_2m" in rk_feature_names

        if nwp_wind_requested:
            if nwp_path:
                logger.info("=== Loading ICON-D2 NWP wind speed ===")
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

        if ecmwf_wind_requested:
            db_url = os.environ.get("ECMWF_WIND_SL_URL")
            if not db_url:
                logger.warning("ECMWF wind speed requested but ECMWF_WIND_SL_URL not set — skipping.")
            else:
                logger.info("=== Loading ECMWF wind speed ===")
                ecmwf_matrix = load_ecmwf_wind_speed(
                    station_ids=station_ids,
                    station_lats=lats,
                    station_lons=lons,
                    timestamps=pivot.index,
                    db_url=db_url,
                )
                rk_dynamic_features["ecmwf_wind_speed"] = ecmwf_matrix
                logger.info("Loaded RK feature 'ecmwf_wind_speed' as (T, N) matrix.")

        # Keep only loaded features
        rk_feature_names = [
            f for f in rk_feature_names
            if f in rk_static_features or f in rk_dynamic_features
        ]
        logger.info(f"Final RK feature list: {rk_feature_names}")

        # Drop timestamps with NaN in dynamic features
        if rk_dynamic_features:
            valid_mask = np.ones(len(pivot), dtype=bool)
            for fname, arr in rk_dynamic_features.items():
                valid_mask &= ~np.any(np.isnan(arr), axis=1)
            n_dropped = int((~valid_mask).sum())
            if n_dropped > 0:
                logger.info(
                    f"Dropping {n_dropped} timestamps with NaN in dynamic RK features "
                    f"({100.0 * n_dropped / len(pivot):.1f}%)."
                )
                pivot = pivot.iloc[valid_mask]
                rk_dynamic_features = {k: v[valid_mask] for k, v in rk_dynamic_features.items()}

    return pivot, lats, lons, alts, station_ids, rk_feature_names, rk_static_features, rk_dynamic_features


def regression_kriging_prediction(
    target_idx: int,
    ws_matrix: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    dist_matrix: np.ndarray,
    variogram_params: dict,
    k: int,
    rk_feature_names: list,
    rk_static_features: dict,
    rk_dynamic_features: dict,
):
    """Perform regression kriging for a single target station across all timestamps.

    Returns predicted wind speed array of shape (T,)
    """
    from sklearn.linear_model import LinearRegression

    N = ws_matrix.shape[1]
    T = ws_matrix.shape[0]

    # Use all OTHER stations for training (exclude target)
    train_mask = np.ones(N, dtype=bool)
    train_mask[target_idx] = False
    train_indices = np.where(train_mask)[0]

    # Build feature matrix for regression
    # X: (T, N_train, n_features)
    n_features = len(rk_feature_names)
    X_train = np.zeros((T, len(train_indices), n_features))
    X_target = np.zeros((T, n_features))

    for i, fname in enumerate(rk_feature_names):
        if fname in rk_static_features:
            # Static feature: replicate across time
            feat_vals = rk_static_features[fname]
            X_train[:, :, i] = feat_vals[train_indices][np.newaxis, :]
            X_target[:, i] = feat_vals[target_idx]
        elif fname in rk_dynamic_features:
            # Dynamic feature: (T, N)
            feat_vals = rk_dynamic_features[fname]
            X_train[:, :, i] = feat_vals[:, train_indices]
            X_target[:, i] = feat_vals[:, target_idx]

    # Fit OLS per timestamp to get residuals
    y_train = ws_matrix[:, train_indices]  # (T, N_train)
    residuals = np.zeros_like(y_train)
    trend_predictions = np.zeros(T)

    for t in range(T):
        # Skip if any NaN in this timestamp
        valid = ~np.isnan(y_train[t, :])
        if not valid.any():
            trend_predictions[t] = np.nan
            continue

        lr = LinearRegression()
        lr.fit(X_train[t, valid, :], y_train[t, valid])

        # Residuals for kriging
        residuals[t, valid] = y_train[t, valid] - lr.predict(X_train[t, valid, :])

        # Trend prediction for target
        trend_predictions[t] = lr.predict(X_target[t:t+1, :])[0]

    # Kriging on residuals
    # Find k nearest neighbors to target
    target_dists = dist_matrix[target_idx, :]
    sorted_indices = np.argsort(target_dists)
    # Exclude the target itself and take k nearest
    neighbor_indices = [idx for idx in sorted_indices if train_mask[idx]][:k]

    # Map to indices in train_indices array
    neighbor_positions = [np.where(train_indices == idx)[0][0] for idx in neighbor_indices]

    # Kriging weights
    nugget = variogram_params["nugget"]
    psill = variogram_params["psill"]
    range_val = variogram_params["range"]
    model_name = variogram_params.get("model", "gaussian")

    from utils.interpolation import _get_variogram_fn
    variogram_fn = _get_variogram_fn(model_name)

    # Build kriging system for these k neighbors
    k_actual = len(neighbor_indices)
    K = np.zeros((k_actual + 1, k_actual + 1))

    for i, idx_i in enumerate(neighbor_indices):
        for j, idx_j in enumerate(neighbor_indices):
            h = dist_matrix[idx_i, idx_j]
            K[i, j] = variogram_fn(h, nugget, psill, range_val)
        K[i, k_actual] = 1.0
        K[k_actual, i] = 1.0
    K[k_actual, k_actual] = 0.0

    # RHS: covariances from neighbors to target
    k_vec = np.zeros(k_actual + 1)
    for i, idx_i in enumerate(neighbor_indices):
        h = dist_matrix[idx_i, target_idx]
        k_vec[i] = variogram_fn(h, nugget, psill, range_val)
    k_vec[k_actual] = 1.0

    # Solve for weights
    try:
        weights = np.linalg.solve(K, k_vec)
        weights = weights[:k_actual]  # exclude Lagrange multiplier
    except np.linalg.LinAlgError:
        logger.warning(f"Singular kriging matrix for target {target_idx}")
        weights = np.ones(k_actual) / k_actual

    # Interpolate residuals
    residual_predictions = np.zeros(T)
    for t in range(T):
        neighbor_residuals = residuals[t, neighbor_positions]
        if np.isnan(neighbor_residuals).all():
            residual_predictions[t] = 0.0
        else:
            # Use only valid neighbors
            valid = ~np.isnan(neighbor_residuals)
            if valid.any():
                w_valid = weights[valid]
                w_valid = w_valid / w_valid.sum()
                residual_predictions[t] = np.dot(w_valid, neighbor_residuals[valid])

    # Final prediction = trend + kriged residual
    predictions = trend_predictions + residual_predictions

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate kriging timeseries for all stations")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_spatial_interpolation.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--log", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_config(args.config)

    # Load data
    pivot, lats, lons, alts, station_ids, rk_feature_names, rk_static_features, rk_dynamic_features = load_data(config)

    ws_matrix = pivot.values  # (T, N)
    timestamps = pivot.index

    # Compute distance matrix
    interp_cfg = config["interpolation"]

    if interp_cfg.get("anisotropy", {}).get("enabled", False):
        logger.info("Computing anisotropic distance matrix")
        aniso_cfg = interp_cfg["anisotropy"]
        dist_matrix = compute_anisotropic_distance_matrix(
            lats, lons,
            angle_deg=aniso_cfg.get("angle", 0),
            ratio=aniso_cfg.get("ratio", 1.0),
        )
    else:
        logger.info("Computing isotropic distance matrix")
        dist_matrix = compute_distance_matrix(lats, lons)

    # Compute global variogram
    logger.info("=== Computing global variogram ===")
    variogram_cfg = {
        "model": interp_cfg.get("variogram_model", "gaussian"),
        "n_lags": interp_cfg.get("n_variogram_lags", 30),
        "max_dist": interp_cfg.get("max_variogram_dist"),
        "detrend": interp_cfg.get("variogram_detrend", True),
    }

    lags_emp, sv_emp = compute_empirical_semivariance(
        ws_matrix,
        dist_matrix,
        n_lags=variogram_cfg["n_lags"],
        max_dist=variogram_cfg["max_dist"],
        detrend=variogram_cfg["detrend"],
    )

    variogram_params = fit_global_variogram(
        lags_emp,
        sv_emp,
        model=variogram_cfg["model"],
    )

    logger.info(f"Variogram: {variogram_params}")

    # Parameters
    k = interp_cfg.get("k_neighbors", 159)

    # Output directory
    output_dir = "data/kriging_stations"
    os.makedirs(output_dir, exist_ok=True)

    # Process each station
    logger.info(f"=== Generating timeseries for {len(station_ids)} stations ===")

    for i, sid in enumerate(tqdm(station_ids, desc="Interpolating stations")):
        predictions = regression_kriging_prediction(
            target_idx=i,
            ws_matrix=ws_matrix,
            lats=lats,
            lons=lons,
            dist_matrix=dist_matrix,
            variogram_params=variogram_params,
            k=k,
            rk_feature_names=rk_feature_names,
            rk_static_features=rk_static_features,
            rk_dynamic_features=rk_dynamic_features,
        )

        # Save to Parquet
        result_df = pd.DataFrame({
            "timestamp": timestamps,
            "wind_speed_kriging": predictions,
        })

        output_path = os.path.join(output_dir, f"kriging_{sid}.parquet")
        result_df.to_parquet(output_path, index=False)

    logger.info(f"=== Done! Saved {len(station_ids)} files to {output_dir} ===")


if __name__ == "__main__":
    main()
