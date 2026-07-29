#!/usr/bin/env python3
"""Regenerate the cached spatial-KNN imputation parquets for wind_speed/wind_direction.

Mirrors the KNN-imputation block of run_spatial_interpolation.py (same station
population, same hashing scheme, same knn_k=10), but skips the expensive
kriging LOO-CV so it only rebuilds the two cache files consumed by
utils/imputation.load_knn_imputation() at train time.

Resolution: the original cache was fit at native 10-min resolution — sklearn's
KNNImputer has no NaN-aware tree algorithm, so it falls back to brute-force
O(T^2) pairwise distances. At T ~ 156k (full 3-year range) that's multiple
hours per feature. Every consumer (load_knn_imputation) resamples the cache
to 1h via .resample(freq).mean() anyway, so fitting directly on the
hourly-resampled series is numerically equivalent at the point of use and
~36x cheaper (T ~ 26k). This script therefore resamples to 1h before the
KNNImputer fit.

Run this whenever the raw station data has grown past the cached coverage
(the cache never auto-refreshes — matches[-1] in load_knn_imputation just
picks the newest-looking filename, it does not check freshness).

Usage:
    python geostatistics/regen_knn_imputation.py
"""
import hashlib
import logging
import os

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "/mnt/lambda1/nvme1/synthetic/raw/wind"
CACHE_DIR = "/mnt/lambda1/nvme1/synthetic/knnimputer/wind"
EXCLUDE = {"14138"}
KNN_K = 10


def main() -> None:
    station_ids = sorted(
        f.replace("Station_", "").replace(".parquet", "")
        for f in os.listdir(DATA_PATH)
        if f.startswith("Station_") and f.endswith(".parquet")
    )
    station_ids = [s for s in station_ids if s not in EXCLUDE]
    logger.info("Station population: %d stations", len(station_ids))

    sid_hash = hashlib.md5(",".join(sorted(station_ids)).encode()).hexdigest()[:8]
    logger.info("Station-set hash: %s (expect 67558851 to match existing cache)", sid_hash)

    all_dfs = []
    for sid in station_ids:
        fpath = os.path.join(DATA_PATH, f"Station_{sid}.parquet")
        df = pd.read_parquet(fpath, columns=["wind_speed", "wind_direction"])
        df = df.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in df.columns:
            df = df.rename(columns={df.columns[0]: "timestamp"})
        df["station_id"] = sid
        all_dfs.append(df[["timestamp", "station_id", "wind_speed", "wind_direction"]])

    logger.info("Concatenating %d station frames ...", len(all_dfs))
    combined = pd.concat(all_dfs, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    logger.info("Combined range: %s -> %s", combined["timestamp"].min(), combined["timestamp"].max())

    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- wind_speed ---
    ws_cache = os.path.join(CACHE_DIR, f"wind_speed_knn{KNN_K}_start_end_{sid_hash}.parquet")
    ws_pivot = combined.pivot_table(index="timestamp", columns="station_id", values="wind_speed", aggfunc="mean")
    ws_pivot = ws_pivot.reindex(columns=station_ids).sort_index()
    ws_pivot = ws_pivot.resample("1h", closed="left", label="left").mean()
    logger.info("Resampled wind_speed to 1h: %d timestamps", len(ws_pivot))
    n_missing = int(ws_pivot.isna().sum().sum())
    logger.info("wind_speed: %d missing values (%.2f%%) — fitting KNNImputer(k=%d) ...",
                n_missing, 100.0 * n_missing / ws_pivot.size, KNN_K)
    imputer = KNNImputer(n_neighbors=KNN_K)
    ws_imputed = imputer.fit_transform(ws_pivot.values)
    ws_pivot = pd.DataFrame(ws_imputed, index=ws_pivot.index, columns=ws_pivot.columns)
    ws_pivot.to_parquet(ws_cache)
    logger.info("Saved -> %s", ws_cache)

    # --- wind_direction via sin/cos ---
    dir_cache = os.path.join(CACHE_DIR, f"wind_direction_knn{KNN_K}_start_end_{sid_hash}.parquet")
    dir_pivot = combined.pivot_table(index="timestamp", columns="station_id", values="wind_direction", aggfunc="mean")
    dir_pivot = dir_pivot.reindex(columns=station_ids).sort_index()
    # Circular resample to 1h: average sin/cos components, not raw degrees
    # (a plain degree mean is wrong across the 360deg/0deg wrap).
    rad_10min = np.deg2rad(dir_pivot.values)
    sin_10min = pd.DataFrame(np.sin(rad_10min), index=dir_pivot.index, columns=dir_pivot.columns)
    cos_10min = pd.DataFrame(np.cos(rad_10min), index=dir_pivot.index, columns=dir_pivot.columns)
    sin_pivot = sin_10min.resample("1h", closed="left", label="left").mean()
    cos_pivot = cos_10min.resample("1h", closed="left", label="left").mean()
    dir_pivot = pd.DataFrame(
        np.rad2deg(np.arctan2(sin_pivot.values, cos_pivot.values)) % 360,
        index=sin_pivot.index, columns=sin_pivot.columns,
    )
    dir_pivot[sin_pivot.isna() | cos_pivot.isna()] = np.nan
    logger.info("Resampled wind_direction to 1h: %d timestamps", len(dir_pivot))
    n_missing_dir = int(dir_pivot.isna().sum().sum())
    logger.info("wind_direction: %d missing values (%.2f%%) — fitting KNNImputer(k=%d) ...",
                n_missing_dir, 100.0 * n_missing_dir / dir_pivot.size, KNN_K)
    rad = np.deg2rad(dir_pivot.values)
    sin_vals, cos_vals = np.sin(rad), np.cos(rad)
    combined_sc = np.concatenate([sin_vals, cos_vals], axis=1)
    imputer_dir = KNNImputer(n_neighbors=KNN_K)
    imputed_sc = imputer_dir.fit_transform(combined_sc)
    sin_imp, cos_imp = imputed_sc[:, :len(station_ids)], imputed_sc[:, len(station_ids):]
    dir_imp = np.rad2deg(np.arctan2(sin_imp, cos_imp)) % 360
    dir_pivot = pd.DataFrame(dir_imp, index=dir_pivot.index, columns=dir_pivot.columns)
    dir_pivot.to_parquet(dir_cache)
    logger.info("Saved -> %s", dir_cache)

    logger.info("Done.")


if __name__ == "__main__":
    main()
