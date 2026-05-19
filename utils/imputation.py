"""
utils/imputation.py — Shared spatial imputation utilities.

Provides two layers:

Array-based (used by train_dcrnn.py / hpo_dcrnn.py via train_stgnn2.py):
  load_interpol_imputation   — load Kriging rk_pred → (T, N) float32
  load_knn_imputation        — load spatial-KNN parquet → (T, N) float32
  apply_interpol_imputation  — fill NaN in (T, N, M) meas_raw with rk_pred
  apply_knn_imputation       — fill NaN in (T, N, M) meas_raw with knn_arr

DataFrame-based (used by preprocessing.py / train_cl.py / hpo_cl.py):
  impute_dfs_with_kriging    — fill NaN in target_col of {key: DataFrame} dict
  impute_dfs_with_knn        — fill NaN in feature cols using spatial-KNN parquets
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _station_id_from_key(key: str) -> str:
    """Extract 5-digit station ID from dict keys used by preprocessing.get_data().

    Handles patterns like 'synth_01234.csv', 'Station_01234.parquet', '01234'.
    """
    key = os.path.basename(key)
    key = key.replace("synth_", "").replace("Station_", "")
    key = os.path.splitext(key)[0]
    return key


# ---------------------------------------------------------------------------
# Array-based functions  (T, N, M) — used by the DCRNN/STGNN pipeline
# ---------------------------------------------------------------------------

def load_interpol_imputation(
    interpol_path: str,
    station_ids: list[str],
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    """Load Regression-Kriging predictions (rk_pred) and align to timestamps.

    Returns
    -------
    rk_pred : (T, N) float32 — NaN where no interpol file exists for a station.
    """
    series = []
    for sid in station_ids:
        fpath = os.path.join(interpol_path, f"Station_{sid}.parquet")
        if not os.path.exists(fpath):
            series.append(pd.Series(np.nan, index=timestamps, name=sid, dtype="float32"))
            continue
        df = pd.read_parquet(fpath, columns=["timestamp", "rk_pred"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        s = df.set_index("timestamp")["rk_pred"].rename(sid)
        series.append(s)

    pivot = pd.concat(series, axis=1).reindex(timestamps)
    return pivot.values.astype(np.float32)


def load_knn_imputation(
    knnimputer_path: str,
    feature: str,
    station_ids: list[str],
    timestamps: pd.DatetimeIndex,
    freq: str = "1h",
) -> np.ndarray:
    """Load pre-computed spatial-KNN parquet and align to timestamps.

    The parquet is stored at 10-min resolution and resampled to *freq* by mean.

    Returns
    -------
    arr : (T, N) float32 — NaN where the file has no entry.
    """
    pattern = os.path.join(knnimputer_path, f"{feature}_knn*.parquet")
    matches = sorted(glob.glob(pattern))
    if not matches:
        logger.warning(
            "KNN imputation: no parquet found for '%s' in %s — skipping",
            feature, knnimputer_path,
        )
        return np.full((len(timestamps), len(station_ids)), np.nan, dtype=np.float32)

    fpath = matches[-1]
    logger.debug("KNN imputation: loading %s", os.path.basename(fpath))
    df = pd.read_parquet(fpath)

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.resample(freq, closed="left", label="left").mean()

    available = [s for s in station_ids if s in df.columns]
    missing   = [s for s in station_ids if s not in df.columns]
    if missing:
        logger.warning(
            "KNN imputation '%s': %d stations not in parquet: %s",
            feature, len(missing), missing[:10],
        )
    df = df.reindex(columns=station_ids).reindex(timestamps)
    logger.info(
        "KNN imputation '%s': loaded %d/%d stations, NaN remaining: %d",
        feature, len(available), len(station_ids),
        int(np.isnan(df.values).sum()),
    )
    return df.values.astype(np.float32)


def apply_interpol_imputation(
    meas_raw: np.ndarray,
    rk_pred: np.ndarray,
    measurement_cols: list[str],
    target_col: str = "wind_speed",
) -> np.ndarray:
    """Fill NaN in *target_col* channel of *meas_raw* with *rk_pred* values.

    Parameters
    ----------
    meas_raw         : (T, N, M) float32
    rk_pred          : (T, N) float32
    measurement_cols : ordered list of column names (last dim of meas_raw)
    target_col       : which column to fill

    Returns *meas_raw* modified in-place.
    """
    if target_col not in measurement_cols:
        return meas_raw
    idx  = measurement_cols.index(target_col)
    mask = np.isnan(meas_raw[:, :, idx])
    meas_raw[:, :, idx][mask] = rk_pred[mask]
    return meas_raw


def apply_knn_imputation(
    meas_raw: np.ndarray,
    knn_arr: np.ndarray,
    measurement_cols: list[str],
    feature: str,
) -> np.ndarray:
    """Fill NaN in *feature* channel of *meas_raw* with *knn_arr* values.

    Parameters
    ----------
    meas_raw         : (T, N, M) float32
    knn_arr          : (T, N) float32
    measurement_cols : ordered list of column names (last dim of meas_raw)
    feature          : which column to fill

    Returns *meas_raw* modified in-place.
    """
    if feature not in measurement_cols:
        return meas_raw
    idx  = measurement_cols.index(feature)
    mask = np.isnan(meas_raw[:, :, idx]) & ~np.isnan(knn_arr)
    meas_raw[:, :, idx][mask] = knn_arr[mask]
    return meas_raw


# ---------------------------------------------------------------------------
# DataFrame-based functions  {key: DataFrame} — used by the CL pipeline
# ---------------------------------------------------------------------------

def impute_dfs_with_kriging(
    dfs: Dict[str, pd.DataFrame],
    interpol_path: str,
    target_col: str,
) -> Dict[str, pd.DataFrame]:
    """Fill NaN in *target_col* of each per-station DataFrame using Kriging rk_pred.

    Reads per-station parquets from *interpol_path*/Station_{sid}.parquet.
    Alignment is done via the DataFrame's own DatetimeIndex — no resampling.

    Parameters
    ----------
    dfs          : {file_key: DataFrame} as returned by preprocessing.get_data()
    interpol_path: directory with Station_XXXXX.parquet files containing 'rk_pred'
    target_col   : column to fill (e.g. 'ghi', 'wind_speed')
    """
    filled_total = 0
    for key, df in dfs.items():
        if target_col not in df.columns:
            continue
        nan_mask = df[target_col].isna()
        if not nan_mask.any():
            continue
        sid = _station_id_from_key(key)
        fpath = os.path.join(interpol_path, f"Station_{sid}.parquet")
        if not os.path.exists(fpath):
            logger.debug("Kriging imputation: no file for station %s — skipping", sid)
            continue
        rk_df = pd.read_parquet(fpath, columns=["timestamp", "rk_pred"])
        rk_df["timestamp"] = pd.to_datetime(rk_df["timestamp"], utc=True)
        rk_series = rk_df.set_index("timestamp")["rk_pred"].reindex(df.index)
        fill_mask = nan_mask & rk_series.notna()
        n_filled = int(fill_mask.sum())
        if n_filled:
            df.loc[fill_mask, target_col] = rk_series[fill_mask].values
            filled_total += n_filled
        nan_after = int(df[target_col].isna().sum())
        logger.debug(
            "Kriging imputation station %s '%s': %d NaN → %d NaN (%d filled)",
            sid, target_col, int(nan_mask.sum()), nan_after, n_filled,
        )
    logger.info(
        "Kriging imputation: filled %d NaN values in '%s' across %d stations",
        filled_total, target_col, len(dfs),
    )
    return dfs


def impute_dfs_with_knn(
    dfs: Dict[str, pd.DataFrame],
    knnimputer_path: str,
    features: list[str],
    freq: str = "1h",
) -> Dict[str, pd.DataFrame]:
    """Fill NaN in *features* columns using pre-computed spatial-KNN parquets.

    For each feature a single wide parquet (columns = station IDs) is loaded,
    resampled to *freq*, then applied per station.

    Parameters
    ----------
    dfs             : {file_key: DataFrame}
    knnimputer_path : directory with {feature}_knn*.parquet files
    features        : list of column names to impute (e.g. ['dhi', 'wind_direction'])
    freq            : target frequency matching the DataFrames' index
    """
    for feature in features:
        pattern = os.path.join(knnimputer_path, f"{feature}_knn*.parquet")
        matches = sorted(glob.glob(pattern))
        if not matches:
            logger.warning(
                "KNN imputation: no parquet found for '%s' in %s — skipping",
                feature, knnimputer_path,
            )
            continue
        knn_wide = pd.read_parquet(matches[-1])
        knn_wide.index = pd.to_datetime(knn_wide.index, utc=True)
        knn_wide = knn_wide.resample(freq, closed="left", label="left").mean()

        filled_total = 0
        for key, df in dfs.items():
            if feature not in df.columns:
                continue
            nan_mask = df[feature].isna()
            if not nan_mask.any():
                continue
            sid = _station_id_from_key(key)
            if sid not in knn_wide.columns:
                continue
            knn_series = knn_wide[sid].reindex(df.index)
            fill_mask = nan_mask & knn_series.notna()
            n_filled = int(fill_mask.sum())
            if n_filled:
                df.loc[fill_mask, feature] = knn_series[fill_mask].values
                filled_total += n_filled
        logger.info(
            "KNN imputation: filled %d NaN values in '%s'", filled_total, feature,
        )
    return dfs
