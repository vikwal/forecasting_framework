"""
train_stgnn2.py — Training script for the heterogeneous STGNN.

Graph structure
---------------
  station nodes    : DWD weather stations (train + val)
  icond2 nodes     : ICON-D2 NWP grid points (k nearest to each station)
  ecmwf nodes      : ECMWF HRES grid points (from PostGIS DB)

Temporal structure (run-based)
------------------------------
Each training sample is one ICON-D2 run at time t_run.  The 96-step NWP
sequence is built from TWO consecutive runs:

    t_run - 48h              t_run            t_run + 48h
         |                     |                     |
         [== hist run lead 1..48 ==][== curr run lead 1..48 ==]
         [===== measurements (available) ======]    (unknown)
                                               [==== target ====]

Measurements are only available for [t_run-48h, t_run).
The forecast period uses NWP only (meas zeroed in the model input).

Usage
-----
    python geostatistics/train_stgnn2.py \
        --config configs/config_wind_stgcn.yaml \
        --suffix v1
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pyproj import Geod
from tqdm import tqdm

# Make the repo root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from geostatistics.stgnn import (
    HeterogeneousGraphBuilder,
    ModelConfig,
    STGNN,
)
from geostatistics.stgnn.training.sampler import TrainingSampler
from geostatistics.stgnn.training.trainer import InductiveTrainer
from geostatistics.stgnn.utils.normalization import StandardScaler
from geostatistics.evaluation import evaluate as run_evaluation, find_ws_feat_idx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_stgnn2")

_GEOD = Geod(ellps="WGS84")

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Station measurements
# ---------------------------------------------------------------------------

def load_station_measurements(
    data_path: str,
    station_ids: list[str],
    cols: list[str],
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load per-station measurement CSVs and return a (T, N, M) array.

    Returns
    -------
    meas :       (T, N, M) float32, NaN where data is missing
    timestamps : hourly DatetimeIndex (UTC)
    """
    pivots = []
    common_index = None
    for col in cols:
        dfs = []
        for sid in station_ids:
            fpath = os.path.join(data_path, f"Station_{sid}.csv")
            df = pd.read_csv(fpath, usecols=["timestamp", col],
                             parse_dates=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            dfs.append(df.set_index("timestamp")[col].rename(sid))
        # closed="left", label="left": hour 0 = [00:00, 00:50], hour 1 = [01:00, 01:50], …
        pivot = (
            pd.concat(dfs, axis=1)
            .sort_index()
            .resample("1h", closed="left", label="left")
            .mean()
        )
        if common_index is None:
            common_index = pivot.index
        pivots.append(pivot.values.astype(np.float32))

    return np.stack(pivots, axis=-1), common_index   # (T, N, M)


def load_interpol_imputation(
    interpol_path: str,
    station_ids: list[str],
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Load Regression-Kriging predictions (``rk_pred``) from pre-computed
    interpolation CSVs and align them to ``timestamps``.

    Used to impute missing ``wind_speed`` values in ``meas_raw``:
    wherever ``meas_raw[:, n, ws_idx]`` is NaN the corresponding
    ``rk_pred`` value fills the gap.

    Returns
    -------
    rk_pred : (T, N) float32  — NaN where the interpol file has no entry
              for that timestamp.
    """
    series = []
    for sid in station_ids:
        fpath = os.path.join(interpol_path, f"Station_{sid}.csv")
        if not os.path.exists(fpath):
            # Station has no interpol file → all-NaN column (no imputation)
            series.append(pd.Series(np.nan, index=timestamps, name=sid, dtype="float32"))
            continue
        df = pd.read_csv(fpath, usecols=["timestamp", "rk_pred"], parse_dates=["timestamp"])
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
) -> np.ndarray:
    """
    Load a pre-computed KNN-imputed parquet from ``knnimputer_path`` for the
    given ``feature`` (e.g. ``"wind_direction"`` or ``"wind_speed"``) and
    return a (T, N) float32 array aligned to ``timestamps``.

    The parquet contains 10-min resolution data; it is resampled to 1 h by
    mean before alignment.  Columns are station IDs (5-digit strings).
    Missing stations or timestamps are left as NaN.

    Parameters
    ----------
    knnimputer_path : directory that contains the parquet files
    feature         : e.g. ``"wind_direction"``
    station_ids     : ordered list of station IDs to extract
    timestamps      : hourly UTC DatetimeIndex to align to

    Returns
    -------
    arr : (T, N) float32 — NaN where the file has no entry
    """
    import glob

    pattern = os.path.join(knnimputer_path, f"{feature}_knn*.parquet")
    matches = sorted(glob.glob(pattern))
    if not matches:
        logger.warning(
            "KNN imputation: no parquet found for '%s' in %s — skipping",
            feature, knnimputer_path,
        )
        return np.full((len(timestamps), len(station_ids)), np.nan, dtype=np.float32)

    fpath = matches[-1]   # use the most recent file
    logger.info("KNN imputation: loading %s", os.path.basename(fpath))
    df = pd.read_parquet(fpath)

    # Resample 10-min → 1 h (closed="left", label="left" matches measurement loader)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.resample("1h", closed="left", label="left").mean()

    # Select requested stations (fill missing stations with NaN)
    available = [s for s in station_ids if s in df.columns]
    missing   = [s for s in station_ids if s not in df.columns]
    if missing:
        logger.warning(
            "KNN imputation '%s': %d stations not in parquet: %s",
            feature, len(missing), missing[:10],
        )
    df = df.reindex(columns=station_ids)   # missing → NaN columns

    # Align to timestamps
    df = df.reindex(timestamps)
    logger.info(
        "KNN imputation '%s': loaded %d/%d stations, NaN remaining: %d",
        feature, len(available), len(station_ids),
        int(np.isnan(df.values).sum()),
    )
    return df.values.astype(np.float32)   # (T, N)


def apply_knn_imputation(
    meas_raw: np.ndarray,
    knn_arr: np.ndarray,
    measurement_cols: list[str],
    feature: str,
) -> np.ndarray:
    """
    Fill NaN entries in the ``feature`` channel of ``meas_raw`` with the
    corresponding values from ``knn_arr``.

    Parameters
    ----------
    meas_raw         : (T, N, M) float32 — may contain NaN
    knn_arr          : (T, N)   float32 — pre-imputed values
    measurement_cols : list of column names (same order as last dim of meas_raw)
    feature          : column to impute (e.g. ``"wind_direction"``)

    Returns
    -------
    meas_raw with NaN in the feature channel replaced (modified in-place).
    """
    if feature not in measurement_cols:
        return meas_raw
    idx  = measurement_cols.index(feature)
    mask = np.isnan(meas_raw[:, :, idx]) & ~np.isnan(knn_arr)
    meas_raw[:, :, idx][mask] = knn_arr[mask]
    return meas_raw


def apply_interpol_imputation(
    meas_raw: np.ndarray,
    rk_pred: np.ndarray,
    measurement_cols: list[str],
    target_col: str = "wind_speed",
) -> np.ndarray:
    """
    Fill NaN entries in the ``target_col`` channel of ``meas_raw`` with
    the corresponding ``rk_pred`` values.

    Parameters
    ----------
    meas_raw         : (T, N, M) float32 — may contain NaN
    rk_pred          : (T, N)   float32 — interpolated values
    measurement_cols : list of column names (same order as last dim of meas_raw)
    target_col       : column to impute (default: "wind_speed")

    Returns
    -------
    meas_raw with NaN in the target channel replaced by rk_pred.
    The array is modified in-place and returned.
    """
    if target_col not in measurement_cols:
        return meas_raw
    idx = measurement_cols.index(target_col)
    mask = np.isnan(meas_raw[:, :, idx])
    meas_raw[:, :, idx][mask] = rk_pred[mask]
    return meas_raw


def load_station_metadata(
    data_path: str,
    station_ids: list[str],
    meta_path: str = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load lat, lon, altitude for the given station IDs.

    Supports two CSV formats (auto-detected by columns present):
      - stations_master.csv  : station_id, latitude, longitude, station_height  (comma-sep)
      - wind_parameter.csv   : park_id, latitude, longitude, altitude           (semicolon-sep)

    Parameters
    ----------
    data_path  : fallback directory (used only when meta_path is None)
    station_ids: station IDs to look up
    meta_path  : explicit path to the metadata CSV; overrides data_path if given
    """
    if meta_path is None:
        meta_path = os.path.join(data_path, "wind_parameter.csv")

    # Try comma separator first, then semicolon
    meta = pd.read_csv(meta_path, dtype=str, sep=None, engine="python")

    # Normalise column names to a common internal set
    col_map: dict[str, str] = {}
    for c in meta.columns:
        lc = c.strip().lower()
        if lc in ("station_id", "park_id"):
            col_map[c] = "id"
        elif lc == "latitude":
            col_map[c] = "latitude"
        elif lc == "longitude":
            col_map[c] = "longitude"
        elif lc in ("altitude", "station_height"):
            col_map[c] = "altitude"
    meta = meta.rename(columns=col_map)
    meta["id"] = meta["id"].str.strip().str.zfill(5)
    meta = meta.set_index("id")

    lats = meta.loc[station_ids, "latitude"].astype(float).values.astype(np.float32)
    lons = meta.loc[station_ids, "longitude"].astype(float).values.astype(np.float32)
    alts = meta.loc[station_ids, "altitude"].astype(float).values.astype(np.float32)
    return lats, lons, alts


# ---------------------------------------------------------------------------
# ICON-D2 ML loader — run-based
# ---------------------------------------------------------------------------

# Height levels: (bottomlevel_m, toplevel_m, midpoint_m)
_ML_LEVELS = [
    (0,       20.0,    10),
    (20.0,    55.212,  38),
    (55.212,  100.277, 78),
    (100.277, 153.438, 127),
    (153.438, 213.746, 184),
    (213.746, 280.598, 247),
]
# midpoint → toplevel (used to filter CSV rows)
_MIDPOINT_TO_TOPLEVEL: dict[int, float] = {mid: top for _, top, mid in _ML_LEVELS}
_ALL_MIDPOINTS = sorted(_MIDPOINT_TO_TOPLEVEL)


def _nearest_ml_midpoint(height_m: int) -> int:
    return min(_ALL_MIDPOINTS, key=lambda x: abs(x - height_m))


def _parse_ml_feature_spec(features: list[str]) -> list[tuple[str, float, str]]:
    """
    Parse YAML feature names into (feat_name, toplevel, kind) triples.

    Supported format: ``{quantity}_{height}m``
    Quantities: u, v, wind_speed, temperature, pressure, qs

    Examples
    --------
    'u_10m'          → ('u_10m',          20.0,    'u')
    'wind_speed_38m' → ('wind_speed_38m', 55.212,  'wind_speed')
    'temperature_78m'→ ('temperature_78m', 100.277, 'temperature')
    """
    import re
    pattern = re.compile(r'^(u|v|wind_speed|temperature|pressure|qs)_(\d+)m$')
    specs = []
    for feat in features:
        m = pattern.match(feat)
        if not m:
            raise ValueError(
                f"Cannot parse ICON-D2 ML feature: {feat!r}. "
                f"Expected format: {{quantity}}_{{height}}m  "
                f"(e.g. 'u_10m', 'wind_speed_38m'). "
                f"Available quantities: u, v, wind_speed, temperature, pressure, qs. "
                f"Available heights: {_ALL_MIDPOINTS}"
            )
        kind = m.group(1)
        height = int(m.group(2))
        mid = _nearest_ml_midpoint(height)
        if abs(mid - height) > 50:
            logger.warning(
                "Feature %s: requested height %dm, nearest level midpoint is %dm",
                feat, height, mid,
            )
        specs.append((feat, _MIDPOINT_TO_TOPLEVEL[mid], kind))
    return specs


def _load_icond2_ml_parquet(
    fpath: Path,
    feature_specs: list[tuple[str, float, str]],
) -> tuple[list[pd.Timestamp], np.ndarray]:
    """
    Load one ICON-D2 ML grid point Parquet file.

    Returns
    -------
    run_times : sorted list of run start timestamps (UTC)
    array     : (R, 48, F) float32 — leads 1..48 only (lead 0 excluded)
    """
    df = pd.read_parquet(fpath)
    df["starttime"] = pd.to_datetime(df["starttime"], utc=True)

    # Keep lead 1..48 only (exclude analysis time lead=0)
    df = df[(df["forecasttime"] >= 1) & (df["forecasttime"] <= 48)].copy()
    df["forecasttime"] = df["forecasttime"].astype(int)
    df["lead_idx"] = df["forecasttime"] - 1   # 0-indexed: 0..47

    run_times_raw = sorted(df["starttime"].unique())
    R = len(run_times_raw)
    run_idx_map = {t: i for i, t in enumerate(run_times_raw)}
    df["run_idx"] = df["starttime"].map(run_idx_map)

    F = len(feature_specs)
    result = np.full((R, 48, F), np.nan, dtype=np.float32)

    for fi, (_, toplevel, kind) in enumerate(feature_specs):
        sub = df[np.isclose(df["toplevel"], toplevel, atol=0.01)]
        r_idx = sub["run_idx"].values
        l_idx = sub["lead_idx"].values
        if kind == "u":
            vals = sub["u_wind"].values
        elif kind == "v":
            vals = sub["v_wind"].values
        elif kind == "wind_speed":
            vals = np.sqrt(sub["u_wind"].values ** 2 + sub["v_wind"].values ** 2)
        elif kind == "temperature":
            vals = sub["temperature"].values
        elif kind == "pressure":
            vals = sub["pressure"].values
        elif kind == "qs":
            vals = sub["qs"].values
        else:
            raise ValueError(f"Unknown kind: {kind!r}")
        result[r_idx, l_idx, fi] = vals.astype(np.float32)

    return [pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC") for t in run_times_raw], result


def _parse_latlon(stem: str) -> tuple[float, float]:
    """Parse lat/lon from ICON-D2 filename stem, e.g. '52_9065_12_8820'."""
    parts = stem.split("_")
    lat = float(f"{parts[0]}.{parts[1]}")
    lon = float(f"{parts[2]}.{parts[3]}")
    return lat, lon


def _select_nearest_grid_files(
    sid_dir: Path,
    station_lat: float,
    station_lon: float,
    k: int,
) -> list[tuple[Path, str, float, float]]:
    """
    List all *_ML.csv in sid_dir, rank by geodesic distance, return k nearest.
    Returns list of (fpath, stem, grid_lat, grid_lon).
    """
    candidates = []
    for fpath in sid_dir.glob("*_ML.parquet"):
        stem = fpath.stem.replace("_ML", "")
        try:
            glat, glon = _parse_latlon(stem)
        except Exception:
            continue
        candidates.append((fpath, stem, glat, glon))

    if not candidates:
        return []

    # Geodesic distances
    lons1 = np.full(len(candidates), station_lon)
    lats1 = np.full(len(candidates), station_lat)
    glats = np.array([c[2] for c in candidates])
    glons = np.array([c[3] for c in candidates])
    _, _, dists = _GEOD.inv(lons1, lats1, glons, glats)
    order = np.argsort(dists)
    return [candidates[i] for i in order[:k]]


def load_icond2_ml_runs(
    nwp_path: str,
    station_ids: list[str],
    station_coords: np.ndarray,          # (N, 2) [lat, lon]
    features: list[str],
    run_hours: tuple[int, ...] = (6, 9, 12, 15),
    next_n_grid: int = 4,
    n_workers: int = 8,
    cutoff: pd.Timestamp | None = None,  # exclude runs after this time
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all ICON-D2 ML runs across all run-hours for all stations.

    Strategy (mirrors preprocessing.py _process_forecast_hour):
    - Iterate over stations (outer loop, tqdm)
    - Per station: find k nearest grid files per run-hour via geodesic distance
    - Load those CSVs in a thread pool (inner, per station)
    - Deduplicate grid nodes by (stem, rh) — same CSV loaded only once

    Returns
    -------
    run_times            : DatetimeIndex (R,) sorted
    grid_coords          : (N_grid, 2) float32 [lat, lon]
    grid_runs            : (R, 48, N_grid, F) float32
    station_nearest_grid : (N_stations,) int64 — index into grid_coords
    """
    feature_specs = _parse_ml_feature_spec(features)
    F = len(features)
    N = len(station_ids)
    ml_base = Path(nwp_path) / "ML"

    # ------------------------------------------------------------------
    # Phase 1: per station, find k nearest grid files per run-hour
    # Collect unique (stem, rh) → fpath (no duplicates)
    # ------------------------------------------------------------------
    # station_grid_keys[si] = list of (stem, rh) for that station's k nearest per rh
    station_grid_keys: list[list[tuple[str, int]]] = [[] for _ in range(N)]
    unique_grid_paths: dict[tuple[str, int], Path] = {}

    for si, sid in enumerate(tqdm(station_ids, desc="Scanning ICON-D2 dirs")):
        s_lat = float(station_coords[si, 0])
        s_lon = float(station_coords[si, 1])
        for rh in run_hours:
            sid_dir = ml_base / f"{rh:02d}" / sid
            if not sid_dir.exists():
                continue
            nearest = _select_nearest_grid_files(sid_dir, s_lat, s_lon, next_n_grid)
            for fpath, stem, _, _ in nearest:
                key = (stem, rh)
                station_grid_keys[si].append(key)
                if key not in unique_grid_paths:
                    unique_grid_paths[key] = fpath

    if not unique_grid_paths:
        raise FileNotFoundError(
            f"No ICON-D2 ML Parquet files found under {ml_base}. "
            f"Check run_hours={list(run_hours)} and station_ids."
        )
    logger.info(
        "ICON-D2 ML: %d unique (grid, run-hour) Parquet files to load (%d stations × %d run-hours × %d grid pts)",
        len(unique_grid_paths), N, len(run_hours), next_n_grid,
    )

    # ------------------------------------------------------------------
    # Phase 2: parallel Parquet loading
    # ------------------------------------------------------------------
    csv_results: dict[tuple[str, int], tuple[list, np.ndarray]] = {}

    def _load(key_fpath):
        key, fpath = key_fpath
        return key, _load_icond2_ml_parquet(fpath, feature_specs)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_load, kp): kp[0] for kp in unique_grid_paths.items()}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Loading ICON-D2 ML Parquet files", unit="files"):
            key = futures[fut]
            try:
                _, (run_times_h, arr_h) = fut.result()
                csv_results[key] = (run_times_h, arr_h)
            except Exception as e:
                logger.warning("Failed loading %s: %s", key, e)

    # ------------------------------------------------------------------
    # Phase 3: global run time index
    # ------------------------------------------------------------------
    all_run_times: set[pd.Timestamp] = set()
    for run_times_h, _ in csv_results.values():
        for t in run_times_h:
            if t.hour not in run_hours:
                continue
            if cutoff is not None and t > cutoff:
                continue
            all_run_times.add(t)

    run_times_global = pd.DatetimeIndex(sorted(all_run_times))
    R = len(run_times_global)
    run_idx_map = {t: i for i, t in enumerate(run_times_global)}
    logger.info("ICON-D2 ML: %d total runs (%s … %s)",
                R, run_times_global[0].date(), run_times_global[-1].date())

    # ------------------------------------------------------------------
    # Phase 4: unique grid nodes and dense array
    # ------------------------------------------------------------------
    unique_stems_rh = sorted(unique_grid_paths.keys())           # (stem, rh) sorted
    unique_stems    = sorted({stem for stem, _ in unique_stems_rh})
    grid_coords     = np.array([_parse_latlon(s) for s in unique_stems], dtype=np.float32)
    N_grid          = len(unique_stems)
    stem_to_gi      = {s: i for i, s in enumerate(unique_stems)}

    logger.info("ICON-D2 ML: %d unique grid nodes", N_grid)
    grid_runs = np.full((R, 48, N_grid, F), np.nan, dtype=np.float32)

    for (stem, rh), (run_times_h, arr_h) in csv_results.items():
        gi = stem_to_gi[stem]
        for local_ri, t in enumerate(run_times_h):
            global_ri = run_idx_map.get(t)
            if global_ri is not None:
                # Fill NaN slots only (multiple rh can share the same grid stem;
                # last-write wins, which is fine since they're different runs)
                grid_runs[global_ri, :, gi, :] = arr_h[local_ri, :, :]

    # ------------------------------------------------------------------
    # Phase 5: nearest grid node index per station
    # (use station_grid_keys[si][0] = nearest from rh[0] as the canonical nearest)
    # ------------------------------------------------------------------
    station_nearest_grid = np.zeros(N, dtype=np.int64)
    for si in range(N):
        keys = station_grid_keys[si]
        if keys:
            # First key is the nearest grid point for the first available run-hour
            stem = keys[0][0]
            station_nearest_grid[si] = stem_to_gi.get(stem, 0)
        else:
            logger.warning("Station %s: no grid keys found", station_ids[si])

    # Drop runs that have any NaN across grid nodes or leads
    complete_mask = ~np.isnan(grid_runs).any(axis=(1, 2, 3))  # (R,)
    n_dropped = int((~complete_mask).sum())
    if n_dropped:
        logger.info("ICON-D2 ML: dropping %d incomplete run(s) with missing grid data", n_dropped)
        grid_runs        = grid_runs[complete_mask]
        run_times_global = run_times_global[complete_mask]

    return run_times_global, grid_coords, grid_runs, station_nearest_grid


# ---------------------------------------------------------------------------
# ECMWF loader (absolute time — latest-run merged per valid_time)
# ---------------------------------------------------------------------------

def _compute_derived_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Compute wind_speed_Xm = sqrt(u² + v²) on-the-fly if requested but not present.
    Works for both ICON-D2 (u_Xm) and ECMWF (u_windXm / u_wind_Xm) naming.
    """
    import re
    for feat in features:
        if feat in df.columns:
            continue
        m = re.match(r"wind_speed_(\d+m)$", feat)
        if not m:
            continue
        level = m.group(1)
        candidates_u = [f"u_{level}", f"u_wind{level}", f"u_wind_{level}"]
        candidates_v = [f"v_{level}", f"v_wind{level}", f"v_wind_{level}"]
        u_col = next((c for c in candidates_u if c in df.columns), None)
        v_col = next((c for c in candidates_v if c in df.columns), None)
        if u_col and v_col:
            df[feat] = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2).astype(np.float32)
    return df


# ECMWF DB raw columns (as stored in ecmwf_wind_sl)
_ECMWF_DB_COLS = {
    "u_wind_10m", "v_wind_10m",
    "u_wind_100m", "v_wind_100m",
    "u_wind_200m", "v_wind_200m",
    "temp_2m", "dew_point_2m", "specific_rho", "friction_velocity",
}


def _ecmwf_raw_db_cols(features: list[str]) -> list[str]:
    """
    Determine which raw DB columns are needed for the requested features.

    For direct features (e.g. 'u_wind10m' → DB col 'u_wind_10m'), include directly.
    For derived features (e.g. 'wind_speed_10m'), include the u+v components needed.
    Returns deduplicated list of DB column names.
    """
    import re
    needed: set[str] = set()
    for feat in features:
        # Normalise Python name → DB name: insert _ before digit group
        # e.g. u_wind10m → u_wind_10m, v_wind100m → v_wind_100m
        db_name = re.sub(r"([a-z])(\d)", r"\1_\2", feat)
        if db_name in _ECMWF_DB_COLS:
            needed.add(db_name)
            continue
        # Derived: wind_speed_Xm → need u_wind_Xm + v_wind_Xm
        m = re.match(r"wind_speed_(\d+m)$", feat)
        if m:
            level = m.group(1)
            u_col = f"u_wind_{level}"
            v_col = f"v_wind_{level}"
            if u_col in _ECMWF_DB_COLS:
                needed.add(u_col)
            if v_col in _ECMWF_DB_COLS:
                needed.add(v_col)
    return sorted(needed)


def _fetch_ecmwf_with_coords(
    station_lat: float,
    station_lon: float,
    next_n_grid_points: int,
    db_url: str,
    features: list[str],
    ts_min: pd.Timestamp,
    ts_max: pd.Timestamp,
) -> pd.DataFrame:
    """
    Query ECMWF DB for the k nearest grid points to a station.

    Only fetches raw DB columns needed for ``features`` (including u/v for
    derived wind_speed_Xm). Derived features are computed by the caller via
    ``_compute_derived_features``.

    Returns DataFrame with columns:
        starttime, forecasttime, valid_time, grid_lat, grid_lon, rank,
        <raw db cols with normalised names, e.g. u_wind10m>
    """
    from sqlalchemy import create_engine

    raw_db_cols = _ecmwf_raw_db_cols(features)
    if not raw_db_cols:
        raise ValueError(
            f"None of the requested ECMWF features {features} map to known DB columns. "
            f"Available: {sorted(_ECMWF_DB_COLS)}"
        )
    select_cols = ", ".join(f"e.{c}" for c in raw_db_cols)
    ts_min_s = ts_min.strftime("%Y-%m-%d %H:%M:%S%z")
    ts_max_s = ts_max.strftime("%Y-%m-%d %H:%M:%S%z")

    query = f"""
    WITH nearest_points AS (
        SELECT geom,
               ST_Y(geom) AS grid_lat,
               ST_X(geom) AS grid_lon,
               ROW_NUMBER() OVER (
                   ORDER BY geom <-> ST_SetSRID(ST_MakePoint({station_lon}, {station_lat}), 4326)
               ) AS rank
        FROM ecmwf_grid_points
        ORDER BY geom <-> ST_SetSRID(ST_MakePoint({station_lon}, {station_lat}), 4326)
        LIMIT {next_n_grid_points}
    )
    SELECT e.starttime, e.forecasttime, n.grid_lat, n.grid_lon, n.rank, {select_cols}
    FROM ecmwf_wind_sl e
    JOIN nearest_points n ON e.geom = n.geom
    WHERE e.starttime BETWEEN '{ts_min_s}' AND '{ts_max_s}'
    ORDER BY n.rank, e.starttime, e.forecasttime
    """
    engine = create_engine(db_url)
    try:
        df = pd.read_sql(query, engine)
    finally:
        engine.dispose()

    if df.empty:
        return df

    df["starttime"]    = pd.to_datetime(df["starttime"], utc=True)
    df["forecasttime"] = df["forecasttime"].astype(int)
    df["valid_time"]   = df["starttime"] + pd.to_timedelta(df["forecasttime"], unit="h")

    # Normalise DB column names: u_wind_10m → u_wind10m
    import re as _re
    df.columns = [_re.sub(r"_(\d+m)", r"\1", c) for c in df.columns]
    return df


def load_ecmwf_at_stations_and_grid(
    db_url: str,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    features: list[str],
    timestamps: pd.DatetimeIndex,
    next_n_grid_per_station: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ECMWF NWP (absolute hourly, latest-run merged) at all stations
    and at all k-nearest grid point nodes.

    Returns
    -------
    station_nwp : (T, N_stations, F) float32
    grid_coords : (N_unique, 2) [lat, lon]
    grid_nwp    : (T, N_unique, F) float32
    grid_alts   : (N_unique,) float32 zeros
    """
    T  = len(timestamps)
    F  = len(features)
    Ns = len(station_lats)

    ts_min = timestamps[0].normalize()
    ts_max = timestamps[-1].normalize() + pd.Timedelta(hours=13)

    def _to_hourly(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates("valid_time", keep="last")
        return df.set_index("valid_time").sort_index()

    station_nwp = np.full((T, Ns, F), np.nan, dtype=np.float32)
    grid_cache: dict[tuple[float, float], pd.DataFrame] = {}

    for si in tqdm(range(Ns), desc="Loading ECMWF (stations + grid nodes)"):
        lat, lon = float(station_lats[si]), float(station_lons[si])
        try:
            df = _fetch_ecmwf_with_coords(
                station_lat=lat, station_lon=lon,
                next_n_grid_points=next_n_grid_per_station,
                db_url=db_url, features=features,
                ts_min=ts_min, ts_max=ts_max,
            )
        except Exception as e:
            logger.warning("ECMWF fetch failed station %d: %s", si, e)
            continue

        if df.empty:
            continue

        # Nearest grid point → station NWP
        df1 = _to_hourly(df[df["rank"] == 1].copy())
        df1 = _compute_derived_features(df1, features)
        for fi, feat in enumerate(features):
            if feat in df1.columns:
                station_nwp[:, si, fi] = df1[feat].reindex(timestamps).values

        # All k ranks → grid node cache
        for (glat, glon), grp in df.groupby(["grid_lat", "grid_lon"]):
            key = (round(float(glat), 5), round(float(glon), 5))
            if key not in grid_cache:
                grid_cache[key] = _compute_derived_features(_to_hourly(grp.copy()), features)

    # Build grid node arrays
    grid_keys   = sorted(grid_cache.keys())
    N_grid      = len(grid_keys)
    grid_coords = np.array(grid_keys, dtype=np.float32)
    grid_alts   = np.zeros(N_grid, dtype=np.float32)
    grid_nwp    = np.full((T, N_grid, F), np.nan, dtype=np.float32)

    for gi, key in enumerate(grid_keys):
        gdf = grid_cache[key]
        for fi, feat in enumerate(features):
            if feat in gdf.columns:
                grid_nwp[:, gi, fi] = gdf[feat].reindex(timestamps).values

    logger.info("ECMWF: %d unique grid nodes from %d station queries", N_grid, Ns)
    return station_nwp, grid_coords, grid_nwp, grid_alts


def load_ecmwf_parquet_at_stations_and_grid(
    parquet_path: str,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    features: list[str],
    timestamps: pd.DatetimeIndex,
    next_n_grid_per_station: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ECMWF NWP natively from a Parquet file.
    Does not require PostgreSQL. Extremely fast memory-mapped loading.
    """
    logger.info("Loading ECMWF from parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    ts_min = timestamps[0].normalize()
    ts_max = timestamps[-1].normalize() + pd.Timedelta(hours=13)
    df = df[(df["valid_time"] >= ts_min) & (df["valid_time"] <= ts_max)].copy()

    df = _compute_derived_features(df, features)

    unique_grids = df[["grid_lat", "grid_lon"]].drop_duplicates().values

    Ns = len(station_lats)
    Ng = len(unique_grids)

    station_nearest = []
    global_needed_grids = set()

    # Calculate nearest nodes using pyproj Geod
    for si in range(Ns):
        s_lat = np.full(Ng, station_lats[si])
        s_lon = np.full(Ng, station_lons[si])
        g_lats = unique_grids[:, 0]
        g_lons = unique_grids[:, 1]
        _, _, dists = _GEOD.inv(s_lon, s_lat, g_lons, g_lats)

        order = np.argsort(dists)
        top_k_idx = order[:next_n_grid_per_station]

        station_nearest.append([tuple(unique_grids[i]) for i in top_k_idx])
        for i in top_k_idx:
            global_needed_grids.add(tuple(unique_grids[i]))

    grid_keys = sorted(list(global_needed_grids))
    N_grid = len(grid_keys)
    grid_coords = np.array(grid_keys, dtype=np.float32)
    grid_alts = np.zeros(N_grid, dtype=np.float32)

    T = len(timestamps)
    F = len(features)

    station_nwp = np.full((T, Ns, F), np.nan, dtype=np.float32)
    grid_nwp = np.full((T, N_grid, F), np.nan, dtype=np.float32)

    df["grid_key"] = list(zip(df["grid_lat"].round(5), df["grid_lon"].round(5)))
    needed_keys_round = { (round(float(k[0]), 5), round(float(k[1]), 5)) for k in grid_keys }

    df = df[df["grid_key"].isin(needed_keys_round)]

    logger.info("Executing Pandas Hash Grouping to avoid expensive DataFrame iteration...")
    grouped = df.groupby("grid_key")

    logger.info("Filling ECMWF grid arrays ...")
    from tqdm import tqdm
    for gi, key in enumerate(tqdm(grid_keys, desc="Filling ECMWF Grid Arrays")):
        k_round = (round(float(key[0]), 5), round(float(key[1]), 5))
        try:
            gdf = grouped.get_group(k_round)
        except KeyError:
            continue
        gdf = gdf.drop_duplicates("valid_time", keep="last")
        gdf = gdf.set_index("valid_time").sort_index()
        for fi, feat in enumerate(features):
            if feat in gdf.columns:
                grid_nwp[:, gi, fi] = gdf[feat].reindex(timestamps).values

    for si in tqdm(range(Ns), desc="Filling ECMWF Station Tensors"):
        nearest_key = station_nearest[si][0]
        k_round = (round(float(nearest_key[0]), 5), round(float(nearest_key[1]), 5))
        try:
            gdf = grouped.get_group(k_round)
        except KeyError:
            continue
        gdf = gdf.drop_duplicates("valid_time", keep="last")
        gdf = gdf.set_index("valid_time").sort_index()
        for fi, feat in enumerate(features):
            if feat in gdf.columns:
                station_nwp[:, si, fi] = gdf[feat].reindex(timestamps).values

    logger.info("ECMWF Parquet: %d unique grid nodes allocated for %d stations", N_grid, Ns)
    return station_nwp, grid_coords, grid_nwp, grid_alts


# ---------------------------------------------------------------------------
# NWP elevation loader
# ---------------------------------------------------------------------------

def _load_elevations_from_table(
    db_url: str,
    table: str,
    coords: np.ndarray,          # (N, 2) [lat, lon]
    tol: float = 0.02,
) -> np.ndarray:
    """
    Read the ``elevation`` column from a grid-point table and match to *coords*.

    Coordinates are extracted via PostGIS ST_Y / ST_X.  Each entry in *coords*
    is matched to the nearest table row within *tol* degrees; unmatched nodes
    get elevation 0 with a warning.

    Returns (N,) float32 array.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(f"SELECT ST_Y(geom) AS lat, ST_X(geom) AS lon, elevation FROM {table}"),
                conn,
            )
    except Exception as exc:
        logger.warning(
            "Could not read elevation from %s (%s) — using 0 m for all nodes", table, exc
        )
        return np.zeros(len(coords), dtype=np.float32)
    finally:
        engine.dispose()

    if df.empty or df["elevation"].isna().all():
        logger.warning(
            "%s has no elevation data — run populate_nwp_elevations.py first; using 0 m", table
        )
        return np.zeros(len(coords), dtype=np.float32)

    alts = np.zeros(len(coords), dtype=np.float32)
    n_missing = 0
    db_lats = df["lat"].values
    db_lons = df["lon"].values
    db_elev = df["elevation"].fillna(0).values.astype(np.float32)

    for i, (lat, lon) in enumerate(coords):
        dists = np.sqrt((db_lats - lat) ** 2 + (db_lons - lon) ** 2)
        j = int(np.argmin(dists))
        if dists[j] <= tol:
            alts[i] = db_elev[j]
        else:
            n_missing += 1

    if n_missing:
        logger.warning(
            "%s: %d / %d nodes had no match within %.3f° — using 0 m",
            table, n_missing, len(coords), tol,
        )
    return alts


def load_nwp_elevations(
    weather_db_url: str,
    ecmwf_db_url: str,
    icond2_coords: np.ndarray,   # (N_i, 2) [lat, lon]
    ecmwf_coords: np.ndarray,    # (N_e, 2) [lat, lon]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load NWP node elevations from the ``elevation`` column of the grid-point tables.

    icon_d2_grid_points lives in WeatherDB (WEATHER_DB_URL).
    ecmwf_grid_points   lives in the ECMWF DB (ECMWF_WIND_SL_URL).

    Returns
    -------
    icond2_alts : (N_i,) float32
    ecmwf_alts  : (N_e,) float32
    """
    icond2_alts = _load_elevations_from_table(
        weather_db_url, "icon_d2_grid_points", icond2_coords,
    )
    ecmwf_alts = _load_elevations_from_table(
        ecmwf_db_url, "ecmwf_grid_points", ecmwf_coords,
    )
    logger.info(
        "NWP elevations — ICON-D2: %.0f–%.0f m  ECMWF: %.0f–%.0f m",
        icond2_alts.min(), icond2_alts.max(),
        ecmwf_alts.min(),  ecmwf_alts.max(),
    )
    return icond2_alts, ecmwf_alts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train heterogeneous STGNN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument(
        "--eval", action="store_true",
        help="Run 4-pass LOO evaluation on val run pairs after training and save results in the pkl.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg  = cfg["data"]
    stgnn_cfg = cfg.get("stgnn2", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config_stem = Path(args.config).stem.replace("config_", "")
    model_name  = f"{config_stem}_stgnn2_{args.suffix}.pt" if args.suffix else f"{config_stem}_stgnn2.pt"
    model_path  = Path("models") / model_name

    # Set up dynamic log file
    log_stem = Path(args.config).stem
    log_name = f"train_stgnn2_{log_stem}_{args.suffix}.log" if args.suffix else f"train_stgnn2_{log_stem}.log"
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
    icond2_features   = stgnn_cfg.get("icond2_features")
    ecmwf_features    = stgnn_cfg.get("ecmwf_features")
    measurement_cols  = stgnn_cfg.get("measurement_features")
    target_col        = stgnn_cfg.get("target_col")
    if target_col not in measurement_cols:
        raise ValueError(f"target_col '{target_col}' must be in measurement_features")
    target_feat_idx = measurement_cols.index(target_col)

    run_hours = tuple(stgnn_cfg.get("icond2_run_hours", [6, 9, 12, 15]))
    next_n_icond2 = stgnn_cfg.get("next_n_icond2", 4)
    n_workers = stgnn_cfg.get("n_workers", 8)
    nwp_path  = data_cfg.get("nwp_path")
    data_path = data_cfg["path"]

    H   = stgnn_cfg.get("history_length", 48)
    F_h = stgnn_cfg.get("forecast_horizon", 48)

    # ------------------------------------------------------------------
    # Station measurements  (T, N_all, M)
    # ------------------------------------------------------------------
    logger.info("Loading station measurements …")
    meas_raw, timestamps = load_station_measurements(data_path, all_ids, cols=measurement_cols)
    T = len(timestamps)
    logger.info("Timestamps: %d  (%s … %s)", T, timestamps[0], timestamps[-1])

    # NaN audit — report which stations have gaps so we can catch data issues early
    nan_counts = np.isnan(meas_raw[:, :, 0]).sum(axis=0)   # (N,) per station
    bad = [(all_ids[i], int(nan_counts[i])) for i in np.where(nan_counts > 0)[0]]
    if bad:
        logger.warning(
            "Measurement NaN audit: %d stations have gaps (will skip affected windows): %s",
            len(bad), bad,
        )
    else:
        logger.info("Measurement NaN audit: no missing values ✓")
    # Boolean mask: True where any station has NaN at that timestep
    _meas_nan_any = np.isnan(meas_raw[:, :, 0]).any(axis=1)  # (T,)

    # Temporal split
    test_start = data_cfg.get("test_start")
    test_end   = data_cfg.get("test_end")
    if test_start:
        ts_cutoff = pd.Timestamp(test_start, tz="UTC")
        split_t = int(np.searchsorted(timestamps, ts_cutoff, side="left"))
    else:
        val_frac = data_cfg.get("val_frac", 0.2)
        split_t = int(T * (1 - val_frac))
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
    R        = len(run_times)
    I2       = len(icond2_features)
    N_igrid  = len(icond2_coords)
    logger.info("ICON-D2 grid nodes: %d  runs: %d  features: %s",
                N_igrid, R, icond2_features)

    # ------------------------------------------------------------------
    # ECMWF NWP  →  (T, N_all, E2) + grid nodes (T, N_egrid, E2)
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
                next_n_grid_per_station=stgnn_cfg.get("next_n_ecmwf", 4),
            )
        logger.info("ECMWF grid nodes: %d  features: %s", len(ecmwf_coords), ecmwf_features)
    else:
        logger.warning(f"Parquet file {ecmwf_parquet_file} not found. Using fallbacks...")
        # if db_url:
        #     logger.info("Loading ECMWF NWP data from database …")
        #     station_ecmwf_nwp, ecmwf_coords, ecmwf_nwp, ecmwf_alts = \
        #         load_ecmwf_at_stations_and_grid(
        #             db_url=db_url,
        #             station_lats=lats, station_lons=lons,
        #             features=ecmwf_features, timestamps=timestamps,
        #             next_n_grid_per_station=stgnn_cfg.get("next_n_ecmwf", 4),
        #         )
        #     logger.info("ECMWF grid nodes: %d  features: %s", len(ecmwf_coords), ecmwf_features)
        # else:
        logger.warning("ECMWF_WIND_SL_URL not set — using zero ECMWF features")
        station_ecmwf_nwp = np.zeros((T, len(all_ids), len(ecmwf_features)), dtype=np.float32)
        ec_lats = np.arange(47.5, 55.0, 0.5)
        ec_lons = np.arange(6.0, 15.5, 0.5)
        eg, lg  = np.meshgrid(ec_lats, ec_lons)
        ecmwf_coords = np.stack([eg.ravel(), lg.ravel()], axis=1).astype(np.float32)
        ecmwf_nwp    = np.zeros((T, len(ecmwf_coords), len(ecmwf_features)), dtype=np.float32)
        ecmwf_alts   = np.zeros(len(ecmwf_coords), dtype=np.float32)

    # ------------------------------------------------------------------
    # NWP elevations from DB
    # ------------------------------------------------------------------
    weather_db_url = os.environ.get("WEATHER_DB_URL")
    if stgnn_cfg.get("use_altitude_diff", False):
        if weather_db_url and db_url:
            logger.info("Loading NWP elevations from grid-point tables …")
            icond2_alts, ecmwf_alts = load_nwp_elevations(
                weather_db_url=weather_db_url,
                ecmwf_db_url=db_url,        # ECMWF_WIND_SL_URL, already read above
                icond2_coords=icond2_coords,
                ecmwf_coords=ecmwf_coords,
            )
            # Offset grid elevations by +10.0m because grid coords are ground-level and NWP features are 10m wind
            icond2_alts += 10.0
            ecmwf_alts += 10.0
        else:
            missing = []
            if not weather_db_url: missing.append("WEATHER_DB_URL")
            if not db_url:         missing.append("ECMWF_WIND_SL_URL")
            logger.warning(
                "%s not set — altitude for NWP edges not available, using 0 m",
                " and ".join(missing),
            )
            icond2_alts = np.zeros(N_igrid, dtype=np.float32)
            # ecmwf_alts already set above
    else:
        icond2_alts = np.zeros(N_igrid, dtype=np.float32)
        # ecmwf_alts already set above

    # ------------------------------------------------------------------
    # Scalers — fit on training data only
    # ------------------------------------------------------------------
    logger.info("Fitting scalers on training data …")

    M_meas = len(measurement_cols)
    meas_scaler = StandardScaler()
    meas_scaler.fit(meas_raw[:split_t, :N_train].reshape(-1, M_meas))
    meas_scaled = meas_scaler.transform(
        meas_raw.reshape(-1, M_meas)
    ).reshape(T, len(all_ids), M_meas)

    # ICON-D2 scaler: fit on training runs (t_run < split_time) over all grid nodes
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
    # Build run pairs  (r_curr, r_hist, t_run_abs)
    # ------------------------------------------------------------------
    # t_run_abs: index of the run start time in the measurement timestamps array
    ts_lookup = pd.Series(np.arange(T), index=timestamps)
    run_times_np = run_times  # pd.DatetimeIndex

    train_run_pairs: list[tuple[int, int, int]] = []
    val_run_pairs:   list[tuple[int, int, int]] = []
    skipped = 0

    for r_curr in range(R):
        t_run = run_times_np[r_curr]

        # Map run start time to absolute measurement index
        if t_run not in ts_lookup.index:
            skipped += 1
            continue
        t_run_abs = int(ts_lookup[t_run])

        # Need H steps of history and F_h steps of GT in measurement array
        if t_run_abs < H or t_run_abs + F_h > T:
            skipped += 1
            continue

        # Find history run: the run from exactly H hours ago
        t_hist_target = t_run - pd.Timedelta(hours=H)
        diffs_s = np.abs((run_times_np - t_hist_target).total_seconds().values)
        r_hist  = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600:   # tolerance: 3 h (one run step)
            skipped += 1
            continue

        # Skip if any station has NaN in history or forecast window
        if _meas_nan_any[t_run_abs - H : t_run_abs + F_h].any():
            skipped += 1
            continue

        pair = (r_curr, r_hist, t_run_abs)
        if t_run < split_time:
            train_run_pairs.append(pair)
        else:
            val_run_pairs.append(pair)

    logger.info(
        "Run pairs — train: %d  val: %d  skipped: %d",
        len(train_run_pairs), len(val_run_pairs), skipped,
    )

    # ------------------------------------------------------------------
    # Model config
    # ------------------------------------------------------------------
    model_cfg = ModelConfig.from_yaml(
        stgnn_cfg,
        icond2_features=icond2_features,
        ecmwf_features=ecmwf_features,
        measurement_features=measurement_cols,
        n_train=N_train,
        n_val=N_val,
        checkpoint_path=str(model_path),
    )

    # ------------------------------------------------------------------
    # Build heterogeneous graph
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
    # Model, Sampler, Trainer
    # ------------------------------------------------------------------
    model = STGNN(model_cfg)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    sampler = TrainingSampler(
        model_cfg, builder, base_graph,
        target_feat_idx=target_feat_idx,
        station_coords=station_coords,   # (N_all, 2) raw [lat, lon] for radius filtering
    )
    trainer = InductiveTrainer(model, sampler, model_cfg, device)

    train_station_indices = list(range(N_train))
    val_station_indices   = list(range(N_train, N_train + N_val))

    logger.info("Starting training …")
    logger.info("Hyperparameters:\n%s", yaml.dump(stgnn_cfg, default_flow_style=False))
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
        trainer.load_best()
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
    pkl_stem = Path(model_name).stem   # e.g. "wind_stgcn_stgnn2_v1"
    pkl_path = Path("results") / f"{pkl_stem}_{timestamp}.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    pkl_data = {
        "model_name":   pkl_stem,
        "architecture": "stgnn2",
        "config":       stgnn_cfg,
        "train_ids":    train_ids,
        "val_ids":      val_ids,
        "n_train":      N_train,
        "n_val":        N_val,
        "history":      fit_result["history"],
        "best_val_loss": fit_result["best_val_loss"],
        "stopped_epoch": fit_result["stopped_epoch"],
        "evaluation":   eval_df,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)
    logger.info("Training results saved → %s", pkl_path)


if __name__ == "__main__":
    main()