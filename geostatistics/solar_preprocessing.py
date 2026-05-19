"""
solar_preprocessing.py — NWP data loading for the solar forecasting use case.

Analogous to load_icond2_ml_runs() in train_stgnn2.py, but reads ICON-D2
surface-level (SL) parquet files containing shortwave radiation fields.

File structure (mirrors the ML wind layout)
-------------------------------------------
  {nwp_path}/SL/{run_hour:02d}/{station_id}/{stem}_SL.parquet

Each file covers one grid point.  Columns include:
  starttime, forecasttime, longitude, latitude,
  aswdifd_s_avg, aswdir_s_avg, aswdifd_s, aswdir_s,
  alb_rad, clct, t_2m, td_2m, t_g, u_10m, v_10m, ...
  delivery_hour

forecasttime is in fractional hours (15-min steps: 0.0, 0.25, 0.5, …, 48.0).

Resampling
----------
15-minute NWP data is resampled to 1 h by grouping on floor(forecasttime)
and averaging all columns.  Leads 1..48 are kept (lead 0 = analysis time
is discarded), giving the same (R, 48, N_grid, F) dense array expected by
the DCRNN trainer.

Derived features
----------------
The following special feature names can be requested in the config and are
computed before resampling:

  ghi_nwp   = aswdir_s + aswdifd_s   (global horizontal irradiance)
  dhi_nwp   = aswdifd_s               (diffuse horizontal irradiance)
  bhi_nwp   = aswdir_s                (direct beam on horizontal)
  wind_speed_nwp = sqrt(u_10m² + v_10m²)

All other names are treated as direct column names in the parquet file.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from geostatistics.train_stgnn2 import _GEOD, _parse_latlon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DERIVED = {"ghi_nwp", "dhi_nwp", "bhi_nwp", "wind_speed_nwp"}


def _add_derived_cols(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute derived columns in-place before resampling."""
    if "ghi_nwp" in features:
        df["ghi_nwp"] = (
            df.get("aswdir_s", pd.Series(0.0, index=df.index))
            + df.get("aswdifd_s", pd.Series(0.0, index=df.index))
        ).clip(lower=0)
    if "dhi_nwp" in features:
        df["dhi_nwp"] = df.get("aswdifd_s", pd.Series(np.nan, index=df.index)).clip(lower=0)
    if "bhi_nwp" in features:
        df["bhi_nwp"] = df.get("aswdir_s", pd.Series(np.nan, index=df.index)).clip(lower=0)
    if "wind_speed_nwp" in features:
        u = df.get("u_10m", pd.Series(0.0, index=df.index))
        v = df.get("v_10m", pd.Series(0.0, index=df.index))
        df["wind_speed_nwp"] = np.sqrt(u**2 + v**2)
    return df


def _select_nearest_sl_files(
    sid_dir: Path,
    station_lat: float,
    station_lon: float,
    k: int,
) -> list[tuple[Path, str, float, float]]:
    """
    List all *_SL.parquet in sid_dir, rank by geodesic distance, return k nearest.
    Returns list of (fpath, stem, grid_lat, grid_lon).
    """
    candidates = []
    for fpath in sid_dir.glob("*_SL.parquet"):
        stem = fpath.stem.replace("_SL", "")
        try:
            glat, glon = _parse_latlon(stem)
        except Exception:
            continue
        candidates.append((fpath, stem, glat, glon))

    if not candidates:
        return []

    lons1 = np.full(len(candidates), station_lon)
    lats1 = np.full(len(candidates), station_lat)
    glats = np.array([c[2] for c in candidates])
    glons = np.array([c[3] for c in candidates])
    _, _, dists = _GEOD.inv(lons1, lats1, glons, glats)
    order = np.argsort(dists)
    return [candidates[i] for i in order[:k]]


def _load_solar_sl_parquet(
    fpath: Path,
    features: list[str],
    freq_h: float = 1.0,
) -> tuple[list[pd.Timestamp], np.ndarray]:
    """
    Load one ICON-D2 SL grid-point parquet, resample 15 min → freq_h hours,
    return (run_times, array) where array has shape (R, n_leads, F).
    n_leads = int(48 / freq_h) — e.g. 48 for 1 h, 96 for 30 min.

    Resampling strategy: group by (starttime, floor(forecasttime / freq_h)) and mean.
    """
    df = pd.read_parquet(fpath)
    df["starttime"] = pd.to_datetime(df["starttime"], utc=True)

    df = _add_derived_cols(df, features)

    n_leads_max = int(round(48.0 / freq_h))
    df["lead_bin"] = np.floor(df["forecasttime"] / freq_h).astype(int)

    available = [f for f in features if f in df.columns]
    if not available:
        raise ValueError(
            f"None of the requested features {features} found in {fpath}. "
            f"Available columns: {list(df.columns)}"
        )
    grouped = (
        df.groupby(["starttime", "lead_bin"])[available]
        .mean()
        .reset_index()
    )

    # Keep leads 1..n_leads_max (discard analysis-time bin 0)
    grouped = grouped[(grouped["lead_bin"] >= 1) & (grouped["lead_bin"] <= n_leads_max)].copy()
    grouped["lead_idx"] = grouped["lead_bin"] - 1

    run_times_raw = sorted(df["starttime"].unique())
    R = len(run_times_raw)
    run_idx_map = {t: i for i, t in enumerate(run_times_raw)}
    grouped["run_idx"] = grouped["starttime"].map(run_idx_map)

    F = len(features)
    result = np.full((R, n_leads_max, F), np.nan, dtype=np.float32)

    for fi, feat in enumerate(features):
        if feat not in grouped.columns:
            continue
        r_idx = grouped["run_idx"].values
        l_idx = grouped["lead_idx"].values
        result[r_idx, l_idx, fi] = grouped[feat].values.astype(np.float32)

    run_times_utc = [
        t.tz_convert("UTC") if getattr(t, "tzinfo", None) else pd.Timestamp(t, tz="UTC")
        for t in run_times_raw
    ]
    return run_times_utc, result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_solar_sl_runs(
    nwp_path: str,
    station_ids: list[str],
    station_coords: np.ndarray,           # (N, 2) [lat, lon]
    features: list[str],
    run_hours: tuple[int, ...] = (6,),
    next_n_grid: int = 4,
    n_workers: int = 8,
    cutoff: pd.Timestamp | None = None,
    freq_h: float = 1.0,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all ICON-D2 SL runs for the solar use case.

    Mirrors load_icond2_ml_runs() in train_stgnn2.py; returns the same
    array contract so the DCRNN trainer needs no modification.

    Returns
    -------
    run_times            : DatetimeIndex (R,) sorted UTC
    grid_coords          : (N_grid, 2) float32 [lat, lon]
    grid_runs            : (R, n_leads, N_grid, F) float32  (n_leads = int(48/freq_h))
    station_nearest_grid : (N_stations,) int64 — index into grid_coords
    """
    F = len(features)
    N = len(station_ids)
    sl_base = Path(nwp_path) / "SL"

    # ------------------------------------------------------------------
    # Phase 1: per station, find k nearest SL files per run-hour
    # ------------------------------------------------------------------
    station_grid_keys: list[list[tuple[str, int]]] = [[] for _ in range(N)]
    unique_grid_paths: dict[tuple[str, int], Path] = {}

    for si, sid in enumerate(tqdm(station_ids, desc="Scanning ICON-D2 SL dirs")):
        s_lat = float(station_coords[si, 0])
        s_lon = float(station_coords[si, 1])
        for rh in run_hours:
            sid_dir = sl_base / f"{rh:02d}" / sid
            if not sid_dir.exists():
                continue
            nearest = _select_nearest_sl_files(sid_dir, s_lat, s_lon, next_n_grid)
            for fpath, stem, _, _ in nearest:
                key = (stem, rh)
                station_grid_keys[si].append(key)
                if key not in unique_grid_paths:
                    unique_grid_paths[key] = fpath

    if not unique_grid_paths:
        raise FileNotFoundError(
            f"No ICON-D2 SL parquet files found under {sl_base}. "
            f"Check run_hours={list(run_hours)} and station_ids."
        )
    logger.info(
        "ICON-D2 SL: %d unique (grid, run-hour) parquet files (%d stations × %d run-hours × %d grid pts)",
        len(unique_grid_paths), N, len(run_hours), next_n_grid,
    )

    # ------------------------------------------------------------------
    # Phase 2: parallel loading and 15-min → freq_h resampling
    # ------------------------------------------------------------------
    sl_results: dict[tuple[str, int], tuple[list, np.ndarray]] = {}

    def _load(key_fpath):
        key, fpath = key_fpath
        return key, _load_solar_sl_parquet(fpath, features, freq_h=freq_h)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_load, kp): kp[0] for kp in unique_grid_paths.items()}
        for fut in tqdm(
            as_completed(futures), total=len(futures),
            desc="Loading ICON-D2 SL parquet files", unit="files",
        ):
            key = futures[fut]
            try:
                _, (run_times_h, arr_h) = fut.result()
                sl_results[key] = (run_times_h, arr_h)
            except Exception as exc:
                logger.warning("Failed loading %s: %s", key, exc)

    # ------------------------------------------------------------------
    # Phase 3: global run-time index
    # ------------------------------------------------------------------
    all_run_times: set[pd.Timestamp] = set()
    for run_times_h, _ in sl_results.values():
        for t in run_times_h:
            if t.hour not in run_hours:
                continue
            if cutoff is not None and t > cutoff:
                continue
            all_run_times.add(t)

    run_times_global = pd.DatetimeIndex(sorted(all_run_times))
    R = len(run_times_global)
    run_idx_map = {t: i for i, t in enumerate(run_times_global)}
    logger.info(
        "ICON-D2 SL: %d total runs (%s … %s)",
        R, run_times_global[0].date(), run_times_global[-1].date(),
    )

    # ------------------------------------------------------------------
    # Phase 4: unique grid nodes and dense array
    # ------------------------------------------------------------------
    unique_stems_rh = sorted(unique_grid_paths.keys())
    unique_stems    = sorted({stem for stem, _ in unique_stems_rh})
    grid_coords     = np.array([_parse_latlon(s) for s in unique_stems], dtype=np.float32)
    N_grid          = len(unique_stems)
    stem_to_gi      = {s: i for i, s in enumerate(unique_stems)}
    logger.info("ICON-D2 SL: %d unique grid nodes", N_grid)

    n_leads_max = int(round(48.0 / freq_h))
    grid_runs = np.full((R, n_leads_max, N_grid, F), np.nan, dtype=np.float32)

    for (stem, rh), (run_times_h, arr_h) in sl_results.items():
        gi = stem_to_gi[stem]
        for local_ri, t in enumerate(run_times_h):
            global_ri = run_idx_map.get(t)
            if global_ri is not None:
                grid_runs[global_ri, :, gi, :] = arr_h[local_ri, :, :]

    # ------------------------------------------------------------------
    # Phase 5: nearest grid node per station
    # ------------------------------------------------------------------
    station_nearest_grid = np.zeros(N, dtype=np.int64)
    for si in range(N):
        keys = station_grid_keys[si]
        if keys:
            stem = keys[0][0]
            station_nearest_grid[si] = stem_to_gi.get(stem, 0)
        else:
            logger.warning("Station %s: no SL grid keys found", station_ids[si])

    nan_count = int(np.isnan(grid_runs).sum())
    if nan_count > 0:
        nan_runs      = np.where(np.isnan(grid_runs).any(axis=(1, 2, 3)))[0]
        nan_grid_mask = np.isnan(grid_runs).any(axis=(0, 1, 3))
        nan_grid_idx  = np.where(nan_grid_mask)[0]
        affected_sta  = np.where(np.isin(station_nearest_grid, nan_grid_idx))[0]
        logger.warning(
            "ICON-D2 SL: %d NaN value(s) in grid data "
            "(%d runs, %d grid nodes, %d affected stations). "
            "Consider filling gaps in the SL parquet pipeline.",
            nan_count, len(nan_runs), len(nan_grid_idx), len(affected_sta),
        )

    return run_times_global, grid_coords, grid_runs, station_nearest_grid
