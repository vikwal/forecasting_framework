"""
solar_preprocessing.py — NWP data loading for the solar forecasting use case.

Analogous to load_icond2_ml_runs() in train_stgnn2.py, but reads ICON-D2
surface-level (SL) parquet files containing shortwave radiation fields.

File structure (does NOT mirror the ML wind layout)
---------------------------------------------------
  {nwp_path}/SL/{run_hour:02d}/{lon}_{lat}_SL.parquet

Zwei Abweichungen gegenüber den ML-Dateien, die beide verifiziert sind:

1. **Flach statt nach Station gruppiert.**  ML liegt unter
   ``ML/{run_hour}/{station_id}/``, SL dagegen direkt in ``SL/{run_hour}/`` —
   alle ~1218 Gitterpunkte in einem Verzeichnis.
2. **Dateinamen sind lon-first.**  ML: ``52_9057_12_9151_ML.parquet`` → lat 52.9057,
   lon 12.9151.  SL: ``10_0000_47_8000_SL.parquet`` → **lon 10.0, lat 47.8**
   (Feld 0 hat den Wertebereich 6.01–14.98, Feld 1 47.38–55.03).  Auch die Spalten
   *innerhalb* der SL-Datei sind vertauscht (``longitude`` enthält die Breite).

Each file covers one grid point.  Columns include:
  starttime, forecasttime, longitude, latitude,
  aswdifd_s_avg, aswdir_s_avg, aswdifd_s, aswdir_s,
  alb_rad, clct, t_2m, relhum_2m, td_2m, t_g, u_10m, v_10m, ...
  delivery_hour

forecasttime is in fractional hours (15-min steps: 0.0, 0.25, 0.5, …, 48.0);
einzelne ältere Läufe liefern nur volle Stunden.

Resampling
----------
``aswdir_s``/``aswdifd_s`` sind **Intervallmittel, die auf forecasttime enden**
(verifiziert: Mittel über ft ∈ (1, 2] = ``aswdifd_s_avg@2·2 − aswdifd_s_avg@1·1``).
Die übrigen Felder sind Momentanwerte.  Entsprechend gilt für den Lead-Index
eines linksbündig gelabelten Stundenintervalls:

  akkumuliert:  lead_idx = ceil(ft) - 1        Momentanwert:  lead_idx = floor(ft)

Beide Zweige werden getrennt gemittelt und auf (starttime, lead_idx) zusammengeführt.
Ein gemeinsames ``floor``-Mapping würde die Strahlung um eine volle Stunde
verschieben — messbar: die Kreuzkorrelation Messung↔Prognose fällt dann von
0.960 auf 0.936 (Station 00183, Apr–Sep 2025).
Ergebnis sind Leads 0..47, also dasselbe (R, 48, N_grid, F) dichte Array, das der
DCRNN-Trainer erwartet.

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
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from geostatistics.train_stgnn2 import _GEOD
from utils import solar
from utils.solar import parse_sl_latlon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DERIVED = {"ghi_nwp", "dhi_nwp", "bhi_nwp", "wind_speed_nwp"}

#: Felder, deren Wert das auf ``forecasttime`` endende Intervall mittelt (siehe
#: Modul-Docstring). Alles andere ist ein Momentanwert.
#:
#: ``aswdir_s_avg``/``aswdifd_s_avg`` gehören **nicht** hierher: sie tragen das
#: Mittel seit Vorhersagebeginn [0, ft]. Für sie passt weder ``ceil`` noch
#: ``floor``, deshalb werden sie in ``utils.solar._reject_running_means`` abgelehnt.
_ACCUMULATED = {"aswdir_s", "aswdifd_s", "ghi_nwp", "dhi_nwp", "bhi_nwp"}


def _is_accumulated(feature: str) -> bool:
    return feature in _ACCUMULATED


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


@lru_cache(maxsize=32)
def _scan_sl_dir(sl_dir_str: str) -> tuple[tuple[str, float, float], ...]:
    """Alle Gitterpunkte eines flachen SL-Laufverzeichnisses als (stem, lat, lon).

    Gecached, weil dasselbe Verzeichnis (~1218 Dateien) sonst für jede Station
    erneut gelistet würde.
    """
    sl_dir = Path(sl_dir_str)
    if not sl_dir.is_dir():
        return ()
    out = []
    for fpath in sl_dir.glob("*_SL.parquet"):
        stem = fpath.stem.replace("_SL", "")
        try:
            glat, glon = parse_sl_latlon(stem)   # SL: lat-first, wie ML
        except Exception:
            continue
        out.append((stem, glat, glon))
    return tuple(sorted(out))


def _select_nearest_sl_files(
    sl_dir: Path,
    station_lat: float,
    station_lon: float,
    k: int,
) -> list[tuple[Path, str, float, float]]:
    """
    List all *_SL.parquet in the flat run-hour directory, rank by geodesic
    distance, return k nearest.
    Returns list of (fpath, stem, grid_lat, grid_lon).
    """
    grid = _scan_sl_dir(str(sl_dir))
    if not grid:
        return []

    glats = np.array([g[1] for g in grid])
    glons = np.array([g[2] for g in grid])
    lons1 = np.full(len(grid), station_lon)
    lats1 = np.full(len(grid), station_lat)
    _, _, dists = _GEOD.inv(lons1, lats1, glons, glats)
    order = np.argsort(dists)[:k]
    return [(sl_dir / f"{grid[i][0]}_SL.parquet", grid[i][0], grid[i][1], grid[i][2])
            for i in order]


def _load_solar_sl_parquet(
    fpath: Path,
    features: list[str],
    freq_h: float = 1.0,
    sub_hourly_fill: str = "ffill",
) -> tuple[list[pd.Timestamp], np.ndarray]:
    """
    Load one ICON-D2 SL grid-point parquet, resample 15 min → freq_h hours,
    return (run_times, array) where array has shape (R, n_leads, F).
    n_leads = int(48 / freq_h) — e.g. 48 for 1 h, 96 for 30 min.

    Resampling: akkumulierte Felder über ``ceil(ft/freq_h) - 1``, Momentanwerte
    über ``floor(ft/freq_h)`` mitteln (siehe Modul-Docstring), Lead-Index 0..n_leads-1,
    Gültigkeitszeitpunkt ``starttime + lead_idx · freq_h``.

    Bei ``freq_h < 1`` greift zusätzlich ``sub_hourly_fill``: nur die Strahlungsfelder
    sind viertelstündlich abgelegt, alle übrigen (``clct``, ``t_2m``, ``alb_rad``,
    ``u_10m``, ``h_snow``, ``prr_gsp``, …) nur zur vollen Stunde. Ohne Auffüllung
    blieben drei von vier Leads dort NaN und die NaN-Behandlung würde sie verwerfen —
    ``freq: '15min'`` fiele also stillschweigend auf ein Stundenraster zurück.
    Siehe ``utils.solar._fill_sub_hourly``.
    """
    solar._reject_running_means(list(features))

    df = pd.read_parquet(fpath)
    df["starttime"] = pd.to_datetime(df["starttime"], utc=True)

    df = _add_derived_cols(df, features)

    n_leads_max = int(round(48.0 / freq_h))
    ft = df["forecasttime"].astype(float) / freq_h
    acc_bin = np.ceil(ft).astype(int) - 1
    inst_bin = np.floor(ft).astype(int)

    available = [f for f in features if f in df.columns]
    if not available:
        raise ValueError(
            f"None of the requested features {features} found in {fpath}. "
            f"Available columns: {list(df.columns)}"
        )

    acc_feats = [f for f in available if _is_accumulated(f)]
    inst_feats = [f for f in available if not _is_accumulated(f)]

    parts = []
    for feats, bins in ((acc_feats, acc_bin), (inst_feats, inst_bin)):
        if not feats:
            continue
        tmp = df[["starttime"] + feats].copy()
        tmp["lead_idx"] = bins.values
        tmp = tmp[(tmp["lead_idx"] >= 0) & (tmp["lead_idx"] < n_leads_max)]
        parts.append(tmp.groupby(["starttime", "lead_idx"], as_index=False)[feats].mean())

    grouped = parts[0]
    for extra in parts[1:]:
        grouped = grouped.merge(extra, on=["starttime", "lead_idx"], how="outer")

    if freq_h < 1.0 and inst_feats:
        grouped = grouped.rename(columns={"lead_idx": "forecasttime"})
        grouped = solar._fill_sub_hourly(grouped, inst_feats, freq_h, sub_hourly_fill)
        grouped = grouped.rename(columns={"forecasttime": "lead_idx"})

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
    sub_hourly_fill: str = "ffill",
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
            # SL ist flach: alle Gitterpunkte liegen direkt in SL/{run_hour}/,
            # es gibt kein Unterverzeichnis je Station (anders als bei ML).
            sid_dir = sl_base / f"{rh:02d}"
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
        return key, _load_solar_sl_parquet(fpath, features, freq_h=freq_h,
                                            sub_hourly_fill=sub_hourly_fill)

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
    # SL-Stems sind lat-first (wie ML); grid_coords wird als [lat, lon] erwartet.
    grid_coords     = np.array([parse_sl_latlon(s) for s in unique_stems], dtype=np.float32)
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
