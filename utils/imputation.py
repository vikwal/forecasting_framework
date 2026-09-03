"""
utils/imputation.py — Shared spatial imputation utilities.

WIND-SPEED IMPUTATION SOURCE (since 2026-09-02, docs/imputation_tft_switch.md)
-----------------------------------------------------------------------------
Gaps in `wind_speed` are filled from the per-station Parquets under
`data.interpol_path` (…/synthetic/interpol/wind), column **`imputed`** —
the prediction of the Temporal-Fusion-Transformer wind closing model
(203 stations, ERA5 + neighbour stations + statics, 2023-07-24 00:00 UTC …
2026-07-31 23:00 UTC hourly).

That replaces BOTH earlier paths:
  * Regression-Kriging (`rk_pred`) — the column no longer exists in those
    files at all, so anything still reading it raises instead of degrading
    silently. Kept only for reference in
    …/synthetic/interpol/wind_vor_tft_20260902.
  * the per-station ERA5-OLS of `utils/era5_imputation.py`
    (docs/imputation_era5_only.md) — that module is no longer called by any
    pipeline; it covered 153 stations and ended 2026-06-30, the TFT covers
    203 stations to 2026-07-31.

`imputed` is non-NaN EXACTLY at the hours where `wind_speed_raw` is NaN, so
it slots into the same "fill NaN in meas_raw" contract the previous arrays
had. There is NO fallback layered on top of it: cells it does not cover
(hours outside the file's window, stations without a file) stay NaN by
design, same rule as the ERA5-only path before it.

The imputed values carry NO validated error metric — the closing model's
skill was measured on artificially hidden but truly observed hours, not on
the real gaps. `kontextfrei` (True = no own measurement anywhere in the
48-h window, ~36 % of all filled hours) is the honest way to separate
well-supported from weakly-supported fills in an evaluation; it is carried
through the diagnostics of every function below.

`wind_direction` is NOT covered — it stays on the spatial-KNN path
(load_knn_imputation / apply_knn_imputation), unchanged.

SOLAR is not affected: `interpol/solar` still carries `rk_pred`, and the
loaders below resolve the value column from the file's own schema
(`imputed` preferred, `rk_pred` accepted) instead of hard-coding one.

Provides two layers:

Array-based (used by train_dcrnn.py / hpo_dcrnn.py via train_stgnn2.py):
  load_gap_imputation          — load the gap-filling column → (T, N) float32
  load_knn_imputation          — load spatial-KNN parquet → (T, N) float32
  apply_imputation             — fill NaN in (T, N, M) meas_raw with a (T, N) array
  apply_knn_imputation         — fill NaN in (T, N, M) meas_raw with knn_arr
  impute_meas_raw_from_interpol— load + apply + diagnostics, one call

DataFrame-based (used by preprocessing.py / train_cl.py / hpo_cl.py):
  impute_dfs_from_interpol   — fill NaN in target_col of {key: DataFrame} dict
  impute_dfs_with_knn        — fill NaN in feature cols using spatial-KNN parquets
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interpol-file schema
# ---------------------------------------------------------------------------

# Value column holding the gap fill, in order of preference. 'imputed' is the
# TFT prediction written on 2026-09-02 for wind; 'rk_pred' is the older
# Regression-Kriging column, still what interpol/solar carries.
IMPUTATION_VALUE_COLUMNS = ("imputed", "rk_pred")

# Welche Spalte die Luecke einer bestimmten Messgroesse fuellt, in
# Vorzugsreihenfolge. Seit dem 2026-09-03 fuehrt der Wind-Baum
# (interpol/wind_richtung) neben 'imputed' auch 'imputed_dir' — die Richtung in
# Grad, aus demselben Abschlussmodell. Eine Messgroesse ohne Eintrag hier wird
# aus dem Interpol-Baum NICHT gefuellt; fuer sie bleibt der KNN-Pfad zustaendig
# (so bleibt Solar unberuehrt, dessen Baum nur 'rk_pred' kennt).
IMPUTATION_COLUMN_BY_FEATURE = {
    "wind_speed":     ("imputed", "rk_pred"),
    "wind_direction": ("imputed_dir",),
}

# Per-hour flag written by the TFT closing model: True = the whole 48-h window
# around this hour held no measurement of this station, so the value rests on
# ERA5 + neighbours + statics alone. Absent in the older Kriging files.
CONTEXTFREE_COLUMN = "kontextfrei"


def _parquet_columns(fpath: str) -> list[str]:
    """Column names of a Parquet file, read from its footer only."""
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(fpath).schema_arrow.names)
    except Exception:  # pragma: no cover — engine without a footer reader
        return list(pd.read_parquet(fpath).columns)


def resolve_imputation_column(fpath: str, feature: str | None = None) -> str | None:
    """Pick the gap-fill value column of one interpol Parquet.

    Without *feature* this resolves the target column as before and raises
    KeyError if the file carries none of IMPUTATION_VALUE_COLUMNS —
    deliberately loud: a file without a usable column means the directory is
    not what the caller thinks it is.

    With *feature* it resolves that measurement's column from
    IMPUTATION_COLUMN_BY_FEATURE and returns None when the tree does not carry
    it. None is a normal answer here, not an error: it says "this directory has
    nothing for this measurement", and the caller decides what follows.
    """
    cols = _parquet_columns(fpath)
    if feature is not None:
        for cand in IMPUTATION_COLUMN_BY_FEATURE.get(feature, ()):
            if cand in cols:
                return cand
        return None
    for cand in IMPUTATION_VALUE_COLUMNS:
        if cand in cols:
            return cand
    raise KeyError(
        f"{fpath} carries none of the known imputation columns "
        f"{IMPUTATION_VALUE_COLUMNS} — found {cols}. "
        "Wind files written before 2026-09-02 had 'rk_pred'; the TFT files "
        "written since have 'imputed' (docs/imputation_tft_switch.md)."
    )


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

def load_gap_imputation(
    interpol_path: str,
    station_ids: list[str],
    timestamps: pd.DatetimeIndex,
    value_col: Optional[str] = None,
    with_kontextfrei: bool = False,
):
    """Load the gap-filling values from *interpol_path* and align to timestamps.

    For wind this is the TFT column `imputed` (non-NaN only at the hours the
    station has no raw measurement); for the older solar files it is
    `rk_pred`. The column is resolved from each file's own schema unless
    *value_col* is given.

    Parameters
    ----------
    interpol_path    : directory with Station_XXXXX.parquet files
    station_ids      : station IDs, defines the N axis order
    timestamps       : DatetimeIndex, defines the T axis
    value_col        : force a column name instead of resolving it per file
    with_kontextfrei : also return the (T, N) bool mask of context-free fills
                       (all-False where a file has no such column)

    Returns
    -------
    values : (T, N) float32 — NaN where the file has no value for that hour
             (including: no file for that station, hour outside the file).
    kontextfrei : (T, N) bool — only if with_kontextfrei=True.
    """
    val_series, ctx_series = [], []
    n_files = 0
    cols_seen: set[str] = set()
    for sid in station_ids:
        fpath = os.path.join(interpol_path, f"Station_{sid}.parquet")
        empty = pd.Series(np.nan, index=timestamps, name=sid, dtype="float32")
        if not os.path.exists(fpath):
            val_series.append(empty)
            ctx_series.append(pd.Series(False, index=timestamps, name=sid, dtype=bool))
            continue
        n_files += 1
        col = value_col or resolve_imputation_column(fpath)
        cols_seen.add(col)
        wanted = ["timestamp", col]
        has_ctx = with_kontextfrei and CONTEXTFREE_COLUMN in _parquet_columns(fpath)
        if has_ctx:
            wanted.append(CONTEXTFREE_COLUMN)
        df = pd.read_parquet(fpath, columns=wanted)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        val_series.append(df[col].astype("float32").rename(sid))
        if with_kontextfrei:
            ctx = df[CONTEXTFREE_COLUMN] if has_ctx else pd.Series(False, index=df.index)
            ctx_series.append(ctx.astype(bool).rename(sid))

    values = pd.concat(val_series, axis=1).reindex(timestamps).values.astype(np.float32)
    logger.info(
        "Interpol imputation: %d/%d stations have a file under %s, column(s) %s",
        n_files, len(station_ids), interpol_path, sorted(cols_seen) or ["—"],
    )
    if not with_kontextfrei:
        return values
    ctx = pd.concat(ctx_series, axis=1).reindex(timestamps).astype("boolean").fillna(False)
    return values, ctx.to_numpy(dtype=bool)


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


def apply_imputation(
    meas_raw: np.ndarray,
    values: np.ndarray,
    measurement_cols: list[str],
    target_col: str = "wind_speed",
) -> np.ndarray:
    """Fill NaN in *target_col* channel of *meas_raw* with *values*.

    Parameters
    ----------
    meas_raw         : (T, N, M) float32
    values           : (T, N) float32 — the imputation source
    measurement_cols : ordered list of column names (last dim of meas_raw)
    target_col       : which column to fill

    Returns *meas_raw* modified in-place. Cells where *values* is NaN stay
    NaN — no fallback (docs/imputation_tft_switch.md).
    """
    if target_col not in measurement_cols:
        return meas_raw
    idx  = measurement_cols.index(target_col)
    mask = np.isnan(meas_raw[:, :, idx])
    meas_raw[:, :, idx][mask] = values[mask]
    return meas_raw


def impute_meas_raw_from_interpol(
    meas_raw: np.ndarray,
    station_ids: list[str],
    timestamps: pd.DatetimeIndex,
    measurement_cols: list[str],
    interpol_path: str,
    target_col: str = "wind_speed",
    secondary_cols: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Fill NaN in *target_col* — and, if the tree carries them, in the other
    measurement columns — of *meas_raw* from the interpol directory.

    The single entry point every geostatistics pipeline uses — replaces the
    former `load_interpol_imputation` + `load_era5_imputation` +
    `apply_interpol_imputation` sequence (docs/imputation_tft_switch.md).

    Returns
    -------
    meas_raw : modified in place
    diag     : counts for the switch report —
               value_col, n_cells_missing_total, n_cells_filled,
               n_cells_still_missing, n_cells_filled_kontextfrei,
               n_cells_offered_unused (an imputed value existed where the
               measurement was NOT missing — a raster mismatch between the
               interpol files and load_station_measurements, expected 0);
               handled_cols (every measurement column this call filled, so the
               caller can skip the KNN path for exactly those) and
               secondary, a per-column breakdown of the same counts.

    secondary_cols
        When True (default), every measurement column other than *target_col*
        that IMPUTATION_COLUMN_BY_FEATURE knows AND the tree actually carries is
        filled from the same files — for wind that is 'wind_direction' from
        'imputed_dir' since 2026-09-03. Columns the tree does not cover are left
        untouched and stay the KNN path's business. There is no fallback in
        either direction: a column filled from here is NOT topped up by KNN, so
        a gap the tree does not cover reaches the NaN audit and stops the run.
    """
    if target_col not in measurement_cols:
        logger.info(
            "Interpol imputation skipped: '%s' not in measurement_cols=%s",
            target_col, measurement_cols,
        )
        return meas_raw, {"value_col": None, "n_cells_filled": 0,
                          "handled_cols": [], "secondary": {}}

    tidx = measurement_cols.index(target_col)
    # Resolve the value column once, from the first station file that exists,
    # and use it for the whole directory: a directory where only some files
    # carry 'imputed' is not a state any pipeline should quietly average over.
    value_col = None
    probe_path = None
    for sid in station_ids:
        fpath = os.path.join(interpol_path, f"Station_{sid}.parquet")
        if os.path.exists(fpath):
            probe_path = fpath
            value_col = resolve_imputation_column(fpath)
            break
    if value_col is None:
        raise FileNotFoundError(
            f"interpol_path {interpol_path} holds no Station_*.parquet for any of the "
            f"{len(station_ids)} requested stations."
        )
    values, kontextfrei = load_gap_imputation(
        interpol_path, station_ids, timestamps,
        value_col=value_col, with_kontextfrei=True,
    )
    have = ~np.isnan(values)
    missing = np.isnan(meas_raw[:, :, tidx])

    n_missing = int(missing.sum())
    fill_mask = missing & have
    n_filled = int(fill_mask.sum())

    meas_raw = apply_imputation(meas_raw, values, measurement_cols, target_col)

    diag: Dict[str, object] = {
        "value_col": value_col,
        "n_stations": len(station_ids),
        "n_cells_missing_total": n_missing,
        "n_cells_filled": n_filled,
        "n_cells_still_missing": n_missing - n_filled,
        "n_cells_filled_kontextfrei": int((fill_mask & kontextfrei).sum()),
        "n_cells_offered_unused": int((have & ~missing).sum()),
    }
    logger.info(
        "Interpol imputation '%s' from column '%s': %d NaN → %d NaN (%d filled, of them %d "
        "kontextfrei; %d offered values fell on non-missing cells). No fallback — remaining "
        "NaN stay NaN (docs/imputation_tft_switch.md).",
        target_col, value_col, n_missing, n_missing - n_filled, n_filled,
        diag["n_cells_filled_kontextfrei"], diag["n_cells_offered_unused"],
    )

    # ---- Sekundaerspalten aus demselben Baum -----------------------------
    # Bis zum 2026-09-03 kam die Windrichtung ausschliesslich aus dem
    # KNN-Cache. Seither fuehrt interpol/wind_richtung sie als 'imputed_dir'
    # aus demselben Abschlussmodell wie die Geschwindigkeit. Was hier gefuellt
    # wird, steht in diag['handled_cols'] — der Aufrufer laesst genau diese
    # Spalten beim KNN-Schritt aus, damit keine zwei Quellen in einer Spalte
    # landen.
    handled = [target_col]
    sec_diag: Dict[str, Dict[str, object]] = {}
    if secondary_cols:
        for col in measurement_cols:
            if col == target_col:
                continue
            col_name = resolve_imputation_column(probe_path, feature=col)
            if col_name is None:
                continue
            sec_values = load_gap_imputation(
                interpol_path, station_ids, timestamps, value_col=col_name,
            )
            cidx = measurement_cols.index(col)
            sec_missing = np.isnan(meas_raw[:, :, cidx])
            sec_have = ~np.isnan(sec_values)
            n_sec_missing = int(sec_missing.sum())
            n_sec_filled = int((sec_missing & sec_have).sum())
            meas_raw = apply_imputation(meas_raw, sec_values, measurement_cols, col)
            sec_diag[col] = {
                "value_col": col_name,
                "n_cells_missing_total": n_sec_missing,
                "n_cells_filled": n_sec_filled,
                "n_cells_still_missing": n_sec_missing - n_sec_filled,
            }
            handled.append(col)
            logger.info(
                "Interpol imputation '%s' from column '%s': %d NaN → %d NaN (%d filled). "
                "No KNN fallback for this column.",
                col, col_name, n_sec_missing, n_sec_missing - n_sec_filled, n_sec_filled,
            )
    diag["handled_cols"] = handled
    diag["secondary"] = sec_diag
    return meas_raw, diag


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

def impute_dfs_from_interpol(
    dfs: Dict[str, pd.DataFrame],
    interpol_path: str,
    target_col: str,
) -> Dict[str, pd.DataFrame]:
    """Fill NaN in *target_col* of each per-station DataFrame from *interpol_path*.

    Reads per-station parquets from *interpol_path*/Station_{sid}.parquet and
    takes the gap-fill column resolved from the file's schema — `imputed`
    (TFT, wind since 2026-09-02) or `rk_pred` (older Kriging files, solar).
    Alignment is done via the DataFrame's own DatetimeIndex — no resampling.

    Cells the file does not cover stay NaN; there is no fallback
    (docs/imputation_tft_switch.md).

    Parameters
    ----------
    dfs          : {file_key: DataFrame} as returned by preprocessing.get_data()
    interpol_path: directory with Station_XXXXX.parquet files
    target_col   : column to fill (e.g. 'ghi', 'wind_speed')
    """
    filled_total = 0
    ctxfree_total = 0
    remaining_total = 0
    cols_seen: set[str] = set()
    for key, df in dfs.items():
        if target_col not in df.columns:
            continue
        nan_mask = df[target_col].isna()
        if not nan_mask.any():
            continue
        sid = _station_id_from_key(key)
        fpath = os.path.join(interpol_path, f"Station_{sid}.parquet")
        if not os.path.exists(fpath):
            logger.debug("Interpol imputation: no file for station %s — skipping", sid)
            remaining_total += int(nan_mask.sum())
            continue
        file_cols = _parquet_columns(fpath)
        col = resolve_imputation_column(fpath)
        cols_seen.add(col)
        wanted = ["timestamp", col]
        if CONTEXTFREE_COLUMN in file_cols:
            wanted.append(CONTEXTFREE_COLUMN)
        src = pd.read_parquet(fpath, columns=wanted)
        src["timestamp"] = pd.to_datetime(src["timestamp"], utc=True)
        src = src.set_index("timestamp")
        values = src[col].reindex(df.index)
        fill_mask = nan_mask & values.notna()
        n_filled = int(fill_mask.sum())
        if n_filled:
            df.loc[fill_mask, target_col] = values[fill_mask].values
            filled_total += n_filled
            if CONTEXTFREE_COLUMN in src.columns:
                ctx = src[CONTEXTFREE_COLUMN].reindex(df.index).astype("boolean").fillna(False).astype(bool)
                ctxfree_total += int((fill_mask & ctx).sum())
        nan_after = int(df[target_col].isna().sum())
        remaining_total += nan_after
        logger.debug(
            "Interpol imputation station %s '%s' (column '%s'): %d NaN → %d NaN (%d filled)",
            sid, target_col, col, int(nan_mask.sum()), nan_after, n_filled,
        )
    logger.info(
        "Interpol imputation: filled %d NaN values in '%s' across %d stations "
        "(column(s) %s, %d of the fills kontextfrei); %d NaN remain — no fallback.",
        filled_total, target_col, len(dfs), sorted(cols_seen) or ["—"],
        ctxfree_total, remaining_total,
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
