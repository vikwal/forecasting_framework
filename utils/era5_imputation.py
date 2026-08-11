"""
utils/era5_imputation.py — ERA5-reanalysis-based imputation for wind_speed.

The SOLE imputation path for wind_speed. Regression-Kriging and the KNN
imputer are no longer used for wind_speed anywhere in this module or at its
call sites (docs/imputation_era5_only.md) — replaces the earlier two-stage
setup (docs/imputation_era5_switch.md) that still fell back to KNN for
stations/hours outside ERA5 coverage. Established by the read-only comparison
analysis in docs/imputation_era5_comparison.md (per-station OLS on four
ERA5-derived features beats Kriging by ~19% RMSE on 20034 artificially-hidden
but truly observed station-hours) and the composite-vs-pure-OLS measurement in
docs/imputation_era5_switch.md, which did NOT find a clear win for a
blended OLS/quantile-mapping estimator -- so this module implements pure
per-station Linear Regression only, per that measurement's fallback rule.

wind_direction is explicitly NOT covered here: the comparison analysis found
raw ERA5 direction loses to the existing KNN imputer in every wind class
(31.5 deg vs 9.1 deg mean error) -- direction imputation stays on the KNN
path (utils/imputation.load_knn_imputation / apply_knn_imputation), wired
unchanged at each call site.

Data source: a local per-station Parquet cache
(/mnt/lambda1/nvme1/synthetic/era5_wind_cache/Station_<sid>.parquet), NOT
Postgres. Built once by mirroring public.era5_wind for the 151 pool stations
it covers, plus GRIB-extracted data for the 2 pool stations absent from
era5_wind ('03196', '15813' -- see docs/imputation_era5_only.md section 1-2
for the calibration that established nearest-grid-point as the reproducing
extraction method, and section 2 for the extraction itself). Every cache
file carries the same 9 columns: u_wind_10m, v_wind_10m, u_wind_100m,
v_wind_100m, wind_gust_10m, friction_wind, temp_2m, pressure, dew_point_2m
-- a DatetimeIndex named 'timestamp', tz-aware UTC, hourly. Only the first
four of the wind-related columns are used as OLS features here (ERA5_FEATURES
below); temp_2m/pressure/dew_point_2m are present in the cache for future use
but deliberately NOT added to the feature set by this change (would be a
design decision outside this task's scope -- flagged, not made).

NO FALLBACK for wind_speed (docs/imputation_era5_only.md):
    - stations absent from the cache (none currently -- the cache was built
      to cover the full 153-station pool from configs/mtgnn/stdhp/
      config_wind_mtgnn_nwp_stdhp_fold1.yaml; this module makes no
      station-list assumption, so a station missing its cache file simply
      contributes no ERA5 predictions, same as before)
    - hours after ERA5_COVERAGE_END (the cache currently ends 2026-06-30
      23:00 UTC for all 153 stations; raw station measurements can extend
      past that) are DELIBERATELY left NaN. This is intentional, not a bug:
      the session that requested this change verified it costs zero
      train/val run-pairs under the fold1 config (test_end=2026-03-31 caps
      the loaded window well before ERA5's end).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# Feature order exactly as in docs/imputation_era5_comparison.md section 2(b)
# -- UNCHANGED by the cache switch (see module docstring: the cache carries
# three additional columns not used here, on purpose).
ERA5_FEATURES = ["mag10", "ratio_100_10", "friction_wind", "wind_gust_10m"]

# Per-station Parquet cache built by docs/imputation_era5_only.md step 3.
# One file per pool station: Station_<sid>.parquet, DatetimeIndex 'timestamp'
# (UTC, hourly), columns = RAW_CACHE_COLUMNS below.
# Die Ablage ist EIN Speicher, aber unter zwei Pfadkonventionen sichtbar:
# l1 ist der Besitzer und sieht sie nativ als /mnt/nvme1, l2 und ws mounten sie
# per NFS als /mnt/lambda1/nvme1. Der Config-Pfad-Rewrite auf l1 erfasst nur
# configs/, nicht Python-Konstanten — deshalb hier zur Laufzeit aufloesen.
_ERA5_CACHE_CANDIDATES = (
    "/mnt/lambda1/nvme1/synthetic/era5_wind_cache",   # l2, ws
    "/mnt/nvme1/synthetic/era5_wind_cache",           # l1
)
ERA5_CACHE_DIR = next(
    (c for c in _ERA5_CACHE_CANDIDATES if os.path.isdir(c)),
    _ERA5_CACHE_CANDIDATES[0],
)

RAW_CACHE_COLUMNS = [
    "u_wind_10m", "v_wind_10m", "u_wind_100m", "v_wind_100m",
    "wind_gust_10m", "friction_wind", "temp_2m", "pressure", "dew_point_2m",
]

# Cache coverage end (docs/imputation_era5_only.md step 3: identical across
# all 153 stations -- 151 mirror public.era5_wind's end, the 2 GRIB-extracted
# stations were extracted for the GRIB archive's own end, which coincides).
# Kept as an explicit constant only for diagnostics/logging, not as a hard
# filter -- reading each station's own cached index already yields nothing
# past what's actually cached, ERA5_COVERAGE_END never needs to be "correct"
# for correctness, only for the diagnostic breakdown.
ERA5_COVERAGE_END = pd.Timestamp("2026-06-30 23:00:00", tz="UTC")

# Same physical plausibility guard as the (now-removed) Kriging/KNN
# wind_speed imputation paths (docs/imputation_plausibility_guard.md):
# absolute ceiling for an hourly wind-speed mean at 10 m height, not a
# station-relative bound. Applied here because an OLS extrapolation can
# overshoot just like Kriging/IDW could.
WIND_SPEED_LOWER_BOUND = 0.0
WIND_SPEED_UPPER_BOUND = 40.0

# Stations with fewer than this many observed+ERA5-covered fit hours don't
# get an ERA5 model (their hours simply stay NaN -- no fallback, see module
# docstring). Reuses the same "<30 points" floor the comparison analysis
# already applied to its monthly-stratified quantile mapping (section 2c)
# rather than inventing a new threshold.
MIN_FIT_ROWS = 30


def load_era5_wind_features(
    station_ids: List[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load the per-station ERA5 Parquet cache for *station_ids* and compute
    the derived features. Replaces the former direct public.era5_wind query
    (docs/imputation_era5_switch.md) -- no Postgres connection is made here
    anymore.

    Parameters
    ----------
    station_ids : station IDs (any zero-padding; normalised to 5 digits here).
    start, end  : optional inclusive UTC timestamp bounds.
    cache_dir   : overrides ERA5_CACHE_DIR if given.

    Returns
    -------
    DataFrame with columns ['station_id', 'timestamp'] + ERA5_FEATURES,
    'timestamp' tz-aware UTC, 'station_id' zero-padded to 5 digits. Empty
    (but correctly-columned) if station_ids is empty or no cache files are
    found for any of them.
    """
    cols = ["station_id", "timestamp"] + ERA5_FEATURES
    if not station_ids:
        return pd.DataFrame(columns=cols)

    sids = sorted({str(s).zfill(5) for s in station_ids})
    cdir = cache_dir or ERA5_CACHE_DIR

    start_utc = pd.Timestamp(start).tz_convert("UTC") if start is not None and pd.Timestamp(start).tzinfo else (
        pd.Timestamp(start, tz="UTC") if start is not None else None
    )
    end_utc = pd.Timestamp(end).tz_convert("UTC") if end is not None and pd.Timestamp(end).tzinfo else (
        pd.Timestamp(end, tz="UTC") if end is not None else None
    )

    frames = []
    n_missing_files = 0
    for sid in sids:
        path = os.path.join(cdir, f"Station_{sid}.parquet")
        if not os.path.isfile(path):
            n_missing_files += 1
            continue
        df = pd.read_parquet(path, columns=["u_wind_10m", "v_wind_10m", "u_wind_100m", "v_wind_100m",
                                             "wind_gust_10m", "friction_wind"])
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        df.index = idx
        if start_utc is not None:
            df = df[df.index >= start_utc]
        if end_utc is not None:
            df = df[df.index <= end_utc]
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in df.columns:
            df = df.rename(columns={df.columns[0]: "timestamp"})
        df.insert(0, "station_id", sid)
        frames.append(df)

    if n_missing_files:
        logger.info(
            "ERA5 cache: %d/%d requested station(s) have no cache file under %s "
            "(no ERA5 predictions for those -- stays NaN, no fallback).",
            n_missing_files, len(sids), cdir,
        )

    if not frames:
        return pd.DataFrame(columns=cols)

    out = pd.concat(frames, axis=0, ignore_index=True)

    mag10 = np.hypot(out["u_wind_10m"].to_numpy(dtype=float), out["v_wind_10m"].to_numpy(dtype=float))
    mag100 = np.hypot(out["u_wind_100m"].to_numpy(dtype=float), out["v_wind_100m"].to_numpy(dtype=float))
    out["mag10"] = mag10
    out["ratio_100_10"] = mag100 / np.maximum(mag10, 0.1)
    # wind_gust_10m, friction_wind already present as raw columns.

    return out[cols]


def fit_station_ols(
    era5_df: pd.DataFrame,
    truth_long: pd.DataFrame,
    min_fit_rows: int = MIN_FIT_ROWS,
) -> Tuple[Dict[str, LinearRegression], pd.DataFrame]:
    """Fit one per-station LinearRegression: wind_speed ~ ERA5_FEATURES.

    Parameters
    ----------
    era5_df     : output of load_era5_wind_features (columns: station_id,
                  timestamp, ERA5_FEATURES...).
    truth_long  : long-format observed measurements, columns
                  ['station_id', 'timestamp', 'wind_speed'] -- wind_speed
                  must already be NaN-free here (caller filters to observed
                  hours only; this is the fit set, hidden/missing hours must
                  not be in it).
    min_fit_rows: stations with fewer merged fit rows than this get no model.

    Returns
    -------
    models : {station_id: fitted LinearRegression}
    coefs  : DataFrame, one row per fitted station, columns
             ['station_id', 'n_fit_rows', 'intercept', 'coef_mag10',
              'coef_ratio_100_10', 'coef_friction_wind', 'coef_wind_gust_10m']
             -- the "Koeffizienten je Station protokollieren" audit trail.
    """
    merged = truth_long.merge(era5_df, on=["station_id", "timestamp"], how="inner")
    merged = merged.dropna(subset=["wind_speed"] + ERA5_FEATURES)

    models: Dict[str, LinearRegression] = {}
    rows = []
    for sid, g in merged.groupby("station_id"):
        if len(g) < min_fit_rows:
            continue
        X = g[ERA5_FEATURES].to_numpy(dtype=float)
        y = g["wind_speed"].to_numpy(dtype=float)
        model = LinearRegression()
        model.fit(X, y)
        models[sid] = model
        rows.append({
            "station_id": sid,
            "n_fit_rows": len(g),
            "intercept": model.intercept_,
            **{f"coef_{f}": c for f, c in zip(ERA5_FEATURES, model.coef_)},
        })
    coefs = pd.DataFrame(rows, columns=["station_id", "n_fit_rows", "intercept"] + [f"coef_{f}" for f in ERA5_FEATURES])
    return models, coefs


def predict_station_ols(
    models: Dict[str, LinearRegression],
    era5_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the fitted per-station models to *era5_df* rows.

    Returns a long DataFrame ['station_id', 'timestamp', 'era5_pred'] --
    only for stations that have a fitted model AND have finite features
    (NaN features, e.g. a friction_wind gap, are left out, not zero-filled).
    """
    out = []
    for sid, g in era5_df.groupby("station_id"):
        model = models.get(sid)
        if model is None:
            continue
        feat = g[ERA5_FEATURES]
        valid = feat.notna().all(axis=1)
        if not valid.any():
            continue
        gv = g.loc[valid]
        pred = model.predict(gv[ERA5_FEATURES].to_numpy(dtype=float))
        # NOTE: build the result frame from a column *slice* of gv, not
        # gv["timestamp"].values -- .values on a tz-aware datetime64 Series
        # can silently drop the UTC tzinfo depending on pandas version,
        # which then breaks the exact-Timestamp dict lookup in
        # load_era5_imputation() below (every row would map to nothing).
        result = gv[["station_id", "timestamp"]].reset_index(drop=True).copy()
        result["era5_pred"] = pred
        out.append(result)
    if not out:
        return pd.DataFrame(columns=["station_id", "timestamp", "era5_pred"])
    return pd.concat(out, axis=0, ignore_index=True)


def _apply_plausibility_guard(pred: np.ndarray) -> np.ndarray:
    """Same guard as the (now-removed) Kriging/KNN wind_speed path
    (docs/imputation_plausibility_guard.md): negative -> 0.0, > 40.0 -> 40.0.
    NaN passes through untouched."""
    finite = ~np.isnan(pred)
    neg_mask = finite & (pred < WIND_SPEED_LOWER_BOUND)
    hi_mask = finite & (pred > WIND_SPEED_UPPER_BOUND)
    n_neg = int(np.count_nonzero(neg_mask))
    n_hi = int(np.count_nonzero(hi_mask))
    if n_neg:
        pred = np.where(neg_mask, WIND_SPEED_LOWER_BOUND, pred)
    if n_hi:
        pred = np.where(hi_mask, WIND_SPEED_UPPER_BOUND, pred)
    if n_neg or n_hi:
        logger.warning(
            "ERA5 imputation plausibility guard: clipped %d value(s) < %.1f -> %.1f, "
            "%d value(s) > %.1f -> %.1f",
            n_neg, WIND_SPEED_LOWER_BOUND, WIND_SPEED_LOWER_BOUND,
            n_hi, WIND_SPEED_UPPER_BOUND, WIND_SPEED_UPPER_BOUND,
        )
    return pred


def load_era5_imputation(
    station_ids: List[str],
    timestamps: pd.DatetimeIndex,
    meas_raw: np.ndarray,
    measurement_cols: List[str],
    target_col: str = "wind_speed",
    cache_dir: Optional[str] = None,
    min_fit_rows: int = MIN_FIT_ROWS,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, int]]:
    """High-level entry point: fit + apply per-station ERA5 OLS imputation
    for *target_col*, aligned to (T, N) like the removed
    load_interpol_imputation / load_knn_imputation wind_speed paths, so it
    slots into apply_interpol_imputation() UNCHANGED at call sites -- only
    the array passed in changes.

    This is now the ONLY imputation source for wind_speed -- call sites must
    NOT layer a KNN (or any other) fallback on top of this function's output
    for target_col='wind_speed' (docs/imputation_era5_only.md). Cells this
    function leaves NaN are meant to stay NaN.

    Parameters
    ----------
    station_ids      : station IDs, same order as meas_raw's N axis.
    timestamps        : DatetimeIndex, same order/length as meas_raw's T axis.
    meas_raw          : (T, N, M) float32, NOT yet imputed -- the observed
                         mask for the fit set is derived from this directly
                         (target_col column, non-NaN = observed).
    measurement_cols  : column names for meas_raw's last axis.
    target_col        : which column to impute (only 'wind_speed' is
                         meaningful; ERA5 direction is deliberately unused,
                         see module docstring).
    cache_dir         : overrides ERA5_CACHE_DIR if given.
    min_fit_rows      : passed through to fit_station_ols.

    Returns
    -------
    era5_pred : (T, N) float32, NaN wherever the cache doesn't cover the
                station/hour or the station had too few fit rows.
    coefs     : per-station coefficient audit table (see fit_station_ols).
    diag      : {'n_stations_fitted', 'n_stations_no_era5_rows',
                 'n_stations_below_min_fit_rows', 'n_cells_filled',
                 'n_cells_missing_total'} -- for the switch-report tables.
    """
    if target_col not in measurement_cols:
        raise ValueError(f"target_col '{target_col}' not in measurement_cols={measurement_cols}")
    tidx = measurement_cols.index(target_col)

    T, N, _ = meas_raw.shape
    assert N == len(station_ids)
    assert T == len(timestamps)

    era5_df = load_era5_wind_features(
        station_ids, start=timestamps[0], end=timestamps[-1], cache_dir=cache_dir,
    )

    stations_with_era5 = set(era5_df["station_id"].unique().tolist())
    n_stations_no_era5_rows = len(set(str(s).zfill(5) for s in station_ids) - stations_with_era5)

    # Fit set: observed hours (target_col not NaN in meas_raw), long format.
    observed_mask = ~np.isnan(meas_raw[:, :, tidx])
    obs_t, obs_n = np.where(observed_mask)
    truth_long = pd.DataFrame({
        "station_id": [str(station_ids[j]).zfill(5) for j in obs_n],
        "timestamp": timestamps[obs_t],
        "wind_speed": meas_raw[obs_t, obs_n, tidx],
    })

    models, coefs = fit_station_ols(era5_df, truth_long, min_fit_rows=min_fit_rows)
    n_stations_below_min = len(stations_with_era5) - len(models)

    pred_long = predict_station_ols(models, era5_df)

    era5_pred = np.full((T, N), np.nan, dtype=np.float32)
    if not pred_long.empty:
        sid_to_col = {str(s).zfill(5): j for j, s in enumerate(station_ids)}
        ts_to_row = {ts: i for i, ts in enumerate(timestamps)}
        rows = pred_long["timestamp"].map(ts_to_row)
        colidx = pred_long["station_id"].map(sid_to_col)
        valid = rows.notna() & colidx.notna()
        r = rows[valid].to_numpy(dtype=int)
        c = colidx[valid].to_numpy(dtype=int)
        vals = _apply_plausibility_guard(pred_long.loc[valid, "era5_pred"].to_numpy(dtype=float))
        era5_pred[r, c] = vals.astype(np.float32)

    # Diagnostics: how many currently-missing target_col cells get filled.
    missing_mask = np.isnan(meas_raw[:, :, tidx])
    n_cells_missing_total = int(missing_mask.sum())
    n_cells_filled = int((missing_mask & ~np.isnan(era5_pred)).sum())

    diag = {
        "n_stations_total": N,
        "n_stations_no_era5_rows": n_stations_no_era5_rows,
        "n_stations_below_min_fit_rows": n_stations_below_min,
        "n_stations_fitted": len(models),
        "n_cells_missing_total": n_cells_missing_total,
        "n_cells_filled": n_cells_filled,
        "n_cells_still_missing": n_cells_missing_total - n_cells_filled,
    }
    logger.info(
        "ERA5 imputation '%s': %d/%d stations fitted (%d without ERA5 cache rows, %d below "
        "min_fit_rows=%d) -- filled %d/%d missing cells, %d remain NaN (no fallback).",
        target_col, diag["n_stations_fitted"], N, diag["n_stations_no_era5_rows"],
        diag["n_stations_below_min_fit_rows"], min_fit_rows,
        diag["n_cells_filled"], diag["n_cells_missing_total"], diag["n_cells_still_missing"],
    )
    return era5_pred, coefs, diag
