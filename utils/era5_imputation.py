"""
utils/era5_imputation.py — ERA5-reanalysis-based imputation for wind_speed.

Replaces Regression-Kriging as the primary gap-filling source for missing
wind_speed measurements. Established by the read-only comparison analysis in
docs/imputation_era5_comparison.md (per-station OLS on four ERA5-derived
features beats Kriging by ~19% RMSE on 20034 artificially-hidden but truly
observed station-hours) and the composite-vs-pure-OLS measurement in
docs/imputation_era5_switch.md, which did NOT find a clear win for a
blended OLS/quantile-mapping estimator -- so this module implements pure
per-station Linear Regression only, per that measurement's fallback rule.

wind_direction is explicitly NOT covered here: the comparison analysis found
raw ERA5 direction loses to the existing KNN imputer in every wind class
(31.5 deg vs 9.1 deg mean error) -- direction imputation stays on the KNN
path (utils/imputation.load_knn_imputation / apply_knn_imputation).

Fallback chain (see docs/imputation_era5_switch.md):
    1. ERA5 + per-station OLS correction, wherever ERA5 covers the station/hour.
    2. Otherwise: existing KNN imputer (unchanged, wired at each call site).

Two things ERA5 does not cover, both handled by the KNN fallback:
    - stations absent from public.era5_wind (as of the comparison analysis:
      '03196', '15813' among the 153-station pool -- but this module makes
      no station-list assumption; whichever stations the query returns rows
      for are considered covered, so a growing era5_wind table is picked up
      automatically without a code change)
    - hours after ERA5_COVERAGE_END (public.era5_wind currently ends
      2026-06-30 23:00 UTC; raw station measurements extend past that)
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# Feature order exactly as in docs/imputation_era5_comparison.md section 2(b).
ERA5_FEATURES = ["mag10", "ratio_100_10", "friction_wind", "wind_gust_10m"]

# public.era5_wind coverage end at the time of the comparison analysis
# (docs/imputation_era5_comparison.md section 1.2: 2023-07-01 00:00 --
# 2026-06-30 23:00 UTC). Hours after this never have ERA5 rows; kept as an
# explicit constant only for diagnostics/logging, not as a hard filter --
# the SQL query and the per-timestamp merge already yield NaN for anything
# not actually present in the table, ERA5_COVERAGE_END never needs to be
# "correct" for correctness, only for the diagnostic breakdown.
ERA5_COVERAGE_END = pd.Timestamp("2026-06-30 23:00:00", tz="UTC")

# Same physical plausibility guard as the Kriging/KNN imputation paths
# (docs/imputation_plausibility_guard.md): absolute ceiling for an hourly
# wind-speed mean at 10 m height, not a station-relative bound. Applied here
# because an OLS extrapolation can overshoot just like Kriging/IDW can.
WIND_SPEED_LOWER_BOUND = 0.0
WIND_SPEED_UPPER_BOUND = 40.0

# Stations with fewer than this many observed+ERA5-covered fit hours don't
# get an ERA5 model (falls back to KNN for all of that station's hours).
# Reuses the same "<30 points" floor the comparison analysis already applied
# to its monthly-stratified quantile mapping (section 2c) rather than
# inventing a new threshold.
MIN_FIT_ROWS = 30


def _to_naive_utc(ts) -> pd.Timestamp:
    """era5_wind.timestamp is stored tz-naive but is UTC-content (verified by
    the comparison-analysis session via an offset test) -- query bounds must
    be naive too, so tz-aware bounds are stripped after converting to UTC."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def load_era5_wind_features(
    station_ids: List[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    db_url: Optional[str] = None,
) -> pd.DataFrame:
    """Load public.era5_wind for *station_ids* and compute the derived features.

    Parameters
    ----------
    station_ids : station IDs (any zero-padding; normalised to 5 digits here).
    start, end  : optional inclusive UTC timestamp bounds.
    db_url      : overrides WEATHER_DB_URL if given.

    Returns
    -------
    DataFrame with columns ['station_id', 'timestamp'] + ERA5_FEATURES,
    'timestamp' tz-aware UTC, 'station_id' zero-padded to 5 digits. Empty
    (but correctly-columned) if station_ids is empty or the query returns
    no rows.
    """
    cols = ["station_id", "timestamp"] + ERA5_FEATURES
    if not station_ids:
        return pd.DataFrame(columns=cols)

    sids = sorted({str(s).zfill(5) for s in station_ids})
    url = db_url or os.environ.get("WEATHER_DB_URL")
    if not url:
        raise ValueError(
            "WEATHER_DB_URL environment variable not set -- required to load "
            "public.era5_wind for ERA5-based wind_speed imputation."
        )

    where = ["station_id IN ({})".format(",".join(f"'{s}'" for s in sids))]
    params: list = []
    if start is not None:
        where.append("timestamp >= %s")
        params.append(_to_naive_utc(start))
    if end is not None:
        where.append("timestamp <= %s")
        params.append(_to_naive_utc(end))
    query = f"""
        SELECT station_id, timestamp, u_wind_10m, v_wind_10m, u_wind_100m, v_wind_100m,
               wind_gust_10m, friction_wind
        FROM public.era5_wind
        WHERE {' AND '.join(where)}
    """

    conn = psycopg2.connect(url)
    try:
        df = pd.read_sql(query, conn, params=params or None)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame(columns=cols)

    df["station_id"] = df["station_id"].astype(str).str.zfill(5)
    ts = pd.to_datetime(df["timestamp"])
    df["timestamp"] = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")

    mag10 = np.hypot(df["u_wind_10m"].to_numpy(dtype=float), df["v_wind_10m"].to_numpy(dtype=float))
    mag100 = np.hypot(df["u_wind_100m"].to_numpy(dtype=float), df["v_wind_100m"].to_numpy(dtype=float))
    df["mag10"] = mag10
    df["ratio_100_10"] = mag100 / np.maximum(mag10, 0.1)
    # wind_gust_10m, friction_wind already present as raw columns.

    return df[cols]


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
    """Same guard as Kriging/KNN (docs/imputation_plausibility_guard.md):
    negative -> 0.0, > 40.0 -> 40.0. NaN passes through untouched."""
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
    db_url: Optional[str] = None,
    min_fit_rows: int = MIN_FIT_ROWS,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, int]]:
    """High-level entry point: fit + apply per-station ERA5 OLS imputation
    for *target_col*, aligned to (T, N) like load_interpol_imputation /
    load_knn_imputation, so it slots into apply_interpol_imputation()
    UNCHANGED at call sites -- only the array passed in changes.

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
    db_url            : overrides WEATHER_DB_URL if given.
    min_fit_rows      : passed through to fit_station_ols.

    Returns
    -------
    era5_pred : (T, N) float32, NaN wherever ERA5 doesn't cover the
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
        station_ids, start=timestamps[0], end=timestamps[-1], db_url=db_url,
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
        "ERA5 imputation '%s': %d/%d stations fitted (%d without ERA5 rows, %d below "
        "min_fit_rows=%d) -- filled %d/%d missing cells, %d remain for KNN fallback.",
        target_col, diag["n_stations_fitted"], N, diag["n_stations_no_era5_rows"],
        diag["n_stations_below_min_fit_rows"], min_fit_rows,
        diag["n_cells_filled"], diag["n_cells_missing_total"], diag["n_cells_still_missing"],
    )
    return era5_pred, coefs, diag
