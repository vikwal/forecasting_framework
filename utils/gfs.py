"""
gfs.py — NOAA GFS 0.25° as an additional NWP source next to ICON-D2 (and ECMWF), for the
synthetic wind parks (parks_v1) and PV parks (pv_park).

Data: per-site, per-run point files of the point extraction ``NWP-points`` (``points.cli gfs``,
from the GFS Zarr store of ``NWP/GFS``):

| | wind (``data.gfs_path``) | PV (``data.gfs_solar_path``) |
|---|---|---|
| files | ``<gfs_path>/SL/{06,12}/<prefix><id>/<lat>_<lon>_wind_sl.parquet`` | ``<gfs_solar_path>/{06,12}/<prefix><id>/<lat>_<lon>_solar_sl.parquet`` |
| columns | ``u_wind{10,20,30,40,50,80,100}m``, ``v_wind…m``, ``gust``, ``temp2m``, ``dew_point2m``, ``relhum2m``, ``sp``, ``tsfc``, ``hpbl`` | ``dswrf``, ``uswrf``, ``dlwrf``, ``albdo``, ``tcc``, ``lcc``, ``mcc``, ``hcc``, ``t_2m``, ``td_2m``, ``relhum_2m``, ``tsfc``, ``sp``, ``u_10m``, ``v_10m``, ``tp``, ``prate``, ``sde`` |

``forecasttime`` 0..57 h (int), lat-first file stems like ECMWF. ``dswrf``/``uswrf``/``dlwrf`` (W/m²)
and ``tp`` (kg/m² per hour) are means/sums over the hour **ending** at ``forecasttime`` (NaN at 0),
everything else is instantaneous.

Features (``params.known_features`` / ``observed_features``), suffix ``_<rank>`` per grid point:

* ``gfs_wind_speed_h<NNN>`` — |u, v| at NNN m (10, 20, 30, 40, 50, 80, 100; no wind above 100 m)
* ``gfs_ghi`` — ``dswrf`` (GFS has no direct/diffuse split and no clear-sky field)
* ``gfs_<column>`` — any column of the files unchanged, e.g. ``gfs_tcc``, ``gfs_t_2m``, ``gfs_gust``

Run assignment: the latest GFS run <= ICON starttime (06 for ICON 06/09, 12 for ICON 12/15, the
12 UTC run of the previous day before 06). With leads to 57 h a 48 h ICON horizon is always covered.

Config (all optional; GFS is only read when ``'gfs' in params.nwp_models`` and a ``gfs_*`` feature is
requested): ``data.gfs_path`` / ``data.gfs_solar_path``, ``params.next_n_grid_gfs`` (default 1).
A configured but missing GFS directory raises instead of silently dropping the features.
"""
from __future__ import annotations

import logging
import os
import re

import numpy as np
import pandas as pd
from geopy.distance import geodesic

logger = logging.getLogger(__name__)

RUNS = ('06', '12')
PREFIX = 'gfs_'
#: columns that hold the mean/sum over the hour ending at forecasttime
ACCUMULATED_COLS = frozenset({'dswrf', 'uswrf', 'dlwrf', 'tp'})
#: derived features -> raw columns
_DERIVED = {'gfs_ghi': ('dswrf',)}
_STEM_RE = re.compile(r'^(-?\d+)_(\d+)_(-?\d+)_(\d+)_(wind_sl|solar_sl)$')
_WS_RE = re.compile(r'^gfs_wind_speed_h(\d+)$')


def requested(features: dict | None) -> list[str]:
    """``gfs_*`` entries of the known/observed features, in order."""
    if not features:
        return []
    feats = (features.get('known') or []) + (features.get('observed') or [])
    return list(dict.fromkeys(f for f in feats if f.startswith(PREFIX)))


def enabled(params_cfg: dict, features: dict | None) -> bool:
    return 'gfs' in (params_cfg.get('nwp_models') or []) and bool(requested(features))


def raw_columns(feats: list[str]) -> list[str]:
    """Raw file columns needed for the requested ``gfs_*`` features."""
    out = []
    for f in feats:
        m = _WS_RE.match(f)
        if m:
            out += [f'u_wind{m.group(1)}m', f'v_wind{m.group(1)}m']
        elif f in _DERIVED:
            out += list(_DERIVED[f])
        else:
            out.append(f[len(PREFIX):])
    return list(dict.fromkeys(out))


def run_start(icon_starttime: pd.Series) -> pd.Series:
    """Latest GFS run (06/12 UTC) <= ICON starttime."""
    st = pd.to_datetime(icon_starttime, utc=True)
    day, hour = st.dt.normalize(), st.dt.hour
    out = day + pd.to_timedelta(np.where(hour >= 12, 12, 6), unit='h')
    return out.where(hour >= 6, day - pd.Timedelta(hours=12))


def site_points(root: str, site_dir: str, suffix: str, lat: float, lon: float, k: int,
                runs=RUNS) -> list[list[str]]:
    """Files of the k nearest grid points (both runs per point), nearest first."""
    by_point: dict[tuple, list[str]] = {}
    for run in runs:
        d = os.path.join(root, run, site_dir)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            m = _STEM_RE.match(fname[:-len('.parquet')]) if fname.endswith(suffix) else None
            if m:
                key = (float(f'{m.group(1)}.{m.group(2)}'), float(f'{m.group(3)}.{m.group(4)}'))
                by_point.setdefault(key, []).append(os.path.join(d, fname))
    if not by_point:
        raise FileNotFoundError(f'no GFS files *{suffix} for {site_dir} under {root}/{{{",".join(runs)}}}')
    ranked = sorted(by_point.items(), key=lambda kv: geodesic((lat, lon), kv[0]).kilometers)
    return [paths for _, paths in ranked[:max(1, k)]]


def load_points(root: str, site_dir: str, suffix: str, lat: float, lon: float, feats: list[str], k: int,
                lower=None, upper=None) -> pd.DataFrame:
    """Wide frame (starttime, forecasttime, <feature>_<rank>) of the k nearest GFS grid points."""
    cols = raw_columns(feats)
    merged = None
    for rank, paths in enumerate(site_points(root, site_dir, suffix, lat, lon, k), start=1):
        df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        df['starttime'] = pd.to_datetime(df['starttime'], utc=True)
        if lower is not None:
            df = df[df['starttime'] >= lower]
        if upper is not None:
            df = df[df['starttime'] <= upper]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f'{site_dir}: GFS files {paths[0]} lack columns {missing}')
        df = df.drop_duplicates(subset=['starttime', 'forecasttime'], keep='last')
        out = df[['starttime', 'forecasttime']].copy()
        out['forecasttime'] = out['forecasttime'].astype(int)
        for f in feats:
            m = _WS_RE.match(f)
            if m:
                h = m.group(1)
                val = np.hypot(df[f'u_wind{h}m'], df[f'v_wind{h}m'])
            elif f == 'gfs_ghi':
                val = df['dswrf'].clip(lower=0)
            else:
                val = df[f[len(PREFIX):]]
            out[f'{f}_{rank}'] = val.to_numpy()
        merged = out if merged is None else merged.merge(out, on=['starttime', 'forecasttime'], how='outer')
    return merged.sort_values(['starttime', 'forecasttime']).reset_index(drop=True)


def is_accumulated(feat: str) -> bool:
    return feat in _DERIVED or feat[len(PREFIX):] in ACCUMULATED_COLS


def merge(df: pd.DataFrame, gfs: pd.DataFrame, feats: list[str], timestamp_col: str = 'timestamp',
          left_labelled: bool = True) -> pd.DataFrame:
    """Attach the GFS columns to the ICON frame by (GFS run, GFS lead).

    left_labelled (PV path): a step labelled T describes [T, T + freq); hourly means ending at the
    lead use lead = floor(h) + 1, instantaneous fields lead = floor(h), h = hours from the GFS run
    to T. Sub-hourly steps keep the hourly value. left_labelled=False (wind path): ICON values are
    instantaneous at T, all GFS fields use lead = h (whole hours).
    """
    out = df.reset_index(drop=True).copy()
    st = run_start(out['starttime'])
    out['_gfs_st'] = st
    offset_min = ((pd.to_datetime(out[timestamp_col], utc=True) - st).dt.total_seconds() / 60).round()
    hours = np.floor_divide(offset_min.to_numpy().astype('int64'), 60)
    acc = [f for f in feats if is_accumulated(f)] if left_labelled else []
    inst = [f for f in feats if f not in acc]
    part = gfs.rename(columns={'starttime': '_gfs_st', 'forecasttime': '_gfs_ft'})
    part['_gfs_st'] = pd.to_datetime(part['_gfs_st'], utc=True)
    for fs, lead in ((acc, hours + 1), (inst, hours)):
        cols = [c for c in gfs.columns if any(c.startswith(f + '_') and c[len(f) + 1:].isdigit() for f in fs)]
        if not cols:
            continue
        # the right side is unique on the keys, so the left join keeps row count and order
        out['_gfs_ft'] = lead.astype('int64')
        out = out.merge(part[['_gfs_st', '_gfs_ft'] + cols], on=['_gfs_st', '_gfs_ft'], how='left')
        out = out.drop(columns=['_gfs_ft'])
    return out.drop(columns=['_gfs_st'])
