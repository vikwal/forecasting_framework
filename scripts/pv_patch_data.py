"""ICON-D2 16x16 patches around a PV park as extra known columns of the park frame (Viktor 2026-11-12).

The patch store ``<patch_path>/SL/09/<prefix><id>/<lat>_<lon>_SL.parquet`` plus ``points.csv`` (row, col per file;
row 0 south, col 0 west, nearest point at (8, 8)) comes from ``NWP/DWD/zarr/icon_export.py --patch 16``.
Per grid point the plane-of-array irradiance of the park (``gti``, same transposition as ``gti_nwp``) and the total
cloud cover (``clct``) are added as columns ``p<channel>_<row*16+col>``; ``stat_*`` columns hold patch summaries
(mean and std over the central 3x3, 7x7 and the full patch) for the grid-feature baseline.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import pv_parks, solar as _solar

CHANNELS = ('gti', 'clct')
SIZE = 16
WINDOWS = {'c3': (7, 10), 'c7': (5, 12), 'all': (0, 16)}            # row/col slices around the centre (8, 8)


def patch_cols(channels=CHANNELS, size=SIZE):
    return [f'p{c}_{i}' for c in channels for i in range(size * size)]


def stat_cols(channels=CHANNELS):
    return [f'stat_{c}_{w}_{s}' for c in channels for w in WINDOWS for s in ('mean', 'std')]


def _load_point(args):
    path, step_h, n_leads, lower, upper = args
    return _solar.load_sl_grid_point(path, ['ghi_nwp', 'dhi_nwp', 'clct'], step_h, n_leads, lower, upper, 'ffill')


def load_patch(cfg: dict, park_id: str, patch_path: str, procs: int = 32) -> pd.DataFrame:
    """Frame indexed by (starttime, forecasttime) with columns pgti_*, pclct_* (row-major), runs with a complete
    patch only."""
    data_cfg = cfg['data']
    meta = pv_parks.park_meta(data_cfg, park_id)
    step_h, n_leads = _solar.resolve_freq(data_cfg['freq'], warn=False)
    lower = pd.Timestamp(data_cfg['train_start'], tz='UTC') - pd.Timedelta(days=7)
    upper = pd.Timestamp(data_cfg.get('data_cutoff') or data_cfg['test_end'], tz='UTC')
    site = Path(patch_path) / 'SL' / '09' / f"{data_cfg.get('nwp_site_prefix', 'park_')}{pv_parks.base_id(park_id)}"
    pts = pd.read_csv(site / 'points.csv')
    pts = pts.sort_values(['row', 'col']).reset_index(drop=True)
    if len(pts) != SIZE * SIZE:
        raise ValueError(f'{park_id}: {len(pts)} patch points, expected {SIZE * SIZE}')
    files = [site / f'{k}_SL.parquet' for k in pts['point_key']]
    with ProcessPoolExecutor(procs) as ex:
        frames = list(ex.map(_load_point, [(str(f), step_h, n_leads, lower, upper) for f in files]))
    key = ['starttime', 'forecasttime']
    base = frames[0][key].copy()
    step_min = int(round(step_h * 60))
    base['timestamp'] = base['starttime'] + pd.to_timedelta(base['forecasttime'].astype('int64') * step_min, unit='m')
    geo = _solar.solar_geometry(pd.DatetimeIndex(base['timestamp'].unique()), meta['lat'], meta['lon'], meta['altitude'],
                                freq=data_cfg['freq'])
    geo = base.merge(geo, left_on='timestamp', right_index=True, how='left')
    out = {}
    for i, fr in enumerate(frames):
        fr = base[key].merge(fr, on=key, how='left')
        d = geo.copy()
        d['ghi_nwp_1'], d['dhi_nwp_1'] = fr['ghi_nwp'].values, fr['dhi_nwp'].values
        d = pv_parks._add_park_irradiance(d, meta)
        out[f'pgti_{i}'] = d['gti_nwp'].values
        out[f'pclct_{i}'] = fr['clct'].values
    df = pd.concat([base[key], pd.DataFrame(out, index=base.index)[patch_cols()]], axis=1)
    df = df.dropna()
    full = df.groupby('starttime').size()
    df = df[df['starttime'].isin(full[full == n_leads].index)]
    return df.set_index(key).sort_index()


def add_stats(patch: pd.DataFrame) -> pd.DataFrame:
    """Patch summaries per step: mean and std of each channel over the central 3x3, 7x7 and the full patch."""
    out = {}
    for c in CHANNELS:
        a = patch[[f'p{c}_{i}' for i in range(SIZE * SIZE)]].to_numpy().reshape(-1, SIZE, SIZE)
        for w, (lo, hi) in WINDOWS.items():
            sub = a[:, lo:hi, lo:hi].reshape(len(a), -1)
            out[f'stat_{c}_{w}_mean'], out[f'stat_{c}_{w}_std'] = sub.mean(1), sub.std(1)
    return pd.DataFrame(out, index=patch.index)


def attach(frame: pd.DataFrame, extra: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Inner join on (starttime, forecasttime); returns the frame and the number of runs lost."""
    runs = frame.index.get_level_values('starttime').nunique()
    idx = frame.index
    joined = frame.reset_index().merge(extra.reset_index(), on=['starttime', 'forecasttime'], how='inner')
    joined = joined.set_index(idx.names).sort_index()
    return joined, runs - joined.index.get_level_values('starttime').nunique()


def extended_scaler(sx: StandardScaler, base_cols: list, frames: list, train_end: pd.Timestamp,
                    shared: dict) -> StandardScaler:
    """sx for the original columns; new columns scaled on the pooled training rows, one mean/std per group in
    ``shared`` (all grid points of a channel share it, so the spatial structure survives scaling) and per column
    otherwise."""
    tr = pd.concat([f[f.index.get_level_values('starttime') <= train_end] for f in frames])
    new = [c for c in frames[0].columns if c not in base_cols and c != 'power']
    mean, var = [], []
    grouped = {c: g for g, cols in shared.items() for c in cols}
    stats = {g: (np.nanmean(tr[cols].to_numpy()), np.nanvar(tr[cols].to_numpy())) for g, cols in shared.items()}
    for c in new:
        m, v = stats[grouped[c]] if c in grouped else (tr[c].mean(), tr[c].var(ddof=0))
        mean.append(m)
        var.append(max(v, 1e-12))
    s = StandardScaler()
    s.mean_ = np.concatenate([sx.mean_, mean])
    s.var_ = np.concatenate([sx.var_, var])
    s.scale_ = np.sqrt(s.var_)
    s.n_features_in_ = len(s.mean_)
    s.n_samples_seen_ = sx.n_samples_seen_
    s._ff_feature_cols = list(base_cols) + new
    return s


def cache_path(root: str, park_id: str) -> str:
    return os.path.join(root, f'patch_{park_id}.parquet')
