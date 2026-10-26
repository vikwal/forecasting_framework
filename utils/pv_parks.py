"""
pv_parks.py — PV parks (synthetic release parks_pv_v1_30min, Trianel twins and meters) in the solar
TFT path. ``data.use_case: 'pv_park'``.

Per park the frame has the solar contract (MultiIndex starttime / forecasttime / timestamp, one row per
step of every ICON-D2 run, ``df.attrs['target_cols']``):

* target ``power``: ``power_park`` of ``synth_<id>.parquet`` (``data.target_source: synthetic``) or the
  metered power of ``meter_<id>.parquet`` (``meter``), normalised by the DC capacity (``parks.csv``,
  ``capacity_kwp_dc`` or ``kwp_dc``) — one sample per park whatever its number of orientations
* ICON-D2 single-level forecasts from the per-site store ``<nwp_path>/SL/<HH>/<prefix><id>/`` written by
  ``NWP-points`` (``points.cli icon-sl-*``): nearest grid point(s) by filename (lat-first stems), loaded with
  :func:`utils.solar.load_sl_grid_point` on ``data.freq`` (radiation = 15-min means ending at the lead,
  instantaneous fields at the interval start, both mapped to left-labelled steps)
* ECMWF HRES solar from ``<ecmwf_solar_path>/{00,12}/<prefix><id>/`` (``points.cli ecmwf --kind SOLAR``),
  assigned per ICON run as in the solar path (latest ECMWF run <= ICON start)
* geometry-invariant irradiance for a central model: the plane-of-array irradiance of the park's arrays
  (``pv_arrays.csv`` or the twin manifest), share-weighted, Hay-Davies via pvlib, for ICON (``gti_nwp``),
  ECMWF (``ecmwf_gti``) and the clear sky (``gti_clearsky``), plus the clear-sky indices ``kc_gti_nwp``,
  ``ecmwf_kc_gti`` and ``kt_nwp``. A park of several orientations is one sample with the weighted
  irradiance (the POA sum of the chain itself), its target is never split.
* solar geometry and cyclic time features of :func:`utils.solar.solar_geometry`
* statics: capacity_kwp, tilt_eff, azimuth_eff_sin/cos, latitude, longitude, altitude
"""
from __future__ import annotations

import json
import logging
import os
import re

import numpy as np
import pandas as pd
import pvlib
from geopy.distance import geodesic

from . import solar as _solar
from . import gfs as _gfs
from . import solar_ecmwf as _se

logger = logging.getLogger(__name__)

_SL_STEM_RE = re.compile(r'^(-?\d+)_(\d+)_(-?\d+)_(\d+)_SL$')
_SOLAR_STEM_RE = re.compile(r'^(-?\d+)_(\d+)_(-?\d+)_(\d+)_solar_sl$')
#: park-level irradiance features computed here (no grid-point suffix)
PARK_FEATURES = ('gti_nwp', 'gti_clearsky', 'kc_gti_nwp', 'kt_nwp', 'ecmwf_gti', 'ecmwf_kc_gti')


# ---------------------------------------------------------------------------
# park metadata
# ---------------------------------------------------------------------------

def _park_id_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split('_', 1)[1] if stem.startswith(('synth_', 'meter_')) else stem


METER_SUFFIX = '_meter'


def base_id(park_id: str) -> str:
    """Park id without the ``_meter`` suffix that marks the metered copy of a twin park in a config
    (so that ``pv1`` (synthetic twin, training) and ``pv1_meter`` (Zähler, holdout) can coexist)."""
    return park_id[:-len(METER_SUFFIX)] if park_id.endswith(METER_SUFFIX) else park_id


def park_source(data_cfg: dict, park_id: str) -> tuple[str, str]:
    """(directory, target source) of a park: ``data.path`` with ``data.target_source`` by default;
    parks listed in ``data.meter_files`` come from ``data.meter_path`` with the metered target and parks in
    ``data.twin_files`` from ``data.twin_path`` with the synthetic target - so one config can pretrain on
    the release plus the synthetic twins and evaluate on the Trianel meters (holdout_files)."""
    if park_id in (data_cfg.get('meter_files') or []):
        return data_cfg['meter_path'], 'meter'
    if park_id in (data_cfg.get('twin_files') or []):
        return data_cfg['twin_path'], 'synthetic'
    return data_cfg['path'], data_cfg.get('target_source', 'synthetic')


def resolve_path(data_cfg: dict, park_id: str) -> str:
    """File that identifies the park for the router (synth_<id>.parquet of its source directory)."""
    return os.path.join(park_source(data_cfg, park_id)[0], f'synth_{base_id(park_id)}.parquet')


def park_meta(data_cfg: dict, park_id: str) -> dict:
    """lat/lon/altitude, DC capacity [W] and the arrays [(tilt, azimuth, share)] of one park from the
    release tables (parks.csv, sites.csv, pv_arrays.csv) or the twin manifest."""
    base = park_source(data_cfg, park_id)[0]
    park_id = base_id(park_id)
    parks = pd.read_csv(data_cfg.get('parks_file') or os.path.join(base, 'parks.csv'), dtype={'park_id': str}).set_index('park_id')
    if park_id not in parks.index:
        raise ValueError(f'park {park_id} not in {base}/parks.csv')
    row = parks.loc[park_id]
    cap_col = next(c for c in ('capacity_kwp_dc', 'kwp_dc', 'capacity_kw') if c in parks.columns)
    cap_w = float(row[cap_col]) * 1000.0
    sites_file = data_cfg.get('sites_file') or next(
        (os.path.join(base, f) for f in ('sites.csv', 'sites_confidential.csv') if os.path.exists(os.path.join(base, f))), None)
    if sites_file is None:
        raise FileNotFoundError(f'no sites.csv / sites_confidential.csv under {base}')
    sites = pd.read_csv(sites_file, dtype=str)
    idc = 'site_id' if 'site_id' in sites.columns else 'park_id'
    s = sites[sites[idc] == park_id]
    if s.empty:
        raise ValueError(f'park {park_id} not in {sites_file}')
    lat, lon = float(s['lat'].iloc[0]), float(s['lon'].iloc[0])
    alt = float(s['altitude'].iloc[0]) if 'altitude' in s.columns and pd.notna(s['altitude'].iloc[0]) else 0.0
    arrays_file = data_cfg.get('arrays_file') or os.path.join(base, 'pv_arrays.csv')
    if os.path.exists(arrays_file):
        a = pd.read_csv(arrays_file, dtype={'park_id': str})
        a = a[a.park_id == park_id]
        arrays = [(float(r.tilt), float(r.azimuth), float(r.share)) for r in a.itertuples()]
    else:
        man = json.load(open(os.path.join(base, f'manifest_{park_id}.json')))
        arrays = [(float(x['tilt']), float(x['azimuth']), float(x['share'])) for x in man['arrays']]
    if not arrays:
        raise ValueError(f'park {park_id}: no arrays')
    tot = sum(w for _, _, w in arrays)
    arrays = [(t, a, w / tot) for t, a, w in arrays]
    # effective orientation = share-weighted mean normal
    n = np.array([[np.sin(np.radians(t)) * np.sin(np.radians(a)), np.sin(np.radians(t)) * np.cos(np.radians(a)), np.cos(np.radians(t))] for t, a, _ in arrays])
    m = (n * np.array([w for _, _, w in arrays])[:, None]).sum(axis=0)
    tilt_eff = float(np.degrees(np.arctan2(np.hypot(m[0], m[1]), m[2])))
    az_eff = float(np.degrees(np.arctan2(m[0], m[1])) % 360.0)
    return {'lat': lat, 'lon': lon, 'altitude': alt, 'capacity_w': cap_w, 'arrays': arrays,
            'tilt_eff': tilt_eff, 'azimuth_eff': az_eff}


# ---------------------------------------------------------------------------
# target
# ---------------------------------------------------------------------------

def load_target(data_cfg: dict, park_id: str, capacity_w: float, freq: str,
                lower: pd.Timestamp | None, upper: pd.Timestamp | None) -> pd.DataFrame:
    """``power`` (fraction of the DC capacity) on left-labelled steps of ``freq``."""
    base, src = park_source(data_cfg, park_id)
    park_id = base_id(park_id)
    if src == 'synthetic':
        df = pd.read_parquet(os.path.join(base, f'synth_{park_id}.parquet'), columns=['power_park'])
        s = df['power_park']
    elif src == 'meter':
        df = pd.read_parquet(os.path.join(base, f'meter_{park_id}.parquet'))
        s = df['power_w']
    else:
        raise ValueError(f"data.target_source must be 'synthetic' or 'meter', got {src!r}")
    s.index = pd.to_datetime(s.index, utc=True)
    s = s.resample(pd.Timedelta(freq), closed='left', label='left').mean()
    if lower is not None:
        s = s[s.index >= lower]
    if upper is not None:
        s = s[s.index <= upper]
    out = pd.DataFrame({'power': s / capacity_w})
    out.index.name = 'timestamp'
    return out


# ---------------------------------------------------------------------------
# ICON-D2 single level, per-site store
# ---------------------------------------------------------------------------

def _site_points(site_dir: str, stem_re: re.Pattern, suffix: str) -> list[tuple[str, float, float]]:
    out = []
    if not os.path.isdir(site_dir):
        return out
    for f in os.listdir(site_dir):
        if not f.endswith(suffix):
            continue
        m = stem_re.match(f[:-len('.parquet')])
        if m:
            la, lf, lo, lof = m.groups()
            out.append((f, float(f'{la}.{lf}'), float(f'{lo}.{lof}')))
    return out


def load_icon_sl(data_cfg: dict, params_cfg: dict, park_id: str, lat: float, lon: float, features: list[str],
                 step_h: float, n_leads: int, lower, upper) -> tuple[pd.DataFrame, str | None]:
    """All ICON-D2 runs of the configured forecast hours at the k nearest grid points of the site directory;
    columns ``<feature>_<rank>``, starttime, forecasttime (lead index), timestamp (left label)."""
    prefix = data_cfg.get('nwp_site_prefix', 'park_')
    hours = [str(h).zfill(2) for h in data_cfg.get('forecast_hours', ['09'])]
    k = int(params_cfg.get('next_n_grid_points', 1))
    per_rank: dict[int, list[pd.DataFrame]] = {}
    for hh in hours:
        site_dir = os.path.join(data_cfg['nwp_path'], 'SL', hh, f'{prefix}{base_id(park_id)}')
        pts = _site_points(site_dir, _SL_STEM_RE, '_SL.parquet')
        if not pts:
            logger.warning('park %s: no ICON-D2 SL files under %s', park_id, site_dir)
            continue
        pts.sort(key=lambda p: geodesic((lat, lon), (p[1], p[2])).kilometers)
        for rank, (f, _la, _lo) in enumerate(pts[:k], start=1):
            frame = _solar.load_sl_grid_point(os.path.join(site_dir, f), features, step_h, n_leads, lower, upper,
                                              params_cfg.get('sub_hourly_fill', 'ffill'))
            if not frame.empty:
                per_rank.setdefault(rank, []).append(frame)
    if not per_rank:
        raise FileNotFoundError(f'park {park_id}: no ICON-D2 SL data for hours {hours}')
    merged = None
    for rank in sorted(per_rank):
        frame = pd.concat(per_rank[rank], ignore_index=True).drop_duplicates(subset=['starttime', 'forecasttime'], keep='last')
        feat_cols = [c for c in frame.columns if c not in ('starttime', 'forecasttime')]
        frame = frame.rename(columns={c: f'{c}_{rank}' for c in feat_cols})
        merged = frame if merged is None else merged.merge(frame, on=['starttime', 'forecasttime'], how='outer')
    step_min = int(round(step_h * 60))
    merged['timestamp'] = merged['starttime'] + pd.to_timedelta(merged['forecasttime'].astype('int64') * step_min, unit='m')
    return merged.sort_values(['starttime', 'forecasttime']).reset_index(drop=True), '1'


# ---------------------------------------------------------------------------
# ECMWF solar, per-site run store
# ---------------------------------------------------------------------------

def load_ecmwf_solar_site(ecmwf_path: str, prefix: str, park_id: str, lat: float, lon: float, features: list[str],
                          k: int, lower, upper, runs=('00', '12')) -> pd.DataFrame:
    """Same frame as :func:`utils.solar_ecmwf.load_ecmwf_solar_for_station`, read from
    ``<ecmwf_path>/<run>/<prefix><id>/<lat>_<lon>_solar_sl.parquet`` (both runs concatenated per point)."""
    raw_cols = _se._required_ecmwf_columns(features)
    by_point: dict[tuple, list[str]] = {}
    for run in runs:
        d = os.path.join(ecmwf_path, run, f'{prefix}{base_id(park_id)}')
        for f, la, lo in _site_points(d, _SOLAR_STEM_RE, '_solar_sl.parquet'):
            by_point.setdefault((la, lo), []).append(os.path.join(d, f))
    if not by_point:
        raise FileNotFoundError(f'park {park_id}: no ECMWF solar files under {ecmwf_path}')
    ranked = sorted(by_point.items(), key=lambda kv: geodesic((lat, lon), kv[0]).kilometers)
    merged = None
    for rank, (_c, paths) in enumerate(ranked[:k], start=1):
        df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        df['starttime'] = pd.to_datetime(df['starttime'], utc=True)
        if lower is not None:
            df = df[df['starttime'] >= lower]
        if upper is not None:
            df = df[df['starttime'] <= upper]
        if df.empty:
            continue
        missing = [c for c in raw_cols if c not in df.columns]
        if missing:
            logger.warning('park %s: ECMWF solar without columns %s', park_id, missing)
        df = df.drop_duplicates(subset=['starttime', 'forecasttime'], keep='last')
        df['forecasttime'] = df['forecasttime'].astype(int)
        df = _se._add_ecmwf_derived(df, features)
        present = [f for f in features + _se.geometry_intermediates(features) if f in df.columns]
        if not present:
            continue
        frame = df[['starttime', 'forecasttime'] + present].rename(columns={c: f'{c}_{rank}' for c in present})
        merged = frame if merged is None else merged.merge(frame, on=['starttime', 'forecasttime'], how='outer')
    if merged is None:
        raise ValueError(f'park {park_id}: ECMWF solar files empty in the window')
    return merged.sort_values(['starttime', 'forecasttime']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# plane-of-array irradiance of the park
# ---------------------------------------------------------------------------

def park_gti(arrays: list, zenith: pd.Series, azimuth: pd.Series, ghi: pd.Series, dhi: pd.Series, dni: pd.Series,
             dni_extra: pd.Series, albedo: float = 0.2) -> pd.Series:
    """Share-weighted plane-of-array global irradiance (Hay-Davies) of the park's arrays."""
    out = pd.Series(0.0, index=ghi.index)
    z = zenith.clip(upper=90.0)
    for tilt, az, share in arrays:
        poa = pvlib.irradiance.get_total_irradiance(tilt, az, z, azimuth, dni.fillna(0.0), ghi.fillna(0.0), dhi.fillna(0.0),
                                                    dni_extra=dni_extra, albedo=albedo, model='haydavies')
        out = out + share * poa['poa_global'].fillna(0.0).clip(lower=0.0)
    return out.where(ghi.notna())


def _add_park_irradiance(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """gti_nwp / gti_clearsky / kc_gti_nwp / kt_nwp (ICON rank 1) and ecmwf_gti / ecmwf_kc_gti where present."""
    zen, az = df['solar_zenith'], np.degrees(np.arctan2(df['solar_azimuth_sin'], df['solar_azimuth_cos'])) % 360.0
    df['gti_clearsky'] = park_gti(meta['arrays'], zen, az, df['ghi_clearsky'], df['dhi_clearsky'], df['dni_clearsky'], df['dni_extra'])
    if 'ghi_nwp_1' in df and 'dhi_nwp_1' in df:
        d = _solar.derive_irradiance_components(df['ghi_nwp_1'], df['dhi_nwp_1'], zen, df['dni_clearsky'])
        df['gti_nwp'] = park_gti(meta['arrays'], zen, az, df['ghi_nwp_1'], df['dhi_nwp_1'], d['dni'], df['dni_extra'])
        df['kc_gti_nwp'] = _solar.clearsky_index(df['gti_nwp'], df['gti_clearsky'])
        df['kt_nwp'] = _solar.clearsky_index(df['ghi_nwp_1'], df['ghi_clearsky'])
    if 'ecmwf_ghi_1' in df and 'ecmwf_dhi_1' in df:
        d = _solar.derive_irradiance_components(df['ecmwf_ghi_1'], df['ecmwf_dhi_1'], zen, df['dni_clearsky'])
        df['ecmwf_gti'] = park_gti(meta['arrays'], zen, az, df['ecmwf_ghi_1'], df['ecmwf_dhi_1'], d['dni'], df['dni_extra'])
        df['ecmwf_kc_gti'] = _solar.clearsky_index(df['ecmwf_gti'], df['gti_clearsky'])
    return df


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def preprocess_pv_park(path: str, config: dict, freq: str = '30min', features: dict | None = None,
                       park_id: str | None = None) -> pd.DataFrame:
    """``park_id`` (the config's file id, e.g. ``pv1_meter``) overrides the id derived from ``path``, which
    cannot carry the ``_meter`` suffix because the router checks the synthetic file for existence."""
    data_cfg, params_cfg = config['data'], config.get('params', {})
    features = features or {'known': [], 'observed': [], 'static': []}
    park_id = park_id or _park_id_from_path(path)
    meta = park_meta(data_cfg, park_id)
    step_h, n_leads = _solar.resolve_freq(freq, warn=False)

    lower = pd.Timestamp(data_cfg['train_start'], tz='UTC') - pd.Timedelta(days=7) if data_cfg.get('train_start') else None
    _up = data_cfg.get('data_cutoff') or data_cfg.get('test_end')
    upper = pd.Timestamp(_up, tz='UTC') if _up else None

    target = load_target(data_cfg, park_id, meta['capacity_w'], freq, lower, upper)

    requested = list(dict.fromkeys((features.get('known') or []) + (features.get('observed') or [])))
    icon_feats = [f for f in (params_cfg.get('icond2_features') or ['ghi_nwp', 'dhi_nwp', 'bhi_nwp', 't_2m', 'clct', 'h_snow'])]
    if any(f in requested for f in ('gti_nwp', 'kc_gti_nwp', 'kt_nwp')):
        icon_feats = list(dict.fromkeys(icon_feats + ['ghi_nwp', 'dhi_nwp']))
    df, nearest_label = load_icon_sl(data_cfg, params_cfg, park_id, meta['lat'], meta['lon'], icon_feats, step_h, n_leads, lower, upper)

    feat_cols = [c for c in df.columns if c not in ('starttime', 'forecasttime', 'timestamp')]
    counts = df.dropna(subset=feat_cols).groupby('starttime').size()
    valid = counts[counts == n_leads].index
    n_drop = df['starttime'].nunique() - len(valid)
    if n_drop:
        logger.info('park %s: %d of %d ICON runs incomplete (< %d leads at %s) and dropped', park_id, n_drop, df['starttime'].nunique(), n_leads, freq)
    df = df[df['starttime'].isin(valid)].copy()
    if df.empty:
        raise ValueError(f'park {park_id}: no complete ICON-D2 runs')

    df = df.merge(target, left_on='timestamp', right_index=True, how='left')
    geo = _solar.solar_geometry(pd.DatetimeIndex(df['timestamp'].unique()), meta['lat'], meta['lon'], meta['altitude'], freq=freq)
    df = df.merge(geo, left_on='timestamp', right_index=True, how='left')
    _solar._add_nwp_geometry_features(df, icon_feats)

    ecmwf_path = data_cfg.get('ecmwf_solar_path')
    ecmwf_feats = [str(f) for f in (params_cfg.get('ecmwf_features') or [])]
    if ecmwf_path and ecmwf_feats:
        need = list(dict.fromkeys(ecmwf_feats + (['ecmwf_ghi', 'ecmwf_dhi'] if any(f in requested for f in ('ecmwf_gti', 'ecmwf_kc_gti')) else [])))
        try:
            ec = load_ecmwf_solar_site(ecmwf_path, data_cfg.get('nwp_site_prefix', 'park_'), park_id, meta['lat'], meta['lon'], need,
                                       int(params_cfg.get('next_n_grid_points_ecmwf', 1)), lower, upper)
            df = _se.merge_ecmwf(df, ec, need, int(round(step_h * 60)))
            df = _se.add_ecmwf_geometry_features(df, need)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning('park %s: ECMWF solar not loaded (%s)', park_id, exc)

    # GFS (utils/gfs.py): <gfs_solar_path>/{06,12}/<prefix><id>/<lat>_<lon>_solar_sl.parquet; a configured
    # but missing source raises, unlike ECMWF above
    if _gfs.enabled(params_cfg, features):
        gfs_feats = _gfs.requested(features)
        gfs_path = data_cfg.get('gfs_solar_path')
        if not gfs_path:
            raise ValueError('gfs_* features requested but data.gfs_solar_path is not set')
        g = _gfs.load_points(gfs_path, f"{data_cfg.get('nwp_site_prefix', 'park_')}{base_id(park_id)}", '_solar_sl.parquet',
                             meta['lat'], meta['lon'], gfs_feats, int(params_cfg.get('next_n_grid_gfs', 1)),
                             None if lower is None else lower - pd.Timedelta(days=1), upper)
        df = _gfs.merge(df, g, gfs_feats)

    df = _add_park_irradiance(df, meta)

    df = df.set_index(['starttime', 'forecasttime', 'timestamp']).sort_index()
    static_data = {'latitude': meta['lat'], 'longitude': meta['lon'], 'altitude': meta['altitude'],
                   'capacity_kwp': meta['capacity_w'] / 1000.0, 'tilt_eff': meta['tilt_eff'],
                   'azimuth_eff_sin': float(np.sin(np.radians(meta['azimuth_eff']))), 'azimuth_eff_cos': float(np.cos(np.radians(meta['azimuth_eff'])))}
    static_features = features.get('static', []) or []
    from .preprocessing import static_categories
    cats = static_categories(config)
    if static_features and 'park_id' in cats:
        if park_id not in cats['park_id']:
            raise ValueError(f'park {park_id} missing in the park_id categories')
        static_data['park_id'] = cats['park_id'].index(park_id)
    for feat in static_features:
        if feat in static_data:
            df[feat] = static_data[feat]
        else:
            logger.warning('park %s: static feature %s not available', park_id, feat)

    df = _solar._select_columns(df, features, ['power'], static_features, params_cfg)
    before = len(df)
    df = df.dropna()
    if before and len(df) < before:
        logger.debug('park %s: %d of %d rows dropped as NaN', park_id, before - len(df), before)
    run_lengths = df.groupby(level='starttime').size()
    if len(run_lengths):
        expected = int(run_lengths.value_counts().idxmax())
        df = df[df.index.get_level_values('starttime').isin(run_lengths[run_lengths == expected].index)]
    if df.empty:
        raise ValueError(f'park {park_id}: nothing left after preprocessing')
    df.attrs['nwp_nearest_label'] = nearest_label
    df.attrs['target_cols'] = ['power']
    return df.sort_index()


def station_folds(sites: pd.DataFrame, n_folds: int = 3, seed: int = 42, lat_bins: int = 3, lon_bins: int = 2) -> list[list[str]]:
    """Disjoint station folds for the HPO (hpo.station_folds), stratified over a lat/lon grid like the CAMS
    correction model (cams_bc.splits): within every grid cell the stations are permuted with the seed and dealt
    round-robin to the folds, so each fold covers north/south and east/west alike. `sites` needs columns
    site_id, lat, lon. Returns n_folds sorted id lists."""
    t = sites[['site_id', 'lat', 'lon']].drop_duplicates('site_id').reset_index(drop=True)
    cell = pd.qcut(t['lat'], lat_bins, labels=False).astype(str) + '_' + pd.qcut(t['lon'], lon_bins, labels=False).astype(str)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for _, grp in t.groupby(cell):
        order = grp['site_id'].to_numpy()[rng.permutation(len(grp))]
        start = int(rng.integers(n_folds))
        for i, s in enumerate(order):
            folds[(i + start) % n_folds].append(str(s))
    return [sorted(f) for f in folds]
