"""utils/gfs.py: run assignment, point loading from the per-site run store, lead mapping."""
import numpy as np
import pandas as pd
import pytest

from utils import gfs


def _write_point(root, run, site_dir, key, suffix, starts, n_lead=58, base=0.0):
    """Point file whose values encode (run start hour, lead): value = base + 100*hour + lead."""
    rows = []
    for st in starts:
        for ft in range(n_lead):
            v = base + 100 * st.hour + ft
            rows.append({'starttime': st, 'forecasttime': ft, 'u_wind100m': 3.0, 'v_wind100m': 4.0,
                         'temp2m': v, 'dswrf': np.nan if ft == 0 else v, 'tcc': v})
    d = root / run / site_dir
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / f'{key}{suffix}', index=False)


def test_run_start_takes_latest_run_not_after_icon_start():
    st = pd.Series(pd.to_datetime(['2024-01-02 06:00', '2024-01-02 09:00', '2024-01-02 12:00',
                                   '2024-01-02 15:00', '2024-01-02 03:00'], utc=True))
    got = gfs.run_start(st)
    want = pd.to_datetime(['2024-01-02 06:00', '2024-01-02 06:00', '2024-01-02 12:00',
                           '2024-01-02 12:00', '2024-01-01 12:00'], utc=True)
    assert list(got) == list(want)


def test_raw_columns_and_requested():
    feats = {'known': ['wind_speed_h127', 'gfs_wind_speed_h100', 'gfs_tcc'], 'observed': ['gfs_ghi']}
    assert gfs.requested(feats) == ['gfs_wind_speed_h100', 'gfs_tcc', 'gfs_ghi']
    assert gfs.raw_columns(gfs.requested(feats)) == ['u_wind100m', 'v_wind100m', 'tcc', 'dswrf']
    assert not gfs.enabled({'nwp_models': ['icon-d2']}, feats)
    assert gfs.enabled({'nwp_models': ['icon-d2', 'gfs']}, feats)


def test_load_points_ranks_by_distance_and_pools_runs(tmp_path):
    starts06 = pd.to_datetime(['2024-01-02 06:00'], utc=True)
    starts12 = pd.to_datetime(['2024-01-02 12:00'], utc=True)
    for run, st in (('06', starts06), ('12', starts12)):
        _write_point(tmp_path, run, 'park_A', '50_0000_10_0000', '_wind_sl.parquet', st, base=0.0)
        _write_point(tmp_path, run, 'park_A', '50_2500_10_0000', '_wind_sl.parquet', st, base=1e4)
    df = gfs.load_points(str(tmp_path), 'park_A', '_wind_sl.parquet', 50.2, 10.0,
                         ['gfs_wind_speed_h100', 'gfs_temp2m'], k=2)
    assert df.starttime.nunique() == 2 and len(df) == 2 * 58
    assert np.allclose(df['gfs_wind_speed_h100_1'], 5.0)
    # rank 1 is the point at 50.25 (nearer to 50.2), its values carry base 1e4
    r = df[df.starttime == starts12[0]].set_index('forecasttime')
    assert r.loc[7, 'gfs_temp2m_1'] == 1e4 + 1200 + 7 and r.loc[7, 'gfs_temp2m_2'] == 1200 + 7
    with pytest.raises(FileNotFoundError):
        gfs.load_points(str(tmp_path), 'park_B', '_wind_sl.parquet', 50.2, 10.0, ['gfs_temp2m'], k=1)


def test_merge_wind_uses_whole_hour_offset_from_gfs_run(tmp_path):
    _write_point(tmp_path, '06', 'park_A', '50_0000_10_0000', '_wind_sl.parquet',
                 pd.to_datetime(['2024-01-02 06:00'], utc=True))
    g = gfs.load_points(str(tmp_path), 'park_A', '_wind_sl.parquet', 50.0, 10.0, ['gfs_temp2m'], k=1)
    icon_st = pd.Timestamp('2024-01-02 09:00', tz='UTC')
    icon = pd.DataFrame({'starttime': icon_st, 'forecasttime': np.arange(1, 49)})
    icon['timestamp'] = icon.starttime + pd.to_timedelta(icon.forecasttime, unit='h')
    out = gfs.merge(icon, g, ['gfs_temp2m'], left_labelled=False)
    assert len(out) == 48 and out.gfs_temp2m_1.notna().all()
    # ICON 09 + 1 h = GFS 06 + 4 h; ICON 09 + 48 h = GFS 06 + 51 h
    assert out.gfs_temp2m_1.iloc[0] == 600 + 4 and out.gfs_temp2m_1.iloc[-1] == 600 + 51


def test_merge_pv_accumulated_ends_inside_step_instantaneous_at_start(tmp_path):
    _write_point(tmp_path, '12', 'park_A', '50_0000_10_0000', '_solar_sl.parquet',
                 pd.to_datetime(['2024-01-02 12:00'], utc=True))
    feats = ['gfs_ghi', 'gfs_tcc']
    g = gfs.load_points(str(tmp_path), 'park_A', '_solar_sl.parquet', 50.0, 10.0, feats, k=1)
    icon_st = pd.Timestamp('2024-01-02 15:00', tz='UTC')
    steps = np.arange(96)
    icon = pd.DataFrame({'starttime': icon_st, 'forecasttime': steps,
                         'timestamp': icon_st + pd.to_timedelta(30 * steps, unit='m')})
    out = gfs.merge(icon, g, feats)
    # step 0 = [15:00, 15:30): GFS 12 + 3 h; mean over (15, 16] -> lead 4, cloud cover at 15:00 -> lead 3
    assert out.gfs_ghi_1.iloc[0] == 1200 + 4 and out.gfs_tcc_1.iloc[0] == 1200 + 3
    assert out.gfs_ghi_1.iloc[1] == 1200 + 4                       # sub-hourly step keeps the hourly value
    assert out.gfs_ghi_1.iloc[-1] == 1200 + 51 and out[['gfs_ghi_1', 'gfs_tcc_1']].notna().all().all()
