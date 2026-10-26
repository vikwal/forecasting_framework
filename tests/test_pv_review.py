"""Review 2026-10-17 of the PV pretraining line: test windows start at data.test_start for sub-hourly NWP
data, wind_speed_nwp survives the TFT feature mapping in the SL path, few-shot window arithmetic."""
import numpy as np
import pandas as pd
import pytest

from utils import preprocessing, finetune


def _nwp_frame(freq: str, steps: int, start='2023-10-01', end='2023-12-31', run_hour=9) -> pd.DataFrame:
    """Daily runs at run_hour UTC with `steps` leads of `freq`; known features k1/k2, target power."""
    runs = pd.date_range(f'{start} {run_hour:02d}:00', f'{end} {run_hour:02d}:00', freq='24h', tz='UTC')
    rows = []
    for st in runs:
        ts = st + pd.to_timedelta(np.arange(steps) * pd.Timedelta(freq))
        rows.append(pd.DataFrame({'starttime': st, 'forecasttime': np.arange(steps), 'timestamp': ts,
                                  'k1': np.sin(np.arange(steps) / 7.0) + st.dayofyear / 400.0,
                                  'k2': np.cos(np.arange(steps) / 5.0),
                                  'power': (np.sin(ts.hour.values / 24.0 * 2 * np.pi) + 1) / 2}))
    return pd.concat(rows).set_index(['starttime', 'forecasttime', 'timestamp']).sort_index()


def _split(df, steps, test_start, train_end):
    return preprocessing.prepare_data_for_tft(
        data=df, history_length=steps, future_horizon=steps, known_future_cols=['k1', 'k2'],
        observed_past_cols=['power'], static_cols=[], step_size=steps, train_frac=1.0, target_col='power',
        train_start=pd.Timestamp('2023-10-01', tz='UTC'), train_end=pd.Timestamp(train_end, tz='UTC'),
        test_start=pd.Timestamp(test_start, tz='UTC'), test_end=pd.Timestamp('2023-12-31 23:00', tz='UTC'))


def test_sub_hourly_test_windows_start_at_test_start():
    """freq 30min, 96 steps: before the fix the split reached 96 h back and the runs of 11-29 and 11-30
    became test windows (93 instead of 91), both inside the training window."""
    df = _nwp_frame('30min', 96)
    res = _split(df, 96, '2023-12-01', '2023-11-30 23:30')
    idx = pd.DatetimeIndex(res['index_test'])
    assert idx.min() == pd.Timestamp('2023-12-01 09:00', tz='UTC')
    assert len(idx) == 31
    assert pd.DatetimeIndex(res['index_train']).max() == pd.Timestamp('2023-11-30 09:00', tz='UTC')
    assert res['X_test']['known'].shape == (31, 192, 2) and res['y_test'].shape == (31, 96)
    assert res['X_test']['observed'].shape[0] == 31


def test_hourly_split_unchanged():
    """freq 1h, 48 steps: the hours-based extension was already right, same windows as before."""
    df = _nwp_frame('1h', 48)
    res = _split(df, 48, '2023-12-01', '2023-11-30 23:00')
    idx = pd.DatetimeIndex(res['index_test'])
    assert idx.min() == pd.Timestamp('2023-12-01 09:00', tz='UTC') and len(idx) == 31


def test_index_step_and_mask_helpers():
    ts = pd.DatetimeIndex(['2023-01-01 00:00', '2023-01-01 00:30', '2023-01-01 01:00'], tz='UTC').repeat(3)
    assert preprocessing._index_step(ts) == pd.Timedelta('30min')
    assert preprocessing._index_step(pd.DatetimeIndex([], tz='UTC')) == pd.Timedelta('1h')
    idx = pd.DatetimeIndex(['2023-11-30 09:00', '2023-12-01 09:00'], tz='UTC')
    assert preprocessing.test_windows_from(idx, '2023-12-01').tolist() == [False, True]          # naive vs aware
    assert preprocessing.test_windows_from(idx.tz_localize(None), pd.Timestamp('2023-12-01', tz='UTC')).tolist() == [False, True]


def test_wind_speed_nwp_mapping_wind_and_sl_paths():
    wind_cols = ['wind_speed_h10_1', 'wind_speed_h100_1', 't_2m_1', 'power']
    known, _ = preprocessing.prepare_features_for_tft(wind_cols, ['wind_speed_nwp', 't_2m'], ['power'])
    assert known == ['wind_speed_h10_1', 'wind_speed_h100_1', 't_2m_1']
    sl_cols = ['ghi_nwp_1', 'wind_speed_nwp_1', 't_2m_1', 'power']
    known, observed = preprocessing.prepare_features_for_tft(sl_cols, ['ghi_nwp', 'wind_speed_nwp', 't_2m'], ['power'])
    assert known == ['ghi_nwp_1', 'wind_speed_nwp_1', 't_2m_1'] and observed == ['power']


def test_fewshot_window_ends_at_first_test_run():
    start, end = finetune.window('2023-12-01', 7, '30min', 96, ['09'])
    assert end == '2023-11-29 09:00'            # targets [11-29 09:00, 12-01 09:00) end at the first test run
    assert start == '2023-11-21 09:00'          # 7 daily windows need the run 48 h before the first of them
    start1, end1 = finetune.window('2023-12-01', 1, '30min', 96, ['06', '09', '12', '15'])
    assert (start1, end1) == ('2023-11-27 06:00', '2023-11-29 06:00')
