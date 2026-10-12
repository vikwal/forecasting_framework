"""data.station_history_start on the real-park path: rows before the history start are dropped, test rows kept."""

import numpy as np
import pandas as pd

from utils.preprocessing import history_cut_mask


def test_cut_before_history_start_within_training_period():
    ts = pd.date_range('2024-02-10', '2024-03-05', freq='D', tz='UTC')
    m = history_cut_mask(ts, '2024-02-15', '2024-03-01')
    assert m.dtype == bool and m.shape == (len(ts),)
    assert m[ts < pd.Timestamp('2024-02-15', tz='UTC')].all()
    assert not m[ts >= pd.Timestamp('2024-02-15', tz='UTC')].any()


def test_test_rows_kept_even_if_history_starts_after_test():
    ts = pd.date_range('2024-02-20', '2024-03-10', freq='D', tz='UTC')
    m = history_cut_mask(ts, '2024-03-05', '2024-03-01')   # history start beyond test_start: only training rows cut
    assert m[ts < pd.Timestamp('2024-03-01', tz='UTC')].all()
    assert not m[ts >= pd.Timestamp('2024-03-01', tz='UTC')].any()


def test_naive_timestamps_are_treated_as_utc():
    ts = pd.Series(pd.date_range('2024-02-14 22:00', periods=4, freq='h'))      # naive
    m = history_cut_mask(ts, '2024-02-15', '2024-03-01')
    assert m.tolist() == [True, True, False, False]
    assert isinstance(m, np.ndarray)
