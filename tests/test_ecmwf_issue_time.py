"""ECMWF run assignment by issue time for the Trianel path (preprocessing._merge_ecmwf_by_issue_time)."""

import numpy as np
import pandas as pd

from utils import preprocessing as pp


def _icon(start: str, leads=(1, 2, 3)) -> pd.DataFrame:
    s = pd.Timestamp(start, tz='UTC')
    return pd.DataFrame({'starttime': [s] * len(leads), 'forecasttime': list(leads),
                         'timestamp': [s + pd.Timedelta(hours=h) for h in leads], 'wind_speed_h78_1': 5.0})


def _runs(start: str, leads, value) -> pd.DataFrame:
    s = pd.Timestamp(start, tz='UTC')
    return pd.DataFrame({'starttime': [s] * len(leads), 'validtime': [s + pd.Timedelta(hours=h) for h in leads],
                         'wind_speed_h100': [value] * len(leads)})


def test_icon_09_takes_the_00_run_and_icon_15_the_12_run():
    icon = pd.concat([_icon('2024-01-05 09:00'), _icon('2024-01-05 15:00')], ignore_index=True)
    runs = pd.concat([_runs('2024-01-05 00:00', range(0, 30), 1.0),      # covers 09+3 h (lead 12)
                      _runs('2024-01-05 12:00', range(0, 30), 2.0)], ignore_index=True)
    out = pp._merge_ecmwf_by_issue_time(icon, runs)
    assert len(out) == 6
    assert (out.loc[out.starttime == pd.Timestamp('2024-01-05 09:00', tz='UTC'), 'wind_speed_h100'] == 1.0).all()
    assert (out.loc[out.starttime == pd.Timestamp('2024-01-05 15:00', tz='UTC'), 'wind_speed_h100'] == 2.0).all()


def test_no_look_ahead_into_a_later_run():
    """An ICON 09 UTC run must not use the 12 UTC ECMWF run even if that one also covers the hours."""
    icon = _icon('2024-01-05 09:00', leads=(4, 5))                        # valid 13, 14 UTC
    runs = pd.concat([_runs('2024-01-05 00:00', range(0, 20), 1.0), _runs('2024-01-05 12:00', range(0, 20), 2.0)])
    out = pp._merge_ecmwf_by_issue_time(icon, runs)
    assert (out['wind_speed_h100'] == 1.0).all()


def test_runs_without_a_complete_match_are_dropped_not_filled():
    icon = pd.concat([_icon('2024-01-05 09:00'), _icon('2024-01-06 09:00')], ignore_index=True)
    runs = _runs('2024-01-05 00:00', range(0, 30), 1.0)                   # no run on 2024-01-06
    out = pp._merge_ecmwf_by_issue_time(icon, runs)
    assert out['starttime'].nunique() == 1 and np.isfinite(out['wind_speed_h100']).all()
    assert out['starttime'].iloc[0] == pd.Timestamp('2024-01-05 09:00', tz='UTC')
