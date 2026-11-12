"""Cold-start calendar boundaries and identity with existing preprocessing."""
import copy
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'scripts'), str(root)]
import pv_quantile_coldstart as m


def test_calendar_budget_and_pipeline_identity(monkeypatch):
    monkeypatch.setenv('DATA_ROOT', '/tmp')
    # Explicit inclusion of lower bound, exclusion of upper bound and spanning windows.
    lo = pd.to_datetime(['2024-01-24 23:30', '2024-01-25', '2024-01-30', '2024-01-30'], utc=True, format='mixed')
    hi = pd.to_datetime(['2024-01-26', '2024-01-26', '2024-02-01', '2024-01-31 23:30'], utc=True, format='mixed')
    item = dict(X_train={'known': np.arange(4)[:, None]}, y_train=np.arange(4)[:, None], index_train=lo)
    (x, y), info = m.subset(item, lo, hi, 7)
    np.testing.assert_array_equal(y[:, 0], [1, 3])
    np.testing.assert_array_equal(x['known'][:, 0], [1, 3])
    assert info['windows'] == 2
    try:
        m.subset(item, lo - pd.Timedelta(days=365), hi - pd.Timedelta(days=365), 7)
    except ValueError:
        pass
    else:
        raise AssertionError('Empty budget was accepted')
    print('PASS explicit boundary and empty-budget checks')
    
    # Real preprocessing code, with timestamp-encoded labels and complete daily NWP runs.
    cfg = m.ft.base.load_cfg(str(root / m.ft.base.DATA_CFG))
    cfg['model'] = copy.deepcopy(m.ft.base.load_cfg(str(root / m.ft.base.Q_CFG))['model'])
    features = m.ft.preprocessing.get_features(config=cfg)
    issues = pd.date_range('2023-07-24 09:00', '2024-06-01 09:00', freq='D', tz='UTC')
    starts = issues.repeat(96)
    leads = np.tile(np.arange(96) / 2, len(issues))
    valid = starts + pd.to_timedelta(leads, unit='h')
    index = pd.MultiIndex.from_arrays([starts, leads, valid], names=['starttime', 'forecasttime', 'timestamp'])
    vals = valid.asi8 / 1e9
    frame = pd.DataFrame({k: np.sin(np.arange(len(index)) / (i + 10)) for i, k in enumerate(features['known'])}, index=index)
    frame['power'] = vals
    prepared, _ = m.ft.preprocessing.pipeline(frame, cfg, features['known'], features['observed'], features['static'], 'power')
    original = m.ft.base.windows({'park': frame}, cfg, features, None, m.ft.base.CAL, m.ft.base.EV)['park']
    for part in ['train', 'test']:
        np.testing.assert_array_equal(prepared['y_' + part], original['y_' + part])
        for k in prepared['X_' + part]:
            np.testing.assert_array_equal(prepared['X_' + part][k], original['X_' + part][k])
    np.testing.assert_array_equal(prepared['index_test'], original['index_test'])
    np.testing.assert_array_equal(prepared['nwp_raw_test'], original['nwp_raw_test'])
    print('PASS existing generator array, index, and raw NWP identity')
    
    lower, upper = m.label_bounds(frame, prepared['index_train'])
    # Labels are encoded as seconds since epoch; float32 has ~128 second precision.
    np.testing.assert_allclose(prepared['y_train'][:, 0], lower.asi8 / 1e9, rtol=0, atol=128)
    np.testing.assert_allclose(prepared['y_train'][:, -1], upper.asi8 / 1e9, rtol=0, atol=128)
    counts = []
    for days in [7, 14, 30, 60, 90]:
        (_, own), info = m.subset(prepared, lower, upper, days)
        start = (pd.Timestamp('2024-02-01', tz='UTC') - pd.Timedelta(days=days)).timestamp()
        end = pd.Timestamp('2024-02-01', tz='UTC').timestamp()
        assert own.min() >= start and own.max() < end
        counts.append(info['windows'])
    assert counts == sorted(counts)
    print('PASS actual target bound mapping and older target exclusion for all five budgets:', counts)
