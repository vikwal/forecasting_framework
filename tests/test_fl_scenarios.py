"""FL scenario extensions: masked loss (data.target_mask), mask flags of the curtailed
releases, seeding, per-client holdout of the local baseline, cache key, hub wind target."""

import numpy as np
import pandas as pd
import pytest
import torch

from utils import hpo, local_training, tools
from utils.preprocessing import park_hub_wind, target_mask_flags


def _cfg(mask=None, quantiles=(0.5,)):
    return {'data': {'target_mask': mask} if mask else {},
            'model': {'tft': {'quantiles': list(quantiles)} if quantiles else {}}}


@pytest.mark.parametrize('quantiles', [(0.5,), None])
def test_criterion_without_mask_is_unchanged(quantiles):
    pred, tgt = torch.rand(4, 6), torch.rand(4, 6)
    crit, _ = tools.make_criterion(_cfg(None, quantiles))
    ref = tools._pinball_loss(pred, tgt, list(quantiles)) if quantiles else torch.nn.MSELoss()(pred, tgt)
    assert torch.equal(crit(pred, tgt), ref)


@pytest.mark.parametrize('quantiles', [(0.5,), None])
def test_masked_criterion_ignores_masked_targets(quantiles):
    pred, tgt = torch.rand(4, 6), torch.rand(4, 6)
    masked = tgt.clone()
    masked[:, :3] = tools.TARGET_MASK_VALUE
    crit, _ = tools.make_criterion(_cfg('all', quantiles))
    base, _ = tools.make_criterion(_cfg(None, quantiles))
    assert torch.allclose(crit(pred, masked), base(pred[:, 3:], tgt[:, 3:]))
    # gradients only through unmasked elements
    p = pred.clone().requires_grad_(True)
    crit(p, masked).backward()
    assert torch.all(p.grad[:, :3] == 0) and torch.any(p.grad[:, 3:] != 0)
    # all masked: zero loss, no NaN
    allm = torch.full_like(tgt, tools.TARGET_MASK_VALUE)
    assert crit(pred, allm).item() == 0.0


def test_target_mask_flags():
    df = pd.DataFrame({'curt_flag': [0, 1, 1, 1, 0], 'loss_mkt': [0, 5, 0, 0, 0],
                       'loss_env': [0, 0, 2, 0, 0], 'loss_grid': [0, 0, 0, 7, 0]})
    assert target_mask_flags(df, 'all').tolist() == [False, True, True, True, False]
    assert target_mask_flags(df, 'market_env').tolist() == [False, True, True, False, False]
    assert target_mask_flags(df, 'grid').tolist() == [False, False, False, True, False]
    with pytest.raises(ValueError):
        target_mask_flags(df, 'nonsense')


def test_set_seed_reproducible():
    tools.set_seed(7)
    a = (torch.rand(3), np.random.rand(3))
    tools.set_seed(7)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_local_configs_keep_own_holdout():
    cfg = {'data': {'files': ['a', 'b', 'c', 'd'], 'val_files': ['x', 'y'], 'holdout_files': ['z']},
           'model': {'fl': True},
           'fl': {'clients': {'C1': ['a', 'b'], 'C2': ['c', 'd']}, 'client_holdout': {'C1': ['x']}}}
    out = local_training.client_configs(cfg)
    assert out['C1']['data']['files'] == ['a', 'b'] and out['C1']['data']['holdout_files'] == ['x']
    assert 'holdout_files' not in out['C2']['data'] and 'val_files' not in out['C2']['data']


def test_cache_key_has_train_start_and_mask_for_real_parks(monkeypatch):
    import copy
    from utils.data_cache import DataCache
    monkeypatch.setenv('DATA_ROOT', '/data')
    base = tools.load_config('configs/parks_v1/config_parks_v1_curt_v11_cl80_static.yaml')
    dc = DataCache.__new__(DataCache)
    feats = {'known': [], 'observed': [], 'static': []}
    k = lambda c: dc._get_config_hash(c, feats, 'tft')  # noqa: E731
    c2 = copy.deepcopy(base); c2['data']['train_start'] = '2024-07-01'
    c3 = copy.deepcopy(base); c3['data']['target_mask'] = 'all'
    c4 = copy.deepcopy(base); del c4['data']['power_col']
    c5 = copy.deepcopy(c4); c5['data']['train_start'] = '2024-07-01'
    assert len({k(base), k(c2), k(c3)}) == 3
    assert k(c4) == k(c5)          # other paths: key unchanged by train_start (as before)


def test_grid50_mask_is_deterministic_half():
    from utils.preprocessing import _known_events
    ids = pd.Series([f'N{i}:2024' for i in range(2000)] + ['', None])
    a, b = _known_events(ids, 0.5), _known_events(ids, 0.5)
    assert a.equals(b) and 0.45 < a[:2000].mean() < 0.55 and not a.iloc[-1] and not a.iloc[-2]
    df = pd.DataFrame({'curt_flag': [1, 1, 1], 'loss_mkt': [1, 0, 0], 'loss_env': [0, 0, 0],
                       'loss_grid': [0, 5, 5], 'grid_event_id': ['', 'A', 'B']})
    m = target_mask_flags(df, 'market_env_grid50')
    assert m.iloc[0] and m.iloc[1] == _known_events(pd.Series(['A']), 0.5).iloc[0]


def test_training_pipeline_starts_from_initial_weights():
    """Fine-tuning must start from the given weights; with lr 0 they stay unchanged."""
    import inspect
    sig = inspect.signature(tools.training_pipeline)
    assert 'initial_weights' in sig.parameters and 'trainable' in sig.parameters
    src = open('train_fl.py').read()
    assert 'initial_weights=global_weights' in src


def test_cache_key_station_history(monkeypatch):
    import copy
    from utils.data_cache import DataCache
    monkeypatch.setenv('DATA_ROOT', '/data')
    base = tools.load_config('configs/parks_v1/config_parks_v1_curt_v11_cl80_static.yaml')
    dc = DataCache.__new__(DataCache)
    k = lambda c: dc._get_config_hash(c, {'known': [], 'observed': [], 'static': []}, 'tft')  # noqa: E731
    c2 = copy.deepcopy(base); c2['data']['station_history_start'] = {'SEL976062210315': '2024-07-18'}
    assert k(base) != k(c2)


def test_park_hub_wind_is_capacity_weighted():
    df = pd.DataFrame({'wind_speed_hub_t1': [4.0, 8.0], 'wind_speed_hub_t2': [6.0, 10.0],
                       'wind_speed_hub_t3': [99.0, 99.0]})
    groups = pd.DataFrame({'turbine': ['t1', 't2'], 'n_turbines': [3, 1], 'rated_kw': [2000.0, 2000.0]})
    assert np.allclose(park_hub_wind(df, groups).to_numpy(), [4.5, 8.5])
    with pytest.raises(ValueError):
        park_hub_wind(df[['wind_speed_hub_t1']], groups)


def test_kfolds_by_dates_expanding_without_target_overlap():
    t = pd.date_range('2023-07-26 09:00', '2025-07-31 09:00', freq='D', tz='UTC')
    n = len(t)
    ds = [{'X_train': {'known': np.arange(n)[:, None] + 1000 * s, 'observed': np.zeros((n, 1))},
           'y_train': np.arange(n)[:, None] + 1000 * s, 'index_train': t} for s in range(2)]
    b = ['2024-08-01', '2024-12-01', '2025-04-01', '2025-08-01']
    folds = hpo.kfolds_by_dates(ds, b, horizon_hours=48)
    assert len(folds) == 3
    for i, ((Xt, yt), (Xv, yv)) in enumerate(folds):
        lo, hi = pd.Timestamp(b[i], tz='UTC'), pd.Timestamp(b[i + 1], tz='UTC')
        tt, tv = t[yt[:, 0] % 1000], t[yv[:, 0] % 1000]
        assert (tt + pd.Timedelta(hours=48) <= lo).all() and (tv >= lo).all() and (tv < hi).all()
        assert len(yv) == 2 * ((t >= lo) & (t < hi)).sum()                 # both stations
        assert np.array_equal(Xt['known'], yt) and np.array_equal(Xv['known'], yv)
    assert len(folds[0][0][1]) < len(folds[1][0][1]) < len(folds[2][0][1])   # expanding
    # last training issue before the first boundary: 2024-07-30 09:00 + 48 h > 2024-08-01 -> 07-29
    assert t[folds[0][0][1][:, 0] % 1000].max() == pd.Timestamp('2024-07-29 09:00', tz='UTC')


def test_split_bounds_strict_and_default():
    from utils.preprocessing import split_bounds
    cfg = {'data': {'train_end': '2025-07-31 23:00', 'test_start': '2025-08-01', 'test_end': '2026-07-31',
                    'freq': '1h'}, 'model': {'horizon': 48}}
    tr, te = split_bounds(cfg)                                            # default: unchanged
    assert tr == pd.Timestamp('2025-07-31 23:00') and te == pd.Timestamp('2026-07-31')
    cfg['data']['strict_split'] = True
    tr, te = split_bounds(cfg)
    assert tr == pd.Timestamp('2025-07-30 00:00')                         # issue + 48 h <= test_start
    assert te == pd.Timestamp('2026-07-31 23:59:59')                      # whole last day


def test_load_study_required_raises(tmp_path, monkeypatch):
    monkeypatch.delenv('OPTUNA_STORAGE', raising=False)
    from utils import hpo as _hpo
    path = str(tmp_path / 's.db')
    assert _hpo.load_study(path, 'missing') is None
    with pytest.raises(RuntimeError):
        _hpo.load_study(path, 'missing', required=True)


def test_filter_train_runs_keeps_issue_hours():
    from utils.preprocessing import filter_train_runs
    t = pd.DatetimeIndex([f'2024-01-0{d} {h}:00' for d in (1, 2) for h in ('06', '09', '12', '15')], tz='UTC')
    n = len(t)
    prep = {'X_train': {'known': np.arange(n)[:, None], 'static': np.zeros((n, 2))}, 'y_train': np.arange(n)[:, None],
            'index_train': t, 'X_test': None}
    out = filter_train_runs(prep, ['09'])
    assert list(out['index_train'].hour) == [9, 9] and out['y_train'][:, 0].tolist() == [1, 5]
    assert out['X_train']['known'][:, 0].tolist() == [1, 5] and len(out['X_train']['static']) == 2
    with pytest.raises(ValueError):
        filter_train_runs(prep, ['03'])
