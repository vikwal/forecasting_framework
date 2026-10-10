"""FL scenario extensions: masked loss (data.target_mask), mask flags of the curtailed
releases, seeding, per-client holdout of the local baseline, cache key."""

import numpy as np
import pandas as pd
import pytest
import torch

from utils import local_training, tools
from utils.preprocessing import target_mask_flags


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
