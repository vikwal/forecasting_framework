"""Client-drift diagnostics (utils/fl_drift.py, fl.drift_metrics)."""

import math
import os

import numpy as np
import pytest
import torch

from utils import fedgradient as fg
from utils import federated, fl_drift
from test_fedgradient import (_client, _data, _local_dispatch, _partitions, _sim_config, _sim_hp, _tft,
                              cpu_only)  # noqa: F401


def _vec(*xs):
    return torch.tensor(xs, dtype=torch.float32)


def test_identical_vectors_are_fully_aligned():
    s = fl_drift.step_stats([_vec(1, 2, 3)] * 3, [1, 2, 3])
    assert np.allclose(s['cos'], 1.0) and math.isclose(s['pairwise_cos'], 1.0, abs_tol=1e-6)
    assert math.isclose(s['alignment'], 1.0, abs_tol=1e-6)
    assert math.isclose(s['mean_norm'], math.sqrt(14), abs_tol=1e-5)


def test_orthogonal_vectors():
    s = fl_drift.step_stats([_vec(1, 0), _vec(0, 1)], [1, 1])
    assert np.allclose(s['cos'], [math.sqrt(0.5)] * 2, atol=1e-6)
    assert abs(s['pairwise_cos']) < 1e-6
    assert math.isclose(s['alignment'], math.sqrt(0.5), abs_tol=1e-6)
    assert np.allclose(s['norm'], [1.0, 1.0])


def test_opposite_vectors_cancel():
    s = fl_drift.step_stats([_vec(2, 0), _vec(-2, 0)], [1, 1])
    assert s['alignment'] < 1e-6 and math.isclose(s['pairwise_cos'], -1.0, abs_tol=1e-6)


def test_weighted_mean_follows_the_weights():
    s = fl_drift.step_stats([_vec(1, 0), _vec(0, 1)], [3, 1])        # mean = (0.75, 0.25)
    assert s['cos'][0] > s['cos'][1]
    assert math.isclose(s['mean_norm'], math.sqrt(0.75 ** 2 + 0.25 ** 2), abs_tol=1e-6)


def test_flatten_keeps_key_order_and_skips_missing():
    t = {'b': torch.ones(2, 2), 'a': torch.zeros(3), 'i': torch.tensor([1, 2])}
    v = fl_drift.flatten(t, ['a', 'b', 'c', 'i'])
    assert v.tolist() == [0, 0, 0, 1, 1, 1, 1]                        # 'c' missing, 'i' not float


def test_drift_log_round_bookkeeping():
    log = fl_drift.DriftLog(['w'])
    log.add(['A', 'B'], [_vec(1, 0), _vec(1, 0)], [1, 1])
    log.add(['A'], [_vec(0, 1)], [1])                                # B exhausted
    log.add_val({'A': 0.5, 'B': 0.7, 'C': 0.9})                      # C: evaluation only
    log.end_round(3)
    c, r = log.frames()
    assert list(c['client']) == ['A', 'B', 'C'] and list(c['n_steps']) == [2, 1, 0]
    assert list(c['val_rmse']) == [0.5, 0.7, 0.9] and np.isnan(c.loc[2, 'cos_mean'])
    assert r.loc[0, 'n_steps'] == 2 and r.loc[0, 'round'] == 3
    assert math.isclose(r.loc[0, 'pairwise_cos'], 1.0, abs_tol=1e-6)  # only the two-client step counts
    assert math.isclose(r.loc[0, 'alignment'], 1.0, abs_tol=1e-6)
    log2 = fl_drift.DriftLog.from_dict(log.to_dict())
    assert log2.frames()[1].equals(r) and log2.keys == ['w']


def test_single_client_step_has_no_pairwise_cos():
    log = fl_drift.DriftLog(['w'])
    log.add(['A'], [_vec(1, 0)], [1])
    log.end_round(1)
    assert np.isnan(log.frames()[1].loc[0, 'pairwise_cos'])


def test_run_round_feeds_the_log_per_step():
    model = _tft()
    Xa, ya = _data(40, [0, 1], seed=1)                                # 5 batches
    Xb, yb = _data(27, [2, 3], seed=2)                                # 3 batches
    clients = [_client(_tft(seed=5), Xa, ya, cid='A'), _client(_tft(seed=6), Xb, yb, cid='B')]
    log = fl_drift.DriftLog(fl_drift.trainable_keys(model, skip=['static_embed.0.weight']))
    optim = fg.build_server_optimizer(model.parameters(), {'server_optimizer': 'sgd', 'server_lr': 0.01})
    _, steps = fg.run_round(_local_dispatch(clients), ['A', 'B'], model, optim, seed=1, drift=log)
    log.end_round(1)
    c, r = log.frames()
    assert steps == 5 and r.loc[0, 'n_steps'] == 5
    assert dict(zip(c['client'], c['n_steps'])) == {'A': 5, 'B': 3}
    assert c['cos_mean'].between(-1, 1).all() and (c['norm_mean'] > 0).all()
    assert 'static_embed.0.weight' not in log.keys


def _run(strategy, n_rounds=2, tmp_path=None, resume=None):
    cfg = _sim_config(strategy, tmp_path, n_rounds)
    cfg['fl']['drift_metrics'] = True
    if resume:
        cfg['fl']['resume'] = resume
    hp = _sim_hp(strategy)
    hp['n_rounds'] = n_rounds
    return federated.run_simulation(_partitions(), cfg, hp)


@pytest.mark.parametrize('strategy', ['fedavg', 'fedgradient'])
def test_run_simulation_writes_drift_frames(cpu_only, strategy):
    history, _ = _run(strategy)
    c, r = history['drift_clients'], history['drift_rounds']
    assert list(r['round']) == [1, 2]
    assert list(r['n_steps']) == ([4, 4] if strategy == 'fedgradient' else [1, 1])
    assert r['alignment'].between(0, 1 + 1e-6).all() and r['pairwise_cos'].between(-1, 1).all()
    assert sorted(c['client'].unique()) == ['A', 'B'] and len(c) == 4
    assert np.isfinite(c['cos_mean']).all() and np.isfinite(c['val_rmse']).all()


def test_drift_rows_survive_a_resume(cpu_only, tmp_path):
    _run('fedgradient', n_rounds=2, tmp_path=tmp_path)
    resumed, _ = _run('fedgradient', n_rounds=3, resume=os.path.join(tmp_path, 'last.pt'))
    assert list(resumed['drift_rounds']['round']) == [1, 2, 3]
    assert len(resumed['drift_clients']) == 6


def test_default_leaves_history_without_drift(cpu_only):
    hp = _sim_hp('fedavg')
    hp['n_rounds'] = 2
    history, _ = federated.run_simulation(_partitions(), _sim_config('fedavg'), hp)
    assert 'drift_rounds' not in history
