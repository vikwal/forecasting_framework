"""scripts/pv_hpo_local.py: search space, variant-B folds, model wrapper (2026-11-13)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pv_hpo_local as h  # noqa: E402


def test_pilot_configs_reproducible_and_valid():
    a, b = h.pilot_configs(), h.pilot_configs()
    assert a == b and len(a) == 2 * h.PILOT_CONFIGS
    assert {c['conv_type'] for c in a[:h.PILOT_CONFIGS]} == {'cnn'}
    for c in a:
        assert c['kernel_size'] % 2 == 1
        assert c['n_rnn_layers'] >= 2 or c['dropout'] == 0.0
        assert c['conv_type'] == 'cnn' or c['n_cnn_layers'] <= 7


def test_net_step_output_shape():
    for c in h.pilot_configs()[::12]:
        net = h.Net(4, 7, c).eval()
        x = torch.randn(2, 192, 4)
        assert net(None, x).shape == (2, 96, 7)
        y = x.clone()
        y[:, :96] += 5.0                                     # past window is not an input
        assert torch.equal(net(None, x), net(None, y))


def _park(n1=30, n2=360):
    t1 = pd.date_range('2023-07-24 09:00', periods=n1, freq='D', tz='UTC')
    t2 = pd.date_range('2024-08-01 09:00', periods=n2, freq='D', tz='UTC')
    t2 = t2[t2 + pd.Timedelta(hours=48) <= pd.Timestamp('2025-08-01', tz='UTC')]
    X = lambda n, off: dict(observed=np.zeros((n, 96, 0)), known=np.full((n, 192, 1), off, float))  # noqa: E731
    y2 = np.arange(len(t2), dtype=float)[:, None].repeat(96, 1)
    return dict(t1=t1, X1=X(n1, -1.0), y1=-np.ones((n1, 96)), t2=t2, X2=X(len(t2), 1.0), y2=y2,
                nwp2=y2 + 100)


def test_pilot_split_never_returns_j3_and_separates_test():
    d = _park()
    d.update(X3='j3', y3='j3', t3='j3', nwp3='j3')
    dev, ts = h.pilot_split(d)
    assert not {'X3', 'y3', 't3', 'nwp3'} & set(dev)
    assert (dev['t2'] + pd.Timedelta(hours=48) <= h.utc(h.PILOT_TEST)).all()
    assert (ts['t'] >= h.utc(h.PILOT_TEST)).all() and len(ts['y']) > 0
    folds = h.folds_b(dev, h.PILOT_BOUNDS)
    assert len(folds) == 2
    for _, va, _ in folds:                                    # no pilot-test target in any selection score
        assert va[1].max() < ts['y'].min()


def test_folds_b_expanding_without_target_overlap():
    d = _park()
    folds = h.folds_b(d, h.B_BOUNDS)
    assert len(folds) == 3
    sizes = [len(tr[1]) for tr, _, _ in folds]
    assert sizes[0] == 30 and sizes[0] < sizes[1] < sizes[2]
    for (tr, va, nwp), lo, hi in zip(folds, h.B_BOUNDS, h.B_BOUNDS[1:]):
        m = np.asarray((d['t2'] >= h.utc(lo)) & (d['t2'] < h.utc(hi)))
        assert np.array_equal(va[1], d['y2'][m]) and np.array_equal(nwp, d['nwp2'][m])
        # J2 windows in training end before the fold start (issue + 48 h <= lo)
        used = tr[1][tr[1][:, 0] >= 0][:, 0].astype(int)
        assert (d['t2'][used] + pd.Timedelta(hours=48) <= h.utc(lo)).all()


def test_space_v2_bounds():
    import optuna
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(200):
        hp = h.space_v2(study.ask())
        assert hp['kernel_size'] % 2 == 1 and 2 <= hp['n_cnn_layers'] <= 6
        assert hp['lr'] <= 2e-3 and hp['weight_decay'] <= 1e-4
        assert hp['n_rnn_layers'] >= 2 or hp['dropout'] == 0.0


def test_hpo_parks_exclude_own_fold_and_test():
    split = dict(test=['t1', 't2'], folds={'1': ['a', 'b'], '2': ['c', 'd'], '3': ['e']})
    for k in (1, 2, 3):
        ps = h.hpo_parks(split, k)
        assert not set(ps) & set(split['folds'][str(k)]) and not set(ps) & set(split['test'])
        assert sorted(ps) == sorted(p for j, v in split['folds'].items() if j != str(k) for p in v)
        assert ps == h.hpo_parks(split, k)                  # fixed order: pruning compares the same parks


def test_no_j3_drops_test_year():
    d = dict(X1=1, y1=1, X2=2, y2=2, t2=2, nwp2=2, X3=3, y3=3, t3=3, nwp3=3, park='p', group='N0')
    assert set(h.no_j3(d)) == {'X1', 'y1', 'X2', 'y2', 't2', 'nwp2', 'park', 'group'}
