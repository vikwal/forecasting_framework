"""Date folds (hpo.fold_boundaries) shared by hpo_cl.py / data_cache.py and hpo_fl.py:
fold layout per client, scaler cutoff and the fold-value reduction."""

import numpy as np
import pandas as pd
import pytest

from utils import hpo

B = ['2024-08-01', '2024-12-01', '2025-04-01', '2025-08-01']


def _prepared(start: str, end: str, horizon: int = 48) -> dict:
    """One station: hourly issue times, X as the TFT dict layout, y = issue hour."""
    t = pd.date_range(start, end, freq='24h', tz='UTC')
    n = len(t)
    return {'index_train': t,
            'X_train': {'known': np.arange(n, dtype=float)[:, None, None].repeat(horizon, 1),
                        'static': np.zeros((n, 1))},
            'y_train': np.arange(n, dtype=float)}


def _cfg(boundaries=B, kfolds=3, test_start='2025-08-01'):
    return {'hpo': {'fold_boundaries': boundaries, 'kfolds': kfolds, 'min_train_date': None},
            'model': {'horizon': 48}, 'data': {'test_start': test_start}}


def _expected(t: pd.DatetimeIndex, b0, b1, horizon=48):
    tr = t[t + pd.Timedelta(hours=horizon) <= b0]
    va = t[(t >= b0) & (t < b1)]
    return len(tr), len(va)


def test_client_folds_expanding_window_without_target_overlap():
    cfg = _cfg()
    t1 = pd.date_range('2023-07-24', '2025-07-31', freq='24h', tz='UTC')
    t2 = pd.date_range('2023-10-01', '2025-07-31', freq='24h', tz='UTC')
    folds = hpo.client_folds([_prepared('2023-07-24', '2025-07-31'), _prepared('2023-10-01', '2025-07-31')], cfg)
    assert len(folds) == 3 == hpo.n_folds(cfg)
    b = hpo.fold_boundaries(cfg)
    prev_train = 0
    for i, ((X_tr, y_tr), (X_va, y_va)) in enumerate(folds):
        n_tr1, n_va1 = _expected(t1, b[i], b[i + 1])
        n_tr2, n_va2 = _expected(t2, b[i], b[i + 1])
        assert len(y_tr) == n_tr1 + n_tr2 and len(y_va) == n_va1 + n_va2    # issue + 48 h <= b_i / [b_i, b_i+1)
        assert len(y_tr) > prev_train                                        # expanding window
        prev_train = len(y_tr)
        assert X_tr['known'].shape[0] == len(y_tr) and X_va['known'].shape[0] == len(y_va)
        assert X_tr['static'].shape[0] == len(y_tr)
    # validation windows are 4 months each (one boundary step), training grows by one window per fold
    assert [len(f[1][1]) for f in folds] == [_expected(t1, b[i], b[i + 1])[1] + _expected(t2, b[i], b[i + 1])[1]
                                             for i in range(3)]


def test_client_folds_positional_without_boundaries():
    cfg = _cfg(boundaries=None, kfolds=2)
    folds = hpo.client_folds([_prepared('2023-07-24', '2025-07-31')], cfg)
    assert len(folds) == 2 == hpo.n_folds(cfg)


def test_scaler_fit_end_is_first_boundary_or_test_start():
    assert hpo.scaler_fit_end(_cfg()) == pd.Timestamp('2024-08-01', tz='UTC')
    assert hpo.scaler_fit_end(_cfg(boundaries=None)) == pd.Timestamp('2025-08-01', tz='UTC')
    assert hpo.scaler_fit_end(_cfg(test_start='2024-06-01')) == pd.Timestamp('2024-06-01', tz='UTC')


def test_reduce_fold_metric():
    rounds = [0.30, 0.20, 0.25, 0.28]
    assert hpo.reduce_fold_metric(rounds, {'direction': 'minimize'}, 'last') == 0.28
    assert hpo.reduce_fold_metric(rounds, {'direction': 'minimize'}, 'best') == 0.20
    assert hpo.reduce_fold_metric(pd.Series(rounds), {'direction': 'maximize'}, 'best') == 0.30
    with pytest.raises(ValueError):
        hpo.reduce_fold_metric(rounds, {'direction': 'minimize'}, 'mean')


class _Trial:
    """Records the suggested names; returns the lower bound / first choice."""
    def __init__(self):
        self.params = {}

    def suggest_float(self, name, lo, hi, log=False):
        self.params[name] = lo
        return lo

    def suggest_int(self, name, lo, hi, step=1, log=False):
        self.params[name] = lo
        return lo

    def suggest_categorical(self, name, choices):
        self.params[name] = choices[0]
        return choices[0]


def _fl_cfg(strategy='fedgradient'):
    return {'model': {'lr': 5e-4, 'tft': {'clipnorm': 7.0}},
            'fl': {'strategy': strategy, 'n_local_epochs': 1,
                   'fedgradient': {'server_optimizer': 'adam', 'server_lr': 0.01, 'beta_1': 0.9, 'beta_2': 0.99,
                                   'eps': 1e-3, 'momentum': 0.9, 'weight_decay': 0.0, 'clipnorm': 1.0,
                                   'weighting': 'batch_size'}},
            'hpo': {'fl': {'fedgradient': {'server_optimizer': ['sgd_momentum', 'adam'], 'server_lr': [1e-4, 1e-1],
                                           'clipnorm': [0.3, 10.0]},
                           'n_local_epochs': [1, 5]}}}


def test_fedgradient_trial_searches_server_step_and_client_clipping():
    t = _Trial()
    hp = hpo._fedgradient_trial(_fl_cfg(), t)
    assert hp['server_optimizer'] == 'sgd_momentum' and hp['server_lr'] == 1e-4 and hp['client_clipnorm'] == 0.3
    assert set(t.params) == {'server_optimizer', 'server_lr', 'client_clipnorm'}


def test_fedavg_overrides_only_for_weight_averaging():
    assert hpo.fedavg_trial_overrides(_fl_cfg('fedavg'), _Trial()) == {'n_local_epochs': 1}
    assert hpo.fedavg_trial_overrides(_fl_cfg('fedgradient'), _Trial()) == {}
    cfg = _fl_cfg('fedavg')
    cfg['hpo']['fl'].pop('n_local_epochs')
    assert hpo.fedavg_trial_overrides(cfg, _Trial()) == {}


def test_prune_fold_is_the_penultimate_fold():
    assert hpo.prune_fold(3) == 1 and hpo.prune_fold(2) == 0 and hpo.prune_fold(5) == 3
    assert hpo.prune_fold(1) is None


def test_sampler_seed_differs_per_process_and_keeps_none():
    import os
    assert hpo._sampler_seed(None) is None
    s = hpo._sampler_seed(42)
    assert 0 <= s < 2 ** 31 - 1 and s == hpo._sampler_seed(42)          # deterministic within a process
    assert s != (42 * 100003 + os.getpid() + 1) % (2 ** 31 - 1)


def _hp_cfg(fl: bool):
    return {'model': {'name': 'tft', 'shuffle': True, 'fl': fl, 'lookback': 48, 'horizon': 48,
                      'tft': {'num_quantiles': 1, 'clipnorm': 7.0}},
            'fl': {'strategy': 'fedavg', 'n_rounds': 100},
            'hpo': {'batch_size': [32, 256], 'epochs': [10, 100], 'n_layers': [1, 4], 'n_cnn_layers': [1, 8],
                    'n_rnn_layers': [1, 4], 'learning_rate': [1e-4, 1e-2], 'attention_heads': [1, 8],
                    'cnn': {'filters': [16, 128], 'kernel_size': [2, 5], 'increase_filters': False},
                    'rnn': {'dropout': [0.0, 0.5], 'units': [16, 128]}, 'fnn': {'units': [32, 256]},
                    'tft': {'n_heads': [1, 8], 'hidden_dim': [16, 128], 'dropout': [0.01, 0.5],
                            'n_lstm_layers': [1, 2], 'static_embedding_dim': [8, 64], 'clipnorm': [1.0, 10.0]},
                    'weight_decay': [1e-6, 1e-3],
                    'fl': {'server_lr': [1e-4, 1e-1], 'beta_1': [0, 1], 'beta_2': [0, 1], 'tau': [1e-4, 1e-1]}}}


def test_fl_search_space_has_no_dead_dimensions():
    t = _Trial()
    hp = hpo.get_hyperparameters(_hp_cfg(fl=True), hpo=True, trial=t)
    assert 'weight_decay' not in t.params and 'clipnorm' not in t.params      # not consumed by FL clients
    assert hp['clipnorm'] == 7.0 and 'weight_decay' not in hp
    t2 = _Trial()
    hp2 = hpo.get_hyperparameters(_hp_cfg(fl=False), hpo=True, trial=t2)
    assert 'weight_decay' in t2.params and 'clipnorm' in t2.params and hp2['clipnorm'] == 1.0


def test_fold_without_samples_raises():
    cfg = _cfg(boundaries=['2026-01-01', '2026-05-01'])         # after the data: no validation samples
    with pytest.raises(ValueError):
        hpo.client_folds([_prepared('2023-07-24', '2025-07-31')], cfg)


def test_make_pruner_percentile_before_median():
    import optuna
    assert hpo.make_pruner(None) is None and hpo.make_pruner({}) is None
    p = hpo.make_pruner({'percentile': 75.0, 'use_median_pruner': False, 'n_startup_trials': 10, 'n_warmup_steps': 1})
    assert isinstance(p, optuna.pruners.PercentilePruner) and p._percentile == 75.0 and p._n_startup_trials == 10
    assert isinstance(hpo.make_pruner({'use_median_pruner': True}), optuna.pruners.MedianPruner)
    assert isinstance(hpo.make_pruner({'disable_pruning': True}), optuna.pruners.NopPruner)


def test_station_folds_rotate_validation_group_with_expanding_window():
    """hpo.station_folds: fold i validates only group i, trains on the other stations, window still expands."""
    ds = [_prepared('2023-07-24', '2025-07-31') for _ in range(3)]
    keys = ['/x/synth_S1.csv', '/x/synth_S2.csv', '/x/synth_S3.csv']
    folds = hpo.kfolds_by_dates(ds, B, 48, station_groups=[['S1'], ['S2'], ['S3']], keys=keys)
    b = hpo.fold_boundaries(_cfg())
    t = ds[0]['index_train']
    for i, ((X_tr, y_tr), (X_va, y_va)) in enumerate(folds):
        n_tr, n_va = _expected(t, b[i], b[i + 1])
        assert len(y_tr) == 2 * n_tr and len(y_va) == n_va            # two training stations, one validation station
        assert X_tr['static'].shape[0] == len(y_tr)
    with pytest.raises(ValueError):                                      # one list per fold
        hpo.kfolds_by_dates(ds, B, 48, station_groups=[['S1'], ['S2']], keys=keys)
    with pytest.raises(ValueError):                                      # a station in two folds
        hpo.kfolds_by_dates(ds, B, 48, station_groups=[['S1'], ['S1'], ['S3']], keys=keys)


def test_horizon_hours_counts_steps_of_freq():
    assert hpo.horizon_hours({'model': {'horizon': 96}, 'data': {'freq': '30min'}}) == 48.0
    assert hpo.horizon_hours({'model': {'horizon': 48}, 'data': {}}) == 48.0


def test_fixed_window_rotates_stations_with_one_time_split():
    """hpo.fold_mode 'fixed_window': every fold uses [b0, b1) for validation; fold i validates group i and
    trains the other stations on issue + horizon <= b0 only (no pre-b0 sample of the val station in training)."""
    ds = [_prepared('2023-07-24', '2025-07-31') for _ in range(3)]
    for j, d in enumerate(ds):                       # encode the station in y to trace where samples go
        d['y_train'] = d['y_train'] + 1e6 * (j + 1)
    keys = ['/x/synth_S1.parquet', '/x/synth_S2.parquet', '/x/synth_S3.parquet']
    groups = [['S1'], ['S2'], ['S3']]
    B2 = ['2024-08-01', '2025-08-01']
    folds = hpo.kfolds_by_dates(ds, B2, 48, station_groups=groups, keys=keys, fold_mode='fixed_window')
    assert len(folds) == 3
    b0, b1 = hpo.fold_boundaries({'hpo': {'fold_boundaries': B2}})
    t = ds[0]['index_train']
    n_tr, n_va = _expected(t, b0, b1)
    for i, ((X_tr, y_tr), (X_va, y_va)) in enumerate(folds):
        assert len(y_tr) == 2 * n_tr and len(y_va) == n_va
        assert X_tr['static'].shape[0] == len(y_tr) and X_va['static'].shape[0] == len(y_va)
        st_tr = set((y_tr // 1e6).astype(int)); st_va = set((y_va // 1e6).astype(int))
        assert st_va == {i + 1} and st_tr == {1, 2, 3} - {i + 1}
    with pytest.raises(ValueError):                  # fixed_window needs exactly two boundaries
        hpo.kfolds_by_dates(ds, B, 48, station_groups=groups, keys=keys, fold_mode='fixed_window')
    with pytest.raises(ValueError):                  # ... and station groups
        hpo.kfolds_by_dates(ds, B2, 48, fold_mode='fixed_window')
    with pytest.raises(ValueError):                  # unknown mode
        hpo.kfolds_by_dates(ds, B2, 48, station_groups=groups, keys=keys, fold_mode='rolling')
    with pytest.raises(ValueError):                  # a key matching no station id
        hpo.kfolds_by_dates(ds, B2, 48, station_groups=groups, keys=keys[:2] + ['/x/synth_S9.parquet'],
                            fold_mode='fixed_window')
    with pytest.raises(ValueError):                  # a station id matching no key
        hpo.kfolds_by_dates(ds, B2, 48, station_groups=[['S1'], ['S2'], ['S3', 'S4']], keys=keys,
                            fold_mode='fixed_window')


def test_station_of_prefers_exact_stem_match():
    ids = {'pv1', 'pv10'}
    assert hpo._station_of('/x/synth_pv1.parquet', ids) == 'pv1'
    assert hpo._station_of('/x/synth_pv10.parquet', ids) == 'pv10'
    assert hpo._station_of('/x/pv10.csv', ids) == 'pv10'


def test_n_folds_and_client_folds_follow_fold_mode():
    cfg = _cfg(boundaries=['2024-08-01', '2025-08-01'])
    cfg['hpo'].update({'fold_mode': 'fixed_window', 'station_folds': [['S1'], ['S2'], ['S3']]})
    assert hpo.n_folds(cfg) == 3
    assert hpo.n_folds(_cfg()) == 3
    with pytest.raises(ValueError):                  # per-client folds cannot rotate stations
        hpo.client_folds([_prepared('2023-07-24', '2025-07-31')], cfg)
