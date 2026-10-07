"""Local training baseline (utils/local_training.py, train_local.py)."""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from utils import local_training as lt
from utils import preprocessing


def _fl_config():
    return {'data': {'path': '/x/parks_v1', 'files': ['P0', 'P1', 'P2', 'P3'],
                     'val_files': ['H0'], 'holdout_files': ['H1'], 'client_files': ['P0']},
            'params': {'static_features': ['park_id'], 'static_categorical': ['park_id']},
            'model': {'fl': True, 'epochs': 100},
            'fl': {'strategy': 'fedgradient', 'clients': {'A': ['P0', 'P1'], 'B': ['P2', 'P3']}}}


def test_client_configs_keep_everything_but_the_station_lists():
    cfgs = lt.client_configs(_fl_config())
    assert list(cfgs) == ['A', 'B']
    assert cfgs['B']['data']['files'] == ['P2', 'P3']
    for cfg in cfgs.values():
        assert not set(lt.DROPPED_DATA_KEYS) & set(cfg['data'])
        assert cfg['model']['fl'] is False and cfg['model']['epochs'] == 100
        assert cfg['params'] == _fl_config()['params']
    # local park-id codes 0..n_local-1, the embedding covers only the client's parks
    assert preprocessing.static_categories(cfgs['B']) == {'park_id': ['P2', 'P3']}


def test_client_subset_and_unknown_client():
    assert list(lt.client_configs(_fl_config(), ['B'])) == ['B']
    with pytest.raises(ValueError):
        lt.client_configs(_fl_config(), ['C'])
    with pytest.raises(ValueError):
        lt.client_configs({'data': {}, 'model': {}, 'fl': {}})


def test_local_name():
    assert lt.local_name('configs/parks_v1/config_parks_v1_fl_fedgradient_parkid.yaml',
                         'fedgradient') == 'config_parks_v1_local_parkid'
    assert lt.local_name('configs/x/config_abc.yaml', 'fedavg') == 'config_abc_local'


def test_write_configs_roundtrip(tmp_path):
    paths = lt.write_configs(lt.client_configs(_fl_config()), str(tmp_path / 'cfg'), 'config_t_local')
    with open(paths['A'] + '.yaml') as f:
        assert yaml.safe_load(f)['data']['files'] == ['P0', 'P1']
    with pytest.raises(ValueError):
        lt.write_configs(lt.client_configs(_fl_config()), str(tmp_path / 'a.b'), 'x')


def test_parse_gpus():
    assert lt.parse_gpus('0-3', 8) == [0, 1, 2, 3]
    assert lt.parse_gpus('1-2,5', 8) == [1, 2, 5]
    assert lt.parse_gpus(None, 4) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        lt.parse_gpus('0,0', 4)


def test_run_jobs_queue_limits_concurrency_and_survives_failures(tmp_path):
    """6 jobs on 2 GPUs x 1 slot: never more than 2 at once, each sees its own
    CUDA_VISIBLE_DEVICES, a failing job does not stop the rest."""
    script = ("import os, sys, time; print(os.environ['CUDA_VISIBLE_DEVICES']); "
              "time.sleep(0.4); sys.exit(int(sys.argv[1]))")
    jobs = [lt.Job(name=f'j{i}', cmd=[sys.executable, '-c', script, '1' if i == 2 else '0'],
                   log=str(tmp_path / f'j{i}.out')) for i in range(6)]
    lt.run_jobs(jobs, gpus=[3, 5], per_gpu=1, poll=0.05)
    assert [j.returncode for j in jobs] == [0, 0, 1, 0, 0, 0]
    for j in jobs:
        assert open(j.log).read().strip() == str(j.gpu) and j.gpu in (3, 5)
    events = sorted([(j.start, 1) for j in jobs] + [(j.end, -1) for j in jobs])
    level = peak = 0
    for _, d in events:
        level += d
        peak = max(peak, level)
    assert peak <= 2


def _fake_result(path, keys, r2, val_rmse):
    ev = pd.DataFrame({'R^2': r2, 'RMSE': [0.1] * len(keys), 'MAE': [0.05] * len(keys), 'key': keys},
                      index=pd.Index(['TFT'] * len(keys), name='Models'))
    ev.loc['mean'] = ev.mean(numeric_only=True)
    with open(path, 'wb') as f:
        pickle.dump({'config': {'data': {'files': keys}}, 'evaluation': ev,
                     'history': {'val_rmse': val_rmse}}, f)


def test_result_path_and_merge(tmp_path):
    pa, pb = str(tmp_path / 'a.pkl'), str(tmp_path / 'b.pkl')
    _fake_result(pa, ['synth_P0.csv', 'synth_P1.csv'], [0.8, 0.9], [0.3, 0.2, 0.25])
    _fake_result(pb, ['synth_P2.csv'], [0.7], [0.4, 0.35])
    log = tmp_path / 'a.out'
    log.write_text(f"...\nTraining completed! Results saved to: {pa}\n")
    assert lt.result_path_from_log(str(log)) == pa
    assert lt.result_path_from_log(str(tmp_path / 'missing.out')) is None

    merged = lt.merge_results({'A': pa, 'B': pb})
    ev = merged['evaluation']
    rows = ev[ev['key'].notna()]
    assert rows['client_id'].tolist() == ['A', 'A', 'B']
    assert np.isclose(ev.loc['mean', 'R^2'], 0.8)
    assert merged['clients']['A'] == {'result': pa, 'n_stations': 2, 'epochs': 3, 'best_epoch': 2,
                                      'runtime_s': None, 'gpu': None}
    out = str(tmp_path / 'res' / 'local.pkl')
    lt.save_merged(merged, out)
    assert os.path.exists(out) and os.path.exists(out.replace('.pkl', '_clients.json'))
    assert 'R^2 mean 0.8000' in lt.summary_line(merged)
