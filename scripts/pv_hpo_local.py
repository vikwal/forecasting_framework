#!/usr/bin/env python3
"""Local CNNRNN models per synthetic PV park: CV pilot before the per-park HPO (Viktor 2026-11-13).

Years: J1 2023-07-24..2024-07-31 (training), J2 Aug 2024--Jul 2025 (validation), J3 Aug 2025--Jul 2026 (test, the
additional year of release v4, released for this study by Viktor 2026-11-13). The final model of a configuration trains
on J1+J2 (24 months) with early stopping and evaluation on J3 (agreed ES protocol). Known features as in the partner map
(ICON-D2 09 UTC, ECMWF 00 UTC, 30 min), but the CNNRNN sees only the 96 forecast steps (no power history, so no past
window); seven quantiles, pinball loss. Plan and decisions: ~/Work/pv_sim2real/docs/plan_hpo_local_cnnrnn_2026-11-13.md.

Modes
  prepare  windows of all parks, one pickle per park (x scaler fitted on the 60 source parks until 2024-08-01, as in
           the synthetic transfer study)
  pilot    which selection protocol to use, without touching J3: the setting shifted by one block (pseudo test
           Apr--Jul 2025). Fixed random configurations (same for every park) under variant A (J1 -> Aug 2024..Mar 2025),
           variant B (two expanding 4-month folds, hpo.kfolds_by_dates) and the pseudo-final model F (20 months ->
           Apr--Jul 2025); one job = one park and one chunk of configurations, resumable
  launch   pilot jobs over GPU slots (utils.local_training.run_jobs)

    frcst/bin/python scripts/pv_hpo_local.py prepare --output results/parks_pv_v4_30min/hpo_local_cnnrnn_2026-11-13_v1
    frcst/bin/python scripts/pv_hpo_local.py launch --output <same> --gpus 0-2 --per-gpu 3
"""
import argparse
import copy
import json
import logging
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import hpo, local_training, preprocessing, tools  # noqa: E402
from utils.data_cache import _fit_global_scaler_x  # noqa: E402
from utils.models import CNNRNN  # noqa: E402
import pv_conformal as base  # noqa: E402
from pv_quantile_finetune import train, snapshot  # noqa: E402
from pv_quantile_metrics import metrics  # noqa: E402
from pv_synthetic_transfer import QS, park_id  # noqa: E402

J1 = ('2023-07-24', '2024-07-31 23:30')
J2 = ('2024-08-01', '2025-07-31 23:30')
J3 = ('2025-08-01', '2026-07-31 23:30')
B_BOUNDS = ['2024-08-01', '2024-12-01', '2025-04-01', '2025-08-01']      # variant B: three 4-month blocks of J2
# The pilot does not touch J3 (Viktor 2026-11-13): it mirrors the setting one block earlier. Pseudo test = Apr--Jul 2025,
# selection on Aug 2024--Mar 2025 (A: one block of 8 months, B: two expanding folds), pseudo-final model on J1 plus
# Aug 2024--Mar 2025 (20 months).
PILOT_BOUNDS = ['2024-08-01', '2024-12-01', '2025-04-01']
PILOT_TEST = '2025-04-01'
H = 96
INPUT_STEPS = 96          # CNNRNN input: the 96 forecast steps only (no lagged target, so no past window; Viktor 13.11.)
STEP = pd.Timedelta('30min')
EPOCHS = 200
PILOT_CONFIGS = 24                                                       # per conv type
PILOT_SEED = 2026
PILOT_PER_GROUP = 3                                                      # 3 x 8 groups = 24 pilot parks


def utc(s):
    return pd.Timestamp(s, tz='UTC')


def code_pin():
    return os.environ.get('CODE_PIN') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


# --------------------------------------------------------------------------------------------------------- data

def prepare(out):
    cfg = base.load_cfg('configs/pv_parks/config_pv_v4_central_split1_nohist')
    clients = base.load_cfg('configs/pv_parks/config_pv_v4_fl_clients_split1_nohist')['fl']['clients']
    source, target = cfg['data']['holdout_files'], cfg['data']['files']
    assert len(source) == 60 and len(target) == 166 and not set(source) & set(target)
    cfg['data'].update(train_start=J1[0], train_end=J1[1], test_start=J2[0], test_end=J3[1], data_cutoff=J3[1],
                       strict_split=True, holdout_files=[])
    cfg['model']['tft']['quantiles'] = QS
    cfg['model']['epochs'] = EPOCHS
    assert cfg['params']['observed_features'] == []
    features = preprocessing.get_features(config=cfg)
    group = {p: g for g, ids in clients.items() for p in ids} | {p: 'source' for p in source}

    def load(ids):
        c = copy.deepcopy(cfg)
        c['data']['files'] = ids
        dfs = preprocessing.get_data(c['data']['path'], c['data']['freq'], c, features)
        for k, df in dfs.items():
            t = pd.DatetimeIndex(df.index.get_level_values('timestamp'))
            dfs[k] = df.loc[t < pd.Timestamp('2026-08-01', tz=t.tz)]
        return dfs, c

    (out / 'parks').mkdir(parents=True, exist_ok=False)
    dfs, scfg = load(source)
    sx = _fit_global_scaler_x(dfs, scfg, logging.getLogger(__name__), fit_until='2024-08-01', restrict_train_end=True)
    counts, skipped = {}, []
    for ids in (source, target):
        if ids is target:
            dfs, scfg = load(target)
        items = base.windows(dfs, scfg, features, sx, J1, (J2[0], J3[1]))
        del dfs
        for key, it in items.items():
            pid = park_id(key)
            t1 = pd.DatetimeIndex(it['index_train'])
            t = pd.DatetimeIndex(it['index_test'])
            end = t + H * STEP                                           # end of the target window
            m2 = np.asarray((t >= utc(J2[0])) & (end <= utc('2025-08-01')))
            m3 = np.asarray((t >= utc(J3[0])) & (end <= utc('2026-08-01')))
            assert (t1 + H * STEP <= utc(J2[0])).all()
            d = dict(park=pid, group=group[pid], X1=it['X_train'], y1=it['y_train'], t1=t1)
            for j, m in (('2', m2), ('3', m3)):
                d['X' + j] = {k: v[m] for k, v in it['X_test'].items()}
                d['y' + j], d['t' + j] = it['y_test'][m], t[m]
                d['nwp' + j] = np.asarray(it['nwp_raw_test'])[m]
            n = {j: len(d['y' + j]) for j in '123'}
            if min(n.values()) == 0:
                skipped.append(pid)
                continue
            counts[pid] = dict(group=group[pid], **{f'n{j}': v for j, v in n.items()})
            with open(out / 'parks' / f'{pid}.pkl', 'wb') as f:
                pickle.dump(d, f)
        logging.info('prepared %d parks so far', len(counts))
    cfg['model']['feature_dim'] = tools.get_feature_dim(X=d['X1'])
    with open(out / 'meta.pkl', 'wb') as f:
        pickle.dump(dict(cfg=cfg, scaler=sx, features=features), f)
    meta = dict(J1=J1, J2=J2, J3=J3, quantiles=QS, parks=counts, skipped=sorted(skipped),
                skipped_no_windows=sorted(set(source + target) - set(counts) - set(skipped)),
                code_pin=code_pin(), date=pd.Timestamp.now().isoformat())
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    logging.info('prepared %d parks, %d without J1/J2/J3 windows', len(counts), len(set(source + target) - set(counts)))


def load_park(out, pid):
    with open(out / 'parks' / f'{pid}.pkl', 'rb') as f:
        return pickle.load(f)


def take(X, m):
    return {k: v[m] for k, v in X.items()}


def join(*parts):
    return ({k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]},
            np.concatenate([p[1] for p in parts]))


def folds_b(d, bounds):
    """Variant B: expanding folds over J2 (utils.hpo.kfolds_by_dates); fold i validates the J2 windows issued in
    [b_i, b_i+1) and trains on J1 plus all J2 windows whose targets end before b_i. Returns (train, val, nwp_val)."""
    X, y = join((d['X1'], d['y1']), (d['X2'], d['y2']))
    t = d['t1'].append(d['t2'])
    folds = hpo.kfolds_by_dates([dict(index_train=t, X_train=X, y_train=y)], bounds, H * STEP / pd.Timedelta('1h'))
    out = []
    for (tr, va), lo, hi in zip(folds, bounds, bounds[1:]):
        m = np.asarray((d['t2'] >= utc(lo)) & (d['t2'] < utc(hi)))
        assert np.array_equal(va[1], d['y2'][m], equal_nan=True)         # fold order = J2 order
        out.append((tr, va, d['nwp2'][m]))
    return out


def pilot_split(d):
    """Pilot data without J3: dev = J1 plus the J2 windows whose targets end before PILOT_TEST, test = J2 windows
    issued from PILOT_TEST on."""
    dev = np.asarray(d['t2'] + H * STEP <= utc(PILOT_TEST))
    tst = np.asarray(d['t2'] >= utc(PILOT_TEST))
    part = lambda m: dict(X=take(d['X2'], m), y=d['y2'][m], t=d['t2'][m], nwp=d['nwp2'][m])  # noqa: E731
    dv, ts = part(dev), part(tst)
    dev_park = dict(d, X2=dv['X'], y2=dv['y'], t2=dv['t'], nwp2=dv['nwp'])
    for k in ('X3', 'y3', 't3', 'nwp3'):
        dev_park.pop(k, None)
    return dev_park, ts


# -------------------------------------------------------------------------------------------------- model, space

class Net(nn.Module):
    """utils.models.CNNRNN with one quantile output per forecast step (step_output), TFT call signature."""
    def __init__(self, n_in, n_q, hp):
        super().__init__()
        self.m = CNNRNN(n_features=n_in, output_dim=H, conv_type=hp['conv_type'], rnn_type='lstm',
                        bidirectional=hp['bidirectional'], num_quantiles=n_q,
                        hyperparameters=dict(n_cnn_layers=hp['n_cnn_layers'], filters=hp['filters'],
                                             kernel_size=hp['kernel_size'], units=hp['units'],
                                             n_rnn_layers=hp['n_rnn_layers'], dropout=hp['dropout'],
                                             increase_filters=False, step_output=True))

    def forward(self, observed, known, static=None):
        return self.m(known[:, -INPUT_STEPS:])


def space(trial, conv_type):
    """CNNRNN search space (plan section 4). Odd kernels only: 'same' padding with an even kernel lengthens the sequence
    by one step per layer and shifts the step head. Dropout acts only between LSTM layers."""
    hp = dict(conv_type=conv_type,
              n_cnn_layers=trial.suggest_int('n_cnn_layers', 2, 10 if conv_type == 'cnn' else 7),
              filters=trial.suggest_int('filters', 16, 256, log=True),
              kernel_size=trial.suggest_categorical('kernel_size', [3, 5, 7]),
              n_rnn_layers=trial.suggest_int('n_rnn_layers', 1, 3),
              units=trial.suggest_int('units', 16, 256, log=True),
              bidirectional=trial.suggest_categorical('bidirectional', [False, True]),
              lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
              batch_size=trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
              weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True))
    hp['dropout'] = trial.suggest_float('dropout', 0.0, 0.5) if hp['n_rnn_layers'] >= 2 else 0.0
    return hp


def pilot_configs():
    """PILOT_CONFIGS random configurations per conv type, reproducible (RandomSampler, fixed seed)."""
    cfgs = []
    for conv in ('cnn', 'tcn'):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=PILOT_SEED + (conv == 'tcn')))
        for _ in range(PILOT_CONFIGS):
            cfgs.append(dict(id=len(cfgs), **space(study.ask(), conv)))
    return cfgs


# ------------------------------------------------------------------------------------------------------ training

def mse(pred, y):
    return float(np.mean((pred[..., 0] - y) ** 2))


def fit(cfg, hp, train_data, val, seed, device, out, loss='pinball'):
    """Train on train_data = (X, y) with early stopping on val = (X, y): all-cell pinball over the seven quantiles (as
    the earlier studies), or with loss='mse' a point model (one output) trained and early-stopped on MSE."""
    if loss == 'mse':
        cfg = copy.deepcopy(cfg)
        cfg['model']['tft']['quantiles'] = None                         # tools.make_criterion -> MSELoss
    n_q = len(QS) if loss == 'pinball' else 1
    factory = lambda: Net(cfg['model']['feature_dim']['known_dim'], n_q, hp)  # noqa: E731
    tools.set_seed(seed)
    state = snapshot(factory())
    t0 = time.time()
    model, log, _ = train('local_hpo', [train_data], [dict(X_test=val[0], y_test=val[1])], state, cfg, hp, device,
                          seed, QS if loss == 'pinball' else [0.5], out, factory=factory, lr=hp['lr'],
                          batch_size=hp['batch_size'], weight_decay=hp['weight_decay'], save=False,
                          val_fn=mse if loss == 'mse' else None)
    info = dict(best_epoch=log['best_epoch'], epochs=log['epochs'], seconds=round(time.time() - t0, 1),
                n_params=sum(p.numel() for p in model.parameters()), n_train=len(train_data[1]))
    return model, info


def evaluate(model, X, y, nwp, device):
    pred = base.predict(model, X, device)
    return metrics(pred, y, QS, (nwp > 10) & np.isfinite(y))


METRIC_COLS = ['n_day', 'pinball_all', 'pinball_day_sorted', 'crossing_rate', 'median_rmse', 'median_r2'] + [
    f'{m}_{lv}' for lv in (90, 80, 50) for m in ('coverage', 'width', 'interval_score')]
ROW_COLS = ['park', 'group', 'config', 'conv_type', 'variant', 'score', 'best_epoch', 'epochs', 'seconds', 'n_params',
            'n_train', 'a_test', *METRIC_COLS, 'code_pin']                # one column set for A, B and F rows


def pilot(a):
    out = Path(a.output)
    with open(out / 'meta.pkl', 'rb') as f:
        cfg = pickle.load(f)['cfg']
    cfgs = json.loads((out / 'pilot' / 'configs.json').read_text())
    lo, hi = map(int, a.configs.split('-'))
    d, ts = pilot_split(load_park(out, a.park))                           # J3 is dropped here and never read
    path = out / 'pilot' / 'rows' / f'{a.park}_c{lo:02d}-{hi:02d}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set(map(tuple, pd.read_csv(path)[['config', 'variant']].values)) if path.exists() else set()
    device = torch.device('cuda:0')
    pin = code_pin()
    folds = folds_b(d, PILOT_BOUNDS)

    def record(hp, variant, info, score, extra=None):
        row = dict(park=a.park, group=d['group'], config=hp['id'], conv_type=hp['conv_type'], variant=variant,
                   score=score, **info, **(extra or {}), code_pin=pin)
        unknown = set(row) - set(ROW_COLS)
        assert not unknown, unknown
        pd.DataFrame([row]).reindex(columns=ROW_COLS).to_csv(path, mode='a', header=not path.exists(), index=False)
        print(f"{a.park} config {hp['id']} {variant} score {score:.6f} epoch {info['best_epoch']}/{info['epochs']} "
              f"{info['seconds']} s", flush=True)

    for hp in cfgs[lo:hi + 1]:
        seed = 42 + hp['id']
        if (hp['id'], 'A') not in done:                                   # J1 -> Aug 2024..Mar 2025
            model, info = fit(cfg, hp, (d['X1'], d['y1']), (d['X2'], d['y2']), seed, device, out)
            test = evaluate(model, ts['X'], ts['y'], ts['nwp'], device)['pinball_day']
            record(hp, 'A', info, evaluate(model, d['X2'], d['y2'], d['nwp2'], device)['pinball_day'],
                   dict(a_test=test))
        for i, (tr, va, nwp) in enumerate(folds, 1):
            if (hp['id'], f'B{i}') not in done:
                model, info = fit(cfg, hp, tr, va, seed, device, out)
                record(hp, f'B{i}', info, evaluate(model, *va, nwp, device)['pinball_day'])
        if (hp['id'], 'F') not in done:                                   # pseudo-final: 20 months -> Apr..Jul 2025
            model, info = fit(cfg, hp, join((d['X1'], d['y1']), (d['X2'], d['y2'])), (ts['X'], ts['y']), seed,
                              device, out)
            m = evaluate(model, ts['X'], ts['y'], ts['nwp'], device)
            record(hp, 'F', info, m['pinball_day'], {k: v for k, v in m.items() if k != 'pinball_day'})


# Loss comparison (Viktor 13.11.): the partner-map recipe trained once with pinball (median as point forecast) and once
# with MSE (one output), pseudo-final setting of the pilot (20 months -> Apr..Jul 2025, no J3), seeds 42..44.
RECIPE = dict(conv_type='cnn', n_cnn_layers=5, filters=64, kernel_size=3, n_rnn_layers=2, units=64, dropout=0.1,
              bidirectional=False, lr=3e-4, batch_size=64, weight_decay=0.0)
LOSSCMP_SEEDS = (42, 43, 44)


def point_metrics(p, y, nwp):
    """Point forecast p (n, horizon), clipped at 0, on daytime cells: RMSE, MAE, bias, R2."""
    m = (nwp > 10) & np.isfinite(y)
    e = np.maximum(p[m], 0) - y[m]
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                r2=float(1 - np.sum(e ** 2) / np.sum((y[m] - y[m].mean()) ** 2)))


def losscmp(a):
    out = Path(a.output)
    with open(out / 'meta.pkl', 'rb') as f:
        cfg = pickle.load(f)['cfg']
    d, ts = pilot_split(load_park(out, a.park))                           # J3 is dropped here and never read
    path = out / 'losscmp' / 'rows' / f'{a.park}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set(map(tuple, pd.read_csv(path)[['loss', 'seed']].values)) if path.exists() else set()
    device, pin = torch.device('cuda:0'), code_pin()
    train_data = join((d['X1'], d['y1']), (d['X2'], d['y2']))
    for seed in LOSSCMP_SEEDS:
        for loss in ('pinball', 'mse'):
            if (loss, seed) in done:
                continue
            model, info = fit(cfg, RECIPE, train_data, (ts['X'], ts['y']), seed, device, out, loss=loss)
            pred = base.predict(model, ts['X'], device)
            point = np.sort(pred, axis=-1)[..., QS.index(.5)] if loss == 'pinball' else pred[..., 0]
            row = dict(park=a.park, group=d['group'], loss=loss, seed=seed, **info,
                       **point_metrics(point, ts['y'], ts['nwp']), code_pin=pin)
            if loss == 'pinball':
                q = metrics(pred, ts['y'], QS, (ts['nwp'] > 10) & np.isfinite(ts['y']))
                row.update(pinball_day=q['pinball_day'], coverage_90=q['coverage_90'])
            pd.DataFrame([row]).reindex(columns=LOSSCMP_COLS).to_csv(path, mode='a', header=not path.exists(),
                                                                     index=False)
            print(f"{a.park} {loss} seed {seed} rmse {row['rmse']:.5f} mae {row['mae']:.5f}", flush=True)


LOSSCMP_COLS = ['park', 'group', 'loss', 'seed', 'best_epoch', 'epochs', 'seconds', 'n_params', 'n_train', 'rmse', 'mae',
                'bias', 'r2', 'pinball_day', 'coverage_90', 'code_pin']


def pilot_parks(out):
    """PILOT_PER_GROUP parks per client group N0..R3 (target pool) with a full J1 year, drawn with a fixed seed."""
    parks = json.loads((out / 'meta.json').read_text())['parks']
    full = max(v['n1'] for v in parks.values())
    rng = np.random.default_rng(PILOT_SEED)
    groups = sorted({v['group'] for v in parks.values()} - {'source'})
    return [str(p) for g in groups for p in rng.choice(
        sorted(p for p, v in parks.items() if v['group'] == g and v['n1'] == full), PILOT_PER_GROUP, replace=False)]


def launch(a):
    out = Path(a.output)
    if a.stage == 'losscmp':
        return launch_losscmp(a, out)
    if a.stage in ('hpo', 'eval'):
        return launch_split_stage(a, out)
    (out / 'pilot').mkdir(exist_ok=True)
    cpath = out / 'pilot' / 'configs.json'
    if not cpath.exists():
        cpath.write_text(json.dumps(pilot_configs(), indent=1))
    ppath = out / 'pilot' / 'parks.json'
    if not ppath.exists():
        ppath.write_text(json.dumps(pilot_parks(out)))
    parks, n = json.loads(ppath.read_text()), len(json.loads(cpath.read_text()))
    edges = np.linspace(0, n, a.chunks + 1).astype(int)
    jobs = [local_training.Job(name=f'{p}_c{lo}', log=str(out / 'pilot' / 'logs' / f'{p}_c{lo:02d}.out'),
                               cmd=[sys.executable, __file__, 'pilot', '--output', str(out), '--park', p,
                                    '--configs', f'{lo}-{hi - 1}'])
            for lo, hi in zip(edges, edges[1:]) for p in parks]
    gpus = local_training.parse_gpus(a.gpus, torch.cuda.device_count())
    logging.info('%d pilot jobs (%d parks x %d chunks) on GPUs %s x %d', len(jobs), len(parks), a.chunks, gpus,
                 a.per_gpu)
    local_training.run_jobs(jobs, gpus, per_gpu=a.per_gpu, cwd=str(Path(__file__).resolve().parents[1]))
    failed = [j.name for j in jobs if j.returncode != 0]
    (out / 'pilot' / 'manifest.json').write_text(json.dumps(
        {j.name: dict(gpu=j.gpu, returncode=j.returncode, seconds=round(j.end - j.start)) for j in jobs}, indent=1))
    if failed:
        raise SystemExit(f'failed pilot jobs: {failed}')
    (out / 'pilot' / 'COMPLETE').write_text(pd.Timestamp.now().isoformat())


def launch_split_stage(a, out):
    """hpo: a.workers Optuna workers per fold study; eval: one job per park of a.parks (test | pool | fold<k>)."""
    me = [sys.executable, __file__]
    if a.stage == 'hpo':
        jobs = [local_training.Job(name=f'fold{k}_w{w}', log=str(out / 'hpo' / 'logs' / f'fold{k}_w{w}.out'),
                                   cmd=me + ['hpo', '--output', str(out), '--fold', str(k), '--worker', str(w),
                                             '--trials', str(a.trials)])
                for w in range(a.workers) for k in (1, 2, 3)]
    else:
        split = load_split(out)
        parks = (split['test'] if a.parks == 'test' else sum(split['folds'].values(), []) if a.parks == 'pool'
                 else split['folds'][a.parks.removeprefix('fold')])
        jobs = [local_training.Job(name=p, log=str(out / 'eval' / a.name / 'logs' / f'{p}.out'),
                                   cmd=me + ['eval', '--output', str(out), '--park', p, '--name', a.name,
                                             '--configs-json', a.configs_json, '--protocol', a.protocol,
                                             '--seeds', *map(str, a.seeds)])
                for p in parks]
    gpus = local_training.parse_gpus(a.gpus, torch.cuda.device_count())
    logging.info('%d %s jobs on GPUs %s x %d', len(jobs), a.stage, gpus, a.per_gpu)
    local_training.run_jobs(jobs, gpus, per_gpu=a.per_gpu, cwd=str(Path(__file__).resolve().parents[1]))
    failed = [j.name for j in jobs if j.returncode != 0]
    stage_dir = out / 'hpo' if a.stage == 'hpo' else out / 'eval' / a.name
    (stage_dir / 'manifest.json').write_text(json.dumps(
        {j.name: dict(gpu=j.gpu, returncode=j.returncode, seconds=round(j.end - j.start)) for j in jobs}, indent=1))
    if failed:
        raise SystemExit(f'failed {a.stage} jobs: {failed}')
    (stage_dir / 'COMPLETE').write_text(pd.Timestamp.now().isoformat())


def launch_losscmp(a, out):
    parks = json.loads((out / 'pilot' / 'parks.json').read_text())         # same 24 parks as the pilot
    jobs = [local_training.Job(name=p, log=str(out / 'losscmp' / 'logs' / f'{p}.out'),
                               cmd=[sys.executable, __file__, 'losscmp', '--output', str(out), '--park', p])
            for p in parks]
    gpus = local_training.parse_gpus(a.gpus, torch.cuda.device_count())
    local_training.run_jobs(jobs, gpus, per_gpu=a.per_gpu, cwd=str(Path(__file__).resolve().parents[1]))
    failed = [j.name for j in jobs if j.returncode != 0]
    if failed:
        raise SystemExit(f'failed losscmp jobs: {failed}')
    (out / 'losscmp' / 'COMPLETE').write_text(pd.Timestamp.now().isoformat())


# ------------------------------------------------------------------------------------- shared HPO over park folds

HPO_TRIALS = 50
PRUNE_EVERY = 10          # report the running mean after every 10 parks (same park order in every trial)
STUDY = 'pv_hpo_local_cnnrnn_2026-11-13_v1_fold{k}'


def space_v2(trial):
    """Search space after the pilot (plan section 11): weight decay and lr capped, at most 6 conv layers; conv type
    and bidirectional as search parameters."""
    conv = trial.suggest_categorical('conv_type', ['cnn', 'tcn'])
    hp = dict(conv_type=conv,
              n_cnn_layers=trial.suggest_int('n_cnn_layers', 2, 6),
              filters=trial.suggest_int('filters', 16, 256, log=True),
              kernel_size=trial.suggest_categorical('kernel_size', [3, 5, 7]),
              n_rnn_layers=trial.suggest_int('n_rnn_layers', 1, 3),
              units=trial.suggest_int('units', 16, 256, log=True),
              bidirectional=trial.suggest_categorical('bidirectional', [False, True]),
              lr=trial.suggest_float('lr', 1e-4, 2e-3, log=True),
              batch_size=trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
              weight_decay=trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True))
    hp['dropout'] = trial.suggest_float('dropout', 0.0, 0.5) if hp['n_rnn_layers'] >= 2 else 0.0
    return hp


def load_split(out):
    return json.loads((out / 'split.json').read_text())


def hpo_parks(split, k):
    """Parks the HPO of fold k optimises on: the two other folds, in a fixed shuffled order."""
    parks = sorted(p for j in ('1', '2', '3') if j != str(k) for p in split['folds'][j])
    return [str(p) for p in np.random.default_rng(PILOT_SEED + k).permutation(parks)]


def no_j3(d):
    return {k: v for k, v in d.items() if not k.endswith('3')}


def hpo_worker(a):
    """One Optuna worker of the fold-k study: each trial trains the local model of every HPO park (J1 -> J2, variant
    A) and is scored by the mean daytime pinball over the parks; median pruning on the running mean."""
    out, k = Path(a.output), a.fold
    with open(out / 'meta.pkl', 'rb') as f:
        cfg = pickle.load(f)['cfg']
    parks = hpo_parks(load_split(out), k)[:a.limit_parks]                # limit_parks: smoke tests only
    data = {p: no_j3(load_park(out, p)) for p in parks}
    study = hpo.create_or_load_study(
        str(out / 'hpo' / 'studies.db'), STUDY.format(k=k),
        pruning_config=dict(use_median_pruner=True, n_startup_trials=10, n_warmup_steps=0, interval_steps=1),
        config=dict(hpo=dict(objectives=[dict(metric='pinball_day', direction='minimize')]),
                    params=dict(random_seed=42)))
    path = out / 'hpo' / f'fold{k}' / f'rows_w{a.worker}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    device, pin, failures = torch.device('cuda:0'), code_pin(), 0
    while len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))) < a.trials:
        trial = study.ask()
        hp, seed, scores = space_v2(trial), 42 + trial.number, []
        try:
            for i, p in enumerate(parks):
                d = data[p]
                model, info = fit(cfg, hp, (d['X1'], d['y1']), (d['X2'], d['y2']), seed, device, out)
                scores.append(evaluate(model, d['X2'], d['y2'], d['nwp2'], device)['pinball_day'])
                pd.DataFrame([dict(fold=k, trial=trial.number, park=p, score=scores[-1], **info, code_pin=pin)]).to_csv(
                    path, mode='a', header=not path.exists(), index=False)
                if (i + 1) % PRUNE_EVERY == 0 and i + 1 < len(parks):
                    trial.report(float(np.mean(scores)), step=i + 1)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            study.tell(trial, float(np.mean(scores)))
            print(f'fold {k} trial {trial.number} mean pinball {np.mean(scores):.6f} {json.dumps(hp)}', flush=True)
            failures = 0
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            print(f'fold {k} trial {trial.number} pruned after {len(scores)} parks', flush=True)
        except Exception as e:                                            # e.g. a nonfinite validation loss
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f'fold {k} trial {trial.number} failed: {e}', flush=True)
            failures += 1
            if failures >= 10:
                raise


def candidates(a):
    """Top-3 completed trials of each fold study plus the earlier recipe -> hpo/candidates.json."""
    out = Path(a.output)
    cands = [dict(id='recipe', **RECIPE)]
    for k in (1, 2, 3):
        study = hpo.load_study(str(out / 'hpo' / 'studies.db'), STUDY.format(k=k), required=True)
        done = sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE), key=lambda t: t.value)
        for t in done[:3]:
            hp = dict(t.params)
            hp.setdefault('dropout', 0.0)
            cands.append(dict(id=f'f{k}_t{t.number}', fold=k, hpo_value=t.value, **hp))
    path = out / 'hpo' / 'candidates.json'
    if path.exists():
        raise SystemExit(f'{path} exists')
    path.write_text(json.dumps(cands, indent=1))
    print(json.dumps(cands, indent=1))


def evaluate_configs(a):
    """Configurations x seeds on one park. protocol A: J1 -> J2 (pool parks). protocol final: J1+J2 -> J3 with early
    stopping on J3 (agreed ES protocol); only allowed for the test parks of split.json."""
    out = Path(a.output)
    with open(out / 'meta.pkl', 'rb') as f:
        cfg = pickle.load(f)['cfg']
    split = load_split(out)
    if a.protocol == 'final':
        assert a.park in split['test'], 'J3 is only evaluated on test parks'
        d = load_park(out, a.park)
    else:
        assert a.park not in split['test'], 'test parks are not used before the final evaluation'
        d = no_j3(load_park(out, a.park))
    cfgs = json.loads(Path(a.configs_json).read_text())
    path = out / 'eval' / a.name / 'rows' / f'{a.park}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set(map(tuple, pd.read_csv(path)[['config', 'seed']].astype(str).values)) if path.exists() else set()
    device, pin = torch.device('cuda:0'), code_pin()
    for hp in cfgs:
        for seed in a.seeds:
            if (str(hp['id']), str(seed)) in done:
                continue
            if a.protocol == 'final':
                tr, va, nwp = join((d['X1'], d['y1']), (d['X2'], d['y2'])), (d['X3'], d['y3']), d['nwp3']
            else:
                tr, va, nwp = (d['X1'], d['y1']), (d['X2'], d['y2']), d['nwp2']
            model, info = fit(cfg, hp, tr, va, seed, device, out)
            m = evaluate(model, *va, nwp, device)
            point = np.sort(base.predict(model, va[0], device), axis=-1)[..., QS.index(.5)]
            row = dict(park=a.park, group=d['group'], protocol=a.protocol, config=hp['id'], seed=seed, **info, **m,
                       **{f'point_{k}': v for k, v in point_metrics(point, va[1], nwp).items()}, code_pin=pin)
            pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)
            print(f"{a.park} {hp['id']} seed {seed} pinball {m['pinball_day']:.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'pilot', 'losscmp', 'hpo', 'candidates', 'eval', 'launch'])
    ap.add_argument('--stage', choices=['pilot', 'losscmp', 'hpo', 'eval'], default='pilot', help='what launch starts')
    ap.add_argument('--fold', type=int, choices=[1, 2, 3])
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8, help='Optuna workers per fold study (launch --stage hpo)')
    ap.add_argument('--trials', type=int, default=HPO_TRIALS, help='completed trials per fold study')
    ap.add_argument('--name', help='evaluation name (eval)')
    ap.add_argument('--configs-json', help='configurations to evaluate (eval)')
    ap.add_argument('--protocol', choices=['A', 'final'], default='A')
    ap.add_argument('--parks', default='pool', help='test | pool | fold<k> (launch --stage eval)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--limit-parks', type=int, default=None, help='smoke tests: first n HPO parks only')
    ap.add_argument('--output', required=True)
    ap.add_argument('--park')
    ap.add_argument('--configs', help="'lo-hi' (inclusive) indices into pilot/configs.json")
    ap.add_argument('--gpus', default=None)
    ap.add_argument('--per-gpu', type=int, default=3)
    ap.add_argument('--chunks', type=int, default=3, help='configuration chunks per park (jobs = parks x chunks)')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.set_num_threads(4)
    if a.mode == 'prepare':
        prepare(Path(a.output))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; refusing accidental CPU training')
    if a.mode == 'candidates':
        candidates(a)
    elif a.mode in ('pilot', 'losscmp', 'hpo', 'eval'):
        logging.getLogger().setLevel(logging.WARNING)                     # train() logs every epoch
        dict(pilot=pilot, losscmp=losscmp, hpo=hpo_worker, eval=evaluate_configs)[a.mode](a)
    else:
        launch(a)


if __name__ == '__main__':
    main()
