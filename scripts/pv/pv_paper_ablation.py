#!/usr/bin/env python3
"""P5 of the CNN-LSTM computation order (Viktor 2026-11-17, ~/Work/pv_sim2real/docs/rechenauftrag_cnnlstm_2026-11-17.md):
ablations of the global CNN-LSTM of P1 (scripts/pv/pv_paper_cnnlstm.py; same recipe, seeds, ES protocol and test windows).

Cases (jan = client parks until Jan 2024 -> six meters Feb..May 2024; split1 = client parks J1 -> 56 test parks J2):
  hist_jan, hist_split1   with power history. The windows are issued daily at 09 UTC, so the target of the window
                          issued two days earlier is exactly the measured power of the 48 h before the issue time. The
                          model sees the full 192-step window: four known inputs + power channel (past 96 steps =
                          that power, future 96 = 0) + indicator (1 where the power is observed); missing history
                          (first days, gaps) -> 0 with indicator 0. Step head on the last 96 steps as before.
  ghi_jan, ghi_split1     GHI instead of GTI: known inputs ghi_nwp, ghi_clearsky, ecmwf_ghi, solar_zenith_cos (as the
                          TFT ablation of 09.11.); own x scaler fitted like the HPO one (60 source parks until
                          2024-08-01); masks (gti_nwp > 10, shutdowns) from the GTI data so every case is scored on the
                          same cells.
  gen_<ablation>          generator components (~/Work/pv_sim2real/benchmark_tilt/export_ablation.py): client fleet
                          generated with raw CAMS / without snow / without horizon / full (control), O1 arrays; jan case.

    frcst/bin/python scripts/pv/pv_paper_ablation.py prepare-ghi --output results/parks_pv_v4_30min/paper_cnnlstm_2026-11-17_p5
    frcst/bin/python scripts/pv/pv_paper_ablation.py prepare-gen --output <same> --ablation raw_cams   (per ablation)
    frcst/bin/python scripts/pv/pv_paper_ablation.py launch --output <same> --cases hist_jan,hist_split1,... --gpus 0,1
"""
import argparse
import copy
import json
import logging
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import local_training, preprocessing, pv_parks, tools  # noqa: E402
from utils.data_cache import _fit_global_scaler_x  # noqa: E402
from utils.models import CNNRNN  # noqa: E402
import pv_conformal as base  # noqa: E402
import pv_hpo_local as HL  # noqa: E402
import pv_geometry_forecast as GF  # noqa: E402
import pv_paper_cnnlstm as PC  # noqa: E402
from pv_quantile_finetune import train, snapshot  # noqa: E402
from pv_synthetic_transfer import QS, park_id  # noqa: E402

P1_DIR = Path('results/parks_pv_v4_30min/paper_cnnlstm_2026-11-17_p1')
GHI_KNOWN = ['ghi_nwp', 'ghi_clearsky', 'ecmwf_ghi', 'solar_zenith_cos']
GEN_ROOT = os.path.join(os.environ.get('DATA_ROOT', '/mnt/lambda1/nvme1'), 'synthetic', 'solar', 'generator_ablation_v4', 'benchmark_tilt_2026-11-13')
GEN_ABLATIONS = ('full', 'raw_cams', 'no_snow', 'no_horizon')
CASES = ('hist_jan', 'hist_split1', 'ghi_jan', 'ghi_split1') + tuple(f'gen_{x}' for x in GEN_ABLATIONS)
SEEDS = PC.SEEDS
TWO_DAYS = pd.Timedelta('2D')


# ------------------------------------------------------------------------------------------------------ history

class HistNet(nn.Module):
    """HL.Net over the full 192-step window (known + power + indicator); step head on the last 96 steps."""
    def __init__(self, n_in, n_q, hp):
        super().__init__()
        self.m = CNNRNN(n_features=n_in, output_dim=HL.H, conv_type=hp['conv_type'], rnn_type='lstm',
                        bidirectional=hp['bidirectional'], num_quantiles=n_q,
                        hyperparameters=dict(n_cnn_layers=hp['n_cnn_layers'], filters=hp['filters'],
                                             kernel_size=hp['kernel_size'], units=hp['units'],
                                             n_rnn_layers=hp['n_rnn_layers'], dropout=hp['dropout'],
                                             increase_filters=False, step_output=True))

    def forward(self, observed, known, static=None):
        return self.m(known)


def with_history(X, t, series):
    """X['known'] (n, 192, k) -> (n, 192, k + 2): power of the 48 h before each issue time from series {issue time: y}."""
    n = len(t)
    pw, ind = np.zeros((n, 2 * HL.H), np.float32), np.zeros((n, 2 * HL.H), np.float32)
    for i, t0 in enumerate(pd.DatetimeIndex(t)):
        y = series.get(t0 - TWO_DAYS)
        if y is not None:
            ok = np.isfinite(y)
            pw[i, :HL.H] = np.where(ok, y, 0.0)
            ind[i, :HL.H] = ok
    k = np.asarray(X['known'])
    assert k.shape[1] == 2 * HL.H, k.shape
    return dict(observed=X['observed'], known=np.concatenate([k, pw[..., None], ind[..., None]], axis=-1))


def series_of(*parts):
    """{issue time: target row} from (t, y) pairs."""
    s = {}
    for t, y in parts:
        s.update(zip(pd.DatetimeIndex(t), y))
    return s


# ------------------------------------------------------------------------------------------------------ prepare

def prepare_ghi(out):
    """Client parks (J1), split-1 test parks (J2) and meters (Jul..Jan / Feb..May) with GHI inputs; own scaler."""
    with open(HPO_DIR() / 'meta.pkl', 'rb') as f:
        hm = pickle.load(f)
    cfg = copy.deepcopy(hm['cfg'])
    cfg['params']['known_features'] = GHI_KNOWN
    features = preprocessing.get_features(config=cfg)
    assert features['known'] == GHI_KNOWN and not features['observed'], features
    src_cfg = base.load_cfg('configs/pv_parks/config_pv_v4_central_split1_nohist')
    source = src_cfg['data']['holdout_files']
    (out / 'prepared').mkdir(parents=True, exist_ok=True)
    assert not (out / 'prepared' / 'ghi_clients.pkl').exists()

    def load(ids, c0=cfg):
        c = copy.deepcopy(c0)
        c['data']['files'] = ids
        dfs = preprocessing.get_data(c['data']['path'], c['data']['freq'], c, features)
        for k, df in dfs.items():
            t = pd.DatetimeIndex(df.index.get_level_values('timestamp'))
            dfs[k] = df.loc[t < pd.Timestamp('2025-08-01', tz=t.tz)]
        return dfs, c

    dfs, scfg = load(source)
    sx = _fit_global_scaler_x(dfs, scfg, logging.getLogger(__name__), fit_until='2024-08-01', restrict_train_end=True)
    S = {}
    for key, it in base.windows(dfs, scfg, features, sx, HL.J1, HL.J2).items():
        p = park_id(key)
        if not (HPO_DIR() / 'parks' / f'{p}.pkl').exists():
            continue
        g = HL.load_park(HPO_DIR(), p)                          # GTI data of the HPO: masks and issue times
        t = pd.DatetimeIndex(it['index_test'])
        m = np.asarray(t.isin(g['t2']))
        assert m.sum() == len(g['t2']), (p, m.sum(), len(g['t2']))
        S[p] = dict(X1=it['X_train'], y1=it['y_train'], t1=pd.DatetimeIndex(it['index_train']),
                    X2=HL.take(it['X_test'], m), y2=np.asarray(it['y_test'])[m], t2=t[m], nwp2=g['nwp2'],
                    shut2=np.zeros(g['y2'].shape, bool))
        assert np.allclose(S[p]['y2'], g['y2'], equal_nan=True), p
    del dfs
    dfs, ccfg = load(cfg['data']['files'])
    C = {}
    for key, it in base.windows(dfs, ccfg, features, sx, HL.J1, HL.J2).items():
        t = pd.DatetimeIndex(it['index_train'])
        if len(t):
            C[park_id(key)] = dict(X1=it['X_train'], y1=it['y_train'], t1=t)
    del dfs
    # meters: the level-3 data path (twins of the tilt benchmark, O1 directory), GHI inputs
    mcfg = base.load_cfg(GF.DATA_CFG)
    mcfg['params']['known_features'] = GHI_KNOWN
    root = os.path.join(GF.TWIN_ROOT, 'O1_ours')
    mcfg['data'].update(twin_path=root, meter_path=root, files=[p + '_meter' for p in GF.PARKS], holdout_files=[],
                        twin_files=list(GF.PARKS), meter_files=[p + '_meter' for p in GF.PARKS])
    mdfs = preprocessing.get_data(data_dir=mcfg['data']['path'], config=mcfg, freq=mcfg['data']['freq'], features=features, files_key='files')
    G = PC.meters('O1_ours')
    M = {}
    for key, it in base.windows(mdfs, mcfg, features, sx, GF.TRAIN, GF.TEST).items():
        pv = Path(key).name.removeprefix('synth_').removesuffix('.csv').removesuffix('.parquet').removesuffix('_meter')
        g = G[pv]
        tm, t2 = pd.DatetimeIndex(it['index_train']), pd.DatetimeIndex(it['index_test'])
        assert tm.equals(g['tm']) and t2.equals(g['t2']), pv
        assert np.allclose(np.asarray(it['y_test']), g['y2'], equal_nan=True), pv
        M[pv] = dict(Xm=it['X_train'], ym=it['y_train'], tm=tm, X2=it['X_test'], y2=g['y2'], t2=t2, nwp2=g['nwp2'], shut2=g['shut2'])
    assert sorted(M) == sorted(GF.PARKS), sorted(M)
    for name, obj in (('ghi_clients', C), ('ghi_source', S), ('ghi_meters', M)):
        with open(out / 'prepared' / f'{name}.pkl', 'wb') as f:
            pickle.dump(obj, f)
    cfg['model']['feature_dim'] = tools.get_feature_dim(X=next(iter(C.values()))['X1'])
    with open(out / 'prepared' / 'ghi_meta.pkl', 'wb') as f:
        pickle.dump(dict(cfg=cfg, scaler=sx, features=features), f)
    print(f'GHI: {len(C)} client parks, {len(S)} test parks, {len(M)} meters', flush=True)


def prepare_gen(out, ablation):
    """Client parks of one generator ablation (jan case: train until Jan, P3 windows), HPO scaler."""
    with open(HPO_DIR() / 'meta.pkl', 'rb') as f:
        hm = pickle.load(f)
    cfg, sx = copy.deepcopy(hm['cfg']), hm['scaler']
    cfg['data']['path'] = os.path.join(GEN_ROOT, ablation)
    for f in cfg['data']['files']:
        assert pv_parks.park_source(cfg['data'], f)[0] == cfg['data']['path'], (ablation, f)
    features = preprocessing.get_features(config=cfg)
    name = out / 'prepared' / f'gen_{ablation}_clients.pkl'
    (out / 'prepared').mkdir(parents=True, exist_ok=True)
    assert not name.exists(), name
    dfs = preprocessing.get_data(cfg['data']['path'], cfg['data']['freq'], cfg, features)
    C = {}
    for key, it in base.windows(dfs, cfg, features, sx, PC.P3_TRAIN, PC.P3_TEST).items():
        t = pd.DatetimeIndex(it['index_train'])
        if len(t):
            C[park_id(key)] = dict(X1=it['X_train'], y1=it['y_train'], t1=t)
    with open(name, 'wb') as f:
        pickle.dump(C, f)
    print(f'{ablation}: {len(C)} client parks, {sum(len(d["y1"]) for d in C.values())} windows', flush=True)


def HPO_DIR():
    return GF.HPO_DIR


# ---------------------------------------------------------------------------------------------------------- data

def jan_cut(d):
    return np.asarray(d['t1'] + HL.H * HL.STEP <= PC.JAN_END)


def load(out, name):
    with open(out / 'prepared' / f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


def case_data(out, case):
    """(train (X, y), {park: test dict}, known_dim, hist)"""
    kind, split = case.split('_', 1)
    if kind == 'gen':
        C = load(out, f'gen_{split}_clients')
        tr = HL.join(*[(HL.take(d['X1'], jan_cut(d)), d['y1'][jan_cut(d)]) for d in C.values() if jan_cut(d).any()])
        return tr, PC.test_parks('jan'), 4, False
    if kind == 'ghi':
        C = load(out, 'ghi_clients')
        T = load(out, 'ghi_meters') if split == 'jan' else load(out, 'ghi_source')
        cut = jan_cut if split == 'jan' else (lambda d: np.ones(len(d['t1']), bool))
        tr = HL.join(*[(HL.take(d['X1'], cut(d)), d['y1'][cut(d)]) for d in C.values() if cut(d).any()])
        return tr, T, 4, False
    # hist: the P1 data plus the power channel
    with open(P1_DIR / 'prepared' / 'clients.pkl', 'rb') as f:
        C = pickle.load(f)
    cut = jan_cut if split == 'jan' else (lambda d: np.ones(len(d['t1']), bool))
    parts = []
    for d in C.values():
        m = cut(d)
        if m.any():
            parts.append((with_history(HL.take(d['X1'], m), d['t1'][m], series_of((d['t1'], d['y1']))), d['y1'][m]))
    tr = HL.join(*parts)
    T = {}
    if split == 'jan':
        for p, d in PC.meters('O1_ours').items():
            s = series_of((d['tm'], d['ym']), (d['t2'], d['y2']))
            T[p] = dict(X2=with_history(d['X2'], d['t2'], s), y2=d['y2'], t2=d['t2'], nwp2=d['nwp2'], shut2=d['shut2'])
    else:
        for p, d in PC.test_parks('split1').items():
            g = HL.load_park(HPO_DIR(), p)
            s = series_of((g['t1'], g['y1']), (g['t2'], g['y2']))
            T[p] = dict(d, X2=with_history(d['X2'], d['t2'], s))
    return tr, T, 6, True


# ------------------------------------------------------------------------------------------------------------ run

def run(a):
    out = Path(a.output)
    with open(P1_DIR / 'meta.pkl', 'rb') as f:
        cfg = pickle.load(f)['cfg']
    hp = GF.recipe()
    tr, T, k, hist = case_data(out, a.case)
    parks = sorted(T)
    device, pin = torch.device('cuda:0'), HL.code_pin()
    factory = (lambda: HistNet(k, len(QS), hp)) if hist else (lambda: HL.Net(k, len(QS), hp))  # noqa: E731
    tools.set_seed(a.seed)
    t0 = time.time()
    model, log, _ = train(f'p5_{a.case}', [tr], [dict(X_test=T[p]['X2'], y_test=T[p]['y2']) for p in parks],
                          snapshot(factory()), cfg, hp, device, a.seed, QS, out, factory=factory, lr=hp['lr'],
                          batch_size=hp['batch_size'], weight_decay=hp['weight_decay'], save=False)
    info = dict(best_epoch=log['best_epoch'], epochs=log['epochs'], seconds=round(time.time() - t0, 1), n_train=len(tr[1]))
    print(f'{a.case} seed {a.seed}: {info}', flush=True)
    for d in ('pred', 'rows', 'models'):
        (out / d).mkdir(exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), hp=hp, quantiles=QS, known_dim=k, hist=hist), out / 'models' / f'{a.case}_seed{a.seed}.pt')
    rows = []
    for p in parks:
        m, pred = GF.scores(model, T[p], device)
        np.save(out / 'pred' / f'{a.case}_{p}_seed{a.seed}.npy', pred.astype(np.float32))
        rows.append(dict(arm=a.case, park=p, seed=a.seed, **info, **m, code_pin=pin))
    pd.DataFrame(rows).to_csv(out / 'rows' / f'{a.case}_seed{a.seed}.csv', index=False)


def launch(a):
    out = Path(a.output)
    jobs = [local_training.Job(name=f'{c}_seed{s}', log=str(out / 'logs' / f'{c}_seed{s}.out'),
                               cmd=[sys.executable, __file__, 'run', '--output', str(out), '--case', c, '--seed', str(s)])
            for c in a.cases.split(',') for s in SEEDS]
    gpus = local_training.parse_gpus(a.gpus, torch.cuda.device_count())
    local_training.run_jobs(jobs, gpus, per_gpu=a.per_gpu, cwd=str(Path(__file__).resolve().parents[2]))
    failed = [j.name for j in jobs if j.returncode != 0]
    if failed:
        raise SystemExit(f'failed jobs: {failed}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare-ghi', 'prepare-gen', 'run', 'launch'])
    ap.add_argument('--output', required=True)
    ap.add_argument('--case', choices=CASES)
    ap.add_argument('--cases', default=','.join(CASES))
    ap.add_argument('--ablation', choices=GEN_ABLATIONS)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--per-gpu', type=int, default=3)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    os.chdir(Path(__file__).resolve().parents[2])
    {'prepare-ghi': lambda: prepare_ghi(Path(a.output)), 'prepare-gen': lambda: prepare_gen(Path(a.output), a.ablation),
     'run': lambda: run(a), 'launch': lambda: launch(a)}[a.mode]()


if __name__ == '__main__':
    main()
