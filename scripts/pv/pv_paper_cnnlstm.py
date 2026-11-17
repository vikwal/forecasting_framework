#!/usr/bin/env python3
"""Paper experiments with the CNN-LSTM (Viktor 2026-11-17, ~/Work/pv_sim2real/docs/rechenauftrag_cnnlstm_2026-11-17.md).

P1, application: one global CNNRNN (recipe f3_t64 of the local HPO, step output, seven quantiles, pinball, no power
history) pre-trained on the 166 synthetic client parks of release parks_pv_v4 (geometry O1, store v4), then applied
zero-shot. Two cases:
  jan     train on the client parks 2023-07-24..2024-01-31; early stopping and test on the six Trianel meters
          Feb..May 2024 (pooled, agreed protocol; the meter inputs are those of level 3 = O1 arrays,
          results/trianel/geometry_forecast_2026-11-16_v2/prepared/O1_ours.pkl)
  split1  train on the client parks 2023-07-24..2024-07-31; early stopping and test on the 56 synthetic test parks of
          split 1, Aug 2024..Jul 2025 (J2 of the local HPO; J3 stays untouched)
x scaler, windows and test parks are those of the local HPO (results/parks_pv_v4_30min/hpo_local_cnnrnn_2026-11-13_v1),
so P1 is comparable with the local models and with level 3.
P2, LOPO (case lopo_<pv>): the same model trained on the other five meters 2023-07-24..2024-01-31 (meter inputs of
level 3), early stopping and test on the held-out meter Feb..May 2024; one job per held-out park and seed.
P3, geometry -> application (case p3_<variant>): as jan, but pre-trained on one synthetic fleet per geometry variant
(~/Work/pv_sim2real/benchmark_tilt/export_ew.py --fleet: release chain with the variant's arrays, the gti inputs from
the same arrays) and tested on the meters with that variant's inputs (level-3 prepared/<variant>.pkl):
    prepare --output <p3 dir> --variant <variant>      (once per variant)
    launch  --output <p3 dir> --p3
Batch: the recipe's 16; --batch 64 scales the learning rate linearly (Viktor 2026-11-17: allowed if 16 is too slow).

    frcst/bin/python scripts/pv/pv_paper_cnnlstm.py prepare --output results/parks_pv_v4_30min/paper_cnnlstm_2026-11-17_p1
    frcst/bin/python scripts/pv/pv_paper_cnnlstm.py run --output <same> --case jan --seed 42 --max-epochs 1   (timing)
    frcst/bin/python scripts/pv/pv_paper_cnnlstm.py launch --output <same> --gpus 0,1 --per-gpu 3
    frcst/bin/python scripts/pv/pv_paper_cnnlstm.py collect --output <same>
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import local_training, preprocessing, pv_parks, tools  # noqa: E402
import pv_conformal as base  # noqa: E402
import pv_hpo_local as HL  # noqa: E402
import pv_geometry_forecast as GF  # noqa: E402
from pv_quantile_finetune import train, snapshot  # noqa: E402
from pv_synthetic_transfer import QS, park_id  # noqa: E402

HPO_DIR = GF.HPO_DIR
METERS = Path('results/trianel/geometry_forecast_2026-11-16_v2/prepared/O1_ours.pkl')
J1 = HL.J1
JAN_END = pd.Timestamp('2024-02-01', tz='UTC')            # jan case: targets end before Feb 2024
FLEET_ROOT = os.path.join(os.environ.get('DATA_ROOT', '/mnt/lambda1/nvme1'), 'synthetic', 'solar', 'benchmark_tilt_fleet', 'benchmark_tilt_2026-11-13')
P3_VARIANTS = ('O1_ours', 'A1_register', 'A2_fixed20', 'B3_ndsm_gradient', 'O6_largest_array')
P3_TRAIN, P3_TEST = ('2023-07-24', '2024-01-31 23:30'), ('2024-02-01', '2024-05-31')
CASES = ('jan', 'split1')
METER_PARKS = ('pv1', 'pv2', 'pv3', 'pv5', 'pv6', 'pv7')
ALL_CASES = CASES + tuple(f'lopo_{p}' for p in METER_PARKS) + tuple(f'p3_{v}' for v in P3_VARIANTS)
SEEDS = (42, 43, 44)


def prepare(out, variant=None):
    """Training windows of the 166 client parks over J1 (the jan case cuts them later); same scaler as the HPO.
    variant: read the client parks from that variant's fleet export instead of the release (P3)."""
    with open(HPO_DIR / 'meta.pkl', 'rb') as f:
        hm = pickle.load(f)
    cfg, sx = copy.deepcopy(hm['cfg']), hm['scaler']
    assert len(cfg['data']['files']) == 166 and not cfg['data']['holdout_files'] and cfg['params']['observed_features'] == []
    if variant:
        cfg['data']['path'] = os.path.join(FLEET_ROOT, variant)
        for f in cfg['data']['files']:                          # every park must come from the variant directory
            assert pv_parks.park_source(cfg['data'], f)[0] == cfg['data']['path'], (variant, f)
    features = preprocessing.get_features(config=cfg)
    assert features['known'] == hm['features']['known']
    name = f'clients_{variant}.pkl' if variant else 'clients.pkl'
    (out / 'prepared').mkdir(parents=True, exist_ok=True)
    assert not (out / 'prepared' / name).exists(), name
    dfs = preprocessing.get_data(cfg['data']['path'], cfg['data']['freq'], cfg, features)
    # the variant fleets end 2024-06-30: train until Jan, test window Feb..May (the generator drops a park without test
    # windows); the release keeps J1 / J2 as the HPO
    items = base.windows(dfs, cfg, features, sx, *((P3_TRAIN, P3_TEST) if variant else (J1, HL.J2)))
    C = {}
    for key, it in items.items():
        t = pd.DatetimeIndex(it['index_train'])
        if len(t) == 0:
            continue
        assert (t + HL.H * HL.STEP <= (JAN_END if variant else HL.utc(HL.J2[0]))).all()
        C[park_id(key)] = dict(X1=it['X_train'], y1=it['y_train'], t1=t)
    with open(out / 'prepared' / name, 'wb') as f:
        pickle.dump(C, f)
    cfg['model']['feature_dim'] = tools.get_feature_dim(X=next(iter(C.values()))['X1'])
    with open(out / 'meta.pkl', 'wb') as f:
        pickle.dump(dict(cfg=cfg), f)
    src = sorted(p for p, v in json.loads((HPO_DIR / 'meta.json').read_text())['parks'].items() if v['group'] == 'source')
    n_jan = {p: int((d['t1'] + HL.H * HL.STEP <= JAN_END).sum()) for p, d in C.items()}
    meta = dict(clients=sorted(C), n_train_split1={p: len(d['y1']) for p, d in C.items()}, n_train_jan=n_jan,
                test_parks_split1=src, meters=str(METERS), recipe=GF.recipe(), seeds=SEEDS, scaler=str(HPO_DIR / 'meta.pkl'),
                code_pin=HL.code_pin(), date=pd.Timestamp.now().isoformat())
    (out / (f'meta_{variant}.json' if variant else 'meta.json')).write_text(json.dumps(dict(meta, data_path=cfg['data']['path']), indent=1))
    print(f"{len(C)} client parks, windows split1 {sum(meta['n_train_split1'].values())}, jan {sum(n_jan.values())}, "
          f"{sum(v > 0 for v in n_jan.values())} parks with jan windows; {len(src)} split-1 test parks", flush=True)


def meters(variant='O1_ours'):
    with open(METERS.parent / f'{variant}.pkl', 'rb') as f:
        return pickle.load(f)


def test_parks(case):
    """{park: dict(X2, y2, nwp2, shut2)} of the evaluation window."""
    if case.startswith('lopo_'):
        d = meters()[case.removeprefix('lopo_')]
        return {case.removeprefix('lopo_'): dict(X2=d['X2'], y2=d['y2'], t2=d['t2'], nwp2=d['nwp2'], shut2=d['shut2'])}
    if case == 'jan' or case.startswith('p3_'):
        D = meters(case.removeprefix('p3_') if case.startswith('p3_') else 'O1_ours')
        return {p: dict(X2=d['X2'], y2=d['y2'], t2=d['t2'], nwp2=d['nwp2'], shut2=d['shut2']) for p, d in D.items()}
    meta = json.loads((HPO_DIR / 'meta.json').read_text())
    T = {}
    for p in sorted(k for k, v in meta['parks'].items() if v['group'] == 'source'):
        d = HL.load_park(HPO_DIR, p)
        T[p] = dict(X2=d['X2'], y2=d['y2'], t2=d['t2'], nwp2=d['nwp2'], shut2=np.zeros(d['y2'].shape, bool))
    return T


def train_data(out, case):
    if case.startswith('lopo_'):
        D = meters()
        return HL.join(*[(D[p]['Xm'], D[p]['ym']) for p in METER_PARKS if p != case.removeprefix('lopo_')])
    name = f"clients_{case.removeprefix('p3_')}.pkl" if case.startswith('p3_') else 'clients.pkl'
    with open(out / 'prepared' / name, 'rb') as f:
        C = pickle.load(f)
    parts = []
    for p in sorted(C):
        d = C[p]
        m = np.asarray(d['t1'] + HL.H * HL.STEP <= JAN_END) if case != 'split1' else np.ones(len(d['t1']), bool)
        if m.any():
            parts.append((HL.take(d['X1'], m), d['y1'][m]))
    return HL.join(*parts)


def run(a):
    out = Path(a.output)
    with open(out / 'meta.pkl', 'rb') as f:
        cfg = pickle.load(f)['cfg']
    hp = GF.recipe()
    if a.batch and a.batch != hp['batch_size']:
        hp['lr'] *= a.batch / hp['batch_size']                       # linear scaling rule
        hp['batch_size'] = a.batch
    if a.max_epochs:
        cfg['model']['epochs'] = a.max_epochs
    device, pin = torch.device('cuda:0'), HL.code_pin()
    tr = train_data(out, a.case)
    T = test_parks(a.case)
    parks = sorted(T)
    factory = lambda: HL.Net(cfg['model']['feature_dim']['known_dim'], len(QS), hp)  # noqa: E731
    tools.set_seed(a.seed)
    t0 = time.time()
    model, log, _ = train(f'global_{a.case}', [tr], [dict(X_test=T[p]['X2'], y_test=T[p]['y2']) for p in parks],
                          snapshot(factory()), cfg, hp, device, a.seed, QS, out, factory=factory, lr=hp['lr'],
                          batch_size=hp['batch_size'], weight_decay=hp['weight_decay'], save=False)
    info = dict(best_epoch=log['best_epoch'], epochs=log['epochs'], seconds=round(time.time() - t0, 1),
                n_train=len(tr[1]), batch=hp['batch_size'], lr=hp['lr'])
    print(f"{a.case} seed {a.seed}: {info}", flush=True)
    if a.max_epochs:                                                   # timing run: nothing is stored
        return
    tag = f'{a.case}_b{hp["batch_size"]}'
    (out / 'pred').mkdir(exist_ok=True)
    (out / 'models').mkdir(exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), hp=hp, quantiles=QS), out / 'models' / f'{tag}_seed{a.seed}.pt')
    rows = []
    for p in parks:
        m, pred = GF.scores(model, T[p], device)
        np.save(out / 'pred' / f'{tag}_{p}_seed{a.seed}.npy', pred.astype(np.float32))
        rows.append(dict(arm=f'global_{tag}', case=a.case, park=p, seed=a.seed, **info, **m, code_pin=pin))
    path = out / 'rows' / f'{tag}_seed{a.seed}.csv'
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def launch(a):
    out = Path(a.output)
    jobs = []
    cases = [f'lopo_{p}' for p in METER_PARKS] if a.lopo else [f'p3_{v}' for v in P3_VARIANTS] if a.p3 else CASES
    for case in cases:
        for s in SEEDS:
            name = f'{case}_b{a.batch or 16}_seed{s}'
            cmd = [sys.executable, __file__, 'run', '--output', str(out), '--case', case, '--seed', str(s)]
            jobs.append(local_training.Job(name=name, log=str(out / 'logs' / f'{name}.out'),
                                           cmd=cmd + (['--batch', str(a.batch)] if a.batch else [])))
    gpus = local_training.parse_gpus(a.gpus, torch.cuda.device_count())
    local_training.run_jobs(jobs, gpus, per_gpu=a.per_gpu, cwd=str(Path(__file__).resolve().parents[2]))
    failed = [j.name for j in jobs if j.returncode != 0]
    if failed:
        raise SystemExit(f'failed jobs: {failed}')
    (out / 'COMPLETE').write_text(pd.Timestamp.now().isoformat())


def collect(a):
    out = Path(a.output)
    df = pd.concat([pd.read_csv(f) for f in sorted((out / 'rows').glob('*.csv'))], ignore_index=True)
    df.to_csv(out / 'rows_all.csv', index=False)
    pd.set_option('display.width', 250)
    print(df.groupby(['arm', 'seed'])[['best_epoch', 'epochs', 'seconds', 'pinball_day_clean', 'point_r2_clean']].mean().round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'run', 'launch', 'collect'])
    ap.add_argument('--output', required=True)
    ap.add_argument('--case', choices=ALL_CASES)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch', type=int, default=None, help='override the recipe batch (lr scaled linearly)')
    ap.add_argument('--max-epochs', type=int, default=None, help='timing run: train this many epochs, store nothing')
    ap.add_argument('--lopo', action='store_true', help='launch the LOPO jobs instead of P1')
    ap.add_argument('--p3', action='store_true', help='launch the P3 jobs (one fleet per geometry variant)')
    ap.add_argument('--variant', choices=P3_VARIANTS, help='prepare: client parks from this variant fleet (P3)')
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--per-gpu', type=int, default=3)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    os.chdir(Path(__file__).resolve().parents[2])
    {'prepare': lambda: prepare(Path(a.output), a.variant), 'run': lambda: run(a), 'launch': lambda: launch(a), 'collect': lambda: collect(a)}[a.mode]()


if __name__ == '__main__':
    main()
