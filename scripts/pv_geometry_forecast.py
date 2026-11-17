#!/usr/bin/env python3
"""Effect of the park geometry on the forecast (Viktor 2026-11-16, level 3 of the tilt benchmark,
~/Work/pv_sim2real/docs/benchmark_tilt_ergebnisse_2026-11-16.md).

A local CNNRNN (recipe f3_t64 of the local HPO, ~/Work/pv_sim2real/docs/plan_hpo_local_cnnrnn_2026-11-13.md §12) is
trained per Trianel park on the synthetic twin of that park, built with the angles of one geometry variant, and tested
zero-shot on the real meter. The twins come from the tilt benchmark (same chain for every variant, only the arrays
differ): ${DATA_ROOT}/synthetic/solar/benchmark_tilt_twins/benchmark_tilt_2026-11-13/<variant>/. The known input
gti_nwp / gti_clearsky / ecmwf_gti is computed with the same variant's arrays (twin manifest), as in operation, where
only the assumed angles are known (Viktor: consistent).

Arms per park and seed:
  zs_<variant>   train on the twin (variant) 2023-07-24..2024-01-31, early stopping and test on the meter Feb..May 2024
  real           train on the meter 2023-07-24..2024-01-31 (inputs with the O1 arrays), same ES and test
  chain_<variant> (v2, Viktor 2026-11-16) one job per variant and seed that writes three arms:
                 zs_<variant>    the zero-shot model as above
                 zscal_<variant> its quantiles conformalised on the meter 2023-07-24..2024-01-31 (CQR per interval pair
                                 0.05/0.95, 0.1/0.9, 0.25/0.75, Mondrian classes of gti_nwp as pv_conformal, own park,
                                 clean daytime cells; the median stays)
                 ft_<variant>    the zero-shot model fine-tuned on the meter 2023-07-24..2024-01-31 (same recipe and lr,
                                 ES on the test window; epoch 0 = the zero-shot model is a candidate)
Early stopping on the test window is the agreed protocol (Viktor 2026-11-16). x scaler: the one of the local HPO
(60 source parks of parks_pv_v4 until 2024-08-01), so every arm sees the same scaling.
Metrics on the meter Feb..May 2024: daytime pinball (gti_nwp > 10), median RMSE/MAE/R2; '_clean' variants without
shutdown cells (meter < 2 % kWp at corrected GHI > 300 W/m2, the mask of the tilt benchmark).

    frcst/bin/python scripts/pv_geometry_forecast.py prepare --output results/trianel/geometry_forecast_2026-11-16_v1
    frcst/bin/python scripts/pv_geometry_forecast.py launch  --output <same> --gpus 0-2 --per-gpu 3
    frcst/bin/python scripts/pv_geometry_forecast.py collect --output <same>
    v2: prepare --output results/trianel/geometry_forecast_2026-11-16_v2; launch --output <v2> --chain

--study ew (next step 1): the same chain on the east-west parks of the fleet with >= 90 days before the test (29 of 56) (twins of
~/Work/pv_sim2real/benchmark_tilt/export_ew.py, the O1 series is the meter = truth, no shutdowns); one job per
variant and seed:
    prepare --study ew --output results/parks_pv_v4_30min/geometry_forecast_ew_2026-11-16_v1; launch --study ew --chain --output <same>

--budget (v3, Viktor 2026-11-17, cold start): per variant and seed the zero-shot model, then fine-tuned on only the last
7/14/30/60 meter days before the test (Dec/Jan; arms ft<n>_<variant>, ES on the test window as before). Uses the
prepared data of v2 (symlink prepared/ and meta.pkl):
    launch --budget [--study ew] --output <v3 dir>
--study ew_summer (P4 of ~/Work/pv_sim2real/docs/rechenauftrag_cnnlstm_2026-11-17.md, summer start): the same 29
east-west parks and budget chain, shifted by half a year: pre-training on the variant twin 2023-07-24..2024-07-31 (as
the winter study, twin and meter budget overlap in time), meter budget = the last 7/14/30/60 days before 2024-08-01
(June/July), early stopping and test Aug..Nov 2024. Twins of export_ew.py --long:
    prepare --study ew_summer --output results/parks_pv_v4_30min/geometry_forecast_ew_summer_2026-11-17_v1
    launch  --study ew_summer --budget --output <same>
"""
import argparse
import copy
import json
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import local_training, preprocessing, pv_parks, tools  # noqa: E402
import pv_conformal as base  # noqa: E402
import pv_hpo_local as HL  # noqa: E402
from pv_quantile_metrics import metrics  # noqa: E402
from pv_quantile_finetune import train, snapshot  # noqa: E402
from pv_synthetic_transfer import QS  # noqa: E402

DATA_CFG = 'configs/pv_parks/config_pv_v4_trianel_real_jan_nohist'
HPO_DIR = Path('results/parks_pv_v4_30min/hpo_local_cnnrnn_2026-11-13_v1')
TWIN_ROOT = os.path.join(os.environ.get('DATA_ROOT', '/mnt/lambda1/nvme1'), 'synthetic', 'solar', 'benchmark_tilt_twins', 'benchmark_tilt_2026-11-13')
VARIANTS = ['O1_ours', 'A0_operator', 'A1_register', 'A1b_register_fleet_median', 'A2_fixed20', 'B3_ndsm_gradient',
            'C1_pvanalytics', 'C1_pvanalytics_train', 'O6_largest_array']   # C1 saw the test window (upper bound), C1_train only Jul-Jan
REAL_INPUTS = 'O1_ours'
PARKS = ['pv1', 'pv2', 'pv3', 'pv5', 'pv6', 'pv7']
TRAIN = ('2023-07-24', '2024-01-31 23:30')
TEST = ('2024-02-01', '2024-05-31')
SEEDS = (42, 43, 44)
SHUT_FRAC, SHUT_GHI = 0.02, 300.0          # benchmark_tilt.common.shutdown_mask
EW_ROOT = os.path.join(os.environ.get('DATA_ROOT', '/mnt/lambda1/nvme1'), 'synthetic', 'solar', 'benchmark_tilt_ew_fleet', 'benchmark_tilt_2026-11-13')
EW_LONG_ROOT = os.path.join(os.environ.get('DATA_ROOT', '/mnt/lambda1/nvme1'), 'synthetic', 'solar', 'benchmark_tilt_ew_fleet_long', 'benchmark_tilt_2026-11-13')
SUMMER_TRAIN, SUMMER_TEST = ('2023-07-24', '2024-07-31 23:30'), ('2024-08-01', '2024-11-30')
EW_MIN_TRAIN_DAYS = 90                      # 29 of the 56 east-west parks (the others start later)
EW_VARIANTS = ['O1_ours', 'A1_register', 'A1b_register_fleet_median', 'A2_fixed20', 'B3_ndsm_gradient', 'O6_largest_array', 'SP_best_single']


def use_study(study):
    """Switch the module constants to the synthetic east-west study."""
    global TWIN_ROOT, VARIANTS, PARKS, TRAIN, TEST
    if study in ('ew', 'ew_summer'):
        TWIN_ROOT, VARIANTS = EW_ROOT, EW_VARIANTS
        PARKS = [pv for pv in sorted(pd.read_csv(os.path.join(EW_ROOT, 'O1_ours', 'parks.csv')).park_id) if ew_history_ok(pv)]
    if study == 'ew_summer':                    # same 29 parks as the winter study, then shift the windows
        TWIN_ROOT, TRAIN, TEST = EW_LONG_ROOT, SUMMER_TRAIN, SUMMER_TEST


def ew_history_ok(pv, min_days=EW_MIN_TRAIN_DAYS):
    """Release drivers start at the park's history start: keep parks with >= min_days of training before the test."""
    idx = pd.read_parquet(os.path.join(EW_ROOT, 'O1_ours', f'synth_{pv}.parquet'), columns=['power_park']).index
    return len(idx) > 0 and (pd.Timestamp(TEST[0], tz='UTC') - max(idx.min(), pd.Timestamp(TRAIN[0], tz='UTC'))).days >= min_days


def recipe():
    c = next(x for x in json.loads((HPO_DIR / 'hpo' / 'candidates.json').read_text()) if x['id'] == 'f3_t64')
    return {k: c[k] for k in ('id', 'conv_type', 'n_cnn_layers', 'filters', 'kernel_size', 'n_rnn_layers', 'units',
                              'bidirectional', 'lr', 'batch_size', 'weight_decay', 'dropout')}


def shutdown_cells(variant, pv, t0):
    """(n, H) mask of shutdown cells for windows whose first target step is t0 (index_test)."""
    d = os.path.join(TWIN_ROOT, variant)
    cap = float(pd.read_csv(os.path.join(d, 'parks.csv')).set_index('park_id').loc[pv, 'kwp_dc']) * 1000.0
    y = pd.read_parquet(os.path.join(d, f'meter_{pv}.parquet'))['power_w'] / cap
    syn = pd.read_parquet(os.path.join(d, f'synth_{pv}.parquet'))
    if 'ghi' in syn:
        ghi = syn['ghi']
        shut = ((y < SHUT_FRAC) & (ghi.reindex(y.index) > SHUT_GHI)).reindex(ghi.index, fill_value=False)
    else:                                                       # synthetic truth (study ew): no shutdowns
        shut = pd.Series(False, index=y.index)
    cells = pd.DatetimeIndex(t0).values[:, None] + pd.to_timedelta(np.arange(HL.H) * 30, unit='min').values[None, :]
    return shut.reindex(pd.DatetimeIndex(cells.ravel(), tz='UTC'), fill_value=False).values.reshape(cells.shape), y, cells


def gti_column(sx, X, nwp):
    """Scaler column of gti_nwp (known feature 0): the one whose inverse reproduces nwp_raw_test."""
    K = X['known'][:, -HL.H:, 0]
    err = [float(np.nanmax(np.abs(K * sx.scale_[i] + sx.mean_[i] - nwp))) for i in range(len(sx.mean_))]
    i = int(np.argmin(err))
    assert err[i] < 1e-6, err
    return i


def gti_raw(sx, X, i):
    return X['known'][:, -HL.H:, 0] * sx.scale_[i] + sx.mean_[i]


def prepare(out):
    with open(HPO_DIR / 'meta.pkl', 'rb') as f:
        hm = pickle.load(f)
    sx, feat_hpo = hm['scaler'], hm['features']
    (out / 'prepared').mkdir(parents=True, exist_ok=False)
    summary = {}
    for v in VARIANTS:
        cfg = base.load_cfg(DATA_CFG)
        cfg['data'].update(twin_path=os.path.join(TWIN_ROOT, v), meter_path=os.path.join(TWIN_ROOT, v),
                           files=PARKS + [p + '_meter' for p in PARKS], holdout_files=[],
                           twin_files=list(PARKS), meter_files=[p + '_meter' for p in PARKS])
        if TRAIN == SUMMER_TRAIN:                               # ew_summer: the config's dates are those of the winter study
            cfg['data'].update(train_start=TRAIN[0], train_end=TRAIN[1], test_start=TEST[0], test_end=TEST[1],
                               data_cutoff=f"{pd.Timestamp(TEST[1]) + pd.Timedelta(days=3):%Y-%m-%d}")
        for f in cfg['data']['files']:                          # every park must come from the variant directory
            assert pv_parks.park_source(cfg['data'], f)[0] == os.path.join(TWIN_ROOT, v), (v, f, pv_parks.park_source(cfg['data'], f))
        features = preprocessing.get_features(config=cfg)
        assert features['known'] == feat_hpo['known'] and not features['observed'], (features, feat_hpo)
        dfs = preprocessing.get_data(data_dir=cfg['data']['path'], config=cfg, freq=cfg['data']['freq'], features=features, files_key='files')
        items = base.windows(dfs, cfg, features, sx, TRAIN, TEST)
        D = {}
        for key, it in items.items():
            name = Path(key).name.removeprefix('synth_').removesuffix('.csv').removesuffix('.parquet')
            pv = name.removesuffix('_meter')
            d = D.setdefault(pv, {})
            if name.endswith('_meter'):
                t = pd.DatetimeIndex(it['index_test'])
                shut, ymeter, cells = shutdown_cells(v, pv, t)
                yt = np.asarray(it['y_test'])
                ref = ymeter.reindex(pd.DatetimeIndex(cells.ravel(), tz='UTC')).values.reshape(cells.shape)
                ok = np.isfinite(yt) & np.isfinite(ref)
                err = float(np.nanmax(np.abs(yt[ok] - ref[ok]))) if ok.any() else np.nan
                assert err < 1e-4, f'{v} {pv}: target cells do not line up with the meter (max |d| {err})'
                nwp2 = np.asarray(it['nwp_raw_test'])
                ig = gti_column(sx, it['X_test'], nwp2)
                tm = pd.DatetimeIndex(it['index_train'])
                d.update(Xm=it['X_train'], ym=it['y_train'], X2=it['X_test'], y2=yt, t2=t, nwp2=nwp2, shut2=shut,
                         tm=tm, nwpm=gti_raw(sx, it['X_train'], ig), shutm=shutdown_cells(v, pv, tm)[0])
            else:
                d.update(X1=it['X_train'], y1=it['y_train'], t1=pd.DatetimeIndex(it['index_train']))
        assert sorted(D) == PARKS and all({'X1', 'X2'} <= set(D[p]) for p in PARKS), {p: sorted(D[p]) for p in D}
        with open(out / 'prepared' / f'{v}.pkl', 'wb') as f:
            pickle.dump(D, f)
        summary[v] = {p: dict(n1=len(D[p]['y1']), nm=len(D[p]['ym']), n2=len(D[p]['y2']), shut_cells=int(D[p]['shut2'].sum())) for p in PARKS}
        print(v, summary[v], flush=True)
    cfg = copy.deepcopy(hm['cfg'])
    cfg['model']['feature_dim'] = tools.get_feature_dim(X=D[PARKS[0]]['X1'])
    with open(out / 'meta.pkl', 'wb') as f:
        pickle.dump(dict(cfg=cfg), f)
    (out / 'meta.json').write_text(json.dumps(dict(variants=VARIANTS, parks=PARKS, train=TRAIN, test=TEST, seeds=SEEDS, recipe=recipe(),
                                                   twin_root=TWIN_ROOT, scaler=str(HPO_DIR / 'meta.pkl'), counts=summary,
                                                   code_pin=HL.code_pin(), date=pd.Timestamp.now().isoformat()), indent=1))


def scores(model, d, device):
    pred = base.predict(model, d['X2'], device)
    y, nwp = d['y2'], d['nwp2']
    day = (nwp > 10) & np.isfinite(y)
    out = {}
    for tag, m in (('', day), ('_clean', day & ~d['shut2'])):
        q = metrics(pred, y, QS, m)
        point = np.sort(pred, axis=-1)[..., QS.index(.5)]
        pm = {}
        e = np.maximum(point[m], 0) - y[m]
        pm = dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                  r2=float(1 - np.sum(e ** 2) / np.sum((y[m] - y[m].mean()) ** 2)))
        out.update({f'{k}{tag}': q[k] for k in ('n_day', 'pinball_day', 'median_rmse', 'median_r2')})
        out.update({f'point_{k}{tag}': v for k, v in pm.items()})
        for lv, (lo, hi) in ((90, (0.05, 0.95)),):
            ps = np.maximum(np.sort(pred, axis=-1), 0)
            out[f'coverage_{lv}{tag}'] = float(((y[m] >= ps[m][:, QS.index(lo)]) & (y[m] <= ps[m][:, QS.index(hi)])).mean())
    return out, pred


def conformal(pc, yc, nwpc, keep, pred, nwp):
    """CQR per interval pair on the calibration cells (daytime, finite, keep), Mondrian classes of gti_nwp; night cells
    and classes with < 30 calibration cells stay as they are. Returns the corrected (n, H, Q) quantiles, sorted."""
    qc, q = np.sort(pc, axis=-1), np.sort(pred, axis=-1).copy()
    m = (nwpc > 10) & np.isfinite(yc) & keep
    bc, be = np.digitize(nwpc, base.BINS), np.digitize(nwp, base.BINS)
    corr = {}
    for level, (ql, qh) in base.LEVELS.items():
        il, ih = QS.index(ql), QS.index(qh)
        s = np.maximum(qc[..., il] - yc, yc - qc[..., ih])
        for b in range(1, len(base.BINS)):
            sb = s[m & (bc == b)]
            c = base.cqr_quantile(sb, 1 - level) if len(sb) >= 30 else 0.0
            sel = be == b
            lo, hi = q[..., il], q[..., ih]
            lo[sel] = np.maximum(lo[sel] - c, 0.0)
            hi[sel] = hi[sel] + c
            corr[f'{level}_{b}'] = c
    return np.sort(q, axis=-1), corr


class Fixed(torch.nn.Module):
    """Stand-in model that returns stored predictions (scores() calls base.predict on it)."""
    def __init__(self, pred):
        super().__init__()
        self.pred = pred


def scores_of(pred, d):
    orig = base.predict
    base.predict = lambda model, X, device, bs=256: model.pred   # noqa: E731
    try:
        return scores(Fixed(pred), d, None)
    finally:
        base.predict = orig


def finetune(cfg, hp, model, tr, val, seed, device, out):
    factory = lambda: HL.Net(cfg['model']['feature_dim']['known_dim'], len(QS), hp)  # noqa: E731
    t0 = time.time()
    m, log, _ = train('local_ft', [tr], [dict(X_test=val[0], y_test=val[1])], snapshot(model), cfg, hp, device, seed, QS, out,
                      factory=factory, lr=hp['lr'], batch_size=hp['batch_size'], weight_decay=hp['weight_decay'], save=False)
    return m, dict(best_epoch=log['best_epoch'], epochs=log['epochs'], seconds=round(time.time() - t0, 1), n_train=len(tr[1]))


def run_chain(a, out, cfg, hp, device, pin):
    v, _, only = a.arm.removeprefix('chain_').partition('@')
    seeds = (int(only),) if only else SEEDS
    with open(out / 'prepared' / f'{v}.pkl', 'rb') as f:
        D = pickle.load(f)
    path = out / 'rows' / f'{a.arm}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set(map(tuple, pd.read_csv(path).query("arm.str.startswith('ft_')", engine='python')[['park', 'seed']].astype(str).values)) if path.exists() else set()
    (out / 'pred').mkdir(exist_ok=True)
    for pv in PARKS:
        d = D[pv]
        for seed in seeds:
            if (pv, str(seed)) in done:
                continue
            rows = []
            model, info = HL.fit(cfg, hp, (d['X1'], d['y1']), (d['X2'], d['y2']), seed, device, out)
            m, pred = scores(model, d, device)
            rows.append(dict(arm=f'zs_{v}', variant=v, park=pv, seed=seed, **info, **m, code_pin=pin))
            np.save(out / 'pred' / f'zs_{v}_{pv}_seed{seed}.npy', pred.astype(np.float32))
            pc = base.predict(model, d['Xm'], device)
            qcal, corr = conformal(pc, np.asarray(d['ym'], float), d['nwpm'], ~d['shutm'], pred, d['nwp2'])
            m2, _ = scores_of(qcal, d)
            rows.append(dict(arm=f'zscal_{v}', variant=v, park=pv, seed=seed, **info, **m2, **{f'cqr_{k}': c for k, c in corr.items()}, code_pin=pin))
            np.save(out / 'pred' / f'zscal_{v}_{pv}_seed{seed}.npy', qcal.astype(np.float32))
            ft, finfo = finetune(cfg, hp, model, (d['Xm'], d['ym']), (d['X2'], d['y2']), seed, device, out)
            m3, pred3 = scores(ft, d, device)
            rows.append(dict(arm=f'ft_{v}', variant=v, park=pv, seed=seed, **{**info, **finfo}, **m3, code_pin=pin))
            np.save(out / 'pred' / f'ft_{v}_{pv}_seed{seed}.npy', pred3.astype(np.float32))
            df = pd.DataFrame(rows)
            if path.exists():
                df = pd.concat([pd.read_csv(path), df], ignore_index=True)
            df.to_csv(path, index=False)
            print(f"{v} {pv} seed {seed} pinball zs {m['pinball_day']:.5f} cal {m2['pinball_day']:.5f} ft {m3['pinball_day']:.5f} "
                  f"(ft epoch {finfo['best_epoch']})", flush=True)


BUDGETS = (7, 14, 30, 60)                    # meter days before the test (Viktor 2026-11-17: cold start)


def last_days(d, n):
    """Meter training windows of the last n calendar days before the test (windows = one per issue day)."""
    tm = pd.DatetimeIndex(d['tm'])
    m = np.asarray(tm >= tm.max().normalize() - pd.Timedelta(days=n - 1))
    return {k: x[m] for k, x in d['Xm'].items()}, np.asarray(d['ym'])[m]


def run_budget(a, out, cfg, hp, device, pin):
    """Zero-shot model of the variant, then fine-tuned on the last n meter days (n in BUDGETS); arms ft<n>_<variant>."""
    v, _, only = a.arm.removeprefix('budget_').partition('@')
    seeds = (int(only),) if only else SEEDS
    with open(out / 'prepared' / f'{v}.pkl', 'rb') as f:
        D = pickle.load(f)
    path = out / 'rows' / f'{a.arm}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    last = f'ft{BUDGETS[-1]}_{v}'
    done = set(map(tuple, pd.read_csv(path).query('arm == @last')[['park', 'seed']].astype(str).values)) if path.exists() else set()
    (out / 'pred').mkdir(exist_ok=True)
    for pv in PARKS:
        d = D[pv]
        for seed in seeds:
            if (pv, str(seed)) in done:
                continue
            rows = []
            model, info = HL.fit(cfg, hp, (d['X1'], d['y1']), (d['X2'], d['y2']), seed, device, out)
            m, pred = scores(model, d, device)
            rows.append(dict(arm=f'zs_{v}', variant=v, park=pv, seed=seed, budget=0, **info, **m, code_pin=pin))
            np.save(out / 'pred' / f'zs_{v}_{pv}_seed{seed}.npy', pred.astype(np.float32))
            msg = [f"zs {m['pinball_day']:.5f}"]
            for n in BUDGETS:
                Xn, yn = last_days(d, n)
                ft, finfo = finetune(cfg, hp, model, (Xn, yn), (d['X2'], d['y2']), seed, device, out)
                mn, predn = scores(ft, d, device)
                rows.append(dict(arm=f'ft{n}_{v}', variant=v, park=pv, seed=seed, budget=n, **{**info, **finfo}, **mn, code_pin=pin))
                np.save(out / 'pred' / f'ft{n}_{v}_{pv}_seed{seed}.npy', predn.astype(np.float32))
                msg.append(f"ft{n} {mn['pinball_day']:.5f} (ep {finfo['best_epoch']}, n {len(yn)})")
            df = pd.DataFrame(rows)
            if path.exists():
                df = pd.concat([pd.read_csv(path), df], ignore_index=True)
            df.to_csv(path, index=False)
            print(v, pv, 'seed', seed, ' '.join(msg), flush=True)


def run(a):
    out = Path(a.output)
    with open(out / 'meta.pkl', 'rb') as f:
        cfg = pickle.load(f)['cfg']
    hp = recipe()
    device, pin = torch.device('cuda:0'), HL.code_pin()
    arm = a.arm
    if arm.startswith('chain_'):
        return run_chain(a, out, cfg, hp, device, pin)
    if arm.startswith('budget_'):
        return run_budget(a, out, cfg, hp, device, pin)
    v = REAL_INPUTS if arm == 'real' else arm.removeprefix('zs_')
    with open(out / 'prepared' / f'{v}.pkl', 'rb') as f:
        D = pickle.load(f)
    path = out / 'rows' / f'{arm}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set(map(tuple, pd.read_csv(path)[['park', 'seed']].astype(str).values)) if path.exists() else set()
    (out / 'pred').mkdir(exist_ok=True)
    for pv in PARKS:
        d = D[pv]
        for seed in SEEDS:
            if (pv, str(seed)) in done:
                continue
            tr = (d['Xm'], d['ym']) if arm == 'real' else (d['X1'], d['y1'])
            model, info = HL.fit(cfg, hp, tr, (d['X2'], d['y2']), seed, device, out)
            m, pred = scores(model, d, device)
            np.save(out / 'pred' / f'{arm}_{pv}_seed{seed}.npy', pred.astype(np.float32))
            row = dict(arm=arm, variant=v, park=pv, seed=seed, **info, **m, code_pin=pin)
            pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)
            print(f"{arm} {pv} seed {seed} pinball {m['pinball_day']:.5f} clean {m['pinball_day_clean']:.5f}", flush=True)


def launch(a):
    out = Path(a.output)
    arms = ['real'] + [f'{"chain" if a.chain else "zs"}_{v}' for v in VARIANTS]
    if a.study in ('ew', 'ew_summer'):                          # no meter of its own; O1 = truth; one job per seed
        arms = [f'chain_{v}@{s}' for v in VARIANTS for s in SEEDS]
    if a.budget:                                                # v3: fine-tuning on the last n meter days
        arms = [f'budget_{v}@{s}' for v in VARIANTS for s in SEEDS]
    jobs = [local_training.Job(name=x, log=str(out / 'logs' / f'{x}.out'),
                               cmd=[sys.executable, __file__, 'run', '--study', a.study, '--output', str(out), '--arm', x]) for x in arms]
    gpus = local_training.parse_gpus(a.gpus, torch.cuda.device_count())
    local_training.run_jobs(jobs, gpus, per_gpu=a.per_gpu, cwd=str(Path(__file__).resolve().parents[1]))
    failed = [j.name for j in jobs if j.returncode != 0]
    if failed:
        raise SystemExit(f'failed jobs: {failed}')
    (out / 'COMPLETE').write_text(pd.Timestamp.now().isoformat())


def collect(a):
    out = Path(a.output)
    df = pd.concat([pd.read_csv(f) for f in sorted((out / 'rows').glob('*.csv'))], ignore_index=True)
    df.to_csv(out / 'rows_all.csv', index=False)
    pd.set_option('display.width', 250)
    for m in ('pinball_day', 'pinball_day_clean', 'point_rmse_clean', 'point_r2_clean'):
        t = df.groupby(['arm', 'park'])[m].mean().unstack()
        t['mean'] = t.mean(axis=1)
        print(m), print(t.round(5).sort_values('mean').to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'run', 'launch', 'collect'])
    ap.add_argument('--output', required=True)
    ap.add_argument('--arm')
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--per-gpu', type=int, default=3)
    ap.add_argument('--chain', action='store_true', help='v2: zero-shot + conformal + fine-tuning per variant')
    ap.add_argument('--budget', action='store_true', help='v3: fine-tuning on the last 7/14/30/60 meter days per variant')
    ap.add_argument('--study', choices=['trianel', 'ew', 'ew_summer'], default='trianel')
    a = ap.parse_args()
    use_study(a.study)
    os.chdir(Path(__file__).resolve().parents[1])
    {'prepare': lambda: prepare(Path(a.output)), 'run': lambda: run(a), 'launch': lambda: launch(a), 'collect': lambda: collect(a)}[a.mode]()


if __name__ == '__main__':
    main()
