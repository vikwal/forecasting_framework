#!/usr/bin/env python3
"""Calibrate P1 on historical meters, using level-3 CQR and fixed model weights.

Uses CPU only. Exports each seed and a directly calibrated seed ensemble.
The median is preserved by projecting corrected quantiles around it.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_paper_cnnlstm as P
import pv_geometry_forecast as G
import pv_conformal as C
import pv_hpo_local as H


def calibrate(pc, d, pred):
    q, corr = G.conformal(pc, d['ym'], d['nwpm'], ~d['shutm'], pred, d['nwp2'])
    # Sorting in the shared routine can move the median; retain the raw point forecast.
    mid = P.QS.index(0.5)
    median = np.maximum(np.sort(pred, axis=-1)[..., mid], 0)
    q[..., :mid] = np.minimum(q[..., :mid], median[..., None])
    q[..., mid] = median
    q[..., mid + 1:] = np.maximum(q[..., mid + 1:], median[..., None])
    q = np.maximum(q, 0)
    assert np.all(np.diff(q, axis=-1) >= 0)
    assert np.array_equal(q[..., mid], median)
    return q, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    torch.set_num_threads(4)
    out = Path(a.output)
    cfg = pickle.load(open(out / 'meta.pkl', 'rb'))['cfg']
    data = P.meters()
    cal, pred = {}, {}
    rows = []
    for seed in P.SEEDS:
        ck = torch.load(out / 'models' / f'jan_b16_seed{seed}.pt', map_location='cpu', weights_only=False)
        assert list(ck['quantiles']) == list(P.QS)
        model = H.Net(cfg['model']['feature_dim']['known_dim'], len(P.QS), ck['hp'])
        model.load_state_dict(ck['state_dict'])
        model.eval()
        for pv in P.METER_PARKS:
            d = data[pv]
            assert (pd.DatetimeIndex(d['tm']) + pd.Timedelta('48h') <= P.JAN_END).all()
            cal[seed, pv] = C.predict(model, d['Xm'], torch.device('cpu'))
            pred[seed, pv] = np.load(out / 'pred' / f'jan_b16_{pv}_seed{seed}.npy')
            q, corr = calibrate(cal[seed, pv], d, pred[seed, pv])
            np.save(out / 'pred' / f'jan_cal_b16_{pv}_seed{seed}.npy', q.astype(np.float32))
            rows.append(dict(park=pv, seed=str(seed), **corr))
            print(f'calibrated {pv} seed {seed}', flush=True)
    for pv in P.METER_PARKS:
        pc = np.mean([np.maximum(np.sort(cal[s, pv], axis=-1), 0) for s in P.SEEDS], axis=0)
        pe = np.mean([np.maximum(np.sort(pred[s, pv], axis=-1), 0) for s in P.SEEDS], axis=0)
        q, corr = calibrate(pc, data[pv], pe)
        np.save(out / 'pred' / f'jan_cal_ensemble_{pv}.npy', q.astype(np.float32))
        rows.append(dict(park=pv, seed='ensemble', **corr))
    pd.DataFrame(rows).to_csv(out / 'calibration_corrections.csv', index=False)
    (out / 'calibration_meta.json').write_text(json.dumps(dict(
        date=pd.Timestamp.now(tz='UTC').isoformat(), code_pin=H.code_pin(),
        calibration='historical clean daytime meters Jul-Jan, GTI bins as level 3',
        method='CQR with median-preserving projection', device='cpu'), indent=2))


if __name__ == '__main__':
    main()
