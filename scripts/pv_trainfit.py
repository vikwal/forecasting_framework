#!/usr/bin/env python3
"""Is the shared (central) model capacity-limited? (Viktor 2026-11-12) Pinball on the TRAINING windows of each meter park,
central model vs the park's local model, for already trained checkpoints (no training here):
  patch_cnn_2026-11-12_v1      point and stats arms, scratch
  boundary_real_2026-11-11_v1  P1 full history, sim prior (local vs federated over the six parks)
A central model that fits a park's training data clearly worse than the park's own model underfits (capacity or
optimisation); equal fit means the shared model can represent every park.

    frcst/bin/python scripts/pv_trainfit.py --gpu 0 --output results/parks_pv_v4_30min/trainfit_2026-11-12_v1
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_quantile_finetune as ft  # noqa: E402
import pv_patch_cnn as pc  # noqa: E402
from pv_quantile_metrics import metrics  # noqa: E402

RES = Path('results/parks_pv_v4_30min')


def daylight(item, base_cols, scaler, part):
    """gti_nwp > 10 W/m2 over the horizon, unscaled from the known input (gti_nwp is the first known feature)."""
    j = base_cols.index('gti_nwp')
    g = item[f'X_{part}']['known'][:, -item[f'y_{part}'].shape[1]:, 0] * scaler.scale_[j] + scaler.mean_[j]
    return g > 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    device = torch.device(f'cuda:{a.gpu}')
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (42, 43, 44):
        cfg = pc.load_cfg()
        qs = cfg['model']['tft']['quantiles']
        _, (sx, _) = ft.finetune.load_init(ft.base.Q_MODEL.format(seed=seed))
        data, keys, base_known, _, base_cols = pc.build_data(cfg, sx, RES / 'patch_cnn_2026-11-12_v1', arms=('point', 'stats'))
        hp = ft.hpo.get_hyperparameters(config=cfg)

        def model_for(arm, ckpt):
            c = dict(cfg, model=dict(cfg['model'], feature_dim=ft.tools.get_feature_dim(X=data[arm][keys[0]]['X_train'])))
            m = ft.models.get_model(config=c, hyperparameters=hp)
            m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict'])
            return m.to(device).eval()

        sd = RES / 'patch_cnn_2026-11-12_v1' / f'seed{seed}'
        br = RES / 'boundary_real_2026-11-11_v1' / f'seed{seed}'
        plans = []
        for arm in ('point', 'stats'):
            plans.append((f'scratch_{arm}', 'central', arm, {k: sd / 'central' / arm / 'fedgradient_best.pt' for k in keys}))
            plans.append((f'scratch_{arm}', 'local', arm, {k: sd / 'local' / arm / pc.park_id(k) / 'local_client_best.pt' for k in keys}))
        park = {k: pc.park_id(k).replace('_meter', '') for k in keys}
        plans.append(('prior_point', 'central', 'point', {k: br / f'P1/{park[k]}/full/federated/fedgradient_best.pt' for k in keys}))
        plans.append(('prior_point', 'local', 'point', {k: br / f'P1/{park[k]}/full/local/local_client_best.pt' for k in keys}))
        for study, scope, arm, ckpts in plans:
            for k in keys:
                if not ckpts[k].exists():
                    print('missing', ckpts[k])
                    continue
                m = model_for(arm, ckpts[k])
                it = data[arm][k]
                for part in ('train', 'test'):
                    pred = ft.base.predict(m, it[f'X_{part}'], device)
                    rows.append(dict(study=study, scope=scope, park=park[k], seed=seed, part=part,
                                     **metrics(pred, it[f'y_{part}'], qs, daylight(it, base_cols, sx, part))))
                del m
        pd.DataFrame(rows).to_csv(a.output / 'trainfit.csv', index=False)


if __name__ == '__main__':
    main()
