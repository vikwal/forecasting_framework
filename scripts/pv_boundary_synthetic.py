#!/usr/bin/env python3
"""Part B of the dependence-boundary study (Viktor 2026-11-11): a two-dimensional map "parks per client x months of own data"
of the gain from federated partners, on fully synthetic parks (no reality gap).

Per seed: 20 partner parks with the full training year and one target client per size P = 1, 2, 5, 10 (disjoint, drawn
from the 154 client parks of the synthetic transfer study). Own data: M = 1, 3, 6, 12 months of the training year
2023-07-24..2024-07-31, starting in August / November / February / May (M = 6: August, February; M = 12: whole year).
Four arms, ES and evaluation on the client's parks August 2024--July 2025 (agreed protocol):
  local_scratch, federated_scratch     random initialisation
  local_prior,   federated_prior       start from the model pretrained on the 56 held-out source parks (synthetic transfer)
Federated = pooled training of client and partners (identical to FedGradient). Windows are chronological, one per 09 UTC
run; positions are used as day offsets (prepared.pkl predates index_train; a few dropped runs shift by days).

    frcst/bin/python scripts/pv_boundary_synthetic.py --seed 42 --gpu 0 --output results/parks_pv_v4_30min/boundary_synthetic_2026-11-11_v1
"""
import argparse
import os
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import hpo, models  # noqa: E402
import pv_conformal as base  # noqa: E402
from pv_quantile_finetune import train, snapshot  # noqa: E402
from pv_quantile_metrics import metrics  # noqa: E402
from pv_synthetic_scaling import PREPARED  # noqa: E402
from pv_cnnlstm_pers import FrameworkCNNLSTM  # noqa: E402


def bundle_source(path):
    with path.open('rb') as f:
        return pickle.load(f)['source']

PRETRAINED = Path('results/parks_pv_v4_30min/synthetic_transfer_2026-11-09_v1/seed{seed}/pretrained.pt')
SIZES = [1, 2, 5, 10]
STARTS = {'aug': 8, 'nov': 100, 'feb': 192, 'may': 282}                # day offsets from 2023-07-24
ARMS = ['local_scratch', 'federated_scratch', 'local_prior', 'federated_prior']
PLAN = [(1, s) for s in STARTS] + [(3, s) for s in STARTS] + [(6, 'aug'), (6, 'feb'), (12, 'year')]


def own(item, months, start):
    n = len(item['y_train'])
    lo = 0 if start == 'year' else STARTS[start]
    hi = n if start == 'year' else min(n, lo + int(round(30.4 * months)))
    return {k: v[lo:hi] for k, v in item['X_train'].items()}, item['y_train'][lo:hi]


def pool(parts):
    return {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}, np.concatenate([p[1] for p in parts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--partners', type=int, default=20)
    ap.add_argument('--sizes', type=int, nargs='+', default=SIZES)
    ap.add_argument('--months', type=int, nargs='+', default=[1, 3, 6, 12])
    ap.add_argument('--arms', nargs='+', default=ARMS, help='subset of the four arms (local arms do not depend on partners)')
    ap.add_argument('--model', choices=['tft', 'cnnlstm'], default='tft',
                    help='cnnlstm: utils.models.CNNRNN with quantile step output (5 conv layers); its prior is pretrained here on the 56 source parks')
    a = ap.parse_args()
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{a.gpu}')
    with PREPARED.open('rb') as f:
        bundle = pickle.load(f)
    cfg, pool_parks = bundle['cfg'], bundle['target']
    del bundle
    qs = cfg['model']['tft']['quantiles']
    cfg['params']['random_seed'] = a.seed
    hp = hpo.get_hyperparameters(config=cfg)
    hp['lr'] = 3e-4
    order = sorted(pool_parks)
    perm = [order[i] for i in np.random.default_rng(a.seed).permutation(len(order))]
    targets, pos = {}, 20                                                  # targets fixed at positions 20..37 for every
    for P in SIZES:                                                        # partner count; partners = first 20, then the
        targets[P], pos = perm[pos:pos + P], pos + P                       # parks after the targets
    partners = (perm[:20] + perm[pos:])[:a.partners]
    if a.model == 'tft':
        factory = lambda: models.get_model(config=cfg, hyperparameters=hp)  # noqa: E731
    else:
        factory = lambda: FrameworkCNNLSTM(cfg['model']['feature_dim']['known_dim'], cfg['model']['horizon'], len(qs), 5)  # noqa: E731
    torch.manual_seed(a.seed)
    init = factory()
    scratch = snapshot(init)
    del init
    out = a.output / f'seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if a.model == 'tft':
        prior = torch.load(str(PRETRAINED).format(seed=a.seed), map_location='cpu', weights_only=False)['state_dict']
    elif (a.output.parent / f'cnnlstm_prior_seed{a.seed}.pt').exists():          # shared by all partner counts
        prior = torch.load(a.output.parent / f'cnnlstm_prior_seed{a.seed}.pt', map_location='cpu', weights_only=False)['state_dict']
    else:                                              # same protocol as the TFT prior (pv_synthetic_transfer.py)
        src = bundle_source(PREPARED)
        model, log, _ = train('local_pretrain', [pool([own(it, 12, 'year') for it in src.values()])], list(src.values()),
                              scratch, cfg, hp, device, a.seed, qs, out, factory=factory)
        prior = snapshot(model)
        torch.save(dict(state_dict=prior, log=log), a.output.parent / f'cnnlstm_prior_seed{a.seed}.pt')
        del model, src
    partner_data = pool([own(pool_parks[k], 12, 'year') for k in partners])
    pin = os.environ.get('CODE_PIN') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() + ' + dirty'
    rows = pd.read_csv(out / 'metrics.csv').to_dict('records') if (out / 'metrics.csv').exists() else []
    done = {r['case'] for r in rows}
    for P in a.sizes:
        val = [pool_parks[k] for k in targets[P]]
        for months, start in [x for x in PLAN if x[0] in a.months]:
            mine = pool([own(pool_parks[k], months, start) for k in targets[P]])
            for arm, state in [x for x in (('local_scratch', scratch), ('federated_scratch', scratch), ('local_prior', prior),
                                           ('federated_prior', prior)) if x[0] in a.arms]:
                case = f'P{P:02d}/m{months:02d}_{start}/{arm}'
                if case in done:
                    continue
                cdir = out / case
                cdir.mkdir(parents=True, exist_ok=True)
                data = mine if arm.startswith('local') else pool([mine, partner_data])
                model, log, _ = train('local_' + arm, [data], val, state, cfg, hp, device, a.seed, qs, cdir, factory=factory)
                model.eval()
                per = []
                for it in val:
                    pred = base.predict(model, it['X_test'], device)
                    per.append(metrics(pred, it['y_test'], qs, np.asarray(it['nwp_raw_test']) > 10))
                rows.append(dict(case=case, seed=a.seed, partners=a.partners, model=a.model, n_parks=P, months=months, start=start, arm=arm,
                                 own_windows=len(mine[1]), best_epoch=log['best_epoch'], epochs=log['epochs'],
                                 date='2026-11-11', code_pin=pin, **pd.DataFrame(per).mean().to_dict()))
                pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
                del model
                torch.cuda.empty_cache()
    (out / 'clients.txt').write_text(f'partners {partners}\n' + ''.join(f'P{P} {targets[P]}\n' for P in SIZES))
    (out / 'COMPLETE').write_text('done\n')


if __name__ == '__main__':
    main()
