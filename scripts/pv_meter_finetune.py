#!/usr/bin/env python3
"""Decision test (Viktor 2026-11-09): what do the six meters add to the synthetically pretrained model, and does
sharing them pay? All variants start from the same pretrained model (generic v4 fleet, no power history) and its
scaler, and train with the same engine, utils.fedgradient (per-batch gradients, server-side Adam step); the variants
differ only in how the meter data are partitioned into clients:

  zeroshot      no training
  affine_local  zero-shot plus a per-park affine recalibration fitted on that park's fine-tuning window
  local_<p>     one client = one meter, evaluated on that meter
  central       one client holding all six meters (pooled data, batch 64)
  loo_<p>       one client holding the other five meters, evaluated on the left-out meter (newcomer)
  fedgrad       six clients, per-batch gradient mean over the six, batch 64 per client (effective 384)
  fedgrad_b11   six clients, batch 11 per client (effective 66, ~ the central batch)

Time split (Viktor, protocol): fine-tuning 2023-07-24..2024-01-31, validation February..May 2024; early stopping on the
whole validation window of the training parks (left-out park: of the five others), evaluation on the same window.
Curtailment stays in.

    frcst/bin/python scripts/pv_meter_finetune.py -c configs/pv_parks/config_pv_v4_trianel_real_jan_nohist \
        --init-model models/cl_m-tft_out-96_freq-30min_pv_v4_central_jan_nohist.pt --seed 42 --gpu 0
Output: results/parks_pv_v4_30min/meterft_v3/<variant>_seed<s>.pkl (predictions in the train_cl layout) and
training_seed<s>.csv (epochs, best validation RMSE per variant)."""
import argparse
import copy
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import fedgradient as fg, finetune, hpo, models, preprocessing, tools  # noqa: E402
from utils.final_training import predict  # noqa: E402

FT_END = '2024-01-31 23:30'
EV_START, EV_END = '2024-02-01', '2024-05-31'


def windows(dfs, cfg, features, sx, train_end, test_start, test_end):
    """key -> prepared item (X_train, y_train, X_test, y_test, index_test, nwp_raw_test) for one time split."""
    c = copy.deepcopy(cfg)
    c['data'].update(train_end=train_end, test_start=test_start, test_end=test_end)
    return {it['key']: it for it in tools.create_data_generator(dfs, c, features, scaler_x=sx, scaler_y=None)}


def cat(items, part):
    X = {k: np.concatenate([it[f'X_{part}'][k] for it in items]) for k in items[0][f'X_{part}']}
    return X, np.concatenate([it[f'y_{part}'] for it in items])


def point(p):
    return p[..., 0] if p.ndim == 3 else p


def train_variant(name, clients, val, init_sd, cfg, hp, device, seed, batch_size, log, start_candidate=True):
    """clients: list of (X, y) per client; val: (X, y) pooled validation windows (Feb-May) of the training parks."""
    tools.set_seed(seed)
    criterion, median_idx = tools.make_criterion(cfg)
    glob = models.get_model(config=cfg, hyperparameters=hp)
    glob.load_state_dict(init_sd)
    glob = glob.to(device)
    opt = torch.optim.Adam(glob.parameters(), lr=hp['lr'])
    workers = []
    for i, (X, y) in enumerate(clients):
        m = models.get_model(config=cfg, hyperparameters=hp)
        m.load_state_dict(init_sd)
        workers.append(fg.GradientClient(m, X, y, batch_size, criterion, device, is_tft=True,
                                         clipnorm=cfg['model']['tft'].get('clipnorm'), median_idx=median_idx,
                                         client_id=i, dropout_seed_offset=i))

    def dispatch(method, idx, arg):
        out = []
        for i in idx:
            w = workers[i]
            out.append(w.prepare_round(*arg) if method == 'prepare_round' else w.step(arg) if method == 'step'
                       else w.round_metrics())
        return out

    es = cfg['model']['early_stopping']
    best, best_sd, wait, steps_total = np.inf, copy.deepcopy(glob.state_dict()), 0, 0
    p0, t0 = predict(glob, val[0], val[1], device, 256)
    rmse0 = float(np.sqrt(np.mean((point(p0) - t0) ** 2)))
    best, best_ep = (rmse0, 0) if start_candidate else (np.inf, 0)       # 'no fine-tuning' as candidate (default)
    for ep in range(cfg['model']['epochs']):
        _, steps = fg.run_round(dispatch, list(range(len(workers))), glob, opt, fg.round_seed(seed, ep), weighting='batch_size')
        steps_total += steps
        p, t = predict(glob, val[0], val[1], device, 256)
        rmse = float(np.sqrt(np.mean((point(p) - t) ** 2)))
        if rmse < best - es['min_delta']:
            best, best_sd, wait, best_ep = rmse, copy.deepcopy(glob.state_dict()), 0, ep + 1
        else:
            wait += 1
            if wait >= es['patience']:
                break
    glob.load_state_dict(best_sd)
    log.append(dict(variant=name, seed=seed, clients=len(workers), batch_size=batch_size, epochs=ep + 1,
                    best_epoch=best_ep, val_rmse_init=rmse0, val_rmse_best=best,
                    server_steps=steps_total))
    logging.info(f"{name} seed {seed}: {ep + 1} epochs, best {best_ep}, validation RMSE {rmse0:.4f} -> {best:.4f}")
    return glob


def predictions(model, items, keys, device):
    out = {}
    for k in keys:
        it = items[k]
        p, t = predict(model, it['X_test'], it['y_test'], device, 256)
        cols = [f't+{i + 1}' for i in range(t.shape[1])]
        out[(k, 'power')] = {'pred': pd.DataFrame(np.clip(point(p), 0, None), index=it['index_test'], columns=cols),
                             'true': pd.DataFrame(t, index=it['index_test'], columns=cols),
                             'baseline::NWP (gti_nwp)': pd.DataFrame(it['nwp_raw_test'], index=it['index_test'], columns=cols)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--init-model', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--lr', type=float, default=3e-4, help='fine-tuning lr (protocol: 3e-4 moves the weights, 1e-5 does not)')
    ap.add_argument('--no-start-candidate', action='store_true',
                    help='best epoch >= 1 as in train_cl (the start state is not an early-stopping candidate)')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    device = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    cfg = tools.handle_freq(config=tools.load_config(a.config.split('.yaml')[0] + '.yaml'))
    cfg['model']['name'], cfg['model']['fl'] = 'tft', False
    features = preprocessing.get_features(config=cfg)
    init_sd, (sx, _) = finetune.load_init(a.init_model)
    dfs = preprocessing.get_data(data_dir=cfg['data']['path'], config=cfg, freq=cfg['data']['freq'], features=features, files_key='files')
    keys = sorted(dfs)
    ft = windows(dfs, cfg, features, sx, FT_END, EV_START, EV_END)       # X_train = fine-tuning, X_test = validation Feb-May (ES)
    ev = ft                                                                # evaluation on the same validation window
    cfg['model']['feature_dim'] = tools.get_feature_dim(X=ft[keys[0]]['X_train'])
    hp = hpo.get_hyperparameters(config=cfg)
    hp['lr'] = a.lr
    bs = int(hp.get('batch_size', cfg['model']['batch_size']))
    for k in keys:
        logging.info(f"{k}: fine-tuning windows {len(ft[k]['y_train'])}, validation {len(ft[k]['y_test'])}")
    out_dir = os.path.join('results', os.path.basename(cfg['data']['path']), 'meterft_v3' + ('_nostart' if a.no_start_candidate else ''))   # meterft, meterft_v2: ES on January, training only to December (2026-11-09, wrong split, superseded)
    os.makedirs(out_dir, exist_ok=True)
    save = lambda name, preds, extra=None: pickle.dump({'variant': name, 'seed': a.seed, 'init_model': a.init_model, 'lr': a.lr,
                                                        'predictions': preds, **(extra or {})},
                                                       open(os.path.join(out_dir, f'{name}_seed{a.seed}.pkl'), 'wb'))
    log = []
    base = models.get_model(config=cfg, hyperparameters=hp)
    base.load_state_dict(init_sd)
    base = base.to(device)
    if a.seed == 42:                                                       # deterministic, once
        save('zeroshot', predictions(base, ev, keys, device))
        aff = {}
        for k in keys:                                                     # per-park affine on the fine-tuning window
            p, t = predict(base, ft[k]['X_train'], ft[k]['y_train'], device, 256)
            p, t = point(p).ravel(), t.ravel()
            m = (t > 0) | (p > 0.005)
            coef = np.polyfit(p[m], t[m], 1)
            pr = predictions(base, ev, [k], device)
            pr[(k, 'power')]['pred'] = (pr[(k, 'power')]['pred'] * coef[0] + coef[1]).clip(lower=0)
            pr[(k, 'power')]['pred'] = pr[(k, 'power')]['pred'].where(pr[(k, 'power')]['pred'] > 0.0, 0.0)
            aff.update(pr)
            logging.info(f"affine {k}: slope {coef[0]:.3f} intercept {coef[1]:.4f}")
        save('affine_local', aff)
    val_all = cat([ft[k] for k in keys], 'test')
    # local
    preds = {}
    for k in keys:
        m = train_variant(f'local_{k}', [cat([ft[k]], 'train')], cat([ft[k]], 'test'), init_sd, cfg, hp, device, a.seed, bs, log, not a.no_start_candidate)
        preds.update(predictions(m, ev, [k], device))
    save('local', preds)
    # leave one out (newcomer)
    preds = {}
    for k in keys:
        rest = [j for j in keys if j != k]
        m = train_variant(f'loo_{k}', [cat([ft[j] for j in rest], 'train')], cat([ft[j] for j in rest], 'test'),
                          init_sd, cfg, hp, device, a.seed, bs, log, not a.no_start_candidate)
        preds.update(predictions(m, ev, [k], device))
    save('loo', preds)
    # central (pooled) and federated (per-batch gradient mean over six clients)
    m = train_variant('central', [cat([ft[k] for k in keys], 'train')], val_all, init_sd, cfg, hp, device, a.seed, bs, log, not a.no_start_candidate)
    save('central', predictions(m, ev, keys, device))
    for name, b in (('fedgrad', bs), ('fedgrad_b11', max(1, round(bs / len(keys))))):
        m = train_variant(name, [cat([ft[k]], 'train') for k in keys], val_all, init_sd, cfg, hp, device, a.seed, b, log, not a.no_start_candidate)
        save(name, predictions(m, ev, keys, device))
    pd.DataFrame(log).to_csv(os.path.join(out_dir, f'training_seed{a.seed}.csv'), index=False)
    logging.info('done')


if __name__ == '__main__':
    main()
