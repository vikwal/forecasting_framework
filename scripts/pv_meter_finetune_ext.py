#!/usr/bin/env python3
"""Extension of the fine-tuning decision test (Viktor 2026-11-09), same split and engine as pv_meter_finetune.py
(fine-tuning 2023-07-24..2024-01-31, early stopping and evaluation on February..May 2024, start = generic model
without power history, lr 3e-4, per-batch gradients with a server-side Adam step):

  1  synthetic rehearsal on the server: every server step mixes the meter gradient with the gradient of one batch of the
     synthetic training fleet (v4 client parks, same window), g = (1 - w) * mean(meter clients) + w * synthetic, w = 0.5
       local_srv     one meter client + server batch, per park
       fedgrad_srv   six meter clients + server batch
  2  personalised FedGradient: shared body aggregated over the six clients, the output layer stays with each client
     (own Adam step on its own gradient)
       pfl_head      six clients, shared body + personal output layer
       local_head    per park, only the output layer is trained (body frozen) - the local counterpart

    frcst/bin/python scripts/pv_meter_finetune_ext.py -c configs/pv_parks/config_pv_v4_trianel_real_jan_nohist \
        --fleet-config configs/pv_parks/config_pv_v4_central_jan_nohist \
        --init-model models/cl_m-tft_out-96_freq-30min_pv_v4_central_jan_nohist.pt --seed 42 --gpu 0 [--no-start-candidate]
Output: results/parks_pv_v4_30min/meterft_v3[_nostart]/<variant>_seed<s>.pkl, training_ext_seed<s>.csv"""
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import fedgradient as fg, finetune, hpo, models, preprocessing, tools  # noqa: E402
from utils.final_training import predict  # noqa: E402
from pv_meter_finetune import EV_END, EV_START, FT_END, cat, point, predictions, windows  # noqa: E402


def is_head(name):
    return name.startswith('output_layer.')


class Run:
    """Clients with own model copies, a global model and its server optimizer on the shared parameters."""

    def __init__(self, parts, init_sd, cfg, hp, device, seed, bs, personal=False, freeze_body=False, server_data=None):
        tools.set_seed(seed)
        self.cfg, self.hp, self.device, self.seed = cfg, hp, device, seed
        criterion, median_idx = tools.make_criterion(cfg)
        self.glob = self.model(init_sd)
        self.personal = personal
        shared = [p for n, p in self.glob.named_parameters() if not (personal and is_head(n))]
        self.opt = torch.optim.Adam(shared, lr=0.0 if freeze_body else hp['lr'])
        mk = lambda X, y, i: fg.GradientClient(self.model(init_sd), X, y, bs, criterion, device, is_tft=True,
                                               clipnorm=cfg['model']['tft'].get('clipnorm'), median_idx=median_idx,
                                               client_id=i, dropout_seed_offset=i)
        self.clients = [mk(X, y, i) for i, (X, y) in enumerate(parts)]
        self.head_opt = [torch.optim.Adam([p for n, p in c.model.named_parameters() if is_head(n)], lr=hp['lr'])
                         for c in self.clients] if personal else None
        self.server = mk(*server_data, 99) if server_data is not None else None
        self.server_round = 0

    def model(self, sd):
        m = models.get_model(config=self.cfg, hyperparameters=self.hp)
        m.load_state_dict(sd)
        return m.to(self.device)

    def shared_weights(self):
        return {n: p.detach().to('cpu', copy=True) for n, p in self.glob.named_parameters() if not (self.personal and is_head(n))}

    def server_batch(self, weights):
        if self.server.remaining <= 0:
            self.server.prepare_round(fg.round_seed(self.seed + 7, self.server_round))
            self.server_round += 1
        return self.server.step(weights)

    def epoch(self, ep, w_srv):
        for c in self.clients:
            c.prepare_round(fg.round_seed(self.seed, ep))
        active = [i for i, c in enumerate(self.clients) if c.remaining > 0]
        while active:
            weights = self.shared_weights()
            res = []
            for i in active:
                res.append(self.clients[i].step(weights))
                if self.personal:
                    self.head_opt[i].step()                                # personal output layer, own gradient
            grads, _ = fg.aggregate_gradients(res, None, 'batch_size')
            if self.server is not None:
                g_s = self.server_batch(weights)['gradients']
                grads = {k: (1 - w_srv) * v + w_srv * g_s[k].float() for k, v in grads.items()}
            if self.personal:
                grads = {k: v for k, v in grads.items() if not is_head(k)}
            fg.server_step(self.glob, self.opt, grads)
            active = [i for i, r in zip(active, res) if r['remaining'] > 0]

    def client_model(self, i):
        """Global parameters plus (personal) head of client i."""
        m = self.clients[i].model
        fg.load_params(m, self.shared_weights())
        return m if self.personal else self.glob

    def state(self):
        return copy.deepcopy(self.glob.state_dict()), [copy.deepcopy(c.model.state_dict()) for c in self.clients]

    def restore(self, st):
        self.glob.load_state_dict(st[0])
        for c, sd in zip(self.clients, st[1]):
            c.model.load_state_dict(sd)


def val_rmse(run, vals):
    se, n = 0.0, 0
    for i, (X, y) in enumerate(vals):
        p, t = predict(run.client_model(i), X, y, run.device, 256)
        se += float(((point(p) - t) ** 2).sum()); n += t.size
    return float(np.sqrt(se / n))


def train(name, run, vals, cfg, seed, log, start_candidate, w_srv=0.5):
    es = cfg['model']['early_stopping']
    rmse0 = val_rmse(run, vals)
    best, best_ep, best_st = (rmse0, 0, run.state()) if start_candidate else (np.inf, 0, run.state())
    wait = 0
    for ep in range(cfg['model']['epochs']):
        run.epoch(ep, w_srv)
        r = val_rmse(run, vals)
        if r < best - es['min_delta']:
            best, best_ep, best_st, wait = r, ep + 1, run.state(), 0
        else:
            wait += 1
            if wait >= es['patience']:
                break
    run.restore(best_st)
    log.append(dict(variant=name, seed=seed, clients=len(run.clients), epochs=ep + 1, best_epoch=best_ep,
                    val_rmse_init=rmse0, val_rmse_best=best))
    logging.info(f"{name} seed {seed}: {ep + 1} epochs, best {best_ep}, validation RMSE {rmse0:.4f} -> {best:.4f}")
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--fleet-config', required=True, help='config of the pretraining fleet (data.files = synthetic parks)')
    ap.add_argument('--init-model', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--server-weight', type=float, default=0.5)
    ap.add_argument('--no-start-candidate', action='store_true')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    device = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    load = lambda p: tools.handle_freq(config=tools.load_config(p.split('.yaml')[0] + '.yaml'))
    cfg, fcfg = load(a.config), load(a.fleet_config)
    for c in (cfg, fcfg):
        c['model']['name'], c['model']['fl'] = 'tft', False
    features = preprocessing.get_features(config=cfg)
    init_sd, (sx, _) = finetune.load_init(a.init_model)
    dfs = preprocessing.get_data(data_dir=cfg['data']['path'], config=cfg, freq=cfg['data']['freq'], features=features, files_key='files')
    keys = sorted(dfs)
    ft = windows(dfs, cfg, features, sx, FT_END, EV_START, EV_END)
    fdfs = preprocessing.get_data(data_dir=fcfg['data']['path'], config=fcfg, freq=fcfg['data']['freq'], features=features, files_key='files')
    fleet = windows(fdfs, fcfg, features, sx, FT_END, EV_START, EV_END)
    syn = cat([it for it in fleet.values() if len(it['y_train'])], 'train')
    del fdfs, fleet
    logging.info(f"synthetic server data: {len(syn[1])} windows")
    cfg['model']['feature_dim'] = tools.get_feature_dim(X=ft[keys[0]]['X_train'])
    hp = hpo.get_hyperparameters(config=cfg)
    hp['lr'] = a.lr
    bs = int(hp.get('batch_size', cfg['model']['batch_size']))
    sc = not a.no_start_candidate
    out_dir = os.path.join('results', os.path.basename(cfg['data']['path']), 'meterft_v3' + ('' if sc else '_nostart'))
    os.makedirs(out_dir, exist_ok=True)
    save = lambda name, preds: pickle.dump({'variant': name, 'seed': a.seed, 'init_model': a.init_model, 'lr': a.lr,
                                           'server_weight': a.server_weight, 'predictions': preds},
                                          open(os.path.join(out_dir, f'{name}_seed{a.seed}.pkl'), 'wb'))
    log, parts, vals = [], {k: cat([ft[k]], 'train') for k in keys}, {k: cat([ft[k]], 'test') for k in keys}
    # 1 synthetic rehearsal on the server
    preds = {}
    for k in keys:
        run = train(f'local_srv_{k}', Run([parts[k]], init_sd, cfg, hp, device, a.seed, bs, server_data=syn), [vals[k]],
                    cfg, a.seed, log, sc, a.server_weight)
        preds.update(predictions(run.glob, ft, [k], device))
    save('local_srv', preds)
    run = train('fedgrad_srv', Run([parts[k] for k in keys], init_sd, cfg, hp, device, a.seed, bs, server_data=syn),
                [vals[k] for k in keys], cfg, a.seed, log, sc, a.server_weight)
    save('fedgrad_srv', predictions(run.glob, ft, keys, device))
    # 2 personalised FedGradient (shared body, personal output layer) and its local counterpart
    run = train('pfl_head', Run([parts[k] for k in keys], init_sd, cfg, hp, device, a.seed, bs, personal=True),
                [vals[k] for k in keys], cfg, a.seed, log, sc)
    preds = {}
    for i, k in enumerate(keys):
        preds.update(predictions(run.client_model(i), ft, [k], device))
    save('pfl_head', preds)
    preds = {}
    for k in keys:
        run = train(f'local_head_{k}', Run([parts[k]], init_sd, cfg, hp, device, a.seed, bs, personal=True, freeze_body=True),
                    [vals[k]], cfg, a.seed, log, sc)
        preds.update(predictions(run.client_model(0), ft, [k], device))
    save('local_head', preds)
    pd.DataFrame(log).to_csv(os.path.join(out_dir, f'training_ext_seed{a.seed}.csv'), index=False)
    logging.info('done')


if __name__ == '__main__':
    main()
