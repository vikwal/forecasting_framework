#!/usr/bin/env python3
"""Shared CNN encoder, personal LSTM head (Viktor 2026-11-12): does FL-style personalisation of a quantile CNN-LSTM
beat purely local training on the six real meters? One ICON-D2 grid point (the four point features of the quantile
TFT), same split and protocol as before (train 2023-07-24..2024-01-31, early stopping and evaluation Feb--May 2024),
scratch init, lr 3e-4, no HPO. Model utils.models.CNNRNN with quantile output (--arch), 5 and 10 conv layers. Arms:
  local        one CNN-LSTM per park
  global       one CNN-LSTM over all six parks (FedGradient = pooled)
  global_head  global, then encoder frozen, LSTM head fine-tuned per park (sequential FedRep-style personalisation)
  global_ft    global, then all layers fine-tuned per park (control for global_head)
  fedper       joint training: encoder gradients pooled over the parks, one LSTM head per park (FedPer)
Early stopping: local, *_head, *_ft and fedper select per park on that park's evaluation window; global once for all.

    frcst/bin/python scripts/pv_cnnlstm_pers.py --arch framework_step --seed 42 --gpu 0 --output results/parks_pv_v4_30min/cnnlstm_pers_framework_step_2026-11-12_v1
"""
import argparse
import copy
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_quantile_finetune as ft  # noqa: E402
import pv_patch_cnn as pc  # noqa: E402
from pv_quantile_metrics import metrics, pinball  # noqa: E402
from utils import fedgradient as fg  # noqa: E402
from utils.models import CNNRNN  # noqa: E402

RES = Path('results/parks_pv_v4_30min')


def freeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    return model


class _Enc(nn.Module):
    def __init__(self, conv_stack):
        super().__init__()
        self.conv_stack = conv_stack

    def forward(self, x):                                  # (B, T, F) -> (B, T, C), as CNNRNN.forward
        return self.conv_stack(x.transpose(1, 2)).transpose(1, 2)


class _Head(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.rnn, self.output_fc, self.horizon, self.n_q, self.step = m.rnn, m.output_fc, m.horizon, m.num_quantiles, m.step_output

    def forward(self, z):                                  # as CNNRNN.forward without attention
        out, _ = self.rnn(z)
        if self.step:
            return self.output_fc(out[:, -self.horizon:])
        return self.output_fc(out[:, -1]).view(-1, self.horizon, self.n_q)


class FrameworkCNNLSTM(nn.Module):
    """utils.models.CNNRNN ('cnn-lstm' of get_model; conv stack and LSTM as in the PV configs: 64 filters, kernel 3,
    2 LSTM layers of 64) with quantile output, split into encoder (conv stack) and head (LSTM + output layer); same
    computation as CNNRNN.forward, TFT call signature. step_output: one output per forecast step."""
    def __init__(self, n_in, horizon, n_q, depth, step_output=True):
        super().__init__()
        m = CNNRNN(n_features=n_in, output_dim=horizon, conv_type='cnn', rnn_type='lstm', bidirectional=False,
                   hyperparameters=dict(n_cnn_layers=depth, filters=64, kernel_size=3, units=64, n_rnn_layers=2,
                                        dropout=0.1, step_output=step_output), num_quantiles=n_q)
        self.encoder, self.head = _Enc(m.conv_stack), _Head(m)

    def forward(self, observed, known, static=None):
        return self.head(self.encoder(known))


class FedPer(nn.Module):
    """Shared encoder plus one head per park; ``park`` selects the head."""
    def __init__(self, make, parks):
        super().__init__()
        base = make()
        self.encoder = base.encoder
        self.heads = nn.ModuleDict({p: make().head for p in parks})

    def forward(self, known, park):
        return self.heads[park](self.encoder(known))


def train_fedper(make, data, keys, cfg, qs, device, seed, out, lr=3e-4, bs=11):
    """Each step: one batch per park, loss weighted by batch size (encoder gradient = pooled gradient), heads see
    only their park. Per park the snapshot (encoder + own head) of its best evaluation epoch is kept."""
    ft.tools.set_seed(seed)
    criterion, _ = ft.tools.make_criterion(cfg)
    names = {k: pc.park_id(k) for k in keys}
    model = FedPer(make, list(names.values())).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    clip = cfg['model']['tft']['clipnorm']
    tens = {k: (torch.as_tensor(data[k]['X_train']['known'], dtype=torch.float32, device=device),
                torch.as_tensor(data[k]['y_train'], dtype=torch.float32, device=device)) for k in keys}
    es = cfg['model']['early_stopping']

    def val(k):
        model.eval()
        X = torch.as_tensor(data[k]['X_test']['known'], dtype=torch.float32, device=device)
        with torch.no_grad():
            p = torch.cat([model(X[i:i + 256], names[k]) for i in range(0, len(X), 256)]).cpu().numpy()
        return pinball(p, data[k]['y_test'], qs), p

    best = {k: (val(k)[0], 0, copy.deepcopy(model.state_dict())) for k in keys}
    mean_best, wait, hist = np.mean([b[0] for b in best.values()]), 0, []
    for ep in range(1, cfg['model']['epochs'] + 1):
        model.train()
        g = torch.Generator().manual_seed(seed * 1000 + ep)
        perms = {k: torch.randperm(len(tens[k][1]), generator=g) for k in keys}
        steps = max(int(np.ceil(len(tens[k][1]) / bs)) for k in keys)
        for s in range(steps):
            opt.zero_grad()
            parts = [(k, perms[k][s * bs:(s + 1) * bs]) for k in keys if s * bs < len(perms[k])]
            n = sum(len(i) for _, i in parts)
            loss = sum(criterion(model(tens[k][0][i.to(device)], names[k]), tens[k][1][i.to(device)]) * len(i) / n
                       for k, i in parts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        cur = {k: val(k)[0] for k in keys}
        for k in keys:
            if cur[k] < best[k][0]:
                best[k] = (cur[k], ep, copy.deepcopy(model.state_dict()))
        m = float(np.mean(list(cur.values())))
        hist.append(dict(epoch=ep, mean_val=m, **{names[k]: cur[k] for k in keys}))
        if m < mean_best - es['min_delta']:
            mean_best, wait = m, 0
        else:
            wait += 1
        logging.info('fedper seed %s epoch %s mean pinball %.6f', seed, ep, m)
        if wait >= es['patience']:
            break
    pd.DataFrame(hist).to_csv(out / 'fedper_history.csv', index=False)
    preds = {}
    for k in keys:
        model.load_state_dict(best[k][2])
        preds[k] = (val(k)[1], best[k][1])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--depths', type=int, nargs='+', default=[5, 10])
    ap.add_argument('--arch', choices=['framework_step', 'framework'], default='framework_step',
                    help='utils.models.CNNRNN with quantile output; framework_step: step_output=True (one output per '
                         'step), framework: whole horizon from the last LSTM state. The residual variant of the first run '
                         '(cnnlstm_pers_2026-11-12_v1) used archiv/cnn_lstm_q_2026-11-12.py')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{a.gpu}')
    cfg = pc.load_cfg()
    qs = cfg['model']['tft']['quantiles']
    _, (sx, _) = ft.finetune.load_init(ft.base.Q_MODEL.format(seed=a.seed))
    data, keys, base_known, _, _ = pc.build_data(cfg, sx, RES / 'patch_cnn_2026-11-12_v1', arms=('point',))
    data = data['point']
    hp = ft.hpo.get_hyperparameters(config=cfg)
    out = a.output / f'seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() + ' + dirty'
    rows = pd.read_csv(out / 'metrics.csv').to_dict('records') if (out / 'metrics.csv').exists() else []
    done = {r['case'] for r in rows}

    def record(case, arm, depth, k, pred, best_epoch, n_params):
        it = data[k]
        rows.append(dict(case=case, arm=arm, arch=a.arch, depth=depth, park=pc.park_id(k).replace('_meter', ''), seed=a.seed,
                         best_epoch=best_epoch, n_params=n_params, date='2026-11-12', code_pin=pin,
                         **metrics(pred, it['y_test'], qs, np.asarray(it['nwp_raw_test']) > 10)))
        pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)

    for depth in a.depths:
        make = lambda: FrameworkCNNLSTM(len(base_known), cfg['model']['horizon'], len(qs), depth,  # noqa: E731
                                        step_output=a.arch == 'framework_step')
        ft.tools.set_seed(a.seed)
        m0 = make()
        state0, n_params = {k: v.clone() for k, v in m0.state_dict().items()}, sum(p.numel() for p in m0.parameters())
        n_enc = sum(p.numel() for p in m0.encoder.parameters())
        (out / f'd{depth:02d}_params.json').write_text(json.dumps(dict(total=n_params, encoder=n_enc, head=n_params - n_enc)))
        del m0
        for k in keys:
            case = f'd{depth:02d}/local/{pc.park_id(k)}'
            if case in done:
                continue
            (out / case).mkdir(parents=True, exist_ok=True)
            model, log, _ = ft.train('local_client', [ft.cat([data[k]], 'train')], [data[k]], state0, cfg, hp, device,
                                     a.seed, qs, out / case, factory=make)
            record(case, 'local', depth, k, ft.base.predict(model.eval(), data[k]['X_test'], device), log['best_epoch'], n_params)
        gdir = out / f'd{depth:02d}/global'
        gdir.mkdir(parents=True, exist_ok=True)
        if (gdir / 'fedgradient_best.pt').exists():
            gstate = torch.load(gdir / 'fedgradient_best.pt', map_location='cpu', weights_only=False)['state_dict']
        else:
            model, log, _ = ft.train('fedgradient', [ft.cat([data[k]], 'train') for k in keys], [data[k] for k in keys],
                                     state0, cfg, hp, device, a.seed, qs, gdir, factory=make)
            gstate = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            for k in keys:
                record(f'd{depth:02d}/global', 'global', depth, k, ft.base.predict(model.eval(), data[k]['X_test'], device),
                       log['best_epoch'], n_params)
        for arm, fac in (('global_head', lambda: freeze_encoder(make())), ('global_ft', make)):
            for k in keys:
                case = f'd{depth:02d}/{arm}/{pc.park_id(k)}'
                if case in done:
                    continue
                (out / case).mkdir(parents=True, exist_ok=True)
                model, log, _ = ft.train('local_client', [ft.cat([data[k]], 'train')], [data[k]], gstate, cfg, hp, device,
                                         a.seed, qs, out / case, factory=fac)
                record(case, arm, depth, k, ft.base.predict(model.eval(), data[k]['X_test'], device), log['best_epoch'], n_params)
        case = f'd{depth:02d}/fedper'
        if case not in done:
            (out / case).mkdir(parents=True, exist_ok=True)
            preds = train_fedper(make, data, keys, cfg, qs, device, a.seed, out / case)
            for k in keys:
                record(case, 'fedper', depth, k, preds[k][0], preds[k][1], n_params)
        torch.cuda.empty_cache()
    (out / 'COMPLETE').write_text('done\n')


if __name__ == '__main__':
    main()
