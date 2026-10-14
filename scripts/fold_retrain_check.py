"""Does retraining every 4 months help a local model? For every FL client (own stations only, best trial
of its hpo_local.py study) train the model of each expanding-window date fold (hpo.fold_boundaries) and
evaluate every fold model on its own and on all later validation windows:

    fold 0 model -> windows 0, 1, 2      (the "train once" model over the whole validation year)
    fold 1 model -> windows 1, 2
    fold 2 model -> window 2             (the "retrain every window" scheme = fold j on window j)

Early stopping is the one of the HPO (on the fold's own window, restore best weights), so a fold model
is evaluated in-sample w.r.t. early stopping on its own window and out-of-sample on the later ones.

Usage (per GPU one process, clients split between them):
    python scripts/fold_retrain_check.py -c configs/parks_v1/split2/config_parks_v1_curt_v11_fl_fedgradient_static_s2 \
        --clients R0 R1 --gpu 0 --seeds 42 43 44 --forecast-hours 06 09 12 15 --out results/fold_retrain/s2_local.csv
    python scripts/fold_retrain_check.py --summary results/fold_retrain/s2_local.csv
"""
import argparse
import gc
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import data_cache, hpo, local_training, preprocessing, tools   # noqa: E402
from hpo_local import client_hpo_configs                                  # noqa: E402
from train_local import local_name_from                                   # noqa: E402

COLS = ['client', 'seed', 'train_fold', 'eval_window', 'n', 'mse', 'rmse', 'mae', 'epochs', 'best_epoch',
        'train_samples', 'train_min']


def predict(model, X, y, device, batch_size: int):
    """Point predictions and targets of a TFT model for one (X, y) set, as in the training loop."""
    loader = tools.create_pytorch_dataloader(X, y, batch_size=batch_size, shuffle=False, device=device,
                                             num_workers=0, drop_last=False)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                obs, known, static, t = [b.to(device) for b in batch]
            elif len(batch) == 3:
                obs, known, t = [b.to(device) for b in batch]
                static = None
            else:
                raise ValueError(f'unexpected TFT batch of length {len(batch)}')
            preds.append(model(obs, known, static).detach().cpu())
            targets.append(t.detach().cpu())
    p = torch.cat(preds).numpy()
    t = torch.cat(targets).numpy()
    if p.ndim == 3 and t.ndim == 2:          # single quantile
        p = p[..., 0]
    if p.shape != t.shape:
        raise ValueError(f'prediction shape {p.shape} != target shape {t.shape}')
    return p, t


def metrics(p, t) -> dict:
    err = p - t
    return {'n': int(t.shape[0]), 'mse': float(np.mean(err ** 2)), 'rmse': float(np.sqrt(np.mean(err ** 2))),
            'mae': float(np.mean(np.abs(err)))}


def run_client(cid: str, cfg_path: str, seeds: list, device: str, out: str, model_name: str) -> None:
    cfg = tools.load_config(cfg_path)
    cfg['model']['fl'] = False            # as hpo_cl.py sets them after loading
    cfg['model']['name'] = model_name
    cfg['model']['verbose'] = 0
    stem = os.path.basename(cfg_path).replace('config_', '').replace('.yaml', '')
    study_name = f"cl_m-{model_name}_out-{cfg['model']['horizon']}_freq-{cfg['data']['freq']}_{stem}"
    study = hpo.load_study(cfg['hpo']['studies_path'], study_name, required=True)
    hp = hpo.get_hyperparameters(config=cfg, study=study)
    logging.info(f'[{cid}] study {study_name}: best trial {study.best_trial.number} '
                 f'(value {study.best_value:.4f}); hyperparameters {hp}')
    features = preprocessing.get_features(config=cfg)
    lazy, cache_id = data_cache.create_or_load_preprocessed_data(config=cfg, features=features, model_name=model_name,
                                                                 force_reprocess=False, use_cache=True)
    n_folds = len(lazy)
    logging.info(f'[{cid}] cache {cache_id}, {n_folds} folds: '
                 + ', '.join(f"{i}: {lazy.get_fold_info(i)['train_samples']}/{lazy.get_fold_info(i)['val_samples']}"
                             for i in range(n_folds)))
    vals = [lazy[j][1] for j in range(n_folds)]                # (X_val, y_val) per window, kept in memory (small)
    for seed in seeds:
        for i in range(n_folds):
            tools.set_seed(int(seed))
            train, val = lazy[i]
            cfg['model_name'] = f'fold_retrain_{stem}_s{seed}_f{i}'
            t0 = time.time()
            history, model = tools.training_pipeline(train=train, val=val, hyperparameters=hp, config=cfg,
                                                     device=device)
            train_min = (time.time() - t0) / 60
            epochs = len(history['val_rmse'])
            best_epoch = int(np.argmin(history['val_rmse'])) + 1
            rows = []
            for j in range(i, n_folds):
                p, t = predict(model, vals[j][0], vals[j][1], device, hp['batch_size'])
                m = metrics(p, t)
                rows.append({'client': cid, 'seed': seed, 'train_fold': i, 'eval_window': j, **m, 'epochs': epochs,
                             'best_epoch': best_epoch, 'train_samples': len(train[1]), 'train_min': round(train_min, 2)})
                logging.info(f"[{cid}] seed {seed} fold {i} -> window {j}: rmse {m['rmse']:.4f} mae {m['mae']:.4f} "
                             f"(n {m['n']}, {epochs} epochs, best {best_epoch}, {train_min:.1f} min)")
            pd.DataFrame(rows, columns=COLS).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
            del model, history
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def summary(csv: str) -> str:
    """Markdown: per client and pooled, RMSE of the fold-0 model vs. the retrained model on every window,
    plus the whole validation year (train once = fold 0 everywhere; retrain = fold j on window j)."""
    df = pd.read_csv(csv)
    n_folds = int(df.eval_window.max()) + 1
    # seed-mean MSE per (client, train_fold, eval_window); n is identical across seeds
    g = df.groupby(['client', 'train_fold', 'eval_window']).agg(mse=('mse', 'mean'), n=('n', 'first'),
                                                                 seeds=('seed', 'nunique')).reset_index()
    out = []
    out.append(f'Seeds per cell: {int(g.seeds.min())}–{int(g.seeds.max())}; RMSE in capacity units; '
               f'Δ = retrained − fold-0 model (negative = retraining helps).\n')
    head = '| Client | ' + ' | '.join(f'W{j} fold0 | W{j} fold{j} | Δ' for j in range(1, n_folds)) \
           + ' | Jahr einmal | Jahr retrain | Δ |'
    out.append(head)
    out.append('|' + '---|' * (1 + 3 * (n_folds - 1) + 3))
    pooled = {'once': [], 'retrain': [], 'n': []}
    per_window = {j: {'once': [], 'retrain': [], 'n': []} for j in range(1, n_folds)}
    for cid, gc_ in g.groupby('client'):
        cell = {(r.train_fold, r.eval_window): (r.mse, r.n) for r in gc_.itertuples()}
        row = [cid]
        once_mse, retr_mse, ns = [], [], []
        for j in range(n_folds):
            m0, n = cell[(0, j)]
            mj, _ = cell[(j, j)]
            once_mse.append(m0); retr_mse.append(mj); ns.append(n)
            if j >= 1:
                row += [f'{np.sqrt(m0):.4f}', f'{np.sqrt(mj):.4f}', f'{np.sqrt(mj) - np.sqrt(m0):+.4f}']
                per_window[j]['once'].append(m0); per_window[j]['retrain'].append(mj); per_window[j]['n'].append(n)
        ns = np.array(ns)
        y_once = np.sqrt(np.sum(np.array(once_mse) * ns) / ns.sum())
        y_retr = np.sqrt(np.sum(np.array(retr_mse) * ns) / ns.sum())
        row += [f'{y_once:.4f}', f'{y_retr:.4f}', f'{y_retr - y_once:+.4f}']
        out.append('| ' + ' | '.join(row) + ' |')
        pooled['once'].append(np.sum(np.array(once_mse) * ns)); pooled['retrain'].append(np.sum(np.array(retr_mse) * ns))
        pooled['n'].append(ns.sum())
    row = ['**alle (gepoolt)**']
    for j in range(1, n_folds):
        n = np.array(per_window[j]['n'])
        o = np.sqrt(np.sum(np.array(per_window[j]['once']) * n) / n.sum())
        r = np.sqrt(np.sum(np.array(per_window[j]['retrain']) * n) / n.sum())
        row += [f'{o:.4f}', f'{r:.4f}', f'{r - o:+.4f}']
    o = np.sqrt(np.sum(pooled['once']) / np.sum(pooled['n']))
    r = np.sqrt(np.sum(pooled['retrain']) / np.sum(pooled['n']))
    row += [f'{o:.4f}', f'{r:.4f}', f'{r - o:+.4f}']
    out.append('| ' + ' | '.join(row) + ' |')
    # all fold models on the last window: does the model age with distance to its training end?
    out.append(f'\nLetztes Fenster W{n_folds - 1}, RMSE je Trainingsstand (Seed-Mittel, gepoolt über Clients):\n')
    out.append('| Modell | ' + ' | '.join(sorted(g.client.unique())) + ' | gepoolt |')
    out.append('|' + '---|' * (2 + g.client.nunique()))
    for i in range(n_folds):
        sub = g[(g.train_fold == i) & (g.eval_window == n_folds - 1)].set_index('client')
        cells = [f'{np.sqrt(sub.loc[c, "mse"]):.4f}' if c in sub.index else '–' for c in sorted(g.client.unique())]
        pooled_i = np.sqrt(np.sum(sub.mse * sub.n) / sub.n.sum()) if len(sub) else float('nan')
        out.append(f'| fold {i} | ' + ' | '.join(cells) + f' | {pooled_i:.4f} |')
    return '\n'.join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-c', '--config', help='FL config with fl.clients and the hpo section (date folds)')
    ap.add_argument('-m', '--model', default='tft')
    ap.add_argument('--clients', nargs='+', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--out', default='results/fold_retrain/fold_retrain.csv')
    ap.add_argument('--run-dir', default=None, help='where the client configs are written (default runs/fold_retrain/<name>)')
    ap.add_argument('--forecast-hours', nargs='+', default=None,
                    help='NWP issue hours used for training and evaluation (default: the config\'s, e.g. 09 only)')
    ap.add_argument('--summary', metavar='CSV', help='only print the markdown summary of an existing result file')
    args = ap.parse_args()
    if args.summary:
        print(summary(args.summary))
        return 0
    cfg_path = args.config if args.config.endswith('.yaml') else f'{args.config}.yaml'
    config = tools.load_config(cfg_path)
    name = local_name_from(cfg_path, config)
    run_dir = args.run_dir or os.path.join('runs', 'fold_retrain', name)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(run_dir, 'logs',
                                                                   f"fold_retrain_{'-'.join(args.clients or ['all'])}.log")),
                                  logging.StreamHandler()])
    # same round trip as hpo_local.py (write + load), so the data-cache key matches the HPO's
    configs = client_hpo_configs(config, args.clients, kfolds=config['hpo'].get('kfolds'))
    if args.forecast_hours:          # e.g. 06 09 12 15: four samples per day instead of the 09 run only
        for cfg in configs.values():
            cfg['data']['forecast_hours'] = [str(h).zfill(2) for h in args.forecast_hours]
            cfg['data'].pop('train_forecast_hours', None)
    paths = local_training.write_configs(configs, os.path.join(run_dir, 'configs'), name)
    device = f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else \
        ('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'{len(paths)} clients, seeds {args.seeds}, device {device}, forecast hours '
                 f"{next(iter(configs.values()))['data'].get('forecast_hours')}, results -> {args.out}")
    for cid, path in paths.items():
        run_client(cid, f'{path}.yaml', args.seeds, device, args.out, args.model)
    print(summary(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
