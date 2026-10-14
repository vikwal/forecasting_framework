"""Apply the trained split-2 models (local per client and central) to the holdout parks they have
never seen: the fold models (scripts/final_folds.py, models/split2_folds/) on the fold windows and the
test-year models (train_cl.py --save_model, models/<study>_seed<s>.pt) on the test year. Each model
scales the holdout data with its own scaler.

    python scripts/apply_holdout.py -c configs/parks_v1/split2/final/config_parks_v1_curt_v11_fl_fedgradient_static_s2 \
        --central-config configs/parks_v1/split2/final/config_parks_v1_curt_v11_cl80_static_s2 \
        --which folds test --seeds 42 43 44 --gpu 0
    python scripts/apply_holdout.py --summary results/split2_holdout/holdout_eval.csv

Output: results/split2_holdout/holdout_eval.csv (unit, kind, seed, window, park, n, rmse, mae, r2) and
one predictions pickle per (unit, kind, seed).
"""
import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import final_training as ft, hpo, models, preprocessing, tools   # noqa: E402

COLS = ['unit', 'kind', 'seed', 'window', 'park', 'n', 'rmse', 'mae', 'r2']


def clean_state(state: dict) -> dict:
    return {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state.items()}


def load_model(cfg: dict, hp: dict, path: str, device: str, X=None):
    if X is not None:
        cfg['model']['feature_dim'] = tools.get_feature_dim(X=X)
    model = models.get_model(config=cfg, hyperparameters=hp)
    model.load_state_dict(clean_state(torch.load(path, map_location='cpu')))
    return model.to(device).eval()


def test_mask(prepared: dict, boundaries, fold: int) -> np.ndarray:   # unused signature parity
    return np.ones(len(prepared['y_test']), dtype=bool)


def predict_test(model, prepared_clients: dict, target: str, device: str, batch_size: int):
    preds, rows = {}, []
    for cid, c in prepared_clients.items():
        for key, prepared in c['stations'].items():
            if len(prepared.get('y_test', [])) == 0:
                continue
            times = hpo._sample_times(prepared['index_test'])
            p, t = ft.predict(model, prepared['X_test'], prepared['y_test'], device, batch_size)
            pred, true = ft._frames(p, t, times, c['scaler_y'])
            preds[(key, target)] = {'pred': pred, 'true': true}
            rows.append({'key': key, **ft.metrics(pred, true)})
    return preds, pd.DataFrame(rows)


def run_unit(unit: str, setting: str, cfg_path: str, config: dict, holdout: list, args, device: str) -> list:
    cfg = ft.final_config(config, args.model, setting)      # final configs already carry the four ICON runs
    study_name = ft.study_names(cfg_path, config, args.model, setting, [unit] if setting == 'local' else None)[unit]
    hp, study = ft.best_hyperparameters(cfg, study_name)
    features = preprocessing.get_features(config=cfg)
    boundaries = hpo.fold_boundaries(cfg)
    target = preprocessing.get_target_cols(cfg)[0]
    rows = []
    os.makedirs(args.out_dir, exist_ok=True)
    if 'folds' in args.which:
        with open(os.path.join(args.model_dir, f'scaler_{study_name}.pkl'), 'rb') as f:
            sc = pickle.load(f)[unit]
        prepared = ft.prepare_clients(cfg, features, {'holdout': holdout}, pool_scalers=False, scalers={'holdout': sc})
        for seed in args.seeds:
            preds_all = {}
            for fold in range(len(boundaries) - 1):
                path = os.path.join(args.model_dir, f'{study_name}_fold{fold}_seed{seed}.pt')
                model = load_model(cfg, hp, path, device, next(iter(prepared['holdout']['stations'].values()))['X_train'])
                preds, ev = ft.predict_window(model, prepared, boundaries, fold, target, device, hp['batch_size'])
                preds_all[fold] = preds
                for r in ev.itertuples():
                    rows.append({'unit': unit, 'kind': 'fold', 'seed': seed, 'window': fold, 'park': r.key, 'n': r.n,
                                 'rmse': r.rmse, 'mae': r.mae, 'r2': r.r2})
                agg = ft.pooled(ev)
                logging.info(f"[{unit}] fold model {fold} seed {seed} on {len(ev)} holdout parks: rmse {agg['rmse']:.4f} "
                             f"mae {agg['mae']:.4f} (n {agg['n']})")
                del model
            pd.to_pickle({'unit': unit, 'kind': 'fold', 'seed': seed, 'study': study_name, 'predictions': preds_all},
                         os.path.join(args.out_dir, f'{study_name}_folds_seed{seed}_holdout.pkl'))
        del prepared
    if 'test' in args.which:
        for seed in args.seeds:
            with open(os.path.join('models', 'scaler', f'scaler_{study_name}_seed{seed}.pkl'), 'rb') as f:
                sx = pickle.load(f)
            prepared = ft.prepare_clients(cfg, features, {'holdout': holdout}, pool_scalers=False,
                                          scalers={'holdout': {'scaler_x': sx, 'scaler_y': None}})
            model = load_model(cfg, hp, os.path.join('models', f'{study_name}_seed{seed}.pt'), device,
                               next(iter(prepared['holdout']['stations'].values()))['X_test'])
            preds, ev = predict_test(model, prepared, target, device, hp['batch_size'])
            for r in ev.itertuples():
                rows.append({'unit': unit, 'kind': 'test', 'seed': seed, 'window': 'test', 'park': r.key, 'n': r.n,
                             'rmse': r.rmse, 'mae': r.mae, 'r2': r.r2})
            agg = ft.pooled(ev)
            logging.info(f"[{unit}] test model seed {seed} on {len(ev)} holdout parks: rmse {agg['rmse']:.4f} "
                         f"mae {agg['mae']:.4f} (n {agg['n']})")
            pd.to_pickle({'unit': unit, 'kind': 'test', 'seed': seed, 'study': study_name, 'predictions': preds},
                         os.path.join(args.out_dir, f'{study_name}_test_seed{seed}_holdout.pkl'))
            del model, prepared
    return rows


def summary(csv: str) -> str:
    d = pd.read_csv(csv)
    d['mse'] = d.rmse ** 2
    def pooled(g):
        return pd.Series({'rmse': np.sqrt((g.n * g.mse).sum() / g.n.sum()), 'mae': (g.n * g.mae).sum() / g.n.sum(),
                          'r2': g.r2.mean()})
    per = d.groupby(['unit', 'kind', 'window', 'seed']).apply(pooled).groupby(['unit', 'kind', 'window']).mean().reset_index()
    per['group'] = np.where(per.unit == 'central', 'central', 'local')
    out = ['Holdout parks: pooled RMSE / MAE over parks, mean over seeds; "local" = mean over the 8 client models.\n']
    t = per.groupby(['kind', 'window', 'group'])[['rmse', 'mae', 'r2']].mean().round(4).reset_index()
    out.append('| kind | window | central rmse | local rmse | Δ | central mae | local mae | central r2 | local r2 |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for (k, w), g in t.groupby(['kind', 'window'], sort=False):
        c = g[g.group == 'central'].iloc[0] if (g.group == 'central').any() else None
        l = g[g.group == 'local'].iloc[0] if (g.group == 'local').any() else None
        f = lambda x, col: f'{x[col]:.4f}' if x is not None else ''
        delta = f'{l.rmse - c.rmse:+.4f}' if c is not None and l is not None else ''
        out.append(f'| {k} | {w} | {f(c, "rmse")} | {f(l, "rmse")} | {delta} | {f(c, "mae")} | {f(l, "mae")} | '
                   f'{f(c, "r2")} | {f(l, "r2")} |')
    out.append('\nPer unit (RMSE, pooled over holdout parks, mean over seeds):\n')
    piv = per.pivot_table(index='unit', columns=['kind', 'window'], values='rmse').round(4)
    piv.columns = [f'{k} {w}' for k, w in piv.columns]
    out.append('| unit | ' + ' | '.join(piv.columns) + ' |')
    out.append('|---|' + '---|' * len(piv.columns))
    for u, r in piv.iterrows():
        out.append(f'| {u} | ' + ' | '.join('' if v != v else f'{v:.4f}' for v in r) + ' |')
    return '\n'.join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--config', help='FL final config (fl.clients, data.val_files = holdout parks)')
    ap.add_argument('--central-config', default=None, help='cl80 final config (adds the central models)')
    ap.add_argument('-m', '--model', default='tft')
    ap.add_argument('--clients', nargs='+', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    ap.add_argument('--which', nargs='+', default=['folds', 'test'], choices=['folds', 'test'])
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--model-dir', default=os.path.join('models', 'split2_folds'))
    ap.add_argument('--out-dir', default=os.path.join('results', 'split2_holdout'))
    ap.add_argument('--summary', default=None, help='only print the summary of this CSV')
    args = ap.parse_args()
    if args.summary:
        print(summary(args.summary))
        return 0
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else 'cuda'
    cfg_path = args.config if args.config.endswith('.yaml') else f'{args.config}.yaml'
    config = tools.load_config(cfg_path)
    holdout = list(config['data'].get('val_files') or config['data'].get('holdout_files') or [])
    if not holdout:
        raise SystemExit('no holdout parks (data.val_files / holdout_files) in the config')
    logging.info(f'{len(holdout)} holdout parks, seeds {args.seeds}, {args.which}')
    out_csv = os.path.join(args.out_dir, 'holdout_eval.csv')
    units = [(c, 'local', cfg_path, config) for c in (args.clients or config['fl']['clients'])]
    if args.central_config:
        cp = args.central_config if args.central_config.endswith('.yaml') else f'{args.central_config}.yaml'
        units.append(('central', 'central', cp, tools.load_config(cp)))
    for unit, setting, path, conf in units:
        rows = run_unit(unit, setting, path, conf, holdout, args, device)
        os.makedirs(args.out_dir, exist_ok=True)
        pd.DataFrame(rows, columns=COLS).to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
    print(summary(out_csv))
    return 0


if __name__ == '__main__':
    sys.exit(main())
