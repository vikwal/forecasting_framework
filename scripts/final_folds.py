"""Final models on the three HPO date folds with the best trial of each study: central (cl80),
local (one model per FL client) or federated (FedAvg / FedGradient, global weights).

Per unit, seed and fold: train on the expanding window (early stopping on the fold's own window
as in the HPO), predict every station in the window, save weights
(models/split2_folds/<study>_fold<i>_seed<s>.pt), the result pickle
(results/split2_folds/<study>_fold<i>_seed<s>.pkl: hyperparameters, config, history, predictions
{(station, target): {'pred', 'true'}} with issue-time index and t+1..t+H columns, evaluation per
station) and one summary row (results/split2_folds/summary.csv).

    python scripts/final_folds.py --setting central -c configs/parks_v1/split2/config_parks_v1_curt_v11_cl80_static_s2 \
        --seeds 42 43 44 --gpu 0 --forecast-hours 06 09 12 15
    python scripts/final_folds.py --setting local -c configs/parks_v1/split2/config_parks_v1_curt_v11_fl_fedgradient_static_s2 \
        --clients R0 R1 --gpu 1 --forecast-hours 06 09 12 15
    python scripts/final_folds.py --setting fl -c configs/parks_v1/split2/config_parks_v1_curt_v11_fl_fedavg_static_s2 \
        --forecast-hours 06 09 12 15
"""
import argparse
import gc
import logging
import os
import pickle
import sys
from datetime import datetime

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import final_training as ft, hpo, preprocessing, tools   # noqa: E402

COLS = ['setting', 'unit', 'study', 'trial', 'seed', 'fold', 'n', 'rmse', 'mae', 'epochs', 'best', 'train_samples',
        'train_min', 'result']


def run_unit(setting: str, unit: str, study_name: str, cfg: dict, args, device: str) -> None:
    cfg = ft.final_config(cfg, args.model, setting, args.forecast_hours)
    hp, study = ft.best_hyperparameters(cfg, study_name)
    logging.info(f'[{unit}] study {study_name}: best trial {study.best_trial.number} (value {study.best_value:.4f}); '
                 f'hyperparameters {hp}')
    features = preprocessing.get_features(config=cfg)
    boundaries = hpo.fold_boundaries(cfg)
    clients = ft.unit_clients(cfg, setting, unit)
    pool = setting == 'fl' and bool(cfg['fl'].get('global_scaler', False))
    prepared = ft.prepare_clients(cfg, features, clients, pool_scalers=pool)
    os.makedirs(args.model_dir, exist_ok=True)
    scalers = {cid: {'scaler_x': c['scaler_x'], 'scaler_y': c['scaler_y']} for cid, c in prepared.items()}
    with open(os.path.join(args.model_dir, f'scaler_{study_name}.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
    target = preprocessing.get_target_cols(cfg)[0]
    folds = args.folds if args.folds is not None else list(range(len(boundaries) - 1))
    for seed in args.seeds:
        for fold in folds:
            model, state, info = ft.train_fold(setting, cfg, hp, prepared, boundaries, fold, seed, device, args.gpu)
            preds, evaluation = ft.predict_window(model, prepared, boundaries, fold, target, device, hp['batch_size'])
            agg = ft.pooled(evaluation)
            payload = {'setting': setting, 'unit': unit, 'study': study_name, 'trial': study.best_trial.number,
                       'seed': seed, 'fold': fold, 'boundaries': [str(b.date()) for b in boundaries],
                       'hyperparameters': hp, 'config': cfg, 'history': info.pop('history'),
                       'comm_stats': info.pop('comm_stats', None), 'info': info,
                       'evaluation': evaluation, 'pooled': agg, 'predictions': preds,
                       'created': datetime.now().isoformat(timespec='seconds')}
            path = ft.save_fold(args.out_dir, args.model_dir, study_name, fold, seed, state, payload)
            row = {'setting': setting, 'unit': unit, 'study': study_name, 'trial': study.best_trial.number,
                   'seed': seed, 'fold': fold, 'n': agg['n'], 'rmse': round(agg['rmse'], 5), 'mae': round(agg['mae'], 5),
                   'epochs': info['epochs'], 'best': info['best'], 'train_samples': info['train_samples'],
                   'train_min': info['train_min'], 'result': path}
            summary = os.path.join(args.out_dir, 'summary.csv')
            pd.DataFrame([row], columns=COLS).to_csv(summary, mode='a', header=not os.path.exists(summary), index=False)
            logging.info(f"[{unit}] seed {seed} fold {fold}: rmse {agg['rmse']:.4f} mae {agg['mae']:.4f} "
                         f"(n {agg['n']}, {info['epochs']} epochs/rounds, best {info['best']}, "
                         f"{info['train_min']:.1f} min) -> {path}")
            del model, state, preds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    del prepared
    gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--setting', required=True, choices=ft.SETTINGS)
    ap.add_argument('-c', '--config', required=True, help='cl80 config (central) or FL config (local, fl)')
    ap.add_argument('-m', '--model', default='tft')
    ap.add_argument('--clients', nargs='+', default=None, help='local: subset of clients (default: all)')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    ap.add_argument('--folds', nargs='+', type=int, default=None, help='subset of folds (default: all)')
    ap.add_argument('--gpu', type=int, default=None, help='GPU index (central/local: device; fl: FL_GPU_ORDER)')
    ap.add_argument('--forecast-hours', nargs='+', default=None, help="ICON runs, e.g. 06 09 12 15 (default: config)")
    ap.add_argument('--out-dir', default=os.path.join('results', 'split2_folds'))
    ap.add_argument('--model-dir', default=os.path.join('models', 'split2_folds'))
    args = ap.parse_args()

    cfg_path = args.config if args.config.endswith('.yaml') else f'{args.config}.yaml'
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    config = tools.load_config(cfg_path)
    if args.setting != 'central' and not (config.get('fl') or {}).get('clients'):
        raise SystemExit('local / fl need an FL config with fl.clients')
    device = f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else \
        ('cuda' if torch.cuda.is_available() else 'cpu')
    studies = ft.study_names(cfg_path, config, args.model, args.setting, args.clients)
    logging.info(f'{args.setting}: {len(studies)} unit(s), seeds {args.seeds}, device {device}, '
                 f'forecast hours {args.forecast_hours or config["data"].get("forecast_hours")}')
    for unit, study_name in studies.items():
        run_unit(args.setting, unit, study_name, config, args, device)
    return 0


if __name__ == '__main__':
    sys.exit(main())
