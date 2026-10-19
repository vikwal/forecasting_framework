"""Evaluate fold checkpoints on held-out (zero-shot) and in-sample farms.

For each fold k the checkpoint trained by train_cl.py on the 80 training
farms is evaluated per farm on the test window, with the fold's own training
scaler. Metrics are reported for the seasonal sub-periods P1/P2 and the full
window (masks over forecast-window start times — the model is NOT retrained
between periods).

Usage (from repo root, venv frcst):
  python iaims26/eval_zeroshot.py --folds 0 1 2 3 4 --splits heldout insample
  python iaims26/eval_zeroshot.py --smoke            # smoke configs instead

Output: iaims26/results/zeroshot/zeroshot_fold{k}.csv
  columns: station_id, fold, split, period, R2, RMSE, MAE, n_windows
"""

import argparse
import gc
import logging

import numpy as np
import torch

import common

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

COLUMNS = ['station_id', 'fold', 'split', 'period', 'R2', 'RMSE', 'MAE', 'n_windows']


def evaluate_fold(fold: int, splits, smoke: bool = False, device: str = None,
                  config_name: str = None, out_tag: str = None):
    """Evaluate one fold config — or, with config_name set, any generated
    config (e.g. config_wind_iaims26g100 with splits=['insample'])."""
    from utils import tools, eval as eval_utils, preprocessing

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    name = config_name or common.config_name_for_fold(fold, smoke=smoke)
    config = common.load_generated_config(name)
    ckpt = common.checkpoint_path(name)
    scaler = common.load_scaler(common.scaler_path(name))
    model, hyperparameters, dims = common.load_pretrained_tft(config, ckpt, device=device)
    logging.info(f'fold {fold}: loaded {ckpt.name} (dims={dims})')

    tag = out_tag or (f'{"smoke" if smoke else "fold"}{fold}')
    out_csv = common.RESULTS_DIR / 'zeroshot' / f'zeroshot_{tag}.csv'
    rows = []

    station_groups = {}
    if 'heldout' in splits and config['data'].get('val_files'):
        station_groups['heldout'] = list(config['data']['val_files'])
    if 'insample' in splits:
        station_groups['insample'] = list(config['data']['files'])

    features = preprocessing.get_features(config=config)
    for split, stations in station_groups.items():
        dfs = common.get_station_dfs(config, stations)
        eval_cfg = common.make_eval_config(config, config['data']['test_start'],
                                           config['data']['test_end'])
        generator = tools.create_data_generator(dfs, eval_cfg, features, scaler_x=scaler)
        for data_dict in generator:
            key = common.station_key_to_id(data_dict['key'])
            X_test, y_test = data_dict['X_test'], data_dict['y_test']
            index_test = data_dict['index_test']
            y_true, y_pred = tools.get_y(X_test=X_test, y_test=y_test, model=model,
                                         scaler_y=data_dict['scalers']['y'], device=device)
            masks = common.period_masks(index_test)
            for period, mask in masks.items():
                if not mask.any():
                    continue
                metrics = eval_utils.get_metrics(y_pred=y_pred[mask], y_true=y_true[mask])
                rows.append([key, fold, split, period,
                             metrics['R^2'][0], metrics['RMSE'][0], metrics['MAE'][0],
                             int(mask.sum())])
            logging.info(f'fold {fold} {split} {key}: '
                         f'R2={rows[-1][4]:.4f} RMSE={rows[-1][5]:.4f} (full)')
        del dfs
        gc.collect()

    if out_csv.exists():
        out_csv.unlink()
    common.append_rows_atomic(out_csv, rows, COLUMNS)
    logging.info(f'fold {fold}: wrote {len(rows)} rows to {out_csv}')
    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--splits', nargs='+', default=['heldout'],
                        choices=['heldout', 'insample'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default=None)
    parser.add_argument('--config-name', default=None,
                        help='evaluate a single generated config instead of '
                             'fold configs (e.g. config_wind_iaims26g100)')
    parser.add_argument('--out-tag', default=None,
                        help='output CSV tag when using --config-name (e.g. g100)')
    args = parser.parse_args()

    if args.config_name:
        evaluate_fold(-1, args.splits, device=args.device,
                      config_name=args.config_name,
                      out_tag=args.out_tag or args.config_name.split('iaims26')[-1])
        return

    for fold in args.folds:
        evaluate_fold(fold, args.splits, smoke=args.smoke, device=args.device)


if __name__ == '__main__':
    main()
