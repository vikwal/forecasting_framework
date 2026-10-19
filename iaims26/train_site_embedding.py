"""Train and evaluate the site-ID-embedding control model.

Modes:
  --global100   train on all 100 farms, per-farm in-sample evaluation
                (comparison partner for the physical-static global model)
  --fold K      train on fold K's 80 farms, zero-shot eval on the 20 held-out
                farms with mean-embedding imputation for unseen IDs
                (expected failure case: IDs cannot transfer)

The data pipeline runs with static_features=[] (the integer index must never
pass through the global scaler); the station index is injected as the static
tensor after windowing, using the station_index map from folds.json.

Output: iaims26/results/site_embedding/siteid_{mode}.csv
        iaims26/models/siteid_{mode}.pt
"""

import argparse
import copy
import gc
import logging

import numpy as np
import torch

import common
import site_embedding
import trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

COLUMNS = ['station_id', 'mode', 'split', 'period', 'R2', 'RMSE', 'MAE', 'n_windows']


station_key_to_id = common.station_key_to_id


def build_arrays(config, stations, station_index, scaler, features):
    """Load + window per farm, inject the station index as static feature.

    Returns (X_train, y_train, X_val, y_val, test_data) where test_data maps
    station id -> (X_test, y_test, index_test).
    """
    from utils import tools

    dfs = common.get_station_dfs(config, stations)
    gen = tools.create_data_generator(dfs, copy.deepcopy(config), features,
                                      scaler_x=scaler)
    Xtr_parts = {'observed': [], 'known': [], 'static': []}
    ytr_parts = []
    Xte_parts = {'observed': [], 'known': [], 'static': []}
    yte_parts = []
    test_data = {}
    for d in gen:
        sid = station_key_to_id(d['key'])
        idx = float(station_index[sid])
        for part, X, y_list in (('train', d['X_train'], ytr_parts),
                                ('test', d['X_test'], yte_parts)):
            n = X['observed'].shape[0]
            X = dict(X)
            X['static'] = np.full((n, 1), idx, dtype=np.float32)
            target = Xtr_parts if part == 'train' else Xte_parts
            for k in target:
                target[k].append(X[k])
            y_list.append(d['y_train'] if part == 'train' else d['y_test'])
            if part == 'test':
                test_data[sid] = (X, d['y_test'], d['index_test'])
    X_train = {k: np.concatenate(v) for k, v in Xtr_parts.items()}
    y_train = np.concatenate(ytr_parts)
    X_val = {k: np.concatenate(v) for k, v in Xte_parts.items()}
    y_val = np.concatenate(yte_parts)
    del dfs
    gc.collect()
    return X_train, y_train, X_val, y_val, test_data


def main():
    from utils import tools, eval as eval_utils, preprocessing
    import hpo_zeroshot

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--global100', action='store_true')
    group.add_argument('--fold', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    folds = common.load_folds()
    station_index = folds['station_index']

    # Config: fold config (if HPO params were applied they are inherited),
    # but with static features disabled for the pipeline.
    if args.global100:
        config = common.load_generated_config('config_wind_iaims26g100')
        train_stations = sorted(str(s) for s in config['data']['files'])
        eval_groups = {'insample': train_stations}
        mode = 'global100'
    else:
        config = common.load_generated_config(common.config_name_for_fold(args.fold))
        train_stations = list(config['data']['files'])
        eval_groups = {'heldout': list(config['data']['val_files']),
                       'insample': train_stations}
        mode = f'fold{args.fold}'
    config['params']['static_features'] = []
    features = preprocessing.get_features(config=config)
    assert features['static'] == []

    hyperparameters = common.resolve_hyperparameters(config)

    # Scaler over the 80/100 training farms (static-free feature space)
    logging.info(f'{mode}: loading training farms + fitting scaler…')
    dfs_train = common.get_station_dfs(config, train_stations)
    scaler = hpo_zeroshot.fit_fold_scaler(dfs_train, config)
    del dfs_train
    gc.collect()

    X_train, y_train, X_val, y_val, test_data_train = build_arrays(
        config, train_stations, station_index, scaler, features)
    logging.info(f'{mode}: train windows={len(y_train)}, val windows={len(y_val)}')

    model = site_embedding.build_site_embedding_tft(
        config, hyperparameters, num_stations=len(station_index),
        known_dim=X_train['known'].shape[-1])
    model.set_known_ids([station_index[s] for s in train_stations])

    history, model, epochs_run = trainer.train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=hyperparameters['batch_size'], lr=hyperparameters['lr'],
        device=device, epochs=args.epochs, patience=args.patience,
        clipnorm=hyperparameters.get('clipnorm'), verbose=True)
    logging.info(f'{mode}: trained {epochs_run} epochs, '
                 f'best val_rmse={min(history["val_rmse"]):.5f}')

    common.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), common.MODELS_DIR / f'siteid_{mode}.pt')

    # Per-farm evaluation
    rows = []
    model.eval()
    for split, stations in eval_groups.items():
        if split == 'insample':
            per_farm = {s: test_data_train[s] for s in stations}
        else:  # heldout farms need their own data pass (scaled with the training scaler)
            _, _, _, _, per_farm = build_arrays(
                config, stations, station_index, scaler, features)
        for sid, (X_test, y_test, index_test) in per_farm.items():
            y_true, y_pred = tools.get_y(X_test=X_test, y_test=y_test,
                                         model=model, scaler_y=None, device=device)
            for period, mask in common.period_masks(index_test).items():
                if not mask.any():
                    continue
                m = eval_utils.get_metrics(y_pred=y_pred[mask], y_true=y_true[mask])
                rows.append([sid, mode, split, period,
                             m['R^2'][0], m['RMSE'][0], m['MAE'][0], int(mask.sum())])

    out_csv = common.RESULTS_DIR / 'site_embedding' / f'siteid_{mode}.csv'
    if out_csv.exists():
        out_csv.unlink()
    common.append_rows_atomic(out_csv, rows, COLUMNS)
    logging.info(f'{mode}: wrote {len(rows)} rows to {out_csv}')


if __name__ == '__main__':
    main()
