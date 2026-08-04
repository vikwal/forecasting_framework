#!/usr/bin/env python3
"""
Final training for the TFT non-GNN benchmark (tft_bc), using the best
hyperparameters found by an existing hpo_cl_tft_bc.py Optuna study.

Mirrors hpo_cl_tft_bc.py's data pipeline (data_cache.create_or_load_preprocessed_data,
next_n_grid_points/next_n_grid_ecmwf/next_n_stations preprocessing-level params) but
trains a single final model instead of running trials, analogous to what
geostatistics/train_dcrnn.py does with --hpo-study for the graph models.

Usage
-----
    python train_cl_tft_bc.py -c configs/tft_bc/config_wind_tft_base_fold1.yaml \
        --hpo-study cl_m-tft-bc_out-48_freq-1h_wind_tft_base --gpu 2

    python train_cl_tft_bc.py -c configs/tft_bc/config_wind_tft_base_test.yaml \
        --hpo-study cl_m-tft-bc_out-48_freq-1h_wind_tft_base --gpu 2 --test-mode
"""

import os
import json
import pickle
import argparse
import logging

import torch
import optuna

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import preprocessing, tools, hpo, data_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Final training with best HPO params (tft_bc)")
    parser.add_argument('-m', '--model', type=str, default='tft')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-s', '--suffix', type=str, default='')
    parser.add_argument('--hpo-study', type=str, required=True,
                         help='Exact Optuna study name to load best_trial params from')
    parser.add_argument('--test-mode', action='store_true', default=False,
                         help='Merge val_files into the training pool (final held-out test run)')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--max-cache-gb', type=float, default=150.0)
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory for preprocessed-data cache entries (defaults to '
                             'utils.data_cache.DEFAULT_CACHE_DIR; set on hosts without that mount)')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    suffix = f'_{args.suffix}' if args.suffix else ''
    if '.yaml' in args.config:
        args.config = args.config.split('.')[0]
    config_name = args.config.split('/')[-1] if '/' in args.config else args.config
    if config_name.startswith('config_'):
        config_name = config_name[7:]
    log_file = f'logs/train_cl_tft_bc_m-{args.model}_c-{config_name}{suffix}.log'

    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"FINAL TRAINING (tft_bc) - Model: {args.model}, Config: {args.config}, "
                f"HPO study: {args.hpo_study}, test-mode: {args.test_mode}")
    logger.info("=" * 80)

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    config = tools.load_config(f'{args.config}.yaml')
    config['model']['verbose'] = 0
    freq = config['data']['freq']
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']
    config['model']['fl'] = False
    config['model']['name'] = args.model

    if args.test_mode:
        if not config['data'].get('test_files'):
            raise ValueError("--test-mode requires 'test_files' to be set in the config.")
        logger.info("--test-mode: merging val_files into training set (files += val_files).")
        config['data']['files'] = list(config['data'].get('files', [])) + list(config['data'].get('val_files', []))
        config['data']['val_files'] = []  # skip _replace_val_with_val_files, fall back to plain val_split

    storage_url = os.environ.get('OPTUNA_STORAGE')
    if not storage_url:
        raise RuntimeError("OPTUNA_STORAGE env var must be set to load the HPO study.")
    study = optuna.load_study(study_name=args.hpo_study, storage=storage_url)
    best = study.best_trial
    logger.info(f"Loaded study '{args.hpo_study}': best_trial={best.number}, "
                f"best_value={best.value:.6f}, params={json.dumps(best.params)}")

    for key in ('next_n_grid_points', 'next_n_grid_ecmwf', 'next_n_stations'):
        if key in best.params:
            config['params'][key] = best.params[key]
    logger.info(
        f"Preprocessing params from best trial: "
        f"next_n_grid_points={config['params']['next_n_grid_points']}, "
        f"next_n_grid_ecmwf={config['params']['next_n_grid_ecmwf']}, "
        f"next_n_stations={config['params']['next_n_stations']}"
    )

    hyperparameters = hpo.get_hyperparameters(config=config, hpo=False, study=study)
    hyperparameters['epochs'] = config['model']['epochs']
    logger.info(f"Hyperparameters: {json.dumps(hyperparameters, default=str)}")

    features = preprocessing.get_features(config=config)

    model_tag = f'train_tft_bc_m-{args.model}_c-{config_name}{suffix}'
    use_cache = not args.no_cache

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    logger.info(f"Using device: {device}")

    # cv_mode='spatial' (see utils/data_cache.py::create_or_load_preprocessed_data_spatial):
    # a fold-specific config (config_wind_tft_sp_*_foldN.yaml) already carries that
    # fold's train-role/target-role station split verbatim in data.files/data.val_files
    # (populated from configs/spatial_folds.yaml when the config was written) — so
    # unlike hpo_tft_bc.py's HPO loop, this script does NOT need to import
    # geostatistics.spatial_cv or re-derive the fold at all, it just has to call the
    # matching data_cache function. This is deliberate: get_test_results_tft_bc.py must
    # recover the SAME cache_id (and therefore the same fitted scaler_x) this call
    # produces, and the surest way to guarantee that is for both scripts to do the
    # exact same thing with the config they were handed, with no divergent
    # re-derivation logic that could drift apart (the GNN-side review finding N1 this
    # task was asked not to repeat).
    cv_mode = str(config.get('hpo', {}).get('cv_mode', 'temporal')).lower()
    if cv_mode == 'spatial':
        if args.test_mode:
            raise ValueError(
                "--test-mode is not supported with cv_mode='spatial': it merges "
                "val_files into files for a final full-network retrain, which has no "
                "defined meaning for a fold-specific spatial split. Use a temporal "
                "config for the final full-network model."
            )
        logger.info(
            f"cv_mode='spatial': using data.files/data.val_files from {args.config}.yaml "
            f"as-is ({len(config['data'].get('files', []))} train / "
            f"{len(config['data'].get('val_files', []))} target stations), fixed time "
            f"window (train < {config['data'].get('val_start')}, val >= that)."
        )
        lazy_fold_loader, cache_id = data_cache.create_or_load_preprocessed_data_spatial(
            cache_dir=args.cache_dir,
            config=config,
            features=features,
            model_name=model_tag,
            force_reprocess=False,
            use_cache=use_cache,
        )
    else:
        lazy_fold_loader, cache_id = data_cache.create_or_load_preprocessed_data(
            cache_dir=args.cache_dir,
            config=config,
            features=features,
            model_name=model_tag,
            force_reprocess=False,
            use_cache=use_cache,
        )
    logger.info(f"Using data with cache ID: {cache_id}, {len(lazy_fold_loader)} fold(s)")
    if len(lazy_fold_loader) != 1:
        raise ValueError(
            f"Expected exactly 1 fold for final training (set hpo.kfolds: 1 in the config "
            f"for cv_mode='temporal'; cv_mode='spatial' always produces exactly 1 fold "
            f"per fold-specific config), got {len(lazy_fold_loader)}."
        )

    train, val = lazy_fold_loader[0]
    X_train, y_train = train
    X_val, y_val = val if val and val[0] is not None else (None, None)
    logger.info(f"Train samples: {len(y_train)}, Val samples: {0 if y_val is None else len(y_val)}")

    history, model = tools.training_pipeline(
        train=train, val=val, hyperparameters=hyperparameters, config=config, device=device,
    )

    final_val_rmse = history['val_rmse'][-1] if history.get('val_rmse') else None
    logger.info(f"Training finished. Final val_rmse (scaled units): {final_val_rmse}")

    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_path = os.path.join('models', f'{model_tag}.pt')
    torch.save(raw_model.state_dict(), model_path)
    logger.info(f"Saved model state_dict to {model_path}")

    metadata = {
        'model_tag': model_tag,
        'config_path': f'{args.config}.yaml',
        'hpo_study': args.hpo_study,
        'best_trial_number': best.number,
        'best_trial_value': best.value,
        'best_trial_params': best.params,
        'hyperparameters': hyperparameters,
        'feature_dim': config['model']['feature_dim'],
        'test_mode': args.test_mode,
        'final_val_rmse_scaled': final_val_rmse,
    }
    meta_path = os.path.join('models', f'{model_tag}_meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved metadata to {meta_path}")

    results_dir = os.path.join('results', 'tft_bc')
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, f'{model_tag}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    logger.info(f"Saved training history to {history_path}")

    logger.info("Done.")


if __name__ == '__main__':
    main()
