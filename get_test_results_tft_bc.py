#!/usr/bin/env python3
"""
get_test_results_tft_bc.py — Evaluate a trained tft_bc model (from train_cl_tft_bc.py)
on held-out test_files, in PHYSICAL units, analogous to
geostatistics/get_test_results_dcrnn.py and train_mtgnn.py::_metrics (RMSE_phys).

The target ('wind_speed', scale_target=False) is never scaled, so scaler_y is always
None and tools.get_y() skips inverse-transforming y — no scaler_y handling needed here.

Feature scaling (scaler_x) is a different story: since the v3 preprocessing change
(utils/data_cache.py::_fit_global_scaler_x), training uses ONE StandardScaler fitted
across all training stations, injected into the pipeline via config['scaler_x'] — not a
scaler fitted per-station or per-call. Evaluation MUST reuse that exact fitted scaler,
or prepare_data_for_tft silently falls back to its "LOCAL SCALING STRATEGY" branch (a
fresh per-station scaler fit on the fly) and every feature — including the 13 static
features, which the local branch collapses to 0 — would be scaled differently than
during training, with no error or warning. This script recovers the training scaler by
recomputing the training run's cache_id (DataCache._get_config_hash, same inputs as
train_cl_tft_bc.py) and loading it from that run's cached metadata, where it was stored
as a side effect of caching config['scaler_x'] alongside the rest of the config.

Also computes R² and Skill (vs. persistence baseline = last actual measurement before
forecast start), identical definitions to geostatistics/homo_sampler.py::evaluate_homo_model,
and writes data/test_results/<name>.csv + data/raw_preds/<name>_raw.parquet in the same
schema as get_test_results_dcrnn.py / get_test_results_wavenet.py, so TFT results can be
loaded into a fold_evaluation.ipynb-style comparison notebook alongside the graph models.

Usage
-----
    python get_test_results_tft_bc.py -c configs/tft_bc/config_wind_tft_base_fold1.yaml \
        --hpo-study cl_m-tft-bc_out-48_freq-1h_wind_tft_base --model-tag train_tft_bc_m-tft_c-wind_tft_base_fold1 \
        --raw-out-name tft_wind_tft_base_fold1 --gpu 2
"""

import os
import copy
import json
import pickle
import argparse
import logging
import math

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import preprocessing, tools, models, data_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Test-set evaluation for tft_bc models (physical units)")
    parser.add_argument('-m', '--model', type=str, default='tft')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--hpo-study', type=str, required=True)
    parser.add_argument('--model-tag', type=str, required=True,
                         help='model_tag used by train_cl_tft_bc.py (models/<tag>.pt / <tag>_meta.pkl)')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--raw-out-name', default=None,
                         help="Stem for data/test_results/<name>.csv and data/raw_preds/<name>_raw.parquet "
                              "(e.g. 'tft_wind_tft_base_fold1' or 'tft_wind_tft_base_test_fold0'). "
                              "Defaults to model_tag if omitted.")
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory for preprocessed-data cache entries (defaults to '
                             'utils.data_cache.DEFAULT_CACHE_DIR; set on hosts without that mount)')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    if '.yaml' in args.config:
        args.config = args.config.split('.')[0]
    config_name = args.config.split('/')[-1] if '/' in args.config else args.config
    if config_name.startswith('config_'):
        config_name = config_name[7:]
    log_file = f'logs/eval_tft_bc_c-{config_name}.log'

    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"TEST EVALUATION (tft_bc) - Config: {args.config}, model_tag: {args.model_tag}")
    logger.info("=" * 80)

    config = tools.load_config(f'{args.config}.yaml')
    config['model']['verbose'] = 0
    config = tools.handle_freq(config=config)
    config['model']['fl'] = False
    config['model']['name'] = args.model

    if not config['data'].get('test_files'):
        raise ValueError("Config has no 'test_files' — nothing to evaluate on.")

    storage_url = os.environ.get('OPTUNA_STORAGE')
    if not storage_url:
        raise RuntimeError("OPTUNA_STORAGE env var must be set to load the HPO study.")
    study = optuna.load_study(study_name=args.hpo_study, storage=storage_url)
    best = study.best_trial
    for key in ('next_n_grid_points', 'next_n_grid_ecmwf', 'next_n_stations'):
        if key in best.params:
            config['params'][key] = best.params[key]
    logger.info(
        f"Preprocessing params from best trial: "
        f"next_n_grid_points={config['params']['next_n_grid_points']}, "
        f"next_n_grid_ecmwf={config['params']['next_n_grid_ecmwf']}, "
        f"next_n_stations={config['params']['next_n_stations']}"
    )

    meta_path = os.path.join('models', f'{args.model_tag}_meta.pkl')
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    hyperparameters = metadata['hyperparameters']
    config['model']['feature_dim'] = metadata['feature_dim']
    logger.info(f"Loaded metadata from {meta_path} (best_trial={metadata['best_trial_number']}, "
                f"best_value={metadata['best_trial_value']:.6f})")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    logger.info(f"Using device: {device}")

    model = models.get_model(config=config, hyperparameters=hyperparameters)
    model_path = os.path.join('models', f'{args.model_tag}.pt')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model weights from {model_path}")

    features = preprocessing.get_features(config=config)
    freq = config['data']['freq']

    # --- Recover the training run's global scaler_x (utils/data_cache.py::_fit_global_scaler_x) ---
    # Training injects config['scaler_x'] before calling preprocessing.pipeline() (see
    # data_cache.create_or_load_preprocessed_data); if we don't do the same here,
    # prepare_data_for_tft silently falls back to fitting a fresh per-station scaler
    # (LOCAL SCALING STRATEGY) — wrong feature scaling with no error. The fitted scaler
    # is recoverable from the training run's own cache metadata: reproduce the exact
    # cache_id train_cl_tft_bc.py would have computed (same config/features/model_name
    # inputs to DataCache._get_config_hash) and pull config['scaler_x'] back out of it.
    hash_config = copy.deepcopy(config)
    if metadata.get('test_mode'):
        # Mirrors train_cl_tft_bc.py --test-mode: files += val_files, val_files cleared,
        # before the training run's hash/cache lookup — must match here bit-for-bit or
        # we compute the wrong cache_id.
        hash_config['data']['files'] = (list(hash_config['data'].get('files', []))
                                         + list(hash_config['data'].get('val_files', [])))
        hash_config['data']['val_files'] = []

    # cv_mode='spatial': train_cl_tft_bc.py computed its cache_id from the config
    # exactly as given (data.files/data.val_files already ARE that fold's train-role/
    # target-role split, read straight from config_wind_tft_sp_*_foldN.yaml — see
    # create_or_load_preprocessed_data_spatial's docstring). No test_mode-style
    # files/val_files rewrite happened there, so none must happen here either — hashing
    # anything other than `config` unmodified would recompute a DIFFERENT cache_id than
    # training used and silently recover the wrong (or no) scaler, exactly the N1-style
    # mismatch this function's docstring above already guards against for temporal mode.
    cv_mode = str(config.get('hpo', {}).get('cv_mode', 'temporal')).lower()
    if cv_mode == 'spatial':
        if metadata.get('test_mode'):
            raise RuntimeError(
                "Model metadata has test_mode=True but cv_mode='spatial' — "
                "train_cl_tft_bc.py refuses that combination, so this should be "
                "unreachable. Refusing to guess the training cache_id."
            )
        hash_config = copy.deepcopy(config)

    cache = data_cache.DataCache(args.cache_dir or data_cache.DEFAULT_CACHE_DIR)
    train_cache_id = cache._get_config_hash(hash_config, features, model_name=args.model_tag)
    train_cache_paths = cache.get_cache_paths(train_cache_id)
    if not os.path.exists(train_cache_paths['metadata']):
        raise RuntimeError(
            f"Could not recover the training scaler_x: no cache metadata found at "
            f"{train_cache_paths['metadata']} for recomputed training cache_id "
            f"{train_cache_id}. Refusing to fall back to a freshly-fit per-station "
            f"scaler, which would silently mismatch the trained model's feature scaling. "
            f"Check that the config/hpo-study/model-tag match the original training run "
            f"exactly and that its cache entry hasn't been evicted."
        )
    with open(train_cache_paths['metadata'], 'rb') as f:
        train_cache_meta = pickle.load(f)
    scaler_x = train_cache_meta['config'].get('scaler_x')
    if scaler_x is None or not hasattr(scaler_x, 'mean_'):
        raise RuntimeError(
            f"Training cache metadata at {train_cache_paths['metadata']} (cache_id "
            f"{train_cache_id}) has no fitted scaler_x. Refusing to fall back to a "
            f"freshly-fit per-station scaler."
        )
    config['scaler_x'] = scaler_x
    logger.info(
        f"Recovered global scaler_x from training cache_id {train_cache_id} "
        f"({len(getattr(scaler_x, '_ff_feature_cols', []))} feature columns, "
        f"has_target_feature_scaler={hasattr(scaler_x, '_ff_target_feature_scaler')})"
    )

    # Neighbour pool at test time: every station in the experiment. The test stations'
    # own measurements are model INPUTS here (their future values are what gets scored),
    # and in deployment the full observation network is available — so unlike training
    # (files only) and validation (files + val_files), nothing has to be withheld.
    # Set before get_data: the neighbour merge happens during loading.
    config['data']['neighbor_pool'] = (list(config['data'].get('files', []))
                                       + list(config['data'].get('val_files', []))
                                       + list(config['data'].get('test_files', [])))
    logger.info(f"Neighbour pool for test stations: "
                f"{len(config['data']['neighbor_pool'])} stations (files + val_files + test_files)")

    test_dfs = preprocessing.get_data(
        data_dir=config['data']['path'],
        config=config,
        freq=freq,
        features=features,
        target_col=config['data']['target_col'],
        files_key='test_files',
    )
    logger.info(f"Loaded {len(test_dfs)} test stations from {config['data']['path']} "
                f"(test window {config['data']['test_start']} .. {config['data']['test_end']})")

    freq_delta = pd.Timedelta(freq)
    target_col = config['data']['target_col']

    per_station = []
    raw_records = []
    all_y_true, all_y_pred, all_y_nwp = [], [], []

    for station_id, df in test_dfs.items():
        prepared, _ = preprocessing.pipeline(
            data=df,
            config=config,
            known_cols=features['known'],
            observed_cols=features['observed'],
            static_cols=features['static'],
            target_col=target_col,
        )
        X_test, y_test = prepared.get('X_test'), prepared.get('y_test')
        scaler_y = prepared.get('scalers', {}).get('y')
        if X_test is None or y_test is None or len(y_test) == 0:
            logger.warning(f"Station {station_id}: no test samples in window, skipping.")
            continue

        y_true, y_pred = tools.get_y(X_test=X_test, y_test=y_test, model=model,
                                      scaler_y=scaler_y, device=device)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        r2 = float(r2_score(y_true.ravel(), y_pred.ravel()))

        nwp_raw = prepared.get('nwp_raw_test')
        rmse_nwp = None
        if nwp_raw is not None and len(nwp_raw) == len(y_true):
            rmse_nwp = float(np.sqrt(np.mean((nwp_raw - y_true) ** 2)))
            all_y_nwp.append(nwp_raw)
        else:
            nwp_raw = None

        # --- Persistence baseline: last actual measurement before forecast start (run_time - 1 step) ---
        # (Same definition as geostatistics/homo_sampler.py::evaluate_homo_model for DCRNN/MTGNN/WaveNet.)
        run_times = prepared.get('index_test')
        pers_ref = None
        skill = None
        if run_times is not None and len(run_times) == len(y_true):
            target_series = df[target_col]
            pers_vals = target_series.reindex(pd.DatetimeIndex(run_times) - freq_delta).to_numpy()
            pers_ref = np.repeat(pers_vals[:, None], y_true.shape[1], axis=1)
            valid_p = ~(np.isnan(pers_ref) | np.isnan(y_true))
            if valid_p.sum() >= 2:
                rmse_pers = float(math.sqrt(mean_squared_error(y_true[valid_p], pers_ref[valid_p])))
                skill = (1.0 - rmse / rmse_pers) if rmse_pers > 0 else None

            for i, run_ts in enumerate(run_times):
                for h in range(y_true.shape[1]):
                    raw_records.append({
                        'station_id': station_id,
                        'run_time':   run_ts,
                        'valid_time': run_ts + freq_delta * (h + 1),
                        'horizon':    h + 1,
                        'pred':       float(y_pred[i, h]),
                        'gt':         float(y_true[i, h]),
                        'nwp_ref':    float(nwp_raw[i, h]) if nwp_raw is not None else np.nan,
                        'pers_ref':   float(pers_ref[i, h]) if pers_ref is not None else np.nan,
                    })

        per_station.append({
            'station_id': station_id,
            'n_samples': int(len(y_true)),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'rmse_nwp': rmse_nwp,
            'skill_nwp': (1 - rmse / rmse_nwp) if rmse_nwp else None,
            'skill': skill,
        })
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        logger.info(f"Station {station_id}: n={len(y_true)}, RMSE={rmse:.4f} m/s, R2={r2:.4f}"
                    + (f", RMSE_NWP={rmse_nwp:.4f}, Skill_NWP={1 - rmse / rmse_nwp:.4f}" if rmse_nwp else "")
                    + (f", Skill={skill:.4f}" if skill is not None else ""))

    if not per_station:
        raise RuntimeError("No test stations produced samples — check test_start/test_end vs. data coverage.")

    y_true_all = np.concatenate(all_y_true, axis=0)
    y_pred_all = np.concatenate(all_y_pred, axis=0)
    pooled_rmse = float(np.sqrt(np.mean((y_pred_all - y_true_all) ** 2)))
    pooled_mae = float(np.mean(np.abs(y_pred_all - y_true_all)))
    mean_station_rmse = float(np.mean([s['rmse'] for s in per_station]))

    result = {
        'model_tag': args.model_tag,
        'config_path': f'{args.config}.yaml',
        'hpo_study': args.hpo_study,
        'test_start': str(config['data']['test_start']),
        'test_end': str(config['data']['test_end']),
        'n_stations': len(per_station),
        'pooled_rmse': pooled_rmse,
        'pooled_mae': pooled_mae,
        'mean_station_rmse': mean_station_rmse,
        'per_station': per_station,
    }
    if all_y_nwp:
        y_nwp_all = np.concatenate(all_y_nwp, axis=0)
        pooled_rmse_nwp = float(np.sqrt(np.mean((y_nwp_all - y_true_all) ** 2)))
        result['pooled_rmse_nwp'] = pooled_rmse_nwp
        result['pooled_skill_nwp'] = 1 - pooled_rmse / pooled_rmse_nwp

    logger.info(f"Pooled test RMSE (physical units, m/s): {pooled_rmse:.4f} "
                f"(mean-of-station: {mean_station_rmse:.4f}) over {len(per_station)} stations")
    if 'pooled_skill_nwp' in result:
        logger.info(f"Pooled Skill_NWP: {result['pooled_skill_nwp']:.4f}")

    results_dir = os.path.join('results', 'tft_bc')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f'{args.model_tag}_test_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    json_path = os.path.join(results_dir, f'{args.model_tag}_test_results.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved results to {out_path} / {json_path}")

    # --- CSV + raw-predictions parquet, same schema/location as get_test_results_dcrnn.py /
    # get_test_results_wavenet.py (data/test_results/*.csv, data/raw_preds/*_raw.parquet) so
    # fold_evaluation.ipynb-style notebooks can load TFT alongside the graph models. ---
    out_stem = args.raw_out_name if args.raw_out_name else args.model_tag

    test_results_dir = os.path.join('data', 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    station_df = pd.DataFrame(per_station).drop(columns=['rmse_nwp'])
    station_csv_path = os.path.join(test_results_dir, f'{out_stem}.csv')
    station_df.to_csv(station_csv_path, index=False)
    logger.info(f"Saved per-station CSV to {station_csv_path}")

    if raw_records:
        raw_preds_dir = os.path.join('data', 'raw_preds')
        os.makedirs(raw_preds_dir, exist_ok=True)
        raw_df = pd.DataFrame(raw_records)
        raw_path = os.path.join(raw_preds_dir, f'{out_stem}_raw.parquet')
        raw_df.to_parquet(raw_path, index=False)
        logger.info(f"Saved raw predictions to {raw_path}")


if __name__ == '__main__':
    main()
