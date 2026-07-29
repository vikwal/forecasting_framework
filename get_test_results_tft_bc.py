#!/usr/bin/env python3
"""
get_test_results_tft_bc.py — Evaluate a trained tft_bc model (from train_cl_tft_bc.py)
on held-out test_files, in PHYSICAL units (inverse-transformed via each station's own
StandardScaler), analogous to geostatistics/get_test_results_dcrnn.py and
train_mtgnn.py::_metrics (RMSE_phys).

This is deliberately per-station: prepare_data_for_tft fits a *per-station* scaler_y
(no global scaler is injected in the tft_bc pipeline), so each test station's
predictions/targets must be inverse-transformed with that station's own scaler before
computing errors — never with a scaler from a different station or a pooled scaler.

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

from utils import preprocessing, tools, models


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
