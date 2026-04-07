#!/usr/bin/env python3
"""
Training Script
"""

import os
import math
import copy
import json
import warnings
import argparse

warnings.filterwarnings('ignore', message=".*GetPrototype.*")
warnings.filterwarnings('ignore', message='X does not have valid feature names', category=UserWarning)
import pandas as pd
import numpy as np
import logging
import pickle
import gc
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.multiprocessing

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import preprocessing, tools, hpo, eval


def main() -> None:
    logger = logging.getLogger(__name__)

    # Argument parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument('-m', '--model', type=str, default='tft', help='Select Model (default: tft)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    parser.add_argument('-s', '--suffix', type=str, default='', help='Define suffix for study name (default: empty)')
    parser.add_argument('--save_model', action='store_true', default=False, help='Save trained model to models directory (default: False)')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    suffix = ''
    if args.suffix:
        suffix = f'_{args.suffix}'
    if '.yaml' in args.config:
        args.config = args.config.split('.')[0]
    if '/' in args.config:
        config_name = args.config.split('/')[-1]
    else:
        config_name = args.config
    log_file = f'logs/train_cl_m-{args.model}_{("_").join(config_name.split("_")[1:])}{suffix}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # GPU initialization
    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("Using CPU")

    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    config = tools.load_config(f'{args.config}.yaml')
    freq = config['data']['freq']
    params = config['params']
    config = tools.handle_freq(config=config)
    output_dim = config['model']['output_dim']
    lookback = config['model']['lookback']
    horizon = config['model']['horizon']

    logging.info(f'Model: PyTorch {args.model.upper()}, Output dim: {output_dim}, Frequency: {freq}, '
                f'Lookback: {lookback}, Horizon: {horizon}, Step size: {config["model"]["step_size"]}')

    # Set model name for preprocessing pipeline
    config['model']['name'] = args.model

    # ── Chronos-2 setup ──────────────────────────────────────────────────────
    is_chronos = args.model == 'chronos'
    chronos_pipeline = None
    if is_chronos:
        from chronos import BaseChronosPipeline
        chronos_cfg = config['model']['chronos']
        repo_id = chronos_cfg.get('repo_id', 'amazon/chronos-2')
        device_map = chronos_cfg.get('device_map', 'cuda')
        logging.info(f"Loading Chronos-2 from {repo_id} …")
        chronos_pipeline = BaseChronosPipeline.from_pretrained(repo_id, device_map=device_map)
        logging.info("Chronos-2 pipeline loaded.")
    # ─────────────────────────────────────────────────────────────────────────

    # Extract retrain settings with defaults
    retrain_interval = config['data'].get('retrain_interval', 1)
    eval_interval = config['data'].get('eval_interval', retrain_interval)

    # Validate eval_interval configuration
    if retrain_interval > 1 and eval_interval != retrain_interval:
        logging.warning(
            f"Invalid configuration: retrain_interval={retrain_interval} and eval_interval={eval_interval}. "
            f"eval_interval can only differ from retrain_interval when retrain_interval=1. "
            f"Setting eval_interval to {retrain_interval}."
        )
        eval_interval = retrain_interval
    elif retrain_interval == 1 and eval_interval < 1:
        logging.warning(f"Invalid eval_interval={eval_interval}. Setting to 1.")
        eval_interval = 1

    logging.info(f"Retrain interval: {retrain_interval}, Eval interval: {eval_interval}")

    # Get features
    data_dir = config['data']['path']
    features = preprocessing.get_features(config=config)

    # Setup paths
    base_dir = os.path.basename(data_dir)
    target_dir = os.path.join('results', base_dir)
    os.makedirs(target_dir, exist_ok=True)
    study_name_suffix = '_'.join(config_name.split('_')[1:])
    if args.suffix:
        study_name_suffix += f'_{args.suffix}'
    study_name = f'cl_m-{args.model}_out-{output_dim}_freq-{freq}_{study_name_suffix}' # .yaml delete later when renamed the hpo studies

    # Load data - REUSE EXISTING PREPROCESSING
    dfs = preprocessing.get_data(
        data_dir=data_dir,
        config=config,
        freq=freq,
        features=features
    )

    # Optional separate validation/test stations (val_files).
    # When present: training sequences come from dfs (up to test_start) and
    # test/validation sequences come from val_dfs (test_start → test_end),
    # scaled with the scaler fitted on dfs training data.
    val_dfs = None
    if config['data'].get('val_files'):
        logging.info("val_files found — loading separate validation/test stations.")
        val_dfs = preprocessing.get_data(
            data_dir=data_dir,
            config=config,
            freq=freq,
            features=features,
            files_key='val_files'
        )
        logging.info(f"Loaded {len(val_dfs)} val stations, {len(dfs)} training stations.")

    logging.info(f'Config: {json.dumps(params, indent=2)}')

    # Calculate test periods for evaluation
    test_start = pd.Timestamp(config['data']['test_start'], tz='UTC')
    test_end = pd.Timestamp(config['data']['test_end'], tz='UTC')

    # If test_end is at 00:00 (only date specified), extend to include the whole day (23:00)
    if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
        test_end = test_end.replace(hour=23, minute=0, second=0)

    if eval_interval > 1:
        # Custom date-based splitting logic for MultiIndex data
        # 1. Extract unique dates from the data
        all_starttimes = set()
        for df in dfs.values():
            if isinstance(df.index, pd.MultiIndex):
                # Extract unique dates (normalized to 00:00:00)
                dates = pd.to_datetime(df.index.get_level_values('starttime')).normalize().unique()
                all_starttimes.update(dates)
            else:
                # Fallback for non-MultiIndex (though this path is for Icon-D2 mainly)
                dates = pd.to_datetime(df.index).normalize().unique()
                all_starttimes.update(dates)

        # 2. Filter dates within test range
        sorted_dates = sorted(list(all_starttimes))
        # Ensure test_start/end are timezone-aware if dates are
        if sorted_dates and sorted_dates[0].tzinfo:
             ts_test_start = test_start.tz_convert(sorted_dates[0].tzinfo)
             ts_test_end = test_end.tz_convert(sorted_dates[0].tzinfo)
        else:
             ts_test_start = test_start.tz_localize(None)
             ts_test_end = test_end.tz_localize(None)

        test_dates = [d for d in sorted_dates if d >= ts_test_start and d <= ts_test_end]

        if not test_dates:
            logging.warning("No data found within test range! Fallback to standard calculation.")
            test_periods = tools.calculate_retrain_periods(test_start, test_end, eval_interval, freq)
        else:
            # 3. Split into chunks
            chunks = np.array_split(test_dates, eval_interval)
            test_periods = []
            for chunk in chunks:
                if len(chunk) > 0:
                    p_start = chunk[0]
                    p_end = chunk[-1]
                    p_end = p_end + pd.Timedelta(hours=23, minutes=59, seconds=59)
                    test_periods.append((p_start, p_end))

        if retrain_interval > 1:
            logging.info(f"Test periods (Date-based, with retraining): {len(test_periods)}")
        else:
            logging.info(f"Test periods (Date-based, evaluation only): {len(test_periods)}")
        for i, (start, end) in enumerate(test_periods):
            logging.info(f"  Period {i+1}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    else:
        test_periods = [(test_start, test_end)]
        logging.info(f"Single test period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")

    # Storage for multiple retraining results
    all_evaluations = []
    all_histories = []

    # --- GLOBAL SCALING (same as TensorFlow version) ---
    global_scaler_x = StandardScaler()
    global_scaler_y = StandardScaler()

    # Get features for lag feature creation
    features = preprocessing.get_features(config=config)

    # Get hyperparameters using hpo.get_hyperparameters
    study = None
    if config['model']['lookup_hpo']:
        logging.info(f"Looking up hyperparameters for study: {study_name}")
        study = hpo.load_study(config['hpo']['studies_path'], study_name)
    hyperparameters = hpo.get_hyperparameters(config=config,
                                                study=study)
    logging.info(f"Hyperparameters: {json.dumps(hyperparameters, indent=2)}")

    # Determine if we need to train in this loop or just evaluate
    # Train model once when retrain_interval=1, or for each period when retrain_interval>1
    train_once_eval_multiple = (retrain_interval == 1 and eval_interval > 1)
    trained_model = None  # Store model when training once

    # Loop over test periods (training and/or evaluation)
    for period_idx, (current_test_start, current_test_end) in enumerate(test_periods):
        logging.info(f"\n{'='*80}")
        if train_once_eval_multiple:
            if period_idx == 0:
                logging.info(f"Training cycle 1/1 (will evaluate on {len(test_periods)} periods)")
            else:
                logging.info(f"Evaluation cycle {period_idx + 1}/{len(test_periods)}")
        else:
            logging.info(f"Training cycle {period_idx + 1}/{len(test_periods)}")
        logging.info(f"Test period: {current_test_start.strftime('%Y-%m-%d')} to {current_test_end.strftime('%Y-%m-%d')}")
        logging.info(f"{'='*80}\n")

        # Create period-specific config (deep copy to avoid modifying original)
        period_config = copy.deepcopy(config)

        # For train_once_eval_multiple: use original test_start for training,
        # but period-specific test_start/end for evaluation
        if train_once_eval_multiple:
            # Keep original test_start for training, use period boundaries for evaluation
            training_test_start = test_start.strftime('%Y-%m-%d %H:%M')
            eval_test_start = current_test_start.strftime('%Y-%m-%d %H:%M')
            eval_test_end = current_test_end.strftime('%Y-%m-%d %H:%M')
        else:
            # For retraining: use period-specific boundaries for both training and evaluation
            training_test_start = current_test_start.strftime('%Y-%m-%d %H:%M')
            eval_test_start = current_test_start.strftime('%Y-%m-%d %H:%M')
            eval_test_end = current_test_end.strftime('%Y-%m-%d %H:%M')

        # Set test boundaries
        period_config['data']['test_start'] = training_test_start
        period_config['data']['test_end'] = eval_test_end

        # Determine if we need to train in this iteration
        should_train = (period_idx == 0) if train_once_eval_multiple else True

        # Fit global scaler when training
        if should_train:
            logging.debug(f"Fitting global scaler for period {period_idx + 1}...")

            # Reset scalers for each period when retraining (fresh start each time)
            global_scaler_x = StandardScaler()
            global_scaler_y = StandardScaler()
            target_col = period_config['data']['target_col']
            fit_scaler_y = target_col != 'power'  # power is pre-normalised to [0,1]

            for key, df in tqdm(dfs.items(), desc="Fitting Global Scaler"):
                df_temp = df.copy()

                # For non-TFT models, create lag features before fitting scaler
                if period_config['model']['name'] not in ['tft', 'tcn-tft', 'stemgnn', 'chronos']:
                    # Create lag features for observed columns (same as in pipeline)
                    for col in features['observed']:
                        all_observed_cols = [new_col for new_col in df_temp.columns if new_col == col or new_col.startswith(col + '_lag_')]
                        for new_col in all_observed_cols:
                            df_temp = preprocessing.lag_features(
                                data=df_temp,
                                lookback=period_config['model']['lookback'],
                                horizon=period_config['model']['horizon'],
                                lag_in_col=period_config['data']['lag_in_col'],
                                target_col=new_col
                            )
                            # Drop the original column if it's not the target and not known
                            if new_col != target_col and new_col not in features['known']:
                                df_temp.drop(new_col, axis=1, inplace=True, errors='ignore')

                t_0 = 0 if period_config['eval']['eval_on_all_test_data'] else period_config['eval']['t_0']
                df_train, _ = preprocessing.split_data(
                    data=df_temp,
                    train_frac=period_config['data']['train_frac'],
                    test_start=pd.Timestamp(period_config['data']['test_start'], tz='UTC'),
                    test_end=pd.Timestamp(period_config['data']['test_end'], tz='UTC'),
                    t_0=t_0
                )

                # Fit X scaler (exclude target)
                df_train_x = df_train.drop(columns=[target_col], errors='ignore')
                global_scaler_x.partial_fit(df_train_x.values)

                # Fit Y scaler on target column
                if fit_scaler_y and target_col in df_train.columns:
                    global_scaler_y.partial_fit(df_train[[target_col]].values)

                del df_temp, df_train, df_train_x
                gc.collect()

            logging.debug("Global scalers fitted.")

        # Prepare data for training and/or evaluation
        if should_train:
            # Use memory-efficient data processing (REUSE existing functions)
            logging.debug("Starting memory-efficient data processing...")
            scaler_y_arg = global_scaler_y if fit_scaler_y else None

            if val_dfs is not None:
                # Separate sources: train sequences from dfs, test sequences from val_dfs.
                # Scalers were fitted on dfs training data only; val_dfs is scaled with
                # the same pre-fitted scalers (no re-fitting on val data).
                train_generator = tools.create_data_generator(
                    dfs, period_config, features,
                    scaler_x=global_scaler_x, scaler_y=scaler_y_arg
                )
                val_generator = tools.create_data_generator(
                    val_dfs, period_config, features,
                    scaler_x=global_scaler_x, scaler_y=scaler_y_arg
                )
                X_train, y_train, _, _, _, _ = tools.combine_datasets_efficiently(train_generator)
                _, _, X_test, y_test, test_data, _ = tools.combine_datasets_efficiently(val_generator)
            else:
                data_generator = tools.create_data_generator(
                    dfs, period_config, features,
                    scaler_x=global_scaler_x, scaler_y=scaler_y_arg
                )
                X_train, y_train, X_test, y_test, test_data, _ = tools.combine_datasets_efficiently(data_generator)

            # Note: Don't delete dfs yet - needed for evaluation pipeline later
            gc.collect()
            logging.debug("Data processing completed.")

            # Log shapes
            if isinstance(X_train, dict):
                if 'known' in X_train:
                    logging.info(f'Data shape: X_train known: {X_train["known"].shape}, X_test known: {X_test["known"].shape}')
                if 'static' in X_train:
                    logging.info(f'Data shape: X_train static: {X_train["static"].shape}, X_test static: {X_test["static"].shape}')
                logging.info(f'Data shape: X_train observed: {X_train["observed"].shape}, X_test observed: {X_test["observed"].shape}')
                logging.info(f'Data shape: y_train: {y_train.shape}, y_test: {y_test.shape}')
            else:
                logging.info(f'Data shape: X_train: {X_train.shape}, X_test: {X_test.shape}')
                logging.info(f'Data shape: y_train: {y_train.shape}, y_test: {y_test.shape}')

            # Device selection
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f"Using device: {device}")

            if is_chronos:
                # --- FINE-TUNING CHRONOS-2 ---
                chronos_cfg = period_config['model']['chronos']
                lr = hyperparameters.get('learning_rate', hyperparameters.get('lr', period_config['model']['lr']))
                batch_size = hyperparameters.get('batch_size', period_config['model'].get('batch_size', 256))
                finetune_mode = chronos_cfg.get('fine_tune_mode', 'full')
                lora_config = chronos_cfg.get('lora_config', None)
                # num_steps computed after inputs are built (needs n_sequences for steps_per_epoch)

                # Build fit inputs (full training set)
                known_cols_fit = period_config['params'].get('known_features', [])
                if isinstance(X_train, dict):
                    inputs_fit = tools.build_chronos_fit_inputs(X_train, y_train, known_cols_fit)
                else:
                    inputs_fit = [X_train[i].flatten() for i in range(X_train.shape[0])]

                n_covariates_fit = len(known_cols_fit)
                steps_per_epoch = math.ceil(len(inputs_fit) / batch_size)
                epochs = hyperparameters.get('epochs', period_config['model']['epochs'])
                num_steps = chronos_cfg.get('num_steps') or epochs * steps_per_epoch
                logging.info(
                    f"Fine-tuning Chronos-2 | mode={finetune_mode}, epochs={epochs}, "
                    f"steps={num_steps}, steps_per_epoch={steps_per_epoch}, "
                    f"lr={lr}, batch_size={batch_size}, "
                    f"n_train={len(inputs_fit)}, n_val={y_test.shape[0]}, "
                    f"n_known_covariates={n_covariates_fit}"
                )

                # Per-epoch metrics callback — validated on test set
                es_cfg = period_config['model'].get('early_stopping', {})
                chronos_cb = tools.ChronosEarlyStoppingCallback(
                    pipeline=chronos_pipeline,
                    X_train=X_train, y_train=y_train,
                    X_val=X_test, y_val=y_test,
                    config=period_config,
                    known_cols=known_cols_fit,
                    early_stopping_cfg=es_cfg,
                    steps_per_epoch=steps_per_epoch,
                )

                has_observed_features = X_train['observed'].shape[2] > 0
                chronos_pipeline = chronos_pipeline.fit(
                    inputs=inputs_fit,
                    prediction_length=period_config['model']['horizon'],
                    finetune_mode=finetune_mode,
                    lora_config=lora_config,
                    context_length=period_config['model']['lookback'],
                    min_past=period_config['model']['horizon'] if has_observed_features else 0,
                    learning_rate=lr,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    output_dir=chronos_cfg.get('output_dir', 'models/'),
                    finetuned_ckpt_name=chronos_cfg.get('finetuned_ckpt_name', 'finetuned-ckpt'),
                    logging_steps=steps_per_epoch,
                    callbacks=[chronos_cb.callback],
                    disable_tqdm=True,
                    report_to='none',
                    remove_printer_callback=True,
                )
                history = chronos_cb.history
                model = chronos_pipeline
                logging.info("Chronos-2 fine-tuning complete.")
            else:
                # --- TRAINING WITH PYTORCH ---
                history, model = tools.training_pipeline(
                    train=(X_train, y_train),
                    val=(X_test, y_test),
                    hyperparameters=hyperparameters,
                    config=period_config,
                    device=device
                )

            # Store model if training once
            if train_once_eval_multiple:
                trained_model = model
                logging.info(f"Model trained and will be reused for all {len(test_periods)} evaluation periods.")
                # For train_once_eval_multiple in the first period, we need to prepare eval data separately
                # Create a separate config for evaluation with period-specific boundaries
                eval_period_config = config.copy()
                eval_period_config['data']['test_start'] = eval_test_start
                eval_period_config['data']['test_end'] = eval_test_end
                # Regenerate test data with evaluation boundaries
                logging.debug("Preparing evaluation data for first period...")
                eval_data_generator = tools.create_data_generator(dfs, eval_period_config, features, scaler_x=global_scaler_x,
                                                                   scaler_y=global_scaler_y if fit_scaler_y else None)
                _, _, _, _, test_data, _ = tools.combine_datasets_efficiently(eval_data_generator)
                gc.collect()

            logging.info(f"Training completed. Final metrics:")
            if 'train_loss' in history:
                logging.info(f"  Train - MSE: {history['train_loss'][-1]:.6f}, RMSE: {history['train_rmse'][-1]:.4f}, R²: {history['train_r2'][-1]:.4f}")
                if 'val_loss' in history:
                    logging.info(f"  Val   - MSE: {history['val_loss'][-1]:.6f}, RMSE: {history['val_rmse'][-1]:.4f}, R²: {history['val_r2'][-1]:.4f}")
            else:
                logging.info("  (no epoch-level metrics available for this model type)")
        else:
            # Evaluation only: prepare test data for this period with period-specific boundaries
            logging.info("Evaluation only (using previously trained model)...")
            eval_period_config = config.copy()
            eval_period_config['data']['test_start'] = eval_test_start
            eval_period_config['data']['test_end'] = eval_test_end
            data_generator = tools.create_data_generator(dfs, eval_period_config, features, scaler_x=global_scaler_x,
                                                           scaler_y=global_scaler_y if fit_scaler_y else None)
            _, _, _, _, test_data, _ = tools.combine_datasets_efficiently(data_generator)
            model = trained_model
            history = None  # No training history for evaluation-only periods
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            gc.collect()

        # --- GENERATE PREDICTIONS AND EVALUATE PER PARK ---
        logging.info('Start evaluation pipeline...')

        eval_results = []
        for park_key, (X_test_park, y_test_park, index_test_park, scaler_y_park) in test_data.items():
            logging.debug(f'Evaluating park: {park_key}')

            park_df = dfs.get(park_key) or (val_dfs.get(park_key) if val_dfs else None)
            if park_df is None:
                logging.warning(f"No raw data found for park {park_key}, skipping evaluation.")
                continue

            if train_once_eval_multiple:
                test_start_str = eval_test_start
            else:
                test_start_str = period_config['data']['test_start']
            t_0 = 0 if period_config['eval']['eval_on_all_test_data'] else period_config['eval']['t_0']

            if is_chronos:
                known_cols = period_config['params'].get('known_features', [])
                y_true_park, y_pred_park = tools.get_y_chronos2(
                    X_test=X_test_park,
                    y_test=y_test_park,
                    chronos_pipeline=model,
                    config=period_config,
                    known_cols=known_cols,
                    scaler_y=scaler_y_park,
                )
                df_pred_park = tools.y_to_df(
                    y=y_pred_park, output_dim=output_dim, horizon=horizon,
                    index=index_test_park,
                    t_0=None if period_config['eval']['eval_on_all_test_data'] else t_0,
                )
                df_true_park = tools.y_to_df(
                    y=y_true_park, output_dim=output_dim, horizon=horizon,
                    index=index_test_park,
                    t_0=None if period_config['eval']['eval_on_all_test_data'] else t_0,
                )
                park_eval = eval.evaluate_models(
                    pred=df_pred_park,
                    true=df_true_park,
                    persistence={},
                    main_model_name='CHRONOS',
                    drop_except_main=True,
                )
            else:
                park_eval = eval.evaluation_pipeline(
                    data=park_df,
                    model=model,
                    model_name=f'{args.model.upper()}',
                    X_test=X_test_park,
                    y_test=y_test_park,
                    scaler_y=scaler_y_park,
                    output_dim=output_dim,
                    horizon=horizon,
                    index_test=index_test_park,
                    test_start=test_start_str,
                    t_0=t_0,
                    park_id=park_key,
                    synth_dir=None,
                    get_physical_persistence=False,
                    target_col=period_config['data']['target_col'],
                    evaluate_on_all_test_data=period_config['eval']['eval_on_all_test_data'],
                    device=device
                )
            park_eval['key'] = park_key
            eval_results.append(park_eval)

        if eval_results:
            period_evaluation = pd.concat(eval_results, ignore_index=False)
        else:
            period_evaluation = pd.DataFrame()

        logging.info(f"Per-park evaluation completed: {len(eval_results)} parks evaluated")

        if not period_evaluation.empty:
            period_evaluation.loc['mean'] = period_evaluation.mean(numeric_only=True)
            period_evaluation.loc['std'] = period_evaluation.std(numeric_only=True)
            logging.info(f"\n{period_evaluation.to_string()}")

        period_evaluation['output_dim'] = output_dim
        period_evaluation['freq'] = freq
        if period_config['eval']['eval_on_all_test_data']:
            period_evaluation['t_0'] = None
        else:
            period_evaluation['t_0'] = period_config['eval']['t_0']

        all_evaluations.append(period_evaluation)
        if history is not None:
            all_histories.append(history)

        if train_once_eval_multiple:
            if period_idx == 0:
                logging.info(f"Completed training cycle 1/1")
            else:
                logging.info(f"Completed evaluation cycle {period_idx + 1}/{len(test_periods)}")
        else:
            logging.info(f"Completed training cycle {period_idx + 1}/{len(test_periods)}")

        # --- PER-PERIOD SAVING (only for multiple retraining cycles) ---
        if len(test_periods) > 1 and should_train and args.save_model:
            period_n = period_idx + 1
            os.makedirs(os.path.join('models', 'scaler'), exist_ok=True)
            period_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            period_pkl = os.path.join('results', base_dir, f'{study_name}_{period_n}_{period_timestamp}.pkl')
            with open(period_pkl, 'wb') as f:
                pickle.dump({
                    'hyperparameters': hyperparameters,
                    'config': config,
                    'history': history,
                    'evaluation': period_evaluation,
                    'test_dates': [(current_test_start, current_test_end)],
                }, f)
            logging.info(f"Period {period_n} results saved to: {period_pkl}")
            if not is_chronos:
                period_model_path = os.path.join('models', f'{study_name}_{period_n}.pt')
                torch.save(model.state_dict(), period_model_path)
                logging.info(f"Period {period_n} model saved to: {period_model_path}")
                period_scaler_path = os.path.join('models', 'scaler', f'scaler_{study_name}_{period_n}.pkl')
                with open(period_scaler_path, 'wb') as f:
                    pickle.dump(global_scaler_x, f)
                if config['data']['target_col'] != 'power':
                    with open(period_scaler_path.replace('scaler_', 'scaler_y_'), 'wb') as f:
                        pickle.dump(global_scaler_y, f)
                logging.info(f"Period {period_n} scaler saved to: {period_scaler_path}")

    # Save final model if requested
    if args.save_model:
        os.makedirs(os.path.join('models', 'scaler'), exist_ok=True)

        if is_chronos:
            model_path = os.path.join('models', f'chronos_{study_name}')
            chronos_pipeline.save_pretrained(model_path)
            logging.info(f"Chronos-2 model saved to: {model_path}/")
            scaler_path = os.path.join('models', 'scaler', f'scaler_chronos_{study_name}.pkl')
        else:
            model_path = os.path.join('models', f'{study_name}.pt')
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model saved to: {model_path}")
            scaler_path = os.path.join('models', 'scaler', f'scaler_{study_name}.pkl')

        with open(scaler_path, 'wb') as f:
            pickle.dump(global_scaler_x, f)
        logging.info(f"Scaler saved to: {scaler_path}")

        if config['data']['target_col'] != 'power':
            scaler_y_path = scaler_path.replace('scaler_', 'scaler_y_')
            with open(scaler_y_path, 'wb') as f:
                pickle.dump(global_scaler_y, f)
            logging.info(f"Scaler Y saved to: {scaler_y_path}")

    # --- AGGREGATE RESULTS ACROSS ALL RETRAINING PERIODS ---
    logging.info(f"\n{'='*80}")

    if len(all_evaluations) > 1:
        # Calculate mean evaluation across all retrainings/evaluations
        # Average DataFrames row-wise to maintain structure

        # Separate numeric and non-numeric columns from first DataFrame
        first_eval = all_evaluations[0]
        numeric_cols = first_eval.select_dtypes(include=[np.number]).columns
        non_numeric_cols = first_eval.select_dtypes(exclude=[np.number]).columns

        # Create empty DataFrame with same structure as input
        evaluation = first_eval.copy()

        # Average numeric columns row-wise across all evaluations
        for col in numeric_cols:
            # Extract this column from all evaluations and compute mean
            col_data = pd.concat([eval_df[col] for eval_df in all_evaluations], axis=1)
            evaluation[col] = col_data.mean(axis=1)

        # For non-numeric columns, keep values from first evaluation
        # (they should be the same across all periods)

        logging.info(f"Aggregated evaluation:")
        logging.info(evaluation)

        # Build results with individual results
        results = {
            'hyperparameters': hyperparameters,
            'config': config,
            'history': all_histories if all_histories else None,  # List of dictionaries, or None if eval-only
            'evaluation': evaluation,
            'individual_evaluations': all_evaluations,
            'test_dates': test_periods
        }
    else:
        # Single training run (backward compatible)
        evaluation = all_evaluations[0]
        history_result = all_histories[0] if all_histories else None
        results = {
            'hyperparameters': hyperparameters,
            'config': config,
            'history': history_result,  # Single dictionary or None
            'evaluation': evaluation,
            'test_dates': test_periods
        }

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path_to_pkl = os.path.join('results', base_dir, f'{study_name}_{timestamp}.pkl')

    with open(path_to_pkl, 'wb') as f:
        pickle.dump(results, f)

    logging.info(f"\nTraining completed! Results saved to: {path_to_pkl}")

    # Log final metrics (handle both single dict and list of dicts, or None)
    metrics_data = []
    histories = results['history']

    if not histories or (isinstance(histories, list) and len(histories) == 0):
        # No training history (eval-only mode)
        logging.info("\nNo training history (evaluation-only mode with single trained model)")
    else:
        if not isinstance(histories, list):
            histories = [histories]

        for i, hist in enumerate(histories):
            row = {
                'Cycle': i + 1 if len(histories) > 1 else 'Final',
                'train_loss': hist['train_loss'][-1],
                'val_loss': hist['val_loss'][-1] if 'val_loss' in hist else None,
                'train_rmse': hist['train_rmse'][-1],
                'val_rmse': hist['val_rmse'][-1] if 'val_rmse' in hist else None,
                'train_r2': hist['train_r2'][-1],
                'val_r2': hist['val_r2'][-1] if 'val_r2' in hist else None,
            }
            metrics_data.append(row)

        metrics_df = pd.DataFrame(metrics_data)
        # Define the desired column order for logging
        ordered_cols = ['Cycle', 'train_loss', 'val_loss', 'train_rmse', 'val_rmse', 'train_r2', 'val_r2']
        # Filter for columns that actually exist in the DataFrame
        final_cols = [col for col in ordered_cols if col in metrics_df.columns]
        metrics_df = metrics_df[final_cols]

        logging.info("\nTraining and Validation Metrics:")
        logging.info(metrics_df.to_string(index=False, float_format="%.6f", na_rep='N/A'))

    # Cleanup
    del dfs
    gc.collect()


if __name__ == '__main__':
    main()

