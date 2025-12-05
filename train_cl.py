#!/usr/bin/env python3
"""
Training Script
"""

import os
import json
import argparse
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
    parser.add_argument('-i', '--index', type=str, default='', help='Define index')
    parser.add_argument('--save_model', action='store_true', default=False, help='Save trained model to models directory (default: False)')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    index = ''
    if args.index:
        index = f'_{args.index}'
    log_file = f'logs/train_cl_m-{args.model}_{("_").join(args.config.split("_")[1:])}{index}.log'
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

    # Load config
    if '.yaml' in args.config:
        args.config = args.config.split('.')[0]
    config = tools.load_config(f'configs/{args.config}.yaml')
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

    # Extract retrain settings with defaults
    retrain_enabled = config['data'].get('retrain', False)
    retrain_interval = config['data'].get('retrain_interval', 1) if retrain_enabled else 1

    # Get features
    data_dir = config['data']['path']
    features = preprocessing.get_features(config=config)

    # Setup paths
    base_dir = os.path.basename(data_dir)
    target_dir = os.path.join('results', base_dir)
    os.makedirs(target_dir, exist_ok=True)
    study_name_suffix = '_'.join(args.config.split('_')[1:])
    study_name = f'cl_m-{args.model}_out-{output_dim}_freq-{freq}_{study_name_suffix}'

    # Load data - REUSE EXISTING PREPROCESSING
    dfs = preprocessing.get_data(
        data_dir=data_dir,
        config=config,
        freq=freq,
        features=features
    )
    logging.info(f'Config: {json.dumps(params, indent=2)}')

    # Calculate test periods for retraining
    test_start = pd.Timestamp(config['data']['test_start'], tz='UTC')
    test_end = pd.Timestamp(config['data']['test_end'], tz='UTC')

    # If test_end is at 00:00 (only date specified), extend to include the whole day (23:00)
    if test_end.hour == 0 and test_end.minute == 0 and test_end.second == 0:
        test_end = test_end.replace(hour=23, minute=0, second=0)

    if retrain_enabled and retrain_interval > 1:
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
            test_periods = tools.calculate_retrain_periods(test_start, test_end, retrain_interval, freq)
        else:
            # 3. Split into chunks
            chunks = np.array_split(test_dates, retrain_interval)
            test_periods = []
            for chunk in chunks:
                if len(chunk) > 0:
                    # Start is the first date
                    p_start = chunk[0]
                    # End is the last date (we want to include all runs on this day)
                    # split_data logic now handles inclusive upper bound for starttime
                    p_end = chunk[-1]
                    # Ensure p_end covers the full day (23:59:59) if needed,
                    # but since we compare starttime <= p_end, and p_end is 00:00:00 of the last day,
                    # we might miss runs later in that day if we don't adjust.
                    # Actually, if p_end is 2025-10-21 00:00:00, and we have a run at 2025-10-21 06:00:00,
                    # starttime <= p_end would be False for that run!
                    # So we MUST set p_end to end of day.
                    p_end = p_end + pd.Timedelta(hours=23, minutes=59, seconds=59)

                    test_periods.append((p_start, p_end))

        logging.info(f"Retrain enabled: {retrain_enabled}, Test periods (Date-based): {len(test_periods)}")
        for i, (start, end) in enumerate(test_periods):
            logging.info(f"  Period {i+1}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    else:
        test_periods = [(test_start, test_end)]
        logging.info(f"Retrain enabled: {retrain_enabled}, Single test period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")

    # Storage for multiple retraining results
    all_evaluations = []
    all_histories = []

    # --- GLOBAL SCALING (same as TensorFlow version) ---
    global_scaler_x = StandardScaler()

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

    # Loop over test periods (retraining)
    for period_idx, (current_test_start, current_test_end) in enumerate(test_periods):
        logging.info(f"\n{'='*80}")
        logging.info(f"Training cycle {period_idx + 1}/{len(test_periods)}")
        logging.info(f"Test period: {current_test_start.strftime('%Y-%m-%d')} to {current_test_end.strftime('%Y-%m-%d')}")
        logging.info(f"{'='*80}\n")

        # Create period-specific config
        period_config = config.copy()
        # Use full timestamp format to preserve hours (e.g., 23:00 for period ends)
        period_config['data']['test_start'] = current_test_start.strftime('%Y-%m-%d %H:%M')
        period_config['data']['test_end'] = current_test_end.strftime('%Y-%m-%d %H:%M')

        # Fit global scaler for each period when retraining, or once if no retraining
        if period_idx == 0 or retrain_enabled:
            logging.debug(f"Fitting global scaler for period {period_idx + 1}...")

            # Reset scaler for each period when retraining (fresh start each time)
            if retrain_enabled:
                global_scaler_x = StandardScaler()

            for key, df in tqdm(dfs.items(), desc="Fitting Global Scaler"):
                df_temp = df.copy()
                target_col = period_config['data']['target_col']

                # For non-TFT models, create lag features before fitting scaler
                if period_config['model']['name'] not in ['tft', 'stemgnn']:
                    # Create lag features for observed columns (same as in pipeline)
                    for col in features['observed']:
                        all_observed_cols = [new_col for new_col in df_temp.columns if col in new_col]
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

                # Drop target column
                if target_col in df_temp.columns:
                    df_temp.drop(target_col, axis=1, inplace=True)

                t_0 = 0 if period_config['eval']['eval_on_all_test_data'] else period_config['eval']['t_0']
                df_train, _ = preprocessing.split_data(
                    data=df_temp,
                    train_frac=period_config['data']['train_frac'],
                    test_start=pd.Timestamp(period_config['data']['test_start'], tz='UTC'),
                    test_end=pd.Timestamp(period_config['data']['test_end'], tz='UTC'),
                    t_0=t_0
                )
                global_scaler_x.partial_fit(df_train)

                del df_temp, df_train
                gc.collect()

            logging.debug("Global scaler fitted.")

        # Use memory-efficient data processing (REUSE existing functions)
        logging.debug("Starting memory-efficient data processing...")
        data_generator = tools.create_data_generator(dfs, period_config, features, scaler_x=global_scaler_x)
        X_train, y_train, X_test, y_test, test_data = tools.combine_datasets_efficiently(data_generator)

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

        # --- TRAINING WITH PYTORCH ---
        history, model = tools.training_pipeline(
            train=(X_train, y_train),
            val=(X_test, y_test),
            hyperparameters=hyperparameters,
            config=period_config,
            device=device
        )

        logging.info(f"Training completed. Final metrics:")
        logging.info(f"  Train - MSE: {history['train_loss'][-1]:.6f}, RMSE: {history['train_rmse'][-1]:.4f}, R²: {history['train_r2'][-1]:.4f}")
        if 'val_loss' in history:
            logging.info(f"  Val   - MSE: {history['val_loss'][-1]:.6f}, RMSE: {history['val_rmse'][-1]:.4f}, R²: {history['val_r2'][-1]:.4f}")

        # --- GENERATE PREDICTIONS AND EVALUATE PER PARK ---
        logging.info('Start evaluation pipeline...')

        # Evaluate each park individually
        eval_results = []
        for park_key, (X_test_park, y_test_park, index_test_park, scaler_y_park) in test_data.items():
            logging.debug(f'Evaluating park: {park_key}')

            # Get park data
            park_df = dfs[park_key]

            # Determine test_start from period_config
            test_start_str = period_config['data']['test_start']
            t_0 = 0 if period_config['eval']['eval_on_all_test_data'] else period_config['eval']['t_0']

            # Run evaluation pipeline for this park
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
            # Add park identifier
            eval_results.append(park_eval)

        # Combine all evaluations for this period
        if eval_results:
            period_evaluation = pd.concat(eval_results, ignore_index=False)
        else:
            period_evaluation = pd.DataFrame()

        logging.info(f"Per-park evaluation completed: {len(eval_results)} parks evaluated")

        if not period_evaluation.empty:
            # Add aggregated statistics
            period_evaluation.loc['mean'] = period_evaluation.mean(numeric_only=True)
            period_evaluation.loc['std'] = period_evaluation.std(numeric_only=True)
            logging.info(period_evaluation)

        period_evaluation['output_dim'] = output_dim
        period_evaluation['freq'] = freq
        if period_config['eval']['eval_on_all_test_data']:
            period_evaluation['t_0'] = None
        else:
            period_evaluation['t_0'] = period_config['eval']['t_0']

        # Store results for this period
        all_evaluations.append(period_evaluation)
        all_histories.append(history)

        logging.info(f"Completed training cycle {period_idx + 1}/{len(test_periods)}")

    # --- AGGREGATE RESULTS ACROSS ALL RETRAINING PERIODS ---
    logging.info(f"\n{'='*80}")

    if len(all_evaluations) > 1:
        # Calculate mean evaluation across all retrainings
        # Only average numeric columns; for string columns (like 'freq'), take the first value
        concatenated = pd.concat(all_evaluations)
        grouped = concatenated.groupby(level=0)

        # Average numeric columns
        evaluation = grouped.mean(numeric_only=True)

        # For non-numeric columns, take the first value (they should be the same across periods)
        non_numeric_cols = concatenated.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            evaluation[col] = grouped[col].first()

        logging.info(f"Aggregated evaluation:")
        logging.info(evaluation)

        # Build results with individual results
        results = {
            'hyperparameters': hyperparameters,
            'config': config,
            'history': all_histories,  # List of dictionaries
            'evaluation': evaluation,
            'individual_evaluations': all_evaluations,
            'test_dates': test_periods
        }
    else:
        # Single training run (backward compatible)
        evaluation = all_evaluations[0]
        results = {
            'hyperparameters': hyperparameters,
            'config': config,
            'history': all_histories[0],  # Single dictionary
            'evaluation': evaluation,
            'test_dates': test_periods
        }

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path_to_pkl = os.path.join('results', base_dir, f'{study_name}_{timestamp}.pkl')

    with open(path_to_pkl, 'wb') as f:
        pickle.dump(results, f)

    logging.info(f"\nTraining completed! Results saved to: {path_to_pkl}")

    # Save model if requested
    if args.save_model:
        model_path = os.path.join('models', f'{study_name}.pt')
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved to: {model_path}")

    # Log final metrics (handle both single dict and list of dicts)
    metrics_data = []
    histories = results['history']
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

