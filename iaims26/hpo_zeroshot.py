"""Zero-shot HPO: tune TFT hyperparameters for generalization to unseen farms.

Objective (minimize): mean RMSE over the 5 site-folds, where each fold trains
on its 80 training farms and validates on the 20 *held-out* farms over the
validation window 2024-08-01 .. 2025-07-31 (all data before that window is
training data; the paper's test window 2025-08+ is never touched).

Pruning: after each fold the running mean is reported to Optuna
(trial.report(mean_so_far, step=fold)), so the MedianPruner kills bad trials
after 1-2 folds. HPO trainings use reduced epochs/patience.

The per-fold tensors are hyperparameter-independent and are built ONCE per
worker process, then reused across all trials (data loading dominates
otherwise).

Usage (one worker per GPU, l1 and l2 share the study via OPTUNA_STORAGE):
  CUDA_VISIBLE_DEVICES=g python iaims26/hpo_zeroshot.py --worker
  python iaims26/hpo_zeroshot.py --status
  python iaims26/hpo_zeroshot.py --export   # -> configs/hpo_best.json
"""

import argparse
import copy
import gc
import json
import logging
from datetime import datetime

import numpy as np
import torch

import common
import trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

STUDY_NAME = 'iaims26_zeroshot_tft'
VAL_START = '2024-08-01'
VAL_END = '2025-07-31'
MAX_TOTAL_TRIALS = 100
# Reduced training budget per HPO fold (full protocol would be ~2h/fold —
# untenable for 100 trials x 5 folds). Final fold models train full-length.
HPO_MAX_EPOCHS = 40
HPO_PATIENCE = 5
HPO_BEST_PATH = common.IAIMS_DIR / 'configs' / 'hpo_best.json'


def fit_fold_scaler(dfs_train: dict, config: dict):
    """Replicates train_cl.py's global scaler fitting (partial_fit per farm
    on the training range)."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from utils import preprocessing

    scaler = StandardScaler()
    target_col = config['data']['target_col']
    for df in dfs_train.values():
        df_train, _ = preprocessing.split_data(
            data=df,
            train_frac=config['data']['train_frac'],
            test_start=pd.Timestamp(config['data']['test_start'], tz='UTC'),
            test_end=pd.Timestamp(config['data']['test_end'], tz='UTC'),
            t_0=0,
        )
        scaler.partial_fit(df_train.drop(columns=[target_col], errors='ignore').values)
    return scaler


def build_fold_tensors(base_config: dict, folds: dict, dfs_all: dict):
    """Build (X_train, y_train, X_val, y_val) per fold for the HPO window."""
    from utils import tools, preprocessing

    fold_data = {}
    key_by_id = {common.station_key_to_id(key): key for key in dfs_all}
    all_stations = sorted(key_by_id)
    for k in range(folds['n_folds']):
        heldout = folds['folds'][str(k)]
        train_stations = [s for s in all_stations if s not in heldout]
        assert len(train_stations) + len(heldout) == len(all_stations)

        cfg = common.make_eval_config(base_config, VAL_START, VAL_END)
        features = preprocessing.get_features(config=cfg)
        dfs_train = {key_by_id[s]: dfs_all[key_by_id[s]] for s in train_stations}
        dfs_val = {key_by_id[s]: dfs_all[key_by_id[s]] for s in heldout}

        scaler = fit_fold_scaler(dfs_train, cfg)

        gen = tools.create_data_generator(dfs_train, copy.deepcopy(cfg),
                                          features, scaler_x=scaler)
        X_train, y_train, _, _, _, _ = tools.combine_datasets_efficiently(gen)
        gen = tools.create_data_generator(dfs_val, copy.deepcopy(cfg),
                                          features, scaler_x=scaler)
        _, _, X_val, y_val, _, _ = tools.combine_datasets_efficiently(gen)

        fold_data[k] = (X_train, y_train, X_val, y_val)
        logging.info(f'fold {k}: train windows={len(y_train)}, '
                     f'heldout-val windows={len(y_val)}')
        gc.collect()
    return fold_data


def make_objective(config: dict, fold_data: dict, device: str):
    import optuna
    from utils import hpo as hpo_utils

    def objective(trial):
        cfg = copy.deepcopy(config)
        cfg['model']['fl'] = False
        hyperparameters = hpo_utils.get_hyperparameters(config=cfg, hpo=True, trial=trial)
        logging.info(f'trial {trial.number}: {trial.params}')

        rmses = []
        for step, k in enumerate(sorted(fold_data)):
            X_train, y_train, X_val, y_val = fold_data[k]
            dims = {
                'observed_dim': X_train['observed'].shape[-1],
                'known_dim': X_train['known'].shape[-1],
                'static_dim': X_train['static'].shape[-1] if 'static' in X_train else 0,
            }
            model = common.build_tft(cfg, hyperparameters, dims)
            try:
                history, model, epochs_run = trainer.train_model(
                    model, X_train, y_train, X_val, y_val,
                    batch_size=hyperparameters['batch_size'],
                    lr=hyperparameters['lr'],
                    device=device,
                    epochs=HPO_MAX_EPOCHS,
                    patience=HPO_PATIENCE,
                    clipnorm=hyperparameters.get('clipnorm'),
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logging.warning(f'trial {trial.number} fold {k}: CUDA OOM — pruning')
                raise optuna.TrialPruned()
            best_rmse = float(min(history['val_rmse']))
            rmses.append(best_rmse)
            del model
            torch.cuda.empty_cache()
            gc.collect()

            mean_so_far = float(np.mean(rmses))
            trial.report(mean_so_far, step)
            logging.info(f'trial {trial.number} fold {k}: val_rmse={best_rmse:.5f} '
                         f'(mean so far {mean_so_far:.5f}, {epochs_run} epochs)')
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(rmses))

    return objective


def get_study(config: dict):
    from utils import hpo as hpo_utils
    return hpo_utils.create_or_load_study(
        path=config['hpo']['studies_path'],
        study_name=STUDY_NAME,
        pruning_config=config['hpo']['pruning'],
        config=config,
    )


def cmd_worker(args):
    import optuna
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = common.load_base_config()
    folds = common.load_folds()
    stations = sorted(str(s) for s in config['data']['files'])

    logging.info(f'loading data for {len(stations)} stations (once per worker)…')
    dfs_all = common.get_station_dfs(config, stations)
    fold_data = build_fold_tensors(config, folds, dfs_all)
    del dfs_all
    gc.collect()

    study = get_study(config)
    objective = make_objective(config, fold_data, device)
    study.optimize(
        objective,
        n_trials=args.trials,
        callbacks=[MaxTrialsCallback(MAX_TOTAL_TRIALS,
                                     states=(TrialState.COMPLETE, TrialState.PRUNED))],
        gc_after_trial=True,
    )
    logging.info('worker finished.')


def cmd_status(args):
    from optuna.trial import TrialState
    config = common.load_base_config()
    study = get_study(config)
    trials = study.trials
    by_state = {}
    for t in trials:
        by_state[t.state.name] = by_state.get(t.state.name, 0) + 1
    print(f'study {STUDY_NAME}: {len(trials)} trials {by_state}')
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    if complete:
        best = study.best_trial
        print(f'best: value={best.value:.5f} trial={best.number}')
        for k, v in best.params.items():
            print(f'  {k}: {v}')


def cmd_export(args):
    config = common.load_base_config()
    study = get_study(config)
    best = study.best_trial
    payload = {
        'study': STUDY_NAME,
        'exported': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'trial_number': best.number,
        'value_mean_heldout_rmse': best.value,
        'params': best.params,
    }
    with open(HPO_BEST_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'wrote {HPO_BEST_PATH}:')
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--worker', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--export', action='store_true')
    parser.add_argument('--trials', type=int, default=MAX_TOTAL_TRIALS,
                        help='max trials for THIS worker (global cap: 100)')
    args = parser.parse_args()
    if args.worker:
        cmd_worker(args)
    elif args.status:
        cmd_status(args)
    else:
        cmd_export(args)


if __name__ == '__main__':
    main()
