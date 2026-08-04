# Hyperparameter optimization for the TFT non-GNN wind-speed-forecasting
# benchmark (geostatistics study) — forked from hpo_cl.py.
#
# Difference vs. hpo_cl.py: next_n_grid_points / next_n_grid_ecmwf /
# next_n_stations (preprocessing-level, data-shape-changing parameters) are
# sampled BY OPTUNA per trial, not fixed once before the trial loop. hpo_cl.py
# calls data_cache.create_or_load_preprocessed_data() exactly once, before the
# Optuna loop, so it cannot vary these — see plan "Lücke 3"
# (~/.claude/plans/hidden-booping-salamander.md). Here the call moves inside
# the trial loop; utils/data_cache.py::_get_config_hash already hashes these
# four params (fixed alongside "Lücke 1"), so repeated values hit the cache
# and only genuinely new combinations trigger reprocessing.
#
# Everything else (architecture/optimizer HPO via hpo.get_hyperparameters,
# fold loop, pruning, study bookkeeping) is unchanged from hpo_cl.py.
#
# Restored from archiv/hpo_cl_tft_bc.py (removed from the tree by commit
# a65285c "new GNN HPO") and renamed hpo_tft_bc.py — repo-root, alongside the
# other top-level CL/TFT HPO scripts (hpo_cl.py, hpo_fl.py,
# hpo_trianel_cl.py), NOT under geostatistics/ where the GNN HPO scripts
# (hpo_dcrnn.py/hpo_mtgnn.py/hpo_wavenet.py) live: this script is forked from
# hpo_cl.py's family (its own docstring says so) and always lived at the repo
# root before being archived — moving it under geostatistics/ now would just
# be a second, gratuitous path change alongside the real fix below.
#
# cv_mode='spatial' (added on top of the archived script, see the fold-loop
# below and utils/data_cache.py::create_or_load_preprocessed_data_spatial):
# mirrors geostatistics/hpo_dcrnn.py's spatial 3-fold CV so the TFT benchmark
# uses the same CV axis as DCRNN/MTGNN/WaveNet — same time window in every
# fold (train < data.val_start, val in [val_start, test_start)), rotating
# TARGET STATIONS instead of a fixed files/val_files split. Default is
# cv_mode='temporal' (bitwise the old behaviour); see configs/tft_bc/
# config_wind_tft_sp_{base,hist}.yaml for the new spatial studies.

import os
import re
import json
import argparse
import copy
import shutil
import time
import fcntl
import pandas as pd
import numpy as np
import logging
import gc

import torch
import optuna

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import preprocessing, tools, hpo, data_cache
from geostatistics.spatial_cv import load_spatial_folds, station_pool, build_folds, resolve_cv_mode

optuna.logging.set_verbosity(optuna.logging.INFO)


# ── Cache-budget enforcement ─────────────────────────────────────────────────
#
# next_n_grid_points/next_n_grid_ecmwf/next_n_stations vary per trial (see
# module docstring above), so every genuinely new combination reprocesses and
# caches a full dataset under utils/data_cache.py's DataCache (files named
# "<cache_id>_{metadata,prepared,fold_manifest}.pkl" + "<cache_id>_folds/" in
# the shared, flat data_cache/ dir). With ranges [1,7]x[0,4]x[0,8] that's up
# to 315 distinct combinations *per study*, ~300-600MB each — left unchecked
# this can fill the shared disk over a long HPO run. This tracks only the
# cache_ids THIS script creates (in a small JSON manifest with cross-process
# file locking, since 2 parallel workers per study share one cache_dir) and
# evicts the least-recently-used ones once the tracked total exceeds
# --max-cache-gb. Never touches cache entries it didn't create itself (e.g.
# DCRNN/MTGNN's data_cache/gnns/ or other users' CL-pipeline caches).

CACHE_MANIFEST_NAME = '.tft_bc_cache_manifest.json'
CACHE_EVICTION_GRACE_SECONDS = 300  # never evict an entry touched in the last 5 min


def _cache_entry_paths(cache_dir: str, cache_id: str) -> list:
    base = os.path.join(cache_dir, cache_id)
    return [
        f'{base}_metadata.pkl',
        f'{base}_prepared.pkl',
        f'{base}_prepared.npy',
        f'{base}_fold_manifest.pkl',
        f'{base}_kfolds.pkl',
        f'{base}_kfolds.npy',
        f'{base}_folds',  # directory
    ]


def _cache_entry_size_bytes(cache_dir: str, cache_id: str) -> int:
    total = 0
    for p in _cache_entry_paths(cache_dir, cache_id):
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        elif os.path.exists(p):
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
    return total


def _delete_cache_entry(cache_dir: str, cache_id: str) -> None:
    for p in _cache_entry_paths(cache_dir, cache_id):
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def enforce_cache_budget(cache_dir: str, cache_id: str, max_cache_gb: float, logger: logging.Logger) -> None:
    """Register `cache_id` as just-used and evict LRU entries over budget.

    Safe across the 2 parallel workers per study: the manifest read-modify-write
    is wrapped in an flock on the manifest file itself.
    """
    manifest_path = os.path.join(cache_dir, CACHE_MANIFEST_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    # Ensure the file exists before opening 'r+' (flock needs an existing fd).
    if not os.path.exists(manifest_path):
        with open(manifest_path, 'a'):
            pass

    with open(manifest_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            try:
                manifest = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                manifest = {}

            now = time.time()
            manifest[cache_id] = now  # register/refresh last-used time

            total_bytes = sum(
                _cache_entry_size_bytes(cache_dir, cid) for cid in manifest
            )
            max_bytes = max_cache_gb * (1024 ** 3)

            if total_bytes > max_bytes:
                # Evict oldest-first, skipping the entry we just used and
                # anything touched very recently (still being written by a
                # concurrent worker).
                by_age = sorted(
                    (cid for cid in manifest if cid != cache_id),
                    key=lambda cid: manifest[cid],
                )
                for cid in by_age:
                    if total_bytes <= max_bytes:
                        break
                    if now - manifest[cid] < CACHE_EVICTION_GRACE_SECONDS:
                        continue
                    size = _cache_entry_size_bytes(cache_dir, cid)
                    _delete_cache_entry(cache_dir, cid)
                    del manifest[cid]
                    total_bytes -= size
                    logger.info(
                        f'Cache budget: evicted {cid} ({size / 1e9:.2f} GB), '
                        f'tracked total now {total_bytes / 1e9:.2f} GB (budget {max_cache_gb:.0f} GB)'
                    )

            f.seek(0)
            f.truncate()
            json.dump(manifest, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization (TFT non-GNN benchmark)")
    parser.add_argument('-m', '--model', type=str, default='tft', help='Select Model (default: tft)')
    parser.add_argument('-c', '--config', type=str, help='Select config')
    parser.add_argument('-s', '--suffix', type=str, default='', help='Define suffix')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching for small datasets')
    parser.add_argument('--gpu', type=int, default=None, help='GPU to use (default: auto-select)')
    parser.add_argument('--max-cache-gb', type=float, default=150.0,
                        help='Cap on disk space used by this script\'s cache entries in data_cache/ '
                             '(shared budget across all parallel workers on the same host via a locked '
                             'manifest); least-recently-used entries are evicted once exceeded (default: 150)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory for preprocessed-data cache entries. Defaults to '
                             'utils.data_cache.DEFAULT_CACHE_DIR, which is host-specific — '
                             'set this on hosts where that mount does not exist.')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    suffix = f'_{args.suffix}' if args.suffix else ''
    if '.yaml' in args.config:
        args.config = args.config.split('.')[0]
    config_name = args.config.split('/')[-1] if '/' in args.config else args.config
    if config_name.startswith('config_'):
        config_name = config_name[7:]
    log_file = f'logs/hpo_cl_tft_bc_m-{args.model}_c-{config_name}{suffix}.log'

    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"NEW HPO SESSION (tft_bc) - Model: {args.model}, Config: {args.config}")
    logger.info("=" * 80)

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)

    base_config = tools.load_config(f'{args.config}.yaml')
    base_config['model']['verbose'] = 0
    freq = base_config['data']['freq']
    base_config = tools.handle_freq(config=base_config)
    output_dim = base_config['model']['output_dim']
    lookback = base_config['model']['lookback']
    horizon = base_config['model']['horizon']
    base_config['model']['fl'] = False
    base_config['model']['name'] = args.model

    # Ranges for the preprocessing-level params — read once, sampled per trial below.
    # Falls back to the config's fixed value (as a single-point range) if no
    # hpo.<param> range is given, so this script also works on configs that
    # don't want these varied.
    def _range(key, default_low, default_high):
        r = base_config['hpo'].get(key)
        if r is None:
            fixed = base_config['params'].get(key, default_low)
            return [fixed, fixed]
        return r

    grid_points_range = _range('next_n_grid_points', 1, 7)
    grid_ecmwf_range = _range('next_n_grid_ecmwf', 0, 4)
    stations_range = _range('next_n_stations', 0, 8)

    logging.info(
        f'HPO for Model: {args.model}, Output dim: {output_dim}, Frequency: {freq}, '
        f'Lookback: {lookback}, Horizon: {horizon}, Step size: {base_config["model"]["step_size"]}, '
        f'next_n_grid_points range: {grid_points_range}, next_n_grid_ecmwf range: {grid_ecmwf_range}, '
        f'next_n_stations range: {stations_range}'
    )

    features = preprocessing.get_features(config=base_config)

    data_dir = base_config['data']['path']
    base_dir = os.path.basename(data_dir)
    os.makedirs(os.path.join('results', base_dir), exist_ok=True)
    # Strip a trailing _fold<N> from the config name so fold-specific configs (used by
    # the spatial-CV retrain path, train_cl_tft_bc.py / get_test_results_tft_bc.py)
    # read/write the SAME Optuna study as the foldless HPO config — matches
    # geostatistics/hpo_dcrnn.py's hpo_stem derivation. This HPO script itself is only
    # ever invoked with the foldless config (config_wind_tft_sp_base.yaml, not
    # _fold1/2/3.yaml — those exist purely for the retrain/eval scripts), so this is a
    # no-op for every current call site; it exists so a future 'point HPO at a fold
    # config by mistake' still lands in the right, shared study instead of quietly
    # opening a stray per-fold one.
    hpo_stem = re.sub(r'_fold\d+$', '', config_name)
    study_name_suffix = hpo_stem + suffix
    study_name = f'cl_m-{args.model}-bc_out-{output_dim}_freq-{freq}_{study_name_suffix}'

    pruning_config = base_config.get('hpo', {}).get('pruning', {})
    study = hpo.create_or_load_study(
        base_config['hpo']['studies_path'],
        study_name,
        pruning_config=pruning_config,
        config=base_config,
    )

    # ── cv_mode: temporal (default, old behaviour) | spatial (new: rotating target
    # stations, fixed time window — see geostatistics/spatial_cv.py / hpo_dcrnn.py) ──
    cv_mode, spatial_folds_path = resolve_cv_mode(base_config.get('hpo', {}))
    spatial_folds = None
    all_station_ids = None
    if cv_mode == 'spatial':
        spatial_fold_defs = load_spatial_folds(spatial_folds_path)
        all_station_ids = station_pool(spatial_fold_defs)
        spatial_folds = build_folds(spatial_fold_defs, all_station_ids,
                                     max_val_stations=base_config['hpo'].get('n_val_stations'))
        logging.info(
            "CV-Modus: raeumlich — %d Folds aus %s, %d Stationen im Pool. "
            "config['data']['files']/['val_files'] (%d/%d Stationen) werden IGNORIERT "
            "und je Fold durch die Stationsrollen aus spatial_folds.yaml ersetzt.",
            len(spatial_folds), spatial_folds_path, len(all_station_ids),
            len(base_config['data'].get('files', [])), len(base_config['data'].get('val_files', [])),
        )
        for sf in spatial_folds:
            logging.info("  %s: %d train / %d target stations", sf.name, len(sf.train_idx), len(sf.val_idx))
    else:
        logging.info("CV-Modus: zeitlich (temporal) — unveraendertes Verhalten.")

    objectives, is_multi_objective = hpo.get_objectives_from_config(base_config)
    logging.info(f"HPO Mode: {'Multi-Objective' if is_multi_objective else 'Single-Objective'}")
    obj_strs = [f"{obj['metric']} ({obj['direction']})" for obj in objectives]
    logging.info(f"Objectives: {obj_strs}")

    use_cache = not args.no_cache

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    logging.info(f"Using device: {device}")

    len_trials = len(study.trials)
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    logging.info(f'Starting HPO with {base_config["hpo"]["trials"] - completed_trials} new trials.')
    logging.info(f'Previous trials: {len_trials} total, {completed_trials} completed, {pruned_trials} pruned.')

    trial_counter = 0
    while completed_trials < base_config['hpo']['trials']:
        trial = study.ask()
        trial_number = len_trials + trial_counter

        # ── Preprocessing-level params: sampled FIRST, own config copy per trial ──
        config = copy.deepcopy(base_config)
        n_grid_points = trial.suggest_int('next_n_grid_points', grid_points_range[0], grid_points_range[1])
        n_grid_ecmwf = trial.suggest_int('next_n_grid_ecmwf', grid_ecmwf_range[0], grid_ecmwf_range[1])
        n_stations = trial.suggest_int('next_n_stations', stations_range[0], stations_range[1])
        config['params']['next_n_grid_points'] = n_grid_points
        config['params']['next_n_grid_ecmwf'] = n_grid_ecmwf
        config['params']['next_n_stations'] = n_stations

        hyperparameters = hpo.get_hyperparameters(config=config, hpo=True, trial=trial)

        existing_params = [t.params for t in study.trials]
        current_params = trial.params
        param_count = sum(1 for params in existing_params if params == current_params)
        if param_count > 1:
            logging.warning(
                f"Trial number {trial_number}: Duplicate parameters detected "
                f"(found {param_count} times), marking as failed..."
            )
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue

        logging.info(
            f"Complete trial number {completed_trials+1}: "
            f"grid_points={n_grid_points}, grid_ecmwf={n_grid_ecmwf}, stations={n_stations}, "
            f"{json.dumps(hyperparameters)}"
        )

        try:
            # ── Data loading: temporal (single call, kfolds folds already inside) vs.
            # spatial (one call per fold — each fold is its own DataCache entry, since
            # station role determines both the scaler and the neighbour-restricted
            # station data; see create_or_load_preprocessed_data_spatial's docstring). ──
            fold_data = []  # list of (fold_idx, (train, val))
            if cv_mode == 'temporal':
                logging.info(
                    f"Loading/creating preprocessed data for next_n_grid_points={n_grid_points}, "
                    f"next_n_grid_ecmwf={n_grid_ecmwf}, next_n_stations={n_stations} …"
                )
                lazy_fold_loader, cache_id = data_cache.create_or_load_preprocessed_data(
                    config=config,
                    features=features,
                    model_name=args.model,
                    force_reprocess=False,
                    use_cache=use_cache,
                    cache_dir=args.cache_dir,
                )
                logging.info(f"Using data with cache ID: {cache_id}, {len(lazy_fold_loader)} folds")
                if use_cache and cache_id != "no_cache":
                    enforce_cache_budget(args.cache_dir or data_cache.DEFAULT_CACHE_DIR, cache_id,
                                         args.max_cache_gb, logging.getLogger())
                for fold_idx in range(len(lazy_fold_loader)):
                    fold_data.append((fold_idx, lazy_fold_loader[fold_idx]))
            else:  # spatial
                logging.info(
                    f"Loading/creating preprocessed data (spatial CV, {len(spatial_folds)} folds) for "
                    f"next_n_grid_points={n_grid_points}, next_n_grid_ecmwf={n_grid_ecmwf}, "
                    f"next_n_stations={n_stations} …"
                )
                for fold_idx, sf in enumerate(spatial_folds):
                    fold_config = copy.deepcopy(config)
                    fold_config['data']['files'] = [all_station_ids[i] for i in sf.train_idx]
                    fold_config['data']['val_files'] = [all_station_ids[i] for i in sf.val_idx]
                    lazy_fold_loader, cache_id = data_cache.create_or_load_preprocessed_data_spatial(
                        config=fold_config,
                        features=features,
                        model_name=args.model,
                        force_reprocess=False,
                        use_cache=use_cache,
                        cache_dir=args.cache_dir,
                    )
                    logging.info(
                        f"  {sf.name}: cache ID {cache_id}, {len(sf.train_idx)} train / "
                        f"{len(sf.val_idx)} target stations"
                    )
                    if use_cache and cache_id != "no_cache":
                        enforce_cache_budget(args.cache_dir or data_cache.DEFAULT_CACHE_DIR, cache_id,
                                             args.max_cache_gb, logging.getLogger())
                    fold_data.append((fold_idx, lazy_fold_loader[0]))

            accuracies = []
            for fold_idx, fold in fold_data:
                train, val = fold

                X_train, y_train = train
                if val and val[0] is not None:
                    X_val, y_val = val
                    logging.debug(f"Fold {fold_idx}: Train samples: {len(y_train)}, Val samples: {len(y_val)}")
                else:
                    logging.warning(f"Fold {fold_idx}: No validation data! Train samples: {len(y_train)}")

                config['model_name'] = (
                    f'hpo_cl_tft_bc_m-{args.model}_out-{output_dim}_freq-{freq}'
                    f'trial-{trial_number}_fold-{fold_idx}'
                )

                history, model = tools.training_pipeline(
                    train=train, val=val, hyperparameters=hyperparameters, config=config, device=device,
                )

                metric_map = {
                    'loss': 'val_loss', 'val_loss': 'val_loss', 'mse': 'val_loss',
                    'rmse': 'val_rmse', 'mae': 'val_mae', 'r2': 'val_r2', 'val_r2': 'val_r2',
                }

                # Report the BEST epoch's value, not the last one: early stopping runs
                # `patience` further epochs after the optimum and restores the best
                # weights (restore_best_weights=True), so history[-1] describes a model
                # that is by construction worse than the one actually kept.
                def _best(values, direction):
                    return min(values) if direction == 'minimize' else max(values)

                fold_metrics = {}
                for obj in objectives:
                    metric_key = obj['metric']
                    if metric_key in history and len(history[metric_key]) > 0:
                        fold_metrics[metric_key] = _best(history[metric_key], obj['direction'])
                    elif metric_map.get(metric_key) in history and len(history[metric_map.get(metric_key, '')]) > 0:
                        fold_metrics[metric_key] = _best(history[metric_map[metric_key]], obj['direction'])
                    else:
                        available = [k for k in history.keys() if len(history[k]) > 0]
                        empty = [k for k in history.keys() if len(history[k]) == 0]
                        raise ValueError(
                            f"Could not find metric '{metric_key}' with values. "
                            f"Available (non-empty): {available}. Empty: {empty}."
                        )

                if not accuracies:
                    accuracies = {obj['metric']: [] for obj in objectives}
                for obj in objectives:
                    accuracies[obj['metric']].append(fold_metrics[obj['metric']])

                if not is_multi_objective:
                    # Intermediate value for the pruner. Under cv_mode='spatial' this is the
                    # RUNNING MEAN over the folds finished so far, matching
                    # geostatistics/hpo_dcrnn.py:1295 (trial.report(float(np.mean(fold_losses)),
                    # step=fold_idx)) — the DCRNN/MTGNN/WaveNet studies this benchmark exists to be
                    # compared against. Two reasons it matters here specifically:
                    #   * the running mean is a running estimate of the value actually reported to
                    #     Optuna at the end (study.tell(trial, average_accuracy) below); the raw
                    #     single-fold value is not.
                    #   * under spatial CV the folds hold out DIFFERENT target stations and differ
                    #     markedly in difficulty, so a raw fold value carries a large fold-specific
                    #     offset. MedianPruner only ever compares trials at equal steps, so raw
                    #     values are not *invalid* — but they make the pruning signal much noisier
                    #     than the graph studies', i.e. the TFT benchmark would be searched under a
                    #     systematically worse selection rule than the models it is benchmarked
                    #     against, which is exactly the asymmetry cv_mode='spatial' exists to remove.
                    # cv_mode='temporal' deliberately keeps reporting the raw fold value: the
                    # existing temporal studies already hold intermediate values on that scale, and
                    # MedianPruner compares within a study, so switching mid-study would pit two
                    # different statistics against each other at the same step.
                    if cv_mode == 'spatial':
                        report_value = float(np.mean(accuracies[objectives[0]['metric']]))
                    else:
                        report_value = fold_metrics[objectives[0]['metric']]
                    trial.report(report_value, step=fold_idx)

                metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in fold_metrics.items()])
                # len(fold_data), not len(lazy_fold_loader): under cv_mode='spatial' the
                # last lazy_fold_loader in scope is a single-fold loader for the LAST
                # spatial fold only (each fold is its own create_or_load_preprocessed_data_spatial
                # call), so it would misreport "fold N/1" instead of "fold N/3".
                logging.info(f'Processed fold {fold_idx + 1}/{len(fold_data)}, {metrics_str}')

                del model, history
                gc.collect()

                if not is_multi_objective and trial.should_prune():
                    raise optuna.TrialPruned()

            else:
                if is_multi_objective:
                    average_values, metrics_summary = [], []
                    for obj in objectives:
                        avg_value = np.mean(accuracies[obj['metric']])
                        average_values.append(avg_value)
                        metrics_summary.append(f"{obj['metric']}: {avg_value:.4f}")
                    logging.info(f'Fold averages: {accuracies}')
                    study.tell(trial, values=average_values)
                    logging.info(f'Trial number {trial_number+1} completed with: {", ".join(metrics_summary)}')
                else:
                    metric_name = objectives[0]['metric']
                    fold_values = accuracies[metric_name]
                    average_accuracy = np.mean(fold_values)
                    logging.info(f'Accuracies for the folds: {fold_values}')
                    study.tell(trial, average_accuracy)
                    logging.info(
                        f'Trial number {trial_number+1} completed with average {metric_name}: {average_accuracy:.4f}'
                    )

                logging.info(f'Progress: {completed_trials}/{base_config["hpo"]["trials"]} successful trials completed.')
                completed_trials += 1

        except optuna.TrialPruned:
            logging.info(f'Trial number {trial_number+1} was pruned by Optuna')
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)

        except KeyboardInterrupt:
            logging.warning(f'Trial number {trial_number+1} interrupted by user. Marking as failed.')
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise

        except Exception as e:
            logging.error(f'Trial number {trial_number+1} failed with error: {str(e)}. Marking as failed.')
            logging.exception("Full traceback:")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise

        trial_counter += 1

    logging.info("=" * 80)
    logging.info("HPO COMPLETED")
    logging.info("=" * 80)

    if is_multi_objective:
        best_trials = study.best_trials
        logging.info(f"Pareto Front: {len(best_trials)} non-dominated solutions")
        for i, trial in enumerate(best_trials[:10]):
            values_str = ", ".join(
                f"{obj['metric']}: {trial.values[j]:.6f}" for j, obj in enumerate(objectives)
            )
            logging.info(f"  Solution {i+1}: {values_str}")
            if i == 0:
                logging.info(f"  Params: {json.dumps(trial.params, indent=2)}")
    else:
        logging.info(f"Best trial: {study.best_trial.number}")
        logging.info(f"Best {objectives[0]['metric']}: {study.best_value:.6f}")
        logging.info(f"Best params: {json.dumps(study.best_params, indent=2)}")

    logging.info("=" * 80)


if __name__ == '__main__':
    main()
