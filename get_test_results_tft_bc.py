#!/usr/bin/env python3
"""
get_test_results_tft_bc.py — Evaluate a trained tft_bc model (from train_cl_tft_bc.py)
on held-out test_files, in PHYSICAL units, analogous to
geostatistics/get_test_results_dcrnn.py and train_mtgnn.py::_metrics (RMSE_phys).

The target (scale_target=False) is never scaled, so scaler_y is always None and
tools.get_y() skips inverse-transforming y — no scaler_y handling needed here.

Mehrere Zielgroessen (Solar: data.target_cols: [ghi, dhi]) werden je Zielgroesse
getrennt ausgewertet; die Ergebniszeilen und das Roh-Parquet tragen dann eine
Spalte 'target', genau wie bei den Graphmodellen. Drei Solar-Eigenheiten sind
dabei beruecksichtigt, alle drei still, wenn man sie uebersieht:

* ``params.target_transform: nwp_residual`` — die Zielspalte ist 'Messung - NWP'.
  tools.get_y darf dann NICHT bei 0 clippen (rund die Haelfte der Zielwerte ist
  negativ), RMSE und MAE sind in beiden Raeumen identisch, die NWP-Baseline ist
  die Nullreihe, und fuer pred/gt im Parquet wird die physikalische Skala ueber
  die abgezogene Basisspalte zurueckgerechnet (solar.resolve_residual_baseline_col).
* ``eval.exclude_imputed`` — gefuellte Zielpositionen fliegen elementweise ueber
  '<target>_observed' aus Metriken und Parquet.
* Lead-0-Label — ICON-D2 SL labelt linksbuendig auf die Laufzeit, ML erst eine
  Stunde danach (shared.resolution.lead0_offset). Das bestimmt valid_time und
  den Bezugspunkt der Persistenz.

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
from utils.eval import _column_by_run
from utils.solar import resolve_residual_baseline_col
from geostatistics.shared.resolution import lead0_offset

#: Einheit je Zielgroesse, nur fuer die Log-Ausgabe. Gleiche Zuordnung wie
#: geostatistics/get_test_results_dcrnn.py.
EINHEIT = {'wind_speed': 'm/s', 'power': 'kW',
           'ghi': 'W/m²', 'dhi': 'W/m²', 'bhi': 'W/m²', 'dni': 'W/m²'}


def _spalte_je_lauf(df: pd.DataFrame, col: str, run_times, horizon: int):
    """Spalte ``col`` als (n_runs, horizon)-Array, laufweise auf ``run_times`` gelegt.

    Nutzt eval._column_by_run: bei NWP-Daten baut create_tft_sequences genau EIN
    Fenster je Vorhersagelauf, Lead j ist also 'forecasttime j' desselben
    starttime. Ein Pivot auf (starttime x forecasttime) trifft diese Zuordnung —
    ein groupby('timestamp') wuerde ueber ueberlappende Laeufe mitteln und die
    Baseline glaetten, die die Vorhersage nicht bekommt (eval.py:620).
    """
    if col is None or col not in df.columns:
        return None
    vorlage = pd.DataFrame(index=pd.Index(run_times),
                           columns=[f't+{i + 1}' for i in range(horizon)])
    piv = _column_by_run(df, col, vorlage)
    return None if piv is None else piv.to_numpy(dtype=float)


def _persistenz(df: pd.DataFrame, target_col: str, basis_col, run_times,
                freq_delta, lead0_off: int, nwp_residual: bool):
    """Letzter Messwert vor Prognosestart, in physikalischen Einheiten.

    Der Prognosestart liegt bei ``run_time + lead0_off * freq`` (Wind: eine
    Stunde nach dem Lauf, Solar: der Lauf selbst — shared.resolution.lead0_offset),
    der Referenzwert also einen Schritt davor. Dieselbe Definition wie
    homo_sampler.evaluate_homo_model (``meas_raw[t_run_abs - 1]``) fuer die
    Graphmodelle.

    Bei ``target_transform: nwp_residual`` steht in der Zielspalte bereits das
    Residuum; die Messreihe ist daraus nur mit der Basisspalte zurueckzugewinnen.
    """
    if target_col not in df.columns:
        return None
    reihe = df[target_col]
    if nwp_residual:
        if basis_col is None or basis_col not in df.columns:
            return None
        reihe = reihe + df[basis_col]
    if isinstance(reihe.index, pd.MultiIndex):
        if 'timestamp' not in (reihe.index.names or []):
            return None
        reihe = reihe.droplevel([n for n in reihe.index.names if n != 'timestamp'])
    reihe = reihe[~reihe.index.duplicated(keep='first')].sort_index()
    zeitpunkte = pd.DatetimeIndex(run_times) + freq_delta * (lead0_off - 1)
    return reihe.reindex(zeitpunkte).to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test-set evaluation for tft_bc models (physical units)")
    parser.add_argument('-m', '--model', type=str, default='tft')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--eval-split', choices=('test', 'val'), default='test',
                        help="'test': test_files im Fenster [test_start, test_end] (Default). "
                             "'val': die val_files des Folds im Fenster [val_start, test_start), "
                             "also das Validierungsfenster — fuer Trockenlaeufe, die den "
                             "Testsatz unangetastet lassen sollen.")
    parser.add_argument('--hpo-study', type=str, default=None,
                        help='Optuna study to take best_trial params from. Omit for a standard-hyperparameter dry run — train and eval must omit it together.')
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

    # --hpo-study optional, siehe train_cl_tft_bc.py. Ohne Studie stammen die
    # next_n_*-Werte aus der Config und muessen mit denen des Trainingslaufs
    # uebereinstimmen, sonst zeigt die cache_id auf einen anderen Datensatz.
    study = None
    best = None
    if args.hpo_study:
        storage_url = os.environ.get('OPTUNA_STORAGE')
        if not storage_url:
            raise RuntimeError("OPTUNA_STORAGE env var must be set to load the HPO study.")
        study = optuna.load_study(study_name=args.hpo_study, storage=storage_url)
        best = study.best_trial
        for key in ('next_n_grid_points', 'next_n_grid_ecmwf', 'next_n_stations'):
            if key in best.params:
                config['params'][key] = best.params[key]
    else:
        logger.info("Kein --hpo-study: Preprocessing-Parameter aus der Config (Trockenlauf).")
    logger.info(
        f"Preprocessing params ({'best trial' if best else 'Config, Trockenlauf'}): "
        f"next_n_grid_points={config['params']['next_n_grid_points']}, "
        f"next_n_grid_ecmwf={config['params']['next_n_grid_ecmwf']}, "
        f"next_n_stations={config['params']['next_n_stations']}"
    )

    meta_path = os.path.join('models', f'{args.model_tag}_meta.pkl')
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    hyperparameters = metadata['hyperparameters']
    config['model']['feature_dim'] = metadata['feature_dim']
    if metadata.get('best_trial_number') is not None:
        logger.info(f"Loaded metadata from {meta_path} (best_trial={metadata['best_trial_number']}, "
                    f"best_value={metadata['best_trial_value']:.6f})")
    else:
        logger.info(f"Loaded metadata from {meta_path} (Trockenlauf, keine HPO-Studie)")

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

    # --- Auswertungsfenster festlegen (NACH der cache_id-Berechnung!) ---
    # test_start geht in DataCache._get_config_hash ein. Es hier zu aendern ist nur
    # deshalb unbedenklich, weil train_cache_id oben bereits berechnet und der
    # scaler_x daraus schon geladen ist.
    if args.eval_split == 'val':
        val_start = config['data'].get('val_start')
        if not val_start:
            raise ValueError("--eval-split val braucht data.val_start in der Config.")
        if not config['data'].get('val_files'):
            raise ValueError("--eval-split val braucht data.val_files in der Config.")
        eval_start, eval_end = str(val_start), str(config['data']['test_start'])
        config['data']['test_files'] = list(config['data']['val_files'])
        config['data']['test_start'] = eval_start
        config['data']['test_end'] = eval_end
        # Nur die Trainingsstationen des Folds als Nachbarn — dieselbe Menge, die
        # create_or_load_preprocessed_data_spatial den Zielstationen im Training
        # zugestanden hat. Sonst saehe die Auswertung mehr als die Validierung.
        config['data']['neighbor_pool'] = list(config['data'].get('files', []))
        logger.info(
            f"--eval-split val: {len(config['data']['test_files'])} Zielstationen des Folds "
            f"im Fenster {eval_start} .. {eval_end}; Nachbar-Pool "
            f"{len(config['data']['neighbor_pool'])} Trainingsstationen (train-role only)"
        )
    else:
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

    # target_col ist hier nur der Default fuer get_data: data.target_col hat drin
    # ohnehin Vorrang, und bei Solar steht dort None, weil die Ziele in
    # data.target_cols stehen. get_target_cols loest beides einheitlich auf.
    test_dfs = preprocessing.get_data(
        data_dir=config['data']['path'],
        config=config,
        freq=freq,
        features=features,
        target_col=preprocessing.get_target_cols(config)[0],
        files_key='test_files',
    )
    logger.info(f"Loaded {len(test_dfs)} test stations from {config['data']['path']} "
                f"(test window {config['data']['test_start']} .. {config['data']['test_end']})")

    freq_delta = pd.Timedelta(freq)
    # Mehrere Zielgroessen (Solar: ghi + dhi) stehen in data.target_cols; data.target_col
    # ist dort None. get_target_cols bevorzugt target_cols und liefert fuer Wind
    # unveraendert ['wind_speed'], der Single-Target-Pfad bleibt also wie er war.
    target_cols = preprocessing.get_target_cols(config)
    multi_target = len(target_cols) > 1
    target_col = target_cols[0]

    params_cfg = config.get('params', {})
    nwp_residual = str(params_cfg.get('target_transform', 'none')) == 'nwp_residual'
    exclude_imputed = bool(config.get('eval', {}).get('exclude_imputed', False))
    # Versatz zwischen Laufzeit und erstem Lead: ML (wind) laesst forecasttime=0 weg
    # und beginnt bei t_run+1h, SL (solar) labelt linksbuendig auf t_run. Derselbe
    # Parameter, mit dem der GNN-Pfad am 15.09.2026 repariert wurde (handoff §3/§4.1).
    lead0_off = lead0_offset(config['data'].get('use_case', 'wind'))
    logger.info(f"Zielgroessen: {target_cols}; target_transform="
                f"{params_cfg.get('target_transform', 'none')}, exclude_imputed="
                f"{exclude_imputed}, lead0_offset={lead0_off}")

    per_station = []
    raw_records = []
    # Rohwerte je Zielgroesse sammeln (Schluessel = Zielspalte), fuer die gepoolten Zahlen.
    pool = {tgt: {'true': [], 'pred': [], 'nwp': []} for tgt in target_cols}

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

        # Beim Residuum-Ziel darf NICHT bei 0 abgeschnitten werden — rund die Haelfte
        # der Zielwerte ist dort negativ (utils/solar._to_nwp_residual, tools.get_y).
        y_true, y_pred = tools.get_y(X_test=X_test, y_test=y_test, model=model,
                                      scaler_y=scaler_y, device=device,
                                      clip_negative=not nwp_residual)
        run_times = prepared.get('index_test')
        if run_times is None or len(run_times) != len(y_true):
            logger.warning(f"Station {station_id}: index_test fehlt oder passt nicht zu "
                           f"y_test — Station uebersprungen.")
            continue
        horizon_len = y_true.shape[1]

        for j, tgt in enumerate(target_cols):
            y_t = y_true[:, :, j] if y_true.ndim == 3 else y_true
            y_p = y_pred[:, :, j] if y_pred.ndim == 3 else y_pred

            # --- NWP-Baseline und Rueckrechnung in physikalische Einheiten ---
            # Im Residuumsraum ist die Zielspalte 'Messung - NWP'; die Umkehrung
            # braucht genau die Spalte, die abgezogen wurde (resolve_residual_
            # baseline_col liefert sie, inklusive params.nwp_baseline_col und der
            # _1-Vorzugsregel). RMSE und MAE sind in beiden Raeumen identisch,
            # R2 und die Rohwerte im Parquet sind es nicht — letztere muessen
            # physikalisch sein, damit sie neben den DCRNN-Parquets stehen koennen.
            basis_col = (resolve_residual_baseline_col(df.columns, tgt, params_cfg)
                         if nwp_residual else None)
            if nwp_residual:
                nwp = _spalte_je_lauf(df, basis_col, run_times, horizon_len)
                if nwp is None:
                    raise RuntimeError(
                        f"Station {station_id}, Ziel '{tgt}': target_transform="
                        f"'nwp_residual', aber die Basisspalte "
                        f"{basis_col!r} laesst sich nicht laufweise ausrichten. "
                        f"Ohne sie liesse sich weder die physikalische Skala "
                        f"rekonstruieren noch Skill_NWP bilden — Abbruch statt "
                        f"stillschweigend falscher Zahlen.")
                gt_abs = y_t + nwp
                pred_abs = y_p + nwp
                # Im Residuumsraum IST die rohe NWP-Prognose die Nullreihe (eval.py:607).
                nwp_err = np.zeros_like(y_t) - y_t
            else:
                nwp_raw = prepared.get('nwp_raw_test') if j == 0 else None
                if nwp_raw is None or len(nwp_raw) != len(y_t):
                    nwp_raw = _spalte_je_lauf(
                        df, preprocessing.nwp_baseline_prefixes(
                            params_cfg.get('nwp_baseline_col'), 'wind_speed_h10')[0],
                        run_times, horizon_len)
                nwp = nwp_raw
                gt_abs, pred_abs = y_t, y_p
                nwp_err = None if nwp is None else (nwp - y_t)

            # --- Maske: imputierte Zielpositionen und Fehlwerte heraus ---
            # '<target>_observed' legt utils/solar.preprocess_solar_icond2 an
            # (True = echter Messwert). Elementweise, nicht laufweise — dieselbe
            # Regel wie eval._evaluate_single_target.
            maske = np.isfinite(y_t) & np.isfinite(y_p)
            if exclude_imputed:
                beobachtet = _spalte_je_lauf(df, f'{tgt}_observed', run_times, horizon_len)
                if beobachtet is None:
                    raise RuntimeError(
                        f"Station {station_id}, Ziel '{tgt}': eval.exclude_imputed ist "
                        f"gesetzt, aber '{tgt}_observed' laesst sich nicht laufweise "
                        f"ausrichten. Ungefiltert weiterzurechnen waere die falsche Zahl.")
                maske &= beobachtet > 0.5
            if maske.sum() < 2:
                logger.warning(f"Station {station_id}, Ziel '{tgt}': nach Maske nur "
                               f"{int(maske.sum())} Werte — uebersprungen.")
                continue

            rmse = float(np.sqrt(np.mean((y_p[maske] - y_t[maske]) ** 2)))
            mae = float(np.mean(np.abs(y_p[maske] - y_t[maske])))
            r2 = float(r2_score(gt_abs[maske], pred_abs[maske]))
            rmse_nwp = (float(np.sqrt(np.mean(nwp_err[maske] ** 2)))
                        if nwp_err is not None else None)

            # --- Persistenz: letzter Messwert vor Prognosestart ---
            pers_vals = _persistenz(df, tgt, basis_col, run_times, freq_delta,
                                    lead0_off, nwp_residual)
            pers_ref = None
            skill = None
            if pers_vals is not None:
                pers_ref = np.repeat(pers_vals[:, None], horizon_len, axis=1)
                gueltig = maske & np.isfinite(pers_ref)
                if gueltig.sum() >= 2:
                    rmse_pers = float(math.sqrt(mean_squared_error(
                        gt_abs[gueltig], pers_ref[gueltig])))
                    skill = (1.0 - rmse / rmse_pers) if rmse_pers > 0 else None

            for i in range(len(run_times)):
                run_ts = run_times[i]
                for h in range(horizon_len):
                    if not maske[i, h]:
                        continue
                    satz = {
                        'station_id': station_id,
                        'run_time':   run_ts,
                        # Lead 0 haengt an run_time + lead0_off Schritten, horizon zaehlt ab 1.
                        'valid_time': run_ts + freq_delta * (h + lead0_off),
                        'horizon':    h + 1,
                        'pred':       float(pred_abs[i, h]),
                        'gt':         float(gt_abs[i, h]),
                        'nwp_ref':    float(nwp[i, h]) if nwp is not None else np.nan,
                        'pers_ref':   float(pers_ref[i, h]) if pers_ref is not None else np.nan,
                    }
                    if multi_target:
                        satz['target'] = tgt
                    raw_records.append(satz)

            eintrag = {
                'station_id': station_id,
                # n_samples zaehlt wie bisher die Vorhersagefenster (Laeufe mit
                # mindestens einer bewerteten Zelle), n_values die tatsaechlich
                # bewerteten Einzelwerte — letzteres ist das, was
                # get_test_results_dcrnn.py 'n_samples' nennt.
                'n_samples': int(maske.any(axis=1).sum()),
                'n_values': int(maske.sum()),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'rmse_nwp': rmse_nwp,
                'skill_nwp': (1 - rmse / rmse_nwp) if rmse_nwp else None,
                'skill': skill,
            }
            if multi_target:
                eintrag['target'] = tgt
            per_station.append(eintrag)

            pool[tgt]['true'].append(gt_abs[maske])
            pool[tgt]['pred'].append(pred_abs[maske])
            if nwp_err is not None:
                pool[tgt]['nwp'].append(nwp_err[maske])

            einheit = EINHEIT.get(tgt, '')
            logger.info(f"Station {station_id} [{tgt}]: Laeufe={int(maske.any(axis=1).sum())}, "
                        f"Werte={int(maske.sum())}, "
                        f"RMSE={rmse:.4f} {einheit}, R2={r2:.4f}"
                        + (f", RMSE_NWP={rmse_nwp:.4f}, Skill_NWP={1 - rmse / rmse_nwp:.4f}"
                           if rmse_nwp else "")
                        + (f", Skill={skill:.4f}" if skill is not None else ""))

    if not per_station:
        raise RuntimeError("No test stations produced samples — check test_start/test_end vs. data coverage.")

    # Gepoolte Zahlen je Zielgroesse. Ueber ghi und dhi gemittelt beschriebe eine
    # einzelne Zahl keine der beiden Groessen — dieselbe Trennung wie in
    # geostatistics/get_test_results_dcrnn.py.
    je_ziel = {}
    for tgt in target_cols:
        if not pool[tgt]['true']:
            continue
        y_true_all = np.concatenate(pool[tgt]['true'])
        y_pred_all = np.concatenate(pool[tgt]['pred'])
        zeilen = [r for r in per_station if not multi_target or r.get('target') == tgt]
        block = {
            'n_stations': len(zeilen),
            'pooled_rmse': float(np.sqrt(np.mean((y_pred_all - y_true_all) ** 2))),
            'pooled_mae': float(np.mean(np.abs(y_pred_all - y_true_all))),
            'mean_station_rmse': float(np.mean([r['rmse'] for r in zeilen])),
        }
        if pool[tgt]['nwp']:
            nwp_err_all = np.concatenate(pool[tgt]['nwp'])
            block['pooled_rmse_nwp'] = float(np.sqrt(np.mean(nwp_err_all ** 2)))
            block['pooled_skill_nwp'] = 1 - block['pooled_rmse'] / block['pooled_rmse_nwp']
        je_ziel[tgt] = block

    result = {
        'model_tag': args.model_tag,
        'config_path': f'{args.config}.yaml',
        'eval_split': args.eval_split,
        'hpo_study': args.hpo_study,
        'target_cols': target_cols,
        'target_transform': params_cfg.get('target_transform', 'none'),
        'exclude_imputed': exclude_imputed,
        'lead0_offset': lead0_off,
        'test_start': str(config['data']['test_start']),
        'test_end': str(config['data']['test_end']),
        'per_target': je_ziel,
        'per_station': per_station,
    }
    # Single-Target behaelt die bisherigen Schluessel auf oberster Ebene, damit
    # bestehende Wind-Auswertungen unveraendert weiterlesen koennen.
    if not multi_target:
        result.update({k: v for k, v in je_ziel.get(target_cols[0], {}).items()})

    for tgt, block in je_ziel.items():
        einheit = EINHEIT.get(tgt, '')
        logger.info(f"[{tgt}] Pooled RMSE ({einheit}): {block['pooled_rmse']:.4f} "
                    f"(mean-of-station: {block['mean_station_rmse']:.4f}) ueber "
                    f"{block['n_stations']} Stationen"
                    + (f", Skill_NWP: {block['pooled_skill_nwp']:.4f}"
                       if 'pooled_skill_nwp' in block else ""))

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
