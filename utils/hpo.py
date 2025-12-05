import ray
import time
import optuna
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.model_selection import TimeSeriesSplit

from . import tools, models


def apply_min_train_len_per_file(prepared_datasets: List[Dict[str, Any]],
                                min_train_date: str = None) -> Dict[str, Any]:
    """
    Wendet min_train_date auf jede einzelne Datei an, bevor sie kombiniert werden.

    Args:
        prepared_datasets: Liste von prepared_data Dictionaries von jeder Datei
        min_train_date: End-Datum für minimum training block (z.B. '2024-07-31')
                       Alle Daten mit starttime <= diesem Datum werden als min_block verwendet

    Returns:
        Dictionary mit 'min_blocks' (kombinierte minimum blocks) und 'remaining_datasets'
        (verbleibende Daten pro Datei nach Abzug der min blocks)
    """
    if min_train_date is None:
        # Wenn kein min_train_date, gib alle Daten als "remaining" zurück
        return {
            'min_blocks': None,
            'remaining_datasets': prepared_datasets
        }

    min_blocks = []
    remaining_datasets = []

    for i, prepared_data in enumerate(prepared_datasets):
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        index_train = prepared_data.get('index_train')

        if index_train is None:
            raise ValueError(f"File {i}: 'index_train' not found in prepared_data. "
                           "Cannot use min_train_date without index information.")

        # Bestimme min_block_end basierend auf min_train_date
        # Ensure min_train_date is a string first (YAML might load it as datetime object or number)
        if not isinstance(min_train_date, str):
            min_train_date = str(min_train_date)

        min_date = pd.to_datetime(min_train_date, errors='coerce')

        if pd.isna(min_date):
            raise ValueError(f"File {i}: Could not parse min_train_date '{min_train_date}' as a valid date")

        # Handle timezone
        if min_date.tzinfo is None:
            min_date = min_date.tz_localize('UTC')
        else:
            min_date = min_date.tz_convert('UTC')

        if isinstance(index_train, pd.MultiIndex):
            # MultiIndex: Filtere auf 'starttime' level
            mask = index_train.get_level_values('starttime') <= min_date
            min_block_end = mask.sum()
            logging.debug(f"File {i}: MultiIndex detected, filtering on 'starttime' <= {min_date}")
        else:
            # Regular DatetimeIndex: Filtere direkt
            mask = index_train <= min_date
            min_block_end = mask.sum()
            logging.debug(f"File {i}: Regular DatetimeIndex detected, filtering on index <= {min_date}")

        if min_block_end == 0:
            logging.warning(f"File {i}: No data found with dates <= {min_train_date}. "
                          f"Skipping this file for min_blocks.")
            # Erstelle leeres min_block
            if isinstance(X_train, dict):
                X_min_block = {key: np.array([]).reshape(0, *value.shape[1:]) for key, value in X_train.items() if len(value) > 0}
            else:
                X_min_block = np.array([]).reshape(0, *X_train.shape[1:])
            y_min_block = np.array([])

            min_blocks.append({
                'X_train': X_min_block,
                'y_train': y_min_block,
                'X_test': prepared_data.get('X_test', None),
                'y_test': prepared_data.get('y_test', None)
            })
            remaining_datasets.append(prepared_data)
            continue

        if len(y_train) <= min_block_end:
            logging.debug(f"File {i}: All data ({len(y_train)} samples) is within min_train_date ({min_train_date}). "
                        f"Using all data as minimum block.")
            # Wenn die ganze Datei innerhalb des min_train_date liegt, verwende alles als min_block
            min_blocks.append(prepared_data)
            # Erstelle leeres "remaining" dataset
            if isinstance(X_train, dict):
                X_remaining = {key: np.array([]).reshape(0, *value.shape[1:]) for key, value in X_train.items() if len(value) > 0}
            else:
                X_remaining = np.array([]).reshape(0, *X_train.shape[1:])
            y_remaining = np.array([])

            remaining_datasets.append({
                'X_train': X_remaining,
                'y_train': y_remaining,
                'X_test': prepared_data.get('X_test', None),
                'y_test': prepared_data.get('y_test', None)
            })
        else:
            # Teile die Datei in minimum block + remaining
            # Minimum block
            if isinstance(X_train, dict):
                X_min_block = {key: value[:min_block_end] for key, value in X_train.items() if len(value) > 0}
                X_remaining = {key: value[min_block_end:] for key, value in X_train.items() if len(value) > 0}
            else:
                X_min_block = X_train[:min_block_end]
                X_remaining = X_train[min_block_end:]

            y_min_block = y_train[:min_block_end]
            y_remaining = y_train[min_block_end:]

            min_blocks.append({
                'X_train': X_min_block,
                'y_train': y_min_block,
                'X_test': prepared_data.get('X_test', None),
                'y_test': prepared_data.get('y_test', None)
            })

            remaining_datasets.append({
                'X_train': X_remaining,
                'y_train': y_remaining,
                'X_test': prepared_data.get('X_test', None),
                'y_test': prepared_data.get('y_test', None)
            })

            logging.debug(f"File {i}: Split into min_block ({len(y_min_block)} samples up to {min_train_date}) + "
                        f"remaining ({len(y_remaining)} samples)")

    return {
        'min_blocks': min_blocks,
        'remaining_datasets': remaining_datasets
    }


def kfolds_with_per_file_min_train_len(prepared_datasets: List[Dict[str, Any]],
                                      n_splits: int,
                                      val_split: float = None,
                                      min_train_date: str = None) -> List:
    """
    Erstellt k-folds mit per-file minimum training date.

    Args:
        prepared_datasets: Liste von prepared_data Dictionaries von jeder Datei
        n_splits: Anzahl der Splits
        val_split: Validation split ratio (nur für n_splits=1)
        min_train_date: End-Datum für minimum training block (z.B. '2024-07-31')

    Returns:
        Liste von ((X_train_combined, y_train_combined), (X_val_combined, y_val_combined)) Tupeln
    """
    if n_splits == 1:
        # Für n_splits=1, kombiniere einfach alle Daten und verwende val_split
        X_train_all = None
        y_train_all = None

        for prepared_data in prepared_datasets:
            if X_train_all is None:
                X_train_all = prepared_data['X_train']
                y_train_all = prepared_data['y_train']
            else:
                X_train_all = tools.concatenate_data(old=X_train_all, new=prepared_data['X_train'])
                y_train_all = np.concatenate((y_train_all, prepared_data['y_train']))

        X_train, y_train, X_val, y_val = split_val(X=X_train_all, y=y_train_all, val_split=val_split)
        return [((X_train, y_train), (X_val, y_val))]

    # Wende min_train_date pro Datei an
    split_result = apply_min_train_len_per_file(prepared_datasets, min_train_date)
    min_blocks = split_result['min_blocks']
    remaining_datasets = split_result['remaining_datasets']

    if min_blocks is None:
        # Kein min_train_date spezifiziert, verwende normale Kombination + k-folds
        X_train_all = None
        y_train_all = None

        for prepared_data in prepared_datasets:
            if X_train_all is None:
                X_train_all = prepared_data['X_train']
                y_train_all = prepared_data['y_train']
            else:
                X_train_all = tools.concatenate_data(old=X_train_all, new=prepared_data['X_train'])
                y_train_all = np.concatenate((y_train_all, prepared_data['y_train']))

        return kfolds(X=X_train_all, y=y_train_all, n_splits=n_splits, val_split=val_split)

    # Kombiniere die minimum blocks effizienter
    X_min_combined = None
    y_min_combined = None

    # Sammle erst alle nicht-leeren Blöcke
    valid_min_blocks = [block for block in min_blocks if len(block['y_train']) > 0]

    if valid_min_blocks:
        # Extrahiere alle y_train Arrays und konkateniere sie auf einmal
        y_parts = [block['y_train'] for block in valid_min_blocks]
        y_min_combined = np.concatenate(y_parts, axis=0)

        # Für X_train: Prüfe den Typ des ersten gültigen Blocks
        first_block = valid_min_blocks[0]
        if isinstance(first_block['X_train'], dict):
            # TFT case: Konkateniere jedes Feature separat
            X_min_combined = {}
            for key in first_block['X_train'].keys():
                X_parts = [block['X_train'][key] for block in valid_min_blocks
                          if key in block['X_train'] and len(block['X_train'][key]) > 0]
                if X_parts:
                    X_min_combined[key] = np.concatenate(X_parts, axis=0)
        else:
            # Standard numpy array case
            X_parts = [block['X_train'] for block in valid_min_blocks]
            X_min_combined = np.concatenate(X_parts, axis=0)

    # Erstelle k-folds für die remaining datasets - optimiert
    remaining_kfolds = []
    for prepared_data in remaining_datasets:
        if len(prepared_data['y_train']) > 0:  # Nur wenn noch Daten vorhanden sind
            fold_data = kfolds(X=prepared_data['X_train'], y=prepared_data['y_train'],
                             n_splits=n_splits, val_split=val_split)
            remaining_kfolds.append(fold_data)

    # Kombiniere minimum block mit jedem fold - optimiert für bessere Performance
    combined_kfolds = []
    for fold_idx in range(n_splits):
        # Sammle alle fold-Teile erst in Listen, dann batch-konkateniere
        train_X_parts = []
        train_y_parts = []
        val_X_parts = []
        val_y_parts = []

        # Füge min_block hinzu (falls vorhanden)
        if X_min_combined is not None:
            train_X_parts.append(X_min_combined)
            train_y_parts.append(y_min_combined)

        # Sammle fold-spezifische Teile von allen Dateien
        for file_kfolds in remaining_kfolds:
            if fold_idx < len(file_kfolds):
                (X_train_fold, y_train_fold), (X_val_fold, y_val_fold) = file_kfolds[fold_idx]

                # Sammle Training-Teile
                if len(y_train_fold) > 0:
                    train_X_parts.append(X_train_fold)
                    train_y_parts.append(y_train_fold)

                # Sammle Validation-Teile
                if len(y_val_fold) > 0:
                    val_X_parts.append(X_val_fold)
                    val_y_parts.append(y_val_fold)

        # Batch-Konkatenation für Training
        if train_y_parts:
            y_fold_combined = np.concatenate(train_y_parts, axis=0)
            if isinstance(train_X_parts[0], dict):
                # TFT case
                X_fold_combined = {}
                for key in train_X_parts[0].keys():
                    X_key_parts = [X_part[key] for X_part in train_X_parts
                                  if key in X_part and len(X_part[key]) > 0]
                    if X_key_parts:
                        X_fold_combined[key] = np.concatenate(X_key_parts, axis=0)
            else:
                # Standard numpy case
                X_fold_combined = np.concatenate(train_X_parts, axis=0)
        else:
            X_fold_combined, y_fold_combined = None, None

        # Batch-Konkatenation für Validation
        if val_y_parts:
            y_val_combined = np.concatenate(val_y_parts, axis=0)
            if isinstance(val_X_parts[0], dict):
                # TFT case
                X_val_combined = {}
                for key in val_X_parts[0].keys():
                    X_key_parts = [X_part[key] for X_part in val_X_parts
                                  if key in X_part and len(X_part[key]) > 0]
                    if X_key_parts:
                        X_val_combined[key] = np.concatenate(X_key_parts, axis=0)
            else:
                # Standard numpy case
                X_val_combined = np.concatenate(val_X_parts, axis=0)
        else:
            X_val_combined, y_val_combined = None, None

        combined_kfolds.append(((X_fold_combined, y_fold_combined), (X_val_combined, y_val_combined)))

        # Logging
        if X_min_combined is not None:
            min_samples = len(y_min_combined)
            fold_samples = len(y_fold_combined) - min_samples
            val_samples = len(y_val_combined) if y_val_combined is not None else 0
            logging.debug(f"Combined Fold {fold_idx+1}: {len(y_fold_combined)} training total (min_blocks: {min_samples}, fold_parts: {fold_samples}), {val_samples} validation")

    return combined_kfolds



def get_objectives_from_config(config: dict) -> tuple:
    """
    Extract optimization objectives from config.

    Supports both single-objective (backward compatible) and multi-objective optimization.

    Args:
        config: Configuration dictionary

    Returns:
        tuple: (objectives_list, is_multi_objective)

        objectives_list format:
        - Single: [{'metric': 'val_loss', 'direction': 'minimize'}]
        - Multi: [{'metric': 'val_r2', 'direction': 'maximize'},
                  {'metric': 'val_rmse', 'direction': 'minimize'}]
        is_multi_objective: bool indicating if multi-objective mode

    Examples:
        # Single objective (backward compatible)
        config = {'hpo': {'metric': 'val_loss'}}
        >> [{'metric': 'val_loss', 'direction': 'minimize'}], False

        # Multi-objective
        config = {'hpo': {'objectives': [
            {'metric': 'val_r2', 'direction': 'maximize'},
            {'metric': 'val_rmse', 'direction': 'minimize'}
        ]}}
        >> [...], True
    """
    hpo_config = config.get('hpo', {})

    # Check for multi-objective format (new)
    if 'objectives' in hpo_config:
        objectives = hpo_config['objectives']
        if not isinstance(objectives, list) or len(objectives) == 0:
            raise ValueError("'objectives' must be a non-empty list")

        # Validate each objective
        for obj in objectives:
            if 'metric' not in obj or 'direction' not in obj:
                raise ValueError("Each objective must have 'metric' and 'direction' keys")
            if obj['direction'] not in ['maximize', 'minimize']:
                raise ValueError(f"Direction must be 'maximize' or 'minimize', got: {obj['direction']}")

        is_multi = len(objectives) > 1
        return objectives, is_multi

    # Single objective format (backward compatible)
    elif 'metric' in hpo_config:
        metric = hpo_config['metric']

        # Infer direction from metric name
        if 'loss' in metric.lower() or 'mse' in metric.lower() or 'mae' in metric.lower() or 'rmse' in metric.lower():
            direction = 'minimize'
        elif 'r2' in metric.lower() or 'accuracy' in metric.lower():
            direction = 'maximize'
        else:
            # Default to minimize
            direction = 'minimize'
            logging.warning(f"Could not infer direction for metric '{metric}', using 'minimize'")

        objectives = [{'metric': metric, 'direction': direction}]
        return objectives, False

    else:
        raise ValueError("Config must contain either 'hpo.metric' (single) or 'hpo.objectives' (multi)")


def create_or_load_study(path, study_name, direction=None, pruning_config=None, config=None):
    """
    Create or load an Optuna study with flexible single/multi-objective support.

    Args:
        path: Path to study database
        study_name: Name of the study
        direction: (Deprecated) Single objective direction - use config instead
        pruning_config: Pruning configuration
        config: Full config dict (required for multi-objective)

    Returns:
        optuna.Study object
    """
    storage = f'sqlite:///{path}'

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        n_ei_candidates=24,
        gamma=lambda x: min(int(0.25 * x), 30),
        multivariate=True,
        group=True,
        warn_independent_sampling=False
    )

    pruner = None
    if pruning_config and pruning_config.get('use_median_pruner', False):
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruning_config.get('n_startup_trials', 10),
            n_warmup_steps=pruning_config.get('n_warmup_steps', 1),
            interval_steps=pruning_config.get('interval_steps', 1)
        )
        logging.debug(f"Using MedianPruner with n_startup_trials={pruner._n_startup_trials}, "
                     f"n_warmup_steps={pruner._n_warmup_steps}, interval_steps={pruner._interval_steps}")
    elif pruning_config and pruning_config.get('disable_pruning', False):
        pruner = optuna.pruners.NopPruner()
        logging.debug("Using NopPruner (no pruning)")

    # Determine if multi-objective from config
    if config is not None:
        objectives, is_multi = get_objectives_from_config(config)

        if is_multi:
            # Multi-objective mode
            directions = [obj['direction'] for obj in objectives]
            objective_names = [obj['metric'] for obj in objectives]
            logging.info(f"Multi-Objective HPO: {len(objectives)} objectives - {objective_names}")
        else:
            # Single objective mode (from config)
            directions = objectives[0]['direction']
            logging.info(f"Single-Objective HPO: {objectives[0]['metric']} ({directions})")
    else:
        # Backward compatibility: use direction parameter
        if direction is None:
            raise ValueError("Either 'config' or 'direction' must be provided")
        directions = direction
        is_multi = False
        logging.warning("Using deprecated 'direction' parameter. Please provide 'config' instead.")

    try:
        existing_study = optuna.load_study(study_name=study_name, storage=storage)
        completed_trials = len([t for t in existing_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logging.info(f"Loaded existing study '{study_name}' with {completed_trials} completed trials.")
        if pruner is not None and existing_study:
            logging.debug(f"Overriding pruner for existing study '{study_name}' (session-only)")
            existing_study.pruner = pruner
        study = existing_study
    except:
        logging.info(f"Creating new study '{study_name}'.")

        if is_multi:
            # Create multi-objective study
            study = optuna.create_study(
                storage=storage,
                study_name=study_name,
                directions=directions,  # List of directions
                load_if_exists=False,
                sampler=sampler,
                pruner=pruner
            )
        else:
            # Create single-objective study
            study = optuna.create_study(
                storage=storage,
                study_name=study_name,
                direction=directions,  # Single direction
                load_if_exists=False,
                sampler=sampler,
                pruner=pruner
            )

    return study

def load_study(studies_path: str,
               study_name: str):
    storage = f'sqlite:///{studies_path}'
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
    except:
        #os.remove(f'{path}.db')
        study = None
    return study

def split_val(X: Any,
              y: np.ndarray,
              val_split):
    if val_split == 0:
        return X, y, None, None
    val_index = int(len(y)*(1-val_split))
    # case for tft
    if type(X) == dict:
        X_train, X_val = {}, {}
        for key, value in X.items():
            if len(value) == 0:
                continue
            X_train[key] = value[:val_index]
            X_val[key] = value[val_index:]
    else:
        X_train = X[:val_index]
        X_val = X[val_index:]
    y_train = y[:val_index]
    y_val = y[val_index:]
    return X_train, y_train, X_val, y_val


def kfolds(X: Any,
           y: np.ndarray,
           n_splits: int,
           val_split: float = None) -> List:
    """
    Erstellt k-folds für Zeitserien.

    Args:
        X: Features (numpy array oder dict für TFT)
        y: Target values
        n_splits: Anzahl der Splits
        val_split: Validation split ratio (nur für n_splits=1)

    Returns:
        Liste von ((X_train, y_train), (X_val, y_val)) Tupeln
    """
    kfolds = []
    if n_splits == 1: # if not kfolds
        X_train, y_train, X_val, y_val = split_val(X=X, y=y, val_split=val_split)
        kfolds.append(((X_train, y_train), (X_val, y_val)))
        return kfolds

    # Standard TimeSeriesSplit ohne minimum training length
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, val_index in tscv.split(y):
        if type(X) == dict:
            X_train, X_val = {}, {}
            for key, value in X.items():
                if len(value) == 0:
                    continue
                X_train[key] = value[train_index]
                X_val[key] = value[val_index]
        else:
            X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        kfolds.append(((X_train, y_train), (X_val, y_val)))
    return kfolds

def concatenate_processed_data(data_parts: List) -> Any:
    """Konkateniert eine Liste von verarbeiteten Daten (numpy arrays oder dicts)."""
    # Filtere None-Werte heraus, die durch Fehler/leere Splits entstanden sein könnten
    valid_parts = [p for p in data_parts if p is not None]
    if not valid_parts:
        return None
    first_item = valid_parts[0]
    if isinstance(first_item, np.ndarray):
        # Prüfe auf leere Arrays, bevor konkateniert wird
        valid_parts = [p for p in valid_parts if p.shape[0] > 0]
        return np.concatenate(valid_parts, axis=0)
    elif isinstance(first_item, dict):
         # Stelle sicher, dass alle Teile Dictionaries sind
         if not all(isinstance(p, dict) for p in valid_parts):
             raise TypeError("Mische Typen beim Konkatenieren von Dictionaries")
         combined_dict = {}
         # Nehme an, alle dicts haben dieselben Keys wie das erste
         keys = first_item.keys()
         for key in keys:
             arrays_to_concat = []
             for d in valid_parts:
                arrays_to_concat.append(d[key])
             if arrays_to_concat: # Nur konkatenieren, wenn es etwas gibt
                 # Sicherstellen, dass alle Teile für diesen Key numpy arrays sind
                 if not all(isinstance(arr, np.ndarray) for arr in arrays_to_concat):
                     raise TypeError(f"Nicht-Numpy-Array gefunden für Key '{key}' beim Konkatenieren innerhalb des Dictionaries.")
                 combined_dict[key] = np.concatenate(arrays_to_concat, axis=0)
             else:
                 # Fallback: Leeres Array oder None, je nach Anforderung
                 # Hier verwenden wir None, um anzuzeigen, dass keine Daten für diesen Key vorhanden waren
                 combined_dict[key] = None
         return combined_dict

def combine_kfolds(n_splits: int,
                   kfolds_per_series: list):
    combined_kfolds = []
    for i in range(n_splits):
        xtr_parts = []
        ytr_parts = []
        xval_parts = []
        yval_parts = []
        for series_folds in kfolds_per_series:
            (xtr, ytr), (xval, yval) = series_folds[i]
            xtr_parts.append(xtr)
            ytr_parts.append(ytr)
            xval_parts.append(xval)
            yval_parts.append(yval)
        y_train_combined = np.concatenate([p for p in ytr_parts if p is not None and p.shape[0]>0], axis=0)
        y_val_combined = np.concatenate([p for p in yval_parts if p is not None and p.shape[0]>0], axis=0)
        X_train_combined = concatenate_processed_data(xtr_parts)
        X_val_combined = concatenate_processed_data(xval_parts)
        combined_kfolds.append(((X_train_combined, y_train_combined), (X_val_combined, y_val_combined)))
    return combined_kfolds


def get_hyperparameters(config: dict,
                        hpo=False,
                        trial=None,
                        study=None) -> dict:
    hyperparameters = {}
    # general hyperparameters
    hyperparameters['shuffle'] = config['model']['shuffle']
    model_name = config['model']['name']
    batch_size = config['hpo']['batch_size']
    epochs = config['hpo']['epochs']
    n_layers = config['hpo']['n_layers']
    # cnn / rnn / fnn specific hyperparameters
    n_cnn_layers = config['hpo']['n_cnn_layers']
    n_rnn_layers = config['hpo']['n_rnn_layers']
    learning_rate = config['hpo']['learning_rate']
    attention_heads = config['hpo']['attention_heads']
    filters = config['hpo']['cnn']['filters']
    kernel_size = config['hpo']['cnn']['kernel_size']
    dropout_rnn = config['hpo']['rnn']['dropout']
    rnn_units = config['hpo']['rnn']['units']
    fnn_units = config['hpo']['fnn']['units']
    increase_filters = config['hpo']['cnn']['increase_filters']
    # tft specific hyperparameters
    n_heads = config['hpo']['tft']['n_heads']
    hidden_dim = config['hpo']['tft']['hidden_dim']
    dropout_tft = config['hpo']['tft']['dropout']
    n_lstm_layers = config['hpo']['tft'].get('n_lstm_layers', [1, 3])
    static_embedding_dim = config['hpo']['tft'].get('static_embedding_dim')
    num_quantiles = config['model']['tft'].get('num_quantiles')
    clipnorm = config['hpo']['tft'].get('clipnorm')
    lookback = config['model']['lookback']#config['hpo']['tft']['lookback']
    horizon = config['model']['horizon']
    # fl specific hyperparameters
    n_rounds = config['hpo']['fl']['n_rounds']
    strategy = config['hpo']['fl']['strategy']
    server_lr = config['hpo']['fl']['server_lr']
    beta_1 = config['hpo']['fl']['beta_1']
    beta_2 = config['hpo']['fl']['beta_2']
    tau = config['hpo']['fl']['tau']
    # boolean variables
    is_cnn_type = 'cnn' in model_name or 'tcn' in model_name
    is_rnn_type = 'lstm' in model_name or 'gru' in model_name
    is_fnn_type = model_name == 'fnn'
    is_tft_type = model_name == 'tft'
    is_gnn_type = 'gnn' in model_name

    if hpo:
        hyperparameters['batch_size'] = trial.suggest_int('batch_size', batch_size[0], batch_size[1])
        #hyperparameters['epochs'] = trial.suggest_int('epochs', epochs[0], epochs[1])
        hyperparameters['lr'] = trial.suggest_float('lr', learning_rate[0], learning_rate[1], log=True)
        if config['model']['fl']:
            hyperparameters['n_rounds'] = trial.suggest_int('n_rounds', n_rounds[0], n_rounds[1])
            hyperparameters['strategy'] = strategy
            if strategy in ['fedavgm', 'fedadam', 'fedyogi']:
                hyperparameters['server_lr'] = trial.suggest_float('server_lr', server_lr[0], server_lr[1], log=True)
                hyperparameters['beta_1'] = trial.suggest_float('beta_1', beta_1[0], beta_1[1])
            if strategy == 'fedadam':
                hyperparameters['beta_2'] = trial.suggest_float('beta_2', beta_2[0], beta_2[1])
                hyperparameters['tau'] = trial.suggest_float('tau', tau[0], tau[1], log=True)
        if is_cnn_type:
            hyperparameters['filters'] = trial.suggest_int('filters', filters[0], filters[1])
            hyperparameters['kernel_size'] = trial.suggest_int('kernel_size', kernel_size[0], kernel_size[1])
            hyperparameters['n_cnn_layers'] = trial.suggest_int('n_cnn_layers', n_cnn_layers[0], n_cnn_layers[1])
            hyperparameters['increase_filters'] = increase_filters
        if is_rnn_type or is_gnn_type:
            hyperparameters['dropout'] = trial.suggest_float('dropout', dropout_rnn[0], dropout_rnn[1])
            hyperparameters['units'] = trial.suggest_int('units', rnn_units[0], rnn_units[1])
            hyperparameters['n_rnn_layers'] = trial.suggest_int('n_rnn_layers', n_rnn_layers[0], n_rnn_layers[1])
        if is_fnn_type:
            hyperparameters['n_layers'] = trial.suggest_int('n_layers', n_layers[0], n_layers[1])
            hyperparameters['units'] = trial.suggest_int('units', fnn_units[0], fnn_units[1])
        if is_tft_type:
            hyperparameters['horizon'] = horizon
            hyperparameters['lookback'] = lookback#trial.suggest_categorical('lookback', lookback)
            n_heads_hpo = trial.suggest_int('n_heads', n_heads[0], n_heads[1])
            hyperparameters['n_heads'] = n_heads_hpo
            minimum_head_dim = 4
            hyperparameters['hidden_dim'] = trial.suggest_int(
                'hidden_dim',
                n_heads_hpo * minimum_head_dim, # minimum hidden dim per attention head
                hidden_dim[1],
                step=n_heads_hpo
            )
            hyperparameters['dropout'] = trial.suggest_float('dropout', dropout_tft[0], dropout_tft[1])
            hyperparameters['num_lstm_layers'] = trial.suggest_int('num_lstm_layers', n_lstm_layers[0], n_lstm_layers[1])

            # Static embedding dimension (can be None for auto-sizing)
            if static_embedding_dim is not None and isinstance(static_embedding_dim, list):
                hyperparameters['static_embedding_dim'] = trial.suggest_int('static_embedding_dim', static_embedding_dim[0], static_embedding_dim[1])
            else:
                hyperparameters['static_embedding_dim'] = static_embedding_dim

            # Clipnorm: can be None, a scalar, or a range for HPO
            if clipnorm is not None and isinstance(clipnorm, list):
                hyperparameters['clipnorm'] = trial.suggest_float('clipnorm', clipnorm[0], clipnorm[1])
            else:
                hyperparameters['clipnorm'] = clipnorm
        if 'attn' in model_name:
            hyperparameters['attention_heads'] = trial.suggest_int('attention_heads', attention_heads[0], attention_heads[1])

        # StemGNN HPO
        if is_gnn_type:
             hyperparameters['n_nodes'] = config['params'].get('next_n_grid_points', 12)
             hyperparameters['units'] = trial.suggest_int('units', rnn_units[0], rnn_units[1])
             hyperparameters['stack_cnt'] = config['model']['stemgnn']['stack_cnt']
             hyperparameters['multi_layer'] = config['model']['stemgnn']['multi_layer']
             hyperparameters['dropout'] = trial.suggest_float('dropout', dropout_rnn[0], dropout_rnn[1])

    else:
        if study and study.best_trial:
            hyperparameters.update(study.best_trial.params)
            hyperparameters['lookback'] = lookback
            hyperparameters['horizon'] = horizon
        else:
            hyperparameters['batch_size'] = config['model']['batch_size']
            #hyperparameters['epochs'] = config['model']['epochs']
            hyperparameters['lr'] = config['model']['lr']
            if 'attn' in model_name:
                hyperparameters['attention_heads'] = config['model']['attention_heads']
            if config['model'].get('fl', False):
                hyperparameters['n_rounds'] = config['fl']['n_rounds']
                hyperparameters['strategy'] = config['fl']['strategy']
                if config['fl']['strategy'].lower() in ['fedavgm', 'fedadam', 'fedyogi']:
                    hyperparameters['server_lr'] = config['fl']['fedopt']['server_lr']
                    hyperparameters['beta_1'] = config['fl']['fedopt']['beta_1']
                if config['fl']['strategy'].lower() in ['fedadam', 'fedyogi']:
                    hyperparameters['beta_2'] = config['fl']['fedopt']['beta_2']
                if config['fl']['strategy'].lower() in ['fedadam']:
                    hyperparameters['tau'] = config['fl']['fedopt']['tau']
            if is_cnn_type:
                hyperparameters['filters'] = config['model']['cnn']['filters']
                hyperparameters['kernel_size'] = config['model']['cnn']['kernel_size']
                hyperparameters['n_cnn_layers'] = config['model']['cnn']['n_cnn_layers']
                hyperparameters['increase_filters'] = config['model']['cnn']['increase_filters']
            if is_rnn_type:
                hyperparameters['dropout'] = config['model']['rnn']['dropout']
                hyperparameters['units'] = config['model']['rnn']['units']
                hyperparameters['n_rnn_layers'] = config['model']['rnn']['n_rnn_layers']
            if is_fnn_type:
                hyperparameters['n_layers'] = config['model']['fnn']['n_layers']
                hyperparameters['units'] = config['model']['fnn']['units']
            if is_tft_type:
                hyperparameters['horizon'] = horizon
                hyperparameters['lookback'] = lookback
                hyperparameters['n_heads'] = config['model']['tft']['n_heads']
                hyperparameters['hidden_dim'] = config['model']['tft']['hidden_dim']
                hyperparameters['dropout'] = config['model']['tft']['dropout']
                hyperparameters['num_lstm_layers'] = config['model']['tft'].get('n_lstm_layers', 1)
                hyperparameters['static_embedding_dim'] = config['model']['tft'].get('static_embedding_dim')
                hyperparameters['num_quantiles'] = config['model']['tft']['num_quantiles']
                hyperparameters['clipnorm'] = config['model']['tft']['clipnorm']
            if is_gnn_type:
                hyperparameters['n_nodes'] = config['params'].get('next_n_grid_points', 12)
                hyperparameters['units'] = config['model']['stemgnn']['units']
                hyperparameters['stack_cnt'] = config['model']['stemgnn']['stack_cnt']
                hyperparameters['multi_layer'] = config['model']['stemgnn']['multi_layer']
                hyperparameters['dropout'] = config['model']['stemgnn']['dropout']

    return hyperparameters

def load_hyperparams(study_name: str,
                     config: dict):
    studies_dir = config['hpo']['studies_dir']
    study = load_study(studies_dir=studies_dir,
                       study_name=study_name)
    if study:
        return study.best_trial.params
    return None