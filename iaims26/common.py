"""Shared helpers for the iAIMS26 experiments.

All scripts in iaims26/ import from here. This module never modifies code
outside iaims26/; it only reads the repo's utils/ modules and artifacts
produced by train_cl.py (checkpoints, scalers, result pickles).
"""

import copy
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
IAIMS_DIR = REPO_ROOT / 'iaims26'
FOLDS_PATH = IAIMS_DIR / 'folds.json'
GENERATED_DIR = IAIMS_DIR / 'configs' / 'generated'
RESULTS_DIR = IAIMS_DIR / 'results'
MODELS_DIR = IAIMS_DIR / 'models'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Seasonal sub-periods inside the single test window (analysis-level split;
# the models are trained once, without retraining between periods).
TEST_START = '2025-08-01'
TEST_END = '2026-01-31'
PERIODS = {
    'P1': ('2025-08-01', '2025-10-31'),  # late summer / early autumn
    'P2': ('2025-11-01', '2026-01-31'),  # winter
    'full': (TEST_START, TEST_END),
}

# Few-shot fine-tuning budgets: days of history before TEST_START.
BUDGETS = {'7d': 7, '30d': 30, '91d': 91, 'full': None}

# Hyperparameters of the published IGEC 100-farm model (fallback/reference).
IGEC_HYPERPARAMETERS = {
    'batch_size': 46,
    'lr': 1.95e-4,
    'n_heads': 8,
    'hidden_dim': 40,
    'dropout': 0.0992,
    'n_lstm_layers': 1,
    'static_embedding_dim': 128,
    'clipnorm': 6.9179,
}

STATIC_FEATURES = ['park_age', 'cut_in', 'cut_out', 'rated_wind_speed',
                   'hub_height', 'rotor_diameter', 'altitude']


# ---------------------------------------------------------------------------
# Host handling (l1 has /mnt/nvme*, l2 mounts them under /mnt/lambda1/nvme*)
# ---------------------------------------------------------------------------

_PATH_PREFIX_MAP = [('/mnt/nvme1', '/mnt/lambda1/nvme1'),
                    ('/mnt/nvme2', '/mnt/lambda1/nvme2')]


def host_path(path: str) -> str:
    """Rewrite an l1 data path for the current host if necessary."""
    if path and not os.path.exists(path):
        for src, dst in _PATH_PREFIX_MAP:
            if path.startswith(src):
                candidate = dst + path[len(src):]
                if os.path.exists(candidate):
                    return candidate
    return path


def localize_config_paths(config: dict) -> dict:
    for key in ('path', 'power_curves_path', 'nwp_path', 'ecmwf_path'):
        if key in config['data'] and isinstance(config['data'][key], str):
            config['data'][key] = host_path(config['data'][key])
    return config


# ---------------------------------------------------------------------------
# Configs / folds
# ---------------------------------------------------------------------------

def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(config: dict, path) -> None:
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None)


def load_base_config() -> dict:
    config = load_yaml(IAIMS_DIR / 'configs' / 'base_iaims26.yaml')
    config['model']['name'] = 'tft'
    return localize_config_paths(config)


def load_generated_config(name: str) -> dict:
    """Load a generated fold config by its config name (without .yaml)."""
    config = load_yaml(GENERATED_DIR / f'{name}.yaml')
    config['model']['name'] = 'tft'
    return localize_config_paths(config)


def load_folds() -> dict:
    with open(FOLDS_PATH) as f:
        folds = json.load(f)
    all_ids = [s for stations in folds['folds'].values() for s in stations]
    assert len(all_ids) == len(set(all_ids)), 'folds are not disjoint'
    return folds


def fold_of_station(folds: dict, station: str) -> int:
    for k, stations in folds['folds'].items():
        if station in stations:
            return int(k)
    raise KeyError(f'station {station} not in any fold')


# ---------------------------------------------------------------------------
# train_cl.py artifact naming (must replicate train_cl.py lines 130-133, 667)
# ---------------------------------------------------------------------------

def config_name_for_fold(fold: int, smoke: bool = False) -> str:
    return f'config_wind_iaims26{"smoke" if smoke else "f"}{fold}'


def study_name_parts(config_name: str, model: str = 'tft',
                     output_dim: int = 48, freq: str = '1h'):
    """Replicates the study/artifact naming of train_cl.py.

    Returns (study_name, study_name_suffix). The final checkpoint is saved as
    models/{study_name}{study_name_suffix}.pt (suffix appears twice — quirk of
    train_cl.py) and the scaler as models/scaler/scaler_{same}.pkl.
    """
    suffix = '_'.join(config_name.split('_')[1:])
    study_name = f'cl_m-{model}_out-{output_dim}_freq-{freq}_{suffix}'
    return study_name, suffix


def checkpoint_path(config_name: str) -> Path:
    # train_cl.py until 2026-07 appended the suffix twice; the current version
    # saves models/{study_name}.pt. Prefer the doubled name if it exists.
    study_name, suffix = study_name_parts(config_name)
    doubled = REPO_ROOT / 'models' / f'{study_name}{suffix}.pt'
    return doubled if doubled.exists() else REPO_ROOT / 'models' / f'{study_name}.pt'


def scaler_path(config_name: str) -> Path:
    study_name, suffix = study_name_parts(config_name)
    doubled = REPO_ROOT / 'models' / 'scaler' / f'scaler_{study_name}{suffix}.pkl'
    return doubled if doubled.exists() else REPO_ROOT / 'models' / 'scaler' / f'scaler_{study_name}.pkl'


def results_pkl_glob(config_name: str, data_path: str) -> list:
    """All result pickles train_cl.py wrote for this config, newest first."""
    study_name, _ = study_name_parts(config_name)
    base_dir = os.path.basename(data_path)
    result_dir = REPO_ROOT / 'results' / base_dir
    hits = sorted(result_dir.glob(f'{study_name}_2*.pkl'), reverse=True)
    return hits


# ---------------------------------------------------------------------------
# Checkpoint / model handling
# ---------------------------------------------------------------------------

def load_state_dict_clean(path) -> dict:
    """Load a state_dict saved by train_cl.py, stripping torch.compile's
    '_orig_mod.' prefix (same handling as get_test_results.py)."""
    sd = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    elif isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def infer_feature_dims(state_dict: dict) -> dict:
    """Infer TFT input dims from embedding-layer counts in a state_dict."""
    def count(prefix):
        idx = set()
        for key in state_dict:
            if key.startswith(prefix + '.'):
                idx.add(int(key.split('.')[1]))
        return len(idx)

    return {
        'observed_dim': count('observed_embed'),
        'known_dim': count('known_embed'),
        'static_dim': count('static_embed'),
    }


def resolve_hyperparameters(config: dict) -> dict:
    """Resolve hyperparameters exactly like train_cl.py with lookup_hpo: False
    (fallback to the values baked into the config)."""
    from utils import hpo
    return hpo.get_hyperparameters(config=config, study=None)


def build_tft(config: dict, hyperparameters: dict, feature_dims: dict):
    from utils import models
    cfg = copy.deepcopy(config)
    cfg['model']['name'] = 'tft'
    cfg['model']['feature_dim'] = feature_dims
    return models.get_model(config=cfg, hyperparameters=hyperparameters)


def load_pretrained_tft(config: dict, ckpt_path, device: str = 'cpu'):
    """Rebuild the TFT from a train_cl.py checkpoint and load its weights."""
    sd = load_state_dict_clean(ckpt_path)
    dims = infer_feature_dims(sd)
    hyperparameters = resolve_hyperparameters(config)
    model = build_tft(config, hyperparameters, dims)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model, hyperparameters, dims


def load_scaler(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def station_key_to_id(key: str) -> str:
    """Map generator/dfs keys like 'synth_15813.csv' to the 5-digit id."""
    return key.split('synth_')[-1].split('.')[0]


def get_station_dfs(config: dict, stations: list) -> dict:
    """Load raw per-station dataframes for an explicit station list."""
    from utils import preprocessing
    cfg = copy.deepcopy(config)
    cfg['model']['name'] = 'tft'
    cfg['data']['files'] = list(stations)
    cfg['data'].pop('val_files', None)
    features = preprocessing.get_features(config=cfg)
    return preprocessing.get_data(
        data_dir=cfg['data']['path'], config=cfg, freq=cfg['data']['freq'],
        features=features, files_key='files')


def make_eval_config(config: dict, test_start: str, test_end: str,
                     train_start: str = None) -> dict:
    """Period-specific config for create_data_generator (deepcopy — the
    generator writes the scaler into the config)."""
    cfg = copy.deepcopy(config)
    cfg['model']['name'] = 'tft'
    cfg['data']['test_start'] = test_start
    cfg['data']['test_end'] = test_end
    if train_start is not None:
        cfg['data']['train_start'] = train_start
    else:
        cfg['data'].pop('train_start', None)
    return cfg


def budget_train_start(budget_key: str) -> str:
    """Map a budget key to the train_start date (None for full history)."""
    days = BUDGETS[budget_key]
    if days is None:
        return None
    ts = pd.Timestamp(TEST_START) - pd.Timedelta(days=days)
    return ts.strftime('%Y-%m-%d %H:%M')


def period_masks(index_test) -> dict:
    """Boolean masks per period over forecast windows (index_test[:, 0] =
    window start timestamps)."""
    import numpy as np
    starts = pd.DatetimeIndex(index_test[:, 0] if getattr(index_test, 'ndim', 1) > 1
                              else index_test)
    if starts.tz is None:
        starts = starts.tz_localize('UTC')
    masks = {}
    for name, (p_start, p_end) in PERIODS.items():
        lo = pd.Timestamp(p_start, tz='UTC')
        hi = pd.Timestamp(p_end, tz='UTC') + pd.Timedelta(hours=23, minutes=59)
        masks[name] = np.array((starts >= lo) & (starts <= hi))
    return masks


def append_rows_atomic(csv_path, rows: list, columns: list) -> None:
    """Append result rows to a CSV, writing the header once."""
    import pandas as pd
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode='a', header=header, index=False)
