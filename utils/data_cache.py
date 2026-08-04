"""
Data caching utilities für efficient multi-GPU training.
Implements memory-mapped loading to avoid duplicating preprocessed data across processes.
"""
import os
import copy
import pickle
import hashlib
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from . import preprocessing, tools, hpo

# Single source of truth for the cache location. Anything that needs to know
# where cache entries live (size accounting, eviction, cache_manager.py) must
# import this instead of repeating the literal — a second hardcoded copy in
# hpo_cl_tft_bc.py once silently pointed the eviction logic at an empty relative
# directory, so nothing was ever evicted and the root partition filled up.
DEFAULT_CACHE_DIR = "/mnt/nvme2/data_cache"
DEFAULT_GNN_CACHE_DIR = f"{DEFAULT_CACHE_DIR}/gnns"


class DataCache:
    """
    Manages disk-based caching of preprocessed data with memory-mapped loading.
    """

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def _get_config_hash(self, config: Dict, features: Dict, model_name: str = None) -> str:
        """
        Generate a hash from config, features and model to identify cached data.

        Args:
            config: Configuration dictionary
            features: Features dictionary
            model_name: Name of the model (affects preprocessing)

        Returns:
            Hash string identifying this specific preprocessing configuration
        """
        # Create a hashable representation of config, features and model
        hash_data = {
            # Bump whenever the *semantics* of preprocessing change without any
            # config key changing, so previously cached (differently-built) entries
            # can never be silently reused. v2: known-window past run offset fixed
            # to future_len, val folds chunked temporally instead of station-major.
            # v3: single global scaler_x fitted across all training stations instead of
            # one StandardScaler per station (see _fit_global_scaler_x) — this also
            # activates the static-feature scaling in prepare_data_for_tft.
            # v4: next_n_stations neighbours are drawn from a split-dependent pool
            # (training stations see `files` only, validation stations see
            # `files + val_files`) instead of the whole stations_master.csv. The pool is a
            # deterministic function of `files`/`val_files`, both already hashed below.
            'preprocessing_version': 4,
            'data_path': config['data']['path'],
            'files': sorted(config['data'].get('files', [])),
            'val_files': sorted(config['data'].get('val_files', [])),
            'freq': config['data']['freq'],
            'target_col': config['data']['target_col'],
            'test_start': config['data'].get('test_start', None),
            'features': features,
            'next_n_grid_points': config['params']['next_n_grid_points'],
            'get_next_grid_points_method': config['params'].get('get_next_grid_points_method', None),
            'next_n_grid_ecmwf': config['params'].get('next_n_grid_ecmwf', None),
            'next_n_stations': config['params'].get('next_n_stations', None),
            'nwp_models': config['params'].get('nwp_models', None),
            'ecmwf_features': config['params'].get('ecmwf_features', None),
            'model': {
                'name': model_name,
                'lookback': config['model']['lookback'],
                'horizon': config['model']['horizon'],
                'step_size': config['model'].get('step_size', 1),
                'output_dim': config['model']['output_dim']
            },
            'hpo': {
                'min_train_date': config['hpo'].get('min_train_date', None),
                'kfolds': config['hpo'].get('kfolds', None)
            }
        }

        # cv_mode='spatial' (create_or_load_preprocessed_data_spatial): station ROLES
        # rotate per fold instead of being fixed for the whole study, and the val window
        # is a fixed [val_start, test_start) instead of the temporal n_splits+1 chunking.
        # 'files'/'val_files' above already differ per spatial fold (different station
        # sets) and already differ from any temporal config in practice, so this key is
        # redundant for actually distinguishing entries — it exists purely as an explicit
        # guard so a spatial and a temporal config could never collide on hash even in a
        # hypothetical edge case where their station lists coincided. Only added when
        # cv_mode is truly 'spatial' (a key absent from every existing temporal config),
        # so every existing temporal-mode hash_data dict — and therefore every existing
        # cache_id — is unchanged bit-for-bit.
        cv_mode = str(config.get('hpo', {}).get('cv_mode', 'temporal')).lower()
        if cv_mode == 'spatial':
            hash_data['cv_mode'] = 'spatial'
            hash_data['val_start'] = config['data'].get('val_start')

        # Convert to string and hash
        hash_string = str(sorted(hash_data.items()))
        return hashlib.md5(hash_string.encode()).hexdigest()

    def get_cache_paths(self, cache_id: str) -> Dict[str, str]:
        """Get paths for cached data files."""
        base_path = os.path.join(self.cache_dir, cache_id)
        return {
            'metadata': f"{base_path}_metadata.pkl",
            'prepared_datasets': f"{base_path}_prepared.npy",
            'combined_kfolds': f"{base_path}_kfolds.npy"
        }

    def is_cached(self, config: Dict, features: Dict, model_name: str = None) -> Tuple[bool, str]:
        """
        Check if preprocessed data exists in cache.

        Returns:
            (is_cached, cache_id)
        """
        cache_id = self._get_config_hash(config, features, model_name)
        paths = self.get_cache_paths(cache_id)

        # Check if required files exist (support both .npy and .pkl for datasets)
        datasets_pkl = paths['prepared_datasets'].replace('.npy', '.pkl')
        datasets_exists = os.path.exists(datasets_pkl) or os.path.exists(paths['prepared_datasets'])

        required_files = [paths['metadata']]

        # Check for k-folds - either new individual files system or old system
        manifest_file = os.path.join(self.cache_dir, f"{cache_id}_fold_manifest.pkl")
        kfolds_pkl = paths['combined_kfolds'].replace('.npy', '.pkl')

        # New system: check for fold manifest AND fold directory
        if os.path.exists(manifest_file):
            fold_dir = os.path.join(self.cache_dir, f"{cache_id}_folds")
            kfolds_exists = os.path.exists(fold_dir) and len(os.listdir(fold_dir)) > 0
            self.logger.debug(f"Checking new fold system: manifest={os.path.exists(manifest_file)}, folds_dir={kfolds_exists}")
        else:
            # Old system: check for single pickle/npy files
            kfolds_exists = os.path.exists(kfolds_pkl) or os.path.exists(paths['combined_kfolds'])
            self.logger.debug(f"Checking old fold system: pkl={os.path.exists(kfolds_pkl)}, npy={os.path.exists(paths['combined_kfolds'])}")

        all_exist = all(os.path.exists(path) for path in required_files) and datasets_exists and kfolds_exists

        return all_exist, cache_id

    def save_preprocessed_data(self,
                             config: Dict,
                             features: Dict,
                             prepared_datasets: List[Dict],
                             combined_kfolds: List,
                             model_name: str = None) -> str:
        """
        Save preprocessed data to cache.

        Args:
            config: Configuration dictionary
            features: Features dictionary
            prepared_datasets: List of prepared datasets
            combined_kfolds: Combined k-folds data
            model_name: Name of the model (affects preprocessing)

        Returns:
            cache_id: Unique identifier for this cached data
        """
        cache_id = self._get_config_hash(config, features, model_name)
        paths = self.get_cache_paths(cache_id)

        self.logger.info(f"Saving preprocessed data to cache (ID: {cache_id})")

        # Save metadata
        metadata = {
            'config': config,
            'features': features,
            'model_name': model_name,
            'cache_id': cache_id,
            'n_datasets': len(prepared_datasets),
            'n_folds': len(combined_kfolds) if combined_kfolds else 0
        }

        with open(paths['metadata'], 'wb') as f:
            pickle.dump(metadata, f)

        # Save prepared datasets as memory-mapped array
        # We'll save each dataset as a separate section in the file
        self._save_prepared_datasets_mmap(prepared_datasets, paths['prepared_datasets'])

        # Save combined k-folds as memory-mapped array
        if combined_kfolds:
            self._save_combined_kfolds_mmap(combined_kfolds, paths['combined_kfolds'])

        self.logger.info(f"Successfully cached data with ID: {cache_id}")
        return cache_id

    def _save_prepared_datasets_mmap(self, prepared_datasets: List[Dict], file_path: str):
        """Save prepared datasets in memory-mappable format."""
        # Save as pickle since the datasets have complex nested structure
        with open(file_path.replace('.npy', '.pkl'), 'wb') as f:
            pickle.dump(prepared_datasets, f)

    def _save_combined_kfolds_mmap(self, combined_kfolds: List, file_path: str):
        """Save combined k-folds as individual files for true lazy loading."""
        cache_id = os.path.basename(file_path).replace('_kfolds.npy', '').replace('_kfolds.pkl', '')

        # Create individual files for each fold
        fold_dir = os.path.join(self.cache_dir, f"{cache_id}_folds")
        os.makedirs(fold_dir, exist_ok=True)

        fold_manifest = []

        self.logger.info(f"Saving {len(combined_kfolds)} folds as individual files...")

        for i, fold_data in enumerate(combined_kfolds):
            fold_file = os.path.join(fold_dir, f"fold_{i:03d}.pkl")

            with open(fold_file, 'wb') as f:
                pickle.dump(fold_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Store metadata for quick access
            (X_train, y_train), (X_val, y_val) = fold_data
            fold_info = {
                'file': fold_file,
                'train_samples': len(y_train),
                'val_samples': len(y_val),
                'train_shape': y_train.shape if hasattr(y_train, 'shape') else 'unknown',
                'val_shape': y_val.shape if hasattr(y_val, 'shape') else 'unknown'
            }
            fold_manifest.append(fold_info)

            self.logger.debug(f"Saved fold {i} to {fold_file}")

        # Save fold manifest for quick access
        manifest_file = os.path.join(self.cache_dir, f"{cache_id}_fold_manifest.pkl")
        with open(manifest_file, 'wb') as f:
            pickle.dump(fold_manifest, f)

        self.logger.info(f"Saved fold manifest with {len(fold_manifest)} folds")

    def load_preprocessed_data(self, cache_id: str) -> Tuple[List[Dict], List, Dict]:
        """
        Load preprocessed data from cache using memory mapping.

        Args:
            cache_id: Cache identifier

        Returns:
            (prepared_datasets, combined_kfolds, metadata)
        """
        paths = self.get_cache_paths(cache_id)

        self.logger.info(f"Loading preprocessed data from cache (ID: {cache_id})")

        # Load metadata
        with open(paths['metadata'], 'rb') as f:
            metadata = pickle.load(f)

        # Load data - check for both .npy and .pkl files for backward compatibility
        datasets_path_npy = paths['prepared_datasets']
        datasets_path_pkl = datasets_path_npy.replace('.npy', '.pkl')

        if os.path.exists(datasets_path_pkl):
            # Load from pickle file (newer format)
            with open(datasets_path_pkl, 'rb') as f:
                prepared_datasets = pickle.load(f)
        else:
            # Load from numpy file (legacy format)
            prepared_datasets = np.load(paths['prepared_datasets'],
                                      allow_pickle=True,
                                      mmap_mode='r')

        # Try to load fold manifest (new individual-file system)
        manifest_file = os.path.join(self.cache_dir, f"{cache_id}_fold_manifest.pkl")
        if os.path.exists(manifest_file):
            # Load from new individual-file system
            self.logger.info("Loading fold data from individual fold files")
            with open(manifest_file, 'rb') as f:
                combined_kfolds = pickle.load(f)  # This is just the manifest, not the data
        else:
            # Fallback to old system
            combined_kfolds = None
            # Check for both .npy and .pkl files for backward compatibility
            kfolds_path_npy = paths['combined_kfolds']
            kfolds_path_pkl = kfolds_path_npy.replace('.npy', '.pkl')

            if os.path.exists(kfolds_path_pkl):
                # Load from pickle file (old format)
                self.logger.info("Loading fold data from single pickle file (legacy)")
                with open(kfolds_path_pkl, 'rb') as f:
                    combined_kfolds = pickle.load(f)
            elif os.path.exists(kfolds_path_npy):
                # Load from numpy file (legacy format)
                self.logger.info("Loading fold data from numpy file (legacy)")
                combined_kfolds = np.load(kfolds_path_npy,
                                        allow_pickle=True,
                                        mmap_mode='r')

        self.logger.info(f"Successfully loaded cached data with {len(prepared_datasets)} datasets")

        return prepared_datasets, combined_kfolds, metadata


class LazyFoldLoader:
    """
    Lazy loader for k-fold data that only loads folds when needed.
    """

    def __init__(self, fold_manifest_or_data):
        """
        Initialize lazy fold loader.

        Args:
            fold_manifest_or_data: Either fold manifest (new system) or combined kfolds (old system)
        """
        self.logger = logging.getLogger(__name__)

        if isinstance(fold_manifest_or_data, list) and len(fold_manifest_or_data) > 0:
            # Check if it's a manifest or actual data
            if isinstance(fold_manifest_or_data[0], dict) and 'file' in fold_manifest_or_data[0]:
                # New system: manifest with file paths
                self.fold_manifest = fold_manifest_or_data
                self.is_new_system = True
                self.logger.info(f"Initialized LazyFoldLoader with {len(self.fold_manifest)} individual fold files")
            else:
                # Old system: actual fold data
                self.combined_kfolds_mmap = fold_manifest_or_data
                self.is_new_system = False
                self.logger.info(f"Initialized LazyFoldLoader with {len(self.combined_kfolds_mmap)} folds (legacy)")
        else:
            # Empty or None
            self.fold_manifest = []
            self.is_new_system = True
            self.logger.warning("Initialized empty LazyFoldLoader")

    def __len__(self):
        """Return number of folds."""
        if self.is_new_system:
            return len(self.fold_manifest)
        else:
            return len(self.combined_kfolds_mmap)

    def __getitem__(self, fold_idx: int) -> Tuple:
        """
        Load a specific fold on demand.

        Args:
            fold_idx: Index of fold to load

        Returns:
            ((X_train, y_train), (X_val, y_val))
        """
        if self.is_new_system:
            # New system: load from individual file
            if fold_idx >= len(self.fold_manifest):
                raise IndexError(f"Fold index {fold_idx} out of range (max: {len(self.fold_manifest)-1})")

            fold_file = self.fold_manifest[fold_idx]['file']
            self.logger.debug(f"Loading fold {fold_idx} from {fold_file}")

            with open(fold_file, 'rb') as f:
                fold_data = pickle.load(f)

            return fold_data
        else:
            # Old system: access memory-mapped data
            self.logger.debug(f"Loading fold {fold_idx} from memory-mapped data")
            fold_data = self.combined_kfolds_mmap[fold_idx]

            if isinstance(fold_data, (list, tuple)):
                return fold_data
            else:
                return fold_data.item() if hasattr(fold_data, 'item') else fold_data

    def get_fold_info(self, fold_idx: int) -> Dict:
        """
        Get information about a fold without fully loading it.

        Args:
            fold_idx: Index of fold

        Returns:
            Dictionary with fold information
        """
        if self.is_new_system:
            # New system: return cached info from manifest
            if fold_idx >= len(self.fold_manifest):
                raise IndexError(f"Fold index {fold_idx} out of range")

            fold_info = self.fold_manifest[fold_idx].copy()
            fold_info['fold_idx'] = fold_idx
            return fold_info
        else:
            # Old system: load data to get info (not ideal)
            fold_data = self[fold_idx]
            (X_train, y_train), (X_val, y_val) = fold_data

            info = {
                'fold_idx': fold_idx,
                'train_samples': len(y_train),
                'val_samples': len(y_val),
                'train_shape': y_train.shape,
                'val_shape': y_val.shape
            }

            if isinstance(X_train, dict):
                info['feature_keys'] = list(X_train.keys())
                info['feature_shapes'] = {k: v.shape for k, v in X_train.items()}
            else:
                info['input_shape'] = X_train.shape

            return info


def _fit_global_scaler_x(dfs, config, logger, fit_until=None):
    """
    Fit ONE StandardScaler over the training rows of ALL training stations.

    Args:
        fit_until: Optional cutoff (anything pd.Timestamp() accepts) used INSTEAD of
            config['data']['test_start'] as the train/not-train boundary passed to
            split_data(). Additive, defaults to None (= old behaviour, cutoff at
            test_start unchanged). Used by create_or_load_preprocessed_data_spatial for
            cv_mode='spatial': there the scaler must be fit only on data strictly before
            data.val_start (the fold's val chunk starts there), not before test_start —
            otherwise the val chunk would leak into the very scaler used to transform it.

    Mirrors train_cl.py:326-370 and the GNN pipelines' `meas_scaler`
    (geostatistics/train_dcrnn.py:750), which likewise fit a single scaler across all
    training stations rather than one per station.

    Why this matters (and not just for tidiness):
      * Per-station scaling z-scores every station's observed history against its own
        mean/std, while the target stays in raw m/s (scale_target=False for
        target_col='wind_speed', preprocessing.py:377). The station's absolute wind
        level is thereby erased from the features but still demanded of the output —
        so the near-persistence relation y ≈ own_history, which is the whole point of
        the 'hist' variant's `wind_speed` observed feature, becomes unlearnable for
        unseen stations. Measured on held-out stations, the own-history advantage went
        from +1.4 % (per-station) to +9.7 % (global), and at leads 1-6 h from -1.8 %
        to +29 %.
      * Static features are per-station CONSTANTS. A per-station scaler would collapse
        them all to exactly 0, which is why the static-scaling block in
        prepare_data_for_tft (preprocessing.py:3682) is guarded by `if scaler_x` — with
        a per-station scaler it is skipped entirely and statics reach the model raw
        (altitude up to 2956 next to z0 at 0.0005). A global scaler makes that block
        both applicable and correct.

    The scaler is deliberately fitted on the *training* stations only (`files`), never
    on `val_files`: _replace_val_with_val_files reuses the same config and therefore the
    same pre-fitted scaler, so validation stations are transformed but never fitted on.

    The target column is excluded, matching prepare_data_for_tft's
    `feature_cols = [c for c in train_df.columns if c != target_col]`.
    """
    from sklearn.preprocessing import StandardScaler

    target_col = config['data']['target_col']
    t_0 = 0 if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
    history_length = config['model']['lookback']

    scaler = StandardScaler()
    # The target column is excluded from `scaler` (see below), but for target_col
    # 'wind_speed' the very same column is ALSO an observed input feature in the 'hist'
    # variant. It therefore needs its own global scaler, or it would reach the model raw
    # while every other observed column is standardised — an asymmetry that would hit
    # 'hist' only. y itself is taken from the UNSCALED frame in prepare_data_for_tft, so
    # this never touches the target.
    target_feature_scaler = StandardScaler()
    ref_cols, n_stations, n_rows = None, 0, 0

    for key in list(dfs.keys()):
        # Impute here rather than inside pipeline(): the scaler must see exactly the
        # values prepare_data_for_tft will later transform, and NaNs would poison
        # partial_fit. knn_imputer preserves the column set and returns early when
        # there is nothing to impute, so pipeline()'s own call becomes a no-op.
        dfs[key] = preprocessing.knn_imputer(data=dfs[key],
                                             n_neighbors=config['data']['n_neighbors'])
        df = dfs[key]

        # Mirror prepare_data_for_tft's split exactly, including the backwards
        # extension of test_start for NWP data (preprocessing.py:3657-3666) — otherwise
        # the scaler would be fitted on rows that end up in the test split.
        cutoff = fit_until if fit_until is not None else config['data'].get('test_start', None)
        cutoff = pd.Timestamp(cutoff) if cutoff is not None else None
        is_nwp = isinstance(df.index, pd.MultiIndex) and 'starttime' in df.index.names
        if is_nwp and cutoff is not None:
            cutoff = cutoff - pd.Timedelta(hours=history_length)

        df_train, _ = preprocessing.split_data(
            data=df,
            train_frac=config['data']['train_frac'],
            train_start=pd.Timestamp(config['data'].get('train_start', None)),
            test_start=cutoff,
            test_end=pd.Timestamp(config['data'].get('test_end', None)),
            t_0=t_0,
        )
        if len(df_train) == 0:
            logger.warning(f"global scaler_x: station {key} contributed no training rows — skipped.")
            continue

        df_train_x = df_train.drop(columns=[target_col], errors='ignore')

        # A silently differing column set (e.g. a neighbour station missing a feature,
        # which preprocess_synth_wind_icond2 only logs a warning for) would make
        # partial_fit either raise or — worse — align the wrong columns across stations.
        if ref_cols is None:
            ref_cols = df_train_x.columns.tolist()
        elif df_train_x.columns.tolist() != ref_cols:
            missing = sorted(set(ref_cols) - set(df_train_x.columns))
            extra = sorted(set(df_train_x.columns) - set(ref_cols))
            raise ValueError(
                f"global scaler_x: station {key} has a different feature column set than the "
                f"first station. Missing: {missing}. Unexpected: {extra}. Column order/content "
                "must be identical across stations or the shared scaler misaligns features."
            )

        scaler.partial_fit(df_train_x.values)
        if target_col in df_train.columns:
            target_feature_scaler.partial_fit(df_train[[target_col]].values)
        n_stations += 1
        n_rows += len(df_train_x)
        del df_train, df_train_x

    if ref_cols is None:
        raise ValueError("global scaler_x: no training station produced any training rows.")

    logger.info(
        f"Fitted global scaler_x on {n_stations} training stations / {n_rows} rows / "
        f"{len(ref_cols)} feature columns (target '{target_col}' excluded, val_files not seen)."
    )
    # Consumed by prepare_data_for_tft to verify that every station it later transforms
    # presents the same columns in the same order. sklearn only checks the column COUNT
    # when handed a bare ndarray, so a reordered frame would otherwise be scaled with
    # the wrong column's mean/std without any error.
    scaler._ff_feature_cols = ref_cols
    if hasattr(target_feature_scaler, 'mean_'):
        scaler._ff_target_feature_scaler = target_feature_scaler
        logger.info(
            f"Fitted global scaler for '{target_col}' used as an input feature: "
            f"mean={target_feature_scaler.mean_[0]:.4f} scale={target_feature_scaler.scale_[0]:.4f} "
            "(applies to the 'hist' variant's own-history column; y stays raw)."
        )
    return scaler


def _replace_val_with_val_files(combined_kfolds, config, features, logger):
    """
    Replace the val portion of each k-fold with data from val_files stations.

    Val stations are processed for the full training period (X_train, before test_start).
    If ``config['hpo']['min_train_date']`` is set, samples before it are dropped first —
    mirroring ``apply_min_train_len_per_file``'s treatment of the training stations, so
    the val chunk boundaries below line up with the *same* [min_train_date, test_start]
    window the training folds are carved from (otherwise val chunk 0 could start years
    before any training fold's val window, since val stations get no "fixed block").
    The (possibly truncated) sequences are split into n_splits+1 equal temporal chunks;
    fold k gets chunk k+1, mirroring the TimeSeriesSplit structure used for training
    stations.
    """
    freq = config['data']['freq']
    data_dir = config['data']['path']
    n_splits = config['hpo']['kfolds']
    min_train_date = config['hpo'].get('min_train_date')

    # Neighbour pool for the VALIDATION stations: training + validation stations. At
    # validation time the val stations' own measurements are legitimately available (they
    # are inputs, not the thing being scored), so they may serve as each other's
    # neighbours — but test_files must stay invisible. Set before get_data, since the
    # neighbour merge happens while loading.
    config['data']['neighbor_pool'] = (list(config['data'].get('files', []))
                                       + list(config['data'].get('val_files', [])))
    logger.info(f"Neighbour pool for validation stations: "
                f"{len(config['data']['neighbor_pool'])} stations (files + val_files)")

    val_dfs = preprocessing.get_data(
        data_dir=data_dir,
        config=config,
        freq=freq,
        features=features,
        files_key='val_files'
    )

    prepared_list = []
    for key, df in val_dfs.items():
        prepared_data, _ = preprocessing.pipeline(
            data=df,
            config=config,
            known_cols=features['known'],
            observed_cols=features['observed'],
            static_cols=features['static'],
            target_col=config['data']['target_col']
        )
        if prepared_data is None:
            continue
        X_tr = prepared_data.get('X_train')
        y_tr = prepared_data.get('y_train')
        if X_tr is None or y_tr is None or len(y_tr) == 0:
            continue
        prepared_list.append(prepared_data)

    if min_train_date and prepared_list:
        split_result = hpo.apply_min_train_len_per_file(prepared_list, min_train_date)
        prepared_list = split_result['remaining_datasets']
        logger.info(
            f"val_files: dropped samples before min_train_date={min_train_date} "
            f"({len(prepared_list)} stations remaining)."
        )

    X_vals, y_vals, idx_vals = [], [], []
    for prepared_data in prepared_list:
        X_tr = prepared_data.get('X_train')
        y_tr = prepared_data.get('y_train')
        idx_tr = prepared_data.get('index_train')
        if X_tr is None or y_tr is None or len(y_tr) == 0:
            continue
        if idx_tr is None or len(idx_tr) != len(y_tr):
            raise ValueError(
                "val_files: 'index_train' missing or misaligned with y_train "
                f"(index={None if idx_tr is None else len(idx_tr)}, y={len(y_tr)}). "
                "Cannot build temporal validation chunks without per-sample timestamps."
            )
        X_vals.append(X_tr)
        y_vals.append(y_tr)
        idx_vals.append(pd.Index(idx_tr))

    if not X_vals:
        logger.warning("val_files produced no training-period samples — k-fold val unchanged.")
        return combined_kfolds

    if isinstance(X_vals[0], dict):
        X_val_full = {k: np.concatenate([x[k] for x in X_vals], axis=0) for k in X_vals[0]}
    else:
        X_val_full = np.concatenate(X_vals, axis=0)
    y_val_full = np.concatenate(y_vals, axis=0)
    idx_val_full = idx_vals[0].append(idx_vals[1:]) if len(idx_vals) > 1 else idx_vals[0]

    # Sort every val sample chronologically before chunking. The concatenation
    # above is station-major (station 1's full history, then station 2's, …), so
    # slicing index ranges out of it would carve out *stations*, not time windows —
    # every fold would then span the full period and differ only in which val
    # stations it saw. Sorting first makes the chunks below genuine temporal
    # slices, each containing all val stations, so folds differ only in their
    # validation *period*, mirroring the TimeSeriesSplit over the training stations.
    order = np.argsort(idx_val_full.values, kind='stable')
    y_val_full = y_val_full[order]
    if isinstance(X_val_full, dict):
        X_val_full = {k: v[order] for k, v in X_val_full.items()}
    else:
        X_val_full = X_val_full[order]
    idx_val_full = idx_val_full[order]

    n_val = len(y_val_full)
    n_chunks = n_splits + 1
    chunk_size = n_val // n_chunks

    modified_kfolds = []
    for k, (train, _) in enumerate(combined_kfolds):
        start = (k + 1) * chunk_size
        end = (k + 2) * chunk_size if k < n_splits - 1 else n_val
        if isinstance(X_val_full, dict):
            X_chunk = {key: arr[start:end] for key, arr in X_val_full.items()}
        else:
            X_chunk = X_val_full[start:end]
        modified_kfolds.append((train, (X_chunk, y_val_full[start:end])))

    for k, (_, (_, y_chunk)) in enumerate(modified_kfolds):
        start = (k + 1) * chunk_size
        end = (k + 2) * chunk_size if k < n_splits - 1 else n_val
        logger.info(
            f"  val fold {k}: {len(y_chunk)} samples, "
            f"{idx_val_full[start]} → {idx_val_full[end - 1]}"
        )
    logger.info(
        f"Val portions replaced with val_files data: {n_splits} temporal chunks "
        f"(~{chunk_size} samples each) from {len(val_dfs)} val stations."
    )
    return modified_kfolds


def create_or_load_preprocessed_data(config: Dict,
                                   features: Dict,
                                   model_name: str = None,
                                   force_reprocess: bool = False,
                                   use_cache: bool = True,
                                   cache_dir: str = None) -> Tuple[LazyFoldLoader, str]:
    """
    Create or load preprocessed data with caching.

    Args:
        config: Configuration dictionary
        features: Features dictionary
        model_name: Name of the model (affects preprocessing)
        force_reprocess: Force reprocessing even if cache exists
        use_cache: Whether to use caching at all (False for small datasets)

    Returns:
        (lazy_fold_loader, cache_id)
    """
    logger = logging.getLogger(__name__)

    # If caching is disabled, process data directly
    if not use_cache:
        logger.info("Caching disabled, processing data directly")
        # Process data without caching
        prepared_datasets = preprocessing.preprocess_data(config, features)
        combined_kfolds = preprocessing.get_k_folds(prepared_datasets, config)

        # Create lazy loader directly from memory
        lazy_loader = LazyFoldLoader(combined_kfolds)
        logger.info(f"Processed {len(lazy_loader)} folds without caching")
        return lazy_loader, "no_cache"

    cache = DataCache(cache_dir or DEFAULT_CACHE_DIR)

    # Check if data is already cached
    is_cached, cache_id = cache.is_cached(config, features, model_name)

    if is_cached and not force_reprocess:
        logger.info(f"Found cached data (ID: {cache_id}), loading from cache")
        prepared_datasets, combined_kfolds, metadata = cache.load_preprocessed_data(cache_id)

        # Create lazy loader
        if combined_kfolds is not None:
            lazy_loader = LazyFoldLoader(combined_kfolds)
            logger.info(f"Loaded {len(lazy_loader)} folds from cache")
            return lazy_loader, cache_id
        else:
            logger.warning("No k-folds found in cache, need to reprocess")
            force_reprocess = True

    if not is_cached or force_reprocess:
        logger.info("Preprocessing data and saving to cache...")

        # Perform full preprocessing
        freq = config['data']['freq']
        data_dir = config['data']['path']

        # Neighbour pool for the TRAINING stations: training stations only. A neighbour
        # contributes its measured `wind_speed` history as an input feature, and that
        # series is precisely that station's target — so drawing neighbours from
        # val_files/test_files would feed held-out data into training. Must be set BEFORE
        # get_data: the neighbour merge happens during loading, inside
        # preprocess_synth_wind_icond2, not later in the pipeline.
        config['data']['neighbor_pool'] = list(config['data'].get('files', []))
        logger.info(f"Neighbour pool for training stations: "
                    f"{len(config['data']['neighbor_pool'])} stations (files only)")

        # Load raw data
        dfs = preprocessing.get_data(data_dir=data_dir,
                                   config=config,
                                   freq=freq,
                                   features=features)

        # Pass 1: fit ONE scaler across all training stations before preparing any of
        # them, then hand it to pipeline() via config['scaler_x'] — prepare_data_for_tft
        # switches to its GLOBAL SCALING branch (preprocessing.py:3749) when it is set,
        # and _replace_val_with_val_files below reuses the same config, so the val
        # stations are transformed with this scaler instead of fitting their own.
        config['scaler_x'] = _fit_global_scaler_x(dfs, config, logger)

        # Pass 2: preprocess each dataset with the shared scaler
        prepared_datasets = []
        for key, df in dfs.items():
            logger.debug(f'Preprocessing {key}')

            # Ensure model name is set for preprocessing compatibility
            if 'name' not in config['model']:
                config['model']['name'] = 'temp_for_preprocessing'

            prepared_data, processed_df = preprocessing.pipeline(
                data=df,
                config=config,
                known_cols=features['known'],
                observed_cols=features['observed'],
                static_cols=features['static'],
                target_col=config['data']['target_col']
            )
            prepared_datasets.append(prepared_data)

        # Create k-folds
        min_train_date = config['hpo'].get('min_train_date', None)
        combined_kfolds = hpo.kfolds_with_per_file_min_train_len(
            prepared_datasets=prepared_datasets,
            n_splits=config['hpo']['kfolds'],
            val_split=config['hpo']['val_split'],
            min_train_date=min_train_date
        )

        # If val_files is specified, replace each fold's val portion with data from val stations.
        # Val stations are processed for the training period (before test_start) so their
        # temporal windows align with the k-fold val windows from the training stations.
        # TimeSeriesSplit with n_splits folds divides into n_splits+1 parts; fold k uses
        # part k+1. We mirror this: split val_files sequences into n_splits+1 chunks and
        # assign chunk k+1 to fold k. Result is embedded in combined_kfolds before caching.
        if config['data'].get('val_files'):
            combined_kfolds = _replace_val_with_val_files(combined_kfolds, config, features, logger)

        # Save to cache
        cache_id = cache.save_preprocessed_data(config, features, prepared_datasets, combined_kfolds, model_name)

        # Create lazy loader
        lazy_loader = LazyFoldLoader(combined_kfolds)
        logger.info(f"Created and cached {len(lazy_loader)} folds")

        return lazy_loader, cache_id


# ---------------------------------------------------------------------------
# Spatial-CV sibling of create_or_load_preprocessed_data() — additive, does not
# touch DataCache, _fit_global_scaler_x's old call sites, _replace_val_with_val_files,
# or create_or_load_preprocessed_data itself, so cv_mode='temporal' (every existing
# config) is untouched. Mirrors geostatistics/hpo_dcrnn.py's cv_mode='spatial' handling:
# same time window in every fold (train < data.val_start, val in
# [val_start, test_start)), rotating station ROLES instead of a fixed files/val_files
# split. Unlike the GNN pipeline (raw tensors cached once for the whole 153-station
# pool, role applied per-epoch at near-zero cost), the TFT pipeline bakes both feature
# scaling AND next_n_stations neighbour selection into each station's prepared arrays
# at preprocessing time — both are role-dependent under cv_mode='spatial' (a station's
# role, and therefore its legal neighbour pool, differs across the 3 folds), so they
# cannot be computed once and reused unmodified across folds the way the raw NWP/
# measurement loading could be. This function therefore does its own preprocessing
# pass per fold (own DataCache entry, own scaler, own neighbour-restricted station
# data) rather than sharing one pool-wide cache entry across all 3 folds. See the
# implementation report for why this is the deliberate, documented trade-off (keeps
# fold correctness exact; costs ~3x the preprocessing I/O of the temporal path across
# the whole spatial study, paid once, cached from then on).
# ---------------------------------------------------------------------------

def _spatial_fold_time_mask(idx, since=None, until=None):
    """Boolean mask for a per-sample DatetimeIndex-like `idx` inside [since, until)."""
    ts = pd.DatetimeIndex(pd.to_datetime(idx))
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    mask = np.ones(len(ts), dtype=bool)
    if since is not None:
        mask &= ts >= since
    if until is not None:
        mask &= ts < until
    return mask


def _split_prepared_by_time(prepared_data, since=None, until=None):
    """Slice one station's prepared_data['X_train']/y_train to samples whose timestamp
    (index_train) falls in [since, until). Returns (X, y) or None if nothing remains.

    Additive helper for create_or_load_preprocessed_data_spatial: unlike
    _replace_val_with_val_files' n_splits+1 chunking (temporal mode, rotating time
    window over a FIXED station split), spatial CV uses one fixed [val_start,
    test_start) cut on a ROTATING station split, so no chunk-index bookkeeping is
    needed here — just a timestamp mask per station.
    """
    X = prepared_data.get('X_train')
    y = prepared_data.get('y_train')
    idx = prepared_data.get('index_train')
    if X is None or y is None or len(y) == 0:
        return None
    if idx is None or len(idx) != len(y):
        raise ValueError(
            "create_or_load_preprocessed_data_spatial: 'index_train' missing or "
            f"misaligned with y_train (index={None if idx is None else len(idx)}, "
            f"y={len(y)}). Cannot build a time-window fold split without per-sample "
            "timestamps."
        )
    mask = _spatial_fold_time_mask(idx, since=since, until=until)
    if not mask.any():
        return None
    if isinstance(X, dict):
        X_f = {k: v[mask] for k, v in X.items()}
    else:
        X_f = X[mask]
    return X_f, y[mask]


def _concat_fold_split(splits):
    """Concatenate a list of per-station (X, y) splits (skipping empty ones) into one
    (X, y) pair, dict-valued X included. Returns None if every split was empty."""
    splits = [s for s in splits if s is not None]
    if not splits:
        return None
    Xs, ys = zip(*splits)
    if isinstance(Xs[0], dict):
        X_cat = {k: np.concatenate([x[k] for x in Xs], axis=0) for k in Xs[0]}
    else:
        X_cat = np.concatenate(Xs, axis=0)
    return X_cat, np.concatenate(ys, axis=0)


def _build_spatial_fold_data(fold_config: Dict, features: Dict, logger) -> Tuple[List[Dict], List]:
    """Preprocess ONE spatial-CV fold: fold_config['data']['files']/['val_files'] give
    the fold's train-role / target-role station IDs (rotate per fold; caller's
    responsibility — see create_or_load_preprocessed_data_spatial). Returns
    (prepared_datasets, combined_kfolds) with exactly one fold, ready for
    DataCache.save_preprocessed_data.
    """
    freq = fold_config['data']['freq']
    data_dir = fold_config['data']['path']
    train_ids = list(fold_config['data'].get('files', []))
    val_ids = list(fold_config['data'].get('val_files', []))

    val_start = pd.Timestamp(fold_config['data']['val_start'], tz='UTC')
    test_start_raw = fold_config['data'].get('test_start')
    test_start = pd.Timestamp(test_start_raw, tz='UTC') if test_start_raw else None

    # Train-role stations: neighbours restricted to this fold's OTHER train-role
    # stations only — same restriction create_or_load_preprocessed_data already
    # applies to its (fixed, whole-study) 'files' stations; here 'files' just rotates
    # per fold instead of being fixed for the whole study.
    fold_config['data']['neighbor_pool'] = list(train_ids)
    logger.info("Spatial fold: neighbour pool for %d training stations: %d candidates "
                "(train-role only)", len(train_ids), len(fold_config['data']['neighbor_pool']))
    train_dfs = preprocessing.get_data(data_dir=data_dir, config=fold_config, freq=freq,
                                        features=features, files_key='files')

    # Per-fold scaler: fit ONLY on this fold's train-role stations and ONLY on data
    # strictly before val_start (not test_start) — the fold's val chunk starts at
    # val_start, so anything from val_start onward must stay unseen by the scaler.
    fold_config['scaler_x'] = _fit_global_scaler_x(train_dfs, fold_config, logger, fit_until=val_start)

    if 'name' not in fold_config['model']:
        fold_config['model']['name'] = 'temp_for_preprocessing'

    prepared_train = []
    for key, df in train_dfs.items():
        prepared_data, _ = preprocessing.pipeline(
            data=df, config=fold_config, known_cols=features['known'],
            observed_cols=features['observed'], static_cols=features['static'],
            target_col=fold_config['data']['target_col'])
        if prepared_data is not None:
            prepared_train.append(prepared_data)

    # Target-role stations of this fold: neighbours restricted to train-role stations
    # ONLY — not other target-role stations of this fold. Mirrors the GNN sampler
    # (geostatistics/stgnn/training/sampler.py::sample_val(), Z.313-317: a validation
    # target's spatial neighbours are drawn exclusively from training stations). This
    # is the one deliberate behavioural difference from _replace_val_with_val_files,
    # whose val stations pool files+val_files — safe there only because temporal folds
    # never change WHICH stations are held out, so "other val stations" are always the
    # same fixed val_files set, never a rotating target set.
    fold_config['data']['neighbor_pool'] = list(train_ids)
    logger.info("Spatial fold: neighbour pool for %d target stations: %d candidates "
                "(train-role only, NOT other target stations)", len(val_ids),
                len(fold_config['data']['neighbor_pool']))
    val_dfs = preprocessing.get_data(data_dir=data_dir, config=fold_config, freq=freq,
                                      features=features, files_key='val_files')

    prepared_val = []
    for key, df in val_dfs.items():
        prepared_data, _ = preprocessing.pipeline(
            data=df, config=fold_config, known_cols=features['known'],
            observed_cols=features['observed'], static_cols=features['static'],
            target_col=fold_config['data']['target_col'])
        if prepared_data is not None:
            prepared_val.append(prepared_data)

    train_splits = [_split_prepared_by_time(p, until=val_start) for p in prepared_train]
    val_splits = [_split_prepared_by_time(p, since=val_start, until=test_start) for p in prepared_val]

    train_pair = _concat_fold_split(train_splits)
    val_pair = _concat_fold_split(val_splits)
    if train_pair is None or val_pair is None:
        raise ValueError(
            "create_or_load_preprocessed_data_spatial: fold produced an empty train or "
            "val split (train_pair is None: %s, val_pair is None: %s) — check "
            "data.val_start/test_start against the stations' available data range." %
            (train_pair is None, val_pair is None)
        )

    logger.info(
        "Spatial fold assembled: %d train samples (< %s) / %d val samples (%s .. %s)",
        len(train_pair[1]), val_start.date(), len(val_pair[1]), val_start.date(),
        test_start.date() if test_start is not None else 'end of data',
    )

    combined_kfolds = [(train_pair, val_pair)]
    return prepared_train + prepared_val, combined_kfolds


def create_or_load_preprocessed_data_spatial(config: Dict,
                                              features: Dict,
                                              model_name: str = None,
                                              force_reprocess: bool = False,
                                              use_cache: bool = True,
                                              cache_dir: str = None) -> Tuple[LazyFoldLoader, str]:
    """cv_mode='spatial' sibling of create_or_load_preprocessed_data() for ONE fold.

    Reads the fold's station roles straight from config['data']['files'] (train-role)
    / config['data']['val_files'] (target-role) — exactly like the temporal path — so
    callers (HPO fold loop, single-fold retrain, test-set eval) all agree on the same
    fold as long as they pass the same files/val_files, with no separate fold-index
    bookkeeping needed. hpo_tft_bc.py's spatial fold loop builds a fresh config copy
    per fold (from geostatistics/spatial_cv.SpatialFold); train_cl_tft_bc.py /
    get_test_results_tft_bc.py instead read a fold-specific config file
    (config_wind_tft_sp_*_foldN.yaml) whose files/val_files are already that fold's
    station split — both paths converge here unmodified, which is what keeps the
    per-fold scaler used at eval identical to the one used at training (the failure
    mode flagged as GNN review finding N1: a fold's retrain and its evaluation must
    fit/recover literally the same scaler).

    Requires config['data']['val_start'] (fixed Train/Val time boundary — see module
    docstring above for why the window is fixed while roles rotate). Always returns a
    LazyFoldLoader with exactly one fold (train, val), so callers that already assert
    `len(lazy_fold_loader) == 1` (train_cl_tft_bc.py) need no change.
    """
    logger = logging.getLogger(__name__)

    if not config['data'].get('val_start'):
        raise ValueError(
            "cv_mode='spatial' requires data.val_start (fixed Train/Val time boundary, "
            "matching the GNN spatial-CV configs) — none set in this config."
        )
    if config.get('hpo', {}).get('min_train_date'):
        logger.warning(
            "cv_mode='spatial': hpo.min_train_date=%s is ignored — the fold's time "
            "window is fixed by data.val_start/data.test_start instead.",
            config['hpo']['min_train_date'],
        )
    if config.get('hpo', {}).get('val_split') not in (None, 1):
        logger.warning(
            "cv_mode='spatial': hpo.val_split=%s is ignored — there is exactly one "
            "fixed val window per fold, no chunk selection.",
            config['hpo'].get('val_split'),
        )

    if not use_cache:
        logger.info("Caching disabled, processing spatial fold directly")
        fold_config = copy.deepcopy(config)
        prepared_datasets, combined_kfolds = _build_spatial_fold_data(fold_config, features, logger)
        return LazyFoldLoader(combined_kfolds), "no_cache"

    cache = DataCache(cache_dir or DEFAULT_CACHE_DIR)
    fold_config = copy.deepcopy(config)
    is_cached, cache_id = cache.is_cached(fold_config, features, model_name)

    if is_cached and not force_reprocess:
        logger.info(f"Spatial fold: found cached data (ID: {cache_id}), loading from cache")
        _, combined_kfolds, _ = cache.load_preprocessed_data(cache_id)
        if combined_kfolds is not None:
            return LazyFoldLoader(combined_kfolds), cache_id
        logger.warning("Spatial fold: no k-folds found in cache, need to reprocess")

    logger.info("Preprocessing spatial fold and saving to cache...")
    prepared_datasets, combined_kfolds = _build_spatial_fold_data(fold_config, features, logger)
    cache_id = cache.save_preprocessed_data(fold_config, features, prepared_datasets, combined_kfolds, model_name)
    logger.info(f"Created and cached spatial fold (ID: {cache_id})")
    return LazyFoldLoader(combined_kfolds), cache_id


# ---------------------------------------------------------------------------
# GNNCache — memory-mapped caching for DCRNN / STGNN2 data
# ---------------------------------------------------------------------------

class GNNCache:
    """
    Disk-based cache for large GNN tensors (ICON-D2, ECMWF, measurements).

    Layout::

        cache_dir/{key}/
            grid_icond2_runs_scaled.npy   # mmap-able  (R × 48 × N_igrid × I2)
            meas_scaled.npy               # mmap-able  (T × N_all × M_meas)
            station_ecmwf_scaled.npy      # mmap-able  (T × N_all × E2)
            ecmwf_nwp_scaled.npy          # mmap-able  (T × N_ecmwf × E2)
            derived.pkl                   # small: coords, run_times, scalers, pairs …

    All workers share the same OS page-cache for the mmap'd arrays, so
    4 workers don't consume 4 × 15 GB RAM.
    """

    # Arrays stored as .npy (mmap-able) — raw/unscaled so each fold can fit
    # its own scaler on the correct training window (no data leakage).
    ARRAY_NAMES = [
        "grid_icond2_runs",
        "meas_raw",
        "station_ecmwf_nwp",
        "ecmwf_nwp",
    ]

    def __init__(self, cache_dir: str = DEFAULT_GNN_CACHE_DIR):
        from pathlib import Path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    # ── Key generation ────────────────────────────────────────────────────────

    @staticmethod
    def make_key(cfg: dict) -> str:
        """Stable MD5 hash over all fields that affect tensor content."""
        import hashlib, json
        data_cfg  = cfg["data"]
        # hpo_{dcrnn,mtgnn,wavenet}.py all wrap their actual feature lists
        # under a "dcrnn" key before calling this (see their _key_cfg
        # construction) so this lookup always finds them; the mtgnn/wavenet
        # fallbacks here are defense-in-depth only, in case a future caller
        # passes its config section under its own name instead.
        dcrnn_cfg = cfg.get("dcrnn", cfg.get("stgnn2", cfg.get("mtgnn", cfg.get("wavenet", {}))))
        key_dict = {
            "path":             data_cfg.get("path", ""),
            "nwp_path":         data_cfg.get("nwp_path", ""),
            "ecmwf_path":       data_cfg.get("ecmwf_path", ""),
            "files":            sorted(str(s) for s in data_cfg.get("files", [])),
            "val_files":        sorted(str(s) for s in data_cfg.get("val_files", [])),
            "test_start":       str(data_cfg.get("test_start", "")),
            "test_end":         str(data_cfg.get("test_end", "")),
            "interpol_path":    str(data_cfg.get("interpol_path", "")),
            "knnimputer_path":  str(data_cfg.get("knnimputer_path", "")),
            "icond2_features":  sorted(dcrnn_cfg.get("icond2_features", [])),
            "ecmwf_features":   sorted(dcrnn_cfg.get("ecmwf_features", [])),
            "meas_features":    sorted(dcrnn_cfg.get("measurement_features", [])),
            "next_n_icond2":    dcrnn_cfg.get("next_n_icond2", 4),
            "next_n_ecmwf":     dcrnn_cfg.get("next_n_ecmwf", 4),
            "icond2_run_hours": sorted(dcrnn_cfg.get("icond2_run_hours", [6, 9, 12, 15])),
            "use_altitude_diff": dcrnn_cfg.get("use_altitude_diff", False),
        }
        raw = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _dir(self, key: str):
        from pathlib import Path
        return self.cache_dir / key

    def _lock_path(self, key: str):
        """Lock file for *key*, deliberately **outside** the key directory.

        Keeping it out of ``{key}/`` means the cache directory contains nothing
        but the payload, and a reader that lists the directory never sees it.
        """
        return self.cache_dir / f".{key}.write.lock"

    def exists(self, key: str) -> bool:
        """Return True only if the cache is fully written (derived.pkl present)."""
        return (self._dir(key) / "derived.pkl").exists()

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(
        self,
        key: str,
        arrays: dict,
        derived: dict,
    ) -> None:
        """
        Persist arrays as .npy (one file each) and everything else as derived.pkl.

        Concurrency contract
        --------------------
        Several HPO workers MISS the same key at the same time and every one of
        them calls ``save``, while other workers already hold ``mmap`` views of
        the very files being written (``grid_icond2_runs.npy`` is 2.7 GB). The
        previous implementation wrote straight to the destination paths with no
        lock, so a reader could observe a half-written array. Three properties
        are enforced here:

        1. **Mutual exclusion.** An ``fcntl.flock`` on ``.{key}.write.lock``
           (next to, not inside, the cache directory) serialises writers.
        2. **Double check.** Once the lock is held, a cache that another worker
           completed in the meantime is left alone — this is what would have
           prevented the 2026-08-03 incident in which a worker without
           ``WEATHER_DB_URL`` overwrote a good cache with 0 m NWP elevations.
        3. **Atomic publication.** Every file is written to
           ``{name}.tmp.<pid>`` inside the cache directory and then moved into
           place with ``os.replace``, which is atomic on POSIX. ``derived.pkl``
           goes last, so ``exists()`` only becomes true once the payload is
           complete. Readers that already hold an ``mmap`` keep the old inode
           and are unaffected; new readers see either the old or the new file,
           never a truncated one.

        Existing cache directories are never renamed, moved or deleted — a
        directory-level ``os.replace`` would fail on a non-empty target anyway,
        and the running campaign reads from exactly these directories.

        Parameters
        ----------
        key     : cache key returned by ``make_key``
        arrays  : dict with keys from ``ARRAY_NAMES``
        derived : dict with small objects (scalers, coords, pairs, timestamps …)
        """
        import fcntl
        import pickle

        p = self._dir(key)
        p.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_path(key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                if self.exists(key):
                    # Another worker finished this key while we were loading the
                    # raw data. The key is a content hash of the config, so its
                    # payload is by construction the same — rewriting it would
                    # only put a complete cache at risk.
                    self.logger.warning(
                        "GNNCache — %s already complete (written by another "
                        "process); skipping the write.", p,
                    )
                    return

                tmp_paths: list = []
                try:
                    for name, arr in arrays.items():
                        final = p / f"{name}.npy"
                        tmp = p / f"{name}.npy.tmp.{os.getpid()}"
                        tmp_paths.append(tmp)
                        self.logger.info(
                            "GNNCache — saving %s %s …", name, arr.shape
                        )
                        # np.save appends '.npy' to a path that lacks it, so
                        # hand it an open file object instead of the path.
                        with open(tmp, "wb") as fh:
                            np.save(fh, arr)
                            fh.flush()
                            os.fsync(fh.fileno())
                        os.replace(tmp, final)
                        tmp_paths.remove(tmp)

                    # derived.pkl last: exists() keys on it, so the cache only
                    # becomes visible once every array is in place.
                    tmp = p / f"derived.pkl.tmp.{os.getpid()}"
                    tmp_paths.append(tmp)
                    with open(tmp, "wb") as fh:
                        pickle.dump(derived, fh, protocol=4)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, p / "derived.pkl")
                    tmp_paths.remove(tmp)
                finally:
                    for leftover in tmp_paths:
                        try:
                            os.unlink(leftover)
                        except OSError:
                            pass
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

        self.logger.info("GNNCache — written to %s", p)

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_arrays(
        self,
        key: str,
        names: list = None,
        mmap: bool = True,
        mmap_overrides: dict = None,
    ) -> dict:
        """
        Load cached arrays.

        Parameters
        ----------
        key            : cache key
        names          : list of array names to load (default: all ARRAY_NAMES)
        mmap           : default mmap mode for all arrays
        mmap_overrides : per-array override, e.g. ``{"grid_icond2_runs": False}``
                         to load a specific array fully into RAM while keeping the
                         rest memory-mapped.  Useful when an array is always copied
                         by downstream code (e.g. StandardScaler.transform) anyway,
                         making mmap only add overhead without sharing benefit.
        """
        p     = self._dir(key)
        names = names or self.ARRAY_NAMES
        overrides = mmap_overrides or {}
        out: dict = {}
        for name in names:
            path = p / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"GNNCache: missing {path}")
            use_mmap = overrides.get(name, mmap)
            mode = "r" if use_mmap else None
            out[name] = np.load(path, mmap_mode=mode)
            self.logger.info(
                "GNNCache — loaded %s %s (mmap=%s)", name, out[name].shape, use_mmap
            )
        return out

    def load_derived(self, key: str) -> dict:
        import pickle
        with open(self._dir(key) / "derived.pkl", "rb") as fh:
            return pickle.load(fh)

    def load(
        self,
        key: str,
        mmap: bool = True,
        mmap_overrides: dict = None,
    ) -> tuple:
        """Convenience: returns (arrays_dict, derived_dict)."""
        return self.load_arrays(key, mmap=mmap, mmap_overrides=mmap_overrides), self.load_derived(key)