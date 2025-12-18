"""
Data caching utilities für efficient multi-GPU training.
Implements memory-mapped loading to avoid duplicating preprocessed data across processes.
"""
import os
import pickle
import hashlib
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from . import preprocessing, tools, hpo


class DataCache:
    """
    Manages disk-based caching of preprocessed data with memory-mapped loading.
    """

    def __init__(self, cache_dir: str = "data_cache"):
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
            'data_path': config['data']['path'],
            'files': sorted(config['data'].get('files', [])),
            'freq': config['data']['freq'],
            'target_col': config['data']['target_col'],
            'features': features,
            'next_n_grid_points': config['params']['next_n_grid_points'], # newly added 15.12.25
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


def create_or_load_preprocessed_data(config: Dict,
                                   features: Dict,
                                   model_name: str = None,
                                   force_reprocess: bool = False,
                                   use_cache: bool = True) -> Tuple[LazyFoldLoader, str]:
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

    cache = DataCache()

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

        # Load raw data
        dfs = preprocessing.get_data(data_dir=data_dir,
                                   config=config,
                                   freq=freq,
                                   features=features)

        # Preprocess each dataset
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

        # Save to cache
        cache_id = cache.save_preprocessed_data(config, features, prepared_datasets, combined_kfolds, model_name)

        # Create lazy loader
        lazy_loader = LazyFoldLoader(combined_kfolds)
        logger.info(f"Created and cached {len(lazy_loader)} folds")

        return lazy_loader, cache_id