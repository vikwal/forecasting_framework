#!/usr/bin/env python3
"""
Cache management utility for the forecasting framework.
Provides commands to inspect, clean, and manage cached preprocessed data.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_cache import DataCache
from utils import tools


def list_cached_data(cache_dir: str = "data_cache"):
    """List all cached datasets."""
    cache = DataCache(cache_dir)

    if not os.path.exists(cache_dir):
        print(f"Cache directory '{cache_dir}' does not exist.")
        return

    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('_metadata.pkl')]

    if not cache_files:
        print("No cached data found.")
        return

    print(f"Found {len(cache_files)} cached datasets in '{cache_dir}':")
    print("-" * 80)

    for metadata_file in sorted(cache_files):
        cache_id = metadata_file.replace('_metadata.pkl', '')
        paths = cache.get_cache_paths(cache_id)

        try:
            import pickle
            with open(paths['metadata'], 'rb') as f:
                metadata = pickle.load(f)

            # Calculate file sizes
            total_size = 0
            file_sizes = {}
            for name, path in paths.items():
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    file_sizes[name] = size
                    total_size += size

            print(f"Cache ID: {cache_id}")
            print(f"  Data path: {metadata['config']['data']['path']}")
            print(f"  Files: {metadata['config']['data'].get('files', 'N/A')}")
            print(f"  Model: {metadata['config']['model']['lookback']}→{metadata['config']['model']['horizon']} "
                  f"(step: {metadata['config']['model'].get('step_size', 1)})")
            print(f"  Datasets: {metadata.get('n_datasets', 'N/A')}")
            print(f"  Folds: {metadata.get('n_folds', 'N/A')}")
            print(f"  Total size: {total_size / (1024**2):.1f} MB")

            for name, size in file_sizes.items():
                print(f"    {name}: {size / (1024**2):.1f} MB")
            print()

        except Exception as e:
            print(f"Cache ID: {cache_id} (Error reading metadata: {e})")
            print()


def clean_cache(cache_dir: str = "data_cache", cache_id: str = None, all_cache: bool = False):
    """Clean cached data."""
    cache = DataCache(cache_dir)

    if all_cache:
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Removed all cached data from '{cache_dir}'")
        else:
            print(f"Cache directory '{cache_dir}' does not exist.")
    elif cache_id:
        paths = cache.get_cache_paths(cache_id)
        removed_files = []

        for name, path in paths.items():
            if os.path.exists(path):
                os.remove(path)
                removed_files.append(name)

        if removed_files:
            print(f"Removed cache {cache_id}: {', '.join(removed_files)}")
        else:
            print(f"Cache {cache_id} not found or already removed.")
    else:
        print("Please specify either --cache-id or --all")


def cache_info(cache_dir: str = "data_cache", cache_id: str = None):
    """Show detailed information about cached data."""
    cache = DataCache(cache_dir)

    if not cache_id:
        print("Please specify --cache-id")
        return

    paths = cache.get_cache_paths(cache_id)

    if not os.path.exists(paths['metadata']):
        print(f"Cache {cache_id} not found.")
        return

    try:
        import pickle
        import numpy as np

        # Load metadata
        with open(paths['metadata'], 'rb') as f:
            metadata = pickle.load(f)

        print(f"Cache ID: {cache_id}")
        print(f"Configuration:")
        print(f"  Data path: {metadata['config']['data']['path']}")
        print(f"  Files: {metadata['config']['data'].get('files', [])}")
        print(f"  Frequency: {metadata['config']['data']['freq']}")
        print(f"  Target: {metadata['config']['data']['target_col']}")
        print(f"  Model lookback: {metadata['config']['model']['lookback']}")
        print(f"  Model horizon: {metadata['config']['model']['horizon']}")
        print(f"  Step size: {metadata['config']['model'].get('step_size', 1)}")
        print(f"  Output dim: {metadata['config']['model']['output_dim']}")

        print(f"\nData info:")
        print(f"  Datasets: {metadata.get('n_datasets', 'N/A')}")
        print(f"  K-folds: {metadata.get('n_folds', 'N/A')}")

        print(f"\nFeatures:")
        features = metadata.get('features', {})
        for feat_type, feat_list in features.items():
            print(f"  {feat_type}: {feat_list}")

        # Load and inspect fold data if available
        if os.path.exists(paths['combined_kfolds']):
            print(f"\nFold details:")
            combined_kfolds = np.load(paths['combined_kfolds'], allow_pickle=True, mmap_mode='r')

            for i in range(min(3, len(combined_kfolds))):
                fold_data = combined_kfolds[i]
                (X_train, y_train), (X_val, y_val) = fold_data

                print(f"  Fold {i}:")
                print(f"    Train samples: {len(y_train)}")
                print(f"    Val samples: {len(y_val)}")

                if isinstance(X_train, dict):
                    print(f"    Features: {list(X_train.keys())}")
                    for key, arr in X_train.items():
                        print(f"      {key}: {arr.shape}")
                else:
                    print(f"    Input shape: {X_train.shape}")

            if len(combined_kfolds) > 3:
                print(f"    ... and {len(combined_kfolds) - 3} more folds")

    except Exception as e:
        print(f"Error reading cache data: {e}")


def main():
    parser = argparse.ArgumentParser(description="Cache management for forecasting framework")
    parser.add_argument('--cache-dir', default='data_cache', help='Cache directory path')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List command
    list_parser = subparsers.add_parser('list', help='List all cached data')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean cached data')
    clean_group = clean_parser.add_mutually_exclusive_group(required=True)
    clean_group.add_argument('--cache-id', help='Specific cache ID to remove')
    clean_group.add_argument('--all', action='store_true', help='Remove all cached data')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show detailed info about cached data')
    info_parser.add_argument('--cache-id', required=True, help='Cache ID to inspect')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if args.command == 'list':
        list_cached_data(args.cache_dir)
    elif args.command == 'clean':
        clean_cache(args.cache_dir, args.cache_id, args.all)
    elif args.command == 'info':
        cache_info(args.cache_dir, args.cache_id)


if __name__ == '__main__':
    main()