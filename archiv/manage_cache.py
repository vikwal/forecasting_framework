#!/usr/bin/env python3
"""
Cache management utility for the forecasting framework.
Provides commands to inspect, clean, and manage cached data.
"""

import os
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any
import hashlib

def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"

def get_cache_info(cache_dir: str = "data_cache") -> List[Dict[str, Any]]:
    """Get information about all cached datasets."""
    cache_info = []

    if not os.path.exists(cache_dir):
        return cache_info

    # Group files by cache ID
    cache_groups = {}
    for file_path in Path(cache_dir).glob("*"):
        if file_path.is_file():
            name = file_path.name
            # Extract cache ID (everything before the first underscore)
            if '_' in name:
                cache_id = name.split('_')[0]
                if cache_id not in cache_groups:
                    cache_groups[cache_id] = []
                cache_groups[cache_id].append(file_path)

    for cache_id, files in cache_groups.items():
        # Load metadata if available
        metadata_file = Path(cache_dir) / f"{cache_id}_metadata.pkl"
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata for {cache_id}: {e}")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in files)

        # Extract useful info from metadata
        config_name = "unknown"
        model_name = "unknown"
        n_folds = 0
        n_datasets = 0

        if metadata:
            if 'config' in metadata and 'data' in metadata['config']:
                data_path = metadata['config']['data'].get('path', '')
                config_name = Path(data_path).name if data_path else "unknown"

                # Extract more config details
                data_config = metadata['config']['data']
                target_col = data_config.get('target_col', 'unknown')
                freq = data_config.get('freq', 'unknown')
                files = data_config.get('files', [])
                n_files_used = len(files) if isinstance(files, list) else 0

                # Model parameters
                model_config = metadata['config'].get('model', {})
                lookback = model_config.get('lookback', 'unknown')
                horizon = model_config.get('horizon', 'unknown')
            else:
                target_col = 'unknown'
                freq = 'unknown'
                n_files_used = 0
                lookback = 'unknown'
                horizon = 'unknown'

            model_name = metadata.get('model_name', 'unknown')
            n_folds = metadata.get('n_folds', 0)
            n_datasets = metadata.get('n_datasets', 0)
        else:
            target_col = 'unknown'
            freq = 'unknown'
            n_files_used = 0
            lookback = 'unknown'
            horizon = 'unknown'

        cache_info.append({
            'cache_id': cache_id,
            'config_name': config_name,
            'model_name': model_name,
            'target_col': target_col,
            'freq': freq,
            'n_files_used': n_files_used,
            'lookback': lookback,
            'horizon': horizon,
            'n_folds': n_folds,
            'n_datasets': n_datasets,
            'total_size': total_size,
            'n_files': len(files),
            'files': [f.name if hasattr(f, 'name') else str(f) for f in files]
        })

    return sorted(cache_info, key=lambda x: x['total_size'], reverse=True)

def list_caches(cache_dir: str = "data_cache", detailed: bool = False):
    """List all cached datasets with their info."""
    cache_info = get_cache_info(cache_dir)

    if not cache_info:
        print(f"No cached data found in '{cache_dir}'")
        return

    if detailed:
        # Detailed view with configuration information
        print(f"\nDetailed cache overview in '{cache_dir}':")
        print("=" * 100)

        total_size = 0
        for i, info in enumerate(cache_info, 1):
            total_size += info['total_size']
            print(f"\n{i}. Cache ID: {info['cache_id']}")
            print(f"   📊 Dataset: {info['config_name']} ({info['n_files_used']} files, {info['n_datasets']} datasets)")
            print(f"   🧠 Model: {info['model_name']} | Target: {info['target_col']} | Freq: {info['freq']}")
            print(f"   📈 Time Series: {info['lookback']} → {info['horizon']} | Folds: {info['n_folds']}")
            print(f"   💾 Size: {format_size(info['total_size'])} ({info['n_files']} cache files)")

        print("\n" + "=" * 100)
        print(f"Summary: {len(cache_info)} caches, total size: {format_size(total_size)}")

    else:
        # Compact view (original format)
        print(f"\nCached datasets in '{cache_dir}':")
        print("=" * 100)
        print(f"{'ID':<12} {'Dataset':<18} {'Model':<8} {'Target':<8} {'L→H':<8} {'Folds':<5} {'Size':<8} {'Data'}")
        print("-" * 100)

        total_size = 0
        for info in cache_info:
            total_size += info['total_size']
            lookback_horizon = f"{info['lookback']}→{info['horizon']}"
            print(f"{info['cache_id'][:12]:<12} "
                  f"{info['config_name'][:18]:<18} "
                  f"{info['model_name'][:8]:<8} "
                  f"{info['target_col'][:8]:<8} "
                  f"{lookback_horizon:<8} "
                  f"{info['n_folds']:<5} "
                  f"{format_size(info['total_size']):<8} "
                  f"{info['n_files_used']}")

        print("-" * 100)
        print(f"Total: {len(cache_info)} caches, {format_size(total_size)}")
        print(f"\nUse 'list --detailed' for more information")

def clean_cache(cache_dir: str = "data_cache", cache_id: str = None, confirm: bool = False):
    """Clean cached data."""
    if cache_id:
        # Clean specific cache
        if not confirm:
            print(f"This will delete cache {cache_id}. Use --confirm to proceed.")
            return

        deleted_files = 0
        deleted_size = 0

        for file_path in Path(cache_dir).glob(f"{cache_id}_*"):
            if file_path.is_file():
                deleted_size += file_path.stat().st_size
                file_path.unlink()
                deleted_files += 1

        if deleted_files > 0:
            print(f"Deleted cache {cache_id}: {deleted_files} files, {format_size(deleted_size)}")
        else:
            print(f"Cache {cache_id} not found")
    else:
        # Clean all caches
        if not confirm:
            print(f"This will delete ALL cached data in '{cache_dir}'. Use --confirm to proceed.")
            return

        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Deleted entire cache directory: {cache_dir}")
        else:
            print(f"Cache directory '{cache_dir}' does not exist")

def show_cache_config(cache_dir: str = "data_cache", cache_id: str = None):
    """Show the original configuration used to create a cache."""
    cache_info = get_cache_info(cache_dir)

    if not cache_id:
        print("Available caches:")
        for info in cache_info:
            print(f"  {info['cache_id'][:12]}: {info['config_name']} + {info['model_name']}")
        print("\nUse: manage_cache.py config <cache_id> to see the full configuration")
        return

    # Find specific cache
    target_cache = None
    for info in cache_info:
        if info['cache_id'].startswith(cache_id):
            target_cache = info
            break

    if not target_cache:
        print(f"Cache '{cache_id}' not found")
        return

    # Load metadata
    metadata_file = Path(cache_dir) / f"{target_cache['cache_id']}_metadata.pkl"
    if not metadata_file.exists():
        print(f"Metadata file not found for cache {cache_id}")
        return

    try:
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        print(f"\nConfiguration for Cache ID: {target_cache['cache_id']}")
        print("=" * 80)

        if 'config' not in metadata:
            print("No configuration data found in metadata")
            return

        config = metadata['config']

        # Display key configuration sections
        print("\n📊 DATA CONFIGURATION:")
        data_config = config.get('data', {})
        print(f"  Path: {data_config.get('path', 'N/A')}")
        print(f"  Files: {data_config.get('files', 'N/A')}")
        print(f"  Target Column: {data_config.get('target_col', 'N/A')}")
        print(f"  Frequency: {data_config.get('freq', 'N/A')}")

        print("\n🧠 MODEL CONFIGURATION:")
        model_config = config.get('model', {})
        print(f"  Model Name: {metadata.get('model_name', 'N/A')}")
        print(f"  Lookback: {model_config.get('lookback', 'N/A')}")
        print(f"  Horizon: {model_config.get('horizon', 'N/A')}")
        print(f"  Output Dim: {model_config.get('output_dim', 'N/A')}")
        print(f"  Step Size: {model_config.get('step_size', 'N/A')}")

        print("\n🔧 FEATURES:")
        features = metadata.get('features', {})
        if features:
            for key, value in features.items():
                if isinstance(value, (list, tuple)) and len(value) > 5:
                    print(f"  {key}: [{len(value)} items]")
                else:
                    print(f"  {key}: {value}")
        else:
            print("  No feature information available")

        print("\n📈 PREPROCESSING PARAMETERS:")
        params = config.get('params', {})
        if params:
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print("  No preprocessing parameters found")

        print("\n📦 CACHE STATISTICS:")
        print(f"  Total Size: {format_size(target_cache['total_size'])}")
        print(f"  Number of Folds: {target_cache['n_folds']}")
        print(f"  Number of Datasets: {target_cache['n_datasets']}")
        print(f"  Files: {target_cache['n_files']}")

        # Hash verification
        print(f"\n🔐 CACHE HASH: {target_cache['cache_id']}")

    except Exception as e:
        print(f"Error reading metadata: {e}")

def inspect_cache(cache_dir: str = "data_cache", cache_id: str = None):
    """Inspect detailed information about a specific cache."""
    cache_info = get_cache_info(cache_dir)

    if cache_id:
        # Find specific cache
        target_cache = None
        for info in cache_info:
            if info['cache_id'].startswith(cache_id):
                target_cache = info
                break

        if not target_cache:
            print(f"Cache '{cache_id}' not found")
            return

        cache_info = [target_cache]

    for info in cache_info:
        print(f"\nCache ID: {info['cache_id']}")
        print(f"Config: {info['config_name']}")
        print(f"Model: {info['model_name']}")
        print(f"Folds: {info['n_folds']}")
        print(f"Datasets: {info['n_datasets']}")
        print(f"Total Size: {format_size(info['total_size'])}")
        print(f"Files ({info['n_files']}):")
        for filename in sorted(info['files']):
            file_path = Path(cache_dir) / filename
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"  {filename}: {format_size(size)}")

        # Load and display metadata
        metadata_file = Path(cache_dir) / f"{info['cache_id']}_metadata.pkl"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                print("\nMetadata:")
                if 'config' in metadata:
                    config = metadata['config']
                    print(f"  Data path: {config.get('data', {}).get('path', 'unknown')}")
                    print(f"  Target: {config.get('data', {}).get('target_col', 'unknown')}")
                    print(f"  Frequency: {config.get('data', {}).get('freq', 'unknown')}")
                    if 'model' in config:
                        print(f"  Lookback: {config['model'].get('lookback', 'unknown')}")
                        print(f"  Horizon: {config['model'].get('horizon', 'unknown')}")
            except Exception as e:
                print(f"  Could not load metadata: {e}")

def main():
    parser = argparse.ArgumentParser(description="Cache management utility")
    parser.add_argument('--cache-dir', default='data_cache', help='Cache directory (default: data_cache)')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List command
    list_parser = subparsers.add_parser('list', help='List all caches')
    list_parser.add_argument('--detailed', action='store_true', help='Show detailed configuration information')

    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect cache details')
    inspect_parser.add_argument('cache_id', nargs='?', help='Cache ID to inspect (optional)')

    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration used to create cache')
    config_parser.add_argument('cache_id', nargs='?', help='Cache ID to show config for (optional)')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean caches')
    clean_parser.add_argument('cache_id', nargs='?', help='Specific cache ID to clean (optional)')
    clean_parser.add_argument('--confirm', action='store_true', help='Confirm deletion')

    args = parser.parse_args()

    if args.command == 'list':
        list_caches(args.cache_dir, getattr(args, 'detailed', False))
    elif args.command == 'inspect':
        inspect_cache(args.cache_dir, getattr(args, 'cache_id', None))
    elif args.command == 'config':
        show_cache_config(args.cache_dir, getattr(args, 'cache_id', None))
    elif args.command == 'clean':
        clean_cache(args.cache_dir, getattr(args, 'cache_id', None), args.confirm)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()