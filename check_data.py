#!/usr/bin/env python3
"""
Check data completeness across all wind stations.

This script validates that NWP (Known Features) data is complete for each station,
identifying forecast runs with missing or incomplete data that would be skipped during training/evaluation.

Usage:
    python check_data.py
"""

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add utils to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import preprocessing

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(levelname)s: %(message)s'
)


def check_station_data(config_path, global_config):
    """
    Check a single station for data completeness.

    Returns:
        dict: Statistics about missing data
    """
    station_id = config_path.stem.replace('config_wind_', '')

    try:
        with open(config_path, 'r') as f:
            station_config = yaml.safe_load(f)

        # Get features
        features = preprocessing.get_features(config=global_config)

        # Load data
        data_dir = station_config['data']['path']
        dfs = preprocessing.get_data(
            data_dir=data_dir,
            config=station_config,
            freq=global_config['data']['freq'],
            features=features
        )

        if len(dfs) == 0:
            return {
                'station_id': station_id,
                'status': 'ERROR',
                'error': 'No data found',
                'total_forecasts': 0,
                'complete_forecasts': 0,
                'incomplete_forecasts': 0,
                'missing_dates': []
            }

        # Check MultiIndex structure
        for key, df in dfs.items():
            if not isinstance(df.index, pd.MultiIndex) or 'starttime' not in df.index.names:
                return {
                    'station_id': station_id,
                    'status': 'ERROR',
                    'error': 'Data is not in expected MultiIndex format',
                    'total_forecasts': 0,
                    'complete_forecasts': 0,
                    'incomplete_forecasts': 0,
                    'missing_dates': []
                }

            # Get unique starttimes
            starttimes = df.index.get_level_values('starttime').unique()
            total_forecasts = len(starttimes)

            # Expected length per forecast (from config)
            expected_length = global_config['model']['output_dim']

            # Check each forecast for completeness
            incomplete_count = 0
            incomplete_dates = []

            for starttime in starttimes:
                forecast_data = df.xs(starttime, level='starttime')
                actual_length = len(forecast_data)

                if actual_length != expected_length:
                    incomplete_count += 1
                    incomplete_dates.append({
                        'date': str(starttime),
                        'expected': expected_length,
                        'actual': actual_length,
                        'missing': expected_length - actual_length
                    })

            complete_count = total_forecasts - incomplete_count

            return {
                'station_id': station_id,
                'status': 'OK' if incomplete_count == 0 else 'INCOMPLETE',
                'error': None,
                'total_forecasts': total_forecasts,
                'complete_forecasts': complete_count,
                'incomplete_forecasts': incomplete_count,
                'incomplete_dates': incomplete_dates[:5] if incomplete_count > 0 else []  # First 5 examples
            }

    except Exception as e:
        return {
            'station_id': station_id,
            'status': 'ERROR',
            'error': str(e),
            'total_forecasts': 0,
            'complete_forecasts': 0,
            'incomplete_forecasts': 0,
            'incomplete_dates': []
        }


def main():
    """Main function to check all stations."""

    print("="*80)
    print("DATA COMPLETENESS CHECK FOR ALL WIND STATIONS")
    print("="*80)
    print()

    # Load global config
    config_path = Path('configs/config_wind_100.yaml')
    if not config_path.exists():
        print(f"ERROR: Global config not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        global_config = yaml.safe_load(f)

    print(f"Using global config: {config_path}")
    print(f"Expected forecast length: {global_config['model']['output_dim']} timesteps")
    print()

    # Find all station configs
    local_configs_dirs = [Path('configs/wind_50'), Path('configs/wind_100ex50')]
    local_config_files = []
    for config_dir in local_configs_dirs:
        if config_dir.exists():
            local_config_files.extend(list(config_dir.glob('config_wind_*.yaml')))
    local_config_files = sorted(local_config_files)

    print(f"Found {len(local_config_files)} station configs")
    print("-"*80)
    print()

    # Check each station
    results = []
    for i, config_file in enumerate(local_config_files, 1):
        print(f"[{i}/{len(local_config_files)}] Checking station {config_file.stem.replace('config_wind_', '')}...", end=' ', flush=True)

        result = check_station_data(config_file, global_config)
        results.append(result)

        if result['status'] == 'OK':
            print(f"✓ Complete ({result['total_forecasts']} forecasts)")
        elif result['status'] == 'INCOMPLETE':
            print(f"⚠ {result['incomplete_forecasts']}/{result['total_forecasts']} incomplete")
        else:
            print(f"✗ ERROR: {result['error']}")

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()

    # Calculate summary statistics
    total_stations = len(results)
    ok_stations = sum(1 for r in results if r['status'] == 'OK')
    incomplete_stations = sum(1 for r in results if r['status'] == 'INCOMPLETE')
    error_stations = sum(1 for r in results if r['status'] == 'ERROR')

    total_forecasts = sum(r['total_forecasts'] for r in results)
    total_complete = sum(r['complete_forecasts'] for r in results)
    total_incomplete = sum(r['incomplete_forecasts'] for r in results)

    print(f"Stations Summary:")
    print(f"  Total stations:      {total_stations}")
    print(f"  ✓ Complete:          {ok_stations} ({ok_stations/total_stations*100:.1f}%)")
    print(f"  ⚠ Incomplete data:   {incomplete_stations} ({incomplete_stations/total_stations*100:.1f}%)")
    print(f"  ✗ Errors:            {error_stations} ({error_stations/total_stations*100:.1f}%)")
    print()

    print(f"Forecasts Summary:")
    print(f"  Total forecasts:     {total_forecasts}")
    print(f"  ✓ Complete:          {total_complete} ({total_complete/total_forecasts*100:.1f}%)")
    print(f"  ⚠ Incomplete:        {total_incomplete} ({total_incomplete/total_forecasts*100:.1f}%)")
    print()

    # Show stations with most incomplete data
    if incomplete_stations > 0:
        print("-"*80)
        print("STATIONS WITH INCOMPLETE DATA (Top 10):")
        print("-"*80)
        print()

        incomplete_results = [r for r in results if r['status'] == 'INCOMPLETE']
        incomplete_results.sort(key=lambda x: x['incomplete_forecasts'], reverse=True)

        for result in incomplete_results[:10]:
            print(f"Station {result['station_id']}: {result['incomplete_forecasts']}/{result['total_forecasts']} incomplete "
                  f"({result['incomplete_forecasts']/result['total_forecasts']*100:.1f}%)")

            if result['incomplete_dates']:
                print(f"  Example incomplete dates:")
                for date_info in result['incomplete_dates'][:3]:
                    print(f"    - {date_info['date']}: {date_info['actual']}/{date_info['expected']} timesteps "
                          f"({date_info['missing']} missing)")
        print()

    # Show error stations
    if error_stations > 0:
        print("-"*80)
        print("STATIONS WITH ERRORS:")
        print("-"*80)
        print()

        error_results = [r for r in results if r['status'] == 'ERROR']
        for result in error_results:
            print(f"Station {result['station_id']}: {result['error']}")
        print()

    print("="*80)
    print("CHECK COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
