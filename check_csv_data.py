#!/usr/bin/env python3
"""
Check data completeness by analyzing existing CSV files.

This script analyzes the daily_results CSV files to identify
forecast runs with missing data across stations.

Usage:
    python check_csv_data.py
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict


def check_csv_file(csv_path):
    """Check a single CSV file for missing data."""
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        total_forecasts = len(df)
        total_stations = len(df.columns)

        # Count NaN values per forecast (row)
        nan_per_forecast = df.isna().sum(axis=1)
        complete_forecasts = (nan_per_forecast == 0).sum()
        incomplete_forecasts = total_forecasts - complete_forecasts

        # Count NaN values per station (column)
        nan_per_station = df.isna().sum(axis=0)
        complete_stations = (nan_per_station == 0).sum()
        incomplete_stations = total_stations - complete_stations

        # Find most problematic forecasts
        worst_forecasts = nan_per_forecast.nlargest(5)

        # Find most problematic stations
        worst_stations = nan_per_station.nlargest(5)

        return {
            'name': csv_path.name,
            'status': 'OK' if incomplete_forecasts == 0 else 'INCOMPLETE',
            'total_forecasts': total_forecasts,
            'total_stations': total_stations,
            'complete_forecasts': complete_forecasts,
            'incomplete_forecasts': incomplete_forecasts,
            'complete_stations': complete_stations,
            'incomplete_stations': incomplete_stations,
            'worst_forecasts': worst_forecasts.to_dict() if len(worst_forecasts) > 0 else {},
            'worst_stations': worst_stations.to_dict() if len(worst_stations) > 0 else {}
        }

    except Exception as e:
        return {
            'name': csv_path.name,
            'status': 'ERROR',
            'error': str(e),
            'total_forecasts': 0,
            'total_stations': 0,
            'complete_forecasts': 0,
            'incomplete_forecasts': 0,
            'complete_stations': 0,
            'incomplete_stations': 0,
            'worst_forecasts': {},
            'worst_stations': {}
        }


def main():
    """Main function."""

    print("="*80)
    print("CSV DATA COMPLETENESS CHECK")
    print("="*80)
    print()

    # Find all daily_results CSV files
    data_dir = Path('data/test_results')
    csv_files = list(data_dir.glob('daily_results_*.csv'))

    if not csv_files:
        print(f"ERROR: No daily_results_*.csv files found in {data_dir}")
        print("Please run: python get_test_results.py --evaldaily")
        print("         or: python get_test_results.py --localperformance")
        return

    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    print("-"*80)
    print()

    # Check each file
    results = []
    for csv_file in sorted(csv_files):
        print(f"Checking {csv_file.name}...", end=' ', flush=True)

        result = check_csv_file(csv_file)
        results.append(result)

        if result['status'] == 'OK':
            print(f"✓ Complete")
        elif result['status'] == 'INCOMPLETE':
            pct = result['incomplete_forecasts'] / result['total_forecasts'] * 100
            print(f"⚠ {result['incomplete_forecasts']}/{result['total_forecasts']} incomplete forecasts ({pct:.1f}%)")
        else:
            print(f"✗ ERROR: {result.get('error', 'Unknown error')}")

    print()
    print("="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    print()

    for result in results:
        if result['status'] == 'ERROR':
            continue

        print(f"File: {result['name']}")
        print(f"  Forecasts: {result['total_forecasts']} total, "
              f"{result['complete_forecasts']} complete, "
              f"{result['incomplete_forecasts']} incomplete")
        print(f"  Stations:  {result['total_stations']} total, "
              f"{result['complete_stations']} complete, "
              f"{result['incomplete_stations']} incomplete")

        if result['worst_forecasts']:
            print(f"  Worst forecast dates (most missing stations):")
            for date, count in list(result['worst_forecasts'].items())[:3]:
                print(f"    - {date}: {int(count)} stations missing")

        if result['worst_stations']:
            print(f"  Worst stations (most missing forecasts):")
            for station, count in list(result['worst_stations'].items())[:3]:
                print(f"    - Station {station}: {int(count)} forecasts missing")

        print()

    print("="*80)
    print("CHECK COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
