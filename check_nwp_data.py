#!/usr/bin/env python3
"""
Check NWP data completeness for all wind stations.

This script validates raw ICON-D2 NWP CSV files for each station,
using haversine distance to find the nearest grid points.

Usage:
    python check_nwp_data.py [--forecast-hour 06]
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in kilometers."""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def load_station_coordinates():
    """
    Load station coordinates from wind_parameter.csv.

    Returns:
        dict: {station_id: (lat, lon)}
    """
    csv_path = Path('data/wind_parameter.csv')
    df = pd.read_csv(csv_path, sep=';')

    coords = {}
    for _, row in df.iterrows():
        station_id = str(row['park_id']).zfill(5)
        coords[station_id] = (row['latitude'], row['longitude'])

    return coords


def find_nearest_grid_point(station_lat, station_lon, nwp_base_path):
    """
    Find nearest NWP grid point for a station.

    Returns:
        tuple: (csv_filename, distance_km) or (None, None) if not found
    """
    if not os.path.exists(nwp_base_path):
        return None, None

    csv_files = [f for f in os.listdir(nwp_base_path) if f.endswith('_ML.csv')]

    if not csv_files:
        return None, None

    file_distances = []
    for csv_file in csv_files:
        parts = csv_file.replace('_ML.csv', '').split('_')
        if len(parts) >= 4:
            try:
                grid_lat = float(f"{parts[0]}.{parts[1]}")
                grid_lon = float(f"{parts[2]}.{parts[3]}")
                distance = haversine_distance(station_lat, station_lon, grid_lat, grid_lon)
                file_distances.append((csv_file, distance))
            except (ValueError, IndexError):
                continue

    if not file_distances:
        return None, None

    file_distances.sort(key=lambda x: x[1])
    return file_distances[0]


def check_csv_completeness(csv_path, expected_start='2025-08-01', expected_end='2025-10-31'):
    """Check completeness of a single NWP CSV file."""
    try:
        df = pd.read_csv(csv_path)

        if 'starttime' not in df.columns or 'forecasttime' not in df.columns:
            return {
                'status': 'ERROR',
                'error': f'Missing required columns. Found: {list(df.columns)}',
                'total_forecasts': 0,
                'complete_forecasts': 0,
                'incomplete_forecasts': 0
            }

        df['starttime'] = pd.to_datetime(df['starttime'], utc=True)
        start_date = pd.Timestamp(expected_start, tz='UTC')
        end_date = pd.Timestamp(expected_end, tz='UTC')
        df_filtered = df[(df['starttime'] >= start_date) & (df['starttime'] <= end_date)]

        # Group by starttime and count unique forecast hours
        grouped = df_filtered.groupby('starttime')['forecasttime'].nunique()
        expected_length = 49  # 0-48 hours inclusive

        complete_forecasts = (grouped == expected_length).sum()
        incomplete_forecasts = len(grouped) - complete_forecasts
        incomplete_dates = grouped[grouped != expected_length].to_dict()

        return {
            'status': 'OK' if incomplete_forecasts == 0 else 'INCOMPLETE',
            'error': None,
            'total_forecasts': len(grouped),
            'complete_forecasts': complete_forecasts,
            'incomplete_forecasts': incomplete_forecasts,
            'incomplete_dates': incomplete_dates
        }

    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e),
            'total_forecasts': 0,
            'complete_forecasts': 0,
            'incomplete_forecasts': 0
        }


def check_station(station_id, station_coords, nwp_base_dir, forecast_hour):
    """Check NWP data for a single station."""
    try:
        if station_id not in station_coords:
            return {
                'station_id': station_id,
                'status': 'ERROR',
                'error': 'Station not found in wind_parameter.csv',
                'grid_distance_km': None,
                'total_forecasts': 0,
                'complete_forecasts': 0,
                'incomplete_forecasts': 0
            }

        station_lat, station_lon = station_coords[station_id]
        nwp_path = nwp_base_dir / forecast_hour / station_id
        csv_file, distance = find_nearest_grid_point(station_lat, station_lon, nwp_path)

        if csv_file is None:
            return {
                'station_id': station_id,
                'status': 'ERROR',
                'error': 'No NWP files found',
                'grid_distance_km': None,
                'total_forecasts': 0,
                'complete_forecasts': 0,
                'incomplete_forecasts': 0
            }

        csv_path = nwp_path / csv_file
        result = check_csv_completeness(csv_path)

        return {
            'station_id': station_id,
            'status': result['status'],
            'error': result.get('error'),
            'grid_distance_km': round(distance, 2),
            'csv_file': csv_file,
            'total_forecasts': result['total_forecasts'],
            'complete_forecasts': result['complete_forecasts'],
            'incomplete_forecasts': result['incomplete_forecasts'],
            'incomplete_dates': result.get('incomplete_dates', {})
        }

    except Exception as e:
        return {
            'station_id': station_id,
            'status': 'ERROR',
            'error': str(e),
            'grid_distance_km': None,
            'total_forecasts': 0,
            'complete_forecasts': 0,
            'incomplete_forecasts': 0
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Check NWP data completeness')
    parser.add_argument('--forecast-hour', default='06', choices=['06', '09', '12', '15'],
                       help='Forecast hour to check (default: 06)')
    args = parser.parse_args()

    print("="*80)
    print(f"NWP DATA COMPLETENESS CHECK (Forecast Hour: {args.forecast_hour})")
    print("="*80)
    print()

    nwp_base_dir = Path('/mnt/nas/icon-d2/csv/ML')

    print("Loading station coordinates from wind_parameter.csv...")
    station_coords = load_station_coordinates()
    all_station_ids = sorted(station_coords.keys())

    print(f"Found {len(all_station_ids)} stations")
    print(f"NWP data directory: {nwp_base_dir}")
    print("-"*80)
    print()

    results = []
    for i, station_id in enumerate(all_station_ids, 1):
        print(f"[{i}/{len(all_station_ids)}] Checking station {station_id}...", end=' ', flush=True)

        result = check_station(station_id, station_coords, nwp_base_dir, args.forecast_hour)
        results.append(result)

        if result['status'] == 'OK':
            print(f"✓ Complete ({result['total_forecasts']} forecasts, {result['grid_distance_km']}km)")
        elif result['status'] == 'INCOMPLETE':
            print(f"⚠ {result['incomplete_forecasts']}/{result['total_forecasts']} incomplete ({result['grid_distance_km']}km)")
        else:
            print(f"✗ ERROR: {result['error']}")

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()

    total_stations = len(results)
    ok_stations = sum(1 for r in results if r['status'] == 'OK')
    incomplete_stations = sum(1 for r in results if r['status'] == 'INCOMPLETE')
    error_stations = sum(1 for r in results if r['status'] == 'ERROR')

    total_forecasts = sum(r['total_forecasts'] for r in results)
    total_complete = sum(r['complete_forecasts'] for r in results)
    total_incomplete = sum(r['incomplete_forecasts'] for r in results)

    print(f"Stations:")
    print(f"  Total:         {total_stations}")
    print(f"  ✓ Complete:    {ok_stations} ({ok_stations/total_stations*100:.1f}%)")
    print(f"  ⚠ Incomplete:  {incomplete_stations} ({incomplete_stations/total_stations*100:.1f}%)")
    print(f"  ✗ Errors:      {error_stations} ({error_stations/total_stations*100:.1f}%)")
    print()

    if total_forecasts > 0:
        print(f"Forecasts:")
        print(f"  Total:         {total_forecasts}")
        print(f"  ✓ Complete:    {total_complete} ({total_complete/total_forecasts*100:.1f}%)")
        print(f"  ⚠ Incomplete:  {total_incomplete} ({total_incomplete/total_forecasts*100:.1f}%)")
        print()

    if incomplete_stations > 0:
        print("-"*80)
        print("STATIONS WITH MOST INCOMPLETE DATA (Top 10):")
        print("-"*80)
        print()

        incomplete_results = [r for r in results if r['status'] == 'INCOMPLETE']
        incomplete_results.sort(key=lambda x: x['incomplete_forecasts'], reverse=True)

        for result in incomplete_results[:10]:
            pct = result['incomplete_forecasts'] / result['total_forecasts'] * 100
            print(f"Station {result['station_id']}: {result['incomplete_forecasts']}/{result['total_forecasts']} incomplete ({pct:.1f}%)")
            print(f"  Grid distance: {result['grid_distance_km']}km")
            print(f"  CSV file: {result.get('csv_file', 'N/A')}")

            if result.get('incomplete_dates'):
                print(f"  Example incomplete dates:")
                for date, count in list(result['incomplete_dates'].items())[:3]:
                    print(f"    - {date}: {count}/48 timesteps ({48-count} missing)")
        print()

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
