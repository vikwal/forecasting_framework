#!/usr/bin/env python3
"""
Analyze Icon-D2 grid points for a specific station and forecast hour.
Shows distance to station and temporal coverage for each CSV file.
"""

import os
import math
import pandas as pd
from geopy.distance import geodesic
import argparse
from pathlib import Path


def analyze_csv_file(csv_path):
    """Analyze a single CSV file for temporal coverage."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None, None, 0

        df['starttime'] = pd.to_datetime(df['starttime'], utc=True)
        df['timestamp'] = df['starttime'] + pd.to_timedelta(df['forecasttime'], unit='h')

        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        n_rows = len(df)

        return min_date, max_date, n_rows
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, None, 0


def extract_coordinates_from_filename(filename):
    """Extract lat/lon from filename like: 50_15_10_37_ML.csv"""
    parts = filename.replace('_ML.csv', '').split('_')
    if len(parts) >= 4:
        try:
            lat = float(f"{parts[0]}.{parts[1]}")
            lon = float(f"{parts[2]}.{parts[3]}")
            return lat, lon
        except (ValueError, IndexError):
            return None, None
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Analyze Icon-D2 grid points')
    parser.add_argument('--path', type=str,
                       default='/mnt/nas/icon-d2/csv/ML/09/03086',
                       help='Path to the directory with CSV files')
    parser.add_argument('--station-id', type=str, default='03086',
                       help='Station ID for coordinate lookup')
    parser.add_argument('--stations-csv', type=str,
                       default='/home/viktorwalter/Work/DWD/data/stations_list.csv',
                       help='Path to stations master list')
    parser.add_argument('--sort-by', type=str, default='distance',
                       choices=['distance', 'rows', 'filename'],
                       help='Sort results by: distance, rows, or filename')

    args = parser.parse_args()

    # Load station coordinates
    stations_df = pd.read_csv(args.stations_csv)
    station_info = stations_df[stations_df['Stations_id'] == int(args.station_id)]

    if station_info.empty:
        print(f"ERROR: Station {args.station_id} not found in stations list")
        return

    station_lat = station_info['geoBreite'].iloc[0]
    station_lon = station_info['geoLaenge'].iloc[0]

    print(f"Station {args.station_id}: {station_lat:.4f}°N, {station_lon:.4f}°E")
    print(f"Analyzing directory: {args.path}")
    print("=" * 120)

    # Check if directory exists
    if not os.path.exists(args.path):
        print(f"ERROR: Directory {args.path} does not exist!")
        return

    # Get all CSV files
    csv_files = [f for f in os.listdir(args.path) if f.endswith('_ML.csv')]

    if not csv_files:
        print(f"No CSV files found in {args.path}")
        return

    print(f"Found {len(csv_files)} CSV files\n")

    # Analyze each file
    results = []
    for csv_file in csv_files:
        csv_path = os.path.join(args.path, csv_file)

        # Extract coordinates
        grid_lat, grid_lon = extract_coordinates_from_filename(csv_file)

        if grid_lat is None or grid_lon is None:
            print(f"WARNING: Could not extract coordinates from {csv_file}")
            continue

        # Calculate distance
        #distance = geodesic((station_lat, station_lon), (grid_lat, grid_lon)).kilometers
        distance = math.sqrt((grid_lat - station_lat) ** 2 + (grid_lon - station_lon) ** 2)
        # Analyze temporal coverage
        min_date, max_date, n_rows = analyze_csv_file(csv_path)

        results.append({
            'filename': csv_file,
            'lat': grid_lat,
            'lon': grid_lon,
            'distance_km': distance,
            'n_rows': n_rows,
            'min_date': min_date,
            'max_date': max_date,
            'is_empty': n_rows == 0
        })

    # Convert to DataFrame for easy sorting and display
    df_results = pd.DataFrame(results)

    # Sort based on user choice
    if args.sort_by == 'distance':
        df_results = df_results.sort_values('distance_km')
    elif args.sort_by == 'rows':
        df_results = df_results.sort_values('n_rows', ascending=False)
    else:  # filename
        df_results = df_results.sort_values('filename')

    # Display results
    print(f"{'Rank':<5} {'Filename':<25} {'Lat':<8} {'Lon':<8} {'Dist(km)':<10} {'Rows':<8} {'Min Date':<20} {'Max Date':<20} {'Status':<10}")
    print("-" * 120)

    for idx, row in df_results.iterrows():
        rank = idx + 1 if args.sort_by == 'distance' else ''
        status = "EMPTY" if row['is_empty'] else "OK"
        min_date_str = row['min_date'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['min_date']) else 'N/A'
        max_date_str = row['max_date'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['max_date']) else 'N/A'

        print(f"{rank:<5} {row['filename']:<25} {row['lat']:<8.4f} {row['lon']:<8.4f} "
              f"{row['distance_km']:<10.2f} {row['n_rows']:<8} {min_date_str:<20} {max_date_str:<20} {status:<10}")

    # Summary statistics
    print("\n" + "=" * 120)
    print(f"SUMMARY:")
    print(f"  Total files: {len(df_results)}")
    print(f"  Empty files: {df_results['is_empty'].sum()}")
    print(f"  Valid files: {(~df_results['is_empty']).sum()}")
    print(f"  Closest grid point: {df_results.iloc[0]['distance_km']:.2f} km ({df_results.iloc[0]['filename']})")

    # Show first 12 by distance
    print(f"\n  First 12 grid points by distance:")
    for i, row in df_results.head(12).iterrows():
        status = "⚠️  EMPTY" if row['is_empty'] else "✓ OK"
        print(f"    {i+1:2d}. {row['distance_km']:6.2f} km - {row['filename']:<25} ({row['n_rows']:5d} rows) {status}")


if __name__ == '__main__':
    main()
