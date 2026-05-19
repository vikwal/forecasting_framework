#!/usr/bin/env python3
"""
Convert CSV data files to Parquet format for faster loading.

Handles three data types:
  - Real SCADA measurements  (flat dir,  sep=';')
  - Synthetic power data     (flat dir,  sep=';')
  - ICON-D2 NWP forecasts    (nested ML/{hour}/{park}/*.csv, sep=',')

Parquet files are 5-10x smaller and 10-50x faster to read.

Usage:
    python convert_data_to_parquet.py -c configs/config_trianel_fl.yaml
    python convert_data_to_parquet.py -c configs/config_trianel_fl.yaml --nwp-only
    python convert_data_to_parquet.py -c configs/config_trianel_fl.yaml --force
"""
import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import tools


# ---------------------------------------------------------------------------
# Single-file helpers
# ---------------------------------------------------------------------------

def _convert_one(csv_path: str, sep: str, force: bool, compress: str) -> tuple:
    """Convert one CSV → Parquet. Returns (parquet_path, csv_mb, parquet_mb)."""
    parquet_path = csv_path[:-4] + '.parquet'
    if os.path.exists(parquet_path) and not force:
        return parquet_path, 0.0, 0.0  # skipped

    df = pd.read_csv(csv_path, sep=sep)
    if "starttime" in df.columns and df["starttime"].dtype == object:
        df["starttime"] = pd.to_datetime(df["starttime"], utc=True)
    df.to_parquet(parquet_path, engine='pyarrow', compression=compress)

    csv_mb     = os.path.getsize(csv_path)     / 1024 / 1024
    parquet_mb = os.path.getsize(parquet_path) / 1024 / 1024
    return parquet_path, csv_mb, parquet_mb


def _worker(args):
    """Top-level function so ProcessPoolExecutor can pickle it."""
    csv_path, sep, force, compress = args
    try:
        return _convert_one(csv_path, sep, force, compress)
    except Exception as exc:
        return None, 0.0, 0.0, str(exc)


# ---------------------------------------------------------------------------
# Directory converters
# ---------------------------------------------------------------------------

def convert_flat_directory(data_path: str, sep: str = ';',
                            force: bool = False, compress: str = 'snappy') -> dict:
    """Convert all CSVs in a flat directory (real / synthetic)."""
    if not os.path.isdir(data_path):
        logging.warning(f'Directory not found, skipping: {data_path}')
        return {'converted': 0, 'skipped': 0, 'csv_mb': 0.0, 'parquet_mb': 0.0}

    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    stats = {'converted': 0, 'skipped': 0, 'csv_mb': 0.0, 'parquet_mb': 0.0}

    for csv_file in tqdm(csv_files, desc=f'  {os.path.basename(data_path)}', unit='file'):
        csv_path = os.path.join(data_path, csv_file)
        parquet_path = csv_path[:-4] + '.parquet'
        if os.path.exists(parquet_path) and not force:
            stats['skipped'] += 1
            continue
        try:
            _, csv_mb, parquet_mb = _convert_one(csv_path, sep=sep,
                                                  force=force, compress=compress)
            stats['converted'] += 1
            stats['csv_mb']     += csv_mb
            stats['parquet_mb'] += parquet_mb
        except Exception as exc:
            logging.error(f'  Failed: {csv_file}: {exc}')

    return stats


def convert_nwp_directory(nwp_path: str, force: bool = False,
                           compress: str = 'snappy', workers: int = 8,
                           hours: list = None) -> dict:
    """
    Convert ICON-D2 NWP CSVs in nested structure ML/{hour}/{park_id}/*.csv.
    Uses ProcessPoolExecutor for parallel conversion.
    hours: if given, only process these forecast-hour subdirectories (e.g. ['06', '09']).
    """
    ml_root = os.path.join(nwp_path, 'ML')
    if not os.path.isdir(ml_root):
        logging.warning(f'NWP ML directory not found, skipping: {ml_root}')
        return {'converted': 0, 'skipped': 0, 'csv_mb': 0.0, 'parquet_mb': 0.0}

    # Collect all CSV paths
    all_csv = []
    available_hours = sorted(os.listdir(ml_root))
    selected_hours  = [h for h in available_hours if hours is None or h in hours]
    if hours and not selected_hours:
        logging.warning(f'None of the requested hours {hours} found under {ml_root}. '
                        f'Available: {available_hours}')
        return {'converted': 0, 'skipped': 0, 'csv_mb': 0.0, 'parquet_mb': 0.0}

    for hour_dir in selected_hours:
        hour_path = os.path.join(ml_root, hour_dir)
        if not os.path.isdir(hour_path):
            continue
        for park_dir in sorted(os.listdir(hour_path)):
            park_path = os.path.join(hour_path, park_dir)
            if not os.path.isdir(park_path):
                continue
            for fname in os.listdir(park_path):
                if fname.endswith('.csv'):
                    all_csv.append(os.path.join(park_path, fname))

    total = len(all_csv)
    to_convert = [p for p in all_csv
                  if force or not os.path.exists(p[:-4] + '.parquet')]
    skipped = total - len(to_convert)

    logging.info(f'  NWP: {total} CSV files total, '
                 f'{len(to_convert)} to convert, {skipped} already parquet')

    if not to_convert:
        return {'converted': 0, 'skipped': skipped, 'csv_mb': 0.0, 'parquet_mb': 0.0}

    stats = {'converted': 0, 'skipped': skipped, 'csv_mb': 0.0, 'parquet_mb': 0.0}
    job_args = [(p, ',', force, compress) for p in to_convert]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, a): a[0] for a in job_args}
        with tqdm(total=len(to_convert), desc='  NWP CSVs', unit='file') as pbar:
            for future in as_completed(futures):
                result = future.result()
                if len(result) == 4:          # error tuple
                    logging.error(f'  Failed {futures[future]}: {result[3]}')
                else:
                    _, csv_mb, parquet_mb = result
                    if csv_mb > 0:            # was actually converted (not skipped)
                        stats['converted'] += 1
                        stats['csv_mb']     += csv_mb
                        stats['parquet_mb'] += parquet_mb
                pbar.update(1)

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert Trianel CSV data to Parquet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-c', '--config', default=None,
                        help='Path to YAML config (e.g. configs/config_trianel_fl.yaml)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing parquet files')
    parser.add_argument('--compress', default='snappy',
                        choices=['snappy', 'gzip', 'brotli'],
                        help='Parquet compression (default: snappy)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel workers for NWP conversion (default: 8)')
    parser.add_argument('--nwp-only', action='store_true',
                        help='Convert only NWP files (skip real/synthetic)')
    parser.add_argument('--skip-nwp', action='store_true',
                        help='Skip NWP conversion (only real/synthetic)')
    parser.add_argument('--hours', nargs='+', default=None,
                        help='Only convert these NWP forecast-hour dirs, e.g. --hours 06 09')
    parser.add_argument('--nwp-path', default=None,
                        help='Override nwp_path from config (e.g. /mnt/nvme2/icon-d2/csv)')
    args = parser.parse_args()

    if args.config is None and args.nwp_path is None:
        parser.error('Provide --config or --nwp-path (or both)')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    real_path = synth_path = path_fallback = nwp_path = None
    if args.config:
        config        = tools.load_config(args.config)
        real_path     = config['data'].get('real_data_path')
        synth_path    = config['data'].get('synthetic_data_path')
        path_fallback = config['data'].get('path')
        nwp_path      = config['data'].get('nwp_path')

    if args.nwp_path:
        nwp_path = args.nwp_path

    logging.info('Paths:')
    logging.info(f'  Real SCADA:        {real_path}')
    logging.info(f'  Synthetic:         {synth_path}')
    if path_fallback and path_fallback != synth_path:
        logging.info(f'  Synthetic (path):  {path_fallback}')
    logging.info(f'  NWP (ICON-D2):     {nwp_path}')
    logging.info(f'  Compression: {args.compress}  |  Workers: {args.workers}')
    if args.force:
        logging.warning('Force mode — existing parquet files will be overwritten')

    all_stats = {}

    if not args.nwp_only:
        logging.info('\nConverting real SCADA data...')
        all_stats['real'] = convert_flat_directory(
            real_path, sep=';', force=args.force, compress=args.compress)

        logging.info('\nConverting synthetic data...')
        all_stats['synth'] = convert_flat_directory(
            synth_path, sep=';', force=args.force, compress=args.compress)

        if path_fallback and path_fallback != synth_path:
            logging.info('\nConverting synthetic data (data.path fallback)...')
            all_stats['synth_path'] = convert_flat_directory(
                path_fallback, sep=';', force=args.force, compress=args.compress)

    if not args.skip_nwp:
        logging.info('\nConverting NWP (ICON-D2) data...')
        if args.hours:
            logging.info(f'  Hour filter: {args.hours}')
        all_stats['nwp'] = convert_nwp_directory(
            nwp_path, force=args.force, compress=args.compress, workers=args.workers,
            hours=args.hours)

    # Summary
    logging.info('\n' + '=' * 65)
    logging.info('SUMMARY')
    logging.info('=' * 65)
    total_converted = total_csv_mb = total_parquet_mb = 0
    for name, s in all_stats.items():
        logging.info(f'{name.upper():10s}  converted={s["converted"]:4d}  '
                     f'skipped={s["skipped"]:4d}', )
        if s['converted'] > 0:
            ratio = s['csv_mb'] / s['parquet_mb'] if s['parquet_mb'] > 0 else 0
            logging.info(f'            {s["csv_mb"]:.0f} MB CSV → '
                         f'{s["parquet_mb"]:.0f} MB Parquet  ({ratio:.1f}x smaller)')
        total_converted += s['converted']
        total_csv_mb    += s['csv_mb']
        total_parquet_mb += s['parquet_mb']

    if total_converted > 0:
        logging.info(f'\nTotal: {total_converted} files, '
                     f'{total_csv_mb:.0f} MB → {total_parquet_mb:.0f} MB  '
                     f'(saved {total_csv_mb - total_parquet_mb:.0f} MB)')
    else:
        logging.info('Nothing to convert — all files already in Parquet.')
    logging.info('=' * 65)


if __name__ == '__main__':
    main()
