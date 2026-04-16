import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

def convert_single_file(args):
    in_path, out_path, overwrite = args
    if out_path.exists() and not overwrite:
        return True # Skip already converted
    try:
        # Read CSV with exact date parsing we use in training
        df = pd.read_csv(in_path, parse_dates=["starttime"])

        # Ensure out directory exists (could be created concurrently, so exist_ok=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to parquet
        df.to_parquet(out_path, engine="pyarrow", index=False)
        return True
    except Exception as e:
        print(f"Error processing {in_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert ICON-D2 ML CSVs to Parquet format.")
    parser.add_argument("--src", type=str, default="/mnt/lambda1/nvme2/icon-d2/csv", help="Source directory containing the ML csvs")
    parser.add_argument("--dst", type=str, default="/mnt/lambda1/nvme1/icond-d2/parquet", help="Target directory for parquet files")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parquet files")

    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        print(f"Source directory {src_dir} does not exist.")
        return

    print(f"Scanning for CSV files in {src_dir}...")

    # We only care about the ML directory and *_ML.csv files
    ml_src_base = src_dir / "ML"

    if not ml_src_base.exists():
        print(f"Could not find ML base directory at {ml_src_base}")
        return

    csv_files = list(ml_src_base.rglob("*_ML.csv"))

    print(f"Found {len(csv_files)} CSV files. Preparing conversion tasks...")

    tasks = []
    for fpath in csv_files:
        # Compute relative path to ml_src_base
        rel_path = fpath.relative_to(ml_src_base)
        # Create output path with .parquet suffix
        out_path = dst_dir / "ML" / rel_path.with_suffix(".parquet")
        tasks.append((fpath, out_path, args.overwrite))

    print(f"Starting conversion with {args.workers} workers...")

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_single_file, task): task for task in tasks}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting CSV to Parquet"):
            if fut.result():
                success_count += 1
            else:
                fail_count += 1

    print("\n--- Conversion Summary ---")
    print(f"Successfully converted: {success_count} files")
    print(f"Failed conversions: {fail_count} files")
    print(f"Target directory: {dst_dir / 'ML'}")

    print("\nNOTE: Remember to update your config YAML to point to the new parquet directory!")
    print("      And change train_stgnn2.py logic to use pd.read_parquet() and search for *.parquet files.")

if __name__ == "__main__":
    main()
