"""
patch_parquets_from_csv.py — Merge missing CSV rows into existing parquet files.

For each CSV under --csv-path (ML/{rh}/{park}/*.csv), checks whether all its
starttimes are already present in the corresponding parquet under --pq-path.
Rows not yet in the parquet are appended and the parquet is rewritten.

Existing parquet data is never removed — only new rows are added.

Usage:
    cd forecasting_framework/
    python misc/patch_parquets_from_csv.py \
        --csv-path /mnt/nvme2/icon-d2/csv/ML \
        --pq-path  /mnt/nvme1/icon-d2/parquet/ML \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("patch")


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    if "starttime" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["starttime"]):
        df["starttime"] = pd.to_datetime(df["starttime"], utc=True)
    return df


def patch(csv_base: Path, pq_base: Path, dry_run: bool) -> None:
    all_csvs = sorted(csv_base.rglob("*_ML.csv"))
    if not all_csvs:
        log.warning("No *_ML.csv files found under %s", csv_base)
        return

    patched = skipped = errors = 0
    rows_added = 0

    for csv_path in all_csvs:
        rel     = csv_path.relative_to(csv_base)
        pq_path = pq_base / rel.with_suffix(".parquet")

        if not pq_path.exists():
            log.warning("Parquet not found — skipping: %s", pq_path)
            continue

        try:
            csv_df = _to_utc(pd.read_csv(csv_path, sep=","))
            pq_df  = _to_utc(pd.read_parquet(pq_path))

            csv_times = set(csv_df["starttime"])
            pq_times  = set(pq_df["starttime"])
            missing   = csv_times - pq_times

            if not missing:
                skipped += 1
                continue

            new_rows = csv_df[csv_df["starttime"].isin(missing)]
            log.info(
                "%s  →  +%d rows (%d missing starttimes)",
                rel, len(new_rows), len(missing),
            )

            if not dry_run:
                merged = pd.concat([pq_df, new_rows], ignore_index=True)
                merged.sort_values(["starttime", "forecasttime", "toplevel"], inplace=True)
                merged.reset_index(drop=True, inplace=True)
                merged.to_parquet(pq_path, engine="pyarrow", compression="snappy")

            patched    += 1
            rows_added += len(new_rows)

        except Exception as exc:
            log.error("Error processing %s: %s", rel, exc)
            errors += 1

    log.info("─" * 60)
    if dry_run:
        log.info("DRY-RUN — nothing written")
    log.info("Files patched  : %d", patched)
    log.info("Files unchanged: %d  (already complete)", skipped)
    log.info("Rows added     : %d", rows_added)
    log.info("Errors         : %d", errors)
    log.info("─" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge missing CSV rows into parquet files")
    ap.add_argument("--csv-path", required=True,
                    help="CSV base dir, e.g. /mnt/nvme2/icon-d2/csv/ML")
    ap.add_argument("--pq-path",  required=True,
                    help="Parquet base dir, e.g. /mnt/nvme1/icon-d2/parquet/ML")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Show what would be done without writing anything")
    args = ap.parse_args()

    patch(Path(args.csv_path), Path(args.pq_path), args.dry_run)


if __name__ == "__main__":
    main()
