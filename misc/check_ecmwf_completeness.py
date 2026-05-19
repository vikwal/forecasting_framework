"""
check_ecmwf_completeness.py — Audit ECMWF parquet files and/or DB table for completeness.

Parquet checks (per file):
  1. Missing run timestamps — expected every 12 h at 00:00 and 12:00 UTC
  2. Incomplete runs        — fewer rows than expected (n_grid_points × n_forecasttime_steps)
  3. NaN values            — per-starttime, per-column NaN row count

Database checks (table ecmwf_wind_sl via ECMWF_WIND_SL_URL):
  1. Missing run timestamps — same logic as parquet
  2. Incomplete runs        — fewer than expected rows per starttime
  3. NULL values            — per-starttime, per-column NULL row count

Output: JSON
  {
    "parquet": {
      "<file>": {
        "expected_rows_per_run": 44022,
        "missing":    ["2023-07-25 00:00:00+00:00", ...],
        "incomplete": {"2024-01-03 00:00:00+00:00": 43000, ...},
        "nan_runs":   {"2024-01-05 12:00:00+00:00": {"u_wind10m": 3, ...}, ...}
      }
    },
    "database": {
      "table": "ecmwf_wind_sl",
      "expected_rows_per_run": 44022,
      "missing":    [...],
      "incomplete": {...},
      "null_runs":  {"2024-01-05 12:00:00+00:00": {"dew_point_2m": 3, ...}, ...}
    }
  }

Usage:
    cd forecasting_framework/
    # Parquet only
    python misc/check_ecmwf_completeness.py \
        --path /mnt/nvme1/ecmwf/parquet/ecmwf_wind_sl_full_prefect.parquet \
        [--cutoff 2025-10-31] [--output results/ecmwf_completeness.json]

    # DB only  (requires ECMWF_WIND_SL_URL env var)
    python misc/check_ecmwf_completeness.py --no-parquet

    # Both
    python misc/check_ecmwf_completeness.py \
        --path /mnt/nvme1/ecmwf/parquet \
        [--cutoff 2025-10-31]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import psycopg2
from tqdm import tqdm

RUN_HOURS = [0, 12]  # ECMWF runs twice daily (00 UTC and 12 UTC)

# Data columns in ecmwf_wind_sl (DB uses underscore before number suffix)
_DB_DATA_COLS = [
    "dew_point_2m", "temp_2m", "specific_rho",
    "u_wind_10m", "v_wind_10m",
    "u_wind_100m", "v_wind_100m",
    "u_wind_200m", "v_wind_200m",
    "friction_velocity",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ecmwf_completeness")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _expected_timestamps(
    first_ts: pd.Timestamp,
    last_ts: pd.Timestamp,
    run_hours: list[int],
) -> set[pd.Timestamp]:
    """Return all expected run timestamps between first and last (inclusive)."""
    result: set[pd.Timestamp] = set()
    for rh in sorted(run_hours):
        rng = pd.date_range(
            start=first_ts.normalize() + pd.Timedelta(hours=rh),
            end=last_ts,
            freq="1D",
            tz="UTC",
        )
        result.update(rng.tolist())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Parquet audit
# ─────────────────────────────────────────────────────────────────────────────

def _audit_parquet_file(
    fpath: Path,
    run_hours: list[int],
    cutoff: pd.Timestamp | None,
) -> dict | None:
    """Return audit dict for one parquet file, or None if clean."""
    log.info("Loading %s …", fpath.name)
    df = pd.read_parquet(fpath)

    if df.empty:
        log.warning("%s is empty", fpath.name)
        return None

    data_cols = [
        c for c in df.columns
        if c not in ("starttime", "forecasttime", "grid_lat", "grid_lon", "valid_time")
    ]

    if cutoff is not None:
        df = df[df["starttime"] <= cutoff]
    df = df[df["starttime"].dt.hour.isin(run_hours)]
    if df.empty:
        return None

    counts = df.groupby("starttime").size()
    expected_rows = int(counts.mode().iloc[0])

    missing = sorted(
        str(ts)
        for ts in _expected_timestamps(counts.index.min(), counts.index.max(), run_hours)
        if ts not in set(counts.index.tolist())
    )
    incomplete = {str(ts): int(n) for ts, n in counts.items() if n != expected_rows}

    log.info("Checking NaN values in %s …", fpath.name)
    nan_runs: dict[str, dict[str, int]] = {}
    for ts, grp in tqdm(df.groupby("starttime"), desc="NaN scan", leave=False):
        if counts.get(ts, 0) != expected_rows:
            continue
        nan_counts = {
            col: int(grp[col].isna().sum())
            for col in data_cols
            if grp[col].isna().any()
        }
        if nan_counts:
            nan_runs[str(ts)] = nan_counts

    if not missing and not incomplete and not nan_runs:
        return None

    return {
        "expected_rows_per_run": expected_rows,
        "missing":    missing,
        "incomplete": incomplete,
        "nan_runs":   nan_runs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Database audit
# ─────────────────────────────────────────────────────────────────────────────

def _audit_db(
    db_url: str,
    run_hours: list[int],
    cutoff: pd.Timestamp | None,
) -> dict:
    """Audit ecmwf_wind_sl table for missing, incomplete, and NULL-containing runs."""
    log.info("Connecting to ECMWF DB …")
    p = urlparse(db_url)
    conn = psycopg2.connect(
        host=p.hostname, port=p.port,
        database=p.path[1:],
        user=p.username, password=p.password,
    )

    null_exprs = ",\n        ".join(
        f"COUNT(*) - COUNT({col}) AS null_{col}"
        for col in _DB_DATA_COLS
    )
    cutoff_clause = f"WHERE starttime <= '{cutoff}'" if cutoff else ""

    query = f"""
        SELECT starttime, COUNT(*) AS cnt,
        {null_exprs}
        FROM ecmwf_wind_sl
        {cutoff_clause}
        GROUP BY starttime
        ORDER BY starttime
    """

    log.info("Running DB aggregate query (may take a moment) …")
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    conn.close()

    if not rows:
        log.warning("No rows returned from ecmwf_wind_sl")
        return {
            "table": "ecmwf_wind_sl",
            "expected_rows_per_run": 0,
            "missing": [], "incomplete": {}, "null_runs": {},
        }

    # Build per-starttime result
    col_names = ["starttime", "cnt"] + [f"null_{c}" for c in _DB_DATA_COLS]
    counts: dict[pd.Timestamp, int] = {}
    null_data: dict[pd.Timestamp, dict[str, int]] = {}
    for row in rows:
        record = dict(zip(col_names, row))
        ts = pd.Timestamp(record["starttime"]).tz_convert("UTC")
        counts[ts] = int(record["cnt"])
        null_data[ts] = {
            col: int(record[f"null_{col}"])
            for col in _DB_DATA_COLS
            if int(record[f"null_{col}"]) > 0
        }

    # Filter to requested run hours
    counts   = {ts: n for ts, n in counts.items()   if ts.hour in run_hours}
    null_data = {ts: v for ts, v in null_data.items() if ts.hour in run_hours}

    if not counts:
        return {
            "table": "ecmwf_wind_sl",
            "expected_rows_per_run": 0,
            "missing": [], "incomplete": {}, "null_runs": {},
        }

    # Expected rows = modal count
    import statistics
    expected_rows = statistics.mode(counts.values())

    first_ts = min(counts)
    last_ts  = max(counts)
    expected_set = _expected_timestamps(first_ts, last_ts, run_hours)
    observed_set = set(counts.keys())

    missing = sorted(str(ts) for ts in expected_set if ts not in observed_set)
    incomplete = {
        str(ts): int(n)
        for ts, n in counts.items()
        if n != expected_rows
    }
    null_runs = {
        str(ts): v
        for ts, v in null_data.items()
        if v and counts.get(ts, 0) == expected_rows
    }

    log.info(
        "DB audit done — %d starttimes, %d missing, %d incomplete, %d with NULLs",
        len(counts), len(missing), len(incomplete), len(null_runs),
    )

    return {
        "table": "ecmwf_wind_sl",
        "expected_rows_per_run": expected_rows,
        "missing":    missing,
        "incomplete": incomplete,
        "null_runs":  null_runs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path", dest="paths", action="append", default=[],
        metavar="PATH",
        help="Parquet file or directory (can be given multiple times)",
    )
    ap.add_argument("--no-parquet", action="store_true",
                    help="Skip parquet audit even if --path is given")
    ap.add_argument("--no-db",      action="store_true",
                    help="Skip database audit")
    ap.add_argument(
        "--run-hours", type=int, nargs="+", default=RUN_HOURS,
        help=f"Expected run hours UTC (default: {RUN_HOURS})",
    )
    ap.add_argument("--cutoff", default=None, help="Ignore runs after YYYY-MM-DD")
    ap.add_argument("--output", default="results/ecmwf_completeness.json")
    args = ap.parse_args()

    cutoff = pd.Timestamp(args.cutoff, tz="UTC") if args.cutoff else None
    output: dict = {}

    # ── Parquet audit ──────────────────────────────────────────────────────────
    if not args.no_parquet and args.paths:
        files: list[Path] = []
        for p in args.paths:
            p = Path(p)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                found = sorted(p.rglob("*.parquet"))
                log.info("Found %d parquet file(s) in %s", len(found), p)
                files.extend(found)
            else:
                log.warning("Path not found: %s", p)

        pq_results: dict[str, dict] = {}
        miss_total = incomp_total = nan_total = 0
        for fpath in files:
            audit = _audit_parquet_file(fpath, args.run_hours, cutoff)
            if audit is not None:
                pq_results[str(fpath)] = audit
                miss_total   += len(audit["missing"])
                incomp_total += len(audit["incomplete"])
                nan_total    += len(audit["nan_runs"])

        log.info("─" * 60)
        log.info("[Parquet] Files audited              : %d", len(files))
        log.info("[Parquet] Files with issues          : %d", len(pq_results))
        log.info("[Parquet] Total missing run slots    : %d", miss_total)
        log.info("[Parquet] Total incomplete run slots : %d", incomp_total)
        log.info("[Parquet] Total runs with NaN        : %d", nan_total)
        log.info("─" * 60)
        output["parquet"] = pq_results

    # ── Database audit ─────────────────────────────────────────────────────────
    if not args.no_db:
        db_url = os.environ.get("ECMWF_WIND_SL_URL")
        if not db_url:
            log.warning("ECMWF_WIND_SL_URL not set — skipping DB audit")
        else:
            db_audit = _audit_db(db_url, args.run_hours, cutoff)
            log.info("─" * 60)
            log.info("[DB] Missing run slots    : %d", len(db_audit["missing"]))
            log.info("[DB] Incomplete run slots : %d", len(db_audit["incomplete"]))
            log.info("[DB] Runs with NULLs      : %d", len(db_audit["null_runs"]))
            log.info("─" * 60)
            output["database"] = db_audit

    if not output:
        log.error("Nothing to audit — provide --path and/or ensure ECMWF_WIND_SL_URL is set")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("Written → %s", out)


if __name__ == "__main__":
    main()
