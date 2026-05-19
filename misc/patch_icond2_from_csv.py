"""
patch_icond2_from_csv.py — Patch rh=06 ICON-D2 ML parquet files from CSV source.

For each parquet file the script:
  1. Finds missing run timestamps that exist in the CSV  → appends those rows
  2. Finds incomplete run timestamps (< 294 rows) where the CSV has 294 rows  → replaces those rows

Only parquet files that actually change are written back.

Usage:
    cd forecasting_framework/
    python misc/patch_icond2_from_csv.py \
        --config   configs/dcrnn/config_wind_stgcn.yaml \
        --csv-path /mnt/nvme1/icon-d2/csv \
        [--run-hour 6]               # default: 6
        [--first-n-days 2]           # only patch timestamps within first N days of each file
        [--workers 16]
        [--dry-run]                  # show what would change, write nothing
        [--output  results/patch_log.json]
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

EXPECTED_ROWS = 294  # 6 toplevel levels × 49 forecasttime steps

# TEMPORARY — stations restricted for the exception-merge run. DELETE when done.
_EXCEPTION_STATIONS = {'03321', '05426'}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("patch")


# ─────────────────────────────────────────────────────────────────────────────

def _patch_file(
    pq_path: Path,
    csv_path: Path,
    first_n_days: int | None,
    dry_run: bool,
) -> dict:
    """Patch *pq_path* with data from *csv_path*.

    Returns a dict with keys:
        added        — timestamps fully appended from CSV (were missing from parquet)
        replaced     — timestamps fully replaced from CSV (parquet had < 294 rows or NaN)
        merged       — timestamps partially merged (parquet had some rows, CSV had more)
        skipped_csv  — timestamps where CSV has no more rows than parquet (not patched)
        written      — True if the file was actually rewritten
    """
    result = {"added": [], "replaced": [], "merged": [], "skipped_csv": [], "written": False}

    dfp = pd.read_parquet(pq_path)
    dfc = pd.read_csv(csv_path, parse_dates=["starttime"])

    if dfc.empty:
        return result

    # Ensure timezone consistency
    if dfc["starttime"].dt.tz is None:
        dfc["starttime"] = dfc["starttime"].dt.tz_localize("UTC")

    dfc.sort_values(["starttime", "forecasttime", "toplevel"], inplace=True)
    dfc.dropna(inplace=True)
    dfc.drop_duplicates(subset=["starttime", "forecasttime", "toplevel"], keep="first", inplace=True)
    pq_counts  = dfp.groupby("starttime").size()
    csv_counts = dfc.groupby("starttime").size()

    pq_ts_set  = set(pq_counts.index)
    csv_ts_set = set(csv_counts.index)

    # Optional: restrict to first N days of the parquet's date range
    if first_n_days is not None and not dfp.empty:
        cutoff_date = dfp["starttime"].min() + pd.Timedelta(days=first_n_days)
        csv_ts_set = {ts for ts in csv_ts_set if ts <= cutoff_date}

    rows_to_add: list[pd.DataFrame] = []

    # 1. Timestamps missing from parquet but present (and complete) in CSV
    for ts in sorted(csv_ts_set - pq_ts_set):
        csv_n = int(csv_counts.get(ts, 0))
        if csv_n == EXPECTED_ROWS:
            rows_to_add.append(dfc[dfc["starttime"] == ts])
            result["added"].append(str(ts))
        elif csv_n > 0:
            result["skipped_csv"].append(f"{ts} (CSV has {csv_n} rows)")

    data_cols = [c for c in dfp.columns if c != "starttime"]

    # --- TEMPORARY EXCEPTION: row-level merge for these two dates ---------------
    # CSV is incomplete (<294 rows) but has valid values for some (forecasttime, toplevel).
    # Instead of replacing the whole timestamp, update only matching rows in-place.
    # DELETE this block once the source data is properly regenerated.
    _EXCEPTION_DATES = set()
    station_id = pq_path.parent.name
    for ts in (_EXCEPTION_DATES & (csv_ts_set & pq_ts_set)) if station_id in _EXCEPTION_STATIONS else []:
        csv_n = int(csv_counts.get(ts, 0))
        if csv_n == 0:
            continue
        pq_ts = dfp[dfp["starttime"] == ts].copy().set_index(["forecasttime", "toplevel"])
        csv_ts = dfc[dfc["starttime"] == ts].set_index(["forecasttime", "toplevel"])
        common = pq_ts.index.intersection(csv_ts.index)
        if common.empty:
            continue
        for col in data_cols:
            if col in csv_ts.columns:
                pq_ts.loc[common, col] = csv_ts.loc[common, col].values
        pq_ts = pq_ts.reset_index()
        dfp = dfp[dfp["starttime"] != ts]
        rows_to_add.append(pq_ts)
        result["replaced"].append(
            f"{ts} (exception-merge: {len(common)} rows updated from CSV)"
        )
    # --- END TEMPORARY EXCEPTION ------------------------------------------------

    # 2. Timestamps present in both:
    #    a) Full replacement  — CSV has 294 rows, parquet is incomplete or has NaN
    #    b) Partial row merge — CSV has more rows than parquet (fills in missing forecasttime steps)
    for ts in sorted((csv_ts_set & pq_ts_set) - _EXCEPTION_DATES):
        pq_n  = int(pq_counts.get(ts, 0))
        csv_n = int(csv_counts.get(ts, 0))
        pq_rows = dfp[dfp["starttime"] == ts]
        has_nan = pq_rows[data_cols].isna().any(axis=None)

        if csv_n == EXPECTED_ROWS and (pq_n < EXPECTED_ROWS or has_nan):
            # Full replacement: CSV is complete, parquet is not
            reason = f"{pq_n}→{csv_n}" if pq_n < EXPECTED_ROWS else f"NaN@{int(pq_rows[data_cols].isna().any(axis=1).sum())}rows"
            dfp = dfp[dfp["starttime"] != ts]
            rows_to_add.append(dfc[dfc["starttime"] == ts])
            result["replaced"].append(f"{ts} ({reason})")
        elif pq_n < EXPECTED_ROWS and csv_n > pq_n:
            # Partial merge: add (forecasttime, toplevel) combinations from CSV missing in parquet
            csv_ts_rows = dfc[dfc["starttime"] == ts]
            pq_keys = set(zip(pq_rows["forecasttime"], pq_rows["toplevel"]))
            new_rows = csv_ts_rows[
                [(ft, tl) not in pq_keys
                 for ft, tl in zip(csv_ts_rows["forecasttime"], csv_ts_rows["toplevel"])]
            ]
            if not new_rows.empty:
                rows_to_add.append(new_rows)
                result["merged"].append(
                    f"{ts} (+{len(new_rows)} rows, {pq_n}→{pq_n + len(new_rows)})"
                )

    if not rows_to_add:
        return result

    dfp = pd.concat([dfp, *rows_to_add], ignore_index=True)
    dfp = dfp.sort_values(["starttime", "forecasttime", "toplevel"]).reset_index(drop=True)

    if not dry_run:
        dfp.to_parquet(pq_path, engine="pyarrow", index=False)
        result["written"] = True

    return result


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      default=None)
    ap.add_argument("--nwp-path",    default=None,
                    help="Parquet base dir (e.g. /mnt/nvme1/icon-d2/parquet)")
    ap.add_argument("--csv-path",    default="/mnt/nvme1/icon-d2/csv",
                    help="CSV base dir (mirrors parquet structure)")
    ap.add_argument("--run-hour",    type=int, default=6)
    ap.add_argument("--first-n-days", type=int, default=None,
                    help="Only patch timestamps within the first N days of each file's range")
    ap.add_argument("--workers",     type=int, default=8)
    ap.add_argument("--dry-run",     action="store_true",
                    help="Report changes without writing")
    ap.add_argument("--output",      default="results/icond2_patch_log.json")
    args = ap.parse_args()

    nwp_path = args.nwp_path
    if args.config and nwp_path is None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        nwp_path = cfg["data"]["nwp_path"]
    if nwp_path is None:
        ap.error("Provide --nwp-path or --config")

    rh = args.run_hour
    pq_base  = Path(nwp_path) / "ML" / f"{rh:02d}"
    csv_base = Path(args.csv_path) / "ML" / f"{rh:02d}"

    if not pq_base.exists():
        log.error("Parquet rh dir not found: %s", pq_base)
        return
    if not csv_base.exists():
        log.error("CSV rh dir not found: %s", csv_base)
        return

    # Collect all parquet files for this run hour
    pq_files = sorted(pq_base.rglob("*_ML.parquet"))
    log.info("Found %d parquet files in %s", len(pq_files), pq_base)
    if args.dry_run:
        log.info("DRY-RUN mode — no files will be written")

    # Build stem → CSV path index (stem may live in a different station dir on csv_base)
    log.info("Indexing CSV stems in %s …", csv_base)
    stem_to_csv: dict[str, Path] = {}
    for csv_f in csv_base.rglob("*_ML.csv"):
        stem = csv_f.stem.replace("_ML", "")
        if stem not in stem_to_csv:
            stem_to_csv[stem] = csv_f
    log.info("Found %d unique CSV stems", len(stem_to_csv))

    patch_log: dict[str, dict] = {}
    n_written = 0
    n_added_total = 0
    n_replaced_total = 0
    n_merged_total = 0

    def _task(pq_path: Path) -> tuple[str, dict]:
        stem = pq_path.stem.replace("_ML", "")
        csv_path = stem_to_csv.get(stem)
        if csv_path is None:
            return str(pq_path), {"error": "no CSV found"}
        return str(pq_path), _patch_file(
            pq_path, csv_path, args.first_n_days, args.dry_run
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_task, p): p for p in pq_files}
        for fut in tqdm(as_completed(futures), total=len(pq_files), desc="Patching"):
            try:
                path_str, res = fut.result()
                if res.get("added") or res.get("replaced") or res.get("merged") or res.get("error"):
                    patch_log[path_str] = res
                if res.get("written"):
                    n_written += 1
                n_added_total    += len(res.get("added",    []))
                n_replaced_total += len(res.get("replaced", []))
                n_merged_total   += len(res.get("merged",   []))
            except Exception as e:
                pq_path = futures[fut]
                log.warning("Error patching %s: %s", pq_path, e)
                patch_log[str(pq_path)] = {"error": str(e)}

    n_changed = len([v for v in patch_log.values() if v.get("added") or v.get("replaced") or v.get("merged")])
    log.info("─" * 60)
    log.info("Parquet files processed  : %d", len(pq_files))
    log.info("Files with changes       : %d  (written: %d)", n_changed, n_written)
    log.info("Run timestamps added     : %d", n_added_total)
    log.info("Run timestamps replaced  : %d", n_replaced_total)
    log.info("Run timestamps merged    : %d", n_merged_total)
    if args.dry_run:
        log.info("(dry-run — nothing written)")
    log.info("─" * 60)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(sorted(patch_log.items())), f, indent=2, ensure_ascii=False)
    log.info("Patch log → %s", out)


if __name__ == "__main__":
    main()
