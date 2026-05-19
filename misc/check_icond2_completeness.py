"""
check_icond2_completeness.py — Audit ICON-D2 ML or SL parquet files.

ML (Multi-Level fields):
  For each parquet file, checks:
    1. Temporal gaps  — dates between first and last run that are missing entirely
    2. Incomplete runs — starttimes with fewer than 6×49=294 rows
    3. NaN rows       — starttimes with 294 rows but containing NaN in any data column

SL (Single-Level fields):
  For each parquet file, checks:
    1. Temporal gaps        — dates between first and last run that are missing entirely
    2. Incomplete runs      — starttimes with fewer than 193 rows (0h–48h in 15-min steps)
    3. Missing (starttime, forecasttime) combinations — which forecast steps are absent
    4. NaN rows             — NaN in data columns, respecting that at quarter-hour steps
                              only radiation columns (aswdifd_s_avg, aswdir_s_avg,
                              aswdifd_s, aswdir_s) carry values; all others are NaN by design

Output: JSON  { "<path>": { "missing": [...], "incomplete": {ts: n}, ... } }
Only files with problems are included; clean files are omitted.

Usage:
    cd forecasting_framework/
    python misc/check_icond2_completeness.py \\
        --mode     ML|SL
        [--config  configs/dcrnn/config_wind_stgcn.yaml]
        [--nwp-path /mnt/nvme1/icon-d2/parquet]
        [--output   results/icond2_completeness_<mode>.json]
        [--workers  16]
        [--cutoff   2025-10-31]
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from geopy.distance import geodesic
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

EXPECTED_ROWS_PER_RUN_ML = 294   # 6 levels × 49 forecast steps (0–48 h)
EXPECTED_ROWS_PER_RUN_SL = 193   # 1 level  × 193 steps (0–48 h in 0.25 h increments)

# forecasttime values expected for a complete SL run (decimal hours)
EXPECTED_FT_SL = set(np.round(np.arange(0, 48.25, 0.25), 10))

# Radiation columns — only these carry values at sub-hourly (quarter-hour) rows
RADIATION_COLS = {"aswdifd_s_avg", "aswdir_s_avg", "aswdifd_s", "aswdir_s"}

# Columns that are not weather-variable data
META_COLS_SL = {"starttime", "forecasttime", "longitude", "latitude", "delivery_hour"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("completeness")


def _read_parquet(fpath: Path) -> pd.DataFrame:
    df = pd.read_parquet(fpath)
    if "starttime" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["starttime"]):
        df["starttime"] = pd.to_datetime(df["starttime"], utc=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ML audit
# ─────────────────────────────────────────────────────────────────────────────

def _audit_file_ml(fpath: Path, run_hour: int, cutoff: pd.Timestamp | None) -> dict:
    """Return {"missing": [...], "incomplete": {ts: n}, "nan_rows": {ts: n}}."""
    df = _read_parquet(fpath)
    if df.empty:
        return {"missing": [], "incomplete": {}, "nan_rows": {}}

    data_cols = [c for c in df.columns if c != "starttime"]
    counts = df.groupby("starttime").size()

    if cutoff is not None:
        counts = counts[counts.index <= cutoff]
        df = df[df["starttime"] <= cutoff]
    if counts.empty:
        return {"missing": [], "incomplete": {}, "nan_rows": {}}

    first_ts = counts.index.min().tz_convert("UTC")
    last_ts  = counts.index.max().tz_convert("UTC")
    expected = pd.date_range(
        start=first_ts.normalize() + pd.Timedelta(hours=run_hour),
        end=last_ts,
        freq="1D",
        tz="UTC",
    )

    observed_set = set(counts.index.tolist())
    missing = sorted(str(ts) for ts in expected if ts not in observed_set)
    incomplete = {
        str(ts): int(n)
        for ts, n in counts.items()
        if n != EXPECTED_ROWS_PER_RUN_ML
    }

    # Vectorized NaN check — only for complete runs
    complete_idx = counts[counts == EXPECTED_ROWS_PER_RUN_ML].index
    if len(complete_idx):
        df_c = df[df["starttime"].isin(complete_idx)]
        row_has_nan = df_c[data_cols].isna().any(axis=1)
        nan_by_ts = row_has_nan.groupby(df_c["starttime"]).sum()
        nan_rows = {str(ts): int(n) for ts, n in nan_by_ts.items() if n > 0}
    else:
        nan_rows = {}

    return {"missing": missing, "incomplete": incomplete, "nan_rows": nan_rows}


# ─────────────────────────────────────────────────────────────────────────────
# SL audit
# ─────────────────────────────────────────────────────────────────────────────

def _audit_file_sl(fpath: Path, run_hour: int, cutoff: pd.Timestamp | None) -> dict:
    """Return {"missing": [...], "incomplete": {ts: n},
               "missing_forecasttimes": {ts: [ft, ...]}, "nan_rows": {ts: n}}.

    Quarter-hour rows (forecasttime % 1 != 0) are expected to have NaN in all
    non-radiation columns — this is by design and is NOT flagged as an error.
    Only the four radiation columns are checked for NaN at quarter-hour steps.
    """
    df = _read_parquet(fpath)
    if df.empty:
        return {"missing": [], "incomplete": {}, "missing_forecasttimes": {}, "nan_rows": {}}

    data_cols = [c for c in df.columns if c not in META_COLS_SL]
    rad_cols  = [c for c in data_cols if c in RADIATION_COLS]

    counts = df.groupby("starttime").size()

    if cutoff is not None:
        counts = counts[counts.index <= cutoff]
        df = df[df["starttime"] <= cutoff]
    if counts.empty:
        return {"missing": [], "incomplete": {}, "missing_forecasttimes": {}, "nan_rows": {}}

    first_ts = counts.index.min().tz_convert("UTC")
    last_ts  = counts.index.max().tz_convert("UTC")
    expected = pd.date_range(
        start=first_ts.normalize() + pd.Timedelta(hours=run_hour),
        end=last_ts,
        freq="1D",
        tz="UTC",
    )

    observed_set = set(counts.index.tolist())
    missing = sorted(str(ts) for ts in expected if ts not in observed_set)
    incomplete = {
        str(ts): int(n)
        for ts, n in counts.items()
        if n != EXPECTED_ROWS_PER_RUN_SL
    }

    # ── Vectorized NaN check (only complete runs) ─────────────────────────
    complete_idx = counts[counts == EXPECTED_ROWS_PER_RUN_SL].index
    nan_rows: dict[str, int] = {}
    if len(complete_idx) and data_cols:
        df_c  = df[df["starttime"].isin(complete_idx)].copy()
        ft_c  = np.round(df_c["forecasttime"].to_numpy().astype(float) % 1, 10)
        qh    = ft_c != 0   # True = sub-hourly row

        n_nan_series = pd.Series(0, index=df_c.index, dtype=int)
        if data_cols:
            hourly_nan = df_c.loc[~qh, data_cols].isna().any(axis=1)
            n_nan_series.loc[hourly_nan.index] = hourly_nan.astype(int)
        if rad_cols:
            qh_nan = df_c.loc[qh, rad_cols].isna().any(axis=1)
            n_nan_series.loc[qh_nan.index] = qh_nan.astype(int)

        nan_by_ts = n_nan_series.groupby(df_c["starttime"]).sum()
        nan_rows = {str(ts): int(n) for ts, n in nan_by_ts.items() if n > 0}

    # ── Missing (starttime, forecasttime) combinations (incomplete only) ──
    # Stores the count of missing steps, plus the first few values as a sample.
    missing_fts: dict[str, dict] = {}
    if incomplete and "forecasttime" in df.columns:
        incomplete_idx = counts[counts != EXPECTED_ROWS_PER_RUN_SL].index
        df_inc = df[df["starttime"].isin(incomplete_idx)]
        for ts, grp in df_inc.groupby("starttime"):
            observed_ft = set(np.round(grp["forecasttime"].to_numpy().astype(float), 10))
            miss_ft = sorted(EXPECTED_FT_SL - observed_ft)
            if miss_ft:
                missing_fts[str(ts)] = {"count": len(miss_ft), "sample": miss_ft[:5]}

    return {
        "missing": missing,
        "incomplete": incomplete,
        "missing_forecasttimes": missing_fts,
        "nan_rows": nan_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ml_stem(stem: str) -> tuple[float, float]:
    """ML stem '52_9065_12_8820' → (lat=52.9065, lon=12.8820)."""
    p = stem.split("_")
    return float(f"{p[0]}.{p[1]}"), float(f"{p[2]}.{p[3]}")


def _parse_sl_stem(stem: str) -> tuple[float, float]:
    """SL stem '10_0000_47_8000' → (lat=47.8000, lon=10.0000)  [lon comes first in SL names]."""
    p = stem.split("_")
    return float(f"{p[2]}.{p[3]}"), float(f"{p[0]}.{p[1]}")


def _build_summary(
    results: dict[str, dict],
    mode: str,
    mapping_csv: str,
) -> dict:
    """
    Build a compact summary grouped by grid-point stem.

    Uses data/nwp_station_mapping.csv (built by misc/build_nwp_station_mapping.py)
    to look up, for each problematic grid point:
      - which station (park_id) it belongs to
      - its rank among all grid points assigned to that station (1 = nearest)
      - total number of grid points in that station group
      - geodesic distance to the station

    For ML files the park_id is already encoded in the path parent directory.
    For SL files it is resolved via the mapping table.

    Returns dict keyed by stem:
        {lat, lon, park_id, distance_km, rank, total_in_park, files: [...]}
    """
    mapping = pd.read_csv(mapping_csv, dtype={"park_id": str})
    mapping["park_id"] = mapping["park_id"].str.zfill(5)
    # Build lookup: (lat_rounded4, lon_rounded4) → row dict
    lookup: dict[tuple, dict] = {}
    for _, row in mapping.iterrows():
        key = (round(float(row["lat"]), 4), round(float(row["lon"]), 4))
        lookup[key] = row.to_dict()

    parse_stem = _parse_ml_stem if mode == "ML" else _parse_sl_stem

    stem_map: dict[str, dict] = {}

    for fpath_str, audit in results.items():
        fpath = Path(fpath_str)
        stem  = fpath.stem.replace(f"_{mode}", "")

        # Run-hour: one level up for ML (…/rh/park/file), two levels for SL (…/rh/file)
        try:
            rh = int(fpath.parent.name if mode == "SL" else fpath.parent.parent.name)
        except ValueError:
            rh = -1

        # Compact file entry — counts only, no timestamp lists
        file_entry: dict = {
            "path":       fpath_str,
            "run_hour":   rh,
            "missing":    len(audit.get("missing", [])),
            "incomplete": len(audit.get("incomplete", {})),
            "nan_runs":   len(audit.get("nan_rows", {})),
        }
        if mode == "SL":
            file_entry["missing_ft_runs"] = len(audit.get("missing_forecasttimes", {}))

        if stem not in stem_map:
            try:
                lat, lon = parse_stem(stem)
            except (ValueError, IndexError):
                lat, lon = float("nan"), float("nan")

            info: dict = {"lat": lat, "lon": lon}

            key = (round(lat, 4), round(lon, 4))
            if key in lookup:
                row = lookup[key]
                info["park_id"]       = str(row["park_id"]).zfill(5)
                info["distance_km"]   = round(float(row["distance_km"]), 3)
                info["rank"]          = int(row["rank"])
                info["total_in_park"] = int(row["total_in_park"])
            elif mode == "ML" and fpath.parent.name.isdigit() is False:
                # Fallback: park_id from path for ML (if mapping lookup missed)
                info["park_id"]       = fpath.parent.name
                info["distance_km"]   = None
                info["rank"]          = None
                info["total_in_park"] = None
            else:
                info["park_id"]       = None
                info["distance_km"]   = None
                info["rank"]          = None
                info["total_in_park"] = None

            stem_map[stem] = {**info, "files": []}

        stem_map[stem]["files"].append(file_entry)

    for v in stem_map.values():
        v["files"].sort(key=lambda e: e["run_hour"])

    return dict(sorted(stem_map.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",     choices=["ML", "SL"], default="ML",
                    help="ML = Multi-Level fields, SL = Single-Level fields (default: ML)")
    ap.add_argument("--config",   default=None,
                    help="YAML config (used to read nwp_path if --nwp-path is omitted)")
    ap.add_argument("--nwp-path", default=None,
                    help="Base NWP directory (e.g. /mnt/nvme1/icon-d2/parquet)")
    ap.add_argument("--output",   default=None,
                    help="Output JSON path (default: results/icond2_completeness_<MODE>.json)")
    ap.add_argument("--workers",  type=int, default=16)
    ap.add_argument("--cutoff",   default=None,
                    help="Skip runs after YYYY-MM-DD (e.g. 2025-10-31)")
    ap.add_argument("--summary",     action="store_true",
                    help="Generate a compact summary JSON grouped by grid-point stem, "
                         "showing each point's station assignment and rank "
                         "(skips re-audit if audit JSON already exists)")
    ap.add_argument("--mapping-csv", default="data/nwp_station_mapping.csv",
                    help="Precomputed grid-point→station mapping CSV "
                         "(build with misc/build_nwp_station_mapping.py; "
                         "default: data/nwp_station_mapping.csv)")
    args = ap.parse_args()

    nwp_path   = args.nwp_path
    cutoff_str = args.cutoff

    if args.config and nwp_path is None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        nwp_path   = cfg["data"]["nwp_path"]
        cutoff_str = cutoff_str or cfg["data"].get("test_end")

    if nwp_path is None:
        ap.error("Provide --nwp-path or --config")

    cutoff: pd.Timestamp | None = (
        pd.Timestamp(cutoff_str, tz="UTC") if cutoff_str else None
    )

    mode     = args.mode
    base_dir = Path(nwp_path) / mode
    out_path = Path(args.output or f"results/icond2_completeness_{mode}.json")

    if not base_dir.exists():
        log.error("%s directory not found: %s", mode, base_dir)
        return

    # ── Caching: load existing audit JSON if --summary and file already exists ─
    results: dict[str, dict] = {}
    total_files = 0

    if args.summary and out_path.exists():
        log.info("Audit JSON already exists — loading from cache: %s", out_path)
        with open(out_path) as f:
            results = json.load(f)
        log.info("Loaded %d problematic files from cache", len(results))
    else:
        # ── Discover parquet files ────────────────────────────────────────────
        log.info("Scanning %s …", base_dir)
        all_files: list[tuple[Path, int]] = []
        for rh_dir in sorted(base_dir.iterdir()):
            if not rh_dir.is_dir():
                continue
            try:
                rh = int(rh_dir.name)
            except ValueError:
                continue
            pattern = f"*_{mode}.parquet"
            for fpath in rh_dir.rglob(pattern):
                all_files.append((fpath, rh))

        total_files = len(all_files)
        log.info("Found %d parquet files across %d run-hour dirs",
                 total_files, len({rh for _, rh in all_files}))

        # ── Parallel audit ────────────────────────────────────────────────────
        audit_fn = _audit_file_ml if mode == "ML" else _audit_file_sl

        def _task(item):
            fpath, rh = item
            return str(fpath), audit_fn(fpath, rh, cutoff)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_task, item): item for item in all_files}
            for fut in tqdm(as_completed(futures), total=total_files, desc="Auditing"):
                try:
                    path_str, audit = fut.result()
                    has_issues = (
                        audit.get("missing")
                        or audit.get("incomplete")
                        or audit.get("missing_forecasttimes")
                        or audit.get("nan_rows")
                    )
                    if has_issues:
                        results[path_str] = audit
                except Exception as e:
                    fpath, rh = futures[fut]
                    log.warning("Error auditing %s: %s", fpath, e)

        # ── Log stats + write audit JSON ──────────────────────────────────────
        bad_files     = len(results)
        total_miss    = sum(len(v["missing"])    for v in results.values())
        total_incomp  = sum(len(v["incomplete"]) for v in results.values())
        total_miss_ft = sum(len(v.get("missing_forecasttimes", {})) for v in results.values())
        total_nan     = sum(len(v.get("nan_rows", {})) for v in results.values())

        log.info("─" * 60)
        log.info("Mode                        : %s", mode)
        log.info("Total parquet files audited : %d", total_files)
        log.info("Files with issues           : %d  (%.1f%%)",
                 bad_files, 100 * bad_files / max(total_files, 1))
        log.info("Total missing run slots     : %d", total_miss)
        log.info("Total incomplete run slots  : %d", total_incomp)
        if mode == "SL":
            log.info("Total runs with missing fts : %d", total_miss_ft)
        log.info("Total runs with NaN values  : %d", total_nan)
        log.info("─" * 60)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dict(sorted(results.items())), f, indent=2, ensure_ascii=False)
        log.info("Written → %s", out_path)

    # ── Optional summary ──────────────────────────────────────────────────────
    if args.summary:
        summary_path = out_path.with_stem(out_path.stem + "_summary")

        if summary_path.exists():
            log.info("Summary JSON already exists — skipping: %s", summary_path)
        else:
            if not Path(args.mapping_csv).exists():
                log.error("Mapping CSV not found: %s  — run misc/build_nwp_station_mapping.py first",
                          args.mapping_csv)
            else:
                log.info("Building summary …")
                summary = _build_summary(results, mode, args.mapping_csv)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                log.info("Summary → %s  (%d unique grid points)", summary_path, len(summary))


if __name__ == "__main__":
    main()
