"""
diagnose_icond2_runs.py — Trace load_icond2_ml_runs step by step.

Usage:
    cd forecasting_framework/
    python misc/diagnose_icond2_runs.py \
        --config configs/dcrnn/config_wind_stgcn.yaml \
        [--cutoff 2025-10-31] \
        [--n-stations 10]   # limit stations for faster scan

The script mirrors load_icond2_ml_runs phase by phase and prints diagnostics
at each step to identify why R=836 instead of the expected ~3303.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from geostatistics.train_stgnn2 import (
    _load_icond2_ml_parquet,
    _parse_latlon,          # noqa: F401
    _parse_ml_feature_spec,
    _select_nearest_grid_files,
    load_station_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diagnose")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sep(title: str = "") -> None:
    bar = "─" * 70
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dcrnn/config_wind_stgcn.yaml")
    ap.add_argument("--cutoff", default=None,
                    help="Exclude runs after this date (YYYY-MM-DD). "
                         "Defaults to test_end from config.")
    ap.add_argument("--n-stations", type=int, default=None,
                    help="Limit to first N stations (faster scan).")
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg  = cfg["data"]
    dcrnn_cfg = cfg.get("dcrnn", cfg.get("stgnn2", {}))

    nwp_path      = data_cfg["nwp_path"]
    run_hours     = tuple(dcrnn_cfg.get("icond2_run_hours", [6, 9, 12, 15]))
    next_n_grid   = dcrnn_cfg.get("next_n_icond2", 4)
    features      = dcrnn_cfg.get("icond2_features", ["wind_speed_10m"])
    feature_specs = _parse_ml_feature_spec(features)

    cutoff_str = args.cutoff or data_cfg.get("test_end")
    cutoff: pd.Timestamp | None = pd.Timestamp(cutoff_str, tz="UTC") if cutoff_str else None

    # ── Station IDs ──────────────────────────────────────────────────────────
    all_ids_raw = (
        list(data_cfg.get("files", []))
        + list(data_cfg.get("val_files", []))
        + list(data_cfg.get("test_files", []))
    )
    all_ids = [str(s) for s in all_ids_raw]
    if args.n_stations:
        all_ids = all_ids[: args.n_stations]

    data_path  = data_cfg["path"]
    meta_path  = data_cfg.get("stations_master")
    lats, lons, _alts = load_station_metadata(data_path, all_ids, meta_path=meta_path)
    station_coords = np.stack([lats, lons], axis=1).astype(np.float32)
    N = len(all_ids)

    ml_base = Path(nwp_path) / "ML"

    sep("CONFIG")
    print(f"  config       : {args.config}")
    print(f"  nwp_path     : {nwp_path}")
    print(f"  run_hours    : {run_hours}")
    print(f"  next_n_grid  : {next_n_grid}")
    print(f"  features     : {features}")
    print(f"  cutoff       : {cutoff}")
    print(f"  N stations   : {N}")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Scan directories
    # ═════════════════════════════════════════════════════════════════════════
    sep("PHASE 1 — Directory scan")

    unique_grid_paths: dict[tuple[str, int], Path] = {}
    station_grid_keys: list[list[tuple[str, int]]] = [[] for _ in range(N)]

    stems_per_rh: dict[int, set[str]] = {rh: set() for rh in run_hours}
    missing_dirs: list[tuple[str, int]] = []

    for si, sid in enumerate(tqdm(all_ids, desc="Scanning")):
        s_lat = float(station_coords[si, 0])
        s_lon = float(station_coords[si, 1])
        for rh in run_hours:
            sid_dir = ml_base / f"{rh:02d}" / sid
            if not sid_dir.exists():
                missing_dirs.append((sid, rh))
                continue
            nearest = _select_nearest_grid_files(sid_dir, s_lat, s_lon, next_n_grid)
            for fpath, stem, _, _ in nearest:
                key = (stem, rh)
                station_grid_keys[si].append(key)
                stems_per_rh[rh].add(stem)
                if key not in unique_grid_paths:
                    unique_grid_paths[key] = fpath

    print(f"\n  Missing station×rh dirs : {len(missing_dirs)}")
    if missing_dirs[:10]:
        print(f"    (first 10): {missing_dirs[:10]}")

    print(f"\n  unique (stem, rh) keys  : {len(unique_grid_paths)}")
    print(f"\n  Unique stems per run-hour:")
    all_stems = set()
    for rh in run_hours:
        s = stems_per_rh[rh]
        all_stems |= s
        print(f"    rh={rh:02d} → {len(s):5d} unique stems")
    print(f"    union        → {len(all_stems):5d} unique stems")

    # Check stem overlap: which stems are present in all 4 rh vs fewer
    present_in_n_rh: Counter = Counter()
    for stem in all_stems:
        cnt = sum(1 for rh in run_hours if stem in stems_per_rh[rh])
        present_in_n_rh[cnt] += 1
    print(f"\n  Stem coverage across rh dirs:")
    for n in sorted(present_in_n_rh):
        print(f"    present in {n}/{len(run_hours)} rh : {present_in_n_rh[n]} stems")

    # How many unique_grid_paths per rh?
    keys_per_rh: Counter = Counter(rh for _, rh in unique_grid_paths)
    print(f"\n  unique_grid_paths entries per rh:")
    for rh in run_hours:
        print(f"    rh={rh:02d} → {keys_per_rh[rh]:5d}")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Parallel load
    # ═════════════════════════════════════════════════════════════════════════
    sep("PHASE 2 — Parallel Parquet load")

    csv_results: dict[tuple[str, int], tuple[list, np.ndarray]] = {}
    load_failures: list[tuple[str, int, str]] = []

    def _load(key_fpath):
        key, fpath = key_fpath
        return key, _load_icond2_ml_parquet(fpath, feature_specs)

    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(_load, kp): kp[0] for kp in unique_grid_paths.items()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading"):
            key = futures[fut]
            try:
                _, (run_times_h, arr_h) = fut.result()
                csv_results[key] = (run_times_h, arr_h)
            except Exception as e:
                load_failures.append((key[0], key[1], str(e)))

    print(f"\n  Loaded successfully : {len(csv_results)}/{len(unique_grid_paths)}")
    print(f"  Load failures       : {len(load_failures)}")
    if load_failures[:5]:
        for stem, rh, err in load_failures[:5]:
            print(f"    FAIL ({rh:02d}, {stem[:30]}): {err[:80]}")

    # Per-rh count in csv_results
    results_per_rh: Counter = Counter(rh for _, rh in csv_results)
    print(f"\n  csv_results entries per rh:")
    for rh in run_hours:
        print(f"    rh={rh:02d} → {results_per_rh[rh]:5d}")

    # Examine hour distribution inside run_times_h per rh
    print(f"\n  Hour distribution of run_times_h per rh (sample: first loaded file):")
    sample_per_rh: dict[int, tuple[str, list]] = {}
    for (stem, rh), (run_times_h, _) in csv_results.items():
        if rh not in sample_per_rh:
            sample_per_rh[rh] = (stem, run_times_h)
    for rh in run_hours:
        if rh not in sample_per_rh:
            print(f"    rh={rh:02d} → (no results)")
            continue
        stem, rt = sample_per_rh[rh]
        hour_counts = Counter(t.hour for t in rt)
        print(f"    rh={rh:02d} ({stem[:30]}) : {len(rt)} run times, "
              f"hour distribution = {dict(sorted(hour_counts.items()))}")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Global run time union
    # ═════════════════════════════════════════════════════════════════════════
    sep("PHASE 3 — Global run time union")

    all_run_times: set[pd.Timestamp] = set()
    skipped_wrong_hour: dict[int, int] = defaultdict(int)   # keyed by rh
    skipped_after_cutoff: dict[int, int] = defaultdict(int)
    added_per_rh: dict[int, int] = defaultdict(int)

    for (stem, rh), (run_times_h, _) in csv_results.items():
        for t in run_times_h:
            if t.hour not in run_hours:
                skipped_wrong_hour[rh] += 1
                continue
            if cutoff is not None and t > cutoff:
                skipped_after_cutoff[rh] += 1
                continue
            all_run_times.add(t)
            added_per_rh[rh] += 1

    run_times_global = pd.DatetimeIndex(sorted(all_run_times))
    R = len(run_times_global)

    print(f"\n  Total unique run times (R) : {R}")
    if R > 0:
        print(f"  Time range                 : {run_times_global[0]} … {run_times_global[-1]}")
    print(f"\n  Breakdown per rh:")
    for rh in run_hours:
        print(f"    rh={rh:02d} : added={added_per_rh[rh]:6d}  "
              f"wrong_hour={skipped_wrong_hour[rh]:6d}  "
              f"after_cutoff={skipped_after_cutoff[rh]:6d}")

    hour_counts_global = Counter(t.hour for t in run_times_global)
    print(f"\n  Hour distribution in run_times_global:")
    for h in sorted(hour_counts_global):
        print(f"    hour {h:02d} UTC : {hour_counts_global[h]} runs")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 4 — NaN completeness check (without building full array)
    # ═════════════════════════════════════════════════════════════════════════
    sep("PHASE 4 — NaN coverage estimate (without full array)")

    run_idx_map = {t: i for i, t in enumerate(run_times_global)}
    unique_stems = sorted({stem for stem, _ in unique_grid_paths})
    stem_to_gi   = {s: i for i, s in enumerate(unique_stems)}
    N_grid = len(unique_stems)

    print(f"\n  N_grid (unique stems) : {N_grid}")
    print(f"  R (total runs)        : {R}")

    # Track which (run_idx, grid_idx) slots are filled
    filled: set[tuple[int, int]] = set()
    for (stem, rh), (run_times_h, _) in csv_results.items():
        gi = stem_to_gi[stem]
        for t in run_times_h:
            global_ri = run_idx_map.get(t)
            if global_ri is not None:
                filled.add((global_ri, gi))

    total_slots = R * N_grid
    filled_slots = len(filled)
    nan_slots = total_slots - filled_slots

    print(f"  Total (run, grid) slots : {total_slots:,}")
    print(f"  Filled slots            : {filled_slots:,}  ({100*filled_slots/max(total_slots,1):.1f}%)")
    print(f"  NaN (missing) slots     : {nan_slots:,}   ({100*nan_slots/max(total_slots,1):.1f}%)")

    # Count runs with at least one NaN grid slot → would be dropped by complete_mask
    runs_with_nan_grid: set[int] = set()
    for ri in range(R):
        for gi in range(N_grid):
            if (ri, gi) not in filled:
                runs_with_nan_grid.add(ri)
    print(f"\n  Runs with ≥1 missing grid slot (would be dropped) : {len(runs_with_nan_grid):,}")
    print(f"  Runs fully complete                                : {R - len(runs_with_nan_grid):,}")

    # Simulate forward-fill: how many runs would survive?
    runs_with_nan_only_at_start: set[int] = set()
    # A run ri has NaN that can't be forward-filled only if ALL runs before ri
    # also have NaN for that (gi). For a quick estimate: check if the first run
    # (ri=0) has any NaN grid slots.
    first_run_nan_gi: set[int] = set()
    for ri in range(R):
        for gi in range(N_grid):
            if (ri, gi) not in filled:
                if ri == 0:
                    first_run_nan_gi.add(gi)
    # After forward-fill, only runs whose NaN grid nodes are ALL in first_run_nan_gi
    # (never filled) would still have NaN → zeroed.
    # For the complete_mask, after forward-fill + zero-fill, all runs are complete.
    print(f"\n  After forward-fill + zero-fill (NEW behavior in train_stgnn2.py):")
    print(f"    All {R} runs would be fully complete ✓")
    print(f"    (only {len(first_run_nan_gi)} grid nodes might get zeroed at run 0)")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 4b — NaN breakdown by feature and grid node
    # ═════════════════════════════════════════════════════════════════════════
    sep("PHASE 4b — NaN breakdown (feature × grid node)")

    # For each (stem, rh), check per-feature NaN counts in arr_h
    nan_by_feature: list[int] = [0] * len(features)
    nan_by_gi: dict[int, int] = defaultdict(int)

    for (stem, rh), (run_times_h, arr_h) in csv_results.items():
        gi = stem_to_gi[stem]
        # arr_h : (R_local, 48, F)  — only runs in this file
        # Check which runs land inside run_idx_map
        for local_ri, t in enumerate(run_times_h):
            if run_idx_map.get(t) is None:
                continue
            for fi in range(len(features)):
                if np.isnan(arr_h[local_ri, :, fi]).any():
                    nan_by_feature[fi] += 1
                    nan_by_gi[gi] += 1

    print(f"\n  NaN-affected (run, grid) slots by feature:")
    for fi, fname in enumerate(features):
        print(f"    [{fi}] {fname:<30s} : {nan_by_feature[fi]:6d} affected runs")

    # Top-10 grid nodes with most NaN runs
    top_nan_gi = sorted(nan_by_gi.items(), key=lambda x: -x[1])[:15]
    print(f"\n  Top-15 grid nodes by NaN-run count:")
    for gi, cnt in top_nan_gi:
        stem = unique_stems[gi]
        print(f"    gi={gi:4d}  stem={stem:<30s}  NaN runs={cnt:6d}")

    # Runs-with-NaN by hour
    runs_with_nan_by_hour: Counter = Counter()
    for ri in runs_with_nan_grid:
        runs_with_nan_by_hour[run_times_global[ri].hour] += 1
    print(f"\n  Runs-with-NaN by hour:")
    for h in sorted(runs_with_nan_by_hour):
        print(f"    hour {h:02d} UTC : {runs_with_nan_by_hour[h]} runs dropped")

    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    sep("SUMMARY")
    print(f"  Stations                        : {N}")
    print(f"  Run-hours                       : {run_hours}")
    print(f"  unique (stem, rh) keys          : {len(unique_grid_paths)}")
    print(f"  Loaded csv_results              : {len(csv_results)}")
    print(f"  R (runs before NaN drop)        : {R}")
    print(f"  R (runs after NaN drop estimate): {R - len(runs_with_nan_grid)}")
    print(f"  Expected R (with 4 rh)          : ~{len(stems_per_rh[run_hours[0]]) * len(run_hours)}")
    sep()


if __name__ == "__main__":
    main()
