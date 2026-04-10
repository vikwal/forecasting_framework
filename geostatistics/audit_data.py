"""
audit_data.py — Data completeness audit for the STGNN pipeline.

Loads all data sources (measurements, ICON-D2 ML runs, ECMWF NWP) and writes
a detailed completeness report to reports/data_audit_<config_stem>.txt.

Usage
-----
    python geostatistics/audit_data.py --config configs/config_wind_stgcn.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# Re-use loaders from the training script
from geostatistics.train_stgnn2 import (
    load_station_measurements,
    load_station_metadata,
    load_icond2_ml_runs,
)


# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


def section(out: StringIO, title: str) -> None:
    out.write("\n" + "=" * 72 + "\n")
    out.write(f"  {title}\n")
    out.write("=" * 72 + "\n")


# ---------------------------------------------------------------------------

def audit_measurements(
    out: StringIO,
    meas: np.ndarray,          # (T, N, M)
    timestamps: pd.DatetimeIndex,
    station_ids: list[str],
    cols: list[str],
) -> None:
    T, N, M = meas.shape
    section(out, f"MEASUREMENTS  ({T} timesteps × {N} stations × {M} features)")

    out.write(f"\nTime range: {timestamps[0]}  →  {timestamps[-1]}\n")

    for mi, col in enumerate(cols):
        out.write(f"\n--- Feature: {col} ---\n")
        mat = meas[:, :, mi]   # (T, N)
        total_cells = T * N
        total_nan   = int(np.isnan(mat).sum())
        out.write(f"  Total cells : {total_cells:,}   NaN: {total_nan:,}  ({_pct(total_nan, total_cells)})\n")

        # Per-station NaN counts
        nan_per_station = np.isnan(mat).sum(axis=0)   # (N,)
        bad_idx = np.where(nan_per_station > 0)[0]
        if len(bad_idx) == 0:
            out.write("  All stations complete ✓\n")
        else:
            out.write(f"  Stations with missing values: {len(bad_idx)} / {N}\n")
            out.write(f"  {'Station':<12} {'NaN count':>12} {'%':>8}  First NaN timestamp\n")
            out.write("  " + "-" * 56 + "\n")
            for i in bad_idx:
                cnt = int(nan_per_station[i])
                first_nan_t = timestamps[np.where(np.isnan(mat[:, i]))[0][0]]
                out.write(f"  {station_ids[i]:<12} {cnt:>12,} {_pct(cnt, T):>8}  {first_nan_t}\n")

        # Per-timestep: how many stations are NaN
        nan_per_t = np.isnan(mat).sum(axis=1)   # (T,)
        bad_t = np.where(nan_per_t > 0)[0]
        out.write(f"\n  Timesteps with ≥1 missing station: {len(bad_t):,} / {T:,}\n")
        if len(bad_t) > 0:
            runs = []
            start = prev = bad_t[0]
            for idx in bad_t[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    runs.append((start, prev))
                    start = prev = idx
            runs.append((start, prev))
            out.write(f"  Contiguous NaN runs (timestep index ranges):\n")
            for s, e in runs[:50]:    # cap at 50 for readability
                out.write(f"    [{s:5d} – {e:5d}]  {timestamps[s]}  →  {timestamps[e]}  ({e-s+1} steps)\n")
            if len(runs) > 50:
                out.write(f"    ... ({len(runs) - 50} more runs not shown)\n")


def audit_icond2(
    out: StringIO,
    grid_runs: np.ndarray,          # (R, 48, N_grid, F)
    run_times: pd.DatetimeIndex,
    icond2_coords: np.ndarray,      # (N_grid, 2)
    station_nearest_grid: np.ndarray,  # (N,) int
    station_ids: list[str],
    features: list[str],
) -> None:
    R, leads, N_grid, F = grid_runs.shape
    section(out, f"ICON-D2 ML RUNS  ({R} runs × {leads} leads × {N_grid} grid nodes × {F} features)")

    out.write(f"\nRun time range: {run_times[0]}  →  {run_times[-1]}\n")

    # Overall NaN summary
    total_cells = R * leads * N_grid * F
    total_nan   = int(np.isnan(grid_runs).sum())
    out.write(f"Total cells : {total_cells:,}   NaN: {total_nan:,}  ({_pct(total_nan, total_cells)})\n")

    # Per-feature NaN
    out.write("\nPer-feature NaN:\n")
    for fi, feat in enumerate(features):
        n = int(np.isnan(grid_runs[:, :, :, fi]).sum())
        out.write(f"  {feat:<30} {n:>12,}  ({_pct(n, R * leads * N_grid)})\n")

    # Per-grid-node: fraction of (run, lead) slots that are NaN
    nan_per_node = np.isnan(grid_runs[:, :, :, 0]).sum(axis=(0, 1))  # (N_grid,)
    total_per_node = R * leads
    bad_nodes = np.where(nan_per_node > 0)[0]
    out.write(f"\nGrid nodes with ≥1 NaN slot: {len(bad_nodes)} / {N_grid}\n")
    if len(bad_nodes) > 0:
        out.write(f"  {'Node idx':>8} {'lat':>10} {'lon':>10} {'NaN slots':>12} {'%':>8}\n")
        out.write("  " + "-" * 56 + "\n")
        for gi in bad_nodes[:100]:
            n = int(nan_per_node[gi])
            lat, lon = icond2_coords[gi]
            out.write(f"  {gi:>8} {lat:>10.4f} {lon:>10.4f} {n:>12,} {_pct(n, total_per_node):>8}\n")
        if len(bad_nodes) > 100:
            out.write(f"  ... ({len(bad_nodes) - 100} more nodes not shown)\n")

    # Per-run: fraction of (grid_node, lead) slots that are NaN
    nan_per_run = np.isnan(grid_runs[:, :, :, 0]).sum(axis=(1, 2))  # (R,)
    bad_runs = np.where(nan_per_run > 0)[0]
    out.write(f"\nRuns with ≥1 NaN slot: {len(bad_runs)} / {R}\n")
    if len(bad_runs) > 0:
        out.write(f"  First 20 bad runs:\n")
        for ri in bad_runs[:20]:
            n = int(nan_per_run[ri])
            out.write(f"    run {ri:4d}  {run_times[ri]}  NaN slots: {n:,} / {leads * N_grid:,}  ({_pct(n, leads * N_grid)})\n")
        if len(bad_runs) > 20:
            out.write(f"  ... ({len(bad_runs) - 20} more runs not shown)\n")

    # Station nearest-grid check
    out.write(f"\nStation nearest-grid mapping:\n")
    out.write(f"  {'Station':<12} {'Grid idx':>10} {'lat':>10} {'lon':>10} {'NaN slots in nearest node':>28}\n")
    out.write("  " + "-" * 72 + "\n")
    for si, sid in enumerate(station_ids):
        gi = station_nearest_grid[si]
        nan_s = int(nan_per_node[gi])
        lat, lon = icond2_coords[gi]
        flag = " ← HAS NaN" if nan_s > 0 else ""
        out.write(f"  {sid:<12} {gi:>10} {lat:>10.4f} {lon:>10.4f} {nan_s:>28,}{flag}\n")


def audit_run_pairs(
    out: StringIO,
    meas: np.ndarray,          # (T, N, M)
    timestamps: pd.DatetimeIndex,
    run_times: pd.DatetimeIndex,
    split_time: pd.Timestamp,
    H: int,
    F_h: int,
) -> None:
    T = len(timestamps)
    R = len(run_times)
    ts_lookup = pd.Series(np.arange(T), index=timestamps)
    _meas_nan_any = np.isnan(meas[:, :, 0]).any(axis=1)

    section(out, f"RUN PAIR ANALYSIS  (H={H}h history, F={F_h}h forecast)")

    train_pairs, val_pairs = 0, 0
    skip_no_ts, skip_bounds, skip_no_hist_run, skip_nan_meas = 0, 0, 0, 0

    for r_curr in range(R):
        t_run = run_times[r_curr]
        if t_run not in ts_lookup.index:
            skip_no_ts += 1
            continue
        t_run_abs = int(ts_lookup[t_run])
        if t_run_abs < H or t_run_abs + F_h > T:
            skip_bounds += 1
            continue
        t_hist_target = t_run - pd.Timedelta(hours=H)
        diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
        r_hist  = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600:
            skip_no_hist_run += 1
            continue
        if _meas_nan_any[t_run_abs - H : t_run_abs + F_h].any():
            skip_nan_meas += 1
            continue
        if t_run < split_time:
            train_pairs += 1
        else:
            val_pairs += 1

    total_valid = train_pairs + val_pairs
    out.write(f"\n  Runs total           : {R:,}\n")
    out.write(f"  Valid run pairs      : {total_valid:,}  (train: {train_pairs:,}  val: {val_pairs:,})\n")
    out.write(f"\n  Skipped:\n")
    out.write(f"    No matching timestamp in measurement array : {skip_no_ts:,}\n")
    out.write(f"    Out of bounds (need H before + F after)    : {skip_bounds:,}\n")
    out.write(f"    No history run within 3h                   : {skip_no_hist_run:,}\n")
    out.write(f"    NaN in measurement window                  : {skip_nan_meas:,}\n")
    out.write(f"    Total skipped                              : {skip_no_ts + skip_bounds + skip_no_hist_run + skip_nan_meas:,}\n")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="STGNN data completeness audit")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg       = load_yaml(args.config)
    data_cfg  = cfg["data"]
    stgnn_cfg = cfg.get("stgnn2", {})

    train_ids = [str(s) for s in data_cfg["files"]]
    val_ids   = [str(s) for s in data_cfg["val_files"]]
    all_ids   = train_ids + val_ids
    N_train   = len(train_ids)

    measurement_cols = stgnn_cfg.get("measurement_features")
    icond2_features  = stgnn_cfg.get("icond2_features")
    run_hours        = tuple(stgnn_cfg.get("icond2_run_hours", [6, 9, 12, 15]))
    next_n_icond2    = stgnn_cfg.get("next_n_icond2", 4)
    n_workers        = stgnn_cfg.get("n_workers", 8)
    data_path        = data_cfg["path"]
    nwp_path         = data_cfg.get("nwp_path")
    H                = stgnn_cfg.get("history_length", 48)
    F_h              = stgnn_cfg.get("forecast_horizon", 48)

    test_start = data_cfg.get("test_start")
    test_end   = data_cfg.get("test_end")
    run_cutoff = pd.Timestamp(test_end, tz="UTC") if test_end else None

    out = StringIO()
    config_stem = Path(args.config).stem.replace("config_", "")
    out.write(f"DATA AUDIT REPORT — {config_stem}\n")
    out.write(f"Config : {args.config}\n")
    out.write(f"Stations: {len(all_ids)}  (train: {N_train}  val: {len(val_ids)})\n")

    # ----- Measurements -----
    print("[1/3] Loading measurements …")
    meas, timestamps = load_station_measurements(data_path, all_ids, cols=measurement_cols)
    T = len(timestamps)

    test_start_ts = pd.Timestamp(test_start, tz="UTC") if test_start else None
    if test_start_ts:
        split_t = int(np.searchsorted(timestamps, test_start_ts, side="left"))
    else:
        split_t = int(T * 0.8)
    split_time = timestamps[split_t]

    audit_measurements(out, meas, timestamps, all_ids, measurement_cols)

    # ----- ICON-D2 -----
    print("[2/3] Loading ICON-D2 ML runs …")
    lats, lons, alts = load_station_metadata(data_path, all_ids)
    station_coords   = np.stack([lats, lons], axis=1)

    run_times, icond2_coords, grid_runs, station_nearest_grid = load_icond2_ml_runs(
        nwp_path=nwp_path,
        station_ids=all_ids,
        station_coords=station_coords,
        features=icond2_features,
        run_hours=run_hours,
        next_n_grid=next_n_icond2,
        n_workers=n_workers,
        cutoff=run_cutoff,
    )
    audit_icond2(out, grid_runs, run_times, icond2_coords, station_nearest_grid,
                 all_ids, icond2_features)

    # ----- Run pairs -----
    print("[3/3] Analysing run pairs …")
    audit_run_pairs(out, meas, timestamps, run_times, split_time, H, F_h)

    # ----- Write report -----
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"data_audit_{config_stem}.txt"
    report_path.write_text(out.getvalue())
    print(f"\nReport written → {report_path}")
    print(out.getvalue())


if __name__ == "__main__":
    main()