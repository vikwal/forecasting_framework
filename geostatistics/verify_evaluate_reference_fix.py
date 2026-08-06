#!/usr/bin/env python3
"""Verification script for evaluate_reference.py val-mode fix (Aufgabe D).

Run from repo root on l1 with the frcst venv active.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error

RAW_DIR = Path("data/raw_preds")
RES_DIR = Path("data/test_results")
STATION_DIR = Path("/mnt/nvme1/synthetic/raw/wind")

EXPECTED_T0 = pd.Timestamp("2024-08-01 07:00:00", tz="UTC")
EXPECTED_T1 = pd.Timestamp("2025-08-02 15:00:00", tz="UTC")
EXPECTED_ROWS = 3_574_080

_mask_cache: dict[str, pd.Series] = {}


def imputed_mask_for_station(sid: str, valid_times: pd.DatetimeIndex) -> np.ndarray:
    """True where the *original* raw measurement is NaN at that hour (=> imputed)."""
    if sid not in _mask_cache:
        fpath = STATION_DIR / f"Station_{sid}.parquet"
        df = pd.read_parquet(fpath, columns=["wind_speed"])
        df.index = pd.to_datetime(df.index, utc=True)
        hourly = df["wind_speed"].resample("1h", closed="left", label="left").mean()
        _mask_cache[sid] = hourly
    hourly = _mask_cache[sid]
    reindexed = hourly.reindex(valid_times)
    return reindexed.isna().values


def rmse(g, p):
    ok = ~(np.isnan(g) | np.isnan(p))
    if ok.sum() < 2:
        return np.nan
    return float(math.sqrt(mean_squared_error(g[ok], p[ok])))


def per_station_rmse(df: pd.DataFrame, pred_col: str, gt_col: str = "gt") -> pd.Series:
    out = {}
    for sid, g in df.groupby("station_id"):
        out[sid] = rmse(g[gt_col].values, g[pred_col].values)
    return pd.Series(out)


def main():
    fold_config = {
        0: "configs/dcrnn/stdhp/config_wind_dcrnn_stdhp_fold1.yaml",
        1: "configs/dcrnn/stdhp/config_wind_dcrnn_stdhp_fold2.yaml",
        2: "configs/dcrnn/stdhp/config_wind_dcrnn_stdhp_fold3.yaml",
    }

    all_icon_full_station_means = []
    all_icon_masked_station_means = []

    for n, cfgpath in fold_config.items():
        model_pq = RAW_DIR / f"stdhp_dcrnn_wind_dcrnn_fold{n}_raw.parquet"
        icon_pq  = RAW_DIR / f"icon_d2_fold{n}_raw.parquet"
        ecmwf_pq = RAW_DIR / f"ecmwf_fold{n}_raw.parquet"
        icon_csv = RES_DIR / f"icon_d2_fold{n}.csv"
        ecmwf_csv = RES_DIR / f"ecmwf_fold{n}.csv"

        if not model_pq.exists() or not icon_pq.exists():
            print(f"=== FOLD {n}: missing outputs, skipping (model={model_pq.exists()} icon={icon_pq.exists()}) ===")
            continue

        print(f"\n{'='*90}\nFOLD {n}\n{'='*90}")

        with open(cfgpath) as fh:
            cfg = yaml.safe_load(fh)
        val_files = set(str(s) for s in cfg["data"]["val_files"])

        model_df = pd.read_parquet(model_pq)
        icon_df  = pd.read_parquet(icon_pq)
        ecmwf_df = pd.read_parquet(ecmwf_pq) if ecmwf_pq.exists() else None

        # ---- D1: time window ----
        print("--- D1: time window (valid_time min/max) ---")
        for name, df in [("model", model_df), ("icon_d2_new", icon_df)] + ([("ecmwf_new", ecmwf_df)] if ecmwf_df is not None else []):
            vmin, vmax = df.valid_time.min(), df.valid_time.max()
            ok = (vmin == EXPECTED_T0) and (vmax == EXPECTED_T1)
            print(f"  {name:15s} {vmin} .. {vmax}   match_expected={ok}")

        # ---- D2: station sets ----
        print("--- D2: station sets ---")
        model_stations = set(model_df.station_id.unique())
        icon_stations  = set(icon_df.station_id.unique())
        print(f"  val_files (config): {len(val_files)}")
        print(f"  model stations:     {len(model_stations)}  overlap w/ val_files={len(model_stations & val_files)}")
        print(f"  icon_d2 stations:   {len(icon_stations)}  overlap w/ val_files={len(icon_stations & val_files)}")
        print(f"  model == icon_d2 station set: {model_stations == icon_stations}")

        # ---- D3: row counts ----
        print("--- D3: row counts ---")
        print(f"  model rows={len(model_df)}  icon_d2 rows={len(icon_df)}  expected={EXPECTED_ROWS}")
        if ecmwf_df is not None:
            print(f"  ecmwf rows={len(ecmwf_df)}")

        # ---- D4: ICON-D2 RMSE consistency, full sample ----
        print("--- D4: ICON-D2 RMSE consistency (FULL sample) ---")
        icon_csv_df = pd.read_csv(icon_csv) if icon_csv.exists() else None
        if icon_csv_df is not None:
            icon_csv_df = icon_csv_df.assign(station_id=icon_csv_df.station_id.astype(str).str.zfill(5))
            csv_mean = icon_csv_df["rmse"].mean()
            csv_std  = icon_csv_df["rmse"].std()
            print(f"  from icon_d2_fold{n}.csv (script's own per-station rmse):  mean={csv_mean:.4f}  std={csv_std:.4f}  n_stations={len(icon_csv_df)}")

        icon_rmse_own = per_station_rmse(icon_df, "pred", "gt")
        print(f"  recomputed from icon_d2 raw parquet (pred vs gt):            mean={icon_rmse_own.mean():.4f}  std={icon_rmse_own.std():.4f}  n={len(icon_rmse_own)}")

        model_nwp_rmse = per_station_rmse(model_df, "nwp_ref", "gt")
        print(f"  from MODEL raw parquet's nwp_ref column (ICON-D2 as-seen-by-model): mean={model_nwp_rmse.mean():.4f}  std={model_nwp_rmse.std():.4f}  n={len(model_nwp_rmse)}")

        pooled_icon = rmse(icon_df["gt"].values, icon_df["pred"].values)
        pooled_model_nwp = rmse(model_df["gt"].values, model_df["nwp_ref"].values)
        print(f"  [POOLED, not per-station] icon_d2 raw: {pooled_icon:.4f}   model nwp_ref: {pooled_model_nwp:.4f}")

        all_icon_full_station_means.append(icon_rmse_own.mean())

        # ---- D4b: ICON-D2 RMSE consistency, EXCLUDING imputed target hours ----
        print("--- D4b: ICON-D2 RMSE consistency (EXCLUDING imputed target hours) ---")
        # Build mask per row: True = imputed (exclude)
        def add_imputed_flag(df):
            flags = np.zeros(len(df), dtype=bool)
            for sid, idx in df.groupby("station_id").groups.items():
                vt = pd.DatetimeIndex(df.loc[idx, "valid_time"])
                flags[df.index.get_indexer(idx)] = imputed_mask_for_station(sid, vt)
            return flags

        icon_df["_imputed"] = add_imputed_flag(icon_df)
        model_df["_imputed"] = add_imputed_flag(model_df)

        frac_imputed_icon = icon_df["_imputed"].mean()
        frac_imputed_model = model_df["_imputed"].mean()
        print(f"  fraction of rows with imputed target: icon_d2={frac_imputed_icon:.4%}  model={frac_imputed_model:.4%}")

        icon_df_clean = icon_df[~icon_df["_imputed"]]
        model_df_clean = model_df[~model_df["_imputed"]]

        icon_rmse_clean = per_station_rmse(icon_df_clean, "pred", "gt")
        model_nwp_rmse_clean = per_station_rmse(model_df_clean, "nwp_ref", "gt")
        print(f"  icon_d2 raw (masked), per-station:   mean={icon_rmse_clean.mean():.4f}  std={icon_rmse_clean.std():.4f}")
        print(f"  model nwp_ref (masked), per-station: mean={model_nwp_rmse_clean.mean():.4f}  std={model_nwp_rmse_clean.std():.4f}")
        print(f"  delta (full - masked) icon_d2 per-station mean: {icon_rmse_own.mean() - icon_rmse_clean.mean():.4f}")

        pooled_icon_clean = rmse(icon_df_clean["gt"].values, icon_df_clean["pred"].values)
        pooled_model_nwp_clean = rmse(model_df_clean["gt"].values, model_df_clean["nwp_ref"].values)
        print(f"  [POOLED, not per-station] icon_d2 masked: {pooled_icon_clean:.4f}   model nwp_ref masked: {pooled_model_nwp_clean:.4f}")

        all_icon_masked_station_means.append(icon_rmse_clean.mean())

        # ---- D5: ECMWF RMSE per fold ----
        if ecmwf_df is not None:
            print("--- D5: ECMWF RMSE (per station mean +/- SD over stations) ---")
            ecmwf_csv_df = pd.read_csv(ecmwf_csv) if ecmwf_csv.exists() else None
            if ecmwf_csv_df is not None:
                print(f"  FULL sample, from ecmwf_fold{n}.csv:      mean={ecmwf_csv_df['rmse'].mean():.4f}  std={ecmwf_csv_df['rmse'].std():.4f}  n={len(ecmwf_csv_df)}")
            ecmwf_df["_imputed"] = add_imputed_flag(ecmwf_df)
            ecmwf_clean = ecmwf_df[~ecmwf_df["_imputed"]]
            ecmwf_rmse_full = per_station_rmse(ecmwf_df, "pred", "gt")
            ecmwf_rmse_clean = per_station_rmse(ecmwf_clean, "pred", "gt")
            print(f"  FULL sample, recomputed from raw parquet: mean={ecmwf_rmse_full.mean():.4f}  std={ecmwf_rmse_full.std():.4f}")
            print(f"  MASKED (excl. imputed), recomputed:       mean={ecmwf_rmse_clean.mean():.4f}  std={ecmwf_rmse_clean.std():.4f}")

            # ---- D6: stations where ECMWF beats ICON-D2 ----
            print("--- D6: stations where ECMWF RMSE < ICON-D2 RMSE (full sample, per-station) ---")
            common = icon_rmse_own.index.intersection(ecmwf_rmse_full.index)
            better = (ecmwf_rmse_full.loc[common] < icon_rmse_own.loc[common]).sum()
            print(f"  ECMWF better at {better} / {len(common)} stations")
        else:
            print("--- D5/D6: no ECMWF output for this fold ---")

    print(f"\n{'='*90}\nOVERALL (across available folds)\n{'='*90}")
    if all_icon_full_station_means:
        arr = np.array(all_icon_full_station_means)
        print(f"ICON-D2 per-station-mean RMSE, FULL sample, averaged over folds: {arr.mean():.4f} +/- {arr.std():.4f}  (per-fold: {[round(x,4) for x in arr]})")
        print("  Expectation from brief: 1.3252 +/- 0.0288")
    if all_icon_masked_station_means:
        arr2 = np.array(all_icon_masked_station_means)
        print(f"ICON-D2 per-station-mean RMSE, MASKED (excl. imputed), averaged over folds: {arr2.mean():.4f} +/- {arr2.std():.4f}  (per-fold: {[round(x,4) for x in arr2]})")


if __name__ == "__main__":
    main()
