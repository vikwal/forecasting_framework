#!/usr/bin/env python3
"""Runs ON l2 (frcst venv) from ~/Work/forecasting_framework.

Exports the per-station and stratified metrics for the paper
Graphs_Wind_Speed_Forecasting, for BOTH evaluation splits:

  val  : 3-fold spatial CV on the validation year 2024-08-01 .. 2025-07-31
         (51 held-out stations per fold, union = 153), tuned retrains (§15/§16),
         baselines (MOS 2nwp, TFT, raw ICON-D2 / ECMWF from evaluate_reference).
  test : --test-mode, 50 never-seen test stations, test year 2025-08-01 ..
         2026-07-31 (§19); HIST arms = expanding-window retrains (3 chunks);
         raw references from evaluate_reference --test-mode --fold-idx 7.

Filtering identical to docs/evaluation_results.md §14-19: target hours whose
observation was imputed are excluded (build_imputation_mask / _lookup_imputed).
Aggregation: per-station metric first, station mean afterwards (paper convention).

Usage:  python scripts/export_paper_metrics.py [val|test|all] [arm ...]

Outputs (CSV) in paper_export/ (appended/replaced per (split, arm)):
  per_station.csv   split, arm, fold, station_id, n, rmse, mae, r2, bias
  horizon.csv       split, arm, runs(all|09), horizon, rmse   (station mean)
  month.csv         split, arm, month, rmse                   (station mean)
  wsclass.csv       split, arm, classby(gt|nwp), cls, rmse, n  (station mean)
  chunk.csv         split, arm, chunk, rmse                    (station mean)
  coverage.csv      split, arm, rows_raw, rows_kept, n_stations
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from geostatistics.stdrun.make_stdhp_figures import (  # noqa: E402
    norm_station, build_imputation_mask, _lookup_imputed)

RAW = Path("data/raw_preds")
OUT = Path("paper_export"); OUT.mkdir(exist_ok=True)
_CANDIDATES = [Path("/mnt/nvme1/synthetic/raw/wind"), Path("/mnt/lambda1/nvme1/synthetic/raw/wind")]
STATION_RAW_DIR = next((c for c in _CANDIDATES if c.is_dir()), _CANDIDATES[-1])
COLS = ["station_id", "run_time", "valid_time", "horizon", "pred", "gt", "nwp_ref", "pers_ref"]

VAL = {
    "icon":          [f"from_l1/icon_d2_fold{i}" for i in (0, 1, 2)],
    "ecmwf":         [f"from_l1/ecmwf_fold{i}" for i in (0, 1, 2)],
    "dcrnn":         [f"retrain_dcrnn_fold{i}" for i in (1, 2, 3)],
    "dcrnn_idw":     [f"retrain_dcrnn_idw_alt_fold{i}" for i in (1, 2, 3)],
    "dcrnn_base":    [f"retrain_dcrnn_base_fold{i}" for i in (1, 2, 3)],
    "dcrnn_nomeas":  [f"retrain_dcrnn_nomeas_fold{i}" for i in (1, 2, 3)],
    "dcrnn_nograph": [f"retrain_dcrnn_nograph_fold{i}" for i in (1, 2, 3)],
    "dcrnn_hist":    [f"retrain_dcrnn_nwp_hist_fold{i}" for i in (1, 2, 3)],
    "mtgnn":         [f"retrain_mtgnn_nwp_fold{i}" for i in (1, 2, 3)],
    "mtgnn_base":    [f"retrain_mtgnn_fold{i}" for i in (1, 2, 3)],
    "mtgnn_hist":    [f"retrain_mtgnn_nwp_hist_fold{i}" for i in (1, 2, 3)],
    "wavenet":       [f"retrain_wavenet_nwp_fold{i}" for i in (1, 2, 3)],
    "wavenet_base":  [f"retrain_wavenet_fold{i}" for i in (1, 2, 3)],
    "tft":           [f"retrain_tft_sp_base_fold{i}" for i in (1, 2, 3)],
    "tft_hist":      [f"retrain_tft_sp_hist_fold{i}" for i in (1, 2, 3)],
    "mos_reg":       [f"mos_regional_2nwp_fold{i}" for i in (0, 1, 2)],
    "mos_near":      [f"mos_nearest_2nwp_fold{i}" for i in (0, 1, 2)],
    "mos_loc":       [f"mos_local_2nwp_fold{i}" for i in (0, 1, 2)],
    # --- added 2026-09-05: seed replicates of A and D' (new imputation chain, see seedrep_worker.sh on ws)
    "dcrnn_rep2":     [f"rep2_dcrnn_fold{i}" for i in (1, 2, 3)],
    "dcrnn_rep3":     [f"rep3_dcrnn_fold{i}" for i in (1, 2, 3)],
    "dcrnn_idw_rep2": [f"rep2_dcrnn_idw_alt_fold{i}" for i in (1, 2, 3)],
    "dcrnn_idw_rep3": [f"rep3_dcrnn_idw_alt_fold{i}" for i in (1, 2, 3)],
}
TEST = {
    "dcrnn":      ["testmode_dcrnn"],
    "dcrnn_idw":  ["testmode_dcrnn_idw_alt"],
    "mtgnn":      ["testmode_mtgnn_nwp"],
    "dcrnn_hist": [f"testmode_dcrnn_nwp_hist_s{s}" for s in (1, 2, 3)],
    "mtgnn_hist": [f"testmode_mtgnn_nwp_hist_s{s}" for s in (1, 2, 3)],
    "icon":       ["icon_d2_test_fold7"],
    "ecmwf":      ["ecmwf_test_fold7"],
    # --- added 2026-09-05; a missing parquet is skipped, so these may be listed early ---
    # fixed-epoch test-year models (--fixed-epochs, no checkpoint selection on test stations)
    "dcrnn_fe":       ["testmode_fe_dcrnn"],
    "dcrnn_idw_fe":   ["testmode_fe_dcrnn_idw_alt"],
    "mtgnn_fe":       ["testmode_fe_mtgnn_nwp"],
    "dcrnn_hist_fe":  [f"testmode_fe_dcrnn_nwp_hist_s{s}" for s in (1, 2, 3)],
    "mtgnn_hist_fe":  [f"testmode_fe_mtgnn_nwp_hist_s{s}" for s in (1, 2, 3)],
    # HIST arms trained once (step-1 checkpoint), scored over the full year: retraining contrast
    "dcrnn_hist_once": ["testmode_dcrnn_nwp_hist_once"],
    "mtgnn_hist_once": ["testmode_mtgnn_nwp_hist_once"],
    # MOS in --test-mode on the test year (configs/baselines/config_wind_mos_testyear_fold1.yaml)
    "mos_reg":  ["testyear_mos_regional_2nwp_test_fold0"],
    "mos_near": ["testyear_mos_nearest_2nwp_test_fold0"],
    "mos_loc":  ["testyear_mos_local_2nwp_test_fold0"],
    # TFT sp_base / sp_hist in --test-mode on the test year
    "tft":      ["testmode_tft_base"],
    "tft_hist": ["testmode_tft_hist"],
}
CHUNKS = [("Aug-Nov 2025", "2025-08-01", "2025-12-01"),
          ("Dec 2025-Mar 2026", "2025-12-01", "2026-04-01"),
          ("Apr-Jul 2026", "2026-04-01", "2026-08-01")]
VAL_CHUNKS = [("Aug-Nov 2024", "2024-08-01", "2024-12-01"),
              ("Dec 2024-Mar 2025", "2024-12-01", "2025-04-01"),
              ("Apr-Jul 2025", "2025-04-01", "2025-08-01")]
WS_BINS = [0, 2, 4, 6, 8, 10, 12, 100]
WS_LAB = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", ">12"]


def station_metrics(sid_code, n_st, pred, gt):
    e = pred - gt
    n = np.bincount(sid_code, minlength=n_st).astype(float)
    se = np.bincount(sid_code, weights=e, minlength=n_st)
    se2 = np.bincount(sid_code, weights=e * e, minlength=n_st)
    sae = np.bincount(sid_code, weights=np.abs(e), minlength=n_st)
    sg = np.bincount(sid_code, weights=gt, minlength=n_st)
    sg2 = np.bincount(sid_code, weights=gt * gt, minlength=n_st)
    var = sg2 / n - (sg / n) ** 2
    return pd.DataFrame({"n": n.astype(int), "rmse": np.sqrt(se2 / n), "mae": sae / n,
                         "r2": 1 - (se2 / n) / var, "bias": se / n})


def strat_rmse(sid_code, n_st, key_code, n_key, pred, gt):
    """per-station RMSE per stratum, then station mean -> array over strata (nan if empty)"""
    e2 = (pred - gt) ** 2
    flat = key_code * n_st + sid_code
    n = np.bincount(flat, minlength=n_key * n_st).astype(float).reshape(n_key, n_st)
    s = np.bincount(flat, weights=e2, minlength=n_key * n_st).reshape(n_key, n_st)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.sqrt(s / n)
    out = np.nanmean(np.where(n > 0, per, np.nan), axis=1) if n_key else np.array([])
    return out, n.sum(axis=1)


def chunk_code(rt, chunks):
    code = np.full(len(rt), -1, dtype=int)
    for i, (name, lo, hi) in enumerate(chunks):
        sel = (rt >= pd.Timestamp(lo, tz="UTC")) & (rt < pd.Timestamp(hi, tz="UTC"))
        code[sel.to_numpy()] = i
    return code


def load(stem, mask):
    p = RAW / f"{stem}_raw.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=COLS)
    df["station_id"] = df["station_id"].map(norm_station)
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df = df.dropna(subset=["pred", "gt"])
    n0 = len(df)
    df = df.loc[~_lookup_imputed(df, mask)].reset_index(drop=True)
    df.attrs["n_raw"] = n0
    return df


def process(split, arms, chunks, mask, rows, only=None):
    PS, HZ, MO, WC, CH, COV = rows
    for arm, stems in arms.items():
        if only and arm not in only:
            continue
        t0 = time.time()
        parts = []
        for k, stem in enumerate(stems):
            d = load(stem, mask)
            if d is None:
                print(f"  [{split}] {arm}: missing {stem}", flush=True)
                continue
            d["fold"] = k + 1 if split == "val" else 0
            parts.append(d)
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        COV.append(dict(split=split, arm=arm, rows_raw=sum(p.attrs["n_raw"] for p in parts),
                        rows_kept=len(df), n_stations=df["station_id"].nunique()))
        sid_cat = pd.Categorical(df["station_id"])
        sid_code = sid_cat.codes.astype(np.int64)
        sids = list(sid_cat.categories); n_st = len(sids)
        fold_of = df.groupby(sid_code)["fold"].first()
        gt = df["gt"].to_numpy(float)
        hz = df["horizon"].to_numpy(int)
        run09 = (df["run_time"].dt.hour == 9).to_numpy()
        vt = df["valid_time"]
        mcode_raw = (vt.dt.year * 12 + vt.dt.month - 1).to_numpy()
        m_cat = pd.Categorical(mcode_raw); mcode = m_cat.codes.astype(np.int64)
        mlab = [f"{c // 12}-{c % 12 + 1:02d}" for c in m_cat.categories]
        nwp = df["nwp_ref"].to_numpy(float)
        cls_gt = np.clip(np.searchsorted(WS_BINS, gt, side="right") - 1, 0, len(WS_LAB) - 1)
        cls_nwp = np.clip(np.searchsorted(WS_BINS, nwp, side="right") - 1, 0, len(WS_LAB) - 1)
        ccode = chunk_code(df["run_time"], chunks)
        preds = [("pred", arm)]
        if arm == list(arms)[0] or (only and arm == only[0]):
            preds += [("nwp_ref", "icon_ref"), ("pers_ref", "persistence")]
        for col, name in preds:
            pred = df[col].to_numpy(float)
            ok = ~np.isnan(pred)
            m = station_metrics(sid_code[ok], n_st, pred[ok], gt[ok])
            for i, r in m.iterrows():
                PS.append(dict(split=split, arm=name, fold=int(fold_of[i]), station_id=sids[i],
                               n=int(r.n), rmse=r.rmse, mae=r.mae, r2=r.r2, bias=r.bias))
            for runs, sel in (("all", ok), ("09", ok & run09)):
                s, _ = strat_rmse(sid_code[sel], n_st, hz[sel] - 1, 48, pred[sel], gt[sel])
                for h, v in enumerate(s):
                    HZ.append(dict(split=split, arm=name, runs=runs, horizon=h + 1, rmse=v))
            s, _ = strat_rmse(sid_code[ok], n_st, mcode[ok], len(mlab), pred[ok], gt[ok])
            for lab, v in zip(mlab, s):
                MO.append(dict(split=split, arm=name, month=lab, rmse=v))
            for by, cc in (("gt", cls_gt), ("nwp", cls_nwp)):
                s, cnt = strat_rmse(sid_code[ok], n_st, cc[ok], len(WS_LAB), pred[ok], gt[ok])
                for lab, v, c in zip(WS_LAB, s, cnt):
                    WC.append(dict(split=split, arm=name, classby=by, cls=lab, rmse=v, n=int(c)))
            selc = ok & (ccode >= 0)
            s, _ = strat_rmse(sid_code[selc], n_st, ccode[selc], len(chunks), pred[selc], gt[selc])
            for (lab, _, _), v in zip(chunks, s):
                CH.append(dict(split=split, arm=name, chunk=lab, rmse=v))
        print(f"  [{split}] {arm:14s} rows {len(df):>10,}  stations {n_st:3d}  "
              f"RMSE {m.rmse.mean() if preds[-1][1]==arm else station_metrics(sid_code, n_st, df['pred'].to_numpy(float), gt).rmse.mean():.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    only = sys.argv[2:] or None
    todo = []
    if which in ("val", "all"):
        todo.append(("val", VAL, VAL_CHUNKS))
    if which in ("test", "all"):
        todo.append(("test", TEST, CHUNKS))
    sids = set()
    for _, arms, _ in todo:
        for arm, stems in arms.items():
            if only and arm not in only:
                continue
            for s in stems:
                p = RAW / f"{s}_raw.parquet"
                if p.exists():
                    sids |= set(pd.read_parquet(p, columns=["station_id"])["station_id"]
                                .map(norm_station).unique())
    print(f"[i] {len(sids)} stations, building imputation mask from {STATION_RAW_DIR} ...", flush=True)
    mask = build_imputation_mask(sorted(sids), STATION_RAW_DIR)
    rows = ([], [], [], [], [], [])
    for split, arms, chunks in todo:
        process(split, arms, chunks, mask, rows, only)
    for name, r in zip(["per_station", "horizon", "month", "wsclass", "chunk", "coverage"], rows):
        new = pd.DataFrame(r)
        p = OUT / f"{name}.csv"
        if p.exists() and len(new):
            old = pd.read_csv(p, dtype={"station_id": str})
            key = old[["split", "arm"]].apply(tuple, axis=1)
            drop = set(new[["split", "arm"]].apply(tuple, axis=1))
            old = old[~key.isin(drop)]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(p, index=False)
        print(f"[i] wrote {p} ({len(new)} rows)")


if __name__ == "__main__":
    main()
