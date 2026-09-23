#!/usr/bin/env python3
"""MOS-regional mit Messhistorie: window und compact, Validierungs- und Testjahr.

MOS-local bleibt unberuehrt -- es liest bei der Inferenz nichts und wird am Ort
gefittet, eine +hist-Variante ergaebe dort keinen Sinn.
"""
from __future__ import annotations
import json, os, sys, time
from argparse import Namespace
from pathlib import Path

ROOT = Path("~/Work/forecasting_framework").expanduser()
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np
from geostatistics.baselines import dataset as ds
from geostatistics.baselines.evaluate_baselines import run_mos

VAL_CFG = {0: "configs/baselines/config_wind_qrf_local_fold1.yaml",
           1: "configs/baselines/config_wind_qrf_local_fold2.yaml",
           2: "configs/baselines/config_wind_qrf_local_fold3.yaml"}
TEST_CFG = "configs/baselines/config_wind_mos_testyear_pool153.yaml"


def args_for(fold, hist, prefix, test_mode):
    return Namespace(fold_idx=fold, nwp_sources="both", eval_window="val",
                     test_mode=test_mode, out_prefix=prefix, per_lead=False,
                     nwp_geometry=False, i2_hist=False, idw_n=0, n_fit_rows=0,
                     subsample_seed=ds.SUBSAMPLE_SEED_DEFAULT, n_jobs=32,
                     hpo_study="none", mos_features="full", mos_k_i2=4,
                     mos_k_e2=4, mos_hist=hist)


def one(ctx, fold, hist, prefix, test_mode, results):
    N_tr, N_va = ctx["N_train"], ctx["N_val"]
    tr, va = np.arange(N_tr), np.arange(N_tr, N_tr + N_va)
    fit_pairs, _ = ds.build_run_pairs(ctx, None, ctx["val_start"])
    eval_pairs, _ = ds.build_run_pairs(ctx, ctx["val_start"], ctx["test_start"])
    t0 = time.time()
    try:
        sdf, _ = run_mos(ctx, args_for(fold, hist, prefix, test_mode),
                         fit_pairs, eval_pairs, tr, va, "mos_regional")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  FEHLER hist={hist} fold={fold}: {exc!r}", flush=True)
        results.append(dict(fold=fold, hist=hist, error=repr(exc))); return
    rec = dict(fold=fold, hist=hist, split="test" if test_mode else "val",
               n=int(len(sdf)), rmse=float(sdf.rmse.mean()), r2=float(sdf.r2.mean()),
               skill=float(sdf.skill_nwp.mean()), n_nan=int(sdf.rmse.isna().sum()),
               seconds=round(time.time() - t0, 1))
    results.append(rec)
    print(f"  hist={hist:8s} fold={fold} n={rec['n']} rmse={rec['rmse']:.4f} "
          f"r2={rec['r2']:.4f} skill={rec['skill']:.4f} nan={rec['n_nan']} "
          f"({rec['seconds']}s)", flush=True)


def main():
    results = []
    for fold in (0, 1, 2):
        print(f"\n===== VAL FOLD {fold} =====", flush=True)
        ctx = ds.load_context(VAL_CFG[fold])
        for hist in ("window", "compact"):
            one(ctx, fold, hist, "", False, results)
        del ctx
    print("\n===== TESTJAHR =====", flush=True)
    ctx = ds.load_context(TEST_CFG)
    for hist in ("window", "compact"):
        one(ctx, 0, hist, "testyear_", True, results)
    out = ROOT / "data/test_results/mos_hist_summary.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nZusammenfassung -> {out}", flush=True)


if __name__ == "__main__":
    main()
