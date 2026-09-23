#!/usr/bin/env python3
"""Was ist die Konditionierung auf die Tageszeit wert?

MOS-regional mit vollem Praediktorensatz, einmal wie bisher je (Laufstunde,
Vorlaufzeit) und einmal nur je Vorlaufzeit. Die zweite Fassung hat vierfach so
viele Zeilen je Zelle und verliert genau die Information ueber die Tageszeit,
die den Graphmodellen fehlt -- sie haben kein Kalendermerkmal.
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

VAL = {0:"configs/baselines/config_wind_qrf_local_fold1.yaml",
       1:"configs/baselines/config_wind_qrf_local_fold2.yaml",
       2:"configs/baselines/config_wind_qrf_local_fold3.yaml"}
TEST = "configs/baselines/config_wind_mos_testyear_pool153.yaml"

def mk(fold, strat, prefix, test_mode):
    return Namespace(fold_idx=fold, nwp_sources="both", eval_window="val",
                     test_mode=test_mode, out_prefix=prefix, per_lead=False,
                     nwp_geometry=False, i2_hist=False, idw_n=0, n_fit_rows=0,
                     subsample_seed=ds.SUBSAMPLE_SEED_DEFAULT, n_jobs=32,
                     hpo_study="none", mos_features="full", mos_k_i2=4,
                     mos_k_e2=4, mos_hist="none", mos_strat=strat)

res = []
def one(ctx, fold, strat, prefix, test_mode):
    N_tr, N_va = ctx["N_train"], ctx["N_val"]
    fp,_ = ds.build_run_pairs(ctx, None, ctx["val_start"])
    ep,_ = ds.build_run_pairs(ctx, ctx["val_start"], ctx["test_start"])
    t0 = time.time()
    sdf,_ = run_mos(ctx, mk(fold, strat, prefix, test_mode), fp, ep,
                    np.arange(N_tr), np.arange(N_tr, N_tr+N_va), "mos_regional")
    r = dict(fold=fold, strat=strat, split="test" if test_mode else "val",
             rmse=float(sdf.rmse.mean()), r2=float(sdf.r2.mean()),
             n_nan=int(sdf.rmse.isna().sum()), seconds=round(time.time()-t0,1))
    res.append(r)
    print(f"  strat={strat:12s} fold={fold} rmse={r['rmse']:.4f} r2={r['r2']:.4f} "
          f"nan={r['n_nan']} ({r['seconds']}s)", flush=True)

for fold in (0,1,2):
    print(f"\n===== VAL FOLD {fold} =====", flush=True)
    ctx = ds.load_context(VAL[fold])
    one(ctx, fold, "lead", "", False)
    del ctx
print("\n===== TESTJAHR =====", flush=True)
ctx = ds.load_context(TEST)
one(ctx, 0, "lead", "testyear_", True)
(ROOT/"data/test_results/mos_strat_summary.json").write_text(json.dumps(res, indent=1))
print("\nfertig", flush=True)
