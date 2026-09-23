#!/usr/bin/env python3
"""MOS-Feature-Budget auf dem Testjahr: 3 Arme x {ws, full}, ein ctx-Load.

Aufbau des veroeffentlichten Testjahr-Laufs rekonstruiert aus
logs/mos_testyear_mos_regional.log: fit_stations=153 (Pool), eval_stations=50
(Teststationen), Fit bis 2025-08-01, Bewertung 2025-08-01..2026-08-01,
eval_window=val mit test_mode=True.

ws laeuft unter out_prefix "regress_testyear_", full unter "testyear_" mit dem
Stem-Zusatz _full. Beides kollidiert nicht mit den Artefakten vom 05.09.
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

CONFIG = "configs/baselines/config_wind_mos_testyear_pool153.yaml"


def make_args(mos_features, out_prefix):
    return Namespace(
        fold_idx=0, nwp_sources="both", eval_window="val", test_mode=True,
        out_prefix=out_prefix, per_lead=False, nwp_geometry=False, i2_hist=False,
        idw_n=0, n_fit_rows=0, subsample_seed=ds.SUBSAMPLE_SEED_DEFAULT, n_jobs=32,
        hpo_study="none", mos_features=mos_features, mos_k_i2=4, mos_k_e2=4,
    )


def main():
    t0 = time.time()
    ctx = ds.load_context(CONFIG)
    print(f"ctx loaded in {time.time()-t0:.1f}s", flush=True)
    N_train, N_val = ctx["N_train"], ctx["N_val"]
    train_pos = np.arange(N_train)
    val_pos = np.arange(N_train, N_train + N_val)
    print(f"fit_stations={N_train} eval_stations={N_val}", flush=True)
    assert N_train == 153 and N_val == 50, (N_train, N_val)

    fit_pairs, _ = ds.build_run_pairs(ctx, None, ctx["val_start"])
    eval_pairs, _ = ds.build_run_pairs(ctx, ctx["val_start"], ctx["test_start"])
    print(f"fit_pairs={len(fit_pairs)} eval_pairs={len(eval_pairs)}  "
          f"(veroeffentlicht: 2933 / 1450)", flush=True)

    results = []
    for mos_features, prefix in (("ws", "regress_testyear_"), ("full", "testyear_")):
        for arm in ("mos_regional", "mos_nearest", "mos_local"):
            t1 = time.time()
            args = make_args(mos_features, prefix)
            try:
                station_df, _ = run_mos(ctx, args, fit_pairs, eval_pairs, train_pos, val_pos, arm)
            except Exception as exc:                          # noqa: BLE001
                print(f"  FEHLER {arm} {mos_features}: {exc!r}", flush=True)
                results.append(dict(arm=arm, features=mos_features, error=repr(exc)))
                continue
            rec = dict(arm=arm, features=mos_features, n_stations=int(len(station_df)),
                       mean_rmse=float(station_df["rmse"].mean()),
                       mean_r2=float(station_df["r2"].mean()),
                       mean_skill_nwp=float(station_df["skill_nwp"].mean()),
                       n_nan=int(station_df["rmse"].isna().sum()),
                       seconds=round(time.time() - t1, 1))
            results.append(rec)
            print(f"  {arm:14s} {mos_features:5s} n={rec['n_stations']} "
                  f"rmse={rec['mean_rmse']:.4f} r2={rec['mean_r2']:.4f} "
                  f"skill={rec['mean_skill_nwp']:.4f} nan={rec['n_nan']} "
                  f"({rec['seconds']}s)", flush=True)

    out = ROOT / "data/test_results/mos_feature_budget_testyear.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nZusammenfassung -> {out}", flush=True)


if __name__ == "__main__":
    main()
