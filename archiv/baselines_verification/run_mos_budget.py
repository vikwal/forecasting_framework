#!/usr/bin/env python3
"""MOS mit dem Feature-Budget der Modelle -- drei Arme x drei Folds, ein ctx-Load je Fold.

Fuer jeden Fold laufen sechs Kombinationen:
  * --mos-features ws   unter out_prefix "regress_"  -> Regressionstest gegen die
    veroeffentlichten CSVs, ueberschreibt sie NICHT
  * --mos-features full unter dem Stem-Zusatz "_full"

Aufbau uebernommen von archiv/baselines_verification/run_all_mos_scratch.py.
Ruft dieselben run_mos()/assemble_and_save()-Funktionen wie die CLI.
"""
from __future__ import annotations
import json, os, sys, time
from argparse import Namespace
from pathlib import Path

ROOT = Path("~/Work/forecasting_framework").expanduser()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
from geostatistics.baselines import dataset as ds
from geostatistics.baselines.evaluate_baselines import run_mos

CONFIG_BY_FOLD = {
    0: "configs/baselines/config_wind_qrf_local_fold1.yaml",
    1: "configs/baselines/config_wind_qrf_local_fold2.yaml",
    2: "configs/baselines/config_wind_qrf_local_fold3.yaml",
}


def make_args(fold_idx, mos_features, out_prefix):
    return Namespace(
        fold_idx=fold_idx, nwp_sources="both", eval_window="val", test_mode=False,
        out_prefix=out_prefix, per_lead=False, nwp_geometry=False, i2_hist=False,
        idw_n=0, n_fit_rows=0, subsample_seed=ds.SUBSAMPLE_SEED_DEFAULT, n_jobs=32,
        hpo_study="none", mos_features=mos_features, mos_k_i2=4, mos_k_e2=4,
    )


def main():
    results = []
    for fold in (0, 1, 2):
        print(f"\n===== FOLD {fold} =====", flush=True)
        t0 = time.time()
        ctx = ds.load_context(CONFIG_BY_FOLD[fold])
        print(f"ctx loaded in {time.time()-t0:.1f}s", flush=True)
        N_train, N_val = ctx["N_train"], ctx["N_val"]
        train_pos = np.arange(N_train)
        val_pos = np.arange(N_train, N_train + N_val)
        fit_pairs, fit_c = ds.build_run_pairs(ctx, None, ctx["val_start"])
        eval_pairs, eval_c = ds.build_run_pairs(ctx, ctx["val_start"], ctx["test_start"])
        print(f"fit_pairs={len(fit_pairs)} eval_pairs={len(eval_pairs)}", flush=True)
        assert len(fit_pairs) == 1473, len(fit_pairs)
        assert len(eval_pairs) == 1460, len(eval_pairs)

        for mos_features, prefix in (("ws", "regress_"), ("full", "")):
            for arm in ("mos_regional", "mos_nearest", "mos_local"):
                t1 = time.time()
                args = make_args(fold, mos_features, prefix)
                try:
                    station_df, _raw = run_mos(ctx, args, fit_pairs, eval_pairs,
                                               train_pos, val_pos, arm)
                except Exception as exc:                      # noqa: BLE001
                    print(f"  FEHLER {arm} {mos_features} fold={fold}: {exc!r}", flush=True)
                    results.append(dict(fold=fold, arm=arm, features=mos_features,
                                        error=repr(exc)))
                    continue
                n_nan = int(station_df["rmse"].isna().sum())
                rec = dict(fold=fold, arm=arm, features=mos_features,
                           n_stations=int(len(station_df)),
                           mean_rmse=float(station_df["rmse"].mean()),
                           mean_r2=float(station_df["r2"].mean()),
                           mean_skill_nwp=float(station_df["skill_nwp"].mean()),
                           n_nan=n_nan, seconds=round(time.time() - t1, 1))
                results.append(rec)
                print(f"  {arm:14s} {mos_features:5s} fold={fold} "
                      f"n={rec['n_stations']} rmse={rec['mean_rmse']:.4f} "
                      f"r2={rec['mean_r2']:.4f} skill={rec['mean_skill_nwp']:.4f} "
                      f"nan={n_nan} ({rec['seconds']}s)", flush=True)
        del ctx

    out = ROOT / "data/test_results/mos_feature_budget_summary.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nZusammenfassung -> {out}", flush=True)


if __name__ == "__main__":
    main()
