#!/usr/bin/env python3
"""
geostatistics/baselines/evaluate_baselines.py — CLI: fit -> predict -> _save
for the QRF and MOS baselines.

Schwesterskript zu ``evaluate_reference.py`` (Spezifikation 2.1): importiert
``_station_metrics``/``_save`` von dort, statt sie nachzubauen, damit das
Ausgabeformat nicht driftet.

Usage
-----
    cd ~/Work/forecasting_framework
    CUDA_VISIBLE_DEVICES="" nice -n 19 python geostatistics/baselines/evaluate_baselines.py \\
        -c configs/baselines/config_wind_qrf_local_fold1.yaml --fold-idx 0 \\
        --arm mos_regional --nwp-sources both
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from geostatistics.baselines import dataset as ds          # noqa: E402
from geostatistics.baselines import mos as mos_mod          # noqa: E402
from geostatistics.baselines import qrf as qrf_mod          # noqa: E402
from geostatistics.evaluate_reference import _station_metrics, _save  # noqa: E402
from geostatistics.spatial_cv import fold_hash              # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_baselines")

MOS_ARMS = ("mos_regional", "mos_nearest", "mos_local")
QRF_ARMS = ("qrf_local", "qrf_idw")
ALL_ARMS = QRF_ARMS + MOS_ARMS + ("all",)

# Retraining-Szenario (F2, Nutzerentscheidung 2026-08-10): Fit-Verlaengerung bis
# 2025-12-01, Bewertung 2025-12-01 .. 2026-04-01. Nur der CODE-PFAD, in dieser
# Phase NICHT gefahren (--dry-run-Beleg genuegt).
RETRAIN_FIT_END = pd.Timestamp("2025-12-01", tz="UTC")
RETRAIN_EVAL_END = pd.Timestamp("2026-04-01", tz="UTC")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "<unknown>"


def _study_name_for_config(config_path: str, cfg: dict) -> str:
    """Identisch zu hpo_qrf.py — dieselbe Formel wie hpo_mtgnn.py:344, damit
    QRF-local UND QRF-IDW (F1: "uebernimmt die best_params von QRF-local",
    ohne eigene Studie) denselben Studiennamen auflösen."""
    import re
    stem = Path(config_path).stem.replace("config_", "")
    hpo_stem = re.sub(r"_fold\d+$", "", stem)
    qcfg = cfg.get("qrf", {})
    F_h = qcfg.get("forecast_horizon", 48)
    freq = cfg["data"].get("freq", "1h")
    return f"cl_m-qrf_out-{F_h}_freq-{freq}_{hpo_stem}"


def _load_best_params(study_name: str) -> dict | None:
    import os
    import optuna
    storage_url = os.environ.get("OPTUNA_STORAGE")
    if not storage_url:
        raise RuntimeError("OPTUNA_STORAGE not set — required for --hpo-study auto")
    storage = optuna.storages.RDBStorage(
        url=storage_url, engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600},
    )
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        logger.warning("Optuna study %s not found — falling back to fixed defaults", study_name)
        return None
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        logger.warning("Optuna study %s has no COMPLETE trials yet — falling back to fixed defaults", study_name)
        return None
    logger.info("Loaded best_params from %s (best_value=%.4f, %d COMPLETE trials): %s",
                study_name, study.best_value, len(completed), study.best_params)
    return dict(study.best_params)


def _resolve_qrf_params(args, cfg: dict) -> dict:
    defaults = dict(n_estimators=200, min_samples_leaf=2e-5, max_features=0.33, max_depth=30)
    if args.hpo_study == "auto":
        study_name = _study_name_for_config(args.config, cfg)
        best = _load_best_params(study_name)
        if best:
            defaults.update(best)
    defaults["random_state"] = cfg.get("qrf", {}).get("random_state", 20260810)
    defaults["n_jobs"] = args.n_jobs
    return defaults


def _window_bounds(ctx: dict, args) -> tuple[pd.Timestamp | None, pd.Timestamp | None,
                                              pd.Timestamp | None, pd.Timestamp | None]:
    """(fit_lo, fit_hi, eval_lo, eval_hi)."""
    val_start = ctx["val_start"]
    test_start = ctx["test_start"]
    test_end = ctx["test_end"]
    if args.eval_window == "val":
        return None, val_start, val_start, test_start
    if args.eval_window == "test":
        return None, val_start, test_start, test_end
    if args.eval_window == "retrain":
        return None, RETRAIN_FIT_END, RETRAIN_FIT_END, RETRAIN_EVAL_END
    raise ValueError(args.eval_window)


def _stem_suffix(args) -> str:
    if args.test_mode:
        return "_test"
    if args.eval_window == "test":
        return "_tw"
    if args.eval_window == "retrain":
        return "_retrain"
    return ""


def assemble_and_save(
    preds_flat: np.ndarray, meta: pd.DataFrame, val_ids: list[str],
    S: int, P: int, F_h: int, stem: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """preds_flat and meta rows must be in station-major/pair/lead order
    (exactly what ``build_feature_matrix``/``build_mos_rows`` produce)."""
    pred_arr = np.asarray(preds_flat, dtype=np.float64).reshape(S, P, F_h)
    gt_arr = meta["gt"].to_numpy(dtype=np.float64).reshape(S, P, F_h)
    nwp_arr = meta["nwp_ref"].to_numpy(dtype=np.float64).reshape(S, P, F_h)
    pers_arr = meta["pers_ref"].to_numpy(dtype=np.float64).reshape(S, P, F_h)

    preds_acc = [list(pred_arr[i]) for i in range(S)]
    gts_acc = [list(gt_arr[i]) for i in range(S)]
    nwp_acc = [list(nwp_arr[i]) for i in range(S)]
    pers_acc = [list(pers_arr[i]) for i in range(S)]

    station_df = _station_metrics(preds_acc, gts_acc, nwp_acc, pers_acc, val_ids)

    raw_df = meta.copy()
    raw_df["pred"] = preds_flat
    raw_df = raw_df[["station_id", "run_time", "valid_time", "horizon", "pred", "gt", "nwp_ref", "pers_ref"]]
    _save(station_df, raw_df, stem=stem)
    return station_df, raw_df


# ---------------------------------------------------------------------------
# Per-arm fit/predict
# ---------------------------------------------------------------------------

def run_qrf_local(ctx, args, fit_pairs, eval_pairs, train_pos, val_pos, arm_label="qrf_local"):
    idw_n = args.idw_n if arm_label == "qrf_idw" else 0
    idw_neighbour_pos = train_pos if arm_label == "qrf_idw" else None

    X_fit, y_fit, cols, _ = ds.build_feature_matrix(
        ctx, train_pos, fit_pairs, k_i2=4, k_e2=4, need_meta=False,
        nwp_geometry=args.nwp_geometry, i2_hist=args.i2_hist,
        idw_neighbour_pos=idw_neighbour_pos, idw_n=idw_n,
    )
    logger.info("%s fit matrix: %s rows x %d cols", arm_label, f"{X_fit.shape[0]:,}", X_fit.shape[1])

    params = _resolve_qrf_params(args, ctx["cfg"])
    logger.info("%s params: %s", arm_label, params)

    if args.per_lead:
        horizon_col = cols.index("horizon")
        models = qrf_mod.fit_per_lead(X_fit, y_fit, horizon_col, X_fit[:, horizon_col], **params)
    else:
        X_sub, y_sub, sub_idx = qrf_mod.subsample_rows(X_fit, y_fit, args.n_fit_rows, args.subsample_seed)
        logger.info("%s: fitting on %s of %s rows (seed=%d)",
                    arm_label, f"{len(y_sub):,}", f"{len(y_fit):,}", args.subsample_seed)
        model = qrf_mod.fit(X_sub, y_sub, **params)

    X_eval, y_eval, _, meta = ds.build_feature_matrix(
        ctx, val_pos, eval_pairs, k_i2=4, k_e2=4, need_meta=True,
        nwp_geometry=args.nwp_geometry, i2_hist=args.i2_hist,
        idw_neighbour_pos=idw_neighbour_pos, idw_n=idw_n,
    )
    if args.per_lead:
        horizon_col = cols.index("horizon")
        preds = qrf_mod.predict_per_lead(models, X_eval, horizon_col, X_eval[:, horizon_col])
    else:
        preds = qrf_mod.predict(model, X_eval)

    val_ids = [ctx["all_ids"][i] for i in val_pos]
    stem = f"{args.out_prefix}{arm_label}" + (f"_n{idw_n}" if arm_label == "qrf_idw" else "") \
        + f"{_stem_suffix(args)}_fold{args.fold_idx}"
    return assemble_and_save(preds, meta, val_ids, len(val_pos), len(eval_pairs), ctx["F_h"], stem)


def run_mos(ctx, args, fit_pairs, eval_pairs, train_pos, val_pos, arm: str):
    nwp_sources = args.nwp_sources
    val_ids = [ctx["all_ids"][i] for i in val_pos]
    train_ids = [ctx["all_ids"][i] for i in train_pos]
    suffix_2nwp = "_2nwp" if nwp_sources == "both" else ""

    if arm == "mos_regional":
        rows_train = ds.build_mos_rows(ctx, train_pos, fit_pairs, nwp_sources)
        betas = mos_mod.fit_regional(rows_train, nwp_sources)
        rows_eval = ds.build_mos_rows(ctx, val_pos, eval_pairs, nwp_sources)
        preds = mos_mod.predict_with_regional(rows_eval, betas, nwp_sources)

    elif arm == "mos_nearest":
        rows_train = ds.build_mos_rows(ctx, train_pos, fit_pairs, nwp_sources)
        per_station_betas = mos_mod.fit_per_station(rows_train, nwp_sources)
        # Jede Zielstation uebernimmt die Koeffizienten der geodaetisch naechsten
        # Trainingsstation (Spezifikation 3.5) — ueber pairwise_geodesic_km,
        # NICHT euklidisch in Grad (B2).
        target_betas: dict[str, dict[int, np.ndarray]] = {}
        nearest_map: dict[str, tuple[str, float]] = {}
        for pos in val_pos:
            sid = ctx["all_ids"][pos]
            nb_pos, dist_km = ds.nearest_train_station(ctx, pos, train_pos)
            nb_sid = ctx["all_ids"][nb_pos]
            target_betas[sid] = per_station_betas.get(nb_sid, {})
            nearest_map[sid] = (nb_sid, dist_km)
        logger.info("MOS-nearest station->nearest-train map: %s",
                    {k: (v[0], round(v[1], 2)) for k, v in list(nearest_map.items())[:5]})
        rows_eval = ds.build_mos_rows(ctx, val_pos, eval_pairs, nwp_sources)
        preds = mos_mod.predict_with_per_station(rows_eval, target_betas, nwp_sources)

    elif arm == "mos_local":
        # Transduktive Obergrenze (F5/Spezifikation 3.5): Fit UND Auswertung an
        # derselben Zielstation. Fit-Zeilen kommen aus dem TRAININGSFENSTER
        # (fit_pairs) DIESER Station, nicht aus train_pos.
        rows_train = ds.build_mos_rows(ctx, val_pos, fit_pairs, nwp_sources)
        per_station_betas = mos_mod.fit_per_station(rows_train, nwp_sources)
        rows_eval = ds.build_mos_rows(ctx, val_pos, eval_pairs, nwp_sources)
        preds = mos_mod.predict_with_per_station(rows_eval, per_station_betas, nwp_sources)

    else:
        raise ValueError(arm)

    stem = f"{args.out_prefix}{arm}{suffix_2nwp}{_stem_suffix(args)}_fold{args.fold_idx}"
    meta = rows_eval.rename(columns={"y": "gt"})[
        ["station_id", "run_time", "valid_time", "horizon", "gt", "nwp_ref", "pers_ref"]
    ]
    return assemble_and_save(preds, meta, val_ids, len(val_pos), len(eval_pairs), ctx["F_h"], stem)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate QRF/MOS baselines (fit -> predict -> _save)")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--fold-idx", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--arm", required=True,
                   choices=["qrf_local", "qrf_idw", "mos_regional", "mos_nearest", "mos_local", "all"])
    p.add_argument("--eval-window", choices=["val", "test", "retrain"], default="val",
                   help="'retrain': Retraining-Szenario (F2) — Fit bis 2025-12-01, "
                        "Bewertung 2025-12-01..2026-04-01. Code-Pfad, nicht Teil dieser Phase.")
    p.add_argument("--test-mode", action="store_true")
    p.add_argument("--hpo-study", choices=["auto", "none"], default="none")
    p.add_argument("--nwp-sources", choices=["icond2", "both"], default="both")
    p.add_argument("--n-fit-rows", type=int, default=0)
    p.add_argument("--subsample-seed", type=int, default=ds.SUBSAMPLE_SEED_DEFAULT)
    p.add_argument("--n-jobs", type=int, default=32)
    p.add_argument("--ecmwf-features", default=None)
    p.add_argument("--nwp-geometry", action="store_true")
    p.add_argument("--i2-hist", action="store_true")
    p.add_argument("--per-lead", action="store_true")
    p.add_argument("--idw-n", type=int, default=5, choices=[3, 5, 10],
                   help="QRF-IDW neighbour count (F1 sweep n in {3,5,10})")
    p.add_argument("--out-prefix", default="")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.test_mode and args.eval_window != "val":
        raise SystemExit("--test-mode schliesst --eval-window aus (Spezifikation 6.1)")

    ecmwf_override = args.ecmwf_features.split(",") if args.ecmwf_features else None
    ctx = ds.load_context(args.config, ecmwf_features_override=ecmwf_override, test_mode=args.test_mode)

    N_train, N_val = ctx["N_train"], ctx["N_val"]
    train_pos = np.arange(N_train)
    val_pos = np.arange(N_train, N_train + N_val)

    fit_lo, fit_hi, eval_lo, eval_hi = _window_bounds(ctx, args)
    fit_pairs, fit_counters = ds.build_run_pairs(ctx, fit_lo, fit_hi)
    eval_pairs, eval_counters = ds.build_run_pairs(ctx, eval_lo, eval_hi)

    dropped = {k: fit_counters[k] + eval_counters[k] for k in fit_counters}

    fit_stations = set(ctx["train_ids"]) if args.arm != "mos_local" else set(ctx["val_ids"])
    eval_stations = set(ctx["val_ids"])
    overlap = fit_stations & eval_stations if args.arm != "mos_local" else set()

    k_i2, k_e2 = 4, 4
    n_cols = len(ds.feature_columns(ctx, k_i2, k_e2))
    fh = ctx["F_h"]

    fit_rows = len(fit_stations) * len(fit_pairs) * fh
    eval_rows = len(eval_stations) * len(eval_pairs) * fh

    fhash = fold_hash("configs/spatial_folds.yaml") if Path("configs/spatial_folds.yaml").exists() else "n/a"
    logger.info(
        "=== BASELINE fold=%d arm=%s  fit_stations=%d  eval_stations=%d\n"
        "    fit_pairs=%d  eval_pairs=%d  fit_rows=%d (subsample=%d)  eval_rows=%d\n"
        "    k_i2=%d k_e2=%d i2_mode=%s e2_mode=%s  cols=%d\n"
        "    dropped: r_hist=%d  grid_nan=%d  meas_nan=%d\n"
        "    fold_hash=%s  commit=%s  seed=%d  eval_window=%s ===",
        args.fold_idx, args.arm, len(fit_stations), len(eval_stations),
        len(fit_pairs), len(eval_pairs), fit_rows, args.n_fit_rows, eval_rows,
        k_i2, k_e2, ctx["qcfg"].get("icond2_feature_mode", "dir_in_deg"),
        ctx["qcfg"].get("ecmwf_feature_mode", "dir_in_deg"), n_cols,
        dropped["r_hist"], dropped["grid_nan"], dropped["meas_nan"],
        fhash[:12], _git_commit(), args.subsample_seed, args.eval_window,
    )

    if args.eval_window == "val" and not args.test_mode:
        if len(fit_pairs) != 1473:
            raise SystemExit(f"FATAL: fit_pairs={len(fit_pairs)} != 1473 im Standardmodus")
        if len(eval_pairs) != 1460:
            raise SystemExit(f"FATAL: eval_pairs={len(eval_pairs)} != 1460 im Standardmodus")
    if overlap:
        raise SystemExit(f"FATAL: fit_stations ∩ eval_stations = {sorted(overlap)[:5]}... "
                          f"({len(overlap)} Stationen) — Leckage")

    if args.dry_run:
        if args.arm != "all":
            logger.info("--dry-run: baue Design-Matrizen (kein Fit) …")
            t0 = time.time()
            if args.arm in ("qrf_local", "qrf_idw"):
                idw_n = args.idw_n if args.arm == "qrf_idw" else 0
                idw_pos = train_pos if args.arm == "qrf_idw" else None
                X_fit, y_fit, cols, _ = ds.build_feature_matrix(
                    ctx, train_pos, fit_pairs, k_i2=4, k_e2=4, need_meta=False,
                    nwp_geometry=args.nwp_geometry, i2_hist=args.i2_hist,
                    idw_neighbour_pos=idw_pos, idw_n=idw_n,
                )
                X_eval, y_eval, _, _ = ds.build_feature_matrix(
                    ctx, val_pos, eval_pairs, k_i2=4, k_e2=4, need_meta=False,
                    nwp_geometry=args.nwp_geometry, i2_hist=args.i2_hist,
                    idw_neighbour_pos=idw_pos, idw_n=idw_n,
                )
                logger.info("--dry-run: X_fit=%s X_eval=%s cols=%d (%.1fs) — y_fit mean=%.3f y_eval mean=%.3f",
                            X_fit.shape, X_eval.shape, len(cols), time.time() - t0,
                            float(np.mean(y_fit)) if len(y_fit) else float("nan"),
                            float(np.mean(y_eval)) if len(y_eval) else float("nan"))
            else:
                rows_fit_stations = val_pos if args.arm == "mos_local" else train_pos
                rows_fit = ds.build_mos_rows(ctx, rows_fit_stations, fit_pairs, args.nwp_sources)
                rows_eval = ds.build_mos_rows(ctx, val_pos, eval_pairs, args.nwp_sources)
                logger.info("--dry-run: rows_fit=%d rows_eval=%d cols=%s (%.1fs) — y_fit mean=%.3f y_eval mean=%.3f",
                            len(rows_fit), len(rows_eval), list(rows_fit.columns), time.time() - t0,
                            float(rows_fit["y"].mean()) if len(rows_fit) else float("nan"),
                            float(rows_eval["y"].mean()) if len(rows_eval) else float("nan"))
        logger.info("--dry-run: keine Fits, keine Ausgabedateien.")
        return

    arms = list(MOS_ARMS) + ["qrf_local"] if args.arm == "all" else [args.arm]
    for arm in arms:
        logger.info("── running arm=%s ──", arm)
        if arm in ("qrf_local", "qrf_idw"):
            run_qrf_local(ctx, args, fit_pairs, eval_pairs, train_pos, val_pos, arm_label=arm)
        else:
            run_mos(ctx, args, fit_pairs, eval_pairs, train_pos, val_pos, arm)


if __name__ == "__main__":
    main()
