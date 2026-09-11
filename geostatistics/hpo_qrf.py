#!/usr/bin/env python3
"""
geostatistics/hpo_qrf.py — Optuna-Studie fuer QRF-local (Spezifikation 5).

Objective: Mittel ueber die 3 raeumlichen Folds des GEPOOLTEN, unskalierten
Val-RMSE in m/s (identisch zu hpo_mtgnn.py, siehe Spezifikation 1.8/5.1) — NICHT
das Mittel der Stations-RMSEs (Spezifikation 8.5).

Kein GPU-Zugriff, kein Torch-Import. Nur die eigene Studie
(``cl_m-qrf_out-48_freq-1h_wind_qrf_local``) wird geschrieben; die 13
Modellstudien werden nur GELESEN (nirgends in diesem Skript geschrieben).

Trial-Budget zaehlt PRO WORKER (Befund 10.6/hpo_mtgnn.py:1097-1098) — GENAU
EIN Worker fuer diese Studie starten.

Usage
-----
    cd ~/Work/forecasting_framework
    eval "$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE|DATA_ROOT)=' ~/.bashrc)"
    source frcst/bin/activate
    CUDA_VISIBLE_DEVICES="" nice -n 19 python geostatistics/hpo_qrf.py \\
        --config configs/baselines/config_wind_qrf_local_fold1.yaml --n-jobs 32

    # Schritt 0 (Spezifikation 5.3) — Skalierungskurve, dann beenden:
    CUDA_VISIBLE_DEVICES="" nice -n 19 python geostatistics/hpo_qrf.py \\
        --config configs/baselines/config_wind_qrf_local_fold1.yaml --scaling-curve
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import optuna                                            # noqa: E402
from optuna.samplers import TPESampler                    # noqa: E402

from geostatistics.baselines import dataset as ds          # noqa: E402
from geostatistics.baselines import qrf as qrf_mod          # noqa: E402
from geostatistics.spatial_cv import (                      # noqa: E402
    build_folds, fold_hash, load_spatial_folds, station_pool,
)
from geostatistics.train_stgnn2 import load_yaml            # noqa: E402

logger = logging.getLogger("hpo_qrf")


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    optuna.logging.disable_default_handler()
    optuna.logging.enable_propagation()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "<unknown>"


def _suggest(trial: optuna.Trial, name: str, spec: dict):
    ptype = spec["type"]
    if ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if ptype == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if ptype == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    raise ValueError(f"Unknown HPO param type {ptype!r} for '{name}'")


def sample_hyperparameters(trial: optuna.Trial, hpo_params: dict) -> dict:
    return {name: _suggest(trial, name, spec) for name, spec in hpo_params.items()}


# ---------------------------------------------------------------------------
# Schritt 0 — Skalierungskurve (Spezifikation 5.3)
# ---------------------------------------------------------------------------

def run_scaling_curve(ctx: dict, sf0, fit_pairs: list, eval_pairs: list, args) -> list[tuple[int, float, float]]:
    train_pos = np.array(sf0.train_idx)
    val_pos = np.array(sf0.val_idx)

    logger.info("SCALING-CURVE — building fit design matrix (Fold 0, %d train stations x %d pairs x 48 leads) …",
                len(train_pos), len(fit_pairs))
    t0 = time.time()
    X_fit, y_fit, cols, _ = ds.build_feature_matrix(ctx, train_pos, fit_pairs, k_i2=4, k_e2=4, need_meta=False)
    logger.info("SCALING-CURVE — fit matrix: %s rows x %d cols in %.1fs",
                f"{len(y_fit):,}", X_fit.shape[1], time.time() - t0)

    X_eval, y_eval, _, _ = ds.build_feature_matrix(ctx, val_pos, eval_pairs, k_i2=4, k_e2=4, need_meta=False)
    logger.info("SCALING-CURVE — eval matrix: %s rows x %d cols", f"{len(y_eval):,}", X_eval.shape[1])

    n_total = len(y_fit)
    targets = [500_000, 1_000_000, 2_000_000, 4_000_000, n_total]
    results: list[tuple[int, float, float]] = []
    for n in targets:
        n_int = min(int(n), n_total)
        req = 0 if n_int >= n_total else n_int
        X_sub, y_sub, _ = qrf_mod.subsample_rows(X_fit, y_fit, req, args.subsample_seed)
        t0 = time.time()
        model = qrf_mod.fit(
            X_sub, y_sub,
            n_estimators=200, min_samples_leaf=2e-5, max_features=0.33, max_depth=30,
            random_state=20260810, n_jobs=args.n_jobs,
        )
        fit_s = time.time() - t0
        preds = qrf_mod.predict(model, X_eval)
        rmse = qrf_mod.pooled_rmse(preds, y_eval)
        logger.info("SCALING-CURVE  n=%-10d fit=%7.1fs  pooled_val_rmse=%.4f", len(y_sub), fit_s, rmse)
        results.append((len(y_sub), fit_s, rmse))

    logger.info("SCALING-CURVE SUMMARY (n, fit_seconds, pooled_val_rmse): %s", results)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HPO for QRF-local")
    p.add_argument("--config", required=True)
    p.add_argument("--suffix", default="", help="Log-Suffix, wie hpo_mtgnn.py:306")
    p.add_argument("--n-jobs", type=int, default=32)
    p.add_argument("--n-fit-rows", type=int, default=1_000_000)
    p.add_argument("--subsample-seed", type=int, default=ds.SUBSAMPLE_SEED_DEFAULT)
    p.add_argument("--scaling-curve", action="store_true",
                   help="Schritt 0 aus 5.3: fittet die n-Kurve und beendet")
    p.add_argument("--preprocess-only", action="store_true", help="wie hpo_mtgnn.py:308")
    p.add_argument(
        "--max-trials-this-run", type=int, default=None,
        help="NICHT Teil der Spezifikation §6.3 — Test-/Rauchtest-Knopf: begrenzt "
             "diesen Prozessaufruf auf hoechstens N Trials, unabhaengig vom "
             "Studien-Gesamtbudget (qrf.hpo.trials). Ohne diese Option verhaelt sich "
             "das Skript wie spezifiziert (voller 'remaining'-Lauf).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    config_path = Path(args.config)
    config_stem = config_path.stem.replace("config_", "")
    hpo_stem = re.sub(r"_fold\d+$", "", config_stem)
    suffix = f"_{args.suffix}" if args.suffix else ""

    cfg = load_yaml(args.config)
    qcfg = cfg.get("qrf", {})
    hpo_cfg = qcfg.get("hpo", {})
    if not hpo_cfg:
        print("ERROR: No 'hpo' block found under 'qrf' in config.")
        sys.exit(1)

    F_h = qcfg.get("forecast_horizon", 48)
    freq = cfg["data"].get("freq", "1h")
    study_name = f"cl_m-qrf_out-{F_h}_freq-{freq}_{hpo_stem}"

    _setup_logging(Path("logs") / f"hpo_qrf_{config_stem}{suffix}.log")
    logger.info("=" * 70)
    logger.info("HPO QRF — config: %s  study: %s", args.config, study_name)
    logger.info("=" * 70)

    n_trials = hpo_cfg.get("trials", 60)
    pruner_type = hpo_cfg.get("pruner", "median")
    pruner_n_startup = hpo_cfg.get("pruner_n_startup_trials", 20)
    pruner_n_warmup = hpo_cfg.get("pruner_n_warmup_steps", 1)
    spatial_folds_path = hpo_cfg.get("spatial_folds", "configs/spatial_folds.yaml")
    hpo_params = hpo_cfg.get("params", {})

    if pruner_type == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_n_startup, n_warmup_steps=pruner_n_warmup,
        )

    # ── Union pool laden (cv_mode=spatial, Spezifikation 5.1/1.1) — EIN Laden,
    # danach nur noch build_folds() zum Umindizieren pro Fold. ────────────────
    spatial_fold_defs = load_spatial_folds(spatial_folds_path)
    all_ids_pool = station_pool(spatial_fold_defs)
    _prov_fold_hash = fold_hash(spatial_folds_path)
    logger.info("cv_mode=spatial — %d Folds aus %s, %d Stationen im Pool",
                len(spatial_fold_defs), spatial_folds_path, len(all_ids_pool))

    ctx = ds.load_context(args.config, files_override=all_ids_pool, val_files_override=[])
    folds = build_folds(spatial_fold_defs, ctx["all_ids"])
    for sf in folds:
        logger.info("%s — %d Trainings-/%d Zielstationen", sf.name, len(sf.train_idx), len(sf.val_idx))

    val_start = ctx["val_start"]
    test_start = ctx["test_start"]
    fit_pairs_all, fit_counters = ds.build_run_pairs(ctx, None, val_start)
    eval_pairs_all, eval_counters = ds.build_run_pairs(ctx, val_start, test_start)
    logger.info(
        "Fit pairs: %d (dropped r_hist=%d grid_nan=%d meas_nan=%d)  "
        "Eval pairs: %d (dropped r_hist=%d grid_nan=%d meas_nan=%d)",
        len(fit_pairs_all), fit_counters["r_hist"], fit_counters["grid_nan"], fit_counters["meas_nan"],
        len(eval_pairs_all), eval_counters["r_hist"], eval_counters["grid_nan"], eval_counters["meas_nan"],
    )

    if args.preprocess_only:
        logger.info("--preprocess-only done. Exiting.")
        return

    if args.scaling_curve:
        run_scaling_curve(ctx, folds[0], fit_pairs_all, eval_pairs_all, args)
        return

    _prov_host = socket.gethostname()
    _prov_commit = _git_commit()

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr("host", _prov_host)
        trial.set_user_attr("commit", _prov_commit)
        trial.set_user_attr("fold_hash", _prov_fold_hash)
        trial.set_user_attr("n_fit_rows", args.n_fit_rows)
        trial.set_user_attr("subsample_seed", args.subsample_seed)
        trial.set_user_attr("k_i2", 4)
        trial.set_user_attr("k_e2", 4)
        trial.set_user_attr("icond2_feature_mode", qcfg.get("icond2_feature_mode", "dir_in_deg"))
        trial.set_user_attr("ecmwf_feature_mode", qcfg.get("ecmwf_feature_mode", "dir_in_deg"))

        sampled = sample_hyperparameters(trial, hpo_params)
        logger.info("Trial %d — hyperparameters: %s", trial.number, sampled)

        fold_rmses: list[float] = []
        for fold_idx, sf in enumerate(folds):
            train_pos = np.array(sf.train_idx)
            val_pos = np.array(sf.val_idx)

            X_fit, y_fit, _, _ = ds.build_feature_matrix(
                ctx, train_pos, fit_pairs_all, k_i2=4, k_e2=4, need_meta=False,
            )
            X_sub, y_sub, _ = qrf_mod.subsample_rows(X_fit, y_fit, args.n_fit_rows, args.subsample_seed)

            model = qrf_mod.fit(
                X_sub, y_sub,
                n_estimators=sampled.get("n_estimators", 200),
                min_samples_leaf=sampled.get("min_samples_leaf", 2e-5),
                max_features=sampled.get("max_features", 0.33),
                max_depth=sampled.get("max_depth", 30),
                random_state=qcfg.get("random_state", 20260810),
                n_jobs=args.n_jobs,
            )

            X_eval, y_eval, _, _ = ds.build_feature_matrix(
                ctx, val_pos, eval_pairs_all, k_i2=4, k_e2=4, need_meta=False,
            )
            preds = qrf_mod.predict(model, X_eval)
            # Gepooltes, unskaliertes RMSE ueber ALLE Stationen/Paare/Leads dieses
            # Folds — NICHT das Mittel der Stations-RMSEs (Spezifikation 1.8/8.5).
            rmse = float(np.sqrt(np.mean(
                (preds.astype(np.float64) - y_eval.astype(np.float64)) ** 2
            )))
            fold_rmses.append(rmse)
            logger.info("Trial %d %s — pooled val RMSE=%.4f", trial.number, sf.name, rmse)

            trial.report(float(np.mean(fold_rmses)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_rmse = float(np.mean(fold_rmses))
        logger.info("Trial %d done — fold RMSEs: %s  mean=%.4f",
                    trial.number, [f"{v:.4f}" for v in fold_rmses], mean_rmse)
        return mean_rmse

    storage_url = os.environ.get("OPTUNA_STORAGE")
    if not storage_url:
        raise RuntimeError("OPTUNA_STORAGE not set — non-interactive shell? see README K3")
    storage = optuna.storages.RDBStorage(
        url=storage_url, heartbeat_interval=60,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600},
    )
    logger.info("Optuna storage: PostgreSQL (OPTUNA_STORAGE)")

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="minimize",
        sampler=TPESampler(), pruner=pruner, load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
    remaining = max(n_trials - completed, 0)
    if args.max_trials_this_run is not None:
        remaining = min(remaining, args.max_trials_this_run)
        logger.info("--max-trials-this-run=%d — Rauchtest-Begrenzung aktiv", args.max_trials_this_run)
    logger.info("Study loaded — %d completed, %d remaining (this run) of %d total",
                completed, remaining, n_trials)

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, catch=(Exception,))

    logger.info("=" * 70)
    logger.info("HPO RUN DONE")
    try:
        logger.info("Best trial so far: #%d  best_value=%.6f m/s", study.best_trial.number, study.best_value)
        for k, v in study.best_params.items():
            logger.info("  %-30s %s", k, v)
    except ValueError:
        logger.info("No COMPLETE trial yet.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
