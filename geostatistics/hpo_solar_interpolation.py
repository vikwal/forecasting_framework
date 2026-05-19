#!/usr/bin/env python3
"""
hpo_solar_interpolation.py — Hyperparameter optimisation for the spatial
solar GHI interpolation pipeline (IDW / OK / RK).

Analogous to hpo_spatial_interpolation.py but for solar.

Strategy
--------
- Data is loaded once outside the trial loop.
- Each trial re-fits the variogram on a subsampled set of timestamps.
- Objective: minimise mean per-station RMSE of the RK method.

Usage (run from forecasting_framework/)
----------------------------------------
    python geostatistics/hpo_solar_interpolation.py \\
        --config configs/config_solar_interpol.yaml [--suffix v1]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from optuna.samplers import TPESampler

from geostatistics.run_solar_interpolation import load_config, load_data
from utils.interpolation import (
    compute_anisotropic_distance_matrix,
    compute_distance_matrix,
    compute_empirical_semivariance,
    compute_metrics,
    fit_global_variogram,
    run_loo_cv,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# HP sampling
# ---------------------------------------------------------------------------

def _suggest(trial: optuna.Trial, name: str, spec: dict):
    ptype = spec["type"]
    if ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if ptype == "int":
        return trial.suggest_int(name, spec["low"], spec["high"],
                                 step=spec.get("step", 1))
    if ptype == "float":
        return trial.suggest_float(name, spec["low"], spec["high"],
                                   log=spec.get("log", False))
    raise ValueError(f"Unknown HPO param type {ptype!r} for '{name}'")


def sample_hyperparameters(trial: optuna.Trial, hpo_params: dict) -> dict:
    return {name: _suggest(trial, name, spec) for name, spec in hpo_params.items()}


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(
    pivot,
    lats, lons, alts,
    station_ids,
    rk_feature_names,
    rk_static_features,
    rk_dynamic_features,
    dist_matrix,
    hpo_cfg: dict,
    base_interp_cfg: dict,
):
    subsample  = hpo_cfg.get("subsample_hours")
    seed       = int(hpo_cfg.get("seed", 42))
    hpo_params = hpo_cfg["params"]
    rng        = np.random.default_rng(seed)

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hyperparameters(trial, hpo_params)

        k               = int(hp.get("k_neighbors",      base_interp_cfg.get("k_neighbors", 30)))
        idw_power       = float(hp.get("idw_power",      base_interp_cfg.get("idw_power", 2.0)))
        variogram_model = hp.get("variogram_model",      base_interp_cfg.get("variogram_model", "gaussian"))
        n_lags          = int(hp.get("n_variogram_lags", base_interp_cfg.get("n_variogram_lags", 30)))
        detrend         = bool(hp.get("variogram_detrend", base_interp_cfg.get("variogram_detrend", True)))
        max_dist        = base_interp_cfg.get("max_variogram_dist")

        aniso_cfg     = base_interp_cfg.get("anisotropy", {})
        aniso_enabled = bool(hp.get("anisotropy_enabled", aniso_cfg.get("enabled", False)))
        aniso_angle   = float(hp.get("anisotropy_angle",  aniso_cfg.get("angle", 0.0)))
        aniso_ratio   = float(hp.get("anisotropy_ratio",  aniso_cfg.get("ratio", 1.0)))

        if not aniso_enabled:
            trial.set_user_attr("anisotropy_angle", None)
            trial.set_user_attr("anisotropy_ratio", None)

        # --- optional NWP feature toggles ---
        base_rk_features = base_interp_cfg.get("rk_features") or []
        rk_dyn_sub = dict(rk_dynamic_features) if rk_dynamic_features else {}
        for feat in list(rk_dyn_sub.keys()):
            toggle_key = f"use_{feat}"
            if toggle_key in hp:
                if not bool(hp[toggle_key]):
                    rk_dyn_sub.pop(feat, None)

        trial_rk_names = None
        if rk_feature_names:
            trial_rk_names = [
                f for f in rk_feature_names
                if f in (rk_static_features or {}) or f in rk_dyn_sub
            ] or None

        # --- subsample ---
        T = len(pivot)
        if subsample and subsample < T:
            t_idx          = np.sort(rng.choice(T, size=int(subsample), replace=False))
            values_sub     = pivot.values[t_idx]
            timestamps_sub = pivot.index[t_idx]
            rk_dyn_sub     = {k_: v_[t_idx] for k_, v_ in rk_dyn_sub.items()}
        else:
            values_sub     = pivot.values
            timestamps_sub = pivot.index

        # --- anisotropic dist ---
        if aniso_enabled:
            aniso_dist = compute_anisotropic_distance_matrix(lats, lons, aniso_angle, aniso_ratio)
        else:
            aniso_dist = None

        # --- variogram ---
        lags_emp, sv_emp = compute_empirical_semivariance(
            values_sub, dist_matrix, n_lags=n_lags, max_dist=max_dist, detrend=detrend,
        )
        try:
            vp = fit_global_variogram(lags_emp, sv_emp, model=variogram_model)
        except ValueError as exc:
            logger.debug("Trial %d pruned — variogram fit failed: %s", trial.number, exc)
            raise optuna.exceptions.TrialPruned()
        variogram_params_list = [vp]
        segment_indices = np.zeros(len(timestamps_sub), dtype=int)

        # --- LOO-CV (RK only) ---
        predictions = run_loo_cv(
            values_matrix=values_sub,
            timestamps=timestamps_sub,
            lats=lats,
            lons=lons,
            alts=alts,
            station_ids=station_ids,
            dist_matrix=dist_matrix,
            variogram_params_list=variogram_params_list,
            segment_indices=segment_indices,
            k=k,
            idw_power=idw_power,
            rk_feature_names=trial_rk_names,
            rk_static_features=rk_static_features or None,
            rk_dynamic_features=rk_dyn_sub or None,
            aniso_dist_matrix=aniso_dist,
            rk_only=True,
        )

        if predictions.empty or "rk_pred" not in predictions.columns:
            raise optuna.exceptions.TrialPruned()

        per_station_df, summary_df = compute_metrics(predictions)
        rk_row = summary_df[summary_df["method"] == "rk"]
        if rk_row.empty:
            raise optuna.exceptions.TrialPruned()

        rmse = float(rk_row["rmse"].iloc[0])
        logger.info(
            "Trial %d  RK RMSE=%.4f  k=%d  model=%s  detrend=%s  aniso=%s",
            trial.number, rmse, k, variogram_model, detrend, aniso_enabled,
        )
        return rmse

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HPO for spatial solar GHI interpolation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("-s", "--suffix", default="")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config  = load_config(args.config)
    hpo_cfg = config.get("hpo")
    if not hpo_cfg:
        raise ValueError("No 'hpo' section found in config.")

    study_name = hpo_cfg.get("study_name", "solar_interpol")
    if args.suffix:
        study_name = f"{study_name}_{args.suffix}"
    n_trials = int(hpo_cfg.get("trials", 100))

    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"hpo_{study_name}.log"
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("=== HPO study: %s  (%d trials) ===", study_name, n_trials)

    # --- storage ---
    storage_url = os.environ.get("OPTUNA_STORAGE")
    if storage_url:
        storage = optuna.storages.RDBStorage(url=storage_url,
                                             engine_kwargs={"pool_pre_ping": True})
        logger.info("Optuna storage: PostgreSQL")
    else:
        studies_dir = Path(__file__).parent.parent / "studies"
        studies_dir.mkdir(exist_ok=True)
        sqlite_path = studies_dir / f"{study_name}.db"
        storage = f"sqlite:///{sqlite_path}"
        logger.warning("OPTUNA_STORAGE not set — SQLite: %s", sqlite_path)

    # Ensure load_data fetches all possible NWP features that might be toggled
    interp_cfg = config.setdefault("interpolation", {})
    hpo_params = hpo_cfg.get("params", {})
    rk_features = list(interp_cfg.get("rk_features") or [])
    for feat in ("ghi_nwp", "dhi_nwp", "alb_rad", "clct", "t_2m"):
        if f"use_{feat}" in hpo_params and feat not in rk_features:
            rk_features.append(feat)
    interp_cfg["rk_features"] = rk_features

    logger.info("=== Loading data (once for all trials) ===")
    (pivot, _raw_pivot, lats, lons, alts, station_ids,
     rk_feature_names, rk_static_features, rk_dynamic_features) = load_data(config)
    logger.info("Data loaded: %d timestamps × %d stations", len(pivot), len(station_ids))

    dist_matrix = compute_distance_matrix(lats, lons)
    logger.info("Distance matrix computed.")

    sampler = TPESampler()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    already_done = len(study.trials)
    remaining    = max(0, n_trials - already_done)
    logger.info("Study '%s': %d done, %d remaining.", study_name, already_done, remaining)

    objective = make_objective(
        pivot=pivot,
        lats=lats, lons=lons, alts=alts,
        station_ids=station_ids,
        rk_feature_names=rk_feature_names,
        rk_static_features=rk_static_features,
        rk_dynamic_features=rk_dynamic_features,
        dist_matrix=dist_matrix,
        hpo_cfg=hpo_cfg,
        base_interp_cfg=config.get("interpolation", {}),
    )

    study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    best = study.best_trial
    logger.info("=== Best trial #%d  RK RMSE=%.4f ===", best.number, best.value)
    for k_, v_ in best.params.items():
        logger.info("  %-30s %s", k_, v_)

    print("\n=== Best hyperparameters ===")
    for k_, v_ in best.params.items():
        print(f"  {k_}: {v_}")
    print(f"\nRK RMSE: {best.value:.4f}")


if __name__ == "__main__":
    main()
