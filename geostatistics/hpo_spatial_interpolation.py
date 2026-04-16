#!/usr/bin/env python3
"""
hpo_spatial_interpolation.py — Hyperparameter optimisation for the spatial
wind-speed interpolation pipeline (IDW / OK / RK).

Strategy
--------
- Data (station measurements, NWP, ECMWF) is loaded **once** outside the
  trial loop — loading is the expensive I/O step.
- Each trial re-fits the variogram and runs the LOO-CV on a random subsample
  of timestamps (``hpo.subsample_hours``) for speed.
- Objective: minimise mean per-station RMSE of the **RK** method.
- Study is stored in the database given by the ``OPTUNA_STORAGE`` environment
  variable (PostgreSQL), or falls back to a local SQLite file.

Usage (run from forecasting_framework/)
----------------------------------------
    python geostatistics/hpo_spatial_interpolation.py \\
        --config configs/config_wind_interpol.yaml [--suffix v1]

Config
------
Add an ``hpo`` block to your YAML (see config_wind_interpol.yaml for an
example). Tuneable parameters under ``hpo.params``:

    k_neighbors        int   — number of nearest neighbours
    idw_power          float — IDW exponent
    variogram_model    cat   — spherical | gaussian | exponential
    n_variogram_lags   int   — number of empirical semivariance lag bins
    variogram_detrend  cat   — true | false
    anisotropy_enabled cat   — true | false
    anisotropy_angle   float — major-axis direction (0–180°)
    anisotropy_ratio   float — range_minor / range_major (0.1–1.0)
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

from geostatistics.run_spatial_interpolation import load_config, load_data
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
    u_matrix, v_matrix,
    rk_feature_names,
    rk_static_features,
    rk_dynamic_features,
    dist_matrix,
    hpo_cfg: dict,
    base_interp_cfg: dict,
):
    """Return an Optuna objective closure over the pre-loaded data."""

    subsample = hpo_cfg.get("subsample_hours")
    seed      = int(hpo_cfg.get("seed", 42))
    hpo_params = hpo_cfg["params"]
    rng = np.random.default_rng(seed)

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hyperparameters(trial, hpo_params)

        k             = int(hp.get("k_neighbors",      base_interp_cfg.get("k_neighbors", 10)))
        idw_power     = float(hp.get("idw_power",      base_interp_cfg.get("idw_power", 2.0)))
        variogram_model = hp.get("variogram_model",    base_interp_cfg.get("variogram_model", "spherical"))
        n_lags        = int(hp.get("n_variogram_lags", base_interp_cfg.get("n_variogram_lags", 20)))
        detrend       = bool(hp.get("variogram_detrend", base_interp_cfg.get("variogram_detrend", False)))
        max_dist      = base_interp_cfg.get("max_variogram_dist")  # not tuned — keep fixed

        aniso_cfg     = base_interp_cfg.get("anisotropy", {})
        aniso_enabled = bool(hp.get("anisotropy_enabled", aniso_cfg.get("enabled", False)))
        aniso_angle   = float(hp.get("anisotropy_angle", aniso_cfg.get("angle", 0.0)))
        aniso_ratio   = float(hp.get("anisotropy_ratio", aniso_cfg.get("ratio", 1.0)))

        # Prune degenerate anisotropy: angle/ratio only matter when enabled
        if not aniso_enabled:
            trial.set_user_attr("anisotropy_angle", None)
            trial.set_user_attr("anisotropy_ratio", None)

        base_rk_features = base_interp_cfg.get("rk_features") or []
        use_temp2m       = bool(hp.get("use_temperature_2m", "temperature_2m" in base_rk_features))
        use_direction    = bool(hp.get("use_direction", "wind_direction" in base_rk_features))
        do_uv            = bool(hp.get("interpolate_uv", base_interp_cfg.get("interpolate_uv", False)))
        nwp_comps_mode   = str(hp.get("nwp_components",   base_interp_cfg.get("nwp_components", "absolute")))
        ecmwf_comps_mode = str(hp.get("ecmwf_components", base_interp_cfg.get("ecmwf_components", "absolute")))

        # --- subsample timestamps ---
        T = len(pivot)
        if subsample and subsample < T:
            t_idx = np.sort(rng.choice(T, size=int(subsample), replace=False))
            values_sub     = pivot.values[t_idx]
            timestamps_sub = pivot.index[t_idx]
            u_sub          = u_matrix[t_idx] if (do_uv and u_matrix is not None) else None
            v_sub          = v_matrix[t_idx] if (do_uv and v_matrix is not None) else None
            rk_dyn_sub     = {k_: v_[t_idx] for k_, v_ in rk_dynamic_features.items()} \
                             if rk_dynamic_features else {}
        else:
            values_sub     = pivot.values
            timestamps_sub = pivot.index
            u_sub          = u_matrix if (do_uv and u_matrix is not None) else None
            v_sub          = v_matrix if (do_uv and v_matrix is not None) else None
            rk_dyn_sub     = dict(rk_dynamic_features) if rk_dynamic_features else {}

        # Drop optional features not selected for this trial
        if not use_temp2m:
            rk_dyn_sub = {k_: v_ for k_, v_ in rk_dyn_sub.items() if k_ != "temperature_2m"}
        if not use_direction:
            rk_dyn_sub = {k_: v_ for k_, v_ in rk_dyn_sub.items() if k_ != "wind_direction"}
        if nwp_comps_mode == "absolute":
            rk_dyn_sub = {k_: v_ for k_, v_ in rk_dyn_sub.items()
                          if k_ not in ("nwp_u_wind", "nwp_v_wind")}
        elif nwp_comps_mode == "components":
            rk_dyn_sub = {k_: v_ for k_, v_ in rk_dyn_sub.items()
                          if k_ != "nwp_wind_speed"}
        if ecmwf_comps_mode == "absolute":
            rk_dyn_sub = {k_: v_ for k_, v_ in rk_dyn_sub.items()
                          if k_ not in ("ecmwf_u_wind", "ecmwf_v_wind")}
        elif ecmwf_comps_mode == "components":
            rk_dyn_sub = {k_: v_ for k_, v_ in rk_dyn_sub.items()
                          if k_ != "ecmwf_wind_speed"}

        # Build the active feature name list (mirrors load_data logic)
        trial_rk_names = None
        if rk_feature_names:
            trial_rk_names = [
                f for f in rk_feature_names
                if f in (rk_static_features or {}) or f in rk_dyn_sub
            ] or None

        # --- anisotropic distance matrix (if enabled) ---
        if aniso_enabled:
            aniso_dist = compute_anisotropic_distance_matrix(lats, lons, aniso_angle, aniso_ratio)
        else:
            aniso_dist = None

        # --- variogram fitting (one global segment) ---
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

        # --- LOO-CV (RK only — IDW and OK are skipped for speed) ---
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
            u_matrix=u_sub,
            v_matrix=v_sub,
            rk_feature_names=trial_rk_names,
            rk_static_features=rk_static_features or None,
            rk_dynamic_features=rk_dyn_sub or None,
            aniso_dist_matrix=aniso_dist,
            rk_only=True,
        )

        if predictions.empty or "rk_pred" not in predictions.columns:
            raise optuna.exceptions.TrialPruned()

        # --- objective: mean per-station RK RMSE ---
        per_station_df, summary_df = compute_metrics(predictions)
        rk_row = summary_df[summary_df["method"] == "rk"]
        if rk_row.empty:
            raise optuna.exceptions.TrialPruned()

        rmse = float(rk_row["rmse"].iloc[0])
        logger.info(
            "Trial %d finished — RK RMSE=%.4f  k=%d  model=%s  detrend=%s  aniso=%s"
            "  temp2m=%s  dir=%s  uv=%s  nwp_comps=%s  ecmwf_comps=%s",
            trial.number, rmse, k, variogram_model, detrend, aniso_enabled,
            use_temp2m, use_direction, do_uv, nwp_comps_mode, ecmwf_comps_mode,
        )
        return rmse

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HPO for spatial wind-speed interpolation"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("-s", "--suffix", default="",
                        help="Suffix appended to study name, e.g. '_v2'")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    hpo_cfg = config.get("hpo")
    if not hpo_cfg:
        raise ValueError("No 'hpo' section found in config.")

    study_name = hpo_cfg.get("study_name", "wind_interpol")
    if args.suffix:
        study_name = f"{study_name}_{args.suffix}"
    n_trials = int(hpo_cfg.get("trials", 50))

    # --- log file ---
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"hpo_{study_name}.log"
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("=== HPO study: %s  (%d trials) ===", study_name, n_trials)
    logger.info("Log → %s", log_path)

    # --- storage ---
    storage_url = os.environ.get("OPTUNA_STORAGE")
    if storage_url:
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"pool_pre_ping": True},
        )
        logger.info("Optuna storage: PostgreSQL (OPTUNA_STORAGE)")
    else:
        studies_dir = Path(__file__).parent.parent / "studies"
        studies_dir.mkdir(exist_ok=True)
        sqlite_path = studies_dir / f"{study_name}.db"
        storage = f"sqlite:///{sqlite_path}"
        logger.warning("OPTUNA_STORAGE not set — using SQLite: %s", sqlite_path)

    # --- ensure HPO data requirements are loaded ---
    # (If we tune components, force 'both' so load_data fetches everything)
    interp_cfg = config.setdefault("interpolation", {})
    hpo_params = hpo_cfg.get("params", {})
    if "interpolate_uv" in hpo_params:
        interp_cfg["interpolate_uv"] = True
    if "nwp_components" in hpo_params:
        interp_cfg["nwp_components"] = "both"
    if "ecmwf_components" in hpo_params:
        interp_cfg["ecmwf_components"] = "both"
    
    rk_features = interp_cfg.get("rk_features") or []
    if "use_temperature_2m" in hpo_params and "temperature_2m" not in rk_features:
        rk_features.append("temperature_2m")
    if "use_direction" in hpo_params and "wind_direction" not in rk_features:
        rk_features.append("wind_direction")
    interp_cfg["rk_features"] = rk_features

    # --- load data ONCE ---
    logger.info("=== Loading data (once for all trials) ===")
    (pivot, lats, lons, alts, station_ids,
     u_matrix, v_matrix,
     rk_feature_names, rk_static_features, rk_dynamic_features) = load_data(config)

    logger.info("Data loaded: %d timestamps × %d stations", len(pivot), len(station_ids))

    logger.info("Computing geodesic distance matrix (%d×%d) …", len(station_ids), len(station_ids))
    dist_matrix = compute_distance_matrix(lats, lons)

    # --- create / load study ---
    sampler = TPESampler(seed=int(hpo_cfg.get("seed", 42)))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    already_done = len(study.trials)
    remaining    = max(0, n_trials - already_done)
    logger.info(
        "Study '%s': %d trials already done, running %d more.",
        study_name, already_done, remaining,
    )
    if remaining == 0:
        logger.info("Nothing to do — increase hpo.trials to run more.")

    objective = make_objective(
        pivot=pivot,
        lats=lats, lons=lons, alts=alts,
        station_ids=station_ids,
        u_matrix=u_matrix,
        v_matrix=v_matrix,
        rk_feature_names=rk_feature_names,
        rk_static_features=rk_static_features,
        rk_dynamic_features=rk_dynamic_features,
        dist_matrix=dist_matrix,
        hpo_cfg=hpo_cfg,
        base_interp_cfg=config.get("interpolation", {}),
    )

    study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    # --- results ---
    best = study.best_trial
    logger.info("=== Best trial #%d  RK RMSE=%.4f ===", best.number, best.value)
    for k, v in best.params.items():
        logger.info("  %-30s %s", k, v)

    print("\n=== Best hyperparameters ===")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"\nRK RMSE: {best.value:.4f}")


if __name__ == "__main__":
    main()