#!/usr/bin/env python3
"""Entry point for the spatial wind-speed interpolation pipeline.

Runs IDW, Ordinary Kriging, and Regression Kriging in a leave-one-out
cross-validation across all stations defined in the config, then writes
results and diagnostic scatter plots to the configured output directory.

Usage (run from forecasting_framework/):
    python geostatistics/run_spatial_interpolation.py --config configs/config_spatial_interpolation.yaml
"""

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Allow imports from the project root (utils/, etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.interpolation import (
    compute_anisotropic_distance_matrix,
    compute_distance_matrix,
    compute_empirical_semivariance,
    compute_metrics,
    fit_global_variogram,
    run_loo_cv,
    wind_to_uv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & data loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config from *path*."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> tuple:
    """Load station CSVs and metadata; return a (T, N) pivot and coordinate arrays.

    Reads:
      - ``{data.path}/wind_parameter.csv``  → lat, lon, altitude per station
      - ``{data.path}/synth_{station_id}.csv`` → per-station wind-speed time series

    Applies optional ``test_start`` / ``test_end`` filters from the config.
    If ``interpolation.interpolate_uv`` is true, also builds u/v component
    matrices from the wind_direction column (meteorological convention).

    Returns:
        pivot:       DataFrame (T×N) of wind_speed; index=timestamp, cols=station_id.
        lats:        (N,) latitude array.
        lons:        (N,) longitude array.
        alts:        (N,) altitude array in metres.
        station_ids: List[str] of station IDs in column order.
        u_matrix:    (T, N) zonal component array, or None if not requested.
        v_matrix:    (T, N) meridional component array, or None if not requested.
    """
    data_path = config["data"]["path"]
    station_ids = [str(s) for s in config["data"]["files"]]
    interp_cfg = config.get("interpolation", {})
    do_uv = interp_cfg.get("interpolate_uv", False)

    # --- metadata ---
    meta_path = os.path.join(data_path, "wind_parameter.csv")
    meta = pd.read_csv(meta_path, sep=";", dtype={"park_id": str})
    meta["park_id"] = meta["park_id"].astype(str)
    meta = meta.set_index("park_id")

    missing_meta = [s for s in station_ids if s not in meta.index]
    if missing_meta:
        raise ValueError(
            f"{len(missing_meta)} station IDs not found in wind_parameter.csv: "
            f"{missing_meta[:5]} ..."
        )

    lats = meta.loc[station_ids, "latitude"].values.astype(np.float64)
    lons = meta.loc[station_ids, "longitude"].values.astype(np.float64)
    alts = meta.loc[station_ids, "altitude"].values.astype(np.float64)

    # --- per-station time series ---
    cols_to_load = ["station_id", "timestamp", "wind_speed"]
    if do_uv:
        cols_to_load.append("wind_direction")
    # Dynamic RK features (all except 'altitude' which comes from metadata)
    rk_feature_names_cfg = interp_cfg.get("rk_features") or []
    dynamic_rk_cols = [f for f in rk_feature_names_cfg if f != "altitude"]
    cols_to_load.extend(dynamic_rk_cols)

    all_dfs = []
    for sid in station_ids:
        fpath = os.path.join(data_path, f"synth_{sid}.csv")
        df = pd.read_csv(fpath, sep=";", parse_dates=["timestamp"])
        df["station_id"] = sid
        available = [c for c in cols_to_load if c in df.columns]
        all_dfs.append(df[available])

    combined = pd.concat(all_dfs, ignore_index=True)

    # Optional time filter — strip tz from timestamps in data if needed
    if combined["timestamp"].dt.tz is None:
        combined["timestamp"] = combined["timestamp"].dt.tz_localize("UTC")

    test_start = config["data"].get("test_start")
    test_end = config["data"].get("test_end")
    if test_start:
        combined = combined[combined["timestamp"] >= pd.Timestamp(test_start, tz="UTC")]
    if test_end:
        combined = combined[combined["timestamp"] <= pd.Timestamp(test_end, tz="UTC")]

    # Build (T, N) pivot for wind_speed — preserve station order from config
    pivot = combined.pivot_table(
        index="timestamp", columns="station_id", values="wind_speed", aggfunc="first"
    )
    pivot = pivot.reindex(columns=station_ids)
    pivot.index = pd.DatetimeIndex(pivot.index)
    pivot = pivot.sort_index()

    u_matrix = None
    v_matrix = None

    if do_uv and "wind_direction" in combined.columns:
        pivot_dir = combined.pivot_table(
            index="timestamp", columns="station_id", values="wind_direction", aggfunc="first"
        )
        pivot_dir = pivot_dir.reindex(columns=station_ids)
        pivot_dir.index = pd.DatetimeIndex(pivot_dir.index)
        pivot_dir = pivot_dir.sort_index()

        u_arr, v_arr = wind_to_uv(pivot.values, pivot_dir.values)
        u_matrix = u_arr
        v_matrix = v_arr
        logger.info("Computed u/v wind component matrices.")
    elif do_uv:
        logger.warning("interpolate_uv=true but 'wind_direction' column not found in CSVs.")

    # --- RK feature matrices ---
    # 'altitude' is static (from metadata); everything else is loaded from CSVs.
    rk_feature_names = interp_cfg.get("rk_features")  # None if not set
    rk_static_features: dict = {}
    rk_dynamic_features: dict = {}

    if rk_feature_names:
        rk_static_features["altitude"] = alts   # always available as static
        dynamic_names = [f for f in rk_feature_names if f != "altitude"]
        for fname in dynamic_names:
            if fname in combined.columns:
                piv = combined.pivot_table(
                    index="timestamp", columns="station_id", values=fname, aggfunc="first"
                )
                piv = piv.reindex(columns=station_ids)
                piv.index = pd.DatetimeIndex(piv.index)
                piv = piv.sort_index()
                rk_dynamic_features[fname] = piv.values.astype(np.float64)
                logger.info("Loaded RK feature '%s' as (T, N) matrix.", fname)
            else:
                logger.warning("RK feature '%s' not found in CSVs — skipping.", fname)

        # Keep only features that were actually loaded
        rk_feature_names = [
            f for f in rk_feature_names
            if f in rk_static_features or f in rk_dynamic_features
        ]
        logger.info("Final RK feature list: %s", rk_feature_names)

    logger.info(
        "Loaded data: %d timestamps × %d stations", len(pivot), len(station_ids)
    )
    return pivot, lats, lons, alts, station_ids, u_matrix, v_matrix, rk_feature_names, rk_static_features, rk_dynamic_features


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_variogram_plot(
    lags_emp: np.ndarray,
    sv_emp: np.ndarray,
    variogram_params: dict,
    output_dir: str,
    prefix: str,
) -> None:
    """Save a plot of the empirical semivariance and the fitted variogram model."""
    from utils.interpolation import _get_variogram_fn

    model_name = variogram_params.get("model", "spherical")
    model_fn = _get_variogram_fn(model_name)

    h_fit = np.linspace(0, lags_emp[-1] * 1.05, 300)
    sv_fit = model_fn(
        h_fit,
        variogram_params["nugget"],
        variogram_params["psill"],
        variogram_params["range"],
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(lags_emp, sv_emp, color="steelblue", zorder=3, label="Empirisch")
    ax.plot(h_fit, sv_fit, color="tomato", linewidth=1.8, label=f"{model_name.capitalize()} (Fit)")

    p = variogram_params
    ax.axhline(p["nugget"], color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(p["nugget"] + p["psill"], color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(p["range"], color="gray", linewidth=0.8, linestyle="--")
    ax.text(
        p["range"] + lags_emp[-1] * 0.01,
        sv_emp.min() * 0.95,
        f"range={p['range']:.1f} km",
        fontsize=8,
        color="gray",
    )
    param_str = f"nugget={p['nugget']:.3f}  psill={p['psill']:.3f}  range={p['range']:.1f} km"
    ax.set_title(f"Globales Variogramm ({model_name}) — {param_str}", fontsize=9)
    ax.set_xlabel("Lag-Distanz (km)")
    ax.set_ylabel("Semivarianz (m²/s²)")
    ax.legend(fontsize=9)
    plt.tight_layout()

    fname = os.path.join(output_dir, f"{prefix}_variogram.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info("Saved variogram plot → %s", fname)


def save_scatter_plot(df: pd.DataFrame, method: str, output_dir: str, prefix: str) -> None:
    """Save a scatter plot of observed vs predicted for *method*.

    For scalar wind-speed methods ('idw', 'ok', 'rk') the observed column is
    'wind_speed_observed'.  For component methods ('u_idw', 'v_ok', …) the
    observed column is inferred from the method prefix.
    """
    pred_col = f"{method}_pred"

    # Determine the matching observed column
    comp = method.split("_")[0]  # 'idw'/'ok'/'rk' → comp=='idw'; 'u_idw' → comp=='u'
    if comp in ("u", "v"):
        obs_col = f"{comp}_observed"
        xlabel = f"Observed {comp} component (m/s)"
        ylabel = f"{method.upper()} predicted (m/s)"
    else:
        obs_col = "wind_speed_observed"
        xlabel = "Observed wind speed (m/s)"
        ylabel = f"{method.upper()} predicted (m/s)"

    plot_df = df.dropna(subset=[obs_col, pred_col])
    obs = plot_df[obs_col]
    pred = plot_df[pred_col]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs, pred, alpha=0.15, s=4, color="steelblue", rasterized=True)
    lim = [min(obs.min(), pred.min()) - 0.5, max(obs.max(), pred.max()) + 0.5]
    ax.plot(lim, lim, "r--", linewidth=1.0, label="1:1 line")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{method.upper()} — Observed vs. Predicted (LOO-CV)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    fname = os.path.join(output_dir, f"{prefix}_scatter_{method}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info("Saved scatter plot → %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spatial wind-speed interpolation: IDW / OK / RK with LOO-CV"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    interp_cfg = config.get("interpolation", {})
    out_cfg = config.get("output", {})

    k = int(interp_cfg.get("k_neighbors", 8))
    idw_power = float(interp_cfg.get("idw_power", 2.0))
    n_lags = int(interp_cfg.get("n_variogram_lags", 20))
    max_dist = interp_cfg.get("max_variogram_dist")      # None → auto
    variogram_model = interp_cfg.get("variogram_model", "spherical")
    variogram_segments = int(interp_cfg.get("variogram_segments", 1))
    aniso_cfg = interp_cfg.get("anisotropy", {})
    output_dir = out_cfg.get("path", "data/geostatistics")
    prefix = out_cfg.get("prefix", "spatial_interp")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    logger.info("=== Loading data ===")
    pivot, lats, lons, alts, station_ids, u_matrix, v_matrix, \
        rk_feature_names, rk_static_features, rk_dynamic_features = load_data(config)
    values_matrix = pivot.values  # (T, N)
    timestamps = pivot.index
    T = len(timestamps)

    # 2. Distance matrix (geodesic) — used for k-NN selection and IDW
    logger.info("=== Computing %d×%d geodesic distance matrix ===", len(station_ids), len(station_ids))
    dist_matrix = compute_distance_matrix(lats, lons)

    # 3. Anisotropic distance matrix (optional) — used for kriging system only
    aniso_dist_matrix = None
    if aniso_cfg.get("enabled", False):
        angle = float(aniso_cfg["angle"])
        ratio = float(aniso_cfg["ratio"])
        logger.info(
            "=== Computing anisotropic distance matrix (angle=%.1f°, ratio=%.3f) ===",
            angle, ratio,
        )
        aniso_dist_matrix = compute_anisotropic_distance_matrix(lats, lons, angle, ratio)

    # 4. Variogram fitting — one per segment
    #    variogram_segments=1  → one global variogram (current behaviour)
    #    variogram_segments=0  → one per timestamp (T segments)
    #    variogram_segments=N  → N equal time-blocks
    if variogram_segments == 0:
        n_segs = T
        seg_slices = [slice(t, t + 1) for t in range(T)]
        segment_indices = np.arange(T, dtype=int)
        logger.info("=== Per-timestamp mode: fitting %d variograms ===", T)
    elif variogram_segments == 1:
        n_segs = 1
        seg_slices = [slice(0, T)]
        segment_indices = np.zeros(T, dtype=int)
    else:
        n_segs = variogram_segments
        boundaries = np.array_split(np.arange(T), n_segs)
        seg_slices = [slice(int(b[0]), int(b[-1]) + 1) for b in boundaries]
        segment_indices = np.zeros(T, dtype=int)
        for i, b in enumerate(boundaries):
            segment_indices[b] = i
        logger.info("=== Fitting %d segment variograms ===", n_segs)

    variogram_params_list = []
    for seg_i, sl in enumerate(seg_slices):
        seg_vals = values_matrix[sl]
        lags_emp, sv_emp = compute_empirical_semivariance(
            seg_vals, dist_matrix, n_lags=n_lags, max_dist=max_dist
        )
        vp = fit_global_variogram(lags_emp, sv_emp, model=variogram_model)
        variogram_params_list.append(vp)
        if n_segs <= 12 or seg_i == 0 or seg_i == n_segs - 1:
            logger.info("  Segment %d/%d: %s", seg_i + 1, n_segs, vp)

    # Save variogram plot for the first (or only) segment
    lags_emp0, sv_emp0 = compute_empirical_semivariance(
        values_matrix[seg_slices[0]], dist_matrix, n_lags=n_lags, max_dist=max_dist
    )
    save_variogram_plot(lags_emp0, sv_emp0, variogram_params_list[0], output_dir, prefix)

    # 5. LOO-CV
    logger.info("=== Running LOO-CV (%d stations × %d timestamps) ===", len(station_ids), T)
    if u_matrix is not None:
        logger.info("u/v component interpolation enabled.")
    predictions = run_loo_cv(
        values_matrix=values_matrix,
        timestamps=timestamps,
        lats=lats,
        lons=lons,
        alts=alts,
        station_ids=station_ids,
        dist_matrix=dist_matrix,
        variogram_params_list=variogram_params_list,
        segment_indices=segment_indices,
        k=k,
        idw_power=idw_power,
        u_matrix=u_matrix,
        v_matrix=v_matrix,
        rk_feature_names=rk_feature_names or None,
        rk_static_features=rk_static_features or None,
        rk_dynamic_features=rk_dynamic_features or None,
        aniso_dist_matrix=aniso_dist_matrix,
    )

    # 5. Save raw predictions
    pred_path = os.path.join(output_dir, f"{prefix}_loo_predictions.csv")
    predictions.to_csv(pred_path, index=False)
    logger.info("Saved LOO predictions → %s", pred_path)

    # 6. Metrics
    logger.info("=== Computing metrics ===")
    per_station_df, summary_df = compute_metrics(predictions)

    ps_path = os.path.join(output_dir, f"{prefix}_results_per_station.csv")
    per_station_df.to_csv(ps_path, index=False)
    logger.info("Saved per-station metrics → %s", ps_path)

    sm_path = os.path.join(output_dir, f"{prefix}_results_summary.csv")
    summary_df.to_csv(sm_path, index=False)
    logger.info("Saved summary metrics → %s", sm_path)

    # 7. Scatter plots for scalar wind speed
    logger.info("=== Saving scatter plots ===")
    for method in ["idw", "ok", "rk"]:
        save_scatter_plot(predictions, method, output_dir, prefix)

    # Scatter plots for u/v and derived vector speed
    if u_matrix is not None:
        for comp in ("u", "v"):
            for method in ("idw", "ok", "rk"):
                col = f"{comp}_{method}_pred"
                if col in predictions.columns:
                    save_scatter_plot(predictions, f"{comp}_{method}", output_dir, prefix)

    logger.info("=== Done ===")
    print("\nGlobal summary (mean over stations):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
