#!/usr/bin/env python3
"""Directional variogram analysis for spatial anisotropy of wind speed.

Uses scikit-gstat to compute and visualise directional semivariances for
the azimuths specified in the config (config.interpolation.azimuths).
Each azimuth is evaluated with a tolerance of ±22.5°.

Coordinates are projected from WGS84 to ETRS89/UTM zone 32N (EPSG:25832)
before being passed to scikit-gstat. UTM is a conformal projection that
preserves local angles exactly and distances with <0.1 % error across
Germany — both are critical for reliable directional analysis.

Usage (run from forecasting_framework/):
    python geostatistics/directional_variogram.py --config configs/config_spatial_interpolation.yaml
"""

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from skgstat import models as skg_models
import yaml

# Allow imports from the project root (utils/, etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Human-readable labels for the standard azimuths
_AZ_LABELS = {
    0: "0° (N–S)",
    45: "45° (NE–SW)",
    90: "90° (E–W)",
    135: "135° (SE–NW)",
}


# ---------------------------------------------------------------------------
# Config & data loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config from *path*."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_all_data(config: dict) -> tuple:
    """Load station metadata and the full wind-speed time series.

    Returns the complete (T, N) observation matrix so that directional
    semivariances can be computed across all timestamps, preserving the
    full spatio-temporal variability of the wind field.

    Returns:
        coords:        (N, 2) UTM km coordinates (x=East, y=North, EPSG:25832).
        values_matrix: (T, N) wind speed array; NaN where data are missing.
        station_ids:   List of station ID strings in config order.
    """
    data_path = config["data"]["path"]
    station_ids = [str(s) for s in config["data"]["files"]]

    # Metadata
    meta_path = os.path.join(data_path, "wind_parameter.csv")
    meta = pd.read_csv(meta_path, sep=";", dtype={"park_id": str})
    meta = meta.set_index("park_id")

    lats = meta.loc[station_ids, "latitude"].values.astype(np.float64)
    lons = meta.loc[station_ids, "longitude"].values.astype(np.float64)

    # Time series — load all stations
    all_dfs = []
    for sid in station_ids:
        fpath = os.path.join(data_path, f"synth_{sid}.csv")
        df = pd.read_csv(fpath, sep=";", parse_dates=["timestamp"])
        df["station_id"] = sid
        all_dfs.append(df[["station_id", "timestamp", "wind_speed"]])

    combined = pd.concat(all_dfs, ignore_index=True)

    # Localise timestamps if tz-naive, then apply optional time filter
    if combined["timestamp"].dt.tz is None:
        combined["timestamp"] = combined["timestamp"].dt.tz_localize("UTC")

    test_start = config["data"].get("test_start")
    test_end = config["data"].get("test_end")
    if test_start:
        combined = combined[combined["timestamp"] >= pd.Timestamp(test_start, tz="UTC")]
    if test_end:
        combined = combined[combined["timestamp"] <= pd.Timestamp(test_end, tz="UTC")]

    # Build (T, N) pivot — preserve station order from config
    pivot = combined.pivot_table(
        index="timestamp", columns="station_id", values="wind_speed", aggfunc="first"
    )
    pivot = pivot.reindex(columns=station_ids).sort_index()
    values_matrix = pivot.values.astype(np.float64)  # (T, N)

    # Project WGS84 → ETRS89 / UTM zone 32N (EPSG:25832), standard CRS for Germany
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    x_m, y_m = transformer.transform(lons, lats)
    coords = np.column_stack([x_m / 1000, y_m / 1000])  # m → km

    logger.info(
        "Loaded %d stations × %d timestamps.",
        len(station_ids), values_matrix.shape[0],
    )
    return coords, values_matrix, station_ids


# ---------------------------------------------------------------------------
# Distance distribution summary
# ---------------------------------------------------------------------------

def _pairwise_bearings(coords: np.ndarray) -> np.ndarray:
    """Compute bearings in [0, 180) for all upper-triangle pairs.

    Bearings are reduced to [0, 180) because variograms are symmetric:
    the pair i→j carries the same spatial information as j→i.

    Args:
        coords: (N, 2) projected km coordinates (x=East, y=North).

    Returns:
        (M,) bearing array in degrees, same order as scipy.spatial.distance.pdist.
    """
    n = len(coords)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = coords[j_idx, 0] - coords[i_idx, 0]  # East component
    dy = coords[j_idx, 1] - coords[i_idx, 1]  # North component
    bearing = np.degrees(np.arctan2(dx, dy)) % 180  # fold to [0, 180)
    return bearing


def _directional_mask(bearings: np.ndarray, azimuth: float, tolerance: float) -> np.ndarray:
    """Boolean mask for pairs within *tolerance* degrees of *azimuth* (mod 180)."""
    az = azimuth % 180
    diff = np.abs(bearings - az)
    # Wrap around at 0/180 boundary (e.g. azimuth=0°, bearing=175°)
    diff = np.minimum(diff, 180 - diff)
    return diff <= tolerance


def _print_dist_block(dists: np.ndarray, label: str) -> None:
    """Print min/mean/median/max and band counts for a distance array."""
    bins = [0, 10, 50, 100, 200, 300, np.inf]
    band_labels = ["0–10", "10–50", "50–100", "100–200", "200–300", ">300"]

    if len(dists) == 0:
        print(f"  {label}: no pairs")
        return

    print(f"\n  {label}  (N={len(dists):,} pairs)")
    print(f"    min={dists.min():.1f}  mean={dists.mean():.1f}  "
          f"median={np.median(dists):.1f}  max={dists.max():.1f}  km")
    pcts = np.percentile(dists, [10, 25, 50, 75, 90])
    print(f"    p10={pcts[0]:.1f}  p25={pcts[1]:.1f}  p50={pcts[2]:.1f}  "
          f"p75={pcts[3]:.1f}  p90={pcts[4]:.1f}  km")
    band_parts = []
    for bl, lo, hi in zip(band_labels, bins[:-1], bins[1:]):
        cnt = int(((dists >= lo) & (dists < hi)).sum())
        band_parts.append(f"{bl} km: {cnt:,} ({100*cnt/len(dists):.1f}%)")
    print("    " + "  |  ".join(band_parts))


def print_distance_summary(
    coords: np.ndarray,
    azimuths: list = None,
    tolerance: float = 22.5,
) -> None:
    """Print omnidirectional and (optionally) per-azimuth distance statistics.

    Helps decide on appropriate lag bin spacing for variogram analysis.

    Args:
        coords:    (N, 2) projected km coordinates.
        azimuths:  List of azimuth angles to break down separately.
                   If None, only the omnidirectional summary is printed.
        tolerance: Angular half-width used for directional filtering (degrees).
    """
    dists = pdist(coords)
    bearings = _pairwise_bearings(coords)

    print("\n" + "=" * 70)
    print("  Pairwise distance distribution (km)")
    print("=" * 70)
    _print_dist_block(dists, "All directions (omnidirectional)")

    if azimuths:
        print()
        print(f"  Directional breakdown  (tolerance ±{tolerance}°)")
        for az in azimuths:
            mask = _directional_mask(bearings, az, tolerance)
            label = _AZ_LABELS.get(az, f"{az}°")
            _print_dist_block(dists[mask], label)

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Directional variogram computation
# ---------------------------------------------------------------------------

def compute_directional_variograms(
    coords: np.ndarray,
    values_matrix: np.ndarray,
    azimuths: list,
    tolerance: float = 22.5,
    n_lags: int = 10,
    max_dist: float = None,
) -> dict:
    """Compute empirical directional semivariances and fit a spherical model per azimuth.

    Semivariances are averaged over all timestamps for each station pair,
    then filtered by bearing angle before binning — so the full spatio-
    temporal variability of the wind field informs every lag estimate.

    The spherical model is fitted using ``skgstat.models.spherical`` via
    scipy's curve_fit (trust-region reflective).

    Args:
        coords:         (N, 2) projected km coordinates (x=East, y=North).
        values_matrix:  (T, N) observed wind speeds; NaN allowed.
        azimuths:       List of azimuth angles in degrees (geographic: 0=North).
        tolerance:      Angular half-width in degrees.
        n_lags:         Number of equally-spaced lag bins.
        max_dist:       Maximum lag distance in km; None → half of max pairwise dist.

    Returns:
        dict mapping azimuth (int) → {'lags': ndarray, 'sv': ndarray, 'params': dict}.
        'params' contains 'range', 'sill', 'nugget'; None if fitting failed.
    """
    N = coords.shape[0]
    i_idx, j_idx = np.triu_indices(N, k=1)

    # Pairwise distances and bearings (pdist order = upper triangle row-major)
    dists_1d = pdist(coords)
    bearings = _pairwise_bearings(coords)

    if max_dist is None:
        max_dist = dists_1d.max() / 2.0

    bin_edges = np.linspace(0.0, max_dist, n_lags + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Per-pair semivariance averaged over all timestamps — shape (M,)
    vals_i = values_matrix[:, i_idx]  # (T, M)
    vals_j = values_matrix[:, j_idx]  # (T, M)
    sv_pairs = 0.5 * np.nanmean((vals_i - vals_j) ** 2, axis=0)

    results = {}
    for az in azimuths:
        logger.info("  azimuth=%d°  tolerance=±%.1f°", az, tolerance)

        dir_mask = _directional_mask(bearings, az, tolerance)
        dist_mask = dists_1d <= max_dist
        mask = dir_mask & dist_mask

        if mask.sum() < n_lags:
            logger.warning("  azimuth=%d°: only %d pairs — skipped.", az, mask.sum())
            continue

        fd = dists_1d[mask]
        fsv = sv_pairs[mask]

        # Bin
        bin_idx = np.searchsorted(bin_edges[1:], fd).clip(0, n_lags - 1)
        sv_sums = np.zeros(n_lags)
        sv_counts = np.zeros(n_lags)
        np.add.at(sv_sums, bin_idx, fsv)
        np.add.at(sv_counts, bin_idx, 1)

        valid = sv_counts > 0
        lags = bin_centers[valid]
        sv = sv_sums[valid] / sv_counts[valid]

        # Fit spherical model via skgstat.models.spherical(h, r, c0, b)
        params = None
        try:
            sill_init = float(sv.max())
            range_init = float(lags[len(lags) // 2])
            popt, _ = curve_fit(
                skg_models.spherical,
                lags, sv,
                p0=[range_init, sill_init, 0.0],
                bounds=([0, 0, 0], [max_dist * 2, sill_init * 3, sill_init]),
                method="trf",
                maxfev=10_000,
            )
            params = {"range": float(popt[0]), "sill": float(popt[1]), "nugget": float(popt[2])}
            logger.info(
                "    → range=%.1f km  sill=%.4f  nugget=%.4f  (%d bins, %d pairs)",
                params["range"], params["sill"], params["nugget"],
                valid.sum(), int(mask.sum()),
            )
        except Exception as exc:
            logger.warning("  azimuth=%d°: model fit failed (%s).", az, exc)

        results[az] = {"lags": lags, "sv": sv, "params": params}

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_directional_variograms(variograms: dict, output_path: str, tolerance: float = 22.5) -> None:
    """Plot all directional variograms on a single combined figure and save.

    Each direction is shown as:
      - Scatter points  → empirical semivariance per lag bin.
      - Dashed line     → fitted spherical model.

    Args:
        variograms:  {azimuth: DirectionalVariogram} from
                     ``compute_directional_variograms``.
        output_path: Full filesystem path for the output PNG.
    """
    if not variograms:
        logger.warning("No variograms to plot.")
        return

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(variograms)))

    fig, ax = plt.subplots(figsize=(9, 5))

    for (az, result), color in zip(variograms.items(), colors):
        lags = result["lags"]
        sv = result["sv"]
        params = result["params"]
        label = _AZ_LABELS.get(az, f"{az}°")

        # Empirical points
        ax.scatter(lags, sv, color=color, s=45, zorder=3, alpha=0.9)

        # Fitted model curve (skgstat.models.spherical)
        if params is not None:
            h_fit = np.linspace(0, lags.max() * 1.05, 300)
            sv_fit = skg_models.spherical(h_fit, params["range"], params["sill"], params["nugget"])
            ax.plot(h_fit, sv_fit, color=color, linewidth=1.8, linestyle="--",
                    label=label, zorder=2)
        else:
            ax.plot(lags, sv, color=color, linewidth=1.2, label=label, zorder=2)

    ax.set_xlabel("Lag-Distanz (km)")
    ax.set_ylabel("Semivarianz (m²/s²)")
    ax.set_title(
        f"Direktionale Variogramme — Anisotropie-Analyse (Toleranz ±{tolerance:.1f}°)",
        fontsize=10,
    )
    ax.legend(fontsize=9, title="Richtung")
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved directional variogram plot → %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directional variogram analysis for wind speed spatial anisotropy"
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

    azimuths = interp_cfg.get("azimuths", [0, 45, 90, 135])
    n_lags = int(interp_cfg.get("n_variogram_lags", 10))
    max_dist = interp_cfg.get("max_variogram_dist")
    output_dir = out_cfg.get("path", "results/geostatistics")

    # Tolerance = half the minimum step between azimuths → exact tiling of [0, 180)
    az_sorted = sorted(set(a % 180 for a in azimuths))
    steps = np.diff(az_sorted).tolist()
    if len(az_sorted) > 1:
        steps.append(180 - az_sorted[-1] + az_sorted[0])  # wrap-around gap
    tolerance = min(steps) / 2
    logger.info("Azimuths: %s → auto-computed tolerance: ±%.1f°", azimuths, tolerance)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=== Loading data ===")
    coords, values_matrix, station_ids = load_all_data(config)
    print_distance_summary(coords, azimuths=azimuths, tolerance=tolerance)

    logger.info("=== Computing directional variograms for azimuths %s ===", azimuths)
    variograms = compute_directional_variograms(
        coords,
        values_matrix,
        azimuths,
        tolerance=tolerance,
        n_lags=n_lags,
        max_dist=max_dist,
    )

    output_path = os.path.join(output_dir, "variogram_analysis.png")
    plot_directional_variograms(variograms, output_path, tolerance=tolerance)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
