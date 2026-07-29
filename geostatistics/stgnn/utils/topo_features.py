"""
Topographic node-feature loading for graph edge construction.

Loads per-station terrain metrics (elevation, slope, aspect, TPI, TDI, roughness
length z0, distance to coast) produced by the ``synthetic_re_data_generation``
pipeline, joins them onto a station-ID list in a fixed order, and z-score
normalises each column so they can be combined into edge-difference features
(see ``edge_features()`` in ``spatial.py``).

Expects ``topo_dir`` to contain two files:
  topo_features.csv  — location_id, kind, latitude, longitude, elevation, slope,
                        aspect, tpi5, tdi, elev_std, tpi75, z0, clc_class
  dist_coast.csv      — station_id, dist_coast_km
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical, fixed processing order for topographic edge features — independent
# of the order given in a config's ``edge_features`` list.
TOPO_FEATURE_ORDER = [
    "slope", "aspect_sin", "aspect_cos", "tpi5", "tpi75", "tdi", "elev_std",
    "z0", "dist_coast",
]

# Source column in topo_features.csv for each derived feature name
_TOPO_CSV_COLUMNS = {
    "slope": "slope",
    "tpi5": "tpi5",
    "tpi75": "tpi75",
    "tdi": "tdi",
    "elev_std": "elev_std",
    "z0": "z0",
}


def load_topo_node_features(
    topo_dir: str,
    station_ids: list[str],
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    """
    Load and z-score-normalise topographic node features for a list of stations.

    Parameters
    ----------
    topo_dir :      directory containing topo_features.csv and dist_coast.csv
    station_ids :   station IDs in the same order as station_coords/alts
    feature_names : subset of TOPO_FEATURE_ORDER to load (order-independent)

    Returns
    -------
    dict mapping feature name -> (N,) float32 array, aligned to station_ids
    """
    requested = [f for f in TOPO_FEATURE_ORDER if f in feature_names]
    if not requested:
        return {}

    topo_path = Path(topo_dir) / "topo_features.csv"
    coast_path = Path(topo_dir) / "dist_coast.csv"

    topo_df = pd.read_csv(topo_path, dtype={"location_id": str})
    topo_df = topo_df[topo_df["kind"] == "station"].set_index("location_id")

    coast_df = pd.read_csv(coast_path, dtype={"station_id": str})
    coast_df = coast_df.set_index("station_id")

    topo_df["aspect_sin"] = np.sin(np.deg2rad(topo_df["aspect"]))
    topo_df["aspect_cos"] = np.cos(np.deg2rad(topo_df["aspect"]))

    out: dict[str, np.ndarray] = {}
    for name in requested:
        if name == "dist_coast":
            series = coast_df["dist_coast_km"]
        elif name in ("aspect_sin", "aspect_cos"):
            series = topo_df[name]
        else:
            series = topo_df[_TOPO_CSV_COLUMNS[name]]

        values = series.reindex(station_ids)
        n_missing = int(values.isna().sum())
        if n_missing > 0:
            median = values.median()
            logger.warning(
                "Topo feature '%s': %d/%d stations missing, filling with median (%.4g). "
                "Missing IDs: %s",
                name, n_missing, len(station_ids), median,
                [sid for sid, v in zip(station_ids, values.isna()) if v][:10],
            )
            values = values.fillna(median)

        arr = values.to_numpy(dtype=np.float32)
        mean = float(arr.mean())
        std = max(float(arr.std()), 1e-6)
        out[name] = ((arr - mean) / std).astype(np.float32)

    return out
