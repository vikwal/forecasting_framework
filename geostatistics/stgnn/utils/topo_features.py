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


# Variance-stabilising transform per feature, applied before the z-score in
# load_topo_station_features(). The raw columns are heavily right-skewed — tdi
# has median 0.90 but max 361 (sigma 29.7), so a plain z-score leaves 202 of 203
# stations inside +-0.15 sigma and one at +12 sigma. As an absolute node feature
# that constant offset goes straight into every GRU gate / input channel, so the
# skew matters much more here than in a pairwise difference.
#   z0        : enters the log wind profile logarithmically
#   tpi5/75   : signed, roughly symmetric around 0 -> signed log
#   others    : non-negative and long-tailed -> log1p
#   aspect_*  : already bounded in [-1, 1] -> untouched
_TOPO_TRANSFORMS = {
    "z0":         lambda a: np.log(np.maximum(a, 1e-6)),
    "tdi":        lambda a: np.log1p(np.maximum(a, 0.0)),
    "slope":      lambda a: np.log1p(np.maximum(a, 0.0)),
    "elev_std":   lambda a: np.log1p(np.maximum(a, 0.0)),
    "dist_coast": lambda a: np.log1p(np.maximum(a, 0.0)),
    "tpi5":       lambda a: np.sign(a) * np.log1p(np.abs(a)),
    "tpi75":      lambda a: np.sign(a) * np.log1p(np.abs(a)),
}


def _resolve_train_idx(
    n_stations: int,
    n_train: int | None,
    train_idx: list[int] | np.ndarray | None,
) -> np.ndarray:
    """Trainingsstationen als explizite Indizes.

    ``n_train`` (die fuehrenden N Stationen) gilt nur, solange die Stationsliste
    train-zuerst sortiert ist. Bei raeumlicher CV rotieren die Rollen innerhalb
    einer einmal geladenen Stationsmenge, dann muss ``train_idx`` uebergeben
    werden. Genau eines von beiden ist erforderlich.
    """
    if (n_train is None) == (train_idx is None):
        raise ValueError("Genau eines von n_train / train_idx angeben")
    if train_idx is not None:
        idx = np.asarray(train_idx, dtype=int)
        if idx.size == 0:
            raise ValueError("train_idx ist leer")
        if idx.min() < 0 or idx.max() >= n_stations:
            raise IndexError(f"train_idx ausserhalb von 0..{n_stations - 1}")
        return idx
    return np.arange(int(n_train))


def load_topo_station_features(
    topo_dir: str,
    station_ids: list[str],
    feature_names: list[str],
    n_train: int | None = None,
    train_idx: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Load absolute per-station topographic features for use as **node** features.

    Differs from ``load_topo_node_features`` (which feeds edge differences) in
    three ways that only matter for absolute values:

    1. Variance-stabilising transforms (see ``_TOPO_TRANSFORMS``) before scaling.
    2. The z-score is fitted on the **train stations only**, matching every
       other scaler in the pipeline (meas/i2/e2/static) instead of leaking
       val/test station statistics into the normalisation.
    3. ``tdi`` is set to 0 where the terrain is perfectly flat (elev_std == 0)
       rather than median-filled. The ratio is 0/0 on zero relief, so 0 is the
       physically correct value — median-filling hands an offshore location a
       typical inland terrain value. Mirrors utils/preprocessing.py, which the
       TFT baseline already does this way.

    Parameters
    ----------
    topo_dir :      directory containing topo_features.csv and dist_coast.csv
    station_ids :   station IDs, same order as station_coords/alts
    feature_names : subset of TOPO_FEATURE_ORDER (order-independent)
    n_train :       number of leading train stations to fit the z-score on —
                    only valid if ``station_ids`` is sorted train-first
    train_idx :     explicit train-station indices, for spatial CV where the
                    roles rotate within one loaded station set. Exactly one of
                    ``n_train`` / ``train_idx`` must be given.

    Returns
    -------
    (N, F) float32 array and the resolved column names in canonical order
    """
    requested = [f for f in TOPO_FEATURE_ORDER if f in feature_names]
    if not requested:
        return np.zeros((len(station_ids), 0), dtype=np.float32), []

    tr_idx = _resolve_train_idx(len(station_ids), n_train, train_idx)

    topo_df = pd.read_csv(Path(topo_dir) / "topo_features.csv", dtype={"location_id": str})
    topo_df = topo_df[topo_df["kind"] == "station"].set_index("location_id")
    coast_df = pd.read_csv(Path(topo_dir) / "dist_coast.csv", dtype={"station_id": str})
    coast_df = coast_df.set_index("station_id")

    topo_df["aspect_sin"] = np.sin(np.deg2rad(topo_df["aspect"]))
    topo_df["aspect_cos"] = np.cos(np.deg2rad(topo_df["aspect"]))

    # Aspect is undefined on zero relief: the DEM gradient has no direction there,
    # so whatever angle the source reports is an artefact. The zero vector encodes
    # "no preferred direction" and is distinguishable from every real bearing,
    # which a median-filled or artefact angle is not. Same reasoning as tdi below.
    flat_relief = topo_df["elev_std"] == 0
    if flat_relief.any():
        logger.info(
            "Topo features 'aspect_sin'/'aspect_cos': set to 0 for %d perfectly flat "
            "location(s) (%s) — aspect is undefined on zero relief.",
            int(flat_relief.sum()), topo_df.index[flat_relief].tolist()[:5],
        )
        topo_df.loc[flat_relief, ["aspect_sin", "aspect_cos"]] = 0.0

    # Undefined dissection ratio on zero relief is 0, not missing.
    flat = topo_df["tdi"].isna() & flat_relief
    if flat.any():
        logger.info(
            "Topo feature 'tdi': set to 0.0 for %d perfectly flat location(s) (%s) — "
            "undefined ratio on zero relief, not missing data.",
            int(flat.sum()), topo_df.index[flat].tolist()[:5],
        )
        topo_df.loc[flat, "tdi"] = 0.0

    cols: list[np.ndarray] = []
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
            # Train-only median for the same reason the z-score is train-only:
            # a median over all stations leaks val/test statistics into the
            # imputation, contradicting point 2 of this docstring.
            median = values.iloc[tr_idx].median()
            logger.warning(
                "Topo station feature '%s': %d/%d missing, filling with median (%.4g) "
                "over the %d train stations. Missing IDs: %s",
                name, n_missing, len(station_ids), median, len(tr_idx),
                [sid for sid, v in zip(station_ids, values.isna()) if v][:10],
            )
            values = values.fillna(median)

        arr = values.to_numpy(dtype=np.float64)
        transform = _TOPO_TRANSFORMS.get(name)
        if transform is not None:
            arr = transform(arr)

        if name in ("aspect_sin", "aspect_cos"):
            # Do NOT z-score these: they are a unit-circle pair, already
            # bounded in [-1, 1]. Scaling each independently with its own
            # mean/std (as the branch below does for every other feature)
            # applies a different factor to sin and cos, turning the unit
            # circle into an ellipse — the angular distance between two
            # exposures then depends on direction, which defeats the point
            # of encoding aspect as sin/cos in the first place (review
            # brief L4).
            cols.append(arr.astype(np.float32))
            continue

        train_slice = arr[tr_idx]
        mean = float(train_slice.mean())
        std = max(float(train_slice.std()), 1e-6)
        cols.append(((arr - mean) / std).astype(np.float32))

    out = np.stack(cols, axis=1).astype(np.float32)
    logger.info(
        "Loaded %d topographic station features (z-score on %d train stations, "
        "%s): %s",
        out.shape[1], len(tr_idx),
        "leading" if train_idx is None else "explicit indices", requested,
    )
    return out, requested


def load_topo_station_features_dict(
    topo_dir: str,
    station_ids: list[str],
    feature_names: list[str],
    n_train: int | None = None,
    train_idx: list[int] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    ``load_topo_station_features`` in the dict form ``HomoSampler`` expects.

    HomoSampler takes ``topo_feats`` as ``{name: (N,) array}`` and both appends
    those columns to the static tensor (which feeds emb_mlp and, with
    ``broadcast_topo``, the input channels) and derives edge features from them.
    Use this rather than ``load_topo_node_features`` wherever the values end up
    as absolute node features — see that function's docstring for why the
    normalisation differs.
    """
    arr, names = load_topo_station_features(
        topo_dir, station_ids, feature_names, n_train=n_train, train_idx=train_idx,
    )
    return {name: arr[:, i] for i, name in enumerate(names)}


def load_topo_node_features(
    topo_dir: str,
    station_ids: list[str],
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    """
    Load and z-score-normalise topographic node features for a list of stations.

    .. warning::
       Normalises without variance-stabilising transforms and fits the z-score
       over **all** stations. That is tolerable for the pairwise edge
       differences in ``HeterogeneousGraphBuilder`` but wrong for absolute node
       features: heavily right-skewed features (tpi5, tdi, tpi75) collapse the
       bulk of the stations onto a near-constant value plus a single outlier.
       For node/broadcast use call ``load_topo_station_features_dict`` instead.

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
