"""
fixture.py — deterministic, data-free test fixture for the DCRNN ablation checks.

Why this exists
---------------
The verification suite for ablation variants B (``neighbour_meas_available:
false``) and C (``station_connectivity: "none"``) has to run *before* any GPU
time is spent, and long before the Optuna campaign finishes.  It therefore may
not touch the parquet stores under ``/mnt/.../icon-d2`` or the measurement CSVs.

What it does use is everything that is cheap and *real*:

  * the real config (``configs/dcrnn/config_wind_dcrnn.yaml``), parsed by the
    production ``DCRNNConfig.from_yaml``;
  * the real station IDs, coordinates and altitudes from
    ``data/stations_master.csv``;
  * the real topographic node/edge features from ``topo_features_path``;
  * the real ``HeterogeneousGraphBuilder`` and the real ``TrainingSampler``.

Only the *values* of the measurement and NWP tensors are synthetic, drawn from a
seeded ``numpy`` generator.  Shapes, dtypes, feature counts and the whole graph
topology are the production ones.  That is exactly what the checks need: they
assert on masking behaviour, edge sets and information flow, never on physics.

Everything here is deterministic given ``seed``.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from geostatistics.dcrnn import DCRNNConfig
from geostatistics.stgnn import HeterogeneousGraphBuilder
from geostatistics.stgnn.training.sampler import TrainingSampler


# ---------------------------------------------------------------------------
# Helpers mirroring train_dcrnn.py's feature resolution
# ---------------------------------------------------------------------------

def _resolved_features(dcrnn_cfg: dict) -> tuple[list[str], list[str]]:
    """Return (icond2_features, ecmwf_features) after mode + dir encoding.

    Mirrors train_dcrnn.py: ``resolve_feature_mode`` first, then
    ``apply_dir_encoding`` when the mode is ``dir_in_deg``.  The encoding is
    applied to a dummy array purely to recover the resulting *names*, which is
    the only thing that determines I2 / E2.
    """
    from geostatistics.train_dcrnn import apply_dir_encoding, resolve_feature_mode

    i2_mode = dcrnn_cfg.get("icond2_feature_mode", "both")
    e2_mode = dcrnn_cfg.get("ecmwf_feature_mode", "both")
    i2 = resolve_feature_mode(dcrnn_cfg.get("icond2_features") or [], i2_mode)
    e2 = resolve_feature_mode(dcrnn_cfg.get("ecmwf_features") or [], e2_mode)

    if i2_mode == "dir_in_deg":
        _, i2 = apply_dir_encoding(np.zeros((1, 1, 1, len(i2)), dtype=np.float32), i2)
    if e2_mode == "dir_in_deg" and len(e2) > 0:
        _, e2 = apply_dir_encoding(np.zeros((1, 1, len(e2)), dtype=np.float32), e2)
    return list(i2), list(e2)


def _measurement_cols(dcrnn_cfg: dict) -> list[str]:
    """Measurement columns after ``encode_circular_measurements``."""
    cols = list(dcrnn_cfg.get("measurement_features") or [])
    if "wind_direction" in cols:
        i = cols.index("wind_direction")
        cols = cols[:i] + ["sin_wind_direction", "cos_wind_direction"] + cols[i + 1:]
    return cols


def _synthetic_grid(coords: np.ndarray, n_lat: int, n_lon: int) -> np.ndarray:
    """Regular lat/lon grid covering the station bounding box (+ margin)."""
    lat0, lat1 = float(coords[:, 0].min()) - 0.3, float(coords[:, 0].max()) + 0.3
    lon0, lon1 = float(coords[:, 1].min()) - 0.3, float(coords[:, 1].max()) + 0.3
    la = np.linspace(lat0, lat1, n_lat)
    lo = np.linspace(lon0, lon1, n_lon)
    gla, glo = np.meshgrid(la, lo, indexing="ij")
    return np.stack([gla.ravel(), glo.ravel()], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

class Fixture:
    """Container for everything the sampler / model calls need."""

    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)

    # -- convenience ----------------------------------------------------

    def sample_train_kwargs(self) -> dict:
        return dict(
            station_meas=self.station_meas,
            station_nearest_grid=self.station_nearest_grid,
            grid_icond2_runs=self.grid_icond2_runs,
            station_ecmwf_nwp=self.station_ecmwf_nwp,
            station_static=self.station_static,
            ecmwf_nwp=self.ecmwf_nwp,
            icond2_static=self.icond2_static,
            ecmwf_static=self.ecmwf_static,
            train_station_indices=self.train_station_indices,
        )

    def sample_val_kwargs(self) -> dict:
        d = self.sample_train_kwargs()
        d["val_station_indices"] = self.val_station_indices
        return d


def build_fixture(
    config_path: str | Path,
    *,
    seed: int = 20260803,
    overrides: dict | None = None,
    n_stations: int = 60,
    n_grid_lat: int = 14,
    n_grid_lon: int = 14,
    n_ecmwf_lat: int = 6,
    n_ecmwf_lon: int = 6,
    n_runs: int = 4,
) -> Fixture:
    """Build a deterministic fixture from a real DCRNN config.

    Parameters
    ----------
    config_path : path to a ``configs/dcrnn/config_wind_dcrnn*.yaml``
    seed        : seed for the synthetic tensors (numpy Generator)
    overrides   : dict merged into the ``dcrnn:`` section *before* parsing —
                  this is how the ablation variants are switched on
    n_stations  : how many of the config's stations to use (keeps the graph
                  small enough that the whole suite runs in seconds)
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    data_cfg = cfg["data"]
    dcrnn_cfg = copy.deepcopy(cfg.get("dcrnn", {}))
    if overrides:
        dcrnn_cfg.update(overrides)

    rng = np.random.default_rng(seed)

    # ── stations: real IDs, real coordinates, real altitudes ───────────
    from geostatistics.train_stgnn2 import load_station_metadata

    train_ids_all = [str(s) for s in data_cfg["files"]]
    val_ids_all = [str(s) for s in data_cfg["val_files"]]
    n_val = max(4, n_stations // 4)
    n_train = n_stations - n_val
    train_ids = train_ids_all[:n_train]
    val_ids = val_ids_all[:n_val]
    all_ids = train_ids + val_ids
    N_train, N_val = len(train_ids), len(val_ids)
    N_all = len(all_ids)

    lats, lons, alts = load_station_metadata(
        data_cfg["path"], all_ids, meta_path=data_cfg["stations_master"],
    )
    station_coords = np.stack([lats, lons], axis=1).astype(np.float32)
    alts = alts.astype(np.float32)

    # ── feature dimensions, exactly as the training script derives them ─
    icond2_features, ecmwf_features = _resolved_features(dcrnn_cfg)
    measurement_cols = _measurement_cols(dcrnn_cfg)
    target_col = dcrnn_cfg.get("target_col")
    I2, E2 = len(icond2_features), len(ecmwf_features)
    M = len(measurement_cols)

    model_cfg = DCRNNConfig.from_yaml(
        dcrnn_cfg,
        icond2_features=icond2_features,
        ecmwf_features=ecmwf_features,
        measurement_features=measurement_cols,
        target_col=target_col,
        n_train=N_train,
        n_val=N_val,
        checkpoint_path="/dev/null",
        station_node_features=None,
    )

    H_hist = model_cfg.history_length
    H_fore = model_cfg.forecast_horizon

    # ── NWP grids (synthetic geometry, real geodesic machinery) ────────
    icond2_coords = _synthetic_grid(station_coords, n_grid_lat, n_grid_lon)
    if model_cfg.graph.next_n_ecmwf_grid_points > 0:
        ecmwf_coords = _synthetic_grid(station_coords, n_ecmwf_lat, n_ecmwf_lon)
    else:
        ecmwf_coords = np.empty((0, 2), dtype=np.float32)
    N_igrid, N_egrid = len(icond2_coords), len(ecmwf_coords)
    icond2_alts = rng.uniform(0.0, 900.0, size=N_igrid).astype(np.float32)
    ecmwf_alts = rng.uniform(0.0, 900.0, size=N_egrid).astype(np.float32)

    # ── graph (real builder, real topo features) ───────────────────────
    builder = HeterogeneousGraphBuilder(model_cfg.graph)
    base_graph = builder.build(
        station_coords=station_coords,
        station_altitudes=alts,
        icond2_grid_coords=icond2_coords,
        ecmwf_grid_coords=ecmwf_coords,
        icond2_altitudes=icond2_alts,
        ecmwf_altitudes=ecmwf_alts,
        station_ids=all_ids,
    )

    # ── synthetic, seeded tensors with production shapes ───────────────
    T = H_hist + H_fore + 8
    station_meas = rng.normal(0.0, 1.0, size=(T, N_all, M)).astype(np.float32)
    grid_icond2_runs = rng.normal(
        0.0, 1.0, size=(n_runs, 48, N_igrid, I2)).astype(np.float32)
    station_ecmwf_nwp = rng.normal(0.0, 1.0, size=(T, N_all, E2)).astype(np.float32)
    ecmwf_nwp = rng.normal(0.0, 1.0, size=(T, N_egrid, E2)).astype(np.float32)

    # station.static == 3 geo columns + len(station_node_feature_names) topo
    # columns; the sampler appends the type indicator, so this array is one
    # column narrower than model_cfg.station_static_features.
    S_pre = 3 + len(model_cfg.station_node_feature_names)
    station_static = rng.normal(0.0, 1.0, size=(N_all, S_pre)).astype(np.float32)
    icond2_static = rng.normal(0.0, 1.0, size=(N_igrid, 3)).astype(np.float32)
    ecmwf_static = rng.normal(0.0, 1.0, size=(N_egrid, 3)).astype(np.float32)

    from geostatistics.stgnn.utils.spatial import geodesic_knn

    _, nearest = geodesic_knn(icond2_coords, station_coords, k=1)
    station_nearest_grid = nearest[:, 0].astype(np.int64)

    sampler = TrainingSampler(
        model_cfg, builder, base_graph,
        target_feat_idx=model_cfg.target_feat_idx,
        station_coords=station_coords,
    )

    return Fixture(
        cfg=cfg,
        dcrnn_cfg=dcrnn_cfg,
        model_cfg=model_cfg,
        builder=builder,
        base_graph=base_graph,
        sampler=sampler,
        all_ids=all_ids,
        station_coords=station_coords,
        station_altitudes=alts,
        icond2_coords=icond2_coords,
        ecmwf_coords=ecmwf_coords,
        icond2_altitudes=icond2_alts,
        ecmwf_altitudes=ecmwf_alts,
        station_meas=station_meas,
        station_nearest_grid=station_nearest_grid,
        grid_icond2_runs=grid_icond2_runs,
        station_ecmwf_nwp=station_ecmwf_nwp,
        station_static=station_static,
        ecmwf_nwp=ecmwf_nwp,
        icond2_static=icond2_static,
        ecmwf_static=ecmwf_static,
        train_station_indices=list(range(N_train)),
        val_station_indices=list(range(N_train, N_all)),
        N_train=N_train,
        N_val=N_val,
        N_all=N_all,
        M=M,
        I2=I2,
        E2=E2,
        T=T,
        H_hist=H_hist,
        H_fore=H_fore,
        n_runs=n_runs,
        measurement_cols=measurement_cols,
        icond2_features=icond2_features,
        ecmwf_features=ecmwf_features,
    )
