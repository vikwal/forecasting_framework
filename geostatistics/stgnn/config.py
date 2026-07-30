"""
Configuration dataclasses for the heterogeneous STGNN.

The canonical configuration source is the ``stgnn2`` section of the YAML file.
Use ``ModelConfig.from_yaml(stgnn2_dict, ...)`` to build the full config —
do not set values directly on the dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .utils.topo_features import TOPO_FEATURE_ORDER

# Core edge features that have always existed (fixed column order 0,1,2 relied
# on by DCGRUCell.edge_weight_from_attr / direction_to_adj — never reorder).
_CORE_EDGE_FEATURES = ("distance", "direction", "altitude_diff")


def parse_edge_features(d: dict) -> tuple[bool, bool, bool, list[str]]:
    """
    Parse the ``edge_features`` config key (if present) into the three core
    boolean flags plus an ordered list of topographic feature names.

    If ``edge_features`` is absent, falls back to the legacy individual
    ``use_distance_features`` / ``use_direction_features`` / ``use_altitude_diff``
    flags (default True/True/False) with no topographic features — fully
    backward compatible with configs predating this key.
    """
    if "edge_features" in d:
        requested = set(d["edge_features"])
        use_distance = "distance" in requested
        use_direction = "direction" in requested
        use_altitude_diff = "altitude_diff" in requested
        topo_names = [f for f in TOPO_FEATURE_ORDER if f in requested]
    else:
        use_distance = d.get("use_distance_features", True)
        use_direction = d.get("use_direction_features", True)
        use_altitude_diff = d.get("use_altitude_diff", False)
        topo_names = []
    return use_distance, use_direction, use_altitude_diff, topo_names


def parse_station_node_features(d: dict, cli_override=None) -> list[str]:
    """
    Resolve the ``station_node_features`` key into an ordered list of topographic
    feature names, to be used as **absolute** node features on the station nodes.

    Deliberately separate from ``edge_features``: a difference between two nodes
    and a point property of one node are not interchangeable. ``aspect_sin/cos``
    for instance is meaningless as an edge difference (sin(a)-sin(b) is not an
    angular distance) but informative per node, where it can interact with the
    time-varying wind direction.

    Accepts a list of names, or the string ``"all"`` for every feature in
    TOPO_FEATURE_ORDER. Absent or empty means no node features, which keeps the
    declared static width — and therefore existing checkpoints — unchanged.

    ``cli_override`` (same accepted forms) takes precedence, so one config file
    can serve both arms of an A/B run without being edited.
    """
    raw = cli_override if cli_override is not None else d.get("station_node_features")
    if not raw:
        return []
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in ("none", "off", "false"):
            # Explicit empty — needed so the CLI can force the "no topo" arm even
            # when the config file requests features.
            return []
        raw = TOPO_FEATURE_ORDER if token == "all" else [
            s.strip() for s in raw.split(",") if s.strip()
        ]
    requested = set(raw)
    unknown = requested - set(TOPO_FEATURE_ORDER)
    if unknown:
        raise ValueError(
            f"station_node_features contains unknown name(s): {sorted(unknown)}. "
            f"Valid names: {TOPO_FEATURE_ORDER}"
        )
    return [f for f in TOPO_FEATURE_ORDER if f in requested]


@dataclass
class GraphConfig:
    station_connectivity: str
    next_n_icond2_grid_points: int
    next_n_ecmwf_grid_points: int
    station_k: int = 6
    use_distance_features: bool = True
    use_direction_features: bool = True
    use_altitude_diff: bool = False
    topo_features_path: Optional[str] = None
    topo_feature_names: List[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    min_target_stations: int
    max_target_stations: int
    max_neighbor_stations: int
    loss_fn: str
    loss_weights_by_horizon: bool
    horizon_decay: float
    lr: float
    weight_decay: float
    scheduler: str
    max_epochs: int
    batch_size: int
    gradient_clip: float
    patience: int
    checkpoint_path: str
    subsample_neighbors: bool = True
    neighbor_radius_km: Optional[float] = None  # None = no radius limit
    next_n_neighbors: Optional[int] = None      # when set: pick N spatially nearest neighbors instead of random
    val_stations: Optional[List[int]] = None
    test_stations: Optional[List[int]] = None


@dataclass
class ModelConfig:
    graph: GraphConfig
    training: TrainingConfig

    history_length: int
    forecast_horizon: int

    station_meas_features: int
    icond2_features_per_step: int
    ecmwf_features_per_step: int
    station_static_features: int
    icond2_static_features: int
    ecmwf_static_features: int

    latent_dim: int
    num_message_passing_layers: int
    encoder_mlp_layers: int
    decoder_mlp_layers: int
    aggregation: str

    temporal_encoding: str
    gru_hidden_dim: int
    gru_num_layers: int
    cnn_channels: int
    cnn_kernel_size: int

    dropout: float
    update_nwp_nodes: bool
    processor_heads: int = 4          # GATv2 attention heads in ST-blocks

    def station_input_dim(self) -> int:
        T_hist = self.history_length
        T_fore = self.forecast_horizon
        return (
            T_hist * self.station_meas_features
            + (T_hist + T_fore) * self.icond2_features_per_step
            + (T_hist + T_fore) * self.ecmwf_features_per_step
            + self.station_static_features
        )

    def icond2_input_dim(self) -> int:
        T = self.history_length + self.forecast_horizon
        return T * self.icond2_features_per_step + self.icond2_static_features

    def ecmwf_input_dim(self) -> int:
        T = self.history_length + self.forecast_horizon
        return T * self.ecmwf_features_per_step + self.ecmwf_static_features

    def edge_input_dim(self) -> int:
        """
        Dimension of NWP→station (i2s/e2s) edge_attr, consumed by GATv2Conv's
        declared ``edge_dim``. Topographic features are only added to
        station↔station (s2s) edges (see GraphConfig.topo_feature_names),
        which DCGRUCell reads dynamically without a declared width — so they
        must NOT be counted here.
        """
        dim = 0
        if self.graph.use_distance_features:
            dim += 1
        if self.graph.use_direction_features:
            dim += 2
        if self.graph.use_altitude_diff:
            dim += 1
        return max(dim, 1)

    @classmethod
    def from_yaml(
        cls,
        d: dict,
        *,
        icond2_features: list[str],
        ecmwf_features: list[str],
        measurement_features: list[str],
        n_train: int,
        n_val: int,
        checkpoint_path: str,
    ) -> "ModelConfig":
        """
        Build a ModelConfig from the ``stgnn2`` YAML section.

        Parameters
        ----------
        d :                     the ``stgnn2`` dict from the YAML file
        icond2_features :       list of ICON-D2 column names used
        ecmwf_features :        list of ECMWF column names used
        measurement_features :  list of station measurement column names
        n_train :               number of training stations
        n_val :                 number of validation stations
        checkpoint_path :       where to save the best model checkpoint
        """
        use_distance, use_direction, use_altitude_diff, topo_names = parse_edge_features(d)
        graph = GraphConfig(
            station_connectivity=d.get("station_connectivity", "delaunay"),
            next_n_icond2_grid_points=d["next_n_icond2"],
            next_n_ecmwf_grid_points=d["next_n_ecmwf"],
            use_distance_features=use_distance,
            use_direction_features=use_direction,
            use_altitude_diff=use_altitude_diff,
            topo_features_path=d.get("topo_features_path"),
            topo_feature_names=topo_names,
        )
        training = TrainingConfig(
            min_target_stations=d["min_target_stations"],
            max_target_stations=d["max_target_stations"],
            max_neighbor_stations=d.get("max_neighbor_stations", 60),
            neighbor_radius_km=d.get("neighbor_radius_km", None),
            loss_fn=d.get("loss_fn", "mse"),
            loss_weights_by_horizon=d.get("loss_weights_by_horizon", True),
            horizon_decay=d.get("horizon_decay", 0.95),
            lr=d["lr"],
            weight_decay=d.get("weight_decay", 1e-5),
            scheduler=d.get("scheduler", "cosine"),
            max_epochs=d["max_epochs"],
            batch_size=d["batch_size"],
            gradient_clip=d.get("gradient_clip", 1.0),
            patience=d["patience"],
            checkpoint_path=checkpoint_path,
            val_stations=list(range(n_train, n_train + n_val)),
        )
        return cls(
            graph=graph,
            training=training,
            history_length=d["history_length"],
            forecast_horizon=d["forecast_horizon"],
            station_meas_features=len(measurement_features),
            icond2_features_per_step=len(icond2_features),
            ecmwf_features_per_step=len(ecmwf_features),
            station_static_features=4,   # lat, lon, alt, type_indicator
            icond2_static_features=3,    # lat, lon, alt
            ecmwf_static_features=3,
            latent_dim=d["hidden"],
            num_message_passing_layers=d["num_layers"],
            encoder_mlp_layers=d.get("encoder_mlp_layers", 2),
            decoder_mlp_layers=d.get("decoder_mlp_layers", 2),
            aggregation=d.get("aggregation", "mean"),
            temporal_encoding=d.get("temporal_encoding", "cnn"),
            gru_hidden_dim=d.get("gru_hidden", 128),
            gru_num_layers=d.get("gru_layers", 2),
            cnn_channels=d.get("cnn_channels", 64),
            cnn_kernel_size=d.get("cnn_kernel_size", 5),
            dropout=d.get("dropout", 0.1),
            update_nwp_nodes=d.get("update_nwp_nodes", False),
            processor_heads=d.get("processor_heads", 4),
        )