"""
Configuration for the DCRNN Seq2Seq model.

Reuses ``TrainingConfig`` and ``GraphConfig`` from the stgnn package so the
existing TrainingSampler and HeterogeneousGraphBuilder work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from dataclasses import field

from geostatistics.stgnn.config import (
    GraphConfig,
    TrainingConfig,
    parse_edge_features,
    parse_station_node_features,
)


@dataclass
class DCRNNConfig:
    """
    Full configuration for the DCRNN model.

    The ``training`` and ``graph`` sub-configs are the same dataclasses used
    by the STGNN, so the same sampler and graph builder can be reused.

    Attributes
    ----------
    training / graph       : shared with stgnn (sampler-compatible)
    history_length         : encoder steps  (T_hist)
    forecast_horizon       : decoder steps  (T_fore)
    temporal_encoding      : "gru" or "cnn" — controls sampler output format
    target_feat_idx        : index of target col in measurement_features

    station_meas_features      : M
    icond2_features_per_step   : I2
    ecmwf_features_per_step    : E2
    station_static_features    : S  (lat, lon, alt, type_indicator = 4)
    icond2_static_features     : S_i (lat, lon, alt = 3)
    ecmwf_static_features      : S_e (lat, lon, alt = 3)

    hidden_dim             : DCGRU hidden dimension
    num_layers             : stacked DCGRU layers
    K_hop            : diffusion hops in DiffConv (station s2s graph)
    nwp_out_dim            : output dimension of NWPAttentionLayer
                             (= nwp_heads * out_per_head)
    nwp_heads              : GATv2 attention heads for NWP → station
    dropout                : dropout in cells and attention

    teacher_forcing_ratio  : starting probability; linearly decayed to 0
    n_ahead_prefetch       : queue depth for the batch prefetcher (default: 2)
    prefetch_workers       : worker threads for prefetcher (default: 1)
    """

    # Sub-configs
    training: TrainingConfig
    graph: GraphConfig

    # Time
    history_length: int
    forecast_horizon: int
    temporal_encoding: str
    target_feat_idx: int

    # Feature dims
    station_meas_features: int
    icond2_features_per_step: int
    ecmwf_features_per_step: int
    station_static_features: int = 4
    icond2_static_features: int = 3
    ecmwf_static_features: int = 3

    # Absolute topographic node features appended to station.static (after
    # lat/lon/alt, before the type indicator the sampler adds last). Empty by
    # default, so station_static_features stays 4 and old checkpoints load.
    station_node_feature_names: list[str] = field(default_factory=list)

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 2
    K_hop: int = 2
    nwp_out_dim: int = 128     # output of NWPAttentionLayer; must be divisible by nwp_heads
    nwp_heads: int = 4
    dropout: float = 0.1

    # Training
    teacher_forcing_ratio: float = 0.5
    tf_schedule: str = "linear"        # "linear" | "inv_sigmoid" (paper Appendix E)
    tf_tau: float = 3000.0             # τ for inv_sigmoid schedule (paper default: 3000)
    n_ahead_prefetch: int = 2          # queue depth for batch prefetcher
    prefetch_workers: int = 1          # number of worker threads (currently only 1 supported)

    # Edge weight kernel scale (paper Appendix E.1: W_ij = exp(-d²/σ²))
    edge_weight_sigma: float = 0.2

    # Kriging lag feature: when True, pre-scaled Kriging predictions are appended as an extra
    # measurement channel for all nodes; unlike real measurements, this channel is NOT zeroed
    # out for target nodes, giving the model an external prior at inference time.
    interpolate_history: bool = False

    # Ablation B / C: when False the sampler zeroes the measurement channels of
    # EVERY station, not just the target nodes, so the station graph carries only
    # NWP features and statics. target_mask semantics are untouched, so A and the
    # ablation variants are scored at exactly the same nodes.
    # Must not be combined with interpolate_history=True — the Kriging lag channel
    # is appended *after* the zeroing and would smuggle neighbour measurements back
    # in on a path that bypasses the graph entirely (asserted in train_dcrnn.py,
    # get_test_results_dcrnn.py and hpo_dcrnn.py).
    neighbour_meas_available: bool = True

    # NWP injection bypass: when False, nwp_out_dim is forced to 0 so the
    # NWPAttentionLayer is never constructed / called.  Station.x still carries
    # the nearest-grid NWP features put there by the sampler, so the model has
    # NWP information but no learned graph-based NWP attention.
    nwp_injection: bool = True

    # NWP graph nodes toggle: when True (default), NWP grid points are explicit
    # graph nodes aggregated via GATv2 (nwp_injection must also be True).
    # When False, NWP features from station.x (single nearest grid point, all T
    # steps) are concatenated directly to station measurements — no GATv2.
    nwp_nodes: bool = True

    # Directional adjacency: when True, s2s edge weights are recomputed at each
    # timestep using the NWP wind direction at each station's nearest ICON-D2 grid
    # point.  Edges aligned with the wind flow receive higher weight; opposing edges
    # are down-weighted to zero.  Requires use_direction_features=True in GraphConfig
    # and "wind_direction" in measurement_features.
    # Encoder uses measured wind_direction; decoder uses u/v from data["icond2"].wind_uv
    # (loaded separately by the training script, NOT part of model input features I2).
    direction_to_adj: bool = False
    wind_dir_meas_idx: int = -1     # index of sin_wind_direction (or raw wind_direction) in encoded measurement_features
    wind_dir_cos_idx: int = -1      # index of cos_wind_direction (-1 when wind_direction is raw degrees)

    # ------------------------------------------------------------------

    def edge_input_dim(self) -> int:
        """
        Dimension of NWP→station (i2s/e2s) edge_attr, consumed by GATv2Conv's
        declared ``edge_dim`` in NWPAttentionLayer.

        Matches stgnn GraphConfig.edge_input_dim() logic:
          1 (distance) + 2 (sin/cos bearing) + 1 (altitude diff, optional)

        Topographic features (graph.topo_feature_names) are only added to
        station↔station (s2s) edges, which DCGRUCell reads dynamically without
        a declared width — so they must NOT be counted here.
        """
        dim = 0
        if self.graph.use_distance_features:
            dim += 1
        if self.graph.use_direction_features:
            dim += 2
        if self.graph.use_altitude_diff:
            dim += 1
        return max(dim, 1)

    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        d: dict,
        *,
        icond2_features: list[str],
        ecmwf_features: list[str],
        measurement_features: list[str],
        target_col: str,
        n_train: int,
        n_val: int,
        checkpoint_path: str,
        station_node_features=None,
    ) -> "DCRNNConfig":
        use_distance, use_direction, use_altitude_diff, topo_names = parse_edge_features(d)
        node_feat_names = parse_station_node_features(d, station_node_features)
        graph = GraphConfig(
            station_connectivity=d.get("station_connectivity", "delaunay"),
            next_n_icond2_grid_points=d.get("next_n_icond2", 4),
            next_n_ecmwf_grid_points=d.get("next_n_ecmwf", 4),
            use_distance_features=use_distance,
            use_direction_features=use_direction,
            use_altitude_diff=use_altitude_diff,
            topo_features_path=d.get("topo_features_path"),
            topo_feature_names=topo_names,
        )
        training = TrainingConfig(
            min_target_stations=d.get("min_target_stations", 1),
            max_target_stations=d.get("max_target_stations", 10),
            max_neighbor_stations=d.get("max_neighbor_stations", 60),
            neighbor_radius_km=d.get("neighbor_radius_km", None),
            next_n_neighbors=d.get("next_n_neighbors", None),
            loss_fn=d.get("loss_fn", "mse"),
            loss_weights_by_horizon=d.get("loss_weights_by_horizon", True),
            horizon_decay=d.get("horizon_decay", 0.95),
            lr=d["lr"],
            weight_decay=d.get("weight_decay", 1e-5),
            scheduler=d.get("scheduler", "plateau"),
            max_epochs=d["max_epochs"],
            # "grad_accum" ist der kanonische Schluessel, "batch_size" der
            # aeltere Name derselben Groesse (DCRNNTrainer batcht echt, es
            # wird nichts akkumuliert). Beide muessen gelesen werden: die
            # Base-Config tunte "batch_size" per HPO, hier kam aber nur
            # "grad_accum" an — der Parameter war wirkungslos und jeder
            # Base-Trial lief mit dem Default 4, also ~369 statt ~38
            # Optimizer-Schritten pro Epoche.
            batch_size=d.get("grad_accum", d.get("batch_size", 4)),
            gradient_clip=d.get("gradient_clip", 1.0),
            patience=d["patience"],
            checkpoint_path=checkpoint_path,
            val_stations=list(range(n_train, n_train + n_val)),
        )

        hidden_dim     = d.get("hidden")
        nwp_injection  = d.get("nwp_injection", True)
        nwp_heads      = d.get("nwp_heads", 4)
        nwp_out_dim    = d.get("nwp_out_dim") or hidden_dim
        if not nwp_injection:
            nwp_out_dim = 0
            nwp_heads   = 1   # dummy — NWPAttentionLayer not created
        # Enforce divisibility (skipped when nwp_out_dim == 0)
        if nwp_out_dim > 0 and nwp_out_dim % nwp_heads != 0:
            raise ValueError(
                f"nwp_out_dim ({nwp_out_dim}) must be divisible by nwp_heads ({nwp_heads})"
            )

        interpolate_history = d.get("interpolate_history", False)
        # Default True preserves variant A exactly.
        neighbour_meas_available = d.get("neighbour_meas_available", True)

        direction_to_adj = d.get("direction_to_adj", False)
        # Support both raw degrees ("wind_direction") and sin/cos encoding ("sin_wind_direction")
        if "sin_wind_direction" in measurement_features:
            wind_dir_meas_idx = measurement_features.index("sin_wind_direction")
            wind_dir_cos_idx  = measurement_features.index("cos_wind_direction")
        elif "wind_direction" in measurement_features:
            wind_dir_meas_idx = measurement_features.index("wind_direction")
            wind_dir_cos_idx  = -1
        else:
            wind_dir_meas_idx = -1
            wind_dir_cos_idx  = -1
        if direction_to_adj:
            if wind_dir_meas_idx < 0:
                raise ValueError(
                    "direction_to_adj=True requires 'wind_direction' or "
                    "'sin_wind_direction'/'cos_wind_direction' in measurement_features."
                )
            if not use_direction:
                raise ValueError(
                    "direction_to_adj=True requires direction features enabled "
                    "(bearing columns must be present in edge_attr)."
                )

        nwp_nodes = d.get("nwp_nodes", True)

        return cls(
            training=training,
            graph=graph,
            history_length=d.get("history_length", 48),
            forecast_horizon=d.get("forecast_horizon", 48),
            temporal_encoding=d.get("temporal_encoding", "gru"),
            target_feat_idx=measurement_features.index(target_col),
            station_meas_features=len(measurement_features) + (1 if interpolate_history else 0),
            interpolate_history=interpolate_history,
            neighbour_meas_available=neighbour_meas_available,
            icond2_features_per_step=len(icond2_features),
            ecmwf_features_per_step=len(ecmwf_features) if d.get("next_n_ecmwf", 4) > 0 else 0,
            station_static_features=4 + len(node_feat_names),
            station_node_feature_names=node_feat_names,
            icond2_static_features=3,
            ecmwf_static_features=3,
            hidden_dim=hidden_dim,
            num_layers=d.get("num_layers", 2),
            K_hop=d.get("K_hop", 2),
            nwp_out_dim=nwp_out_dim,
            nwp_heads=nwp_heads,
            dropout=d.get("dropout", 0.1),
            teacher_forcing_ratio=d.get("teacher_forcing_ratio", 0.5),
            tf_schedule=d.get("tf_schedule", "linear"),
            tf_tau=float(d.get("tf_tau", 3000.0)),
            edge_weight_sigma=float(d.get("edge_weight_sigma", 0.2)),
            n_ahead_prefetch=d.get("n_ahead_prefetch", 2),
            prefetch_workers=d.get("prefetch_workers", 1),
            nwp_injection=nwp_injection,
            nwp_nodes=nwp_nodes,
            direction_to_adj=direction_to_adj,
            wind_dir_meas_idx=wind_dir_meas_idx,
            wind_dir_cos_idx=wind_dir_cos_idx,
        )
