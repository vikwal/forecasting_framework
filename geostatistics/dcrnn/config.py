"""
Configuration for the DCRNN Seq2Seq model.

Reuses ``TrainingConfig`` and ``GraphConfig`` from the stgnn package so the
existing TrainingSampler and HeterogeneousGraphBuilder work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from geostatistics.stgnn.config import GraphConfig, TrainingConfig


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
    diffusion_K            : diffusion hops in DiffConv (station s2s graph)
    nwp_out_dim            : output dimension of NWPAttentionLayer
                             (= nwp_heads * out_per_head)
    nwp_heads              : GATv2 attention heads for NWP → station
    dropout                : dropout in cells and attention

    teacher_forcing_ratio  : starting probability; linearly decayed to 0
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

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 2
    diffusion_K: int = 2
    nwp_out_dim: int = 128     # output of NWPAttentionLayer; must be divisible by nwp_heads
    nwp_heads: int = 4
    dropout: float = 0.1

    # Training
    teacher_forcing_ratio: float = 0.5

    # ------------------------------------------------------------------

    def edge_input_dim(self) -> int:
        """
        Dimension of edge_attr produced by HeterogeneousGraphBuilder.

        Matches stgnn GraphConfig.edge_input_dim() logic:
          1 (distance) + 2 (sin/cos bearing) + 1 (altitude diff, optional)
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
    ) -> "DCRNNConfig":
        graph = GraphConfig(
            station_connectivity=d.get("station_connectivity", "delaunay"),
            next_n_icond2_grid_points=d.get("next_n_icond2", 4),
            next_n_ecmwf_grid_points=d.get("next_n_ecmwf", 4),
            use_altitude_diff=d.get("use_altitude_diff", False),
        )
        training = TrainingConfig(
            min_target_stations=d.get("min_target_stations", 1),
            max_target_stations=d.get("max_target_stations", 10),
            max_neighbor_stations=d.get("max_neighbor_stations", 60),
            neighbor_radius_km=d.get("neighbor_radius_km", None),
            loss_fn=d.get("loss_fn", "mse"),
            loss_weights_by_horizon=d.get("loss_weights_by_horizon", True),
            horizon_decay=d.get("horizon_decay", 0.95),
            lr=d["lr"],
            weight_decay=d.get("weight_decay", 1e-5),
            scheduler=d.get("scheduler", "plateau"),
            max_epochs=d["max_epochs"],
            batch_size=d.get("batch_size", 8),
            gradient_clip=d.get("gradient_clip", 1.0),
            patience=d["patience"],
            checkpoint_path=checkpoint_path,
            val_stations=list(range(n_train, n_train + n_val)),
        )

        hidden_dim  = d.get("hidden")
        nwp_heads   = d.get("nwp_heads")
        nwp_out_dim = d.get("nwp_out_dim") or hidden_dim
        # Enforce divisibility
        if nwp_out_dim % nwp_heads != 0:
            raise ValueError(
                f"nwp_out_dim ({nwp_out_dim}) must be divisible by nwp_heads ({nwp_heads})"
            )

        return cls(
            training=training,
            graph=graph,
            history_length=d.get("history_length", 48),
            forecast_horizon=d.get("forecast_horizon", 48),
            temporal_encoding=d.get("temporal_encoding", "gru"),
            target_feat_idx=measurement_features.index(target_col),
            station_meas_features=len(measurement_features),
            icond2_features_per_step=len(icond2_features),
            ecmwf_features_per_step=len(ecmwf_features),
            station_static_features=4,
            icond2_static_features=3,
            ecmwf_static_features=3,
            hidden_dim=hidden_dim,
            num_layers=d.get("num_layers", 2),
            diffusion_K=d.get("diffusion_K", 2),
            nwp_out_dim=nwp_out_dim,
            nwp_heads=nwp_heads,
            dropout=d.get("dropout", 0.1),
            teacher_forcing_ratio=d.get("teacher_forcing_ratio", 0.5),
        )
