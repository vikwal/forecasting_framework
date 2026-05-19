"""
Full DCRNN model: Encoder → Decoder (Seq2Seq with NWP node attention).

Data flow
---------
Inputs from the sampler (temporal_encoding="gru"):

  data["station"].x       : (N_s, T, M+I2+E2)  — meas + NWP per station
  data["station"].static  : (N_s, S)
  data["icond2"].x        : (N_i, T, I2)        — full NWP sequence per grid point
  data["ecmwf"].x         : (N_e, T, E2)
  edge types              : s2s, i2s, e2s

NWP columns are still in station.x (put there by the sampler) but the DCRNN
reads the *original* NWP sequences from icond2.x / ecmwf.x so every grid
point speaks for itself through directed graph edges.

Shape conventions
-----------------
  T       = T_hist + T_fore = 96
  M       = station measurement features (e.g. 2: wind_speed, wind_direction)
  I2, E2  = NWP features per step
  S       = static station features (lat, lon, alt, type_indicator = 4)
  d       = hidden_dim
  d_nwp   = nwp_out_dim (output dim of NWPAttentionLayer)

Encoder loop (t = 0 … T_hist−1):
  nwp_msg_t = GATv2(icond2_t, ecmwf_t → station | H[-1])   (N_s, d_nwp)
  input_t   = cat(meas_t, nwp_msg_t, static)                (N_s, M+d_nwp+S)
  H_l       = DCGRUCell(input_t, H_l, s2s_graph)

Decoder loop (t = 0 … T_fore−1):
  nwp_msg_t = GATv2(icond2_fore_t, ecmwf_fore_t → station | H[-1])
  input_t   = cat(y_{t-1}, nwp_msg_t, static)               (N_s, 1+d_nwp+S)
  H_l       = DCGRUCell(input_t, H_l, s2s_graph)
  ŷ_t       = Linear(H[-1])
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from .dcgru_cell import DCGRUCell
from .encoder import DCGRUEncoder
from .decoder import DCGRUDecoder
from ..config import DCRNNConfig


class DCRNN(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network for inductive
    spatiotemporal wind speed forecasting.

    Parameters
    ----------
    config : DCRNNConfig
    """

    def __init__(self, config: DCRNNConfig) -> None:
        super().__init__()
        self.cfg               = config
        self.T_hist            = config.history_length
        self.T_fore            = config.forecast_horizon
        self.M                 = config.station_meas_features
        self.target_feat_idx   = config.target_feat_idx
        self.direction_to_adj  = config.direction_to_adj
        self.wind_dir_meas_idx = config.wind_dir_meas_idx
        self.wind_dir_cos_idx  = config.wind_dir_cos_idx
        self._icond2_k         = config.graph.next_n_icond2_grid_points
        self.nwp_nodes         = config.nwp_nodes
        self.edge_weight_sigma = config.edge_weight_sigma

        edge_dim = config.edge_input_dim()

        if config.nwp_nodes:
            # Standard path: GATv2 over explicit NWP nodes
            enc_meas_dim     = config.station_meas_features          # M
            nwp_out_dim      = config.nwp_out_dim
            station_nwp_dim  = 0
        else:
            # nwp_nodes=False: k nearest NWP grid points concatenated into station.x, no GATv2
            _k_i2 = config.graph.next_n_icond2_grid_points
            enc_meas_dim    = (config.station_meas_features
                               + _k_i2 * config.icond2_features_per_step
                               + config.ecmwf_features_per_step)     # M + k*I2 + E2
            nwp_out_dim     = 0
            station_nwp_dim = (_k_i2 * config.icond2_features_per_step
                               + config.ecmwf_features_per_step)     # k*I2 + E2

        shared_kwargs = dict(
            icond2_dim=config.icond2_features_per_step,
            ecmwf_dim=config.ecmwf_features_per_step,
            nwp_out_dim=nwp_out_dim,
            static_dim=config.station_static_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            K=config.K_hop,
            dropout=config.dropout,
            nwp_heads=config.nwp_heads,
            edge_dim=edge_dim,
            nwp_nodes=config.nwp_nodes,
        )

        self.encoder = DCGRUEncoder(
            meas_dim=enc_meas_dim,
            **shared_kwargs,
        )
        self.decoder = DCGRUDecoder(
            forecast_horizon=config.forecast_horizon,
            station_nwp_dim=station_nwp_dim,
            **shared_kwargs,
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        data: HeteroData,
        target_mask: Tensor,
        teacher_forcing_targets: Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tensor:
        """
        Parameters
        ----------
        data                    : HeteroData from TrainingSampler
        target_mask             : (N_s,) bool — True for target stations
        teacher_forcing_targets : (N_target, T_fore) or None
        teacher_forcing_ratio   : probability of using GT per decoder step

        Returns
        -------
        preds : (N_target, forecast_horizon)
        """
        i2s_key = ("icond2", "informs", "station")
        e2s_key = ("ecmwf",  "informs", "station")
        s2s_key = ("station", "near",   "station")

        # ── Read node features ─────────────────────────────────────────
        # station.x  : (N_s, T, M+I2+E2)  — measurements in first M cols
        x_station = data["station"].x                  # (N_s, T, M+I2+E2)
        static    = data["station"].static             # (N_s, S)

        # icond2 / ecmwf sequences  (N_nwp, T, F_nwp)
        icond2_seq = data["icond2"].x                  # (N_i, T, I2)
        ecmwf_seq  = data["ecmwf"].x                   # (N_e, T, E2)

        # ── Edge indices and attributes ────────────────────────────────
        s2s_ei = data[s2s_key].edge_index
        s2s_ea = data[s2s_key].edge_attr
        s2s_ew = DCGRUCell.edge_weight_from_attr(s2s_ea, sigma=self.edge_weight_sigma)   # (E,)

        i2s_ei = data[i2s_key].edge_index
        i2s_ea = data[i2s_key].edge_attr
        e2s_ei = data[e2s_key].edge_index
        e2s_ea = data[e2s_key].edge_attr

        # ── Build meas_seq and station_nwp_fore ────────────────────────
        if self.nwp_nodes:
            # Standard path: only M measurement channels feed the encoder
            meas      = x_station[:, :self.T_hist, :self.M]          # (N_s, T_hist, M)
            meas_seq  = meas.permute(1, 0, 2)                         # (T_hist, N_s, M)
            station_nwp_fore = None
        else:
            # nwp_nodes=False: all M+I2+E2 channels from station.x feed the encoder
            meas_full        = x_station[:, :self.T_hist, :]          # (N_s, T_hist, M+I2+E2)
            meas_seq         = meas_full.permute(1, 0, 2)             # (T_hist, N_s, M+I2+E2)
            # NWP columns of the forecast window for the decoder
            station_nwp_fore = x_station[:, self.T_hist:, self.M:].permute(1, 0, 2)
            #                                                          # (T_fore, N_s, I2+E2)
            meas = x_station[:, :self.T_hist, :self.M]  # needed for y_last below

        # ── Transpose NWP sequences for loop: (N, T, F) → (T, N, F) ──
        i2_hist = icond2_seq[:, :self.T_hist, :].permute(1, 0, 2)    # (T_hist, N_i, I2)
        e2_hist = ecmwf_seq[:,  :self.T_hist, :].permute(1, 0, 2)    # (T_hist, N_e, E2)
        i2_fore = icond2_seq[:, self.T_hist:, :].permute(1, 0, 2)    # (T_fore, N_i, I2)
        e2_fore = ecmwf_seq[:,  self.T_hist:, :].permute(1, 0, 2)    # (T_fore, N_e, E2)

        # ── Directional adjacency (wind-conditioned edge weights) ──────
        # Encoder: flow direction from measured wind_direction (met. convention, degrees).
        #          Target station measurements are zeroed by IGNNK masking → 0° for targets.
        # Decoder: flow direction derived from NWP u/v components at nearest icond2 node.
        dir_kwargs_enc: dict = {}
        dir_kwargs_dec: dict = {}
        if self.direction_to_adj:
            import math
            s2s_dist_norm = s2s_ea[:, 0]                                          # (E,)
            s2s_bearing   = torch.atan2(s2s_ea[:, 1], s2s_ea[:, 2])              # (E,) rad

            # Encoder: wind direction → flow direction (radians)
            if self.wind_dir_cos_idx >= 0:
                # sin/cos encoded: atan2(sin, cos) recovers the original angle
                sin_wd = meas_seq[:, :, self.wind_dir_meas_idx]                   # (T_hist, N_s)
                cos_wd = meas_seq[:, :, self.wind_dir_cos_idx]
                flow_enc = torch.atan2(sin_wd, cos_wd) + math.pi                  # (T_hist, N_s) rad
            else:
                # raw degrees (legacy path)
                wd_deg = meas_seq[:, :, self.wind_dir_meas_idx]                   # (T_hist, N_s)
                flow_enc = wd_deg * (math.pi / 180.0) + math.pi                   # (T_hist, N_s) rad

            # Decoder: u/v from data["icond2"].wind_uv (raw, NOT scaled model features)
            wind_uv = data["icond2"].wind_uv                                      # (N_grid_batch, T, 2)
            wind_uv_fore = wind_uv[:, self.T_hist:, :].permute(1, 0, 2)          # (T_fore, N_grid_batch, 2)
            nearest_i2 = i2s_ei[0].reshape(-1, self._icond2_k)[:, 0]             # (N_s_total,)
            u_dec = wind_uv_fore[:, nearest_i2, 0]                                # (T_fore, N_s)
            v_dec = wind_uv_fore[:, nearest_i2, 1]                                # (T_fore, N_s)
            flow_dec = torch.atan2(u_dec, v_dec)                                  # (T_fore, N_s) rad

            dir_kwargs_enc = dict(
                wind_dir_seq=flow_enc, s2s_dist_norm=s2s_dist_norm, s2s_bearing=s2s_bearing,
            )
            dir_kwargs_dec = dict(
                wind_dir_fore=flow_dec, s2s_dist_norm=s2s_dist_norm, s2s_bearing=s2s_bearing,
            )

        # ── Encoder ────────────────────────────────────────────────────
        H_list = self.encoder(
            meas_seq=meas_seq,
            icond2_seq=i2_hist,
            ecmwf_seq=e2_hist,
            static=static,
            s2s_edge_index=s2s_ei,
            s2s_edge_weight=s2s_ew,
            i2s_edge_index=i2s_ei,
            i2s_edge_attr=i2s_ea,
            e2s_edge_index=e2s_ei,
            e2s_edge_attr=e2s_ea,
            **dir_kwargs_enc,
        )

        # Last observed value for all nodes (target stations: 0, zeroed by sampler)
        y_last = meas[:, -1, self.target_feat_idx]         # (N_s,)

        # ── Decoder ────────────────────────────────────────────────────
        preds = self.decoder(
            H_init=H_list,
            icond2_fore=i2_fore,
            ecmwf_fore=e2_fore,
            static=static,
            s2s_edge_index=s2s_ei,
            s2s_edge_weight=s2s_ew,
            i2s_edge_index=i2s_ei,
            i2s_edge_attr=i2s_ea,
            e2s_edge_index=e2s_ei,
            e2s_edge_attr=e2s_ea,
            y_last_hist=y_last,
            target_mask=target_mask,
            teacher_forcing_targets=teacher_forcing_targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
            station_nwp_fore=station_nwp_fore,
            **dir_kwargs_dec,
        )

        return preds   # (N_target, T_fore)

    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def predict(self, data: HeteroData, target_mask: Tensor) -> Tensor:
        self.eval()
        return self(data, target_mask)
