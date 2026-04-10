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
        self.cfg   = config
        self.T_hist = config.history_length
        self.T_fore = config.forecast_horizon
        self.M      = config.station_meas_features
        self.target_feat_idx = config.target_feat_idx

        nwp_out_dim = config.nwp_out_dim
        edge_dim    = config.edge_input_dim()

        shared_kwargs = dict(
            icond2_dim=config.icond2_features_per_step,
            ecmwf_dim=config.ecmwf_features_per_step,
            nwp_out_dim=nwp_out_dim,
            static_dim=config.station_static_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            K=config.diffusion_K,
            dropout=config.dropout,
            nwp_heads=config.nwp_heads,
            edge_dim=edge_dim,
        )

        self.encoder = DCGRUEncoder(
            meas_dim=config.station_meas_features,
            **shared_kwargs,
        )
        self.decoder = DCGRUDecoder(
            forecast_horizon=config.forecast_horizon,
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
        meas      = x_station[:, :self.T_hist, :self.M]  # (N_s, T_hist, M)

        # icond2 / ecmwf sequences  (N_nwp, T, F_nwp)
        icond2_seq = data["icond2"].x                  # (N_i, T, I2)
        ecmwf_seq  = data["ecmwf"].x                   # (N_e, T, E2)

        # ── Edge indices and attributes ────────────────────────────────
        s2s_ei = data[s2s_key].edge_index
        s2s_ea = data[s2s_key].edge_attr
        s2s_ew = DCGRUCell.edge_weight_from_attr(s2s_ea)   # (E,)

        i2s_ei = data[i2s_key].edge_index
        i2s_ea = data[i2s_key].edge_attr
        e2s_ei = data[e2s_key].edge_index
        e2s_ea = data[e2s_key].edge_attr

        # ── Transpose sequences for loop: (N, T, F) → (T, N, F) ──────
        meas_seq     = meas.permute(1, 0, 2)              # (T_hist, N_s, M)
        i2_hist      = icond2_seq[:, :self.T_hist, :].permute(1, 0, 2)  # (T_hist, N_i, I2)
        e2_hist      = ecmwf_seq[:,  :self.T_hist, :].permute(1, 0, 2)  # (T_hist, N_e, E2)
        i2_fore      = icond2_seq[:, self.T_hist:, :].permute(1, 0, 2)  # (T_fore, N_i, I2)
        e2_fore      = ecmwf_seq[:,  self.T_hist:, :].permute(1, 0, 2)  # (T_fore, N_e, E2)

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
        )

        return preds   # (N_target, T_fore)

    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def predict(self, data: HeteroData, target_mask: Tensor) -> Tensor:
        self.eval()
        return self(data, target_mask)
