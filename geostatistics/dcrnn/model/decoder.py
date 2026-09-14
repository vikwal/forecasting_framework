"""
Seq2Seq Autoregressive Decoder with NWP node attention.

NWP messages are computed per forecast step using the current last-layer
hidden state H[-1] as the GATv2 attention query, matching the encoder.

Per forecast step t ∈ [0, T_fore):

  1. NWP attention:
       nwp_msg_t = GATv2(icond2_fore_t, ecmwf_fore_t → station | H[-1])

  2. DCGRU step:
       input_t = cat( y_{t-1}, nwp_msg_t, static )   → (N_s, 1 + nwp_out_dim + S)
       H[l]    = DCGRUCell_l( input_t, H[l] )

  3. Output projection:
       ŷ_t = Linear( H[-1] )               → (N_s,)
       → collect target stations → (N_target,)

Teacher forcing:
  During training, y_{t-1} for target stations is replaced by the
  ground-truth value with probability `teacher_forcing_ratio`.
  Neighbour stations always use their true last value (no forcing needed
  since their measurements are observed).
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch import Tensor

from .dcgru_cell import DCGRUCell
from .nwp_attention import NWPAttentionLayer


class DCGRUDecoder(nn.Module):
    """
    Multi-layer autoregressive DCGRU decoder with NWP attention.

    Parameters
    ----------
    nwp_out_dim      : output dimension of NWPAttentionLayer
    static_dim       : S — time-invariant station features
    hidden_dim       : DCGRU hidden dimension (must match encoder)
    forecast_horizon : T_fore
    num_layers       : stacked DCGRU cells (must match encoder)
    K                : diffusion hops
    dropout          : dropout in cells
    icond2_dim       : I2
    ecmwf_dim        : E2
    nwp_heads        : attention heads
    edge_dim         : edge_attr dimension
    nwp_aggregation  : "attention" (default) | "idw" | "idw_alt" — see NWPAttentionLayer
    idw_p            : inverse-distance power (nwp_aggregation in {"idw","idw_alt"} only)
    alpha_alt        : vertical/horizontal trade-off (nwp_aggregation="idw_alt" only)
    alt_col          : edge_attr column holding altitude_diff (nwp_aggregation="idw_alt" only)
    icond2_max_dist_km : physical-km normaliser, icond2 distance column (nwp_aggregation="idw_alt" only)
    ecmwf_max_dist_km  : physical-km normaliser, ecmwf distance column (nwp_aggregation="idw_alt" only)
    """

    def __init__(
        self,
        nwp_out_dim: int,
        static_dim: int,
        hidden_dim: int,
        forecast_horizon: int,
        num_layers: int = 2,
        K: int = 2,
        dropout: float = 0.0,
        icond2_dim: int = 3,
        ecmwf_dim: int = 3,
        nwp_heads: int = 4,
        edge_dim: int = 3,
        nwp_nodes: bool = True,
        station_nwp_dim: int = 0,
        nwp_aggregation: str = "attention",
        idw_p: float = 2.0,
        alpha_alt: float = 10.0,
        alt_col: int | None = None,
        icond2_max_dist_km: float = 0.0,
        ecmwf_max_dist_km: float = 0.0,
        n_targets: int = 1,
        n_geo: int = 0,
    ) -> None:
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.num_layers       = num_layers
        self.nwp_nodes        = nwp_nodes
        self.nwp_out_dim      = nwp_out_dim
        # n_targets > 1: der autoregressive Kanal fuehrt alle Zielgroessen, und
        # die Ausgabeprojektion gibt sie gemeinsam aus. Bei n_targets == 1
        # bleiben Modulaufbau und Tensorformen exakt wie zuvor — der Wind-Pfad
        # ist davon unberuehrt.
        self.n_targets        = int(n_targets)
        #: Sonnengeometrie-Kanaele, die je Dekodierschritt dazukommen.
        self.n_geo            = int(n_geo)

        if nwp_nodes and nwp_out_dim > 0:
            self.nwp_attn = NWPAttentionLayer(
                icond2_dim=icond2_dim,
                ecmwf_dim=ecmwf_dim,
                station_dim=hidden_dim,
                nwp_out_dim=nwp_out_dim,
                heads=nwp_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                aggregation=nwp_aggregation,
                idw_p=idw_p,
                alpha_alt=alpha_alt,
                alt_col=alt_col,
                icond2_max_dist_km=icond2_max_dist_km,
                ecmwf_max_dist_km=ecmwf_max_dist_km,
            )

        if nwp_nodes:
            gru_input_dim = self.n_targets + self.n_geo + nwp_out_dim + static_dim
        else:
            # station_nwp_dim = I2 + E2 from station.x (nearest grid point)
            gru_input_dim = self.n_targets + self.n_geo + station_nwp_dim + static_dim

        self.cells = nn.ModuleList([
            DCGRUCell(
                gru_input_dim if i == 0 else hidden_dim,
                hidden_dim, K, dropout,
            )
            for i in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, self.n_targets)

    def forward(
        self,
        H_init: list[Tensor],               # from encoder, len=num_layers
        icond2_fore: Tensor,                # (T_fore, N_i, I2)
        ecmwf_fore: Tensor,                 # (T_fore, N_e, E2)
        static: Tensor,                     # (N_s, S)
        s2s_edge_index: Tensor,
        s2s_edge_weight: Tensor,
        i2s_edge_index: Tensor,
        i2s_edge_attr: Tensor,
        e2s_edge_index: Tensor,
        e2s_edge_attr: Tensor,
        y_last_hist: Tensor,                # (N_s,) or (N_s, n_targets) — last history value
        target_mask: Tensor,                # (N_s,) bool
        teacher_forcing_targets: Tensor | None = None,  # (N_target, T_fore[, n_targets])
        teacher_forcing_ratio: float = 0.5,
        wind_dir_fore: Tensor | None = None,    # (T_fore, N_s) degrees, met. convention
        s2s_dist_norm: Tensor | None = None,    # (E_s2s,) — normalised distance
        s2s_bearing: Tensor | None = None,      # (E_s2s,) — azimuth src→dst in radians
        station_nwp_fore: Tensor | None = None, # (T_fore, N_s, I2+E2) when nwp_nodes=False
        geo_fore: Tensor | None = None,         # (N_s, T_fore, G) Sonnengeometrie
    ) -> Tensor:
        """
        Returns
        -------
        preds : (N_target, T_fore) bei n_targets == 1,
                (N_target, T_fore, n_targets) sonst.
        """
        T_fore = self.forecast_horizon
        device = icond2_fore.device
        use_tf  = teacher_forcing_targets is not None and self.training
        use_dir = wind_dir_fore is not None

        N_s = static.size(0)
        H = [h.clone() for h in H_init]
        # Intern immer (N_s, n_targets) — der Einziel-Fall kommt als (N_s,) an.
        y_prev = y_last_hist.clone()
        if y_prev.dim() == 1:
            y_prev = y_prev.unsqueeze(-1)     # (N_s, 1)
        tf_targets = teacher_forcing_targets
        if tf_targets is not None and tf_targets.dim() == 2:
            tf_targets = tf_targets.unsqueeze(-1)   # (N_target, T_fore, 1)
        preds_list: list[Tensor] = []

        for t in range(T_fore):
            if self.nwp_nodes and self.nwp_out_dim > 0:
                # GATv2 attention conditioned on current last-layer hidden state
                nwp_msg = self.nwp_attn.forward(
                    icond2_fore[t], ecmwf_fore[t], H[-1],
                    i2s_edge_index, i2s_edge_attr,
                    e2s_edge_index, e2s_edge_attr,
                )                            # (N_s, nwp_out_dim)
            elif self.nwp_nodes:
                # nwp_injection bypass: no NWP at all
                nwp_msg = torch.zeros(N_s, 0, device=device)
            else:
                # nwp_nodes=False: nearest-grid-point NWP from station.x
                nwp_msg = station_nwp_fore[t]  # (N_s, I2+E2)

            ew_t = (
                DCGRUCell.directional_edge_weight(
                    wind_dir_fore[t], s2s_edge_index[0], s2s_dist_norm, s2s_bearing
                )
                if use_dir else s2s_edge_weight
            )

            # --- Build decoder input ---
            teile_t = [y_prev]
            if self.n_geo and geo_fore is not None:
                teile_t.append(geo_fore[:, t, :])          # (N_s, G)
            teile_t += [nwp_msg, static]
            input_t = torch.cat(teile_t, dim=-1)   # (N_s, n_targets + G + nwp_dim + S)

            # --- DCGRU step ---
            x_t = input_t
            for l, cell in enumerate(self.cells):
                H[l] = cell(x_t, H[l], s2s_edge_index, ew_t)
                x_t  = H[l]

            # --- Output projection ---
            y_hat = self.out_proj(H[-1])                # (N_s, n_targets)
            preds_list.append(y_hat[target_mask])       # (N_target, n_targets)

            # --- Update y_prev for next step ---
            y_next = y_hat.detach().clone()
            if use_tf and random.random() < teacher_forcing_ratio:
                y_next[target_mask] = tf_targets[:, t, :]
            y_prev = y_next

        preds = torch.stack(preds_list, dim=1)      # (N_target, T_fore, n_targets)
        # Einziel-Fall auf die alte Form zurueck, damit Trainer, Auswertung und
        # alle gespeicherten Wind-Ergebnisse unveraendert weiterlaufen.
        return preds.squeeze(-1) if self.n_targets == 1 else preds
