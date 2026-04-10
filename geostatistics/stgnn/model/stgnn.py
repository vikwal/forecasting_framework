"""
Full STGNN model: Encoder → ST-Block Processor → Decoder.

Data flow (shapes for temporal_encoding ∈ {"gru", "cnn"})
----------------------------------------------------------
  Encoder
    station.x (N_s, T, M+I2+E2) ──temporal enc──▶ (N_s, T, d_temp)
                                  ──+ static──────▶ (N_s, T, d_temp+S)
                                  ──per-step MLP──▶ (N_s, T, latent)
    Same for icond2 (N_i, T, I2) and ecmwf (N_e, T, E2).

  Processor (L ST-blocks, each block):
    Step 1 – temporal Conv1d:    (N_*, T, d) → (N_*, T, d)  [residual]
    Step 2 – spatial GATv2 (time-synchronous):
      reshape (N, T, d) → (N*T, d),  expand edges T×,
      GATv2 i2s + e2s + s2s → (N_s*T, d),  reshape → (N_s, T, d)  [residual+norm]

  Decoder
    station[:, T_hist:, :]  →  (N_target, T_fore, d)
    per-step Linear(d→1)    →  (N_target, forecast_horizon)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from ..config import ModelConfig
from .decoder import Decoder
from .encoder import HeteroEncoder
from .processor import HeteroProcessor


class STGNN(nn.Module):
    """
    Spatiotemporal Graph Neural Network for inductive wind speed forecasting.

    Architecture:  Encode → Process (ST-blocks) → Decode

    The model is **inductive**: no per-station learned embeddings.
    Generalisation to unseen stations relies on:
      - Geographic edge features (distance + bearing [+ altitude diff])
      - NWP grid point message passing (GATv2, time-synchronous)
      - Station type indicator (1=observed neighbour, 0=target/unobserved)

    Parameters
    ----------
    config : ModelConfig
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config    = config
        self.encoder   = HeteroEncoder(config)
        self.processor = HeteroProcessor(config)
        self.decoder   = Decoder(config)

    def forward(self, data: HeteroData, target_mask: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        data : HeteroData
            Graph with populated node features (.x, .static) and edge features
            (.edge_index, .edge_attr) for all node/edge types.

            Node feature shapes (gru/cnn encoding):
              data["station"].x      — (N_s, T, M+I2+E2)
              data["station"].static — (N_s, S_station)
              data["icond2"].x       — (N_i, T, I2)
              data["icond2"].static  — (N_i, S_icond2)
              data["ecmwf"].x        — (N_e, T, E2)
              data["ecmwf"].static   — (N_e, S_ecmwf)

        target_mask : (N_stations,) boolean tensor
            True for target (unobserved) stations, False for neighbours.

        Returns
        -------
        predictions : (N_target, forecast_horizon) float tensor
        """
        i2s_key = ("icond2", "informs", "station")
        e2s_key = ("ecmwf",  "informs", "station")
        s2s_key = ("station", "near",   "station")

        # 1. Encode all node types → {type: (N, T, latent)}
        node_emb = self.encoder(data)

        # 2. Collect raw edge indices and attributes
        edge_index_dict = {
            s2s_key: data[s2s_key].edge_index,
            i2s_key: data[i2s_key].edge_index,
            e2s_key: data[e2s_key].edge_index,
        }
        edge_attr_dict = {
            s2s_key: data[s2s_key].edge_attr,
            i2s_key: data[i2s_key].edge_attr,
            e2s_key: data[e2s_key].edge_attr,
        }

        # 3. Run ST-block processor
        node_emb = self.processor(node_emb, edge_index_dict, edge_attr_dict)

        # 4. Decode target stations
        predictions = self.decoder(node_emb["station"], target_mask)

        return predictions

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def predict(self, data: HeteroData, target_mask: Tensor) -> Tensor:
        """Inference-mode forward pass (no_grad wrapper)."""
        self.eval()
        return self(data, target_mask)
