"""
Encoder module: maps raw node features to a shared latent sequence.

All temporal encoders now output the *full temporal sequence* ``(N, T, d)``
instead of a single summary vector.  This preserves the time axis for the
downstream ST-block processor.

Supported temporal encoding strategies
--------------------------------------
"gru"  — GRU over the temporal dimension, return full ``out`` sequence
"cnn"  — Causal dilated 1-D convolutions (no global average pool)

The "flat" strategy is no longer supported in the ST-block architecture.

Static features (lat, lon, alt, type_indicator) are broadcast across the time
axis and concatenated with the temporal output *before* the per-timestep
node MLP, giving every forecast step full access to geographic context.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

from ..config import ModelConfig
from .mlp import MLP


# ---------------------------------------------------------------------------
# Causal Conv1d primitive (used by CausalDilatedCNNEncoder)
# ---------------------------------------------------------------------------

class _CausalConv1d(nn.Module):
    """
    Conv1d with left-only (causal) padding so that output at time *t* cannot
    see inputs from t+1 onwards.

    Parameters
    ----------
    in_ch, out_ch : channel dimensions
    kernel_size   : convolution kernel width
    dilation      : dilation factor; receptive field grows as (k-1)*d + 1
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            dilation=dilation, padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, T) — Conv1d format
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


# ---------------------------------------------------------------------------
# Temporal encoders
# ---------------------------------------------------------------------------

class GRUTemporalEncoder(nn.Module):
    """
    Process the temporal dimension with a GRU; return the *full output*
    sequence so that the time axis is preserved.

    Parameters
    ----------
    input_dim_per_step : features per timestep
    hidden_dim         : GRU hidden dimension (= output dim per step)
    num_layers         : stacked GRU layers
    dropout            : dropout between GRU layers (only if num_layers > 1)
    """

    def __init__(
        self,
        input_dim_per_step: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim_per_step,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (N, T, F_per_step)

        Returns:
            (N, T, hidden_dim)  — full sequence, NOT just the last hidden state
        """
        out, _ = self.gru(x)   # out: (N, T, hidden_dim)
        return out


class CausalDilatedCNNEncoder(nn.Module):
    """
    Two causal dilated Conv1d layers with dilation 1 and 2.
    No global average pool — the full temporal sequence is returned.

    Effective receptive field with kernel_size=5:
      layer 0 (dilation=1): 5 steps
      layer 1 (dilation=2): 5 + 2*4 = 13 steps

    Parameters
    ----------
    input_dim_per_step : features per timestep
    channels           : output channels per conv layer
    kernel_size        : kernel width (should be odd)
    dropout            : dropout after each layer's activation
    """

    def __init__(
        self,
        input_dim_per_step: int,
        channels: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dilations = [1, 2]
        layers: list[nn.Module] = []
        in_ch = input_dim_per_step
        for dil in dilations:
            layers.extend([
                _CausalConv1d(in_ch, channels, kernel_size, dilation=dil),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_ch = channels
        self.net = nn.Sequential(*layers)
        self.output_dim = channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (N, T, F_per_step)

        Returns:
            (N, T, channels)  — full temporal sequence preserved
        """
        x = x.permute(0, 2, 1)   # (N, F, T) — Conv1d format
        x = self.net(x)            # (N, channels, T)
        return x.permute(0, 2, 1)  # (N, T, channels)


# ---------------------------------------------------------------------------
# Heterogeneous encoder
# ---------------------------------------------------------------------------

class HeteroEncoder(nn.Module):
    """
    Encodes raw features of all node types into a shared latent sequence
    ``(N, T, latent_dim)``.

    Processing per node type:
      1. Temporal encoder (GRU or CausalDilatedCNN) → (N, T, d_temp)
      2. Broadcast static features to (N, T, S) and concatenate
      3. Per-timestep node MLP: applied as (N*T, d_temp+S) → (N*T, latent)
         then reshaped back to (N, T, latent)

    Edge encoder MLPs are removed — raw edge attributes are passed directly
    to the GATv2Conv layers in the processor.

    Parameters
    ----------
    config : ModelConfig
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.cfg = config
        te = config.temporal_encoding
        if te == "flat":
            raise ValueError(
                "temporal_encoding='flat' is not supported by the ST-block "
                "architecture.  Use 'gru' or 'cnn'."
            )

        latent    = config.latent_dim
        mlp_layers = config.encoder_mlp_layers
        drop      = config.dropout

        i2_fts   = config.icond2_features_per_step
        ecmwf_fts = config.ecmwf_features_per_step
        meas_fts = config.station_meas_features
        station_fts_per_step = meas_fts + i2_fts + ecmwf_fts

        if te == "gru":
            self.temporal_enc_station = GRUTemporalEncoder(
                station_fts_per_step,
                config.gru_hidden_dim,
                config.gru_num_layers,
                drop,
            )
            self.temporal_enc_icond2 = GRUTemporalEncoder(
                i2_fts, config.gru_hidden_dim, config.gru_num_layers, drop
            )
            self.temporal_enc_ecmwf = GRUTemporalEncoder(
                ecmwf_fts, config.gru_hidden_dim, config.gru_num_layers, drop
            )
            d_temp_s = config.gru_hidden_dim
            d_temp_n = config.gru_hidden_dim

        else:  # "cnn"
            self.temporal_enc_station = CausalDilatedCNNEncoder(
                station_fts_per_step,
                config.cnn_channels,
                config.cnn_kernel_size,
                drop,
            )
            self.temporal_enc_icond2 = CausalDilatedCNNEncoder(
                i2_fts, config.cnn_channels, config.cnn_kernel_size, drop
            )
            self.temporal_enc_ecmwf = CausalDilatedCNNEncoder(
                ecmwf_fts, config.cnn_channels, config.cnn_kernel_size, drop
            )
            d_temp_s = config.cnn_channels
            d_temp_n = config.cnn_channels

        self.temporal_encoding = te

        # Per-timestep node MLPs.
        # Input = temporal_output_per_step + static_features
        self.station_encoder = MLP(
            d_temp_s + config.station_static_features,
            latent, latent, mlp_layers, drop,
        )
        self.icond2_encoder = MLP(
            d_temp_n + config.icond2_static_features,
            latent, latent, mlp_layers, drop,
        )
        self.ecmwf_encoder = MLP(
            d_temp_n + config.ecmwf_static_features,
            latent, latent, mlp_layers, drop,
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _encode_node_type(
        self,
        x_temporal: Tensor,   # (N, T, F_per_step)
        static: Tensor,       # (N, S)
        temporal_enc: nn.Module,
        node_mlp: nn.Module,
    ) -> Tensor:
        """
        Returns
        -------
        (N, T, latent_dim)
        """
        t_out = temporal_enc(x_temporal)       # (N, T, d_temp)
        N, T, _ = t_out.shape
        static_exp = static.unsqueeze(1).expand(-1, T, -1)   # (N, T, S)
        x_in = torch.cat([t_out, static_exp], dim=-1)        # (N, T, d_temp+S)
        # Apply MLP independently at each timestep via a single batched call
        emb = node_mlp(x_in.reshape(N * T, -1))              # (N*T, latent)
        return emb.reshape(N, T, -1)                         # (N, T, latent)

    # ------------------------------------------------------------------

    def forward(self, data: HeteroData) -> dict[str, Tensor]:
        """
        Encode all node types.

        For "gru"/"cnn" encoding, node .x tensors must be (N, T, F_per_step)
        3-D tensors as populated by the sampler.

        Returns
        -------
        node_emb : dict mapping node type → (N_type, T, latent_dim) tensor
        """
        station_emb = self._encode_node_type(
            data["station"].x,
            data["station"].static,
            self.temporal_enc_station,
            self.station_encoder,
        )
        icond2_emb = self._encode_node_type(
            data["icond2"].x,
            data["icond2"].static,
            self.temporal_enc_icond2,
            self.icond2_encoder,
        )
        ecmwf_emb = self._encode_node_type(
            data["ecmwf"].x,
            data["ecmwf"].static,
            self.temporal_enc_ecmwf,
            self.ecmwf_encoder,
        )
        return {
            "station": station_emb,   # (N_s, T, latent)
            "icond2":  icond2_emb,    # (N_i, T, latent)
            "ecmwf":   ecmwf_emb,     # (N_e, T, latent)
        }
