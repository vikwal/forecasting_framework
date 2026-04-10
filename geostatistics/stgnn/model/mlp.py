"""
Shared MLP building block used throughout the STGNN architecture.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """
    Multi-layer perceptron with LayerNorm + SiLU activations and optional dropout.

    Architecture per hidden layer: Linear → LayerNorm → SiLU → Dropout
    Final layer: Linear only (no normalisation or activation).

    Parameters
    ----------
    input_dim :  input feature dimension
    hidden_dim : width of hidden layers
    output_dim : output feature dimension
    num_layers : total number of Linear layers (≥ 1)
    dropout :    dropout probability applied after each hidden activation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "num_layers must be at least 1"

        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_d))
                layers.append(nn.SiLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
