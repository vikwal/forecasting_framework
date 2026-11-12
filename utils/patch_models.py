"""TFT with a spatial encoder on NWP patches (Viktor 2026-11-12).

The known input carries the point features first and the patch channels after them, flattened row-major per channel:
``known = [point (n_point) | channel 0 (H*W) | channel 1 (H*W) | ...]``. A small CNN maps the patch of every time
step to an embedding (time is never collapsed); the TFT then sees ``[point | embedding]`` as known features. With
``t_kernel > 1`` the first convolution also spans neighbouring steps ((2+1)D style), so cloud advection between steps
can enter the embedding.
"""
import torch
from torch import nn

from .models import TFT


class PatchEncoder(nn.Module):
    def __init__(self, channels: int, size: int, emb: int, width: int = 32, t_kernel: int = 1, dropout: float = 0.1):
        super().__init__()
        self.t_kernel = t_kernel
        self.first = nn.Conv3d(channels, width // 2, (t_kernel, 3, 3), padding=(t_kernel // 2, 1, 1))
        self.body = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1), nn.GELU(),        # 16 -> 8
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),             # 8 -> 4
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(width * (size // 4) ** 2, emb))

    def forward(self, x):                                   # x (B, T, C, H, W) -> (B, T, emb)
        b, t = x.shape[:2]
        h = self.first(x.transpose(1, 2)).transpose(1, 2)   # Conv3d wants (B, C, T, H, W)
        return self.body(h.reshape(b * t, *h.shape[2:])).reshape(b, t, -1)


class PatchTFT(nn.Module):
    def __init__(self, n_point: int, channels: int, size: int, emb: int, tft_kwargs: dict, t_kernel: int = 1,
                 width: int = 32):
        super().__init__()
        self.n_point, self.channels, self.size = n_point, channels, size
        self.encoder = PatchEncoder(channels, size, emb, width, t_kernel, tft_kwargs.get('dropout', 0.1))
        self.tft = TFT(known_dim=n_point + emb, **tft_kwargs)

    def forward(self, observed, known, static=None):
        b, t = known.shape[:2]
        patch = known[..., self.n_point:].reshape(b, t, self.channels, self.size, self.size)
        z = self.encoder(patch)
        return self.tft(observed, torch.cat([known[..., :self.n_point], z], dim=-1), static)


def tft_kwargs(config: dict, hyperparameters: dict) -> dict:
    """Same TFT arguments as models.get_model, without known_dim."""
    fd = config['model']['feature_dim']
    qs = config['model'].get('tft', {}).get('quantiles') or [0.5]
    return dict(observed_dim=fd['observed_dim'], static_dim=fd['static_dim'],
                hidden_dim=hyperparameters.get('hidden_size', hyperparameters.get('hidden_dim', 32)),
                num_heads=hyperparameters.get('attention_head_size', hyperparameters.get('n_heads', 4)),
                lookback=config['model']['lookback'], horizon=config['model']['horizon'],
                dropout=hyperparameters.get('dropout', 0.1),
                num_lstm_layers=hyperparameters.get('num_lstm_layers', hyperparameters.get('n_lstm_layers', 1)),
                static_embedding_dim=hyperparameters.get('static_embedding_dim', None),
                rnn_type=config['model']['tft'].get('rnn_type', 'lstm'), num_quantiles=len(qs))
