"""Site-ID-embedding control model (reviewer-4 experiment).

Replaces the 7 physical static features by a learned nn.Embedding over the
station index, fed through the *identical* static pathway of the TFT
(static VSN with one input -> context GRNs). If this model matches the
physical-static model in-sample, the gain could come from any site
conditioning; the zero-shot comparison then shows that only physical
covariates transfer to unseen farms.

No forward() override: TFT.forward calls self.static_embed[i](static[:, i:i+1])
with a (batch, 1) float tensor — IndexEmbedding is a drop-in for that call.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.models import TFT


class IndexEmbedding(nn.Module):
    """Drop-in replacement for nn.Linear(1, emb_dim) on a float station index.

    In eval mode, indices registered as *unknown* (never trained) are imputed
    with the mean embedding of the known (trained) stations — the "best case"
    a site-ID model can offer for an unseen farm.
    """

    def __init__(self, num_stations: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_stations, emb_dim)
        # 1 = station was in the training set
        self.register_buffer('known_mask', torch.ones(num_stations, dtype=torch.bool))

    def set_known_ids(self, known_ids):
        mask = torch.zeros_like(self.known_mask)
        mask[list(known_ids)] = True
        self.known_mask.copy_(mask)

    def forward(self, x):  # x: (batch, 1) float index
        idx = x.round().long().squeeze(-1).clamp(0, self.emb.num_embeddings - 1)
        out = self.emb(idx)
        if not self.training and not bool(self.known_mask.all()):
            mean_known = self.emb.weight[self.known_mask].mean(dim=0)
            unknown = ~self.known_mask[idx]
            if unknown.any():
                out = out.clone()
                out[unknown] = mean_known
        return out


class SiteEmbeddingTFT(TFT):
    """TFT whose only static input is a learned per-station embedding."""

    def __init__(self, num_stations: int, observed_dim: int, known_dim: int,
                 hidden_dim: int, num_heads: int, lookback: int, horizon: int,
                 dropout: float = 0.1, num_lstm_layers: int = 1,
                 static_embedding_dim: int = None, num_quantiles: int = 1):
        super().__init__(
            observed_dim=observed_dim, known_dim=known_dim, static_dim=1,
            hidden_dim=hidden_dim, num_heads=num_heads, lookback=lookback,
            horizon=horizon, dropout=dropout, num_lstm_layers=num_lstm_layers,
            static_embedding_dim=static_embedding_dim,
            num_quantiles=num_quantiles)
        self.num_stations = num_stations
        # Replace the single per-feature Linear by the ID embedding; the
        # static VSN (one input -> weight 1) and context GRNs stay identical.
        self.static_embed = nn.ModuleList([
            IndexEmbedding(num_stations, self.static_embedding_dim)
        ])

    def set_known_ids(self, known_ids):
        self.static_embed[0].set_known_ids(known_ids)


def build_site_embedding_tft(config: dict, hyperparameters: dict,
                             num_stations: int, known_dim: int,
                             observed_dim: int = 1):
    return SiteEmbeddingTFT(
        num_stations=num_stations,
        observed_dim=observed_dim,
        known_dim=known_dim,
        hidden_dim=hyperparameters.get('hidden_dim', 40),
        num_heads=hyperparameters.get('n_heads', 8),
        lookback=config['model']['lookback'],
        horizon=config['model']['horizon'],
        dropout=hyperparameters.get('dropout', 0.1),
        num_lstm_layers=hyperparameters.get('num_lstm_layers',
                                            hyperparameters.get('n_lstm_layers', 1)),
        static_embedding_dim=hyperparameters.get('static_embedding_dim'),
        num_quantiles=1,
    )


def self_test():
    torch.manual_seed(0)
    model = SiteEmbeddingTFT(num_stations=10, observed_dim=1, known_dim=6,
                             hidden_dim=16, num_heads=2, lookback=48, horizon=48,
                             static_embedding_dim=32)
    obs = torch.randn(4, 48, 1)
    known = torch.randn(4, 96, 6)
    static = torch.tensor([[0.0], [1.0], [8.0], [9.0]])
    model.train()
    out = model(obs, known, static)
    assert out.shape == (4, 48), out.shape
    # unseen-id imputation: ids 8/9 unknown -> identical mean embedding output
    model.set_known_ids(range(8))
    model.eval()
    with torch.no_grad():
        emb = model.static_embed[0](static)
    assert torch.allclose(emb[2], emb[3]), 'unknown ids must share the mean embedding'
    assert not torch.allclose(emb[0], emb[2]), 'known ids must keep their own embedding'
    # gradient flows into the embedding table
    model.train()
    loss = model(obs, known, static).sum()
    loss.backward()
    assert model.static_embed[0].emb.weight.grad is not None
    print('site_embedding self-test passed.')


if __name__ == '__main__':
    self_test()
