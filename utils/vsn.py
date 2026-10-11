"""
Variable selection weights of a trained TFT (eval.vsn_weights in train_cl.py).

The TFT has three variable selection networks: static (static features), past (observed + known
features over the lookback) and future (known features over the horizon). Each returns a softmax
weight per input and sample/time step; the mean over all test samples and time steps is the share
the network gives each feature (sums to 1 per network). These are the TFT's own selection weights,
not a permutation importance: a feature can carry weight while being redundant with another one.
"""

import logging

import numpy as np
import pandas as pd
import torch

from . import preprocessing


def input_names(columns: list, features: dict) -> dict:
    """Feature names in the order of the TFT inputs ('observed', 'known', 'static') of one station,
    resolved from the frame columns as prepare_data_for_tft does."""
    known, observed = preprocessing.prepare_features_for_tft(
        cols=list(columns), known_future_cols=features.get('known') or [],
        observed_past_cols=features.get('observed') or [])
    return {'observed': observed, 'known': known, 'static': list(features.get('static') or [])}


def _names(names: list, n: int, kind: str) -> list:
    if len(names) == n:
        return names
    logging.warning(f"[vsn] {kind}: {n} inputs but {len(names)} names {names}; using indices")
    return [f'{kind}_{i}' for i in range(n)]


@torch.no_grad()
def mean_weights(model, X: dict, device, names: dict, batch_size: int = 1024) -> pd.DataFrame:
    """Mean VSN weight per feature over all samples and time steps of X (dict with 'observed',
    'known' and optionally 'static' arrays). Returns rows (network, feature, weight)."""
    net = getattr(model, '_orig_mod', model)          # torch.compile wrapper -> eager module
    was_training = net.training
    net.eval()
    sums, n_total = {}, 0
    n = len(X['known'])
    for i in range(0, n, batch_size):
        obs = torch.as_tensor(X['observed'][i:i + batch_size], dtype=torch.float32, device=device)
        known = torch.as_tensor(X['known'][i:i + batch_size], dtype=torch.float32, device=device)
        static = (torch.as_tensor(X['static'][i:i + batch_size], dtype=torch.float32, device=device)
                  if 'static' in X and np.size(X['static']) else None)
        _, att = net(obs, known, static, return_attention_weights=True)
        nb = known.shape[0]
        for key in ('static_weights', 'past_weights', 'future_weights'):
            w = att.get(key)
            if w is None:
                continue
            w = w.reshape(nb, -1, w.shape[-1]).mean(dim=1).sum(dim=0)   # sum over samples of per-sample means
            sums[key] = sums.get(key, 0) + w.double().cpu().numpy()
        n_total += nb
    if was_training:
        net.train()

    labels = {'static_weights': ('static', names['static']),
              'past_weights': ('past', names['observed'] + names['known']),
              'future_weights': ('future', names['known'])}
    rows = []
    for key, total in sums.items():
        network, feats = labels[key]
        for feat, w in zip(_names(feats, len(total), network), total / max(n_total, 1)):
            rows.append({'network': network, 'feature': feat, 'weight': float(w)})
    return pd.DataFrame(rows)
