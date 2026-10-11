"""Client-drift diagnostics of the federated strategies (``fl.drift_metrics: true``).

At every aggregation step the server holds one vector per participating client: the batch
gradient (FedGradient) or the weight update ``w_client - w_global`` of the round (FedAvg /
FedAvgM / FedAdam). Flattened over the aggregated parameters (row-wise embedding tables such
as the park id are left out), the vectors give

  * per client: cosine similarity to the aggregated (weighted mean) vector and the vector norm,
  * per step: mean pairwise cosine similarity between the clients and the alignment
    ``||mean|| / mean(||v_i||)`` in [0, 1] (1 = all clients point the same way, 0 = they cancel),

averaged over the steps of a round (FedGradient: one step per batch, FedAvg: one per round).
With client-side evaluation the validation RMSE of every client and round is kept as well (the
"which client does the global model fail for this round" view of Nachtigall et al. 2024).
Related measures: gradient diversity (Yin et al. 2018), update cosine similarity under
pre-training (Nguyen et al., "Where to Begin?", ICLR 2023).

Results: ``history['drift_clients']`` (round, client, cos_mean, norm_mean, n_steps, val_rmse)
and ``history['drift_rounds']`` (round, pairwise_cos, alignment, mean_norm, n_steps).
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import torch


def flatten(tensors: Dict[str, torch.Tensor], keys: Sequence[str]) -> torch.Tensor:
    """One float32 CPU vector of the given keys in the given order (missing keys are skipped)."""
    parts = [tensors[k].detach().reshape(-1).float().cpu() for k in keys
             if k in tensors and torch.is_floating_point(tensors[k])]
    return torch.cat(parts) if parts else torch.zeros(0)


def step_stats(vectors: Sequence[torch.Tensor], weights: Sequence[float]) -> Dict[str, Any]:
    """Cosine of every vector to the weighted mean, the norms, the mean pairwise cosine and
    the alignment ``||mean|| / sum_i w_i ||v_i||`` (weights normalised to one)."""
    V = torch.stack([v.float() for v in vectors])                      # (n, d)
    w = torch.as_tensor(list(weights), dtype=V.dtype)
    w = w / w.sum()
    mean = (w[:, None] * V).sum(0)
    norms = V.norm(dim=1)
    mean_norm = mean.norm()
    cos = (V @ mean) / (norms * mean_norm).clamp_min(1e-12)
    n = V.shape[0]
    if n > 1:
        Vn = V / norms.clamp_min(1e-12)[:, None]
        G = Vn @ Vn.T
        pairwise = float((G.sum() - G.diagonal().sum()) / (n * (n - 1)))
    else:
        pairwise = float('nan')
    alignment = float(mean_norm / (w * norms).sum().clamp_min(1e-12))
    return {'cos': cos.tolist(), 'norm': norms.tolist(), 'pairwise_cos': pairwise,
            'alignment': alignment, 'mean_norm': float(mean_norm)}


class DriftLog:
    """Accumulates the step statistics of a round and keeps the finished rounds."""

    def __init__(self, keys: Sequence[str]):
        self.keys = list(keys)
        self.clients: List[Dict[str, Any]] = []
        self.rounds: List[Dict[str, Any]] = []
        self._reset()

    def _reset(self) -> None:
        self._client_acc: Dict[Any, List[float]] = {}        # client -> [sum cos, sum norm, n]
        self._round_acc = [0.0, 0.0, 0]                      # sum alignment, sum mean norm, n steps
        self._pair_acc = [0.0, 0]                            # sum pairwise cos, n steps with >= 2 clients
        self._val: Dict[Any, float] = {}

    def flatten(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        return flatten(tensors, self.keys)

    def add(self, client_ids: Sequence[Any], vectors: Sequence[torch.Tensor],
            weights: Sequence[float]) -> Dict[str, Any]:
        """One aggregation step: the vectors of the participating clients and their
        aggregation weights."""
        s = step_stats(vectors, weights)
        for cid, c, nrm in zip(client_ids, s['cos'], s['norm']):
            acc = self._client_acc.setdefault(cid, [0.0, 0.0, 0])
            acc[0] += c
            acc[1] += nrm
            acc[2] += 1
        self._round_acc[0] += s['alignment']
        self._round_acc[1] += s['mean_norm']
        self._round_acc[2] += 1
        if len(vectors) > 1:
            self._pair_acc[0] += s['pairwise_cos']
            self._pair_acc[1] += 1
        return s

    def add_val(self, rmse_by_client: Dict[Any, float]) -> None:
        self._val.update({k: float(v) for k, v in rmse_by_client.items()})

    def end_round(self, round_num: int) -> None:
        for cid in sorted(set(self._client_acc) | set(self._val), key=str):
            acc = self._client_acc.get(cid)
            self.clients.append({'round': int(round_num), 'client': cid,
                                 'cos_mean': acc[0] / acc[2] if acc else float('nan'),
                                 'norm_mean': acc[1] / acc[2] if acc else float('nan'),
                                 'n_steps': acc[2] if acc else 0,
                                 'val_rmse': self._val.get(cid, float('nan'))})
        n, m = self._round_acc[2], self._pair_acc[1]
        self.rounds.append({'round': int(round_num),
                            'pairwise_cos': self._pair_acc[0] / m if m else float('nan'),
                            'alignment': self._round_acc[0] / n if n else float('nan'),
                            'mean_norm': self._round_acc[1] / n if n else float('nan'),
                            'n_steps': n})
        self._reset()

    def frames(self):
        cols_c = ['round', 'client', 'cos_mean', 'norm_mean', 'n_steps', 'val_rmse']
        cols_r = ['round', 'pairwise_cos', 'alignment', 'mean_norm', 'n_steps']
        return (pd.DataFrame(self.clients, columns=cols_c), pd.DataFrame(self.rounds, columns=cols_r))

    def to_dict(self) -> Dict[str, Any]:
        """Finished rounds only (a checkpoint is written at the end of a round)."""
        return {'keys': list(self.keys), 'clients': list(self.clients), 'rounds': list(self.rounds)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DriftLog':
        obj = cls(d['keys'])
        obj.clients = list(d.get('clients', []))
        obj.rounds = list(d.get('rounds', []))
        return obj


def trainable_keys(model: torch.nn.Module, skip: Optional[Iterable[str]] = None) -> List[str]:
    """Names of the trainable parameters in model order, minus ``skip`` (row-wise tables)."""
    skip = set(skip or ())
    return [n for n, p in model.named_parameters() if p.requires_grad and n not in skip]
