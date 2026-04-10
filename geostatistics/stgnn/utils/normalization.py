"""
Simple feature normalization utilities (mean/std, min/max).

These are thin wrappers that operate on numpy arrays and can be
serialised to disk as plain dicts — no sklearn dependency needed
at inference time.
"""
from __future__ import annotations

import numpy as np


class StandardScaler:
    """
    Fit a mean/std scaler on training data and apply to any array.

    Handles 1-D, 2-D, and masked (NaN-containing) arrays.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.eps = eps

    def fit(self, x: np.ndarray) -> "StandardScaler":
        """
        Fit on x.

        Args:
            x: (N, F) or (N,) array — NaNs are ignored.
        """
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Call fit() first."
        return (x - self.mean_) / (self.std_ + self.eps)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Call fit() first."
        return x * (self.std_ + self.eps) + self.mean_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state_dict(self) -> dict:
        return {"mean": self.mean_, "std": self.std_, "eps": self.eps}

    @classmethod
    def from_state_dict(cls, d: dict) -> "StandardScaler":
        obj = cls(eps=d["eps"])
        obj.mean_ = d["mean"]
        obj.std_ = d["std"]
        return obj


class MinMaxScaler:
    """Scale features to [0, 1]."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.eps = eps

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        self.min_ = np.nanmin(x, axis=0)
        self.max_ = np.nanmax(x, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.min_ is not None
        return (x - self.min_) / (self.max_ - self.min_ + self.eps)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self.min_ is not None
        return x * (self.max_ - self.min_ + self.eps) + self.min_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
