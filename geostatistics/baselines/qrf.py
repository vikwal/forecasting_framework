"""
geostatistics/baselines/qrf.py — QRF-local fit / predict.

Deterministisch (Nutzerentscheidung D1 / F3): ``sklearn.ensemble.
RandomForestRegressor``, Punktvorhersage über den bedingten Mittelwert. In
Prosa/Tabellenköpfen "QRF, ausgewertet über den bedingten Mittelwert" bzw.
Kurzform "RF" (F3); Datei-/Arm-/Studienname bleiben ``qrf_local``.

Kein Feature-Scaling (Spezifikation 3.3): ein Entscheidungsbaum ist invariant
gegen jede streng monotone Spaltentransformation.

Ein Wald für alle Lead-Zeiten (Spezifikation 3.4): ``horizon`` ist ein
Merkmal, nicht 48 getrennte Modelle — Kostengründe und Fairness gegenüber den
Graphmodellen (ein Parametersatz für alle Leads). ``--per-lead`` ist als
optionaler, NICHT gefahrener Pfad implementiert (Spezifikation 3.4 letzter
Satz).
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger("baselines.qrf")

# Feste, nicht gesuchte Werte (Spezifikation 5.2)
FIXED_PARAMS = dict(
    bootstrap=True,
    criterion="squared_error",
)


def subsample_rows(
    X: np.ndarray, y: np.ndarray, n_fit_rows: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduzierbare Teilstichprobe (Spezifikation 5.3): ``rng =
    np.random.default_rng(seed)``, ``idx = rng.choice(n_rows, N, replace=False)``.
    ``n_fit_rows <= 0`` oder ``>= len(y)`` heißt: kein Subsampling.

    Returns (X_sub, y_sub, idx) — idx wird für die Provenienz protokolliert.
    """
    n_rows = len(y)
    if n_fit_rows <= 0 or n_fit_rows >= n_rows:
        idx = np.arange(n_rows)
        return X, y, idx
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, n_fit_rows, replace=False)
    idx.sort()
    return X[idx], y[idx], idx


def fit(
    X: np.ndarray, y: np.ndarray,
    n_estimators: int = 200,
    min_samples_leaf: float = 2e-5,
    max_features: float = 0.33,
    max_depth: int = 30,
    random_state: int = 20260810,
    n_jobs: int = 32,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        **FIXED_PARAMS,
    )
    model.fit(X, y)
    return model


def predict(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    return model.predict(X).astype(np.float32)


def pooled_rmse(preds: np.ndarray, y: np.ndarray) -> float:
    """Gepooltes, unskaliertes RMSE ueber ALLE Zeilen (Stationen x Paare x
    Leads) in m/s — die HPO-Objective-Konvention (Spezifikation 1.8/5.1/8.5),
    NICHT das Mittel der Stations-RMSEs. Einzige Formel-Stelle, von
    ``hpo_qrf.py`` UND der Verifikationssuite (V8) importiert, damit ein Test
    nicht bloss dieselbe Kopie der Formel gegen sich selbst prueft."""
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean((p - g) ** 2)))


def fit_per_lead(
    X: np.ndarray, y: np.ndarray, horizon_col: int, horizons: np.ndarray,
    **fit_kwargs,
) -> dict[int, RandomForestRegressor]:
    """Optionaler Pfad (``--per-lead``, Spezifikation 3.4): 48 getrennte
    Wälder statt einem gepoolten mit ``horizon`` als Merkmal. NICHT gefahren
    in dieser Phase — Kostenfaktor 48 (Spezifikation 3.4 Grund 1). Die
    ``horizon``-Spalte wird pro Teilmodell entfernt, da sie darin konstant ist.
    """
    models: dict[int, RandomForestRegressor] = {}
    keep = np.ones(X.shape[1], dtype=bool)
    keep[horizon_col] = False
    for h in np.unique(horizons):
        mask = horizons == h
        models[int(h)] = fit(X[mask][:, keep], y[mask], **fit_kwargs)
    return models


def predict_per_lead(
    models: dict[int, RandomForestRegressor], X: np.ndarray, horizon_col: int,
    horizons: np.ndarray,
) -> np.ndarray:
    keep = np.ones(X.shape[1], dtype=bool)
    keep[horizon_col] = False
    preds = np.full(len(horizons), np.nan, dtype=np.float32)
    for h in np.unique(horizons):
        mask = horizons == h
        model = models.get(int(h))
        if model is not None:
            preds[mask] = predict(model, X[mask][:, keep])
    return preds
