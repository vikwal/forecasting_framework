"""
nwp_baseline.py — Auswahl der ICON-D2-Feature-Spalte, gegen die Skill_NWP gerechnet wird.

Die Baseline ist die rohe NWP-Prognose *derselben physikalischen Größe* wie das
Vorhersageziel. Bis zum Solar-Use-Case war das implizit immer die
10-m-Windgeschwindigkeit; dieselbe Auswahllogik stand fünfmal kopiert in
train_mtgnn / train_wavenet / get_test_results_mtgnn / get_test_results_wavenet /
evaluate_reference, mit einem stillen ``, 0)``-Fallback auf das erste Feature.
Für Solar wäre dieser Fallback zufällig richtig (``ghi_nwp`` steht meist vorn) —
also genau die Sorte Zufall, die man nicht in einer Metrik haben will.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Bevorzugte ICON-D2-Featurenamen je Zielgröße, absteigend nach Passgenauigkeit.
_PREFERRED: dict[str, tuple[str, ...]] = {
    "wind_speed": ("wind_speed_10m", "wind_speed"),
    "ghi":        ("ghi_nwp",),
    "dhi":        ("dhi_nwp",),
    "bhi":        ("bhi_nwp",),
    "dni":        ("dni_nwp", "bhi_nwp"),
    "kt":         ("kt_nwp", "ghi_nwp"),
    "kd":         ("kd_nwp", "dhi_nwp"),
}


def nwp_baseline_feature_idx(icond2_features: list[str],
                             target_col: str = "wind_speed") -> int:
    """Index der NWP-Baselinespalte in ``icond2_features``.

    Sucht zuerst nach exakten Namen aus ``_PREFERRED[target_col]``, dann nach einem
    Teilstring-Treffer, und fällt zuletzt mit einer Warnung auf Index 0 zurück.
    """
    if not icond2_features:
        raise ValueError("icond2_features ist leer — keine NWP-Baseline möglich.")

    preferred = _PREFERRED.get(target_col, (target_col,))

    for name in preferred:
        for i, feat in enumerate(icond2_features):
            if feat == name:
                return i

    for name in preferred:
        for i, feat in enumerate(icond2_features):
            if name in feat:
                logger.warning(
                    "Kein exaktes NWP-Baselinefeature %s für target_col='%s' gefunden — "
                    "verwende '%s' (Index %d).", list(preferred), target_col, feat, i,
                )
                return i

    logger.warning(
        "Kein NWP-Baselinefeature für target_col='%s' in %s gefunden — "
        "Skill_NWP fällt auf '%s' (Index 0) zurück und ist damit nicht aussagekräftig.",
        target_col, icond2_features, icond2_features[0],
    )
    return 0
