"""Auflösung von ``data.freq`` gegen das native NWP-Ausgaberaster.

Ersetzt acht kopierte Lookup-Tabellen der Form::

    _freq_h_map = {"1h": 1.0, "30min": 0.5, "15min": 0.25, ...}
    freq_h = _freq_h_map.get(freq, 1.0)

Der stille ``.get(freq, 1.0)``-Fallback ist der eigentliche Grund für dieses Modul:
ein Tippfehler oder ein nicht gelisteter Wert (``'2h'``, und in
``evaluate_reference.py`` sogar ``'15min'``) wurde kommentarlos zu einer Stunde.
Das Training lief dann mit einem Lead-Raster, das nicht zur Config passt.

Zusätzlich sind die beiden ICON-D2-Ebenen unterschiedlich fein:

===========  ==================  =========================
Ebene        Use-Case            natives ``forecasttime``
===========  ==================  =========================
ML           wind                stündlich (49 Schritte)
SL           solar               15-minütig (193 Schritte)
===========  ==================  =========================

``freq: '15min'`` auf einer Wind-Config ist deshalb keine feinere Vorhersage,
sondern ein leerer Datensatz — ML kann nur jeden vierten Lead füllen, und die
Vollständigkeitsprüfung verwirft anschließend jeden Lauf.  Das wird hier
abgefangen statt später als ``RuntimeError: expected a non-empty list of Tensors``.
"""
from __future__ import annotations

import pandas as pd

#: Natives Ausgaberaster je Use-Case, in Minuten.
NATIVE_STEP_MIN = {
    "wind": 60,    # ICON-D2 ML
    "solar": 15,   # ICON-D2 SL
}

#: Länge eines ICON-D2-Vorhersagelaufs in Stunden.
FORECAST_HORIZON_H = 48


def freq_to_hours(freq: str, use_case: str = "wind") -> float:
    """``data.freq`` → Schrittweite in Stunden.

    Args:
        freq: Config-Wert, z. B. ``'1h'``, ``'30min'``, ``'15min'``.
        use_case: ``'wind'`` (ICON-D2 ML) oder ``'solar'`` (ICON-D2 SL).

    Raises:
        ValueError: wenn ``freq`` feiner als das native Raster ist, gegen dieses
            versetzt liegt oder den 48-h-Horizont nicht ganzzahlig teilt.
    """
    native = NATIVE_STEP_MIN.get(str(use_case).lower())
    if native is None:
        raise ValueError(
            f"Unbekannter data.use_case='{use_case}' — erwartet: "
            f"{sorted(NATIVE_STEP_MIN)}"
        )

    minutes = pd.Timedelta(freq).total_seconds() / 60.0
    if minutes <= 0 or minutes != int(minutes):
        raise ValueError(f"data.freq='{freq}' ist keine ganzzahlige Minutenangabe.")
    minutes = int(minutes)

    if minutes % native != 0:
        raise ValueError(
            f"data.freq='{freq}' ({minutes} min) ist kein Vielfaches des nativen "
            f"ICON-D2-Rasters für use_case='{use_case}' ({native} min). "
            + ("ML liefert nur Stundenschritte; für Viertelstunden braucht es "
               "die SL-Daten (use_case: solar)." if native == 60 else
               "Zulässig sind '15min', '30min', '45min', '1h', '2h', …")
        )

    total_min = FORECAST_HORIZON_H * 60
    if total_min % minutes != 0:
        raise ValueError(
            f"data.freq='{freq}' teilt den {FORECAST_HORIZON_H}-h-Vorhersagehorizont "
            "nicht ganzzahlig."
        )
    return minutes / 60.0


def n_leads(freq: str, use_case: str = "wind") -> int:
    """Anzahl Lead-Indizes je Lauf (48 bei ``'1h'``, 192 bei ``'15min'``)."""
    return int(round(FORECAST_HORIZON_H / freq_to_hours(freq, use_case)))
