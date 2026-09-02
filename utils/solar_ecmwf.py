"""
solar_ecmwf.py — ECMWF-HRES-Prognosen als zweite NWP-Quelle für den Solar-Use-Case.

Pendant zum ECMWF-Zweig der Wind-Pipeline
(``preprocessing._fetch_ecmwf_data_from_split_parquets``), aber für die
Strahlungsfelder.  Wird von :func:`solar.preprocess_solar_icond2` aufgerufen und
hängt ``ecmwf_*``-Spalten an den ICON-D2-Datensatz.

Datenquelle
-----------
``{data.ecmwf_solar_path}/<lat>_<lat_frac>_<lon>_<lon_frac>_solar_sl.parquet``

| | |
|---|---|
| Gitterpunkte | 759, flaches Verzeichnis |
| Läufe | 00 und 12 UTC |
| ``forecasttime`` | 0 … 57, **ganze Stunden** (int) |
| Spalten | ``ssrd``, ``fdir``, ``ssrc``, ``cdir``, ``tisr``, ``ssrdc``, ``tp``, ``t_2m``, ``td_2m``, ``sp``, ``fal``, ``grid_lat``, ``grid_lon``, ``valid_time`` |

Drei Unterschiede zu ICON-D2 SL, die hier bewusst behandelt sind:

1. **Dateinamen sind lat-first** — ``54_2600_10_1100`` → lat 54.26, lon 10.11.
   Das ist die *umgekehrte* Konvention zu den ICON-SL-Dateien (dort lon-first).
   Bestätigt über die Spalten ``grid_lat``/``grid_lon`` in der Datei selbst.
2. **Die Strahlungsfelder sind bereits dekumuliert und in W/m².**  Die in
   ``docs/predict_solar.md`` genannte Formel ``(F[i] − F[i-1]) / 3600`` ist beim
   Erzeugen der Parquets schon angewandt worden.  Gegenprobe: ``tisr`` erreicht
   1124 W/m², die rechnerische TOA-Einstrahlung für lat 54.26 am 1. Juli liegt bei
   1164 W/m².  Wären es noch J/m²-Summen, stünde dort das 3600-fache.
3. **``forecasttime = 0`` ist bei allen Strahlungsfeldern NaN** (kein vorangehendes
   Intervall).  Die Momentanfelder (``t_2m``, ``sp``, …) haben dort einen Wert —
   genau die Aufteilung, die auch die Lead-Zuordnung unten verwendet.

Zeitliche Zuordnung
-------------------
Wie bei ICON-D2 sind die Strahlungsfelder **Intervallmittel, die auf
``forecasttime`` enden**, der Rest Momentanwerte.  Bei linksbündigen Ziel-Labels
(Wert bei *T* beschreibt [T, T+freq)) gilt für den Stundenversatz *h* eines
Zeitstempels gegenüber dem ECMWF-Lauf:

* akkumuliert:  ``ecmwf_ft = floor(h) + 1``   — das Intervall (ft−1, ft] überdeckt [T, T+freq)
* Momentanwert: ``ecmwf_ft = floor(h)``       — Zustand zu Beginn der Stunde

Die Zuordnung ICON-Lauf → ECMWF-Lauf folgt derselben Regel wie im Wind-Pfad:
**der jüngste ECMWF-Lauf ≤ ICON-``starttime``**, also 12 UTC ab ICON-Stunde 12,
sonst 00 UTC desselben Tages.

Bei ``freq`` < 1 h wird der Stundenwert über die Teilschritte konstant gehalten.
Feiner kann HRES nicht — eine Interpolation würde eine Auflösung vortäuschen, die
das Modell nicht hergibt.  Über die Stunde bleibt die Energiebilanz erhalten.
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from geopy.distance import geodesic

logger = logging.getLogger(__name__)


#: ECMWF-Felder, deren Wert das auf ``forecasttime`` **endende** Intervall mittelt.
ECMWF_ACCUMULATED_COLS = frozenset({
    'ssrd', 'fdir', 'ssrc', 'cdir', 'tisr', 'ssrdc', 'tp',
})

#: Abgeleitete ECMWF-Features → benötigte Rohspalten.
_DERIVED_FROM_ECMWF = {
    'ecmwf_ghi': ('ssrd',),
    'ecmwf_bhi': ('fdir',),
    'ecmwf_dhi': ('ssrd', 'fdir'),
    'ecmwf_bhi_clearsky': ('cdir',),
    'ecmwf_toa': ('tisr',),
}

#: Abgeleitete ECMWF-Features, die zusätzlich den Sonnenstand brauchen.
#: Sie entstehen erst in :func:`add_ecmwf_geometry_features` **nach** dem Merge —
#: ``solar_zenith``/``ghi_clearsky``/``dni_clearsky`` hängen an der Station, nicht
#: am ECMWF-Gitterpunkt, und liegen im ECMWF-Frame noch gar nicht vor.
ECMWF_GEOMETRY_DERIVED = ('ecmwf_dni', 'ecmwf_kt', 'ecmwf_kd')

#: Rohspalten, die unverändert (nur mit Präfix) durchgereicht werden.
_ECMWF_PASSTHROUGH = ('t_2m', 'td_2m', 'sp', 'fal', 'tp', 'tisr', 'cdir', 'ssrc')

_ECMWF_SOLAR_RE = re.compile(r'^(-?\d+)_(\d+)_(-?\d+)_(\d+)_solar_sl$')


def parse_ecmwf_solar_latlon(stem: str) -> tuple[float, float]:
    """``'54_2600_10_1100_solar_sl'`` → ``(lat, lon) = (54.26, 10.11)``.

    **lat-first** — anders als bei den ICON-D2-SL-Dateien.
    """
    m = _ECMWF_SOLAR_RE.match(stem)
    if not m:
        raise ValueError(f"Kein ECMWF-Solar-Dateiname: '{stem}'")
    lat = float(f"{m.group(1)}.{m.group(2)}")
    lon = float(f"{m.group(3)}.{m.group(4)}")
    return lat, lon


@lru_cache(maxsize=8)
def scan_ecmwf_solar_dir(path: str) -> tuple[tuple[str, float, float], ...]:
    """Alle Gitterpunkte als ``(stem, lat, lon)``; gecached je Verzeichnis."""
    if not os.path.isdir(path):
        return ()
    out = []
    for fname in os.listdir(path):
        if not fname.endswith('.parquet'):
            continue
        stem = fname[:-len('.parquet')]
        try:
            lat, lon = parse_ecmwf_solar_latlon(stem)
        except ValueError:
            continue
        out.append((stem, lat, lon))
    return tuple(sorted(out))


def select_nearest_ecmwf_points(path: str, station_lat: float, station_lon: float,
                                k: int) -> list[tuple[str, float, float, float]]:
    """Die ``k`` nächstgelegenen Gitterpunkte als ``(stem, lat, lon, dist_km)``."""
    grid = scan_ecmwf_solar_dir(path)
    if not grid:
        return []
    scored = [(stem, lat, lon, geodesic((station_lat, station_lon), (lat, lon)).kilometers)
              for stem, lat, lon in grid]
    scored.sort(key=lambda x: x[3])
    return scored[:max(1, k)]


def _required_ecmwf_columns(features: list[str]) -> list[str]:
    """Welche Rohspalten für die angeforderten ``ecmwf_*``-Features nötig sind."""
    needed: set[str] = set()
    for feat in features:
        if feat in _DERIVED_FROM_ECMWF:
            needed.update(_DERIVED_FROM_ECMWF[feat])
        elif feat in ECMWF_GEOMETRY_DERIVED:
            needed.update(('ssrd', 'fdir'))
        elif feat.startswith('ecmwf_'):
            needed.add(feat[len('ecmwf_'):])
        else:
            needed.add(feat)
    return sorted(needed)


def _is_accumulated_ecmwf(raw_col: str) -> bool:
    return raw_col in ECMWF_ACCUMULATED_COLS


def geometry_intermediates(features: list[str]) -> list[str]:
    """Zwischenspalten, die bis **nach** dem Merge durchgereicht werden müssen.

    ``ecmwf_kt`` & Co. werden erst dort berechnet, brauchen aber ``ecmwf_ghi``
    bzw. ``ecmwf_dhi`` als Eingang. Ohne diese Liste fallen die Zwischenschritte
    heraus, sobald sie nicht *zusätzlich* selbst angefordert sind: sowohl
    :func:`load_ecmwf_solar_for_station` als auch :func:`merge_ecmwf` behalten
    ausschliesslich Spalten, die zu einem Eintrag in ``features`` passen.
    """
    if not any(f in ECMWF_GEOMETRY_DERIVED for f in features):
        return []
    need = ['ecmwf_ghi']
    if any(f in ('ecmwf_dni', 'ecmwf_kd') for f in features):
        need.append('ecmwf_dhi')
    return [c for c in need if c not in features]


def _add_ecmwf_derived(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Aus den Rohspalten ableitbare ``ecmwf_*``-Felder berechnen (in-place)."""
    def col(name):
        return df[name] if name in df.columns else None

    ssrd, fdir = col('ssrd'), col('fdir')
    if 'ecmwf_ghi' in features and ssrd is not None:
        df['ecmwf_ghi'] = ssrd.clip(lower=0)
    if 'ecmwf_bhi' in features and fdir is not None:
        df['ecmwf_bhi'] = fdir.clip(lower=0)
    if 'ecmwf_dhi' in features and ssrd is not None and fdir is not None:
        df['ecmwf_dhi'] = (ssrd - fdir).clip(lower=0)
    if 'ecmwf_bhi_clearsky' in features and col('cdir') is not None:
        df['ecmwf_bhi_clearsky'] = col('cdir').clip(lower=0)
    if 'ecmwf_toa' in features and col('tisr') is not None:
        df['ecmwf_toa'] = col('tisr').clip(lower=0)
    # Zwischenschritte für die geometrieabhängigen Ableitungen
    if any(f in features for f in ECMWF_GEOMETRY_DERIVED):
        if 'ecmwf_ghi' not in df.columns and ssrd is not None:
            df['ecmwf_ghi'] = ssrd.clip(lower=0)
        if 'ecmwf_bhi' not in df.columns and fdir is not None:
            df['ecmwf_bhi'] = fdir.clip(lower=0)
        if 'ecmwf_dhi' not in df.columns and ssrd is not None and fdir is not None:
            df['ecmwf_dhi'] = (ssrd - fdir).clip(lower=0)
    for raw in _ECMWF_PASSTHROUGH:
        name = f'ecmwf_{raw}'
        if name in features and raw in df.columns and name not in df.columns:
            df[name] = df[raw]
    return df


def load_ecmwf_solar_for_station(ecmwf_path: str,
                                 station_lat: float,
                                 station_lon: float,
                                 features: list[str],
                                 next_n_grid_points: int = 1,
                                 starttime_min: pd.Timestamp | None = None,
                                 starttime_max: pd.Timestamp | None = None) -> pd.DataFrame:
    """ECMWF-Solarprognosen einer Station laden.

    Returns
    -------
    DataFrame mit ``starttime`` (ECMWF-Lauf, UTC), ``forecasttime`` (ganze Stunden),
    getrennten Spalten je Feldtyp und Gitterpunkt-Rang (``<feature>_<rang>``),
    zusätzlich der Spalte ``_acc`` … siehe :func:`merge_ecmwf_into`, die daraus die
    Zuordnung auf das Zielraster übernimmt.
    """
    raw_cols = _required_ecmwf_columns(features)
    nearest = select_nearest_ecmwf_points(ecmwf_path, station_lat, station_lon,
                                          next_n_grid_points)
    if not nearest:
        raise FileNotFoundError(
            f"Keine ECMWF-Solar-Gitterpunkte unter {ecmwf_path} gefunden."
        )

    merged = None
    for rank, (stem, _lat, _lon, _dist) in enumerate(nearest, start=1):
        fpath = os.path.join(ecmwf_path, f'{stem}.parquet')
        try:
            df = pd.read_parquet(fpath)
        except Exception as exc:
            logger.warning("ECMWF-Solar-Datei nicht lesbar: %s (%s)", fpath, exc)
            continue
        if df.empty:
            continue

        df['starttime'] = pd.to_datetime(df['starttime'], utc=True)
        if starttime_min is not None:
            df = df[df['starttime'] >= starttime_min]
        if starttime_max is not None:
            df = df[df['starttime'] <= starttime_max]
        if df.empty:
            continue

        missing = [c for c in raw_cols if c not in df.columns]
        if missing:
            logger.warning("ECMWF-Solar-Datei %s ohne Spalten %s", fpath, missing)
        df = df.drop_duplicates(subset=['starttime', 'forecasttime'], keep='last')
        df['forecasttime'] = df['forecasttime'].astype(int)
        df = _add_ecmwf_derived(df, features)

        # Zwischenschritte der geometrieabhaengigen Groessen mitnehmen, auch wenn
        # sie nicht selbst angefordert sind - sie werden erst nach dem Merge gebraucht.
        present = [f for f in features + geometry_intermediates(features)
                   if f in df.columns]
        if not present:
            continue
        frame = df[['starttime', 'forecasttime'] + present].copy()
        frame = frame.rename(columns={c: f'{c}_{rank}' for c in present})
        merged = frame if merged is None else merged.merge(
            frame, on=['starttime', 'forecasttime'], how='outer'
        )

    if merged is None:
        raise ValueError("Alle ECMWF-Solar-Dateien waren leer oder unlesbar.")
    return merged.sort_values(['starttime', 'forecasttime']).reset_index(drop=True)


def merge_ecmwf(df: pd.DataFrame,
                     ecmwf: pd.DataFrame,
                     features: list[str],
                     step_min: int) -> pd.DataFrame:
    """ECMWF-Spalten an einen ICON-D2-Datensatz hängen.

    ``df`` braucht die Spalten ``starttime`` (ICON-Lauf) und ``timestamp``.

    Die Zuordnung ICON-Lauf → ECMWF-Lauf ist dieselbe wie im Wind-Pfad: der jüngste
    ECMWF-Lauf ≤ ICON-``starttime``.  Innerhalb des Laufs werden akkumulierte und
    Momentanfelder getrennt zugeordnet (``floor(h)+1`` bzw. ``floor(h)``), weil sie
    verschiedene Zeitintervalle beschreiben — dieselbe Unterscheidung wie bei
    ICON-D2.  Bei ``step_min`` < 60 gilt der Stundenwert über die Teilschritte fort.
    """
    if ecmwf.empty:
        return df

    out = df.reset_index(drop=True).copy()

    icon_st = pd.to_datetime(out['starttime'], utc=True)
    run_hour = np.where(icon_st.dt.hour >= 12, 12, 0)
    ecmwf_st = icon_st.dt.normalize() + pd.to_timedelta(run_hour, unit='h')
    if ecmwf_st.dt.tz is None:
        ecmwf_st = ecmwf_st.dt.tz_localize('UTC')
    # Als Series zuweisen, nicht ueber .values: to_numpy() auf einer tz-aware Spalte
    # liefert datetime64[ns] ohne Zeitzone, und der Merge bricht dann mit
    # "trying to merge on datetime64[ns] and datetime64[ns, UTC]" ab.
    out['_ecmwf_st'] = ecmwf_st

    # Versatz des Zielzeitstempels gegenueber dem ECMWF-Lauf, in ganzen Minuten
    offset_min = ((pd.to_datetime(out['timestamp'], utc=True) - ecmwf_st)
                  .dt.total_seconds() / 60).round().astype('int64')
    hours_floor = np.floor_divide(offset_min.to_numpy(), 60)

    def _is_acc(feat: str) -> bool:
        raw = feat[len('ecmwf_'):] if feat.startswith('ecmwf_') else feat
        return (_is_accumulated_ecmwf(raw)
                or feat in _DERIVED_FROM_ECMWF
                or feat in ECMWF_GEOMETRY_DERIVED)

    feats_all = list(features) + geometry_intermediates(features)
    acc_feats = [f for f in feats_all if _is_acc(f)]
    inst_feats = [f for f in feats_all if not _is_acc(f)]

    for feats, lead in ((acc_feats, hours_floor + 1), (inst_feats, hours_floor)):
        cols = [c for c in ecmwf.columns
                if any(c == f or c.startswith(f + '_') for f in feats)]
        if not cols:
            continue
        part = ecmwf[['starttime', 'forecasttime'] + cols].rename(
            columns={'starttime': '_ecmwf_st', 'forecasttime': '_ecmwf_ft'}
        ).drop_duplicates(subset=['_ecmwf_st', '_ecmwf_ft'], keep='last')
        part['_ecmwf_st'] = pd.to_datetime(part['_ecmwf_st'], utc=True)
        # lead positionsweise zuweisen: der Left-Join oben erhaelt Zeilenzahl und
        # -reihenfolge, weil die rechte Seite auf den Schluesseln eindeutig ist.
        out['_ecmwf_ft'] = lead.astype('int64')
        out = out.merge(part, on=['_ecmwf_st', '_ecmwf_ft'], how='left')
        out = out.drop(columns=['_ecmwf_ft'])

    return out.drop(columns=['_ecmwf_st'])


#: Stationsbezogene Spalten, die :func:`add_ecmwf_geometry_features` voraussetzt.
_GEOMETRY_INPUT_COLS = ('solar_zenith', 'ghi_clearsky', 'dni_clearsky')


def add_ecmwf_geometry_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """``ecmwf_dni`` / ``ecmwf_kt`` / ``ecmwf_kd`` je Gitterpunkt-Rang (in-place).

    Pendant zu ``solar._add_nwp_geometry_features`` und bewusst **zeilengleich** in
    den Konventionen, damit ``ecmwf_kt`` und ``kt_nwp`` dieselbe Groesse in
    derselben Skala messen und das Modell sie direkt gegeneinander abwaegen kann:

    * ``dni`` ueber ``pvlib.irradiance.dni()`` mit Deckelung an der Clear-Sky-DNI,
    * ``kt`` als Clear-Sky-Index gegen ``ghi_clearsky`` — **nicht** als klassischer
      Clearness-Index gegen die TOA-Einstrahlung. ECMWF liefert mit ``ssrdc`` zwar
      eine eigene Clear-Sky-GHI; sie als Referenz zu nehmen waere in sich
      stimmiger, wuerde ``ecmwf_kt`` aber gegen eine andere Bezugsgroesse messen
      als ``kt_nwp`` und den Vergleich der beiden Quellen entwerten.
    * ``kd`` als Diffusanteil ``dhi/ghi``, auf [0, 1] begrenzt.

    Ein Unterschied zum ICON-Pendant ist Absicht: wo ECMWF keine Deckung hat,
    bleibt das Ergebnis ``NaN`` statt 0. ``pvlib.irradiance.dni()`` liefert dort
    NaN, und das ``fillna(0.0)`` in ``derive_irradiance_components`` wuerde eine
    Nullstrahlung behaupten, die nur eine Datenluecke ist. Die ECMWF-Abdeckung
    waechst noch (Stand Aug 2026 ab Juli 2023), deshalb ist die Maske hier
    wesentlich — ``dropna()`` weiter unten in der Pipeline verwirft solche Zeilen.

    Zwischenspalten aus :func:`geometry_intermediates`, die nicht selbst
    angefordert waren, werden am Ende entfernt.
    """
    wanted = [f for f in features if f in ECMWF_GEOMETRY_DERIVED]
    if not wanted:
        return df

    # Spaeter Import: solar.py importiert dieses Modul selbst (dort ebenfalls lazy).
    from . import solar as _solar

    missing = [c for c in _GEOMETRY_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"ECMWF-Geometrieableitung {wanted} braucht die Spalten {missing}. "
            "add_ecmwf_geometry_features() gehoert hinter die Sonnengeometrie."
        )

    ranks = sorted({
        col.rsplit('_', 1)[1] for col in df.columns
        if col.startswith('ecmwf_ghi_') and col.rsplit('_', 1)[1].isdigit()
    }, key=int)

    for rank in ranks:
        ghi_col, dhi_col = f'ecmwf_ghi_{rank}', f'ecmwf_dhi_{rank}'
        if ghi_col not in df.columns:
            continue
        gap = df[ghi_col].isna()

        if 'ecmwf_dni' in wanted and dhi_col in df.columns:
            comp = _solar.derive_irradiance_components(
                ghi=df[ghi_col], dhi=df[dhi_col], zenith=df['solar_zenith'],
                dni_clearsky=df['dni_clearsky'],
            )
            df[f'ecmwf_dni_{rank}'] = comp['dni'].mask(gap)
        if 'ecmwf_kt' in wanted:
            df[f'ecmwf_kt_{rank}'] = _solar.clearsky_index(
                df[ghi_col], df['ghi_clearsky']
            ).mask(gap)
        if 'ecmwf_kd' in wanted and dhi_col in df.columns:
            kd = np.where(df[ghi_col] > 0,
                          (df[dhi_col] / df[ghi_col]).clip(0, 1), 0.0)
            df[f'ecmwf_kd_{rank}'] = pd.Series(kd, index=df.index).mask(gap)

    drop = [f'{stem}_{rank}' for stem in geometry_intermediates(features)
            for rank in ranks if f'{stem}_{rank}' in df.columns]
    return df.drop(columns=drop) if drop else df
