"""
solar.py — Preprocessing-Pipeline für den Solar-Irradiance-Use-Case.

Pendant zu ``preprocessing.preprocess_synth_wind_icond2()``: verbindet
DWD-Stationsmessungen (Global-/Diffusstrahlung) mit ICON-D2-Surface-Level-Prognosen
und liefert denselben Datenkontrakt wie die Wind-Pipeline —
MultiIndex ``['starttime', 'forecasttime', 'timestamp']``, gefilterte Feature-Spalten,
``df.attrs['nwp_nearest_label']``.

Datenquellen
------------
Messungen : ``{data.path}/Station_<id>.parquet``
    10-min-Auflösung, Spalten ``ghi``, ``dhi`` (DWD ``GS_10``/``DS_10``) in
    **J/cm² je Messintervall**, zusätzlich ``temperature_2m``, ``wind_speed``,
    ``precipitation_rate``, ``precipitation_duration``.
    Von 204 Stationen führen nur 93 überhaupt Strahlungsdaten.

NWP : ``{data.nwp_path}/SL/{forecast_hour}/<lon>_<lat>_SL.parquet``
    15-min-Auflösung — ``forecasttime`` 0.0 … 48.0 in fraktionalen Stunden, also
    193 Schritte je Lauf.  Die frühesten Läufe (bis ~2023-08-08, rund 2 %) sind nur
    stündlich abgelegt und fallen bei ``freq`` < 1 h aus der Vollständigkeitsprüfung.

Zwei Konventionen weichen von den ICON-D2-**ML**-Dateien (Wind) ab und sind hier
bewusst explizit behandelt:

1. **Dateinamen sind lon-first.**  ML heißt ``<lat>_<lon>_ML.parquet``
   (``52_9057_12_9151`` → lat 52.9057), SL dagegen ``<lon>_<lat>_SL.parquet``
   (``10_0000_47_8000`` → lon 10.0, lat 47.8).  Auch die Spalten *innerhalb* der
   SL-Datei sind vertauscht: ``longitude`` enthält die Breite, ``latitude`` die Länge.
   Deshalb werden die Koordinaten ausschließlich aus dem Dateinamen gelesen.
2. **SL ist flach.**  Es gibt kein ``SL/06/<station_id>/``-Unterverzeichnis wie bei ML;
   alle 1218 Gitterpunkte liegen direkt in ``SL/06/``.

Zeitliche Zuordnung
-------------------
``aswdir_s``/``aswdifd_s`` sind **Intervallmittel, die auf ``forecasttime`` enden**
(verifiziert: Mittel über ft ∈ (1, 2] entspricht exakt
``aswdifd_s_avg@2 · 2 − aswdifd_s_avg@1 · 1``).  Die übrigen SL-Felder
(``t_2m``, ``clct``, ``alb_rad``, ``u_10m``, …) sind Momentanwerte.

Die DWD-Rohzeitstempel markieren das Intervall**ende**
(``params.measurement_time_label: 'right'``, Default) und werden deshalb vor dem
Resampling um ein Messintervall zurückgeschoben.  Aggregiert wird dann mit
``closed='left', label='left'`` — der Wert bei Zeitstempel *T* beschreibt also das
Intervall [T, T+freq).  Damit gilt:

* akkumulierte Felder:  ``lead_idx = ceil(ft / step) - 1``   (Intervall (ft-step, ft])
* Momentanwerte:        ``lead_idx = floor(ft / step)``      (Zustand zu Beginn von [T, T+step))

und einheitlich ``timestamp = starttime + lead_idx · step``.  Beide Zweige werden
getrennt aggregiert und anschließend auf ``(starttime, lead_idx)`` zusammengeführt.
Ein gemeinsames ``floor``/``+ft``-Mapping (wie in der Wind-Pipeline, wo alle Felder
Momentanwerte sind) würde die Strahlung um eine volle Stunde verschieben.

Zeitliche Auflösung (``data.freq``)
-----------------------------------
SL liefert **Viertelstundenschritte**, ML (Wind) nur Stundenschritte — ``freq`` feiner
als 1 h ist deshalb ausschließlich im Solar-Zweig möglich.  :func:`resolve_freq`
prüft ``freq`` gegen dieses native Raster und gibt ``(step_h, n_leads)``;
``'1h'`` → 48 Leads, ``'30min'`` → 96, ``'15min'`` → 192.

Auf der Messseite ist ``freq='15min'`` **nicht** durch einfaches Resampling zu haben:
die Rohwerte liegen im 10-min-Raster und nesten nicht in 15-min-Intervalle.
:func:`resample_interval_mean` mittelt deshalb flächengewichtet über das gemeinsame
5-min-Feinraster.  Für ``'1h'`` (ganzzahliges Vielfaches von 10 min) bleibt es beim
arithmetischen Mittel, die Ergebnisse ändern sich also nicht.
"""
from __future__ import annotations

import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import pandas as pd
import pvlib
from geopy.distance import geodesic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature-Katalog
# ---------------------------------------------------------------------------

#: SL-Felder, deren Wert das auf ``forecasttime`` **endende** Intervall mittelt.
ACCUMULATED_SL_COLS = frozenset({
    'aswdir_s', 'aswdifd_s',
})

#: SL-Felder, die das Mittel **seit Vorhersagebeginn** [0, ft] tragen, nicht das
#: Intervall (ft−step, ft].  Verifiziert: das kumulative Mittel von ``aswdifd_s``
#: bis ft ist auf 4 Nachkommastellen gleich ``aswdifd_s_avg`` bei ft.
#:
#: Für diese Spalten passt **keine** der beiden Lead-Zuordnungen — weder
#: ``ceil`` (Intervallende) noch ``floor`` (Momentanwert).  Sie als Feature
#: durchzureichen hieße, ein über den Lauf laufendes Mittel als Wert eines
#: einzelnen Zeitschritts zu labeln.  Die dekumulierte Fassung steht ohnehin
#: in derselben Datei, deshalb werden sie abgelehnt statt still umgedeutet.
RUNNING_MEAN_SL_COLS = frozenset({
    'aswdir_s_avg', 'aswdifd_s_avg',
})

#: Laufmittel → dekumulierte Entsprechung, für die Fehlermeldung.
_RUNNING_MEAN_REPLACEMENT = {
    'aswdir_s_avg': 'aswdir_s',
    'aswdifd_s_avg': 'aswdifd_s',
}

#: Abgeleitete NWP-Features, die allein aus SL-Spalten berechenbar sind
#: (vor dem Merge, je Gitterpunkt).
_DERIVED_FROM_SL = {
    'ghi_nwp': ('aswdir_s', 'aswdifd_s'),
    'dhi_nwp': ('aswdifd_s',),
    'bhi_nwp': ('aswdir_s',),
    'wind_speed_nwp': ('u_10m', 'v_10m'),
}

#: Abgeleitete NWP-Features, die zusätzlich Sonnenstand/Clear-Sky brauchen
#: (nach dem Merge, je Gitterpunkt-Rang).
_DERIVED_FROM_GEOMETRY = ('dni_nwp', 'kt_nwp', 'kd_nwp')

#: Stationsbezogene Sonnenstands-/Zeitfeatures (kein Gitterpunkt-Suffix).
SOLAR_GEOMETRY_FEATURES = (
    'solar_zenith', 'solar_zenith_cos', 'solar_azimuth_sin', 'solar_azimuth_cos',
    'airmass', 'dni_extra', 'ghi_clearsky', 'dni_clearsky', 'dhi_clearsky',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
)

#: Aus den Messungen ableitbare Zielgrößen.
DERIVED_TARGETS = ('bhi', 'dni', 'kt', 'kd')

#: Zenitwinkel, ab dem DNI definitionsgemäß auf 0 gesetzt wird.
_ZENITH_ZERO_DNI = 88.0


# ---------------------------------------------------------------------------
# Zeitliche Auflösung
# ---------------------------------------------------------------------------

#: Natives Ausgaberaster der ICON-D2-**SL**-Läufe (nur die Strahlungsfelder; der
#: Rest ist stündlich, s. :func:`_fill_sub_hourly`).  Die Wind-Seite (ML) ist
#: durchgehend stündlich — ``data.freq`` ist deshalb nur im Solar-Zweig feiner
#: als 1 h sinnvoll.
NWP_NATIVE_STEP_MIN = 15

#: Abtastraster der DWD-Stationsmessungen.
MEASUREMENT_STEP_MIN = 10

#: Feinstes Raster, in das **beide** Quellen exakt nesten: kgV(10, 15) = 30 min.
#: Alles darunter erfordert auf genau einer Seite eine flächengewichtete
#: Umverteilung — bei ``'15min'`` auf der Messseite, bei ``'10min'`` auf der
#: NWP-Seite.  Das ist keine Einstellungssache, sondern folgt aus den beiden
#: Rastern.
EXACT_STEP_MIN = 30

#: Feinste überhaupt darstellbare Schrittweite: ggT(10, 15) = 5 min.  Darunter
#: hat keine der beiden Quellen Information.
_MIN_STEP_MIN = 5

#: Länge eines ICON-D2-Vorhersagelaufs in Stunden.
FORECAST_HORIZON_H = 48


def resolution_report(freq: str) -> dict:
    """Welche Seite muss für ``freq`` genähert werden?

    Returns
    -------
    dict mit ``minutes``, ``measurement_exact``, ``nwp_exact`` und ``approximated``
    (``'none'`` | ``'measurement'`` | ``'nwp'``).
    """
    minutes = int(round(pd.Timedelta(freq).total_seconds() / 60.0))
    meas_exact = minutes % MEASUREMENT_STEP_MIN == 0
    nwp_exact = minutes % NWP_NATIVE_STEP_MIN == 0
    if meas_exact and nwp_exact:
        approx = 'none'
    elif nwp_exact:
        approx = 'measurement'      # z. B. 15 min: 10-min-Messung wird umverteilt
    elif meas_exact:
        approx = 'nwp'              # z. B. 10 min: 15-min-Prognose wird umverteilt
    else:
        approx = 'both'             # z. B. 5 min: beide Seiten müssen umverteilt werden
    return {'minutes': minutes, 'measurement_exact': meas_exact,
            'nwp_exact': nwp_exact, 'approximated': approx}


def resolve_freq(freq: str, warn: bool = True) -> tuple[float, int]:
    """Prüft ``data.freq`` gegen die beiden nativen Raster.

    Zulässig ist jedes ganzzahlige Vielfache von 5 min (= ggT(10, 15)), das den
    48-h-Horizont ganzzahlig teilt.  Exakt ohne Umverteilung sind nur Vielfache
    von 30 min (= kgV(10, 15)); bei allen anderen wird geloggt, welche Seite
    genähert wird.

    Returns
    -------
    (step_h, n_leads)
        Schrittweite in Stunden und Anzahl Lead-Indizes je Lauf
        (48 bei ``'1h'``, 96 bei ``'30min'``, 192 bei ``'15min'``, 288 bei ``'10min'``).
    """
    minutes = pd.Timedelta(freq).total_seconds() / 60.0
    if minutes <= 0 or minutes != int(minutes):
        raise ValueError(f"data.freq='{freq}' ist keine ganzzahlige Minutenangabe.")
    minutes = int(minutes)

    if minutes % _MIN_STEP_MIN != 0:
        raise ValueError(
            f"data.freq='{freq}' ({minutes} min) ist kein Vielfaches von "
            f"{_MIN_STEP_MIN} min = ggT(Messraster {MEASUREMENT_STEP_MIN} min, "
            f"ICON-D2-SL {NWP_NATIVE_STEP_MIN} min). Feiner hat keine der beiden "
            "Quellen Information."
        )
    total_min = FORECAST_HORIZON_H * 60
    if total_min % minutes != 0:
        raise ValueError(
            f"data.freq='{freq}' teilt den {FORECAST_HORIZON_H}-h-Vorhersagehorizont "
            "nicht ganzzahlig."
        )

    if warn:
        rep = resolution_report(freq)
        if rep['approximated'] == 'measurement':
            logger.warning(
                "data.freq='%s' ist kein Vielfaches des Messrasters von %d min — die "
                "Messungen werden flächengewichtet umverteilt (jeder Zielwert mischt "
                "zwei Ablesungen). Exakt für beide Quellen sind nur Vielfache von "
                "%d min.", freq, MEASUREMENT_STEP_MIN, EXACT_STEP_MIN,
            )
        elif rep['approximated'] == 'both':
            logger.warning(
                "data.freq='%s' ist Vielfaches weder des Mess- (%d min) noch des "
                "ICON-D2-SL-Rasters (%d min) — **beide** Seiten werden umverteilt. "
                "Neue Information entsteht dabei auf keiner.",
                freq, MEASUREMENT_STEP_MIN, NWP_NATIVE_STEP_MIN,
            )
        elif rep['approximated'] == 'nwp':
            logger.warning(
                "data.freq='%s' ist kein Vielfaches des ICON-D2-SL-Rasters von %d min — "
                "die Prognosefelder werden entlang der Lead-Achse flächengewichtet "
                "umverteilt. Das erzeugt eine Auflösung, die das NWP-Modell nicht "
                "hergibt. Exakt für beide Quellen sind nur Vielfache von %d min.",
                freq, NWP_NATIVE_STEP_MIN, EXACT_STEP_MIN,
            )
    return minutes / 60.0, total_min // minutes


def infer_sample_seconds(index: pd.DatetimeIndex, default: float = 600.0) -> float:
    """Abtastintervall der Rohmessungen aus dem Index bestimmen (Modus der Diffs)."""
    deltas = pd.DatetimeIndex(index).to_series().diff().dropna()
    if not len(deltas):
        return default
    seconds = float(deltas.mode().iloc[0].total_seconds())
    return seconds if seconds > 0 else default


def resample_interval_mean(df: pd.DataFrame,
                           freq: str,
                           sample_seconds: float,
                           max_nan_frac: float = 0.5) -> pd.DataFrame:
    """Flächengewichtetes Intervallmittel von ``sample_seconds`` auf ``freq``.

    Die DWD-Rohwerte sind **Intervallmittel** über [t, t+Δt).  Solange ``freq`` ein
    ganzzahliges Vielfaches von Δt ist (10 min → 1 h), ist das arithmetische Mittel
    der Rohwerte exakt — dieser Fall wird direkt durchgereicht.

    Bei ``freq='15min'`` trifft das **nicht** zu: 10-min-Werte liegen nicht im
    15-min-Raster.  Ein einfaches ``.mean()`` würde das Intervall [00:00, 00:15) aus
    den zwei Werten 00:00 und 00:10 ungewichtet mitteln (statt 10:5) und
    [00:15, 00:30) allein aus dem Wert 00:20 bilden.  Beides ist kein Intervallmittel.

    Deshalb wird hier auf das gemeinsame Feinraster (ggT von Δt und ``freq``, also
    5 min für 10↔15) expandiert — jeder Rohwert gilt über seine eigene Intervalllänge
    fort — und erst dort gemittelt.  Da alle Feinzellen gleich lang sind, ist das
    arithmetische Mittel über sie exakt das flächengewichtete Intervallmittel.

    Lücken breiten sich dabei nicht aus: ein Rohwert gilt nur bis ``t + Δt``,
    danach bleibt die Feinzelle NaN.  Ein Zielintervall wird verworfen, sobald
    weniger als ``1 − max_nan_frac`` seiner Dauer durch Messwerte gedeckt ist.
    """
    target_seconds = int(round(pd.Timedelta(freq).total_seconds()))
    sample_int = int(round(sample_seconds))
    if target_seconds < sample_int:
        raise ValueError(
            f"data.freq='{freq}' ist feiner als das Messraster von {sample_int} s — "
            "die Stationsmessungen können das nicht auflösen."
        )

    min_coverage = 1.0 - max_nan_frac

    # Ganzzahliges Vielfaches: Rohwerte nesten sauber, arithmetisches Mittel ist exakt.
    if target_seconds % sample_int == 0:
        resampler = df.resample(freq, closed='left', label='left', origin='epoch')
        agg = resampler.mean()
        expected = max(1.0, target_seconds / sample_int)
        return agg.mask(resampler.count() < min_coverage * expected)

    # Versetztes Raster: über das Feinraster flächengewichtet mitteln.
    fine_seconds = math.gcd(sample_int, target_seconds)
    fine_step = pd.Timedelta(seconds=fine_seconds)
    grid = pd.date_range(
        df.index[0].floor(freq),
        df.index[-1] + pd.Timedelta(seconds=sample_int),
        freq=fine_step, inclusive='left',
    )

    # Jede Feinzelle bekommt den Rohwert des Intervalls, in dem sie liegt.  Bewusst
    # kein ``ffill``: das würde einen *fehlenden* Messwert durch den vorherigen
    # ersetzen und die Lücke damit zuschütten, statt sie sichtbar zu lassen.  Die
    # ``tolerance`` begrenzt die Zuordnung auf die eigene Intervalllänge, sodass
    # Feinzellen ohne zugehörigen Rohwert NaN bleiben.
    pos = pd.Series(np.arange(len(df)), index=df.index).reindex(
        grid, method='ffill',
        tolerance=pd.Timedelta(seconds=sample_int) - pd.Timedelta(1, 'ns'),
    )
    covered = pos.notna().to_numpy()
    values = df.to_numpy(dtype=float)[pos.fillna(0).to_numpy().astype(int)]
    values[~covered] = np.nan
    fine = pd.DataFrame(values, index=grid, columns=df.columns)
    resampler = fine.resample(freq, closed='left', label='left', origin='epoch')
    agg = resampler.mean()
    expected = target_seconds / fine_seconds
    return agg.mask(resampler.count() < min_coverage * expected)


# ---------------------------------------------------------------------------
# ICON-D2 Surface-Level: Gitterpunkt-Scan
# ---------------------------------------------------------------------------

_SL_STEM_RE = re.compile(r'^(-?\d+)_(\d+)_(-?\d+)_(\d+)$')


def parse_sl_lonlat(stem: str) -> tuple[float, float]:
    """``'10_0000_47_8000'`` → ``(lon, lat) = (10.0, 47.8)``.

    Achtung: umgekehrte Reihenfolge gegenüber den ML-Dateinamen (dort lat-first),
    siehe Modul-Docstring.
    """
    m = _SL_STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Unparsbarer SL-Dateiname: '{stem}'")
    lon = float(f"{m.group(1)}.{m.group(2)}")
    lat = float(f"{m.group(3)}.{m.group(4)}")
    return lon, lat


@lru_cache(maxsize=32)
def _scan_sl_grid(nwp_path: str, forecast_hour: str) -> tuple[tuple[str, float, float], ...]:
    """Alle Gitterpunkte eines SL-Laufverzeichnisses als ``(stem, lat, lon)``.

    Gecached: das Verzeichnis enthält ~1218 Dateien und wird sonst für jede der
    ~90 Stationen erneut gelistet.
    """
    sl_dir = os.path.join(nwp_path, 'SL', forecast_hour)
    if not os.path.isdir(sl_dir):
        logger.warning("ICON-D2 SL-Verzeichnis nicht gefunden: %s", sl_dir)
        return ()

    out = []
    for fname in os.listdir(sl_dir):
        if not fname.endswith('_SL.parquet'):
            continue
        stem = fname[:-len('_SL.parquet')]
        try:
            lon, lat = parse_sl_lonlat(stem)
        except ValueError:
            continue
        out.append((stem, lat, lon))

    if not out:
        logger.warning("Keine _SL.parquet-Dateien in %s", sl_dir)
    return tuple(sorted(out))


def select_nearest_sl_points(nwp_path: str,
                             forecast_hour: str,
                             station_lat: float,
                             station_lon: float,
                             k: int) -> list[tuple[str, float, float, float]]:
    """Die ``k`` geodätisch nächsten SL-Gitterpunkte einer Station.

    Returns
    -------
    list of ``(stem, lat, lon, distance_km)``, aufsteigend nach Distanz.
    """
    grid = _scan_sl_grid(nwp_path, forecast_hour)
    if not grid:
        return []
    scored = [
        (stem, lat, lon, geodesic((station_lat, station_lon), (lat, lon)).kilometers)
        for stem, lat, lon in grid
    ]
    scored.sort(key=lambda x: x[3])
    return scored[:k]


# ---------------------------------------------------------------------------
# ICON-D2 Surface-Level: Laden und zeitliche Aggregation
# ---------------------------------------------------------------------------

def _required_sl_columns(features: list[str]) -> list[str]:
    """Rohspalten, die für die gewünschten Feature-Namen aus der Datei gelesen
    werden müssen (Spalten-Pruning beim Parquet-Lesen)."""
    needed: set[str] = set()
    for feat in features:
        if feat in _DERIVED_FROM_SL:
            needed.update(_DERIVED_FROM_SL[feat])
        elif feat == 'dni_nwp':
            needed.update(('aswdir_s', 'aswdifd_s'))
        elif feat in ('kt_nwp', 'kd_nwp'):
            needed.update(('aswdir_s', 'aswdifd_s'))
        elif feat in SOLAR_GEOMETRY_FEATURES:
            continue  # rein aus Zeit/Ort berechnet
        else:
            needed.add(feat)
    return sorted(needed)


def _add_sl_derived(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Berechnet die allein aus SL-Spalten ableitbaren Features in-place."""
    def _col(name):
        return df[name] if name in df.columns else None

    if 'ghi_nwp' in features:
        dirs, diff = _col('aswdir_s'), _col('aswdifd_s')
        if dirs is not None and diff is not None:
            df['ghi_nwp'] = (dirs + diff).clip(lower=0)
    if 'dhi_nwp' in features:
        diff = _col('aswdifd_s')
        if diff is not None:
            df['dhi_nwp'] = diff.clip(lower=0)
    if 'bhi_nwp' in features:
        dirs = _col('aswdir_s')
        if dirs is not None:
            df['bhi_nwp'] = dirs.clip(lower=0)
    if 'wind_speed_nwp' in features:
        u, v = _col('u_10m'), _col('v_10m')
        if u is not None and v is not None:
            df['wind_speed_nwp'] = np.sqrt(u ** 2 + v ** 2)
    # dni_nwp/kt_nwp/kd_nwp brauchen ghi_nwp bzw. dhi_nwp als Zwischenschritt
    if any(f in features for f in _DERIVED_FROM_GEOMETRY):
        dirs, diff = _col('aswdir_s'), _col('aswdifd_s')
        if dirs is not None and diff is not None:
            if 'ghi_nwp' not in df.columns:
                df['ghi_nwp'] = (dirs + diff).clip(lower=0)
            if 'dhi_nwp' not in df.columns:
                df['dhi_nwp'] = diff.clip(lower=0)
            if 'bhi_nwp' not in df.columns:
                df['bhi_nwp'] = dirs.clip(lower=0)
    return df


def _fill_sub_hourly(out: pd.DataFrame,
                     inst_feats: list[str],
                     step_h: float,
                     method: str) -> pd.DataFrame:
    """Füllt die nur stündlich abgelegten Momentanfelder auf ein feineres Raster.

    In den SL-Dateien sind **nur die Strahlungsfelder** (``aswdir_s``,
    ``aswdifd_s``, ``*_avg``) viertelstündlich abgelegt.  ``clct``, ``t_2m``,
    ``alb_rad``, ``u_10m``, ``h_snow``, ``prr_gsp`` und alle übrigen Momentanfelder
    stehen ausschließlich auf der vollen Stunde — bei ``freq='15min'`` sind drei von
    vier Leads dort leer.  Ohne diese Behandlung verwirft das nachgelagerte
    ``dropna()`` genau diese drei Viertel, und ``freq='15min'`` liefert stillschweigend
    wieder ein Stundenraster.

    method
        ``'ffill'``       — Zustand der vollen Stunde gilt bis zur nächsten fort
                            (Default; erfindet keine Werte und passt zur ohnehin
                            verwendeten Momentanwert-Semantik).
        ``'interpolate'`` — linear zwischen den Stundenwerten; der Rest nach dem
                            letzten Stundenanker wird fortgeschrieben, sonst fiele
                            das Laufende weg.
        ``'none'``        — nichts füllen (Momentanfelder bleiben lückenhaft).

    Gefüllt wird ausschließlich **innerhalb eines Laufs** und höchstens über eine
    native Stunde hinweg, damit eine echte Lücke in der Stundenreihe nicht
    beliebig weit verschmiert.
    """
    if step_h >= 1.0 or not inst_feats or method == 'none':
        return out

    # Größte Lücke zwischen zwei Stundenankern in Lead-Schritten.  Die Anker liegen
    # bei floor(k·60 / step_min); teilt step_min die Stunde nicht (z. B. 45 min),
    # ist der Abstand nicht konstant, und ``round(1/step_h) - 1`` unterschätzt ihn
    # (bei 45 min sogar auf 0, womit ein Viertel der Leads NaN bliebe).
    step_min = int(round(step_h * 60))
    limit = -(-60 // step_min) - 1              # ceil(60/step_min) - 1; 3 bei 15 min
    if limit <= 0:
        return out

    out = out.sort_values(['starttime', 'forecasttime'])
    grouped = out.groupby('starttime')[inst_feats]

    if method == 'ffill':
        out[inst_feats] = grouped.ffill(limit=limit)
    elif method == 'interpolate':
        out[inst_feats] = grouped.transform(
            lambda s: s.interpolate(method='linear', limit=limit, limit_area='inside')
        )
        # Nach dem letzten Stundenanker gibt es keinen rechten Stützpunkt mehr —
        # dort bleibt nur Fortschreiben, sonst verliert jeder Lauf seine Endleads.
        out[inst_feats] = out.groupby('starttime')[inst_feats].ffill(limit=limit)
    else:
        raise ValueError(
            f"Unbekannte params.sub_hourly_fill: '{method}' "
            "(erlaubt: 'ffill', 'interpolate', 'none')"
        )
    return out


@lru_cache(maxsize=16)
def _lead_redistribution_matrix(src_step_min: int, dst_step_min: int,
                                n_src: int, n_dst: int) -> np.ndarray:
    """Gewichtsmatrix ``(n_dst, n_src)`` für die Umverteilung entlang der Lead-Achse.

    Spiegelbild von :func:`resample_interval_mean`, nur auf der Prognoseseite:
    Quell-Lead *i* beschreibt die Minuten ``[i·src, (i+1)·src)`` seit ``starttime``,
    Ziel-Lead *j* die Minuten ``[j·dst, (j+1)·dst)``.  Die Gewichte sind die
    Überlappungsanteile, die Zeilensumme ist 1 — das Ergebnis ist damit exakt das
    flächengewichtete Intervallmittel und die Energiebilanz bleibt erhalten.
    """
    W = np.zeros((n_dst, n_src), dtype=float)
    for j in range(n_dst):
        lo, hi = j * dst_step_min, (j + 1) * dst_step_min
        i_lo = max(0, lo // src_step_min)
        i_hi = min(n_src, -(-hi // src_step_min))
        for i in range(i_lo, i_hi):
            ov = min(hi, (i + 1) * src_step_min) - max(lo, i * src_step_min)
            if ov > 0:
                W[j, i] = ov
    return W / dst_step_min


def _redistribute_acc_leads(frame: pd.DataFrame,
                            feats: list[str],
                            src_step_min: int,
                            dst_step_min: int,
                            n_src: int,
                            n_dst: int) -> pd.DataFrame:
    """Akkumulierte Felder vom nativen auf ein feineres Lead-Raster umverteilen.

    Wird nur gebraucht, wenn das Zielraster **kein** Vielfaches des nativen
    SL-Rasters ist (Variante C, z. B. ``freq='10min'``).  Innerhalb eines
    15-min-Blocks entsteht dabei keine neue Information — drei 10-min-Werte, die
    aus zwei 15-min-Werten stammen, sind zwangsläufig teils identisch bzw.
    Mischungen.  Genau das ist der methodische Preis dieser Variante.
    """
    W = _lead_redistribution_matrix(src_step_min, dst_step_min, n_src, n_dst)
    # factorize statt np.unique: erhält den tz-aware datetime64-Dtype, sonst
    # käme eine Objekt-Spalte heraus und der spätere Merge auf 'starttime' liefe leer.
    inv, uniq = pd.factorize(frame['starttime'], sort=True)
    lead = frame['forecasttime'].to_numpy().astype(int)

    out = {'starttime': pd.Index(uniq).repeat(n_dst),
           'forecasttime': np.tile(np.arange(n_dst), len(uniq))}
    for f in feats:
        src = np.full((len(uniq), n_src), np.nan)
        src[inv, lead] = frame[f].to_numpy(dtype=float)
        # NaN würde über die Matrix die ganze Zielzeile vergiften; stattdessen
        # nur über die tatsächlich vorhandenen Quellzellen normieren.
        mask = ~np.isnan(src)
        filled = np.where(mask, src, 0.0)
        num = filled @ W.T
        den = mask.astype(float) @ W.T
        with np.errstate(invalid='ignore', divide='ignore'):
            val = np.where(den > 0, num / den, np.nan)
        out[f] = val.reshape(-1)
    result = pd.DataFrame(out)
    # Zielleads ohne *jede* Quelldeckung dürfen nicht als Zeile stehenbleiben: im
    # Zweig ohne Umverteilung entsteht für eine fehlende Prognose gar keine Zeile,
    # und genau darauf beruht die Vollständigkeitsprüfung (n_leads Zeilen je Lauf)
    # in preprocess_solar_icond2(). Ohne dieses dropna liefert ein nur stündlich
    # abgelegter Altlauf trotzdem volle n_dst Zeilen, gilt damit als vollständig,
    # und die anschließende Modus-Heuristik über die Lauflängen kann auf die
    # verstümmelte Länge kippen — dann überleben ausgerechnet die kaputten Läufe.
    return result.dropna(subset=feats, how='all')


def load_sl_grid_point(fpath: str,
                       features: list[str],
                       step_h: float,
                       n_leads: int,
                       starttime_min: pd.Timestamp | None = None,
                       starttime_max: pd.Timestamp | None = None,
                       sub_hourly_fill: str = 'ffill') -> pd.DataFrame:
    """Lädt eine SL-Gitterpunkt-Datei und aggregiert sie auf ``step_h``-Raster.

    Ist ``step_h`` ein Vielfaches des nativen 15-min-Rasters, werden die nativen
    Schritte einfach zusammengemittelt (exakt).  Andernfalls — Variante C, etwa
    ``freq='10min'`` — laufen die akkumulierten Felder zusätzlich durch
    :func:`_redistribute_acc_leads`.

    Returns
    -------
    DataFrame mit ``starttime`` (UTC), ``forecasttime`` (Lead-Index 0…n_leads-1)
    und einer Spalte je Feature.
    """
    raw_cols = _required_sl_columns(features)
    read_cols = sorted(set(raw_cols) | {'starttime', 'forecasttime'})

    df = pd.read_parquet(fpath, engine='pyarrow', columns=read_cols)
    if df.empty:
        return pd.DataFrame(columns=['starttime', 'forecasttime'])

    # starttime ist mit festem +02:00-Offset gespeichert (ganzjährig, keine
    # Sommerzeitumstellung) — nach UTC entspricht die Stunde dem delivery_hour.
    df['starttime'] = pd.to_datetime(df['starttime'], utc=True)
    if starttime_min is not None:
        df = df[df['starttime'] >= starttime_min]
    if starttime_max is not None:
        df = df[df['starttime'] <= starttime_max]
    if df.empty:
        return pd.DataFrame(columns=['starttime', 'forecasttime'])

    df = df.drop_duplicates(subset=['starttime', 'forecasttime'], keep='last')
    df = _add_sl_derived(df, features)

    step_min = int(round(step_h * 60))
    native = NWP_NATIVE_STEP_MIN
    # Nur wenn das Zielraster kein Vielfaches des nativen ist, muss umverteilt
    # werden; sonst mittelt das ceil-Mapping ganze native Schritte zusammen.
    needs_redistribution = step_min % native != 0

    present = [f for f in features if f in df.columns]
    acc_feats = [f for f in present if _is_accumulated_feature(f)]
    inst_feats = [f for f in present if not _is_accumulated_feature(f)]

    # Akkumulierte Felder: bei Bedarf erst auf das native Raster, dann umverteilen.
    acc_step_h = native / 60.0 if needs_redistribution else step_h
    acc_n_leads = int(round(FORECAST_HORIZON_H * 60 / native)) if needs_redistribution else n_leads

    # In Minuten rechnen: ``forecasttime`` ist ein Vielfaches von 0.25 h, also von
    # 15 min — ganzzahlig und frei von den Rundungsfehlern, die ``1/(1/6)`` bei
    # step_h = 10 min erzeugen würde.
    ft_min = np.rint(df['forecasttime'].astype(float) * 60).astype(int)
    acc_step_min = int(round(acc_step_h * 60))
    acc_lead = -(-ft_min // acc_step_min) - 1     # ceil, Intervall endet auf ft
    inst_lead = ft_min // step_min                # floor, Momentanwert zu Intervallbeginn

    parts = []
    for feats, lead, limit in ((acc_feats, acc_lead, acc_n_leads),
                               (inst_feats, inst_lead, n_leads)):
        if not feats:
            continue
        tmp = df[['starttime'] + feats].copy()
        tmp['forecasttime'] = lead.values
        tmp = tmp[(tmp['forecasttime'] >= 0) & (tmp['forecasttime'] < limit)]
        agg = tmp.groupby(['starttime', 'forecasttime'], as_index=False)[feats].mean()
        if feats is acc_feats and needs_redistribution:
            agg = _redistribute_acc_leads(agg, feats, native, step_min,
                                          acc_n_leads, n_leads)
        parts.append(agg)

    if not parts:
        return pd.DataFrame(columns=['starttime', 'forecasttime'])

    out = parts[0]
    for extra in parts[1:]:
        out = out.merge(extra, on=['starttime', 'forecasttime'], how='outer')
    return _fill_sub_hourly(out, inst_feats, step_h, sub_hourly_fill)


def _is_accumulated_feature(feature: str) -> bool:
    """True, wenn das Feature aus akkumulierten/gemittelten SL-Feldern stammt."""
    if feature in ACCUMULATED_SL_COLS:
        return True
    if feature in ('ghi_nwp', 'dhi_nwp', 'bhi_nwp', 'dni_nwp', 'kt_nwp', 'kd_nwp'):
        return True
    return False


def load_icond2_sl_for_station(nwp_path: str,
                               station_lat: float,
                               station_lon: float,
                               features: list[str],
                               forecast_hours: list[str],
                               next_n_grid_points: int,
                               step_h: float,
                               n_leads: int,
                               starttime_min: pd.Timestamp | None = None,
                               starttime_max: pd.Timestamp | None = None,
                               n_workers: int = 8,
                               sub_hourly_fill: str = 'ffill') -> tuple[pd.DataFrame, str | None]:
    """Lädt alle ICON-D2-SL-Läufe einer Station über alle Vorhersagestunden.

    Spaltenbenennung folgt der Wind-Pipeline: ``<feature>_<rang>`` mit Rang 1 =
    nächstgelegener Gitterpunkt.

    ``sub_hourly_fill`` greift nur bei ``step_h < 1`` und steuert, wie die allein
    stündlich abgelegten Momentanfelder auf das feinere Raster kommen —
    siehe :func:`_fill_sub_hourly`.

    Returns
    -------
    (DataFrame mit starttime/forecasttime/timestamp + Featurespalten,
     Label des nächstgelegenen Gitterpunkts für die NWP-Baseline)
    """
    jobs: list[tuple[str, str, int]] = []   # (fpath, forecast_hour, rank)
    nearest_label = None

    for fh in forecast_hours:
        nearest = select_nearest_sl_points(nwp_path, fh, station_lat, station_lon,
                                           next_n_grid_points)
        if not nearest:
            logger.warning("Keine SL-Gitterpunkte für Vorhersagestunde %s gefunden.", fh)
            continue
        nearest_label = '1'
        for rank, (stem, _lat, _lon, _dist) in enumerate(nearest, start=1):
            fpath = os.path.join(nwp_path, 'SL', fh, f'{stem}_SL.parquet')
            jobs.append((fpath, fh, rank))

    if not jobs:
        raise FileNotFoundError(
            f"Keine ICON-D2-SL-Daten unter {os.path.join(nwp_path, 'SL')} "
            f"für forecast_hours={forecast_hours}."
        )

    def _load(job):
        fpath, fh, rank = job
        frame = load_sl_grid_point(fpath, features, step_h, n_leads,
                                   starttime_min, starttime_max, sub_hourly_fill)
        return fh, rank, frame

    per_rank: dict[int, list[pd.DataFrame]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(n_workers, len(jobs)))) as ex:
        futures = [ex.submit(_load, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                _fh, rank, frame = fut.result()
            except Exception as exc:  # eine defekte Datei darf den Lauf nicht killen
                logger.warning("SL-Datei konnte nicht geladen werden: %s", exc)
                continue
            if not frame.empty:
                per_rank.setdefault(rank, []).append(frame)

    if not per_rank:
        raise ValueError("Alle ICON-D2-SL-Dateien waren leer oder nicht lesbar.")

    merged = None
    for rank in sorted(per_rank):
        frame = pd.concat(per_rank[rank], ignore_index=True)
        frame = frame.drop_duplicates(subset=['starttime', 'forecasttime'], keep='last')
        feat_cols = [c for c in frame.columns if c not in ('starttime', 'forecasttime')]
        frame = frame.rename(columns={c: f'{c}_{rank}' for c in feat_cols})
        merged = frame if merged is None else merged.merge(
            frame, on=['starttime', 'forecasttime'], how='outer'
        )

    # In ganzen Minuten rechnen: step_h ist bei 10 min = 1/6 h binär nicht exakt,
    # und ``lead · step_h`` verfehlt dann das 10-min-Raster der Messungen um
    # Nanosekunden — der Merge auf 'timestamp' liefe für zwei Drittel der Leads leer.
    step_min_out = int(round(step_h * 60))
    merged['timestamp'] = (
        merged['starttime']
        + pd.to_timedelta(merged['forecasttime'].astype('int64') * step_min_out, unit='m')
    )
    return merged.sort_values(['starttime', 'forecasttime']).reset_index(drop=True), nearest_label


# ---------------------------------------------------------------------------
# Stationsmessungen
# ---------------------------------------------------------------------------

def load_station_measurements(station_parquet: str,
                              freq: str = '1h',
                              time_label: str = 'right',
                              lower: pd.Timestamp | None = None,
                              upper: pd.Timestamp | None = None,
                              max_nan_frac: float = 0.5) -> pd.DataFrame:
    """Lädt eine DWD-Solarstation und aggregiert sie auf ``freq``.

    ``ghi``/``dhi`` liegen als **J/cm² je Messintervall** vor und werden hier auf
    W/m² umgerechnet:  ``W/m² = J/cm² · 10⁴ / Δt[s]``.  Δt wird aus dem tatsächlichen
    Abtastintervall bestimmt, nicht fest verdrahtet.

    Parameters
    ----------
    time_label : 'left' | 'right'
        Bedeutung des Zeitstempels der Rohdaten.  ``'right'`` (Default) = der
        Zeitstempel markiert das Intervall*ende*, der Index wird vor dem Resampling
        um ein Intervall zurückgeschoben.

        Belegt über den Verschiebungstest gegen ``ghi_nwp`` (7 Stationen,
        Apr–Sep 2024, beide Auflösungen): mit ``'right'`` liegt das RMSE-Minimum
        symmetrisch auf Verschiebung 0 (z. B. Station 00183 bei 15 min:
        80.7 / **78.5** / 80.7 W/m² für −15/0/+15 min), mit ``'left'`` dagegen
        asymmetrisch nach +1 Schritt gezogen (84.4 / 79.7 / **78.8**).
        Bei stündlicher Auflösung mittelt sich der 10-min-Versatz weitgehend heraus
        (69.0 → 68.4 W/m²); erst ``freq='15min'`` macht ihn deutlich sichtbar.
    max_nan_frac : float
        Anteil fehlender Rohwerte, ab dem das aggregierte Intervall auf NaN gesetzt
        wird (statt stillschweigend über wenige vorhandene Werte zu mitteln).

    Notes
    -----
    Die Aggregation läuft über :func:`resample_interval_mean`, weil das 10-min-Raster
    der Messungen nicht in das 15-min-Raster der SL-Prognosen nestet.
    """
    df = pd.read_parquet(station_parquet, engine='pyarrow')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.drop(columns=['station_id'], errors='ignore').select_dtypes(include='number')
    df = df.sort_index()

    if df.empty:
        return df

    # Abtastintervall aus den Daten bestimmen (10 min bei den DWD-Solardateien).
    sample_seconds = infer_sample_seconds(df.index)

    if time_label == 'right':
        df.index = df.index - pd.Timedelta(seconds=sample_seconds)
    elif time_label != 'left':
        raise ValueError(f"time_label muss 'left' oder 'right' sein, nicht '{time_label}'")

    # J/cm² je Intervall  →  W/m²
    j_per_cm2_to_w_per_m2 = 1e4 / sample_seconds
    for col in ('ghi', 'dhi'):
        if col in df.columns:
            df[col] = df[col] * j_per_cm2_to_w_per_m2

    if lower is not None:
        df = df[df.index >= lower]
    if upper is not None:
        df = df[df.index <= upper]
    if df.empty:
        return df

    return resample_interval_mean(df, freq, sample_seconds, max_nan_frac)


# ---------------------------------------------------------------------------
# Sonnengeometrie und abgeleitete Zielgrößen
# ---------------------------------------------------------------------------

def solar_geometry(times: pd.DatetimeIndex,
                   latitude: float,
                   longitude: float,
                   altitude: float,
                   freq: str = '1h') -> pd.DataFrame:
    """Sonnenstand, Clear-Sky-Strahlung und zyklische Zeitfeatures.

    Die Sonnenposition wird zur **Intervallmitte** ausgewertet (die Zeitstempel sind
    linksbündige Intervalllabels), sonst wäre der Zenitwinkel bei stündlicher
    Auflösung systematisch um eine halbe Stunde versetzt.
    """
    times = pd.DatetimeIndex(times).sort_values().unique()
    times = pd.DatetimeIndex(times)
    mid = times + pd.Timedelta(freq) / 2

    solpos = pvlib.solarposition.get_solarposition(mid, latitude, longitude,
                                                   altitude=altitude)
    solpos.index = times

    zenith = solpos['zenith']
    app_zenith = solpos['apparent_zenith']
    azimuth = solpos['azimuth']

    airmass_rel = pvlib.atmosphere.get_relative_airmass(app_zenith)
    pressure = pvlib.atmosphere.alt2pres(altitude)
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel, pressure)

    linke = pvlib.clearsky.lookup_linke_turbidity(mid, latitude, longitude)
    linke.index = times
    dni_extra = pd.Series(pvlib.irradiance.get_extra_radiation(mid).values, index=times)

    clearsky = pvlib.clearsky.ineichen(
        apparent_zenith=app_zenith,
        airmass_absolute=airmass_abs.fillna(0.0),
        linke_turbidity=linke,
        altitude=altitude,
        dni_extra=dni_extra,
    )

    doy = times.dayofyear.values
    hour = times.hour.values + times.minute.values / 60.0

    out = pd.DataFrame(index=times)
    out['solar_zenith'] = zenith.values
    out['solar_zenith_cos'] = np.cos(np.deg2rad(zenith.values)).clip(min=0.0)
    out['solar_azimuth_sin'] = np.sin(np.deg2rad(azimuth.values))
    out['solar_azimuth_cos'] = np.cos(np.deg2rad(azimuth.values))
    out['airmass'] = airmass_abs.values
    out['airmass'] = out['airmass'].fillna(0.0)
    out['dni_extra'] = dni_extra.values
    out['ghi_clearsky'] = clearsky['ghi'].values
    out['dni_clearsky'] = clearsky['dni'].values
    out['dhi_clearsky'] = clearsky['dhi'].values
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    out['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    out.index.name = 'timestamp'
    return out


def derive_irradiance_components(ghi: pd.Series,
                                 dhi: pd.Series,
                                 zenith: pd.Series,
                                 dni_clearsky: pd.Series | None = None) -> dict[str, pd.Series]:
    """``bhi = ghi − dhi`` und ``dni = bhi / cos θ_z``.

    DNI wird über ``pvlib.irradiance.dni()`` berechnet, das bei θ_z > 88° auf 0 setzt
    und optional gegen die Clear-Sky-DNI deckelt — ohne diese Deckelung explodiert
    der Quotient bei tiefstehender Sonne.
    """
    bhi = (ghi - dhi).clip(lower=0)
    dni = pvlib.irradiance.dni(
        ghi=ghi, dhi=dhi, zenith=zenith,
        clearsky_dni=dni_clearsky,
        zenith_threshold_for_zero_dni=_ZENITH_ZERO_DNI,
    )
    dni = pd.Series(dni, index=ghi.index).fillna(0.0).clip(lower=0)
    return {'bhi': bhi, 'dni': dni}


def clearsky_index(value: pd.Series,
                   clearsky: pd.Series,
                   min_clearsky: float = 20.0,
                   clip_max: float = 1.5) -> pd.Series:
    """``k = value / clearsky``, robust gegen Nacht/Dämmerung.

    Unterhalb ``min_clearsky`` W/m² (Nacht und tiefe Dämmerung) ist der Quotient
    numerisch wertlos und wird auf 0 gesetzt; oben wird auf ``clip_max`` gedeckelt
    (Cloud-Enhancement erreicht real bis ~1.4).
    """
    out = pd.Series(0.0, index=value.index, dtype=float)
    usable = clearsky > min_clearsky
    out[usable] = (value[usable] / clearsky[usable]).clip(lower=0.0, upper=clip_max)
    return out


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def preprocess_solar_icond2(path: str,
                            config: dict,
                            freq: str = '1h',
                            features: dict | None = None) -> pd.DataFrame:
    """Preprocessing einer Solarstation: Messungen + ICON-D2-SL-Prognosen.

    Pendant zu ``preprocessing.preprocess_synth_wind_icond2()`` mit identischem
    Rückgabekontrakt: MultiIndex ``['starttime', 'forecasttime', 'timestamp']``.

    Args:
        path: Pfad zur Stationsdatei (``Station_<id>.parquet``); die ID wird aus dem
            Dateinamen extrahiert.
        config: Konfigurations-Dictionary.
        freq: Zielfrequenz (Default ``'1h'``).
        features: Dict mit ``known``/``observed``/``static``.
    """
    # Später Import, um einen Zirkelbezug preprocessing ↔ solar zu vermeiden.
    from . import preprocessing as _prep

    features = features or {'known': [], 'observed': [], 'static': []}
    data_cfg = config.get('data', {})
    params_cfg = config.get('params', {})

    station_id = _station_id_from_path(path)

    # Auflösung zuerst prüfen — sonst schlägt ein ungültiges freq erst beim
    # Resampling der Messungen mit einer weniger aussagekräftigen Meldung fehl.
    step_h, n_leads = resolve_freq(freq)

    # ------------------------------------------------------------------
    # Zeitfenster (7 Tage Vorlauf, damit die ersten Sequenzen genug Historie haben)
    # ------------------------------------------------------------------
    lower_str = data_cfg.get('train_start')
    upper_str = data_cfg.get('data_cutoff') or data_cfg.get('test_end')
    lower = pd.Timestamp(lower_str, tz='UTC') - pd.Timedelta(days=7) if lower_str else None
    upper = pd.Timestamp(upper_str, tz='UTC') if upper_str else None

    # ------------------------------------------------------------------
    # Stationsmetadaten
    # ------------------------------------------------------------------
    stations_df = pd.read_csv(data_cfg['stations_master'], sep=',', dtype={'station_id': str})
    station_info = stations_df[stations_df['station_id'] == station_id]
    if station_info.empty:
        raise ValueError(f"Station {station_id} nicht in {data_cfg['stations_master']} gefunden")
    station_lat = float(station_info['latitude'].iloc[0])
    station_lon = float(station_info['longitude'].iloc[0])
    altitude = float(station_info['station_height'].iloc[0])

    # ------------------------------------------------------------------
    # 1. Messungen
    # ------------------------------------------------------------------
    station_parquet = os.path.join(data_cfg['path'], f'Station_{station_id}.parquet')
    df_meas = load_station_measurements(
        station_parquet,
        freq=freq,
        time_label=params_cfg.get('measurement_time_label', 'right'),
        lower=lower,
        upper=upper,
        max_nan_frac=params_cfg.get('max_nan_frac', 0.5),
    )
    if df_meas.empty:
        raise ValueError(f"Station {station_id}: keine Messdaten im Zeitraum.")
    if 'ghi' not in df_meas.columns or df_meas['ghi'].notna().sum() == 0:
        raise ValueError(
            f"Station {station_id} führt keine Strahlungsdaten (ghi vollständig NaN). "
            "Nur 93 der 204 Stationen haben ghi/dhi — Stationsliste prüfen."
        )

    # ------------------------------------------------------------------
    # 2. ICON-D2 Surface-Level
    # ------------------------------------------------------------------
    icond2_features = _resolve_nwp_features(features, params_cfg)
    forecast_hours = [str(h).zfill(2) for h in data_cfg.get('forecast_hours', ['06', '09', '12', '15'])]

    df_nwp, nearest_label = load_icond2_sl_for_station(
        nwp_path=data_cfg['nwp_path'],
        station_lat=station_lat,
        station_lon=station_lon,
        features=icond2_features,
        forecast_hours=forecast_hours,
        next_n_grid_points=params_cfg.get('next_n_grid_points', 1),
        step_h=step_h,
        n_leads=n_leads,
        starttime_min=lower,
        starttime_max=upper,
        n_workers=params_cfg.get('nwp_workers', 8),
        sub_hourly_fill=params_cfg.get('sub_hourly_fill', 'ffill'),
    )

    # Unvollständige Läufe verwerfen (n_leads Schritte je starttime).
    # Gezählt werden nur Leads, für die *alle* NWP-Spalten belegt sind. Die bloße
    # Zeilenzahl genügt nicht: akkumulierte und Momentanfelder werden getrennt
    # aggregiert und per Outer-Join zusammengeführt, und bei den nur stündlich
    # abgelegten Altläufen landen die beiden Hälften auf *verschiedenen* Leads
    # (bei freq='30min' die Strahlung auf den ungeraden, die Momentanwerte auf den
    # geraden). Der Lauf hat dann volle n_leads Zeilen, von denen aber jede zur
    # Hälfte NaN ist. Das anschließende dropna() halbiert ihn, und die
    # Modus-Heuristik über die Lauflängen weiter unten kann auf diese halbierte
    # Länge kippen — dann überleben ausgerechnet die Altläufe und die vollständigen
    # Viertelstundenläufe fliegen raus.
    _nwp_feat_cols = [c for c in df_nwp.columns
                      if c not in ('starttime', 'forecasttime', 'timestamp')]
    counts = df_nwp.dropna(subset=_nwp_feat_cols).groupby('starttime').size()
    valid_starts = counts[counts == n_leads].index
    n_runs_total = df_nwp['starttime'].nunique()
    n_dropped = n_runs_total - len(valid_starts)
    if n_dropped:
        # Bei step_h < 1 fallen zusätzlich die frühen, nur stündlich abgelegten SL-Läufe
        # heraus (bis ~2023-08-08 rund 2 % der Läufe) — das ist beabsichtigt, aber
        # sichtbar zu machen, sonst wirkt der Datensatz grundlos kürzer als bei '1h'.
        logger.info(
            "Station %s: %d von %d NWP-Läufen unvollständig (< %d Leads bei freq='%s') "
            "und verworfen.", station_id, n_dropped, n_runs_total, n_leads, freq,
        )
    df_nwp = df_nwp[df_nwp['starttime'].isin(valid_starts)].copy()
    if df_nwp.empty:
        raise ValueError(f"Station {station_id}: keine vollständigen ICON-D2-SL-Läufe.")

    # ------------------------------------------------------------------
    # 3. Merge Messungen ↔ Prognosen
    # ------------------------------------------------------------------
    df = df_nwp.merge(df_meas, left_on='timestamp', right_index=True, how='left')

    # ------------------------------------------------------------------
    # 4. Sonnengeometrie und abgeleitete Größen
    # ------------------------------------------------------------------
    geo = solar_geometry(pd.DatetimeIndex(df['timestamp'].unique()),
                         station_lat, station_lon, altitude, freq=freq)
    df = df.merge(geo, left_on='timestamp', right_index=True, how='left')

    if 'dhi' in df.columns:
        derived = derive_irradiance_components(
            ghi=df['ghi'], dhi=df['dhi'], zenith=df['solar_zenith'],
            dni_clearsky=df['dni_clearsky'],
        )
        df['bhi'] = derived['bhi']
        df['dni'] = derived['dni']
        df['kd'] = np.where(df['ghi'] > 0, (df['dhi'] / df['ghi']).clip(0, 1), 0.0)
    df['kt'] = clearsky_index(df['ghi'], df['ghi_clearsky'])

    # NWP-seitige, geometrieabhängige Ableitungen je Gitterpunkt-Rang
    _add_nwp_geometry_features(df, icond2_features)

    # ------------------------------------------------------------------
    # 4b. ECMWF-HRES als zweite NWP-Quelle (optional)
    # ------------------------------------------------------------------
    df = _merge_ecmwf(df, config, station_lat, station_lon, station_id,
                      lower, upper, step_h)

    # ------------------------------------------------------------------
    # 5. Nachbarstationen (optional)
    # ------------------------------------------------------------------
    df = _merge_neighbor_stations(df, config, features, stations_df, station_id,
                                  station_lat, station_lon, freq)

    # ------------------------------------------------------------------
    # 6. Zielgrößen-Transformation
    # ------------------------------------------------------------------
    target_cols = get_target_cols(config)
    transform = params_cfg.get('target_transform', 'none')
    if transform == 'clearsky_index':
        cs_map = {'ghi': 'ghi_clearsky', 'dhi': 'dhi_clearsky',
                  'bhi': 'dni_clearsky', 'dni': 'dni_clearsky'}
        # Kappungsgrenze je Zielgroesse einstellbar (Skalar oder Dict).
        #
        # Der Default 1.5 taugt nur fuer ghi: Cloud Enhancement erreicht dort real
        # bis ~1.4, gemessen sind 2.8-4.3 % der Tagsamples darueber. Fuer dhi ist er
        # falsch — Diffusstrahlung ist UNTER Wolken maximal, nicht bei klarem
        # Himmel. Gemessen an zwei Teststationen liegen 29.5 % bzw. 48.3 % der
        # Tagsamples ueber 1.5, der Median bei 1.10 bzw. 1.46, das Maximum bei 5.29.
        # Mit der Kappung wird die Zielgroesse abgeschnitten statt transformiert:
        # der RMSE faellt um 54 %, aber die NWP-Baseline sinkt gleich mit
        # (47.77 -> 41.61 W/m²) — die Zahlen messen dann eine andere, leichtere
        # Aufgabe und sind mit den uebrigen Laeufen nicht vergleichbar.
        clip_cfg = params_cfg.get('clearsky_clip_max', 1.5)
        for tgt in target_cols:
            if tgt in cs_map and tgt in df.columns:
                clip = (clip_cfg.get(tgt, 1.5) if isinstance(clip_cfg, dict) else clip_cfg)
                df[tgt] = clearsky_index(df[tgt], df[cs_map[tgt]], clip_max=float(clip))
                logger.debug("Station %s: '%s' als Clear-Sky-Index, Kappung bei %.1f.",
                             station_id, tgt, float(clip))
    elif transform == 'nwp_residual':
        df = _to_nwp_residual(df, target_cols, params_cfg, station_id)
    elif transform != 'none':
        raise ValueError(f"Unbekannte params.target_transform: '{transform}' "
                         "(erlaubt: 'none', 'clearsky_index', 'nwp_residual')")

    # ------------------------------------------------------------------
    # 7. MultiIndex, statische Features, Spaltenauswahl
    # ------------------------------------------------------------------
    df = df.set_index(['starttime', 'forecasttime', 'timestamp']).sort_index()

    static_features = features.get('static', []) or []
    static_data = {'latitude': station_lat, 'longitude': station_lon,
                   'altitude': altitude, 'station_height': altitude}
    static_data.update(_prep._get_topo_features(station_id, params_cfg.get('topo_features_path')))
    for feat in static_features:
        if feat in static_data:
            df[feat] = static_data[feat]
        else:
            logger.warning("Station %s: statisches Feature '%s' nicht verfügbar.",
                           station_id, feat)

    df = _select_columns(df, features, target_cols, static_features, params_cfg)

    # ------------------------------------------------------------------
    # 8. Aufräumen
    # ------------------------------------------------------------------
    before = len(df)
    df = df.dropna()
    if before and len(df) < before:
        logger.debug("Station %s: %d von %d Zeilen wegen NaN verworfen (%.1f %%).",
                     station_id, before - len(df), before, 100 * (before - len(df)) / before)

    # Randläufe, die durch dropna() unvollständig geworden sind, ganz entfernen
    run_lengths = df.groupby(level='starttime').size()
    if len(run_lengths):
        expected = int(run_lengths.value_counts().idxmax())
        keep = run_lengths[run_lengths == expected].index
        df = df[df.index.get_level_values('starttime').isin(keep)]

    if df.empty:
        raise ValueError(f"Station {station_id}: nach dem Preprocessing keine Daten übrig.")

    if nearest_label is not None:
        df.attrs['nwp_nearest_label'] = nearest_label
    df.attrs['target_cols'] = target_cols
    return df.sort_index()


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def get_target_cols(config: dict) -> list[str]:
    """Zielspalten aus der Config: ``data.target_cols`` (Liste) oder ``data.target_col``."""
    data_cfg = config.get('data', {})
    cols = data_cfg.get('target_cols')
    if cols:
        return [str(c) for c in (cols if isinstance(cols, (list, tuple)) else [cols])]
    return [str(data_cfg.get('target_col', 'ghi'))]


def _station_id_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith('Station_') or stem.startswith('synth_'):
        return stem.split('_')[-1]
    return stem


def _resolve_nwp_features(features: dict, params_cfg: dict) -> list[str]:
    """Welche ICON-D2-SL-Features geladen werden müssen.

    Bevorzugt ``params.icond2_features``; sonst werden sie aus den known/observed-
    Features abgeleitet (alles, was weder Messgröße, Sonnengeometrie noch
    Nachbarfeature ist).
    """
    explicit = params_cfg.get('icond2_features')
    if explicit:
        return _reject_running_means([str(f) for f in explicit])

    measurement_like = {'ghi', 'dhi', 'bhi', 'dni', 'kt', 'kd',
                        'temperature_2m', 'wind_speed',
                        'precipitation_rate', 'precipitation_duration'}
    out = []
    for feat in (features.get('known', []) or []) + (features.get('observed', []) or []):
        if feat in measurement_like or feat in SOLAR_GEOMETRY_FEATURES:
            continue
        if feat.endswith('_next') or feat.startswith('ecmwf_'):
            continue
        if feat not in out:
            out.append(feat)
    if not out:
        out = ['ghi_nwp', 'dhi_nwp']
    return _reject_running_means(out)


def _reject_running_means(feats: list[str]) -> list[str]:
    """Bricht ab, wenn ein Laufmittel-Feld als Feature angefordert wurde."""
    bad = [f for f in feats if f in RUNNING_MEAN_SL_COLS]
    if bad:
        hints = ', '.join(f"'{f}' → '{_RUNNING_MEAN_REPLACEMENT[f]}'" for f in bad)
        raise ValueError(
            f"params.icond2_features enthält Laufmittel-Felder: {bad}. "
            "Diese tragen das Mittel seit Vorhersagebeginn ([0, ft]), nicht den Wert "
            "des Zeitschritts — als Feature wären sie ein über den Lauf wanderndes "
            f"Mittel mit falschem Zeitstempel. Dekumulierte Entsprechung nutzen: {hints}."
        )
    return feats


def _add_nwp_geometry_features(df: pd.DataFrame, icond2_features: list[str]) -> None:
    """Berechnet ``dni_nwp``/``kt_nwp``/``kd_nwp`` je Gitterpunkt-Rang (in-place)."""
    wanted = [f for f in icond2_features if f in _DERIVED_FROM_GEOMETRY]
    if not wanted:
        return
    ranks = sorted({
        col.rsplit('_', 1)[1] for col in df.columns
        if col.startswith('ghi_nwp_') and col.rsplit('_', 1)[1].isdigit()
    }, key=int)

    for rank in ranks:
        ghi_col, dhi_col = f'ghi_nwp_{rank}', f'dhi_nwp_{rank}'
        if ghi_col not in df.columns or dhi_col not in df.columns:
            continue
        if 'dni_nwp' in wanted:
            comp = derive_irradiance_components(
                ghi=df[ghi_col], dhi=df[dhi_col], zenith=df['solar_zenith'],
                dni_clearsky=df['dni_clearsky'],
            )
            df[f'dni_nwp_{rank}'] = comp['dni']
        if 'kt_nwp' in wanted:
            df[f'kt_nwp_{rank}'] = clearsky_index(df[ghi_col], df['ghi_clearsky'])
        if 'kd_nwp' in wanted:
            df[f'kd_nwp_{rank}'] = np.where(
                df[ghi_col] > 0, (df[dhi_col] / df[ghi_col]).clip(0, 1), 0.0
            )




#: Baselinespalte je Zielgroesse fuer das Residuum-Ziel. Bewusst dieselbe Zuordnung
#: wie eval._NWP_BASELINE_BY_TARGET — waeren es verschiedene, wuerde Skill_NWP gegen
#: eine andere Groesse gemessen als die, die vom Ziel abgezogen wurde.
_RESIDUAL_BASELINE_BY_TARGET = {
    'ghi': 'ghi_nwp', 'dhi': 'dhi_nwp', 'bhi': 'bhi_nwp', 'dni': 'dni_nwp',
    'kt': 'kt_nwp', 'kd': 'kd_nwp',
}


def resolve_residual_baseline_col(df_columns, target_col: str,
                                  params_cfg: dict | None = None) -> str | None:
    """Spalte, die fuer ``target_col`` als Residuumsbasis dient (naechster Gitterpunkt).

    Beruecksichtigt ``params.nwp_baseline_col`` (Skalar oder Dict) und faellt sonst
    auf die Standardzuordnung zurueck. Bevorzugt die ``_1``-Variante, also den
    naechstgelegenen Gitterpunkt — dieselbe Wahl, die auch eval.py trifft.
    """
    cfg_val = (params_cfg or {}).get('nwp_baseline_col')
    # Nur die Dict-Form ist zielspezifisch. Ein Skalar waere bei mehreren Zielgroessen
    # mehrdeutig — dieselbe Regel wie in eval._resolve_nwp_baseline, damit Abzug und
    # Skill_NWP garantiert dieselbe Spalte verwenden.
    prefix = cfg_val.get(target_col) if isinstance(cfg_val, dict) else None
    if not prefix:
        prefix = _RESIDUAL_BASELINE_BY_TARGET.get(target_col)
    if not prefix:
        return None
    cands = [c for c in df_columns if c == prefix or c.startswith(prefix + '_')]
    if not cands:
        return None
    return next((c for c in cands if c.endswith('_1')), cands[0])


def _to_nwp_residual(df: pd.DataFrame,
                     target_cols: list[str],
                     params_cfg: dict,
                     station_id: str) -> pd.DataFrame:
    """Zielgroessen auf das Residuum gegen die ICON-D2-Prognose umstellen.

    Aus ``ghi`` wird ``ghi - ghi_nwp_1``. Damit ist die rohe NWP-Prognose durch die
    Ausgabe **null** reproduziert: Skill_NWP >= 0 wird zum Boden statt zur Decke.
    Genau das ist Bias Correction im Wortsinn.

    Die Metriken bleiben dabei mit dem Absolutraum vergleichbar, weil Differenzen
    erhalten bleiben:  ``(y_pred + nwp) - (y_true + nwp) = y_pred - y_true``.
    RMSE und MAE sind also identisch zu einer Auswertung in W/m², und die Baseline
    ist im Residuumsraum die Nullreihe — deren RMSE ist genau der NWP-Fehler.
    ``eval._evaluate_single_target`` setzt sie deshalb auf null statt auf die
    NWP-Spalte (Parameter ``nwp_residual``).

    Zwei Dinge aendern ihre Bedeutung und sind so gewollt:

    * **R²** bezieht sich auf die Varianz des Residuums, nicht auf die von ``ghi``.
      Das ist der strengere und aussagekraeftigere Wert.
    * Die Vorhersage darf **negativ** werden. ``tools.get_y`` muss deshalb mit
      ``clip_negative=False`` laufen, sonst wird die halbe Verteilung abgeschnitten.
    """
    missing = []
    for tgt in target_cols:
        if tgt not in df.columns:
            continue
        base_col = resolve_residual_baseline_col(df.columns, tgt, params_cfg)
        if base_col is None:
            missing.append(tgt)
            continue
        df[tgt] = df[tgt] - df[base_col]
        logger.debug("Station %s: Ziel '%s' auf Residuum gegen '%s' umgestellt.",
                     station_id, tgt, base_col)
    if missing:
        raise ValueError(
            f"Station {station_id}: params.target_transform='nwp_residual', aber fuer "
            f"{missing} ist keine NWP-Baselinespalte im Datensatz. Die entsprechenden "
            "*_nwp-Features muessen in params.icond2_features stehen."
        )
    return df


def _merge_ecmwf(df: pd.DataFrame,
                 config: dict,
                 station_lat: float,
                 station_lon: float,
                 station_id: str,
                 lower: pd.Timestamp | None,
                 upper: pd.Timestamp | None,
                 step_h: float) -> pd.DataFrame:
    """ECMWF-HRES-Strahlungsprognosen anhaengen, wenn konfiguriert.

    Aktiv, sobald ``data.ecmwf_solar_path`` **und** ``params.ecmwf_features`` gesetzt
    sind. Fehlt eines von beiden, bleibt der Datensatz unveraendert — der Solar-Pfad
    laeuft also weiter rein auf ICON-D2.

    Schlaegt das Laden fehl, wird gewarnt statt abgebrochen: ECMWF ist eine
    Zusatzquelle, deren Abdeckung noch waechst (Stand Aug 2026 erst Juli 2023).
    Ein fehlender Monat soll nicht das ganze Training verhindern. Die angeforderten
    Spalten fehlen dann schlicht, und ``dropna()`` weiter unten wuerde alles
    verwerfen — deshalb ist die Warnung deutlich formuliert.
    """
    data_cfg = config.get('data', {})
    params_cfg = config.get('params', {})
    ecmwf_path = data_cfg.get('ecmwf_solar_path')
    ecmwf_features = [str(f) for f in (params_cfg.get('ecmwf_features') or [])]
    if not ecmwf_path or not ecmwf_features:
        return df

    from . import solar_ecmwf as _se

    try:
        ecmwf = _se.load_ecmwf_solar_for_station(
            ecmwf_path=ecmwf_path,
            station_lat=station_lat,
            station_lon=station_lon,
            features=ecmwf_features,
            next_n_grid_points=params_cfg.get('next_n_grid_points_ecmwf', 1),
            starttime_min=lower,
            starttime_max=upper,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Station %s: ECMWF-Solar nicht geladen (%s) — es wird nur "
                       "ICON-D2 verwendet. Die in params.ecmwf_features gelisteten "
                       "Spalten fehlen dann im Datensatz.", station_id, exc)
        return df

    before = len(df)
    df = _se.merge_ecmwf(df, ecmwf, ecmwf_features, int(round(step_h * 60)))
    # Erst hier, nicht schon beim Laden: dni/kt/kd brauchen den Sonnenstand der
    # Station, und der haengt am Zeitstempel des Zieldatensatzes.
    df = _se.add_ecmwf_geometry_features(df, ecmwf_features)
    added = [c for c in df.columns if c.startswith('ecmwf_')]
    n_nan = int(df[added].isna().all(axis=1).sum()) if added else before
    if n_nan:
        logger.warning("Station %s: %d von %d Zeilen ohne ECMWF-Deckung (%.1f %%) — "
                       "die ECMWF-Abdeckung ist kuerzer als der Trainingszeitraum.",
                       station_id, n_nan, before, 100 * n_nan / max(before, 1))
    return df


def _merge_neighbor_stations(df: pd.DataFrame,
                             config: dict,
                             features: dict,
                             stations_df: pd.DataFrame,
                             station_id: str,
                             station_lat: float,
                             station_lon: float,
                             freq: str) -> pd.DataFrame:
    """``<feature>_next_<rang>``-Spalten der nächstgelegenen Stationen anhängen.

    Der Kandidatenpool wird — wie in der Wind-Pipeline — auf ``data.neighbor_pool``
    beschränkt. Ohne diese Beschränkung zöge eine Trainingsstation Nachbarn aus dem
    Val-/Test-Split, deren ``ghi``-Historie genau die zurückgehaltene Zielgröße ist.
    """
    params_cfg = config.get('params', {})
    n_next = params_cfg.get('next_n_stations') or 0
    requested = [f for f in (features.get('known', []) or []) + (features.get('observed', []) or [])
                 if f.endswith('_next')]
    if n_next <= 0 or not requested:
        return df

    base_feats = [f[:-len('_next')] for f in requested]
    candidates = stations_df[stations_df['station_id'] != station_id].copy()

    pool = config.get('data', {}).get('neighbor_pool')
    if pool:
        pool = {str(s) for s in pool}
        candidates = candidates[candidates['station_id'].astype(str).isin(pool)]
        if len(candidates) < n_next:
            logger.warning(
                "Station %s: nur %d zulässige Nachbarkandidaten (angefordert: %d).",
                station_id, len(candidates), n_next,
            )

    candidates['_distance_km'] = candidates.apply(
        lambda row: geodesic((station_lat, station_lon),
                             (row['latitude'], row['longitude'])).km, axis=1
    )
    nearest = candidates.nsmallest(n_next, '_distance_km')

    data_path = config['data']['path']
    for rank, (_, row) in enumerate(nearest.iterrows(), start=1):
        neighbor_id = str(row['station_id'])
        neighbor_file = os.path.join(data_path, f'Station_{neighbor_id}.parquet')
        if not os.path.exists(neighbor_file):
            logger.warning("Nachbarstation %s: Datei nicht gefunden, übersprungen.", neighbor_id)
            continue
        df_n = load_station_measurements(
            neighbor_file, freq=freq,
            time_label=params_cfg.get('measurement_time_label', 'right'),
            lower=df['timestamp'].min(), upper=df['timestamp'].max(),
            max_nan_frac=params_cfg.get('max_nan_frac', 0.5),
        )
        for base in base_feats:
            if base not in df_n.columns:
                logger.warning("Nachbarstation %s: Feature '%s' fehlt, übersprungen.",
                               neighbor_id, base)
                continue
            col = f'{base}_next_{rank}'
            df = df.merge(df_n[[base]].rename(columns={base: col}),
                          left_on='timestamp', right_index=True, how='left')
    return df


def _select_columns(df: pd.DataFrame,
                    features: dict,
                    target_cols: list[str],
                    static_features: list[str],
                    params_cfg: dict) -> pd.DataFrame:
    """Auf die angeforderten Features + Zielspalten reduzieren.

    Die Feature-Namen matchen — wie in der Wind-Pipeline — optional mit
    Gitterpunkt-/Nachbar-Suffix (``ghi_nwp`` → ``ghi_nwp_1``, ``ghi_next`` → ``ghi_next_1``).
    """
    requested = list(dict.fromkeys(
        (features.get('known', []) or []) + (features.get('observed', []) or [])
    ))
    if not requested:
        return df

    patterns = [re.compile(rf'^{re.escape(feat)}(_[A-Z0-9]+)?$') for feat in requested]
    keep = [col for col in df.columns if any(p.match(col) for p in patterns)]

    for tgt in target_cols:
        if tgt not in keep:
            keep.append(tgt)
    for feat in static_features:
        if feat in df.columns and feat not in keep:
            keep.append(feat)

    # NWP-Baseline-Spalte für Skill_NWP erhalten, auch wenn sie kein Modell-Feature ist
    # nwp_baseline_col darf ein Dict {Zielgroesse: Spalte} sein — dann muessen ALLE
    # Baselinespalten erhalten bleiben, nicht nur eine.
    from .preprocessing import nwp_baseline_prefixes
    for baseline in nwp_baseline_prefixes(params_cfg.get('nwp_baseline_col'), 'ghi_nwp_1'):
        for cand in (baseline, f'{baseline}_1'):
            if cand in df.columns and cand not in keep:
                keep.append(cand)

    # Clear-Sky-Bezugsspalten erhalten, wenn das Ziel als Clear-Sky-Index vorliegt.
    # eval._evaluate_single_target rechnet damit vor den Metriken nach W/m² zurueck;
    # ohne sie stuende der RMSE in k-Einheiten und die NWP-Baseline in W/m².
    # Bewusst hier statt in known_features: so bleibt die Spalte erhalten, ohne
    # Modell-Eingang zu werden — sonst haette der Clear-Sky-Lauf ein Feature mehr
    # als die Vergleichslaeufe und der Unterschied waere nicht mehr sauber zuzuordnen.
    if params_cfg.get('target_transform') == 'clearsky_index':
        cs_map = {'ghi': 'ghi_clearsky', 'dhi': 'dhi_clearsky',
                  'bhi': 'dni_clearsky', 'dni': 'dni_clearsky'}
        for tgt in target_cols:
            cand = cs_map.get(tgt)
            if cand and cand in df.columns and cand not in keep:
                keep.append(cand)

    missing = [c for c in keep if c not in df.columns]
    if missing:
        logger.warning("Angeforderte Spalten fehlen im Datensatz: %s", missing)
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    return df[keep]
