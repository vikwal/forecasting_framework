#!/usr/bin/env python3
"""Horizontüberhöhung und Sky-View-Faktor je Station aus SRTM3.

Die einzige topografische Größe mit einem *direkten* Mechanismus für
Einstrahlung auf die Horizontale: ein erhöhter Horizont verdeckt einen Teil des
Himmels (trifft die Diffusstrahlung) und verschiebt Sonnenauf- und -untergang
(trifft die Direktstrahlung). ``slope`` und ``aspect`` aus
``topo_features.csv`` tun das nicht — sie beschreiben die Neigung des Geländes,
nicht die des Sensors, und ein Pyranometer liegt waagerecht.

Berechnet je Station und Azimut den Winkel des höchsten Geländepunkts über der
Horizontalen und leitet daraus drei Kennzahlen ab:

``svf``
    Sky-View-Faktor für eine horizontale Fläche,
    ``SVF = mean_φ cos²(θ(φ))``. Herleitung: sichtbar ist der Himmel vom
    Horizontwinkel θ bis zum Zenit; das über die Zenitdistanz z gewichtete
    Integral ``∫₀^{90°−θ} cos z · sin z dz = ½cos²θ`` normiert auf den
    ungestörten Wert ½. 1.0 = freier Horizont.

``horizon_mean``
    Mittlerer Horizontwinkel über alle Azimute, in Grad.

``horizon_solar``
    Mittlerer Horizontwinkel über den Azimutsektor 90°…270° (Ost über Süd nach
    West) — der Bereich, in dem in Deutschland die Sonne steht. Für die
    Direktstrahlung die aussagekräftigere Zahl als das Rundummittel, weil ein
    hoher Nordhorizont die Einstrahlung praktisch nicht berührt.

Berücksichtigt Erdkrümmung und Standardrefraktion über den effektiven
Erdradius ``R/0.87``: ein Punkt in Entfernung d liegt um ``d²/(2·R_eff)``
scheinbar tiefer. Ohne das überschätzt die Rechnung ferne Horizonte.

**Grenze der Datenquelle.** SRTM3 löst ~90 m auf und ist ein
Oberflächenmodell aus dem Jahr 2000. Nahverschattung durch Gebäude, Masten
oder Bewuchs ist darin nicht verlässlich enthalten. Die Zahlen beschreiben den
*Geländehorizont*, nicht den tatsächlichen Sichthorizont des Sensors.

Aufruf:
    frcst/bin/python scripts/make_horizon_features.py
    frcst/bin/python scripts/make_horizon_features.py --stations 00183 00427
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: Erdradius und Refraktion. 0.87 ist der übliche Koeffizient für
#: Standardatmosphäre (Refraktion lässt den Horizont scheinbar weiter wirken).
R_EARTH = 6_371_000.0
R_EFF = R_EARTH / 0.87

#: Azimutauflösung. 5° = 72 Richtungen; feiner ändert SVF in flachem Gelände
#: nur in der dritten Nachkommastelle.
AZIMUT_SCHRITT = 5.0

#: Profil entlang eines Azimuts. 100 m Schrittweite liegt über der
#: SRTM3-Auflösung (~90 m), es wird also nichts übersprungen. 20 km Reichweite:
#: darüber hinaus müsste ein Berg schon > 300 m über der Station liegen, um den
#: Horizont nach Krümmungskorrektur überhaupt noch anzuheben.
SCHRITT_M = 100.0
REICHWEITE_M = 20_000.0

#: Sektor, in dem in Deutschland (47–55° N) die Sonne steht.
SOLAR_SEKTOR = (90.0, 270.0)

VOID = -32768  # SRTM-Fehlwert


def _kachel_array(datei) -> np.ndarray:
    """Eine SRTM-Kachel als (n, n) float-Array, Voids als NaN."""
    n = datei.square_side
    arr = np.frombuffer(datei.data, dtype='>i2').astype(np.float32)
    arr = arr[: n * n].reshape(n, n)
    return np.where(arr == VOID, np.nan, arr)


class Gelaende:
    """Höhenabfrage über SRTM3 mit kachelweisem numpy-Cache."""

    def __init__(self, cache_dir: str | None = None):
        import srtm
        self._src = srtm.get_data(local_cache_dir=cache_dir) if cache_dir else srtm.get_data()
        self._kacheln: dict = {}

    def hoehe(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Höhen für Punktfelder, nearest neighbour. NaN, wo keine Kachel greift."""
        out = np.full(lat.shape, np.nan, dtype=np.float32)
        # Kachel = ganzzahliger Grad-Block; gruppiert abfragen statt punktweise.
        klat, klon = np.floor(lat).astype(int), np.floor(lon).astype(int)
        for kl, kn in {(a, b) for a, b in zip(klat.ravel(), klon.ravel())}:
            m = (klat == kl) & (klon == kn)
            arr = self._hole(kl, kn)
            if arr is None:
                continue
            n = arr.shape[0]
            # Zeile 0 ist der NORDrand der Kachel.
            row = np.clip(((kl + 1 - lat[m]) * (n - 1)).round().astype(int), 0, n - 1)
            col = np.clip(((lon[m] - kn) * (n - 1)).round().astype(int), 0, n - 1)
            out[m] = arr[row, col]
        return out

    def _hole(self, klat: int, klon: int):
        key = (klat, klon)
        if key not in self._kacheln:
            try:
                datei = self._src.get_file(float(klat) + 0.5, float(klon) + 0.5)
                self._kacheln[key] = _kachel_array(datei) if datei is not None else None
            except Exception:
                self._kacheln[key] = None
        return self._kacheln[key]


def horizont(gelaende: Gelaende, lat: float, lon: float) -> dict:
    """Horizontwinkel je Azimut und die drei abgeleiteten Kennzahlen."""
    azimute = np.arange(0.0, 360.0, AZIMUT_SCHRITT)
    dist = np.arange(SCHRITT_M, REICHWEITE_M + SCHRITT_M, SCHRITT_M)

    # Stationshöhe aus DERSELBEN Quelle wie das Profil. Die barometrische Höhe
    # aus stations_master.csv wäre inkonsistent — eine Differenz von wenigen
    # Metern zwischen Bezugspunkt und Profil verschiebt den Horizontwinkel in
    # der Nahzone um Grad.
    h0 = float(gelaende.hoehe(np.array([lat]), np.array([lon]))[0])
    if not np.isfinite(h0):
        return {}

    az_rad = np.deg2rad(azimute)[:, None]
    d = dist[None, :]
    # Lokale Ebenennäherung; auf 20 km liegt ihr Fehler weit unter der
    # SRTM-Auflösung.
    dlat = np.rad2deg(d * np.cos(az_rad) / R_EARTH)
    dlon = np.rad2deg(d * np.sin(az_rad) / (R_EARTH * np.cos(np.deg2rad(lat))))
    h = gelaende.hoehe(lat + dlat, lon + dlon)

    # Erdkrümmung + Refraktion: scheinbare Absenkung mit der Entfernung.
    ueberhoehung = h - h0 - d ** 2 / (2 * R_EFF)
    with np.errstate(invalid='ignore'):
        winkel = np.degrees(np.arctan2(ueberhoehung, d))
    winkel = np.where(np.isfinite(winkel), winkel, -90.0)
    theta = np.clip(winkel.max(axis=1), 0.0, None)   # negativ = freie Sicht

    solar = (azimute >= SOLAR_SEKTOR[0]) & (azimute <= SOLAR_SEKTOR[1])
    return {
        'svf': float(np.mean(np.cos(np.deg2rad(theta)) ** 2)),
        'horizon_mean': float(theta.mean()),
        'horizon_solar': float(theta[solar].mean()),
        'horizon_max': float(theta.max()),
        'srtm_elevation': h0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stations-master', default=str(REPO / 'data/stations_master.csv'))
    ap.add_argument('--out', default=str(REPO / 'data/horizon_features.csv'))
    ap.add_argument('--stations', nargs='*', help='nur diese IDs (Default: alle)')
    ap.add_argument('--srtm-cache', default=None, help='Kachelverzeichnis (Default: ~/.srtm)')
    args = ap.parse_args()

    sm = pd.read_csv(args.stations_master, dtype={'station_id': str})
    if args.stations:
        sm = sm[sm['station_id'].isin(args.stations)]
    print(f'{len(sm)} Stationen, {int(360 / AZIMUT_SCHRITT)} Azimute, '
          f'{REICHWEITE_M / 1000:.0f} km Reichweite in {SCHRITT_M:.0f}-m-Schritten')

    gelaende = Gelaende(args.srtm_cache)
    zeilen, fehlend = [], []
    for i, r in enumerate(sm.itertuples(index=False), start=1):
        werte = horizont(gelaende, float(r.latitude), float(r.longitude))
        if not werte:
            fehlend.append(r.station_id)
            continue
        zeilen.append({'station_id': r.station_id, **werte})
        if i % 25 == 0 or i == len(sm):
            print(f'  {i}/{len(sm)}')

    out = pd.DataFrame(zeilen).set_index('station_id').sort_index()
    out.round(5).to_csv(args.out)
    print(f'\ngeschrieben: {args.out}  ({len(out)} Stationen)')
    if fehlend:
        print(f'ohne SRTM-Abdeckung: {fehlend}')

    print('\nVerteilung:')
    print(out[['svf', 'horizon_mean', 'horizon_solar', 'horizon_max']]
          .describe().loc[['mean', 'std', 'min', '50%', 'max']].round(4).to_string())
    eng = out.nsmallest(8, 'svf')[['svf', 'horizon_mean', 'horizon_solar', 'srtm_elevation']]
    print('\nEngster Horizont (kleinster SVF):')
    print(eng.round(4).to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
