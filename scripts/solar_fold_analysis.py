#!/usr/bin/env python3
"""Kennzahlen der Solar-TFT-Fold-Laeufe — Datenaufbereitung und Aggregate.

Traegt die Rohvorhersagen der drei Folds zusammen und reichert sie um die
Groessen an, entlang derer sich Solarprognosefehler ueberhaupt erklaeren lassen:
Sonnenstand, Clear-Sky-Strahlung und daraus der **Clear-Sky-Index** kt der
Messung. kt ist das Solar-Pendant zur Windklasse der Wind-Auswertung
(``scripts/eval_testyear.py``, Abbildung 05): er trennt wolkenlose von bedeckten
und von wechselhaften Situationen, und genau an dieser Achse entscheidet sich,
ob eine Nachbearbeitung gegen ICON-D2 gewinnt.

Die drei Fold-Zielmengen sind **disjunkt** (rotierende raeumliche Folds,
``configs/solar_folds.yaml``): zusammengenommen decken sie alle 62 Poolstationen
ab, jede bewertet von einem Modell, das sie nie im Training gesehen hat. Die
Vereinigung ist damit eine zero-shot-Auswertung ueber den ganzen Pool.

Aufruf als Modul (``report_solar_folds.py`` tut das) oder direkt:
    frcst/bin/python scripts/solar_fold_analysis.py --check
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.solar import solar_geometry, clearsky_index  # noqa: E402
from geostatistics.stdrun.make_stdhp_figures import norm_station  # noqa: E402

RAW = REPO / "data/raw_preds"
STATIONS = REPO / "data/stations_master.csv"
ZIELE = ("ghi", "dhi")
SCHRITT_MIN = 30

#: Clear-Sky-Index der MESSUNG, kt = GHI / GHI_clearsky. Die Grenzen trennen die
#: vier Regime, die in der Solarprognose unterschiedliche Fehlerquellen haben:
#: bedeckt (Modell und NWP liegen beide niedrig, absolute Fehler klein),
#: trübe/wechselhaft (die schwierigste Klasse — Wolkenfelder, die kein NWP auf
#: 2 km aufloest), heiter und wolkenlos (NWP ist dort ohnehin gut).
#:
#: Die Grenzen sind an der gemessenen Verteilung ausgerichtet (Median 0.705 ueber
#: alle Tagesschritte), nicht an Lehrbuchwerten. Zum Niveau: ``ghi_clearsky``
#: stammt aus pvlib-Ineichen mit Linke-Truebung aus der Klimatologie — dieselbe
#: Funktion, die auch das Feature im Modell-Input erzeugt (utils.solar.
#: solar_geometry). Sie liegt an sehr klaren Tagen etwas zu niedrig; zusammen mit
#: Cloud Enhancement liegen rund 11 % der Tagesschritte ueber kt = 1.1. Der
#: Zeitbezug ist geprueft: ein Verschiebungstest ueber ±60 min hat sein Optimum
#: bei 0 (q99 von kt minimal), die Intervallmitten-Konvention passt also.
KT_GRENZEN = [0.0, 0.3, 0.6, 0.85, 1.6]
KT_LABELS = ["bedeckt\n(kt<0.3)", "trüb\n(0.3–0.6)", "heiter\n(0.6–0.85)", "klar\n(kt>0.85)"]

#: Gemessene Einstrahlung in Klassen — das direkte Pendant zu den Windklassen der
#: Wind-Auswertung (scripts/eval_testyear.py, Abbildung 05).
GHI_GRENZEN = [0, 50, 150, 300, 500, 700, 2000]
GHI_LABELS = ["0–50", "50–150", "150–300", "300–500", "500–700", ">700"]

#: Nur Situationen mit nennenswerter Einstrahlungsmoeglichkeit. Nachtwerte sind
#: echte Nullen, die jede Fehlerstatistik nach unten ziehen, ohne dass irgendein
#: Modell dort etwas leisten koennte; in der Daemmerung ist kt numerisch wertlos.
TAG_CLEARSKY_MIN = 50.0


def lade_fold(fold: int, stem: str = "tft_solar_tft_fold") -> pd.DataFrame:
    """Rohvorhersagen eines Folds, Stations-ID normalisiert."""
    pfad = RAW / f"{stem}{fold}_raw.parquet"
    df = pd.read_parquet(pfad)
    df["station_id"] = df["station_id"].map(norm_station)
    df["fold"] = fold
    return df


def stationsmeta() -> pd.DataFrame:
    meta = pd.read_csv(STATIONS, dtype={"station_id": str})
    meta["station_id"] = meta["station_id"].map(norm_station)
    meta["station_height"] = pd.to_numeric(meta["station_height"], errors="coerce")
    return meta.set_index("station_id")


def _geometrie_je_station(zeiten: pd.DatetimeIndex, lat: float, lon: float,
                          hoehe: float) -> pd.DataFrame:
    """Sonnenstand und Clear-Sky fuer die Zeitstempel EINER Station.

    Nutzt dieselbe Funktion wie das Preprocessing (``utils.solar.solar_geometry``),
    damit ``ghi_clearsky`` hier und im Modell-Input identisch definiert sind —
    Intervallmitte, Ineichen mit Linke-Truebung.
    """
    geo = solar_geometry(zeiten, latitude=lat, longitude=lon, altitude=hoehe,
                         freq=f"{SCHRITT_MIN}min")
    return geo[["solar_zenith", "ghi_clearsky", "dhi_clearsky"]]


def reichere_an(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Zeit-, Sonnenstands- und Regime-Spalten ergaenzen.

    Neue Spalten:
      lead_h     Vorlaufzeit in Stunden (horizon 1 = Laufzeitpunkt, also 0 h)
      run_hour   Laufstunde (06/09/12/15 UTC)
      hour       Stunde der Gueltigkeitszeit, UTC
      month      Kalendermonat der Gueltigkeitszeit
      zenith     Sonnenzenitwinkel
      ghi_cs     Clear-Sky-GHI am Ort
      kt         Clear-Sky-Index der Messung (0 bei Nacht)
      kt_klasse  gebinnte Fassung, s. KT_GRENZEN
      tag        True, wenn ghi_cs > TAG_CLEARSKY_MIN
      err/err_nwp/err_pers   Vorhersage minus Messung, je Referenz
    """
    df = df.copy()
    df["lead_h"] = (df["horizon"] - 1) * (SCHRITT_MIN / 60.0)
    df["run_hour"] = df["run_time"].dt.hour
    df["hour"] = df["valid_time"].dt.hour
    df["month"] = df["valid_time"].dt.month

    teile = []
    for sid, teil in df.groupby("station_id", sort=False):
        if sid not in meta.index:
            raise KeyError(f"Station {sid} fehlt in {STATIONS.name}")
        zeile = meta.loc[sid]
        zeiten = pd.DatetimeIndex(teil["valid_time"].unique()).sort_values()
        geo = _geometrie_je_station(zeiten, float(zeile["latitude"]),
                                    float(zeile["longitude"]),
                                    float(zeile["station_height"]))
        teil = teil.join(geo, on="valid_time")
        teile.append(teil)
    df = pd.concat(teile, ignore_index=True)

    df = df.rename(columns={"ghi_clearsky": "ghi_cs"})
    # kt gegen die jeweils passende Clear-Sky-Groesse: GHI gegen ghi_cs, DHI
    # ebenfalls gegen ghi_cs — der Diffusanteil hat keinen eigenen sinnvollen
    # Index (dhi/dhi_cs wird bei Bewoelkung > 1 und ist kein Truebungsmass).
    df["kt"] = clearsky_index(df["gt"].where(df["target"] == "ghi", np.nan),
                              df["ghi_cs"], min_clearsky=TAG_CLEARSKY_MIN)
    # Fuer DHI-Zeilen kt aus der GHI-Zeile desselben (Station, run, horizon)
    # uebernehmen, damit beide Zielgroessen im selben Regime einsortiert werden.
    schluessel = ["station_id", "run_time", "horizon"]
    kt_ghi = (df.loc[df["target"] == "ghi", schluessel + ["kt"]]
                .rename(columns={"kt": "kt_ref"}))
    df = df.merge(kt_ghi, on=schluessel, how="left")
    df["kt"] = df["kt_ref"].fillna(df["kt"])
    df = df.drop(columns=["kt_ref"])

    # Dasselbe fuer die PROGNOSTIZIERTE Lage: kt_nwp = ICON-Prognose / Clear-Sky.
    # kt oben bedingt auf die Wahrheit und zeigt deshalb, wo die Prognose danebenlag;
    # kt_nwp ist vorab bekannt und beantwortet die operative Frage — welcher
    # Vorhersage kann ich trauen, bevor ich die Messung kenne.
    df["kt_nwp"] = clearsky_index(df["nwp_ref"].where(df["target"] == "ghi", np.nan),
                                  df["ghi_cs"], min_clearsky=TAG_CLEARSKY_MIN)
    kt_nwp_ghi = (df.loc[df["target"] == "ghi", schluessel + ["kt_nwp"]]
                    .rename(columns={"kt_nwp": "kt_nwp_ref"}))
    df = df.merge(kt_nwp_ghi, on=schluessel, how="left")
    df["kt_nwp"] = df["kt_nwp_ref"].fillna(df["kt_nwp"])
    df = df.drop(columns=["kt_nwp_ref"])

    df["kt_klasse"] = pd.cut(df["kt"], bins=KT_GRENZEN, labels=KT_LABELS, right=False)
    df["kt_nwp_klasse"] = pd.cut(df["kt_nwp"], bins=KT_GRENZEN, labels=KT_LABELS, right=False)
    # Einstrahlungsklasse der Messung, je Zielgroesse auf deren eigener Skala.
    df["ghi_klasse"] = pd.cut(df["gt"], bins=GHI_GRENZEN, labels=GHI_LABELS, right=False)
    df["tag"] = df["ghi_cs"] > TAG_CLEARSKY_MIN
    df["err"] = df["pred"] - df["gt"]
    df["err_nwp"] = df["nwp_ref"] - df["gt"]
    df["err_pers"] = df["pers_ref"] - df["gt"]
    return df


# ---------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------

def _rmse(x: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def metriken(teil: pd.DataFrame) -> pd.Series:
    """RMSE, MAE, Bias und Skill gegen ICON-D2 und Persistenz fuer eine Gruppe."""
    rmse_m = _rmse(teil["err"])
    rmse_n = _rmse(teil["err_nwp"])
    pers = teil["err_pers"].dropna()
    rmse_p = _rmse(pers) if len(pers) else np.nan
    return pd.Series({
        "n": len(teil),
        "rmse": rmse_m,
        "mae": float(np.mean(np.abs(teil["err"]))),
        "bias": float(np.mean(teil["err"])),
        "rmse_nwp": rmse_n,
        "bias_nwp": float(np.mean(teil["err_nwp"])),
        "rmse_pers": rmse_p,
        "skill_nwp": 1 - rmse_m / rmse_n if rmse_n > 0 else np.nan,
        "skill_pers": 1 - rmse_m / rmse_p if rmse_p and rmse_p > 0 else np.nan,
        "anteil_besser": float(np.mean(np.abs(teil["err"]) < np.abs(teil["err_nwp"]))),
    })


def nach(df: pd.DataFrame, spalten) -> pd.DataFrame:
    """Metriken je Gruppe, gruppiert nach `spalten`."""
    if isinstance(spalten, str):
        spalten = [spalten]
    return (df.groupby(spalten, observed=True)
              .apply(metriken, include_groups=False)
              .reset_index())


def lade_alles(folds=(1, 2, 3), stem: str = "tft_solar_tft_fold",
               nur_tag: bool = True) -> pd.DataFrame:
    """Alle Folds laden, anreichern und optional auf Tageslicht beschraenken."""
    meta = stationsmeta()
    df = pd.concat([reichere_an(lade_fold(f, stem), meta) for f in folds],
                   ignore_index=True)
    return df[df["tag"]].copy() if nur_tag else df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="nur Fold 1, Kurzbericht")
    args = ap.parse_args()

    folds = (1,) if args.check else (1, 2, 3)
    daten = lade_alles(folds=folds)
    print(f"Zeilen (Tageslicht): {len(daten):,}")
    print(f"Stationen: {daten['station_id'].nunique()}, Ziele: {sorted(daten['target'].unique())}")
    print(f"kt-Verteilung:\n{daten['kt_klasse'].value_counts(normalize=True).round(3)}")
    print(nach(daten, ["target"]).to_string(index=False))
