#!/usr/bin/env python3
"""Gemeinsame Auswertung der Solar-Architekturen: DCRNN gegen TFT, Fold 1.

Beide Pfade bringen ihre eigene Auswertung mit, und beide sind fuer sich
richtig — aber sie messen nicht dasselbe: der TFT filtert imputierte Ziele
ueber ``eval.exclude_imputed``, rechnet im Residuumsraum und mittelt ueber
seine eigenen Laeufe; das DCRNN fuehrt seine Metriken ueber Array-Positionen
und seinen eigenen Laufbestand. Gegeneinander gehalten waeren das zwei Zahlen
ueber zwei Stichproben.

Dieses Skript nimmt deshalb von beiden Seiten nur die **Rohvorhersagen** und
legt Filter, Baselines und Aggregation einmal darueber:

1. Beobachtungsmaske aus der Rohmessung (``load_station_measurements``, genau
   die Funktion, mit der auch der GNN-Pfad seine Maske bildet): NaN = keine
   echte Messung, also imputiert oder Nachtnull.
2. Zeitachse belegen statt glauben: beide Seiten werden ueber ±2 Schritte
   verschoben, das RMSE-Minimum muss bei 0 liegen, sonst bricht das Skript ab
   (``--weich`` warnt nur). Die DCRNN-Seite laeuft dabei gegen die Rohmessung,
   die TFT-Seite ueber ihr Residuum gegen die NWP-Spalte des DCRNN — auf ``gt``
   ginge es dort nicht, das entsteht hier erst durch Nachschlagen. Genau dieser
   Test hat den 30-min-Versatz der v4-Laeufe gefunden (docs/handoff.md §3).
3. Gemeinsame Menge ueber ``(station_id, target, run_time, horizon)``, damit
   beide Architekturen auf denselben Zeilen stehen.
4. RMSE je Station, dann Stationsmittel, dann Wilcoxon (gepaart ueber die
   Stationen) mit Holm-Korrektur — dasselbe Vorgehen wie bei der
   Wind-Testauswertung in ``scripts/eval_testmode.py``.

Zur TFT-Seite: im Ergebnis-Pickle stehen ``pred``/``true`` im **Residuumsraum**
(``target_transform: nwp_residual``), die NWP-Baseline ist dort konstant 0 und
die Persistenz NaN. Die absolute Skala wird deshalb rekonstruiert:

    nwp      = ziel(vt) - true_residuum          (ziel = gefuellte Messreihe)
    pred_abs = pred_residuum + nwp
    gt_abs   = ziel(vt)

Das ist keine Naeherung, sondern die Umkehrung der Transformation — und sie
ist pruefbar: das rekonstruierte ``nwp`` muss die ``nwp_ref``-Spalte des DCRNN
treffen, die aus derselben ICON-D2-Spalte am selben naechsten Gitterpunkt
kommt. Genau das macht der Verschiebungstest der TFT-Seite.

Aufruf:
    frcst/bin/python scripts/eval_solar_arch.py \\
        --dcrnn v5 --tft results/solar/cl_m-tft_..._tft_f1_....pkl
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geostatistics.stdrun.make_stdhp_figures import norm_station  # noqa: E402
from geostatistics.train_stgnn2 import load_station_measurements  # noqa: E402
from utils.imputation import impute_meas_raw_solar  # noqa: E402

RAW = REPO / "data/raw_preds"
OUT = REPO / "data/test_results"
ARME = ["a", "base", "nomeas", "nograph", "idw_alt", "nwp_hist"]
ZIELE = ["ghi", "dhi"]
SCHRITT_MIN = 30
#: Spalten, auf denen beide Seiten zusammengefuehrt werden.
SCHLUESSEL = ["station_id", "target", "run_time", "horizon"]


# ─────────────────────────────────────────────────────────────────────────────
# Messreihe, Maske
# ─────────────────────────────────────────────────────────────────────────────
def lade_messreihen(ids: list[str], roh_pfad: Path, interpol_pfad: Path,
                    stations_master: str) -> tuple[dict, dict]:
    """Rohmessung und gefuellte Reihe je Station.

    Returns:
        (beobachtet, ziel) — beide ``{station_id: DataFrame(index=timestamp,
        columns=ZIELE)}``. ``beobachtet`` ist bool (True = echte Messung),
        ``ziel`` traegt die Werte nach derselben Lueckenfuellung, die auch das
        Training gesehen hat.

    Die Maske entsteht VOR dem Fuellen — danach ist die Information weg, weil
    Modellwerte und Nachtnullen in dieselben Zellen geschrieben werden. Das ist
    dieselbe Konstruktion wie in ``train_dcrnn.py``; Resampling und
    Zeitlabel-Korrektur kommen aus ``load_station_measurements``, damit hier
    nicht eine zweite, leicht abweichende Regel entsteht (die DWD-Zeitstempel
    markieren das Intervallende, und 10-min-Messungen nesten nicht in 30-min-
    Intervalle — beides erledigt die Funktion).
    """
    meas, ts = load_station_measurements(
        str(roh_pfad), ids, cols=ZIELE, freq=f"{SCHRITT_MIN}min",
        use_case="solar", stations_master=stations_master,
    )
    beobachtet = {sid: pd.DataFrame(~np.isnan(meas[:, i, :]), index=ts, columns=ZIELE)
                  for i, sid in enumerate(ids)}
    gefuellt, _ = impute_meas_raw_solar(
        meas.copy(), ids, ts, list(ZIELE), str(interpol_pfad), fill_night=True,
    )
    ziel = {sid: pd.DataFrame(gefuellt[:, i, :], index=ts, columns=ZIELE)
            for i, sid in enumerate(ids)}
    return beobachtet, ziel


def _reihe(tab: dict, sid: str, target: str) -> pd.Series:
    return tab[sid][target]


def spalte_aus_reihen(df: pd.DataFrame, tab: dict, zeitspalte: str,
                      versatz_schritte: int = 0) -> np.ndarray:
    """Werte aus ``tab`` an ``(station_id, target, <zeitspalte>)`` nachschlagen.

    Der Zeitindex der Reihen ist ein lueckenloses ``SCHRITT_MIN``-Raster, die
    Position also ausrechenbar — ``reindex`` je Gruppe kostet auf sechs Armen
    à 5.9 Mio. Zeilen ein Vielfaches davon. Das Raster wird einmal geprueft,
    damit die Annahme nicht stillschweigend trägt.
    """
    out = np.full(len(df), np.nan)
    # tz-aware -> naiv (alles ist UTC), sonst warnt numpy beim Cast und wird
    # ihn kuenftig verweigern.
    zeiten = df[zeitspalte].dt.tz_localize(None).to_numpy().astype("datetime64[m]")
    zeiten = zeiten + np.timedelta64(SCHRITT_MIN * versatz_schritte, "m")
    for (sid, target), idx in df.groupby(["station_id", "target"], observed=True).indices.items():
        if sid not in tab:
            continue
        s = _reihe(tab, sid, target)
        index = s.index.to_numpy().astype("datetime64[m]")
        if len(index) > 1:
            schritte = np.diff(index).astype("timedelta64[m]").astype(int)
            if schritte.min() != SCHRITT_MIN or schritte.max() != SCHRITT_MIN:
                raise SystemExit(
                    f"[FATAL] Zeitindex von {sid}/{target} ist kein lueckenloses "
                    f"{SCHRITT_MIN}-min-Raster — die Positionsrechnung waere falsch.")
        pos = ((zeiten[idx] - index[0]).astype(int) // SCHRITT_MIN)
        gueltig = (pos >= 0) & (pos < len(index))
        werte = s.to_numpy(dtype=float)
        ziel_idx = idx[gueltig]
        out[ziel_idx] = werte[pos[gueltig]]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Die beiden Quellen
# ─────────────────────────────────────────────────────────────────────────────
def lade_dcrnn(stem: str, ziel: dict) -> pd.DataFrame:
    """Ein DCRNN-Roh-Parquet als long-Frame mit selbst gerechneter ``valid_time``.

    ``valid_time`` wird aus ``run_time`` und ``horizon`` neu gebildet und gegen
    die gespeicherte Spalte geprueft: die Laeufe vor der Reparatur vom
    15.09.2026 fuehren dort ``run + horizon * 1 h`` statt ``* 30 min``, was jeden
    Join still um Stunden verschoebe. Massgeblich ist ``horizon`` — die Achse,
    an der Modell, Ziel und NWP im Array haengen.
    """
    pfad = RAW / f"{stem}_raw.parquet"
    if not pfad.exists():
        raise SystemExit(f"[FATAL] Roh-Parquet fehlt: {pfad}")
    df = pd.read_parquet(pfad)
    if "target" not in df.columns:
        df["target"] = ZIELE[0]
    df["station_id"] = df["station_id"].map(norm_station)
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    # horizon ist 1-basiert, Schritt 1 ist bei Solar Lead 0 (linksbuendig auf
    # der Laufzeit) — siehe geostatistics/shared/resolution.lead0_offset.
    vt_neu = df["run_time"] + pd.to_timedelta((df["horizon"] - 1) * SCHRITT_MIN, unit="m")
    if "valid_time" in df.columns:
        abw = (pd.to_datetime(df["valid_time"], utc=True) != vt_neu).mean()
        if abw > 0:
            print(f"    Hinweis: valid_time-Spalte weicht in {abw:6.1%} der Zeilen ab "
                  f"(Lauf vor der Zeitachsen-Reparatur?) — es gilt horizon.")
    df["valid_time"] = vt_neu
    # ``gt`` bleibt der Wert AUS DER DATEI — also der, gegen den das Modell
    # bewertet wurde. Ihn hier aus der Zielreihe neu zu setzen waere bequem,
    # machte den Verschiebungstest aber tautologisch: er verglichen die Reihe
    # dann mit sich selbst und meldete an jeder Zeitachse eine 0.
    return df[SCHLUESSEL + ["valid_time", "pred", "gt", "nwp_ref", "pers_ref"]]


def lade_tft(pkl: Path, ziel: dict, stationen: set[str]) -> pd.DataFrame:
    """TFT-Ergebnis-Pickle als long-Frame in physikalischen Einheiten.

    Die Rueckrechnung aus dem Residuumsraum braucht die Zielreihe; Positionen,
    an denen sie fehlt, fallen ohnehin durch die Beobachtungsmaske.
    """
    with open(pkl, "rb") as fh:
        erg = pickle.load(fh)
    teile = []
    for (datei, target), block in erg["predictions"].items():
        sid = norm_station(datei)
        if sid not in stationen or target not in ZIELE:
            continue
        pred, true = block["pred"], block["true"]
        h = np.array([int(c.split("+")[1]) for c in pred.columns])
        n = len(pred.index)
        teile.append(pd.DataFrame({
            "station_id": sid,
            "target": target,
            "run_time": pd.DatetimeIndex(np.repeat(pred.index.to_numpy(), len(h))),
            "horizon": np.tile(h, n),
            "pred_res": pred.to_numpy().ravel(),
            "true_res": true.to_numpy().ravel(),
        }))
    if not teile:
        raise SystemExit(f"[FATAL] Keine passenden Stationen in {pkl}")
    df = pd.concat(teile, ignore_index=True)
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    df["valid_time"] = df["run_time"] + pd.to_timedelta((df["horizon"] - 1) * SCHRITT_MIN, unit="m")

    # Residuum -> Absolutskala. nwp faellt dabei als Nebenprodukt ab und ist
    # damit pruefbar (--kreuztest).
    zielwert = spalte_aus_reihen(df, ziel, "valid_time")
    df["nwp_ref"] = zielwert - df["true_res"].to_numpy()
    df["pred"] = df["pred_res"].to_numpy() + df["nwp_ref"].to_numpy()
    df["gt"] = zielwert
    # Persistenz wie im GNN-Pfad: der letzte Schritt VOR dem Vorhersagefenster,
    # ueber alle Leads konstant (evaluation._pers_ref). Im Pickle steht dafuer
    # NaN, der TFT fuehrt seine Persistenz an anderer Stelle.
    df["pers_ref"] = spalte_aus_reihen(df, ziel, "run_time", versatz_schritte=-1)
    # true_res bleibt erhalten: der Verschiebungstest der TFT-Seite laeuft
    # ueber das Residuum, nicht ueber das rekonstruierte gt.
    return df[SCHLUESSEL + ["valid_time", "pred", "gt", "nwp_ref", "pers_ref", "true_res"]]


# ─────────────────────────────────────────────────────────────────────────────
# Zeitachse belegen
# ─────────────────────────────────────────────────────────────────────────────
def verschiebungstest(df: pd.DataFrame, beobachtet: dict, ziel: dict,
                      name: str, spanne: int = 2, streng: bool = True) -> pd.DataFrame:
    """RMSE von ``gt`` gegen die Messreihe bei ±``spanne`` Schritten Versatz.

    Liegt das Minimum nicht bei 0, ist die Zeitachse verschoben — und zwar
    still: alle Zahlen sehen weiter plausibel aus. Deshalb Abbruch statt
    Warnung.
    """
    zeilen = []
    maske = spalte_aus_reihen(df, beobachtet, "valid_time")
    echt = np.nan_to_num(maske, nan=0.0).astype(bool)
    gt = df["gt"].to_numpy()
    for k in range(-spanne, spanne + 1):
        ref = spalte_aus_reihen(df, ziel, "valid_time", versatz_schritte=k)
        m = echt & ~np.isnan(ref) & ~np.isnan(gt)
        zeilen.append(dict(offset_schritte=k, offset_min=k * SCHRITT_MIN,
                           rmse=float(np.sqrt(np.mean((gt[m] - ref[m]) ** 2))), n=int(m.sum())))
    t = pd.DataFrame(zeilen)
    bestes = int(t.loc[t["rmse"].idxmin(), "offset_schritte"])
    print(f"  Verschiebungstest {name}:")
    for _, r in t.iterrows():
        mark = "  <<<" if int(r["offset_schritte"]) == bestes else ""
        print(f"    {int(r['offset_min']):+4d} min   RMSE {r['rmse']:9.4f}   n={int(r['n'])}{mark}")
    if bestes != 0:
        msg = (f"[FATAL] {name}: RMSE-Minimum bei {bestes * SCHRITT_MIN:+d} min statt 0. "
               f"Die Zeitachse ist verschoben — keine Auswertung auf verschobenen Reihen.")
        if streng:
            raise SystemExit(msg)
        print(msg)
    return t


def verschiebungstest_tft(tft_roh: pd.DataFrame, dcrnn: pd.DataFrame, ziel: dict,
                          spanne: int = 2, streng: bool = True) -> pd.DataFrame:
    """Zeitachse der TFT-Seite, ueber das Residuum statt ueber ``gt``.

    Auf ``gt`` laesst sich die TFT-Seite nicht pruefen: im Pickle stehen nur
    Residuen, ``gt`` entsteht hier erst durch Nachschlagen in der Zielreihe —
    ein Vergleich gegen dieselbe Reihe waere zirkulaer. Pruefbar ist dagegen
    das Nebenprodukt der Rueckrechnung:

        nwp(k) = ziel(vt + k) - true_residuum

    Nur bei richtigem ``k`` ist das die ICON-D2-Prognose. Gemessen wird gegen
    die ``nwp_ref``-Spalte des DCRNN, die aus derselben ICON-Spalte am selben
    naechsten Gitterpunkt kommt — bei richtiger Ausrichtung also praktisch
    bitgleich sein muss, bei falscher um den Tagesgang daneben liegt.
    """
    g = tft_roh.merge(dcrnn[SCHLUESSEL + ["nwp_ref"]], on=SCHLUESSEL,
                      suffixes=("", "_dcrnn"))
    if g.empty:
        raise SystemExit("[FATAL] TFT und DCRNN haben keine gemeinsame Zeile — "
                         "Stationen, Laufzeiten oder horizon passen nicht zueinander.")
    ref_d = g["nwp_ref_dcrnn"].to_numpy()
    zeilen = []
    for k in range(-spanne, spanne + 1):
        zielwert = spalte_aus_reihen(g, ziel, "valid_time", versatz_schritte=k)
        nwp_k = zielwert - g["true_res"].to_numpy()
        m = ~np.isnan(nwp_k) & ~np.isnan(ref_d)
        zeilen.append(dict(offset_schritte=k, offset_min=k * SCHRITT_MIN,
                           rmse=float(np.sqrt(np.mean((nwp_k[m] - ref_d[m]) ** 2))),
                           anteil_negativ=float((nwp_k[m] < -1).mean()), n=int(m.sum())))
    t = pd.DataFrame(zeilen)
    bestes = int(t.loc[t["rmse"].idxmin(), "offset_schritte"])
    print("  Verschiebungstest TFT (rekonstruiertes NWP gegen DCRNN-nwp_ref):")
    for _, r in t.iterrows():
        mark = "  <<<" if int(r["offset_schritte"]) == bestes else ""
        print(f"    {int(r['offset_min']):+4d} min   RMSE {r['rmse']:9.4f}   "
              f"Anteil nwp<-1: {r['anteil_negativ']:6.4f}   n={int(r['n'])}{mark}")
    if bestes != 0:
        msg = (f"[FATAL] TFT: RMSE-Minimum bei {bestes * SCHRITT_MIN:+d} min statt 0. "
               f"Die beiden Architekturen meinen bei gleichem horizon verschiedene "
               f"Zeitpunkte — kein Vergleich auf verschobenen Reihen.")
        if streng:
            raise SystemExit(msg)
        print(msg)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Filter, Metriken, Signifikanz
# ─────────────────────────────────────────────────────────────────────────────
def filtere(df: pd.DataFrame, beobachtet: dict, name: str) -> pd.DataFrame:
    """Imputierte Zielpositionen elementweise entfernen."""
    maske = spalte_aus_reihen(df, beobachtet, "valid_time")
    echt = np.nan_to_num(maske, nan=0.0).astype(bool)
    brauchbar = echt & ~np.isnan(df["pred"].to_numpy()) & ~np.isnan(df["gt"].to_numpy())
    print(f"  {name:22s} {brauchbar.sum():>9,} von {len(df):>9,} Zeilen "
          f"({100 * brauchbar.mean():5.2f} %) — {(~echt).sum():,} imputiert, "
          f"{int((echt & ~brauchbar).sum()):,} ohne Vorhersage")
    return df.loc[brauchbar].copy()


def rmse_je_station(df: pd.DataFrame, pred_spalte: str = "pred") -> pd.Series:
    e2 = (df[pred_spalte].to_numpy() - df["gt"].to_numpy()) ** 2
    return np.sqrt(pd.Series(e2, index=df.index)
                   .groupby([df["target"], df["station_id"]], observed=True).mean())


def r2_je_station(df: pd.DataFrame, pred_spalte: str = "pred") -> pd.Series:
    """Bestimmtheitsmass je (Zielgroesse, Station) auf der Absolutskala.

    Bezugsgroesse ist das stationseigene Mittel der Messung, passend zur
    RMSE-Aggregation (erst je Station, dann ueber die Stationen mitteln). Ohne
    ``groupby.apply`` gerechnet, damit es auch auf aelteren pandas laeuft.
    """
    key = [df["target"], df["station_id"]]
    y, p = df["gt"], df[pred_spalte]
    ss_res = ((y - p) ** 2).groupby(key, observed=True).sum()
    ss_tot = ((y - y.groupby(key, observed=True).transform("mean")) ** 2
              ).groupby(key, observed=True).sum()
    return 1.0 - ss_res / ss_tot.replace(0.0, np.nan)


def holm(t: pd.DataFrame) -> pd.DataFrame:
    t = t.sort_values("p_raw").reset_index(drop=True)
    m = len(t)
    t["p_holm"] = np.maximum.accumulate(
        [(m - i) * p for i, p in enumerate(t["p_raw"])]).clip(max=1.0)
    return t


def gepaarter_wilcoxon(je_station: dict, labels: list) -> pd.DataFrame:
    """Gepaart ueber die Stationen, je Zielgroesse getrennt."""
    zeilen = []
    for target in ZIELE:
        for A, B in itertools.combinations(labels, 2):
            if A not in je_station or B not in je_station:
                continue
            a, b = je_station[A], je_station[B]
            if target not in a.index.get_level_values(0):
                continue
            d = (a.loc[target] - b.loc[target]).dropna()
            if len(d) < 6:
                continue
            _, p = wilcoxon(d)
            zeilen.append(dict(target=target, A=A, B=B, n=len(d),
                               median_diff=float(d.median()),
                               anteil_A_besser=float((d < 0).mean()), p_raw=float(p)))
    return holm(pd.DataFrame(zeilen)) if zeilen else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dcrnn", default="v5",
                    help="Suffix der DCRNN-Roh-Parquets: data/raw_preds/solar_dcrnn_<suffix>_<arm>_raw.parquet")
    ap.add_argument("--arme", default=",".join(ARME))
    ap.add_argument("--tft", required=True, help="TFT-Ergebnis-Pickle (Fold 1)")
    ap.add_argument("--config", default="configs/solar_dcrnn/config_solar_dcrnn_a_fold1.yaml",
                    help="liefert Stationsliste und Datenpfade")
    ap.add_argument("--keine-verschiebung", action="store_true",
                    help="Verschiebungstest ueberspringen (nur zur Fehlersuche)")
    ap.add_argument("--weich", action="store_true",
                    help="bei verschobener Zeitachse warnen statt abzubrechen")
    ap.add_argument("--out", default=None, help="Praefix der CSV-Ausgaben")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from utils.tools import load_config  # noqa: E402  (braucht DATA_ROOT)
    cfg = load_config(str(REPO / args.config))
    d = cfg["data"]
    stationen = [str(s).zfill(5) for s in d["val_files"]]
    print(f"Fold 1: {len(stationen)} Zielstationen, Ziele {ZIELE}, Schritt {SCHRITT_MIN} min\n")

    print("Messreihen und Beobachtungsmaske …")
    beobachtet, ziel = lade_messreihen(
        stationen, Path(d["path"]), Path(d["interpol_path"]), d["stations_master"])
    anteil = np.mean([beobachtet[s][z].mean() for s in stationen for z in ZIELE])
    print(f"  {anteil:6.2%} der Zeitschritte sind echte Messungen (ueber alle Stationen und Ziele)\n")

    quellen: dict[str, pd.DataFrame] = {}
    print("Rohvorhersagen laden …")
    for arm in args.arme.split(","):
        stem = f"solar_dcrnn_{args.dcrnn}_{arm}"
        quellen[f"DCRNN {arm}"] = lade_dcrnn(stem, ziel)
        print(f"  DCRNN {arm:9s} {len(quellen[f'DCRNN {arm}']):>9,} Zeilen")
    quellen["TFT"] = lade_tft(Path(args.tft), ziel, set(stationen))
    print(f"  TFT       {len(quellen['TFT']):>9,} Zeilen\n")

    if not args.keine_verschiebung:
        print("Zeitachse:")
        erster = next(iter(quellen))
        # DCRNN: gegen die Rohmessung, ``gt`` kommt unveraendert aus dem Parquet.
        verschiebungstest(quellen[erster], beobachtet, ziel, erster, streng=not args.weich)
        # TFT: ueber das Residuum gegen die NWP-Spalte des DCRNN.
        verschiebungstest_tft(quellen["TFT"], quellen[erster], ziel, streng=not args.weich)
        print()
    quellen["TFT"] = quellen["TFT"].drop(columns=["true_res"])

    print("Filter (imputierte Zielpositionen elementweise):")
    quellen = {k: filtere(v, beobachtet, k) for k, v in quellen.items()}

    # Gemeinsame Menge ueber alle Quellen.
    gemeinsam = None
    for v in quellen.values():
        s = v.set_index(SCHLUESSEL).index
        gemeinsam = s if gemeinsam is None else gemeinsam.intersection(s)
    print(f"\nGemeinsame Menge: {len(gemeinsam):,} Zeilen "
          f"({len(gemeinsam) / max(len(next(iter(quellen.values()))), 1):5.1%} der ersten Quelle)")

    je_station, je_station_r2, zeilen = {}, {}, []
    basis: dict[str, pd.Series] = {}     # RMSE der Baselines je (Ziel, Station)
    referenz_gt: tuple[str, np.ndarray] | None = None
    for name, v in quellen.items():
        g = v.set_index(SCHLUESSEL).loc[gemeinsam].reset_index()
        # Beide Seiten muessen auf derselben Menge dieselbe Wahrheit meinen.
        # Taten sie es nicht, verglichen die RMSE-Spalten zwei verschiedene
        # Zielreihen — genau so sah die v4-Tabelle aus, in der die NWP-Baseline
        # der beiden Architekturen um 6.5 W/m² auseinanderlag, obwohl die
        # NWP-Werte selbst bitgleich sind.
        if referenz_gt is None:
            referenz_gt = (name, g["gt"].to_numpy())
        else:
            d = np.abs(g["gt"].to_numpy() - referenz_gt[1])
            if np.nanmax(d) > 1e-6:
                msg = (f"[FATAL] '{name}' und '{referenz_gt[0]}' tragen an denselben "
                       f"(Station, Ziel, Lauf, horizon) verschiedene gt-Werte "
                       f"(max |Δ| = {np.nanmax(d):.4f} W/m², "
                       f"{int((d > 1e-6).sum()):,} Zeilen). Ein RMSE-Vergleich ueber "
                       f"zwei verschiedene Wahrheiten sagt nichts aus.")
                if args.weich:
                    print(msg)
                else:
                    raise SystemExit(msg)
        je_station[name] = rmse_je_station(g)
        r2 = r2_je_station(g)
        je_station_r2[name] = r2
        nwp = rmse_je_station(g, "nwp_ref")
        pers = rmse_je_station(g, "pers_ref")
        r2_nwp = r2_je_station(g, "nwp_ref")
        r2_pers = r2_je_station(g, "pers_ref")
        # Baselines sind ueber alle Quellen identisch (gemeinsame Menge) —
        # einmal aufheben, um daraus die Skills je Station zu bilden.
        basis.setdefault("rmse_nwp", nwp)
        basis.setdefault("rmse_pers", pers)
        for target in ZIELE:
            r = je_station[name].loc[target]
            zeilen.append(dict(
                modell=name, target=target, n_stationen=len(r),
                rmse=float(r.mean()), r2=float(r2.loc[target].mean()),
                rmse_nwp=float(nwp.loc[target].mean()),
                r2_nwp=float(r2_nwp.loc[target].mean()),
                rmse_pers=float(pers.loc[target].mean()),
                r2_pers=float(r2_pers.loc[target].mean()),
                skill_nwp=float(1 - (r / nwp.loc[target]).mean()),
                skill_pers=float(1 - (r / pers.loc[target]).mean()),
                n_punkte=int(len(g) / len(ZIELE)),
            ))
    tab = pd.DataFrame(zeilen).sort_values(["target", "rmse"])
    print("\nStationsmittel des RMSE auf der gemeinsamen Menge (W/m²):")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    sig = gepaarter_wilcoxon(je_station, list(quellen))
    if not sig.empty:
        print("\nGepaart ueber die Stationen (Wilcoxon, Holm-korrigiert), "
              "negative median_diff = A besser:")
        print(sig.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    # ── Streuung des R2 ueber die Stationen ──────────────────────────────
    # Das Stationsmittel allein sagt wenig: R2 misst gegen die stationseigene
    # Varianz, und die haengt bei Strahlung stark an Lage und Bewoelkungsregime.
    r2_tab = pd.DataFrame(je_station_r2)
    print("\nR2 je Station — Streuung (min … max, Spannweite, Standardabweichung):")
    for target in ZIELE:
        t = r2_tab.loc[target]
        print(f"  {target}:")
        for name in t.columns:
            v = t[name].dropna()
            print(f"    {name:16s} Mittel {v.mean():.4f} | Median {v.median():.4f} | "
                  f"{v.min():.4f} ({v.idxmin()}) … {v.max():.4f} ({v.idxmax()}) | "
                  f"Spannweite {v.max() - v.min():.4f} | SD {v.std():.4f}")

    # ── Skill je Station: wo ist ein Modell schlechter als ICON-D2? ──────
    skill = pd.DataFrame({name: 1.0 - je_station[name] / basis["rmse_nwp"]
                          for name in quellen})
    print("\nSkill_NWP je Station — Stationen mit NEGATIVEM Skill "
          "(Modell schlechter als ICON-D2 roh):")
    for target in ZIELE:
        st = skill.loc[target]
        print(f"  {target}:")
        for name in st.columns:
            schlechter = st[name][st[name] < 0]
            if len(schlechter):
                orte = ", ".join(f"{i} ({v:+.3f})" for i, v in
                                 schlechter.sort_values().items())
                print(f"    {name:16s} {len(schlechter):2d} von {len(st)}: {orte}")
            else:
                print(f"    {name:16s}  0 von {len(st)} — an jeder Station besser als ICON-D2")
        schwaechste = st.min(axis=1).sort_values().head(3)
        print(f"    schwaechste Stationen (Minimum ueber die Modelle): "
              + ", ".join(f"{i} {v:+.3f}" for i, v in schwaechste.items()))

    praefix = args.out or f"solar_arch_{args.dcrnn}"
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / f"{praefix}_metriken.csv", index=False)
    je_st = pd.DataFrame(je_station)
    for k, v in basis.items():
        je_st[k] = v
    je_st.to_csv(OUT / f"{praefix}_je_station.csv")
    skill.to_csv(OUT / f"{praefix}_skill_je_station.csv")
    r2_tab.to_csv(OUT / f"{praefix}_r2_je_station.csv")
    if not sig.empty:
        sig.to_csv(OUT / f"{praefix}_wilcoxon.csv", index=False)
    print(f"\nGeschrieben: {OUT}/{praefix}_{{metriken,je_station,wilcoxon}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
