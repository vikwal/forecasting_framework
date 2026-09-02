#!/usr/bin/env python3
"""Raeumliche Fold-Aufteilung der Solar-Stationen.

Uebertrag von ``geostatistics/make_spatial_folds.py`` (Wind, 153 Stationen) auf
den Solar-Fall (85 Stationen). Die Strategien werden von dort importiert, die
Bewertung ist eine andere — und zwar aus einem konkreten Grund:

**Wind-GNN**   ``next_n_stations > 0``: die Nachbarstationen sind ein *Eingang*
               des Modells. Eine Val-Station ohne nahe Train-Station verliert
               einen Messkanal. Die Zielgroesse "Distanz zur naechsten
               Train-Station" misst dort Datenverfuegbarkeit.

**Solar-TFT**  ``next_n_stations: 0``: das Modell sieht ausschliesslich die
               eigene Station, ICON-D2/ECMWF am Gitterpunkt und drei statische
               Merkmale. Kein Nachbar ist Eingang. Dieselbe Distanz misst hier
               etwas anderes — naemlich, wie **aehnlich** der naechste bekannte
               Ort ist, also wie schwer die Aufgabe ist.

Daraus folgt, dass es nicht *eine* richtige Aufteilung gibt, sondern zwei
Fragen mit je eigener Aufteilung:

    dispersed  Interpolation — neuer Standort *innerhalb* des Messnetzes.
               Zu jeder Val-Station liegt ihr raeumlicher Partner im Training.
               Der operativ relevante Fall in Deutschland (dichtes DWD-Netz).

    blocked    Extrapolation — ganze Region unbekannt. Laengengrad-Streifen,
               der naechste Trainingsnachbar ist systematisch weit weg.
               Die pessimistische Schranke.

Beide werden berichtet; welche man als *die* Zahl nimmt, ist eine Entscheidung
ueber den Einsatzfall, keine statistische.

Balanciert wird nicht auf Terrain-Features (die gibt es nur fuer die
Wind-Stationen), sondern auf dem, was das Solar-Modell tatsaechlich als
statische Eingaenge sieht — ``altitude``/``latitude``/``longitude`` — plus dem
mittleren gemessenen GHI **im Trainingszeitraum** als Regimevariable. Letzteres
nutzt nur Trainingsdaten, es fliesst nichts aus dem Testfenster ein.

Usage:
    python scripts/make_solar_folds.py --compare
    python scripts/make_solar_folds.py --n-folds 4 --write configs/solar_folds.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "geostatistics" / "stgnn" / "utils"))

from spatial import pairwise_geodesic_km                      # noqa: E402
from geostatistics.make_spatial_folds import (                # noqa: E402
    make_blocked, make_random, make_dispersed, nearest_train_dist, norm_id,
)

DEFAULT_CONFIG = REPO / "configs/solar_baseline/config_solar_base_lag.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Stationspool
# ──────────────────────────────────────────────────────────────────────────────

def station_pool(cfg: dict) -> list[str]:
    """Alle Stations-IDs der Config, egal in welcher Rolle sie dort stehen."""
    d = cfg["data"]
    ids = set()
    for key in ("files", "val_files", "test_files"):
        ids.update(norm_id(x) for x in d.get(key, []))
    return sorted(ids)


def usable(cfg: dict, ids: list[str], min_pct: float = 50.0) -> tuple[list[str], dict]:
    """Stationen aussortieren, die eines der beiden Fenster nicht tragen.

    Entscheidend ist die Abdeckung mit **nicht-NaN Messwerten** je Fenster, nicht
    der Indexbereich der Datei: alle 85 Parquets laufen von 2023-07 bis 2026-08
    durch, ``ghi`` ist darin aber teils fast vollstaendig NaN. Gemessen im Fenster
    Aug23..Jul25 trennt die Verteilung sauber — 04642 hat 0 % im Training
    (Messungen beginnen erst 2025-06), 04887 0.7 % im Test; die uebrigen 83 liegen
    bei mindestens 66 %, 81 davon ueber 95 %. Eine Schwelle bei 50 % schneidet
    genau dazwischen.

    Das ist ein Vorabtest. Das harte Kriterium — mindestens ein vollstaendiges
    Fenster aus lookback + horizon Schritten — prueft ``preprocessing.py`` erst
    beim Fenstern (der Zaehler dort faengt Reste ab, statt den Lauf abzubrechen).
    """
    d = cfg["data"]
    path = Path(d["path"])
    g = {k: pd.Timestamp(d[k], tz="UTC")
         for k in ("train_start", "train_end", "test_start", "test_end")}
    ziel = d.get("target_cols", ["ghi"])[0]
    gut, raus = [], {}
    for sid in ids:
        f = path / f"Station_{sid}.parquet"
        if not f.exists():
            raus[sid] = "keine Datei"
            continue
        m = pd.read_parquet(f, columns=[ziel])[ziel]
        tr = m[(m.index >= g["train_start"]) & (m.index <= g["train_end"])]
        te = m[(m.index >= g["test_start"]) & (m.index < g["test_end"])]
        p_tr = 100 * tr.notna().mean() if len(tr) else 0.0
        p_te = 100 * te.notna().mean() if len(te) else 0.0
        if p_tr < min_pct:
            raus[sid] = f"nur {p_tr:.1f}% '{ziel}' im Trainingsfenster"
        elif p_te < min_pct:
            raus[sid] = f"nur {p_te:.1f}% '{ziel}' im Testfenster"
        else:
            gut.append(sid)
    return gut, raus


def balance_features(cfg: dict, ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """(N, F) Merkmale, ueber die die Folds ausgeglichen werden.

    Die drei statischen Modelleingaenge plus das mittlere GHI im
    Trainingszeitraum. Rohparquet steht in J/cm^2 je Messintervall; fuer einen
    reinen Balance-Vergleich ist der gemeinsame Faktor egal, deshalb keine
    Umrechnung.
    """
    d = cfg["data"]
    m = pd.read_csv(REPO / d.get("stations_master", "data/stations_master.csv"),
                    dtype={"station_id": str})
    m["station_id"] = m["station_id"].str.zfill(5)
    m = m.set_index("station_id")
    lo, hi = pd.Timestamp(d["train_start"], tz="UTC"), pd.Timestamp(d["train_end"], tz="UTC")
    ghi = []
    for sid in ids:
        s = pd.read_parquet(Path(d["path"]) / f"Station_{sid}.parquet", columns=["ghi"])
        s = s[(s.index >= lo) & (s.index <= hi)]["ghi"]
        ghi.append(float(s.mean()) if len(s) else np.nan)
    ghi = np.array(ghi)
    ghi = np.where(np.isnan(ghi), np.nanmean(ghi), ghi)
    feats = np.c_[m.loc[ids, "station_height"].values.astype(float),
                  m.loc[ids, "latitude"].values.astype(float),
                  m.loc[ids, "longitude"].values.astype(float),
                  ghi]
    return feats, ["hoehe", "breite", "laenge", "ghi_mittel_train"]


# ──────────────────────────────────────────────────────────────────────────────
# Bericht
# ──────────────────────────────────────────────────────────────────────────────

def report(name: str, D: np.ndarray, feats: np.ndarray, namen: list[str],
           fold: np.ndarray, n_folds: int) -> tuple[float, float, float]:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"{'Fold':>5} {'n_val':>6} {'n_train':>8} | Distanz Val -> naechste Train-Station (km)")
    print(f"{'':>5} {'':>6} {'':>8} | {'median':>8} {'p90':>8} {'max':>8}")
    med, mx = [], []
    for f in range(n_folds):
        dd = nearest_train_dist(D, fold, f)
        med.append(np.median(dd)); mx.append(dd.max())
        print(f"{f + 1:>5} {int((fold == f).sum()):>6} {int((fold != f).sum()):>8} | "
              f"{np.median(dd):8.1f} {np.percentile(dd, 90):8.1f} {dd.max():8.1f}")
    gm, gs = feats.mean(axis=0), feats.std(axis=0) + 1e-9
    dev = np.array([np.abs(feats[fold == f].mean(axis=0) - gm) / gs for f in range(n_folds)])
    print(f"\n  Merkmals-Ungleichgewicht (max |Fold-Mittel - Gesamt| / sigma): {dev.max():.3f}"
          f"   (schlechtestes: {namen[int(dev.max(axis=0).argmax())]})")
    print(f"  Median der Nachbardistanz ueber die Folds: {np.mean(med):.1f} km"
          f"  | groesste Luecke: {max(mx):.1f} km")
    return float(np.mean(med)), float(max(mx)), float(dev.max())


def write_folds(path: Path, ids: list[str], fold: np.ndarray, n_folds: int,
                test_ids: list[str] | None = None) -> None:
    out = {}
    if test_ids:
        # Bewusst als eigener Top-Level-Schluessel: spatial_cv.load_spatial_folds
        # liest nur 'spatial_fold*' und ignoriert ihn, das Dashboard und die
        # Abschlussauswertung finden ihn hier trotzdem an einer Stelle.
        out["test_files"] = sorted(test_ids)
    for f in range(n_folds):
        out[f"spatial_fold{f + 1}"] = {
            "files": sorted(ids[i] for i in range(len(ids)) if fold[i] != f),
            "val_files": sorted(ids[i] for i in range(len(ids)) if fold[i] == f),
        }
    path.write_text(yaml.safe_dump(out, default_flow_style=False, sort_keys=False))
    n1 = out["spatial_fold1"]
    print(f"\n-> {path} geschrieben ({n_folds} Folds, "
          f"je {len(n1['files'])} train / {len(n1['val_files'])} val)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--n-folds", type=int, default=3,
                   help="Folds fuer die HPO-Kreuzvalidierung (auf dem Pool, "
                        "also nach Abzug des Testsatzes)")
    p.add_argument("--n-test", type=int, default=21,
                   help="zurueckgehaltene Teststationen; 0 = kein Testsatz")
    p.add_argument("--n-val", type=int, default=None,
                   help="Val-Stationen pro Fold (Default: volle Partition — "
                        "jede Station ist genau einmal Ziel)")
    p.add_argument("--strategy", choices=("dispersed", "blocked", "random"),
                   default="dispersed")
    p.add_argument("--write", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    alle = station_pool(cfg)
    ids, raus = usable(cfg, alle)
    print(f"{len(alle)} Stationen in der Config, {len(ids)} brauchbar im Fenster "
          f"{cfg['data']['train_start']}..{cfg['data']['test_end']}")
    for sid, grund in raus.items():
        print(f"  aussortiert {sid}: {grund}")

    m = pd.read_csv(REPO / cfg["data"].get("stations_master", "data/stations_master.csv"),
                    dtype={"station_id": str})
    m["station_id"] = m["station_id"].str.zfill(5)
    m = m.set_index("station_id")
    coords = np.c_[m.loc[ids, "latitude"].values, m.loc[ids, "longitude"].values]
    D = pairwise_geodesic_km(coords, coords)
    n = len(ids)
    nn = np.sort(D + np.eye(n) * 1e9, axis=1)[:, 0]
    print(f"mittlerer Nachbarabstand im Netz: {np.median(nn):.1f} km (geodaetisch, WGS-84)")

    feats, namen = balance_features(cfg, ids)

    # Referenz: der geerbte feste Wind-Split
    d = cfg["data"]
    if d.get("val_files"):
        fest = np.array([1 if s in {norm_id(x) for x in d["val_files"]} else 0 for s in ids])
        dcur = nearest_train_dist(D, fest, 1)
        print(f"\nGeerbter fester Split als Referenz "
              f"({int((fest == 0).sum())} train / {int((fest == 1).sum())} val):")
        print(f"  Val -> naechste Train: median {np.median(dcur):.1f} km, "
              f"p90 {np.percentile(dcur, 90):.1f} km, max {dcur.max():.1f} km")

    res = {}
    if args.compare:
        res["blocked"] = report("A) GEBLOCKT (Laengengrad-Streifen) — Extrapolation",
                                D, feats, namen, make_blocked(coords, args.n_folds), args.n_folds)
        res["random"] = report("B) ZUFAELLIG — ohne raeumliches Argument",
                               D, feats, namen, make_random(n, args.n_folds, args.seed), args.n_folds)

    # ── Stufe 1: Testsatz zuruecklegen ──────────────────────────────────────
    # Ausgewaehlt mit derselben gestreuten Logik wie die Folds, nur mit einer
    # Gruppengroesse K = round(N / n_test): make_dispersed bildet raeumlich
    # benachbarte K-Tupel und verteilt je einen Partner pro Gruppe. Nimmt man
    # davon Gruppe 0 als Test, hat jede Teststation ihre unmittelbaren
    # raeumlichen Nachbarn im Pool — und der Testsatz deckt das Gebiet
    # gleichmaessig ab, statt (wie bei reinem Kennard-Stone) auf dem Rand zu
    # sitzen, wo das Modell extrapolieren muesste.
    test_ids: list[str] = []
    if args.n_test > 0:
        if args.n_test >= n:
            raise SystemExit(f"--n-test {args.n_test} >= {n} Stationen")
        K = max(2, round(n / args.n_test))
        vor = make_dispersed(D, feats, K, None, args.seed)
        test_mask = vor == 0
        test_ids = [ids[i] for i in range(n) if test_mask[i]]
        pool_idx = [i for i in range(n) if not test_mask[i]]
        print(f"\nTestsatz zurueckgehalten: {len(test_ids)} Stationen "
              f"(Gruppengroesse K={K}), Pool fuer die HPO: {len(pool_idx)}")
        dt = D[np.ix_([i for i in range(n) if test_mask[i]], pool_idx)].min(axis=1)
        print(f"  Test -> naechste Pool-Station: median {np.median(dt):.1f} km, "
              f"p90 {np.percentile(dt, 90):.1f} km, max {dt.max():.1f} km")
        gm, gs = feats.mean(axis=0), feats.std(axis=0) + 1e-9
        dev = np.abs(feats[test_mask].mean(axis=0) - gm) / gs
        print(f"  Merkmalsabweichung des Testsatzes vom Gesamtmittel: "
              f"max {dev.max():.3f} ({namen[int(dev.argmax())]})")
    else:
        pool_idx = list(range(n))

    # ── Stufe 2: Folds auf dem Pool ─────────────────────────────────────────
    pool_ids = [ids[i] for i in pool_idx]
    Dp = D[np.ix_(pool_idx, pool_idx)]
    fp = feats[pool_idx]
    coords_p = coords[pool_idx]
    fold_pool = {"dispersed": lambda: make_dispersed(Dp, fp, args.n_folds, args.n_val, args.seed),
                 "blocked": lambda: make_blocked(coords_p, args.n_folds),
                 "random": lambda: make_random(len(pool_idx), args.n_folds, args.seed)}[args.strategy]()
    ids, feats, D, fold, n = pool_ids, fp, Dp, fold_pool, len(pool_idx)
    res[args.strategy] = report(
        f"C) {args.strategy.upper()} — {args.n_val or 'volle Partition'} Val/Fold",
        D, feats, namen, fold, args.n_folds)

    if len(res) > 1:
        print(f"\n{'=' * 78}\nZusammenfassung\n{'=' * 78}")
        print(f"{'Strategie':>12} | {'median Nachbardist.':>20} | {'groesste Luecke':>16} | "
              f"{'Ungleichgewicht':>16}")
        for k, (med, gap, imb) in res.items():
            print(f"{k:>12} | {med:17.1f} km | {gap:13.1f} km | {imb:16.3f}")

    if args.write:
        write_folds(args.write, ids, fold, args.n_folds, test_ids)


if __name__ == "__main__":
    main()
