"""Raeumliche n-Fold-Aufteilung der HPO-Stationen (Default: 153 -> 3 Folds).

Vergleicht die Strategien anhand der Groessen, die fuer dieses Modell
tatsaechlich zaehlen:

  * Distanz jeder Val-Station zu ihrer naechsten Train-Station (= ob der
    Nachbar-Messkanal ueberhaupt noch existiert)
  * Terrain-Balance zwischen den Folds (= Vergleichbarkeit der Fold-Mittel)

Distanzen geodaetisch (WGS-84, ``pairwise_geodesic_km``) — Grad-Koordinaten
duerfen nicht euklidisch verrechnet werden, 1 deg Laenge ist auf 51 deg N nur
etwa 0.63 * 1 deg Breite.

Die Val-Menge pro Fold ist ueber ``--n-val`` frei waehlbar. Bei
``n_val * n_folds < N`` rotiert nur eine Teilmenge der Stationen; die uebrigen
sind in jedem Fold Trainingsnachbarn. Welche Gruppen rotieren, waehlt eine
Max-Min-Auswahl (Kennard-Stone) auf den Gruppen-Schwerpunkten, damit die
Val-Stationen das Gebiet weiterhin gleichmaessig abdecken.

Usage:
    python geostatistics/make_spatial_folds.py                      # nur Report
    python geostatistics/make_spatial_folds.py --n-val 30
    python geostatistics/make_spatial_folds.py --n-val 51 --write configs/spatial_folds.yaml
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "geostatistics" / "stgnn" / "utils"))

from spatial import pairwise_geodesic_km  # noqa: E402
from geostatistics.stgnn.utils.topo_features import (  # noqa: E402
    load_topo_station_features, TOPO_FEATURE_ORDER,
)

DEFAULT_CONFIG = REPO / "configs/mtgnn/config_wind_mtgnn_nwp_fold1.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Daten
# ──────────────────────────────────────────────────────────────────────────────

def norm_id(x) -> str:
    return str(x).zfill(5)


def load_coords(meta_path: Path, ids: list[str]) -> np.ndarray:
    """(N, 2) [lat, lon] in der Reihenfolge von ``ids``."""
    m = pd.read_csv(meta_path, dtype={"station_id": str})
    m["station_id"] = m["station_id"].str.zfill(5)
    m = m.set_index("station_id")
    missing = [s for s in ids if s not in m.index]
    if missing:
        raise KeyError(f"{len(missing)} Stationen fehlen in {meta_path}: {missing[:5]}")
    return np.c_[m.loc[ids, "latitude"].values, m.loc[ids, "longitude"].values]


# ──────────────────────────────────────────────────────────────────────────────
# Metriken
# ──────────────────────────────────────────────────────────────────────────────

def nearest_train_dist(D: np.ndarray, fold: np.ndarray, f: int) -> np.ndarray:
    """Fuer jede Val-Station in Fold f: Distanz zur naechsten Train-Station.

    ``fold == -1`` markiert Stationen, die in jedem Fold Train sind.
    """
    val = np.where(fold == f)[0]
    trn = np.where(fold != f)[0]
    return D[np.ix_(val, trn)].min(axis=1)


def terrain_imbalance(topo: np.ndarray, fold: np.ndarray, n_folds: int) -> np.ndarray:
    """(n_folds, F) |Fold-Mittel - Gesamtmittel| / sigma je Feature."""
    gm, gs = topo.mean(axis=0), topo.std(axis=0) + 1e-9
    return np.array([np.abs(topo[fold == f].mean(axis=0) - gm) / gs
                     for f in range(n_folds)])


def report(name, D, topo, fold, n_folds=3):
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    n_fix = int((fold == -1).sum())
    print(f"{'Fold':>5} {'n_val':>6} {'n_train':>8} | Distanz Val->naechste Train-Station (km)")
    print(f"{'':>5} {'':>6} {'':>8} | {'median':>8} {'p90':>8} {'max':>8}")
    allmax = []
    for f in range(n_folds):
        d = nearest_train_dist(D, fold, f)
        allmax.append(d.max())
        print(f"{f+1:>5} {int((fold==f).sum()):>6} {int((fold!=f).sum()):>8} | "
              f"{np.median(d):8.1f} {np.percentile(d,90):8.1f} {d.max():8.1f}")
    if n_fix:
        print(f"  ({n_fix} Stationen sind in jedem Fold Trainingsnachbar)")

    dev = terrain_imbalance(topo, fold, n_folds)
    print(f"\n  Terrain-Ungleichgewicht (max |Fold-Mittel - Gesamt| / sigma): {dev.max():.3f}")
    print(f"  schlechtestes Feature: {TOPO_FEATURE_ORDER[int(dev.max(axis=0).argmax())]}")
    print(f"  groesste Nachbar-Luecke ueber alle Folds: {max(allmax):.1f} km")
    return max(allmax), dev.max()


# ──────────────────────────────────────────────────────────────────────────────
# Strategien
# ──────────────────────────────────────────────────────────────────────────────

def make_blocked(coords, n_folds=3):
    """Laengengrad-Streifen — die Lehrbuch-Variante."""
    lon = coords[:, 1]
    order = np.argsort(lon)
    fold = np.empty(len(lon), dtype=int)
    for i, idx in enumerate(order):
        fold[idx] = min(i * n_folds // len(lon), n_folds - 1)
    return fold


def make_random(n, n_folds=3, seed=0):
    rng = np.random.default_rng(seed)
    fold = np.repeat(np.arange(n_folds), int(np.ceil(n / n_folds)))[:n]
    rng.shuffle(fold)
    return fold


def _build_groups(D: np.ndarray, n_folds: int) -> tuple[list[list[int]], list[int]]:
    """Raeumlich benachbarte n_folds-Tupel bilden (isolierteste Station zuerst)."""
    n = D.shape[0]
    unassigned = set(range(n))
    groups: list[list[int]] = []
    Dw = D.copy()
    np.fill_diagonal(Dw, np.inf)

    while len(unassigned) >= n_folds:
        # Isolierteste Station zuerst: ihre Partnerwahl ist am staerksten
        # eingeschraenkt, deshalb bekommt sie den ersten Zugriff.
        cand = np.array(sorted(unassigned))
        sub = Dw[np.ix_(cand, cand)]
        seed_i = cand[np.argmax(np.sort(sub, axis=1)[:, min(n_folds - 1, len(cand) - 1)])]
        others = [c for c in cand if c != seed_i]
        nearest = sorted(others, key=lambda j: Dw[seed_i, j])[:n_folds - 1]
        grp = [int(seed_i)] + [int(x) for x in nearest]
        groups.append(grp)
        unassigned -= set(grp)
    return groups, sorted(unassigned)


def _select_rotating(groups, D, n_val, seed_free=True):
    """Max-Min-Auswahl (Kennard-Stone) von ``n_val`` Gruppen ueber ihre Schwerpunkte.

    Sorgt dafuer, dass die rotierenden Val-Stationen das Gebiet abdecken statt
    sich zu klumpen. Distanz zwischen zwei Gruppen = kleinste Stationsdistanz.
    """
    G = len(groups)
    if n_val >= G:
        return list(range(G))
    GD = np.zeros((G, G))
    for a in range(G):
        for b in range(a + 1, G):
            d = D[np.ix_(groups[a], groups[b])].min()
            GD[a, b] = GD[b, a] = d
    # Start: das am weitesten auseinanderliegende Gruppenpaar
    a, b = np.unravel_index(np.argmax(GD), GD.shape)
    sel = [int(a), int(b)]
    while len(sel) < n_val:
        rest = [g for g in range(G) if g not in sel]
        nxt = max(rest, key=lambda g: GD[g, sel].min())
        sel.append(int(nxt))
    return sorted(sel)


def make_dispersed(D, topo, n_folds=3, n_val=None, seed=0):
    """Raeumlich gestreut: benachbarte Tupel bilden, je einen pro Fold.

    Damit liegt zu jeder Val-Station garantiert ihr unmittelbarer raeumlicher
    Partner im Train-Satz. Die Zuordnung innerhalb der Tupel wird anschliessend
    so permutiert, dass die Terrain-Mittel der Folds moeglichst gleich sind.

    ``n_val`` = Val-Stationen pro Fold. Default (None) = alle Stationen rotieren.
    Ist ``n_val`` kleiner, rotieren nur ``n_val`` raeumlich gestreute Gruppen,
    alle uebrigen Stationen sind in jedem Fold Trainingsnachbar (fold == -1).
    """
    n = D.shape[0]
    groups, leftovers = _build_groups(D, n_folds)
    if n_val is None:
        n_val = len(groups) + (1 if leftovers else 0)

    rotating = _select_rotating(groups, D, n_val)
    rot_set = set(rotating)

    rng = np.random.default_rng(seed)
    fold = np.full(n, -1, dtype=int)
    for gi in rotating:
        perm = rng.permutation(n_folds)
        for k, idx in enumerate(groups[gi]):
            fold[idx] = perm[k]
    # Restgruppen und Reststationen sind in jedem Fold Trainingsnachbar,
    # ausser die Aufteilung geht exakt auf — dann fuellen die Leftovers auf.
    if len(rotating) == len(groups) and leftovers:
        for k, idx in enumerate(leftovers):
            fold[idx] = k % n_folds

    def imbalance(fl):
        gm, gs = topo.mean(axis=0), topo.std(axis=0) + 1e-9
        return max(np.abs(topo[fl == f].mean(axis=0) - gm).max() / gs.max()
                   for f in range(n_folds))

    best = imbalance(fold)
    for _ in range(40):
        improved = False
        for gi in rotating:
            grp = groups[gi]
            cur = [fold[i] for i in grp]
            for perm in itertools.permutations(range(n_folds)):
                if list(perm) == cur:
                    continue
                for k, idx in enumerate(grp):
                    fold[idx] = perm[k]
                val = imbalance(fold)
                if val < best - 1e-9:
                    best, cur, improved = val, list(perm), True
                else:
                    for k, idx in enumerate(grp):
                        fold[idx] = cur[k]
        if not improved:
            break
    return fold


# ──────────────────────────────────────────────────────────────────────────────
# Ausgabe
# ──────────────────────────────────────────────────────────────────────────────

def write_folds(path: Path, ids: list[str], fold: np.ndarray, n_folds: int) -> None:
    out = {}
    for f in range(n_folds):
        # sortiert, damit YAML-Diffs zwischen Laeufen lesbar bleiben; die
        # train-vor-val-Reihenfolge entsteht erst beim Zusammenbau in den
        # train_*/hpo_*-Skripten (all_ids = files + val_files).
        val = sorted(ids[i] for i in range(len(ids)) if fold[i] == f)
        trn = sorted(ids[i] for i in range(len(ids)) if fold[i] != f)
        out[f"spatial_fold{f+1}"] = {"files": trn, "val_files": val}
    path.write_text(yaml.safe_dump(out, default_flow_style=False, sort_keys=False))
    print(f"\n-> {path} geschrieben "
          f"({n_folds} Folds, je {len(out['spatial_fold1']['files'])} train / "
          f"{len(out['spatial_fold1']['val_files'])} val)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="Config, aus der files/val_files und die Pfade kommen")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--n-val", type=int, default=None,
                   help="Val-Stationen pro Fold (Default: volle Partition)")
    p.add_argument("--write", type=Path, default=None,
                   help="Zielpfad fuer die YAML-Fold-Definition")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", action="store_true",
                   help="Auch geblockt/zufaellig zum Vergleich rechnen")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    d = cfg["data"]
    ids = [norm_id(x) for x in d["files"]] + [norm_id(x) for x in d["val_files"]]
    meta = REPO / d.get("stations_master", "data/stations_master.csv")
    coords = load_coords(meta, ids)

    arch = next(k for k in ("mtgnn", "dcrnn", "wavenet", "stgnn") if k in cfg)
    topo, _ = load_topo_station_features(cfg[arch]["topo_features_path"], ids,
                                         TOPO_FEATURE_ORDER, n_train=len(ids))

    D = pairwise_geodesic_km(coords, coords)
    n = len(ids)
    nn = np.sort(D + np.eye(n) * 1e9, axis=1)[:, 0]
    print(f"{n} Stationen, mittlerer Nachbarabstand {np.median(nn):.1f} km "
          f"(geodaetisch, WGS-84)")

    # Referenz: der aktuelle feste Split (103 train / 50 val)
    cur = np.zeros(n, dtype=int)
    cur[len(d["files"]):] = 1
    dcur = nearest_train_dist(D, cur, 1)
    print(f"\nAktueller fester Split ({len(d['files'])}/{len(d['val_files'])}) als Referenz:")
    print(f"  Val->naechste Train: median {np.median(dcur):.1f} km, "
          f"p90 {np.percentile(dcur,90):.1f} km, max {dcur.max():.1f} km")

    # Referenz: die finale Testauswertung (test_files gegen alle 153)
    if d.get("test_files"):
        test_ids = [norm_id(x) for x in d["test_files"]]
        tc = load_coords(meta, test_ids)
        dt = pairwise_geodesic_km(tc, coords).min(axis=1)
        print(f"Finale Testauswertung ({len(test_ids)} Ziele gegen {n} Nachbarn):")
        print(f"  Test->naechste Train: median {np.median(dt):.1f} km, "
              f"p90 {np.percentile(dt,90):.1f} km, max {dt.max():.1f} km")

    res = {}
    if args.compare:
        res["blocked"] = report("A) GEBLOCKT (Laengengrad-Streifen) — Lehrbuch-Variante",
                                D, topo, make_blocked(coords, args.n_folds), args.n_folds)
        res["random"] = report("B) ZUFAELLIG", D, topo,
                               make_random(n, args.n_folds, args.seed), args.n_folds)

    label = (f"C) GESTREUT + terrain-balanciert, "
             f"{args.n_val if args.n_val else 'volle Partition'} Val/Fold")
    fold = make_dispersed(D, topo, args.n_folds, args.n_val, args.seed)
    res["dispersed"] = report(label, D, topo, fold, args.n_folds)

    if len(res) > 1:
        print(f"\n{'='*72}\nZusammenfassung\n{'='*72}")
        print(f"{'Strategie':>12} | {'groesste Nachbar-Luecke':>24} | {'Terrain-Ungleichgew.':>20}")
        for k, (gap, imb) in res.items():
            print(f"{k:>12} | {gap:21.1f} km | {imb:20.3f}")

    if args.write:
        write_folds(args.write, ids, fold, args.n_folds)


if __name__ == "__main__":
    main()
