#!/usr/bin/env python3
"""
Zeichnet Trainings-Stichproben des Stationsgraphen als Bildfolge fuer Vortraege.

Zweck
-----
Sichtbar machen, wie das induktive Setting im Training hergestellt wird. Pro
Trainingsbeispiel passiert dreierlei:

1. ``n_target`` zufaellige Trainingsstationen (gleichverteilt zwischen
   ``min_target_stations`` und ``max_target_stations``) werden zu Zielen.
   Ihre Messhistorie wird genullt (IGNNK-Masking).
2. Aus den uebrigen Trainingsstationen werden die ``next_n_neighbors``
   raeumlich naechsten als Kontext behalten. Das ist deterministisch, nicht
   zufaellig: gerankt wird nach dem Minimalabstand zu IRGENDEINEM Ziel.
3. Der Stationsgraph wird neu verdrahtet. Die Ziele werden aus dem
   Fold-Backbone herausgeschnitten und so wieder angehaengt, wie ein einzelner
   neuer Standort im Betrieb angehaengt wuerde (Delaunay(train + {Ziel})).
   Ziel-zu-Ziel-Kanten gibt es bewusst nicht.

Punkt 3 ist der eigentliche Grund, warum das Setting induktiv ist, und der
Teil, den eine Folie zeigen sollte.

Kein Nachbau der Logik
----------------------
Die Topologie kommt aus ``HeterogeneousGraphBuilder.build_fold_topology`` und
``sample_station_edges``, die Nachbarauswahl aus
``TrainingSampler._nearest_neighbors`` bzw. ``select_val_neighbours``. Damit
kann das Bild nicht vom Training abweichen. Fuer die beiden Sampler-Methoden
genuegt ein Traeger-Objekt mit ``station_coords`` und ``tc.next_n_neighbors``;
mehr lesen sie nicht.

Aufruf
------
::

    python geostatistics/viz/sample_graph_sequence.py \
        --config configs/dcrnn/config_wind_dcrnn.yaml \
        --fold 1 --mode samples --n-samples 6 --seed 0 \
        --out ~/graph_frames

Modi
----
``backbone``  Referenzbild: Delaunay ueber die 102 Trainingsstationen des Folds.
``samples``   N Trainingsbeispiele. Die Folie fuer die Induktivitaet.
``kseries``   Dieselbe Zielmenge, ``next_n_neighbors`` variiert. Zeigt, dass die
              Graphdichte ein Hyperparameter ist.
``zoom``      Ein einzelnes Ziel, stark herangezoomt, mit seinen Attach-Kanten.
``cutout``    Zwei Frames, vorher und nachher. Vorher: die spaeteren Ziele sind
              gewoehnliche Knoten im Fold-Backbone. Nachher: sie sind Ziele, aus
              dem Backbone geschnitten und per Delaunay-Einfuegung wieder
              angebunden. Der Unterschied sind die weggefallenen
              Ziel-zu-Ziel-Kanten. Das schaerfste Bild fuer den Mechanismus.
``val``       Validierungs-Layout zum Vergleich: alle 51 Zielstationen des Folds
              gleichzeitig.

Ausgabe
-------
Je Frame ein PDF (vektoriell, fuer Beamer) und optional ein PNG. Zusaetzlich
``manifest.txt`` mit den Kennzahlen je Frame, damit die Bildunterschriften nicht
von Hand abgetippt werden muessen.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------
# Farbschema, identisch zum Foliensatz (siehe docs/presentation_concept.md 5.3)
# ----------------------------------------------------------------------
C_POOL      = "#d9d9d9"   # Stationen des Pools, im Fold nicht im Training
C_UNUSED    = "#bdbdbd"   # Trainingsstation, in diesem Sample nicht als Kontext gewaehlt
C_NEIGHBOUR = "#1f5fa9"   # Kontextstation mit echten Messungen
C_TARGET    = "#d95f0e"   # maskiertes Ziel, Historie genullt
C_BACKBONE  = "#9ecae1"   # Kante zwischen zwei Kontextstationen
C_ATTACH    = "#d95f0e"   # Kante Ziel zu Kontext, im Betrieb genauso verdrahtet


def _repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "geostatistics").is_dir():
            return p
    raise SystemExit("Repo-Wurzel nicht gefunden (kein Verzeichnis geostatistics/).")


def load_fold(folds_path: Path, fold: int) -> tuple[list[str], list[str]]:
    d = yaml.safe_load(folds_path.read_text())
    key = f"spatial_fold{fold}"
    if key not in d:
        raise SystemExit(f"{key} fehlt in {folds_path}. Vorhanden: {list(d)}")
    return [str(s) for s in d[key]["files"]], [str(s) for s in d[key]["val_files"]]


def build_topology(repo: Path, cfg: dict, coords: np.ndarray, train_idx: list[int]):
    """Backbone und Attach-Nachbarschaften, aus dem Trainingscode selbst."""
    from geostatistics.stgnn.config import GraphConfig
    from geostatistics.stgnn.graph_builder import HeterogeneousGraphBuilder

    gcfg = GraphConfig(
        station_connectivity=cfg.get("station_connectivity", "delaunay"),
        station_graph_mode=cfg.get("station_graph_mode", "attach"),
        next_n_icond2_grid_points=int(cfg.get("next_n_icond2", 1)),
        next_n_ecmwf_grid_points=int(cfg.get("next_n_ecmwf", 0)),
    )
    builder = HeterogeneousGraphBuilder(gcfg)
    # build() waere hier zu teuer, es zieht das gesamte NWP-Gitter. Die beiden
    # benutzten Methoden lesen nur station_coords und station_pair_attr; letzteres
    # liefert ausschliesslich edge_attr, das fuer ein Bild nicht gebraucht wird.
    builder.station_coords = coords
    builder.station_pair_attr = np.zeros((len(coords), len(coords), 1), dtype=np.float32)
    builder._s2s_feat_dim = 1
    return builder, builder.build_fold_topology(train_idx)


def sampler_shim(coords: np.ndarray, k: int | None):
    """Traeger fuer die beiden wiederverwendeten TrainingSampler-Methoden."""
    from geostatistics.stgnn.training.sampler import TrainingSampler

    shim = SimpleNamespace(
        station_coords=coords,
        tc=SimpleNamespace(next_n_neighbors=k),
    )
    return (
        TrainingSampler._nearest_neighbors.__get__(shim),
        TrainingSampler.select_val_neighbours.__get__(shim),
    )


def draw_sample(
    train_idx: list[int],
    k: int | None,
    n_target: int,
    nearest_fn,
    rng: random.Random,
    targets: list[int] | None = None,
):
    """Ein Trainingsbeispiel, Zeile fuer Zeile wie in sampler.sample_train."""
    n_train = len(train_idx)
    if targets is None:
        target_local = sorted(rng.sample(range(n_train), n_target))
        targets = [train_idx[i] for i in target_local]
    cands = [g for g in train_idx if g not in set(targets)]
    neighbours = nearest_fn(targets, cands, k) if k is not None else cands
    return sorted(targets), sorted(neighbours)


def plot_frame(
    ax,
    coords: np.ndarray,
    pool_idx: list[int],
    train_idx: list[int],
    neighbours: list[int],
    targets: list[int],
    edges_global: np.ndarray,
    title: str,
    subtitle: str = "",
    zoom_on: int | None = None,
    zoom_km: float = 120.0,
):
    lat, lon = coords[:, 0], coords[:, 1]
    tset, nset = set(targets), set(neighbours)

    used = nset | tset
    pool_rest = [i for i in pool_idx if i not in set(train_idx)]
    unused    = [i for i in train_idx if i not in used]

    # Kanten zuerst, damit die Knoten darauf liegen
    for a, b in edges_global:
        is_attach = a in tset or b in tset
        ax.plot(
            [lon[a], lon[b]], [lat[a], lat[b]],
            color=C_ATTACH if is_attach else C_BACKBONE,
            lw=1.5 if is_attach else 0.7,
            zorder=3 if is_attach else 2,
            solid_capstyle="round",
            alpha=0.95 if is_attach else 0.8,
        )

    ax.scatter(lon[pool_rest], lat[pool_rest], s=9, c=C_POOL, zorder=4,
               linewidths=0, label="Pool, im Fold Zielstation")
    ax.scatter(lon[unused], lat[unused], s=11, c=C_UNUSED, zorder=5,
               linewidths=0, label="Trainingsstation, nicht im Kontextbudget")
    ax.scatter(lon[neighbours], lat[neighbours], s=26, c=C_NEIGHBOUR, zorder=6,
               linewidths=0, label="Kontext, echte Messungen")
    ax.scatter(lon[targets], lat[targets], s=95, c=C_TARGET, zorder=7,
               marker="o", edgecolors="white", linewidths=1.4,
               label="Ziel, Historie genullt")

    mean_lat = float(np.mean(lat))
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
    ax.set_axis_off()

    if zoom_on is not None:
        dlat = zoom_km / 111.0
        dlon = dlat / np.cos(np.deg2rad(lat[zoom_on]))
        ax.set_xlim(lon[zoom_on] - dlon, lon[zoom_on] + dlon)
        ax.set_ylim(lat[zoom_on] - dlat, lat[zoom_on] + dlat)
    else:
        ax.set_xlim(lon.min() - 0.4, lon.max() + 0.4)
        ax.set_ylim(lat.min() - 0.3, lat.max() + 0.3)

    ax.set_title(title, fontsize=15, loc="left", pad=30 if subtitle else 10)
    if subtitle:
        ax.text(0.0, 1.006, subtitle, transform=ax.transAxes, fontsize=10.5,
                color="#555555", va="bottom", ha="left")


def legend_handles():
    mk = lambda c, s, ec=None: Line2D([], [], marker="o", ls="", markersize=s,
                                      markerfacecolor=c, markeredgecolor=ec or c,
                                      markeredgewidth=1.2 if ec else 0)
    return (
        [mk(C_TARGET, 10, "white"), mk(C_NEIGHBOUR, 6), mk(C_UNUSED, 4.5), mk(C_POOL, 4),
         Line2D([], [], color=C_ATTACH, lw=1.8), Line2D([], [], color=C_BACKBONE, lw=1.2)],
        ["Ziel, Historie genullt", "Kontext, echte Messungen",
         "Trainingsstation ausserhalb des Budgets", "Zielstation des Folds",
         "Anbindung des Ziels (Delaunay-Einfuegung)", "Backbone zwischen Kontextstationen"],
    )


def save(fig, out_dir: Path, name: str, png: bool, dpi: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02, transparent=True)
    if png:
        fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.close(fig)
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dcrnn/config_wind_dcrnn.yaml")
    ap.add_argument("--folds", default="configs/spatial_folds.yaml")
    ap.add_argument("--fold", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--mode", default="samples",
                    choices=("backbone", "samples", "kseries", "zoom", "cutout", "val"))
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--k", type=int, default=None,
                    help="next_n_neighbors. Default: Wert aus der Config.")
    ap.add_argument("--k-list", default="10,25,50,90",
                    help="nur fuer --mode kseries")
    ap.add_argument("--n-target", type=int, default=None,
                    help="feste Zielanzahl statt zufaellig aus [min,max]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="graph_frames")
    ap.add_argument("--png", action="store_true", help="zusaetzlich PNG schreiben")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--figsize", default="7.0,8.4")
    ap.add_argument("--legend", action="store_true",
                    help="Legende in jedes Bild. Fuer Folien meist besser: einmal separat.")
    args = ap.parse_args()

    repo = _repo_root(Path(__file__).resolve().parent)
    sys.path.insert(0, str(repo))

    cfg_all = yaml.safe_load((repo / args.config).read_text())
    data_cfg = cfg_all["data"]
    mcfg = next(v for k, v in cfg_all.items() if k in ("dcrnn", "mtgnn", "wavenet"))

    train_ids, val_ids = load_fold(repo / args.folds, args.fold)
    all_ids = sorted(set(train_ids) | set(val_ids))

    from geostatistics.train_stgnn2 import load_station_metadata
    lats, lons, _ = load_station_metadata(
        data_cfg["path"], all_ids, meta_path=data_cfg.get("stations_master"),
    )
    coords = np.stack([lats, lons], axis=1)

    pos = {sid: i for i, sid in enumerate(all_ids)}
    train_idx = sorted(pos[s] for s in train_ids)
    val_idx   = sorted(pos[s] for s in val_ids)
    pool_idx  = list(range(len(all_ids)))

    k_cfg = mcfg.get("next_n_neighbors")
    k = args.k if args.k is not None else k_cfg
    n_min = int(mcfg.get("min_target_stations", 1))
    n_max = int(mcfg.get("max_target_stations", 10))

    builder, topo = build_topology(repo, mcfg, coords, train_idx)
    nearest_fn, val_fn = sampler_shim(coords, k)
    rng = random.Random(args.seed)

    out_dir = Path(args.out).expanduser()
    fw, fh = (float(x) for x in args.figsize.split(","))
    manifest: list[str] = []

    def edges_of(all_global: list[int], targets: list[int]) -> np.ndarray:
        ei, _ = builder.sample_station_edges(topo, all_global, targets)
        e = ei.numpy()
        g = np.asarray(all_global)
        und = e[:, e[0] < e[1]]          # jede Kante nur einmal zeichnen
        return np.stack([g[und[0]], g[und[1]]], axis=1)

    def frame(name, targets, neighbours, title, subtitle, zoom_on=None, k_used=None):
        all_global = sorted(set(neighbours) | set(targets))
        eg = edges_of(all_global, targets)
        fig, ax = plt.subplots(figsize=(fw, fh))
        plot_frame(ax, coords, pool_idx, train_idx, neighbours, targets, eg,
                   title, subtitle, zoom_on=zoom_on)
        if args.legend:
            h, lb = legend_handles()
            ax.legend(h, lb, loc="lower left", fontsize=8, frameon=False)
        p = save(fig, out_dir, name, args.png, args.dpi)
        n_attach = sum(1 for a, b in eg if a in set(targets) or b in set(targets))
        manifest.append(
            f"{p.name}\tziele={len(targets)}\tkontext={len(neighbours)}\t"
            f"kanten={len(eg)}\tdavon_anbindung={n_attach}\tk={k if k_used is None else k_used}"
        )
        print(manifest[-1])

    if args.mode == "backbone":
        bb = topo.backbone
        fig, ax = plt.subplots(figsize=(fw, fh))
        plot_frame(ax, coords, pool_idx, train_idx, train_idx, [], bb,
                   f"Fold {args.fold}: Backbone",
                   f"Delaunay ueber die {len(train_idx)} Trainingsstationen, "
                   f"{len(bb)} Kanten. Die {len(val_idx)} Zielstationen sind nicht beteiligt.")
        if args.legend:
            h, lb = legend_handles()
            ax.legend(h, lb, loc="lower left", fontsize=8, frameon=False)
        save(fig, out_dir, "backbone", args.png, args.dpi)
        manifest.append(f"backbone.pdf\ttrain={len(train_idx)}\tkanten={len(bb)}")
        print(manifest[-1])

    elif args.mode == "samples":
        for i in range(args.n_samples):
            n_t = args.n_target or rng.randint(min(n_min, len(train_idx)),
                                               min(n_max, len(train_idx)))
            t, nb = draw_sample(train_idx, k, n_t, nearest_fn, rng)
            frame(f"sample_{i+1:02d}", t, nb,
                  f"Trainingsbeispiel {i+1}",
                  f"{len(t)} Ziel(e) maskiert, {len(nb)} Kontextstationen (k = {k})")

    elif args.mode == "kseries":
        n_t = args.n_target or rng.randint(n_min, n_max)
        t, _ = draw_sample(train_idx, None, n_t, nearest_fn, rng)
        for kk in (int(x) for x in args.k_list.split(",")):
            nf, _ = sampler_shim(coords, kk)
            _, nb = draw_sample(train_idx, kk, n_t, nf, rng, targets=t)
            frame(f"k_{kk:03d}", t, nb,
                  f"next_n_neighbors = {kk}",
                  f"dieselben {len(t)} Ziele, {len(nb)} Kontextstationen", k_used=kk)

    elif args.mode == "zoom":
        t, nb = draw_sample(train_idx, k, 1, nearest_fn, rng)
        frame("zoom_single_target", t, nb,
              "Ein Ziel, so verdrahtet wie ein neuer Standort",
              "Anbindung = Delaunay(Trainingsnetz + Ziel). Keine Ziel-zu-Ziel-Kanten.",
              zoom_on=t[0])

    elif args.mode == "cutout":
        n_t = args.n_target or max(4, rng.randint(n_min, n_max))
        tg, nb = draw_sample(train_idx, k, n_t, nearest_fn, rng)
        node_set = set(tg) | set(nb)
        bb = topo.backbone
        keep = np.array([a in node_set and b in node_set for a, b in bb], dtype=bool)
        bb_sub = bb[keep]
        tset = set(tg)
        n_tt = int(sum(1 for a, b in bb_sub if a in tset and b in tset))

        fig, ax = plt.subplots(figsize=(fw, fh))
        plot_frame(ax, coords, pool_idx, train_idx, sorted(node_set), [], bb_sub,
                   "Vorher: gewoehnliche Knoten im Backbone",
                   f"{len(node_set)} Stationen, {len(bb_sub)} Backbone-Kanten")
        save(fig, out_dir, "cutout_1_vorher", args.png, args.dpi)
        manifest.append(f"cutout_1_vorher.pdf\tknoten={len(node_set)}\tkanten={len(bb_sub)}")
        print(manifest[-1])

        frame("cutout_2_nachher", tg, nb,
              "Nachher: dieselben Knoten als Ziele",
              f"aus dem Backbone geschnitten, per Delaunay-Einfuegung angebunden. "
              f"{n_tt} Ziel-zu-Ziel-Kante(n) entfallen.")
        manifest.append(f"# entfernte_ziel_ziel_kanten={n_tt}")
        print(manifest[-1])

    elif args.mode == "val":
        nb = val_fn(val_idx, train_idx)
        frame("val_layout", val_idx, nb,
              f"Validierung, Fold {args.fold}",
              f"alle {len(val_idx)} Zielstationen gleichzeitig, "
              f"{len(nb)} Kontextstationen (Vereinigung der je k naechsten)")

    # Legende separat, damit sie auf der Folie einmal steht und nicht je Bild
    fig, ax = plt.subplots(figsize=(6.2, 1.5))
    ax.set_axis_off()
    h, lb = legend_handles()
    ax.legend(h, lb, loc="center", fontsize=10, frameon=False, ncol=2)
    save(fig, out_dir, "legende", args.png, args.dpi)

    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"\n{len(manifest)} Frames in {out_dir}")


if __name__ == "__main__":
    main()
