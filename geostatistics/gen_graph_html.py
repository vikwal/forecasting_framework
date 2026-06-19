"""
Standalone-Skript: Erzeugt eine interaktive HTML-Karte der Wetterstationen
inkl. Graphstruktur (Delaunay + Radius-Filter) und NWP-Gitterpunkten aus der DB.

Quellen:
  - Stationskoordinaten: data/stations_master.csv (Pfad aus Config)
  - Graphstruktur: Delaunay-Triangulation mit neighbor_radius_km-Filter
  - ICON-D2 Gitterpunkte: icon_d2_grid_points in WeatherDB (WEATHER_DB_URL)
  - ECMWF Gitterpunkte: ecmwf_grid_points in ECMWF_WIND_SL (ECMWF_WIND_SL_URL)

Usage (aus forecasting_framework/ aufrufen):
    python geostatistics/gen_graph_html.py
    python geostatistics/gen_graph_html.py \\
        --config configs/dcrnn/config_wind_dcrnn_base.yaml \\
        --output stgnn_graph.html \\
        --k-icond2 7 --k-ecmwf 4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Import direkt aus stgnn/utils/ – vermeidet stgnn/__init__.py (benötigt torch)
sys.path.insert(0, str(Path(__file__).parent / "stgnn" / "utils"))

from plot_graph import plot_hetero_graph   # noqa: E402
from spatial import delaunay_edges, geodesic_km, geodesic_knn  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate station graph HTML map")
    p.add_argument(
        "--config",
        default="configs/dcrnn/config_wind_dcrnn_base.yaml",
        help="Path to DCRNN config YAML",
    )
    p.add_argument(
        "--output",
        default="stgnn_graph.html",
        help="Output HTML path (default: stgnn_graph.html)",
    )
    p.add_argument(
        "--k-icond2",
        type=int,
        default=7,
        help="k nearest ICON-D2 grid points per station (default: 7)",
    )
    p.add_argument(
        "--k-ecmwf",
        type=int,
        default=4,
        help="k nearest ECMWF grid points per station (default: 4)",
    )
    p.add_argument(
        "--radius",
        type=float,
        default=None,
        help=(
            "Max. Kantenlänge in km für Delaunay-Edges (überschreibt Config). "
            "Default: neighbor_radius_km aus Config, sonst 500 km."
        ),
    )
    p.add_argument(
        "--no-icond2",
        action="store_true",
        help="Skip ICON-D2 grid points (useful if DB not available)",
    )
    p.add_argument(
        "--no-ecmwf",
        action="store_true",
        help="Skip ECMWF grid points (useful if DB not available)",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_icond2_grid_coords() -> np.ndarray | None:
    """
    Loads all ICON-D2 grid point coordinates from WeatherDB.

    Returns (N, 2) array of [lat, lon], or None if DB is unavailable.
    """
    db_url = os.environ.get("WEATHER_DB_URL")
    if not db_url:
        print("Warning: WEATHER_DB_URL not set — skipping ICON-D2 grid points")
        return None
    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        # Standard PostGIS: ST_X = longitude, ST_Y = latitude
        cur.execute(
            "SELECT ST_Y(geom) AS lat, ST_X(geom) AS lon "
            "FROM icon_d2_grid_points "
            "ORDER BY ST_Y(geom), ST_X(geom)"
        )
        rows = cur.fetchall()
        conn.close()
        coords = np.array([[r[0], r[1]] for r in rows], dtype=np.float64)
        print(f"Loaded {len(coords)} ICON-D2 grid points from WeatherDB")
        return coords
    except Exception as e:
        print(f"Warning: Could not load ICON-D2 grid points from DB ({e}) — skipping")
        return None


def _load_ecmwf_grid_coords() -> np.ndarray | None:
    """
    Loads all ECMWF grid point coordinates from ECMWF_WIND_SL.

    Returns (N, 2) array of [lat, lon], or None if DB is unavailable.
    """
    db_url = os.environ.get("ECMWF_WIND_SL_URL")
    if not db_url:
        print("Warning: ECMWF_WIND_SL_URL not set — skipping ECMWF grid points")
        return None
    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(
            "SELECT ST_Y(geom) AS lat, ST_X(geom) AS lon "
            "FROM ecmwf_grid_points "
            "ORDER BY ST_Y(geom), ST_X(geom)"
        )
        rows = cur.fetchall()
        conn.close()
        coords = np.array([[r[0], r[1]] for r in rows], dtype=np.float64)
        print(f"Loaded {len(coords)} ECMWF grid points from ECMWF_WIND_SL")
        return coords
    except Exception as e:
        print(f"Warning: Could not load ECMWF grid points from DB ({e}) — skipping")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Graph helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_station_edges_hierarchical(
    station_coords: np.ndarray,
    cats: list[str],
    neighbor_radius_km: float | None,
) -> dict[str, np.ndarray]:
    """
    Hierarchisches Delaunay pro Kategorie:

      train — Delaunay nur auf Train-Stationen
      val   — neue Kanten aus Delaunay(Train+Val), die nicht in Train-only vorkamen
      test  — neue Kanten aus Delaunay(alle), die nicht in Train+Val vorkamen

    So bleibt der Train-Graph unverändert, wenn Val/Test hinzukommen.

    Returns
    -------
    dict {'train'/'val'/'test': (2, E)} int64 edge indices
    """
    cats_arr = np.array(cats)

    def compute_global_edges(subset_idx: np.ndarray) -> np.ndarray:
        """Delaunay auf Teilmenge → globale Kantenliste (E, 2)."""
        if len(subset_idx) < 3:
            return np.empty((0, 2), dtype=np.int64)
        local_coords = station_coords[subset_idx]
        local_edges  = delaunay_edges(local_coords)   # (E, 2), i < j (lokal)
        if len(local_edges) == 0:
            return np.empty((0, 2), dtype=np.int64)
        global_edges = subset_idx[local_edges]         # (E, 2) globale Indizes
        if neighbor_radius_km is not None:
            dists = geodesic_km(
                station_coords[global_edges[:, 0], 0], station_coords[global_edges[:, 0], 1],
                station_coords[global_edges[:, 1], 0], station_coords[global_edges[:, 1], 1],
            )
            global_edges = global_edges[dists <= neighbor_radius_km]
        return global_edges

    def to_set(edges: np.ndarray) -> set[tuple[int, int]]:
        return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges}

    def to_ei(edge_set: set[tuple[int, int]]) -> np.ndarray:
        """set of (a, b) → (2, E) int64"""
        if not edge_set:
            return np.empty((2, 0), dtype=np.int64)
        arr = np.array(sorted(edge_set), dtype=np.int64)
        return arr.T

    train_idx     = np.where(cats_arr == "train")[0]
    train_val_idx = np.where(np.isin(cats_arr, ["train", "val"]))[0]
    all_idx       = np.arange(len(cats), dtype=np.int64)

    # Schritt 1: Nur Train
    train_edges = compute_global_edges(train_idx)
    train_set   = to_set(train_edges)

    # Schritt 2: Train + Val → neue Kanten sind Val-zugehörig
    tv_edges  = compute_global_edges(train_val_idx)
    tv_set    = to_set(tv_edges)
    val_set   = tv_set - train_set

    # Schritt 3: Alle → neue Kanten sind Test-zugehörig
    all_edges = compute_global_edges(all_idx)
    all_set   = to_set(all_edges)
    test_set  = all_set - tv_set

    result = {
        "train": to_ei(train_set),
        "val":   to_ei(val_set),
        "test":  to_ei(test_set),
    }

    n_total = sum(ei.shape[1] for ei in result.values())
    radius_info = f", Radius-Filter {neighbor_radius_km:.0f} km" if neighbor_radius_km else ""
    print(
        f"Hierarchisches Delaunay: {result['train'].shape[1]} train / "
        f"{result['val'].shape[1]} val / {result['test'].shape[1]} test Kanten "
        f"(∑={n_total}{radius_info})"
    )
    return result


def _build_nwp_edges(
    nwp_coords_all: np.ndarray,
    station_coords: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    k-NN NWP→Station-Kanten. Gibt nur die tatsächlich verbundenen NWP-Knoten zurück.

    Returns
    -------
    nwp_coords_filtered : (N_connected, 2) — nur verbundene NWP-Punkte
    edge_index          : (2, N_s * k) — Zeile 0 = NWP-Index, Zeile 1 = Station-Index
    """
    _, nwp_idx = geodesic_knn(nwp_coords_all, station_coords, k=k)
    # nwp_idx: (N_stations, k)

    N_s = len(station_coords)
    station_idx_flat = np.repeat(np.arange(N_s), k)  # (N_s * k,)
    nwp_idx_flat     = nwp_idx.reshape(-1)            # (N_s * k,)

    # Nur verbundene NWP-Knoten behalten und Indizes neu abbilden
    unique_nwp_idx = np.unique(nwp_idx_flat)
    nwp_coords_filtered = nwp_coords_all[unique_nwp_idx]

    idx_map = {int(old): int(new) for new, old in enumerate(unique_nwp_idx)}
    nwp_idx_remapped = np.array([idx_map[int(i)] for i in nwp_idx_flat], dtype=np.int64)

    edge_index = np.stack([nwp_idx_remapped, station_idx_flat.astype(np.int64)], axis=0)
    return nwp_coords_filtered, edge_index


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]

    # --- Stationen (IDs + Kategorien) ---
    train_ids = [str(x).strip() for x in data.get("files",      [])]
    val_ids   = [str(x).strip() for x in data.get("val_files",  [])]
    test_ids  = [str(x).strip() for x in data.get("test_files", [])]
    all_ids   = train_ids + val_ids + test_ids
    all_cats  = (
        ["train"] * len(train_ids)
        + ["val"]  * len(val_ids)
        + ["test"] * len(test_ids)
    )

    # --- Koordinaten aus stations_master.csv ---
    master_path = Path(data.get("stations_master", "data/stations_master.csv"))
    if not master_path.is_absolute():
        master_path = Path.cwd() / master_path

    master = pd.read_csv(master_path, dtype={"station_id": str})
    master.set_index("station_id", inplace=True)

    coords: list[list[float]] = []
    labels: list[str]         = []
    cats:   list[str]         = []
    missing: list[str]        = []

    for sid, cat in zip(all_ids, all_cats):
        if sid in master.index:
            row = master.loc[sid]
            coords.append([float(row["latitude"]), float(row["longitude"])])
            labels.append(sid)
            cats.append(cat)
        else:
            missing.append(sid)

    if missing:
        print(
            f"Warning: {len(missing)} Station(en) nicht in master CSV, übersprungen: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if not coords:
        raise RuntimeError("Keine Stationskoordinaten gefunden — Config-Pfade prüfen.")

    station_coords = np.array(coords, dtype=np.float64)
    print(
        f"Stationen geladen: {len(station_coords)} "
        f"(train={sum(c=='train' for c in cats)}, "
        f"val={sum(c=='val' for c in cats)}, "
        f"test={sum(c=='test' for c in cats)})"
    )

    # --- Stationsgraph: hierarchisches Delaunay + Radius-Filter ---
    dcrnn_cfg = cfg.get("dcrnn", {})
    if args.radius is not None:
        neighbor_radius = args.radius
    else:
        neighbor_radius = dcrnn_cfg.get("neighbor_radius_km", 500.0)
    s2s_edge_by_cat = _build_station_edges_hierarchical(station_coords, cats, neighbor_radius)

    # --- ICON-D2 Gitterpunkte ---
    icond2_coords: np.ndarray | None = None
    i2s_edge_index: np.ndarray | None = None

    if not args.no_icond2:
        icond2_all = _load_icond2_grid_coords()
        if icond2_all is not None and args.k_icond2 > 0:
            icond2_coords, i2s_edge_index = _build_nwp_edges(
                icond2_all, station_coords, k=args.k_icond2
            )
            print(
                f"ICON-D2: {len(icond2_coords)} verbundene Gitterpunkte, "
                f"{i2s_edge_index.shape[1]} Kanten (k={args.k_icond2})"
            )

    # --- ECMWF Gitterpunkte ---
    ecmwf_coords: np.ndarray | None = None
    e2s_edge_index: np.ndarray | None = None

    if not args.no_ecmwf:
        ecmwf_all = _load_ecmwf_grid_coords()
        if ecmwf_all is not None and args.k_ecmwf > 0:
            ecmwf_coords, e2s_edge_index = _build_nwp_edges(
                ecmwf_all, station_coords, k=args.k_ecmwf
            )
            print(
                f"ECMWF: {len(ecmwf_coords)} verbundene Gitterpunkte, "
                f"{e2s_edge_index.shape[1]} Kanten (k={args.k_ecmwf})"
            )

    # --- Karte erzeugen ---
    plot_hetero_graph(
        station_coords=station_coords,
        icond2_coords=icond2_coords,
        ecmwf_coords=ecmwf_coords,
        s2s_edge_by_cat=s2s_edge_by_cat,
        i2s_edge_index=i2s_edge_index,
        e2s_edge_index=e2s_edge_index,
        station_categories=cats,
        station_labels=labels,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
