"""
register_missing_grid_points.py — Insert ICON-D2 grid-point stems that are used
by the current training config but are not yet in icon_d2_grid_points.

After running this script, execute populate_nwp_elevations.py to fill the
elevation column for the newly inserted rows:

    python geostatistics/populate_nwp_elevations.py --only icond2

Usage
-----
    cd forecasting_framework/
    python misc/register_missing_grid_points.py \
        --config configs/dcrnn/config_wind_stgcn.yaml \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from geostatistics.train_stgnn2 import (
    _select_nearest_grid_files,
    load_station_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("register_grid_points")


def _parse_latlon(stem: str) -> tuple[float, float]:
    p = stem.split("_")
    return float(f"{p[0]}.{p[1]}"), float(f"{p[2]}.{p[3]}")


def collect_training_stems(cfg: dict) -> set[str]:
    """Return the unique ICON-D2 grid-point stems used by the current config."""
    data_cfg  = cfg["data"]
    dcrnn_cfg = cfg["dcrnn"]

    train_ids = [str(s) for s in data_cfg["files"]]
    val_ids   = [str(s) for s in data_cfg.get("val_files", [])]
    all_ids   = train_ids + val_ids

    lats, lons, _ = load_station_metadata(
        data_cfg["path"], all_ids,
        meta_path=data_cfg.get("stations_master"),
    )
    station_coords = np.stack([lats, lons], axis=1)

    ml_base   = Path(data_cfg["nwp_path"]) / "ML"
    run_hours = dcrnn_cfg.get("icond2_run_hours", [9])
    next_n    = dcrnn_cfg.get("next_n_icond2", 4)

    stems: set[str] = set()
    for si, sid in enumerate(all_ids):
        for rh in run_hours:
            sid_dir = ml_base / f"{rh:02d}" / sid
            if not sid_dir.exists():
                continue
            nearest = _select_nearest_grid_files(
                sid_dir,
                float(station_coords[si, 0]),
                float(station_coords[si, 1]),
                next_n,
            )
            for _, stem, _, _ in nearest:
                stems.add(stem)

    log.info(
        "Training stems (next_n_icond2=%d, run_hours=%s): %d unique",
        next_n, run_hours, len(stems),
    )
    return stems


def load_registered_coords(db_url: str) -> tuple[np.ndarray, np.ndarray]:
    """Return lat/lon arrays of all rows in icon_d2_grid_points."""
    from urllib.parse import urlparse
    p = urlparse(db_url)
    conn = psycopg2.connect(
        host=p.hostname, port=p.port,
        database=p.path[1:],
        user=p.username, password=p.password,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT ST_Y(geom), ST_X(geom) FROM icon_d2_grid_points")
        rows = cur.fetchall()
    conn.close()
    if not rows:
        return np.empty(0), np.empty(0)
    lats = np.array([r[0] for r in rows], dtype=np.float64)
    lons = np.array([r[1] for r in rows], dtype=np.float64)
    return lats, lons


def find_missing(
    stems: set[str],
    db_lats: np.ndarray,
    db_lons: np.ndarray,
    tol: float = 0.020,
) -> list[tuple[str, float, float]]:
    """
    Return (stem, lat, lon) for stems that have no matching DB row within *tol* degrees.
    """
    missing = []
    for s in sorted(stems):
        lat, lon = _parse_latlon(s)
        if len(db_lats) == 0:
            missing.append((s, lat, lon))
            continue
        d = np.sqrt((db_lats - lat) ** 2 + (db_lons - lon) ** 2).min()
        if d > tol:
            missing.append((s, lat, lon))
    return missing


def insert_grid_points(
    db_url: str,
    rows: list[tuple[str, float, float]],
    dry_run: bool,
) -> int:
    """Insert (stem, lat, lon) tuples into icon_d2_grid_points. Returns count inserted."""
    if not rows:
        return 0
    if dry_run:
        log.info("DRY-RUN: would insert %d rows", len(rows))
        for stem, lat, lon in rows[:5]:
            log.info("  %s  (%.4f, %.4f)", stem, lat, lon)
        if len(rows) > 5:
            log.info("  … and %d more", len(rows) - 5)
        return len(rows)

    from urllib.parse import urlparse
    p = urlparse(db_url)
    conn = psycopg2.connect(
        host=p.hostname, port=p.port,
        database=p.path[1:],
        user=p.username, password=p.password,
    )
    inserted = 0
    try:
        # icon_d2_grid_points stores POINT(lon, lat) like the rest of the DB
        data = [
            (lon, lat)
            for _, lat, lon in rows
        ]
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO icon_d2_grid_points (geom)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                data,
                template="(ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
                page_size=500,
            )
            inserted = cur.rowcount
        conn.commit()
        log.info("Inserted %d new rows into icon_d2_grid_points", inserted)
    except Exception as exc:
        conn.rollback()
        log.error("Insert failed: %s", exc)
        raise
    finally:
        conn.close()
    return inserted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Path to YAML config, e.g. configs/dcrnn/config_wind_stgcn.yaml")
    ap.add_argument("--tol", type=float, default=0.020,
                    help="Match tolerance in degrees (default: 0.020)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be inserted without writing to DB")
    args = ap.parse_args()

    db_url = os.environ.get("WEATHER_DB_URL")
    if not db_url:
        log.error("WEATHER_DB_URL not set")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    training_stems = collect_training_stems(cfg)
    db_lats, db_lons = load_registered_coords(db_url)
    log.info("icon_d2_grid_points: %d registered rows", len(db_lats))

    missing = find_missing(training_stems, db_lats, db_lons, tol=args.tol)
    log.info("Stems missing from icon_d2_grid_points: %d", len(missing))

    if not missing:
        log.info("Nothing to do — all training stems are registered.")
        return

    n = insert_grid_points(db_url, missing, dry_run=args.dry_run)

    if not args.dry_run:
        log.info(
            "Done. Run populate_nwp_elevations.py to fill elevations:\n"
            "  python geostatistics/populate_nwp_elevations.py --only icond2"
        )


if __name__ == "__main__":
    main()
