"""
populate_nwp_elevations.py — Add SRTM elevation to NWP grid point tables.

Adds an ``elevation`` column (metres a.s.l.) to:
  - icon_d2_grid_points  in WeatherDB   (env: WEATHER_DB_URL)
  - ecmwf_grid_points    in ECMWF DB    (env: ECMWF_WIND_SL_URL)

Data source: SRTM3 (~90 m) via the ``srtm.py`` package, which downloads
individual tiles on demand and caches them locally — no bulk-download limit.
90 m resolution is more than sufficient for NWP grid points (ICON-D2 ~2 km,
ECMWF ~9 km spacing).

Dependencies
------------
    pip install srtm.py sqlalchemy psycopg2-binary

Usage
-----
    python geostatistics/populate_nwp_elevations.py

Optional flags
--------------
    --only icond2     only update icon_d2_grid_points
    --only ecmwf      only update ecmwf_grid_points
    --force           overwrite existing non-NULL elevation values
    --srtm-cache DIR  directory for SRTM tile cache (default: ~/.srtm)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("populate_nwp_elevations")


# ---------------------------------------------------------------------------
# SRTM elevation helper
# ---------------------------------------------------------------------------

def get_srtm_data(cache_dir: str | None):
    try:
        import srtm
    except ImportError:
        logger.error("srtm.py not installed. Run:  pip install srtm.py")
        sys.exit(1)
    kwargs = {"local_cache_dir": cache_dir} if cache_dir else {}
    logger.info("Initialising SRTM data (tiles downloaded on first use, cached locally) …")
    return srtm.get_data(**kwargs)


def lookup_elevations(srtm_data, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Look up SRTM elevation for each (lat, lon) pair.
    Returns (N,) float32 array; points with no coverage get 0 m.
    """
    elevations = np.zeros(len(lats), dtype=np.float32)
    n_missing  = 0
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        val = srtm_data.get_elevation(float(lat), float(lon))
        if val is None:
            n_missing += 1
        else:
            elevations[i] = float(val)
    if n_missing:
        logger.warning("%d / %d points had no SRTM coverage — elevation set to 0 m",
                       n_missing, len(lats))
    return elevations


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_grid_points(engine, table: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read all rows from *table*, returning (geom_hex_list, lats, lons)."""
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT geom, ST_Y(geom) AS lat, ST_X(geom) AS lon FROM {table}")
        ).fetchall()
    geoms = [str(r[0]) for r in rows]
    lats  = np.array([float(r[1]) for r in rows], dtype=np.float64)
    lons  = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return geoms, lats, lons


def add_elevation_column(engine, table: str) -> None:
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS elevation FLOAT;"
        ))
    logger.info("  Column 'elevation' ensured on %s", table)


def already_filled(engine, table: str) -> set[str]:
    """Return set of geom hex values that already have a non-NULL elevation."""
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT geom FROM {table} WHERE elevation IS NOT NULL")
        ).fetchall()
    return {str(r[0]) for r in rows}


def write_elevations(
    engine,
    table: str,
    geoms: list[str],
    elevations: np.ndarray,
    batch_size: int = 1000,
) -> None:
    from sqlalchemy import text
    updates = [
        {"geom_hex": g, "elev": float(e)}
        for g, e in zip(geoms, elevations)
    ]
    batches = [updates[i:i + batch_size] for i in range(0, len(updates), batch_size)]
    for bi, batch in enumerate(batches, 1):
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    UPDATE {table}
                    SET elevation = :elev
                    WHERE geom = ST_GeomFromEWKB(decode(:geom_hex, 'hex'))
                """),
                batch,
            )
        logger.info("  Batch %d/%d written (%d rows)", bi, len(batches), len(batch))


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------

def process_table(
    db_url: str,
    table: str,
    srtm_data,
    force: bool,
) -> None:
    from sqlalchemy import create_engine

    logger.info("Processing %s …", table)
    engine = create_engine(db_url)
    try:
        geoms, lats, lons = load_grid_points(engine, table)
        logger.info("  %d grid points found", len(geoms))
        if not geoms:
            logger.warning("  Table %s is empty — nothing to do", table)
            return

        add_elevation_column(engine, table)

        if not force:
            filled = already_filled(engine, table)
            idx = [i for i, g in enumerate(geoms) if g not in filled]
            if not idx:
                logger.info(
                    "  All rows already have elevation — nothing to do (use --force to overwrite)"
                )
                return
            geoms = [geoms[i] for i in idx]
            lats  = lats[idx]
            lons  = lons[idx]
            logger.info("  %d rows need elevation lookup", len(geoms))

        elevations = lookup_elevations(srtm_data, lats, lons)
        logger.info(
            "  Elevation range: %.0f m – %.0f m  (mean %.0f m)",
            elevations.min(), elevations.max(), elevations.mean(),
        )
        write_elevations(engine, table, geoms, elevations)
        logger.info("  Done: %d rows updated in %s", len(geoms), table)
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate NWP grid point elevation columns from SRTM (~90 m)"
    )
    parser.add_argument("--only", choices=["icond2", "ecmwf"], default=None)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing non-NULL elevation values")
    parser.add_argument("--srtm-cache", default=None,
                        help="Directory for SRTM tile cache (default: srtm.py default)")
    args = parser.parse_args()

    weather_db_url = os.environ.get("WEATHER_DB_URL")
    ecmwf_db_url   = os.environ.get("ECMWF_WIND_SL_URL")

    do_icond2 = args.only in (None, "icond2")
    do_ecmwf  = args.only in (None, "ecmwf")

    if do_icond2 and not weather_db_url:
        logger.error("WEATHER_DB_URL not set — cannot update icon_d2_grid_points")
        do_icond2 = False
    if do_ecmwf and not ecmwf_db_url:
        logger.error("ECMWF_WIND_SL_URL not set — cannot update ecmwf_grid_points")
        do_ecmwf = False
    if not do_icond2 and not do_ecmwf:
        logger.error("Nothing to do — check environment variables")
        sys.exit(1)

    srtm_data = get_srtm_data(args.srtm_cache)

    if do_icond2:
        process_table(weather_db_url, "icon_d2_grid_points", srtm_data, args.force)
    if do_ecmwf:
        process_table(ecmwf_db_url,   "ecmwf_grid_points",  srtm_data, args.force)

    logger.info("All done.")


if __name__ == "__main__":
    main()
