"""
fill_db_from_parquet.py — Fill gaps in multilevelfields DB table from parquet files.

For each unique grid-point stem:
  1. Parse (lat, lon) from filename stem
  2. Find canonical geom in DB via KNN on icon_d2_grid_points
  3. Load parquet, filter to run-hour + cutoff
  4. Query DB for already-complete (starttime, forecasttime) pairs — no aggregation,
     just a lightweight key-only scan filtered by geom + NOT NULL on data columns
  5. Insert only the rows that are missing or incomplete in the DB

Performance rationale:
  - The DB query returns only 2 small columns (no data columns, no aggregation)
  - The index ml_geom_time_idx (geom, starttime, forecasttime) makes the geom
    filter an efficient range scan
  - Set-difference (parquet keys − DB keys) is done in Python
  - Only actual gap rows are sent to the DB → minimal write volume

Usage:
    cd forecasting_framework/
    python misc/fill_db_from_parquet.py \
        --nwp-path /mnt/nvme1/icon-d2/parquet \
        [--run-hours 6 9 12 15] \
        [--cutoff 2025-10-31] \
        [--workers 8] \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Exception-station filter
# ─────────────────────────────────────────────────────────────────────────────

def _build_allowed_stems(
    station_ids: list[str],
    stations_master_path: str,
    all_stems: list[str],
    next_n_grid: int,
) -> set[str]:
    """
    Return the set of ICON-D2 grid-point stems that are among the *next_n_grid*
    nearest grid points to any station in *station_ids*.

    Parameters
    ----------
    station_ids         : list of station IDs to include (exception dict keys)
    stations_master_path: path to stations_master.csv (columns: station_id, latitude, longitude, …)
    all_stems           : all unique stems collected from the parquet directory
    next_n_grid         : how many nearest stems to keep per station
    """
    master = pd.read_csv(stations_master_path, dtype={"station_id": str})
    master["station_id"] = master["station_id"].str.zfill(5)
    sid_set = {str(s).zfill(5) for s in station_ids}

    missing = sid_set - set(master["station_id"])
    if missing:
        log.warning("Exception stations not found in stations_master: %s", sorted(missing))

    stations = master[master["station_id"].isin(sid_set)][["station_id", "latitude", "longitude"]]
    if stations.empty:
        log.warning("No matching stations found — skipping exception filter (all stems allowed)")
        return set(all_stems)

    # Parse lat/lon from all stems
    stem_coords = np.array([_parse_latlon(s) for s in all_stems])  # (N, 2): lat, lon
    stem_arr    = np.array(all_stems)

    allowed: set[str] = set()
    for _, row in stations.iterrows():
        s_lat, s_lon = float(row["latitude"]), float(row["longitude"])
        dists = np.sqrt((stem_coords[:, 0] - s_lat) ** 2 + (stem_coords[:, 1] - s_lon) ** 2)
        top_k = np.argsort(dists)[:next_n_grid]
        allowed.update(stem_arr[top_k].tolist())

    log.info(
        "Exception filter: %d stations → %d unique grid-point stems (next_n_grid=%d)",
        len(stations), len(allowed), next_n_grid,
    )
    return allowed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fill_db")

_thread_local = threading.local()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_latlon(stem: str) -> tuple[float, float]:
    """Parse lat/lon from ICON-D2 filename stem, e.g. '52_9065_12_8820'."""
    parts = stem.split("_")
    return float(f"{parts[0]}.{parts[1]}"), float(f"{parts[2]}.{parts[3]}")


def _get_conn(db_url: str) -> psycopg2.extensions.connection:
    """Return a per-thread DB connection (creates one if needed)."""
    if not hasattr(_thread_local, "conn") or _thread_local.conn.closed:
        p = urlparse(db_url)
        _thread_local.conn = psycopg2.connect(
            host=p.hostname, port=p.port,
            database=p.path[1:],
            user=p.username, password=p.password,
        )
    return _thread_local.conn


def _nan_to_none(series: pd.Series) -> list:
    """Replace NaN/inf with None for SQL NULL."""
    out = []
    for v in series:
        try:
            out.append(None if (v is None or np.isnan(v) or np.isinf(v)) else v)
        except (TypeError, ValueError):
            out.append(v)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-stem worker
# ─────────────────────────────────────────────────────────────────────────────

def _process_stem(
    stem: str,
    pq_path: Path,
    run_hour: int,
    db_url: str,
    cutoff: pd.Timestamp | None,
    dry_run: bool,
    force: bool = False,  # skip the db_count < 10_000 guard (used for exception stems)
    first_n_days: int | None = None,  # per-parquet relative cutoff
) -> dict:
    result = {"rows_sent": 0, "skipped": False, "error": None}

    try:
        lat, lon = _parse_latlon(stem)
        conn = _get_conn(db_url)

        # 1. Canonical geom from icon_d2_grid_points (tiny table, fast KNN)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ST_X(geom), ST_Y(geom),
                       ST_Distance(geom::geography,
                                   ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography)
                FROM icon_d2_grid_points
                ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                LIMIT 1
            """, (lon, lat, lon, lat))
            row = cur.fetchone()

        if row is None:
            result["error"] = "no geom in icon_d2_grid_points"
            return result

        canon_lon, canon_lat, dist_m = row
        if dist_m > 5000:
            log.warning("stem %s: nearest DB geom is %.0f m away (suspicious)", stem, dist_m)

        # 1b. Check that this geom has sufficient data in the DB — if not,
        #     the grid point was never populated and there is nothing to fill.
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM multilevelfields
                WHERE geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                LIMIT 10001
            """, (canon_lon, canon_lat))
            db_count = cur.fetchone()[0]

        if db_count < 10_000 and not force:
            result["skipped"] = True
            return result

        # 2. Load parquet, filter to correct run-hour and cutoff
        df = pd.read_parquet(pq_path)
        if df.empty:
            return result

        if not pd.api.types.is_datetime64_any_dtype(df["starttime"]):
            df["starttime"] = pd.to_datetime(df["starttime"], utc=True)

        df = df[df["starttime"].dt.hour == run_hour]
        if cutoff is not None:
            df = df[df["starttime"] <= cutoff]
        if first_n_days is not None and not df.empty:
            per_file_cutoff = df["starttime"].min() + pd.Timedelta(days=first_n_days)
            df = df[df["starttime"] <= per_file_cutoff]
        if df.empty:
            return result

        pq_start = df["starttime"].min()
        pq_end   = df["starttime"].max()

        # 3. Fetch already-complete (starttime, forecasttime) pairs from DB.
        #    Returns only 2 small key columns — no aggregation, no data columns.
        #    Rows with any NULL data column are excluded → treated as incomplete.
        with conn.cursor() as cur:
            cur.execute("""
                SELECT starttime, forecasttime
                FROM multilevelfields
                WHERE geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                  AND starttime >= %s AND starttime <= %s
                  AND u_wind       IS NOT NULL
                  AND v_wind       IS NOT NULL
                  AND temperature  IS NOT NULL
                  AND pressure     IS NOT NULL
                  AND qs           IS NOT NULL
            """, (canon_lon, canon_lat, pq_start, pq_end))
            complete_keys: set[tuple] = {
                (pd.Timestamp(r[0]).tz_convert("UTC"), float(r[1]))
                for r in cur.fetchall()
            }

        # 4. Identify parquet rows not yet complete in DB
        df["_st_utc"] = df["starttime"].dt.tz_convert("UTC")
        df["_ft"]     = df["forecasttime"].astype(float)
        mask = df.apply(
            lambda r: (r["_st_utc"], r["_ft"]) not in complete_keys, axis=1
        )
        ins = df[mask].drop(columns=["_st_utc", "_ft"])
        df.drop(columns=["_st_utc", "_ft"], inplace=True)
        if ins.empty:
            return result

        ins.sort_values(["starttime", "forecasttime", "toplevel"], inplace=True)
        n = len(ins)
        result["rows_sent"] = n

        if dry_run or n == 0:
            return result

        # 5. Insert only the missing/incomplete rows
        rows = list(zip(
            ins["starttime"].apply(lambda x: x.to_pydatetime()),
            _nan_to_none(ins["forecasttime"]),
            [canon_lon] * n,
            [canon_lat] * n,
            _nan_to_none(ins["toplevel"]),
            _nan_to_none(ins["bottomlevel"]),
            _nan_to_none(ins["u_wind"]),
            _nan_to_none(ins["v_wind"]),
            _nan_to_none(ins["temperature"]),
            _nan_to_none(ins["pressure"]),
            _nan_to_none(ins["qs"]),
        ))

        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO multilevelfields
                    (starttime, forecasttime, geom, toplevel, bottomlevel,
                     u_wind, v_wind, temperature, pressure, qs)
                VALUES %s
                ON CONFLICT (starttime, forecasttime, geom, toplevel) DO UPDATE
                    SET bottomlevel  = COALESCE(multilevelfields.bottomlevel,  EXCLUDED.bottomlevel),
                        u_wind       = COALESCE(multilevelfields.u_wind,       EXCLUDED.u_wind),
                        v_wind       = COALESCE(multilevelfields.v_wind,       EXCLUDED.v_wind),
                        temperature  = COALESCE(multilevelfields.temperature,  EXCLUDED.temperature),
                        pressure     = COALESCE(multilevelfields.pressure,     EXCLUDED.pressure),
                        qs           = COALESCE(multilevelfields.qs,           EXCLUDED.qs)
                """,
                rows,
                template=(
                    "(%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326),"
                    " %s, %s, %s, %s, %s, %s, %s)"
                ),
                page_size=2000,
            )
        conn.commit()

    except Exception as exc:
        result["error"] = str(exc)
        try:
            _get_conn(db_url).rollback()
        except Exception:
            pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwp-path",  required=True,
                    help="Parquet base dir, e.g. /mnt/nvme1/icon-d2/parquet")
    ap.add_argument("--run-hours", type=int, nargs="+", default=[6, 9, 12, 15])
    ap.add_argument("--cutoff",    default=None, help="Ignore runs after YYYY-MM-DD")
    ap.add_argument("--workers",   type=int, default=8)
    ap.add_argument("--dry-run",   action="store_true",
                    help="Report rows that would be sent without writing to DB")
    ap.add_argument(
        "--parks", default=None,
        help=(
            "Comma-separated park IDs to process, e.g. '05426,00096'. "
            "Bypasses the db_count guard — useful for targeted fills."
        ),
    )
    ap.add_argument(
        "--first-n-days", type=int, default=None,
        help=(
            "Per parquet: only send the first N days worth of runs "
            "(relative to each file's earliest starttime). "
            "Useful for seeding/testing without writing full history."
        ),
    )
    ap.add_argument(
        "--exception-stations", default=None,
        help=(
            "Comma-separated station IDs (e.g. '00183,00460,00856'). "
            "When set, only the grid-point stems nearest to these stations are processed. "
            "Requires --stations-master."
        ),
    )
    ap.add_argument(
        "--stations-master", default="data/stations_master.csv",
        help="Path to stations_master.csv (columns: station_id, latitude, longitude). "
             "Used with --exception-stations.",
    )
    ap.add_argument(
        "--next-n-grid", type=int, default=4,
        help="Number of nearest grid-point stems to collect per exception station (default: 4).",
    )
    args = ap.parse_args()

    db_url = os.environ.get("WEATHER_DB_URL")
    if not db_url:
        log.error("WEATHER_DB_URL environment variable not set")
        return

    cutoff = pd.Timestamp(args.cutoff, tz="UTC") if args.cutoff else None
    ml_base = Path(args.nwp_path) / "ML"

    park_set: set[str] = set()
    if args.parks:
        park_set = {p.strip().zfill(5) for p in args.parks.split(",") if p.strip()}
        log.info("Park filter active: %s", sorted(park_set))

    # Collect unique (stem, run_hour) → one representative parquet file per stem
    tasks: list[tuple[str, Path, int]] = []
    for rh in args.run_hours:
        rh_dir = ml_base / f"{rh:02d}"
        if not rh_dir.exists():
            log.warning("rh=%02d directory not found: %s", rh, rh_dir)
            continue
        seen: set[str] = set()
        for fpath in sorted(rh_dir.rglob("*_ML.parquet")):
            if park_set and fpath.parent.name.zfill(5) not in park_set:
                continue
            stem = fpath.stem.replace("_ML", "")
            if stem not in seen:
                seen.add(stem)
                tasks.append((stem, fpath, rh))
        log.info("rh=%02d: %d unique stems", rh, len(seen))

    log.info("Total tasks before filter: %d", len(tasks))

    force = bool(park_set)   # bypass db_count guard when targeting specific parks
    if args.exception_stations:
        station_ids = [s.strip() for s in args.exception_stations.split(",") if s.strip()]
        all_stems   = list({stem for stem, _, _ in tasks})
        allowed     = _build_allowed_stems(
            station_ids, args.stations_master, all_stems, args.next_n_grid
        )
        tasks = [(stem, fpath, rh) for stem, fpath, rh in tasks if stem in allowed]
        log.info("Total tasks after exception filter: %d", len(tasks))
        force = True   # bypass the db_count < 10_000 guard for these stems

    if args.dry_run:
        log.info("DRY-RUN mode — no data will be written")

    total_rows = total_skipped = total_errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_process_stem, stem, fpath, rh, db_url, cutoff, args.dry_run, force,
                      args.first_n_days): stem
            for stem, fpath, rh in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Filling DB"):
            stem = futures[fut]
            try:
                res = fut.result()
                total_rows    += res["rows_sent"]
                total_skipped += int(res.get("skipped", False))
                if res["error"]:
                    total_errors += 1
                    log.warning("stem %-30s  error: %s", stem, res["error"])
            except Exception as exc:
                total_errors += 1
                log.warning("stem %-30s  FAILED: %s", stem, exc)

    log.info("─" * 60)
    log.info("Tasks processed  : %d  (%s)", len(tasks),
             f"exception filter: {args.exception_stations}" if args.exception_stations else "no filter")
    log.info("Stems skipped    : %d  (geom not in DB or < 10 000 rows)", total_skipped)
    log.info("Rows sent to DB  : %d", total_rows)
    log.info("Errors           : %d", total_errors)
    if args.dry_run:
        log.info("(dry-run — nothing written)")
    log.info("─" * 60)


if __name__ == "__main__":
    main()
