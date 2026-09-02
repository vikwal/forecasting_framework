"""
update_icond2_from_db.py — Extend existing ICON-D2 ML parquet caches from the DB.

For each unique grid-point stem in each run-hour directory:
  1. Parse (lat, lon) from filename stem
  2. Resolve the stem's OWN grid point in icon_d2_grid_points — an exact match
     within GEOM_MATCH_DIST_M, geodesic, never a nearest-neighbour search. Stems
     whose grid point is not in the DB (everything outside the next-6 set) are
     skipped, not aliased onto a neighbour. See _resolve_geom for why.
  3. Determine the current max starttime across the stem's parquet file(s)
     (a stem can appear under several park_id folders — read once, reuse)
  4. Fetch only rows newer than that max, up to --target-date, for the given
     run-hour, from multilevelfields
  5. Append the new rows to each affected file's own missing slice and
     rewrite it atomically (temp file + os.replace) — existing rows are
     never modified, dropped, or rewritten with different values.

This is the read/append mirror of misc/fill_db_from_parquet.py (which pushes
parquet -> DB) and reuses the diff/append pattern from
misc/patch_parquets_from_csv.py, with atomic writes added since a crash
mid-write on an in-place write would corrupt years of existing history.

Usage:
    cd forecasting_framework/
    python misc/update_icond2_from_db.py \
        --nwp-path /mnt/lambda1/nvme1/icon-d2/parquet \
        --run-hours 09,12,06,15 \
        --target-date 2026-04-30 \
        [--workers 8] \
        [--dry-run] \
        [--parks 00298,00161] \
        [--output results/icond2_update_from_db_log.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import stat
import sys
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import psycopg2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("update_icond2")

_thread_local = threading.local()

DATA_COLS = ["starttime", "forecasttime", "toplevel", "bottomlevel",
             "u_wind", "v_wind", "temperature", "pressure", "qs"]
EXPECTED_ROWS_PER_RUN = 294  # 6 levels x 49 lead steps (0-48h)
# A stem must resolve to its OWN grid point. The only tolerated difference is the
# 4-decimal rounding in the filename (~4 m at these latitudes) plus the DB's own
# rounding variants; 25 m stays far below the ~2100 m ICON-D2 grid spacing, so it
# can never reach a neighbouring point.
GEOM_MATCH_DIST_M = 25


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


def _resolve_geom(conn, lat: float, lon: float) -> tuple[float, float] | None:
    """
    Resolve the grid point in icon_d2_grid_points that IS (lat, lon) — not the
    one nearest to it.

    The parquet filename encodes the grid point's own coordinates, rounded to 4
    decimals, so the DB value can sit a few metres off; GEOM_MATCH_DIST_M covers
    that and nothing more. A stem whose own grid point is absent from the DB
    returns None and is skipped by the caller.

    Do NOT relax this into a nearest-neighbour lookup. icon_d2_grid_points holds
    only the next-6 set (1218 points) while the ML parquet tree carries 4459
    stems, so an unbounded KNN silently resolved 3241 of them onto a *different*
    grid point — median 2227 m, max 4980 m away — and filled their files with
    that neighbour's values. Every alias sat below the old 5000 m warning
    threshold, so nothing was ever logged. Corrupted 2026-03-07 onward before it
    was caught on 2026-09-01.

    Distance is geodesic (geography), never planar: the `<->` operator on
    `geometry` orders by degrees, which mixes unequal lat/lon scales and is not
    a distance at these latitudes.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ST_X(geom), ST_Y(geom),
                   ST_Distance(geom::geography,
                               ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) AS dist_m
            FROM icon_d2_grid_points
            WHERE ST_DWithin(geom::geography,
                             ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
            ORDER BY geom::geography <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
            LIMIT 1
        """, (lon, lat, lon, lat, GEOM_MATCH_DIST_M, lon, lat))
        row = cur.fetchone()

    if row is None:
        return None

    canon_lon, canon_lat, _dist_m = row
    return canon_lon, canon_lat


def _atomic_write_parquet(df: pd.DataFrame, dest_path: Path) -> None:
    """Write parquet to a temp file in the same directory, then atomically
    replace the destination. On any error the original file is untouched."""
    fd, tmp_path = tempfile.mkstemp(
        dir=dest_path.parent, prefix=dest_path.stem + ".", suffix=".tmp",
    )
    os.close(fd)
    try:
        df.to_parquet(tmp_path, engine="pyarrow", compression="snappy", index=False)
        # mkstemp creates the temp file 0600. Without restoring the destination's own
        # mode first, os.replace would hand those permissions to the parquet file and
        # make it unreadable for every other user and host consuming this shared
        # dataset (l1 training runs, the prefect pipeline) — silently, since the writer
        # itself still reads it fine.
        # Group/other read is forced on top of the destination's own mode: the
        # rewritten file changes owner (to whoever runs this script), so a restrictive
        # mode that was fine while the original owner held it would lock everyone else
        # out of this shared dataset.
        try:
            mode = stat.S_IMODE(os.stat(dest_path).st_mode) | 0o044
        except OSError:
            mode = 0o664
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Per-stem worker
# ─────────────────────────────────────────────────────────────────────────────

def _process_stem(
    stem: str,
    files: list[Path],
    run_hour: int,
    db_url: str,
    target_ts: pd.Timestamp,
    geom_cache: dict[tuple[float, float], tuple[float, float] | None],
    geom_lock: threading.Lock,
    dry_run: bool,
    from_ts: pd.Timestamp | None = None,
) -> dict:
    result = {
        "stem": stem, "rows_fetched": 0, "files_updated": 0, "files_skipped": 0,
        "no_db_point": 0, "warnings": [], "error": None,
    }

    try:
        lat, lon = _parse_latlon(stem)
        conn = _get_conn(db_url)

        with geom_lock:
            cached = geom_cache.get((lat, lon), "MISS")
        if cached == "MISS":
            resolved = _resolve_geom(conn, lat, lon)
            with geom_lock:
                geom_cache[(lat, lon)] = resolved
            cached = resolved

        if cached is None:
            # Expected for every stem outside the next-6 set: the DB holds only
            # the 6 nearest grid points per station, while the ML parquet tree
            # also carries the next-22 neighbourhood. Those files have no source
            # in the DB and are left untouched — this is a skip, not an error.
            result["no_db_point"] = 1
            result["files_skipped"] = len(files)
            return result
        canon_lon, canon_lat = cached

        gap_mode = from_ts is not None

        # 1. Per file: current max starttime, plus (gap mode only) the set of runs
        #    already stored complete inside the requested window. Runs that exist
        #    but are short are treated as missing so they get topped up.
        file_maxes: dict[Path, pd.Timestamp | None] = {}
        file_complete_runs: dict[Path, set] = {}
        for f in files:
            st = pd.read_parquet(f, columns=["starttime"])["starttime"]
            if st.empty:
                result["warnings"].append(f"{f}: file has 0 rows — skipping (out of scope)")
                file_maxes[f] = None
                continue
            if not pd.api.types.is_datetime64_any_dtype(st):
                st = pd.to_datetime(st, utc=True)
            else:
                st = st.dt.tz_convert("UTC")
            file_maxes[f] = st.max()
            if gap_mode:
                in_window = st[(st >= from_ts) & (st <= target_ts)]
                counts = in_window.value_counts()
                file_complete_runs[f] = set(counts[counts >= EXPECTED_ROWS_PER_RUN].index)

        candidate_files = {f: m for f, m in file_maxes.items() if m is not None}
        if not candidate_files:
            result["files_skipped"] = len(files)
            return result

        lower_bound = min(candidate_files.values())
        # In gap mode the requested window lies *inside* the existing range, so the
        # "already up to date" shortcut must not apply — that check is what makes the
        # default (append-only) mode blind to interior holes in the first place.
        if not gap_mode and lower_bound >= target_ts:
            result["files_skipped"] = len(files)
            return result

        # 2. One query for the whole missing range for this stem/run_hour.
        #    NOTE: the DB's geom point-order convention flipped from
        #    (lon, lat) to (lat, lon) for data ingested from 2026-03-15
        #    onward (confirmed empirically — a real upstream inconsistency,
        #    not a bug in this script). Since German lon [~6,15] and lat
        #    [~47,56] ranges never overlap, matching against *both* orderings
        #    is unambiguous and safe, and transparently covers data written
        #    under either convention.
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT {", ".join(DATA_COLS)}
                FROM multilevelfields
                WHERE (geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                       OR geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326))
                  AND starttime >= %s AND starttime <= %s
                  AND EXTRACT(HOUR FROM (starttime AT TIME ZONE 'UTC')) = %s
                ORDER BY starttime, forecasttime, toplevel
            """, (canon_lon, canon_lat, canon_lat, canon_lon,
                  from_ts if gap_mode else lower_bound + pd.Timedelta(microseconds=1),
                  target_ts, run_hour))
            rows = cur.fetchall()

        if not rows:
            result["files_skipped"] = len(files)
            return result

        df_new = pd.DataFrame(rows, columns=DATA_COLS)
        df_new["starttime"] = pd.to_datetime(df_new["starttime"], utc=True)
        result["rows_fetched"] = len(df_new)

        counts = df_new.groupby("starttime").size()
        bad = counts[counts != EXPECTED_ROWS_PER_RUN]
        if len(bad):
            result["warnings"].append(
                f"{len(bad)} fetched run(s) with != {EXPECTED_ROWS_PER_RUN} rows "
                f"(appended anyway): {[str(t) for t in bad.index[:5]]}"
            )

        # 3. Insert per file. Default mode appends only rows newer than that file's
        #    own max; gap mode instead inserts the runs the file is missing inside the
        #    window. Either way the merge below keeps existing rows on conflict, so
        #    nothing already stored is ever modified.
        for f in files:
            file_max = file_maxes.get(f)
            if file_max is None:
                result["files_skipped"] += 1
                continue

            if gap_mode:
                have = file_complete_runs.get(f, set())
                file_new = df_new[~df_new["starttime"].isin(have)]
            else:
                file_new = df_new[df_new["starttime"] > file_max]
            if file_new.empty:
                result["files_skipped"] += 1
                continue

            if dry_run:
                result["files_updated"] += 1
                continue

            existing = pd.read_parquet(f)
            if not pd.api.types.is_datetime64_any_dtype(existing["starttime"]):
                existing["starttime"] = pd.to_datetime(existing["starttime"], utc=True)
            else:
                existing["starttime"] = existing["starttime"].dt.tz_convert("UTC")

            merged = pd.concat([existing, file_new], ignore_index=True)
            merged.sort_values(["starttime", "forecasttime", "toplevel"], inplace=True)
            merged.drop_duplicates(subset=["starttime", "forecasttime", "toplevel"],
                                    keep="first", inplace=True)
            merged.reset_index(drop=True, inplace=True)
            _atomic_write_parquet(merged, f)
            result["files_updated"] += 1

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
    ap.add_argument("--nwp-path", default="/mnt/lambda1/nvme1/icon-d2/parquet",
                    help="Parquet base dir (default: /mnt/lambda1/nvme1/icon-d2/parquet)")
    ap.add_argument("--run-hours", default="09,12,06,15",
                    help="Comma-separated run hours, processed in this exact order "
                         "(default: 09,12,06,15)")
    ap.add_argument("--target-date", required=True,
                    help="Inclusive upper bound date, e.g. 2026-04-30 "
                         "(combined with each run-hour's own time)")
    ap.add_argument("--from-date", default=None,
                    help="Enables GAP-FILL mode: inclusive lower bound date, e.g. "
                         "2026-03-07. Without it the script only appends runs newer "
                         "than each file's current max, so holes *inside* the existing "
                         "range (e.g. a period re-ingested into the DB after the "
                         "parquets had already moved past it) are never noticed. With "
                         "it, every run in [from-date, target-date] that the file is "
                         "missing (or stores incompletely) is inserted in place. "
                         "Existing rows are still never modified.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be appended without writing anything")
    ap.add_argument("--parks", default=None,
                    help="Comma-separated park IDs to restrict processing to, "
                         "e.g. '00298,00161' — for targeted runs/testing")
    ap.add_argument("--output", default="results/icond2_update_from_db_log.json")
    args = ap.parse_args()

    db_url = os.environ.get("WEATHER_DB_URL")
    if not db_url:
        log.error("WEATHER_DB_URL environment variable not set")
        return

    run_hours = [int(rh) for rh in args.run_hours.split(",") if rh.strip()]
    ml_base = Path(args.nwp_path) / "ML"

    park_set: set[str] = set()
    if args.parks:
        park_set = {p.strip().zfill(5) for p in args.parks.split(",") if p.strip()}
        log.info("Park filter active: %s", sorted(park_set))

    # ── Pre-flight: confirm DB actually has data up to the target date ────────
    from utils.db_connector import WeatherDBConnector

    with WeatherDBConnector().get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(starttime) FROM multilevelfields;")
            latest_in_db = cur.fetchone()[0]
    if latest_in_db.tzinfo is None:
        latest_in_db = pd.Timestamp(latest_in_db, tz="UTC")
    else:
        latest_in_db = pd.Timestamp(latest_in_db).tz_convert("UTC")

    max_target_ts = max(
        pd.Timestamp(f"{args.target_date} {rh:02d}:00:00", tz="UTC") for rh in run_hours
    )
    if args.from_date:
        log.info("GAP-FILL mode: inserting missing runs in [%s, %s] for run hours %s",
                 args.from_date, args.target_date, run_hours)
    if latest_in_db < max_target_ts:
        log.error(
            "Pre-flight check failed: DB latest starttime (%s) is earlier than "
            "the requested target (%s). Aborting.", latest_in_db, max_target_ts,
        )
        return
    log.info("Pre-flight OK: DB latest starttime = %s (>= target %s)", latest_in_db, max_target_ts)

    if args.dry_run:
        log.info("DRY-RUN mode — no data will be written")

    geom_cache: dict[tuple[float, float], tuple[float, float] | None] = {}
    geom_lock = threading.Lock()
    all_results: dict[int, list[dict]] = {}

    for rh in run_hours:
        target_ts = pd.Timestamp(f"{args.target_date} {rh:02d}:00:00", tz="UTC")
        from_ts = (pd.Timestamp(f"{args.from_date} {rh:02d}:00:00", tz="UTC")
                   if args.from_date else None)
        rh_dir = ml_base / f"{rh:02d}"
        if not rh_dir.exists():
            log.warning("rh=%02d directory not found: %s — skipping", rh, rh_dir)
            continue

        stem_to_files: dict[str, list[Path]] = defaultdict(list)
        for fpath in sorted(rh_dir.rglob("*_ML.parquet")):
            if park_set and fpath.parent.name.zfill(5) not in park_set:
                continue
            stem = fpath.stem.replace("_ML", "")
            stem_to_files[stem].append(fpath)

        log.info("rh=%02d: %d unique stems, target=%s", rh, len(stem_to_files), target_ts)

        rh_results: list[dict] = []
        rows_fetched = files_updated = files_skipped = errors = no_db = 0

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(_process_stem, stem, files, rh, db_url, target_ts,
                          geom_cache, geom_lock, args.dry_run, from_ts): stem
                for stem, files in stem_to_files.items()
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"rh={rh:02d}"):
                stem = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {"stem": stem, "rows_fetched": 0, "files_updated": 0,
                           "files_skipped": 0, "no_db_point": 0,
                           "warnings": [], "error": str(exc)}

                rh_results.append(res)
                rows_fetched += res["rows_fetched"]
                files_updated += res["files_updated"]
                files_skipped += res["files_skipped"]
                no_db += res.get("no_db_point", 0)
                if res["error"]:
                    errors += 1
                    log.warning("stem %-30s error: %s", stem, res["error"])
                for w in res["warnings"]:
                    log.warning("stem %-30s %s", stem, w)

        all_results[rh] = rh_results

        log.info("─" * 60)
        log.info("rh=%02d summary: rows_fetched=%d files_updated=%d files_skipped=%d "
                 "no_db_point=%d errors=%d",
                  rh, rows_fetched, files_updated, files_skipped, no_db, errors)
        log.info("─" * 60)

        # persist after each run-hour so a later crash doesn't lose earlier progress
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {str(k): v for k, v in all_results.items()}, f, indent=2, ensure_ascii=False,
            )

    if args.dry_run:
        log.info("(dry-run — nothing written)")
    log.info("Done. Log written to %s", args.output)


if __name__ == "__main__":
    main()
