"""
build_nwp_station_mapping.py — Build NWP grid-point → station mapping table.

Scans the ML parquet directory (ML/06/{park_id}/*.parquet), which already
encodes the station assignment in the directory structure.  For each grid
point the geodesic distance to its station is computed, and grid points are
ranked within each station group (rank 1 = nearest).

The resulting table is needed by check_icond2_completeness.py --summary so
it can report "this problematic grid point is the Nth nearest of M grid
points around station <park_id>".

ML stem format: {lat_int}_{lat_dec}_{lon_int}_{lon_dec}  (e.g. 52_9065_12_8820)
SL stem format: {lon_int}_{lon_dec}_{lat_int}_{lat_dec}  (reversed, e.g. 10_0000_47_8000)
Both formats are normalised to (lat, lon) and stored in the output CSV.

Output: data/nwp_station_mapping.csv
Columns: lat, lon, park_id, station_lat, station_lon, distance_km, rank, total_in_park

Usage:
    cd forecasting_framework/
    python misc/build_nwp_station_mapping.py
    python misc/build_nwp_station_mapping.py --nwp-path /mnt/nvme1/icon-d2/parquet
    python misc/build_nwp_station_mapping.py --stations-csv data/wind_parameter.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mapping")


def _parse_ml_stem(stem: str) -> tuple[float, float]:
    """ML stem '52_9065_12_8820' → (lat=52.9065, lon=12.8820)."""
    p = stem.split("_")
    return float(f"{p[0]}.{p[1]}"), float(f"{p[2]}.{p[3]}")


def build_mapping(nwp_path: str, stations_csv: str) -> pd.DataFrame:
    ml_base = Path(nwp_path) / "ML"

    # Use a single run-hour directory (stems are identical across run-hours)
    ref_hour_dir = None
    for d in sorted(ml_base.iterdir()):
        if d.is_dir():
            ref_hour_dir = d
            break

    if ref_hour_dir is None:
        raise FileNotFoundError(f"No run-hour directories found under {ml_base}")

    log.info("Reading grid-point stems from %s", ref_hour_dir)

    stations = pd.read_csv(stations_csv, sep=";", dtype={"park_id": str})
    stations["park_id"] = stations["park_id"].str.zfill(5)
    station_lookup = stations.set_index("park_id")[["latitude", "longitude"]].to_dict("index")

    rows = []
    park_dirs = sorted(p for p in ref_hour_dir.iterdir() if p.is_dir())

    for park_dir in tqdm(park_dirs, desc="Parks"):
        park_id = park_dir.name.zfill(5)
        if park_id not in station_lookup:
            log.warning("park_id %s not found in stations CSV — skipping", park_id)
            continue

        st = station_lookup[park_id]
        station_coord = (st["latitude"], st["longitude"])

        stems = []
        for fpath in park_dir.glob("*_ML.parquet"):
            stem = fpath.stem.replace("_ML", "")
            try:
                lat, lon = _parse_ml_stem(stem)
                stems.append((stem, lat, lon))
            except (ValueError, IndexError):
                log.warning("Cannot parse stem %s — skipping", stem)

        # Compute distances and rank
        for stem, lat, lon in stems:
            dist_km = geodesic((lat, lon), station_coord).km
            rows.append({
                "lat": lat,
                "lon": lon,
                "park_id": park_id,
                "station_lat": st["latitude"],
                "station_lon": st["longitude"],
                "distance_km": round(dist_km, 4),
            })

    df = pd.DataFrame(rows)

    # Rank within each park group (1 = nearest to station)
    df["rank"] = (
        df.groupby("park_id")["distance_km"]
        .rank(method="min")
        .astype(int)
    )
    df["total_in_park"] = df.groupby("park_id")["distance_km"].transform("count").astype(int)

    df.sort_values(["park_id", "rank"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build NWP grid-point → station mapping CSV")
    ap.add_argument("--nwp-path",     default="/mnt/nvme1/icon-d2/parquet",
                    help="Base NWP parquet directory (default: /mnt/nvme1/icon-d2/parquet)")
    ap.add_argument("--stations-csv", default="data/wind_parameter.csv",
                    help="Stations CSV with park_id;longitude;latitude (default: data/wind_parameter.csv)")
    ap.add_argument("--output",       default="data/nwp_station_mapping.csv",
                    help="Output CSV path (default: data/nwp_station_mapping.csv)")
    args = ap.parse_args()

    df = build_mapping(args.nwp_path, args.stations_csv)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    log.info("─" * 60)
    log.info("Grid points total : %d", len(df))
    log.info("Parks (stations)  : %d", df["park_id"].nunique())
    log.info("Avg per park      : %.1f", len(df) / max(df["park_id"].nunique(), 1))
    log.info("Written → %s", out)
    log.info("─" * 60)


if __name__ == "__main__":
    main()
