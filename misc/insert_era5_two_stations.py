#!/usr/bin/env python3
"""Traegt die beiden fehlenden Stationen 03196 und 15813 in public.era5_wind nach.

Hintergrund
-----------
Die ERA5-Extraktion (/home/meghnanegi/Era5/extract_grib_to_db.py) liest ihre
Stationsliste aus der eingefrorenen Datei coordinates.txt vom 2026-02-24 mit
201 Stationen. masterdata hat 203. Die beiden zuletzt aufgenommenen Stationen
fehlen deshalb in era5_wind, obwohl Register, Messungen und GRIB-Rohdaten sie
kennen. Quelle hier ist der Parquet-Cache, in den sie am 2026-08-11 direkt aus
/mnt/nas/era5_raw/wind/ extrahiert wurden (Methode: naechster Gitterpunkt,
kalibriert gegen die vorhandenen 201 Stationen, max|Delta| 1e-6 bis 1e-7).

Eigenschaften
-------------
* Trockenlauf ist die Voreinstellung. Geschrieben wird nur mit --commit.
* Idempotent: ON CONFLICT (station_id, timestamp, geom) DO NOTHING. Ein
  zweiter Lauf aendert nichts, bestehende Zeilen werden nie angetastet.
* Zeitraum wird auf den der Tabelle beschnitten (Beginn 2023-07-01), damit
  nicht zwei von 203 Stationen einen laengeren Verlauf haben als alle anderen.
* geom ist der ERA5-GITTERPUNKT, nicht die Stationskoordinate: die
  Stationskoordinate auf 0.25 Grad gerundet. Verifiziert an allen 201
  vorhandenen Stationen, 201/201 exakt.

Aufruf
------
    source ~/Work/forecasting_framework/frcst/bin/activate
    eval "$(grep -E '^export WEATHER_DB_URL=' ~/.bashrc)"
    python ~/insert_era5_two_stations.py              # Trockenlauf
    python ~/insert_era5_two_stations.py --commit     # schreibt
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import sqlalchemy as sa
import psycopg2
from psycopg2.extras import execute_values

STATIONS = ("03196", "15813")
CACHE = "/mnt/lambda1/nvme1/synthetic/era5_wind_cache"
COLS = ["u_wind_10m", "v_wind_10m", "u_wind_100m", "v_wind_100m",
        "wind_gust_10m", "friction_wind", "temp_2m", "pressure", "dew_point_2m"]
GRID = 0.25


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="tatsaechlich schreiben")
    args = ap.parse_args()

    url = os.environ.get("WEATHER_DB_URL")
    if not url:
        print("FEHLER: WEATHER_DB_URL nicht gesetzt."); return 1

    eng = sa.create_engine(url)
    with eng.connect() as c:
        udt = c.execute(sa.text(
            "select udt_name from information_schema.columns "
            "where table_name='era5_wind' and column_name='geom'")).scalar()
        lo, hi = c.execute(sa.text("select min(timestamp), max(timestamp) from era5_wind")).fetchone()
        n_before = c.execute(sa.text("select count(distinct station_id) from era5_wind")).scalar()
        present = [r[0] for r in c.execute(sa.text(
            "select distinct station_id from era5_wind where station_id = any(:s)"),
            {"s": list(STATIONS)})]
        ref_id = c.execute(sa.text("select station_id from era5_wind limit 1")).scalar()
        n_ref = c.execute(sa.text("select count(*) from era5_wind where station_id=:s"),
                          {"s": ref_id}).scalar()
        mast = pd.read_sql("select station_id, longitude, latitude from masterdata", c)

    mast["sid"] = mast.station_id.astype(str).str.zfill(5)
    print(f"Tabelle era5_wind: {n_before} Stationen, Zeitraum {lo} .. {hi}")
    print(f"Referenzstation {ref_id}: {n_ref} Zeilen")
    if present:
        print(f"ABBRUCH: {present} liegen bereits in era5_wind. Nichts zu tun.")
        return 0

    cast = "::geography" if udt == "geography" else ""
    tmpl = ("(%s,%s,ST_SetSRID(ST_MakePoint(%s,%s),4326)" + cast + ","
            + ",".join(["%s"] * len(COLS)) + ")")
    sql = ("INSERT INTO era5_wind (station_id,timestamp,geom," + ",".join(COLS) + ") "
           "VALUES %s ON CONFLICT (station_id,timestamp,geom) DO NOTHING")

    batches = []
    for sid in STATIONS:
        path = os.path.join(CACHE, f"Station_{sid}.parquet")
        if not os.path.exists(path):
            print(f"ABBRUCH: {path} fehlt."); return 1
        d = pd.read_parquet(path)
        d.index = pd.DatetimeIndex(d.index)
        d = d.loc[(d.index >= pd.Timestamp(lo, tz="UTC")) & (d.index <= pd.Timestamp(hi, tz="UTC"))]
        missing = [c for c in COLS if c not in d.columns]
        if missing:
            print(f"ABBRUCH: {sid} fehlen Spalten {missing}"); return 1
        r = mast.loc[mast.sid == sid]
        if r.empty:
            print(f"ABBRUCH: {sid} nicht in masterdata."); return 1
        r = r.iloc[0]
        glon = float(np.round(float(r.longitude) / GRID) * GRID)
        glat = float(np.round(float(r.latitude) / GRID) * GRID)
        rows = [(sid, ts.tz_convert("UTC").tz_localize(None), glon, glat,
                 *[None if pd.isna(d.at[ts, c]) else float(d.at[ts, c]) for c in COLS])
                for ts in d.index]
        flag = "OK" if len(rows) == n_ref else f"ABWEICHUNG gegen Referenz ({n_ref})"
        print(f"  {sid}: {len(rows)} Zeilen  {d.index.min()} .. {d.index.max()}  "
              f"Station ({r.longitude:.4f},{r.latitude:.4f}) -> Gitter ({glon},{glat})  [{flag}]")
        batches.append((sid, rows))

    total = sum(len(b) for _, b in batches)
    if not args.commit:
        print(f"\nTROCKENLAUF: {total} Zeilen wuerden eingefuegt. Mit --commit ausfuehren.")
        return 0

    conn = psycopg2.connect(url)
    conn.autocommit = False
    try:
        for sid, rows in batches:
            with conn.cursor() as cur:
                execute_values(cur, sql, rows, template=tmpl, page_size=2000)
            print(f"  {sid}: uebergeben")
        conn.commit()
    except Exception as ex:
        conn.rollback()
        print(f"FEHLER, Transaktion zurueckgerollt: {type(ex).__name__}: {ex}")
        return 1
    finally:
        conn.close()

    with eng.connect() as c:
        n_after = c.execute(sa.text("select count(distinct station_id) from era5_wind")).scalar()
        chk = pd.read_sql(
            "select station_id, count(*) n, min(timestamp) mn, max(timestamp) mx "
            "from era5_wind where station_id = any(%(s)s) group by station_id order by station_id",
            c, params={"s": list(STATIONS)})
    print(f"\nStationen: {n_before} -> {n_after}")
    print(chk.to_string(index=False))
    print("OK" if n_after == n_before + 2 else "WARNUNG: Stationszahl nicht wie erwartet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
