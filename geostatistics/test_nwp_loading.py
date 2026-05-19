#!/usr/bin/env python3
"""Testet das NWP-Loading für die erste Wetterstation in der Config.

Ausführung (vom forecasting_framework/ Verzeichnis):
    python geostatistics/test_nwp_loading.py --config configs/config_spatial_interpolation.yaml
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geostatistics.run_spatial_interpolation import _parse_coords_from_filename, load_nwp_wind_speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["path"]
    nwp_path  = config["data"]["nwp_path"]
    sid       = str(config["data"]["files"][0])
    test_start = config["data"].get("test_start")
    test_end   = config["data"].get("test_end")

    # Koordinaten der Station aus wind_parameter.csv
    meta = pd.read_csv(os.path.join(data_path, "wind_parameter.csv"), sep=";", dtype={"park_id": str})
    meta = meta.set_index("park_id")
    lat = float(meta.loc[sid, "latitude"])
    lon = float(meta.loc[sid, "longitude"])

    print(f"\n{'='*60}")
    print(f"Station:   {sid}")
    print(f"Koordinaten: lat={lat}, lon={lon}")
    print(f"Zeitraum:  {test_start} → {test_end}")
    print(f"NWP-Pfad:  {nwp_path}")
    print(f"{'='*60}\n")

    # --- 1. Synth-CSV prüfen ---
    print("[ 1 ] Synth-CSV")
    synth_path = os.path.join(data_path, f"synth_{sid}.csv")
    df_synth = pd.read_csv(synth_path, sep=";", parse_dates=["timestamp"])
    if df_synth["timestamp"].dt.tz is None:
        df_synth["timestamp"] = df_synth["timestamp"].dt.tz_localize("UTC")
    if test_start:
        df_synth = df_synth[df_synth["timestamp"] >= pd.Timestamp(test_start, tz="UTC")]
    if test_end:
        df_synth = df_synth[df_synth["timestamp"] <= pd.Timestamp(test_end, tz="UTC")]

    print(f"  Zeilen:      {len(df_synth)}")
    print(f"  Spalten:     {list(df_synth.columns)}")
    print(f"  wind_speed NaN: {df_synth['wind_speed'].isna().sum()} / {len(df_synth)}")
    print(f"  Zeitraum:    {df_synth['timestamp'].min()} → {df_synth['timestamp'].max()}")
    print()

    # --- 2. NWP-Ordnerstruktur prüfen ---
    print("[ 2 ] NWP-Ordnerstruktur")
    forecast_hours = ("06", "09", "12", "15")
    for fh in forecast_hours:
        folder = os.path.join(nwp_path, "ML", fh, sid)
        if os.path.isdir(folder):
            csvs = [f for f in os.listdir(folder) if f.endswith("_ML.csv")]
            print(f"  {folder}: {len(csvs)} CSV-Dateien")
            if csvs:
                print(f"    Beispiel: {csvs[0]}")
                coords = _parse_coords_from_filename(csvs[0])
                print(f"    Geparste Koordinaten: {coords}")
        else:
            print(f"  {folder}: NICHT GEFUNDEN")
    print()

    # --- 3. Nächsten Gitterpunkt finden ---
    print("[ 3 ] Nächster Gitterpunkt")
    from geopy.distance import geodesic
    nearest_fname = None
    min_dist = np.inf
    for fh in forecast_hours:
        folder = os.path.join(nwp_path, "ML", fh, sid)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.endswith("_ML.csv"):
                continue
            coords = _parse_coords_from_filename(fname)
            if coords is None:
                continue
            dist = geodesic((lat, lon), coords).km
            if dist < min_dist:
                min_dist = dist
                nearest_fname = fname
        break

    if nearest_fname:
        print(f"  Dateiname:  {nearest_fname}")
        print(f"  Distanz:    {min_dist:.2f} km")
    else:
        print("  KEIN Gitterpunkt gefunden!")
        return
    print()

    # --- 4. Eine NWP-CSV einlesen und prüfen ---
    print("[ 4 ] Inhalt der NWP-CSV (erster Forecast-Lauf)")
    for fh in forecast_hours:
        fpath = os.path.join(nwp_path, "ML", fh, sid, nearest_fname)
        if not os.path.isfile(fpath):
            continue
        df_nwp = pd.read_csv(fpath, parse_dates=["starttime"])
        df_nwp = df_nwp[df_nwp["forecasttime"] > 0]
        print(f"  Pfad:      {fpath}")
        print(f"  Zeilen:    {len(df_nwp)}")
        print(f"  Spalten:   {list(df_nwp.columns)}")
        print(f"  Startzeiten (erste 3): {sorted(df_nwp['starttime'].unique())[:3]}")
        heights = sorted((((df_nwp["toplevel"] + df_nwp["bottomlevel"]) / 2).round()).unique())
        print(f"  Verfügbare Höhen (m): {[int(h) for h in heights]}")
        print(f"  NaN in u_wind: {df_nwp['u_wind'].isna().sum()}")
        print(f"  NaN in v_wind: {df_nwp['v_wind'].isna().sum()}")
        print()
        break

    # --- 5. load_nwp_wind_speed für diese eine Station ---
    print("[ 5 ] load_nwp_wind_speed")
    timestamps = pd.DatetimeIndex(df_synth["timestamp"].values)
    hub_height = float(config.get("interpolation", {}).get("nwp_hub_height", 10.0))

    nwp_matrix = load_nwp_wind_speed(
        nwp_path=nwp_path,
        station_ids=[sid],
        station_lats=np.array([lat]),
        station_lons=np.array([lon]),
        timestamps=timestamps,
        hub_height=hub_height,
    )

    col = nwp_matrix[:, 0]
    n_nan  = int(np.isnan(col).sum())
    n_ok   = int((~np.isnan(col)).sum())
    print(f"  Timestamps gesamt: {len(col)}")
    print(f"  Werte vorhanden:   {n_ok} ({100*n_ok/len(col):.1f}%)")
    print(f"  NaN:               {n_nan} ({100*n_nan/len(col):.1f}%)")
    if n_ok > 0:
        valid = col[~np.isnan(col)]
        print(f"  wind_speed min/mean/max: {valid.min():.2f} / {valid.mean():.2f} / {valid.max():.2f} m/s")

    # Lücken analysieren
    if n_nan > 0:
        nan_mask = np.isnan(col)
        nan_ts = timestamps[nan_mask]
        print(f"\n  Erste 5 fehlende Timestamps:")
        for ts in nan_ts[:5]:
            print(f"    {ts}")
        # Prüfen ob die Lücken zusammenhängen
        gaps = np.diff(np.where(nan_mask)[0])
        print(f"  Lücken-Struktur: {np.unique(gaps, return_counts=True)}")

    print(f"\n{'='*60}")
    print("Test abgeschlossen.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
