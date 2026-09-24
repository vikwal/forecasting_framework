"""Verify the run-indexed HRES loader against the valid-time indexed one."""
import numpy as np, pandas as pd, glob, os
from geostatistics.train_stgnn2 import (
    load_ecmwf_runs_at_stations_and_grid,
    load_ecmwf_parquet_at_stations_and_grid,
    _ecmwf_run_for,
)

PATH = "/mnt/lambda1/nvme1/ecmwf/parquet/SL"
FEATS = ["u_wind10m", "v_wind10m", "wind_speed_10m"]
lats = np.array([52.5129, 52.1029]); lons = np.array([11.3942, 11.5827])

# ICON-D2 run times: 06/09/12/15 UTC over a few days
days = pd.date_range("2024-03-04", periods=4, freq="D", tz="UTC")
run_times = pd.DatetimeIndex(sorted(d + pd.Timedelta(hours=h) for d in days for h in (6, 9, 12, 15)))
print("ICON-Laeufe:", len(run_times), run_times[0], "…", run_times[-1])

st_runs, coords, grid_runs, _ = load_ecmwf_runs_at_stations_and_grid(
    parquet_path=PATH, station_lats=lats, station_lons=lons,
    features=FEATS, run_times=run_times, freq_h="1h", horizon=48, next_n_grid_per_station=4)
print("grid_runs:", grid_runs.shape, " NaN-Anteil %.3f" % np.isnan(grid_runs).mean())

# --- Invariante 1: gewaehlter HRES-Lauf immer <= ICON-Lauf
f = sorted(glob.glob(PATH + "/*_sl.parquet"))[0]
d = pd.read_parquet(f, columns=["starttime", "forecasttime"])
d["starttime"] = pd.to_datetime(d["starttime"], utc=True)
starts = np.sort(d["starttime"].unique())
bad = 0
print()
print("ICON-Lauf            gewaehlter HRES-Lauf   Versatz   HRES-Vorlauf 1..48")
for t0 in run_times[:8]:
    hs = _ecmwf_run_for(t0, starts)
    off = (hs - t0).total_seconds() / 3600
    if off > 0: bad += 1
    lead0 = int((t0 - hs).total_seconds() // 3600)
    print("%s  %s  %+5.0f h   %2d..%2d h" % (t0, hs, off, lead0 + 1, lead0 + 48))
print("Laeufe mit HRES aus der Zukunft:", bad, "von", len(run_times))

# --- Invariante 2: Werte stimmen mit der Quelle ueberein
gi = 0
key = (round(float(coords[gi, 0]), 5), round(float(coords[gi, 1]), 5))
src = [p for p in glob.glob(PATH + "/*_sl.parquet")
       if abs(pd.read_parquet(p, columns=["grid_lat"]).iloc[0, 0] - key[0]) < 1e-4][:1]
print()
if src:
    g = pd.read_parquet(src[0])
    g["starttime"] = pd.to_datetime(g["starttime"], utc=True)
    g["forecasttime"] = g["forecasttime"].astype(int)
    g["wind_speed_10m"] = np.hypot(g["u_wind10m"], g["v_wind10m"])
    ok = miss = 0
    for ri, t0 in enumerate(run_times):
        hs = _ecmwf_run_for(t0, starts); lead0 = int((t0 - hs).total_seconds() // 3600)
        for h in (1, 24, 48):
            row = g[(g["starttime"] == hs) & (g["forecasttime"] == lead0 + h)]
            if row.empty: miss += 1; continue
            a = grid_runs[ri, h - 1, gi, 2]; b = float(row["wind_speed_10m"].iloc[0])
            if np.isfinite(a) and abs(a - b) < 1e-4: ok += 1
            else: miss += 1
    print("Stichprobe gegen die Quelldatei: %d korrekt, %d abweichend/fehlend" % (ok, miss))

# --- Vergleich: wie weit liegt der alte Loader daneben?
ts = pd.date_range(run_times[0] + pd.Timedelta(hours=1),
                   run_times[-1] + pd.Timedelta(hours=48), freq="1h")
_, coords_old, grid_old, _ = load_ecmwf_parquet_at_stations_and_grid(
    parquet_path=PATH, station_lats=lats, station_lons=lons,
    features=FEATS, timestamps=ts, next_n_grid_per_station=4)
tpos = {t: i for i, t in enumerate(ts)}
diffs = []
for ri, t0 in enumerate(run_times):
    for h in range(1, 49):
        j = tpos.get(t0 + pd.Timedelta(hours=h))
        if j is None: continue
        a = grid_runs[ri, h - 1, 0, 2]; b = grid_old[j, 0, 2]
        if np.isfinite(a) and np.isfinite(b): diffs.append((h, a - b))
if diffs:
    arr = np.array(diffs)
    print()
    print("Windgeschwindigkeit neu minus alt, je Vorlaufstunde (m/s):")
    for lo, hi in ((1, 6), (7, 12), (13, 24), (25, 36), (37, 48)):
        m = (arr[:, 0] >= lo) & (arr[:, 0] <= hi)
        if m.any():
            print("  %2d-%2d h  Mittel %+.3f  mittl. Betrag %.3f  n=%d"
                  % (lo, hi, arr[m, 1].mean(), np.abs(arr[m, 1]).mean(), m.sum()))
