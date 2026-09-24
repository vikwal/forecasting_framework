"""30-Minuten-Zielachse: wird der Stundenwert konstant gehalten, wie frueher?"""
import numpy as np, pandas as pd, glob
from geostatistics.train_stgnn2 import (
    load_ecmwf_runs_at_stations_and_grid, _ecmwf_run_for, _list_ecmwf_split_files)

PATH = "/mnt/lambda1/nvme1/ecmwf/parquet/SL"
FEATS = ["u_wind10m", "v_wind10m", "wind_speed_10m"]
lats = np.array([52.5129]); lons = np.array([11.3942])
days = pd.date_range("2024-03-04", periods=2, freq="D", tz="UTC")
run_times = pd.DatetimeIndex(sorted(d + pd.Timedelta(hours=h) for d in days for h in (6, 9, 12, 15)))

h1, _, g1, _ = load_ecmwf_runs_at_stations_and_grid(
    parquet_path=PATH, station_lats=lats, station_lons=lons, features=FEATS,
    run_times=run_times, freq_h="1h", horizon=48, next_n_grid_per_station=1)
h2, coords, g2, _ = load_ecmwf_runs_at_stations_and_grid(
    parquet_path=PATH, station_lats=lats, station_lons=lons, features=FEATS,
    run_times=run_times, freq_h="30min", horizon=96, next_n_grid_per_station=1)
print("1h  :", g1.shape, " NaN %.3f" % np.isnan(g1).mean())
print("30m :", g2.shape, " NaN %.3f" % np.isnan(g2).mean())

# floor-Semantik: Schritt i traegt den Stundenwert von floor(i*0.5)
off = np.floor(np.arange(1, 97) * 0.5).astype(int)
bad = 0
for ri in range(len(run_times)):
    for i in range(96):
        o = off[i]                       # Versatz in Stunden
        if o == 0:
            continue                     # Vorlauf 0 kommt in der 1h-Achse nicht vor
        a = g2[ri, i, 0, 2]
        b = g1[ri, o - 1, 0, 2]          # 1h-Achse: Index o-1 hat Versatz o
        if not (np.isnan(a) and np.isnan(b)) and abs(a - b) > 1e-6:
            bad += 1
print("30-min-Schritte, die nicht dem passenden Stundenwert entsprechen:", bad)
pairs = sum(1 for i in range(1, 96) if off[i] == off[i - 1]
            and abs(g2[0, i, 0, 2] - g2[0, i - 1, 0, 2]) < 1e-9)
print("Paare aufeinanderfolgender Halbstunden mit identischem Wert:", pairs, "von 48 erwartet")
