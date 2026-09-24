import numpy as np, pandas as pd, glob
from geostatistics.train_stgnn2 import (
    load_ecmwf_runs_at_stations_and_grid, _ecmwf_run_for, _list_ecmwf_split_files)

PATH = "/mnt/lambda1/nvme1/ecmwf/parquet/SL"
FEATS = ["u_wind10m", "v_wind10m", "wind_speed_10m"]
lats = np.array([52.5129]); lons = np.array([11.3942])
days = pd.date_range("2024-03-04", periods=3, freq="D", tz="UTC")
run_times = pd.DatetimeIndex(sorted(d + pd.Timedelta(hours=h) for d in days for h in (6, 9, 12, 15)))

st, coords, grid_runs, _ = load_ecmwf_runs_at_stations_and_grid(
    parquet_path=PATH, station_lats=lats, station_lons=lons,
    features=FEATS, run_times=run_times, freq_h="1h", horizon=48, next_n_grid_per_station=2)

# Datei exakt ueber lat UND lon finden, so wie der Loader es tut
files = {(round(la, 5), round(lo, 5)): fp for la, lo, fp in _list_ecmwf_split_files(PATH)}
ok = bad = 0
first_bad = None
for gi in range(len(coords)):
    key = (round(float(coords[gi, 0]), 5), round(float(coords[gi, 1]), 5))
    fp = files.get(key)
    if fp is None:
        print("keine Datei fuer", key); continue
    g = pd.read_parquet(fp)
    g["starttime"] = pd.to_datetime(g["starttime"], utc=True)
    g["forecasttime"] = g["forecasttime"].astype(int)
    g = g.set_index(["starttime", "forecasttime"]).sort_index()
    starts = pd.DatetimeIndex(g.index.get_level_values(0).unique()).sort_values()
    for ri, t0 in enumerate(run_times):
        hs = _ecmwf_run_for(t0, starts)
        lead0 = int((t0 - hs).total_seconds() // 3600)
        for h in (1, 12, 24, 36, 48):
            try:
                row = g.loc[(hs, lead0 + h)]
            except KeyError:
                continue
            u, v = float(row["u_wind10m"]), float(row["v_wind10m"])
            exp = {"u_wind10m": u, "v_wind10m": v, "wind_speed_10m": float(np.hypot(u, v))}
            for fi, feat in enumerate(FEATS):
                a = float(grid_runs[ri, h - 1, gi, fi])
                if np.isfinite(a) and abs(a - exp[feat]) < 1e-3:
                    ok += 1
                else:
                    bad += 1
                    if first_bad is None:
                        first_bad = (key, t0, hs, lead0 + h, feat, a, exp[feat])
print("Abgleich gegen die Quelldateien: %d korrekt, %d abweichend" % (ok, bad))
if first_bad:
    print("erste Abweichung:", first_bad)
