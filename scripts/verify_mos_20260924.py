"""MOS-Pfad: stammt der HRES-Praediktor aus dem richtigen Lauf?"""
import numpy as np, pandas as pd, glob
import geostatistics.baselines.dataset as ds
from geostatistics.train_stgnn2 import _ecmwf_run_for, _list_ecmwf_split_files

CFG = "configs/baselines/config_wind_qrf_local_fold1.yaml"
ctx = ds.load_context(CFG)
print("Kontext geladen. HRES-Form:", None if ctx["grid_ecmwf_raw"] is None
      else ctx["grid_ecmwf_raw"].shape)
assert ctx["grid_ecmwf_raw"] is not None, "kein HRES im Kontext"
assert ctx["grid_ecmwf_raw"].ndim == 4, "HRES ist nicht lauf-indiziert!"

run_times = pd.DatetimeIndex(ctx["run_times"])
F_h = ctx["F_h"]
pairs_all = ds.build_run_pairs(ctx, None, None)
pairs = (pairs_all[0] if isinstance(pairs_all, tuple) else pairs_all)[:4]
print("Laufpaare fuer den Test:", len(pairs))

rows = ds.build_mos_rows(ctx, np.array([0, 1]), pairs, nwp_sources="both")
print("MOS-Zeilen:", len(rows), " Spalten:", list(rows.columns))
print("ws_e2 NaN-Anteil: %.3f" % rows["ws_e2"].isna().mean())

# unabhaengig gegen die Quelldatei pruefen
PATH = "/mnt/lambda1/nvme1/ecmwf/parquet/SL"
files = {(round(la, 5), round(lo, 5)): fp for la, lo, fp in _list_ecmwf_split_files(PATH)}
coords = ctx["ecmwf_coords"]; nearest = ctx["nearest_e2_idx"]
ws_idx = ctx["ecmwf_ws_feat_idx"]
gi = int(nearest[0][0])
key = (round(float(coords[gi][0]), 5), round(float(coords[gi][1]), 5))
g = pd.read_parquet(files[key])
g["starttime"] = pd.to_datetime(g["starttime"], utc=True)
g["forecasttime"] = g["forecasttime"].astype(int)
g["ws"] = np.hypot(g["u_wind10m"], g["v_wind10m"])
g = g.set_index(["starttime", "forecasttime"]).sort_index()
starts = pd.DatetimeIndex(g.index.get_level_values(0).unique()).sort_values()

sid = ctx["all_ids"][0]
sub = rows[rows["station_id"] == sid]
ok = bad = 0
for _, r in sub.iterrows():
    t0 = pd.Timestamp(r["run_time"]); h = int(r["horizon"])
    hs = _ecmwf_run_for(t0, starts)
    lead = int((t0 - hs).total_seconds() // 3600) + h
    try:
        exp = float(g.loc[(hs, lead), "ws"])
    except KeyError:
        continue
    got = float(r["ws_e2"])
    if np.isfinite(got) and abs(got - exp) < 1e-3:
        ok += 1
    else:
        bad += 1
        if bad == 1:
            print("erste Abweichung: run=%s h=%d HRES-Lauf=%s lead=%d  ws_e2=%.4f erwartet=%.4f"
                  % (t0, h, hs, lead, got, exp))
print("ws_e2 gegen die Quelldatei: %d korrekt, %d abweichend" % (ok, bad))
