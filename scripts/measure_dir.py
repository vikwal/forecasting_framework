"""Read-only: vergleicht interpol/wind mit interpol/wind_richtung und misst,
was imputed_dir gegenueber dem KNN-Cache an wind_direction fuellt."""
import sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import yaml
from geostatistics.train_stgnn2 import load_station_measurements
from utils.imputation import load_knn_imputation

BASE = "/mnt/nvme1/synthetic/interpol"

# 1. Sind die Geschwindigkeitsspalten in beiden Baeumen gleich?
n_same = n_diff = 0
maxdiff = 0.0
for f in sorted(glob.glob(f"{BASE}/wind/Station_*.parquet")):
    sid = f.split("Station_")[1]
    g = f"{BASE}/wind_richtung/Station_{sid}"
    a = pd.read_parquet(f, columns=["timestamp", "imputed"])
    b = pd.read_parquet(g, columns=["timestamp", "imputed"])
    if len(a) != len(b):
        n_diff += 1; continue
    va, vb = a["imputed"].to_numpy(), b["imputed"].to_numpy()
    both = ~np.isnan(va) & ~np.isnan(vb)
    if (np.isnan(va) != np.isnan(vb)).any():
        n_diff += 1
    elif np.allclose(va[both], vb[both], equal_nan=True):
        n_same += 1
    else:
        n_diff += 1
    if both.any():
        maxdiff = max(maxdiff, float(np.nanmax(np.abs(va[both] - vb[both]))))
print(f"[1] Spalte 'imputed' (wind_speed): {n_same} Stationen identisch, {n_diff} abweichend, "
      f"max. Abweichung {maxdiff:.6g} m/s")

# 2. Richtungsluecken: was fuellt imputed_dir, was der KNN-Cache?
c = yaml.safe_load(open("configs/testmode/full/config_wind_dcrnn_fold1.yaml"))["data"]
ids = [str(s) for s in c["files"]] + [str(s) for s in c["val_files"]] + [str(s) for s in c["test_files"]]
meas, ts = load_station_measurements(c["path"], ids, cols=["wind_speed", "wind_direction"],
                                     freq="1h", use_case="wind",
                                     stations_master=c.get("stations_master"), time_label="right")
dir_raw = meas[:, :, 1]
gaps = np.isnan(dir_raw)
print(f"[2] Zeitachse {ts[0]} .. {ts[-1]}, {gaps.size} Zellen, davon {int(gaps.sum())} Richtungsluecken "
      f"({100*gaps.mean():.3f} %)")

# KNN
knn = load_knn_imputation(c["knnimputer_path"], "wind_direction", ids, ts, freq="1h")
knn_fill = gaps & ~np.isnan(knn)
print(f"    KNN-Cache fuellt {int(knn_fill.sum())} ({100*knn_fill.sum()/gaps.sum():.1f} %), "
      f"offen {int((gaps & ~knn_fill).sum())}")

# imputed_dir
tft = np.full_like(dir_raw, np.nan)
guete = np.full_like(dir_raw, np.nan)
tmap = pd.Series(np.arange(len(ts)), index=ts)
missing_files = 0
for j, sid in enumerate(ids):
    p = f"{BASE}/wind_richtung/Station_{sid}.parquet"
    try:
        d = pd.read_parquet(p, columns=["timestamp", "imputed_dir", "imputed_dir_guete"])
    except FileNotFoundError:
        missing_files += 1; continue
    idx = tmap.reindex(pd.to_datetime(d["timestamp"], utc=True)).to_numpy()
    ok = ~np.isnan(idx)
    ii = idx[ok].astype(int)
    tft[ii, j] = d["imputed_dir"].to_numpy()[ok]
    guete[ii, j] = d["imputed_dir_guete"].to_numpy()[ok]
tft_fill = gaps & ~np.isnan(tft)
print(f"    imputed_dir fuellt {int(tft_fill.sum())} ({100*tft_fill.sum()/gaps.sum():.1f} %), "
      f"offen {int((gaps & ~tft_fill).sum())}, fehlende Dateien: {missing_files}")
only_tft = int((tft_fill & ~knn_fill).sum()); only_knn = int((knn_fill & ~tft_fill).sum())
print(f"    nur imputed_dir: {only_tft}   nur KNN: {only_knn}")

both = tft_fill & knn_fill
if both.any():
    d_ang = np.abs(tft[both] - knn[both]) % 360
    d_ang = np.minimum(d_ang, 360 - d_ang)
    print(f"    auf {int(both.sum())} gemeinsamen Zellen: mittlere Winkeldifferenz "
          f"{d_ang.mean():.1f} Grad, Median {np.median(d_ang):.1f}, p95 {np.percentile(d_ang,95):.1f}")
g = guete[tft_fill]
print(f"    imputed_dir_guete an den Fuellstellen: Median {np.nanmedian(g):.3f}, "
      f"p05 {np.nanpercentile(g,5):.3f}, Anteil < 0.5: {100*np.nanmean(g<0.5):.1f} %")

# 3. Auditfenster der 9 Configs
for te in ("2025-12-01", "2026-04-01", "2026-07-31"):
    a = int(np.searchsorted(ts, pd.Timestamp(te, tz="UTC"), side="right"))
    rest_knn = int((gaps[:a] & ~knn_fill[:a]).sum())
    rest_tft = int((gaps[:a] & ~tft_fill[:a]).sum())
    print(f"[3] bis {te}: Restluecken Richtung — KNN {rest_knn}, imputed_dir {rest_tft}")
