"""Belegmessung zur Umstellung der wind_speed-Imputation auf die TFT-Werte.

Gehoert zu docs/imputation_tft_switch.md Abschnitt 3. Rein lesend, aendert nichts.

Read-only. Laedt die Stationsmessungen einer echten Fold-Config und vergleicht
drei Fuellquellen auf DEMSELBEN meas_raw:
  (a) Kriging  rk_pred aus der Sicherung wind_vor_tft_20260902  (Stand vor 2026-08-11)
  (b) ERA5-OLS utils/era5_imputation.py                          (Stand bis 2026-09-02)
  (c) TFT      imputed  aus interpol/wind                        (neu)
"""
import sys, os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from geostatistics.train_stgnn2 import load_yaml, load_station_measurements
from utils.imputation import load_gap_imputation, resolve_imputation_column
from utils.era5_imputation import load_era5_imputation

CFG = os.environ.get("MEASURE_CFG", "configs/expwin/step1/config_wind_mtgnn_nwp_fold1.yaml")
NEW_DIR = "/mnt/lambda1/nvme1/synthetic/interpol/wind"
OLD_DIR = "/mnt/lambda1/nvme1/synthetic/interpol/wind_vor_tft_20260902"

cfg = load_yaml(str(REPO / CFG))
d = cfg["data"]
m = cfg["mtgnn"]
all_ids = [str(s) for s in d["files"] + d["val_files"] + d["test_files"]]
cols = list(m["measurement_features"])
target = m["target_col"]

print(f"Config      : {CFG}")
print(f"Stationen   : {len(all_ids)}  (files+val_files+test_files)")
print(f"Spalten     : {cols}, Ziel '{target}', freq {d.get('freq','1h')}")

meas_raw, timestamps = load_station_measurements(
    d["path"], all_ids, cols=cols, freq=d.get("freq", "1h"),
    use_case=d.get("use_case", "wind"),
    stations_master=str(REPO / d["stations_master"]),
    time_label=cfg.get("params", {}).get("measurement_time_label", "right"),
)
tidx = cols.index(target)
print(f"Zeitraster  : T={len(timestamps)}  {timestamps[0]} … {timestamps[-1]}")

base = meas_raw[:, :, tidx].copy()
missing = np.isnan(base)
n_missing = int(missing.sum())
n_cells = base.size
print(f"Zellen      : {n_cells}   davon fehlend: {n_missing}  ({100*n_missing/n_cells:.3f} %)")
print()


def dist(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "  (keine)"
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    return (f"n={v.size:>6d}  mean={v.mean():6.3f}  sd={v.std():6.3f}  "
            f"min={q[0]:6.3f}  p05={q[1]:5.3f}  p25={q[2]:5.3f}  med={q[3]:5.3f}  "
            f"p75={q[4]:5.3f}  p95={q[5]:5.3f}  max={q[6]:6.3f}")


rows = []

# (a) Kriging aus der Sicherung
if os.path.isdir(OLD_DIR):
    col = resolve_imputation_column(os.path.join(OLD_DIR, f"Station_{all_ids[0]}.parquet"))
    rk = load_gap_imputation(OLD_DIR, all_ids, timestamps)
    used = np.where(missing & ~np.isnan(rk), rk, np.nan)
    rows.append((f"(a) Kriging '{col}' (Sicherung)", used, None))

# (b) ERA5-OLS
era5, _coefs, era5_diag = load_era5_imputation(all_ids, timestamps, meas_raw.copy(), cols, target)
used_e = np.where(missing & ~np.isnan(era5), era5, np.nan)
rows.append(("(b) ERA5-OLS (bis 2026-09-02)", used_e, era5_diag))

# (c) TFT
col_new = resolve_imputation_column(os.path.join(NEW_DIR, f"Station_{all_ids[0]}.parquet"))
tft, ctx = load_gap_imputation(NEW_DIR, all_ids, timestamps, with_kontextfrei=True)
used_t = np.where(missing & ~np.isnan(tft), tft, np.nan)
rows.append((f"(c) TFT '{col_new}' (neu)", used_t, None))

print("Gefuellte Zellen und Verteilung der EINGESETZTEN Werte")
print("=" * 118)
for name, used, extra in rows:
    n_fill = int(np.isfinite(used).sum())
    print(f"{name:<34s} gefuellt {n_fill:>6d}/{n_missing} "
          f"({100*n_fill/max(n_missing,1):5.1f} %)  offen {n_missing-n_fill:>6d}")
    print(f"{'':34s} {dist(used)}")
    n_stat = int(np.isfinite(used).any(axis=0).sum())
    print(f"{'':34s} betroffene Stationen: {n_stat}/{len(all_ids)}")
    print()

fill_t = missing & np.isfinite(used_t)
print(f"davon kontextfrei (TFT): {int((fill_t & ctx).sum())} "
      f"({100*(fill_t & ctx).sum()/max(fill_t.sum(),1):.1f} % der TFT-Fuellungen)")
print(f"TFT-Werte, die auf eine NICHT fehlende Zelle fielen: "
      f"{int((np.isfinite(tft) & ~missing).sum())}  (Rastergleichheit, erwartet 0)")

both = np.isfinite(used_e) & np.isfinite(used_t)
if both.any():
    de = used_t[both] - used_e[both]
    print(f"\nUeberlappung ERA5 ∩ TFT: {int(both.sum())} Zellen  "
          f"MAE |TFT-ERA5| = {np.abs(de).mean():.3f} m/s  "
          f"Korrelation = {np.corrcoef(used_t[both], used_e[both])[0,1]:.4f}")
only_t = np.isfinite(used_t) & ~np.isfinite(used_e)
only_e = np.isfinite(used_e) & ~np.isfinite(used_t)
print(f"nur TFT: {int(only_t.sum())}   nur ERA5: {int(only_e.sum())}")

# Aufschluesselung der nach der TFT-Fuellung offen gebliebenen Zellen
open_mask = missing & ~np.isfinite(tft)
tft_end = pd.Timestamp("2026-07-31 23:00", tz="UTC")
after_end = (timestamps > tft_end)[:, None] & open_mask
have_file = np.array([os.path.exists(os.path.join(NEW_DIR, f"Station_{s}.parquet")) for s in all_ids])
no_file = (~have_file)[None, :] & open_mask
print(f"\nOffen nach TFT: {int(open_mask.sum())}")
print(f"  davon Stunden nach {tft_end} (TFT-Ende): {int(after_end.sum())}")
print(f"  davon Stationen ohne TFT-Datei ({(~have_file).sum()}: "
      f"{[s for s,h in zip(all_ids,have_file) if not h]}): {int(no_file.sum())}")
print(f"  Rest: {int((open_mask & ~after_end & ~no_file).sum())}")
