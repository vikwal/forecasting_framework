#!/usr/bin/env python3
"""Prueft, ob die beiden Aggregationswege bei freq='1h' dasselbe liefern.

l1 aggregiert mit  raw.resample("1h", closed="left", label="left").mean()
l2 aggregiert mit  solar.resample_interval_mean(raw, "1h", sample_seconds, max_nan_frac=1.0)

Der Kommentar auf l2 behauptet Gleichheit bei 1 h. Das hier misst es.
Auf l2 auszufuehren, dort liegt utils/solar.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.home() / "Work" / "forecasting_framework"
sys.path.insert(0, str(REPO))
from utils import solar  # noqa: E402

CANDS = ["/mnt/lambda1/nvme1/synthetic/raw/wind", "/mnt/nvme1/synthetic/raw/wind"]
PATH = next((c for c in CANDS if Path(c).is_dir()), CANDS[0])
STATIONS = ["00183", "04625", "03925", "00096", "05426"]


def agg_left(frame):
    return frame.resample("1h", closed="left", label="left").mean()


def agg_solar(frame, ss):
    return solar.resample_interval_mean(frame, "1h", ss, max_nan_frac=1.0)


def main():
    print(f"Rohdaten: {PATH}")
    head = f"{'Station':9s} {'Spalte':15s} {'max|Delta|':>11s} {'gleich':>7s} {'Zeilen':>7s} {'NaN-Abw.':>9s}"
    print(head)
    print("-" * len(head))
    worst = 0.0
    for sid in STATIONS:
        for col in ["wind_speed", "wind_direction"]:
            df = pd.read_parquet(f"{PATH}/Station_{sid}.parquet", columns=[col])
            df.index = pd.to_datetime(df.index, utc=True)
            raw = df.rename(columns={col: sid}).sort_index()
            ss = solar.infer_sample_seconds(raw.index)
            if col == "wind_direction":
                rad = np.deg2rad(raw.values)
                s = pd.DataFrame(np.sin(rad), index=raw.index, columns=raw.columns)
                c = pd.DataFrame(np.cos(rad), index=raw.index, columns=raw.columns)
                a1 = np.rad2deg(np.arctan2(agg_left(s).values, agg_left(c).values)) % 360
                a2 = np.rad2deg(np.arctan2(agg_solar(s, ss).values, agg_solar(c, ss).values)) % 360
            else:
                a1 = agg_left(raw).values
                a2 = agg_solar(raw, ss).values
            if a1.shape != a2.shape:
                print(f"{sid:9s} {col:15s}  FORM ABWEICHEND {a1.shape} gegen {a2.shape}")
                continue
            m = np.isfinite(a1) & np.isfinite(a2)
            d = np.abs(a1[m] - a2[m]) if m.any() else np.array([0.0])
            nan_diff = int((np.isnan(a1) != np.isnan(a2)).sum())
            worst = max(worst, float(d.max()))
            print(f"{sid:9s} {col:15s} {d.max():11.3e} "
                  f"{str(bool(np.array_equal(a1[m], a2[m]))):>7s} {a1.shape[0]:7d} {nan_diff:9d}")
    print(f"\ngroesste Abweichung ueber alle Faelle: {worst:.3e}")
    print("Bewertung: unter 1e-6 ist es Rundung, darueber ein echter Unterschied.")


if __name__ == "__main__":
    main()
