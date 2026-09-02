#!/usr/bin/env python3
"""Gefilterte Papierzahlen fuer die Retrains: per Station, ohne imputierte
Zielstunden, gemittelt ueber die drei Folds.

Benutzt die Maskierungs- und Mittelungslogik aus
archiv/baselines_verification/verify_baselines.py, damit es genau EINE Formel
gibt (siehe compute_filtered_mos.py). Zusaetzlich zu 'pred' werden 'nwp_ref'
(ICON-D2) und 'pers_ref' (Persistenz) auf derselben gefilterten Basis
gerechnet: sie muessen die dokumentierten 1.304 bzw. 2.240 treffen, sonst
stimmt die Filterung nicht und keine der Zahlen ist zu gebrauchen.

    filtered_table.py <REPO> <arm>[,<arm>...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/viktor/Work/forecasting_framework"
ARMS = (sys.argv[2].split(",") if len(sys.argv) > 2
        else ["dcrnn", "dcrnn_base", "dcrnn_idw_alt"])
sys.path.insert(0, REPO)

from archiv.baselines_verification.verify_baselines import (   # noqa: E402
    build_imputation_mask, _lookup_imputed, _IMPUTATION_MASK_CACHE,
)
from geostatistics.train_stgnn2 import load_yaml               # noqa: E402

RAW_DIR = load_yaml(f"{REPO}/configs/dcrnn/config_wind_dcrnn_fold1.yaml")["data"]["path"]
print(f"Rohdaten fuer die Imputationsmaske: {RAW_DIR}\n")

COLS = ["station_id", "valid_time", "horizon", "pred", "gt", "nwp_ref", "pers_ref"]


def filtered_per_station_mean(pq: Path) -> dict:
    """Identische Formel wie _imputation_filtered_station_mean_rmse, nur auf
    mehrere Vorhersagespalten angewandt (dieselbe Maske, dieselbe Mittelung)."""
    df = pd.read_parquet(pq, columns=COLS)
    df["station_id"] = df["station_id"].astype(str).str.zfill(5)
    sids = sorted(df["station_id"].unique())
    missing = [s for s in sids if s not in _IMPUTATION_MASK_CACHE]
    if missing:
        _IMPUTATION_MASK_CACHE.update(build_imputation_mask(missing, Path(RAW_DIR)))
    is_imp = _lookup_imputed(df, {s: _IMPUTATION_MASK_CACHE[s] for s in sids})
    d = df[~is_imp]
    out = {"n_rows": len(df), "share_imputed": float(is_imp.mean()),
           "n_stations": d["station_id"].nunique()}
    gt = "gt"
    for col in ("pred", "nwp_ref", "pers_ref"):
        per_st = d.groupby("station_id").apply(
            lambda g, c=col: float(np.sqrt(np.mean(
                (g[c].to_numpy(dtype=np.float64) - g[gt].to_numpy(dtype=np.float64)) ** 2))),
            include_groups=False)
        out[col] = float(per_st.mean())
    return out


rows = []
for arm in ARMS:
    per_fold = []
    for fold in (1, 2, 3):
        pq = Path(REPO) / "data/raw_preds" / f"retrain_{arm}_fold{fold}_raw.parquet"
        if not pq.exists():
            print(f"  {arm} fold{fold}: parquet fehlt noch ({pq.name})")
            continue
        r = filtered_per_station_mean(pq)
        per_fold.append(r)
        print(f"  {arm:14s} fold{fold}: modell={r['pred']:.4f}  icon_d2={r['nwp_ref']:.4f}  "
              f"persistenz={r['pers_ref']:.4f}  stationen={r['n_stations']}  "
              f"imputiert={r['share_imputed']:.2%}")
    if per_fold:
        rows.append(dict(
            arm=arm, n_folds=len(per_fold),
            modell=np.mean([r["pred"] for r in per_fold]),
            icon_d2=np.mean([r["nwp_ref"] for r in per_fold]),
            persistenz=np.mean([r["pers_ref"] for r in per_fold]),
        ))

if rows:
    print("\n=== gefiltert, per Station, Mittel ueber die vorhandenen Folds ===")
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Kontrolle der Filterung gegen die dokumentierten Referenzen ===")
    print(f"  ICON-D2    gerechnet {t['icon_d2'].mean():.4f}  erwartet 1.304  "
          f"Abweichung {abs(t['icon_d2'].mean()-1.304):.4f}")
    print(f"  Persistenz gerechnet {t['persistenz'].mean():.4f}  erwartet 2.240  "
          f"Abweichung {abs(t['persistenz'].mean()-2.240):.4f}")
    print("  Treffen diese beiden nicht, ist die Filterung falsch und die "
          "Modellzahlen sind nicht vergleichbar.")
