#!/usr/bin/env python3
"""Runs ON l2 (frcst venv) from ~/Work/forecasting_framework.

Roh-Skill von ICON-D2 je ZIELstation ueber das TRAININGSfenster — die
ex-ante-Kovariate der Terzil-Analyse im Paper Graphs_Wind_Speed_Forecasting
(Fig. "nwpskill", Abschnitt Results/"The site without a record").

Je Station: R² und RMSE des rohen ICON-D2 am naechsten Gitterpunkt gegen die
eigene Messung, ueber alle Laeufe mit Laufzeit VOR der Fenstergrenze des
jeweiligen Splits (Validierung: val_start der Fold-Configs, 2023-07..2024-08;
Testjahr: test_start, 2023-07..2025-08). Die Kovariate braucht die Messreihe
der Station, aber kein gefittetes Modell, und beruehrt das Bewertungsfenster
nicht. Rohmessungen ohne Imputation; Stunden mit NaN werden uebersprungen.

Ausgabe: paper_export/train_window_skill.csv (station_id, train_r2,
train_rmse, n, split, fold) — lokal gespiegelt nach
Graphs_Wind_Speed_Forecasting/pictures/data/ fuer mkfigs_paper.py.

Aufruf:  DATA_ROOT=/mnt/lambda1/nvme1 ./frcst/bin/python scripts/train_window_skill.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geostatistics.train_stgnn2 import (load_station_measurements,
                                        load_station_metadata, load_icond2_ml_runs)

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
DATA_PATH = f"{DATA_ROOT}/synthetic/raw/wind"
NWP_PATH = f"{DATA_ROOT}/icon-d2/parquet"
META = "data/stations_master.csv"
OUT = Path("paper_export/train_window_skill.csv")


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_multi_constructor(
    "!", lambda lo, s, n: lo.construct_scalar(n) if n.id == "scalar" else None)


def _data_cfg(path):
    with open(path) as f:
        return yaml.load(f, Loader=_Loader)["data"]


def _z5(ids):
    return [str(s).zfill(5) for s in ids]


def train_window_skill(ids, boundary):
    """Pro Station R²/RMSE des naechsten ICON-D2-Gitterpunkts ueber alle
    Laeufe < boundary, Leads 1..48, gegen die unimputierte Messung."""
    meas, ts = load_station_measurements(DATA_PATH, ids, cols=["wind_speed"], freq="1h")
    lats, lons, _ = load_station_metadata(DATA_PATH, ids, meta_path=META)
    coords = np.stack([lats, lons], axis=1)
    run_times, _gcoords, gruns, nearest = load_icond2_ml_runs(
        NWP_PATH, ids, coords, ["wind_speed_10m"], next_n_grid=1,
        n_workers=8, cutoff=pd.Timestamp(boundary, tz="UTC"))
    T = len(ts)
    lookup = pd.Series(np.arange(T), index=ts)
    fc_rows, ob_rows = [], []
    for r, rt in enumerate(run_times):
        if rt not in lookup.index:
            continue
        t_abs = int(lookup[rt]) + 1
        if t_abs + 48 > T:
            continue
        ob_rows.append(meas[t_abs:t_abs + 48, :, 0])
        fc_rows.append(gruns[r, :48, :, 0][:, nearest])
    FC = np.concatenate(fc_rows, axis=0)
    OB = np.concatenate(ob_rows, axis=0)
    out = {}
    for j, sid in enumerate(ids):
        o, f = OB[:, j], FC[:, j]
        ok = ~np.isnan(o) & ~np.isnan(f)
        o, f = o[ok], f[ok]
        mse = float(np.mean((f - o) ** 2))
        out[sid] = dict(train_r2=1 - mse / float(o.var()),
                        train_rmse=float(np.sqrt(mse)), n=int(ok.sum()))
    return pd.DataFrame(out).T


def main():
    parts = []
    for fold in (1, 2, 3):
        cfg = _data_cfg(f"configs/baselines/config_wind_qrf_local_fold{fold}.yaml")
        sk = train_window_skill(_z5(cfg["val_files"]), cfg["val_start"])
        sk["split"], sk["fold"] = "val", fold
        parts.append(sk)
        print(f"[fold {fold}] {len(sk)} Stationen, Median train-R2 {sk['train_r2'].median():.3f}",
              flush=True)
    cfg = _data_cfg("configs/baselines/config_wind_qrf_local_fold1.yaml")
    sk = train_window_skill(_z5(cfg["test_files"]), cfg["test_start"])
    sk["split"], sk["fold"] = "test", 0
    parts.append(sk)
    print(f"[test] {len(sk)} Stationen, Median train-R2 {sk['train_r2'].median():.3f}", flush=True)
    df = pd.concat(parts)
    df.index.name = "station_id"
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT)
    print(f"geschrieben: {OUT} ({len(df)} Zeilen)")


if __name__ == "__main__":
    main()
