"""Preflight der finalen Testauswertung (docs/handoff_testmode.md).

Laedt fuer jede Config unter configs/testmode/ die Messreihen, die
TFT-Imputation (wind_speed) und die KNN-Imputation der Sekundaerspalten —
genau die Kette aus train_dcrnn.py, aber ohne NWP, ohne GPU, ohne Training —
und meldet die NaN-Bilanz JE SPALTE im jeweiligen Auditfenster [0, test_end].

Der Sinn ist der Spaltenschnitt: die Gesamtzahl allein verdeckt, dass
wind_speed und wind_direction aus verschiedenen Quellen mit verschiedener
Reichweite kommen. Genau daran ist der Lauf vom 2026-09-02 zuerst gescheitert
(KNN-Cache fuer wind_direction endete 2026-07-14, s.
docs/imputation_knn_regen_20260902.md).

Read-only und wiederholbar. Vor jedem Start der Queue laufen lassen.
"""
import sys, glob, yaml
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from geostatistics.train_stgnn2 import load_station_measurements
from utils.imputation import (impute_meas_raw_from_interpol,
                              load_knn_imputation, apply_knn_imputation)

cfgs = sorted(glob.glob("configs/testmode/*/*.yaml"))
cache = {}
for f in cfgs:
    c = yaml.safe_load(open(f)); d = c["data"]
    key = "dcrnn" if "dcrnn" in f else "mtgnn"
    mcfg = c.get(key, {})
    cols = list(mcfg.get("measurement_features", ["wind_speed", "wind_direction"]))
    target = mcfg.get("target_col", "wind_speed")
    ids = [str(s) for s in d["files"]] + [str(s) for s in d["val_files"]] + [str(s) for s in d["test_files"]]
    ck = (tuple(ids), tuple(cols))
    if ck not in cache:
        meas, ts = load_station_measurements(
            d["path"], ids, cols=cols, freq=d.get("freq", "1h"),
            use_case=d.get("use_case", "wind"), stations_master=d.get("stations_master"),
            time_label=c.get("params", {}).get("measurement_time_label", "right"))
        meas, diag = impute_meas_raw_from_interpol(meas, ids, ts, cols, d["interpol_path"], target)
        for sec in [x for x in cols if x != target]:
            if int(np.isnan(meas[:, :, cols.index(sec)]).sum()) == 0:
                continue
            knn = load_knn_imputation(d["knnimputer_path"], sec, ids, ts, freq=d.get("freq", "1h"))
            meas = apply_knn_imputation(meas, knn, cols, sec)
        cache[ck] = (meas, ts)
        print(f"  [geladen] {len(ids)} St., T={len(ts)}, {ts[0]} .. {ts[-1]}", flush=True)
    meas, ts = cache[ck]
    te = pd.Timestamp(d["test_end"], tz="UTC")
    cut = int(np.searchsorted(ts, te + pd.Timedelta(days=2), side="right"))
    audit = int(np.searchsorted(ts[:cut], te, side="right"))
    m = meas[:cut]
    per_col = {cols[ci]: int(np.isnan(m[:audit, :, ci]).sum()) for ci in range(len(cols))}
    nan_any = np.isnan(m[:audit]).any(axis=2)
    bad = int((nan_any.sum(axis=0) > 0).sum())
    beyond = int(np.isnan(m[audit:]).any(axis=2).sum())
    print(f"{f:62s} test={d['test_start']}..{d['test_end']} audit_bis={ts[audit-1]} "
          f"NaN-Stationen={bad} NaN-jenseits={beyond} per_col={per_col}", flush=True)
