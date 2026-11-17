#!/usr/bin/env python
"""Zero-shot TabPFN-TS on the six operating parks with the forecaster's NWP inputs as known covariates (2026-10-20),
same series and windows as scripts/pv/chronos2_zeroshot.py. Every (park, issue time) is one item of a single predict_df
call: context = past power and covariates (48 h as the TFT, or 14 days), future = the 96 covariate rows of the run.
Local TabPFN (GPU), median point prediction, no fine-tuning.

Run on l1:  CUDA_VISIBLE_DEVICES=1 ~/venv_fm/bin/python scripts/pv/tabpfn_zeroshot.py [--context 48h|14d]
Output: ${DATA_ROOT}/synthetic/solar/pc_nwp_baseline/fm/tabpfnts_<context>.pkl  {park: {pred: DataFrame(issue x t+1..t+96)}}
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chronos2_zeroshot import COVS, FRAMES, H, OUT, PARKS, WINDOWS, series  # noqa: E402

CONTEXTS = {"48h": 96, "14d": 96 * 14}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="48h", choices=list(CONTEXTS))
    a = ap.parse_args()
    import tabpfn_time_series as T
    Pipe = getattr(T, "TabPFNTSPipeline", None) or getattr(T, "TabPFNTimeSeriesPipeline")
    pipe = Pipe(max_context_length=CONTEXTS[a.context] + 8)
    clen = CONTEXTS[a.context]
    ctx_rows, fut_rows, meta = [], [], {}
    for park in PARKS:
        hist, runs = series(park)
        issues = [st for st in sorted(runs) if any(pd.Timestamp(x, tz="UTC") <= st <= pd.Timestamp(y, tz="UTC") + pd.Timedelta("23h") for x, y in WINDOWS)]
        for st in issues:
            r = runs[st]
            if len(r) < H:
                continue
            past = hist.loc[:st - pd.Timedelta("30min")].tail(clen)
            if past["power"].isna().mean() > 0.2 or len(past) < 48:
                continue
            past = past.copy()
            past["power"] = past["power"].interpolate(limit_direction="both")
            for c in COVS:
                past[c] = past[c].interpolate(limit_direction="both").fillna(0.0)
            item = f"{park}|{st.isoformat()}"
            ctx_rows.append(pd.DataFrame({"item_id": item, "timestamp": past.index.tz_convert(None), "target": past["power"].values,
                                          **{c: past[c].values for c in COVS}}))
            fut_rows.append(pd.DataFrame({"item_id": item, "timestamp": pd.to_datetime(r["timestamp"]).dt.tz_convert(None).values[:H],
                                          **{c: r[c].fillna(0.0).values[:H] for c in COVS}}))
            meta[item] = (park, st)
    ctx = pd.concat(ctx_rows, ignore_index=True)
    fut = pd.concat(fut_rows, ignore_index=True)
    print(a.context, "items", len(meta), "context rows", len(ctx), flush=True)
    out = pipe.predict_df(ctx, future_df=fut, prediction_length=H, quantiles=[0.5])
    out = out.reset_index() if "item_id" not in out.columns else out
    preds = {p: {} for p in PARKS}
    for item, g in out.groupby("item_id"):
        park, st = meta[item]
        g = g.sort_values("timestamp")
        col = "target" if "target" in g.columns else ("0.5" if "0.5" in g.columns else g.columns[-1])
        preds[park][st] = np.clip(g[col].values[:H].astype(float), 0.0, None)
    result = {}
    for park in PARKS:
        P = pd.DataFrame.from_dict(preds[park], orient="index", columns=[f"t+{k + 1}" for k in range(H)]).sort_index()
        result[park] = {"pred": P}
        print(park, P.shape, flush=True)
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(result, open(os.path.join(OUT, f"tabpfnts_{a.context}.pkl"), "wb"))
    print("done")


if __name__ == "__main__":
    main()
