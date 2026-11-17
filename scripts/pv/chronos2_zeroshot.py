#!/usr/bin/env python
"""Zero-shot Chronos-2 on the six operating parks with the forecaster's NWP inputs as covariates (2026-10-20).

Series per park from the PC frames (pc_nwp_baseline/frames/<pv>_meter.parquet: one row per issue time and lead, with
the measured power and the covariates gti_nwp, ecmwf_gti, gti_clearsky): the gap-free 30-min history is horizon day 0
of every run, the future covariates of a run are its own leads 0..95. For every issue time of the two test windows the
model gets the past power and past covariates (context 48 h as the TFT, and 14 days as a second variant) and the
96 future covariates, and returns the median of the next 96 steps. No fine-tuning.

Run on l1 with the framework env (chronos-forecasting 2.2.2, amazon/chronos-2 cached):
  CUDA_VISIBLE_DEVICES=0 frcst/bin/python scripts/pv/chronos2_zeroshot.py
Output: ${DATA_ROOT}/synthetic/solar/pc_nwp_baseline/fm/chronos2_<context>.pkl  {park: {pred: DataFrame(issue x t+1..t+96)}}
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/nvme1")
FRAMES = f"{DATA_ROOT}/synthetic/solar/pc_nwp_baseline/frames"
OUT = f"{DATA_ROOT}/synthetic/solar/pc_nwp_baseline/fm"
PARKS = ["pv1_meter", "pv2_meter", "pv3_meter", "pv5_meter", "pv6_meter", "pv7_meter"]
WINDOWS = [("2023-12-01", "2024-02-29"), ("2024-03-01", "2024-05-31")]
COVS = ["gti_nwp", "ecmwf_gti", "gti_clearsky"]
CONTEXTS = {"48h": 96, "14d": 96 * 14}
H = 96


def series(park: str):
    f = pd.read_parquet(os.path.join(FRAMES, f"{park}.parquet"))
    f["starttime"] = pd.to_datetime(f["starttime"], utc=True)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    hist = f[f.forecasttime < 48].drop_duplicates("timestamp").set_index("timestamp").sort_index()
    hist = hist[["power"] + COVS].reindex(pd.date_range(hist.index[0], f.timestamp.max(), freq="30min", tz="UTC"))
    runs = {st: g.sort_values("forecasttime") for st, g in f.groupby("starttime")}
    return hist, runs


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    for cname, clen in CONTEXTS.items():
        result = {}
        for park in PARKS:
            hist, runs = series(park)
            issues = [st for st in sorted(runs) if any(pd.Timestamp(a, tz="UTC") <= st <= pd.Timestamp(b, tz="UTC") + pd.Timedelta("23h") for a, b in WINDOWS)]
            preds = {}
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
                ctx = pd.DataFrame({"id": park, "timestamp": past.index.tz_convert(None), "target": past["power"].values,
                                    **{c: past[c].values for c in COVS}})
                fut = pd.DataFrame({"id": park, "timestamp": pd.to_datetime(r["timestamp"]).dt.tz_convert(None).values[:H],
                                    **{c: r[c].fillna(0.0).values[:H] for c in COVS}})
                out = pipe.predict_df(ctx, future_df=fut, prediction_length=H, quantile_levels=[0.5], id_column="id",
                                      timestamp_column="timestamp", target="target")
                col = "0.5" if "0.5" in out.columns else [c for c in out.columns if "0.5" in str(c)][0]
                preds[st] = np.clip(out[col].values[:H], 0.0, None)
            P = pd.DataFrame.from_dict(preds, orient="index", columns=[f"t+{k + 1}" for k in range(H)]).sort_index()
            result[park] = {"pred": P}
            print(cname, park, P.shape, flush=True)
        pickle.dump(result, open(os.path.join(OUT, f"chronos2_{cname}.pkl"), "wb"))
    print("done")


if __name__ == "__main__":
    main()
