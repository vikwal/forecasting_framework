#!/usr/bin/env python3
"""Gain from the trailing 48 h of the target site as a function of the conditions inside that window.

Wagner (2026-09-21, #26): are two calm days worth as much as two windy days?
For every (station, run) pair the RMSE over the 48 lead times of the arm without
site observations and of the same arm reading the trailing 48 h is computed, and
the mean and standard deviation of the observed 10 m wind speed inside the window
[run - 48 h, run) are attached. Aggregation per station first, then over stations.
Output: paper_export/window_conditions.csv and .json.
"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
RAW = "data/raw_preds"; ST = "/mnt/lambda1/nvme1/synthetic/raw/wind"
PAIRS = {"test": [("testmode_mtgnn_nwp", "testmode_mtgnn_nwp_hist_once"),
                  ("testmode_dcrnn_nwp", "testmode_dcrnn_nwp_hist_once")],
         "val":  [(f"retrain_mtgnn_fold{i}", f"retrain_mtgnn_nwp_hist_fold{i}") for i in (1, 2, 3)]
               + [(f"retrain_dcrnn_fold{i}", f"retrain_dcrnn_nwp_hist_fold{i}") for i in (1, 2, 3)]}
def load(stem):
    d = pd.read_parquet(f"{RAW}/{stem}_raw.parquet", columns=["station_id", "run_time", "horizon", "pred", "gt", "nwp_ref"])
    d["station_id"] = d.station_id.astype(str).str.zfill(5)
    return d
win_cache = {}
def window_stats(sid):
    if sid in win_cache: return win_cache[sid]
    s = pd.read_parquet(f"{ST}/Station_{sid}.parquet", columns=["wind_speed"])["wind_speed"]
    h = s.resample("1h").mean()
    m = h.rolling(48, min_periods=40).mean().shift(1)
    sd = h.rolling(48, min_periods=40).std().shift(1)
    win_cache[sid] = (m, sd); return win_cache[sid]
def agg_bin(gb):
    st = gb.groupby("station_id").agg(n=("n", "size"), mse_a=("mse_a", "mean"), mse_b=("mse_b", "mean"), mse_n=("mse_n", "mean"))
    st = st[st.n >= 10]
    ra, rb, rn = np.sqrt(st.mse_a), np.sqrt(st.mse_b), np.sqrt(st.mse_n)
    return dict(n_pairs=int(len(gb)), n_stations=int(len(st)), rmse_raw=float(rn.mean()), rmse_noobs=float(ra.mean()),
                rmse_obs=float(rb.mean()), gain=float((ra - rb).mean()), share_stations_gain_pos=float(((ra - rb) > 0).mean()),
                skill_obs_vs_noobs=float((1 - rb / ra).mean()))
rows_out, summary = [], {}
for split, pairs in PAIRS.items():
    for fam in ("mtgnn", "dcrnn"):
        parts = []
        for a, b in pairs:
            if fam not in a: continue
            try: A, B = load(a), load(b)
            except Exception as e: print("skip", a, b, e, flush=True); continue
            m = A.merge(B, on=["station_id", "run_time", "horizon"], suffixes=("_a", "_b"))
            m = m[m.gt_a.notna()]
            m["se_a"] = (m.pred_a - m.gt_a) ** 2; m["se_b"] = (m.pred_b - m.gt_a) ** 2; m["se_n"] = (m.nwp_ref_a - m.gt_a) ** 2
            g = m.groupby(["station_id", "run_time"]).agg(n=("se_a", "size"), mse_a=("se_a", "mean"), mse_b=("se_b", "mean"), mse_n=("se_n", "mean")).reset_index()
            g = g[g.n >= 40]
            wm, wsd = [], []
            for sid, sub in g.groupby("station_id"):
                mm, ss = window_stats(sid)
                rt = pd.to_datetime(sub.run_time, utc=True)
                wm.append(pd.Series(mm.reindex(rt).values, index=sub.index)); wsd.append(pd.Series(ss.reindex(rt).values, index=sub.index))
            g["win_mean"] = pd.concat(wm); g["win_sd"] = pd.concat(wsd)
            parts.append(g)
        if not parts: continue
        g = pd.concat(parts, ignore_index=True).dropna(subset=["win_mean"])
        g["gain"] = np.sqrt(g.mse_a) - np.sqrt(g.mse_b)   # positive: observations help
        edges = np.quantile(g.win_mean, [0, .2, .4, .6, .8, 1.0])
        g["bin"] = pd.cut(g.win_mean, edges, include_lowest=True, labels=False)
        for b in range(5):
            rows_out.append(dict(split=split, family=fam, by="win_mean", bin=b, lo=float(edges[b]), hi=float(edges[b + 1]), **agg_bin(g[g.bin == b])))
        edges_sd = np.quantile(g.win_sd.dropna(), [0, 1/3, 2/3, 1.0])
        g["bin_sd"] = pd.cut(g.win_sd, edges_sd, include_lowest=True, labels=False)
        for b in range(3):
            rows_out.append(dict(split=split, family=fam, by="win_sd", bin=b, lo=float(edges_sd[b]), hi=float(edges_sd[b + 1]), **agg_bin(g[g.bin_sd == b])))
        cs = [spearmanr(sub.win_mean, sub.gain).correlation for _, sub in g.groupby("station_id") if len(sub) >= 50]
        cs_sd = [spearmanr(sub.win_sd, sub.gain).correlation for _, sub in g.groupby("station_id") if sub.win_sd.notna().sum() >= 50]
        summary[f"{split}_{fam}"] = dict(n_pairs=int(len(g)), n_stations=int(g.station_id.nunique()),
                                         rho_gain_winmean_median=float(np.nanmedian(cs)), rho_gain_winmean_share_pos=float(np.mean(np.array(cs) > 0)),
                                         rho_gain_winsd_median=float(np.nanmedian(cs_sd)), rho_gain_winsd_share_pos=float(np.mean(np.array(cs_sd) > 0)),
                                         pooled_rho_winmean=float(spearmanr(g.win_mean, g.gain).correlation),
                                         gain_mean_of_station_means=float(g.groupby("station_id").gain.mean().mean()))
        print(split, fam, summary[f"{split}_{fam}"], flush=True)
pd.DataFrame(rows_out).to_csv("paper_export/window_conditions.csv", index=False)
json.dump(summary, open("paper_export/window_conditions.json", "w"), indent=1)
print("done", flush=True)
