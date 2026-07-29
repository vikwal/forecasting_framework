"""
Diagnostic ONLY — no config/code/DB changes, no training jobs.

Question: given that the 'hist' observed tensor demonstrably contains the correct
own-station history, how much val_rmse improvement was *available* at all?

Method: ridge-regression probe on the real prepared tensors, mirroring the actual
HPO setup (train on one set of stations, validate on HELD-OUT stations; features
per-station standardised, target raw m/s). Compares
    base-like : NWP known @ lead h  +  neighbour observed history
    hist-like : same + own-station observed history
per forecast lead and averaged over the 48 h horizon.
"""
import sys, logging, pickle, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, '.')
from utils import tools, preprocessing

logging.basicConfig(level=logging.ERROR)
CACHE = '/tmp/claude-1003/-home-viktor-Work/367db02d-72d0-4c0d-be17-dd6e0a6c8db8/scratchpad/probe_tensors.pkl'

TRAIN_ST = ['00183', '00460', '00856', '05404']
VAL_ST   = ['01550', '02907']          # from config data.val_files → unseen stations


def load(stations):
    cfg = tools.load_config('configs/tft_bc/config_wind_tft_hist.yaml')
    feats = preprocessing.get_features(cfg)
    out = {}
    for s in stations:
        c = tools.load_config('configs/tft_bc/config_wind_tft_hist.yaml')
        c['data']['files'] = [s]
        c['data']['val_files'] = []
        dfs = preprocessing.get_data(data_dir=c['data']['path'], freq=c['data']['freq'],
                                     config=c, features=feats,
                                     target_col=c['data']['target_col'])
        df = list(dfs.values())[0]
        prep, _ = preprocessing.pipeline(data=df, config=c,
                                         known_cols=feats['known'],
                                         observed_cols=feats['observed'],
                                         static_cols=feats['static'],
                                         target_col=c['data']['target_col'])
        out[s] = (prep['X_train']['observed'], prep['X_train']['known'], prep['y_train'])
        print(f"  loaded {s}: obs{out[s][0].shape} known{out[s][1].shape} y{out[s][2].shape}", flush=True)
    return out


if os.path.exists(CACHE):
    print("loading cached tensors")
    data = pickle.load(open(CACHE, 'rb'))
else:
    print("preprocessing stations (this takes a few minutes)…", flush=True)
    data = load(TRAIN_ST + VAL_ST)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pickle.dump(data, open(CACHE, 'wb'))

LOOKBACK = 48
OWN = 0                                  # observed column index of 'wind_speed'
NEI = slice(1, None)                     # neighbour columns


def design(obs, known, h, use_own):
    """Features for predicting lead h (0-based)."""
    parts = [known[:, LOOKBACK + h, :]]                       # NWP at that lead
    parts.append(obs[:, -3:, NEI].reshape(len(obs), -1))      # neighbours, last 3 h
    parts.append(obs[:, :, NEI].mean(axis=1))                 # neighbours, 48 h mean
    if use_own:
        parts.append(obs[:, -3:, OWN])                        # own, last 3 h
        parts.append(obs[:, :, OWN].mean(axis=1, keepdims=True))
    return np.nan_to_num(np.concatenate(parts, axis=1))


def stack(stations):
    o = np.concatenate([data[s][0] for s in stations])
    k = np.concatenate([data[s][1] for s in stations])
    y = np.concatenate([data[s][2] for s in stations])
    return o, k, y


otr, ktr, ytr = stack(TRAIN_ST)
ova, kva, yva = stack(VAL_ST)
print(f"\ntrain windows {len(ytr)} ({TRAIN_ST})   val windows {len(yva)} ({VAL_ST}, held-out stations)")

rows = []
for h in range(48):
    r = {'lead': h + 1}
    for name, use_own in (('base', False), ('hist', True)):
        Xt, Xv = design(otr, ktr, h, use_own), design(ova, kva, h, use_own)
        m = Ridge(alpha=1.0).fit(Xt, ytr[:, h])
        r[name] = float(np.sqrt(np.mean((m.predict(Xv) - yva[:, h]) ** 2)))
    # persistence for context: last observed own value (per-station scaled → skip)
    rows.append(r)

df = pd.DataFrame(rows)
df['gain'] = df['base'] - df['hist']

print("\n  lead   base_rmse   hist_rmse    gain(m/s)")
for _, r in df.iterrows():
    if r['lead'] in (1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48):
        print(f"  {int(r['lead']):4d}     {r['base']:.4f}      {r['hist']:.4f}     {r['gain']:+.4f}")

print(f"\n  HORIZON MEAN (all 48 leads):  base={df['base'].mean():.4f}  "
      f"hist={df['hist'].mean():.4f}  gain={df['gain'].mean():+.4f} m/s")
print(f"  gain over leads  1–6 h : {df[df.lead <= 6]['gain'].mean():+.4f} m/s")
print(f"  gain over leads  7–24 h: {df[(df.lead > 6) & (df.lead <= 24)]['gain'].mean():+.4f} m/s")
print(f"  gain over leads 25–48 h: {df[df.lead > 24]['gain'].mean():+.4f} m/s")

# How redundant is own history given the neighbours?
own = otr[:, :, OWN].reshape(-1)
nei = otr[:, :, NEI].reshape(-1, otr.shape[-1] - 1)
m = np.isfinite(own) & np.isfinite(nei).all(axis=1)
r2 = Ridge(alpha=1.0).fit(nei[m], own[m]).score(nei[m], own[m])
print(f"\n  R² of own wind_speed regressed on the 5 neighbour series: {r2:.3f}")
print(f"  → {r2*100:.0f}% of the own-history signal is already present in 'base'.")
