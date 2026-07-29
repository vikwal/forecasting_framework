"""
Diagnostic ONLY — no config/code/DB changes, no training jobs.

Compares the TFT-BC HPO scaling regime against the DCRNN/MTGNN/WaveNet regime
(geostatistics/train_dcrnn.py:750-796), using the same ridge probe on real tensors,
validated on HELD-OUT stations. All RMSEs reported in m/s so the variants are
directly comparable.

  A  per-station scaler_x, raw-m/s target      <- current hpo_cl_tft_bc.py path
  B  global scaler_x, raw-m/s target           <- isolates the feature-scaling change
  C  global scaler_x, globally z-scored target <- the DCRNN/MTGNN recipe
"""
import sys, os, pickle, logging
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
from utils import tools, preprocessing

logging.basicConfig(level=logging.ERROR)
CACHE = '/tmp/claude-1003/-home-viktor-Work/367db02d-72d0-4c0d-be17-dd6e0a6c8db8/scratchpad/probe_phys.pkl'
TRAIN_ST = ['00183', '00460', '00856', '05404']
VAL_ST   = ['01550', '02907']
LOOKBACK, OWN, NEI = 48, 0, slice(1, None)


def load(stations):
    """Return tensors in PHYSICAL units (per-station scaling undone)."""
    feats = preprocessing.get_features(tools.load_config('configs/tft_bc/config_wind_tft_hist.yaml'))
    out = {}
    for s in stations:
        c = tools.load_config('configs/tft_bc/config_wind_tft_hist.yaml')
        c['data']['files'] = [s]; c['data']['val_files'] = []
        dfs = preprocessing.get_data(data_dir=c['data']['path'], freq=c['data']['freq'],
                                     config=c, features=feats, target_col=c['data']['target_col'])
        prep, _ = preprocessing.pipeline(data=list(dfs.values())[0], config=c,
                                         known_cols=feats['known'], observed_cols=feats['observed'],
                                         static_cols=feats['static'], target_col=c['data']['target_col'])
        o, k, y = prep['X_train']['observed'], prep['X_train']['known'], prep['y_train']
        so, sk = prep['scalers']['x_observed'], prep['scalers']['x_known']
        o_phys = so.inverse_transform(o.reshape(-1, o.shape[-1])).reshape(o.shape)
        k_phys = sk.inverse_transform(k.reshape(-1, k.shape[-1])).reshape(k.shape)
        out[s] = (o_phys, k_phys, y)
        print(f"  {s}: obs{o.shape} y mean={y.mean():.2f}", flush=True)
    return out


if os.path.exists(CACHE):
    data = pickle.load(open(CACHE, 'rb')); print("cached tensors loaded")
else:
    print("preprocessing (physical units)…", flush=True)
    data = load(TRAIN_ST + VAL_ST)
    pickle.dump(data, open(CACHE, 'wb'))


def design(obs, known, h, use_own):
    p = [known[:, LOOKBACK + h, :], obs[:, -3:, NEI].reshape(len(obs), -1), obs[:, :, NEI].mean(axis=1)]
    if use_own:
        p += [obs[:, -3:, OWN], obs[:, :, OWN].mean(axis=1, keepdims=True)]
    return np.nan_to_num(np.concatenate(p, axis=1))


def build(variant):
    """Returns (otr,ktr,ytr, ova,kva,yva, y_backscale)."""
    def cat(st, i): return np.concatenate([data[s][i] for s in st])
    otr_p, ktr_p, ytr = cat(TRAIN_ST, 0), cat(TRAIN_ST, 1), cat(TRAIN_ST, 2)
    ova_p, kva_p, yva = cat(VAL_ST, 0),  cat(VAL_ST, 1),  cat(VAL_ST, 2)

    if variant == 'A':      # per-station scaling, exactly as the pipeline produces it
        def per_st(st, i):
            outs = []
            for s in st:
                a = data[s][i]
                sc = StandardScaler().fit(a.reshape(-1, a.shape[-1]))
                outs.append(sc.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape))
            return np.concatenate(outs)
        return per_st(TRAIN_ST, 0), per_st(TRAIN_ST, 1), ytr, \
               per_st(VAL_ST, 0),   per_st(VAL_ST, 1),   yva, 1.0

    # global scalers, fit on TRAIN stations only (mirrors meas_scaler/i2_scaler in train_dcrnn.py)
    # all wind-speed columns share one mean/std, like meas_scaler does across stations
    so_mean, so_std = otr_p.mean(), otr_p.std()
    sk = StandardScaler().fit(ktr_p.reshape(-1, ktr_p.shape[-1]))
    g = lambda a: (a - so_mean) / so_std
    gk = lambda a: sk.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape)

    if variant == 'B':      # global features, raw-m/s target
        return g(otr_p), gk(ktr_p), ytr, g(ova_p), gk(kva_p), yva, 1.0
    if variant == 'C':      # global features AND target z-scored with the same mean/std
        return g(otr_p), gk(ktr_p), (ytr - so_mean) / so_std, \
               g(ova_p), gk(kva_p), (yva - so_mean) / so_std, so_std


LBL = {'A': 'per-station scaler_x, raw-m/s target   [CURRENT hpo_cl_tft_bc]',
       'B': 'GLOBAL scaler_x, raw-m/s target        [statics would work here]',
       'C': 'GLOBAL scaler_x + z-scored target      [DCRNN/MTGNN recipe]'}

for v in ('A', 'B', 'C'):
    otr, ktr, ytr, ova, kva, yva, bs = build(v)
    res = {}
    for name, use_own in (('base', False), ('hist', True)):
        rm = []
        for h in range(48):
            m = Ridge(alpha=1.0).fit(design(otr, ktr, h, use_own), ytr[:, h])
            rm.append(np.sqrt(np.mean((m.predict(design(ova, kva, h, use_own)) - yva[:, h]) ** 2)))
        res[name] = np.array(rm) * bs          # back to m/s
    b, hh = res['base'], res['hist']
    print(f"\n=== {v}: {LBL[v]} ===")
    print(f"  horizon-mean RMSE [m/s]  base={b.mean():.4f}  hist={hh.mean():.4f}")
    print(f"  gain  {b.mean()-hh.mean():+.4f} m/s   ({100*(b.mean()-hh.mean())/b.mean():+.2f} %)")
    for lo, hi, lb in ((1, 6, ' 1– 6h'), (7, 24, ' 7–24h'), (25, 48, '25–48h')):
        gg = (b[lo-1:hi] - hh[lo-1:hi]).mean()
        print(f"    leads {lb}: {gg:+.4f} m/s  ({100*gg/b[lo-1:hi].mean():+.2f} %)")
