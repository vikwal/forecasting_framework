"""
Diagnostic ONLY. Follow-up to repro_tft_hist_probe.py.

Hypothesis: observed features are standardised PER STATION (apply_scaling in
prepare_data_for_tft's local-scaling branch), while the target y stays in RAW m/s
(scale_target=False for target_col='wind_speed', preprocessing.py:377). The own-station
history's dominant value is a near-persistence relation y ≈ own_history, which is
scale-dependent — after per-station z-scoring the model can only exploit it if it can
recover that station's mean/std, which it is never given. Neighbour columns are far less
affected (they are a different station's series anyway).

Test: same probe, but with the target also standardised per station. If the own-history
advantage is much larger in that space, the raw-target scale mismatch is what eats it.
"""
import sys, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

CACHE = '/tmp/claude-1003/-home-viktor-Work/367db02d-72d0-4c0d-be17-dd6e0a6c8db8/scratchpad/probe_tensors.pkl'
data = pickle.load(open(CACHE, 'rb'))

TRAIN_ST = ['00183', '00460', '00856', '05404']
VAL_ST   = ['01550', '02907']
LOOKBACK, OWN, NEI = 48, 0, slice(1, None)


def design(obs, known, h, use_own):
    parts = [known[:, LOOKBACK + h, :],
             obs[:, -3:, NEI].reshape(len(obs), -1),
             obs[:, :, NEI].mean(axis=1)]
    if use_own:
        parts.append(obs[:, -3:, OWN])
        parts.append(obs[:, :, OWN].mean(axis=1, keepdims=True))
    return np.nan_to_num(np.concatenate(parts, axis=1))


def stack(stations, ystd):
    """ystd=True -> target standardised with that station's own y mean/std."""
    o, k, y = [], [], []
    for s in stations:
        oo, kk, yy = data[s]
        if ystd:
            yy = (yy - yy.mean()) / yy.std()
        o.append(oo); k.append(kk); y.append(yy)
    return np.concatenate(o), np.concatenate(k), np.concatenate(y)


for ystd in (False, True):
    otr, ktr, ytr = stack(TRAIN_ST, ystd)
    ova, kva, yva = stack(VAL_ST, ystd)
    res = {}
    for name, use_own in (('base', False), ('hist', True)):
        rm = []
        for h in range(48):
            m = Ridge(alpha=1.0).fit(design(otr, ktr, h, use_own), ytr[:, h])
            rm.append(np.sqrt(np.mean((m.predict(design(ova, kva, h, use_own)) - yva[:, h]) ** 2)))
        res[name] = np.array(rm)
    unit = 'σ (per-station z-scored y)' if ystd else 'm/s (raw y — CURRENT pipeline)'
    g = res['base'].mean() - res['hist'].mean()
    print(f"\n=== target in {unit} ===")
    print(f"  horizon-mean RMSE  base={res['base'].mean():.4f}  hist={res['hist'].mean():.4f}")
    print(f"  absolute gain      {g:+.4f}")
    print(f"  RELATIVE gain      {100*g/res['base'].mean():+.2f} %")
    for lo, hi, lbl in ((1, 6, ' 1– 6h'), (7, 24, ' 7–24h'), (25, 48, '25–48h')):
        gg = (res['base'][lo-1:hi] - res['hist'][lo-1:hi]).mean()
        bb = res['base'][lo-1:hi].mean()
        print(f"    leads {lbl}: gain {gg:+.4f}  ({100*gg/bb:+.2f} %)")

# Is the station's absolute wind level recoverable from what the model actually sees?
print("\n=== per-station y statistics (the constants the model would need) ===")
for s in TRAIN_ST + VAL_ST:
    yy = data[s][2]
    tag = 'train' if s in TRAIN_ST else 'VAL  '
    print(f"  {tag} {s}: y mean={yy.mean():6.3f} m/s  std={yy.std():5.3f}")
