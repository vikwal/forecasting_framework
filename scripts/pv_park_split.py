#!/usr/bin/env python3
"""Park split for the shared local CNNRNN HPO (Viktor 2026-11-13): test parks and three HPO folds.

Only parks with a full J1 year (prepared by pv_hpo_local.py prepare). Features: standardised latitude, longitude,
altitude, capacity-weighted tilt, azimuth (sin, cos) and DC/AC ratio, all static (no statistic of the series, which
would include J3).
  1. Test: KMeans with n_test clusters, one random park per cluster (representative test set; Kennard-Stone would
     draw the extreme parks first and make the test an extrapolation).
  2. Folds: KMeans with 50 clusters on the remaining pool; the parks of each cluster are spread over the three folds
     (fewest-members-first, random ties), as in the graph-paper split, so neighbours in feature space land in
     different folds.
  Of 500 random draws (clusters fixed per test set) the most balanced split is kept.

    frcst/bin/python scripts/pv_park_split.py --output results/parks_pv_v4_30min/hpo_local_cnnrnn_2026-11-13_v1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

FEATURES = ['latitude', 'longitude', 'altitude', 'tilt', 'az_sin', 'az_cos', 'dc_ac_ratio']


def park_features(data_dir, ids):
    parks = pd.read_csv(Path(data_dir) / 'parks.csv')
    sites = pd.read_csv(Path(data_dir) / 'sites.csv')
    f = parks.merge(sites[['site_id', 'altitude', 'tilt_capw_true', 'azimuth_mean_true']], on='site_id', how='left')
    f = f.set_index('park_id').loc[ids]
    az = np.deg2rad(f['azimuth_mean_true'])
    out = pd.DataFrame(dict(latitude=f['latitude'], longitude=f['longitude'], altitude=f['altitude'],
                            tilt=f['tilt_capw_true'], az_sin=np.sin(az), az_cos=np.cos(az),
                            dc_ac_ratio=f['dc_ac_ratio']), index=f.index)
    assert not out.isna().any().any()
    return out


def standardise(f):
    return ((f - f.mean()) / f.std()).to_numpy()


def split(feats, n_test=60, n_clusters=50, n_folds=3, seed=2026, draws=500):
    """Clusters are fixed (one KMeans each); of `draws` random assignments the one with the smallest largest
    set-mean deviation (in sd, over test and folds) is kept, as the graph-paper split balances its folds."""
    ids = np.asarray(feats.index)
    lab_t = KMeans(n_test, n_init=20, random_state=seed).fit_predict(standardise(feats))
    best = None
    for d in range(draws):
        rng = np.random.default_rng([seed, d])
        test = sorted(str(rng.choice(ids[lab_t == c])) for c in range(n_test))
        pool = feats.drop(index=test)
        lab = KMeans(n_clusters, n_init=5, random_state=seed).fit_predict(standardise(pool))
        fold, size = {}, np.zeros(n_folds, int)
        for c in rng.permutation(n_clusters):
            for p in rng.permutation(np.asarray(pool.index)[lab == c]):
                k = int(rng.choice(np.flatnonzero(size == size.min())))
                fold[str(p)], size[k] = k + 1, size[k] + 1
        sets = {'test': test, **{f'fold{k}': [p for p, v in fold.items() if v == k] for k in range(1, n_folds + 1)}}
        worst = max(max(b.values()) for b in balance(feats, sets).values())
        if best is None or worst < best[0]:
            best = (worst, test, fold, d)
    return best[1], best[2], best[3]


def balance(feats, sets):
    """Largest deviation of a set mean from the overall mean, in standard deviations, per feature."""
    mu, sd = feats.mean(), feats.std()
    return {name: ((feats.loc[ids].mean() - mu).abs() / sd).round(3).to_dict() for name, ids in sets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, help='pv_hpo_local.py output directory (meta.json)')
    ap.add_argument('--n-test', type=int, default=60)
    ap.add_argument('--clusters', type=int, default=50)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.output)
    path = out / 'split.json'
    if path.exists():
        raise SystemExit(f'{path} exists; versions are never overwritten')
    meta = json.loads((out / 'meta.json').read_text())
    full = max(v['n1'] for v in meta['parks'].values())
    ids = sorted(p for p, v in meta['parks'].items() if v['n1'] == full)
    short = sorted(p for p, v in meta['parks'].items() if v['n1'] < full)
    import pickle
    with open(out / 'meta.pkl', 'rb') as f:
        data_dir = pickle.load(f)['cfg']['data']['path']
    feats = park_features(data_dir, ids)
    test, fold, draw = split(feats, a.n_test, a.clusters, seed=a.seed)
    sets = {'test': test, **{f'fold{k}': sorted(p for p, v in fold.items() if v == k) for k in (1, 2, 3)}}
    bal = balance(feats, sets)
    res = dict(test=test, folds={k: sets[f'fold{k}'] for k in (1, 2, 3)}, excluded_short_j1=short,
               features=FEATURES, n_test_clusters=a.n_test, n_fold_clusters=a.clusters, seed=a.seed, draw=draw,
               balance_sd=bal, groups={p: meta['parks'][p]['group'] for p in ids})
    path.write_text(json.dumps(res, indent=1))
    print(f'{len(ids)} parks with full J1 ({len(short)} excluded): test {len(test)}, folds',
          [len(sets[f"fold{k}"]) for k in (1, 2, 3)])
    for name, b in bal.items():
        g = pd.Series({p: meta['parks'][p]['group'] for p in sets[name]}).value_counts().sort_index().to_dict()
        print(f'  {name}: max deviation {max(b.values()):.3f} sd ({max(b, key=b.get)}), groups {g}')


if __name__ == '__main__':
    main()
