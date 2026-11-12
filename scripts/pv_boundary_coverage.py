#!/usr/bin/env python3
"""Coverage instead of months (Viktor 2026-11-11): does the dependence boundary collapse onto one curve when the client's
own data are measured by how much of the evaluation period's weather they cover?

Per window a target day (the calendar day after the 09 UTC issue, steps 30..77) with forecast features (daily clear-sky
index sum(gti_nwp)/sum(ghi_clear) and its intraday std) -> regime (k-means, k = 3, fitted per dataset on all window days),
sun level (quartile of the daily clear-sky irradiation, i.e. solar geometry) and season (DJF/MAM/JJA/SON). Coverage of a client = mean over its parks of sum over sun-level x regime cells (and, as variant, season x regime) of the
evaluation share of the cell, counted where the own training data hold at least K_MIN days of that cell. Computed for the
real boundary (TFT and LR clients) and the synthetic maps (TFT seeds, LR draws); then the gain of federated over local is
related to coverage, months and own windows (rank correlation and isotonic R^2). No training.

    frcst/bin/python scripts/pv_boundary_coverage.py --output ~/Work/pv_sim2real/results/pv_boundary_2026-11-11
"""
import argparse
import itertools
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_conformal as base  # noqa: E402
from pv_robustness_infer import train_gti  # noqa: E402
from pv_quantile_coldstart import label_bounds  # noqa: E402
from pv_fl_story_real import FEB, SERIES_START  # noqa: E402
from pv_boundary_synthetic import SIZES, PLAN, STARTS  # noqa: E402
from pv_synthetic_scaling import PREPARED  # noqa: E402
from utils import preprocessing  # noqa: E402

H = 96
DAY = slice(30, 78)                                   # steps of the calendar day after the issue (00:00..23:30 UTC)
K_MIN = 3
FW = Path('results/parks_pv_v4_30min')
FLEET = Path('/mnt/lambda1/nvme1/synthetic/solar/parks_pv_v4_30min')
TWIN = Path('/mnt/lambda1/nvme1/synthetic/solar/trianel_twins_30min')
SEASON = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}


def day_features(issue, gti, clear_file):
    """issue: window start times (tz-naive UTC); gti: (windows, H). -> DataFrame date, k_day, k_var per window."""
    clear = pd.read_parquet(clear_file, columns=['ghi_clear_cams'])['ghi_clear_cams']
    clear.index = clear.index.tz_convert('UTC').tz_localize(None)
    vt = issue.values[:, None] + np.arange(H)[DAY] * np.timedelta64(30, 'm')
    gc = clear.reindex(pd.DatetimeIndex(vt.ravel())).values.reshape(vt.shape)
    g = np.asarray(gti, float)[:, DAY]
    m = gc > 50
    k = np.where(m, g / np.where(m, gc, 1), np.nan)
    out = pd.DataFrame({'date': pd.DatetimeIndex(vt[:, 0]).normalize(),
                        'k_day': np.nansum(np.where(m, g, 0), 1) / np.maximum(np.nansum(np.where(m, gc, 0), 1), 1),
                        'k_var': np.nanstd(k, 1), 'clear_day': np.nansum(np.where(np.isfinite(gc), gc, 0), 1)})
    out['season'] = out.date.dt.month.map(SEASON)
    return out


def fit_regimes(frames):
    F = pd.concat(frames, ignore_index=True).dropna(subset=['k_day', 'k_var'])
    X = F[['k_day', 'k_var']].values
    mu, sd = X.mean(0), X.std(0)
    km = KMeans(3, n_init=20, random_state=0).fit((X - mu) / sd)
    order = np.argsort(km.cluster_centers_[:, 0])
    remap = {int(c): i for i, c in enumerate(order)}
    edges = np.quantile(F.clear_day, [0.25, 0.5, 0.75])                  # solar-geometry level of the day (clear-sky energy)
    return lambda df: df.assign(regime=[remap[int(c)] for c in km.predict((df[['k_day', 'k_var']].fillna(0).values - mu) / sd)],
                                sun=np.digitize(df.clear_day, edges))


def coverage(own, ev, by=('sun', 'regime')):
    """Evaluation share of the cells (sun level x regime by default) that the own data hold with >= K_MIN days; and the
    calendar variant (season x regime)."""
    def cov(cols):
        w = ev.groupby(list(cols)).size() / len(ev)
        n = own.groupby(list(cols)).size()
        return float(sum(wi for cell, wi in w.items() if n.get(cell, 0) >= K_MIN))
    return cov(by), cov(('season', 'regime'))


def real():
    cfg = base.load_cfg(base.DATA_CFG)
    _, (sx, _) = base.finetune.load_init(base.Q_MODEL.format(seed=42))
    cfg['scaler_x'] = sx
    features = preprocessing.get_features(config=cfg)
    dfs = preprocessing.get_data(data_dir=cfg['data']['path'], config=cfg, freq=cfg['data']['freq'], features=features, files_key='files')
    data, bounds = {}, {}
    for key, frame in dfs.items():
        prep, _ = preprocessing.pipeline(data=frame, config=cfg, known_cols=features['known'], observed_cols=features['observed'],
                                         static_cols=features['static'], target_col=preprocessing.get_target_cols(cfg)[0])
        data[key], bounds[key] = prep, label_bounds(frame, prep['index_train'])
    keys = sorted(data)
    park = {k: k.replace('synth_', '').replace('_meter.csv', '') for k in keys}
    tr, ev = {}, {}
    for k in keys:
        f = TWIN / f'synth_{park[k]}.parquet'
        naive = lambda idx: pd.DatetimeIndex(idx).tz_convert('UTC').tz_localize(None)
        tr[k] = day_features(naive(data[k]['index_train']), train_gti(data[k], H), f)
        ev[k] = day_features(naive(data[k]['index_test']), data[k]['nwp_raw_test'], f)
    assign = fit_regimes(list(tr.values()) + list(ev.values()))
    tr = {k: assign(v) for k, v in tr.items()}
    ev = {k: assign(v) for k, v in ev.items()}
    rows = []
    for cl in [c for p in (1, 2, 3) for c in itertools.combinations(keys, p)]:
        for days, season in [(30, 'winter'), (30, 'summer'), (90, 'winter'), (90, 'summer'), (None, 'full')]:
            cov, sea = [], []
            for k in cl:
                if season == 'full':
                    keep = np.ones(len(tr[k]), bool)
                else:
                    s, e = (FEB - pd.Timedelta(days=days), FEB) if season == 'winter' else (SERIES_START, SERIES_START + pd.Timedelta(days=days))
                    lo, hi = pd.to_datetime(bounds[k][0], utc=True), pd.to_datetime(bounds[k][1], utc=True)
                    keep = np.asarray((lo >= s) & (hi < e))
                c, se = coverage(tr[k][keep], ev[k])
                cov.append(c); sea.append(se)
            rows.append(dict(client='+'.join(park[k] for k in cl), n_parks=len(cl), days=days or -1, season=season,
                             coverage=np.mean(cov), season_regime_coverage=np.mean(sea)))
    return pd.DataFrame(rows)


def synthetic():
    with PREPARED.open('rb') as f:
        bundle = pickle.load(f)
    pool = bundle['target']
    del bundle
    order = sorted(pool)
    tr, ev = {}, {}
    for k in order:
        it = pool[k]
        f = FLEET / f"{Path(k).name.replace('.csv', '')}.parquet"
        n = len(it['y_train'])
        issue = pd.DatetimeIndex([pd.Timestamp('2023-07-24 09:00') + pd.Timedelta(days=i) for i in range(n)])   # positions as days
        tr[k] = day_features(issue, train_gti(it, H), f)
        ev[k] = day_features(pd.DatetimeIndex(it['index_test']).tz_convert('UTC').tz_localize(None), it['nwp_raw_test'], f)
    assign = fit_regimes(list(tr.values()) + list(ev.values()))
    tr = {k: assign(v) for k, v in tr.items()}
    ev = {k: assign(v) for k, v in ev.items()}

    def cov_of(parks, months, start):
        cs, ss = [], []
        for k in parks:
            n = len(tr[k])
            lo = 0 if start == 'year' else STARTS[start]
            hi = n if start == 'year' else min(n, lo + int(round(30.4 * months)))
            c, s = coverage(tr[k].iloc[lo:hi], ev[k])
            cs.append(c); ss.append(s)
        return np.mean(cs), np.mean(ss)

    rows = []
    for label, seeds in (('tft', [42, 43, 44]), ('lr', [100 + d for d in range(12)])):
        for s in seeds:
            perm = [order[i] for i in np.random.default_rng(s).permutation(len(order))]
            pos = 20
            for P in SIZES:
                parks, pos = perm[pos:pos + P], pos + P
                for months, start in PLAN:
                    c, se = cov_of(parks, months, start)
                    rows.append(dict(model=label, seed=s, n_parks=P, months=months, start=start, coverage=c, season_regime_coverage=se))
    return pd.DataFrame(rows)


def explain(df, x_cols, label):
    out = []
    for x in x_cols:
        d = df.dropna(subset=[x, 'gain'])
        rho = spearmanr(d[x], d.gain).correlation
        iso = IsotonicRegression(increasing='auto').fit(d[x], d.gain)
        r2 = 1 - ((d.gain - iso.predict(d[x])) ** 2).sum() / ((d.gain - d.gain.mean()) ** 2).sum()
        pred = np.empty(len(d))                                           # 5-fold cross-validated isotonic fit (fair across
        folds = np.random.default_rng(0).permutation(len(d)) % 5          # predictors with few or many distinct values)
        for f in range(5):
            m = folds == f
            fit = IsotonicRegression(increasing='auto', out_of_bounds='clip').fit(d[x].values[~m], d.gain.values[~m])
            pred[m] = fit.predict(d[x].values[m])
        r2cv = 1 - ((d.gain.values - pred) ** 2).sum() / ((d.gain - d.gain.mean()) ** 2).sum()
        out.append(dict(case=label, predictor=x, spearman=rho, isotonic_r2=r2, isotonic_r2_cv=r2cv, n=len(d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rc, sc = real(), synthetic()
    rc.to_csv(a.output / 'coverage_real_clients_2026-11-11.csv', index=False)
    sc.to_csv(a.output / 'coverage_synthetic_clients_2026-11-11.csv', index=False)
    res = []
    # real TFT (sim prior)
    t = pd.concat([pd.read_csv(FW / f'boundary_real_2026-11-11_v1/seed{s}/metrics.csv') for s in (42, 43, 44)])
    g = t.groupby(['client', 'days', 'season', 'arm']).pinball_day.mean().unstack()
    g = (100 * (g.federated / g.local - 1)).rename('gain').reset_index().merge(rc, on=['client', 'days', 'season'])
    g['log_days'] = np.log(g.days.replace(-1, 190))
    res += explain(g, ['coverage', 'season_regime_coverage', 'log_days'], 'real_tft_prior')
    # real LR
    l = pd.read_csv(FW / 'boundary_lr_2026-11-11_v1/real_boundary_lr.csv')
    for pr in ('scratch', 'prior'):
        x = l[l.arm.str.endswith(pr)].groupby(['client', 'days', 'season', 'arm']).pinball_day.mean().unstack()
        x = (100 * (x[f'federated_{pr}'] / x[f'local_{pr}'] - 1)).rename('gain').reset_index().merge(rc, on=['client', 'days', 'season'])
        x['log_days'] = np.log(x.days.replace(-1, 190))
        res += explain(x, ['coverage', 'season_regime_coverage', 'log_days'], f'real_lr_{pr}')
    # synthetic TFT and LR
    st = pd.concat([pd.read_csv(FW / f'boundary_synthetic_2026-11-11_v1/seed{s}/metrics.csv') for s in (42, 43, 44)])
    sl = pd.read_csv(FW / 'boundary_lr_2026-11-11_v1/synthetic_boundary_lr.csv')
    sl['seed'] = 100 + sl.draw
    for label, d, key in (('tft', st, ['seed', 'n_parks', 'months', 'start']), ('lr', sl, ['seed', 'n_parks', 'months', 'start'])):
        for pr in ('scratch', 'prior'):
            x = d[d.arm.str.endswith(pr)].groupby(key + ['arm']).pinball_day.mean().unstack()
            x = (100 * (x[f'federated_{pr}'] / x[f'local_{pr}'] - 1)).rename('gain').reset_index()
            x = x.merge(sc[sc.model == label], on=key)
            x['log_months'] = np.log(x.months)
            x['log_parks'] = np.log(x.n_parks)
            res += explain(x, ['coverage', 'season_regime_coverage', 'log_months', 'log_parks'], f'synthetic_{label}_{pr}')
            x.to_csv(a.output / f'coverage_gain_synthetic_{label}_{pr}_2026-11-11.csv', index=False)
    r = pd.DataFrame(res)
    r.to_csv(a.output / 'coverage_explains_2026-11-11.csv', index=False)
    print(r.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
