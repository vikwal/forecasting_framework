"""Summarize the GFS test (configs/gfs_test): ICON+ECMWF vs ICON+ECMWF+GFS per fleet, scope and seed.

Reads the newest results/*/cl_*gfs_*.pkl per (config, seed), prints mean metrics per arm and the paired
per-park difference (GFS arm minus base arm, averaged over seeds) for the 90-park runs.

    python configs/gfs_test/summarize.py
"""
import glob
import os
import pickle
import re

import pandas as pd

RES = 'results'   # train_cl.py writes to results/<basename of data.path>/, not eval.results_path
_NAME = re.compile(r'gfs_(wind|pv)_(one|all)_(icon_ecmwf(?:_gfs)?)_seed(\d+)_\d{8}_\d{6}\.pkl$')


def load() -> pd.DataFrame:
    """Per-park rows (fleet, scope, arm, seed, park, R2, RMSE, MAE, Skill), newest file per run."""
    newest = {}
    for f in sorted(glob.glob(os.path.join(RES, '*', 'cl_*gfs_*.pkl')), key=os.path.getmtime):
        m = _NAME.search(f)
        if m:
            newest[m.groups()] = f
    rows = []
    for (fleet, scope, arm, seed), f in newest.items():
        ev = pickle.load(open(f, 'rb'))['evaluation']
        ev = ev[ev['key'].notna()]
        for _, r in ev.iterrows():
            park = re.sub(r'^synth_|\.(csv|parquet)$', '', str(r['key']))
            rows.append({'fleet': fleet, 'scope': scope, 'arm': arm, 'seed': int(seed), 'park': park,
                         'R2': r['R^2'], 'RMSE': r['RMSE'], 'MAE': r['MAE'], 'Skill': r['Skill']})
    return pd.DataFrame(rows)


def main():
    df = load()
    if df.empty:
        print('no results yet')
        return
    pd.set_option('display.width', 200)
    print('Runs found (seeds per fleet/scope/arm):')
    print(df.groupby(['fleet', 'scope', 'arm'])['seed'].unique().to_string(), '\n')

    # mean over parks per seed, then mean/std over seeds
    per_seed = df.groupby(['fleet', 'scope', 'arm', 'seed'])[['R2', 'RMSE', 'MAE', 'Skill']].mean()
    agg = per_seed.groupby(['fleet', 'scope', 'arm']).agg(['mean', 'std']).round(4)
    print('Mean over parks; mean/std over seeds (RMSE/MAE in units of capacity):')
    print(agg.to_string(), '\n')

    # paired per-park difference, seeds averaged first
    park = df.groupby(['fleet', 'scope', 'arm', 'park'])[['R2', 'RMSE', 'MAE']].mean().unstack('arm')
    for (fleet, scope), g in park.groupby(level=[0, 1]):
        if ('RMSE', 'icon_ecmwf_gfs') not in g or g[('RMSE', 'icon_ecmwf_gfs')].isna().all():
            continue
        d = (g[('RMSE', 'icon_ecmwf_gfs')] - g[('RMSE', 'icon_ecmwf')]).dropna()
        rel = 100 * d / g[('RMSE', 'icon_ecmwf')].loc[d.index]
        print(f'{fleet}/{scope}: {len(d)} parks, RMSE change with GFS: mean {rel.mean():+.2f} %, '
              f'median {rel.median():+.2f} %, better in {(d < 0).sum()} of {len(d)} parks')


if __name__ == '__main__':
    main()
