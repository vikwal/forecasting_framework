#!/usr/bin/env python3
"""Evaluation of the CV pilot of scripts/pv_hpo_local.py (2026-11-13).

Truth per (park, configuration) is the pseudo-final model F (20 months -> Apr--Jul 2025). For each selection protocol
(A: one 8-month validation block, B: mean of the two expanding folds) and each park:
  regret     F of the configuration the protocol selects minus the best F of all configurations
  regret_top3 mean F of the three selected configurations minus the best F
  spearman   rank correlation of the protocol score with F over the configurations
Also: cnn against tcn, and one configuration per park against one shared configuration for all parks.

    frcst/bin/python scripts/pv_hpo_local_pilot_report.py results/parks_pv_v4_30min/hpo_local_cnnrnn_2026-11-13_v1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROTOCOLS = ('A', 'B')


def table(rows):
    """One row per (park, config): A, B (mean of the folds), F and the metadata."""
    rows = rows.copy()
    rows['variant'] = rows['variant'].str.replace(r'^B\d$', 'B', regex=True)
    wide = rows.pivot_table(index=['park', 'group', 'config', 'conv_type'], columns='variant', values='score',
                            aggfunc='mean').reset_index()
    nb = rows[rows.variant == 'B'].groupby(['park', 'config']).size()
    a_test = rows[rows.variant == 'A'].set_index(['park', 'config'])['a_test']
    wide = wide.join(nb.rename('n_b'), on=['park', 'config']).join(a_test, on=['park', 'config'])
    return wide


def selection(wide, configs=None):
    """Regret, top-3 regret and Spearman per park and protocol (optionally restricted to a configuration subset)."""
    out = []
    for park, g in wide.groupby('park'):
        if configs is not None:
            g = g[g.config.isin(configs)]
        g = g.dropna(subset=['F', *PROTOCOLS])
        if len(g) < 3:
            continue
        best = g.F.min()
        for p in PROTOCOLS:
            s = g.sort_values(p)
            out.append(dict(park=park, protocol=p, regret=s.F.iloc[0] - best,
                            regret_top3=s.F.iloc[:3].mean() - best, rel_regret=(s.F.iloc[0] - best) / best,
                            spearman=spearmanr(g[p], g.F).statistic, chosen=int(s.config.iloc[0])))
    return pd.DataFrame(out)


def shared_vs_per_park(wide, protocol):
    """F of the per-park choice against the configuration with the best mean protocol score over all parks."""
    shared = wide.groupby('config')[protocol].mean().idxmin()
    per = wide.loc[wide.groupby('park')[protocol].idxmin()].set_index('park').F
    sh = wide[wide.config == shared].set_index('park').F
    return pd.DataFrame(dict(per_park=per, shared=sh)), int(shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output')
    a = ap.parse_args()
    out = Path(a.output) / 'pilot'
    rows = pd.concat([pd.read_csv(f) for f in sorted((out / 'rows').glob('*.csv'))], ignore_index=True)
    hp = pd.DataFrame(json.loads((out / 'configs.json').read_text())).rename(columns={'id': 'config'})
    wide = table(rows).merge(hp.drop(columns='conv_type'), on='config')
    complete = wide.dropna(subset=['A', 'B', 'F'])
    complete = complete[complete.n_b == 2]
    print(f'{rows.park.nunique()} parks, {rows.config.nunique()} configurations, {len(rows)} trainings, '
          f'{len(complete)} of {len(wide)} (park, config) pairs complete')
    wide.to_csv(out / 'pilot_wide.csv', index=False)

    sel = selection(complete)
    sel.to_csv(out / 'pilot_selection.csv', index=False)
    print('\nSelection quality per protocol (mean over parks; F level', round(complete.F.min(), 5), '..):')
    print(sel.groupby('protocol')[['regret', 'regret_top3', 'rel_regret', 'spearman']].agg(['mean', 'median'])
          .round(5).to_string())
    r = sel.pivot(index='park', columns='protocol', values='regret')
    print(f"B regret < A in {(r.B < r.A).sum()} of {len(r)} parks, equal in {(r.B == r.A).sum()}")

    print('\nArchitecture (F, pseudo test):')
    for conv, g in complete.groupby('conv_type'):
        best = g.groupby('park').F.min()
        print(f'  {conv}: oracle best per park {best.mean():.5f}, median over configs {g.F.median():.5f}, '
              f'share diverged (F > 2 x park best) {(g.F > 2 * g.park.map(complete.groupby("park").F.min())).mean():.2f}')
    for conv in ('cnn', 'tcn'):
        ids = complete[complete.conv_type == conv].config.unique()
        s = selection(complete, ids).groupby('protocol')[['regret', 'spearman']].mean()
        print(f'  selection within {conv}:', s.round(5).to_dict('index'))
    for p in PROTOCOLS:
        picks = complete.loc[complete.groupby('park')[p].idxmin()]
        print(f'  protocol {p} picks conv_type: {picks.conv_type.value_counts().to_dict()}, '
              f'bidirectional: {picks.bidirectional.value_counts().to_dict()}')

    print('\nOne configuration per park vs one shared configuration (F):')
    for p in PROTOCOLS:
        cmp, shared = shared_vs_per_park(complete, p)
        print(f'  {p}: per park {cmp.per_park.mean():.5f}, shared (config {shared}) {cmp.shared.mean():.5f}, '
              f'per park better in {(cmp.per_park < cmp.shared).sum()} of {len(cmp)} parks')

    print('\nTraining data, same configuration (A model 12 months vs F 20 months, pseudo test):')
    print(f'  median a_test / F = {(complete.a_test / complete.F).median():.3f}')

    lc = sorted((Path(a.output) / 'losscmp' / 'rows').glob('*.csv'))
    if lc:
        loss_comparison(pd.concat([pd.read_csv(f) for f in lc], ignore_index=True), Path(a.output) / 'losscmp')


def loss_comparison(rows, out):
    """Pinball median against an MSE point model (same recipe, seeds 42..44), daytime cells of the pseudo test."""
    m = rows.groupby(['park', 'loss'])[['rmse', 'mae', 'bias', 'r2']].mean().unstack('loss')
    rel = {k: (m[k]['pinball'] / m[k]['mse'] - 1) * 100 for k in ('rmse', 'mae')}
    m.to_csv(out / 'losscmp_by_park.csv')
    print(f"\nLoss comparison ({m.shape[0]} parks, mean over seeds; pinball median vs MSE point model):")
    for k in ('rmse', 'mae', 'r2', 'bias'):
        print(f"  {k}: pinball {m[k]['pinball'].mean():.5f}  mse {m[k]['mse'].mean():.5f}")
    for k, r in rel.items():
        print(f"  {k} of the median relative to MSE: mean {r.mean():+.2f} %, median {r.median():+.2f} %, "
              f"median worse in {(r > 0).sum()} of {len(r)} parks")


if __name__ == '__main__':
    main()
