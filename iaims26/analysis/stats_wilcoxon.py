"""Paired two-sided Wilcoxon signed-rank tests over the 100 farms.

Mirrors the IGEC analysis (power_evaluation/synth_power_evaluation.ipynb):
paired per-farm RMSE differences, two-sided test, Holm correction within each
comparison family. Reads results/master_per_farm.csv (run aggregate first).

Output: results/wilcoxon.csv (one row per comparison x period) — LaTeX-ready.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

MASTER = common.RESULTS_DIR / 'master_per_farm.csv'
OUT = common.RESULTS_DIR / 'wilcoxon.csv'
PERIODS = ['full', 'P1', 'P2']


def slice_model(master, spec):
    kind, budget, variant = spec
    df = master[master.model_kind == kind]
    if budget is not None:
        df = df[df.budget == budget]
    if variant is not None:
        df = df[df.variant == variant]
    return df.groupby('station_id').mean(numeric_only=True)


def compare(master, name_a, spec_a, name_b, spec_b, period, family):
    """Paired comparison A vs B on RMSE (negative delta = A better)."""
    a = slice_model(master[master.period == period], spec_a)
    b = slice_model(master[master.period == period], spec_b)
    stations = a.index.intersection(b.index)
    if len(stations) < 10:
        return None
    d_rmse = a.loc[stations, 'RMSE'] - b.loc[stations, 'RMSE']
    d_r2 = a.loc[stations, 'R2'] - b.loc[stations, 'R2']
    try:
        stat, p = wilcoxon(d_rmse, alternative='two-sided')
    except ValueError:
        return None
    return {
        'family': family, 'period': period,
        'A': name_a, 'B': name_b, 'n': len(stations),
        'median_dRMSE': float(np.median(d_rmse)),
        'median_dR2': float(np.median(d_r2)),
        'win_rate_A': float((d_rmse < 0).mean()),
        'W': float(stat), 'p': float(p),
    }


def holm(rows):
    """Holm-Bonferroni within each (family, period) group."""
    df = pd.DataFrame(rows)
    df['p_holm'] = np.nan
    for _, idx in df.groupby(['family', 'period']).groups.items():
        sub = df.loc[idx].sort_values('p')
        m = len(sub)
        adj, running_max = [], 0.0
        for rank, p in enumerate(sub['p']):
            val = min(1.0, (m - rank) * p)
            running_max = max(running_max, val)
            adj.append(running_max)
        df.loc[sub.index, 'p_holm'] = adj
    df['significant'] = df['p_holm'] < 0.05
    return df


def main():
    master = pd.read_csv(MASTER, dtype={'station_id': str})
    rows = []

    zeroshot = ('zeroshot', None, None)
    local = ('local', None, None)
    g100 = ('global100', None, None)

    for period in PERIODS:
        # family: zero-shot positioning
        for name_b, spec_b in [('local', local), ('global100', g100),
                               ('insample80', ('insample80', None, None))]:
            r = compare(master, 'zeroshot', zeroshot, name_b, spec_b, period, 'zeroshot')
            if r:
                rows.append(r)
        # family: site-ID control
        for name_a, spec_a, name_b, spec_b in [
                ('siteid_global', ('siteid_global', None, None), 'global100', g100),
                ('siteid_zeroshot', ('siteid_zeroshot', None, None), 'zeroshot', zeroshot)]:
            r = compare(master, name_a, spec_a, name_b, spec_b, period, 'siteid')
            if r:
                rows.append(r)
        # family: no-static ablation (do the physical statics carry the
        # zero-shot generalization?)
        for name_a, spec_a, name_b, spec_b in [
                ('zeroshot_nostatic', ('zeroshot_nostatic', None, None),
                 'zeroshot', zeroshot),
                ('zeroshot_nostatic', ('zeroshot_nostatic', None, None),
                 'local', local),
                ('insample80_nostatic', ('insample80_nostatic', None, None),
                 'insample80', ('insample80', None, None))]:
            r = compare(master, name_a, spec_a, name_b, spec_b, period, 'nostatic')
            if r:
                rows.append(r)
        # family: fine-tuning vs local and vs zeroshot
        ft = master[master.model_kind == 'finetune']
        for budget in sorted(ft.budget.unique()):
            for variant in sorted(ft.variant.unique()):
                spec = ('finetune', budget, variant)
                name = f'ft_{budget}_{variant}'
                for name_b, spec_b, family in [('local', local, 'ft_vs_local'),
                                               ('zeroshot', zeroshot, 'ft_vs_zeroshot')]:
                    r = compare(master, name, spec, name_b, spec_b, period, family)
                    if r:
                        rows.append(r)

    df = holm(rows)
    df.sort_values(['family', 'period', 'p_holm'], inplace=True)
    df.to_csv(OUT, index=False)
    print(f'wrote {OUT}: {len(df)} comparisons')
    print(df[df.period == 'full'][['family', 'A', 'B', 'n', 'median_dRMSE',
                                   'win_rate_A', 'p', 'p_holm', 'significant']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
