"""LaTeX table fragments for the iAIMS26 paper, generated from the result
CSVs so prose/table numbers can never drift from the data.

Writes booktabs fragments into iaims26/tables/:
  tab_headline.tex   T4 — synthetic study: all configurations vs local
  tab_real.tex       T5 — synthetic-to-real transfer (13 parks)
  tab_hparams.tex    T6 — zero-shot-tuned vs in-sample hyperparameters

Statistical convention (consistent across both tables): paired two-sided
Wilcoxon signed-rank on per-farm/-park RMSE differences vs the LOCAL
baseline, Holm-Bonferroni corrected within each table.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

OUT_DIR = common.IAIMS_DIR / 'tables'
PERIOD = 'full'


def holm(pvals):
    """Holm-Bonferroni: returns adjusted p-values in original order."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def fmt_p(p):
    if np.isnan(p):
        return '--'
    if p < 0.001:
        return '$<$0.001'
    return f'{p:.3f}'


def sig_mark(p):
    return '\\checkmark' if (not np.isnan(p)) and p < 0.05 else '--'


def vs_local_stats(rows):
    """rows: list of (label, series, local_series_aligned). Computes
    delta-median-RMSE, win-rate, Holm-adjusted p per row."""
    stats, pvals = [], []
    for label, s, loc in rows:
        common_idx = s.index.intersection(loc.index)
        d = s[common_idx] - loc[common_idx]
        try:
            _, p = wilcoxon(d)
        except ValueError:
            p = np.nan
        stats.append({'label': label, 'n': len(common_idx),
                      'd_med': float(np.median(d)),
                      'win': float((d < 0).mean()), 'p': p})
        pvals.append(p)
    valid = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    adj = holm([pvals[i] for i in valid]) if valid else []
    for j, i in enumerate(valid):
        stats[i]['p_holm'] = adj[j]
    for s in stats:
        s.setdefault('p_holm', np.nan)
    return stats


def _series(master, kind, budget=None, variant=None, metric='RMSE'):
    df = master[(master.model_kind == kind) & (master.period == PERIOD)]
    if budget is not None:
        df = df[df.budget == budget]
    if variant is not None:
        df = df[df.variant == variant]
    return df.groupby('station_id')[metric].mean()


def tab_headline(master):
    loc_rmse = _series(master, 'local')
    specs = [
        ('Zero-shot (physical statics)', ('zeroshot', None, None)),
        ('Zero-shot, no statics', ('zeroshot_nostatic', None, None)),
        ('Zero-shot, site-ID embedding', ('siteid_zeroshot', None, None)),
        ('Fine-tuned, 7 days', ('finetune', '7d', 'full')),
        ('Fine-tuned, full history', ('finetune', 'full', 'full')),
        ('In-sample (80 farms)', ('insample80', None, None)),
        ('Global reference (100 farms)', ('global100', None, None)),
    ]
    stat_rows = [(label, _series(master, *spec), loc_rmse)
                 for label, spec in specs]
    stats = vs_local_stats(stat_rows)

    # median AND mean: the mean is far more sensitive to the catastrophic
    # tail (single-farm failures), so reporting both exposes tail risk that
    # the median hides.
    lines = ['\\begin{tabular}{lcccccccc}', '\\toprule',
             '& \\multicolumn{2}{c}{$R^2$} & \\multicolumn{2}{c}{RMSE [\\%]} '
             '& & & & \\\\',
             '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
             'Configuration & med. & mean & med. & mean & $\\Delta$RMSE '
             '& Win [\\%] & $p_{\\mathrm{Holm}}$ & sig. \\\\',
             '\\midrule']
    r2_loc = _series(master, 'local', metric='R2')
    lines.append(f'Local (per-farm, ref.) & {r2_loc.median():.3f} & '
                 f'{r2_loc.mean():.3f} & {loc_rmse.median() * 100:.2f} & '
                 f'{loc_rmse.mean() * 100:.2f} & -- & -- & -- & -- \\\\')
    for (label, spec), st in zip(specs, stats):
        r2 = _series(master, *spec, metric='R2')
        rmse = _series(master, *spec)
        lines.append(f'{label} & {r2.median():.3f} & {r2.mean():.3f} & '
                     f'{rmse.median() * 100:.2f} & {rmse.mean() * 100:.2f} & '
                     f'{st["d_med"] * 100:+.2f} & {st["win"] * 100:.0f} & '
                     f'{fmt_p(st["p_holm"])} & {sig_mark(st["p_holm"])} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (OUT_DIR / 'tab_headline.tex').write_text('\n'.join(lines) + '\n')
    print('wrote tab_headline.tex')


def _park_grid(directory, prefix):
    files = [f for f in glob.glob(str(directory / f'{prefix}*.csv'))
             if 'smoke' not in f]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df.period == PERIOD]


def tab_real():
    twin = _park_grid(common.RESULTS_DIR / 'twin_transfer', 'twintr_')

    def sel(df, budget, variant):
        return df[(df.budget == budget) & (df.variant == variant)] \
            .set_index('park_id')

    # The generic-fleet pretraining is reported in the text only (it fails
    # outright); the table covers the twin protocol and its controls.
    # Gentle fine-tuning (0.1x lr) is the primary series — the two full-lr
    # rows are kept as the robustness contrast: better medians, far worse
    # means and worst cases.
    loc = sel(twin, 'local', 'local')
    specs = [
        ('Twin, zero-shot',
         twin[twin.budget == 'zeroshot'].set_index('park_id')),
        ('Twin + 7 days$^{\\ast}$', sel(twin, '7d', 'full')),
        ('Twin + 91 days$^{\\ast}$', sel(twin, '91d', 'full')),
        ('Twin + full history$^{\\ast}$', sel(twin, 'full', 'full')),
        ('Twin + 91 days$^{\\dagger}$', sel(twin, '91d', 'full_fastlr')),
        ('Twin + full history$^{\\dagger}$', sel(twin, 'full', 'full_fastlr')),
        ('From scratch, 7 days', sel(twin, '7d', 'scratch')),
        ('From scratch, full history', sel(twin, 'full', 'scratch')),
    ]
    stats = vs_local_stats([(label, df.RMSE, loc.RMSE) for label, df in specs])

    # No p-column: with n=13 the paired Wilcoxon is underpowered to the point
    # of being uninformative (no comparison survives Holm correction over the
    # eight rows). The table is descriptive; the inferential claims of this
    # paper rest on the n=100 synthetic study. Stated explicitly in the text.
    lines = ['\\begin{tabular}{lcccc}', '\\toprule',
             '& \\multicolumn{3}{c}{$R^2$} & \\\\',
             '\\cmidrule(lr){2-4}',
             'Configuration & med. & mean & worst & Win [\\%] \\\\',
             '\\midrule']
    lines.append(f'Local (per-park, ref.) & {loc.R2.median():.3f} & '
                 f'{loc.R2.mean():.3f} & {loc.R2.min():.2f} & -- \\\\')
    for (label, df), st in zip(specs, stats):
        lines.append(f'{label} & {df.R2.median():.3f} & {df.R2.mean():.3f} & '
                     f'{df.R2.min():.2f} & {st["win"] * 100:.0f} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (OUT_DIR / 'tab_real.tex').write_text('\n'.join(lines) + '\n')
    print('wrote tab_real.tex')


def tab_hparams():
    """Zero-shot HPO optimum only (paper decision, round 4: no contrast
    between in-sample and zero-shot hyperparameters)."""
    with open(common.IAIMS_DIR / 'configs' / 'hpo_best.json') as f:
        zs = json.load(f)['params']
    labels = [('Batch size', 'batch_size', '{:d}'),
              ('Learning rate', 'lr', 'sci'),
              ('Attention heads', 'n_heads', '{:d}'),
              ('Hidden dimension', 'hidden_dim', '{:d}'),
              ('Dropout', 'dropout', '{:.3f}'),
              ('LSTM layers', 'num_lstm_layers', '{:d}'),
              ('Static embedding dimension', 'static_embedding_dim', '{:d}'),
              ('Gradient clipping norm', 'clipnorm', '{:.2f}')]
    lines = ['\\begin{tabular}{lc}', '\\toprule',
             'Hyperparameter & Value \\\\',
             '\\midrule']
    for label, key, fmt in labels:
        if fmt == 'sci':
            mant, exp = f'{zs[key]:.2e}'.split('e')
            val = f'${mant}\\times10^{{{int(exp)}}}$'
        else:
            val = fmt.format(zs[key])
        lines.append(f'{label} & {val} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (OUT_DIR / 'tab_hparams.tex').write_text('\n'.join(lines) + '\n')
    print('wrote tab_hparams.tex')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(common.RESULTS_DIR / 'master_per_farm.csv',
                         dtype={'station_id': str})
    tab_headline(master)
    tab_real()
    tab_hparams()


if __name__ == '__main__':
    main()
