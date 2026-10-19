"""Paper figures for "Forecasting Wind Farms Without History" (iAIMS26).

Style replicates the published-paper conventions (power_evaluation
notebooks): font.size 12 base, FONT=14 labels/titles, no top/right spines,
PDF+PNG at dpi=600 with bbox_inches="tight".

Reads results/master_per_farm.csv, results/wilcoxon.csv and the
per-experiment CSVs; writes into results/figures/.

  F1 fig1_zeroshot.pdf    zero-shot vs local scatter (single column)
  F2 fig2_budget.pdf      few-shot budget curves + parameter inset (full width)
  F3 fig3_real.pdf        synthetic-to-real transfer, 2 panels (full width)
  F4 fig4_mechanism.pdf   statics / no-statics / site-ID mechanism bars
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

# ---- style (identical conventions to the published notebooks) -------------
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})
FONT = 14
DPI = 600

# IAIMS_FIG_STYLE=journal: figures at the Elsevier double-column widths
# (column 238.25 pt = 3.30 in, text 494.5 pt = 6.84 in) into figures_journal/.
JOURNAL = os.environ.get('IAIMS_FIG_STYLE') == 'journal'
W1, W2 = (3.30, 6.84) if JOURNAL else (5.5, 11.0)
if JOURNAL:
    FONT = 11
    plt.rcParams.update({"font.size": 9.5})

C_GLOBAL = "#2166ac"      # global / pretrained / twin
C_NOSTATIC = "#74add1"    # light blue secondary
C_LOCAL = "#2ca02c"       # local baselines
C_BASE1 = "#9467bd"       # tertiary line
C_BASE2 = "#7f7f7f"       # gray
C_RED = "#d62728"         # scratch control
GREEN, RED = "#2ca02c", "#d62728"   # win/loss coloring (as in the companion)

MASTER = common.RESULTS_DIR / 'master_per_farm.csv'
FIG_DIR = common.RESULTS_DIR / ('figures_journal' if JOURNAL else 'figures')
PERIOD = 'full'
BUDGETS = ['zeroshot', '7d', '30d', '91d', 'full']
BUDGET_LABELS = ['zero-shot', '1 week', '1 month', '3 months', 'full']


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'{name}.{ext}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {FIG_DIR / name}.pdf/.png')


def _series(master, kind, budget=None, variant=None, metric='R2', period=PERIOD):
    df = master[(master.model_kind == kind) & (master.period == period)]
    if budget is not None:
        df = df[df.budget == budget]
    if variant is not None:
        df = df[df.variant == variant]
    return df.groupby('station_id')[metric].mean()


# ---------------------------------------------------------------------------
def fig1_zeroshot(master):
    zs = _series(master, 'zeroshot')
    loc = _series(master, 'local')
    stations = zs.index.intersection(loc.index)

    fig, ax = plt.subplots(figsize=(W1, W1 * (0.80 if JOURNAL else 0.65)))
    lo = min(loc[stations].min(), zs[stations].min()) - 0.03
    if JOURNAL:
        lo = 0.25
    ax.plot([lo, 1], [lo, 1], ls='--', lw=1.0, color='gray', zorder=1)
    d = zs[stations] - loc[stations]
    off = zs[stations][zs[stations] < lo] if JOURNAL else zs[stations].iloc[:0]
    # color by which model wins on the farm — the companion papers' convention
    for mask, color, label in ((d > 0, GREEN, 'zero-shot better'),
                               (d <= 0, RED, 'local better')):
        idx = d[mask].index.difference(off.index)
        ax.scatter(loc[idx], zs[idx], s=26 if not JOURNAL else 14, color=color,
                   edgecolors='#555555', linewidths=0.4, zorder=2,
                   label=f'{label} ({mask.mean():.0%})')
    # annotate the single catastrophic zero-shot failure so it cannot be
    # mistaken for a plotting artifact (discussed in the results text)
    if JOURNAL and len(off):
        ax.scatter(loc[off.index], np.full(len(off), lo + 0.012), marker='v',
                   s=28, color=RED, edgecolors='#555555', linewidths=0.4,
                   zorder=3, clip_on=False)
        vals = ', '.join(f'{v:.1f}' for v in sorted(off.values))
        ax.annotate(f'{len(off)} farms off scale\n(zero-shot $R^2$ {vals})',
                    xy=(loc[off.index].mean(), lo + 0.012), xytext=(18, 4),
                    textcoords='offset points', fontsize=8.5, color='#555555',
                    ha='left', va='bottom', arrowprops=dict(arrowstyle='-', color='#999999', lw=0.7))
    elif not JOURNAL:
        worst = zs[stations].idxmin()
        ax.annotate(f'farm {worst}', xy=(loc[worst], zs[worst]),
                    xytext=(-8, 10), textcoords='offset points',
                    fontsize=FONT - 4, color='#555555', ha='right',
                    arrowprops=dict(arrowstyle='-', color='#999999', lw=0.7))
    box = (f'median $R^2$: {zs[stations].median():.3f} '
           f'vs {loc[stations].median():.3f}')
    ax.text(0.03, 0.97, box, transform=ax.transAxes, fontsize=FONT - 3,
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor='#cccccc'))
    ax.set_xlabel('Local model $R^2$ (two years of data)', fontsize=FONT)
    ax.set_ylabel('Zero-shot $R^2$ (unseen farm)', fontsize=FONT)
    ax.tick_params(labelsize=FONT - 1)
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.0)
    ax.legend(fontsize=FONT - 3 if not JOURNAL else 8.5, framealpha=0.95, frameon=not JOURNAL, handletextpad=0.2 if JOURNAL else 0.8, borderaxespad=0.3 if JOURNAL else 0.5,
              loc='upper left' if JOURNAL else 'lower left', bbox_to_anchor=(0.0, 0.89) if JOURNAL else None)
    _save(fig, 'fig1_zeroshot')


# ---------------------------------------------------------------------------
def fig2_budget(master):
    """Few-shot budget curves (single column). The per-variant trainable-
    parameter counts are reported in the text, not as a bar inset (four
    numbers on a log axis were unreadable)."""
    variants = {
        'full':       dict(color=C_GLOBAL, label='full fine-tune'),
        'static_vsn': dict(color=C_BASE2, label='static path + VSNs'),
        'static':     dict(color=C_BASE1, label='static path only'),
        'head':       dict(color=C_NOSTATIC, label='output head only'),
        'scratch':    dict(color=C_RED, label='from scratch (no pretrain)'),
    }
    loc_med = _series(master, 'local').median()
    g100_med = _series(master, 'global100').median()

    fig, ax = plt.subplots(figsize=(W1, W1 * (0.85 if JOURNAL else 0.76)))
    xs = np.arange(len(BUDGETS))
    for variant, style in variants.items():
        med, q1, q3 = [], [], []
        for budget in BUDGETS:
            if budget == 'zeroshot':
                vals = (_series(master, 'zeroshot') if variant != 'scratch'
                        else pd.Series(dtype=float))
            else:
                vals = _series(master, 'finetune', budget=budget, variant=variant)
            med.append(vals.median() if len(vals) else np.nan)
            q1.append(vals.quantile(0.25) if len(vals) else np.nan)
            q3.append(vals.quantile(0.75) if len(vals) else np.nan)
        ax.plot(xs, med, marker='o', ms=5, lw=1.8, **style)
        ax.fill_between(xs, q1, q3, color=style['color'], alpha=0.10, lw=0)
    ax.axhline(loc_med, color=C_LOCAL, ls='--', lw=1.4,
               label='local baseline (median)')
    ax.axhline(g100_med, color=C_BASE2, ls=':', lw=1.4,
               label='global-100 model (in-sample)')
    scr7 = _series(master, 'finetune', budget='7d', variant='scratch').median()
    ax.annotate(f'{scr7:.2f}', xy=(1, scr7), xytext=(8, 0),
                textcoords='offset points', fontsize=FONT - 3, color=C_RED,
                va='center')
    ax.set_xticks(xs)
    ax.set_xticklabels(BUDGET_LABELS, fontsize=FONT - 2, rotation=20)
    ax.set_xlabel('Target-farm data budget', fontsize=FONT)
    ax.set_ylabel('Median $R^2$ (100 unseen farms)', fontsize=FONT)
    ax.tick_params(labelsize=FONT - 1)
    if JOURNAL:
        ax.legend(fontsize=8.5, frameon=False, loc='upper center',
                  bbox_to_anchor=(0.5, -0.32), ncol=2, handlelength=1.6,
                  columnspacing=0.8)
    else:
        ax.legend(fontsize=FONT - 4, framealpha=0.95, loc='lower right')
    _save(fig, 'fig2_budget')


# ---------------------------------------------------------------------------
def _load_park_grid(directory, prefix):
    import glob
    files = glob.glob(str(directory / f'{prefix}*.csv'))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df.period == 'full']


def fig3_real():
    """Real-park transfer: how much on-site data is needed before the
    twin-pretrained model beats the local baselines? Left: accuracy vs
    budget; right: the paired win rate, whose 50 % crossing is the answer.
    The generic-fleet pretraining is text-only (it fails outright)."""
    twin = _load_park_grid(common.RESULTS_DIR / 'twin_transfer', 'twintr_')
    fid = pd.read_csv(common.RESULTS_DIR / 'twin' / 'twin_eval.csv')
    fid_med = fid[(fid.source == 'twin') & (fid.period == 'full')].R2.median()
    loc = twin[twin.variant == 'local'].set_index('park_id').R2

    series = [('synthetic-twin pretrain + fine-tune', 'full', C_GLOBAL),
              ('from scratch (no pretrain)', 'scratch', C_RED)]

    def budget_series(variant):
        """Per-budget R2 series aligned on park_id (zero-shot only exists
        for the pretrained model)."""
        out = []
        for budget in BUDGETS:
            if budget == 'zeroshot':
                s = (twin[twin.budget == 'zeroshot'].set_index('park_id').R2
                     if variant != 'scratch' else pd.Series(dtype=float))
            else:
                s = twin[(twin.budget == budget)
                         & (twin.variant == variant)].set_index('park_id').R2
            out.append(s)
        return out

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(W2, W2 * (0.36 if JOURNAL else 0.32)))
    xs = np.arange(len(BUDGETS))

    # (a) accuracy vs budget
    for label, variant, color in series:
        vals = budget_series(variant)
        med = [s.median() if len(s) else np.nan for s in vals]
        q1 = [s.quantile(0.25) if len(s) else np.nan for s in vals]
        q3 = [s.quantile(0.75) if len(s) else np.nan for s in vals]
        axa.plot(xs, med, marker='o', ms=5, lw=1.8, color=color, label=label)
        axa.fill_between(xs, q1, q3, color=color, alpha=0.10, lw=0)
    axa.axhline(loc.median(), color=C_LOCAL, ls='--', lw=1.4,
                label='local baseline (median)')
    axa.axhline(fid_med, color=C_BASE2, ls=':', lw=1.4,
                label=f'twin fidelity bound ({fid_med:.2f})')
    axa.set_xticks(xs)
    axa.set_xticklabels(BUDGET_LABELS, fontsize=FONT - 2, rotation=20)
    axa.set_xlabel('Real data budget', fontsize=FONT)
    axa.set_ylabel('Median $R^2$ (13 parks)', fontsize=FONT)
    axa.tick_params(labelsize=FONT - 1)
    if not JOURNAL:
        axa.legend(fontsize=FONT - 4, framealpha=0.95, loc='lower right')

    # (b) paired win rate against each park's own local model
    for label, variant, color in series:
        wins = []
        for s in budget_series(variant):
            if not len(s):
                wins.append(np.nan)
                continue
            idx = s.index.intersection(loc.index)
            wins.append((s[idx] > loc[idx]).mean() * 100)
        axb.plot(xs, wins, marker='o', ms=5, lw=1.8, color=color, label=label)
    axb.axhline(50, color=C_LOCAL, ls='--', lw=1.4,
                label='parity with local')
    axb.set_xticks(xs)
    axb.set_xticklabels(BUDGET_LABELS, fontsize=FONT - 2, rotation=20)
    axb.set_xlabel('Real data budget', fontsize=FONT)
    axb.set_ylabel('Parks beating their local model [%]', fontsize=FONT)
    axb.set_ylim(0, 100)
    axb.tick_params(labelsize=FONT - 1)
    if JOURNAL:
        h, l = axa.get_legend_handles_labels()
        hb, lb = axb.get_legend_handles_labels()
        for hh, ll in zip(hb, lb):
            if ll not in l:
                h.append(hh); l.append(ll)
        fig.legend(h, l, fontsize=8.5, frameon=False, loc='upper center',
                   bbox_to_anchor=(0.5, 0.02), ncol=3)
    else:
        axb.legend(fontsize=FONT - 4, framealpha=0.95, loc='upper left')
    fig.tight_layout()
    _save(fig, 'fig3_real')


# ---------------------------------------------------------------------------
def fig4_mechanism(master):
    groups = {
        'In-sample': [('insample80', C_GLOBAL, 'physical statics'),
                      ('insample80_nostatic', C_NOSTATIC, 'no statics'),
                      ('siteid_insample80', C_BASE2, 'site-ID embedding')],
        'Zero-shot': [('zeroshot', C_GLOBAL, 'physical statics'),
                      ('zeroshot_nostatic', C_NOSTATIC, 'no statics'),
                      ('siteid_zeroshot', C_BASE2, 'site-ID embedding')],
    }

    def stars(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    fig, ax = plt.subplots(figsize=(W1, W1 * (0.80 if JOURNAL else 0.65)))
    width = 0.25
    all_meds = []
    for gi, (gname, kinds) in enumerate(groups.items()):
        ref = _series(master, kinds[0][0])
        for bi, (kind, color, label) in enumerate(kinds):
            vals = _series(master, kind)
            med = vals.median()
            all_meds.append(med)
            xpos = gi + (bi - 1) * width
            ax.bar(xpos, med, width * 0.92, color=color,
                   edgecolor='white',
                   label=label if gi == 0 else None)
            if bi > 0:  # paired test vs physical statics within group
                stations = ref.index.intersection(vals.index)
                try:
                    _, p = wilcoxon(vals[stations] - ref[stations])
                except ValueError:
                    p = np.nan
                ax.text(xpos, med + 0.002, stars(p), ha='center',
                        fontsize=FONT - 3)
    lo = min(all_meds) - 0.015
    hi = max(all_meds) + 0.015
    ax.set_ylim(lo, hi)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(groups), fontsize=FONT)
    ax.set_ylabel('Median $R^2$', fontsize=FONT)
    ax.tick_params(labelsize=FONT - 1)
    if JOURNAL:
        ax.legend(fontsize=8.5, frameon=False, loc='lower center',
                  bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.0,
                  columnspacing=0.8)
    else:
        ax.legend(fontsize=FONT - 3, framealpha=0.95, loc='upper right')
    _save(fig, 'fig4_mechanism')


def main():
    master = pd.read_csv(MASTER, dtype={'station_id': str})
    fig1_zeroshot(master)
    fig2_budget(master)
    fig3_real()
    fig4_mechanism(master)


if __name__ == '__main__':
    main()
