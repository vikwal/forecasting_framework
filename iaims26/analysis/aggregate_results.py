"""Merge all iAIMS26 result CSVs into one tidy per-farm table.

Output: iaims26/results/master_per_farm.csv with columns
  station_id, period, model_kind, budget, variant, fold, R2, RMSE, MAE, n_windows

model_kind values:
  zeroshot          fold model on its held-out farm (from results/zeroshot/)
  insample80        fold model on its own 80 training farms
  global100         100-farm reference model (zeroshot_g100.csv, insample)
  finetune          fine-tuned fold model (budget x variant from results/finetune/)
  local             from-scratch local baseline (results/finetune, variant=local)
  siteid_global     site-ID model, in-sample (results/site_embedding/siteid_global100.csv)
  siteid_zeroshot   site-ID fold models on held-out farms (siteid_fold*.csv)

Also runs sanity checks (warnings, not failures).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

OUT = common.RESULTS_DIR / 'master_per_farm.csv'
COLS = ['station_id', 'period', 'model_kind', 'budget', 'variant', 'fold',
        'R2', 'RMSE', 'MAE', 'n_windows']


def _std(df, model_kind, budget='-', variant='-'):
    out = pd.DataFrame({
        'station_id': df['station_id'].astype(str).str.zfill(5),
        'period': df['period'],
        'model_kind': model_kind,
        'budget': df['budget'] if 'budget' in df else budget,
        'variant': df['variant'] if 'variant' in df else variant,
        'fold': df['fold'] if 'fold' in df else -1,
        'R2': df['R2'], 'RMSE': df['RMSE'], 'MAE': df['MAE'],
        'n_windows': df['n_windows'],
    })
    return out


def main():
    frames = []

    # zero-shot / in-sample fold evaluations
    for f in sorted((common.RESULTS_DIR / 'zeroshot').glob('zeroshot_fold*.csv')):
        df = pd.read_csv(f)
        frames.append(_std(df[df.split == 'heldout'], 'zeroshot'))
        frames.append(_std(df[df.split == 'insample'], 'insample80'))
    g100 = common.RESULTS_DIR / 'zeroshot' / 'zeroshot_g100.csv'
    if g100.exists():
        df = pd.read_csv(g100)
        frames.append(_std(df[df.split == 'insample'], 'global100'))

    # fine-tuning grid + local baselines (+ per-farm zero-shot duplicates, dropped)
    for f in sorted((common.RESULTS_DIR / 'finetune').glob('ft_*.csv')):
        if 'smoke' in f.name:
            continue
        df = pd.read_csv(f)
        df = df[df.period != 'no_data']
        frames.append(_std(df[df.variant == 'local'], 'local'))
        ft = df[(df.variant != 'local') & (df.budget != 'zeroshot')]
        frames.append(_std(ft, 'finetune'))

    # site-ID embedding
    sid_dir = common.RESULTS_DIR / 'site_embedding'
    f = sid_dir / 'siteid_global100.csv'
    if f.exists():
        df = pd.read_csv(f)
        frames.append(_std(df[df.split == 'insample'], 'siteid_global'))
    for f in sorted(sid_dir.glob('siteid_fold*.csv')):
        df = pd.read_csv(f)
        frames.append(_std(df[df.split == 'heldout'], 'siteid_zeroshot'))
        frames.append(_std(df[df.split == 'insample'], 'siteid_insample80'))

    master = pd.concat([f for f in frames if len(f)], ignore_index=True)
    master.to_csv(OUT, index=False)
    print(f'wrote {OUT}: {len(master)} rows')
    print(master.groupby(['model_kind', 'period']).size().unstack(fill_value=0))

    # ---- sanity checks (median R2, full period) ----
    med = (master[master.period == 'full']
           .groupby(['model_kind', 'budget', 'variant'])['R2'].median())

    def _get(kind, budget='-', variant='-'):
        try:
            return med.loc[(kind, budget, variant)]
        except KeyError:
            return None

    zs, ins, loc = _get('zeroshot'), _get('insample80'), _get('local')
    if zs is not None and ins is not None and zs > ins:
        print(f'WARN: zero-shot median R2 ({zs:.3f}) > in-sample ({ins:.3f})')
    if zs is not None and loc is not None:
        print(f'median R2 (full): zeroshot={zs:.3f} local={loc:.3f} insample80={ins:.3f}')
    for (kind, budget, variant), value in med.items():
        if kind == 'finetune' and zs is not None and value < zs - 0.05:
            print(f'WARN: finetune {budget}/{variant} median R2 {value:.3f} '
                  f'well below zeroshot {zs:.3f}')


if __name__ == '__main__':
    main()
