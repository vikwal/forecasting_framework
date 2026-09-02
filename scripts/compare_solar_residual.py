#!/usr/bin/env python3
"""Auswertung des Residuum-Experiments (52 Stationen, mit/ohne Residuum-Ziel).

Aufruf:  PYTHONPATH=. frcst/bin/python scripts/compare_solar_residual.py

Beide Laeufe sind identisch bis auf ``params.target_transform``. RMSE und MAE sind
direkt vergleichbar: im Residuumsraum bleiben Differenzen erhalten
(``(y_pred+nwp) - (y_true+nwp) = y_pred - y_true``).

**R² ist NICHT vergleichbar** — im Residuumslauf bezieht es sich auf die Varianz des
Residuums, im Kontrolllauf auf die von ghi/dhi. Ein niedriges R² beim Residuum ist
deshalb kein schlechteres Modell.

Das Skript prueft ausserdem, ob beide Laeufe dieselbe NWP-Baseline melden. Bis
Aug 2026 taten sie das nicht: eval.py hat die NWP-Spalte vor dem Vergleich per
``groupby('timestamp').mean()`` ueber alle ueberlappenden Vorhersagelaeufe gemittelt
und die Baseline damit um ~14 % zu gut gerechnet. Weichen die beiden Werte wieder
voneinander ab, ist der Fix nicht wirksam.
"""
from __future__ import annotations

import glob
import pickle
import sys

import pandas as pd

VARIANTS = {
    'absolut': 'Kontrolle — Ziel in W/m²',
    'residual': 'Ziel = Messung - ICON-D2 (Bias Correction)',
}


def load() -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        hits = sorted(glob.glob(f'results/solar/*res{variant}_2026*.pkl'))
        if not hits:
            print(f'WARNUNG: kein Ergebnis fuer {variant}', file=sys.stderr)
            continue
        ev = pickle.load(open(hits[-1], 'rb'))['evaluation']
        seen = set()
        for idx in ev.index:
            if not isinstance(idx, tuple) or idx in seen:
                continue          # 'mean'/'std' und Wiederholungen ueberspringen
            seen.add(idx)
            target, _model = idx
            r = ev.loc[[idx]].iloc[0]
            skill = r['Skill_NWP']
            rows.append({
                'variante': variant, 'target': target,
                'RMSE': r['RMSE'], 'MAE': r['MAE'], 'R2': r['R^2'],
                'Skill_NWP': skill,
                'RMSE_NWP': r['RMSE'] / (1 - skill) if skill != 1 else float('nan'),
                'datei': hits[-1].split('/')[-1],
            })
    return pd.DataFrame(rows)


def main() -> int:
    df = load()
    if df.empty:
        print('Keine Ergebnisse — laeuft das Training noch? '
              'tail -f logs/solar_residual/train_*.log')
        return 1

    for target in sorted(df['target'].unique()):
        s = df[df.target == target].set_index('variante')
        print(f'\n=== {target} ===')
        print(s[['RMSE', 'MAE', 'R2', 'RMSE_NWP', 'Skill_NWP']]
              .to_string(float_format=lambda x: f'{x:9.4f}'))

        if {'absolut', 'residual'}.issubset(s.index):
            d_rmse = 100 * (s.loc['residual', 'RMSE'] / s.loc['absolut', 'RMSE'] - 1)
            d_mae = 100 * (s.loc['residual', 'MAE'] / s.loc['absolut', 'MAE'] - 1)
            print(f'  Residuum vs Kontrolle:  RMSE {d_rmse:+.2f} %   MAE {d_mae:+.2f} %')

            base = s['RMSE_NWP']
            spread = abs(base.max() - base.min()) / base.mean()
            if spread > 0.01:
                print(f'  ACHTUNG: die beiden Laeufe melden verschiedene NWP-Baselines '
                      f'({base.min():.2f} vs {base.max():.2f}, {100*spread:.1f} % '
                      f'Unterschied). Es ist dieselbe Prognose auf demselben Testset — '
                      f'Skill_NWP ist dann zwischen den Laeufen NICHT vergleichbar.')
            else:
                print(f'  NWP-Baseline konsistent ({base.mean():.2f} W/m²) — '
                      f'Skill_NWP ist vergleichbar.')

        if (s['Skill_NWP'] < 0).any():
            schlecht = list(s.index[s['Skill_NWP'] < 0])
            print(f'  Skill_NWP negativ bei: {schlecht} — dort ist das Modell '
                  f'schlechter als die rohe ICON-D2-Prognose.')

    print('\nR² ist zwischen den Varianten nicht vergleichbar (verschiedene '
          'Bezugsvarianz). Vergleiche RMSE, MAE und Skill_NWP.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
