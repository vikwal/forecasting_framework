#!/usr/bin/env python3
"""Auswertung des Solar-Rasterexperiments (30min / 15min / 10min).

Warum nicht einfach die RMSE nebeneinanderstellen: die drei Laeufe sagen
verschiedene Groessen vorher. Ein 10-min-Wert schwankt staerker als ein
30-min-Mittel, also ist sein RMSE zwangslaeufig hoeher — unabhaengig davon, ob das
Modell besser oder schlechter ist. Belegbar an der NWP-Baseline selbst, die
dieselbe Prognose nur anders aggregiert und trotzdem um ~10 % wandert.

Vergleichbar ist deshalb nur **Skill_NWP = 1 - RMSE_Modell / RMSE_NWP**: Zaehler und
Nenner stammen aus demselben Raster, der Rastereffekt kuerzt sich weitgehend heraus.

Aufruf:  PYTHONPATH=. frcst/bin/python scripts/compare_solar_raster.py
"""
from __future__ import annotations

import glob
import pickle
import sys

import pandas as pd

FREQS = ['30min', '15min', '10min']
VARIANT = {
    '30min': 'Referenz  — beide Quellen exakt (kgV(10,15) = 30)',
    '15min': 'Variante A — Messung 10 → 15 min umverteilt',
    '10min': 'Variante C — ICON 15 → 10 min umverteilt',
}


def load() -> pd.DataFrame:
    rows = []
    for freq in FREQS:
        hits = sorted(glob.glob(f'results/solar/*raster{freq}_2026*.pkl'))
        if not hits:
            print(f'WARNUNG: kein Ergebnis fuer {freq}', file=sys.stderr)
            continue
        ev = pickle.load(open(hits[-1], 'rb'))['evaluation']
        for idx in ev.index:
            if not isinstance(idx, tuple):
                continue           # 'mean'/'std'-Zeilen ueberspringen
            target, _model = idx
            r = ev.loc[[idx]].iloc[0]
            skill_nwp = r['Skill_NWP']
            rows.append({
                'freq': freq, 'target': target,
                'R2': r['R^2'], 'RMSE': r['RMSE'], 'MAE': r['MAE'],
                'Skill_pers': r['Skill'], 'Skill_NWP': skill_nwp,
                # RMSE der Baseline, aus der Skill-Definition zurueckgerechnet
                'RMSE_NWP': r['RMSE'] / (1 - skill_nwp) if skill_nwp != 1 else float('nan'),
            })
    return pd.DataFrame(rows)


def main() -> int:
    df = load()
    if df.empty:
        print('Keine Ergebnisse gefunden — erst scripts/launch_solar_raster.sh laufen lassen.')
        return 1

    for target in df['target'].unique():
        s = df[df.target == target].set_index('freq').reindex(FREQS)
        print(f'\n=== {target} ===')
        print(s[['R2', 'RMSE', 'RMSE_NWP', 'MAE', 'Skill_pers', 'Skill_NWP']]
              .to_string(float_format=lambda x: f'{x:9.4f}'))
        rng = s['Skill_NWP'].max() - s['Skill_NWP'].min()
        best = s['Skill_NWP'].idxmax()
        print(f'  bester Skill_NWP: {best} ({s.loc[best, "Skill_NWP"]:+.4f}), '
              f'Spannweite ueber die drei Raster: {rng:.4f}')
        if (s['Skill_NWP'] < 0).all():
            print('  ACHTUNG: Skill_NWP ist ueberall negativ — das Modell ist in jedem '
                  'Raster schlechter als die rohe ICON-D2-Prognose. Ein Rastervergleich '
                  'zwischen drei Modellen, die alle die Baseline verfehlen, traegt nicht.')

    # Widersprechen sich die Zielgroessen in der Rangfolge?
    order = {t: list(df[df.target == t].sort_values('Skill_NWP', ascending=False)['freq'])
             for t in df['target'].unique()}
    print('\nRangfolge nach Skill_NWP je Zielgroesse:')
    for t, o in order.items():
        print(f'  {t}: {" > ".join(o)}')
    if len({tuple(o) for o in order.values()}) > 1:
        print('  → Die Zielgroessen ordnen die Raster UNTERSCHIEDLICH. Bei einer '
              'Spannweite in der Groessenordnung des Seed-Rauschens heisst das: '
              'das Experiment trennt die Varianten nicht.')

    print('\nHinweis: die drei Laeufe unterscheiden sich zwangslaeufig auch in der '
          'Sequenzlaenge (96/192/288 Schritte bei festem 48-h-Horizont). Gemessen wird '
          'also "Raster inklusive Sequenzlaenge", nicht das Raster isoliert.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
