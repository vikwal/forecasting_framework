#!/usr/bin/env python3
"""Vergleich der Solar-Laeufe vom 13.08.2026 (korrigiertes Fenster, Lookback-Fix).

Aufruf:  PYTHONPATH=. frcst/bin/python scripts/compare_solar_ablation.py

Alle Laeufe teilen Trainingsfenster (2023-08-01..2025-01-27), Testfenster
(2025-02-01..2025-08-01) und Stationen. RMSE/MAE stehen in W/m² und sind
vergleichbar; R² ist es NICHT (verschiedene Bezugsvarianz je Zieltransformation).
"""
from __future__ import annotations
import glob, pickle, sys
import numpy as np
import pandas as pd

RUNS = {
    'absolut':      ('*solar_absolut_resabsolut_2026081[3-9]*.pkl',  'Kontrolle, Ziel in W/m²'),
    'residual':     ('*solar_residual_resresidual_2026081[3-9]*.pkl','Referenz: Bias Correction'),
    'ab_single':    ('*ab_single_ab_single_2026081[3-9]*.pkl',       'nur ghi statt ghi+dhi'),
    'ab_features':  ('*ab_features_ab_features_2026081[3-9]*.pkl',   '+kt_nwp/airmass/dni_cs/dhi_cs'),
    'ab_clearsky':  ('*ab_clearsky_ab_clearsky_2026081[3-9]*.pkl',   'Ziel = ghi/ghi_clearsky'),
    'ab_raster15min':('*ab_raster15min_ab_raster15min_2026081[3-9]*.pkl','freq 15min'),
    'ab_raster10min':('*ab_raster10min_ab_raster10min_2026081[3-9]*.pkl','freq 10min'),
}


def load(pattern: str) -> pd.DataFrame | None:
    hits = sorted(glob.glob(f'results/solar/{pattern}'))
    if not hits:
        return None
    ev = pickle.load(open(hits[-1], 'rb'))['evaluation']
    ev = ev[ev['key'].notna()].copy()
    # Mehrziel-Laeufe tragen die Zielgroesse im Index, Einziel-Laeufe nicht.
    ev['target'] = [i[0] if isinstance(i, tuple) else 'ghi' for i in ev.index]
    ev = ev.drop_duplicates(subset=['key', 'target'])
    ev['datei'] = hits[-1].split('/')[-1]
    return ev


def main() -> int:
    frames = {}
    for name, (pat, _) in RUNS.items():
        df = load(pat)
        if df is None:
            print(f'  – {name}: noch kein Ergebnis', file=sys.stderr)
            continue
        frames[name] = df
    if not frames:
        print('Keine Ergebnisse gefunden.'); return 1

    for target in ('ghi', 'dhi'):
        rows = []
        for name, df in frames.items():
            s = df[df.target == target]
            if s.empty:
                continue
            skill = s['Skill_NWP']
            rows.append({
                'Lauf': name, 'Stationen': len(s),
                'RMSE': s.RMSE.mean(), 'MAE': s.MAE.mean(),
                'Skill_NWP': skill.mean(),
                'Skill_NWP>0': f'{int((skill > 0).sum())}/{len(s)}',
                'RMSE_NWP': (s.RMSE / (1 - skill)).mean(),
                'n_runs': int(s.n_runs.median()) if 'n_runs' in s else -1,
            })
        if not rows:
            continue
        t = pd.DataFrame(rows).set_index('Lauf')
        print(f'\n{"="*78}\n{target.upper()}\n{"="*78}')
        print(t.to_string(float_format=lambda x: f'{x:9.4f}'))

        if 'residual' in t.index:
            base = t.loc['residual']
            print(f'\n  gegen residual (RMSE {base.RMSE:.3f}):')
            for name in t.index:
                if name == 'residual':
                    continue
                d = 100 * (t.loc[name, 'RMSE'] / base.RMSE - 1)
                ds = t.loc[name, 'Skill_NWP'] - base.Skill_NWP
                print(f'    {name:16s} RMSE {d:+6.2f} %   Skill_NWP {ds:+.4f}')

        # Abnahmetest: dieselbe Prognose auf demselben Testset -> dieselbe Baseline
        nb = t['RMSE_NWP']
        vergleichbar = [i for i in nb.index if not i.startswith('ab_raster')]
        if len(vergleichbar) > 1:
            sub = nb.loc[vergleichbar]
            spread = (sub.max() - sub.min()) / sub.mean()
            if spread > 0.01:
                print(f'\n  ACHTUNG: NWP-Baselines weichen um {100*spread:.1f} % voneinander ab '
                      f'({sub.min():.2f} .. {sub.max():.2f}). Bei gleichem Raster und Testset '
                      f'muessen sie gleich sein — Skill_NWP waere sonst nicht vergleichbar.')
            else:
                print(f'\n  NWP-Baseline konsistent ({sub.mean():.2f} W/m²) — Skill_NWP vergleichbar.')

    print('\nR² ist zwischen Zieltransformationen nicht vergleichbar und daher nicht gelistet.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
