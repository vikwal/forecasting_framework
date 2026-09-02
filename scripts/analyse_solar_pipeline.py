#!/usr/bin/env python3
"""Auswertung des Solar-Nachtlaufs. Wird am Ende von run_solar_pipeline.sh aufgerufen.

Beantwortet zwei Fragen, die die Einzellaeufe offen lassen:

1. AUFLOESUNG — Skill_NWP ist ueber Raster hinweg NICHT vergleichbar, weil die
   NWP-Baseline mit der Aufloesung waechst. Hier werden die 15min- und
   10min-Vorhersagen flaechentreu auf 30min aggregiert und gegen den nativ auf
   30min trainierten Lauf auf identischen (Lauf, Lead)-Paaren gehalten.

2. ZIELGROESSE — mit mehreren Wiederholungen je Variante laesst sich der
   Unterschied zwischen den Varianten gegen die Lauf-zu-Lauf-Streuung halten.
"""
from __future__ import annotations
import glob, pickle
import numpy as np
import pandas as pd


#: Nur Laeufe des korrigierten Aufbaus. Die Laeufe vom 12.08. haben ein anderes
#: Testfenster (2025-08..2026-06), die fehlerhafte NWP-Baseline und den
#: Lookback-Fehler — sie miteinander zu mitteln wuerde die Lauf-zu-Lauf-Streuung
#: um Groessenordnungen ueberschaetzen (gemessen 7.8 statt <1 W/m²).
_SETUP = {'test_start': '2025-02-01', 'train_end': '2025-01-27'}


def _passt(d: dict) -> bool:
    cfg = (d.get('config') or {}).get('data', {})
    return all(str(cfg.get(k, '')) == v for k, v in _SETUP.items())


def _load(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(f'results/solar/{pattern}')):
        if 'smoke' in f:
            continue
        try:
            d = {'datei': f.split('/')[-1], **pickle.load(open(f, 'rb'))}
        except Exception as e:
            print(f'  ! {f}: {e}')
            continue
        if _passt(d):
            out.append(d)
    return out


def _metrics(d: dict) -> pd.DataFrame:
    ev = d['evaluation']
    ev = ev[ev['key'].notna()].copy()
    ev['target'] = [i[0] if isinstance(i, tuple) else 'ghi' for i in ev.index]
    return ev.drop_duplicates(subset=['key', 'target'])


# ---------------------------------------------------------------- Block B
def block_b() -> None:
    print('=' * 78)
    print('ZIELGROESSE — Varianten gegen die Lauf-zu-Lauf-Streuung')
    print('=' * 78)
    varianten = {
        'absolut':   '*solar_absolut_*.pkl',
        'residual':  '*solar_residual_*.pkl',
        'clearsky':  '*ab_clearsky_*.pkl',
        'clearsky2': '*clearsky2*.pkl',
        'grid4':     '*grid4*.pkl',
        'ohne_hist': '*noobs*.pkl',
        'features':  '*ab_features*.pkl',
        'feat+grid': '*featgrid*.pkl',
    }
    rows = []
    for name, pat in varianten.items():
        for d in _load(pat):
            m = _metrics(d)
            for tgt in m['target'].unique():
                s = m[m.target == tgt]
                rows.append({'Variante': name, 'target': tgt, 'Lauf': d['datei'][-19:-4],
                             'RMSE': s.RMSE.mean(), 'Skill_NWP': s.Skill_NWP.mean(),
                             'RMSE_NWP': (s.RMSE / (1 - s.Skill_NWP)).mean(),
                             'Stationen': len(s)})
    if not rows:
        print('  keine Ergebnisse'); return
    df = pd.DataFrame(rows)
    for tgt in sorted(df.target.unique()):
        s = df[df.target == tgt]
        agg = s.groupby('Variante').agg(
            Laeufe=('RMSE', 'size'), RMSE_Mittel=('RMSE', 'mean'), RMSE_Streuung=('RMSE', 'std'),
            Skill_Mittel=('Skill_NWP', 'mean'), Skill_Streuung=('Skill_NWP', 'std'))
        print(f'\n--- {tgt} ---')
        print(agg.to_string(float_format=lambda x: f'{x:9.4f}'))
        spann = agg.RMSE_Mittel.max() - agg.RMSE_Mittel.min()
        streu = agg.RMSE_Streuung.max()
        if pd.notna(streu) and streu > 0:
            print(f'\n  Spannweite zwischen den Varianten: {spann:.3f} W/m²')
            print(f'  groesste Streuung INNERHALB einer Variante: {streu:.3f} W/m²')
            if spann < 2 * streu:
                print('  => Der Unterschied zwischen den Varianten liegt in der Groessenordnung')
                print('     der Streuung eines einzelnen Laufs. Aus diesen Daten laesst sich')
                print('     KEINE Zielgroesse als besser ausweisen.')
            else:
                print('  => Der Unterschied uebersteigt die Streuung deutlich.')
        else:
            print('\n  (nur ein Lauf je Variante — Streuung nicht schaetzbar)')

        # Pflichtpruefung: gleiches Raster, gleiches Testset -> gleiche NWP-Baseline.
        # Weicht eine Variante ab, misst sie eine ANDERE Zielgroesse und ihre Zahlen
        # sind nicht vergleichbar. Genau so fiel auf, dass clearsky_index mit
        # clip_max=1.5 die dhi-Zielgroesse abschneidet statt sie zu transformieren:
        # RMSE -54 %, aber die Baseline sank von 47.77 auf 41.61 W/m² mit.
        nb = s.groupby('Variante')['RMSE_NWP'].mean()
        if len(nb) > 1:
            abw = (nb - nb.median()).abs() / nb.median()
            schlecht = abw[abw > 0.01]
            if len(schlecht):
                print('\n  ACHTUNG — abweichende NWP-Baseline:')
                for v, a in schlecht.items():
                    print(f'    {v}: {nb[v]:.2f} W/m² statt {nb.median():.2f} ({100*a:+.1f} %). '
                          f'Diese Variante misst eine andere Zielgroesse; ihr RMSE ist '
                          f'mit den uebrigen NICHT vergleichbar.')
            else:
                print(f'\n  NWP-Baseline konsistent ({nb.median():.2f} W/m²) ueber alle Varianten.')


# ---------------------------------------------------------------- Block A
def _to_30min(df: pd.DataFrame, faktor: int) -> pd.DataFrame:
    """Leads flaechentreu zu 30-Minuten-Mitteln zusammenfassen."""
    a = df.to_numpy()
    n = a.shape[1] // faktor
    a = a[:, :n * faktor].reshape(a.shape[0], n, faktor).mean(axis=2)
    return pd.DataFrame(a, index=df.index, columns=[f't+{i+1}' for i in range(n)])


def block_a() -> None:
    print('\n' + '=' * 78)
    print('AUFLOESUNG — alle Raster auf gemeinsames 30-Minuten-Raster gebracht')
    print('=' * 78)
    laeufe = {'30min': ('*pipe_rast30_pred*.pkl', 1),
              '15min': ('*pipe_rast15_pred*.pkl', 2),
              '10min': ('*pipe_rast10_pred*.pkl', 3)}
    daten = {}
    for name, (pat, faktor) in laeufe.items():
        hits = _load(pat)
        if not hits or not hits[-1].get('predictions'):
            print(f'  ! {name}: keine Vorhersagen gespeichert'); continue
        daten[name] = (hits[-1]['predictions'], faktor)
    if len(daten) < 2:
        print('  zu wenige Raster mit Vorhersagen fuer einen Vergleich'); return

    schluessel = set.intersection(*[set(p) for p, _ in daten.values()])
    print(f'\n  gemeinsame (Station, Zielgroesse): {len(schluessel)}')

    rows = []
    for k in sorted(schluessel):
        station, tgt = k
        agg = {}
        for name, (p, faktor) in daten.items():
            e = p[k]
            agg[name] = (_to_30min(e['pred'], faktor), _to_30min(e['true'], faktor),
                         _to_30min(e[[c for c in e if c.startswith('baseline::NWP')][0]], faktor))
        idx = set.intersection(*[set(v[0].index) for v in agg.values()])
        idx = sorted(idx)
        if not idx:
            continue
        for name, (pr, tr, nw) in agg.items():
            pr, tr, nw = pr.loc[idx], tr.loc[idx], nw.loc[idx]
            ok = np.isfinite(pr.to_numpy()) & np.isfinite(tr.to_numpy()) & np.isfinite(nw.to_numpy())
            e_m = (pr.to_numpy() - tr.to_numpy())[ok]
            e_n = (nw.to_numpy() - tr.to_numpy())[ok]
            rows.append({'Raster': name, 'station': station, 'target': tgt,
                         'RMSE': float(np.sqrt((e_m ** 2).mean())),
                         'RMSE_NWP': float(np.sqrt((e_n ** 2).mean())),
                         'Laeufe': len(idx)})
    if not rows:
        print('  keine gemeinsamen Vorhersagelaeufe'); return
    df = pd.DataFrame(rows)
    df['Skill_NWP'] = 1 - df.RMSE / df.RMSE_NWP
    for tgt in sorted(df.target.unique()):
        s = df[df.target == tgt]
        agg = s.groupby('Raster').agg(Stationen=('RMSE', 'size'), RMSE=('RMSE', 'mean'),
                                      RMSE_NWP=('RMSE_NWP', 'mean'), Skill_NWP=('Skill_NWP', 'mean'))
        print(f'\n--- {tgt} (alles auf 30min aggregiert, identische Samples) ---')
        print(agg.to_string(float_format=lambda x: f'{x:9.4f}'))
        print('  Jetzt ist RMSE direkt vergleichbar: gleiche Aufloesung, gleiche Samples,')
        print('  und die NWP-Baseline muss zwischen den Rastern uebereinstimmen.')
        if '30min' in agg.index:
            b = agg.loc['30min', 'RMSE']
            for r in agg.index:
                if r != '30min':
                    print(f'    {r} gegen nativ 30min: {100*(agg.loc[r,"RMSE"]/b-1):+.2f} % RMSE')


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    print(f'Solar-Nachtlauf — Auswertung {pd.Timestamp.now():%Y-%m-%d %H:%M}\n')
    block_b()
    block_a()
    print('\nHinweis: R² ist zwischen Zieltransformationen nicht vergleichbar und fehlt bewusst.')
