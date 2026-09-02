#!/usr/bin/env python3
"""Wie viel Messung braucht der Forecast? Vergleich auf identischer Stichprobe.

Drei Informationsstaende fuer dieselbe Aufgabe:

  eigene_historie   ghi/dhi der Station selbst als observed features (Referenz)
  nachbarn          die 4 naechsten Stationen aus dem Trainingsbestand,
                    eigene Historie unterdrueckt — Standort ohne eigene Messung
  keine_messung     observed leer, nur ICON-D2 + Geometrie + Statik

Die Nachbar-Variante verliert rund 15 % der Vorhersagelaeufe an Messluecken der
Nachbarn. Ein Vergleich ueber die jeweils eigenen Stichproben waere deshalb
schief; hier wird auf der SCHNITTMENGE aller drei Laeufe gerechnet.
"""
from __future__ import annotations
import glob, pickle
import numpy as np
import pandas as pd

QUELLEN = {
    'eigene_historie': '*pipe_rast30_pred*.pkl',
    'nachbarn':        '*pipe_neigh_r1*.pkl',
    'keine_messung':   '*pipe_noobs_pred*.pkl',
}


def lade(pat: str):
    hits = sorted(glob.glob(f'results/solar/{pat}'))
    if not hits:
        return None
    d = pickle.load(open(hits[-1], 'rb'))
    return d.get('predictions')


def main() -> int:
    daten = {}
    for name, pat in QUELLEN.items():
        p = lade(pat)
        if not p:
            print(f'  ! {name}: keine Vorhersagen gefunden ({pat})')
            continue
        daten[name] = p
    if len(daten) < 2:
        print('zu wenige Laeufe fuer einen Vergleich')
        return 1

    schluessel = sorted(set.intersection(*[set(p) for p in daten.values()]))
    print(f'gemeinsame (Station, Zielgroesse): {len(schluessel)}\n')

    rows = []
    for k in schluessel:
        idx = set.intersection(*[set(daten[n][k]['pred'].index) for n in daten])
        idx = sorted(idx)
        if not idx:
            continue
        for name in daten:
            e = daten[name][k]
            pr, tr = e['pred'].loc[idx], e['true'].loc[idx]
            nw = e[[c for c in e if c.startswith('baseline::NWP')][0]].loc[idx]
            ok = (np.isfinite(pr.to_numpy()) & np.isfinite(tr.to_numpy())
                  & np.isfinite(nw.to_numpy()))
            rows.append({'Quelle': name, 'station': k[0], 'target': k[1],
                         'RMSE': float(np.sqrt(((pr.to_numpy() - tr.to_numpy())[ok] ** 2).mean())),
                         'RMSE_NWP': float(np.sqrt(((nw.to_numpy() - tr.to_numpy())[ok] ** 2).mean())),
                         'Laeufe': len(idx)})
    df = pd.DataFrame(rows)
    if df.empty:
        print('keine gemeinsamen Vorhersagelaeufe'); return 1
    df['Skill_NWP'] = 1 - df.RMSE / df.RMSE_NWP

    for tgt in sorted(df.target.unique()):
        s = df[df.target == tgt]
        agg = s.groupby('Quelle').agg(Stationen=('RMSE', 'size'), RMSE=('RMSE', 'mean'),
                                      RMSE_NWP=('RMSE_NWP', 'mean'),
                                      Skill_NWP=('Skill_NWP', 'mean'),
                                      Laeufe=('Laeufe', 'median'))
        print(f'--- {tgt} (identische Stichprobe) ---')
        print(agg.to_string(float_format=lambda x: f'{x:9.4f}'))
        if 'eigene_historie' in agg.index:
            b = agg.loc['eigene_historie']
            print(f'\n  gegen eigene Historie (RMSE {b.RMSE:.3f}, Skill_NWP {b.Skill_NWP:+.4f}):')
            for n in agg.index:
                if n == 'eigene_historie':
                    continue
                d = 100 * (agg.loc[n, 'RMSE'] / b.RMSE - 1)
                anteil = agg.loc[n, 'Skill_NWP'] / b.Skill_NWP if b.Skill_NWP else float('nan')
                print(f'    {n:16s} RMSE {d:+6.2f} %   Skill_NWP {agg.loc[n,"Skill_NWP"]:+.4f} '
                      f'= {100*anteil:5.1f} % des Referenz-Skills')
            # paarweise je Station
            piv = s.pivot_table(index='station', columns='Quelle', values='RMSE')
            for n in piv.columns:
                if n == 'eigene_historie':
                    continue
                d = piv[n] - piv['eigene_historie']
                print(f'    {n:16s} besser bei {int((d < 0).sum())}/{len(d)} Stationen')
        print()
    return 0


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    print(f'Messnetz-Vergleich {pd.Timestamp.now():%Y-%m-%d %H:%M}\n')
    raise SystemExit(main())
