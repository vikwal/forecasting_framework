#!/usr/bin/env python3
"""Bilanz der Abschlussbewertung: 12 Monate Training, 12 Monate Test.

Zwei Varianten der besten Konfiguration, je 4 Wiederholungen:
    final_lag    observed_features: [ghi, dhi]
    final_nolag  observed_features: []

Ausgegeben wird die Tabelle pro Station, das Mittel ueber die Stationen und der
gepaarte Test zwischen beiden Varianten.

Zur Lesart der Metriken — alle Laeufe nutzen ``target_transform: nwp_residual``,
die Zielgroesse ist also ``Messung - ICON-D2`` und die NWP-Baseline ist die
Nullreihe (utils/eval.py:557):

    Skill_NWP  1 - RMSE_Modell/RMSE_ICON. Die belastbare Zahl: um so viel wird
               die rohe ICON-D2-Prognose besser.
    R^2        R^2 der *Fehlerkorrektur*, nicht der Einstrahlungsprognose. Haengt
               ueber R^2 ~ 1 - (1 - Skill_NWP)^2 direkt am Skill.
    Skill      gegen die Persistenz derselben (Residuen-)Reihe. Schwaechste Aussage.

Zusaetzlich wird R^2 auf der absoluten GHI-Skala ausgewiesen, weil das die Zahl
ist, die man ausserhalb des Residuumsraums erwartet.
"""
from __future__ import annotations
import glob, pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

#: Nur Laeufe der Abschlussbewertung — die Ablationslaeufe haben test_start
#: 2025-02-01 und wuerden die Bilanz sonst verwaessern.
#: Bewusst NUR test_start: train_end wanderte vom 27.07. (Puffer) auf den 31.07.
#: (Monatsende, s. preprocessing.py:407 — der Split greift auf 'starttime', ein
#: Puffer verwirft die Laeufe dazwischen ersatzlos). Ein Filter auf train_end
#: wuerde je nach Aufbau die eine oder andere Haelfte unsichtbar machen.
_SETUP = {'test_start': '2024-08-01'}

#: Zwei Aufbauten mit identischem Zeitfenster — unterschieden wird ueber den
#: Dateinamen, nicht ueber die Config, weil train_end/test_start gleich sind.
#:   baseline  85 Stationen, rein zeitlicher Split (dieselben Stationen im Test)
#:   disjunkt  52 Stationen trainiert, 18 andere ausgewertet
SAETZE = {
    'baseline': {'mit Lag': '*base_lag_r*.pkl', 'ohne Lag': '*base_nolag_r*.pkl'},
    'disjunkt': {'mit Lag': '*final_lag_r*.pkl', 'ohne Lag': '*final_nolag_r*.pkl'},
}
VARIANTEN = SAETZE['baseline']


def _passt(d: dict) -> bool:
    cfg = (d.get('config') or {}).get('data', {})
    return all(str(cfg.get(k, '')) == v for k, v in _SETUP.items())


def _zeitraum(satz: str) -> str:
    """Zeitraum aus der Config lesen statt ihn in den Kopf zu schreiben."""
    import os, yaml
    p = ('configs/solar_baseline/config_solar_base_lag.yaml' if satz == 'baseline'
         else 'configs/solar_final/config_solar_final_lag.yaml')
    if not os.path.exists(p):
        return ''
    d = yaml.safe_load(open(p))['data']
    return (f"Training {d['train_start']}..{d['train_end']}, "
            f"Test {d['test_start']}..{d['test_end']}")


def _laden(varianten: dict) -> pd.DataFrame:
    rows = []
    for name, pat in varianten.items():
        n = 0
        for f in sorted(glob.glob(f'results/solar/{pat}')):
            if 'smoke' in f:
                continue
            try:
                d = pickle.load(open(f, 'rb'))
            except Exception as e:
                print(f'  ! {f}: {e}')
                continue
            if not _passt(d):
                continue
            n += 1
            ev = d['evaluation']
            ev = ev[ev['key'].notna()].copy()
            ev['target'] = [i[0] if isinstance(i, tuple) else 'ghi' for i in ev.index]
            ev = ev.drop_duplicates(subset=['key', 'target'])
            for _, r in ev.iterrows():
                rows.append({'Variante': name, 'target': r['target'],
                             'Station': str(r['key']).replace('synth_', '').replace('.csv', ''),
                             'Lauf': f[-19:-4], 'R2': r['R^2'], 'RMSE': r['RMSE'],
                             'MAE': r['MAE'], 'Skill': r['Skill'],
                             'Skill_NWP': r['Skill_NWP'], 'n_runs': r['n_runs']})
        print(f'  {name:9s} {n} Laeufe')
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--set', dest='satz', choices=sorted(SAETZE), default='baseline',
                    help="'baseline' = 85 Stationen, rein zeitlich; "
                         "'disjunkt' = 52 trainiert / 18 ausgewertet")
    args = ap.parse_args()
    varianten = SAETZE[args.satz]
    kopf = {'baseline': '85 Stationen, rein zeitlicher Split',
            'disjunkt': '52 Stationen trainiert, 18 andere ausgewertet'}[args.satz]
    print('=' * 96)
    print(f'BILANZ ({args.satz}) — {kopf}')
    print(_zeitraum(args.satz))
    print('=' * 96)
    df = _laden(varianten)
    if df.empty:
        print('  keine Ergebnisse')
        return
    print()

    for tgt in sorted(df.target.unique()):
        s = df[df.target == tgt]
        # erst ueber die Wiederholungen mitteln, dann erst vergleichen
        per = s.groupby(['Variante', 'Station'])[
            ['RMSE', 'MAE', 'R2', 'Skill', 'Skill_NWP', 'n_runs']].mean()

        print('=' * 96)
        print(f'{tgt.upper()} — pro Station')
        print('=' * 96)
        tab = per.reset_index().pivot(index='Station', columns='Variante',
                                      values=['RMSE', 'Skill_NWP', 'R2'])
        tab = tab.sort_values(('Skill_NWP', 'mit Lag'), ascending=False)
        print(tab.round(3).to_string())
        print()

        print(f'{tgt.upper()} — ueber die Stationen')
        agg = per.groupby('Variante')[['RMSE', 'MAE', 'R2', 'Skill', 'Skill_NWP']].agg(
            ['mean', 'median', 'min', 'max'])
        print(agg.round(4).to_string())

        # Streuung zwischen den Wiederholungen — Massstab fuer jeden Unterschied
        spread = (s.groupby(['Variante', 'Station', 'Lauf']).RMSE.mean()
                   .groupby(['Variante', 'Station']).std().groupby('Variante').mean())
        print('\n  mittlere Lauf-zu-Lauf-Streuung je Station (RMSE):')
        for k, v in spread.items():
            print(f'    {k:9s} {v:.3f}')

        # gepaarter Vergleich der beiden Varianten
        wide = per.reset_index().pivot(index='Station', columns='Variante', values='RMSE').dropna()
        if wide.shape[1] == 2 and len(wide) > 2:
            a, b = 'mit Lag', 'ohne Lag'
            st, p = wilcoxon(wide[b], wide[a])
            d = 100 * (wide[b].mean() - wide[a].mean()) / wide[a].mean()
            print(f'\n  gepaart ueber {len(wide)} Stationen: ohne Lag {wide[b].mean():.3f} '
                  f'vs mit Lag {wide[a].mean():.3f}  {d:+.2f}%  p={p:.5f}')
            besser = int((wide[b] < wide[a]).sum())
            print(f'  ohne Lag besser an {besser} von {len(wide)} Stationen')

        # absolute Skala: R^2 der Einstrahlungsprognose statt der Fehlerkorrektur
        print('\n  auf absoluter GHI-Skala (sigma aus den Rohdaten des Testfensters):')
        sigma = _sigma_ziel(tgt, args.satz)
        if sigma is None:
            print('    (keine Rohdaten gecached — uebersprungen)')
        else:
            for var in per.index.get_level_values('Variante').unique():
                r = per.loc[var]
                rmse_m = r.RMSE.mean()
                rmse_nwp = (r.RMSE / (1 - r.Skill_NWP)).mean()
                print(f'    {var:9s} RMSE {rmse_m:6.2f} -> R^2 {1 - (rmse_m/sigma)**2:.4f}   '
                      f'| ICON roh {rmse_nwp:6.2f} -> R^2 {1 - (rmse_nwp/sigma)**2:.4f}')
        print()


def _sigma_ziel(tgt: str, satz: str = 'baseline') -> float | None:
    """Standardabweichung der Messreihe im Testfenster, ueber alle Val-Stationen."""
    import os, yaml
    cfg_p = ('configs/solar_baseline/config_solar_base_lag.yaml' if satz == 'baseline'
             else 'configs/solar_final/config_solar_final_lag.yaml')
    if not os.path.exists(cfg_p):
        return None
    cfg = yaml.safe_load(open(cfg_p))
    d = cfg['data']
    # ausgewertet wird, was im Test auftaucht: bei der Baseline die 85 files-Stationen,
    # beim disjunkten Aufbau die 18 val_files.
    stationen = d.get('val_files') or d.get('files', [])
    vals, path = [], d['path']
    for st in stationen:
        f = os.path.join(path, f'Station_{st}.parquet')
        if not os.path.exists(f):
            continue
        try:
            m = pd.read_parquet(f, columns=[tgt])
        except Exception:
            continue
        idx = m.index
        if not isinstance(idx, pd.DatetimeIndex):
            continue
        sel = m[(idx >= d['test_start']) & (idx < d['test_end'])][tgt].dropna()
        if len(sel) > 1:
            # Rohparquet steht in J/cm^2 JE MESSINTERVALL, nicht in W/m^2 — dieselbe
            # Umrechnung wie solar.py:824, mit dem tatsaechlichen Abtastabstand statt
            # einer angenommenen Zehnminutigkeit.
            dt = pd.Series(sel.index).diff().dt.total_seconds().median()
            if not dt or pd.isna(dt):
                continue
            vals.append(sel * (1e4 / dt))
    if not vals:
        return None
    return float(pd.concat(vals).std())


if __name__ == '__main__':
    main()
