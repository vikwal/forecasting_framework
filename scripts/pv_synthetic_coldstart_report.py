"""Check all synthetic cold-start cases and export paired summaries."""
import argparse,json
from pathlib import Path
import pandas as pd
DAYS=[7,14,30,60,90]
def report(run,export):
    frames=[]
    for seed in [42,43,44]:
        root=run/f'seed{seed}';meta=json.loads((root/'metadata.json').read_text())
        if not (root/'COMPLETE').exists():raise RuntimeError(f'Seed {seed} incomplete')
        for days in DAYS:
            for client,keys in meta['groups'].items():
                case=root/f'{days:03d}d'/client
                assert (case/'COMPLETE').exists()
                expected={Path(k).name.removeprefix('synth_').removesuffix('.csv') for k in keys}
                for variant in ['local','central','fedgradient']:
                    df=pd.read_csv(case/f'{variant}_metrics.csv')
                    assert len(df)==2*len(keys) and not df.duplicated(['selection','park']).any()
                    assert set(df.seed)=={seed} and set(df.days)=={days} and set(df.client)=={client} and set(df.variant)=={variant}
                    assert set(df.selection)=={'selected','best_trained'}
                    for _,part in df.groupby('selection'):assert set(part.park)==expected
                    frames.append(df)
    data=pd.concat(frames,ignore_index=True)
    cols=['pinball_day','pinball_all','coverage_90','interval_score_90','median_r2']
    assert data[cols].notna().all().all()
    seeds=data.groupby(['selection','days','variant','seed'])[cols].mean().reset_index()
    clients=data.groupby(['selection','days','variant','seed','client'])[cols].mean().reset_index()
    summary=seeds.groupby(['selection','days','variant'])[cols].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    export.mkdir(parents=True,exist_ok=True)
    for name,df in [('per_park',data),('per_seed',seeds),('per_client',clients),('summary',summary)]:
        df['date']='2026-11-09';df['code_pin']=meta['code_pin'];df.to_csv(export/f'{name}_2026-11-09.csv',index=False)
    pivot=summary[summary.selection=='selected'].pivot(index='days',columns='variant',values='pinball_day_mean')
    lines=['# Synthetischer Kaltstart: wenige eigene Client-Messtage','',
           'Drei Seeds, acht Zielclients. Je Zielclient nur die letzten N Kalendertage vor 01.08.2024; sieben Donorclients mit voller Trainingshistorie. Alle Verfahren starten vom selben separaten synthetischen Vortraining je Seed. ES und Evaluation August2024–Juli2025 wie vereinbart. Keine Leistungshistorie als Input.','',
           'Tages-Pinball: zunächst gleich über alle Zielparks, dann über Seeds. Pro-Client-Werte und beste trainierte Checkpoints zusätzlich in CSVs. Kleine Verbesserungen unter min_delta0.0001 werden bei der Auswahl nicht übernommen.','',
           '| Tage | Lokal | Zentral | FedGradient | FedGradient-Vorteil zu lokal |','|---|---:|---:|---:|---:|']
    for days,r in pivot.iterrows():lines.append(f'| {days} | {r.local:.6f} | {r.central:.6f} | {r.fedgradient:.6f} | {(1-r.fedgradient/r.local)*100:.2f}% |')
    (export/'bericht_2026-11-09.md').write_text('\n'.join(lines)+'\n')
    print(pivot.to_string())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--export',type=Path,required=True);a=p.parse_args();report(a.run,a.export)
