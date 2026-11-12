"""Validate and summarize the paired synthetic transfer experiment."""
import argparse
import json
from pathlib import Path
import pandas as pd


def report(run, export):
    meta=json.loads((run/'metadata.json').read_text())
    frames=[]
    for seed in [42,43,44]:
        root=run/f'seed{seed}'
        if not (root/'COMPLETE').exists(): raise RuntimeError(f'Seed {seed} incomplete')
        df=pd.read_csv(root/'metrics.csv')
        assert len(df)==4*len(meta['target_available'])
        assert not df.duplicated(['variant','park']).any()
        assert set(df.variant)=={'zeroshot','local','central','fedgradient'}
        for _,part in df.groupby('variant'):
            assert set(part.park)==set(meta['target_available']) and set(part.client)==set(meta['groups'])
            assert set(part.seed)=={seed}
        frames.append(df)
    data=pd.concat(frames,ignore_index=True)
    cols=['pinball_day','pinball_all','coverage_90','width_90','interval_score_90','median_r2','crossing_rate']
    assert data[cols].notna().all().all()
    per_seed=data.groupby(['variant','seed'])[cols].mean().reset_index()
    per_client=data.groupby(['variant','client','seed'])[cols].mean().reset_index()
    summary=per_seed.groupby('variant')[cols].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index()
    export.mkdir(parents=True,exist_ok=True)
    for name,table in [('per_park',data),('per_seed',per_seed),('per_client',per_client),('summary',summary)]:
        table['date']='2026-11-09'; table['code_pin']=meta['code_pin']
        table.to_csv(export/f'{name}_2026-11-09.csv',index=False)
    lines=['# Synthetischer Quantiltransfer: separate Quellparks auf Client-Parks\n',
           'Stand 2026-11-09. Seeds 42/43/44; sieben Quantile; keine Leistungshistorie als Eingang.\n',
           f"Vortraining: {len(meta['source_available'])} von 60 Parks. Anpassung und Vergleich: {len(meta['target_available'])} von 166 Parks in acht Clients. Fehlende Parks sind in der Lauf-Metadatei dokumentiert.\n",
           'Training jeweils 24.07.2023–31.07.2024; ES und Evaluation 01.08.2024–31.07.2025. Das zusätzliche Testjahr wurde nicht verwendet.\n',
           'Gleiche Initialisierung und Quell-Scaler je Seed. Lokal ein Modell je Client; zentral/FedGradient gleiche Vereinigung von bis zu acht Teilbatches à elf Fenstern. Adam 3e-4, globales Clipping 7, maximal 100 Epochen, patience 10, min_delta 0.0001. ES anhand Pinball über alle Zellen der jeweiligen Trainingsparks im Auswertefenster.\n',
           'Primär roher Pinball tags (GTI > 10); zunächst gleiches Gewicht je Park, dann Mittel/Standardabweichung über Seeds. Intervalle aus sortierten, nichtnegativen Quantilen. Seed-Streuung ist kein Konfidenzintervall.\n',
           '| Variante | Pinball tags ± SD | 90%-Coverage | Interval Score 90 | Median-R² |\n|---|---:|---:|---:|---:|']
    for v in ['zeroshot','local','central','fedgradient']:
        r=summary.set_index('variant').loc[v]
        lines.append(f"| {v} | {r.pinball_day_mean:.6f} ± {r.pinball_day_std:.6f} | {r.coverage_90_mean:.3%} | {r.interval_score_90_mean:.6f} | {r.median_r2_mean:.4f} |")
    (export/'bericht_2026-11-09.md').write_text('\n\n'.join(lines)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--export',type=Path,required=True)
    a=p.parse_args(); report(a.run,a.export)
