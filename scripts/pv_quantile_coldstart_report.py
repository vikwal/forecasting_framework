#!/usr/bin/env python3
"""Validate completeness and aggregate the PV quantile cold-start comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


DAYS=[7,14,30,60,90]
PARKS=['pv1','pv2','pv3','pv5','pv6','pv7']
METHODS=['local','central','fedgradient']


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    ap.add_argument('--export',type=Path,required=True)
    a=ap.parse_args()
    frames,training,budgets=[],[],[]
    for seed in [42,43,44]:
        root=a.run/f'seed{seed}'
        assert (root/'COMPLETE').exists(), f'Seed {seed} incomplete'
        df=pd.read_csv(root/'metrics.csv')
        expected=pd.MultiIndex.from_product([DAYS,PARKS,METHODS,['selected','best_trained']],names=['days','park','variant','selection'])
        assert not df.duplicated(expected.names).any()
        assert set(map(tuple,df[expected.names].values))==set(expected)
        assert df.seed.eq(seed).all()
        frames.append(df)
        m=json.loads((root/'metadata.json').read_text())
        budgets += [dict(seed=seed,**{k:v for k,v in b.items() if k!='issues'}) for b in m['budgets']]
        training += [{k:v for k,v in row.items() if not isinstance(v,dict)} for row in json.loads((root/'training.json').read_text())]
    df=pd.concat(frames,ignore_index=True)
    measures=['pinball_day','pinball_all','pinball_day_sorted','coverage_90','width_90','interval_score_90','median_r2','crossing_rate']
    assert np.isfinite(df[measures].values).all()
    perseed=df.groupby(['selection','days','variant','seed'])[measures].mean().reset_index()
    summary=perseed.groupby(['selection','days','variant'])[measures].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index()
    selected=df[df.selection=='selected']
    paired=selected.pivot(index=['days','park','seed'],columns='variant',values='pinball_day').reset_index()
    for method in ['central','fedgradient']:
        paired[f'{method}_gain_pct']=100*(paired.local-paired[method])/paired.local
    a.export.mkdir(parents=True,exist_ok=False)
    for name,table in [('per_park',df),('per_seed',perseed),('summary',summary),('paired',paired),('training',pd.DataFrame(training)),('budgets',pd.DataFrame(budgets))]:
        table['date']='2026-11-09'
        table['code_pin']=df.code_pin.iloc[0]
        table.to_csv(a.export/f'{name}_2026-11-09.csv',index=False)
    print(summary[summary.selection=='selected'].to_string(index=False))
    print('\nPaired gains (%), positive favors shared training:')
    print(paired.groupby('days')[['central_gain_pct','fedgradient_gain_pct']].agg(['mean','min','max']).to_string())
    print('\nActual windows:')
    print(pd.DataFrame(budgets).groupby('days').windows.agg(['min','max']).to_string())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    ss=summary[summary.selection=='selected']
    labels={'local':'Lokal','central':'Zentral','fedgradient':'FedGradient'}
    for method in METHODS:
        part=ss[ss.variant==method].sort_values('days')
        for ax,metric in zip(axes,['pinball_day','interval_score_90']):
            ax.errorbar(part.days,part[metric+'_mean'],yerr=part[metric+'_std'],marker='o',capsize=3,label=labels[method])
            ax.set_xlabel('Eigene Messtage im Training')
            ax.set_xticks(DAYS)
            ax.grid(alpha=.2)
    axes[0].set_ylabel('Pinball-Loss tagsüber (kleiner ist besser)')
    axes[1].set_ylabel('Intervall-Score, 90 % (kleiner ist besser)')
    axes[0].legend()
    fig.suptitle('Quantil-Fine-Tuning: fünf Donorparks mit voller Historie\nMittel über sechs Zielparks; Balken: SD über drei Seeds',fontsize=11)
    fig.savefig(a.export/'kaltstart_2026-11-09.png',dpi=180)
    fig.savefig(a.export/'kaltstart_2026-11-09.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
