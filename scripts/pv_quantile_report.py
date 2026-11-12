#!/usr/bin/env python3
"""Aggregate the three paired quantile fine-tuning seeds without rerunning models."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    ap.add_argument('--export',type=Path,required=True)
    a=ap.parse_args()
    frames, training=[],[]
    for seed in [42,43,44]:
        root=a.run/f'seed{seed}'
        if not (root/'COMPLETE').exists():
            raise RuntimeError(f'Seed {seed} incomplete')
        df=pd.read_csv(root/'metrics.csv')
        assert len(df)==42 and not df.duplicated(['variant','selection','park']).any()
        assert set(df.park)=={'pv1','pv2','pv3','pv5','pv6','pv7'}
        frames.append(df)
        for log in json.loads((root/'training.json').read_text()):
            training.append({'seed':seed,**{k:v for k,v in log.items() if not isinstance(v,dict)}})
    df=pd.concat(frames,ignore_index=True)
    metrics=['pinball_day','pinball_all','coverage_90','width_90','interval_score_90','median_r2','crossing_rate']
    seedmean=df.groupby(['selection','variant','seed'])[metrics].mean().reset_index()
    summary=seedmean.groupby(['selection','variant'])[metrics].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index()
    a.export.mkdir(parents=True,exist_ok=False)
    pin=df.code_pin.iloc[0]
    for name,table in [('per_park',df),('per_seed',seedmean),('summary',summary),('training',pd.DataFrame(training))]:
        table['code_pin']=pin
        table['date']='2026-11-09'
        table.to_csv(a.export/f'{name}_2026-11-09.csv',index=False)
    print(summary.to_string(index=False))
    print(pd.DataFrame(training)[['seed','variant','best_epoch','epochs','initial_pinball','selected_pinball','best_trained_pinball']].to_string(index=False))


if __name__=='__main__':
    main()
