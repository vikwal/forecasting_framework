#!/usr/bin/env python3
"""Few-own-label PV quantile fine-tuning with full-history donor clients.

Uses the existing quantile/FedGradient trainer and unchanged February--May ES.
Few days limit training labels only, not validation labels. No conformal step.
"""
import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pv_quantile_finetune as ft
from pv_quantile_metrics import metrics


def label_bounds(frame, indices):
    """Actual target timestamp bounds for the NWP run behind each window."""
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError('Expected NWP MultiIndex')
    idx = frame.index
    table = pd.DataFrame({'issue':idx.get_level_values('starttime'),
                          'valid':idx.get_level_values('timestamp')})
    bounds = table.groupby('issue').valid.agg(['min','max'])
    selected = bounds.loc[pd.DatetimeIndex(indices)]
    return pd.DatetimeIndex(selected['min']), pd.DatetimeIndex(selected['max'])


def subset(item, lower, upper, days):
    """Keep only windows whose entire target horizon is within N calendar days."""
    end = pd.Timestamp('2024-02-01', tz='UTC')
    start = end - pd.Timedelta(days=days)
    lo, hi = pd.to_datetime(lower, utc=True), pd.to_datetime(upper, utc=True)
    keep = np.asarray((lo >= start) & (hi < end))
    if not keep.any():
        raise ValueError(f'No complete target windows for {days} days')
    selected = {k:v[keep] for k,v in item['X_train'].items()}, item['y_train'][keep]
    return selected, dict(days=days, cutoff=str(start), end_exclusive=str(end), windows=int(keep.sum()),
        first_target=str(lo[keep].min()), last_target=str(hi[keep].max()),
        issues=[str(x) for x in item['index_train'][keep]])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--days',type=int,nargs='+',default=[7,14,30,60,90])
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    if any(n<2 for n in a.days) or len(a.days)!=len(set(a.days)):
        ap.error('Days must be unique integers >=2')
    out=Path(a.output)/f'seed{a.seed}'
    out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    device=torch.device(f'cuda:{a.gpu}')
    cfg=ft.base.load_cfg(ft.base.DATA_CFG)
    cfg['model']=copy.deepcopy(ft.base.load_cfg(ft.base.Q_CFG)['model'])
    cfg['model']['early_stopping']['monitor']='val_pinball'
    assert cfg['params']['observed_features']==[] and not cfg['data']['scale_y']
    qs=cfg['model']['tft']['quantiles']
    model_path=ft.base.Q_MODEL.format(seed=a.seed)
    state,(sx,_)=ft.finetune.load_init(model_path)
    cfg['scaler_x']=sx
    features=ft.preprocessing.get_features(config=cfg)
    dfs=ft.preprocessing.get_data(data_dir=cfg['data']['path'],config=cfg,freq=cfg['data']['freq'],features=features,files_key='files')
    data,bounds={},{}
    for key,frame in dfs.items():
        if frame.isna().any().any():
            raise ValueError(f'{key}: missing values could invoke pre-subset target imputation')
        prepared,_=ft.preprocessing.pipeline(data=frame,config=cfg,
            known_cols=features['known'],observed_cols=features['observed'],static_cols=features['static'],
            target_col=ft.preprocessing.get_target_cols(cfg)[0])
        data[key]=prepared
        bounds[key]=label_bounds(frame,prepared['index_train'])
    keys=sorted(data)
    assert len(keys)==6
    # Reusing exactly the original preprocessing is essential for this comparison.
    original=ft.base.windows(dfs,cfg,features,sx,ft.base.CAL,ft.base.EV)
    for key in keys:
        np.testing.assert_array_equal(data[key]['index_test'],original[key]['index_test'])
        np.testing.assert_array_equal(data[key]['nwp_raw_test'],original[key]['nwp_raw_test'])
        for part in ['train','test']:
            np.testing.assert_array_equal(data[key]['y_'+part],original[key]['y_'+part])
            for feature in data[key]['X_'+part]:
                np.testing.assert_array_equal(data[key]['X_'+part][feature],original[key]['X_'+part][feature])
    del original,dfs
    cfg['model']['feature_dim']=ft.tools.get_feature_dim(X=data[keys[0]]['X_train'])
    hp=ft.hpo.get_hyperparameters(config=cfg)
    hp['lr']=3e-4
    pin=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()+' + dirty'
    sources=[__file__,'scripts/pv_quantile_finetune.py','scripts/pv_quantile_metrics.py','scripts/pv_conformal.py',
             'utils/fedgradient.py','utils/preprocessing.py',ft.base.DATA_CFG+'.yaml',ft.base.Q_CFG+'.yaml',model_path]
    meta=dict(seed=a.seed,days=a.days,training=ft.base.CAL,evaluation=ft.base.EV,lr=3e-4,
        quantiles=qs,early_stopping=cfg['model']['early_stopping'],date='2026-11-09',code_pin=pin,
        own_budget='complete training target horizons in last N calendar days; validation unrestricted as agreed',
        donor_budget='full training history through January; no target labels before cutoff',
        batch='local 64; central and FedGradient identical unions of per-client batch11 including remainders',
        sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        full_windows={k:len(data[k]['y_train']) for k in keys},budgets=[])
    rows,logs=[],[]
    def evaluate(model,variant,selection,key,days,case):
        model.eval()
        it=data[key]
        pred=ft.base.predict(model,it['X_test'],device)
        park=key.replace('synth_','').replace('_meter.csv','')
        rows.append(dict(variant=variant,selection=selection,park=park,days=days,seed=a.seed,
            date='2026-11-09',code_pin=pin,**metrics(pred,it['y_test'],qs,np.asarray(it['nwp_raw_test'])>10)))
        np.savez_compressed(case/f'{variant}_{selection}.npz',pred=pred,true=it['y_test'],gti=it['nwp_raw_test'],
            quantiles=qs,index=np.asarray(it['index_test']).astype(str))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    for days in a.days:
        for target in keys:
            park=target.replace('synth_','').replace('_meter.csv','')
            case=out/f'{days:03d}d'/park
            case.mkdir(parents=True)
            own,info=subset(data[target],*bounds[target],days)
            meta['budgets'].append(dict(park=park,**info))
            (out/'metadata.json').write_text(json.dumps(meta,indent=2))
            logging.info('CASE seed=%s days=%s park=%s own_windows=%s',a.seed,days,park,info['windows'])
            entries=[own if k==target else ft.cat([data[k]],'train') for k in keys]
            for variant in ['local','central','fedgradient']:
                name='local_'+park if variant=='local' else variant
                train_data=[own] if variant=='local' else entries
                val=[data[target]] if variant=='local' else [data[k] for k in keys]
                model,log,trained_sd=ft.train(name,train_data,val,state,cfg,hp,device,a.seed,qs,case)
                evaluate(model,variant,'selected',target,days,case)
                model.load_state_dict(trained_sd)
                evaluate(model,variant,'best_trained',target,days,case)
                logs.append(dict(days=days,park=park,seed=a.seed,method=variant,**log))
                (out/'training.json').write_text(json.dumps(logs,indent=2))
                del model
                torch.cuda.empty_cache()
            (case/'COMPLETE').write_text('Three variants complete\n')
    assert len(rows)==len(a.days)*len(keys)*6
    (out/'COMPLETE').write_text('All cold-start comparisons complete\n')


if __name__=='__main__':
    main()
