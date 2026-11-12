"""Client-level synthetic cold start with seven full-history donor clients."""
import argparse, copy, hashlib, json, logging, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pv_quantile_finetune as ft
from pv_quantile_coldstart import label_bounds
from pv_synthetic_transfer import park_id, TRAIN, EVAL, QS
from pv_quantile_metrics import metrics
DAYS=[7,14,30,60,90]

def budget_mask(lower,upper,days):
    end=pd.Timestamp('2024-08-01',tz='UTC')
    lo,hi=pd.to_datetime(lower,utc=True),pd.to_datetime(upper,utc=True)
    return np.asarray((lo>=end-pd.Timedelta(days=days))&(hi<end))

def prepare(source,out):
    with (source/'prepared.pkl').open('rb') as f:b=pickle.load(f)
    cfg=copy.deepcopy(b['cfg']); cfg['data']['files']=b['metadata']['target_available'];cfg['scaler_x']=b['scaler']
    features=ft.preprocessing.get_features(config=cfg)
    dfs=ft.preprocessing.get_data(cfg['data']['path'],cfg['data']['freq'],cfg,features)
    masks={d:{} for d in DAYS}; records=[]
    for key,it in b['target'].items():
        frame=dfs[key];ts=pd.DatetimeIndex(frame.index.get_level_values('timestamp'))
        frame=frame.loc[ts<pd.Timestamp('2025-08-01',tz=ts.tz)]
        if frame.isna().any().any():raise ValueError(f'Missing values before budget restriction: {key}')
        prepared,_=ft.preprocessing.pipeline(data=frame,config=cfg,known_cols=features['known'],observed_cols=features['observed'],static_cols=features['static'],target_col='power')
        for part in ['train','test']:
            np.testing.assert_array_equal(prepared['y_'+part],it['y_'+part])
            for feature in it['X_'+part]:np.testing.assert_array_equal(prepared['X_'+part][feature],it['X_'+part][feature])
        np.testing.assert_array_equal(prepared['index_test'],it['index_test'])
        lo,hi=label_bounds(frame,prepared['index_train'])
        for days in DAYS:
            keep=budget_mask(lo,hi,days);masks[days][key]=keep
            records.append(dict(park=park_id(key),days=days,windows=int(keep.sum()),first_target=str(lo[keep].min()) if keep.any() else None,last_target=str(hi[keep].max()) if keep.any() else None))
    b.pop('source');b['masks']=masks;b['budgets']=records
    for days in DAYS:
        for keys in b['groups'].values():assert sum(masks[days][k].sum() for k in keys)>0
    out.mkdir(parents=True,exist_ok=True)
    (out/'budgets.json').write_text(json.dumps(records,indent=2))
    with (out/'prepared.tmp').open('wb') as f:pickle.dump(b,f)
    (out/'prepared.tmp').replace(out/'prepared.pkl')
    logging.info('PREPARED %s target parks, eight clients, five budgets',len(b['target']))

def run(source,out,seed,gpu):
    torch.set_num_threads(4);ft.tools.set_seed(seed)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    device=torch.device(f'cuda:{gpu}')
    with (out/'prepared.pkl').open('rb') as f:b=pickle.load(f)
    cfg=b['cfg'];cfg['params']['random_seed']=seed
    data,groups=b['target'],b['groups'];hp=ft.hpo.get_hyperparameters(config=cfg);hp['lr']=3e-4
    checkpoint=source/f'seed{seed}/pretrained.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']
    dest=out/f'seed{seed}';dest.mkdir(exist_ok=True)
    (dest/'metadata.json').write_text(json.dumps(dict(seed=seed,days=DAYS,train=TRAIN,evaluation=EVAL,groups=groups,code_pin=b['metadata']['code_pin'],date='2026-11-09',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    full={c:ft.cat([data[k] for k in keys],'train') for c,keys in groups.items()}
    for days in DAYS:
        for client,keys in groups.items():
            case=dest/f'{days:03d}d'/client;case.mkdir(parents=True,exist_ok=True)
            if (case/'COMPLETE').exists():continue
            ownitems=[]
            for key in keys:
                keep=b['masks'][days][key];it=data[key]
                ownitems.append(dict(X_train={k:v[keep] for k,v in it['X_train'].items()},y_train=it['y_train'][keep]))
            own=ft.cat(ownitems,'train')
            logging.info('CASE seed=%s days=%s client=%s windows=%s',seed,days,client,len(own[1]))
            for variant in ['local','central','fedgradient']:
                result=case/f'{variant}_metrics.csv'
                if result.exists():continue
                name='local_'+client if variant=='local' else variant
                entries=[own] if variant=='local' else [own if c==client else full[c] for c in groups]
                val=[data[k] for k in keys] if variant=='local' else list(data.values())
                model,log,trained=ft.train(name,entries,val,state,cfg,hp,device,seed,QS,case)
                (case/f'{variant}_training.json').write_text(json.dumps(log,indent=2))
                rows=[]
                for selection,sd in [('selected',None),('best_trained',trained)]:
                    if sd is not None:model.load_state_dict(sd)
                    model.eval()
                    for key in keys:
                        it=data[key];pred=ft.base.predict(model,it['X_test'],device)
                        rows.append(dict(seed=seed,days=days,client=client,park=park_id(key),variant=variant,selection=selection,date='2026-11-09',code_pin=b['metadata']['code_pin'],**metrics(pred,it['y_test'],QS,np.asarray(it['nwp_raw_test'])>10)))
                        np.savez_compressed(case/f'{variant}_{selection}_{park_id(key)}.npz',pred=pred,true=it['y_test'],gti=it['nwp_raw_test'],quantiles=QS,index=np.asarray(it['index_test']).astype(str))
                tmp=result.with_suffix('.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(result)
                del model;torch.cuda.empty_cache()
            (case/'COMPLETE').write_text('Three variants complete\n')
    (dest/'COMPLETE').write_text('All 40 target-client/budget cases complete\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=[42,43,44]);p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    if a.prepare:prepare(a.source,a.output)
    else:
        if a.seed is None:p.error('--seed required')
        run(a.source,a.output,a.seed,a.gpu)
