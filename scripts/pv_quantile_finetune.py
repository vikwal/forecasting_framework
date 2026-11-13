#!/usr/bin/env python3
"""Zero-shot/local/central/FedGradient comparison of frozen-start PV quantile TFTs.

Train July 2023--January 2024; select and evaluate February--May 2024.
Central and federated variants consume identical unions of client mini-batches.
Clip once on the server after aggregation, rather than clipping clients separately.
Dropout remains enabled, so stochastic trajectories need not be bitwise identical.
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
from utils import fedgradient as fg, tools, models, hpo, finetune, preprocessing
import pv_conformal as base
from pv_meter_finetune import cat
from pv_quantile_metrics import pinball, metrics


class ScheduledClient(fg.GradientClient):
    """Preserve all windows and optionally use the union of client schedules."""
    def __init__(self, *args, sizes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sizes = sizes

    def prepare_round(self, seed, weights=None):
        super().prepare_round(seed, weights)
        sizes = self.sizes or [self.n]
        schedules = [fg.epoch_batches(n, self.batch_size, seed, drop_last=False) for n in sizes]
        offsets = np.cumsum([0] + sizes[:-1])
        self.batches = [torch.cat([b[i]+int(off) for b, off in zip(schedules, offsets) if i<len(b)]).to(self.device)
                        for i in range(max(map(len, schedules)))]
        return len(self.batches)


def snapshot(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def train(name, data, val_items, state, cfg, hp, device, seed, qs, out, factory=None,
          lr=3e-4, batch_size=None, weight_decay=0.0, save=True, val_fn=None):
    """batch_size overrides the default batch (64 local, 11 per client shared); save=False writes no history or
    checkpoints (HPO trials); val_fn(pred (n, horizon, Q), y) replaces the pinball loss as early-stopping metric (e.g.
    MSE for a point model). The defaults reproduce the earlier studies."""
    factory = factory or (lambda: models.get_model(config=cfg, hyperparameters=hp))
    tools.set_seed(seed)
    criterion, mi = tools.make_criterion(cfg)
    model = factory().to(device)
    model.load_state_dict(state)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    clip = cfg['model']['tft']['clipnorm']
    def clip_global(opt, args, kwargs):
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.register_step_pre_hook(clip_global)
    if name == 'central':
        sizes = [len(y) for _, y in data]
        X = {k:np.concatenate([x[k] for x,_ in data]) for k in data[0][0]}
        entries = [(X,np.concatenate([y for _,y in data]))]
    else:
        sizes, entries = None, data
    bs = batch_size or (64 if name.startswith('local_') else 11)
    workers=[]
    for i,(X,y) in enumerate(entries):
        m=factory()
        m.load_state_dict(state)
        workers.append(ScheduledClient(m,X,y,bs,criterion,device,is_tft=True,clipnorm=None,
            median_idx=mi,client_id=i,dropout_seed_offset=i,sizes=sizes))
    def dispatch(method, indices, arg):
        return [workers[i].prepare_round(*arg) if method=='prepare_round' else
                workers[i].step(arg) if method=='step' else workers[i].round_metrics() for i in indices]
    def validate():
        # In a deployed federation, clients return only sums and counts of losses.
        model.eval()
        weighted,n=0.,0
        for it in val_items:
            pred=base.predict(model,it['X_test'],device)
            y=it['y_test']
            weighted += (val_fn(pred,y) if val_fn else pinball(pred,y,qs))*y.size
            n += y.size
        return weighted/n
    initial=validate()
    if not np.isfinite(initial):
        raise ValueError('Nonfinite initial validation loss')
    best,best_ep,best_sd=initial,0,snapshot(model)
    trained_best,trained_ep,trained_sd=float('inf'),0,None
    wait=0
    history=[dict(epoch=0,val_pinball=initial,server_steps=0)]
    comm=fg.CommStats(range(len(workers)))
    es=cfg['model']['early_stopping']
    for ep in range(1,cfg['model']['epochs']+1):
        _,steps=fg.run_round(dispatch,list(range(len(workers))),model,optimizer,fg.round_seed(seed,ep-1),weighting='batch_size',comm=comm)
        loss=validate()
        if not np.isfinite(loss):
            raise ValueError('Nonfinite validation loss')
        if loss<trained_best:
            trained_best,trained_ep,trained_sd=loss,ep,snapshot(model)
        if loss<best-es['min_delta']:
            best,best_ep,best_sd=loss,ep,snapshot(model)
            wait=0
        else:
            wait+=1
        history.append(dict(epoch=ep,val_pinball=loss,server_steps=steps))
        logging.info('%s seed %s epoch %s pinball %.6f best %s',name,seed,ep,loss,best_ep)
        if wait>=es['patience']:
            break
    if save:
        pd.DataFrame(history).to_csv(out/f'{name}_history.csv',index=False)
        fg.save_checkpoint(str(out/f'{name}_best.pt'),{'state_dict':best_sd,'quantiles':qs,'best_epoch':best_ep})
        fg.save_checkpoint(str(out/f'{name}_best_trained.pt'),
            {'state_dict':trained_sd,'quantiles':qs,'best_epoch':trained_ep})
    model.load_state_dict(best_sd)
    return model,dict(variant=name,best_epoch=best_ep,epochs=ep,initial_pinball=initial,
        selected_pinball=best,best_trained_pinball=trained_best,best_trained_epoch=trained_ep,
        **comm.to_dict()),trained_sd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    out=Path(a.output)/f'seed{a.seed}'
    out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; refusing accidental CPU training')
    device=torch.device(f'cuda:{a.gpu}')
    cfg=base.load_cfg(base.DATA_CFG)
    qcfg=base.load_cfg(base.Q_CFG)
    cfg['model']=copy.deepcopy(qcfg['model'])
    cfg['model']['early_stopping']['monitor']='val_pinball'
    qs=cfg['model']['tft']['quantiles']
    model_path=base.Q_MODEL.format(seed=a.seed)
    state,(sx,_)=finetune.load_init(model_path)
    features=preprocessing.get_features(config=cfg)
    dfs=preprocessing.get_data(data_dir=cfg['data']['path'],config=cfg,freq=cfg['data']['freq'],features=features,files_key='files')
    data=base.windows(dfs,cfg,features,sx,base.CAL,base.EV)
    keys=sorted(data)
    assert len(keys)==6
    cfg['model']['feature_dim']=tools.get_feature_dim(X=data[keys[0]]['X_train'])
    hp=hpo.get_hyperparameters(config=cfg)
    hp['lr']=3e-4
    pin=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()+' + dirty'
    files=[__file__,'scripts/pv_quantile_metrics.py','utils/fedgradient.py',base.DATA_CFG+'.yaml',base.Q_CFG+'.yaml',model_path]
    metadata=dict(seed=a.seed,quantiles=qs,calibration=base.CAL,evaluation=base.EV,lr=3e-4,
       early_stopping=cfg['model']['early_stopping'],max_epochs=cfg['model']['epochs'],
       central_fed_client_batch=11,local_batch=64,clip='global norm 7 after gradient aggregation',
       date='2026-11-09',code_pin=pin,
       sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
       sizes={k:{s:len(data[k]['y_'+s]) for s in ['train','test']} for k in keys})
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rows,logs=[],[]
    def evaluate(model,variant,selected,eval_keys):
        model.eval()
        for key in eval_keys:
            it=data[key]
            pred=base.predict(model,it['X_test'],device)
            park=key.replace('synth_','').replace('_meter.csv','')
            row=metrics(pred,it['y_test'],qs,np.asarray(it['nwp_raw_test'])>10)
            rows.append(dict(variant=variant,selection=selected,park=park,seed=a.seed,
                             date='2026-11-09',code_pin=pin,**row))
            np.savez_compressed(out/f'{variant}_{selected}_{park}.npz',pred=pred,true=it['y_test'],
                gti=it['nwp_raw_test'],quantiles=qs,index=np.asarray(it['index_test']).astype(str))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    model=models.get_model(config=cfg,hyperparameters=hp).to(device)
    model.load_state_dict(state)
    evaluate(model,'zeroshot','selected',keys)
    del model
    plans=[('local_'+k,[k],[k]) for k in keys]+[('central',keys,keys),('fedgradient',keys,keys)]
    for name,train_keys,eval_keys in plans:
        model,log,trained_sd=train(name,[cat([data[k]],'train') for k in train_keys],
               [data[k] for k in train_keys],state,cfg,hp,device,a.seed,qs,out)
        variant='local' if name.startswith('local_') else name
        evaluate(model,variant,'selected',eval_keys)
        model.load_state_dict(trained_sd)
        evaluate(model,variant,'best_trained',eval_keys)
        logs.append(log)
        (out/'training.json').write_text(json.dumps(logs,indent=2))
        del model
        torch.cuda.empty_cache()
    (out/'COMPLETE').write_text('All variants completed.\n')


if __name__=='__main__':
    main()
