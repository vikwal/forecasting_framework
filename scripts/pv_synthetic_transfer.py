"""Synthetic source parks -> client parks, paired quantile transfer study."""
import argparse
import copy
import hashlib
import json
import logging
import pickle
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import tools, preprocessing, models, hpo
from utils.fedgradient import save_checkpoint
from utils.data_cache import _fit_global_scaler_x
import pv_conformal as base
from pv_meter_finetune import cat
from pv_quantile_finetune import train, snapshot
from pv_quantile_metrics import metrics

TRAIN = ('2023-07-24', '2024-07-31 23:30')
EVAL = ('2024-08-01', '2025-07-31 23:30')
QS = [.05, .1, .25, .5, .75, .9, .95]


def park_id(key):
    return Path(key).name.removeprefix('synth_').removesuffix('.csv')


def prepare(out):
    cfg = base.load_cfg('configs/pv_parks/config_pv_v4_central_split1_nohist')
    clients = base.load_cfg('configs/pv_parks/config_pv_v4_fl_clients_split1_nohist')['fl']['clients']
    source = cfg['data']['holdout_files']
    target = cfg['data']['files']
    assert len(source) == 60 and len(target) == 166 and not set(source) & set(target)
    assert len(clients) == 8 and sum(map(len, clients.values())) == len(target)
    assert set(sum(clients.values(), [])) == set(target)
    cfg['data'].update(train_start=TRAIN[0], train_end=TRAIN[1], test_start=EVAL[0],
                       test_end=EVAL[1], data_cutoff=EVAL[1], strict_split=True, holdout_files=[])
    cfg['model']['tft']['quantiles'] = QS
    cfg['model']['early_stopping']['monitor'] = 'val_pinball'
    cfg['model']['epochs'] = 100
    assert cfg['params']['observed_features'] == []
    features = preprocessing.get_features(config=cfg)
    def load(ids):
        c = copy.deepcopy(cfg)
        c['data']['files'] = ids
        dfs = preprocessing.get_data(c['data']['path'], c['data']['freq'], c, features)
        # Explicitly exclude all target timestamps in the reserved additional year.
        for k, df in dfs.items():
            t = pd.DatetimeIndex(df.index.get_level_values('timestamp'))
            dfs[k] = df.loc[t < pd.Timestamp('2025-08-01', tz=t.tz)]
        return dfs, c
    dfs, scfg = load(source)
    sx = _fit_global_scaler_x(dfs, scfg, logging.getLogger(__name__),
                             fit_until='2024-08-01', restrict_train_end=True)
    src = base.windows(dfs, scfg, features, sx, TRAIN, EVAL)
    del dfs
    dfs, tcfg = load(target)
    dst = base.windows(dfs, tcfg, features, sx, TRAIN, EVAL)
    del dfs
    groups = {name: sorted(k for k in dst if park_id(k) in ids) for name, ids in clients.items()}
    assert all(groups.values()) and sum(map(len, groups.values())) == len(dst)
    assert not {park_id(k) for k in src} & {park_id(k) for k in dst}
    for it in list(src.values()) + list(dst.values()):
        idx = pd.DatetimeIndex(it['index_test'])
        assert idx.min() >= pd.Timestamp(EVAL[0], tz=idx.tz)
        assert idx.max() + pd.Timedelta(minutes=30*95) <= pd.Timestamp(EVAL[1], tz=idx.tz)
    cfg['model']['feature_dim'] = tools.get_feature_dim(X=next(iter(src.values()))['X_train'])
    cfg['data']['files'] = source
    meta = dict(train=TRAIN, evaluation=EVAL, quantiles=QS, source_requested=source,
                target_requested=target, source_available=[park_id(k) for k in src],
                target_available=[park_id(k) for k in dst], groups=groups,
                source_excluded=sorted(set(source)-{park_id(k) for k in src}),
                target_excluded=sorted(set(target)-{park_id(k) for k in dst}),
                sizes={k:{s:len(it['y_'+s]) for s in ['train','test']} for k,it in {**src,**dst}.items()},
                code_pin=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()+' + dirty',
                date='2026-11-09', lr=3e-4, local_batch=64, shared_batch_per_client=11,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.mkdir(parents=True, exist_ok=True)
    with (out/'prepared.tmp').open('wb') as f:
        pickle.dump(dict(cfg=cfg, source=src, target=dst, groups=groups, scaler=sx, metadata=meta),f)
    (out/'prepared.tmp').replace(out/'prepared.pkl')
    (out/'metadata.json').write_text(json.dumps(meta, indent=2))
    logging.info('Prepared %s source and %s target parks',len(src),len(dst))


def run(out, seed, gpu):
    torch.set_num_threads(4)
    tools.set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    device = torch.device(f'cuda:{gpu}')
    with (out/'prepared.pkl').open('rb') as f:
        bundle = pickle.load(f)
    cfg, src, dst, groups = [bundle[k] for k in ['cfg','source','target','groups']]
    dest = out/f'seed{seed}'
    dest.mkdir(exist_ok=True)
    cfg['params']['random_seed'] = seed
    (dest/'run_metadata.json').write_text(json.dumps(dict(seed=seed, gpu=gpu,
        source_metadata_sha256=hashlib.sha256((out/'metadata.json').read_bytes()).hexdigest(),
        code_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            [__file__, 'scripts/pv_quantile_finetune.py', 'utils/fedgradient.py', 'scripts/pv_quantile_metrics.py']}),indent=2))
    hp = hpo.get_hyperparameters(config=cfg)
    hp['lr'] = 3e-4
    pre = dest/'pretrained.pt'
    if pre.exists():
        state = torch.load(pre,map_location='cpu',weights_only=False)['state_dict']
    else:
        initial = models.get_model(config=cfg,hyperparameters=hp)
        state = snapshot(initial)
        del initial
        model, log, _ = train('local_pretrain', [cat(list(src.values()),'train')],
                             list(src.values()),state,cfg,hp,device,seed,QS,dest)
        state = snapshot(model)
        save_checkpoint(str(pre),dict(state_dict=state,scaler=bundle['scaler'],config=cfg))
        (dest/'pretraining.json').write_text(json.dumps(log,indent=2))
        del model
    del src
    rows = []
    def evaluate(model,variant,keys):
        model.eval()
        for key in keys:
            it = dst[key]
            pred = base.predict(model,it['X_test'],device)
            client = next(c for c,ks in groups.items() if key in ks)
            rows.append(dict(seed=seed,variant=variant,park=park_id(key),client=client,
                         date='2026-11-09',code_pin=bundle['metadata']['code_pin'],
                         **metrics(pred,it['y_test'],QS,np.asarray(it['nwp_raw_test'])>10)))
            np.savez_compressed(dest/f'{variant}_{park_id(key)}.npz',pred=pred,true=it['y_test'],
                                gti=it['nwp_raw_test'],quantiles=QS,index=np.asarray(it['index_test']).astype(str))
        pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
    model = models.get_model(config=cfg,hyperparameters=hp).to(device)
    model.load_state_dict(state)
    evaluate(model,'zeroshot',sorted(dst))
    del model
    plans = [('local_'+c,[ks],ks) for c,ks in groups.items()]
    plans += [(name,list(groups.values()),sorted(dst)) for name in ['central','fedgradient']]
    logs=[]
    for name, parts, keys in plans:
        checkpoint = dest/f'{name}_best.pt'
        if checkpoint.exists():
            model = models.get_model(config=cfg,hyperparameters=hp).to(device)
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'])
        else:
            model, log, _ = train(name,[cat([dst[k] for k in ks],'train') for ks in parts],
                                [dst[k] for k in keys],state,cfg,hp,device,seed,QS,dest)
            logs.append(log)
            (dest/(name+'_training.json')).write_text(json.dumps(log,indent=2))
        evaluate(model,'local' if name.startswith('local_') else name,keys)
        del model
        torch.cuda.empty_cache()
    assert len(rows) == 4*len(dst)
    (dest/'COMPLETE').write_text('All four variants complete.\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--seed',type=int,choices=[42,43,44])
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    if a.prepare: prepare(a.output)
    else:
        if a.seed is None: p.error('--seed required for training')
        run(a.output,a.seed,a.gpu)
