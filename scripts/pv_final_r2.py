"""Final training chain after the round-2 PV HPOs (2026-10-19).

Phase 1 retrains the best trial of a study on its CV folds exactly as the HPO saw them (fold models, early stopping on
the fold's validation set), phase 2 trains the test model on the whole training data with a fixed epoch budget (median
best epoch of the fold models, no early stopping, so the reserved test year 2025-08..2026-07 never steers training) and
evaluates it on the test year. Models, result pickles and raw predictions (--save_model --save-predictions) land in
models/ and results/parks_pv_v1_30min/.

  make  : write the phase-1 configs (reads best params from OPTUNA_STORAGE, freezes them in best_params.json)
  run   : --group transfer|clients --gpus 0,1 [--phase 1|2|all] ; phase 2 derives the epochs from the phase-1 results

Groups: transfer = 60/30 station split (pv_transfer_split_v1.csv), folds = 40 train / 20 val parks, fixed window
2023-07..2024-07 -> 2024-08..2025-07; clients = 8 clients x 10 parks, expanding temporal folds (4-month windows);
fl80 = central model on the 80 client parks, same expanding folds (study pv_v1_hpo_fl80).
"""
import argparse, glob, json, os, pickle, queue, re, subprocess, sys, threading, time
import numpy as np, pandas as pd

FW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PV_RELEASE=v1s: same chain on the snow-fixed release parks_pv_v1s_30min (2026-10-19); names pv_fins_*, own config/log dirs
REL = os.environ.get('PV_RELEASE', 'v1')
os.makedirs(f'{FW}/configs/pv_parks/final_r2' + ('' if REL == 'v1' else 's'), exist_ok=True)
assert REL in ('v1', 'v1s'), REL
N = 'pv_fin' if REL == 'v1' else 'pv_fins'
CFG = f'{FW}/configs/pv_parks/final_r2' + ('' if REL == 'v1' else 's')
BASE = f'{FW}/configs/pv_parks/config_pv_v1_hpo_transfer.yaml'
SPLIT = os.path.expanduser('~/Work/FL_Contribution/data/pv_transfer_split_v1.csv')
LOG = f'{FW}/logs/final_r2' + ('' if REL == 'v1' else 's')
RES = f'{FW}/results/parks_pv_v1' + ('' if REL == 'v1' else 's') + '_30min'
PY = f'{FW}/frcst/bin/python'
PREFIX = 'cl_m-tft_out-96_freq-30min_'
SEED = 42
B = ['2024-08-01', '2024-12-01', '2025-04-01', '2025-08-01']          # client HPO fold boundaries
TEST = ('2025-08-01', '2026-07-31', '2026-08-03')                     # reserved test year (+ data cutoff)
CLIENTS = ['R0', 'R1', 'R2', 'R3', 'N0', 'N1', 'N2', 'N3']
GREP = r'training stations|val stations|holdout stations|Early stopping|Best val|Results saved|Model saved|Traceback|Error|  File'


def data_root():
    return os.environ.get('DATA_ROOT') or ('/mnt/nvme1' if os.path.exists('/mnt/nvme1/synthetic') else '/mnt/lambda1/nvme1')


def parks():
    return pd.read_csv(f'{data_root()}/synthetic/solar/parks_pv_v1_30min/parks.csv', dtype={'park_id': str})


def best_params(study):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t = optuna.load_study(study_name=PREFIX + study, storage=os.environ['OPTUNA_STORAGE']).best_trial
    return {'trial': t.number, 'value': t.value, **t.params}


def _sub(txt, pat, rep, flags=0):
    new, n = re.subn(pat, rep, txt, count=1, flags=flags)
    assert n == 1, f'pattern not found: {pat!r}'
    return new


def write_config(name, header, files, params, test_start, test_end, cutoff, val_files=None, holdout_files=None,
                 epochs=None):
    """One train_cl.py config from the transfer HPO config: fixed best-trial hyperparameters, strict split, no HPO folds.
    epochs=None keeps early stopping (fold run), an int disables it and trains exactly that many epochs (test run)."""
    txt = open(BASE).read()
    if REL == 'v1s':
        txt = txt.replace('parks_pv_v1_30min', 'parks_pv_v1s_30min')
        header = header.replace('# final r2', '# final r2 [release v1s, snow fix]')
    txt = re.sub(r'\A(#.*\n)+', header + '\n', txt)
    extra = ''
    if val_files:
        extra += f'  val_files: {sorted(val_files)!r}\n'
    if holdout_files:
        extra += f'  holdout_files: {sorted(holdout_files)!r}\n'
    txt = _sub(txt, r'  files: \[.*?\] *\n', f'  files: {sorted(files)!r}\n' + extra.replace('\\', r'\\'), re.S)
    txt = _sub(txt, r"  test_start: '.*'\n", f"  test_start: '{test_start}'\n")
    txt = _sub(txt, r"  test_end: '.*'\n", f"  test_end: '{test_end}'\n")
    txt = _sub(txt, r"  data_cutoff: '.*'\n", f"  data_cutoff: '{cutoff}'\n  strict_split: True\n")
    txt = _sub(txt, r'  batch_size: 64\n', f"  batch_size: {params['batch_size']}\n")
    txt = _sub(txt, r'  lr: 0.0005\n', f"  lr: {params['lr']}\n")
    txt = _sub(txt, r'    n_heads: 4\n', f"    n_heads: {params['n_heads']}\n")
    txt = _sub(txt, r'    hidden_dim: 64\n', f"    hidden_dim: {params['hidden_dim']}\n")
    txt = _sub(txt, r'    dropout: 0.1\n', f"    dropout: {params['dropout']}\n")
    txt = _sub(txt, r'    n_lstm_layers: 1\n', f"    n_lstm_layers: {params['num_lstm_layers']}\n")
    if epochs is not None:
        txt = _sub(txt, r'  epochs: 100\n', f'  epochs: {int(epochs)}\n')
        txt = _sub(txt, r'    enabled: True\n', '    enabled: False\n')
    txt = _sub(txt, r"  fold_boundaries: \[.*?\]\n", '')
    txt = _sub(txt, r"  station_folds:\n(    - \[.*?\]\n)+", '', re.S)
    txt = _sub(txt, r"  fold_mode: 'fixed_window'\n", '')
    txt = _sub(txt, r"  results_path: '.*'\n", f"  results_path: 'results/final_r2_{name}'\n")
    os.makedirs(CFG, exist_ok=True)
    open(f'{CFG}/config_{name}.yaml', 'w').write(txt)
    return f'{CFG}/config_{name}.yaml'


def make(groups=('transfer', 'clients')):
    """Phase-1 configs + frozen best params (best_params.json is extended, never truncated)."""
    sp = pd.read_csv(SPLIT, dtype={'park_id': str})
    pk = parks()
    bpf = f'{CFG}/best_params.json'
    bp = json.load(open(bpf)) if os.path.exists(bpf) else {}
    if 'fl80' in groups:
        bp['pv_v1_hpo_fl80'] = best_params('pv_v1_hpo_fl80')
        p = bp['pv_v1_hpo_fl80']; files = pk[pk.client_id != 'holdout'].park_id.tolist(); assert len(files) == 80
        for k in range(3):
            t_end = (pd.Timestamp(B[k + 1]) - pd.Timedelta(days=1)).date()
            cutoff = (pd.Timestamp(B[k + 1]) + pd.Timedelta(days=3)).date()
            write_config(f'{N}_fl80_fold{k}', f'# final r2 2026-10-19, central 80 client parks fold {k}: best trial {p["trial"]} of '
                         f'pv_v1_hpo_fl80, train issue + 48 h <= {B[k]}, early stopping + eval in [{B[k]}, {B[k + 1]}).',
                         files, p, B[k], str(t_end), str(cutoff))
    if 'transfer' not in groups and 'clients' not in groups:
        json.dump(bp, open(bpf, 'w'), indent=1); return
    bp.update({'pv_v1_hpo_transfer': best_params('pv_v1_hpo_transfer'),
               **{f'pv_v1_local_clients_{c}': best_params(f'pv_v1_local_clients_{c}') for c in CLIENTS}})
    json.dump(bp, open(bpf, 'w'), indent=1)
    train = sp[sp.role == 'train']
    p = bp['pv_v1_hpo_transfer']
    for k in range(3):
        va = train[train.fold == k].park_id.tolist(); tr = train[train.fold != k].park_id.tolist()
        write_config(f'{N}_tr_fold{k}', f'# final r2 2026-10-19, transfer fold {k}: best trial {p["trial"]} of pv_v1_hpo_transfer, '
                     f'{len(tr)} train parks, early stopping + eval on the {len(va)} fold parks (val_files) in 2024-08..2025-07.',
                     tr, p, '2024-08-01', '2025-07-31', '2025-08-03', val_files=va)
    for c in CLIENTS:
        files = pk[pk.client_id == c].park_id.tolist(); assert len(files) == 10, c
        p = bp[f'pv_v1_local_clients_{c}']
        for k in range(3):
            t_end = (pd.Timestamp(B[k + 1]) - pd.Timedelta(days=1)).date()
            cutoff = (pd.Timestamp(B[k + 1]) + pd.Timedelta(days=3)).date()
            write_config(f'{N}_cl_{c}_fold{k}', f'# final r2 2026-10-19, client {c} fold {k}: best trial {p["trial"]} of '
                         f'pv_v1_local_clients_{c}, 10 parks, train issue + 48 h <= {B[k]}, early stopping + eval in [{B[k]}, {B[k + 1]}).',
                         files, p, B[k], str(t_end), str(cutoff))
    print(len(glob.glob(f'{CFG}/config_{N}_*_fold*.yaml')), 'fold configs in', CFG)


def best_epoch(name):
    """Epoch of the minimal val_rmse of the latest result pickle of a config (= the early-stopping restore point)."""
    f = sorted(glob.glob(f'{RES}/{PREFIX}{name}_seed{SEED}_*.pkl'))
    if not f:
        raise FileNotFoundError(f'no result for {name} in {RES}')
    h = pickle.load(open(f[-1], 'rb'))['history']
    return int(np.argmin(h['val_rmse'])) + 1


def make_test(group):
    """Phase-2 configs: whole training data, fixed epochs = median best epoch of the three fold models."""
    sp = pd.read_csv(SPLIT, dtype={'park_id': str}); pk = parks()
    bp = json.load(open(f'{CFG}/best_params.json'))
    out = []
    if group == 'transfer':
        ep = [best_epoch(f'{N}_tr_fold{k}') for k in range(3)]; n = max(5, int(round(float(np.median(ep)))))
        write_config(f'{N}_tr_test', f'# final r2 2026-10-19, transfer test model: 60 train parks 2023-07..2025-07, fixed {n} epochs '
                     f'(fold best epochs {ep}, no early stopping), zero-shot eval on the 30 test parks (holdout_files) in 2025-08..2026-07.',
                     sp[sp.role == 'train'].park_id.tolist(), bp['pv_v1_hpo_transfer'], *TEST,
                     holdout_files=sp[sp.role == 'test'].park_id.tolist(), epochs=n)
        out.append((f'{N}_tr_test', ep, n))
    elif group == 'fl80':
        ep = [best_epoch(f'{N}_fl80_fold{k}') for k in range(3)]; n = max(5, int(round(float(np.median(ep)))))
        write_config(f'{N}_fl80_test', f'# final r2 2026-10-19, central 80-park test model: 2023-07..2025-07, fixed {n} epochs '
                     f'(fold best epochs {ep}, no early stopping), eval on the 80 parks in 2025-08..2026-07.',
                     pk[pk.client_id != 'holdout'].park_id.tolist(), bp['pv_v1_hpo_fl80'], *TEST, epochs=n)
        out.append((f'{N}_fl80_test', ep, n))
    else:
        for c in CLIENTS:
            ep = [best_epoch(f'{N}_cl_{c}_fold{k}') for k in range(3)]; n = max(5, int(round(float(np.median(ep)))))
            write_config(f'{N}_cl_{c}_test', f'# final r2 2026-10-19, client {c} test model: 10 parks 2023-07..2025-07, fixed {n} '
                         f'epochs (fold best epochs {ep}, no early stopping), eval in 2025-08..2026-07.',
                         pk[pk.client_id == c].park_id.tolist(), bp[f'pv_v1_local_clients_{c}'], *TEST, epochs=n)
            out.append((f'{N}_cl_{c}_test', ep, n))
    with open(f'{LOG}/epochs_{group}.json', 'w') as f:
        json.dump([{'name': a, 'fold_best_epochs': b, 'test_epochs': c} for a, b, c in out], f, indent=1)
    return [a for a, _, _ in out]


def run(names, gpus, tag, seeds=(SEED,)):
    q = queue.Queue()
    for seed in seeds:
        for n in names:
            q.put((n, seed))
    def lane(gpu):
        while True:
            try:
                name, seed = q.get_nowait()
            except queue.Empty:
                return
            args = [PY, 'train_cl.py', '-c', f'{CFG}/config_{name}', '-m', 'tft', '--save_model', '--save-predictions',
                    '--seed', str(seed)]
            tag_s = f'{tag} seed{seed}'
            env = dict(os.environ, DATA_ROOT=data_root(), CUDA_VISIBLE_DEVICES=gpu)
            t0 = time.time()
            log = f'{LOG}/{name}.log' if seed == SEED else f'{LOG}/{name}_seed{seed}.log'
            with open(log, 'w') as f:
                rc = subprocess.Popen(args, cwd=FW, env=env, stdout=f, stderr=subprocess.STDOUT).wait()
            summary = subprocess.run(['grep', '-E', GREP, log], capture_output=True, text=True).stdout
            with open(f'{LOG}/summary.txt', 'a') as s:
                s.write(f'=== {tag_s} {name} gpu{gpu} rc={rc} {time.time() - t0:.0f}s\n{summary}\n')
            print(f'{time.strftime("%F %T")} {tag_s} {name} gpu{gpu} rc={rc} {time.time() - t0:.0f}s', flush=True)
    th = [threading.Thread(target=lane, args=(g,)) for g in gpus]
    [t.start() for t in th]; [t.join() for t in th]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['make', 'run'])
    ap.add_argument('--group', choices=['transfer', 'clients', 'fl80'])
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--phase', default='all', choices=['1', '2', 'all'])
    ap.add_argument('--seeds', default=str(SEED), help='comma list; extra seeds reuse the existing configs')
    ap.add_argument('--tests-only', action='store_true', help='run the existing test configs (no fold runs, no make_test)')
    a = ap.parse_args()
    if a.mode == 'make':
        make(('fl80',) if a.group == 'fl80' else ('transfer', 'clients')); sys.exit()
    os.makedirs(LOG, exist_ok=True)
    gpus = a.gpus.split(','); seeds = tuple(int(x) for x in a.seeds.split(','))
    if a.tests_only:
        tests = {'transfer': [f'{N}_tr_test'], 'fl80': [f'{N}_fl80_test'], 'clients': [f'{N}_cl_{c}_test' for c in CLIENTS]}[a.group]
        run(tests, gpus, 'P2', seeds); print('ALL DONE', a.group, 'tests-only', seeds, flush=True); sys.exit()
    folds = {'transfer': [f'{N}_tr_fold{k}' for k in range(3)],
             'fl80': [f'{N}_fl80_fold{k}' for k in range(3)],
             'clients': [f'{N}_cl_{c}_fold{k}' for c in CLIENTS for k in range(3)]}[a.group]
    if a.phase in ('1', 'all'):
        print(len(folds), 'fold jobs', flush=True); run(folds, gpus, 'P1', seeds)
    if a.phase in ('2', 'all'):
        tests = make_test(a.group); print(len(tests), 'test jobs', tests, flush=True); run(tests, gpus, 'P2', seeds)
    print('ALL DONE', a.group, flush=True)
