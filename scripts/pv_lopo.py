"""Real-data leave-one-park-out (LOPO) on the six Trianel meters (2026-10-20): the real-data cold-start competitor.
One central TFT trained on the meters of five parks, applied without any training to the sixth, with the round-2
hyperparameters and the two cuts of the A/B study. Early stopping as in realcentral (no val_files): on the test
window of the five training parks, never on the held-out park.

Configs = config_pv_ab_realcentral_cut{1,2}.yaml with data.files = the five other meters and holdout_files = the
held-out meter.  make -> configs/pv_parks/final_r2/config_pv_lopo_<pv>_<cut>.yaml ;  run --gpus 0,1,2,3   (3 seeds)
"""
import argparse
import glob
import os
import queue
import re
import subprocess
import sys
import threading
import time

FW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{FW}/scripts')
import pv_final_r2 as F  # noqa: E402

PARKS = ['pv1', 'pv2', 'pv3', 'pv5', 'pv6', 'pv7']
CUTS = ['cut1', 'cut2']
SEEDS = [42, 43, 44]


def make():
    for pv in PARKS:
        others = [f"'{p}_meter'" for p in PARKS if p != pv]
        for c in CUTS:
            txt = open(f'{F.CFG}/config_pv_ab_realcentral_{c}.yaml').read()
            txt = F._sub(txt, r"\n  files: \[.*?\]\n", f"\n  files: [{', '.join(others)}]\n")
            txt = F._sub(txt, r"\n  holdout_files: \[.*?\]\n", f"\n  holdout_files: ['{pv}_meter']\n")
            txt = F._sub(txt, r"results_path: '.*'", f"results_path: 'results/pv_lopo_{pv}_{c}'")
            txt = re.sub(r'\A#', f'# [LOPO 2026-10-20: trained on the five meters other than {pv}, held out {pv}_meter; otherwise the realcentral config]\n#', txt)
            open(f'{F.CFG}/config_pv_lopo_{pv}_{c}.yaml', 'w').write(txt)
    print(len(PARKS) * len(CUTS), 'configs')


def run(gpus):
    os.makedirs(F.LOG, exist_ok=True)
    q = queue.Queue()
    for s in SEEDS:
        for c in CUTS:
            for pv in PARKS:
                q.put((f'pv_lopo_{pv}_{c}', s))

    def lane(gpu):
        while True:
            try:
                name, seed = q.get_nowait()
            except queue.Empty:
                return
            if glob.glob(f'{FW}/results/parks_pv_v1_30min/*{name}_seed{seed}_*.pkl'):
                print(f'skip {name} seed{seed}: result exists', flush=True)
                continue
            args = [F.PY, 'train_cl.py', '-c', f'{F.CFG}/config_{name}', '-m', 'tft', '--save-predictions', '--seed', str(seed)]
            env = dict(os.environ, DATA_ROOT=F.data_root(), CUDA_VISIBLE_DEVICES=gpu)
            t0 = time.time()
            log = f'{F.LOG}/{name}_seed{seed}.log'
            with open(log, 'w') as f:
                rc = subprocess.Popen(args, cwd=FW, env=env, stdout=f, stderr=subprocess.STDOUT).wait()
            print(f'{time.strftime("%F %T")} LOPO {name} seed{seed} gpu{gpu} rc={rc} {time.time() - t0:.0f}s', flush=True)
    th = [threading.Thread(target=lane, args=(g,)) for g in gpus]
    [t.start() for t in th]
    [t.join() for t in th]
    print('ALL DONE LOPO', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['make', 'run'])
    ap.add_argument('--gpus', default='0')
    a = ap.parse_args()
    make() if a.mode == 'make' else run(a.gpus.split(','))
