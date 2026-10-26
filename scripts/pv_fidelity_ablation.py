"""Generator-fidelity ablation at forecast level (2026-10-19): the A zero-shot model (90 synthetic parks -> six meters,
round-2 hyperparameters, early stopping on the synthetic parks' test window) retrained on fleets with one generator
component removed (pv_sim2real/ablation/gen_fidelity_fleets.py):

  rawcams : uncorrected CAMS irradiance        nosnow : no snow loss        naive : PVGIS-type chain on ERA5

Configs = configs/pv_parks/final_r2/config_pv_ab_zeroshot_cut{1,2}.yaml with data.path switched to the variant
release and results_path renamed; 3 seeds.  make -> configs ;  run --gpus 0,1,2,3   (results as pv_abl_<variant>_<cut>)
"""
import argparse
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

VARIANTS = ['rawcams', 'nosnow', 'naive']
CUTS = ['cut1', 'cut2']
SEEDS = [42, 43, 44]


def make():
    for v in VARIANTS:
        for c in CUTS:
            txt = open(f'{F.CFG}/config_pv_ab_zeroshot_{c}.yaml').read()
            n = txt.count('parks_pv_v1_30min')
            assert n == 1, n
            txt = txt.replace('parks_pv_v1_30min', f'parks_pv_v1_30min_abl_{v}')
            txt = F._sub(txt, r"results_path: '.*'", f"results_path: 'results/pv_abl_{v}_{c}'")
            txt = re.sub(r'\A#', f'# [generator-fidelity ablation 2026-10-19: fleet variant {v}, otherwise the A/B zero-shot config]\n#', txt)
            open(f'{F.CFG}/config_pv_abl_{v}_{c}.yaml', 'w').write(txt)
    print(len(VARIANTS) * len(CUTS), 'configs')


def run(gpus):
    os.makedirs(F.LOG, exist_ok=True)
    q = queue.Queue()
    for s in SEEDS:
        for c in CUTS:
            for v in VARIANTS:
                q.put((f'pv_abl_{v}_{c}', s))

    def lane(gpu):
        while True:
            try:
                name, seed = q.get_nowait()
            except queue.Empty:
                return
            args = [F.PY, 'train_cl.py', '-c', f'{F.CFG}/config_{name}', '-m', 'tft', '--save_model', '--save-predictions', '--seed', str(seed)]
            env = dict(os.environ, DATA_ROOT=F.data_root(), CUDA_VISIBLE_DEVICES=gpu)
            t0 = time.time()
            log = f'{F.LOG}/{name}_seed{seed}.log'
            with open(log, 'w') as f:
                rc = subprocess.Popen(args, cwd=FW, env=env, stdout=f, stderr=subprocess.STDOUT).wait()
            print(f'{time.strftime("%F %T")} ABL {name} seed{seed} gpu{gpu} rc={rc} {time.time() - t0:.0f}s', flush=True)
    th = [threading.Thread(target=lane, args=(g,)) for g in gpus]
    [t.start() for t in th]
    [t.join() for t in th]
    print('ALL DONE ABL', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['make', 'run'])
    ap.add_argument('--gpus', default='0')
    a = ap.parse_args()
    make() if a.mode == 'make' else run(a.gpus.split(','))
