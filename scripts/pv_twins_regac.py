"""A6 on register-only twins (D2, 2026-10-20): the twincentral runs of the A/B study with the twins rebuilt without any
AC limit from the meter (trianel_twins_regac_30min: DC/AC from MaStR, else 1.10). Everything else as config_pv_ab_twincentral.
  make -> configs/pv_parks/final_r2/config_pv_abr_twincentral_<cut>.yaml ;  run --gpus 1,2,3   (3 seeds)
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

CUTS = ['cut1', 'cut2']
SEEDS = [42, 43, 44]


def make():
    for c in CUTS:
        txt = open(f'{F.CFG}/config_pv_ab_twincentral_{c}.yaml').read()
        n = txt.count('trianel_twins_30min')
        assert n == 2, n                      # twin_path and meter_path
        txt = F._sub(txt, r"  twin_path: (.*)trianel_twins_30min", r"  twin_path: \1trianel_twins_regac_30min")
        txt = F._sub(txt, r"results_path: '.*'", f"results_path: 'results/pv_abr_twincentral_{c}'")
        txt = re.sub(r'\A#', '# [A6 on register-only twins 2026-10-20 (D2): twin_path -> trianel_twins_regac_30min, meters unchanged]\n#', txt)
        open(f'{F.CFG}/config_pv_abr_twincentral_{c}.yaml', 'w').write(txt)
    print(len(CUTS), 'configs')


def run(gpus):
    os.makedirs(F.LOG, exist_ok=True)
    q = queue.Queue()
    for s in SEEDS:
        for c in CUTS:
            q.put((f'pv_abr_twincentral_{c}', s))

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
            print(f'{time.strftime("%F %T")} ABR {name} seed{seed} gpu{gpu} rc={rc} {time.time() - t0:.0f}s', flush=True)
    th = [threading.Thread(target=lane, args=(g,)) for g in gpus]
    [t.start() for t in th]
    [t.join() for t in th]
    print('ALL DONE ABR', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['make', 'run'])
    ap.add_argument('--gpus', default='1')
    a = ap.parse_args()
    make() if a.mode == 'make' else run(a.gpus.split(','))
