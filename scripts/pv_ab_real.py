"""Scenario A/B on the real Trianel meters with the round-2 (transfer) hyperparameters (2026-10-19).

A  = zeroshot    : 90 synthetic parks -> meters (holdout), early stopping on the synthetic parks' test window
A6 = twincentral : 6 synthetic twins  -> meters (holdout)
B  = realcentral : 6 meters           -> meters, early stopping on the meters' test window (upper bound for B, as m4 chain)
Cuts as in the m4 chain (strict split): S1 train ..2023-11-30 / test 2023-12-01..2024-02-29, S2 ..2024-02-29 / 2024-03-01..2024-05-31.
Configs = m4 chain configs with model.* set to the best trial of pv_v1_hpo_transfer; 3 seeds.
  make -> configs/pv_parks/final_r2/config_pv_ab_<variant>_<cut>.yaml ;  run --gpus 0,1,2,3
"""
import argparse, json, os, re, sys
FW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{FW}/scripts')
import pv_final_r2 as F
M4 = os.path.expanduser('~/Work/FL_Contribution/reports/pv_m4_chain_2026-10-17/configs')
VARIANTS = ['zeroshot', 'twincentral', 'realcentral'] if F.REL == 'v1' else ['zeroshot', 'twincentral']   # real B does not depend on the release
AB = 'pv_ab' if F.REL == 'v1' else 'pv_abs'
CUTS = ['cut1', 'cut2']
SEEDS = [42, 43, 44]


def make():
    p = json.load(open(f'{F.FW}/configs/pv_parks/final_r2/best_params.json'))['pv_v1_hpo_transfer']   # HPO is not rerun for v1s
    for v in VARIANTS:
        for c in CUTS:
            txt = open(f'{M4}/config_pv_m4_{v}_{c}.yaml').read()
            txt = re.sub(r'\A#.*\n', f'# [A/B real 2026-10-19: model.* = best trial {p["trial"]} of pv_v1_hpo_transfer (batch {p["batch_size"]}, '
                         f'lr {p["lr"]:.2e}, {p["n_heads"]} heads, hidden {p["hidden_dim"]}, dropout {p["dropout"]:.3f}, {p["num_lstm_layers"]} LSTM)]\n', txt)
            txt = F._sub(txt, r'\n  batch_size: 64\n', f"\n  batch_size: {p['batch_size']}\n")
            txt = F._sub(txt, r'\n  lr: 0.0005\n', f"\n  lr: {p['lr']}\n")
            txt = F._sub(txt, r'    n_heads: 4\n', f"    n_heads: {p['n_heads']}\n")
            txt = F._sub(txt, r'    hidden_dim: 64\n', f"    hidden_dim: {p['hidden_dim']}\n")
            txt = F._sub(txt, r'    dropout: 0.1\n', f"    dropout: {p['dropout']}\n")
            txt = F._sub(txt, r'    n_lstm_layers: 1\n', f"    n_lstm_layers: {p['num_lstm_layers']}\n")
            txt = F._sub(txt, r"results_path: '.*'", f"results_path: 'results/{AB}_{v}_{c}'")
            if F.REL == 'v1s':
                txt = txt.replace('parks_pv_v1_30min', 'parks_pv_v1s_30min')
                txt = F._sub(txt, r"  twin_path: (.*)trianel_twins_30min", r"  twin_path: \1trianel_twins_s_30min")
                txt = re.sub(r'\A#', '# [release v1s, snow fix: data.path + twin_path switched, meters unchanged]\n#', txt)
            open(f'{F.CFG}/config_{AB}_{v}_{c}.yaml', 'w').write(txt)
    print(len(VARIANTS) * len(CUTS), 'configs')


def run(gpus):
    import queue, subprocess, threading, time
    os.makedirs(F.LOG, exist_ok=True)
    q = queue.Queue()
    for s in SEEDS:                      # seed-major: every variant gets a first seed early
        for c in CUTS:
            for v in VARIANTS:
                q.put((f'{AB}_{v}_{c}', s))
    def lane(gpu):
        while True:
            try:
                name, seed = q.get_nowait()
            except queue.Empty:
                return
            args = [F.PY, 'train_cl.py', '-c', f'{F.CFG}/config_{name}', '-m', 'tft', '--save_model', '--save-predictions', '--seed', str(seed)]
            env = dict(os.environ, DATA_ROOT=F.data_root(), CUDA_VISIBLE_DEVICES=gpu)
            t0 = time.time(); log = f'{F.LOG}/{name}_seed{seed}.log'
            with open(log, 'w') as f:
                rc = subprocess.Popen(args, cwd=FW, env=env, stdout=f, stderr=subprocess.STDOUT).wait()
            summary = subprocess.run(['grep', '-E', F.GREP, log], capture_output=True, text=True).stdout
            with open(f'{F.LOG}/summary_ab.txt', 'a') as s:
                s.write(f'=== AB {name} seed{seed} gpu{gpu} rc={rc} {time.time() - t0:.0f}s\n{summary}\n')
            print(f'{time.strftime("%F %T")} AB {name} seed{seed} gpu{gpu} rc={rc} {time.time() - t0:.0f}s', flush=True)
    th = [threading.Thread(target=lane, args=(g,)) for g in gpus]
    [t.start() for t in th]; [t.join() for t in th]
    print('ALL DONE AB', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['make', 'run']); ap.add_argument('--gpus', default='0')
    a = ap.parse_args()
    make() if a.mode == 'make' else run(a.gpus.split(','))
