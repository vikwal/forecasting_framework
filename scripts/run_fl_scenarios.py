#!/usr/bin/env python3
"""Run the FL scenario configs (configs/parks_v1/scenarios/manifest.csv) sequentially on one host.

  python scripts/run_fl_scenarios.py --scenarios scarce lopo --methods fl_fedgradient fl_fedavg --gpus 0-7
  python scripts/run_fl_scenarios.py --scenarios mask --methods local --gpus 2 --per-gpu 8

Methods: fl_fedgradient, fl_fedavg (train_fl.py), cl80 (train_cl.py), local (train_local.py on the
FedGradient config). Every run uses --save-predictions. A finished run leaves
logs/scenarios/<config>__<method>.done with the result pickle; reruns skip it (restartable).
Logs: logs/scenarios/<config>__<method>.out.
"""

import argparse
import os
import re
import subprocess
import sys
import time

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGS = os.path.join(REPO, 'logs', 'scenarios')
RESULT = re.compile(r'(results/\S+\.pkl)')


def jobs(man: pd.DataFrame, scenarios, methods):
    out = []
    for m in methods:
        src = 'fl_fedgradient' if m == 'local' else m
        sel = man[man['scenario'].isin(scenarios) & (man['method'] == src)]
        for cfg in sel['config']:
            out.append((cfg, m))
    return out


def command(cfg: str, method: str, args) -> tuple:
    py = sys.executable
    env = dict(os.environ)
    if method == 'local':
        return [py, 'train_local.py', '-c', cfg, '-m', 'tft', '--gpus', args.gpus, '--per-gpu',
                str(args.per_gpu), '--save-predictions'], env
    if args.gpus:
        env['CUDA_VISIBLE_DEVICES'] = expand(args.gpus)
    script = 'train_cl.py' if method == 'cl80' else 'train_fl.py'
    return [py, script, '-c', cfg, '-m', 'tft', '--save-predictions'], env


def expand(spec: str) -> str:
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += [str(i) for i in range(int(a), int(b) + 1)]
        else:
            out.append(part)
    return ','.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', nargs='+', required=True)
    ap.add_argument('--methods', nargs='+', required=True)
    ap.add_argument('--gpus', default=None, help="FL/CL: CUDA_VISIBLE_DEVICES; local: train_local --gpus")
    ap.add_argument('--per-gpu', type=int, default=1, help='local only')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--reverse', action='store_true', help='work the list from the end (second queue)')
    args = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(LOGS, exist_ok=True)
    man = pd.read_csv(os.path.join(REPO, 'configs', 'parks_v1', 'scenarios', 'manifest.csv'))
    todo = jobs(man, args.scenarios, args.methods)
    if args.reverse:
        todo = todo[::-1]
    print(f'{len(todo)} runs', flush=True)
    for i, (cfg, method) in enumerate(todo, 1):
        stem = f'{os.path.basename(cfg)}__{method}'
        done = os.path.join(LOGS, f'{stem}.done')
        log = os.path.join(LOGS, f'{stem}.out')
        if not os.path.exists(done) and os.path.exists(log):      # finished before a restart
            prev = RESULT.findall(open(log, errors='ignore').read())
            if prev and os.path.exists(os.path.join(REPO, prev[-1])):
                with open(done, 'w') as f:
                    f.write(prev[-1] + '\n')
        if os.path.exists(done):
            print(f'[{i}/{len(todo)}] skip {stem} (done)', flush=True)
            continue
        lock = os.path.join(LOGS, f'{stem}.running')
        if not args.dry_run:
            try:                                                     # another queue runs it
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                print(f'[{i}/{len(todo)}] skip {stem} (running elsewhere)', flush=True)
                continue
        cmd, env = command(cfg, method, args)
        print(f'[{i}/{len(todo)}] start {stem}: {" ".join(cmd)}', flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        with open(log, 'w') as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        txt = open(log, errors='ignore').read()
        hits = RESULT.findall(txt)          # Ray workers keep logging after the result line
        ok = rc == 0 and hits
        print(f'[{i}/{len(todo)}] end {stem} rc={rc} {time.time() - t0:.0f} s {hits[-1] if hits else "NO RESULT"}',
              flush=True)
        if ok:
            with open(done, 'w') as f:
                f.write(hits[-1] + '\n')
        os.remove(lock)


if __name__ == '__main__':
    main()
