"""Supervise the three cold-start seeds and export the complete comparison."""
from pathlib import Path
import os,subprocess,sys,time
root=Path(__file__).resolve().parents[1];os.chdir(root)
source=root/'results/parks_pv_v4_30min/synthetic_transfer_2026-11-09_v1'
run=root/'results/parks_pv_v4_30min/synthetic_coldstart_2026-11-09_v1'
if not (run/'prepared.pkl').exists():raise RuntimeError('Prepare data before launch')
jobs=[]
for seed,gpu in [(42,0),(43,1),(44,2)]:
    log=(run/f'seed{seed}.log').open('a')
    p=subprocess.Popen([sys.executable,'-u','scripts/pv_synthetic_coldstart.py','--source',str(source),'--output',str(run),'--seed',str(seed),'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT)
    (run/f'seed{seed}.pid').write_text(str(p.pid));jobs.append((seed,p,log))
errors=[]
for seed,p,log in jobs:
    code=p.wait();log.close()
    if code:errors.append((seed,code))
if errors:(run/'FAILED').write_text(str(errors));sys.exit(1)
export=Path.home()/'Work/pv_sim2real/results/pv_synthetic_coldstart_2026-11-09'
subprocess.run([sys.executable,'scripts/pv_synthetic_coldstart_report.py','--run',str(run),'--export',str(export)],check=True)
(run/'COMPLETE').write_text('360 fits and report completed\n')
