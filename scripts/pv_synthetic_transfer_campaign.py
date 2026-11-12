"""Run three paired seeds and generate the report after successful completion."""
import os
from pathlib import Path
import subprocess
import sys
import time
root=Path(__file__).resolve().parents[1]
os.chdir(root)
run=root/'results/parks_pv_v4_30min/synthetic_transfer_2026-11-09_v1'
while not (run/'prepared.pkl').exists():
    time.sleep(10)
jobs=[]
for seed,gpu in [(42,0),(43,1),(44,2)]:
    log=(run/f'seed{seed}.log').open('a')
    process=subprocess.Popen([sys.executable,'-u','scripts/pv_synthetic_transfer.py','--output',str(run),'--seed',str(seed),'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT)
    (run/f'seed{seed}.pid').write_text(str(process.pid))
    jobs.append((seed,process,log))
errors=[]
for seed,process,log in jobs:
    code=process.wait(); log.close()
    if code: errors.append((seed,code))
if errors:
    (run/'FAILED').write_text(str(errors)); sys.exit(1)
export=Path.home()/'Work/pv_sim2real/results/pv_synthetic_transfer_2026-11-09'
subprocess.run([sys.executable,'scripts/pv_synthetic_transfer_report.py','--run',str(run),'--export',str(export)],check=True)
(run/'COMPLETE').write_text('Three seeds and exported report completed.\n')
