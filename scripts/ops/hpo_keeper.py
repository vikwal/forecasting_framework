#!/usr/bin/env python3
"""Haelt eine Soll-Besetzung je Studie, indem er Plaetze nachbesetzt.

Anders als hpo_watch_restart.py, der Worker an Trial-Grenzen neu startet, sorgt
dieser Halter nur dafuer, dass je Studie so viele Worker laufen wie im Sollplan
steht. Er wartet also darauf, dass die alten Worker mit ihrem r-Suffix nach dem
laufenden Trial aussteigen (siehe .hpo_stop_<suffix>), und besetzt die frei
werdenden Plaetze mit neuen Workern.

Soll steht in ~/hpo_keeper_plan.json:

  {"max_per_gpu": 3,
   "gpus": [0,1,2,3,4,5,6,7],
   "studies": [
     {"model": "mtgnn", "config": "configs/mtgnn/config_wind_mtgnn_nwp.yaml",
      "target": 3, "prefix": "n"}
   ]}

`target` ist die Zahl der Worker DIESES Hosts fuer diese Config, alte wie neue.
Suffixe werden als <prefix><n> durchgezaehlt, bis ein freier Logname gefunden ist.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path.home() / "Work" / "forecasting_framework"
PLAN = Path.home() / "hpo_keeper_plan.json"
LOG = Path.home() / "hpo_keeper.log"
POLL = 120
START_GAP = 45          # s zwischen zwei Starts, damit Cache-Schreiber sich nicht ueberholen


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def running():
    """[(config, gpu, suffix)] der laufenden HPO-Worker."""
    out = []
    for line in sh(["ps", "-eo", "args", "--no-headers"]).splitlines():
        if not re.match(r"^python3?\s+geostatistics/hpo_(dcrnn|mtgnn|wavenet)\.py", line):
            continue
        c = re.search(r"--config (\S+)", line)
        g = re.search(r"--gpu (\S+)", line)
        s = re.search(r"--suffix (\S+)", line)
        if c and g and s:
            out.append((c.group(1), int(g.group(1)), s.group(1)))
    return out


def stem(cfg):
    return Path(cfg).stem.replace("config_", "")


def free_suffix(model, cfg, prefix):
    for i in range(1, 100):
        suf = f"{prefix}{i}"
        if not (REPO / f"logs/hpo_{model}_{stem(cfg)}_{suf}.log").exists():
            return suf
    return None


def start(model, cfg, gpu, suf):
    name = f"hpo_{model}_{stem(cfg)}_{suf}"
    launcher = Path.home() / f"launch_{model}_worker.sh"
    if not launcher.exists():
        log(f"  FEHLER Startskript fehlt: {launcher}")
        return False
    subprocess.run(["screen", "-dmS", name, "bash", str(launcher),
                    str(REPO), cfg, str(gpu), suf], cwd=str(REPO))
    time.sleep(8)
    ok = any(c == cfg and g == gpu and s == suf for c, g, s in running())
    log(f"  {'gestartet' if ok else 'START FEHLGESCHLAGEN'}: {name} gpu={gpu}")
    return ok


def main():
    once = "--once" in sys.argv
    dry = "--dry-run" in sys.argv
    plan = json.load(open(PLAN))
    gpus = plan["gpus"]
    cap = int(plan.get("max_per_gpu", 3))
    studies = plan["studies"]

    log(f"Halter startet: {len(studies)} Studien, GPUs {gpus}, max {cap}/GPU, "
        f"Poll {POLL}s, dry_run={dry}")
    for st in studies:
        log(f"  Soll {st['target']}x  {stem(st['config'])}")

    while True:
        run = running()
        per_gpu = {g: 0 for g in gpus}
        for _, g, _ in run:
            if g in per_gpu:
                per_gpu[g] += 1
        started = 0
        for st in studies:
            cfg, model, target = st["config"], st["model"], int(st["target"])
            have = sum(1 for c, _, _ in run if c == cfg)
            if have >= target:
                continue
            # freieste GPU unterhalb der Kappe
            cands = sorted((n, g) for g, n in per_gpu.items() if n < cap)
            if not cands:
                log(f"{stem(cfg)}: {have}/{target}, aber alle GPUs bei {cap} — warte")
                continue
            n, gpu = cands[0]
            suf = free_suffix(model, cfg, st.get("prefix", "n"))
            if suf is None:
                log(f"{stem(cfg)}: kein freies Suffix")
                continue
            log(f"{stem(cfg)}: {have}/{target} — besetze GPU {gpu} (dort {n} Worker) mit {suf}")
            if not dry:
                if start(model, cfg, gpu, suf):
                    per_gpu[gpu] += 1
                    run.append((cfg, gpu, suf))
                    started += 1
                    time.sleep(START_GAP)
            else:
                per_gpu[gpu] += 1
                run.append((cfg, gpu, suf))
                started += 1
            break        # ein Start je Durchgang, dann Lage neu bewerten
        if once:
            log(f"--once: {started} Start(s), Ende.")
            return
        if started == 0:
            time.sleep(POLL)


if __name__ == "__main__":
    main()
