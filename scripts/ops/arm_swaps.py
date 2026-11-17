#!/usr/bin/env python3
"""Traegt Tauschauftraege ins Wächter-Manifest ein.

Ein Auftrag haengt einen laufenden Worker an seiner naechsten Trial-Grenze auf
eine andere Studie um: der Spender wird beendet, der Ersatz startet auf der
angegebenen GPU. Verlustfrei, weil der Wächter nur an der Grenze eingreift.

  SPEC='[{"donor":"hpo_dcrnn_nomeas_r2","session":"hpo_mtgnn_nwphist_r11",
          "config":"configs/mtgnn/config_wind_mtgnn_nwp_hist.yaml",
          "gpu":3,"suffix":"r11"}]'  [COMMIT=1] python3 arm_swaps.py

Ohne "session" wird derselbe Worker nur auf eine andere GPU verschoben.
"""
import json, os, subprocess, sys
from pathlib import Path

HOME = Path.home()
REPO = HOME / "Work" / "forecasting_framework"
MANIFEST = HOME / "hpo_restart_manifest.json"


def model_of(cfg):
    return Path(cfg).parts[-2]          # configs/<modell>/config_...


def logrel(cfg, suffix):
    stem = Path(cfg).stem.replace("config_", "")
    return f"logs/hpo_{model_of(cfg)}_{stem}_{suffix}.log"


def main():
    spec = json.loads(os.environ.get("SPEC", "[]"))
    if not spec:
        print("SPEC fehlt"); sys.exit(1)
    commit = bool(int(os.environ.get("COMMIT", "0")))
    ws = json.load(open(MANIFEST))
    by = {w["session"]: w for w in ws}
    live = subprocess.run(["ps", "-eo", "args", "--no-headers"],
                          capture_output=True, text=True).stdout

    for job in spec:
        d = by.get(job["donor"])
        if d is None:
            print(f"ABBRUCH: Spender {job['donor']} nicht im Manifest"); sys.exit(1)
        if d.get("replace"):
            print(f"ABBRUCH: {job['donor']} hat schon einen Auftrag"); sys.exit(1)
        if f"--config {d['config']}" not in live or f"--suffix {d['suffix']}" not in live:
            print(f"ABBRUCH: Spender {job['donor']} laeuft nicht"); sys.exit(1)
        cfg = job.get("config", d["config"])
        suffix = job.get("suffix", d["suffix"])
        gpu = int(job.get("gpu", d["gpu"]))
        session = job.get("session", d["session"])
        launcher = HOME / f"launch_{model_of(cfg)}_worker.sh"
        if not launcher.exists():
            print(f"ABBRUCH: Startskript fehlt: {launcher}"); sys.exit(1)
        if not (REPO / cfg).exists():
            print(f"ABBRUCH: Config fehlt: {cfg}"); sys.exit(1)
        lr = logrel(cfg, suffix)
        if session != d["session"] and (REPO / lr).exists():
            print(f"ABBRUCH: Log {lr} existiert schon, Suffix {suffix} doppelt"); sys.exit(1)
        for o in ws:
            if o is not d and o["config"] == cfg and o["suffix"] == suffix and int(o["gpu"]) == gpu:
                print(f"ABBRUCH: kollidiert mit {o['session']}"); sys.exit(1)
        d["replace"] = {"session": session, "config": cfg, "suffix": suffix, "gpu": gpu,
                        "log": lr,
                        "argv": ["SCREEN", "-dmS", session, "bash", str(launcher),
                                 str(REPO), cfg, str(gpu), suffix]}
        print(f"{d['session']} (gpu {d['gpu']}, {Path(d['config']).stem.replace('config_wind_','')})"
              f"  ->  {session} (gpu {gpu}, {Path(cfg).stem.replace('config_wind_','')})")

    nach = {}
    for w in ws:
        c = w["replace"]["config"] if w.get("replace") else w["config"]
        k = Path(c).stem.replace("config_wind_", "")
        nach[k] = nach.get(k, 0) + 1
    print("\nWorker je Studie auf diesem Host nachher:")
    for k, v in sorted(nach.items()):
        print(f"  {k:24s} {v}")

    if commit:
        tmp = MANIFEST.with_suffix(".json.new")
        tmp.write_text(json.dumps(ws, indent=2))
        os.replace(tmp, MANIFEST)
        print(f"\ngeschrieben: {MANIFEST}")
    else:
        print("\nTrockenlauf. COMMIT=1 setzen.")


if __name__ == "__main__":
    main()
