#!/usr/bin/env python3
"""Haengt laufende HPO-Worker an der naechsten Trial-Grenze auf eine andere GPU.

Aufruf:  MOVES="session:neue_gpu,session:neue_gpu" [COMMIT=1] python3 move_gpu.py

Traegt einen "replace"-Block ein, der denselben Worker (gleiche Config, gleiches
Suffix, gleiches Log) auf einer anderen Karte neu startet. Verlustfrei, weil der
Wächter erst an der Trial-Grenze eingreift.
"""
import json, os, re, subprocess, sys
from pathlib import Path

HOME = Path.home()
REPO = HOME / "Work" / "forecasting_framework"
MANIFEST = HOME / "hpo_restart_manifest.json"

LAUNCHER = {"hpo_mtgnn": "launch_mtgnn_worker.sh",
            "hpo_dcrnn": "launch_dcrnn_worker.sh",
            "hpo_wavenet": "launch_wavenet_worker.sh"}


def model_of(cfg):
    part = Path(cfg).parts[-2]          # configs/<modell>/...
    return f"hpo_{part}"


def main():
    moves = [m.strip() for m in os.environ.get("MOVES", "").split(",") if m.strip()]
    if not moves:
        print("MOVES fehlt"); sys.exit(1)
    commit = bool(int(os.environ.get("COMMIT", "0")))
    ws = json.load(open(MANIFEST))
    by = {w["session"]: w for w in ws}

    # aktuelle GPU-Belegung, damit der Zielzustand sichtbar wird
    def belegung():
        c = {}
        for w in ws:
            c[str(w["gpu"])] = c.get(str(w["gpu"]), 0) + 1
        return c

    print("HPO-Worker je GPU vorher: ", dict(sorted(belegung().items())))

    for m in moves:
        session, gpu = m.split(":")
        gpu = int(gpu)
        w = by.get(session)
        if w is None:
            print(f"ABBRUCH: {session} nicht im Manifest"); sys.exit(1)
        if w.get("replace"):
            print(f"ABBRUCH: {session} hat schon einen Tauschauftrag"); sys.exit(1)
        if int(w["gpu"]) == gpu:
            print(f"uebersprungen: {session} liegt schon auf GPU {gpu}"); continue
        launcher = HOME / LAUNCHER[model_of(w["config"])]
        if not launcher.exists():
            print(f"ABBRUCH: Startskript fehlt: {launcher}"); sys.exit(1)
        # Kollision: gleiche Config + Suffix + Ziel-GPU waere nicht unterscheidbar
        for o in ws:
            if o is not w and o["config"] == w["config"] and o["suffix"] == w["suffix"] \
               and int(o["gpu"]) == gpu:
                print(f"ABBRUCH: {session} kollidiert auf GPU {gpu} mit {o['session']}"); sys.exit(1)
        w["replace"] = {"session": session, "config": w["config"], "suffix": w["suffix"],
                        "gpu": gpu, "log": w["log"],
                        "argv": ["SCREEN", "-dmS", session, "bash", str(launcher),
                                 str(REPO), w["config"], str(gpu), w["suffix"]]}
        print(f"Umzug: {session}  GPU {w['gpu']} -> {gpu}")

    nach = {}
    for w in ws:
        g = str(w["replace"]["gpu"]) if w.get("replace") else str(w["gpu"])
        nach[g] = nach.get(g, 0) + 1
    print("HPO-Worker je GPU nachher:", dict(sorted(nach.items())))

    if commit:
        tmp = MANIFEST.with_suffix(".json.new")
        tmp.write_text(json.dumps(ws, indent=2))
        os.replace(tmp, MANIFEST)
        print(f"geschrieben: {MANIFEST}")
    else:
        print("Trockenlauf, nichts geschrieben. COMMIT=1 setzen.")


if __name__ == "__main__":
    main()
