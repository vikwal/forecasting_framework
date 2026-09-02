#!/usr/bin/env python3
"""Restart HPO workers between trials, so no GPU time is lost.

Each worker owns one log file. A completed trial writes exactly one
"Trial <n> done" line. When a new such line appears the worker is at most
POLL seconds into its next trial, so killing it there costs POLL seconds
instead of the hours a mid-trial kill would cost.

One worker at a time. If a relaunch cannot be verified the watcher stops
instead of walking through the rest.

Ein Eintrag darf ein Feld "replace" tragen. Dann wird an der Trial-Grenze
nicht derselbe Worker neu gestartet, sondern der darin beschriebene Ersatz.
So laesst sich ein Worker verlustfrei von einer Studie auf eine andere
umhaengen. Nach dem Tausch wird der Eintrag dauerhaft auf den Ersatz
umgestellt und das Manifest auf die Platte zurueckgeschrieben, damit ein
spaeterer Wächterstart nicht wieder den alten Stand sieht.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

REPO = Path.home() / "Work" / "forecasting_framework"
MANIFEST = Path.home() / "hpo_restart_manifest.json"
DONE = re.compile(r"Trial \d+ done")
POLL = 20
VERIFY_TIMEOUT = 180
LIVENESS_INTERVAL = 300    # s, Lebendpruefung ALLER Worker
RECHECK_AFTER = 60         # s, Nachkontrolle nach einem Neustart


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(Path.home() / "hpo_watch_restart.log", "a") as fh:
        fh.write(line + "\n")


def save_manifest(workers):
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(workers, indent=2))
    os.replace(tmp, MANIFEST)


def done_count(rel):
    p = REPO / rel
    if not p.exists():
        return None
    n = 0
    with open(p, errors="ignore") as fh:
        for line in fh:
            if DONE.search(line):
                n += 1
    return n


def sessions():
    out = {}
    for line in sh(["screen", "-ls"]).splitlines():
        m = re.match(r"\s*(\d+)\.(\S+)\s", line)
        if m:
            out[m.group(2)] = int(m.group(1))
    return out


def worker_pid(cfg, suffix, gpu=None):
    """PID des laufenden Workers, oder None. Muss sich nach dem Neustart
    vom alten PID unterscheiden, sonst haben wir nur den sterbenden
    Vorgaenger gesehen und den Neustart faelschlich als Erfolg gewertet."""
    for line in sh(["ps", "-eo", "pid,args", "--no-headers"]).splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid, args = parts
        if not re.match(r"^python3?\s+geostatistics/hpo_", args):
            continue
        if f"--config {cfg}" not in args or f"--suffix {suffix}" not in args:
            continue
        if gpu is not None and f"--gpu {gpu}" not in args:
            continue
        return int(pid)
    return None


def relaunch(w, workers=None):
    rep = w.get("replace")
    tgt = rep if rep else w
    rest = list(tgt["argv"])[3:]     # ['SCREEN','-dmS',name, ...]
    # Startskript aus /tmp gegen die dauerhafte Kopie tauschen
    rest = [a.replace("/tmp/launch_mtgnn_r4.sh",
                      str(Path.home() / "launch_mtgnn_r4.sh")) for a in rest]
    old_name = w["session"]
    new_name = tgt.get("session", old_name)
    old_pid = w.get("py_pid")
    sh(["screen", "-S", old_name, "-X", "quit"])
    time.sleep(3)
    if old_name in sessions():
        return False, f"Session {old_name} liess sich nicht beenden"
    subprocess.run(["screen", "-dmS", new_name] + rest, cwd=str(REPO))
    t0 = time.time()
    while time.time() - t0 < VERIFY_TIMEOUT:
        pid = worker_pid(tgt["config"], tgt["suffix"], tgt.get("gpu"))
        if new_name in sessions() and pid is not None and pid != old_pid:
            if rep:
                w["session"] = new_name
                w["config"] = tgt["config"]
                w["suffix"] = tgt["suffix"]
                w["gpu"] = tgt.get("gpu")
                w["log"] = tgt["log"]
                w["argv"] = tgt["argv"]
                w.pop("replace", None)
                log(f"  TAUSCH vollzogen: {old_name} -> {new_name} ({tgt['config']})")
            w["py_pid"] = pid
            if workers is not None:
                try:
                    save_manifest(workers)
                except Exception as e:      # Manifest ist Komfort, nicht kritisch
                    log(f"  WARNUNG Manifest nicht geschrieben: {e}")
            what = f"{old_name} -> {new_name}" if new_name != old_name else new_name
            return True, f"{what} laeuft nach {time.time()-t0:.0f}s (PID {old_pid} -> {pid})"
        time.sleep(5)
    return False, f"{new_name} kam innerhalb {VERIFY_TIMEOUT}s NICHT zurueck"


def sweep_liveness(workers):
    """Faengt Worker, die gestartet sind und DANACH abgestuerzt sind.

    Die Verifikation nach einem Neustart prueft nur, dass ein Prozess existiert.
    Am 2026-08-11 starben fuenf l1-Worker Minuten spaeter am Datenladen
    (PermissionError auf den neu geschriebenen Kriging-Dateien) und blieben
    stundenlang unbemerkt. Diese Runde erkennt genau das.

    Ein Eintrag mit "replace" wird hier NICHT getauscht: ein Absturz ist keine
    Trial-Grenze, der laufende Trial ginge verloren. Er wird als er selbst
    wiederbelebt, der Tausch kommt an der naechsten regulaeren Grenze.
    """
    for w in workers:
        name = w["session"]
        if name in sessions() and worker_pid(w["config"], w["suffix"], w.get("gpu")) is not None:
            continue
        log(f"LEBENDPRUEFUNG: {name} hat keinen laufenden Prozess -- wiederbeleben")
        rep = w.pop("replace", None)
        ok, msg = relaunch(w)
        if rep is not None:
            w["replace"] = rep
        log(("  wiederbelebt: " if ok else "  WIEDERBELEBUNG FEHLGESCHLAGEN: ") + msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-restarts", type=int, default=0, help="0 = alle")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-replacements", action="store_true",
                    help="nur Eintraege mit 'replace' an der Trial-Grenze anfassen; "
                         "alle uebrigen nur per Lebendpruefung ueberwachen")
    a = ap.parse_args()

    workers = json.load(open(MANIFEST))
    workers = [w for w in workers if w.get("argv") and w.get("log")]
    base = {}
    for w in workers:
        c = done_count(w["log"])
        if c is None:
            log(f"ABBRUCH: Log fehlt fuer {w['session']} ({w['log']})")
            sys.exit(1)
        base[w["session"]] = c
    log(f"Start: {len(workers)} Worker, Poll {POLL}s, max_restarts={a.max_restarts or 'alle'}, "
        f"dry_run={a.dry_run}, only_replacements={a.only_replacements}")
    for w in workers:
        mark = f"  -> TAUSCH gegen {w['replace']['session']}" if w.get("replace") else ""
        log(f"  {w['session']:24s} gpu={w['suffix']}/{w['gpu']} done_bisher={base[w['session']]}{mark}")

    todo = {w["session"]: w for w in workers
            if not a.only_replacements or w.get("replace")}
    n = 0
    last_sweep = time.time()
    while True:
        if time.time() - last_sweep >= LIVENESS_INTERVAL:
            last_sweep = time.time()
            if not a.dry_run:
                sweep_liveness(workers)
        if not todo:
            time.sleep(POLL)
            continue
        for name in sorted(todo):
            w = todo[name]
            c = done_count(w["log"])
            if c is None or c <= base[name]:
                continue
            log(f"Trial-Abschluss erkannt bei {name} ({base[name]} -> {c})")
            if a.dry_run:
                log(f"  dry-run, kein Eingriff")
                base[name] = c
                continue
            ok, msg = relaunch(w, workers)
            log(("  OK  " if ok else "  FEHLER ") + msg)
            if not ok:
                log("Wächter haelt an, damit der Fehler nicht kaskadiert. Bitte pruefen.")
                sys.exit(2)
            del todo[name]
            n += 1
            time.sleep(RECHECK_AFTER)
            if worker_pid(w["config"], w["suffix"], w.get("gpu")) is None:
                log(f"  NACHKONTROLLE {w['session']}: nach {RECHECK_AFTER}s wieder weg -- Waechter haelt an")
                sys.exit(3)
            log(f"  Nachkontrolle {w['session']}: laeuft")
            if a.max_restarts and n >= a.max_restarts:
                log(f"max_restarts={a.max_restarts} erreicht. Offen: {len(todo)}")
                return
            break
        else:
            time.sleep(POLL)
            continue
        time.sleep(2)


if __name__ == "__main__":
    main()
