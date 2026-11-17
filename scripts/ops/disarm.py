#!/usr/bin/env python3
"""Nimmt Worker aus dem Wächter-Manifest und entschärft offene Tauschauftraege.

  REMOVE="sessionA,sessionB"   Eintraege ganz entfernen (kein Wiederbeleben)
  DISARM="sessionC"            nur den "replace"-Block loeschen
  [COMMIT=1] python3 disarm.py
"""
import json, os, sys
from pathlib import Path

MANIFEST = Path.home() / "hpo_restart_manifest.json"


def main():
    remove = {s.strip() for s in os.environ.get("REMOVE", "").split(",") if s.strip()}
    disarm = {s.strip() for s in os.environ.get("DISARM", "").split(",") if s.strip()}
    commit = bool(int(os.environ.get("COMMIT", "0")))
    ws = json.load(open(MANIFEST))

    by = {w["session"]: w for w in ws}
    for s in remove | disarm:
        if s not in by:
            print(f"ABBRUCH: {s} nicht im Manifest. Vorhanden: {sorted(by)}")
            sys.exit(1)

    for s in disarm:
        rep = by[s].pop("replace", None)
        print(f"entschaerft: {s}" + (f" (haette {rep['session']} gestartet)" if rep else " (kein Auftrag)"))

    keep = [w for w in ws if w["session"] not in remove]
    for s in remove:
        print(f"aus dem Manifest entfernt: {s}")

    print(f"\n{len(ws)} -> {len(keep)} Eintraege, offene Auftraege: "
          f"{[w['session']+' -> '+w['replace']['session'] for w in keep if w.get('replace')] or 'keine'}")

    if commit:
        tmp = MANIFEST.with_suffix(".json.new")
        tmp.write_text(json.dumps(keep, indent=2))
        os.replace(tmp, MANIFEST)
        print(f"geschrieben: {MANIFEST}")
    else:
        print("Trockenlauf. COMMIT=1 setzen.")


if __name__ == "__main__":
    main()
