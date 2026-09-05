#!/usr/bin/env python3
"""Erzeugt die Configs der finalen Testauswertung (--test-mode).

Layout:
  configs/testmode/full/   3 Arme ohne NWP-Historie, ein Modell, 12 Monate Test
  configs/testmode/step1/  HIST-Arme, Retrain 1: Test 2025-08-01 .. 2025-12-01
  configs/testmode/step2/  HIST-Arme, Retrain 2: Test 2025-12-01 .. 2026-04-01
  configs/testmode/step3/  HIST-Arme, Retrain 3: Test 2026-04-01 .. 2026-07-31

Dateinamen bleiben identisch zur Basis-Config (muessen auf _fold<N> enden,
sonst bricht die Optuna-Study-Aufloesung, s. docs/handoff_testmode.md).
Geaendert wird ausschliesslich test_start / test_end.
"""
import re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = {
    "dcrnn":            "configs/dcrnn/config_wind_dcrnn_fold1.yaml",
    "dcrnn_idw_alt":    "configs/dcrnn/config_wind_dcrnn_idw_alt_fold1.yaml",
    "dcrnn_nwp_hist":   "configs/dcrnn/config_wind_dcrnn_nwp_hist_fold1.yaml",
    "mtgnn_nwp":        "configs/mtgnn/config_wind_mtgnn_nwp_fold1.yaml",
    "mtgnn_nwp_hist":   "configs/mtgnn/config_wind_mtgnn_nwp_hist_fold1.yaml",
}
# (Zielverzeichnis, Arme, test_start, test_end, Kopfzeile)
PLAN = [
    ("full",  ["dcrnn", "dcrnn_idw_alt", "mtgnn_nwp"],
     "2025-08-01", "2026-07-31",
     "Ein Modell, Training auf allem vor 2025-08-01, Test ueber das volle Testjahr."),
    ("step1", ["dcrnn_nwp_hist", "mtgnn_nwp_hist"],
     "2025-08-01", "2025-12-01",
     "Expanding-Window-Retrain 1/3: Training < 2025-08-01, Test Aug-Nov 2025."),
    ("step2", ["dcrnn_nwp_hist", "mtgnn_nwp_hist"],
     "2025-12-01", "2026-04-01",
     "Expanding-Window-Retrain 2/3: Training < 2025-12-01, Test Dez 2025-Mrz 2026."),
    ("step3", ["dcrnn_nwp_hist", "mtgnn_nwp_hist"],
     "2026-04-01", "2026-07-31",
     "Expanding-Window-Retrain 3/3: Training < 2026-04-01, Test Apr-Jul 2026."),
    ("fullhist", ["dcrnn_nwp_hist", "mtgnn_nwp_hist"],
     "2025-08-01", "2026-07-31",
     "HIST-Arme EINMAL trainiert (Training < 2025-08-01), Test ueber das volle Testjahr "
     "(2026-09-05): Schritt-1-Checkpoint mit dieser Config auswerten, oder fixed-epochs-Lauf."),
]

HEADER = """# Finale Testauswertung im --test-mode (docs/handoff_testmode.md).
# {desc}
# train_ids = files + val_files (153), Eval zero-shot auf test_files (50).
# val_start ist im --test-mode wirkungslos; die Grenze ist test_start.
# Dateiname absichtlich identisch zur Basis-Config: die Optuna-Study-Aufloesung
# strippt nur ein _fold<N> am Zeilenende. Varianten ueber das Verzeichnis.
"""

def main() -> int:
    written = 0
    for sub, arms, t_start, t_end, desc in PLAN:
        outdir = REPO / "configs" / "testmode" / sub
        outdir.mkdir(parents=True, exist_ok=True)
        for arm in arms:
            src = REPO / BASE[arm]
            if not src.exists():
                print(f"FEHLT: {src}", file=sys.stderr); return 1
            lines = src.read_text().splitlines(keepends=True)
            seen = {"test_start": 0, "test_end": 0}
            out = []
            for ln in lines:
                m = re.match(r"^(\s*)(test_start|test_end):", ln)
                if m:
                    key = m.group(2)
                    seen[key] += 1
                    val = t_start if key == "test_start" else t_end
                    out.append(f"{m.group(1)}{key}: '{val}'\n")
                else:
                    out.append(ln)
            if seen["test_start"] != 1 or seen["test_end"] != 1:
                print(f"UNERWARTET in {src}: {seen}", file=sys.stderr); return 1
            dst = outdir / src.name
            dst.write_text(HEADER.format(desc=desc) + "".join(out))
            written += 1
            print(f"{dst.relative_to(REPO)}  test_start={t_start}  test_end={t_end}")
    print(f"-- {written} Configs geschrieben")
    return 0

if __name__ == "__main__":
    sys.exit(main())
