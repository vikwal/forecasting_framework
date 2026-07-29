#!/usr/bin/env python3
"""
launch_reference_eval.py — Führt evaluate_reference.py für alle 3 Folds sequenziell aus.

Verwendet die NWP-Configs (mit ecmwf_path) damit ICON-D2 und ECMWF beide erzeugt werden.

Verwendung:
  cd /home/viktor/Work/forecasting_framework
  python geostatistics/launch_reference_eval.py
  python geostatistics/launch_reference_eval.py --dry-run
  python geostatistics/launch_reference_eval.py --folds 0,2       # nur bestimmte Folds
  python geostatistics/launch_reference_eval.py --test-mode       # test_files als Val-Set
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORK_DIR      = Path(__file__).parent.parent
VENV_ACTIVATE = WORK_DIR / "frcst/bin/activate"
SCRIPT        = "geostatistics/evaluate_reference.py"

# NWP-Configs für alle 3 Folds (haben ecmwf_path → produziert icon_d2 + ecmwf)
FOLD_CONFIGS = {
    0: "configs/mtgnn/config_wind_mtgnn_nwp_fold1.yaml",
    1: "configs/mtgnn/config_wind_mtgnn_nwp_fold2.yaml",
    2: "configs/mtgnn/config_wind_mtgnn_nwp_fold3.yaml",
}

# Erwartete Output-Dateien pro Fold
def expected_outputs(fold_idx: int) -> list[Path]:
    return [
        WORK_DIR / "data" / "raw_preds" / f"icon_d2_fold{fold_idx}_raw.parquet",
        WORK_DIR / "data" / "test_results" / f"icon_d2_fold{fold_idx}.csv",
        WORK_DIR / "data" / "raw_preds" / f"ecmwf_fold{fold_idx}_raw.parquet",
        WORK_DIR / "data" / "test_results" / f"ecmwf_fold{fold_idx}.csv",
    ]


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def run_fold(fold_idx: int, config: str, test_mode: bool, dry_run: bool) -> int:
    outputs   = expected_outputs(fold_idx)
    done      = [p.exists() for p in outputs]
    done_names = [p.name for p, d in zip(outputs, done) if d]
    miss_names = [p.name for p, d in zip(outputs, done) if not d]

    if all(done):
        _log(f"  SKIP   fold{fold_idx} — alle Outputs vorhanden")
        return 0

    if done_names:
        _log(f"  PARTIAL fold{fold_idx} — vorhanden: {done_names}")
        _log(f"           fehlend:    {miss_names}")

    cmd_parts = (
        f"source {VENV_ACTIVATE} && "
        f"cd {WORK_DIR} && "
        f"python {SCRIPT} "
        f"-c {config} "
        f"--fold-idx {fold_idx}"
        + (" --test-mode" if test_mode else "")
    )

    if dry_run:
        _log(f"  [DRY]  fold{fold_idx}  config={Path(config).name}")
        return 0

    log_dir  = WORK_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"ref_eval_fold{fold_idx}.log"

    _log(f"  START  fold{fold_idx}  config={Path(config).name}  log={log_path.name}")
    t0 = time.time()

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            ["bash", "-c", cmd_parts],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        rc = proc.wait()

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    status = "OK" if rc == 0 else f"FEHLER (exit={rc})"
    _log(f"  FERTIG fold{fold_idx}  {status}  ({m:02d}:{s:02d})")
    if rc != 0:
        _log(f"         → Log: {log_path}")
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(description="NWP-Baselines für alle 3 Folds auswerten")
    parser.add_argument("--folds",     default="0,1,2",
                        help="Komma-separierte Fold-Indizes (0/1/2), default: 0,1,2")
    parser.add_argument("--dry-run",   action="store_true", help="Nur anzeigen, nicht ausführen")
    parser.add_argument("--test-mode", action="store_true",
                        help="test_files als Val-Set verwenden (wie in get_test_results_mtgnn.py)")
    args = parser.parse_args()

    folds = [int(f.strip()) for f in args.folds.split(",")]
    unknown = [f for f in folds if f not in FOLD_CONFIGS]
    if unknown:
        print(f"Unbekannte Fold-Indizes: {unknown}  (gültig: 0, 1, 2)")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  NWP Baseline Evaluation")
    print(f"  Folds:    {folds}")
    print(f"  Script:   {SCRIPT}")
    print(f"  Outputs:  data/raw_preds/  +  data/test_results/")
    print(f"{'─'*60}")
    for f in folds:
        outputs = expected_outputs(f)
        done    = all(p.exists() for p in outputs)
        status  = "✓ fertig" if done else "○ fehlt"
        print(f"  Fold {f}: {FOLD_CONFIGS[f]}  [{status}]")
    print(f"{'═'*60}\n")

    if not args.dry_run:
        yn = input("Starten? [j/N] ").strip().lower()
        if yn not in ("j", "y", "ja", "yes"):
            print("Abgebrochen.")
            return

    t_total = time.time()
    errors  = []

    for fold_idx in folds:
        rc = run_fold(
            fold_idx  = fold_idx,
            config    = FOLD_CONFIGS[fold_idx],
            test_mode = args.test_mode,
            dry_run   = args.dry_run,
        )
        if rc != 0:
            errors.append(fold_idx)

    elapsed = time.time() - t_total
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    print()
    if errors:
        _log(f"Abgeschlossen mit Fehlern in Folds {errors}.  Laufzeit: {h:02d}:{m:02d}:{s:02d}")
        sys.exit(1)
    else:
        _log(f"Alle Folds erfolgreich.  Laufzeit: {h:02d}:{m:02d}:{s:02d}")
        _log(f"Outputs: {WORK_DIR}/data/raw_preds/  +  {WORK_DIR}/data/test_results/")


if __name__ == "__main__":
    main()
