#!/usr/bin/env python3
"""
launch_test_pipeline.py — Retrain + Eval der Modelle im TEST-Modus (Hold-out test_files).

Test-Szenario (Fold 1): Kontext = files + val_files (153 Stationen), Ziel = test_files (50),
Zeitfenster aus den test/-Configs (test_start/test_end). Beide Phasen nutzen --test-mode.

Modelle werden mit denselben HPO-Hyperparametern neu trainiert (--hpo-study auto trifft via
Config-Stem die bestehende Study) und mit --suffix test getrennt von den val-Modellen abgelegt.

Verwendung:
  cd /home/viktor/Work/forecasting_framework
  # Phase 1 — Retrain (DCRNN auf GPU 1, MTGNN auf GPU 2):
  python geostatistics/launch_test_pipeline.py --phase train --gpus 1,2
  # Phase 2 — Eval (erst nachdem die Modelle fertig trainiert sind):
  python geostatistics/launch_test_pipeline.py --phase eval  --gpus 1,2
  # Vorschau:
  python geostatistics/launch_test_pipeline.py --phase train --gpus 1,2 --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

WORK_DIR      = Path(__file__).parent.parent
VENV_ACTIVATE = WORK_DIR / "frcst/bin/activate"


@dataclass
class Spec:
    group:        str   # GPU-Zuweisung (DCRNN → GPU a, MTGNN → GPU b)
    family:       str   # "dcrnn" | "mtgnn"
    config:       str   # test/-Config (Stem trifft via --hpo-study auto die bestehende Study)
    model_stem:   str   # Modell-Dateiname-Stem (Output von train_*, Input -m für eval)
    raw_out_name: str   # Output-Stem: data/test_results/{raw_out_name}.csv + raw_preds/..._raw.parquet


SPECS: list[Spec] = [
    # ── DCRNN (BASE / GRID / GRID+HIST) ───────────────────────────────────────
    Spec("DCRNN", "dcrnn", "configs/dcrnn/test/config_wind_dcrnn_base_fold1.yaml",
         "wind_dcrnn_base_fold1_dcrnn_test",     "dcrnn_wind_dcrnn_base_test_fold0"),
    Spec("DCRNN", "dcrnn", "configs/dcrnn/test/config_wind_dcrnn_fold1.yaml",
         "wind_dcrnn_fold1_dcrnn_test",          "dcrnn_wind_dcrnn_test_fold0"),
    Spec("DCRNN", "dcrnn", "configs/dcrnn/test/config_wind_dcrnn_nwp_hist_fold1.yaml",
         "wind_dcrnn_nwp_hist_fold1_dcrnn_test", "dcrnn_wind_dcrnn_nwp_hist_test_fold0"),
    # ── MTGNN (BASE / GRID / GRID+HIST) ───────────────────────────────────────
    Spec("MTGNN", "mtgnn", "configs/mtgnn/test/config_wind_mtgnn_fold1.yaml",
         "wind_mtgnn_fold1_mtgnn_test",          "mtgnn_wind_mtgnn_test_fold0"),
    Spec("MTGNN", "mtgnn", "configs/mtgnn/test/config_wind_mtgnn_nwp_fold1.yaml",
         "wind_mtgnn_nwp_fold1_mtgnn_test",      "mtgnn_wind_mtgnn_nwp_test_fold0"),
    Spec("MTGNN", "mtgnn", "configs/mtgnn/test/config_wind_mtgnn_nwp_hist_fold1.yaml",
         "wind_mtgnn_nwp_hist_fold1_mtgnn_test", "mtgnn_wind_mtgnn_nwp_hist_test_fold0"),
]


def assign_gpus(gpus: list[int]) -> dict[str, int]:
    groups: list[str] = []
    for s in SPECS:
        if s.group not in groups:
            groups.append(s.group)
    return {g: gpus[i % len(gpus)] for i, g in enumerate(groups)}


def model_path(spec: Spec) -> Path:
    return WORK_DIR / "models" / f"{spec.model_stem}.pt"


def raw_parquet(spec: Spec) -> Path:
    return WORK_DIR / "data" / "raw_preds" / f"{spec.raw_out_name}_raw.parquet"


def is_done(spec: Spec, phase: str) -> bool:
    return model_path(spec).exists() if phase == "train" else raw_parquet(spec).exists()


def session_name(spec: Spec, phase: str) -> str:
    return f"{phase}test_{spec.raw_out_name}"


def build_inner_cmd(spec: Spec, phase: str, gpu: int) -> str:
    if phase == "train":
        script = f"geostatistics/train_{spec.family}.py"
        log    = f"logs/train_{spec.family}_{spec.model_stem}.log"
        cmd = (f"CUDA_VISIBLE_DEVICES={gpu} python {script} "
               f"--config {spec.config} --suffix test --hpo-study auto --test-mode")
    else:
        script = f"geostatistics/get_test_results_{spec.family}.py"
        log    = f"logs/eval_{spec.raw_out_name}.log"
        cmd = (f"CUDA_VISIBLE_DEVICES={gpu} python {script} "
               f"-m {spec.model_stem} -c {spec.config} --hpo-study auto --test-mode "
               f"--raw-out-name {spec.raw_out_name}")
    # Session bleibt bei Fehler offen (exec bash) zum Debuggen.
    return f"bash -c 'source {VENV_ACTIVATE} && cd {WORK_DIR} && {cmd} >> {log} 2>&1 || exec bash'"


def main() -> None:
    ap = argparse.ArgumentParser(description="Test-Modus Retrain + Eval (Hold-out test_files).")
    ap.add_argument("--phase", required=True, choices=["train", "eval"])
    ap.add_argument("--gpus", default="1,2", help="Komma-separierte GPU-IDs, default 1,2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    gpu_of = assign_gpus(gpus)

    print(f"\n{'═'*64}")
    print(f"  TEST-Pipeline — Phase: {args.phase.upper()}")
    print(f"  GPU-Zuweisung: {gpu_of}")
    print(f"{'─'*64}")
    pending: list[Spec] = []
    for s in SPECS:
        done = is_done(s, args.phase)
        mark = "✓ fertig" if done else "○ offen"
        print(f"  [{s.group:5s} GPU{gpu_of[s.group]}] {s.raw_out_name:38s} {mark}")
        if not done:
            pending.append(s)
    print(f"{'═'*64}\n")

    if not pending:
        print("Nichts zu tun — alle Outputs vorhanden.")
        return

    if not args.dry_run:
        yn = input(f"{len(pending)} {args.phase}-Jobs starten? [j/N] ").strip().lower()
        if yn not in ("j", "y", "ja", "yes"):
            print("Abgebrochen.")
            return

    for s in pending:
        gpu = gpu_of[s.group]
        inner = build_inner_cmd(s, args.phase, gpu)
        sess  = session_name(s, args.phase)
        if args.dry_run:
            print(f"  [DRY] screen -dmS {sess}")
            print(f"        {inner[:130]}…")
            continue
        rc = subprocess.run(["screen", "-dmS", sess, "bash", "-c", inner], capture_output=True)
        if rc.returncode != 0:
            print(f"  [FEHLER] {sess}: {rc.stderr.decode()}")
        else:
            print(f"  [OK]  screen -r {sess}")

    if not args.dry_run:
        print("\nÜbersicht:  screen -ls")
        print(f"Logs:       logs/{'train_*' if args.phase=='train' else 'eval_*test*'}.log")


if __name__ == "__main__":
    main()
