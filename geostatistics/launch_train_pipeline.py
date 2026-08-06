#!/usr/bin/env python3
"""
launch_train_pipeline.py — Trainiert alle Varianten × 3 Folds mit HPO-Hyperparametern.

Ablauf:
  - 6 Gruppen in fester Reihenfolge (DCRNN_BASE → MTGNN_BASE → DCRNN_NWP → …)
  - Pro GPU läuft immer genau eine Gruppe
  - Innerhalb einer Gruppe laufen alle 3 Folds PARALLEL auf derselben GPU
  - Sobald alle 3 Folds einer Gruppe fertig sind, startet die nächste Gruppe auf
    der jetzt freien GPU (Work-Stealing aus einer gemeinsamen Queue)

Beispiel mit 2 GPUs:
  --gpus 1,2
  → GPU 1: DCRNN_BASE, dann DCRNN_NWP, dann DCRNN_NWP_HIST
  → GPU 2: MTGNN_BASE, dann MTGNN_NWP, dann MTGNN_NWP_HIST

Modell-Namen:
  models/wind_dcrnn_base_fold1_dcrnn_val.pt   (fold1 = Notebook fold0)
  models/wind_mtgnn_nwp_fold2_mtgnn_val.pt    (fold2 = Notebook fold1)
  …

Verwendung:
  cd /home/viktor/Work/forecasting_framework
  python geostatistics/launch_train_pipeline.py --gpus 1,2
  python geostatistics/launch_train_pipeline.py --gpus 0,1,2,3 --dry-run
  python geostatistics/launch_train_pipeline.py --gpus 1 --groups DCRNN_BASE
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Optional


# ===========================================================================
# ─── GRUPPEN-KONFIGURATION ────────────────────────────────────────────────
# ===========================================================================

WORK_DIR      = Path(__file__).parent.parent   # forecasting_framework/
VENV_ACTIVATE = WORK_DIR / "frcst/bin/activate"
MODEL_SUFFIX  = "val"   # → models/wind_dcrnn_base_fold1_dcrnn_val.pt


@dataclass
class FoldJob:
    """Ein einzelner Trainings-Run (eine Variante, ein Fold)."""
    config: str          # relativ zu WORK_DIR
    fold_label: str      # z.B. "fold1"


@dataclass
class Group:
    """Eine Variante mit allen 3 Folds."""
    name:   str          # z.B. "DCRNN_BASE"
    script: str          # "train_dcrnn" oder "train_mtgnn"
    folds:  list[FoldJob] = field(default_factory=list)
    hpo_study: str = "auto"    # "auto" | None


# ---------------------------------------------------------------------------
# ▼▼▼ REIHENFOLGE UND CONFIGS HIER DEFINIEREN ▼▼▼
# ---------------------------------------------------------------------------
#
# fold1 = Notebook fold0 (test: 2024-07-01 → 2024-11-30)
# fold2 = Notebook fold1 (test: 2024-12-01 → 2025-03-31)
# fold3 = Notebook fold2 (test: 2025-04-01 → 2025-07-31)
#
GROUPS: list[Group] = [

    Group(
        name   = "DCRNN_BASE",
        script = "train_dcrnn",
        folds  = [
            FoldJob("configs/dcrnn/config_wind_dcrnn_base_fold1.yaml", "fold1"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_base_fold2.yaml", "fold2"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_base_fold3.yaml", "fold3"),
        ],
    ),

    Group(
        name   = "MTGNN_BASE",
        script = "train_mtgnn",
        folds  = [
            FoldJob("configs/mtgnn/config_wind_mtgnn_fold1.yaml", "fold1"),
            FoldJob("configs/mtgnn/config_wind_mtgnn_fold2.yaml", "fold2"),
            FoldJob("configs/mtgnn/config_wind_mtgnn_fold3.yaml", "fold3"),
        ],
    ),

    Group(
        name   = "DCRNN_NWP",
        script = "train_dcrnn",
        folds  = [
            FoldJob("configs/dcrnn/config_wind_dcrnn_fold1.yaml", "fold1"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_fold2.yaml", "fold2"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_fold3.yaml", "fold3"),
        ],
    ),

    Group(
        name   = "MTGNN_NWP",
        script = "train_mtgnn",
        folds  = [
            FoldJob("configs/mtgnn/config_wind_mtgnn_nwp_fold1.yaml", "fold1"),
            FoldJob("configs/mtgnn/config_wind_mtgnn_nwp_fold2.yaml", "fold2"),
            FoldJob("configs/mtgnn/config_wind_mtgnn_nwp_fold3.yaml", "fold3"),
        ],
    ),

    Group(
        name   = "DCRNN_NWP_HIST",
        script = "train_dcrnn",
        folds  = [
            FoldJob("configs/dcrnn/config_wind_dcrnn_nwp_hist_fold1.yaml", "fold1"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_nwp_hist_fold2.yaml", "fold2"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_nwp_hist_fold3.yaml", "fold3"),
        ],
    ),

    Group(
        name   = "MTGNN_NWP_HIST",
        script = "train_mtgnn",
        folds  = [
            FoldJob("configs/mtgnn/config_wind_mtgnn_nwp_hist_fold1.yaml", "fold1"),
            FoldJob("configs/mtgnn/config_wind_mtgnn_nwp_hist_fold2.yaml", "fold2"),
            FoldJob("configs/mtgnn/config_wind_mtgnn_nwp_hist_fold3.yaml", "fold3"),
        ],
    ),

    # ── Ablationen des DCRNN-NWP-Arms (Variante A) ─────────────────────────
    # B = keine Nachbar-Messungen, C = zusaetzlich kein Stationsgraph.
    # Bewusst ans Ende gehaengt, damit die Reihenfolge der laufenden Kampagne
    # unveraendert bleibt; im Normalfall gezielt per --groups starten.
    # Studienaufloesung: train_dcrnn.py leitet den Optuna-Studiennamen aus dem
    # Config-Stem ohne _fold<N> ab, also
    #   config_wind_dcrnn_nomeas_fold1.yaml → cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_nomeas
    # Jede Variante braucht daher ihre eigene HPO auf der Basis-Config.
    Group(
        name   = "DCRNN_NOMEAS",
        script = "train_dcrnn",
        folds  = [
            FoldJob("configs/dcrnn/config_wind_dcrnn_nomeas_fold1.yaml", "fold1"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_nomeas_fold2.yaml", "fold2"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_nomeas_fold3.yaml", "fold3"),
        ],
    ),

    Group(
        name   = "DCRNN_NOGRAPH",
        script = "train_dcrnn",
        folds  = [
            FoldJob("configs/dcrnn/config_wind_dcrnn_nograph_fold1.yaml", "fold1"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_nograph_fold2.yaml", "fold2"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_nograph_fold3.yaml", "fold3"),
        ],
    ),

    # ── DCRNN IDW (Ablation D, R6(a) in docs/review_round2_findings.md) ────
    # Ersetzt die gelernte NWP-Attention (GATv2) durch feste inverse-
    # distanzgewichtete Aggregation; leitet sich wie NOMEAS/NOGRAPH vom
    # GRID/NWP-Arm (Variante A) ab. Eigene HPO-Studie ueber den Config-Stem
    # (config_wind_dcrnn_idw*.yaml -> ..._wind_dcrnn_idw), analog zu oben.
    Group(
        name   = "DCRNN_IDW",
        script = "train_dcrnn",
        folds  = [
            FoldJob("configs/dcrnn/config_wind_dcrnn_idw_fold1.yaml", "fold1"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_idw_fold2.yaml", "fold2"),
            FoldJob("configs/dcrnn/config_wind_dcrnn_idw_fold3.yaml", "fold3"),
        ],
    ),

    # ── WaveNet (R4: bisher fehlte hier ein Retrain-Pfad) ──────────────────
    # Exakt analog zu MTGNN_BASE/MTGNN_NWP: die regulaeren Fold-Configs (nicht
    # die stdhp-Kopien — diese Gruppen sind fuer die spaetere echte Kampagne
    # mit HPO-Params gedacht, s. geostatistics/stdrun/ fuer den stdhp-Trockenlauf).
    Group(
        name   = "WAVENET_BASE",
        script = "train_wavenet",
        folds  = [
            FoldJob("configs/wavenet/config_wind_wavenet_fold1.yaml", "fold1"),
            FoldJob("configs/wavenet/config_wind_wavenet_fold2.yaml", "fold2"),
            FoldJob("configs/wavenet/config_wind_wavenet_fold3.yaml", "fold3"),
        ],
    ),

    Group(
        name   = "WAVENET_NWP",
        script = "train_wavenet",
        folds  = [
            FoldJob("configs/wavenet/config_wind_wavenet_nwp_fold1.yaml", "fold1"),
            FoldJob("configs/wavenet/config_wind_wavenet_nwp_fold2.yaml", "fold2"),
            FoldJob("configs/wavenet/config_wind_wavenet_nwp_fold3.yaml", "fold3"),
        ],
    ),

]
# ---------------------------------------------------------------------------
# ▲▲▲ ENDE KONFIGURATION ▲▲▲
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def expected_model_path(group: Group, job: FoldJob) -> Path:
    config_stem = Path(job.config).stem.replace("config_", "")
    script_type = group.script.replace("train_", "")   # "dcrnn" or "mtgnn"
    return WORK_DIR / "models" / f"{config_stem}_{script_type}_{MODEL_SUFFIX}.pt"


def build_cmd(group: Group, job: FoldJob, gpu: int) -> list[str]:
    """Gibt die argv-Liste für den Trainings-Subprocess zurück."""
    script = f"geostatistics/{group.script}.py"
    cmd = [
        "bash", "-c",
        f"source {VENV_ACTIVATE} && "
        f"cd {WORK_DIR} && "
        f"CUDA_VISIBLE_DEVICES={gpu} python {script} "
        f"--config {job.config} "
        f"--suffix {MODEL_SUFFIX} "
        + (f"--hpo-study {group.hpo_study}" if group.hpo_study else "")
    ]
    return cmd


def run_fold(group: Group, job: FoldJob, gpu: int, dry_run: bool) -> int:
    """Startet einen Fold-Training-Prozess und wartet auf sein Ende. Gibt Exit-Code zurück."""
    model_path = expected_model_path(group, job)
    if not dry_run and model_path.exists():
        _log(f"  SKIP   {group.name}/{job.fold_label} (existiert: {model_path.name})")
        return 0

    cmd = build_cmd(group, job, gpu)

    if dry_run:
        exists = "✓ existiert" if model_path.exists() else "fehlt"
        _log(f"  [DRY] {group.name}/{job.fold_label} → GPU {gpu}  [{exists}]")
        return 0

    _log(f"  START  {group.name}/{job.fold_label} GPU={gpu}")

    # Training-Scripts schreiben ihr eigenes Log (logs/train_*.log).
    # Stdout/stderr hier unterdrücken um doppelte Logs zu vermeiden.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ},
    )

    rc = proc.wait()

    status = "OK" if rc == 0 else f"FEHLER (exit={rc})"
    _log(f"  FERTIG {group.name}/{job.fold_label} GPU={gpu}  {status}")
    return rc


def run_group(group: Group, gpu: int, dry_run: bool) -> list[int]:
    """Startet alle Folds einer Gruppe PARALLEL auf einer GPU. Wartet bis alle fertig."""
    _log(f"▶ Starte Gruppe {group.name} auf GPU {gpu}  ({len(group.folds)} Folds parallel)")

    threads: list[threading.Thread] = []
    results: list[int] = [0] * len(group.folds)

    def worker(idx: int, job: FoldJob) -> None:
        results[idx] = run_fold(group, job, gpu, dry_run)

    for i, job in enumerate(group.folds):
        t = threading.Thread(target=worker, args=(i, job), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.3)   # kurze Verzögerung damit die Prozesse nicht gleichzeitig starten

    for t in threads:
        t.join()

    failed = [j.fold_label for j, rc in zip(group.folds, results) if rc != 0]
    if failed:
        _log(f"  ✗ {group.name}: Fehler in Folds {failed}")
    else:
        _log(f"  ✓ {group.name}: alle Folds abgeschlossen")

    return results


def gpu_worker(gpu: int, queue: Queue, dry_run: bool) -> None:
    """Worker-Thread: nimmt Gruppen aus der Queue und verarbeitet sie sequenziell."""
    while True:
        try:
            group: Group = queue.get_nowait()
        except Empty:
            break
        run_group(group, gpu, dry_run)
        queue.task_done()
    _log(f"GPU {gpu}: Queue leer — fertig.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trainiert alle Varianten × 3 Folds mit HPO-Hyperparametern.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpus", "-g", required=True,
                        help="Komma-separierte GPU-Indizes, z.B. 1,2")
    parser.add_argument("--groups", default=None,
                        help="Nur diese Gruppen starten (komma-separiert), z.B. DCRNN_BASE,MTGNN_BASE")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Befehle ausgeben ohne auszuführen")
    args = parser.parse_args()

    gpus = [int(g.strip()) for g in args.gpus.split(",")]

    groups = GROUPS
    if args.groups:
        wanted = {g.strip() for g in args.groups.split(",")}
        groups = [g for g in GROUPS if g.name in wanted]
        if not groups:
            print(f"Keine Gruppen gefunden für: {wanted}")
            sys.exit(1)

    # Übersicht
    print(f"\n{'═'*65}")
    print(f"  GPUs:        {gpus}")
    print(f"  Gruppen:     {len(groups)}  ({', '.join(g.name for g in groups)})")
    print(f"  Folds/GPU:   3 parallel")
    print(f"  Modell-Suffix: _{MODEL_SUFFIX}.pt")
    print(f"  Logs:        logs/  (je Training-Script)")
    print(f"{'─'*65}")

    # Simuliere Queue-Verteilung für die Übersicht
    q_sim: list[list[str]] = [[] for _ in gpus]
    for i, g in enumerate(groups):
        q_sim[i % len(gpus)].append(g.name)
    for gpu_id, gpu_groups in zip(gpus, q_sim):
        print(f"  GPU {gpu_id}: {' → '.join(gpu_groups)}")
    print(f"{'═'*65}\n")

    if not args.dry_run:
        yn = input("Alle Trainings starten? [j/N] ").strip().lower()
        if yn not in ("j", "y", "ja", "yes"):
            print("Abgebrochen.")
            return

    # Queue befüllen (Reihenfolge entspricht GROUPS-Liste)
    queue: Queue = Queue()
    for g in groups:
        queue.put(g)

    # Einen Thread pro GPU starten
    workers: list[threading.Thread] = []
    t_start = time.time()
    for gpu in gpus:
        t = threading.Thread(target=gpu_worker, args=(gpu, queue, args.dry_run), daemon=True)
        t.start()
        workers.append(t)
        time.sleep(0.2)

    for t in workers:
        t.join()

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    _log(f"Pipeline abgeschlossen. Laufzeit: {h:02d}:{m:02d}:{s:02d}")
    _log(f"Modelle: {WORK_DIR / 'models'}/")
    _log(f"Logs:    {WORK_DIR / 'logs'}/")


if __name__ == "__main__":
    main()
