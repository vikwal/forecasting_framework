#!/usr/bin/env python3
"""
launch_eval_pipeline.py — Startet alle Evaluierungs-Jobs parallel in Screen-Sessions.

GPU-Zuweisung:
  Modell-Gruppen werden round-robin auf die verfügbaren GPUs verteilt.
  Alle Jobs einer Gruppe (z.B. alle DCRNN-Folds) laufen auf derselben GPU.

Beispiel:
  --gpus 1,2   →  DCRNN auf GPU 1, MTGNN auf GPU 2
  --gpus 0,1,2 →  DCRNN auf GPU 0, MTGNN auf GPU 1, WaveNet auf GPU 2

Verwendung:
  cd /home/viktor/Work/forecasting_framework
  python geostatistics/launch_eval_pipeline.py --gpus 1,2
  python geostatistics/launch_eval_pipeline.py --gpus 1,2 --dry-run   # nur ausgeben, nicht starten

Sessions überwachen:
  screen -ls                    # alle Sessions
  screen -r eval_dcrnn_fold0    # an Session anhängen
  screen -X -S eval_dcrnn_fold0 quit   # Session beenden
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ===========================================================================
# ─── JOBS KONFIGURIEREN ────────────────────────────────────────────────────
#
#  Jeder Job beschreibt einen Evaluierungslauf:
#    group       : Modellgruppe — bestimmt die GPU-Zuweisung
#    script      : get_test_results_dcrnn / get_test_results_mtgnn
#    model_name  : Pattern für -m (Substring des .pt-Dateinamens)
#    config      : Pfad zur YAML-Config (relativ zu WORK_DIR)
#    hpo_study   : "auto" | Pfad zur .db | None
#    raw_out_name: Stem des Output-Parquets → data/raw_preds/{raw_out_name}_raw.parquet
#                  Muss dem MODEL_META-Schlüssel in fold_evaluation.ipynb entsprechen,
#                  also z.B. "dcrnn_wind_dcrnn_fold0" für fold_evaluation-Zelle 8.
#    extra_args  : Beliebige zusätzliche CLI-Argumente als Liste
#
# ─── DCRNN-Folds (fold1/2/3 in Config = fold0/1/2 im Notebook) ─────────────
#
#  Mehrere HPO-Trial-Files pro Fold existieren — hier den besten Trial eintragen.
#  Naming convention: _hpo_dcrnn_trial{N}_fold{K}.pt
#  → besten Trial aus Optuna-Study ablesen und hier einsetzen.
# ===========================================================================

WORK_DIR = Path(__file__).parent.parent  # forecasting_framework/


@dataclass
class Job:
    group:        str            # Modellgruppe (GPU-Zuweisung)
    script:       str            # "get_test_results_dcrnn" oder "get_test_results_mtgnn"
    model_name:   str            # -m Argument
    config:       str            # -c Argument (relativ zu WORK_DIR)
    raw_out_name: str            # Stem für data/raw_preds/{raw_out_name}_raw.parquet
    hpo_study:    Optional[str] = "auto"
    extra_args:   list[str]     = field(default_factory=list)

    @property
    def session_name(self) -> str:
        return f"eval_{self.raw_out_name}"


# ---------------------------------------------------------------------------
# ▼▼▼ HIER JOBS DEFINIEREN ▼▼▼
# ---------------------------------------------------------------------------

JOBS: list[Job] = [

    # ==========================================================================
    # ── DCRNN ─────────────────────────────────────────────────────────────────
    # Config fold1 → Notebook fold0, fold2 → fold1, fold3 → fold2
    #
    # HINWEIS ZU MODELL-FILES:
    #   Die _hpo_dcrnn_trial*_fold*.pt Files in models/ wurden während des HPO
    #   gespeichert. Da alle 3 DCRNN-Studien (GRID, BASE, NWP+HIST) denselben
    #   Dateinamen-Prefix "_hpo_dcrnn_" nutzen, lässt sich nicht mehr sicher
    #   zuordnen welche Files zu welcher Variante gehören.
    #   → Empfehlung: Beste Trials mit retrain_from_hpo.py nachtrainieren und
    #     mit variantenspezifischen Namen speichern (z.B. wind_dcrnn_fold0.pt).
    #   Bis dahin: beste verfügbare Checkpoints je Fold eintragen.
    # ==========================================================================

    # ── DCRNN GRID (nwp_nodes=true, next_n_icond2=4) ──────────────────────
    # Best trial in study: #243 — Checkpoint fehlt, daher nächstbester nehmen.
    # Verfügbare fold0-Trials: 202, 200, 157, 10, 6  (höchste Nummer zuerst)
    # Verfügbare fold1-Trials: 594, 400, 217, 150, 6
    # Verfügbare fold2-Trials: 399, 224, 223, 169, 153, 145
    Job(
        group       = "DCRNN_GRID",
        script      = "get_test_results_dcrnn",
        model_name  = "_hpo_dcrnn_trial202_fold0",
        config      = "configs/dcrnn/config_wind_dcrnn_fold1.yaml",
        raw_out_name= "dcrnn_wind_dcrnn_fold0",
    ),
    Job(
        group       = "DCRNN_GRID",
        script      = "get_test_results_dcrnn",
        model_name  = "_hpo_dcrnn_trial594_fold1",
        config      = "configs/dcrnn/config_wind_dcrnn_fold2.yaml",
        raw_out_name= "dcrnn_wind_dcrnn_fold1",
    ),
    Job(
        group       = "DCRNN_GRID",
        script      = "get_test_results_dcrnn",
        model_name  = "_hpo_dcrnn_trial399_fold2",
        config      = "configs/dcrnn/config_wind_dcrnn_fold3.yaml",
        raw_out_name= "dcrnn_wind_dcrnn_fold2",
    ),

    # ── DCRNN BASE (nwp_nodes=false, next_n_icond2=1, best trial=#111) ────
    # Checkpoint für trial#111 fehlt — nach Retrain hier ersetzen.
    Job(
        group       = "DCRNN_BASE",
        script      = "get_test_results_dcrnn",
        model_name  = "_hpo_dcrnn_trial157_fold0",   # ← nach Retrain: wind_dcrnn_base_fold0
        config      = "configs/dcrnn/config_wind_dcrnn_base_fold1.yaml",
        raw_out_name= "dcrnn_wind_dcrnn_base_fold0",
    ),
    Job(
        group       = "DCRNN_BASE",
        script      = "get_test_results_dcrnn",
        model_name  = "_hpo_dcrnn_trial150_fold1",   # ← nach Retrain: wind_dcrnn_base_fold1
        config      = "configs/dcrnn/config_wind_dcrnn_base_fold2.yaml",
        raw_out_name= "dcrnn_wind_dcrnn_base_fold1",
    ),
    Job(
        group       = "DCRNN_BASE",
        script      = "get_test_results_dcrnn",
        model_name  = "_hpo_dcrnn_trial145_fold2",   # ← nach Retrain: wind_dcrnn_base_fold2
        config      = "configs/dcrnn/config_wind_dcrnn_base_fold3.yaml",
        raw_out_name= "dcrnn_wind_dcrnn_base_fold2",
    ),

    # ── DCRNN NWP+HIST (hist_wind_available=true, best trial=#90) ──────────
    # Checkpoint für trial#90 fehlt — nach Retrain hier ersetzen.
    Job(
        group       = "DCRNN_NWP_HIST",
        script      = "get_test_results_dcrnn",
        model_name  = "_hpo_dcrnn_trial200_fold0",   # ← nach Retrain: wind_dcrnn_nwp_hist_fold0
        config      = "configs/dcrnn/config_wind_dcrnn_nwp_hist_fold1.yaml",
        raw_out_name= "dcrnn_wind_dcrnn_nwp_hist_fold0",
    ),
    Job(
        group       = "DCRNN_NWP_HIST",
        script      = "get_test_results_dcrnn",
        model_name  = "_hpo_dcrnn_trial217_fold1",   # ← nach Retrain: wind_dcrnn_nwp_hist_fold1
        config      = "configs/dcrnn/config_wind_dcrnn_nwp_hist_fold2.yaml",
        raw_out_name= "dcrnn_wind_dcrnn_nwp_hist_fold1",
    ),
    Job(
        group       = "DCRNN_NWP_HIST",
        script      = "get_test_results_dcrnn",
        model_name  = "_hpo_dcrnn_trial169_fold2",   # ← nach Retrain: wind_dcrnn_nwp_hist_fold2
        config      = "configs/dcrnn/config_wind_dcrnn_nwp_hist_fold3.yaml",
        raw_out_name= "dcrnn_wind_dcrnn_nwp_hist_fold2",
    ),

    # ==========================================================================
    # ── MTGNN ─────────────────────────────────────────────────────────────────
    # Fold-Configs neu erstellt (config_wind_mtgnn_fold1/2/3.yaml etc.)
    # Checkpoints: trial71(fold0), trial73(fold1), trial88(fold2) sind jeweils
    # die einzigen verfügbaren — nach Retrain mit best-trial-Files ersetzen.
    # ==========================================================================

    # ── MTGNN BASE (nwp_nodes=false, hist_wind_available=false) ───────────
    Job(
        group       = "MTGNN_BASE",
        script      = "get_test_results_mtgnn",
        model_name  = "_hpo_mtgnn_trial71_fold0",    # ← nach Retrain: wind_mtgnn_fold0
        config      = "configs/mtgnn/config_wind_mtgnn_fold1.yaml",
        raw_out_name= "mtgnn_wind_mtgnn_fold0",
    ),
    Job(
        group       = "MTGNN_BASE",
        script      = "get_test_results_mtgnn",
        model_name  = "_hpo_mtgnn_trial73_fold1",    # ← nach Retrain: wind_mtgnn_fold1
        config      = "configs/mtgnn/config_wind_mtgnn_fold2.yaml",
        raw_out_name= "mtgnn_wind_mtgnn_fold1",
    ),
    Job(
        group       = "MTGNN_BASE",
        script      = "get_test_results_mtgnn",
        model_name  = "_hpo_mtgnn_trial88_fold2",    # ← nach Retrain: wind_mtgnn_fold2
        config      = "configs/mtgnn/config_wind_mtgnn_fold3.yaml",
        raw_out_name= "mtgnn_wind_mtgnn_fold2",
    ),

    # ── MTGNN NWP (nwp_nodes=true, hist_wind_available=false, best=#61) ───
    # Kein fold-spezifischer Checkpoint vorhanden → nach Retrain eintragen.
    Job(
        group       = "MTGNN_NWP",
        script      = "get_test_results_mtgnn",
        model_name  = "_hpo_mtgnn_trial89_fold0",    # ← nach Retrain: wind_mtgnn_nwp_fold0
        config      = "configs/mtgnn/config_wind_mtgnn_nwp_fold1.yaml",
        raw_out_name= "mtgnn_wind_mtgnn_nwp_fold0",
    ),
    Job(
        group       = "MTGNN_NWP",
        script      = "get_test_results_mtgnn",
        model_name  = "_hpo_mtgnn_trial73_fold1",    # ← nach Retrain: wind_mtgnn_nwp_fold1
        config      = "configs/mtgnn/config_wind_mtgnn_nwp_fold2.yaml",
        raw_out_name= "mtgnn_wind_mtgnn_nwp_fold1",
    ),
    Job(
        group       = "MTGNN_NWP",
        script      = "get_test_results_mtgnn",
        model_name  = "_hpo_mtgnn_trial88_fold2",    # ← nach Retrain: wind_mtgnn_nwp_fold2
        config      = "configs/mtgnn/config_wind_mtgnn_nwp_fold3.yaml",
        raw_out_name= "mtgnn_wind_mtgnn_nwp_fold2",
    ),

    # ── MTGNN NWP+HIST (nwp_nodes=true, hist_wind_available=true, best=#51)
    # Config neu erstellt: config_wind_mtgnn_nwp_hist_fold1/2/3.yaml
    # Kein spezifischer Checkpoint vorhanden → nach Retrain eintragen.
    Job(
        group       = "MTGNN_NWP_HIST",
        script      = "get_test_results_mtgnn",
        model_name  = "_hpo_mtgnn_trial71_fold0",    # ← nach Retrain: wind_mtgnn_nwp_hist_fold0
        config      = "configs/mtgnn/config_wind_mtgnn_nwp_hist_fold1.yaml",
        raw_out_name= "mtgnn_wind_mtgnn_nwp_hist_fold0",
    ),
    Job(
        group       = "MTGNN_NWP_HIST",
        script      = "get_test_results_mtgnn",
        model_name  = "_hpo_mtgnn_trial73_fold1",    # ← nach Retrain: wind_mtgnn_nwp_hist_fold1
        config      = "configs/mtgnn/config_wind_mtgnn_nwp_hist_fold2.yaml",
        raw_out_name= "mtgnn_wind_mtgnn_nwp_hist_fold1",
    ),
    Job(
        group       = "MTGNN_NWP_HIST",
        script      = "get_test_results_mtgnn",
        model_name  = "_hpo_mtgnn_trial88_fold2",    # ← nach Retrain: wind_mtgnn_nwp_hist_fold2
        config      = "configs/mtgnn/config_wind_mtgnn_nwp_hist_fold3.yaml",
        raw_out_name= "mtgnn_wind_mtgnn_nwp_hist_fold2",
    ),

]

# ---------------------------------------------------------------------------
# ▲▲▲ JOBS ENDE ▲▲▲
# ---------------------------------------------------------------------------


VENV_ACTIVATE = WORK_DIR / "frcst/bin/activate"


def assign_gpus(jobs: list[Job], gpus: list[int]) -> dict[str, int]:
    """Weist jeder Modell-Gruppe eine GPU zu (round-robin nach Gruppen-Reihenfolge)."""
    groups: list[str] = []
    for j in jobs:
        if j.group not in groups:
            groups.append(j.group)
    return {g: gpus[i % len(gpus)] for i, g in enumerate(groups)}


def build_command(job: Job, gpu: int) -> str:
    """Baut den Shell-Befehl für eine Screen-Session."""
    script = f"geostatistics/{job.script}.py"
    parts  = [
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "python", script,
        "-m", job.model_name,
        "-c", job.config,
        "--raw-out-name", job.raw_out_name,
    ]
    if job.hpo_study:
        parts += ["--hpo-study", job.hpo_study]
    parts += job.extra_args

    cmd = " ".join(parts)
    log = f"logs/eval_{job.raw_out_name}.log"
    return f"bash -c 'source {VENV_ACTIVATE} && cd {WORK_DIR} && {cmd} 2>&1 | tee {log}; echo EXIT $?; exec bash'"


def launch_session(job: Job, cmd: str, dry_run: bool) -> None:
    session = job.session_name
    screen_cmd = ["screen", "-dmS", session, "bash", "-c", cmd]
    if dry_run:
        print(f"  [DRY] screen -dmS {session}")
        print(f"        {cmd[:120]}…" if len(cmd) > 120 else f"        {cmd}")
    else:
        result = subprocess.run(screen_cmd, capture_output=True)
        if result.returncode != 0:
            print(f"  [FEHLER] screen-Start für {session} fehlgeschlagen: {result.stderr.decode()}")
        else:
            print(f"  [OK]  screen -r {session}")


def check_screen_available() -> bool:
    try:
        subprocess.run(["screen", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Startet alle Evaluierungs-Jobs parallel in Screen-Sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpus", "-g", required=True,
        help="Komma-separierte GPU-Indizes, z.B. 1,2 oder 0,1,2",
    )
    parser.add_argument(
        "--groups", default=None,
        help="Nur diese Modell-Gruppen starten (komma-separiert), z.B. DCRNN oder DCRNN,MTGNN",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Befehle ausgeben ohne auszuführen",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Sekunden zwischen Screen-Starts (Standard: 0.5)",
    )
    args = parser.parse_args()

    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    jobs = JOBS

    if args.groups:
        wanted = {g.strip() for g in args.groups.split(",")}
        jobs = [j for j in jobs if j.group in wanted]
        if not jobs:
            print(f"Keine Jobs für Gruppen: {wanted}")
            sys.exit(1)

    if not jobs:
        print("Keine Jobs definiert.")
        sys.exit(1)

    if not args.dry_run and not check_screen_available():
        print("screen ist nicht installiert oder nicht im PATH.")
        sys.exit(1)

    gpu_map = assign_gpus(jobs, gpus)

    # Übersicht
    print(f"\n{'═'*60}")
    print(f"  GPUs:          {gpus}")
    print(f"  GPU-Zuweisung: {gpu_map}")
    print(f"  Jobs gesamt:   {len(jobs)}")
    print(f"{'─'*60}")

    groups_seen: set[str] = set()
    for job in jobs:
        gpu = gpu_map[job.group]
        if job.group not in groups_seen:
            print(f"\n  ── {job.group} → GPU {gpu} ──────────────────────────")
            groups_seen.add(job.group)
        print(f"  {job.session_name}")

    print(f"\n{'═'*60}\n")

    if not args.dry_run:
        yn = input("Alle Sessions starten? [j/N] ").strip().lower()
        if yn not in ("j", "y", "ja", "yes"):
            print("Abgebrochen.")
            return

    # Sessions starten
    for job in jobs:
        gpu = gpu_map[job.group]
        cmd = build_command(job, gpu)
        launch_session(job, cmd, dry_run=args.dry_run)
        if not args.dry_run:
            time.sleep(args.delay)

    if not args.dry_run:
        print(f"\nAlle {len(jobs)} Sessions gestartet.")
        print("Übersicht:  screen -ls")
        print("Anhängen:   screen -r <session_name>")
        print("Detachen:   Ctrl+A, D")


if __name__ == "__main__":
    main()
