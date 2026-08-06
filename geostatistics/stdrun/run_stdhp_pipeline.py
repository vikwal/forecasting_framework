#!/usr/bin/env python3
"""
run_stdhp_pipeline.py — Trockenlauf-Treiber: Retrain -> Eval mit festen
Standard-Hyperparametern (stdhp) statt HPO-Best-Params, ueber 11 Varianten x
3 Folds = 33 Fold-Jobs (DCRNN {base, nwp, nwp_hist, nomeas, nograph, idw},
MTGNN {base, nwp, nwp_hist}, WaveNet {base, nwp}).

Bewusst ein eigener Treiber neben launch_train_pipeline.py / launch_eval_pipeline.py,
damit die Kampagnen-Launcher (die die laufende HPO-Infrastruktur mitbenutzen)
unangetastet bleiben. Die Job-Tabelle wird aus deren GROUPS/JOBS-Listen
IMPORTIERT statt dupliziert:
  - geostatistics/launch_train_pipeline.py : GROUPS  (Config, Script, Fold)
  - geostatistics/launch_eval_pipeline.py  : JOBS    (raw_out_name-Konvention)
  - geostatistics/stdrun/gen_stdhp_configs.py : new_stem() (Config-Stem-Mapping
    Original -> stdhp, identisch zu den tatsaechlich erzeugten Configs)
Train- und Eval-Jobs werden ueber den *Original*-Config-Pfad verknuepft (nicht
ueber den Gruppennamen — die beiden Launcher benennen dieselbe Variante
unterschiedlich, z.B. Trainingsgruppe "DCRNN_NWP" == Eval-Gruppe "DCRNN_GRID").

Unterschiede zu den Kampagnen-Launchern (bewusst):
  - Granularitaet der Warteschlange ist der einzelne Fold-Job (train+eval),
    nicht die Gruppe — die Launcher starten immer alle 3 Folds einer Gruppe
    gleichzeitig auf einer Karte, das wollen wir hier nicht.
  - Kein --hpo-study, nirgends: alle 30 Configs sind stdhp-Configs mit festen
    YAML-Werten (siehe gen_stdhp_configs.py); train_dcrnn.py:348 und
    get_test_results_dcrnn.py (analog fuer mtgnn/wavenet) ueberspringen den
    Optuna-Override-Block vollstaendig, wenn --hpo-study fehlt.
  - Kollisionsschutz: alle 30 Zielartefakte tragen stdhp im Namen
    (models/*_stdhp.pt) bzw. den Praefix stdhp_ (data/test_results/,
    data/raw_preds/) und sind damit disjunkt vom Bestand aus Juni/Juli. Kein
    Lauf kann eine fremde Datei ueberschreiben (--on-existing skip|abort).
    Ein *eigener* Teilstand wird dagegen bewusst neu erzeugt: ein Checkpoint
    ohne results/<model>_*.pkl stammt aus einem abgebrochenen Training und
    wird neu trainiert; liegt nur eine der beiden Eval-Ausgaben vor, laeuft
    die Eval erneut. Nur vollstaendige Teilergebnisse werden uebersprungen.

Env-Guard: WEATHER_DB_URL, ECMWF_WIND_SL_URL, OPTUNA_STORAGE muessen in der
Umgebung gesetzt sein (siehe Befund K3 — ein Worker ohne WEATHER_DB_URL
schreibt still NWP-Hoehen = 0). Diese Variablen werden NICHT aus ~/.bashrc
nachgeladen (das bricht in ssh-non-interactive-Shells vor Zeile 118-122 ab) —
der Aufrufer muss sie vor dem Start exportiert haben, z.B.:
  eval "$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE)=' ~/.bashrc)"

Reihenfolge: WaveNet base, WaveNet nwp, DCRNN nomeas, DCRNN nograph zuerst
(nie end-to-end gelaufene Pfade), danach der Rest in GROUPS-Reihenfolge.

Verwendung
----------
    cd /home/viktor/Work/forecasting_framework
    python geostatistics/stdrun/run_stdhp_pipeline.py --gpus 0,1 --dry-run
    python geostatistics/stdrun/run_stdhp_pipeline.py --gpus 0,1 --max-concurrent 2
    python geostatistics/stdrun/run_stdhp_pipeline.py --gpus 0 --groups WAVENET_BASE,WAVENET_NWP
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

STDRUN_DIR = Path(__file__).resolve().parent          # geostatistics/stdrun/
GEOSTAT_DIR = STDRUN_DIR.parent                        # geostatistics/
WORK_DIR = GEOSTAT_DIR.parent                           # forecasting_framework/
VENV_ACTIVATE = WORK_DIR / "frcst/bin/activate"

sys.path.insert(0, str(STDRUN_DIR))
sys.path.insert(0, str(GEOSTAT_DIR))

from gen_stdhp_configs import new_stem                  # noqa: E402
from launch_train_pipeline import GROUPS as TRAIN_GROUPS  # noqa: E402
from launch_eval_pipeline import JOBS as EVAL_JOBS       # noqa: E402

REQUIRED_ENV = ["WEATHER_DB_URL", "ECMWF_WIND_SL_URL", "OPTUNA_STORAGE"]

# Groups that never had an end-to-end retrain+eval run before this dry-run —
# scheduled first so the new paths produce results early. DCRNN_IDW (Ablation
# D, R6(a)) is the newest of these: fixed inverse-distance NWP aggregation.
PRIORITY_GROUPS = ["WAVENET_BASE", "WAVENET_NWP", "DCRNN_NOMEAS", "DCRNN_NOGRAPH", "DCRNN_IDW"]


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

@dataclass
class StdhpJob:
    jobname: str            # e.g. "dcrnn_nomeas_fold1" — used for the log filename
    group: str               # original train-group name (for ordering/reporting)
    fold_label: str
    model_type: str          # "dcrnn" | "mtgnn" | "wavenet"
    train_script: str
    eval_script: str
    stdhp_config: str        # relative to WORK_DIR
    model_name: str          # substring for -m / expected checkpoint stem
    checkpoint_path: Path    # models/<model_name>.pt
    raw_out_name: str        # stdhp_<original raw_out_name>
    csv_path: Path
    parquet_path: Path

    # Runtime state (mutated by the driver; not part of identity)
    train_status: str = "pending"   # pending|skipped|running|ok|failed
    eval_status: str = "pending"
    train_rc: int | None = None
    eval_rc: int | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def training_complete(self) -> bool:
        """True iff the training run that produced the checkpoint also ran to
        completion.

        The checkpoint alone is NOT a completion marker: DCRNNTrainer (and the
        MTGNN/WaveNet trainers) torch.save() on every val-RMSE improvement
        (geostatistics/dcrnn/training/trainer.py:453-457), so an aborted run
        leaves a perfectly loadable but under-trained .pt behind. All three
        train_*.py scripts write results/<model_name>_<timestamp>.pkl as their
        very last action (train_dcrnn.py:1150, train_mtgnn.py:855,
        train_wavenet.py:812) — that pkl is the completion marker.
        """
        if not self.checkpoint_path.exists():
            return False
        return any((WORK_DIR / "results").glob(f"{self.model_name}_*.pkl"))

    def eval_complete(self) -> bool:
        """True iff BOTH eval outputs exist. get_test_results_*.py writes the
        CSV first and the parquet second, so a crash in between leaves a CSV
        without a parquet — treating that as 'done' would silently swallow a
        half-finished evaluation."""
        return self.csv_path.exists() and self.parquet_path.exists()

    def to_status_dict(self) -> dict:
        return {
            "jobname": self.jobname,
            "group": self.group,
            "fold": self.fold_label,
            "model_type": self.model_type,
            "stdhp_config": self.stdhp_config,
            "checkpoint": str(self.checkpoint_path.relative_to(WORK_DIR)),
            "csv": str(self.csv_path.relative_to(WORK_DIR)),
            "parquet": str(self.parquet_path.relative_to(WORK_DIR)),
            "train_status": self.train_status,
            "train_rc": self.train_rc,
            "eval_status": self.eval_status,
            "eval_rc": self.eval_rc,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def stdhp_config_for(orig_config: str) -> tuple[str, str, str]:
    """Map an original fold config path to (stdhp_config_path, model_type, stdhp_config_stem)."""
    p = Path(orig_config)
    model_type = p.parts[1]  # configs/<model_type>/config_....yaml
    orig_stem = p.stem.replace("config_", "")
    stdhp_stem = new_stem(orig_stem)
    stdhp_config = f"configs/{model_type}/stdhp/config_{stdhp_stem}.yaml"
    return stdhp_config, model_type, stdhp_stem


def build_jobs() -> list[StdhpJob]:
    eval_by_config = {j.config: j for j in EVAL_JOBS}

    train_configs = {(g.name, f.config) for g in TRAIN_GROUPS for f in g.folds}
    missing_eval = [c for _, c in train_configs if c not in eval_by_config]
    if missing_eval:
        raise RuntimeError(
            f"{len(missing_eval)} train config(s) have no matching entry in "
            f"launch_eval_pipeline.JOBS (joined by config path) — cannot build "
            f"eval commands for: {missing_eval}"
        )

    # Order groups: PRIORITY_GROUPS first (in the given order), then the rest
    # in their original GROUPS order.
    groups_by_name = {g.name: g for g in TRAIN_GROUPS}
    missing_priority = [n for n in PRIORITY_GROUPS if n not in groups_by_name]
    if missing_priority:
        raise RuntimeError(f"PRIORITY_GROUPS references unknown group(s): {missing_priority}")
    ordered_names = list(PRIORITY_GROUPS) + [g.name for g in TRAIN_GROUPS if g.name not in PRIORITY_GROUPS]

    jobs: list[StdhpJob] = []
    for gname in ordered_names:
        group = groups_by_name[gname]
        model_type = group.script.replace("train_", "")
        for foldjob in group.folds:
            stdhp_config, cfg_model_type, stdhp_stem = stdhp_config_for(foldjob.config)
            assert cfg_model_type == model_type, (foldjob.config, model_type, cfg_model_type)

            model_name = f"{stdhp_stem}_{model_type}_stdhp"
            checkpoint_path = WORK_DIR / "models" / f"{model_name}.pt"

            eval_job = eval_by_config[foldjob.config]
            raw_out_name = f"stdhp_{eval_job.raw_out_name}"
            csv_path = WORK_DIR / "data" / "test_results" / f"{raw_out_name}.csv"
            parquet_path = WORK_DIR / "data" / "raw_preds" / f"{raw_out_name}_raw.parquet"

            jobname = f"{gname.lower()}_{foldjob.fold_label}"

            jobs.append(StdhpJob(
                jobname=jobname,
                group=gname,
                fold_label=foldjob.fold_label,
                model_type=model_type,
                train_script=group.script,
                eval_script=eval_job.script,
                stdhp_config=stdhp_config,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                raw_out_name=raw_out_name,
                csv_path=csv_path,
                parquet_path=parquet_path,
            ))
    return jobs


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def build_train_cmd(job: StdhpJob, gpu: int) -> list[str]:
    script = f"geostatistics/{job.train_script}.py"
    inner = (
        f"source {VENV_ACTIVATE} && cd {WORK_DIR} && "
        f"CUDA_VISIBLE_DEVICES={gpu} python {script} "
        f"--config {job.stdhp_config} --suffix stdhp"
    )
    return ["bash", "-c", inner]


def build_eval_cmd(job: StdhpJob, gpu: int) -> list[str]:
    script = f"geostatistics/{job.eval_script}.py"
    inner = (
        f"source {VENV_ACTIVATE} && cd {WORK_DIR} && "
        f"CUDA_VISIBLE_DEVICES={gpu} python {script} "
        f"-m {job.model_name} -c {job.stdhp_config} --raw-out-name {job.raw_out_name}"
    )
    return ["bash", "-c", inner]


def cmd_str(cmd: list[str]) -> str:
    """Human-readable one-liner for --dry-run output (the actual command run
    inside bash -c, without the bash -c wrapper noise)."""
    return cmd[2] if len(cmd) == 3 and cmd[0] == "bash" else " ".join(cmd)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class Driver:
    def __init__(self, jobs: list[StdhpJob], gpus: list[int], max_concurrent: int,
                 on_existing: str, status_path: Path, log_dir: Path):
        self.jobs = jobs
        self.gpus = gpus
        self.max_concurrent = max_concurrent
        self.on_existing = on_existing
        self.status_path = status_path
        self.log_dir = log_dir
        self.queue: Queue = Queue()
        for j in jobs:
            self.queue.put(j)
        self.stop_event = threading.Event()
        self.active_procs: list[subprocess.Popen] = []
        self.active_lock = threading.Lock()
        self.status_lock = threading.Lock()

    # -- status persistence --------------------------------------------
    def write_status(self) -> None:
        with self.status_lock:
            payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "jobs": [j.to_status_dict() for j in self.jobs],
            }
            tmp = self.status_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.status_path)

    # -- collision preflight ---------------------------------------------
    def preflight(self) -> None:
        colliding = []
        for j in self.jobs:
            for p in (j.checkpoint_path, j.csv_path, j.parquet_path):
                if p.exists():
                    colliding.append((j.jobname, str(p)))
        if colliding and self.on_existing == "abort":
            print(f"ABORT: {len(colliding)} target artifact(s) already exist "
                  f"(--on-existing abort):")
            for jobname, p in colliding:
                print(f"  {jobname}: {p}")
            sys.exit(3)
        if colliding:
            print(f"NOTE: {len(colliding)} target artifact(s) already exist "
                  f"(--on-existing skip, default). Per sub-step decision:")
            for j in self.jobs:
                if not any(p.exists() for p in (j.checkpoint_path, j.csv_path, j.parquet_path)):
                    continue
                t = "SKIP train" if j.training_complete() else (
                    "RETRAIN (checkpoint without completion marker)"
                    if j.checkpoint_path.exists() else "train")
                e = "SKIP eval" if j.eval_complete() else (
                    "RE-EVAL (only one of csv/parquet present)"
                    if (j.csv_path.exists() or j.parquet_path.exists()) else "eval")
                print(f"  {j.jobname}: {t} / {e}")

    # -- one job (train, then eval) --------------------------------------
    def run_job(self, job: StdhpJob, gpu: int) -> None:
        job.started_at = datetime.now().isoformat(timespec="seconds")
        log_path = self.log_dir / f"stdhp_{job.jobname}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "a") as logf:
            logf.write(f"\n{'='*80}\n[{datetime.now()}] TRAIN {job.jobname} (GPU {gpu})\n{'='*80}\n")
            logf.flush()

            if job.training_complete():
                job.train_status = "skipped"
                logf.write(f"SKIP — checkpoint + completion marker already exist: "
                           f"{job.checkpoint_path}\n")
            else:
                if job.checkpoint_path.exists():
                    logf.write(
                        f"RETRAIN — {job.checkpoint_path.name} exists but no "
                        f"results/{job.model_name}_*.pkl completion marker was found, i.e. the "
                        f"previous training was interrupted and the checkpoint is under-trained. "
                        f"Retraining from scratch (the stale checkpoint is overwritten; it carries "
                        f"the stdhp naming, so no pre-existing artifact is affected).\n")
                    logf.flush()
                job.train_status = "running"
                self.write_status()
                cmd = build_train_cmd(job, gpu)
                rc = self._run_subprocess(cmd, logf)
                job.train_rc = rc
                job.train_status = "ok" if rc == 0 else "failed"

            self.write_status()

            if job.train_status == "failed":
                logf.write(f"TRAIN FAILED (rc={job.train_rc}) — skipping eval.\n")
                job.eval_status = "skipped"
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                self.write_status()
                return

            logf.write(f"\n{'='*80}\n[{datetime.now()}] EVAL {job.jobname} (GPU {gpu})\n{'='*80}\n")
            logf.flush()

            if job.eval_complete():
                job.eval_status = "skipped"
                logf.write(f"SKIP — both outputs already exist: {job.csv_path} / "
                           f"{job.parquet_path}\n")
            else:
                if job.csv_path.exists() != job.parquet_path.exists():
                    logf.write(
                        f"RE-EVAL — exactly one of the two outputs exists "
                        f"(csv={job.csv_path.exists()}, parquet={job.parquet_path.exists()}), "
                        f"i.e. the previous evaluation was interrupted between the CSV and the "
                        f"parquet write. Re-running (both files carry the stdhp_ prefix, so no "
                        f"pre-existing result is affected).\n")
                    logf.flush()
                job.eval_status = "running"
                self.write_status()
                cmd = build_eval_cmd(job, gpu)
                rc = self._run_subprocess(cmd, logf)
                job.eval_rc = rc
                job.eval_status = "ok" if rc == 0 else "failed"

        job.finished_at = datetime.now().isoformat(timespec="seconds")
        self.write_status()

    def _run_subprocess(self, cmd: list[str], logf) -> int:
        if self.stop_event.is_set():
            return -1
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env={**os.environ})
        with self.active_lock:
            self.active_procs.append(proc)
        rc = proc.wait()
        with self.active_lock:
            if proc in self.active_procs:
                self.active_procs.remove(proc)
        return rc

    def terminate_all(self) -> None:
        with self.active_lock:
            procs = list(self.active_procs)
        for proc in procs:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        deadline = time.time() + 15
        for proc in procs:
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()

    def worker(self, worker_id: int) -> None:
        gpu = self.gpus[worker_id % len(self.gpus)]
        while not self.stop_event.is_set():
            try:
                job = self.queue.get_nowait()
            except Empty:
                break
            self.run_job(job, gpu)
            self.queue.task_done()

    def run(self) -> None:
        # Both the per-job logs and the status file live here; write_status()
        # below is the first write and would crash on a missing directory.
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.preflight()
        self.write_status()
        n_workers = max(1, self.max_concurrent)
        threads = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(n_workers)]
        for t in threads:
            t.start()
            time.sleep(0.2)
        try:
            for t in threads:
                while t.is_alive():
                    t.join(timeout=1.0)
        except KeyboardInterrupt:
            print("\nCtrl-C — terminating running children …")
            self.stop_event.set()
            self.terminate_all()
            self.write_status()
            sys.exit(130)
        self.write_status()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_env_guard() -> None:
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        print(f"FATAL: required environment variable(s) not set: {', '.join(missing)}", file=sys.stderr)
        print(
            "These are silently substituted with unsafe defaults otherwise (K3: a worker "
            "without WEATHER_DB_URL writes NWP heights = 0 without erroring). Export them "
            "before running this driver, e.g. on l2:\n"
            "  eval \"$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE)=' ~/.bashrc)\"",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain -> Eval dry-run driver for stdhp configs (30 fold-jobs, no HPO).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU indices, e.g. 0,1")
    parser.add_argument("--max-concurrent", type=int, default=2,
                         help="Number of fold-jobs (train+eval) run concurrently (default: 2)")
    parser.add_argument("--on-existing", choices=["skip", "abort"], default="skip",
                         help="If a target checkpoint/CSV/parquet already exists: "
                              "'skip' the affected sub-step (resume, default) or 'abort' the whole run.")
    parser.add_argument("--groups", default=None,
                         help="Only run these train-group names (comma-separated), e.g. WAVENET_BASE,WAVENET_NWP")
    parser.add_argument("--dry-run", "-n", action="store_true",
                         help="Print all train/eval command lines without starting anything.")
    parser.add_argument("--status-file", default=None,
                         help="Path to the JSON run-status file (default: logs/stdhp_pipeline_status.json)")
    args = parser.parse_args()

    check_env_guard()

    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    jobs = build_jobs()

    if args.groups:
        wanted = {g.strip() for g in args.groups.split(",")}
        unknown = wanted - {j.group for j in jobs}
        if unknown:
            print(f"Unknown group(s): {unknown}")
            sys.exit(1)
        jobs = [j for j in jobs if j.group in wanted]

    log_dir = WORK_DIR / "logs"
    status_path = Path(args.status_file) if args.status_file else log_dir / "stdhp_pipeline_status.json"

    print(f"\n{'='*90}")
    print(f"  Jobs total:      {len(jobs)}")
    print(f"  GPUs:            {gpus}")
    print(f"  Max concurrent:  {args.max_concurrent}")
    print(f"  On existing:     {args.on_existing}")
    print(f"  Status file:     {status_path}")
    print(f"  Order:           {[j.jobname for j in jobs]}")
    print(f"{'='*90}\n")

    if args.dry_run:
        n_workers = max(1, args.max_concurrent)
        print(f"NOTE: in a real run the GPU is bound to the WORKER SLOT "
              f"(worker_id % len(gpus)), not to the job index — with "
              f"--max-concurrent {n_workers} and --gpus {gpus} the GPUs actually in use are "
              f"{sorted({gpus[w % len(gpus)] for w in range(n_workers)})}. Jobs are pulled from a "
              f"shared queue, so which job lands on which slot is not fixed. The per-job GPU shown "
              f"below is only an illustration of the command shape.\n")
        for i, j in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            train_cmd = build_train_cmd(j, gpu)
            eval_cmd = build_eval_cmd(j, gpu)
            ck_exists = "exists" if j.checkpoint_path.exists() else "missing"
            csv_exists = "exists" if j.csv_path.exists() else "missing"
            print(f"[{i+1:2d}/{len(jobs)}] {j.jobname}  (GPU {gpu})")
            print(f"   TRAIN [{ck_exists}] {cmd_str(train_cmd)}")
            print(f"   EVAL  [{csv_exists}] {cmd_str(eval_cmd)}")
        print(f"\n{'DRY-RUN — nothing started.'}")
        return

    driver = Driver(jobs, gpus, args.max_concurrent, args.on_existing, status_path, log_dir)

    def handle_sigint(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    driver.run()

    n_ok = sum(1 for j in jobs if j.train_status == "ok" and j.eval_status == "ok")
    n_failed = sum(1 for j in jobs if j.train_status == "failed" or j.eval_status == "failed")
    n_skipped = sum(1 for j in jobs if j.train_status == "skipped" and j.eval_status == "skipped")
    print(f"\nDone. ok={n_ok} failed={n_failed} fully-skipped={n_skipped} total={len(jobs)}")
    print(f"Status: {status_path}")


if __name__ == "__main__":
    main()
