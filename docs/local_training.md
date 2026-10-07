# Lokales Training je FL-Client (`train_local.py`)

Stand 2026-10-07. Baseline für FL-Experimente: Jeder Client einer FL-Config trainiert ein eigenes
Modell nur auf seinen Stationen, mit der zentralen Pipeline (`train_cl.py`), ein Prozess je
GPU-Slot, parallel. Damit lässt sich je Station die Kette lokal → föderiert → zentral messen.
Code: `train_local.py` (CLI), `utils/local_training.py` (Config-Ableitung, GPU-Warteschlange,
Zusammenführen). Tests: `tests/test_local_training.py`.

```bash
python train_local.py -c configs/parks_v1/config_parks_v1_fl_fedgradient -m tft --gpus 0-7
python train_local.py -c configs/parks_v1/config_parks_v1_fl_fedgradient_parkid -m tft --gpus 0-3   # 8 Clients auf 4 GPUs (Warteschlange)
python train_local.py -c ... --clients R0 R1 --gpus 0,1 --dry-run                                     # nur Configs + Kommandos
```

Optionen: `--clients` (Teilmenge), `--gpus` (`0-7`, `0,2,3`; Default alle), `--per-gpu`
(gleichzeitige Läufe je GPU, Default 1), `--name`, `--save_model`, `--dry-run`.
Lange Läufe detached starten (`setsid nohup … &`); das Skript bricht nicht ab, wenn ein Client
scheitert, sondern meldet ihn am Ende (`manifest.json`, Exit-Code 1).

## Ableitung der Client-Configs

Die FL-Config (`fl.clients`) ist die einzige Quelle: je Client eine Kopie mit
`data.files` = seine Stationen; `val_files`, `holdout_files`, `test_files`, `client_files`
entfallen (keine fremden Stationen, weder im Training noch in der Bewertung); `model.fl: False`.
Alles andere (Daten, Features, Split, Modell, `model.epochs`, `model.early_stopping`) bleibt
identisch. Folgen: eigener Scaler, eigenes Early Stopping auf dem eigenen Validierungszeitraum,
eigener Adam über alle Epochen; mit Park-ID ein Embedding nur über die eigenen Stationen
(Codes 0…n−1).

Name: `config_<…>_fl_<strategy>[_x]` → `config_<…>_local[_x]` (z. B.
`config_parks_v1_local_parkid`).

## Ausgaben

- `runs/local/<name>_<zeitstempel>/`: `configs/<name>_<client>.yaml` (Pfade aufgelöst, also
  host-spezifisch: Protokoll, keine wiederverwendbare Config), `logs/<client>.out`,
  `train_local.log`, `manifest.json` (GPU, Returncode, Laufzeit, Ergebnisdatei je Client).
- je Client das übliche `train_cl.py`-Ergebnis `results/<data>/cl_m-…_<name>_<client>_<ts>.pkl`.
- zusammengeführt `results/<data>/local_m-<model>_<name ohne config_>_<ts>.pkl`: `evaluation`
  je Station mit Spalte `client_id` (+ Zeilen `mean`/`std`, Layout wie `train_fl.py`),
  `clients` (Epochen, beste Epoche, Laufzeit, GPU), dazu `…_clients.json`.

Die GPU-Warteschlange (`local_training.run_jobs`) ist allgemein: beliebige Kommandos mit
`CUDA_VISIBLE_DEVICES` je Slot, nächster Job sobald ein Slot frei wird, `OMP_NUM_THREADS` =
Kerne / Slots.
