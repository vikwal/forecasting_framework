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

## Ausführen (Anleitung für Agenten)

Wann: Die Frage ist, was Föderation oder zentrales Training gegenüber einem rein lokalen Modell je
Client bringt („lokale Baseline“, „jeder Client nur auf seinen Daten“, „local vs. FL vs. CL“).
Dafür nichts selbst bauen und keine Configs je Client von Hand anlegen: `train_local.py` mit der
**FL-Config** des Vergleichslaufs aufrufen.

1. Host und GPUs: `nvidia-smi` — eine GPU je Client (l1: 8× A6000 für 8 Clients). Bei weniger
   freien GPUs `--gpus` auf die freien setzen, der Rest wartet in der Schlange; `--per-gpu 2`
   nur, wenn der Speicher reicht.
2. Umgebung: im Repo-Root, venv `frcst`, `DATA_ROOT` gesetzt (in nicht-interaktiven Shells
   `eval "$(grep -E '^export DATA_ROOT=' ~/.bashrc)"; export DATA_ROOT`).
3. Erst prüfen, dann starten (detached, das Skript wartet auf alle Clients):
   ```bash
   python train_local.py -c <fl-config> -m tft --gpus 0-7 --dry-run
   setsid nohup python train_local.py -c <fl-config> -m tft --gpus 0-7 > logs/<name>_local.log 2>&1 < /dev/null &
   ```
   Über ssh mit `timeout` und vollständig umgeleiteten Ein-/Ausgaben starten, sonst bleibt
   die ssh-Sitzung hängen.
4. Ende erkennen: im Log `[local] local: N clients, … -> results/…/local_m-….pkl`, je Client
   `[local] end <client> rc=…`; fehlgeschlagene Clients stehen in `manifest.json` (`failed`) und
   führen zu Exit-Code 1. Logs je Client: `runs/local/<name>_<ts>/logs/<client>.out`.
5. Vergleichen: Das zusammengeführte Ergebnis hat das Layout von `train_fl.py` (Zeilen je Station
   mit `client_id`); paarweise je Station gegen FL/CL auf denselben Stationen auswerten,
   z. B. mit `FL_Contribution/pipeline/summarize_fl_fedgradient.py label=<pkl> …`.

Einschränkungen: nur der zentrale Pfad (`train_cl.py`, also auch TFT mit Park-ID); keine
Bewertung auf fremden Stationen (Holdout) — jedes lokale Modell wird nur auf seinen eigenen
Stationen bewertet. Early Stopping läuft wie in `train_cl.py` auf dem Testzeitraum der eigenen
Stationen; mit acht getrennten Auswahlen ist das lokal etwas optimistischer als bei einem
gemeinsamen Modell (parks_v1: [parks_v1.md](parks_v1.md)).

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
