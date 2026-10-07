# parks_v1 — reale MaStR-Windparks im regulären Wind-Pfad

Stand 2026-10-06. Daten: 90 reale Windparks (MaStR-Lokationen, Selektion v1.1) aus
`~/Work/FL_Contribution` — Synthese `${DATA_ROOT}/synthetic/wind/parks_v1/`,
Prognosen `${DATA_ROOT}/nwp_ready/{icon-d2,ecmwf}/` (Punktextraktion `~/Work/NWP/points`).
Berichte: `FL_Contribution/reports/park_synthesis_v1.md`, `nwp_extraction_plan.md`.

## Ablauf

```bash
python scripts/build_parks_v1_inputs.py            # data/parks_v1/{stations_master,wind_parameter,turbine_parameter}.csv
python train_cl.py -c configs/parks_v1/config_parks_v1_cl_smoke -m tft   # 3 Parks, 1 Epoche
python train_cl.py -c configs/parks_v1/config_parks_v1_cl -m tft         # 90 Parks
```

## Erweiterungen in `preprocess_synth_wind_icond2` (nur aktiv, wenn gesetzt)

| Schlüssel | Wirkung |
|---|---|
| `data.power_col` | Ziel = diese Spalte des Synth-Parquets (parks_v1: `power_park`, also mit Wakes), statt Summe aller `power*`-Spalten |
| `data.capacity_col` | Normierung auf `wind_parameter[capacity_col]` kW (registrierte Leistung) |
| `data.wind_parameter_file`, `data.turbine_parameter_file` | Parametertabellen außerhalb von `data.path` |
| `data.nwp_site_prefix` | ICON/ECMWF-Ordner heißen `<prefix><station_id>` (nwp_ready: `park_`) |
| `params.static_categorical: [park_id]` | Park-ID als kategoriales statisches Feature: ganzzahliger Code (Position in `files`+`val_files`+`test_files`, `preprocessing.static_categories`), unskaliert, im TFT `nn.Embedding` statt `nn.Linear(1, d)` (`models._static_cardinalities`); nur `tft` |
| `data.ecmwf_layout: site_runs` | ECMWF aus `<ecmwf_path>/SL/{00,12}/<prefix><id>/<lat>_<lon>_wind_sl.parquet` (`_fetch_ecmwf_data_from_site_runs`) |

Die Laufzuordnung bleibt die des Wind-Pfads: ICON-Lauf < 12 UTC → ECMWF 00 UTC desselben Tages
(bei ICON 09 UTC ohne Blick in die Zukunft). Alle neuen Schlüssel gehen in den Cache-Hash ein,
aber nur wenn gesetzt — bestehende Cache-Einträge bleiben gültig.

## Konfiguration `configs/parks_v1/config_parks_v1_cl.yaml`

ICON-D2 09 UTC `wind_speed_h78/h127/h184` + ECMWF `wind_speed_h100/h200`, je 1 Gitterpunkt
(nächster zum Parkmittelpunkt = Mittel der Turbinenkoordinaten), Leistung der letzten 48 h als
observed, Horizont 48 h, TFT. Training Läufe 2023-07-24 … 2024-07-31, Validierung
2024-08-01 … 2025-07-31.

Variante `config_parks_v1_cl_parkid.yaml`: gleich, plus `static_features: [park_id]` als Embedding.
Numerische Park-Statiken (Nabenhöhe, Rotor, …) sind bewusst nicht drin: ein Park mischt Typen
und Nabenhöhen. Im `power_col`-Modus werden die alten Turbinen-Statiken übersprungen.
Tests: `python -m pytest tests/test_parks_v1.py`.

## Föderiert (FL, 80 Parks in 8 Clients)

Clients aus `FL_Contribution/data/mastr/wind_park_selection_v1_1.csv` (Spalte `client`: R0–R3
regional, N0–N3 überregional, je 10 Parks; `holdout` = 10 Parks). Strategie `fedgradient`
([fedgradient.md](fedgradient.md)), Daten, Features, Split und Modell wie CL.

| Config | Training | Bewertung |
|---|---|---|
| `config_parks_v1_fl_fedgradient.yaml` | 80 Client-Parks | 80 + Holdout (`val_files`, globales Modell) |
| `config_parks_v1_fl_fedgradient_parkid.yaml` | 80, Park-ID-Embedding (Codes global, zeilenweise Aggregation) | nur die 80 |
| `config_parks_v1_cl80{,_parkid}.yaml` | CL-Referenz auf denselben 80 Parks, Early Stopping auf ihnen | wie FL; Holdout über `data.holdout_files` |
| `…_smoke.yaml` | 2 Clients × 2 Parks, 2 Runden | |

```bash
python train_fl.py -c configs/parks_v1/config_parks_v1_fl_fedgradient -m tft   # 8 Clients, je 1 GPU
python train_cl.py -c configs/parks_v1/config_parks_v1_cl80 -m tft
```
Lokale Baseline (je Client ein eigenes Modell, [local_training.md](local_training.md)):
`python train_local.py -c configs/parks_v1/config_parks_v1_fl_fedgradient{,_parkid} -m tft --gpus 0-7`.
Vergleich FedAvg: `config_parks_v1_fl_fedavg{,_parkid}.yaml` (dieselbe Config, `fl.strategy: fedavg`,
`n_local_epochs: 1`).

`data.holdout_files` (nur `train_cl.py`): Stationen, die nach dem Training mit demselben Modell
und Scaler bewertet werden, ohne Training und ohne Early Stopping. `val_files` ersetzt in
`train_cl.py` dagegen die Validierungs- und Testmenge (Early Stopping und Bewertung auf ihnen).
In `train_fl.py` sind `val_files` reine Bewertungsparks.

Ergebnisse 2026-10-06 (R² je Park auf 2024-08 … 2025-07, 80 Client-Parks, Mittel / Median):
CL80 0,833 / 0,845, FedGradient 0,831 / 0,841, FedAvg 0,831 / 0,841; mit Park-ID CL80 0,861 /
0,868, FedGradient 0,857 / 0,866, FedAvg 0,843 / 0,850 (nach 100 Runden noch nicht konvergiert).
Holdout (10 Parks, ohne Park-ID): CL80 0,836, FedGradient 0,839, FedAvg 0,838.
Bericht: `FL_Contribution/reports/fl_fedgradient_parks_v1.md`.
Tests: `python -m pytest tests/test_fedgradient.py`.
