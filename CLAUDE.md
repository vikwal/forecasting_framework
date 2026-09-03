# Forecasting Framework — Claude Context

## Projektübersicht

**Renewable Energy Forecasting** (Wind & PV) mit Machine Learning.
- Input: Numerische Wetterprognosen des **ICON-D2** Modells (NWP)
- Framework: **PyTorch**
- Zwei Lernparadigmen: **Centralized Learning (CL)** und **Federated Learning (FL)**

## Hauptskripte

| Skript | Zweck |
|---|---|
| `train_cl.py` | Zentrales Training mit besten HPO-Hyperparametern |
| `train_fl.py` | Federated Learning Training (nutzt Ray) |
| `hpo_cl.py` | Hyperparameteroptimierung (Optuna), einzelne GPU |
| `hpo_fl.py` | HPO für FL |
| `launch_multi_gpu.sh` | Startet parallele HPO-Prozesse über mehrere GPUs |
| `run_train_all_stations.sh` | Batch-Training für alle Stationen |
| `run_hpo_all_stations.sh` | Batch-HPO für alle Stationen |
| `eval_dashboard.py` | Visualisierung der Ergebnisse aus `results/` |
| `optuna_dashboard.py` | Streamlit-Dashboard für HPO-Studies (PostgreSQL via `OPTUNA_STORAGE`), läuft als systemd-Service auf Port 8503 → `docs/optuna_dashboard.md` |

## Utils (`utils/`)

| Modul | Inhalt |
|---|---|
| `preprocessing.py` | Hauptpipeline für Datenaufbereitung — **modellabhängig** (TFT vs. TCN-GRU) und **quellenabhängig** (Wind vs. PV). Kernfunktionen: `pipeline()`, `prepare_data()`, `prepare_data_for_tft()`, `preprocess_synth_wind_icond2()`, `preprocess_synth_pv()` |
| `db_connector.py` | **NEU**: PostgreSQL/PostGIS Datenbankzugriff für ICON-D2 Daten. Connection Pool, Grid Point KNN-Suche, Multilevel-Datenlader → `docs/icond2_database_integration.md` |
| `models.py` | Modelldefinitionen inkl. TFT, TCN-GRU, FNN, LSTM; Factory-Funktion `get_model()` |
| `federated.py` | FL-Logik: Aggregation (FedAvg), Datenladen pro Client, Ray-Integration |
| `eval.py` | Evaluierungsmetriken, Persistence-Baseline, Ergebnisspeicherung |
| `tools.py` | Config-Laden, Datensplit, DataLoader, Scaler-Handling |
| `hpo.py` | Optuna-Integration, Cross-Validation, Trial-Management |
| `data_cache.py` | Intelligentes Caching mit Hash-basierten Keys |
| `meteo.py` | Meteorologische Hilfsfunktionen (pvlib, Irradianz, Luftdichte) |
| `solar.py` | **NEU (Aug 2026)**: Solar-Irradiance-Pipeline — DWD-Stationsmessungen + ICON-D2 **SL** (Surface Level). `preprocess_solar_icond2()` als Pendant zu `preprocess_synth_wind_icond2()` → `docs/preprocess_icond2_solar.md` |

## Konfiguration (`configs/`)

Alle Experimente werden über **YAML-Dateien** gesteuert. Namenskonvention: `config_wind_<N>.yaml`, `config_wind_<N>fl.yaml` etc.

Wichtige Config-Sektionen:
- `data:` — Datenpfad, Dateien (Stations-IDs), Frequenz, Zielspalte
- `model:` — Architektur, Lookback, Horizon, Feature-Dimensionen, Early Stopping
- `hpo:` — Trials, Folds, Study-Pfad
- `fl:` — Clients-Mapping, Strategie (fedavg), Runden, Fine-Tuning, Early Stopping

## Federated Learning

- Clients sind Gruppen von Stationen (definiert in `fl.clients` in der Config)
- Aggregationsstrategie: FedAvg (erweiterbar)
- **Fine-Tuning** nach FL möglich: globales Modell wird lokal angepasst → `docs/fine_tuning_feature.md`
- Zwei separate Early-Stopping-Konfigurationen (FL vs. Fine-Tuning) → `docs/early_stopping_config.md`

## Preprocessing-Besonderheiten

- **TFT**: Unterscheidet zwischen `observed`, `known` (Wetterprognosen) und `static` Features → eigene `prepare_data_for_tft()` Pipeline
- **Wind**: `preprocess_synth_wind_icond2()` — ICON-D2 NWP Daten, Luftdichte, Rotor-Geometrie
- **PV**: `preprocess_synth_pv()` — pvlib-basierte Irradianzberechnungen
- Skalierung: StandardScaler, scaler wird für Inverse-Transform bei Evaluation gespeichert

## Verzeichnisstruktur

```
forecasting_framework/
├── configs/          # YAML-Konfigurationsdateien
├── docs/             # Feature-Dokumentation (für KI-Kontext)
├── utils/            # Alle Hilfsfunktionen
├── data_cache/       # Gecachte preprocessed Daten (Hash-basiert)
├── models/           # Gespeicherte Modelle
├── results/          # Evaluierungsergebnisse
├── studies/          # Optuna Study-Datenbanken
├── logs/             # Trainings- und HPO-Logs
└── archiv/           # Ältere Versionen
```

## Docs-Verzeichnis

Detaillierte Feature-Dokumentation in `docs/`:
- [handoff.md](docs/handoff.md) — **AKTUELLER STAND (Sep 2026)**: die neun Laeufe der finalen Testauswertung laufen auf l1/l2/ws, wo sie liegen, wie man sie beobachtet und stoppt, was danach zu tun ist; dazu die Imputationskette in ihrer heutigen Form und die offenen Stolperfallen
- [dcrnn_implementation_fixes.md](docs/dcrnn_implementation_fixes.md) — **NEU (Mai 2026)**: Analyse der Paper-Abweichungen, Fixes für BiDirDiffConv + zeitabhängige NWP-Attention, Performanz-Implikationen
- [imputation_tft_switch.md](docs/imputation_tft_switch.md) — **NEU (Sep 2026)**: wind_speed-Imputation auf die TFT-Werte (`imputed` in `interpol/wind`) umgestellt. Loest Regression-Kriging UND den ERA5-OLS-Pfad ab, `IMPUTATION_GUARD_VERSION` 3→4, Belegmessung, `kontextfrei`-Konvention, offener Punkt `dcrnn.interpolate_history`
- [imputation_knn_regen_20260902.md](docs/imputation_knn_regen_20260902.md) — **NEU (Sep 2026)**: KNN-Cache fuer `wind_direction` neu gerechnet (Abdeckung endete 2026-07-14, jetzt 2026-09-01), Stationssatz 203→204, alter Stand unter `knnimputer/wind_vor_regen_20260902`. Dazu der angeglichene NaN-Audit in `train_mtgnn.py`/`train_wavenet.py`
- [early_stopping_config.md](docs/early_stopping_config.md) — Lokales FL Early Stopping vs. Fine-Tuning Early Stopping
- [fine_tuning_feature.md](docs/fine_tuning_feature.md) — Fine-Tuning nach FL
- [global_early_stopping.md](docs/global_early_stopping.md) — Early Stopping über globale FL-Runden
- [preprocess_icond2_wind.md](docs/preprocess_icond2_wind.md) — ICON-D2 Wind Preprocessing: Datenstruktur, Features, Luftdichte, Lookback/Horizon-Zusammenhang, bekannte Limitierungen
- [preprocess_icond2_solar.md](docs/preprocess_icond2_solar.md) — **NEU (Aug 2026)**: ICON-D2 **SL** Solar Preprocessing. Einheiten (J/cm² → W/m²), lon-first-Dateinamen (anders als ML!), flache SL-Struktur, Intervallmittel-Semantik der Strahlungsfelder, **`data.freq` 15 min vs. 1 h** (§ 2b), Messzeitstempel-Konvention (§ 2.6), Datenlücken-Analyse
- [predict_solar.md](docs/predict_solar.md) — **NEU (Aug 2026)**: Zeitraster und Akkumulationssemantik. Wie ICON-D2 (15 min), DWD-Messung (10 min) und die globalen Modelle (stündlich) auf ein gemeinsames Raster kommen; flächentreue Umverteilung (Variante A), Umsetzungsstand, offene Punkte
- [solar_irradiance_plan.md](docs/solar_irradiance_plan.md) — Implementierungsplan des Solar-Use-Cases, getroffene Entscheidungen, offene Punkte
- [predict_wind.md](docs/predict_wind.md) — Wind-Zielgrößen (`wind_speed` & `power`), `extrapolate`-Parameter (Power Law auf Nabenhöhe, `wind_speed_hub_extrap`), NWP-Baseline, Skill_NWP, MultiIndex-Handling, bekannte Fallstricke
- [optuna_dashboard.md](docs/optuna_dashboard.md) — Deployment (systemd), Datenbank-Anbindung via `OPTUNA_STORAGE`, Passwort-geschütztes Study-Löschen
- [fold_dashboard.md](docs/fold_dashboard.md) — **NEU (Aug 2026)**: Streamlit-Dashboard der räumlichen Stationsaufteilungen (Port 8511), Use-Case-Voreinstellung Wind/Solar. Host-weite Dienstübersicht liegt **ausserhalb** des Repos: `~/docs/services.md`
- [station_splits_solar.md](docs/station_splits_solar.md) — **NEU (Aug 2026)**: Entwurf der Solar-Stationsaufteilung. Warum der aus Wind geerbte Split nicht taugt, warum `next_n_stations: 0` den Übertrag ändert, Netzdichte als Grenze (48 km), 21 Testsatz + 62 Pool in 3 rotierenden Folds
- [icond2_database_integration.md](docs/icond2_database_integration.md) — PostgreSQL-Datenbankzugriff für ICON-D2 (Config: `data.icond2_source: 'database'`), Performance, Fehlerbehandlung, Migration CSV→DB
- [train_dcrnn.md](docs/train_dcrnn.md) — DCRNN Training: CLI-Argumente, HPO-Integration (`--hpo-study`), Config-Section, Datenpipeline, Output-Format (`.pt` + Pickle-Dict), **Architektur-Updates (Mai 2026)**

## Use Cases

| Use Case | Config-Schlüssel | Preprocessing | NWP-Quelle |
|---|---|---|---|
| Wind | `data.use_case: wind` (Default) | `preprocessing.preprocess_synth_wind_icond2()` | ICON-D2 **ML** (Multilevel), `ML/{hh}/{station_id}/` |
| Solar | `data.use_case: solar` | `solar.preprocess_solar_icond2()` | ICON-D2 **SL** (Surface), `SL/{hh}/` (flach!) |

Beide liefern denselben Kontrakt: MultiIndex `['starttime','forecasttime','timestamp']`.
Schritte je Lauf = `48 h / data.freq` — bei Wind zwingend 48 (ML ist stündlich), bei Solar
48 / 96 / 192 je nach `freq` (SL ist nativ 15-minütig). Mehrere Zielgrößen über
`data.target_cols` (Multi-Output-Modelle).

**Solar-Stand (Aug 2026):** CL/FL-Pfad läuft; der GNN-Pfad ist durch Messlücken blockiert
und braucht zuerst die Solar-Imputation (`data.interpol_path`/`knnimputer_path`).

## Wichtige Hinweise

- README.md erwähnt TensorFlow — das Framework nutzt aber **PyTorch**
- Stationsdaten liegen auf NAS (`/mnt/nas/synthetic/`)
- **ICON-D2 Daten**: Standardmäßig aus PostgreSQL-Datenbank (`data.icond2_source: 'database'`), CSV-Fallback verfügbar
- **DB-Zugriff**: Umgebungsvariable `WEATHER_DB_URL` muss in `.bashrc` gesetzt sein
- GPU-Nutzung: GPU 0 oft von anderen Prozessen belegt → in Configs ausschließen
- `frcst/` ist das virtuelle Environment
