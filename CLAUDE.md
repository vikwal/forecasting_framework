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

## Utils (`utils/`)

| Modul | Inhalt |
|---|---|
| `preprocessing.py` | Hauptpipeline für Datenaufbereitung — **modellabhängig** (TFT vs. TCN-GRU) und **quellenabhängig** (Wind vs. PV). Kernfunktionen: `pipeline()`, `prepare_data()`, `prepare_data_for_tft()`, `preprocess_synth_wind_icond2()`, `preprocess_synth_pv()` |
| `models.py` | Modelldefinitionen inkl. TFT, TCN-GRU, FNN, LSTM; Factory-Funktion `get_model()` |
| `federated.py` | FL-Logik: Aggregation (FedAvg), Datenladen pro Client, Ray-Integration |
| `eval.py` | Evaluierungsmetriken, Persistence-Baseline, Ergebnisspeicherung |
| `tools.py` | Config-Laden, Datensplit, DataLoader, Scaler-Handling |
| `hpo.py` | Optuna-Integration, Cross-Validation, Trial-Management |
| `data_cache.py` | Intelligentes Caching mit Hash-basierten Keys |
| `meteo.py` | Meteorologische Hilfsfunktionen (pvlib, Irradianz, Luftdichte) |

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
- [early_stopping_config.md](docs/early_stopping_config.md) — Lokales FL Early Stopping vs. Fine-Tuning Early Stopping
- [fine_tuning_feature.md](docs/fine_tuning_feature.md) — Fine-Tuning nach FL
- [global_early_stopping.md](docs/global_early_stopping.md) — Early Stopping über globale FL-Runden
- [preprocess_icond2_wind.md](docs/preprocess_icond2_wind.md) — ICON-D2 Wind Preprocessing: Datenstruktur, Features, Luftdichte, Lookback/Horizon-Zusammenhang, bekannte Limitierungen

## Wichtige Hinweise

- README.md erwähnt TensorFlow — das Framework nutzt aber **PyTorch**
- Stationsdaten liegen auf NAS (`/mnt/nas/synthetic/`)
- GPU-Nutzung: GPU 0 oft von anderen Prozessen belegt → in Configs ausschließen
- `frcst/` ist das virtuelle Environment
