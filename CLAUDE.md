# Forecasting Framework — Claude Context

**Renewable Energy Forecasting** (Wind & PV) mit ML auf **ICON-D2**-NWP-Daten,
PyTorch. Zwei Lernparadigmen: **Centralized (CL)** und **Federated (FL)**.
Alle Experimente werden über YAML in `configs/` gesteuert
(`config_wind_<N>.yaml`, `…fl.yaml`), Sektionen `data:`, `model:`, `hpo:`, `fl:`.

## Code-Landkarte

| Wo | Was |
|---|---|
| `train_cl.py`, `train_fl.py` | Training CL bzw. FL (FL nutzt Ray) |
| `hpo_cl.py`, `hpo_fl.py` | Optuna-HPO; `launch_multi_gpu.sh` startet parallele Prozesse über mehrere GPUs |
| `geostatistics/` | **GNN-Pfad**: `train_dcrnn.py`, `train_mtgnn.py`, `train_wavenet.py`, Kriging, Variogramme, `fold_dashboard.py` |
| `utils/` | `preprocessing.py` (Hauptpipeline, modell- und quellenabhängig), `solar.py` (Solar-Pendant), `models.py` (`get_model()`), `federated.py` (FedAvg, Ray), `hpo.py`, `eval.py`, `tools.py`, `data_cache.py` (Hash-Keys), `meteo.py` (pvlib), `db_connector.py` (PostGIS) |
| `*_dashboard.py` | `optuna_dashboard.py` (8504), `geostatistics/fold_dashboard.py` (8511), `eval_dashboard.py` |
| `deploy/` | systemd-Unit-Vorlagen (host-weite Dienstübersicht: `~/docs/services.md`) |

Namensmuster für alles Weitere: `train_*.py`, `hpo_*.py`, `get_test_results*.py`,
`check_*.py`, `run_*_all_stations.sh`. `trianel_*` gehört zum Trianel-Datensatz.
Nicht selbsterklärende Verzeichnisse: `studies/` (Optuna-DBs), `data_cache/`
(gehashte preprocessed Daten), `runs/`, `reports/`, `archiv/` (alte Stände).

## Use Cases

| Use Case | Config | Preprocessing | NWP-Quelle |
|---|---|---|---|
| Wind | `data.use_case: wind` (Default) | `preprocessing.preprocess_synth_wind_icond2()` | ICON-D2 **ML** (Multilevel), `ML/{hh}/{station_id}/` |
| Solar | `data.use_case: solar` | `solar.preprocess_solar_icond2()` | ICON-D2 **SL** (Surface), `SL/{hh}/` (flach!) |

Beide liefern denselben Kontrakt: MultiIndex `['starttime','forecasttime','timestamp']`.
Schritte je Lauf = `48 h / data.freq` — bei Wind zwingend 48 (ML ist stündlich),
bei Solar 48/96/192 je nach `freq` (SL ist nativ 15-minütig). Mehrere Zielgrößen
über `data.target_cols`. TFT hat eine eigene Pipeline (`prepare_data_for_tft()`,
Trennung `observed`/`known`/`static`).

**Solar-Stand (Aug 2026):** CL/FL-Pfad läuft; der GNN-Pfad ist durch Messlücken
blockiert und braucht zuerst die Solar-Imputation.

## Wo nachschlagen (`docs/`, bei Bedarf lesen)

**Stand & Übergaben** — [handoff.md](docs/handoff.md) (**AKTUELL, Sep 2026**: neun
finale Testläufe auf l1/l2/ws, beobachten/stoppen, Imputationskette, offene
Stolperfallen), [handoff_testmode.md](docs/handoff_testmode.md),
[expanding_window_retrain_handoff.md](docs/expanding_window_retrain_handoff.md)

**Imputation** — [imputation_tft_switch.md](docs/imputation_tft_switch.md) (Sep 2026:
`wind_speed` auf TFT-Werte, `IMPUTATION_GUARD_VERSION` 3→4, `kontextfrei`-Konvention,
offener Punkt `dcrnn.interpolate_history`),
[imputation_knn_regen_20260902.md](docs/imputation_knn_regen_20260902.md) (KNN-Cache
`wind_direction`, 203→204 Stationen), [imputation_richtung_tft_20260903.md](docs/imputation_richtung_tft_20260903.md),
[imputation_plausibility_guard.md](docs/imputation_plausibility_guard.md),
ERA5-Vorgängerstände: `imputation_era5_{switch,only,comparison}.md`

**Preprocessing & Daten** — [preprocess_icond2_wind.md](docs/preprocess_icond2_wind.md)
(ML-Struktur, Luftdichte, Lookback/Horizon), [preprocess_icond2_solar.md](docs/preprocess_icond2_solar.md)
(SL, J/cm²→W/m², lon-first-Dateinamen, `data.freq` 15 min vs. 1 h),
[predict_wind.md](docs/predict_wind.md) (`wind_speed`/`power`, `extrapolate`/Power Law,
Skill_NWP, MultiIndex-Fallstricke), [predict_solar.md](docs/predict_solar.md)
(Zeitraster 15/10/60 min, Akkumulationssemantik), [data.md](docs/data.md),
[icond2_database_integration.md](docs/icond2_database_integration.md),
[spatial_interpolation.md](docs/spatial_interpolation.md)

**Modelle, Baselines, Ablationen** — [train_dcrnn.md](docs/train_dcrnn.md) (CLI,
`--hpo-study`, Output-Format, Architektur-Updates Mai 2026),
[baselines_implementation_spec.md](docs/baselines_implementation_spec.md),
[baselines_verification_results.md](docs/baselines_verification_results.md),
[chronos2.md](docs/chronos2.md), [implementation_plan_ablations.md](docs/implementation_plan_ablations.md),
[ablations_verification_results.md](docs/ablations_verification_results.md)

**Federated Learning** — Clients sind Stationsgruppen (`fl.clients`), Aggregation
FedAvg. [fine_tuning_feature.md](docs/fine_tuning_feature.md),
[early_stopping_config.md](docs/early_stopping_config.md) (FL vs. Fine-Tuning),
[global_early_stopping.md](docs/global_early_stopping.md) (über globale Runden)

**Splits & Evaluation** — [station_splits_solar.md](docs/station_splits_solar.md)
(21 Testsatz + 62 Pool, 3 rotierende Folds, Netzdichte 48 km),
[spatial_cv_implementation_prompt.md](docs/spatial_cv_implementation_prompt.md),
[evaluation_results.md](docs/evaluation_results.md), [study_overview.md](docs/study_overview.md)

**Dashboards** — [optuna_dashboard.md](docs/optuna_dashboard.md) (systemd, `OPTUNA_STORAGE`,
Study-Löschen mit Passwort), [fold_dashboard.md](docs/fold_dashboard.md) (räumliche Splits)

**Reviews & Pläne** — `review_round2_{findings,fixes}.md`, `topo_features_review_brief.md`,
`topo_rehpo_plan.md`, `solar_irradiance_plan.md`, `prompt_baselines_implementation.md`

## Wichtige Hinweise

- README.md erwähnt TensorFlow — das Framework nutzt **PyTorch**.
- Stationsdaten auf dem NAS: `/mnt/nas/synthetic/`.
- ICON-D2 standardmäßig aus PostgreSQL (`data.icond2_source: 'database'`),
  CSV-Fallback vorhanden. `WEATHER_DB_URL` muss in der `.bashrc` gesetzt sein —
  in systemd-Diensten ist sie unsichtbar (siehe `~/docs/services.md`).
- GPU 0 ist oft von anderen Prozessen belegt → in Configs ausschließen.
- `frcst/` ist das virtuelle Environment.
- **`DATA_ROOT`** muss gesetzt sein (`.bashrc`: l1 `/mnt/nvme1`, l2/ws `/mnt/lambda1/nvme1`).
  Datenpfade in Configs stehen als `!ENV '${DATA_ROOT}/…'`, nie als absoluter
  Mountpfad — sonst laufen die Hosts auseinander. Fehlt die Variable, bricht
  `load_config` laut ab; in bereits offenen Terminals `source ~/.bashrc`.
- Skalierung mit StandardScaler; der Scaler wird für den Inverse-Transform bei
  der Evaluation gespeichert.
