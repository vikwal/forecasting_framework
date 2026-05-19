# DCRNN Training — `geostatistics/train_dcrnn.py`

## Überblick

`train_dcrnn.py` trainiert ein **DCRNN Seq2Seq-Modell** (Diffusion Convolutional Recurrent Neural Network) zur räumlichen Windgeschwindigkeits-Interpolation. Das Modell nutzt einen autoregressiven Decoder mit NWP-Injection an jedem Schritt.

Typischer Workflow:

```
hpo_dcrnn.py  →  Optuna-Study (best params)
                          ↓
              train_dcrnn.py --hpo-study auto
                          ↓
              models/<name>.pt  +  results/<name>_<ts>.pkl
```

```bash
# Aufruf aus forecasting_framework/:
python geostatistics/train_dcrnn.py \
    --config configs/config_wind_stgcn.yaml \
    --suffix v1 \
    --hpo-study auto \
    --eval
```

---

## Argumente

| Argument | Typ | Default | Beschreibung |
|---|---|---|---|
| `--config` | str | *(Pflicht)* | Pfad zur YAML-Konfigurationsdatei |
| `--suffix` | str | `""` | Anhang an Modell- und Log-Dateinamen |
| `--hpo-study` | str | `None` | HPO-Studie laden (siehe unten) |
| `--eval` | flag | False | 4-pass LOO-Evaluation nach Training |

---

## HPO-Integration (`--hpo-study`)

`train_dcrnn.py` kann die optimalen Hyperparameter direkt aus einer abgeschlossenen `hpo_dcrnn.py`-Studie laden und damit die Werte aus der YAML-Config überschreiben.

### Verwendung

```bash
# Pfad zur SQLite-DB direkt angeben:
python geostatistics/train_dcrnn.py \
    --config configs/config_wind_stgcn.yaml \
    --hpo-study studies/hpo_dcrnn_wind_stgcn.db

# Oder automatisch ableiten (Standardpfad studies/hpo_dcrnn_<config_stem>.db):
python geostatistics/train_dcrnn.py \
    --config configs/config_wind_stgcn.yaml \
    --hpo-study auto
```

### PostgreSQL

Wenn die Umgebungsvariable `OPTUNA_STORAGE` gesetzt ist **und** `--hpo-study` angegeben wird, wird PostgreSQL automatisch bevorzugt:

```bash
export OPTUNA_STORAGE="postgresql://user:pw@host/db"
python geostatistics/train_dcrnn.py --config ... --hpo-study auto
```

### Logik

| `--hpo-study` | `OPTUNA_STORAGE` | Verhalten |
|---|---|---|
| nicht gesetzt | egal | HPO wird übersprungen, reine YAML-Config |
| gesetzt | nicht gesetzt | SQLite aus angegebenem Pfad (oder `auto`) |
| gesetzt | gesetzt | PostgreSQL via `OPTUNA_STORAGE` |

Der Study-Name wird automatisch aus der Config rekonstruiert (identische Konvention wie in `hpo_dcrnn.py`):

```
cl_m-dcrnn_out-{forecast_horizon}_freq-{data.freq}_{config_stem}
```

Alle überschriebenen Parameter werden im Log ausgegeben:

```
Overriding dcrnn_cfg with HPO best params:
  hidden                         64 → 128
  lr                             3.0e-4 → 8.5e-5
  diffusion_K                    2 → 3
```

---

## Konfiguration

Die `dcrnn`-Section in der YAML-Config. Alle Werte können durch HPO überschrieben werden.

```yaml
dcrnn:
  # Features
  icond2_features: ['u_10m', 'v_10m', 'wind_speed_10m']
  ecmwf_features:  ['u_wind10m', 'v_wind10m', 'wind_speed_10m']
  measurement_features: ['wind_speed', 'wind_direction']
  target_col: 'wind_speed'
  icond2_feature_mode: both   # absolute | components | both
  ecmwf_feature_mode:  both

  # Sequenzlängen
  history_length: 48          # Vergangenheits-Schritte
  forecast_horizon: 48        # Vorhersage-Schritte

  # Modell
  hidden: 128
  num_layers: 2
  diffusion_K: 2
  dropout: 0.1
  teacher_forcing_ratio: 0.5  # Startwert, zerfällt linear auf 0.0
  temporal_encoding: gru

  # Graph
  station_connectivity: delaunay
  next_n_icond2: 4
  next_n_ecmwf: 4
  use_altitude_diff: true
  neighbor_radius_km: 500

  # Sampling
  min_target_stations: 1
  max_target_stations: 10
  max_neighbor_stations: 60

  # Loss
  loss_fn: mse
  loss_weights_by_horizon: true
  horizon_decay: 0.95

  # Optimierer
  lr: 3.0e-4
  weight_decay: 1.0e-5
  scheduler: cosine
  batch_size: 8
  max_epochs: 200
  patience: 15
  gradient_clip: 1.0

  # Daten
  icond2_run_hours: [6, 9, 12, 15]
  n_workers: 16
```

### `icond2_feature_mode` / `ecmwf_feature_mode`

Filtert die NWP-Feature-Liste nach Winddarstellung:

| Modus | Beibehaltene Features |
|---|---|
| `both` | alle (Standard) |
| `absolute` | nur `wind_speed_*` |
| `components` | nur `u_*`, `v_*` |
| *(andere Features wie Temperatur werden immer behalten)* | |

---

## Datenpipeline

Identische Funktionen wie `hpo_dcrnn.py` — alle aus `train_stgnn2.py` importiert.

```
Stationsmessungen     load_station_measurements(...)   → (T, N, M)
Interpol-Imputation   load_interpol_imputation(...)    → NaN-Füllung in target_col
KNN-Imputation        load_knn_imputation(...)         → NaN-Füllung in wind_direction
Stationsmetadaten     load_station_metadata(...)       → lats, lons, alts
ICON-D2 ML runs       load_icond2_ml_runs(...)         → (R, 48, N_grid, I2)
ECMWF NWP             load_ecmwf_parquet_at_stations_and_grid(...)
NWP-Höhen             load_nwp_elevations(...)         → bei use_altitude_diff=true
```

### Temporaler Split

```
test_start  →  split_t (absoluter Index)

[0 ─────────────── split_t ─────────────────── T]
 ←───── Training ──────────→←───── Validation ──→
```

- `test_start` in der Config → `split_t = searchsorted(timestamps, test_start)`
- Fallback: `split_t = int(T * (1 - val_frac))`

### Skalierung

Scaler werden ausschließlich auf `[:split_t, :N_train]` gefittet (kein Data Leakage):

| Daten | Scaler |
|---|---|
| Messdaten `(T, N, M)` | `meas_scaler.fit([:split_t, :N_train])` |
| ICON-D2 Runs | `i2_scaler.fit(runs[train_r_mask])` |
| ECMWF Stationswerte | `e2_scaler.fit([:split_t, :N_train])` |
| Statische Features | `stat_scaler.fit([:N_train])` |

### Run-Pair-Konstruktion

Pro ICON-D2-Run `r` wird ein Tupel `(r_curr, r_hist, t_run_abs)` erstellt:

- `r_curr`: Index des aktuellen Runs (Forecast-Start)
- `r_hist`: Index des historisch passenden Runs (~`H` Stunden vorher)
- `t_run_abs`: absoluter Zeitstempel-Index in `timestamps`

Ein Run-Pair wird übersprungen wenn:
- `t_run` nicht in `timestamps`
- Fenster [t-H, t+F] über Array-Grenzen hinausgeht
- NaN in Messdaten im Fenster vorhanden

Paare werden nach `split_time` in `train_run_pairs` / `val_run_pairs` aufgeteilt.

---

## Modell & Trainer

```
DCRNN(DCRNNConfig)
  Encoder:
    - Diffusionskonvolution: BiDirDiffConv (separate Forward/Backward für s2s)
    - NWP-Attention: per Timestep mit echtem Hidden State H[-1] als Query
    - DCGRU: K Hops über Station-Graph

  Decoder:
    - Autoregressiv: y_{t-1} → nächster Vorhersageschritt
    - NWP-Injection: per Schritt, zeitabhängig
    - Teacher Forcing Ratio: linear zerfallend von teacher_forcing_ratio → 0.0
```

**Architektur-Aktualisierungen (Mai 2026):**

Siehe [`docs/dcrnn_implementation_fixes.md`](dcrnn_implementation_fixes.md) für Details. Zusammengefasst:

| Komponente | Änderung | Grund |
|---|---|---|
| **s2s DiffConv** | Unidirektional → BiDirDiffConv | DCRNN Paper verlangt separate Forward/Backward-Gewichte |
| **NWP Attention** | Zero-Query → Echtes H[-1] | MultiModal Paper: Attention sollte vom Station-Zustand abhängen |
| **Encoder/Decoder** | Pre-computed (1 GATv2 Call) → Per-Timestep (96 Calls) | Zeitinformation vollständig erhalten, statt Batch-Optimierung |

Konfiguration über `DCRNNConfig.from_yaml(dcrnn_cfg, ...)`. Der Trainer (`DCRNNTrainer`) übernimmt:
- Early Stopping (Patience)
- Best-Checkpoint-Sicherung
- TensorBoard-Logging unter `runs/<model_name>/`

**Performance-Hinweis:** Die NWP-Attention ruft jetzt per Timestep auf (96× statt 1× pro Sample). Training ist ~5-15% langsamer, aber Modellqualität sollte besser sein.

---

## Outputs

### Modell

```
models/<config_stem>_dcrnn[_<suffix>].pt
```

Gespeichert als PyTorch State-Dict mit Early-Stopping-Best-Checkpoint.

### Ergebnisse (Pickle)

```
results/<config_stem>_dcrnn[_<suffix>]_<YYYYMMDD_HHMMSS>.pkl
```

Das Pickle enthält ein Dictionary:

| Key | Inhalt |
|---|---|
| `model_name` | Dateiname-Stem des Modells |
| `model_path` | Absoluter Pfad zur `.pt`-Datei |
| `architecture` | `"dcrnn"` |
| `config` | `dcrnn_cfg` nach HPO-Überschreibung |
| `train_ids` | Trainings-Stations-IDs |
| `val_ids` | Validierungs-Stations-IDs |
| `n_train` / `n_val` | Anzahl Stationen |
| `history` | Trainings-History (loss pro Epoche) |
| `best_val_loss` | Bester Val-Loss |
| `stopped_epoch` | Epoche, in der Early Stopping auslöste |
| `evaluation` | `pd.DataFrame` (nur mit `--eval`, sonst `None`) |
| `hpo_study_name` | Name der Optuna-Studie (`None` ohne `--hpo-study`) |
| `hpo_best_params` | Dict der besten HPO-Parameter (`None` ohne `--hpo-study`) |
| `hpo_best_val_loss` | Bester Val-Loss aus der HPO-Studie (`None` ohne `--hpo-study`) |

### Logs

```
logs/train_dcrnn_<config_stem>[_<suffix>].log
```

---

## Evaluation (`--eval`)

4-pass Leave-One-Out-Evaluation auf den Val-Run-Pairs nach dem Training:

1. Besten Checkpoint aus `model_path` laden
2. `evaluate()` aus `geostatistics/evaluation.py` aufrufen
3. Ergebnis-DataFrame in Pickle unter `evaluation` speichern

Die Evaluation ist optional — Standardlauf ohne `--eval` spart Zeit, wenn nur das Modell gefragt ist.

---

## Unterschiede zu `hpo_dcrnn.py`

| Aspekt | `hpo_dcrnn.py` | `train_dcrnn.py` |
|---|---|---|
| Ziel | Hyperparameter-Suche | Finales Training mit festen Params |
| Validation | Time-Series CV (mehrere Folds) | Einmaliger Train/Val-Split |
| Scaler-Fitting | Pro Fold neu, auf Fold-Trainings-Fenster | Einmalig auf `[:split_t]` |
| Modell-Speicherung | Temporäre Checkpoints (gelöscht nach Fold) | Persistentes `.pt` |
| Ergebnis-Speicherung | Optuna-DB | Pickle-Dict |
| Caching | `GNNCache` für schnelle Restarts | Kein Caching |
| Evaluation | Keine | Optional via `--eval` |
