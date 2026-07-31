# Spatiotemporal GNN (STGNN) — `geostatistics/train_stgnn.py`

## Überblick

`train_stgnn.py` implementiert ein **Spatiotemporal Graph Neural Network** zur räumlichen Windgeschwindigkeits-Interpolation an unbekannten Stationen. Jedes Training-Sample ist eine **96-Stunden-Sequenz** (48h Vergangenheit + 48h Zukunft). Das Modell lernt, Messwerte der Nachbarstationen aus der Vergangenheit zu nutzen, um die Windgeschwindigkeit an einer unbekannten Zielstation in der Zukunft vorherzusagen.

```bash
# Aufruf aus forecasting_framework/:
python geostatistics/train_stgnn.py --config configs/config_wind_stgcn.yaml

# Argumente:
#   --config PATH         YAML-Konfigurationsdatei (erforderlich)
#   -s SUFFIX             Anhang ans Output-Prefix
#   --overlap-mode        shortest_lead (default) | mean
#   --log-level           INFO | DEBUG | WARNING
```

---

## Konfiguration (`config_wind_stgcn.yaml`)

Die Config enthält mehrere Sections. Für den STGNN relevant sind:

### `data:`

```yaml
data:
  path: '/mnt/lambda1/nvme1/synthetic/wind/era5_fric_wind_hourly_age'
  files: ['00298', '02907', ...]     # Trainingsstationen (DWD-IDs)
  val_files: ['05930', '07395', ...] # Teststationen — nie im Training gesehen
  test_start: '2025-08-01'           # Zeitfenster für Training UND Evaluation
  test_end:   '2025-10-31'
```

- `files` = Trainingsstationen, `val_files` = Teststationen. Kein zufälliger Split.
- `test_start`/`test_end` filtert den Zeitraum für **alle** Stationen (Train + Test).
- `data.nwp_path` wird vom STGNN nicht direkt verwendet — NWP-Daten kommen über `preprocess_synth_wind_icond2` aus `data.path`.

### `params:` — Feature-Steuerung

```yaml
params:
  measurement_features:
    - wind_speed        # Pflicht: Erstes Feature = Zielgröße
    # - power           # Optional: weitere Messwert-Features (nur im observed Zeitraum)

  nwp_features:
    - wind_speed_h10    # ICON-D2, 10 m Höhe (nächster Gitterpunkt)
    - wind_speed_h38    # ICON-D2, 38 m Höhe
    - ecmwf_wind_speed_h10  # ECMWF, 10 m

  static_features:
    - altitude          # Aus wind_parameter.csv (m ü. NN)
    - latitude          # Aus Metadaten
    - longitude         # Aus Metadaten
    # - hub_height, rotor_diameter, etc.

  # Preprocessing-Parameter für preprocess_synth_wind_icond2:
  next_n_grid_points: 1     # Nur nächster NWP-Gitterpunkt
  next_n_grid_ecmwf: 1      # Nur nächster ECMWF-Gitterpunkt
  nwp_models: ['icon-d2', 'ecmwf']
  ecmwf_features: ['u_wind10m', 'v_wind10m']
  aggregate_nwp_layers: 'weighted_mean'
  get_density: True
```

**`measurement_features`**: Spalten aus `synth_{sid}.csv` die Messwerte enthalten. Das erste Feature ist die Vorhersagegröße. Alle werden mit `mask` multipliziert (d.h. im Zukunftsbereich und für den Zielknoten = 0).

**`nwp_features`**: NWP-abgeleitete Features. Werden über `preprocess_synth_wind_icond2` aus ICON-D2/ECMWF-Rohdaten berechnet. Erkennungsregeln:
- Enthält `_h` (z.B. `wind_speed_h10`, `wind_speed_h38`) → ICON-D2-Verarbeitung
- Beginnt mit `ecmwf_` → ECMWF-Verarbeitung
- Endet auf `_rotor_eq` oder heißt `density` → NWP-Verarbeitung
- Alles andere → direkt aus `synth_{sid}.csv` gelesen

**`static_features`**: Zeitinvariante Node-Features. `latitude`, `longitude`, `altitude` werden direkt aus den bereits geladenen Metadaten genommen; andere Namen werden in `wind_parameter.csv` gesucht.

### `stgnn:` — Modell & Training

```yaml
stgnn:
  k_neighbors: 30              # k-NN Graph (Kanten pro Node)
  val_fraction: 0.2            # Letzter Anteil des Zeitfensters für Early Stopping

  seq_len: 96                  # Fensterlänge gesamt (past_len + forecast_horizon)
  forecast_horizon: 48         # Vorhersage-Schritte → past_len = seq_len - forecast_horizon

  hidden: 128                  # Versteckte Dimension (muss durch heads teilbar sein)
  heads: 4                     # GATv2 Multi-Head Attention
  num_layers: 3                # Anzahl ST-Blöcke
  temporal_kernel_size: 3      # Conv1d Kernel-Größe entlang der Zeitachse
  dropout: 0.1

  lr: 1.0e-3
  weight_decay: 1.0e-5
  batch_size: 32               # Klein wegen großer Sequenzen
  max_epochs: 100
  patience: 15                 # Early-Stopping-Patience (Epochen)
  num_workers: 4               # DataLoader Worker
```

### `output:`

```yaml
output:
  path: 'results/geostatistics'  # Ausgabeverzeichnis
```

---

## Datenpipeline

### Übersicht

```
load_data(_load_cfg, rk_features=None)
  → pivot (T, N_train): wind_speed + Koordinaten/Metadaten
  → Zeitfilterung via test_start / test_end

load_measurement_features(data_path, station_ids, measurement_features, timestamps)
  → {name: (T, N)} direkt aus synth_{sid}.csv

load_nwp_feature_matrices(config, station_ids, nwp_features, timestamps)
  → {name: (T, N)} via preprocess_synth_wind_icond2 (NWP) oder CSV-Fallback

load_static_features(data_path, station_ids, static_features, lats, lons, alts)
  → (N, S) aus Metadaten / wind_parameter.csv

── Identisch für val_files ──────────────────────────────────────────
load_data(val_config)              → val_pivot, val_lats, val_lons, val_alts
load_nwp_feature_matrices(...)     → nwp_val_raw {name: (T, N_val)}
```

### Warum `rk_features=None`?

`load_data` liest normalerweise `interpolation.rk_features` aus der Config (für Regression Kriging) und droppt Timestamps mit NaN in diesen Features. Der STGNN braucht das nicht — daher wird beim Aufruf `"rk_features": None` gesetzt, damit nur `wind_speed` + Metadaten geladen werden und kein Timestamp-Dropping stattfindet.

### NWP-Feature-Loading (`load_nwp_feature_matrices`)

Für jeden Feature-Namen wird geprüft, ob er NWP-typisch ist (enthält `_h`, beginnt mit `ecmwf_`, etc.). Falls ja:

1. `preprocess_synth_wind_icond2` wird für jede Station aufgerufen mit `next_n_grid_points=1` (nur nächster Gitterpunkt)
2. Das Ergebnis hat Spalten mit Suffix `_1` (z.B. `wind_speed_h10_1`) → Suffix wird für den internen Namen entfernt
3. Fallback auf `load_feature_matrices` (direktes CSV-Lesen) für nicht-NWP-Features

```
z.B. Config:  nwp_features: ['wind_speed_h10']
preprocess → Spalte 'wind_speed_h10_1' im DataFrame
→ intern gespeichert als 'wind_speed_h10'
```

### Stationssplit

Explizit aus der Config — kein zufälliger Split:

```
data.files     → Trainingsstationen (80 Stationen)
data.val_files → Teststationen     (80 Stationen)
```

Die Teststationen werden komplett separat geladen (`load_data(val_config)`) und dienen **nur** für die finale Inferenz, nie für Training oder Early Stopping.

### Temporaler Split

```
Zeitfenster [test_start, test_end]:

[─────────────── split_t = T × (1 - val_fraction) ───────][── val_t ──]
  Training (sliding windows, LOO-Masking)                   Early Stopping
```

- `split_t = int(T * (1 - val_fraction))`
- Beide Teile verwenden **dieselben Trainingsstationen**
- Skalierung wird **nur** auf `[:split_t]` gefittet → kein Data Leakage

---

## Skalierung

Pro Feature-Typ ein separater `StandardScaler`, je auf dem Training-Anteil `[:split_t]` gefittet:

| Feature-Typ | Scaler | NaN-Handling |
|---|---|---|
| `measurement_features` | Ein Scaler pro Feature, fit auf beobachtete Werte in `[:split_t]` | NaN → 0 vor Skalierung; `ws_valid`-Maske speichert Original-NaN |
| `nwp_features` | Ein Scaler pro Feature, fit auf `[:split_t]` inkl. Nullen | NaN → 0 vor Skalierung; NaN-Positionen nach Skalierung wieder auf 0 |
| `static_features` | Ein gemeinsamer Scaler, fit auf Trainingsstationen | Keine NaN erwartet |

Der Scaler des ersten `measurement_features`-Eintrags (`ws_sc`) wird für die Inverse-Transformation der Modell-Outputs verwendet.

---

## Feature-Vektor

Pro Node pro Zeitschritt (F = M + K + 1 + S):

```
[ meas_1 × mask | meas_2 × mask | ... | nwp_1 | nwp_2 | ... | mask | static_1 | ... ]
  ←────────────── M ─────────────────→ ←──── K ────────────→   ↑    ←───── S ──────→
                                                                 Einzel-Maske
```

**M = `len(measurement_features)`** — z.B. 1 (`wind_speed`) oder 2 (`wind_speed`, `power`)  
**K = `len(nwp_features)`** — z.B. 3 (`wind_speed_h10`, `wind_speed_h38`, `ecmwf_wind_speed_h10`)  
**S = `len(static_features)`** — z.B. 3 (`altitude`, `latitude`, `longitude`)

### Masking-Logik

Die `mask`-Spalte ist eine binäre Zahl (0 oder 1) **pro Node pro Zeitschritt**:

| Situation | `mask` | `meas_i` |
|---|---|---|
| Nachbar-Node, Vergangenheit (t < past\_len), Messwert beobachtet | 1 | skalierter Messwert |
| Nachbar-Node, Vergangenheit, Messwert fehlend (NaN) | 0 | 0 |
| Nachbar-Node, Zukunft (t ≥ past\_len) | 0 | 0 |
| Ziel-Node, alle Schritte | 0 | 0 |

NWP-Features sind für **alle Nodes, alle Zeitschritte** verfügbar (keine Maskierung). Die Maske gilt nur für Messwerte.

### Zielgröße `y`

Das erste `measurement_features`-Element (typischerweise `wind_speed`) am Ziel-Node für die Forecast-Horizon-Schritte:

```python
y = measurements[0][w + past_len : w + seq_len, s]  # (forecast_horizon,)
```

---

## Modellarchitektur

### Überblick

```
Input: (B, N, seq_len, F)
    │
    ▼
input_proj: Linear(F → H)          →  (B, N, seq_len, H)
    │
    ▼  × num_layers
┌──────────────── ST-Block ──────────────────────────────┐
│                                                         │
│  Temporal: Conv1d(H, H, kernel_size, padding='same')   │
│    (B,N,T,H) → reshape(B*N, H, T) → Conv1d → zurück    │
│    + Residual + LayerNorm                               │
│                                                         │
│  Spatial: GATv2Conv                                     │
│    (B,N,T,H) → permute+reshape(B*T, N, H)              │
│    → make_mega_batch → GATv2Conv → reshape zurück       │
│    + Residual + LayerNorm                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Extraktion Ziel-Node, Forecast-Schritte:
    x[batch, target_idx, past_len:, :]  →  (B, forecast_horizon, H)
    │
    ▼
head: Linear(H → H/2) → GELU → Dropout → Linear(H/2 → 1) → squeeze
    →  (B, forecast_horizon)
```

### ST-Block im Detail

**Temporal Sub-Block:**
- Verarbeitet jeden Node unabhängig entlang der Zeitachse
- `(B, N, T, H)` → `.contiguous().reshape(B*N, T, H)` → `.permute(0,2,1)` → `(B*N, H, T)`
- `Conv1d(H, H, kernel_size=temporal_kernel_size, padding='same')` → gleiche Zeitlänge
- Zurück nach `(B, N, T, H)` + Residual + LayerNorm

**Spatial Sub-Block:**
- Wendet GATv2Conv für jeden Zeitschritt über alle Nodes an
- `(B, N, T, H)` → `.permute(0,2,1,3).contiguous().reshape(B*T, N, H)`
- `make_mega_batch` stapelt B×T unverbundene Graphen zu einem Mega-Graph
- GATv2Conv verarbeitet alle Zeitschritte in einem Durchlauf
- Zurück nach `(B, N, T, H)` + Residual + LayerNorm

### Graph-Topologie

Statisch — gleich für alle Zeitschritte und Batch-Elemente:
- **Kanten**: gerichteter k-NN Graph (Nachbar → Node)
- **Kantenfeatures**: `[dist_norm, sin(bearing), cos(bearing)]` (3-dimensional)
- `edge_dim = 3`

---

## Training

### Epoch-Loop

`run_epoch` gibt `(avg_loss, rmse, r2)` zurück:
- **Loss**: MSE über alle 48 Forecast-Horizon-Schritte × alle Nodes im Batch
- **Gradient Clipping**: `max_norm=1.0`
- **Metriken**: RMSE und R² werden flach über alle Samples und Horizont-Schritte berechnet

### Logging pro Epoche

```
Epoch   5/100  train_loss=0.1234  train_rmse=0.3512  train_r2=0.823
               val_loss=0.1456    val_rmse=0.3815    val_r2=0.798  lr=1.00e-03
```

### Early Stopping & LR-Scheduler

- Early Stopping auf `val_loss`, Patience = `stgnn.patience`
- ReduceLROnPlateau: Patience = `stgnn.patience // 3`, Faktor 0.5, min LR = 1e-6
- Bester Model-State wird am Ende geladen

---

## Inferenz an Teststationen (`predict_test_station`)

### Vorgehen

1. Teststation als neuer Node `test_node_idx = N_train` an Trainingsgraph anhängen
2. `build_test_edges`: k nächste Trainingsstationen → Teststation (Kanten)
3. Alle `T - seq_len + 1` gültigen 96-Schritt-Fenster batched verarbeiten
4. Teststation hat überall `mask=0`, `meas=0` — simuliert unbekannte Station
5. Output: `(B, forecast_horizon)` Vorhersagen

### Überlappende Vorhersagen

Für jeden absoluten Zeitstempel können mehrere Fenster eine Vorhersage liefern:

```
--overlap-mode shortest_lead  (default):
    Nimmt die Vorhersage mit dem kleinsten Forecast-Step-Index h
    = spätestes Fenster = kürzester Lead Time

--overlap-mode mean:
    Mittelt alle Vorhersagen für diesen Zeitstempel
```

### Zeitliche Abdeckung

```
Zeitstempel 0 bis past_len-1 (= 47):  NaN — kein Fenster erzeugt Vorhersage hier
Zeitstempel past_len bis T-1:          abgedeckt durch Fenster-Sliding
```

### NWP-Baseline (Skill Score)

Falls `wind_speed_h10` in `nwp_features` enthalten: automatische Berechnung von `Skill_NWP = 1 - (RMSE_model / RMSE_nwp_h10)` pro Teststation.

---

## Outputs

Alle Dateien in `output.path` mit Prefix `{config_stem}_stgnn` (z.B. `wind_stgcn_stgnn`):

| Datei | Inhalt |
|---|---|
| `*_model.pt` | Model-State-Dict + alle Hyperparameter |
| `*_station_split.csv` | Station-IDs mit Label `train`/`test` |
| `*_predictions.csv` | `station_id, timestamp, wind_speed_observed, stgnn_pred` |
| `*_results_per_station.csv` | `station_id, method, rmse, mae, r2, skill_nwp` pro Teststation |
| `*_results_summary.csv` | Mittelwerte über alle Teststationen |

Log-Datei: `logs/{prefix}.log`

---

## Wiederverwendete Komponenten aus `train_gnn.py`

| Funktion | Beschreibung |
|---|---|
| `build_knn_graph(dist_matrix, k)` | Gerichteter k-NN Graph |
| `build_edge_attr(dist_matrix, lats, lons, edge_index)` | Kanten-Features + max_dist |
| `build_test_edges(...)` | Kanten von k nächsten Trainingsstationen zur Teststation |
| `make_mega_batch(x_batch, edge_index, edge_attr)` | B Graphen → 1 Mega-Graph |

---

## Unterschiede zu `train_gnn.py`

| Aspekt | `train_gnn.py` | `train_stgnn.py` |
|---|---|---|
| Sample-Shape | `(N, F)` — ein Zeitschritt | `(N, 96, F)` — Sequenz |
| Temporale Modellierung | keine | Conv1d pro Node |
| Zielgröße | Skalar | Vektor (48 Schritte) |
| NWP-Features | hardcoded ICON-D2 ws | konfigurierbar via `params.nwp_features` |
| Messwert-Features | hardcoded `wind_speed` | konfigurierbar via `params.measurement_features` |
| Stationssplit | zufällig (seed + fraction) | explizit via `files`/`val_files` |
| Masking | Ziel-Node alle Schritte | Ziel-Node + alle Nodes in Zukunftsschritten |
| Metriken Training | nur Loss | Loss + RMSE + R² |
| NWP-Baseline | nicht berechnet | Skill_NWP wenn `wind_speed_h10` vorhanden |
