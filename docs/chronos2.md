# Chronos-2 Integration

## Überblick

[Chronos-2](https://huggingface.co/amazon/chronos-2) ist ein Foundation Model für Zeitreihenprognosen von Amazon (Oktober 2025).
Es unterstützt nativ **univariate, multivariate und kovariate Vorhersagen** im Zero-Shot-Modus.

Für dieses Framework relevante Eigenschaften:
- 120M Parameter, encoder-only Architektur mit Group Attention
- Bekannte Zukunfts-Kovariaten (NWP-Features) werden nativ unterstützt
- Robuste interne Normalisierung — der Target (Power) wird **nicht extern skaliert**
- Zero-Shot-Einsatz ohne Training möglich; optional Fine-Tuning via `fit()`

---

## Installation

```bash
source frcst/bin/activate
pip install chronos-forecasting
```

---

## Konfiguration

Chronos-spezifische Parameter werden unter `model.chronos` in der bestehenden Config angelegt.
**Keine Dopplungen** — gemeinsame Parameter (Horizon, Lookback, LR, Batch-Size) werden aus den
bereits vorhandenen `model.*`-Feldern bezogen.

```yaml
model:
  # ... bestehende Parameter (epochs, lr, batch_size, lookback, horizon …) …
  chronos:
    repo_id: 'amazon/chronos-2'       # HuggingFace model ID oder lokaler Pfad
    device_map: 'cuda'                 # 'cuda' oder 'cpu'
    fine_tune_mode: 'full'             # 'full' oder 'lora'
    lora_config: null                  # nur relevant bei fine_tune_mode: 'lora'
    num_steps: null                    # null → fällt auf model.epochs zurück
    quantile_levels: [0.5]             # [0.5] = deterministischer Median-Forecast
    cross_learning: false              # true = gemeinsame Gruppen-Attention über Stationen
    output_dir: 'models/'
    finetuned_ckpt_name: 'finetuned-ckpt'
    # Mapping zu Chronos-2 fit()-Parametern:
    #   prediction_length  ← model.horizon
    #   context_length     ← model.lookback
    #   learning_rate      ← model.lr
    #   batch_size         ← model.batch_size
    #   num_steps          ← model.chronos.num_steps  oder  model.epochs
```

---

## Verwendung

### Zero-Shot-Evaluation (`get_test_results.py`)

Ohne vortrainiertes lokales Modell — Chronos-2 direkt von HuggingFace:

```bash
python get_test_results.py \
    --model chronos \
    --config configs/config_wind_80cl.yaml \
    --zero-shot
```

Mit lokal fine-getuntem Modell (falls vorhanden, sonst automatischer Fallback auf Zero-Shot):

```bash
python get_test_results.py \
    --model chronos \
    --config configs/config_wind_80cl.yaml
```

**Modell-Lookup-Logik:**
1. `--zero-shot` → lädt immer von `amazon/chronos-2`
2. Default (kein `--zero-shot`) → sucht `models/chronos_<station_id>/`
3. Falls kein lokales Modell → `logging.warning` + Fallback auf Zero-Shot

Ergebnisse werden in `data/test_results/test_results_chronos.csv` gespeichert.

### Fine-Tuning (`train_cl.py`)

```bash
python train_cl.py \
    -m chronos \
    -c configs/config_wind_80cl \
    --save_model
```

Das fine-getunete Modell wird per `pipeline.save_pretrained()` gespeichert unter:
`models/chronos_cl_m-chronos_out-<N>_freq-<freq>_<suffix>/`

Das Verzeichnis kann anschließend direkt mit `--config` in `get_test_results.py` genutzt werden.

---

## Daten-Pipeline

### `prepare_data_for_chronos2()` (`utils/preprocessing.py`)

Thin Wrapper um `prepare_data_for_tft()` mit `scale_target=False`.
Gibt das gleiche Dict-Format zurück wie TFT:

```
X_train / X_test:
  'observed': (n_windows, lookback, 1)         — unscalierter Power-Verlauf
  'known':    (n_windows, lookback+horizon, K)  — skalierte NWP-Features
  'static':   (n_windows, S)                    — Anlagenparameter (optional)

y_train / y_test: (n_windows, horizon)          — unscalierter Ziel-Power
```

Die NWP-Features (`known_features`) werden skaliert (StandardScaler), der Target bleibt
unscaliert, da Chronos-2 intern eine eigene robuste Normalisierung durchführt.

Routing in `pipeline()`:
```python
config['model']['name'] = 'chronos'  # aktiviert diesen Zweig
```

### `get_y_chronos2()` (`utils/tools.py`)

Konvertiert die numpy-Arrays aus `prepare_data_for_chronos2()` in das
Long-Format-DataFrame-Format von `Chronos2Pipeline.predict_df()`:

- `context_df`: alle Windows als separate `item_id`s mit Power-History
- `future_df`: NWP-Features für den Forecast-Horizont (Spalten = `known_features`)
- Gibt `(y_true, y_pred)` als numpy-Arrays zurück, kompatibel mit `eval.py`

---

## Fine-Tuning Details

### `fit()` Methode (Chronos-2 nativ)

```python
pipeline.fit(
    inputs,             # Liste von 1D-Arrays: [obs_context + target] pro Window
    prediction_length,  # = model.horizon
    finetune_mode,      # 'full' oder 'lora'
    context_length,     # = model.lookback
    learning_rate,      # = model.lr
    num_steps,          # = model.chronos.num_steps || model.epochs
    batch_size,         # = model.batch_size
    output_dir,         # Zwischen-Checkpoints
)
```

**Format der Fine-Tuning-Inputs (multivariate mit NWP):**

```python
{
    'target': np.concatenate([observed_context, y_target]),  # shape: (lookback+horizon,)
    'past_covariates': {col: known_full[i, :, k] for k, col in enumerate(known_cols)},
    # known_full shape: (n, lookback+horizon, n_known) — voller Zeitraum
    'future_covariates': {col: None for col in known_cols},
    # None-Werte signalisieren "zukunftsbekannt"; Chronos-2 lernt beim Slicen aus past_covariates
}
```

Chronos-2 nutzt in TRAIN-Mode die `future_covariates`-Werte nicht direkt —
sie signalisieren nur welche Features als "bekannte Zukunft" behandelt werden.
Die eigentlichen Zukunftswerte der Kovariaten werden aus `past_covariates` (gesamtes Fenster) gesliced.

### LoRA-Fine-Tuning

Für speichereffizientes Fine-Tuning:
```yaml
chronos:
  fine_tune_mode: 'lora'
  lora_config:
    r: 8
    lora_alpha: 16
    target_modules: ['q_proj', 'v_proj']
    lora_dropout: 0.05
```

---

## Modell-Speicherung & Laden

Gespeichert als HuggingFace-Verzeichnis (safetensors + config.json):

```
models/
  chronos_cl_m-chronos_out-48_freq-1h_wind_80cl/
    config.json
    model.safetensors
    ...
```

Laden für Inferenz:
```python
from chronos import BaseChronosPipeline
pipeline = BaseChronosPipeline.from_pretrained('models/chronos_.../', device_map='cuda')
```

---

## Federated Learning (Phase 2)

Chronos-2 Integration in `train_fl.py` und `federated.py` ist für einen späteren Schritt geplant.
Relevante Überlegungen:
- `federated.py` FedAvg aggregiert `state_dict()` — Chronos-2 unterstützt dies via `model.model.state_dict()`
- Alternativ: lokales Fine-Tuning pro Client ohne Aggregation (personalisiertes FL)

---

## Bekannte Limitierungen

- `cross_learning=false` Default: jedes Station-Window wird unabhängig behandelt
- Große Anzahl Test-Windows (n > 1000) → hohes RAM für DataFrame-Konstruktion in `get_y_chronos2()`
- HPO (`hpo_cl.py`) ist für Chronos nicht sinnvoll (Zero-Shot / einfaches Fine-Tuning)
