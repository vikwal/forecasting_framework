# Early Stopping Konfiguration im Federated Learning

## Zwei separate Early Stopping Konfigurationen

### 1. FL Early Stopping (`fl.early_stopping`)
**Zweck:** Steuert Early Stopping während der FL-Trainingsrunden

**Konfiguration:**
```yaml
fl:
  n_local_epochs: 10          # Wird NUR verwendet wenn early_stopping.enabled = False
  early_stopping:              # FL-spezifisches Early Stopping
    enabled: True              # Wenn True: ignoriert n_local_epochs
    patience: 10
    min_delta: 0.0001
    monitor: 'val_rmse'
    mode: 'min'
    restore_best_weights: True
```

**Verhalten:**
- **`enabled: True`**: Clients trainieren mit Early Stopping, `n_local_epochs` wird ignoriert
- **`enabled: False`**: Clients trainieren für genau `n_local_epochs` Epochen

### 2. Fine-Tuning Early Stopping (`model.early_stopping`)
**Zweck:** Steuert Early Stopping während der Fine-Tuning Phase (nach FL)

**Konfiguration:**
```yaml
model:
  epochs: 10                   # Wird für FL UND Fine-Tuning als Maximum verwendet
  early_stopping:              # Fine-Tuning Early Stopping
    enabled: True
    patience: 5
    min_delta: 0.0001
    monitor: 'val_rmse'
    mode: 'min'
    restore_best_weights: True
```

**Verhalten:**
- Wird NUR während Fine-Tuning verwendet (wenn `fl.fine_tune: True`)
- Die `model.epochs` wird als Maximum verwendet
- Early Stopping kann vorher stoppen

## Vollständiges Beispiel

```yaml
model:
  epochs: 50                    # Max Epochen (FL + Fine-Tuning)
  early_stopping:               # Für FINE-TUNING
    enabled: True
    patience: 5
    monitor: 'val_rmse'
    mode: 'min'

fl:
  n_local_epochs: 5             # Nur wenn fl.early_stopping.enabled = False

  early_stopping:               # Für FL-TRAINING
    enabled: True               # Wenn True: n_local_epochs wird ignoriert
    patience: 10
    monitor: 'val_rmse'
    mode: 'min'

  fine_tune: True               # Aktiviert Fine-Tuning nach FL
  fine_tune_epochs: 30          # Optional: Max Epochen für Fine-Tuning (überschreibt model.epochs)
```

## Training Flow

### Szenario 1: Mit FL Early Stopping
```yaml
fl:
  n_rounds: 10
  n_local_epochs: 5              # ❌ WIRD IGNORIERT
  early_stopping:
    enabled: True                # ✅ Aktiv
    patience: 10
```

**Ablauf:**
1. FL Round 1-10: Clients trainieren mit Early Stopping (max 50 Epochen aus `model.epochs`)
2. Aggregation nach jeder Runde
3. Fine-Tuning (wenn `fine_tune: True`): Early Stopping mit `model.early_stopping`

### Szenario 2: Ohne FL Early Stopping
```yaml
fl:
  n_rounds: 10
  n_local_epochs: 5              # ✅ WIRD VERWENDET
  early_stopping:
    enabled: False               # ❌ Inaktiv
```

**Ablauf:**
1. FL Round 1-10: Clients trainieren für **genau 5 Epochen** (aus `n_local_epochs`)
2. Aggregation nach jeder Runde
3. Fine-Tuning (wenn `fine_tune: True`): Early Stopping mit `model.early_stopping`

## Best Practices

**Für FL-Training:**
- Nutze `fl.early_stopping` für variable Trainingszeiten
- Oder nutze `n_local_epochs` für feste Epochenzahlen (z.B. 3-10)

**Für Fine-Tuning:**
- Aktiviere immer `model.early_stopping` um Overfitting zu vermeiden
- `patience: 3-10` je nach Datengröße
- Optional: Setze `fine_tune_epochs` niedriger als `model.epochs`
