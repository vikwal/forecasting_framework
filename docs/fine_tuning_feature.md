# Fine-Tuning Feature für Federated Learning

## Neue Config-Parameter

Füge folgende Parameter unter `fl:` in deiner Config hinzu:

```yaml
fl:
  clients:
    clientA: ['01766', '01200', '01544', '07403']
    clientB: ['15813', '04466', '01759', '02932']
    # ...
  strategy: 'fedavg'
  n_rounds: 10
  personalize: False

  # NEU: Fine-Tuning Parameter
  fine_tune: True              # Aktiviert lokales Fine-Tuning nach FL
  fine_tune_epochs: 50         # Max. Epochen für Fine-Tuning
  fine_tune_patience: 5        # Early Stopping Patience
```

## Wie es funktioniert

### Ohne Fine-Tuning (`fine_tune: False`):
1. FL-Training über N Runden
2. Globales Modell wird aggregiert
3. **Direkte Evaluation** mit globalem Modell

### Mit Fine-Tuning (`fine_tune: True`):
1. FL-Training über N Runden
2. Globales Modell wird aggregiert
3. **🆕 Fine-Tuning Phase**:
   - Jeder Client nimmt das globale Modell
   - Trainiert es auf seinen lokalen Daten
   - Mit Early Stopping (monitored `val_rmse`)
   - Max. `fine_tune_epochs` Epochen
   - Stoppt früher bei `fine_tune_patience` Runden ohne Verbesserung
4. Evaluation mit fine-tuned Modellen

## Vorteile

✅ **Bessere Anpassung**: Globales Modell wird an lokale Daten angepasst
✅ **Overfitting-Schutz**: Early Stopping verhindert Überanpassung
✅ **Flexibilität**: Jeder Client kann unterschiedlich viele Epochen trainieren
✅ **Kompatibel**: Funktioniert mit allen Modellarchitekturen und Personalization

## Empfohlene Config-Werte

```yaml
fine_tune: True
fine_tune_epochs: 50      # Niedrig halten (10-100)
fine_tune_patience: 5     # 3-10 je nach Datengröße
```

## Beispiel

```yaml
fl:
  n_rounds: 10              # FL Training
  fine_tune: True           # Dann Fine-Tuning
  fine_tune_epochs: 30
  fine_tune_patience: 5
```

**Ablauf:**
- 10 FL-Runden mit globaler Aggregation
- Dann feines Tuning auf jedem Client (max 30 Epochen)
- Early Stopping stoppt, wenn 5 Epochen keine Verbesserung
