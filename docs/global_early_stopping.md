# Global Early Stopping für Federated Learning

## Überblick

Das Global Early Stopping beendet das FL-Training vorzeitig, wenn sich die aggregierte Validierungsmetrik über mehrere **globale Runden** nicht mehr verbessert. Die besten Gewichte werden dabei gespeichert und am Ende zurückgegeben.

**Nicht zu verwechseln mit** `fl.early_stopping`, das Early Stopping *lokal* bei den Clients *innerhalb einer Runde* steuert.

## Konfiguration

```yaml
fl:
  n_rounds: 50              # Maximale Anzahl globaler Runden

  # Lokales Early Stopping (pro Client, pro Runde)
  early_stopping:
    enabled: False
    patience: 10
    monitor: 'val_rmse'
    mode: 'min'

  # Globales Early Stopping (über FL-Runden hinweg)
  global_early_stopping:
    enabled: True
    patience: 10            # Runden ohne Verbesserung bis Abbruch
    min_delta: 0.0001       # Minimale Verbesserung die zählt
    monitor: 'val_rmse'     # Metrik: 'val_rmse', 'val_mae', 'val_loss', 'val_r^2'
    mode: 'min'             # 'min' für rmse/mae/loss, 'max' für r^2
```

## Verhalten

1. Nach jeder FL-Runde wird die **aggregierte** `val_rmse` (gewichtet nach Clientgröße) berechnet
2. Verbessert sich die Metrik um mindestens `min_delta` → Counter wird zurückgesetzt, beste Gewichte werden gespeichert
3. Keine Verbesserung → Counter erhöht sich
4. Counter erreicht `patience` → Training bricht ab
5. Die Gewichte der **besten Runde** werden zurückgegeben (nicht der letzten)

## Ablauf

```
Runde 1:  val_rmse=0.120  → best=0.120, counter=0
Runde 2:  val_rmse=0.115  → best=0.115, counter=0  ✓ neue beste Runde
Runde 3:  val_rmse=0.116  → counter=1
Runde 4:  val_rmse=0.117  → counter=2
...
Runde 12: val_rmse=0.118  → counter=10  → STOP, Gewichte von Runde 2 werden verwendet
```

## Unterschied zu lokalem Early Stopping

| | `fl.early_stopping` | `fl.global_early_stopping` |
|---|---|---|
| **Ebene** | Lokal (Client, innerhalb einer Runde) | Global (Server, über Runden) |
| **Einheit** | Epochen | FL-Runden |
| **Effekt** | Client stoppt Training früher | Gesamtes FL stoppt früher |
| **Restore** | Beste Gewichte des Clients | Beste globale Gewichte |

## Implementierung

Die Logik liegt vollständig in `utils/federated.py` in der Funktion `run_simulation()`.
`train_fl.py` benötigt keine Anpassungen — die Funktion gibt automatisch die Gewichte der besten Runde zurück.
