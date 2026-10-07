# FedGradient — Gradienten-Aggregation mit Server-Optimizer

Stand 2026-10-06. Code: `utils/fedgradient.py` (Logik, ohne Ray testbar), eingebunden in
`utils/federated.py` (`ClientActor.fg_*`, `run_simulation`). Tests: `tests/test_fedgradient.py`.
Ersetzt die frühere Strategie `fedsgd` ersatzlos (anderes Verhalten, s. u.).

## Ablauf

```
Runde r (= 1 Epoche):
  jeder Client: Permutation seiner Trainingsfenster, Seed params.random_seed*1000 + r,
                Batches ohne Zurücklegen (drop_last), Cursor
  solange ein Client Batches übrig hat:
    Server -> aktive Clients: aktuelle Parameter
    Client: forward/backward auf dem nächsten Batch, Clipping, Gradienten -> Server
    Server: Mittel der Gradienten, optimizer.step() auf dem globalen Modell
  Evaluation, Logging, Global Early Stopping, Checkpoint: einmal je Runde
```

- Clients mit weniger Batches fallen aus dem Mittel heraus, sobald sie durch sind; die Runde
  endet, wenn alle durch sind.
- Gewichtung nach Batchgröße (`weighting: batch_size`, bei `drop_last` faktisch uniform)
  oder `uniform`.
- Der Client macht keinen Optimizer-Schritt. Modell und Trainingsdaten liegen ab der ersten Runde
  dauerhaft auf der GPU des Actors.

## Konfiguration

```yaml
fl:
  strategy: 'fedgradient'
  n_rounds: 100                 # = maximale Epochen
  global_early_stopping: {enabled: True, patience: 10, monitor: 'val_rmse', mode: 'min'}
  rowwise_embedding: True       # Default; s. Park-ID
  checkpoint: {enabled: True, dir: null}   # null -> checkpoints/fl/<study>/last.pt
  fedgradient:
    server_optimizer: 'adam'    # adam | adamw | sgd | sgd_momentum  (torch.optim)
    server_lr: 0.01
    beta_1: 0.9                 # adam/adamw
    beta_2: 0.99
    eps: 0.001                  # = tau in FLTA26
    momentum: 0.9               # sgd_momentum
    weight_decay: 0.0
    clipnorm: 1.0               # Client-Clipping vor dem Senden; Key fehlt -> model.tft.clipnorm, null -> aus
    weighting: 'batch_size'
```

Die Werte stehen in der Config, im Code gibt es keine versteckten Defaults (fehlende Pflicht-Keys
brechen mit Meldung ab). `hpo.get_hyperparameters` legt sie flach mit Präfix ins
Hyperparameter-Dict (`server_optimizer`, `server_lr`, `server_beta_1`, `server_beta_2`,
`server_eps`, `server_momentum`, `server_weight_decay`, `client_clipnorm`, `gradient_weighting`),
damit sie nicht mit `lr`/`weight_decay`/`clipnorm` des Clients kollidieren.
Optionale HPO-Dimensionen (`hpo_fl.py`): `hpo.fl.fedgradient: {server_optimizer: [adam, sgd_momentum],
server_lr: [1e-4, 1e-1]}`; die übrigen Optimizer-Keys kommen aus `fl.fedgradient` und müssen für
alle Optimizer im Suchraum dort stehen.

## Park-ID (kategoriales Embedding)

- **Codes global:** `train_fl.py`/`hpo_fl.py` laden die Stationen eines Clients über
  `data.client_files` (`federated.client_data_config`, `get_data(files_key='client_files')`),
  `data.files` bleibt die globale Parkliste. Vorher wurde `data.files` je Client überschrieben,
  jeder Client vergab die Codes 0..9 und acht Parks teilten sich eine Embedding-Zeile.
  `federated.check_client_parks` bricht ab, wenn ein Client-Park nicht in
  `files`/`val_files`/`test_files` steht, oder wenn Park-ID und `val_files` zusammen gesetzt sind
  (Holdout-Parks hätten untrainierte Zeilen).
- **Zeilenweise Aggregation:** `static_embed[i]` (`nn.Embedding`) gehört zum globalen Modell, die
  Gradienten je Zeile werden aber nur über die Clients gemittelt, die den Park besitzen (aus den
  Trainings-Statiken abgeleitet). Zeilen ohne beitragenden Client werden nicht angefasst: Wert und
  Optimizer-State (Momentum, zweites Moment) bleiben, also auch kein Weight Decay und kein Drift
  durch Momentum. Gilt analog für FedAvg/FedAdam bei der Gewichtsaggregation
  (`aggregate_weights(..., row_owners, reference_weights)`): eine Zeile kommt nur von ihren
  Besitzern, Zeilen ohne Besitzer behalten den alten Wert.

## Kommunikation, Checkpoint, Resume

- `history['comm_stats']` und `results/<data>/<study>_<ts>_comm_stats.json`: Sync-Schritte
  (FedGradient: Server-Schritte, FedAvg: Runden), Bytes hoch/runter je Client, dazu die
  Gewichts-Downloads für die Client-Evaluation (`eval_download_bytes`). FedGradient je Schritt:
  Download aller Parameter, Upload aller Gradienten (float32), nur für aktive Clients.
- Checkpoint `<dir>/last.pt` nach jeder Runde (atomar): globales Modell, Optimizer-State,
  Early-Stopping-Zustand inkl. bester Gewichte, Metriken, Comm-Stats.
  Fortsetzen: `python train_fl.py -c <config> -m tft --resume checkpoints/fl/<study>/last.pt`.
  Wegen der Rundenseeds ist ein fortgesetzter Lauf identisch mit einem durchgelaufenen (Test).
  Nicht mit `fl.personalize`.

## Baselines zum Vergleich

- **Zentral (CL):** `train_cl.py` auf denselben Stationen (parks_v1: `config_parks_v1_cl80{,_parkid}.yaml`,
  Holdout über `data.holdout_files`).
- **Lokal je Client:** `train_local.py -c <diese FL-Config> -m tft --gpus 0-7` — leitet aus
  `fl.clients` je Client einen zentralen Lauf auf dessen Stationen ab und fährt alle parallel,
  eine GPU je Client ([local_training.md](local_training.md)).
- **FedAvg:** dieselbe Config mit `fl.strategy: fedavg`.

## Unterschiede zur früheren `fedsgd`

| | `fedsgd` (bis 2026-10-06) | `fedgradient` |
|---|---|---|
| Runde | 1 Batch je Client = 1 Schritt | 1 Epoche |
| Sampling | frischer Zufallsbatch je Schritt, mit Zurücklegen, ungeseedet | Permutation je Runde, geseedet |
| Evaluation / ES | nach jedem Batch | je Runde |
| Server-LR | Bug: immer 1,0 (β2 0,999, τ 1e-8) | aus der Config |
| Optimizer | eigener numpy-FedAdam/FedAvgM | `torch.optim`, State im Checkpoint |
| Clipping | nur `fedsgd.gradient_clip` | Client-seitig, Default `model.tft.clipnorm` |
| Personalisierung | ja | nein (`NotImplementedError`) |

Referenz für die Mechanik: `~/Work/FLTA26` auf l1 (`run_fedsgd_round`), dort aber mit fester
Batchzahl je Runde statt Epochen.

## Weitere Änderungen in `run_simulation` (alle Strategien)

- Der Client-Evaluationspfad evaluierte bis 2026-10-06 die Gewichte vom **Rundenanfang**, also das
  Modell der Vorrunde; Global Early Stopping hinkte eine Runde nach, das Training der letzten Runde
  wurde nie bewertet. Jetzt wird das in dieser Runde aggregierte Modell evaluiert.
- Ohne Personalisierung sind die zurückgegebenen `clients_weights` immer die globalen Gewichte
  (vorher mit `global_val_data` die lokal trainierten Client-Gewichte).
- Die Initialisierung des globalen Modells ist mit `params.random_seed` geseedet.
