# Ablations B und C — Implementierung und Verifikationsergebnisse

*Stand 2026-08-03, ausgeführt auf `l2` (`w-lambdablade2`), Branch
`fix/mtgnn-topo-static-dim`, Basis-Commit `674a043`.
Umgesetzt nach `docs/prompt_ablations_implementation.md` und
`docs/implementation_plan_ablations.md`. Alle Zahlen in diesem Dokument sind
reproduzierbar, die Kommandos stehen in §6.*

| Variante | eigene k Gitterpunkte | Nachbar-**Messungen** | Nachbar-**Geometrie / NWP-Kontext** | Config |
|---|---|---|---|---|
| **A** | ja | ja | ja | `config_wind_dcrnn{,_fold1..3}.yaml` |
| **B** | ja | **nein** | ja | `config_wind_dcrnn_nomeas{,_fold1..3}.yaml` |
| **C** | ja | nein | **nein** | `config_wind_dcrnn_nograph{,_fold1..3}.yaml` |

A − B = Wert der Nachbar-Messungen. B − C = Wert des Geometrie- und
Kontextkanals. C = der reine standortweise Downscaling-Boden.

---

## 1. Was geändert wurde

### 1.1 Neues Flag `neighbour_meas_available` (Variante B)

Die Verdrahtung kopiert `hist_wind_available` wörtlich; der Default `True`
erhält Variante A exakt.

| Datei | Änderung |
|---|---|
| `geostatistics/dcrnn/config.py` | neues Dataclass-Feld `neighbour_meas_available: bool = True`, gelesen als `d.get("neighbour_meas_available", True)` |
| `geostatistics/stgnn/training/sampler.py` | Konstruktor-Argument + `self.`-Zuweisung; neuer Maskierungszweig in `sample_train` **und** `sample_val` |
| `geostatistics/evaluation.py` | Parameter in `build_eval_batch` und `evaluate`, Durchreichung, neuer Maskierungszweig |
| `geostatistics/train_dcrnn.py` | 2 Aufrufstellen (`TrainingSampler`, `run_evaluation`) |
| `geostatistics/get_test_results_dcrnn.py` | 1 Aufrufstelle (`evaluate`) |
| `geostatistics/hpo_dcrnn.py` | 1 Aufrufstelle (`TrainingSampler`) |

Maskierung, an allen drei Stellen identisch. **Die Reihenfolge ist tragend:**
B subsumiert das IGNNK-Nullen, der neue Zweig muss zuerst kommen, sonst würden
nur die Zielstationen genullt und die Nachbarn behielten ihre Messungen.

```python
if not self.neighbour_meas_available:
    meas_hist[:, :, :] = 0.0                    # ablation B/C: nobody has measurements
elif not self.hist_wind_available:
    meas_hist[:, target_mask_np, :] = 0.0       # IGNNK masking (variant A)
```

`target_mask` bleibt unangetastet — A und B werden an denselben Knoten bewertet
(belegt in §2, Check `train: target_mask unchanged` und
`train: ground_truth unchanged`).

### 1.2 Neuer Wert `station_connectivity: "none"` (Variante C)

In `geostatistics/stgnn/graph_builder.py::_build_station_edges`, **vor** den
Zweigen `delaunay` und `knn`: leeres `edge_index` `(2, 0)` und leeres
`edge_attr` `(0, F)`.

`F` wird **nicht** hart kodiert, sondern durch einen Aufruf von `edge_features`
auf einer synthetischen Einzelkante abgeleitet — **inklusive der topographischen
Kantenspalten**. Das ist die eine Stelle, an der die Referenz `/tmp/patch_ablations.py`
(30.07.) falsch gewesen wäre: sie zählte nur die drei `use_*`-Flags und hätte
`F = 4` geliefert, während die Kampagnen-Config über
`edge_features: [distance, direction, altitude_diff, z0, slope, tdi, tpi5, tpi75,
aspect_sin, aspect_cos, dist_coast]` tatsächlich **`F = 12`** erzeugt
(1 Distanz + 2 Azimut + 1 Höhendifferenz + 8 Topo). Verifiziert in §2 §4.3.

Warum ein leeres Kantenset in dieser Diffusionsfaltung sicher ist
(`dcrnn/model/dcgru_cell.py`):

* `out = self.lins[0](x)` ist ein unbedingter k = 0 Self-Transform — es gibt
  immer einen Pfad ohne Kanten;
* `out_deg.clamp(min=1e-8)` entfernt die Division durch null;
* `propagate` über null Kanten liefert einen Nulltensor, jeder Term k ≥ 1
  trägt also `lins[k](0)` bei.

Das DCGRU degeneriert damit zu einer gewöhnlichen GRU mit linearer
Eingangstransformation — genau die Absicht von C. Empirisch bestätigt
(§2 §4.4, §4.5).

### 1.3 Assertion und Startzeile

Neu: `geostatistics/ablations/guard.py::check_ablation_flags(dcrnn_cfg, logger)`.
Loggt eine Zeile mit `neighbour_meas_available`, `hist_wind_available`,
`interpolate_history`, `station_connectivity`, `direction_to_adj`, `nwp_nodes`
und der erkannten Variante, und bricht hart ab bei

> `neighbour_meas_available: false` **und** `interpolate_history: true`

Begründung: Der Regression-Kriging-Lag-Kanal wird in `sampler.py` **nach** dem
Nullen angehängt und ist aus den Messungen der anderen Stationen interpoliert.
Er würde Variante B genau die Information zurückgeben, die die Ablation entfernt,
und zwar auf einem Pfad, der den Stationsgraphen komplett umgeht — ohne jedes
sichtbare Symptom. Das ist der einzige Fehlermodus, der eine plausible, aber
bedeutungslose Zahl erzeugt.

Eingehängt in **drei** Entry Points:

| Datei | Stelle |
|---|---|
| `train_dcrnn.py` | nach dem HPO-Override und nach dem Anhängen des File-Handlers, vor jedem Datenladen |
| `get_test_results_dcrnn.py` | nach dem HPO-Override, vor der Feature-Auflösung |
| `hpo_dcrnn.py` | direkt nach der Studiennamen-Ableitung, vor dem Storage-Zugriff |

> **Abweichung vom Plan, bewusst:** Plan Schritt 3 nennt nur `train_dcrnn.py` und
> `get_test_results_dcrnn.py`. Da der Nutzer für §5.1 Ausweg (b) gewählt hat
> (eigene HPO je Variante), ist `hpo_dcrnn.py` jetzt ein vollwertiger Einstiegspunkt
> für B und C und darf den Kriging-Kanal genauso wenig zurückholen können.

### 1.4 Configs

Acht neue Dateien in `configs/dcrnn/`, **generierend** erzeugt aus den
A-Configs durch `geostatistics/ablations/gen_variant_configs.py` (Textchirurgie
ausschließlich innerhalb der `dcrnn:`-Sektion, alle Kommentare und alle
unbeteiligten Einstellungen bleiben byteweise erhalten):

```
config_wind_dcrnn_nomeas.yaml    config_wind_dcrnn_nomeas_fold{1,2,3}.yaml
config_wind_dcrnn_nograph.yaml   config_wind_dcrnn_nograph_fold{1,2,3}.yaml
```

### 1.5 Gruppen

`DCRNN_NOMEAS` und `DCRNN_NOGRAPH` in `launch_train_pipeline.py` (ans Ende der
`GROUPS`-Liste gehängt, damit die Reihenfolge der laufenden Kampagne unverändert
bleibt) und je drei Jobs in `launch_eval_pipeline.py`.

### 1.6 Neue Hilfsdateien

| Datei | Zweck |
|---|---|
| `geostatistics/ablations/guard.py` | Bannerzeile + harte Assertion (§1.3) |
| `archiv/ablations_verification/fixture.py` | deterministische Test-Fixture: echte Config, echte Stations-IDs/Koordinaten/Höhen aus `data/stations_master.csv`, echte Topo-Features, echter Graph-Builder und Sampler; nur die Messwert- und NWP-**Werte** sind geseedet synthetisch |
| `archiv/ablations_verification/batch_fingerprint.py` | Plan §4.1 — SHA-256-Fingerabdruck aller Sampler-Tensoren, vor und nach der Änderung lauffähig |
| `archiv/ablations_verification/verify.py` | Plan §4.2–§4.6 plus Config-Prüfung, 79 Checks |
| `geostatistics/ablations/gen_variant_configs.py` | Config-Generator |

Damit liegen die früher in `/tmp` verstreuten Skripte (Plan §10) im Repository.

---

## 2. Verifikationsergebnisse

Alle Checks laufen ohne Datendateien und ohne GPU auf der CPU, Laufzeit ~40 s.

### §4.1 — Der Flag-Default ist ein No-op (Variante A unangetastet)

Der entscheidende Beleg dafür, dass die Runde-1-Fixes durch diese Arbeit nicht
beschädigt wurden. `batch_fingerprint.py` benutzt **keinerlei** Ablations-API und
ist deshalb auf beiden Seiten der Änderung byteweise identisch lauffähig. Es
zieht mit festem Seed (`random`, `numpy`, `torch` je auf 4711) einen Trainings-
und einen Validierungs-Batch und hasht jeden Tensor.

| Lauf | Zeitpunkt / Host | Ergebnis gegen die Referenz |
|---|---|---|
| Referenz | **vor** dem Patch, `l2`, Arbeitsbaum auf `674a043`, sauber | — |
| 1 | nach dem vollständigen Patch, `l2` | **IDENTICAL** |
| 2 | nach der Config-Generierung, `l2` | **IDENTICAL** |
| 3 | nach dem Rollout, `l2` | **IDENTICAL** |
| 4 | nach dem Pinnen der inerten Parameter, `l2` | **IDENTICAL** |
| 5 | Endstand, `l1` (mit Pfad-Rewrites) | **IDENTICAL** |
| 6 | Endstand, `ws` | **IDENTICAL** |

> `IDENTICAL — 28 tensor fingerprints match bit for bit across 14 + 14 tensors.`

Die Läufe 5 und 6 vergleichen gegen **dieselbe** vor dem Patch auf `l2` erzeugte
Referenzdatei. Damit ist nicht nur belegt, dass Variante A unverändert ist,
sondern auch, dass alle drei Hosts bitgleich samplen — auf `l1` trotz der 79+8
Pfad-Rewrites in `configs/`.

Verglichene Tensoren je Batch (14): `station.x`, `station.static`, `icond2.x`,
`icond2.static`, `ecmwf.x`, `ecmwf.static`, `edge_index` und `edge_attr` für
`station–near–station`, `icond2–informs–station`, `ecmwf–informs–station`,
`target_mask`, `ground_truth`. Verglichen wird der SHA-256 über den rohen
Speicherinhalt, zusätzlich Shape und dtype.

Mitgeprüfte Dimensionen (identisch vor und nach):
`M = 3`, `I2 = 4`, `E2 = 3`, `N_all = 60` (45 Train / 15 Val),
`station_static_features = 13` (4 geo/type + 9 Topo — das ist der A3-Fix aus
Runde 1), `s2s_edges = 328`, `s2s_edge_attr_dim = 12`.

Ein dritter Lauf nach der Config-Generierung ergab erneut `IDENTICAL`.

### §4.2 — B nullt tatsächlich alles

| Check | Zahl |
|---|---|
| Training, Variante A | **6048** von 6480 Messzellen ungleich null |
| Training, Variante B | **0** von 6480 |
| Training, B **+** `hist_wind_available: true` | **0** — beweist die Reihenfolge; bei vertauschten Zweigen wären es 6048 |
| `target_mask` | A: 3 Ziele von 45 Knoten, B: 3 von 45, Masken bitgleich |
| `ground_truth` | Shape (3, 48), max\|Δ\| = **0.000e+00** |
| NWP-Kanäle von `station.x` | 30240 Zellen bitgleich zwischen A und B |
| Validierung, A / B | 6480 von 8640 ≠ 0 / **0** |
| Validierung, B + `hist_wind_available: true` | **0** |
| `evaluation.build_eval_batch`, A / B | 6480 von 8640 ≠ 0 / **0** |
| `evaluation.build_eval_batch`, B + `hist_wind_available: true` | **0** |

Alle drei Maskierungsstellen (`sample_train`, `sample_val`,
`build_eval_batch`) sind damit einzeln geprüft, jeweils auch auf die
Reihenfolge.

### §4.3 — C baut ein leeres Kantenset

| Check | Zahl |
|---|---|
| `edge_index` | A `(2, 328)` → C **`(2, 0)`**, dtype `torch.int64` |
| `edge_attr` | A `(328, 12)` → C **`(0, 12)`** — die Breite ist geprobt, nicht hart kodiert |
| NWP→Station-Kanten unberührt | i2s `(2, 360)`, e2s `(2, 240)` |
| `subgraph_station_edges` auf 30-Knoten-Teilmenge | liefert `(2, 0)` / `(0, 12)`, **keine Exception** |

### §4.4 — C liefert endliche Ausgaben

Ein voller `DCRNN`-Forward-Pass auf einem C-Batch:

```
pred shape (15, 48), finite=True, min=-0.630031 max=0.497454 mean=0.028882,
NaN=0, Inf=0
```

Die in Plan §2.2 als Rückfallebene vorgesehene Variante
`station_connectivity: "self"` wird **nicht** gebraucht.

### §4.5 — C ist wirklich graphfrei (Permutationstest)

Permutiert werden alle drei Kanäle, über die eine Nachbarstation sprechen könnte:

1. `station.x` — Messungen und die stationseigenen NWP-Spalten,
2. `station.static` — lat/lon/alt plus die neun topographischen Knotenfeatures,
3. der Zielindex der `icond2→station` und `ecmwf→station` Kanten, jede
   Nachbarstation bekommt also die Gitterpunkte und die Kantengeometrie einer
   **anderen** Nachbarstation.

Die Zielstationen werden nicht angefasst; der Typindikator in `static` ist über
alle Nachbarn konstant und kann daher auch nicht leaken. Das Modell läuft in
`model.eval()`, die Permutation ist nicht-identisch (alle 45 Nachbarn bewegt).

| Variante | Nachbarn / Ziele | max\|Δpred\| an den Zielstationen |
|---|---|---|
| **C** (kein Graph, keine Messungen) | 45 / 15 | **0.000e+00** |
| B (Graph, keine Messungen) | 45 / 15 | 5.856e-01 |
| A (Graph + Messungen) | 45 / 15 | 5.879e-01 |

C ist **exakt** null, nicht nur innerhalb der Toleranz von 1e-6.

Die beiden Kontrollen sind das eigentlich Wichtige an diesem Test:
Variante B hat, genau wie C, überall genullte Messkanäle — die Permutation
wirkt dort ausschließlich über Geometrie, Statics und NWP-Kontext, und sie
wirkt mit 5.9e-1. C's Null ist also **kein Artefakt der genullten Messungen**,
sondern die Folge des fehlenden Stationsgraphen. Damit ist auch belegt, dass
in C keine zweite Informationsbrücke zwischen Stationen existiert.

Nebenbefund: das ist zugleich der direkte Beweis für Plan §9.2 — `K_hop` und
`next_n_neighbors` können in C nichts bewirken. Auf dieser Grundlage wurden sie
aus C's HPO-Suchraum entfernt, siehe §4.

### §4.6 — Die Assertion feuert

Isoliert:

| Kombination | Ergebnis |
|---|---|
| `neighbour_meas_available=False` + `interpolate_history=True` | `AblationConfigError` |
| `neighbour_meas_available=True` + `interpolate_history=True` (= A) | erlaubt, Variante als `A (full model)` erkannt |
| Variantenerkennung B / C | `B (no neighbour measurements)` / `C (no station graph)` |

End-to-End in den echten Entry Points, mit einer Kopie von
`config_wind_dcrnn_nomeas.yaml`, in der `interpolate_history: true` gesetzt wurde:

| Script | Ergebnis |
|---|---|
| `train_dcrnn.py` | Abbruch in `main()` Zeile 409, **vor** jedem Datenladen, Exit ≠ 0 |
| `get_test_results_dcrnn.py` | Abbruch in `main()` Zeile 196 |
| `hpo_dcrnn.py` | Abbruch in `main()` Zeile 292, vor dem Optuna-Storage-Zugriff |

Die Bannerzeile im Log sieht so aus:

```
[INFO] ABLATION VARIANT B (no neighbour measurements) — neighbour_meas_available=False
       hist_wind_available=False  interpolate_history=False  station_connectivity=delaunay
       direction_to_adj=False  nwp_nodes=True
```

### §4.7 — Kurzer Trainingslauf: **offen**

Plan §4.7 (zwei bis drei Epochen von B auf einer GPU, Loss fällt, Checkpoint
läuft durch den Eval-Pfad) ist als einziger Schritt **nicht** ausgeführt. Grund:
er braucht GPU-Zeit und den vollen Datenpfad, und die Vorgabe für diese Arbeit
war ausdrücklich keine GPU-Zeit, solange die A-Kampagne läuft.

Stand zum Zeitpunkt der Prüfung (2026-08-03 17:25 UTC): alle vier A100 auf `l2`
zwischen 87 % und 99 % ausgelastet, `cl_m-dcrnn_out-48_freq-1h_wind_dcrnn` bei
**0 abgeschlossenen Trials, 3 RUNNING** — es gibt also noch keine
Hyperparameter, aus denen ein Retrain lesen könnte.

**Ersatzweise abgedeckt:** der Forward-Pass in §4.4 und die Permutationsläufe in
§4.5 führen das echte `DCRNN`-Modell auf echten Sampler-Batches aus, inklusive
NWP-Attention, DCGRU-Encoder und autoregressivem Decoder über alle 48
Prognoseschritte. Was §4.7 zusätzlich prüfen würde, ist der Backward-Pass, der
Trainer-Loop und der Checkpoint-Roundtrip.

### Zusatzcheck — die generierten Configs

Semantischer Diff (flach über den kompletten YAML-Baum, nicht Textdiff)
zwischen jeder Variantenconfig und ihrer Quelle:

| Datei | abweichende Schlüssel |
|---|---|
| `config_wind_dcrnn_nomeas.yaml` | `dcrnn.neighbour_meas_available: <absent>→False`, `dcrnn.hpo.trials: 150→60` |
| `config_wind_dcrnn_nomeas_fold{1,2,3}.yaml` | `dcrnn.neighbour_meas_available: <absent>→False` |
| `config_wind_dcrnn_nograph.yaml` | `dcrnn.neighbour_meas_available: <absent>→False`, `dcrnn.station_connectivity: 'delaunay'→'none'`, `dcrnn.hpo.trials: 150→60`, **plus die sechs gepinnten Suchraum-Schlüssel** `dcrnn.hpo.params.{K_hop,next_n_neighbors}.{type,low,high}: …→<absent>` (siehe §4) |
| `config_wind_dcrnn_nograph_fold{1,2,3}.yaml` | `dcrnn.neighbour_meas_available: <absent>→False`, `dcrnn.station_connectivity: 'delaunay'→'none'`, `dcrnn.direction_to_adj: <absent>→False` |

Sonst **kein** einziger Schlüssel. Weiter geprüft, je Datei:

* `station_node_features: all` ist mitgekommen (A3-Fix aus Runde 1);
* `interpolate_history` ist überall `False`;
* die Studienauflösung ergibt `cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_nomeas`
  bzw. `…_nograph` — für die Basis-Config **und** alle drei Fold-Configs;
* die Config parst durch den Produktions-Parser und baut den beabsichtigten
  Graphen: `nomeas` → 88…104 s2s-Kanten, `nograph` → **0** s2s-Kanten.

`dcrnn.direction_to_adj` steht in der A-Basis-Config bereits auf `false`, in
den A-Fold-Configs fehlt der Schlüssel (Parser-Default `False`). In C wird er
überall explizit gesetzt, damit die Variante selbsterklärend ist — der
tatsächliche Wert ändert sich dadurch nirgends.

### Gesamtergebnis

```
79 passed, 0 failed  (of 79 checks)
```
plus `IDENTICAL — 28 tensor fingerprints` für §4.1.

---

## 3. Studienauflösung — Entscheidung (b), begründet

`train_dcrnn.py:338/353` und `hpo_dcrnn.py:268/279` leiten den Optuna-Studiennamen
**identisch** und **immer** aus dem Config-Dateinamen ab (am aktuellen Code
verifiziert, die Zeilennummer 353 aus dem Prompt stimmt, 315–333 aus dem Plan
nicht mehr):

```python
config_stem = Path(config).stem.replace("config_", "")
hpo_stem    = re.sub(r'_fold\d+$', '', config_stem)
study_name  = f"cl_m-dcrnn_out-{H_fore}_freq-{freq}_{hpo_stem}"
```

`--hpo-study` steuert bei Postgres-Storage nur, **ob** geladen wird, nicht welche
Studie.

**Gewählt: Ausweg (b) — eigene HPO je Variante** (Nutzerentscheidung,
Plan §6 Option 2, reduziertes Budget). Konsequenzen:

* Die Namensauflösung passt von allein. `config_wind_dcrnn_nomeas_fold1.yaml`
  → `cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_nomeas`, und die HPO auf
  `config_wind_dcrnn_nomeas.yaml` schreibt in genau diese Studie.
  **Kein neues CLI-Argument nötig**, keine Änderung an der Ableitung.
* Je Variante: erst `hpo_dcrnn.py` auf der Basis-Config, dann
  `launch_train_pipeline.py` für die drei Folds, dann
  `get_test_results_dcrnn.py` in beiden Szenarien.
* **Die Fairness-Frage aus Prompt §5.1 ist damit entschärft:** B und C erben
  A's Hyperparameter *nicht* mehr, jede Variante wird an ihrem eigenen Optimum
  gemessen. Der Vorwurf „die Ablation sieht nur deshalb schlechter aus, weil sie
  mit A's Hyperparametern läuft" trifft nicht mehr zu.

### Trial-Budget: 60 statt 150

Gesetzt in `dcrnn.hpo.trials` der beiden Basis-Configs; die Fold-Configs fahren
nie HPO, ihre `hpo:`-Blöcke bleiben unangetastet.
`trials` ist ein **Studienbudget**, kein Worker-Budget
(`hpo_dcrnn.py`: `remaining = max(n_trials - completed, 0)`), 60 heißt also 60
abgeschlossene Trials insgesamt.

Begründung für 60:

* Der Median-Pruner wird nach `pruner_n_startup_trials: 20` aktiv; TPE beginnt
  nach seinen Default-10 Startup-Trials zu steuern. Bei 60 Trials sind also
  40 pruner-aktiv und 50 TPE-gesteuert — unter etwa 40 Trials bliebe die Suche
  fast reine Zufallssuche, was die Zahlen wertlos machen würde.
* Der Suchraum ist mit 16 Parametern derselbe wie bei A, die Varianten brauchen
  aber nur ihr *eigenes* Optimum, nicht das global beste Modell: in die Aussage
  gehen nur die Differenzen A − B und B − C ein.
* Kosten: 2 × 60 = 120 Trials gegen die laufenden 8 × 150 = 1200 der
  Hauptkampagne, also **+10 %**. Bei 150 je Variante wären es +25 %.

**Restrisiko, das ins Paper gehört:** ein kleineres Suchbudget benachteiligt B
und C weiterhin leicht gegenüber A. Die gemessenen Kanalbeiträge A − B und B − C
sind damit **Obergrenzen**, die Aussage „der Kanal trägt mindestens X" bleibt
konservativ. Wer das vermeiden will, setzt `trials` in beiden Basis-Configs auf
150 — es ist eine Zeile je Datei, und es ist noch keine HPO gelaufen.

---

## 4. Inerte Parameter in Variante C (Plan §9.2) — entschieden

**Entscheidung des Nutzers: `K_hop` und `next_n_neighbors` werden aus dem
HPO-Suchraum von Variante C entfernt; die statischen Werte `K_hop: 2` und
`next_n_neighbors: 90` bleiben stehen.** Umgesetzt mit

```bash
python -m geostatistics.ablations.gen_variant_configs \
    --variant nograph --trials 60 --pin-inert --force
```

`direction_to_adj` war nie Teil des Suchraums und steht in C ohnehin explizit
auf `false`.

### Warum

**Die beiden Parameter sind nachweislich wirkungslos.** Der Permutationstest in
§4.5 liefert für C max\|Δpred\| = **0.000e+00** — exakt null, nicht nur
innerhalb der Toleranz. Ohne station↔station-Kanten kann `K_hop` (Anzahl der
Diffusionssprünge) nichts diffundieren, und `next_n_neighbors` (wie viele
Nachbarknoten in den Teilgraphen kommen) bestimmt nur, wie viele Knoten
rechnen, ohne dass ihr Ergebnis irgendwo ankommt. Beide belegten also
Suchraumdimensionen ohne Gegenwert; `next_n_neighbors` ist zusätzlich **teuer**,
weil die wirkungslosen Nachbarknoten Speicher und Rechenzeit kosten.

**Es gibt den Präzedenzfall im selben Repository.**
`config_wind_dcrnn_base.yaml` wurde aus `config_wind_dcrnn.yaml` generiert und
hat `nwp_heads` / `nwp_out_per_head` aus dem Suchraum entfernt, weil sie ohne
GATv2 wirkungslos sind (`# nwp_heads / nwp_out_per_head entfallen: ohne GATv2
sind sie wirkungslos.`) — exakt dieselbe Situation, dieselbe Konsequenz.

**Bei 60 Trials zählen zwei Dimensionen.** Der Suchraum von C schrumpft von
**17 auf 15** Parameter. Und ein Appendix, der inerte Parameter als „gesucht"
auflistet, liest sich schlecht.

### Was bewusst *nicht* getan wurde

* **`next_n_neighbors` wurde nicht auf sein Minimum gezogen.** Das wäre für das
  Ergebnis nachweislich folgenlos, ändert aber die Knotenzahl je Batch und damit
  den Verbrauch des Dropout-RNG-Stroms — die Trainingstrajektorie wäre eine
  andere (statistisch äquivalent, nicht identisch). Plan §4 nennt es
  ausdrücklich als wünschenswert, dass „C still samples a station subgraph, so
  the batch composition matches A and B"; mit `next_n_neighbors: 90` gilt das.
* **Die drei Fold-Configs von C wurden nicht angefasst.** Sie fahren nie HPO
  (`train_dcrnn.py` liest die besten Parameter aus der Studie, den `hpo:`-Block
  der Fold-Datei liest niemand). `--pin-inert` wirkt deshalb — genau wie
  `--trials` — nur auf die Studien-Config. Der semantische Diff der drei
  Fold-Dateien gegen ihre A-Quellen bleibt damit auf die reine Ablationsachse
  beschränkt (3 Schlüssel), und sie sind gegenüber dem vorherigen Commit
  byteidentisch geblieben.
* **Variante B wurde nicht angefasst.** B benutzt den Stationsgraphen, dort sind
  beide Parameter wirksam. Geprüft: `config_wind_dcrnn_nomeas.yaml` hat
  weiterhin **17** Suchraumparameter, identisch zu A, `K_hop` und
  `next_n_neighbors` inklusive.

### Beleg

Im Suchraum von `config_wind_dcrnn_nograph.yaml` stehen an Stelle der beiden
Blöcke jetzt Kommentarzeilen:

```yaml
      # K_hop entfaellt: ohne station<->station Kanten wirkungslos (Permutationstest: max|dpred| = 0.0).
      # next_n_neighbors entfaellt: ohne station<->station Kanten wirkungslos (Permutationstest: max|dpred| = 0.0).
```

Die Statikwerte sind unverändert:

```
  K_hop: 2
  next_n_neighbors: 90
```

Automatisch geprüft (§2, Zusatzcheck):

| Check | Ergebnis |
|---|---|
| `K_hop` / `next_n_neighbors` aus `hpo.params` verschwunden | ja, Suchraum **17 → 15** Parameter |
| Statikwerte erhalten | `{'K_hop': 2, 'next_n_neighbors': 90}` — identisch zu A |
| B's Suchraum unangetastet | 17 Parameter, beide weiterhin gesucht |

---

## 5. Was noch offen ist

| Punkt | Status |
|---|---|
| Plan §4.7, kurzer Trainingslauf von B | offen, braucht GPU; siehe §2 §4.7 |
| HPO-Läufe für `wind_dcrnn_nomeas` / `wind_dcrnn_nograph` | nicht gestartet |
| 6 Trainingsläufe (2 Varianten × 3 Folds) | nicht gestartet, brauchen die HPO-Ergebnisse |
| Evaluation in `excl_val` / `incl_val` | nicht gestartet |
| Fold-Configs aus einer Quelle regenerieren (Plan §9.1) | weiterhin offen; die A-Fold-Configs haben eine andere Struktur als die A-Basis-Config (zusätzliche `gnn:`/`stgnn:`/`stgnn2:`-Blöcke, andere statische Werte, `hpo.trials: 1000`). Für die Ablation unschädlich, weil jede Variante aus *ihrer* Quelldatei abgeleitet ist — B-fold1 unterscheidet sich von A-fold1 nur im ablatierten Flag |

---

## 6. Rollout

Commit `3f8d6b8` auf Branch `fix/mtgnn-topo-static-dim`, Elternteil `674a043`.
25 Dateien, 4676 Einfügungen, 6 Löschungen.

| Host | Repo | HEAD | Arbeitsbaum |
|---|---|---|---|
| `l2` (`w-lambdablade2`) | `/home/viktor/Work/forecasting_framework` | `3f8d6b8` | sauber (0 Einträge) |
| `l1` (`w-lambdablade1`) | `/home/viktorwalter/Work/forecasting_framework` | `3f8d6b8` | 87 modifizierte Configs, **ausschließlich** Pfad-Rewrites |
| `ws` (`w-lambda-vector`) | `/home/viktor/Work/forecasting_framework` | `3f8d6b8` | sauber (0 Einträge) |

Auf `l1` wurden die Pfad-Rewrites nach dem Pull neu angewendet
(`/mnt/lambda1/nvme1/` → `/mnt/nvme1/`). Kontrolle:

```
git diff -U0 | grep "^[+-]" | grep -v "^[+-][+-]" | grep -cv "mnt/nvme1\|mnt/lambda1"
→ 0
```

87 statt vorher 79 modifizierte Config-Dateien: die acht neuen Varianten-Configs
erben die `/mnt/lambda1/nvme1/`-Pfade aus ihren A-Quellen und werden daher
ebenfalls umgeschrieben.

Die Verifikationssuite wurde auf **allen drei Hosts** ausgeführt, jeweils
**79 passed, 0 failed**, dazu auf jedem Host der §4.1-Fingerabdruck gegen die
vor dem Patch erzeugte Referenz: **IDENTICAL**. Auf `l1` bestätigt das
zusätzlich, dass die Pfad-Rewrites die Varianten-Configs nicht beschädigt haben.

Auf `l1` liegt eine unbeteiligte, nicht versionierte Datei `data_cache.py`
(28.07., älter als diese Arbeit). Sie taucht in `git status` als `??` auf, nicht
im `git diff`, und wurde nicht angefasst.

Aufgeräumt: die veralteten Backups `geostatistics/stgnn/graph_builder.py.bak_ablation`
und `geostatistics/stgnn/training/sampler.py.bak_ablation` (30.07.) auf `l1`
wurden gelöscht — ein Zurückspielen hätte die Runde-1-Fixes A2, A4, B1, B2 und B6
wieder entfernt.

---

## 7. Reproduktion

```bash
ssh l2
cd ~/Work/forecasting_framework && source frcst/bin/activate

# Plan §4.1 — vor und nach einer Codeänderung, Vergleich bitgenau
CUDA_VISIBLE_DEVICES="" python -m archiv.ablations_verification.batch_fingerprint \
    --out /tmp/fp_before.json
#   … Änderung …
CUDA_VISIBLE_DEVICES="" python -m archiv.ablations_verification.batch_fingerprint \
    --out /tmp/fp_after.json --compare archiv/ablations_verification/fp_9808123.json

# Plan §4.2 … §4.6 plus Config-Checks, 79 Checks, ~40 s, keine Daten, keine GPU
CUDA_VISIBLE_DEVICES="" python -m archiv.ablations_verification.verify

# Configs neu erzeugen (idempotent mit --force)
python -m geostatistics.ablations.gen_variant_configs --variant nomeas  --trials 60 --force
python -m geostatistics.ablations.gen_variant_configs --variant nograph --trials 60 --pin-inert --force

# Trockenlauf der Launcher
python geostatistics/launch_train_pipeline.py --gpus 0 --groups DCRNN_NOMEAS,DCRNN_NOGRAPH --dry-run
python geostatistics/launch_eval_pipeline.py  --gpus 0 --groups DCRNN_NOMEAS,DCRNN_NOGRAPH --dry-run
```

Wenn die A-Kampagne durch ist, je Variante in dieser Reihenfolge:

```bash
# 1) HPO (Env-Variablen explizit setzen — .bashrc bricht in nicht-interaktiven
#    Shells vor den export-Zeilen ab, ein Worker ohne WEATHER_DB_URL faellt
#    still auf NWP-Hoehen 0 zurueck)
python geostatistics/hpo_dcrnn.py --config configs/dcrnn/config_wind_dcrnn_nomeas.yaml --gpu N
# 2) Retrain der drei Folds
python geostatistics/launch_train_pipeline.py --gpus ... --groups DCRNN_NOMEAS
# 3) Evaluation
python geostatistics/launch_eval_pipeline.py  --gpus ... --groups DCRNN_NOMEAS
```
