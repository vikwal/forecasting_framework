# Review-Briefing: Topografische Features in DCRNN/MTGNN/WaveNet

**Zweck dieses Dokuments:** Eigenständiges Briefing für ein Code-Review durch
eine andere KI-Session, die den Diff noch nicht kennt. Fasst zusammen, was
geändert wurde, warum, was bereits verifiziert ist, und was gezielt geprüft
werden sollte. Für die vollständige Entstehungsgeschichte (Screening-Ergebnisse,
Entscheidungsverlauf) siehe [`topo_rehpo_plan.md`](topo_rehpo_plan.md) — dieses
Dokument hier ist der kompakte Review-Einstieg.

**Branch:** `fix/mtgnn-topo-static-dim` (gepusht, Basis `origin/main`).
Commits: `e7bb373` … `a748085` (10 Commits, `git log origin/main..HEAD`).
Zusätzlich **uncommitted** Änderungen in `geostatistics/hpo_dcrnn.py`,
`geostatistics/hpo_mtgnn.py`, `geostatistics/hpo_wavenet.py` (siehe Abschnitt 5) —
`git diff` zeigt diese separat von `git diff origin/main..HEAD`.

---

## 1. Aufgabenstellung

DCRNN, MTGNN und WaveNet (Windenergie-Prognose, NWP-Bias-Correction,
`geostatistics/`) nutzten bisher keine topografischen Stationsmerkmale
(Höhe, Hangneigung, Exposition, Rauigkeitslänge, Küstenentfernung etc. — 9
Features, `TOPO_FEATURE_ORDER`), obwohl ein TFT-Vergleichsmodell diese als
statische Features bereits nutzt. Ziel: topografische Information in alle
drei Graph-Architekturen einbauen, und zwar für jede Architektur über **jeden
Kanal, den sie strukturell unterstützt**:

- **Adjazenz** — Topo fließt in eine gelernte Knoten-Embedding-Matrix ein, die
  wiederum die Graphstruktur (Nachbarschaftsgewichte) formt. Nur MTGNN und
  WaveNet haben eine lernbare/adaptive Adjazenz.
- **Feature-Strom** — Topo wird als zusätzlicher, zeitkonstanter Kanal direkt
  in den Zeitreihen-Input gespeist (kann also direkt als Prädiktor wirken,
  nicht nur als Ähnlichkeitsmaß). Bei DCRNN der einzig mögliche Kanal (siehe
  Abschnitt 2), bei MTGNN/WaveNet als **Broadcast** additiv zur Adjazenz.

**Wichtige Design-Entscheidung** (bitte nicht als Inkonsistenz werten): es
wurde *nicht* pro Architektur der empirisch beste Kanal gewählt, sondern
einheitlich der Feature-Strom-Kanal (+ Adjazenz, wo vorhanden) als
HPO-Zielkonfiguration festgelegt — explizit um Cherry-Picking auf einem
Ein-Seed-Screening zu vermeiden (Winner's-Curse-Argument, siehe
`topo_rehpo_plan.md` Abschnitt 1).

## 2. Architektonischer Hintergrund je Modell

### DCRNN (Li et al. 2018, fester Distanz-Graph)
Der Stationsgraph nutzt einen **festen** Gaußkern auf Distanz
(`exp(-d²/σ²)`), keine gelernte Adjazenz. Verifiziert (synthetischer Test,
siehe `topo_rehpo_plan.md`): `edge_weight_from_attr` reduziert `edge_attr`
ohnehin auf Spalte 0 (Distanz) — zusätzliche Kanten-Features (Höhendifferenz,
Richtung) haben nachweislich **keinen** Effekt auf den Output. Ein
Kanten-basierter Topo-Kanal ist bei DCRNN daher architektonisch nicht
sinnvoll umsetzbar, ohne einen gelernten Kanten-Bias einzuführen — das würde
die Fixed-vs-Learned-Graph-Vergleichsachse verwischen und wurde bewusst
**nicht** gemacht. DCRNN bekommt Topo daher ausschließlich über den
Feature-Strom (`station_static` wird bei jedem Zeitschritt in den
DCGRU-Input konkateniert, ohnehin bestehender Mechanismus).

### MTGNN (Wu et al. 2020, "Connecting the Dots")
Selbst-adaptive Adjazenz `A = ReLU(tanh(α(M1M2ᵀ − M2M1ᵀ)))` mit
`M1=tanh(αE1Θ1)`, `M2=tanh(αE2Θ2)`. Die hier verwendete induktive Adaption
(freie Embedding-Matrix `E1,E2` ersetzt durch `emb_mlp(static_features)`) ist
**explizit paper-sanktioniert** (Sec. 4.2: "E1=E2=Z, wobei Z eine statische
Knoten-Feature-Matrix ist").

### Graph WaveNet (Wu et al. 2019)
Analog adaptive Adjazenz `softmax(ReLU(E1E2ᵀ))`. Die induktive Adaption ist
hier **nicht** durch das Original-Paper sanktioniert (das Paper motiviert
freie Embeddings explizit mit "keine Vorabkenntnis nötig"), wird aber über
das MTGNN-Präzedenzurteil gerechtfertigt (selbe Kernautoren, ein Jahr später
publiziert). Zusätzlich wurde die im Paper (Eq. 6) beschriebene, aber im
bisherigen Code fehlende **vorgegebene Distanz-Adjazenz als zweiter
Diffusionszweig** nachgerüstet (`_predefined_adjacency`,
`geostatistics/wavenet/model.py`, Commit `735c4d4`) — per Flag
`predefined_adj` (Default `False`, verifiziert rückwärtskompatibel).

## 3. Der Broadcast-Mechanismus (Kern der Änderung)

In `geostatistics/mtgnn/model.py` und `geostatistics/wavenet/model.py`
(Commit `ffe34f8`) gibt es jeweils ein neues `broadcast_topo: bool = False`
Flag. Wenn aktiv:

```python
proj_in = in_channels + (topo_dim if self.broadcast_topo else 0)
...
# im forward(), nach der NWP-Reassemblierung, vor input_proj:
if self.broadcast_topo:
    topo = static_single[:, 6:6 + self.topo_dim]
    topo = topo.view(1, N, 1, self.topo_dim).expand(B, N, T, self.topo_dim)
    x = torch.cat([x, topo], dim=-1)
```

D. h. dieselben Topo-Spalten, die auch in `emb_mlp` für die Adjazenz genutzt
werden, werden zusätzlich zeitkonstant in jeden Zeitschritt des Modell-Inputs
kopiert. **Wichtig für den Review:** dieser Pfad wurde per synthetischem Test
verifiziert (`/tmp/.../verify_broadcast.py`, nicht Teil des Repos, siehe
`topo_rehpo_plan.md` für die Methodik):
- `state_dict`-Keys identisch bei `broadcast_topo=False` (nur `input_proj`
  breiter bei `True`).
- Alte Checkpoints laden `strict=True` in ein Modell ohne das Flag.
- Störung von `static[:, 6:]` (Topo-Spalten) verändert den Output **nur**
  über die Adjazenz, wenn `broadcast_topo=False`; deutlich stärker, wenn
  `True` — bestätigt, dass der Kanal tatsächlich etwas trägt.
- Der Topo-Kanal ist nachweislich zeitkonstant (kein Leck von Zukunftsinfo).

## 4. Datenpfad: topografische Features laden

Zwei parallele Lader in `geostatistics/stgnn/utils/topo_features.py`:

- `load_topo_node_features(...)` (bereits vorher vorhanden, unverändert) —
  liefert **paarweise Differenzen** für Kanten-Features.
- `load_topo_station_features(...)` (**neu**, Commit `cd59326`) — liefert
  **absolute**, pro-Station z-skalierte Werte für Knoten-Features. Wichtige
  Details, die ein Reviewer prüfen sollte:
  - Varianzstabilisierende Transforms vor der z-Skalierung:
    `log(z0)`, `log1p(slope/tdi/elev_std/dist_coast)`,
    `sign(x)·log1p(|x|)` für `tpi5/tpi75` (stark rechtsschief).
  - `tdi` wird für perfekt flaches Gelände (`elev_std==0`) auf `0` gesetzt,
    nicht mit dem Median aufgefüllt (undefiniertes Verhältnis bei
    Null-Relief, keine fehlende Messung — semantischer Unterschied).
  - z-Skalierung nutzt **nur die ersten `n_train` Stationen** (kein
    Leakage aus Val/Test in die Skalierungsstatistik).
  - `aspect_sin`/`aspect_cos` werden hier berechnet, nicht aus einer
    Rohspalte geladen — bitte Trigonometrie-Vorzeichen gegenrechnen.

`geostatistics/stgnn/config.py::parse_station_node_features(...)` (neu)
löst `--station-node-features` (CLI, `'all'` / `'none'` / kommagetrennte
Liste) gegen `TOPO_FEATURE_ORDER` auf; wirft bei unbekannten Namen.

## 5. CLI-Flags in den Trainings-/HPO-Skripten

Alle sechs Skripte (`train_dcrnn.py`, `train_mtgnn.py`, `train_wavenet.py`,
`hpo_dcrnn.py`, `hpo_mtgnn.py`, `hpo_wavenet.py`) haben jetzt:

- `--station-node-features NAMES` — überschreibt die Config, erlaubt
  derselben Config-Datei beide Arme (mit/ohne Topo) zu bedienen.
- `--broadcast-topo` (nur MTGNN/WaveNet, DCRNN hat keinen separaten Schalter,
  da dessen einziger Kanal ohnehin der Feature-Strom ist) — aktiviert den in
  Abschnitt 3 beschriebenen Mechanismus.
- `--shuffle-node-features` — Permutationskontrolle: vertauscht die
  Topo-Zeilen zufällig (`seed=0`) über die Stationen, behält Parameterzahl
  und Randverteilungen bei, zerstört nur die Stations↔Terrain-Zuordnung.
  Trennt "mehr Kapazität hilft" von "Terrain-Information hilft".

**Kritischer Punkt für den Review — Cache-Platzierung:** in allen drei
`hpo_*.py`-Skripten gibt es einen `GNNCache`-Hit/Miss-Zweig (Rohdaten werden
gecacht, um wiederholtes Parquet-Laden zu vermeiden). Der Topo-Lade-Block
**muss außerhalb beider Zweige** stehen (nach `if/else`, vor der ersten
Nutzung), sonst bekommt ein Cache-Hit eine andere `static_dim` als ein
Cache-Miss, und das beim Retrain aus den HPO-Best-Params gebaute Modell passt
nicht zum während der HPO tatsächlich trainierten. Das war ein realer Bug
(Commit `e7bb373`, ursprünglich nur für MTGNN gefixt) — beim Nachziehen von
`hpo_dcrnn.py`/`hpo_wavenet.py` (uncommitted, siehe unten) wurde bewusst an
derselben Stelle platziert. **Bitte im Diff verifizieren, dass der Topo-Block
tatsächlich nach dem `GNNCache`-Block und vor der ersten
`DCRNNConfig.from_yaml`/`HomoSampler`/Model-Konstruktion steht.**

**Uncommitted (`git diff`, noch kein Commit):**
- `hpo_dcrnn.py`: Import `parse_station_node_features`,
  `load_topo_station_features`; 2 neue CLI-Flags; Topo-Konkatenation nach dem
  Cache-Block; `station_node_features=args.station_node_features` an beide
  `DCRNNConfig.from_yaml(...)`-Aufrufe (Basis-Config und Pro-Trial-Config).
- `hpo_wavenet.py`: identisches Muster zu `train_wavenet.py`
  (`parse_edge_features`-Fallback wenn `--station-node-features` nicht
  gesetzt ist), `topo_feats` an `HomoSampler`, `static_dim`/`topo_dim`/
  `broadcast_topo` an `GraphWaveNetModel`. Vorher hatte `hpo_wavenet.py`
  **überhaupt keinen** Topo-Pfad (`static_dim` war hart auf `6` codiert).
- `hpo_mtgnn.py`: Umstellung vom alten `parse_edge_features`-only-Pfad
  (Zwischenstand aus Commit `e7bb373`, reiner Bugfix ohne Broadcast-Fähigkeit)
  auf den vollen `parse_station_node_features`/`broadcast_topo`-Pfad, jetzt
  identisch zu `train_mtgnn.py`.

Alle drei wurden mit `--preprocess-only --station-node-features all
[--broadcast-topo]` smoke-getestet (Datenladen bis zum Modellbau, kein
Training) — liefen fehlerfrei durch, korrekte finale Breite geloggt (z. B.
DCRNN: `3 geo + 9 topo (+1 type indicator) → 13`).

## 6. Was noch NICHT abgesichert ist (bitte im Review besonders beachten)

- **Keine Permutationskontrolle für MTGNN.** DCRNN und WaveNet haben je einen
  `--shuffle-node-features`-Kontrolllauf gefahren (Ergebnis: Effekt bleibt
  bei DCRNN, WaveNet — Kapazitäts-Confound ausgeschlossen). MTGNN nicht.
- **Screening ist n=1 pro Arm/Architektur**, ein Fold (fold1/GRID), nicht
  signifikanzgetestet. Ergebnisse (siehe `topo_rehpo_plan.md` Abschnitt 1)
  sind Plausibilitätsindikatoren, keine belastbaren Effektgrößen.
- **`ecmwf_feature_mode`/`icond2_feature_mode` u. ä. Config-Interaktionen**
  mit dem neuen Topo-Pfad wurden nicht systematisch durchgetestet, nur die
  Standard-`nwp`-Varianten der drei Architekturen.
- Der WaveNet-`predefined_adj`-Zweig (Eq. 6) ist implementiert und
  rückwärtskompatibel verifiziert, aber **nicht** Teil der geplanten
  Re-HPO-Kampagne (Default `False` bleibt in allen 9 neuen Studien) — falls
  im Code aktiv gesetzt irgendwo auftaucht, ist das ein Fehler.

## 7. GPU-Verteilung für die geplante Re-HPO-Kampagne

9 Studien (DCRNN/MTGNN/WaveNet × `base`/`nwp`/`nwp_hist`), Ziel-Arm
einheitlich Feature-Strom (+Adjazenz wo vorhanden), auf 3 Hosts / 13 GPUs:

| Host | GPUs | Typ | VRAM |
|---|---|---|---|
| w-lambdablade2 ("L2", lokal) | 4 | A100 | 80 GB |
| l1 (w-lambdablade1) | 7 nutzbar (1 fremdbelegt von GPU 3) | RTX A6000 | 49 GB |
| ws (10.166.32.252) | 2 | RTX 4090 | 24 GB |

DCRNN und MTGNN mit 3 Replikaten pro Studie, WaveNet mit 2 (kleinerer
Umfang) → 24 parallele Optuna-Worker:

| GPU | Worker 1 | Worker 2 |
|---|---|---|
| A100 #0 (L2) | DCRNN_BASE r1 | MTGNN_BASE r1 |
| A100 #1 (L2) | DCRNN_NWP r1 | MTGNN_NWP r1 |
| A100 #2 (L2) | DCRNN_NWP_HIST r1 | MTGNN_NWP_HIST r1 |
| A100 #3 (L2) | WaveNet_BASE r1 | WaveNet_NWP r1 |
| A6000 #0 (l1) | WaveNet_NWP_HIST r1 | WaveNet_BASE r2 |
| A6000 #1 (l1) | WaveNet_NWP r2 | WaveNet_NWP_HIST r2 |
| A6000 #2 (l1) | DCRNN_BASE r2 | MTGNN_BASE r2 |
| A6000 #4 (l1) | DCRNN_NWP r2 | MTGNN_NWP r2 |
| A6000 #5 (l1) | DCRNN_NWP_HIST r2 | MTGNN_NWP_HIST r2 |
| A6000 #6 (l1) | DCRNN_BASE r3 | MTGNN_NWP r3 |
| A6000 #7 (l1) | DCRNN_NWP r3 | MTGNN_NWP_HIST r3 |
| RTX4090 #0 (ws) | DCRNN_NWP_HIST r3 | — |
| RTX4090 #1 (ws) | MTGNN_BASE r3 | — |

Prinzipien: nie zwei Replikate derselben Studie auf derselben GPU
(Ausfallsicherheit); alle Worker einer Studie teilen sich einen gemeinsamen
Optuna-Studiennamen über die Postgres-`OPTUNA_STORAGE`-DB — **kritischer
Fallstrick:** nie unterschiedliche `-s`/`--suffix`-Werte zwischen parallelen
Workern derselben Studie verwenden, das fragmentiert eine Studie in mehrere;
nur Log-Dateiname/Screen-Session-Name darf variieren. `ws` braucht NFS-Mounts
von l1 (`/mnt/lambda1/nvme1`, `/mnt/lambda1/nvme2` — l1's `/etc/exports` und
`ufw` mussten für `ws`s IP freigegeben werden) sowie die Python-Pakete
`torch_geometric`, `pyarrow`, `psycopg2-binary`, die dort nachinstalliert
wurden.

## 8. Vorschlag für den Review-Umfang

Sinnvolle Prioritäten für ein fokussiertes Review:

1. `geostatistics/stgnn/utils/topo_features.py` — Korrektheit der Transforms
   und der Train-only-Skalierung (Leakage-Risiko).
2. Cache-Platzierung in allen drei `hpo_*.py` (Abschnitt 5) — Breiten-Bug wie
   in `e7bb373` ist die Art Fehler, die leicht wiederkehrt.
3. `broadcast_topo`-Forward-Pfad in `mtgnn/model.py`/`wavenet/model.py` —
   Indexierung `static_single[:, 6:6+topo_dim]`, Zeitkonstanz, Backward-Kompatibilität
   bei `broadcast_topo=False`.
4. Konsistenz der drei `--station-node-features`/`--shuffle-node-features`-
   Implementierungen zwischen `train_*.py` und `hpo_*.py` (sollen identisch
   sein, damit Retrain aus HPO-Best-Params dasselbe Modell baut wie die HPO
   selbst trainiert hat).

---

# 9. Review-Ergebnis (externe Session, 30. Juli 2026)

Reviewt gegen den Code auf **l2**, Branch `fix/mtgnn-topo-static-dim`, plus die
uncommitteten Änderungen in den drei `hpo_*.py`. Zusätzlich gegen die
Paper-Positionierung in `Graphs_Wind_Speed_Forecasting/story_positioning.md`
und `critical_assessment_and_journals.md` (Literaturaudit vom 27./28. Juli).

**Gesamturteil:** Die Änderung ist konzeptionell richtig und für die
Paper-Story sogar notwendig, nicht bloß nützlich (siehe 9.6). Die Umsetzung ist
in den geprüften Kernpunkten sauber. Es gibt aber **zwei Befunde, die die
geplante Kampagne in der jetzt vorgesehenen Form entwerten würden**, und beide
sind vor dem Start zu beheben. Danach ist das eine solide Basis für die Studie.

## 9.1 BLOCKER — l1 hat den Topo-Code überhaupt nicht

Geprüft und bestätigt:

```
l1:  git rev-parse --abbrev-ref HEAD   ->  iaims26
l1:  git branch -a                     ->  iaims26, main, remotes/origin/main
l1:  ls geostatistics/stgnn/utils/topo_features.py  ->  No such file or directory
l1:  grep -c "broadcast_topo" geostatistics/hpo_mtgnn.py geostatistics/train_mtgnn.py  ->  0, 0
```

Der Branch `fix/mtgnn-topo-static-dim` existiert auf l1 nicht einmal als
Remote-Ref. Der Kampagnenplan in Abschnitt 7 verteilt aber **12 der 24 Worker
auf l1** (A6000 #0, #1, #2, #4, #5, #6, #7).

Warum das nicht auffällt und trotzdem alles zerstört: alle Worker einer Studie
teilen sich über die Postgres-`OPTUNA_STORAGE` einen Studiennamen. Optuna
speichert Hyperparameter und Zielwert, **nicht** die Modellklasse. l1-Worker
würden also Trials eines Modells *ohne* Topo-Features in dieselbe Studie
schreiben wie l2-Worker mit Topo. Kein Absturz, keine Warnung, keine
Dimensionsfehlermeldung, weil jeder Worker sein Modell aus seinem eigenen Code
baut. Das Ergebnis ist eine Studie, deren Trials aus zwei verschiedenen
Modellklassen stammen, und deren „bestes" Trial damit bedeutungslos ist.

Das ist derselbe Fehlertyp wie der Cache-Bug aus `e7bb373`, nur auf Host-Ebene
und deutlich schädlicher, weil er alle neun Studien gleichzeitig betrifft.

**Zu tun vor dem Start:**

1. `fix/mtgnn-topo-static-dim` auf l1 auschecken, und auf `ws` ebenfalls prüfen
   (dort ungeprüft, wurde aber laut Abschnitt 7 aus l1 heraus aufgesetzt).
2. Auf allen drei Hosts denselben Commit-Hash verifizieren, am besten als
   Startbedingung im Launch-Skript: `git rev-parse HEAD` vergleichen und
   abbrechen, wenn sie abweichen.
3. **Achtung beim Auschecken auf l1:** dort liegen uncommittete Änderungen auf
   `iaims26`, unter anderem an `configs/dcrnn/config_wind_dcrnn_{base,fold1,2,3}.yaml`
   und ein gelöschtes `config_wind_dcrnn_nwp_hist.yaml`. Zusätzlich liegt dort
   der Ablations-Patch aus derselben Session (`neighbour_meas_available`,
   `station_connectivity: none`, Backups als `*.bak_ablation`, siehe
   `implementation_plan_ablations.md`). Vor dem Branchwechsel sichern, sonst
   wandert der Patch unbemerkt mit oder geht verloren.

Ergänzend: eine Zeile pro Worker ins Log, die Hostname, Commit-Hash,
`station_node_features` und `broadcast_topo` ausgibt, macht so etwas im
Nachhinein überhaupt erst nachweisbar.

## 9.2 BLOCKER — MTGNN und WaveNet benutzen den falschen Topo-Lader

Das ist der inhaltlich wichtigere Befund. Geprüft in allen sechs Skripten:

| Skript | aufgerufene Funktion |
|---|---|
| `train_dcrnn.py:829`, `hpo_dcrnn.py:629` | `load_topo_station_features` |
| `train_mtgnn.py:434`, `hpo_mtgnn.py:591` | **`load_topo_node_features`** |
| `train_wavenet.py:418`, `hpo_wavenet.py:576` | **`load_topo_node_features`** |

`load_topo_node_features` ist der Kanten-Differenz-Lader. Er hat drei
Eigenschaften, die genau für den Broadcast-Kanal falsch sind:

1. **Keine varianzstabilisierenden Transforms.** `_TOPO_TRANSFORMS` wird dort
   nicht angewandt. Die Begründung im Docstring des anderen Laders trifft
   wörtlich auf den Broadcast zu: *„tdi has median 0.90 but max 361 (sigma
   29.7), so a plain z-score leaves 202 of 203 stations inside ±0.15 sigma and
   one at +12 sigma. As an absolute node feature that constant offset goes
   straight into every GRU gate / input channel, so the skew matters much more
   here than in a pairwise difference."* Der Broadcast **ist** der
   Absolutwert-Fall. MTGNN und WaveNet broadcasten also einen praktisch
   konstanten Kanal plus einen Ausreißer in jeden Zeitschritt.
2. **Skalierung über alle Stationen.** Die Funktion hat keinen `n_train`-Parameter,
   `mean` und `std` werden über `train + val` gebildet. Val-Statistik leckt in
   die Normalisierung. DCRNN vermeidet das, MTGNN und WaveNet nicht.
3. **Keine tdi-Flachland-Behandlung.** Der `elev_std == 0 -> tdi = 0`-Zweig
   fehlt, flaches Gelände bekommt den Median statt der physikalisch richtigen 0.

**Konsequenz für die Paper-Story, und deshalb ist es ein Blocker:** der
Architekturvergleich DCRNN vs. MTGNN vs. WaveNet ist damit durch die
Feature-Vorverarbeitung konfundiert. Jeder Unterschied zwischen den
Architekturen ist teilweise ein Unterschied zwischen zwei Skalierungspipelines.
Genau diese Achse ist im Research Summary als „Kern-Modellvergleich der
Publikation" ausgewiesen, und ein Reviewer aus der ST-GNN-Community, der
ohnehin skeptisch ist (`shao2024basicts` zeigt, dass gemeldete Gewinne unter
fairem Benchmarking verschwinden), wird genau hier hinsehen.

**Zu tun:** MTGNN und WaveNet auf `load_topo_station_features` umstellen und
`n_train` durchreichen. Der Lader liefert bereits ein `(N, F)`-Array in
kanonischer Reihenfolge, `homo_sampler` erwartet ein Dict, also entweder den
Rückgabewert in ein Dict umsetzen oder `_topo_arrays` direkt aus dem Array
füllen. Der Kommentar in `homo_sampler.py:243` („already z-score normalised in
load_topo_node_features") ist dann mit anzupassen.

**Nebenwirkung, die vorher zu bedenken ist:** die fehlende
Permutationskontrolle für MTGNN (Abschnitt 6) ist dadurch nicht weniger,
sondern **mehr** wert. Solange der Topo-Kanal nach der Skalierung nahezu
konstant ist, würde eine Permutation dieses Kanals folgerichtig nichts ändern,
und das Ergebnis „Terrain trägt nichts" wäre ein Artefakt der Skalierung, nicht
ein Befund über das Terrain. Die Kontrolle also erst **nach** dem Umstellen
fahren, sonst misst sie das Falsche.

## 9.3 Zwei kleinere Korrektheitsbefunde

**Median-Imputation leckt, in beiden Ladern.** `values.median()` wird über alle
Stationen gebildet, also inklusive Val, und erst danach greift die
Train-only-z-Skalierung. Das widerspricht der eigenen Zusage im Docstring
(*„instead of leaking val/test station statistics into the normalisation"*).
Betrifft nur einen Skalar pro Feature und nur bei fehlenden Werten, ist aber
genau die Art Detail, die in einem Review auffällt. Korrektur:
`values.iloc[:n_train].median()`.

**`aspect_sin`/`aspect_cos` bei undefinierter Exposition.** Auf perfekt flachem
Gelände ist die Exposition undefiniert und `aspect` vermutlich NaN. Diese NaNs
laufen in die Median-Auffüllung und bekommen damit eine willkürliche
Himmelsrichtung zugewiesen. Semantisch richtig wäre `sin = cos = 0`, also der
Nullvektor „keine Vorzugsrichtung" — exakt analog zur tdi-Behandlung, die an
derselben Stelle bewusst und richtig gelöst wurde. Bitte prüfen, wie viele
Stationen betroffen sind, und die beiden Fälle konsistent behandeln.

## 9.4 Was geprüft wurde und in Ordnung ist

- **Cache-Platzierung**, der explizit angefragte Punkt: in allen drei
  `hpo_*.py` liegt der Topo-Block hinter dem Cache-Schreibblock und vor der
  ersten Modell- beziehungsweise Sampler-Konstruktion
  (`hpo_dcrnn` 614 -> 629 -> 747/895, `hpo_mtgnn` 569 -> 591 -> 742/775,
  `hpo_wavenet` 554 -> 576 -> 726/758). Der Bug aus `e7bb373` ist nicht
  zurückgekehrt.
- **Der Offset 6 im Broadcast** ist korrekt. `homo_sampler._build_static` legt
  `[sin_lat, cos_lat, sin_lon, cos_lon, alt_norm, type_indicator]` an, danach
  Topo in `TOPO_FEATURE_ORDER`. `static_single[:, 6:6+topo_dim]` trifft damit
  genau die Topo-Spalten und überspringt den Type-Indikator.
- **`n_train`** ist korrekt: `all_ids = train_ids + val_ids` und
  `N_train = len(train_ids)`, also selektiert `arr[:n_train]` tatsächlich die
  Trainingsstationen.
- Die Entscheidung, **einheitlich den Feature-Strom-Arm** als HPO-Ziel zu
  nehmen statt pro Architektur den Screening-Sieger, ist richtig und sollte im
  Paper genau mit dem Winner's-Curse-Argument begründet werden. Bei n=1 pro Arm
  wäre die Auswahl sonst nicht verteidigbar.

## 9.5 Ein Fallstrick, der heute noch nicht greift

Die Train-only-Skalierung beruht auf `arr[:n_train]`, setzt also voraus, dass
`station_ids` train-zuerst sortiert ist. Solange die drei Folds **zeitliche**
Folds sind (fold1/2/3 unterscheiden sich in `test_start`/`test_end`, der
Stationssplit ist fix), stimmt das.

`story_positioning.md` Abschnitt 3.2 führt aber **k-fold räumliche
Kreuzvalidierung** als nicht verhandelbaren Punkt 2 der Umsetzungsreihenfolge —
ohne sie ruht die zentrale induktive Aussage auf 50 einmalig gezogenen
Stationen, was genau die Schwäche ist, die `baran2024clustering` hat und die ein
Referee mit Hydrologie-Hintergrund benennen wird. Sobald der Stationssplit pro
Fold wechselt, muss `all_ids` **pro Fold** neu geordnet und die Topo-Skalierung
**pro Fold** neu gefittet werden. Sonst leckt ab diesem Moment jede Fold-Testmenge
in die Skalierungsstatistik. Bitte jetzt schon als Kommentar an der Stelle
vermerken, weil die Umstellung ansonsten still schiefgeht.

## 9.6 Passung zur Paper-Story

Hier ist die Nachricht gut. Contribution (i) in `story_positioning.md`
beansprucht wörtlich, dass die Zuordnung von Standort zu Korrektur gemeinsam
aus *statischen topographischen Deskriptoren*, aus der leadtime-aufgelösten
NWP-Prognose und aus den Live-Beobachtungen der Nachbarstationen gelernt wird.
Ohne Topo-Features in den Graphmodellen war dieser Anspruch für DCRNN, MTGNN
und WaveNet schlicht nicht eingelöst: die Modelle hatten keinerlei Möglichkeit
zu wissen, *an was für einem Standort* sie korrigieren. Die Änderung ist damit
keine Verbesserung, sondern eine Voraussetzung.

Zweitens: das Alleinstellungsargument gegenüber den fünf induktiven Vorarbeiten
ist ausdrücklich, dass bodennaher Wind stärker als Temperatur von lokaler
Rauigkeit, Abschattung und Geländeexposition kontrolliert wird, und dass
deshalb eine *gelernte, terrainkonditionierte* Korrektur hier mehr bringen soll
als in den publizierten Temperaturfällen. Das ist eine testbare Vorhersage, und
die Topo-Features sind der Mechanismus, der sie überhaupt prüfbar macht.

---

# 10. Was zusätzlich in den Experimentplan gehört

Die ersten drei Punkte kosten kaum etwas und ändern, was das Paper behaupten
kann. Der Rest sind Kampagnen-Entscheidungen, die jetzt fallen sollten, weil
sie sonst eine zweite HPO-Runde erzwingen.

## 10.1 Die Permutationskontrolle als berichtetes Experiment, nicht als Sanity-Check

`--shuffle-node-features` ist derzeit als interne Kontrolle gedacht. Sie sollte
ein **Ergebnis im Paper** werden, aus einem Grund, der außerhalb dieses Repos
liegt: die Schwesterstudie (iAIMS26, TFT auf Windleistung) fand, dass physische
Statics die Zero-Shot-Generalisierung **nicht** tragen (0.813 gegen 0.818,
p = 0.44), weil die 48 h Leistungshistorie im Encoder den Standort implizit
identifiziert. Hier ist die Historie der Zielstation maskiert, dieser Kanal
existiert also nicht. Wenn Topo hier trägt und dort nicht, ist das ein
sauberer, mechanistisch erklärter Kontrast zwischen zwei eigenen Studien und
ein starkes Argument dafür, dass die induktive Aufgabe qualitativ anders ist
als die transduktive. Wenn es hier ebenfalls nicht trägt, muss man das wissen,
bevor Contribution (i) so formuliert wird.

Konkret: je einen Shuffle-Lauf pro Architektur mit den finalen
Hyperparametern, gepaart pro Station gegen den echten Lauf, Wilcoxon mit
Holm-Korrektur. Für MTGNN erst nach der Korrektur aus 9.2.

## 10.2 Terrain-Stratifizierung als Auswertung, nicht als eigener Lauf

Axis B der Paper-Story verlangt Skill aufgeschlüsselt nach *Terrain-Unähnlichkeit
zur Trainingsmenge*. Die dafür nötigen Größen liegen nach dieser Änderung
vollständig vor: die neun z-skalierten Topo-Spalten pro Station. Distanzmaß im
Terrain-Raum zur nächsten Trainingsstation, dann Skill dagegen plotten. Kostet
keine GPU-Minute und ist eine der vier Kurven, die Axis B ausmachen. Sollte im
Plan stehen, damit die Topo-Arrays nach dem Training mit abgespeichert werden
und nicht rekonstruiert werden müssen.

## 10.3 Die `next_n_ecmwf`-Randverteilung mitschreiben

Beide Suchräume haben `next_n_ecmwf` mit unterer Schranke 0, das Modell darf
ECMWF also verwerfen. Über neun Studien mit je mehreren Replikaten entsteht
damit ohne Zusatzkosten eine sehr belastbare Sensitivitätsaussage zu
Contribution (iv), der Merkmalsebenen-Fusion zweier NWP-Systeme. Das ist die
stärkere Evidenzform als ein einzelner Ablationslauf, weil das Modell die
Quelle verwerfen *darf* und es nicht tut. Auszuwerten ist die Randverteilung
über alle COMPLETE-Trials plus `optuna.importance`, nicht nur das beste Trial.

Wichtig: wenn die Trials überwiegend 0 wählen, ist Contribution (iv) nicht zu
halten und muss zu einer Methodennotiz degradiert werden. Das sollte man aus
den eigenen Studien erfahren, nicht aus einem Gutachten.

## 10.4 Die Variantenliste jetzt festlegen, nicht nach der Kampagne

Die Kampagne fährt 3 Architekturen × {base, nwp, nwp_hist}. Vier Anmerkungen
dazu, alle mit Konsequenz für die Anzahl der Studien:

- **`base` verteidigt nur Contribution (iv)**, also die am stärksten gefährdete.
  „Erster heterogener Graph mit Gitterpunkt-Knoten" ist durch `yang2025offgrid`
  v1, `wu2024weathergnn`, `blasone2025gnn4cd` und `low2026spatialsupport`
  erledigt. Zusätzlich ist `base` derzeit **keine saubere Ablation**: sein
  HPO-Raum ist enger als der von `nwp` (`next_n_icond2` 1–4 statt 1–7, `K_hop`
  1–2 statt 1–3), die NWP-Variante darf also strikt mehr. Behält man `base`,
  müssen die Räume harmonisiert werden.
- **`nwp_hist` ist keine Ablation mehr, sondern tragende Evidenz** für
  Contribution (iii), die Kurve „was kostet es, keine lokale Historie zu haben".
  Sie interpoliert zwischen `nwp` bei null Tagen und `nwp_hist` bei voller
  Historie. Also unbedingt behalten, aber im Paper anders einordnen.
- **Die Aggregations-Ablation fehlt**, die das Literaturaudit als faktisch
  verpflichtend führt: GATv2 ersetzt durch feste inverse-Distanz-Gewichte über
  dieselben k Gitterpunkte. Sie ist die Ablation, die `yang2025offgrid` fehlt,
  und sie beantwortet den Referee-Einwand „das ist doch nur Interpolation".
  Kostenvorteil gegenüber `base`: identische Architektur mit eingefrorenen
  Attention-Koeffizienten, also gleiche Dimensionen und gleiche Module, damit
  ist das Erben der Hyperparameter inhaltlich vertretbar und es braucht keine
  eigene Studie.
- **Die zwei neuen Ablationen aus `implementation_plan_ablations.md` sind nicht
  im Plan**: `neighbour_meas_available: false` (kein Stationsmesswert irgendwo)
  und `station_connectivity: none` (keine Stationskanten). Code liegt auf l1,
  17 Verifikationstests grün. Zusammen zerlegen sie den Beitrag des
  Stationsgraphen in den Messkanal und den Geometrie-/Kontextkanal und
  entscheiden, ob Contribution (i) ihren zentralen Abgrenzungssatz halten kann.
  Wenn sie nicht in dieser Kampagne mitlaufen, braucht es später eine dritte
  HPO-Runde.

Erwägenswert als Zuschnitt: **volles Variantenraster nur auf der
Headline-Architektur**, MTGNN und WaveNet nur in der `nwp`-Konfiguration als
Architekturvergleich. Das ist übliche Praxis, schrumpft die Studienzahl
deutlich, und die frei werdende Rechenzeit deckt die Ablationen aus dem
vorigen Absatz plus die räumliche k-fold-CV ab, die für die zentrale Aussage
wichtiger ist als die dritte Architektur in der dritten Variante.

## 10.5 Was ohnehin noch aussteht

Aus dem Literaturaudit, unabhängig von dieser Änderung, aber im selben
Kampagnenfenster zu planen, weil beides Rechenzeit braucht:

- **k-fold räumliche Kreuzvalidierung** statt des einzelnen 103/50/50-Splits.
  Siehe auch 9.5, die Topo-Skalierung muss dafür pro Fold neu gefittet werden.
- **Klassische Baselines**: Quantile Regression Forest und ein lokales
  MOS-Modell. Beide trainieren aus deterministischem NWP, beide sind Standard,
  und `schulz2022machine` zeigt, dass der QRF schwer zu schlagen ist. Laut
  Audit die billigste verfügbare Glaubwürdigkeitsinvestition.
- **Saisonale Stratifizierung** oder ein Testfenster über einen Winter. Das
  Fenster August bis Oktober 2025 schließt die Sturmsaison aus, die die
  deutsche Windstatistik dominiert. Diese Nachfrage kommt sicher.

---

# 11. Beschlossener Kampagnenzuschnitt (Stand 30. Juli 2026)

Entscheidungen des Autors nach dem Review in Abschnitt 9. Dieser Abschnitt ist
die verbindliche Fassung und ersetzt die Tabelle in Abschnitt 7.

## 11.1 Studienliste, 11 statt 9

| # | Studie | Config-Stem | Zweck im Paper |
|---|---|---|---|
| 1 | DCRNN_BASE | `wind_dcrnn_base` | Ablation `nwp_nodes` |
| 2 | DCRNN_NWP | `wind_dcrnn` | Headline-Modell |
| 3 | DCRNN_NWP_HIST | `wind_dcrnn_nwp_hist` | Endpunkt der Gauging-Kurve |
| 4 | **DCRNN_B** | `wind_dcrnn_nomeas` | **neu**, kein Stationsmesswert irgendwo |
| 5 | **DCRNN_C** | `wind_dcrnn_nograph` | **neu**, keine Stationskanten |
| 6-8 | MTGNN × base/nwp/nwp_hist | wie bisher | Architekturvergleich |
| 9-11 | WaveNet × base/nwp/nwp_hist | wie bisher | Architekturvergleich |

**B und C laufen nur für DCRNN**, und beide **ohne historische Messwerte an der
Zielstation** (`hist_wind_available: false`, ererbt aus `config_wind_dcrnn.yaml`).
Das ist nicht nur konsistent, es ist notwendig: behielte die Zielstation ihre
eigene Historie, gäbe es weiterhin Messwerte im Graphen und die Aussage „keine
Station trägt Messwerte" wäre schlicht falsch. Die Ablation wäre inkohärent.

Warum B und C eigene Studien bekommen statt die Hyperparameter zu erben: A minus B
ist die tragende Zahl für den Abgrenzungssatz von Contribution (i), also dafür,
dass die Korrektur auch aus den Live-Beobachtungen der Nachbarn gelernt wird.
Erben benachteiligt systematisch die Ablation. Solange die Ablation gut
abschneidet, ist das konservativ und unproblematisch. Das erwartete Ergebnis ist
aber, dass B schlechter ist, und genau dann lässt sich „Nachbarmessungen tragen
etwas" nicht mehr von „die geerbten Hyperparameter passten für B nicht" trennen.
Der Confound greift also präzise die Schlussfolgerung an, die gezogen werden soll.

Keine eigene Studie brauchen: die **IDW-Ablation** (identische Architektur, nur
eingefrorene Attention-Koeffizienten, also gleiche Dimensionen und Module minus
dem Attention-MLP), die **Shuffle-Kontrolle** (muss per Definition dieselben
Hyperparameter nutzen, sonst ist sie keine Kontrolle) und die **Fold-Trainings**
(ziehen sich die Parameter über den `_fold\d+`-Strip aus der Studie ihrer Variante).

## 11.2 `direction_to_adj` wird global entfernt

Der Parameter hat nirgends etwas gebracht und wird aus dem HPO-Suchraum genommen
sowie überall fest auf `false` gesetzt. Betrifft nur DCRNN, MTGNN und WaveNet
haben eigene Adjazenzmechanismen.

Zwei Nebenwirkungen, beide willkommen:

- Der DCRNN-Suchraum verliert eine Dimension, die Suche wird etwas effizienter.
- Variante C muss dadurch nur noch `K_hop` und `next_n_neighbors` pinnen statt
  drei Parameter.

Und eine Klarstellung, die die Interpretation von C **verbessert**: laut
Abschnitt 2 reduziert `edge_weight_from_attr` die `edge_attr` ohnehin auf Spalte 0,
also die Distanz. Ohne `direction_to_adj` ist der DCRNN-Stationsgraph damit ein
reiner fester Distanzkern. Die Differenz B minus C misst also sauber den Wert
eines distanzgewichteten räumlichen Kontexts, ohne Beimischung von Richtungs-
oder Höheninformation. Das ist im Paper leichter zu erklären als die vorherige
Situation.

## 11.3 Suchraum von Variante C

`K_hop` und `next_n_neighbors` werden gepinnt und aus dem `params`-Block
entfernt, `next_n_neighbors` auf das Minimum. Begründung: ohne Stationskanten
kann keiner der beiden das Modell beeinflussen. Das ist nicht vermutet, sondern
verifiziert — der Permutationstest der Ablations-Verifikation zeigt, dass die
Ausgabe an einem Knoten bitgleich bleibt, wenn die Features aller anderen Knoten
permutiert werden (`0.00e+00`), während dieselbe Permutation mit Delaunay-Kanten
sehr wohl durchschlägt (`2.22e+00`).

`next_n_neighbors` ist dabei nicht nur wirkungslos, sondern teuer: die
Nachbarknoten belegen Speicher und Rechenzeit und tragen nachweislich nichts bei.
Am Minimum wird C dadurch spürbar billiger pro Trial als A oder B, was den
Einzel-Worker aus 11.4 teilweise kompensiert.

Im Suchraum von C bleiben: `next_n_icond2`, `next_n_ecmwf`, `nwp_heads`,
`nwp_out_per_head`, `hidden`, `num_layers`, `dropout`, `lr`, `weight_decay`,
`grad_accum`, `horizon_decay`, `teacher_forcing_ratio`, `gradient_clip`. Die
NWP-Attention ist von dieser Ablation nicht betroffen, ihre Parameter bleiben
also alle sinnvoll.

## 11.4 Worker-Verteilung, ersetzt die Tabelle aus Abschnitt 7

Die beiden WaveNet-Doppel, die als **zweiter Worker** auf einer GPU lagen
(`WaveNet_BASE r2` auf A6000 #0 und `WaveNet_NWP_HIST r2` auf A6000 #1), werden
durch B und C ersetzt. Gesamtzahl bleibt 24 Worker.

| GPU | Worker 1 | Worker 2 |
|---|---|---|
| A100 #0 (L2) | DCRNN_BASE r1 | MTGNN_BASE r1 |
| A100 #1 (L2) | DCRNN_NWP r1 | MTGNN_NWP r1 |
| A100 #2 (L2) | DCRNN_NWP_HIST r1 | MTGNN_NWP_HIST r1 |
| A100 #3 (L2) | WaveNet_BASE r1 | WaveNet_NWP r1 |
| A6000 #0 (l1) | WaveNet_NWP_HIST r1 | **DCRNN_B r1** |
| A6000 #1 (l1) | WaveNet_NWP r2 | **DCRNN_C r1** |
| A6000 #2 (l1) | DCRNN_BASE r2 | MTGNN_BASE r2 |
| A6000 #4 (l1) | DCRNN_NWP r2 | MTGNN_NWP r2 |
| A6000 #5 (l1) | DCRNN_NWP_HIST r2 | MTGNN_NWP_HIST r2 |
| A6000 #6 (l1) | DCRNN_BASE r3 | MTGNN_NWP r3 |
| A6000 #7 (l1) | DCRNN_NWP r3 | MTGNN_NWP_HIST r3 |
| RTX4090 #0 (ws) | DCRNN_NWP_HIST r3 | — |
| RTX4090 #1 (ws) | MTGNN_BASE r3 | — |

Replikate danach: DCRNN base/nwp/nwp_hist je 3, MTGNN je 3, WaveNet_NWP 2,
WaveNet_BASE 1, WaveNet_NWP_HIST 1, DCRNN_B 1, DCRNN_C 1.

**Zwei Konsequenzen, die vor dem Start bedacht sein wollen:**

1. **B und C sind mit je einem Worker die langsamsten Studien der Kampagne.** Bei
   gleichem Trial-Budget brauchen sie rund die dreifache Wandzeit der
   Drei-Replikat-Studien und bestimmen damit das Ende der Kampagne. C ist durch
   den beschnittenen Raum und das gepinnte `next_n_neighbors` billiger, B nicht.
   Entweder Trial-Budget für B und C reduzieren, oder einplanen, dass sie
   nachlaufen.
2. **Kein Failover für B und C.** Das Prinzip „nie zwei Replikate derselben
   Studie auf derselben GPU" greift bei einem einzelnen Worker nicht mehr;
   stirbt er, steht die Studie. Da B und C für Contribution (i) tragend sind und
   WaveNet nur den Architekturvergleich stützt, wäre die Priorität streng
   genommen umgekehrt. Wer B und C je zwei Replikate geben will, gibt zusätzlich
   `WaveNet_NWP r2` ab. Auf den beiden 4090 ist kein Platz, die 24 GB tragen nur
   einen Worker.

## 11.5 Pre-Flight-Checkliste

Abzuarbeiten **bevor** ein Worker startet. Die ersten beiden Punkte sind die
Blocker aus 9.1 und 9.2.

**Blocker 1 — Code-Stand auf allen Hosts.** `l1` steht auf Branch `iaims26` und
hat den Topo-Code überhaupt nicht (`topo_features.py` fehlt, null Treffer für
`broadcast_topo`, der Branch `fix/mtgnn-topo-static-dim` existiert dort nicht
einmal als Remote-Ref). Der Plan setzt aber 12 der 24 Worker auf l1. Da alle
Worker einer Studie sich über die Postgres-Storage einen Studiennamen teilen und
Optuna nur Hyperparameter und Zielwert speichert, nicht die Modellklasse, würden
l1-Worker Trials eines Modells ohne Topo in dieselbe Studie schreiben. Kein
Absturz, keine Warnung, das beste Trial wäre bedeutungslos, und zwar in allen
Studien gleichzeitig.

- [ ] `fix/mtgnn-topo-static-dim` auf l1 auschecken, auf `ws` prüfen
- [ ] Vor dem Branchwechsel auf l1 die uncommitteten Änderungen sichern: die vier
      dcrnn-Configs, das gelöschte `config_wind_dcrnn_nwp_hist.yaml`, und den
      Ablations-Patch (`neighbour_meas_available`, `station_connectivity: none`,
      Backups liegen als `*.bak_ablation`)
- [ ] `git rev-parse HEAD` auf allen drei Hosts vergleichen, als Startbedingung
      ins Launch-Skript, Abbruch bei Abweichung
- [ ] Eine Zeile pro Worker ins Log: Hostname, Commit-Hash,
      `station_node_features`, `broadcast_topo`. Ohne das ist ein solcher Fehler
      im Nachhinein nicht nachweisbar

**Blocker 2 — MTGNN und WaveNet auf den richtigen Topo-Lader umstellen.** Alle
vier Skripte rufen `load_topo_node_features` (Kanten-Differenz-Lader), nur DCRNN
ruft `load_topo_station_features`. Damit fehlen MTGNN und WaveNet die
varianzstabilisierenden Transforms, ihre z-Skalierung läuft über `train + val`
statt nur über Train, und die tdi-Flachland-Behandlung fehlt. Der
Architekturvergleich wäre durch die Vorverarbeitung konfundiert.

- [ ] `train_mtgnn.py`, `hpo_mtgnn.py`, `train_wavenet.py`, `hpo_wavenet.py` auf
      `load_topo_station_features` umstellen, `n_train` durchreichen
- [ ] Rückgabewert ist ein `(N, F)`-Array, `homo_sampler` erwartet ein Dict:
      entweder umsetzen oder `_topo_arrays` direkt aus dem Array füllen
- [ ] Kommentar in `homo_sampler.py:243` anpassen („already z-score normalised in
      load_topo_node_features")
- [ ] Danach die fehlende MTGNN-Permutationskontrolle fahren, **nicht vorher**:
      solange der Topo-Kanal nach der Skalierung nahezu konstant ist, ändert eine
      Permutation folgerichtig nichts, und „Terrain trägt nichts" wäre ein
      Skalierungsartefakt statt ein Befund

**Weitere Punkte:**

- [ ] Median-Imputation in **beiden** Ladern auf `values.iloc[:n_train].median()`
      umstellen (leckt derzeit Val-Statistik, entgegen der Zusage im Docstring)
- [ ] `aspect_sin`/`aspect_cos` bei undefinierter Exposition auf 0 setzen statt
      Median, analog zur tdi-Behandlung. Vorher zählen, wie viele Stationen
      betroffen sind
- [ ] `direction_to_adj` aus allen DCRNN-`params`-Blöcken entfernen und in den
      `dcrnn`-Sektionen auf `false` setzen
- [ ] Ablations-Patch auf den Topo-Branch portieren. **Geprüft: alle sechs Anker
      existieren dort mit exakt den erwarteten Trefferzahlen** (Sampler-Konstruktor 1,
      Maskierung train 1, Maskierung eval 1, evaluation-Signatur 2, kwarg-Stellen
      2/1/1, delaunay-Zweig 1). Das Patch-Skript läuft also unverändert durch
- [ ] C-Configs generieren, sobald der Patch auf dem Branch liegt (B-Configs
      existieren bereits auf l1)
- [ ] Falls BASE behalten wird: HPO-Ränge mit NWP harmonisieren
      (`next_n_icond2` derzeit 1–4 gegen 1–7, `K_hop` 1–2 gegen 1–3). Sonst darf
      die NWP-Variante strikt mehr und der Vergleich ist keine saubere Ablation

## 11.6 Eine offene Entscheidung, die die Stationslisten bestimmt

Die HPO validiert auf `n_val_stations: 50` aus `val_files`. Führt das Paper
später räumliche k-fold-Kreuzvalidierung, tauchen genau diese Stationen in
irgendeinem Fold als Teststationen auf, und die Hyperparameter wären in Kenntnis
dieser Stationen gewählt worden. Bei einer Studie, deren Kernaussage „Skill an
vollständig zurückgehaltenen Stationen" lautet, ist das die Art Detail, die
auffällt.

Zwei vertretbare Auswege: entweder eine Gruppe Stationen komplett aus der HPO
heraushalten und nur dort die Headline-Zahl berichten, oder es offenlegen und
begründen. Die Entscheidung fällt vor dem Start, weil sie die Stationslisten der
Kampagne festlegt.

---

# 12. Räumliche Kreuzvalidierung (Stand 31. Juli 2026) — neu zu reviewen

**Dieser Abschnitt ist der eigentliche Gegenstand des zweiten Reviews.** Er
ersetzt die offene Entscheidung aus 11.6 und erledigt den Fallstrick aus 9.5
sowie den ersten Punkt aus 10.5. Der Rest des Dokuments beschreibt weiterhin die
Topo-Features und ist unverändert gültig.

## 12.1 Was sich geändert hat und warum

Die drei Fold-Configs je Architektur unterschieden sich bisher **nur im
Zeitfenster** (`test_start`/`test_end`), der Stationssplit war über alle Folds
identisch. Zusätzlich lief **innerhalb** der HPO eine zweite, davon unabhängige
CV: eine zeitliche Expanding-Window-Schleife über `hpo.n_folds: 3`, deren
Fold-Grenzen aus `test_start` abgeleitet wurden (`hpo_*.py`). Beide Ebenen maßen
zeitliche Robustheit.

Die zentrale Aussage der Studie ist aber Generalisierung auf ungesehene
*Standorte*. Deshalb rotieren jetzt die **Stationen** statt der Zeitfenster:

- Die 153 Stationen aus `files + val_files` sind in `configs/spatial_folds.yaml`
  in drei Gruppen à 51 partitioniert. Jede Station ist genau einmal Val, sonst
  Train (102/51 pro Fold).
- Die zeitliche Expanding-Window-CV in den HPO-Skripten ist durch diese drei
  räumlichen Folds ersetzt: ein Trial trainiert dreimal, Optuna bekommt wie
  bisher den Mittelwert.
- Das Zeitfenster ist in allen drei Folds **identisch** (siehe 12.3).

Die 50 `test_files` bleiben unangetastet und sind in keinem Fold Ziel — die
Frage aus 11.6 ist damit im Sinne von „Gruppe komplett aus der HPO heraushalten"
entschieden.

## 12.2 Warum nicht geblockt (die übliche Empfehlung)

Die Lehrbuchvariante für räumliche CV — zusammenhängende Blöcke zurückhalten —
ist hier **architekturfeindlich**, weil Nachbarstationen bei dieser Architektur
ein bewusster Input sind und kein Leckagekanal. Ein geblockter Split nimmt den
Zielstationen genau den Messkanal weg, den das Modell nutzen soll. Gemessen an
den echten 153 Stationen (geodätisch, WGS-84):

| Strategie | größte Nachbar-Lücke (Val→nächste Train) | Terrain-Ungleichgewicht |
|---|---|---|
| Geblockt (Längengrad-Streifen) | 261.7 km | 0.252 |
| Zufällig | 85.6 km | 0.197 |
| **Gestreut + terrain-balanciert** | **70.8 km** | **0.046** |

Verwendet wird gestreut + terrain-balanciert: räumlich nahe Tripel bilden, aus
jedem Tripel geht eine Station in einen Fold; die Zuordnung innerhalb der Tripel
wird per lokaler Suche so permutiert, dass die Terrain-Mittel der Folds
möglichst gleich sind. Damit liegt zu jeder Val-Station garantiert ihr
unmittelbarer räumlicher Partner im Train-Satz.

Erzeugt von `geostatistics/make_spatial_folds.py` (`--compare` reproduziert die
Tabelle, `--n-val` variiert die Fold-Größe, `--write` schreibt die YAML).

**Zwei Zahlen, die ein Reviewer kennen sollte:**

- Die 51/51/51-Folds erreichen median 40.8 km Abstand Val→nächste Train-Station.
  Die **Untergrenze** des 153er-Netzes (jede Station zur nächsten der 152
  anderen) liegt bei median 39.3 km — die Folds liegen also praktisch auf dem
  Optimum. Eine Variante mit kleineren Val-Mengen (30/30/30, 123 Trainings-
  nachbarn) wurde durchgerechnet und bringt **nichts** (median 40.7 km), kostet
  aber Terrain-Balance (0.120 statt 0.046) und statistische Aussagekraft.
- Die finale Testauswertung ist mit median 23.9 km **dichter vernetzt** als jeder
  HPO-Fold (50 Ziele gegen alle 153 Nachbarn). Das ist eine Eigenschaft des
  fixen, per Kennard-Stone gewählten Testsets, nicht der Fold-Konstruktion, und
  es geht in die unkritische Richtung: die Hyperparameter werden unter etwas
  dünnerer Vernetzung gewählt, als das finale Modell vorfindet. Gehört so in die
  Methodik.

## 12.3 Das Zeitfenster — drei Perioden, in jedem Fold gleich

| Periode | Fenster | Rolle |
|---|---|---|
| Training | 2023-07-24 → 2024-07-31 | ein Jahr |
| Validierung | 2024-08-01 → 2025-07-31 | ein Jahr |
| Test | ab 2025-08-01 | zurückgehalten, nur `--test-mode` |

Die Validierung ist damit räumlich **und** zeitlich out-of-sample: 51 ungesehene
Stationen in einem ungesehenen Jahr. Die Folds unterscheiden sich ausschließlich
in den Stationen.

Umgesetzt über **einen** neuen Config-Schlüssel, weil die Skripte Run-Paare
ohnehin an einer Zeitgrenze teilen — es kam nur die zweite Grenze dazu:

```yaml
data:
  val_start:  '2024-08-01'   # Train davor, Validierung danach
  test_start: '2025-08-01'   # ab hier zurueckgehalten (finaler Test)
```

`--test-mode` schaltet `val_start` bewusst ab: dort *ist* die Testperiode ab
`test_start` die Val-Menge, mit `test_files` als Zielstationen. Damit ist auch
der Einwand aus 10.5 („Testfenster schließt die Sturmsaison aus") erledigt — das
Testfenster umfasst jetzt ein volles Jahr ab August 2025 inklusive Winter.

## 12.4 Geänderte Dateien

| Datei | Änderung |
|---|---|
| `geostatistics/spatial_cv.py` | **neu** (~130 Z.) — lädt/validiert `spatial_folds.yaml`, bildet Stationspool und Fold-Indizes, löst `hpo.cv_mode` auf |
| `geostatistics/make_spatial_folds.py` | überarbeitet — geodätisch statt Haversine, `--n-val`/`--n-folds`/`--compare`/`--write` |
| `geostatistics/fold_dashboard.py` | **neu** — Streamlit-Karte der Train/Val/Test-Aufteilung je Fold, Nachbar-Kennzahlen, Terrain-Balance |
| `geostatistics/stgnn/utils/topo_features.py` | `load_topo_station_features(_dict)` nimmt wahlweise `n_train` **oder** `train_idx` |
| `geostatistics/hpo_{dcrnn,mtgnn,wavenet}.py` | `cv_mode`-Weiche, Fold-Pläne statt `fold_splits`, Scaler und Topo-z-Score pro Fold, Cache-Key auf den Pool abgebildet (je ~200 Z. Diff) |
| `geostatistics/train_{dcrnn,mtgnn,wavenet}.py` | `val_start` als zweite Zeitgrenze (je ~25 Z. Diff) |
| `configs/{dcrnn,mtgnn,wavenet}/*_fold[123].yaml` | 33 Configs: Stationslisten, `val_start`/`test_start`, `hpo.cv_mode: spatial`, `n_val_stations: null`, `test_end`/`val_frac` entfernt |

## 12.5 Der Fallstrick aus 9.5 ist erledigt

9.5 warnte: sobald der Stationssplit pro Fold wechselt, muss die Topo-Skalierung
pro Fold neu gefittet werden, sonst leckt jede Fold-Val-Menge in die
Skalierungsstatistik — und zwar still.

Genau das war der invasivste Teil der Umstellung. `arr[:n_train]` setzte voraus,
dass `station_ids` train-zuerst sortiert ist. Im räumlichen Modus wird der
153er-Pool **einmal** geladen (sortierte Vereinigung, unabhängig davon, welche
Fold-Config übergeben wurde) und pro Fold nur umindiziert; die Rollen sind dann
nicht mehr durch die Reihenfolge kodiert. `load_topo_station_features` nimmt
deshalb jetzt explizite `train_idx`, und `n_train` bleibt nur für den
zeitlichen Pfad. Gilt gleichermaßen für Median-Imputation und z-Score.

Belegt im Log (mtgnn, 1-Trial-Lauf gegen die echte fold1-Config):

```
Zeitfenster je Fold — Train bis 2024-08-01 (1473 Paare), Val 2024-08-01 bis 2025-08-01 (1460 Paare)
Loaded 8 topographic station features (z-score on 102 train stations, explicit indices)
spatial_fold1 — 102 Trainings-/51 Zielstationen
spatial_fold2 — 102 Trainings-/51 Zielstationen
spatial_fold3 — 102 Trainings-/51 Zielstationen
Trial 1 spatial_fold1 — best_val_rmse=1.6670
Trial 1 spatial_fold2 — best_val_rmse=1.8262
Trial 1 spatial_fold3 — best_val_rmse=1.9107
```

Dieselbe Zeile erscheint dreimal mit **je 102** Stationen — nicht einmal mit 153.

## 12.6 Rückwärtskompatibilität (bewusste Anforderung)

Der zeitliche Modus wird nicht verworfen, er bleibt wählbar:

- `hpo.cv_mode` ist per Default `temporal` → Expanding Window wie bisher.
- Ohne `data.val_start` fällt die Val-Grenze wie bisher auf `test_start`.
- 15 Configs (u. a. alle `_fold9`) tragen keinen der neuen Schlüssel und laufen
  unverändert.
- Der `val_frac`-Fallback bleibt bestehen und ist **nicht** toter Code.

## 12.7 Worauf das Review besonders schauen sollte

1. **Scaler-Leckage im räumlichen Modus.** `meas_scaler`, `e2_scaler` und der
   Topo-z-Score fitten jetzt auf `[:scaler_t, fold_train_idx]`. `scaler_t` ist
   der Index von `val_start`, `fold_train_idx` sind die 102 Trainingsstationen
   des Folds. Bitte prüfen, ob irgendwo noch ein `:N_train` steht, das im
   räumlichen Modus 153 statt 102 bedeutet — `stat_scaler.fit(raw_static[:N_train])`
   in `hpo_dcrnn.py` ist ein bewusst belassener Fall (rein geometrische Features,
   wird gecacht, daher fold-unabhängig), aber genau der Typ Stelle, der geprüft
   gehört.
2. **Cache-Key.** Im räumlichen Modus werden `files`/`val_files` im Key durch den
   gemeinsamen Pool ersetzt (sonst drei identische Cache-Einträge für drei
   Fold-Configs), und `val_start` fällt aus dem Key (es teilt nur bereits
   geladene Paare auf). `test_start`/`test_end` bleiben drin, weil sie den
   geladenen Zeitraum kappen. Ist die Argumentation lückenlos?
3. **`--test-mode`-Pfad.** `val_start` wird dort abgeschaltet. Stimmt damit die
   Kette Retrain → `get_test_results_*.py` noch, insbesondere die
   Scaler-Zuschnitte und die Topo-Skalierung im Testlauf? Die
   `get_test_results_*.py` wurden **nicht** angefasst und rufen weiterhin
   `n_train=N_train` — das ist im Testlauf korrekt (dort ist `all_ids`
   train-zuerst), sollte aber gegengeprüft werden.
4. **Studiennamen.** `hpo_stem = re.sub(r'_fold\d+$', '', config_stem)` ist
   unverändert. Im räumlichen Modus ignorieren die HPO-Skripte die
   `files`/`val_files` der übergebenen Config — es ist also egal, welche der drei
   Fold-Configs die HPO bekommt, und alle drei zeigen weiter auf **eine**
   gemeinsame Studie. Ist das gewollt eindeutig oder eine Fußangel?
5. **`hpo.n_val_stations`.** Stand auf 50 und hätte die 51er-Fold-Val-Menge um
   eine Station beschnitten. In den Configs jetzt `null`; `build_folds()` warnt,
   falls doch gesetzt. Reicht die Warnung, oder sollte es ein Fehler sein?
6. **`spatial_folds.yaml` als stille Abhängigkeit.** Die HPO liest die
   Stationslisten aus dieser Datei statt aus der Config. Ändert jemand die Datei
   zwischen zwei Workern derselben Studie, mischen sich unbemerkt zwei
   Fold-Definitionen in einer Optuna-Studie — dieselbe Fehlerklasse wie Blocker 1
   in 11.5. Sollte der Fold-Hash mit ins Log oder in die Trial-`user_attrs`?

## 12.8 Stand der Verifikation

- [x] `--preprocess-only` für mtgnn und wavenet: räumlicher Modus erkannt,
      153-Stationen-Pool, ein gemeinsamer Cache-Key
- [x] 1-Trial/1-Epochen-Lauf mtgnn gegen die echte fold1-Config: alle drei Folds
      durchlaufen, Zeitfenster und Stationszahlen wie oben belegt
- [ ] Dasselbe für wavenet und dcrnn (läuft zum Zeitpunkt des Schreibens)
- [ ] Cache-Key-Vergleich der drei Fold-Configs untereinander (erwartet:
      identisch, weil alle denselben Pool laden)
- [ ] `--test-mode`-Durchlauf gegen einen retrainierten Checkpoint
- [ ] Gegenprobe, dass eine unveränderte Config (`_fold9`) im zeitlichen Modus
      identische Fold-Grenzen erzeugt wie vor der Umstellung

---

# 13. Review-Ergebnis räumliche CV (externe Session, 31. Juli 2026)

Unabhängiges Review ohne Kenntnis der vorherigen Sitzungen. Jeder Punkt aus den
Abschnitten 9–12 wurde am aktuellen Code neu geprüft, nicht aus dem Dokument
übernommen. Geprüft auf **l2** (`fix/mtgnn-topo-static-dim`, HEAD `4f62937` plus
den uncommitteten Änderungen) und auf **l1**. **`ws` war während des gesamten
Reviews nicht erreichbar** (`ssh: connect to host 10.166.32.253 port 22: Network
is unreachable`) — die dortigen 2 Worker aus der Tabelle in 11.4 sind damit
ungeprüft.

## 13.1 Einordnung

**Die räumliche CV selbst ist sauber implementiert.** Der Fallstrick aus 9.5 ist
nicht nur adressiert, sondern numerisch belegt: die Topo-z-Skalierung fittet
nachweislich pro Fold neu, dasselbe gilt für Messwert- und ECMWF-Scaler. Es
wurde **keine** neue Leckage gefunden, und die Behauptungen in 12.1–12.3 halten
der Gegenrechnung stand. Das ist der gute Teil.

**Die Kampagne ist in der geplanten Form trotzdem nicht startfähig**, aus vier
unabhängigen Gründen, von denen drei neu sind und keiner mit der CV-Umstellung
zu tun hat:

1. `hpo_dcrnn.py` bricht bei **jedem** Trial ab — reproduziert, mit Traceback.
   Betrifft alle fünf DCRNN-Studien.
2. Die Studiennamen der Fold-Configs treffen nicht die Namen, die der Retrain
   sucht. 12.7 Punkt 4 ist in dem Punkt sachlich falsch.
3. Der Topo-Kanal ist zwischen den Architekturen **unterschiedlich breit**, und
   in der Mehrzahl der Configs fehlt der Pfad, ohne den er gar nicht lädt.
4. Blocker 9.1 ist unverändert offen und durch die neuen Dateien größer geworden.

Die Reihenfolge unten ist nach Schweregrad, nicht nach Dokumentabschnitt.

## 13.2 Status jedes Punkts aus dem Vor-Review

| Punkt | Status |
|---|---|
| 9.1 l1 hat den Topo-Code nicht | **nicht behoben**, verschärft (13.3 B4) |
| 9.2 MTGNN/WaveNet falscher Topo-Lader | **behoben** (13.7) |
| 9.3 Median-Imputation leckt | **behoben** im Stations-Lader; im Kanten-Lader unverändert, dort aber nicht mehr relevant |
| 9.3 `aspect_sin`/`aspect_cos` | **behoben** (13.7) |
| 9.4 Cache-Platzierung | **weiterhin in Ordnung** (13.7) |
| 9.5 Skalierung pro Fold | **behoben und numerisch verifiziert** (13.6) |
| 10.5 Testfenster ohne Winter | **behoben** durch `test_start: 2025-08-01` ohne `test_end` |
| 11.1 Studien B und C | **nicht umgesetzt** (13.4 H1) |
| 11.2 `direction_to_adj` global entfernt | **teilweise** (13.5 M1) |
| 11.5 Ablations-Patch auf den Branch portieren | **nicht gemacht** (13.4 H1) |
| 11.6 HPO-Val-Stationen | **strukturell gelöst**, verifiziert (13.6) |
| 12.5 Fallstrick 9.5 erledigt | **bestätigt**, mit eigener Messung |
| 12.6 `val_frac`-Fallback „nicht toter Code" | **nicht zutreffend** — in `hpo_dcrnn.py` unerreichbar (13.5 L1) |
| 12.7.1 Scaler-Leckage | **geprüft, sauber** (13.6) |
| 12.7.2 Cache-Key | **geprüft, sauber**, Argumentation aber teils gegenstandslos (13.5 M7) |
| 12.7.3 `--test-mode`-Pfad | im Trainingspfad **korrekt**, im Evaluationspfad **nicht** (13.4 H2) |
| 12.7.4 Studiennamen | **nicht zutreffend, weil** die HPO-Skripte den Fold-Suffix gar nicht strippen (13.3 B2) |
| 12.7.5 `n_val_stations` | in den Configs korrekt `null`, Code-Behandlung asymmetrisch (13.5 M3) |
| 12.7.6 `spatial_folds.yaml` als stille Abhängigkeit | **offen**, und schwerwiegender als vermutet (13.4 H3) |
| 12.8 Verifikationsliste | dcrnn-Punkt ist **fälschlich als „läuft" markiert** — der Lauf ist abgestürzt (13.3 B1) |

## 13.3 Blocker

### B1 — `hpo_dcrnn.py` scheitert an jedem Trial, und die Studie meldet trotzdem Erfolg

Reproduziert am 31.07.2026 mit der Foldcheck-Config aus der vorigen Sitzung:

```
ValueError: GraphConfig.topo_feature_names is set but station_ids was not
            passed to HeterogeneousGraphBuilder.build().
  File "geostatistics/hpo_dcrnn.py", line 1033, in objective
  File "geostatistics/stgnn/graph_builder.py", line 94, in build
```

`geostatistics/graph_builder.py:92-97` verlangt `station_ids`, sobald
`GraphConfig.topo_feature_names` gesetzt ist. Beide `build()`-Aufrufe in
`hpo_dcrnn.py` — `hpo_dcrnn.py:815` (statischer Graph) und `hpo_dcrnn.py:1033`
(Rebuild pro Trial) — übergeben `station_ids` **nicht**. `train_dcrnn.py` tut es
an allen drei Stellen (`train_dcrnn.py:663`, `:677`, `:987`), `hpo_dcrnn.py` an
keiner.

`topo_feature_names` wird aus `dcrnn.edge_features` abgeleitet, und die enthält
in **jeder** Wind-DCRNN-Config topografische Namen. Ausnahme sind ausgerechnet
`config_wind_dcrnn_nwp_hist_fold{1,2,3}.yaml`, die gar kein `edge_features`
haben — die laufen, dafür ohne jede Topo-Information.

Wichtig für die Zuordnung: **das ist keine Regression der CV-Umstellung.** Die
fehlende Übergabe steht identisch im committeten Stand (`git show
HEAD:geostatistics/hpo_dcrnn.py`, Zeilen 759 und 914), und die topografischen
Namen stehen schon auf `HEAD` in `edge_features`. Der Bug ist älter; er trifft
die Kampagne nur jetzt vollständig.

Zweiter, eigenständiger Teil des Befunds: `hpo_dcrnn.py:1211` ruft
`study.optimize(objective, n_trials=remaining, catch=(Exception,))`. Damit wird
jeder Trial-Absturz zu einem FAIL-Trial degradiert, die Schleife läuft weiter
und das Skript loggt am Ende `HPO COMPLETE`. Im echten Log
(`logs/hpo_dcrnn_foldcheck_dcrnn.log`) stehen zwischen „Trial 0 —
hyperparameters" und „HPO COMPLETE" genau **eine Sekunde** und keine einzige
Fold-Zeile. Genau deshalb ist der Punkt in 12.8 als „läuft zum Zeitpunkt des
Schreibens" notiert statt als Fehlschlag. Ein Worker, der so stirbt, ist von
außen nicht von einem unterscheidbar, der arbeitet.

**Zu tun:** `station_ids=all_ids` an beide `build()`-Aufrufe. Danach den
`catch=(Exception,)`-Griff überdenken: mindestens die ersten *n* Fehler laut
loggen und abbrechen, wenn kein Trial COMPLETE wird. Sonst kostet ein
Tippfehler in einer Config eine ganze Kampagnennacht, ohne dass es jemand merkt.

### B2 — Die Fold-Configs erzeugen Studiennamen, die kein Retrain findet

12.7 Punkt 4 sagt, `hpo_stem = re.sub(r'_fold\d+$', '', config_stem)` sei
unverändert und alle drei Fold-Configs zeigten weiter auf **eine** Studie.
Nachgeprüft: `hpo_stem` existiert in den HPO-Skripten überhaupt nicht.

| Skript | Zeile | Studienname |
|---|---|---|
| `hpo_dcrnn.py` | 234, 246 | `cl_m-dcrnn_out-48_freq-1h_{config_stem}` — **ungestrippt** |
| `hpo_mtgnn.py` | 310, 327 | `cl_m-mtgnn_…_{config_stem}` — ungestrippt |
| `hpo_wavenet.py` | 296, 313 | `cl_m-wavenet_…_{config_stem}` — ungestrippt |
| `train_dcrnn.py` | 338, 353 | `…_{hpo_stem}` — **gestrippt** |
| `train_mtgnn.py` | 265, 286 | gestrippt |
| `train_wavenet.py` | 251, 272 | gestrippt |
| `get_test_results_{dcrnn,mtgnn,wavenet}.py` | 173/140/153 | gestrippt |

Belegt im Log: `study: cl_m-dcrnn_out-48_freq-1h_foldcheck_dcrnn` trägt den
Config-Stem wörtlich.

Bisher fiel das nicht auf, weil die HPO immer mit der **Nicht**-Fold-Config lief
(so auch in `geostatistics/run_hpo_study.sh`, das
`config_wind_dcrnn_nwp_hist_new.yaml` startet). Genau diese Nicht-Fold-Configs
sind aber die einzigen, die **nicht** auf räumliche CV umgestellt wurden:

```
config_wind_dcrnn.yaml            cv_mode=None  103/50  test_start 2025-08-01  test_end 2025-10-31  val_frac 0.25
config_wind_dcrnn_fold1.yaml      cv_mode=spatial  102/51  val_start 2024-08-01  test_start 2025-08-01
```

Daraus folgt die Zwickmühle:

- HPO auf einer **Fold-Config** → räumliche CV, aber drei getrennte Studien
  (`…_wind_dcrnn_fold1/2/3`), jede mit denselben drei Folds, also dreifache
  Rechenzeit für dasselbe Ergebnis; und `train_dcrnn.py` sucht danach
  `…_wind_dcrnn` und findet nichts.
- HPO auf der **Nicht-Fold-Config** → der Retrain findet die Studie, aber die
  gesamte räumliche CV ist inaktiv (`cv_mode` fehlt → Default `temporal`), mit
  dem alten 103/50-Split und `test_end: 2025-10-31`.

Es gibt aktuell **keine** Config, die beides erfüllt.

**Zu tun:** entweder `hpo_stem` auch in den drei HPO-Skripten einführen (dann
teilen sich die drei Fold-Configs eine Studie, was im räumlichen Modus korrekt
ist, weil sie ohnehin denselben Pool und dieselben Folds laden) — oder die
neuen Schlüssel in die Nicht-Fold-Configs ziehen und die HPO weiter dort fahren.
Die erste Variante ist die sauberere, weil `n_val_stations`, `val_start` und
`cv_mode` dann nur an einer Stelle gepflegt werden.

### B3 — Der Topo-Kanal ist zwischen den Architekturen ungleich, und meist gar nicht ladbar

Drei getrennte Beobachtungen, die zusammen den Architekturvergleich kippen.

**(a) Unterschiedliche Featurezahl.** `TOPO_FEATURE_ORDER` hat 9 Einträge. Die
`edge_features`-Listen in den Configs enthalten 8 topografische Namen —
`elev_std` fehlt. DCRNN löst ausschließlich über
`parse_station_node_features` auf, hat also **ohne** CLI-Flag null Topo-Features.
MTGNN und WaveNet fallen ohne Flag auf `parse_edge_features` zurück
(`hpo_mtgnn.py:615-618`, `hpo_wavenet.py` analog) und bekommen dann **8**.
Belegt in den Foldcheck-Logs vom 31.07.:

```
hpo_mtgnn_foldcheck_mtgnn.log:19
  Loaded 8 topographic station features (z-score on 102 train stations,
  explicit indices): ['slope','aspect_sin','aspect_cos','tpi5','tpi75','tdi','z0','dist_coast']

hpo_dcrnn_foldcheck_dcrnn.log
  (keine solche Zeile — DCRNN lief mit 0 Topo-Features)
```

Der Vergleich DCRNN vs. MTGNN vs. WaveNet wäre damit 0 gegen 8 gegen 8 Features.
Das ist derselbe Konfundierungstyp wie Blocker 9.2, nur eine Ebene höher.

**(b) `topo_features_path` fehlt in der Mehrzahl der Kampagnen-Configs.** Sobald
`--station-node-features` gesetzt wird, wirft der Ladeblock
(`hpo_dcrnn.py:674-679`, `hpo_mtgnn.py:628-633`, `hpo_wavenet.py:613-618`) ein
`ValueError`. Betroffen:

| Config | `topo_features_path` |
|---|---|
| `dcrnn/config_wind_dcrnn_nwp_hist_fold{1,2,3}` | **fehlt** |
| `mtgnn/config_wind_mtgnn_nwp_hist_fold{1,2,3}` | **fehlt** |
| `wavenet/config_wind_wavenet_fold{1,2,3}` | **fehlt** |
| `wavenet/config_wind_wavenet_nwp_fold1` | vorhanden |
| `wavenet/config_wind_wavenet_nwp_fold{2,3}` | **fehlt** |
| `wavenet/config_wind_wavenet_nwp_hist_fold{1,2,3}` | **fehlt** |

**(c) Fold-interne Inkonsistenz bei WaveNet.** `wavenet_nwp_fold1` hat den Pfad,
`fold2` und `fold3` nicht. Die drei Fold-Configs einer Studie unterscheiden sich
also in etwas, das die Modellbreite bestimmt. Solange die HPO nur eine davon
liest, fällt das nicht auf; beim Retrain der drei Folds entstehen drei
verschieden breite Modelle aus einer Studie.

**Zu tun:** `station_node_features: all` und `topo_features_path` in **alle**
Kampagnen-Configs schreiben statt sich auf CLI-Flags und den
`edge_features`-Fallback zu verlassen. Der Fallback ist für einen
Architekturvergleich der falsche Mechanismus, weil er von einer Liste abhängt,
die für einen ganz anderen Zweck (Kanten-Differenzen) gepflegt wird. Und einen
Vergleich der finalen Featurezahl über die drei Architekturen ins Log, analog zu
der Zeile, die DCRNN schon schreibt.

### B4 — l1 hat weiterhin weder den Topo- noch den CV-Code (offener Blocker 9.1)

Frisch geprüft:

```
l1: git rev-parse --abbrev-ref HEAD   -> iaims26
l1: git rev-parse HEAD                -> 1853a9920e723fcf66fb8d008e5259e4e415f739
l1: git branch -a                     -> iaims26, main, remotes/origin/main
l1: ls geostatistics/stgnn/utils/topo_features.py  -> No such file or directory
l1: ls geostatistics/spatial_cv.py                 -> No such file or directory
l1: ls configs/spatial_folds.yaml                  -> No such file or directory

l2: fix/mtgnn-topo-static-dim @ 4f62937ef296fc2234932c40ea0fecc6011b6f9f
```

`fix/mtgnn-topo-static-dim` existiert auf l1 nach wie vor nicht einmal als
Remote-Ref. Die Argumentation aus 9.1 gilt unverändert und wiegt jetzt schwerer:
zusätzlich zur fehlenden Topo-Dimension würde ein l1-Worker **zeitliche** Folds
rechnen und den Mittelwert daraus in dieselbe Optuna-Studie schreiben wie ein
l2-Worker mit räumlichen Folds. Zwei Zielgrößen mit verschiedener Bedeutung in
einer Studie, ohne Absturz und ohne Warnung.

`ws` konnte nicht geprüft werden (Host nicht erreichbar).

**Zu tun:** unverändert die Checkliste aus 11.5, plus die Startbedingung
`git rev-parse HEAD` im Launch-Skript. Solange die nicht existiert, ist das ein
Fehler, der sich beliebig oft wiederholen kann.

## 13.4 Schwerwiegend

### H1 — Studien B und C existieren nicht, der Ablations-Patch ist nicht portiert

11.1 legt 11 Studien fest, darunter `DCRNN_B` (`wind_dcrnn_nomeas`) und
`DCRNN_C` (`wind_dcrnn_nograph`). Stand jetzt:

- Auf **l2** (Topo-/CV-Branch): weder `nomeas`- noch `nograph`-Configs. Der
  Ablations-Patch selbst ist ebenfalls nicht portiert —
  `grep -rn "neighbour_meas_available" geostatistics/` liefert **null** Treffer,
  und `station_connectivity == "none"` fehlt im `graph_builder.py` (nur
  `delaunay`/`knn`, sonst `ValueError`, `graph_builder.py:166-178`).
- Auf **l1** (`iaims26`): der Patch ist da (`hpo_dcrnn.py:873`,
  `graph_builder.py:146`, `sampler.py:48`, Backups als `*.bak_ablation`), und
  `config_wind_dcrnn_nomeas{,_fold1,_fold2,_fold3}.yaml` existieren. Sie sind
  aber **nicht** auf räumliche CV umgestellt: 103/50 Stationen, kein `val_start`,
  kein `cv_mode`, `test_start: 2024-07-31`, `test_end: 2024-11-30`,
  `n_val_stations: 50`. `neighbour_meas_available: false` ist gesetzt,
  `hist_wind_available` dagegen nicht — 11.1 verlangt dort ausdrücklich `false`;
  der Default ist zwar `False` (`hpo_dcrnn.py:1049`), aber bei einer Ablation,
  deren ganze Aussage an diesem Schalter hängt, sollte er explizit dastehen.
- `wind_dcrnn_nograph` (Variante C) gibt es auf keinem der beiden Hosts.

Damit tragen die beiden Studien, die 11.1 als **tragend für den
Abgrenzungssatz von Contribution (i)** ausweist, im Moment gar nichts. Das ist
kein Code-Fehler, aber es ist der Punkt mit dem größten Abstand zwischen Plan
und Stand.

### H2 — `get_test_results_*.py` kennen `val_start` nicht und werten im Dev-Modus auf der Testperiode aus

12.4 hält fest, dass die Eval-Skripte nicht angefasst wurden, und 12.7 Punkt 3
prüft nur den `--test-mode`-Fall. Der ist korrekt (siehe 13.6). Der
**Dev-Modus** ist es nicht.

`get_test_results_dcrnn.py:256-266` kennt nur `test_start` und `test_end`:

```python
test_start = data_cfg.get("test_start")
if test_start:
    split_t = int(np.searchsorted(timestamps, pd.Timestamp(test_start, tz="UTC"), side="left"))
...
logger.info("Test period starts at %s", split_time)
```

`train_dcrnn.py:565` dagegen:

```python
boundary = test_start if (args.test_mode or not val_start) else val_start
```

Ein Fold-Modell, das mit `val_start: 2024-08-01` trainiert wurde, wird also bei
einem Eval-Lauf **ohne** `--test-mode` gegen den Zeitraum ab `2025-08-01`
gescored — die zurückgehaltene Testperiode. Weil `test_end` in den Fold-Configs
entfernt wurde, ist das zusätzlich das volle Jahr, nicht nur ein Ausschnitt.

Das ist keine Leckage im Training, aber es verbraucht das Testset unbemerkt bei
einem Lauf, der nach seiner Bezeichnung ein Validierungslauf ist. Bei einer
Studie, deren Headline-Zahl ausdrücklich „an vollständig zurückgehaltenen
Stationen und Zeiten" lautet, ist das genau der Fehler, den man nicht machen darf.

**Zu tun:** `val_start` in die drei `get_test_results_*.py` übernehmen, mit
derselben `boundary`-Logik wie in `train_*.py`, und in den Log schreiben, welche
Periode gerade bewertet wird.

### H3 — `spatial_folds.yaml` ist nicht reproduzierbar

12.7 Punkt 6 nennt die Datei eine stille Abhängigkeit. Sie ist schlimmer als
still: sie ist mit dem aktuellen Generator **nicht mehr herstellbar**.

Gemessen an der committeten Datei, mit den Funktionen des Skripts selbst:

```
committed configs/spatial_folds.yaml
  Fold 1: 51/102  median 40.8  p90 54.3  max 57.7
  Fold 2: 51/102  median 40.6  p90 52.1  max 60.8
  Fold 3: 51/102  median 40.8  p90 55.7  max 70.8
  Terrain-Ungleichgewicht 0.046 (tpi5)   groesste Luecke 70.8 km
```

Das bestätigt die Zahlen in 12.2 **exakt** (70.8 km, 0.046). Die Datei ist also
genau die, die dort beschrieben wird. Aber:

```
make_dispersed(D, topo, 3, None, seed=0)  # aktuelles Skript, Default-Seed
  Terrain-Ungleichgewicht 0.029 (tpi75)   groesste Luecke 70.8 km
  Identische Fold-Zuordnung: 87/153 Stationen  -> ABWEICHEND
```

Und `make_spatial_folds.py --compare` liefert für dieselbe Strategie nochmals
andere Zahlen (max 80.1 km, Ungleichgewicht 0.030). Alles bei `seed=0`, also
nicht durch Zufall erklärbar.

Die Ursache ist eine Reihenfolgeabhängigkeit: `make_spatial_folds.py:270` baut
`ids = files + val_files` in **Config-Reihenfolge**, während
`spatial_cv.station_pool()` (`spatial_cv.py:84`) `sorted(union)` liefert. Die
gierige Tripel-Bildung in `_build_groups` und die lokale Suche in
`make_dispersed` hängen an dieser Reihenfolge, also erzeugt jede Config mit
anderer Listenreihenfolge eine andere Partition — und das Skript wurde laut 12.4
seit dem Schreiben der YAML überarbeitet.

Praktisch heißt das: die Fold-Definition ist ein Artefakt, das nicht aus dem
Repository heraus nachvollzogen werden kann. Für die Kampagne ist das
verkraftbar, solange die Datei nicht angefasst wird. Für das Paper ist es das
nicht — die Fold-Konstruktion ist Teil der Methodik, und ein Referee, der die
Reproduzierbarkeit prüft, bekommt andere Folds.

**Zu tun:** `ids` im Generator auf `sorted(...)` umstellen, damit Generator und
Loader dieselbe Reihenfolge verwenden, und die YAML einmal neu erzeugen; oder
die Datei so wie sie ist einfrieren und ihre Herkunft (Skript-Commit,
Argumente) im Kopf der YAML dokumentieren. Zusätzlich, wie in 12.7 Punkt 6
vorgeschlagen: einen Hash über die Fold-Definition in die Trial-`user_attrs` und
ins Log. Das kostet drei Zeilen und macht die Fehlerklasse aus Blocker 9.1 auf
Fold-Ebene überhaupt erst nachweisbar.

## 13.5 Mittel und klein

**M1 — `direction_to_adj` ist nicht global entfernt (11.2).** Noch im
`params`-Block, also weiterhin im Suchraum:
`dcrnn/config_wind_dcrnn.yaml`, `config_wind_dcrnn_nwp_hist_fold{1,2,3}.yaml`,
`config_wind_dcrnn_nwp_hist_new{,_fold1,_fold2,_fold3,_fold9}.yaml`.
Bei `config_wind_dcrnn_fold{1,2,3}.yaml` und `_fold9` wurde der Schlüssel
stattdessen ganz **entfernt**; der Code-Default ist `False`
(`geostatistics/dcrnn/config.py:122` und `:212`), das Verhalten ist also
richtig, aber 11.2 verlangt das explizite `false`. Für ein Paper, das die
Ablation C über „reiner fester Distanzkern" begründet, sollte der Schalter
sichtbar in der Config stehen, nicht implizit im Default.

**M2 — Zwei `nwp_hist`-Familien nebeneinander.** Es gibt
`config_wind_dcrnn_nwp_hist_fold{1,2,3}` **und**
`config_wind_dcrnn_nwp_hist_new_fold{1,2,3}` (analog für MTGNN), beide mit
`cv_mode: spatial`. Die erste Familie hat weder `edge_features` noch
`topo_features_path`, die zweite beides. 11.1 nennt als Stem
`wind_dcrnn_nwp_hist` — das ist die **feature-ärmere**. Vor dem Start
entscheiden und die andere Familie löschen oder umbenennen; sonst hängt das
Ergebnis von Studie 3 daran, welchen Dateinamen ein Worker bekommen hat.

**M3 — `n_val_stations` wird uneinheitlich behandelt.**
`hpo_dcrnn.py:900` ruft `build_folds(..., max_val_stations=hpo_cfg.get("n_val_stations"))`
(Default `None`), `hpo_mtgnn.py:719` und `hpo_wavenet.py:706` reichen dagegen die
Variable `n_val_stations` durch, die weiter oben mit `hpo_cfg.get("n_val_stations", 40)`
belegt wird. Eine Config **ohne** den Schlüssel würde bei MTGNN/WaveNet also 51
Zielstationen auf 40 kürzen — mit Warnung (`spatial_cv.py:107-113`), aber ohne
Abbruch. Alle Kampagnen-Configs setzen `null`, der Fall greift heute nicht.
Zur Frage aus 12.7 Punkt 5: bei `cv_mode: spatial` sollte ein gesetztes
`n_val_stations` ein **Fehler** sein, kein Warning. Die Folds definieren ihre
Zielmenge vollständig; jeder Wert dort ist eine stille Verkleinerung, die
niemand beabsichtigt.

**M4 — `GNNCache.make_key` ignoriert die Modellsektion von MTGNN und WaveNet.**
`utils/data_cache.py` liest `cfg.get("dcrnn", cfg.get("stgnn2", {}))`. MTGNN- und
WaveNet-Configs haben keinen dieser beiden Schlüssel, also fallen
`icond2_features`, `ecmwf_features`, `measurement_features`, `next_n_icond2`,
`next_n_ecmwf`, `icond2_run_hours` und `use_altitude_diff` auf Defaults zurück.
Nachgemessen: `config_wind_mtgnn_nwp_fold1.yaml` und
`config_wind_wavenet_nwp_fold1.yaml` ergeben **denselben** Cache-Key
(`219b3173f80404ac`). Heute harmlos, weil beide Configs identische Featurelisten
haben. Aber jede künftige Änderung an `mtgnn.measurement_features` oder
`mtgnn.icond2_features` invalidiert den Cache **nicht** und liefert stillschweigend
Arrays der falschen Breite — dieselbe Fehlerklasse wie der Bug aus `e7bb373`,
nur über die Cache-Grenze statt über die Zweiggrenze. Vorbestehend, aber im
selben Atemzug zu beheben.

**M5 — `stat_scaler` fittet in der HPO auf 153, im Retrain auf 102 Stationen.**
`hpo_dcrnn.py:598-601` fittet `raw_static[:N_train]`, und `N_train` ist im
räumlichen Modus die volle Poolgröße 153. `train_dcrnn.py:830-832` fittet
dieselben drei Spalten (lat, lon, alt) auf `[:N_train]` mit `N_train = 102`
(`files` der Fold-Config). Das ist der in 12.7 Punkt 1 „bewusst belassene Fall",
und als Leckagefrage ist die Argumentation richtig: Koordinaten und Höhe einer
Zielstation sind bei einem induktiven Modell ohnehin bekannte Eingaben. Der
Punkt ist ein anderer: die HPO tunt gegen eine andere Normierung dieser drei
Kanäle als der Retrain sie herstellt. Der Effekt ist klein (Mittelwert und
Streuung über 153 gegen 102 deutsche Stationen), aber es ist genau die
Diskrepanz zwischen HPO-Modell und Retrain-Modell, vor der Abschnitt 5 des
Briefings warnt. Entweder in beiden Skripten auf den vollen Pool fitten, oder in
beiden auf die Fold-Trainingsmenge — nur nicht verschieden.

**M6 — Kein Provenienz-Log.** Weder Hostname, noch Commit-Hash, noch
`station_node_features`, noch `broadcast_topo`, noch ein Fold-Hash landen im Log
oder in den Trial-`user_attrs` (`grep -n "set_user_attr\|rev-parse\|hostname"`
über die drei HPO-Skripte: keine Treffer). Der Vorschlag aus 9.1 und aus 12.7
Punkt 6 ist unverändert offen — und nach B1 bis B4 gleich viermal der Mechanismus,
der die Fehler im Nachhinein nachweisbar gemacht hätte.

**M7 — Das `val_start`-Pop im Cache-Key ist wirkungslos.**
`hpo_dcrnn.py:365` (und analog in den beiden anderen) entfernt `val_start` aus
der Key-Config, mit ausführlichem Kommentar. `GNNCache.make_key` liest
`val_start` aber gar nicht — der Schlüssel besteht aus `path`, `nwp_path`,
`ecmwf_path`, `files`, `val_files`, `test_start`, `test_end`, `interpol_path`,
`knnimputer_path` und den Modell-Featurelisten. Das Pop ist ein No-op und der
Kommentar behauptet eine Wirkung, die es nicht gibt. Kein Schaden, aber die Art
Kommentar, auf die sich später jemand verlässt.

**L1 — Der `val_frac`-Fallback in `hpo_dcrnn.py` ist toter Code.** 12.6 sagt
ausdrücklich das Gegenteil. `hpo_dcrnn.py:492` erreicht den Zweig nur bei
`cv_mode == "temporal"` **und** fehlendem `test_start` — und genau diese
Kombination bricht wenige hundert Zeilen später unbedingt ab
(`hpo_dcrnn.py:858-862`, „`hpo.cv_mode='temporal'` braucht `data.test_start`").
In `train_*.py` lebt der Fallback dagegen wirklich.

**L2 — `fold_splits` überlebt nur noch in `hpo_dcrnn.py`.** Dort existieren
jetzt zwei Fold-Begriffe nebeneinander: die Zeitfenster-Liste `fold_splits`
(`hpo_dcrnn.py:874-881`, nur im zeitlichen Modus gefüllt) und die neue,
modusübergreifende `fold_plans`-Liste (`:903 ff.`). `fold_splits` wird nur noch
an einer Stelle gelesen (`:945`), um daraus `fold_plans` zu bauen.
`hpo_mtgnn.py` und `hpo_wavenet.py` sind sauberer — dort ist `fold_splits`
komplett verschwunden. Zur ausdrücklichen Frage, ob zwei verwechselbare
Fold-Konzepte im selben Skript koexistieren: **ja, aber ungefährlich.** Der
Trial-Loop iteriert ausschließlich über `fold_plans`, `fold_splits` ist im
räumlichen Modus garantiert leer, und der alte Chunk-Kommentar bei `:830-849`
steht korrekt beim zeitlichen Zweig. Aufräumen ist Kosmetik, keine
Fehlerbehebung.

**L3 — `n_folds` und `min_train_date` werden im räumlichen Modus still
ignoriert.** Die Foldzahl kommt dort aus `spatial_folds.yaml`. Alle
Kampagnen-Configs tragen weiterhin `n_folds: 3`, was zufällig stimmt. Ein
`n_folds: 5` würde kommentarlos nichts bewirken. Eine Warnung wäre billig.

**L4 — `aspect_sin`/`aspect_cos` werden getrennt z-skaliert.** Der Kommentar in
`topo_features.py:52` („`aspect_*`: already bounded in [-1, 1] -> untouched")
bezieht sich nur auf `_TOPO_TRANSFORMS`; die z-Skalierung greift danach trotzdem
und verwendet für Sinus und Cosinus verschiedene Mittelwerte und Streuungen. Die
Kreisgeometrie wird dadurch anisotrop verzerrt — der Einheitskreis wird zur
Ellipse, und der Winkelabstand zwischen zwei Expositionen ist nicht mehr
richtungsunabhängig. Für eine Größe, die genau als Richtung gemeint ist und mit
der zeitvariablen Windrichtung interagieren soll, ist das unschön. Der saubere
Weg wäre, die beiden Kanäle von der z-Skalierung auszunehmen (sie sind bereits
auf [-1, 1] normiert) oder beide mit derselben, gemeinsamen Statistik zu
skalieren. Kein Fehler, aber eine Stelle, an der ein Gutachter mit
Geostatistik-Hintergrund nachfragt.

**L5 — Der Wirkungsbereich der 9.3-Korrekturen ist sehr klein.** Nachgezählt im
153er-Pool: genau **eine** Station (`02961`) hat `elev_std == 0`, und sie ist
auch die einzige mit fehlendem `tdi`. Es gibt **kein** NaN in `aspect`, und
**keine** fehlenden Topo-Werte überhaupt — der Median-Imputationspfad wird für
diesen Pool nie betreten. Die beiden Korrekturen aus 9.3 sind also richtig und
sauber implementiert, aber sie ändern das Ergebnis für eine von 153 Stationen.
Das ist kein Grund, sie zurückzunehmen; es ist ein Grund, sie nicht als
Kernbefund im Paper zu führen.

## 13.6 Nachgerechnet und in Ordnung

Alles Folgende wurde nicht gelesen, sondern gemessen.

**Die Topo-z-Skalierung fittet tatsächlich pro Fold neu** (der zentrale Punkt
aus 9.5 und 12.5). Eigener Lauf gegen `configs/spatial_folds.yaml` und die
echten Topo-CSVs:

```
spatial_fold1: train-mean(|.|) = 2.51e-08   train-std = 1.0000 .. 1.0000
spatial_fold2: train-mean(|.|) = 2.34e-08   train-std = 1.0000 .. 1.0000
spatial_fold3: train-mean(|.|) = 2.57e-08   train-std = 1.0000 .. 1.0000
```

Mittelwert 0 und Streuung 1 gelten also je Fold auf **dessen eigenen** 102
Trainingsstationen, nicht global. Dass das kein Nulleffekt ist, zeigt der
direkte Vergleich der resultierenden Feature-Matrizen:

```
fold1 gegen fold2   max |Diff| je Feature: [0.196 0.043 0.037 0.132 0.104 2.243 0.163 0.327 0.068]
fold1 gegen fold3   max |Diff| je Feature: [0.550 0.057 0.031 0.046 0.058 0.192 0.137 0.041 0.075]
fold1 gegen global-153-Fit                : [0.232 0.022 0.003 0.062 0.038 0.598 0.098 0.101 0.010]
```

Ein global gefitteter z-Score hätte einzelne Stationen um bis zu 0.6 Sigma
(`tdi`) anders normiert. Die Umstellung war notwendig, nicht vorsorglich.

**Die übrigen Scaler sind pro Fold korrekt zugeschnitten.** Geprüft an jeder
`.fit()`-Stelle:

- `hpo_dcrnn.py:1070` `meas_raw[:fold_train_t, fold_train_idx]` — Zeit **und**
  Station korrekt eingeschränkt.
- `hpo_dcrnn.py:1083` `trial_sta_ecmwf[:fold_train_t, fold_train_idx]` — dito;
  hier ist ECMWF stationsinterpoliert `(T, N_stationen, E2)`, die
  Stationsindizierung ist also nötig und vorhanden.
- `hpo_dcrnn.py:1076`, `hpo_mtgnn.py:822`, `hpo_wavenet.py:806`
  `_t_icond2[fold_r]` — Gitterknoten, kein Stationsbezug, Run-Maske korrekt.
- `hpo_mtgnn.py:831` / `hpo_wavenet.py:815` `_t_ecmwf[:fold_t]` **ohne**
  Stationsindex: hier ist ECMWF gitterbasiert, die zweite Achse ist
  `len(ecmwf_coords)`, nicht `len(all_ids)` (belegt durch das
  `reshape(T, len(ecmwf_coords), E2)` zwei Zeilen darunter). Korrekt, nur auf
  den ersten Blick asymmetrisch zu DCRNN.
- `hpo_dcrnn.py:1097` skaliert das Kriging-Lag-Feature mit **denselben**
  Fold-Statistiken wie die Messwerte. Konsistent.
- Kein einziges verbliebenes `[:N_train]` in einem fold-abhängigen Kontext. Die
  einzige `N_train`-Stelle im Scaler-Pfad ist `hpo_dcrnn.py:600` (siehe M5).

**Die HPO-Val-Stationen aus 11.6 sind strukturell entschärft.** Gemessen: die
50 `test_files` und der 153er-Pool sind **disjunkt** (Schnittmenge 0). Die
Fold-Partition ist sauber: 3 × (102 train / 51 val), kein Überlapp innerhalb
eines Folds, die Vereinigung der drei Val-Mengen ist genau der Pool, jede
Station also genau einmal Ziel. Die Wahl aus 11.6 ist damit im Sinne von
„Gruppe komplett aus der HPO heraushalten" getroffen, und zwar tatsächlich und
nicht nur auf dem Papier.

**Der Cache-Key bildet die drei Fold-Configs auf einen Eintrag ab.** Selbst
gerechnet:

```
dcrnn    fold1/2/3  roh: 717a2f12…, 6a6d7cce…, 9f113627…   raeumlich: 5681ff7640bdb2e4 (3x)
mtgnn    fold1/2/3  roh: 5a2699d0…, b796b46a…, f78dcada…   raeumlich: 219b3173f80404ac (3x)
wavenet  fold1/2/3  gleich wie mtgnn                        raeumlich: 219b3173f80404ac (3x)
```

Die Argumentation aus 12.7 Punkt 2 stimmt für `files`/`val_files`; für
`val_start` ist sie gegenstandslos (M7), und die mtgnn/wavenet-Kollision ist ein
eigener Befund (M4).

**Die Cache-Platzierung ist unverändert korrekt** (der explizit nachgefragte
Punkt aus 9.4). Der Topo-Block liegt in allen drei Skripten hinter dem
Cache-Schreibblock und vor der ersten Sampler-/Modellkonstruktion:
`hpo_dcrnn` 655 → 657 ff. → 815/1033, `hpo_mtgnn` 605 → 607 ff. → 839/872,
`hpo_wavenet` analog. Der Bug aus `e7bb373` ist nicht zurückgekehrt.

**Blocker 9.2 ist behoben.** Alle sieben Aufrufstellen benutzen jetzt den
Absolutwert-Lader:

```
train_dcrnn.py:845           load_topo_station_features      (n_train=N_train)
hpo_dcrnn.py:680             load_topo_station_features      (train_idx=…)
train_mtgnn.py:453           load_topo_station_features_dict (n_train=N_train)
hpo_mtgnn.py:634             load_topo_station_features_dict (train_idx=…)
train_wavenet.py:438         load_topo_station_features_dict (n_train=N_train)
hpo_wavenet.py:619           load_topo_station_features_dict (train_idx=…)
get_test_results_{mtgnn,wavenet}.py:257/274                  (n_train=N_train)
```

`load_topo_node_features` wird nur noch aus `graph_builder.py:98` gerufen, also
ausschließlich für Kanten-Differenzen — genau die Rolle, für die er gebaut ist.
Der Kommentar in `homo_sampler.py:244` ist mitgezogen. Die
`_resolve_train_idx`-Weiche (`topo_features.py:64-86`) erzwingt genau eines von
`n_train`/`train_idx` und prüft Grenzen; ein Aufruf ohne beides schlägt fehl
statt still das Falsche zu tun. Sauber gelöst.

**Der `--test-mode`-Pfad im Training ist korrekt.** `train_dcrnn.py:406-418`
setzt dort `train_ids = files + val_files` (102 + 51 = 153) und
`val_ids = test_files` (50), also `all_ids` train-zuerst mit `N_train = 153`.
`n_train=N_train` in `train_dcrnn.py:846` trifft damit tatsächlich die
Trainingsstationen. `train_dcrnn.py:565` schaltet `val_start` im Testmodus ab
und `val_cutoff` auf `None`, die Testperiode läuft also ab `2025-08-01` bis
Datenende ohne Kappung. 12.7 Punkt 3 ist für diesen Teil bestätigt; der Mangel
liegt allein in den Eval-Skripten (H2).

**Das Zeitfenster ist über alle drei Folds identisch**, wie in 12.3 behauptet.
Aus dem eigenen Wiederholungslauf:

```
Zeitfenster je Fold — Train bis 2024-08-01 (1473 Paare), Val 2024-08-01 bis 2025-08-01 (1460 Paare)
spatial_fold1 — 102 Trainings-/51 Zielstationen
spatial_fold2 — 102 Trainings-/51 Zielstationen
spatial_fold3 — 102 Trainings-/51 Zielstationen
```

Dieselben Paarzahlen in allen drei Folds, dieselbe Stationszahl, und eben nicht
153 — die Behauptung aus 12.5 trifft zu.

**Die Zahlen in 12.2 stimmen für die committete Fold-Datei** (siehe H3 für die
Reproduzierbarkeit): median 40.6–40.8 km, größte Lücke 70.8 km,
Terrain-Ungleichgewicht 0.046. Die Rangfolge geblockt ≫ zufällig > gestreut ist
in jedem Lauf stabil, die geblockte Variante reproduziert mit 261.7 km exakt.

**`spatial_cv.py` selbst ist solide.** `load_spatial_folds` prüft, dass alle
Folds denselben Pool abdecken (`:73-75`) und dass `files` und `val_files` je Fold
disjunkt sind (`:76-78`) — beide Prüfungen greifen genau die Fehler ab, die eine
handeditierte YAML produzieren würde. `_norm` mit `zfill(5)` fängt die
YAML-typische int/str-Uneindeutigkeit ab; nachgeprüft, dass PyYAML alle 153 IDs
in der aktuellen Datei als 5-stellige Strings liest (die oktal-mehrdeutigen sind
im Dump korrekt gequotet). `station_pool` liest zwar nur `folds[0]`, was durch
die Poolgleichheitsprüfung davor abgesichert ist.

## 13.7 Was das für die Paper-Story bedeutet

Contribution (i) in `story_positioning.md` §4 verlangt wörtlich, dass die
Abbildung von Standort auf Korrektur **gemeinsam** aus statischen
topographischen Deskriptoren, aus der leadtime-aufgelösten NWP-Prognose und aus
den Live-Beobachtungen der Nachbarstationen gelernt wird, und nennt als
Falsifikationsverfahren ausdrücklich „unter k-fold räumlicher
Kreuzvalidierung".

Die räumliche CV ist damit der zweite von zwei Bausteinen, ohne die der Satz
nicht belegbar wäre, und sie ist implementiert. Was noch fehlt, ist nicht
konzeptionell, sondern mechanisch:

- Der **topographische Deskriptor** liegt in DCRNN aktuell bei null Features und
  in MTGNN/WaveNet bei acht (B3). Solange das so ist, prüft die Kampagne den
  ersten Teilsatz von Contribution (i) für die Headline-Architektur überhaupt
  nicht.
- Die **Nachbarbeobachtungen** sind der Teilsatz, den B und C auseinandernehmen
  sollen. Beide Studien existieren nicht (H1). Ohne sie ist der Abgrenzungssatz
  gegenüber den fünf induktiven Vorarbeiten unbelegt — und das ist laut 11.1
  genau die Zahl, für die eigene Studien statt geerbter Hyperparameter
  beschlossen wurden.
- Der **Wintereinwand** aus 10.5 ist mit `test_start: 2025-08-01` ohne `test_end`
  erledigt, solange die Daten tatsächlich bis in den Winter reichen. Das sollte
  vor der finalen Auswertung einmal gegen den tatsächlichen Datenrand geprüft
  werden, nicht gegen die Config.

Und einen Punkt, der als Stärke ins Paper gehört: die Begründung in 12.2, warum
**nicht** geblockt wurde, ist ungewöhnlich gut belegt und beantwortet einen
Einwand, der sonst sicher gekommen wäre. Die Zahl, die dabei den Ausschlag gibt
— 40.8 km Fold-Median gegen 39.3 km Untergrenze des Netzes — ist das Argument
dafür, dass die Folds die Nachbarschaftsstruktur nicht künstlich ausdünnen. Die
sollte in die Methodik, zusammen mit dem Hinweis, dass die finale Testauswertung
mit 23.9 km **dichter** vernetzt ist und die Hyperparameterwahl damit in die
konservative Richtung verzerrt ist.

## 13.8 Vorschlag für die Reihenfolge

Vor dem ersten Worker, in dieser Reihenfolge, weil jeder Schritt die folgenden
billiger macht:

1. `station_ids=all_ids` an beide `build()`-Aufrufe in `hpo_dcrnn.py` (B1) und
   danach ein 1-Trial-Lauf gegen `config_wind_dcrnn_fold1.yaml`, der drei
   `spatial_fold*`-Zeilen mit `best_val_rmse` produziert. Ohne diesen Nachweis
   ist jede weitere Aussage über die DCRNN-Studien wertlos.
2. `catch=(Exception,)` entschärfen: Abbruch, wenn nach den ersten *k* Trials
   keiner COMPLETE ist (B1, zweiter Teil).
3. Studiennamen vereinheitlichen (B2) — `hpo_stem` in die drei HPO-Skripte.
4. `station_node_features: all` und `topo_features_path` in alle
   Kampagnen-Configs, `nwp_hist`-Dopplung auflösen, `direction_to_adj` explizit
   `false` (B3, M1, M2).
5. `val_start` in die drei `get_test_results_*.py` (H2). Das ist der einzige
   Punkt, der bei Nichtbeachtung das Testset unwiederbringlich verbraucht.
6. l1 auf den Branch bringen, `ws` prüfen, Commit-Hash-Gate ins Launch-Skript,
   Provenienz-Zeile pro Worker inklusive Fold-Hash (B4, M6, H3).
7. Ablations-Patch portieren, B- und C-Configs auf räumliche CV erzeugen (H1).
8. Erst danach: die MTGNN-Permutationskontrolle aus 10.1, die jetzt zum ersten
   Mal das Richtige misst.

Die Punkte 1 bis 5 sind zusammen wenige Stunden Arbeit und entscheiden darüber,
ob die Kampagne überhaupt verwertbare Trials produziert. Punkt 7 entscheidet
darüber, ob Contribution (i) ihren Abgrenzungssatz behalten kann.

**Backup dieses Dokuments vor der Ergänzung:**
`docs/topo_features_review_brief.md.bak_20260731_prereview2`
