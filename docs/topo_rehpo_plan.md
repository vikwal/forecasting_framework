# Plan: Neue HPO-Studien für DCRNN/MTGNN/WaveNet mit topografischen Features

**Zweck:** Vorbereitender Plan für den Fall, dass das laufende Topo-Screening
(fold1/GRID, siehe `geostatistics/compare_topo_screening.py`) bestätigt, dass
topografische Features einen echten Performance-Gewinn bringen. Dieses
Dokument wird **nicht sofort umgesetzt** — es hält fest, was zu tun wäre und
wie die 12 verfügbaren GPUs effizient genutzt werden könnten, damit bei
grünem Licht schnell gestartet werden kann.

## 0. Start-Bedingung

Nicht vor Abschluss von:

- MTGNN-Arme topoA/B/C (laufen aktuell, ~5-7h verbleibend)
- Permutationskontrollen für DCRNN und WaveNet (`--shuffle-node-features`,
  laufen aktuell) — erst wenn der Effekt auch gegen die "mehr-Kapazität"-
  Alternative bestehen bleibt, ist er eine belastbare Grundlage für einen
  teuren Re-HPO-Lauf.
- Idealerweise: Bestätigung, dass der Effekt nicht nur ein Ein-Seed-Zufall
  ist (mind. grobe Plausibilität über mehrere Epochen-Aggregationen, wie
  bereits praktiziert).

Solange diese drei Punkte offen sind, sollte kein GPU-Kontingent für einen
vollen Re-HPO reserviert werden.

## 1. Umfang

3 Architekturen × 3 Varianten (`base`, `nwp`, `nwp_hist`) = **9 Studien**,
zunächst nur auf **fold1**, analog zur bisherigen Praxis (jede fold-spezifische
Config hat ihre eigene Optuna-Studie; `launch_train_pipeline.py` referenziert
diese Studien anschließend für Retrain auf allen 3 Folds). Erst nach Sichtung
der fold1-Ergebnisse entscheiden, ob fold2/fold3 eigene Studien brauchen oder
ob die fold1-Best-Params übernommen werden (wie bisher gehandhabt).

### Entscheidung: einheitlich Arm C als HPO-Ziel

**Regel (a priori, unabhängig vom Screening-Ausgang):** Jede Architektur
erhält die topographischen Features über **jeden Kanal, den sie strukturell
unterstützt** — MTGNN und WaveNet über Adjazenz *und* Feature-Strom (= Arm C),
DCRNN nur über den Feature-Strom, weil sein Stationsgraph konstruktionsbedingt
keinen Kanten-Feature-Mechanismus besitzt (`edge_weight_from_attr` reduziert
`edge_attr` auf die Distanzspalte, Li et al. 2018 wie publiziert).

Begründung — drei Punkte, in der Reihenfolge ihres Gewichts:

1. **Vermeidung von Winner's Curse.** Würde man pro Architektur den im
   Screening besten Arm auswählen und diese Konfigurationen anschließend als
   Hauptergebnis berichten, selektierte man auf derselben Ein-Seed-Stichprobe,
   die man auswertet. Der berichtete Topo-Vorteil wäre systematisch
   überschätzt. Eine vorab fixierte, für alle Architekturen gleiche Regel
   immunisiert dagegen. Das Screening beantwortet damit nur die Frage, für die
   es gedacht war ("bringt Topographie überhaupt etwas?") und wird nicht zur
   Konfigurationsauswahl zweckentfremdet.
2. **C ist ein Superset von B** (verifiziert in den Trainingslogs: WaveNet
   topoB und topoC laden beide dieselben 9 Topo-Features in `emb_mlp` → adaptive
   Adjazenz; C ergänzt nur den Broadcast in die Input-Kanäle). Die Umstellung
   von WaveNet auf C nimmt also nichts weg, was dort gewirkt hat — die Kosten
   der Konsistenz sind nach oben begrenzt und liegen bei WaveNet innerhalb der
   Lauf-zu-Lauf-Streuung.
3. **Vergleichbarkeit der Benchmark-Zeilen.** Ein einheitlicher Kanal macht die
   Manipulation über alle drei Architekturen identisch ("gleiche Information,
   unterschiedlicher Inductive Bias"). WaveNet ist ohnehin kein Flagship-Modell;
   Experimentkonsistenz wiegt dort schwerer als die letzte Zehntelstelle.

**Konsequenz für den Umfang:** HPO-Ziel ist einheitlich Arm C für alle drei
Architekturen, 9 Studien wie oben. Die Ablationen laufen anschließend an C's
Hyperparametern (siehe Abschnitt 3a).

Screening-Befund als Kontext (nicht als Auswahlkriterium) — **finaler Stand,
alle 6 Läufe abgeschlossen** (Zwischenstände während des Laufs von MTGNN A
waren irreführend, siehe Hinweis unten):

| Architektur | A (ohne Topo) | B (Adjazenz) | C (Adj.+Broadcast) |
|---|---|---|---|
| DCRNN | 1.1937 | strukturell n/a | 1.1343 (−4.98 %) |
| MTGNN | 1.1634 | 1.2451 (**+7.03 %**) | 1.1498 (−1.17 %) |
| WaveNet | 1.2897 | 1.1840 (−8.20 %) | 1.2076 (−6.37 %) |

**Wichtige Korrektur gegenüber einem früheren Zwischenstand:** `topoA` (MTGNN,
ohne Topo) lief zum Zeitpunkt einer vorläufigen Auswertung noch (Epoche
78/200) und verbesserte sich danach substanziell weiter bis Early Stop bei
Epoche 155 (1.2401 → 1.1634). Ein Vergleich mit einem noch nicht finalen
Baseline-Wert hatte MTGNNs C-Vorteil zunächst auf −7.28 % beziffert; final
sind es nur −1.17 %. **Lehre:** `best_val_rmse` aus einem noch laufenden Prozess
nie als Endergebnis in eine Vergleichstabelle übernehmen, auch nicht als
vorläufige Zahl mit Sternchen — die Fehleinschätzung entstand genau dadurch.

Bemerkenswert am finalen Bild: das Muster ist architekturabhängig und bei
MTGNN uneinheitlich zu den anderen beiden. DCRNN profitiert robust vom
Feature-Strom, WaveNet robust von der Adjazenz. MTGNN zeigt keinen robusten
Vorteil durch Topographie in keinem Kanal — B ist dort sogar eindeutig
schädlich (+7.03 %), und C's kleiner Vorteil (−1.17 %) liegt näher am
Rauschband (vgl. WaveNets C-vs-B-Differenz, σ≈0.10–0.13) als die Effekte bei
DCRNN/WaveNet. Da für MTGNN nie eine Permutationskontrolle lief (anders als
bei DCRNN und WaveNet), ist der Kapazitäts-Confound für diesen kleinen Effekt
nicht ausgeschlossen. Genau diese Uneinheitlichkeit ist der Grund, die
Konfiguration *nicht* empirisch pro Architektur zu wählen (siehe Entscheidung
oben) — bei MTGNN wäre die datengetriebene Wahl ohnehin uneindeutig.

### 3a. Ablationen

An C's Hyperparametern, nicht neu getunt (Standard-Ablationskonvention: das
volle Modell wird getunt, Komponenten werden bei festen Parametern entfernt):

- **A** (keine Topo-Features) — für alle drei Architekturen.
- **B** (nur Adjazenz, kein Broadcast) — für MTGNN und WaveNet. Bei DCRNN
  strukturell nicht definierbar.

WaveNets B-Befund (−8.20 % gegenüber A, besser als C) bleibt damit als
Ablationszeile erhalten und sollte mit dem Hinweis berichtet werden, dass der
Unterschied C↔B dort innerhalb der Epochen-zu-Epochen-Streuung lag (σ ≈
0.10–0.13, ein Seed).

**Bias-Richtung offenlegen:** Ablationsarme laufen am Optimum des vollen
Modells und sind dadurch benachteiligt. Daraus folgt eine sparsame
Entscheidungsregel: nur wenn ein Ablationsarm *verliert*, ist eine
bestätigende Mini-HPO auf diesem Arm nötig; gewinnt er trotz Handicap, ist das
Ergebnis bereits konservativ.

Erwähnenswert fürs Paper: das hier ausgewertete Screening war sogar noch
konservativer — die verwendeten Best-Params stammten aus den alten Studien, die
*ohne* statische Features optimiert wurden, also aus A's Optimum. Dass B und C
trotzdem klar besser abschnitten, ist ein Ergebnis gegen den Bias.

## 2. Nötige Code-Änderungen vor dem Start

Die HPO-Skripte sind **nicht** auf demselben Stand wie die Trainingsskripte:

- `hpo_dcrnn.py`, `hpo_wavenet.py`: haben aktuell **kein**
  `--station-node-features` / `--broadcast-topo` CLI-Flag. Diese fehlen
  komplett und müssten analog zu `train_dcrnn.py`/`train_wavenet.py` ergänzt
  werden (gleiches Lademuster: `parse_station_node_features`,
  `load_topo_station_features`/`load_topo_node_features`,
  `static_dim`/`topo_dim`/`broadcast_topo` durchreichen).
- `hpo_mtgnn.py`: wurde im Rahmen des Screenings nur soweit gefixt, dass
  `static_dim` konsistent mit dem alten Edge-Diff-Pfad (`parse_edge_features`)
  ist — das war ein reiner Bugfix für ladbare Checkpoints, **nicht** die
  Umstellung auf den neuen Knoten-Feature-/Broadcast-Pfad. Für eine Studie,
  die den Feature-Strom-Kanal mit-tunen soll, muss `hpo_mtgnn.py` auf
  denselben Stand wie `train_mtgnn.py` gehoben werden (broadcast_topo-Flag,
  `topo_dim` etc.), sonst optimiert die Studie ein Modell, das den Kanal gar
  nicht nutzen kann, den wir eigentlich tunen wollen.
- Configs prüfen: `topo_features_path` fehlte z. B. in
  `config_wind_wavenet_nwp_fold1.yaml` bis zum Screening-Fix — vor Start
  gegen alle 9 Ziel-Configs (base/nwp/nwp_hist × 3 Architekturen) prüfen, ob
  der Pfad gesetzt ist.

## 3. Sinnvolle Suchraum-Anpassungen

Mit zusätzlichen 9 statischen Input-Dimensionen ist plausibel, dass sich
optimale Werte verschieben:

- `emb_dim`/`hidden` (MTGNN/WaveNet): die `emb_mlp`, die jetzt aus 6+9 statt
  6 Eingabedimensionen ein Embedding lernt, könnte von etwas mehr Breite
  profitieren — obere Grenze im Suchraum leicht anheben statt fixen Wert
  übernehmen.
- Dropout/Regularisierung: mehr Input-Kapazität ohne mehr Trainingsdaten ist
  ein klassisches Overfitting-Risiko — Suchraum für Dropout ggf. nach oben
  erweitern statt der bisherigen Range.
- `broadcast_topo` **nicht** als Suchraum-Dimension behandeln, sondern pro
  Studie fix auf `true` setzen (Arm C, siehe Entscheidung in Abschnitt 1) —
  eine gesuchte Flag würde die Studie uninterpretierbar machen und die
  Ablationslogik unterlaufen.

## 4. GPU-Landschaft (Stand: 30.07., nach TFT-Stop)

**14 GPUs über 3 Hosts** (nicht 12 — eine dritte Workstation `ws` wurde bei
dieser Analyse entdeckt, war aber wegen eines Tippfehlers in
`~/.ssh/config` nicht erreichbar, ist jetzt gefixt):

| Host | GPUs | Typ | VRAM | Status |
|---|---|---|---|---|
| w-lambdablade2 (lokal) | 4 | A100 | 80 GB | teils durch laufendes Topo-Screening belegt, wird nach dessen Abschluss frei |
| l1 (w-lambdablade1) | 8 | RTX A6000 | 49 GB | **7 GPUs frei** (0,1,4,5,6,7 + GPU2 fast leer), nachdem die 16 TFT-HPO-Worker (`hpo_cl_tft_bc.py`, base+hist × 8 GPUs) auf Anweisung gestoppt wurden; GPU 3 weiterhin von einem anderen Nutzer (~28 GB, 100 % util) belegt |
| ws (10.166.32.252) | 2 | RTX 4090 | 24 GB | **einsatzbereit** (Stand 30.07., siehe unten) |

**ws wurde eingerichtet und smoke-getestet:**

- NFS-Mounts (`/mnt/lambda1/nvme1`, `/mnt/lambda1/nvme2` von l1) eingerichtet —
  dafür musste `/etc/exports` auf l1 um ws's IP ergänzt (`exportfs -ra`) und
  eine `ufw`-Regel auf l1 für Port 2049/111 von ws's IP freigegeben werden;
  ohne diese beiden Freigaben hängt `mount -a` auf ws unendlich im
  Kernel-Wartezustand (`D`), weil das TCP-Connect nie ankommt (nicht sofort als
  Firewall-Problem erkennbar, sieht erst wie ein normales Timeout aus).
- Git auf `ws` war bei `b472d58` (18.12.2025), lokale Änderungen dort gestasht
  (`lokaler ws-Stand vor Topo-Branch-Sync`), Branch
  `fix/mtgnn-topo-static-dim` gepusht und auf `ws` ausgecheckt (jetzt `a748085`,
  synchron mit diesem Host).
- venv war unvollständig: `torch_geometric`, `pyarrow`, `psycopg2-binary`
  fehlten (Repo dort nie für den GNN-Zweig genutzt) — nachinstalliert, keine
  Kollision mit dem vorhandenen `torch==2.9.1+cu128` (die verwendeten
  PyG-Komponenten sind reines Python, keine kompilierten
  torch_scatter/sparse-Extensions im Code).
- Smoke-Test (`hpo_mtgnn.py --preprocess-only`, 3 Min. Laufzeit) lief
  fehlerfrei: korrekter Stations-Split (103/50), KNN-Imputation ok,
  ICON-D2-Parquets wurden über den NFS-Mount geladen.
- **Performance-Hinweis:** Parquet-Laden über NFS lief bei nur ~1–1.7
  Dateien/Sekunde (4284 Dateien) — spürbar langsamer als ein lokales
  NVMe-Read. Betrifft nur den **ersten** Lauf pro Config, danach greift
  `GNNCache` (`data_cache/gnns/`, lokal auf `ws`s eigener SSD, nicht geteilt
  mit l1/w-lambdablade2 — jeder Host baut seinen Cache separat einmalig auf).
  Bei der Verteilung also idealerweise pro Config nur einen Cold-Start auf
  `ws` einplanen, nicht mehrere Configs gleichzeitig neu dort anfangen.
- `OPTUNA_STORAGE` war auf `ws` bereits korrekt gesetzt (zeigt auf diesen
  Host, nicht `localhost`).
- Kapazität ist mit 2× 24 GB klein gegenüber den A100/A6000-Knoten — eher als
  Zusatz-Slots für 1-2 Worker interessant, nicht als Hauptknoten.

**Ergebnis:** **13 von 14 GPUs sofort nutzbar** (4× A100 + 7× A6000 + 2× RTX
4090), ein A6000 (l1 GPU 3) bleibt fremdbelegt.

**VRAM-Fußabdruck ist kein limitierender Faktor:** die aktuell laufenden
Screening-Prozesse brauchen nur ~1.3–2.5 GB pro Prozess (DCRNN-Kontrolllauf
mit größerem Cache ~13 GB), bereits jetzt laufen 2 Prozesse pro lokaler GPU
parallel (bestätigt: topoA+topoB auf GPU 2, topoC+Kontrolle auf GPU 1). Der
Flaschenhals bei mehr Parallelität pro GPU ist SM-Auslastung/Durchsatz, nicht
Speicher — bei 80 GB (A100) bzw. 49 GB (A6000) ist selbst mit 4–6 parallelen
Prozessen kein OOM zu erwarten.

## 5. Verteilungsstrategie (final, Stand 30.07.)

TFT-HPO ist auf beiden Hosts (L2 und l1) gestoppt, das Screening ist komplett
abgeschlossen — alle 13 GPUs sind frei. Auf expliziten Wunsch werden DCRNN und
MTGNN mit **3 Replikaten** pro Studie gefahren, WaveNet mit **2** (24 Worker
über 9 Studien), damit 11 der 13 GPUs doppelt belegt sind und nur 2 (beide auf
`ws`) einfach:

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

Prinzipien:

1. **Nie zwei Replikate derselben Studie auf derselben GPU** — sonst killt ein
   GPU-Ausfall gleich beide/alle Replikate einer Studie statt nur eines.
2. **WaveNet (nur 2 Replikate) auf den schnellsten/nächsten GPUs** (A100 + die
   am wenigsten belasteten A6000), da kleinerer Umfang.
3. Die **r3-Replikate** (nur DCRNN/MTGNN) verteilen sich auf A6000 + beide
   `ws`-Karten, damit `ws` echte dritte Replikate bekommt statt isoliert zu sein.
4. Die 2 einfach belegten GPUs sind bewusst beide auf `ws` — dort ist wegen
   des NFS-Cold-Starts (Abschnitt 4) ohnehin am wenigsten Vorhersagbarkeit,
   das begrenzt das Risiko bei einem `ws`-Ausfall auf 1 Lauf statt 2.

**Gemeinsamer Optuna-Storage (Postgres, `OPTUNA_STORAGE`)** erlaubt beliebig
viele parallele Worker pro Studie ohne Koordinationsaufwand — genau wie beim
Screening. Jede der 9 Studien hat einen festen Studiennamen (automatisch aus
dem Config-Stem generiert); alle Worker einer Studie laufen mit identischem
`--config` gegen dieselbe Studie.

**Fallstrick, der unbedingt zu vermeiden ist**
(siehe `feedback_hpo_worker_suffix.md`): **niemals** unterschiedliche
`-s`/`--suffix`-Werte für parallele Worker derselben Studie verwenden — das
fragmentiert eine Optuna-Studie in mehrere getrennte Studien. Nur der
Log-Dateiname bzw. Screen-Session-Name darf zwischen Workern derselben Studie
variieren, der Studienname (aus `--config` abgeleitet) muss für alle Worker
identisch sein.

## 6. Offene Punkte / nicht Teil dieses Plans

- **Keine MTGNN-Permutationskontrolle.** DCRNN und WaveNet haben je einen
  `--shuffle-node-features`-Lauf, MTGNN nicht (auf Wunsch nicht gestartet).
  Mit dem finalen Screening-Ergebnis (Abschnitt 1: C nur −1.17 % vs. A, B
  sogar +7.03 % schlechter) ist der MTGNN-Effekt ohnehin schon der
  unsicherste der drei — der fehlende Kapazitäts-Ausschluss bleibt daher
  ein offener Punkt, falls MTGNNs C-Vorteil im Re-HPO nicht reproduziert.
- Kein Commit zu einer konkreten Startzeit — abhängig vom Screening-Ausgang.
- Kein automatischer `launch_hpo_pipeline.py`-Launcher wurde geschrieben;
  Abschnitt 5 beschreibt nur das Verteilungsprinzip, Implementierung folgt
  erst nach Freigabe.
- Ob fold2/fold3 eigene Studien brauchen oder fold1-Best-Params übernommen
  werden, wird erst nach Sichtung der fold1-Ergebnisse entschieden.
