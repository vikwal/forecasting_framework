# Graphs Wind Speed Forecasting — Studie und Code

**Zweck dieses Dokuments.** Eine in sich geschlossene Beschreibung dessen, was diese
Studie erreichen soll und wie der Code das tut — Datenfluss, Zeitkonvention, Graph,
Modelle, HPO-Mechanik, Pipeline. Gedacht als Einstiegspunkt für jede neue Session, damit
das alles nicht jedes Mal neu erklärt werden muss.

Alle Angaben in den Abschnitten 1–8 sind am Code verifiziert (Stand `2aaebea`,
Branch `fix/mtgnn-topo-static-dim`, 2026-08-03) und **zeitlos gemeint** — sie ändern sich
nur, wenn sich der Code ändert. Abschnitt 9 ist der **flüchtige Teil**: Kampagnenstand,
Commit-Hashes und offene Befunde. Er altert bewusst; wer ihn liest, sollte ihn nachprüfen.

Dieses Dokument ersetzt `research_summary_dcrnn_mtgnn_wind_bc.md`, das nach `archiv/superseded_docs/` verschoben wurde. Die
Forschungsfrage und die Novelty-Argumentation stammen daraus; sechs dort belegte
Sachfehler sind hier korrigiert und in Abschnitt 10 aufgeführt.

Verwandte Dokumente:
- `story_positioning.md` — Positionierung gegen die Literatur, Contributions, Zieljournale
- `review_round2_findings.md` — Befunde des Code-Reviews, offene Punkte
- `review_round2_fixes.md` — die Fixes daraus, mit Verifikationszahlen
- `implementation_plan_ablations.md`, `ablations_verification_results.md` — Varianten B und C

---

## 1. Worum es geht

### 1.1 Das Problem

NWP-Modelle wie ICON-D2 (DWD) und ECMWF HRES liefern Windprognosen auf einem Gitter, das
nicht mit den interessierenden Standorten zusammenfällt. Zusätzlich haben sie
systematische Fehler gegenüber lokalen Messungen — unzureichend aufgelöste Orographie,
Landnutzung, Rauigkeit. Klassisch sind das zwei getrennte Schritte: räumlich
interpolieren (Regression-Kriging, MOS) und dann den Bias korrigieren.

### 1.2 Das Ziel

Ein einziges, end-to-end lernendes Modell, das NWP-Gitterpunktvorhersagen **direkt** in
kalibrierte, standortscharfe Windgeschwindigkeitsprognosen übersetzt — **ohne
historische Messungen an der Zielstation** als Input. Damit ist es auf neue,
unbeobachtete Standorte übertragbar, ohne Kriging und ohne Retraining pro Standort
(induktives Setting).

### 1.3 Die zentrale Forschungsfrage

**Kann man aus den Gitterpunkten der Wettermodelle brauchbare Information ziehen, oder
reicht es, einfach den nächsten Gitterpunkt zu nehmen?**

Weil Input und Output dieselben Zeitstempel tragen, ist die Aufgabe eher **Bias
Correction als Forecasting**.

### 1.4 Novelty (drei Ansprüche)

1. **DCRNN und MTGNN erstmals für Windenergie-Prognose / NWP-Bias-Correction.** Beide
   stammen aus dem Traffic Forecasting (Li et al. 2018; Wu et al. 2020) und sind dort
   **transduktiv** — feste Knotenmenge mit ID-gebundenen, lernbaren Embeddings. Hier für
   ein **induktives** Setting umgebaut.
2. **Heterogener Graph mit NWP-Gitterpunkten als eigenen Knotentypen.** Statt NWP-Werte
   nur als interpolierte Zusatzfeatures an Stationsknoten zu hängen, sind ICON-D2- und
   ECMWF-Punkte eigene Knoten mit gerichteten Kanten zur Station. Die Aggregation läuft
   nicht über feste Interpolationsgewichte, sondern **gelernt** über bipartite
   Graph-Attention (GATv2). Damit wird das räumliche Downscaling Teil des
   End-to-End-Lernproblems.
3. **IGNNK-artiges Masking kombiniert mit Bias-Correction.** Trainingsaufgabe ist nicht
   Zeitreihenprognose an bekannten Knoten, sondern **induktives räumliches Interpolieren**
   (Wu et al. 2021): pro Trainingsbeispiel wird eine zufällige Teilmenge von Stationen
   als Ziel maskiert, die übrigen dienen als reale Nachbarn mit echten Messungen.

> **Wichtige Einordnung — diese drei Ansprüche halten so nicht.** Ein Volltextdurchgang
> durch die Primärquellen am 2026-08-04 hat sie einzeln geprüft. Das Ergebnis steht in
> `story_positioning.md` §1.6; hier die Kurzfassung:
>
> - **(1) trägt nicht**, aber anders als früher behauptet. `zang2025dstg`, `jiang2023buaa`
>   und `li2025tfdgcn` wenden DCRNN bzw. MTGNN auf Wind an — aber alle drei sind
>   **Leistungsprognose aus SCADA innerhalb von ein bis zwei Windparks**, ohne NWP-Input
>   und ohne räumliche Generalisierung. Bei `zang2025dstg` sind die Knoten sogar
>   *Variablen*, keine Orte. Sie taugen nicht als Prioritätsbeleg, wohl aber als **Beleg
>   für die Lücke**: die gesamte GNN-für-Wind-Literatur ist SCADA-basiert, einzelparkbezogen
>   und transduktiv.
> - **(2) trägt nicht.** `yang2025offgrid` (JAMES 2025, peer-reviewed) baut genau diesen
>   Graphen — Stationsknoten per Delaunay plus die 8 nächsten NWP-Gitterzellen als zweiten
>   Knotentyp — für Wind, aus HRRR-Forecasts, mit Lead Times bis 48 h. **Aber: rein
>   zeitlicher Split über dieselben 358 Stationen, und jede Zielstation bekommt ihre eigene
>   Messhistorie als Input.** Die Architektur ist also besetzt, die Architektur **im
>   induktiven Setting** nicht.
> - **(3) trägt nur verengt.** Tot sind „first inductive post-processing"
>   (`baran2024clustering`), „first *learned* inductive post-processor"
>   (`cho2023downscaling`, `hou2026spatiotemporal`) und „first random node masking in
>   meteorology" (`li2023ssin`, `low2026spatialsupport`).
>
> **Was stattdessen beansprucht wird** — die drei Contributions in `story_positioning.md`
> §4, als Kette aufgebaut:
>
> 1. **Das Problem benennen, formalisieren und vermessen.** Fünf Arbeiten aus vier
>    Methodenfamilien erreichen das induktive Setting, ohne es je zu benennen; eine
>    OpenAlex-Suche danach liefert ab 2022 **null** Treffer. Wir geben ihm einen Namen, das
>    Evaluationsprotokoll und den ersten systematischen Benchmark für bodennahen Wind.
> 2. **Was ein Beobachtungsnetz für einen Standort wert ist, der nicht darin liegt.** Die
>    Zerlegung A/B/C ist an einer nie gemessenen Station **erschöpfend** — mit eigener
>    Historie wäre sie es nicht, weshalb `yang2025offgrid` diese Frage trotz gleichen
>    Graphen nicht stellen kann. Plus die Dosis-Wirkungs-Kurve über die Netzdichte.
> 3. **Der Beleg an 13 realen Windparks.** Ein Windpark hat keine Messhistorie und wird nie
>    eine haben — jedes Verfahren, das die eigene Historie des Zielorts liest, ist dort
>    **konstruktionsbedingt unanwendbar**, `yang2025offgrid` eingeschlossen. Die induktive
>    Fähigkeit ist damit die Anwendbarkeitsbedingung, nicht eine methodische Marotte.
>
> (2) sagt (3) quantitativ vorher — die Parks sitzen bei bekannter Nachbardichte. Trifft es
> zu, ist das eine Konsistenzprüfung über zwei unabhängige Datenwelten.
>
> **Abhängigkeiten:** (1) ist ohne die QRF- und lokalen-MOS-Baselines hohl — Befund **R6**
> ist damit Voraussetzung, nicht Kosmetik. (2) läuft als einzige schon vollständig; die
> Ausdünnungskurve braucht zusätzlich den nie gelaufenen Eval-Pfad. (3) hängt an einer
> Datenanfrage: die SCADA endet am 2024-06-01, also **innerhalb** des Trainingsfensters des
> Windmodells. Geplanter Ersatz, falls sie scheitert: die sub-stündliche Variabilität
> (`std_v_wind`) als neue Zielgröße.

---

## 2. Der Task, konkret

| | |
|---|---|
| Zielgröße | `wind_speed` auf **10 m über Grund** an DWD-Wetterstationen |
| NWP-Input | **ICON-D2** (~2,2 km, regional) und **ECMWF HRES** (global, gröber) |
| Prognosehorizont | **48 h**, durch ICON-D2 vorgegeben |
| Laufstunden | **6, 9, 12, 15 UTC** (`icond2_run_hours`) — sie definieren die Zeitfenster des Modells |
| Historienlänge | 48 h |
| Auflösung | stündlich, UTC |
| Stationspool | **153** (in der Config 103 `files` + 50 `val_files`; die räumliche CV überschreibt das mit 102/51 je Fold) |

**Nächste Gitterpunkte werden geodätisch bestimmt** (WGS-84), nicht euklidisch in Grad.
Wie viele davon berücksichtigt werden, optimiert die HPO (`next_n_icond2` 1…7,
`next_n_ecmwf` 0…4).

### Kanäle

| Block | Symbol | Inhalt | Breite |
|---|---|---|---|
| Messungen | M | `wind_speed`, `wind_direction` → zirkulär kodiert zu `sin`/`cos` | **3** |
| ICON-D2 | I2 | `u_10m`, `v_10m`, `wind_speed_10m`, `wind_speed_38m` | **4** |
| ECMWF | E2 | `u_wind10m`, `v_wind10m`, `wind_speed_10m` | **3** |

`encode_circular_measurements` (`train_dcrnn.py:168`) ersetzt `wind_direction` in Grad
durch `sin`/`cos`, damit der StandardScaler nicht über die 0/360-Sprungstelle stolpert.
Einmal nach der Imputation, vor dem Scaler-Fit.

### Drei Settings je Modell

| Setting | `nwp_nodes` | `hist_wind_available` | Bedeutung |
|---|---|---|---|
| **base** | `false` | `false` | NWP-Gitterprognosen werden an die Knoten-Features gehängt (k nächste Punkte konkateniert), keine GATv2 |
| **nwp** | `true` | `false` | NWP-Gitterpunkte sind eigene Knoten. Damit das Modell hier etwas lernen kann, **müssen die Kanten die relative Position verraten** |
| **nwp+hist** | `true` | `true` | wie nwp, zusätzlich die historische Windgeschwindigkeit am Zielstandort |

Bei `hist_wind_available: true` ist die IGNNK-Maskierung **inaktiv** — das ist gewollt.

---

## 3. Datenfluss

```
load_station_measurements        (T, N, M) Messungen, stündlich, UTC
  -> apply_interpol_imputation   Regression-Kriging fuellt NaN im Zielkanal
  -> apply_knn_imputation        KNN-Fallback fuer Rest und Sekundaerkanaele
  -> encode_circular_measurements  wind_direction -> sin/cos
load_icond2_ml_runs              (R, 48, N_grid, I2), Leads 1..48, geodaetische Dateiauswahl
load_ecmwf_parquet_...           (T, N_grid_e, E2) am Gitter + (T, N, E2) am naechsten Punkt
_load_elevations_from_table      Knotenhoehen, DB + SRTM-Fallback
  -> GNNCache (data_cache/gnns)  rohe, ungeskalierte Tensoren + derived-Dict
```

### Der GNNCache

Hash-basierter Schlüssel über die datenrelevanten Config-Felder. **Wichtig:**
`icond2_alts`, `ecmwf_alts` und die skalierten Statics liegen im `derived`-Teil — ein
Cache-HIT holt sie zurück, ohne sie neu aufzulösen. Ein Worker ohne `WEATHER_DB_URL`
würde NWP-Höhen = 0 hineinschreiben und alle anderen Worker erben das still. Seit
Befund K3 bricht so ein Worker hart ab, statt zu warnen; seit K4 ist das Schreiben
atomar (`os.replace` pro Datei) und per `flock` gesperrt.

`grid_icond2_runs` wird in den RAM geladen, der Rest per mmap.

### Zeitkonvention — die häufigste Fehlerquelle

`t_run_abs` ist der Index des **ersten Prognoseschritts**, also `t_run + 1 h`, **nicht**
der Laufzeit. Grund: ICON-D2 liefert Leads 1…48, gültig `t_run+1 … t_run+48`; Lead 0
wird verworfen.

| Block | Slice | gültige Zeiten |
|---|---|---|
| Messhistorie | `meas[t_run_abs-48 : t_run_abs]` | `t_run-47 … t_run` |
| Zielgröße | `meas[t_run_abs : t_run_abs+48]` | `t_run+1 … t_run+48` |
| ICON-D2 hist | `grid[r_hist]`, Leads 1..48 | `t_run-47 … t_run` |
| ICON-D2 curr | `grid[r_curr]`, Leads 1..48 | `t_run+1 … t_run+48` |
| ECMWF | `ecmwf[t_run_abs-48 : t_run_abs+48]` | `t_run-47 … t_run+48` |

**Wer aus `t_run_abs` die Laufzeit braucht — Logging, Fold-Zuordnung,
Persistenz-Referenz —, muss `t_run_abs - 1` nehmen.**

Empirisch belegt: RMSE(rohes ICON-D2 am nächsten Gitterpunkt vs. Messung) über die
tatsächlich gebauten Run-Paare hat sein Minimum bei Versatz 0 (1,4958 gegen 1,5040 bei
−1 und 1,5550 bei +1). ECMWF ebenso.

---

## 4. Der Graph

### Knoten- und Kantentypen

- **Knoten:** `station`, `icond2`, `ecmwf`
- **Kanten:** `("station","near","station")` bidirektional; `("icond2","informs","station")`
  und `("ecmwf","informs","station")` gerichtet, je Station die k nächsten Gitterpunkte

Der Graph wird **einmal statisch** aus den Koordinaten gebaut (`HeterogeneousGraphBuilder`),
die Zeitreihen füllt der Sampler pro Sample ein.

### Kantenattribute

`[dist_norm, sin(bearing), cos(bearing), alt_diff]` plus optional Topo-Spalten.
In der Kampagnen-Config ergibt `edge_features` **12 Spalten**: 1 Distanz + 2 Azimut
+ 1 Höhendifferenz + 8 Topo.

Dass die NWP→Station-Kanten die Geometrie tragen, ist **zwingend** — sonst wäre die
Attention über die k Gitterpunkte permutationsäquivariant und der nächste Punkt vom
k-ten nicht unterscheidbar. Genau das war Befund A2 aus Review-Runde 1. Belegt durch
einen Permutationstest: k Punkte **mit** Kanten vertauscht → max|Δ| ≈ 1,9e-07;
**ohne** Kanten → 9,5e-02 (MTGNN) bzw. 5,4e-02 (WaveNet).

> **Achtung:** `DCGRUCell.edge_weight_from_attr` reduziert die
> station↔station-Kanten auf **Spalte 0** (Distanz → Gauß-Kernel). DCRNN liest die
> Topo-Spalten der s2s-Kanten also **nicht**; Topo kommt bei allen drei Modellen über
> **Knoten**-Features herein. Die Topo-Spalten im s2s-`edge_attr` sind bei DCRNN toter
> Ballast.

### Knoten-Features

`station.static` hat **13** Spalten: 4 Geo/Typ (lat, lon, alt, Typindikator) + **9 Topo**
(`slope`, `aspect_sin`, `aspect_cos`, `tpi5`, `tpi75`, `tdi`, `elev_std`, `z0`,
`dist_coast`). Gesteuert über `station_node_features: all`.

Der Topo-z-Score wird **nur auf den Trainingsstationen des Folds** gefittet
(`n_train=N_train` bzw. `train_idx=train_idx`). Ein Fit über alle 153 Stationen
verschiebt die Spalten um max|Δ| = 0,4738 — die Unterscheidung ist also tragend.

### IGNNK-Masking

Pro Trainingsbeispiel werden 1–10 zufällige Trainingsstationen als Ziel gewählt und ihre
Messhistorie genullt; die übrigen dienen als Nachbarn mit echten Messungen. In der
Validierung sind **alle** Val-Stationen gleichzeitig Ziel.

---

## 5. Die Modelle

### DCRNN (`geostatistics/dcrnn/`)

Seq2Seq mit **DCGRU-Zellen** (Diffusionskonvolution über den Stationsgraphen, `K_hop`
Diffusionsschritte, bidirektional) statt Standard-GRU-Gates. Decoder **autoregressiv**
mit linear zerfallendem Teacher-Forcing und NWP-Injection an jedem Decoderschritt.

**NWP-Aggregation** über `NWPAttentionLayer` (GATv2): Query = **Hidden State der
Station**, Kantenattribute = `[dist_norm, sin(bearing), cos(bearing), alt_diff]`. Die
Aufmerksamkeit hängt also vom Systemzustand der Station ab und wird pro Zeitschritt neu
berechnet.

Bei `nwp_nodes=false` läuft keine GATv2; stattdessen tragen `k·I2 + k_e·E2` Kanäle
direkt in `station.x`.

Arbeitet auf einem **heterogenen PyG-Graphen** pro Sample.

### MTGNN (`geostatistics/mtgnn/model.py`) und GraphWaveNet (`geostatistics/wavenet/model.py`)

Homogene, gebatchte Tensor-Samples. **Gelernte Adjazenz aus statischen Knotenfeatures**
statt ID-gebundener Embeddings — das ist der induktive Umbau.

NWP-Aggregation über `shared/nwp_gat.py::HomoNWPAttentionLayer` — **Zero-Query**, die
Attention entsteht also allein aus NWP-Features plus Kantenattributen. ICON-D2 und ECMWF
haben je eine **eigene** Attention-Schicht.

> **Korrektur gegenüber der alten Doku:** Die NWP-Einbindung bei MTGNN/WaveNet ist
> **nicht** „analog zu DCRNN". DCRNN nutzt eine Hidden-State-Query, MTGNN und WaveNet
> eine Zero-Query. Das ist ein qualitativer Unterschied und gehört ins Paper.

MTGNN zusätzlich: Graph-Learning-Modul `A = ReLU(tanh(α(M1·M2ᵀ − M2·M1ᵀ)))`, Dilated
Inception, Mixhop-Propagation mit Restart-Wahrscheinlichkeit β, Curriculum Learning.

### Sampler

- `stgnn/training/sampler.py::TrainingSampler` — DCRNN, heterogener PyG-Graph.
  `sample_train`: 1–10 zufällige Trainingsstationen als Ziel, Historie genullt.
  `sample_val`: alle Val-Stationen gleichzeitig Ziel. Nachbarn jeweils die
  `next_n_neighbors` räumlich nächsten.
- `homo_sampler.py::HomoSampler` — MTGNN und WaveNet, homogene Tensor-Batches, gleiche
  Semantik. `_val_layout()` wendet `next_n_neighbors` auch in der Validierung an.

---

## 6. Räumliche Kreuzvalidierung

`configs/spatial_folds.yaml`, **3 Folds**, je **102 Trainings- und 51 Zielstationen**.
Val-Mengen paarweise disjunkt, Vereinigung = 153 = Pool.

> **Korrektur gegenüber der alten Doku:** Es sind **räumliche**, nicht zeitliche Folds.
> Das Zeitfenster ist in allen drei Folds **identisch**: Train bis 2024-08-01
> (1473 Run-Paare), Val 2024-08-01 → 2025-08-01 (1460 Paare). Die Validierung ist damit
> räumlich **und** zeitlich out-of-sample — aber die drei Fold-Ergebnisse sind zeitlich
> vollständig korreliert. **Für eine Wilcoxon-Auswertung über Folds ist das relevant.**

Scaler und Topo-z-Score werden pro Fold auf `[:val_start, train_idx]` gefittet.

**Objective** = Mittel des unskalierten Val-RMSE über die 3 Folds, in m/s.

Zeitlicher Testbereich: `test_start` 2025-08-01 bis `test_end` 2025-10-31.

---

## 7. Wie die HPO funktioniert

### Studien

Acht Studien: DCRNN × {base, nwp, nwp_hist}, MTGNN × {base, nwp, nwp_hist},
WaveNet × {base, nwp}. `wavenet_nwp_hist` wurde bewusst gestrichen.

Storage: **PostgreSQL**, URL in `OPTUNA_STORAGE` (in `~/.bashrc`).

### Der Studienname wird aus dem Config-Dateinamen abgeleitet

```python
hpo_stem   = re.sub(r'_fold\d+$', '', config_stem)
study_name = f"cl_m-dcrnn_out-{H_fore}_freq-{freq}_{hpo_stem}"
```

Das ist die wichtigste Mechanik, die man kennen muss:

- **`--hpo-study` steuert bei Postgres-Storage nur, _ob_ überhaupt geladen wird, nicht
  welche Studie.** Der Name kommt immer aus dem Dateinamen.
- Weil das `_foldN`-Suffix abgeschnitten wird, zeigen `config_X.yaml` und
  `config_X_fold1..3.yaml` auf **dieselbe** Studie. Genau das lässt den Retrain die
  Hyperparameter seiner eigenen Variante finden.
- Wer eine Variante mit eigenen Hyperparametern will, braucht also nur eine eigene
  Config-Datei — kein CLI-Argument.

### Trial-Budget: pro Worker, nicht pro Studie

```python
completed = len([t for t in study.trials if t.state == COMPLETE])
remaining = max(n_trials - completed, 0)      # einmalig beim Start
study.optimize(objective, n_trials=remaining, catch=(Exception,))
```

`remaining` wird **einmal beim Start** berechnet. **N Worker auf derselben Studie holen
sich also je `n_trials` Trials** — die Studie läuft dann auf bis zu N × Budget. Bei
gleich vielen Workern je Studie ist das unschädlich; bei ungleicher Verteilung bekommt
eine Variante mehr Suchbudget als die andere, und ein Vergleich zwischen ihnen wird
verfälscht.

### Pruner und Budget

| | |
|---|---|
| Trials je Studie | 150 (Ablationen: 60) |
| `max_epochs_per_trial` | 100 |
| `patience_per_trial` | 10 |
| Pruner | `median`, `pruner_n_startup_trials: 20`, `pruner_n_warmup_steps: 1` |

Die ersten 20 Trials je Studie laufen also ungeprunt durch — das ist die langsame Phase.

### Suchraum (DCRNN, 17 Parameter)

`K_hop`, `dropout`, `ecmwf_feature_mode`, `grad_accum`, `gradient_clip`, `hidden`,
`horizon_decay`, `icond2_feature_mode`, `lr`, `next_n_ecmwf`, `next_n_icond2`,
`next_n_neighbors`, `num_layers`, `nwp_heads`, `nwp_out_per_head`,
`teacher_forcing_ratio`, `weight_decay`.

`sample_hyperparameters` sampelt **generisch alles**, was unter `hpo.params` steht —
ein neuer Eintrag wirkt also sofort, sofern der Trainingscode den Schlüssel liest.

> **Falle:** Steht ein Parameter in `hpo.params`, wird er aber im Modellbau nicht
> gelesen, ist er wirkungslos und verbrennt trotzdem Suchbudget. Umgekehrt: wird er
> gelesen, steht aber nicht im Suchraum, läuft er still auf dem statischen Wert. Beides
> ist in Runde 1 vorgekommen (Befunde A1 und B1). Bei jeder Suchraumänderung beide
> Richtungen prüfen.

### Fallstrick beim Starten von Läufen

`ssh host '…'` ist eine **nicht-interaktive Shell**; `.bashrc` bricht vor den
`export`-Zeilen ab. Ein so gestarteter Worker hätte kein `WEATHER_DB_URL`.
`/tmp/launch_hpo.sh` liest die Zeilen deshalb explizit aus `~/.bashrc` und bricht ab,
wenn sie fehlen:

```bash
/tmp/launch_hpo.sh <REPO_ROOT> <SUFFIX> "name:script:config:gpu" ...
```

Backslash-Zeilenfortsetzungen funktionieren innerhalb von `ssh host '…'` **nicht** —
Skripte per Heredoc durchpipen (`ssh host bash -s <<'EOF'`).

---

## 8. Die Pipeline: HPO → Retrain → Evaluation

```
1. HPO          hpo_dcrnn.py --config configs/dcrnn/config_wind_dcrnn.yaml --gpu N --suffix rX
                -> Studie cl_m-dcrnn_out-48_freq-1h_wind_dcrnn, Objective = Fold-Mittel des Val-RMSE

2. Retrain      launch_train_pipeline.py, Gruppen DCRNN_BASE / DCRNN_NWP / DCRNN_NWP_HIST ...
                -> train_dcrnn.py je Fold, liest die besten HPO-Parameter aus der Studie,
                   die sich aus dem Config-Dateinamen ergibt

3. Evaluation   launch_eval_pipeline.py -> get_test_results_dcrnn.py
                -> RMSE/MAE/R2 je Station, plus Skill-Scores
```

**Skill-Metriken:**
- `Skill` = 1 − RMSE(Modell)/RMSE(zeitliche Persistenz)
- `skill_icond2` = 1 − RMSE(Modell)/RMSE(rohes ICON-D2 am nächsten Gitterpunkt) —
  quantifiziert direkt den Bias-Correction-Mehrwert
- `skill_ecmwf` analog

Die Persistenz-Referenz ist `meas_raw[t_run_abs - 1]`, also die letzte Beobachtung
**vor** dem Prognosefenster.

### Ablationsvarianten B und C (nur DCRNN)

| Variante | eigene k Gitterpunkte | Nachbar-**Messungen** | Nachbar-**Geometrie / NWP-Kontext** |
|---|---|---|---|
| A = `config_wind_dcrnn.yaml` | ja | ja | ja |
| **B** = `..._nomeas.yaml` | ja | **nein** | ja |
| **C** = `..._nograph.yaml` | ja | nein | **nein** |

**A − B = Wert der Nachbar-Messungen. B − C = Wert des Geometrie- und Kontextkanals.
C = der reine standortweise Downscaling-Boden.**

Nur DCRNN, weil MTGNN bei Variante C aus den statischen Knotenfeatures eine Adjazenz
nachbauen würde — das wäre keine Ein-Variablen-Änderung.

Umgesetzt über zwei Schalter:
- `neighbour_meas_available: false` — nullt die Messhistorie **aller** Knoten. Der Zweig
  muss **vor** `hist_wind_available` greifen, sonst nullt B nur die Zielstationen.
- `station_connectivity: "none"` — leeres `edge_index` `(2,0)` und `edge_attr` `(0,12)`.

`geostatistics/ablations/guard.py::check_ablation_flags` loggt beim Start eine Bannerzeile
mit allen Variantenflags und bricht hart ab, wenn `neighbour_meas_available: false` mit
`interpolate_history: true` zusammenkommt — der Kriging-Kanal würde B sonst über einen
Pfad an Nachbar-Messinformation kommen lassen, der den Graphen komplett umgeht. Der Guard
sitzt in `train_dcrnn.py`, `get_test_results_dcrnn.py` und `hpo_dcrnn.py`, jeweils vor
jedem Datenladen.

Die Varianten-Configs werden **generiert**, nicht handgeschrieben:
```bash
python -m geostatistics.ablations.gen_variant_configs --variant nomeas  --trials 60
python -m geostatistics.ablations.gen_variant_configs --variant nograph --trials 60 --pin-inert
```
`--pin-inert` entfernt `K_hop` und `next_n_neighbors` aus C's Suchraum — ohne
Stationskanten können sie nachweislich nichts bewirken (Permutationstest: max|Δpred| =
exakt 0,0).

### Hosts

| Host | Alias | Repo | GPUs | Besonderheit |
|---|---|---|---|---|
| `w-lambdablade2` | `l2` | `/home/viktor/Work/forecasting_framework` | 4× A100 80 GB | kanonische Kopie, hier committen; Postgres läuft hier |
| `w-lambdablade1` | `l1` | `/home/viktorwalter/Work/forecasting_framework` | 8× RTX A6000 48 GB | **lokale Pfad-Rewrites in `configs/`** (`/mnt/lambda1/nvme1/` → `/mnt/nvme1/`), nach jedem Pull neu anwenden |
| `w-lambda-vector` | `ws` | `/home/viktor/Work/forecasting_framework` | 2× RTX 4090 24 GB | **nur über l2 erreichbar** (`ssh l2 'ssh ws …'`) |

Rollout-Rezept und der Kontrollcheck, dass l1s Diff nur Pfade enthält, stehen in
`review_round2_fixes.md` Abschnitt 8.

### Verifikationswerkzeuge

Liegen unter `archiv/ablations_verification/` (nicht im Laufzeitpfad):

```bash
CUDA_VISIBLE_DEVICES="" python -m archiv.ablations_verification.verify           # Ablations-Suite
CUDA_VISIBLE_DEVICES="" python -m archiv.ablations_verification.verify_review2   # Runde-2-Belege
CUDA_VISIBLE_DEVICES="" python -m archiv.ablations_verification.batch_fingerprint \
    --out /tmp/fp.json --compare archiv/ablations_verification/fp_9808123.json
```

Der Fingerprint vergleicht 28 Sampler-Tensoren bitweise gegen den Stand `674a043` — so
wird belegt, dass Variante A durch spätere Eingriffe unangetastet bleibt. Alles läuft auf
der CPU, ohne GPU und ohne Optuna.

---

## 9. Stand am 2026-08-03 — **dieser Abschnitt altert**

> Alles hier ist eine Momentaufnahme. Vor dem Weiterarbeiten nachprüfen.

**Code:** Branch `fix/mtgnn-topo-static-dim`, Commit `2aaebea` auf l2, l1 und ws.

**Kampagne:** am 2026-08-03 gegen 16:30 von Null neu gestartet (alle vorherigen Studien
waren wegen der Befunde aus Review-Runde 1 ungültig; Snapshot in
`archiv/hpo_r1_invalid/`). 8 Studien × 150 Trials. Die beiden WaveNet-Studien wurden um
21:30 wegen Befund R5 erneut von Null gestartet (Snapshot in `archiv/hpo_wavenet_asym/`).
Die Ablations-HPOs für B und C laufen seit 22:22 mit je 60 Trials, je ein Worker auf l1
GPU 3 und 6.

Größenordnung: mittlere Trial-Dauer rund 4 h in der ungeprunten Anfangsphase. Bis alle
Budgets voll sind, sind eher Wochen als Tage zu erwarten.

**Offene Befunde** (Details in `review_round2_findings.md`):

| # | Was | Fällig |
|---|---|---|
| **N1** | Der Geo-Statik-Scaler wird im Retrain auf allen 153 Stationen gefittet, in `get_test_results_dcrnn.py` aber auf `raw_static[:N_train]`. Ein Fold-Modell sieht bei der Evaluation andere Mittelwerte für lat/lon/alt als beim Training. **Potenziell ergebnisrelevant.** | vor der Auswertung |
| **R3** | Die zugesagten Szenarien `excl_val` / `incl_val` existieren im Code nicht — nur in den Notebooks als Default-Spalte | vor der Auswertung |
| **R4** | WaveNet hat weder Retrain- noch Eval-Gruppe im Launcher; zwei Studien haben keinen Verbraucher | vor dem Retrain |
| **R6** | **Teil B (Baselines):** QRF und MOS fehlen — seit dem Contribution-Umbau vom 2026-08-04 **Voraussetzung** von Contribution (i), nicht Kosmetik. Prompt liegt vor: `docs/prompt_baselines_implementation.md`. **Teil A (Falsifikationen zur alten Contribution (iv)):** (a) und (c) bewusst gestrichen, (b) `next_n_ecmwf: 0` als billige Zugabe im selben Prompt | vor der Auswertung |
| N2–N4 | latent: `--broadcast-topo` fehlt im MTGNN-Eval; `ECMWF_WIND_SL_URL` ungeprüft; `train_stgnn2.py` warnt statt abzubrechen | — |
| §4.7 | Der Eval-Pfad ist **nie end-to-end gelaufen**; Backward-Pass, Trainer-Loop und Checkpoint-Roundtrip von Variante B sind ungeprüft | zusammen mit einem Kurzlauf von B |

**Nicht erneut zu prüfen** (Runde 1 und 2, jeweils belegt): die neun Befunde aus Runde 1
(`review_prompt_round2.md` §6) und K1–K4, R1, R2, R5 aus Runde 2
(`review_round2_fixes.md`).

---

## 10. Was gegenüber `research_summary_dcrnn_mtgnn_wind_bc.md` korrigiert wurde

Die alte Fassung liegt unter `archiv/superseded_docs/research_summary_dcrnn_mtgnn_wind_bc.md`. Diese sechs Punkte waren dort falsch:

1. **§4.3** — „NWP-Einbindung bei MTGNN analog zu DCRNN". Falsch: DCRNN nutzt eine
   Hidden-State-Query, MTGNN und WaveNet eine Zero-Query.
2. **§6** — „drei zeitliche Folds mit `min_train_date: 2024-03-31`". Falsch: räumliche CV
   mit in allen Folds identischem Zeitfenster.
3. **§6** — „7 abgeschlossene HPO-Studien". Es sind 8, und sie liefen zum Zeitpunkt der
   Aussage noch.
4. **§3.3** — „759 ECMWF-Gitterpunkte". Geladen werden **553** für 153 Stationen.
5. **§3.2** — Laufstunden „00/06/09/12/15". Die Configs fahren **[6, 9, 12, 15]**.
6. **§4.1** — „typ. 4" nächste Gitterpunkte. Die HPO sucht **1…7** (ICON-D2) und
   **0…4** (ECMWF).

Zusätzlich enger gefasst: Die alte Fassung beschrieb auch `power` als Zielgröße,
synthetische Leistungswerte für sechs Turbinentypen und Nabenhöhen-Extrapolation. Die
**laufende Kampagne fährt ausschließlich `target_col: wind_speed` auf 10 m** an
DWD-Stationen. Der Leistungspfad existiert im Framework, ist aber nicht Teil dieser
Studie.
