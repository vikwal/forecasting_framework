# Review Runde 2 — Befunde

*Durchgeführt am 2026-08-03 auf Stand `9a0f3b6` (Branch `fix/mtgnn-topo-static-dim`),
geprüft auf `l2` und `l1`. Auftrag: `docs/review_prompt_round2.md`. Der Reviewer hat
keinen Produktivcode verändert, keine Studie beschrieben und keine GPU benutzt.*

**Status der Befunde:** K1–K4, R1 und R2 sind behoben (Commits `7eeb42f` / `463cb23`,
Belege in `docs/review_round2_fixes.md`). R5 ist behoben (Commit `f5234d7`, Abschnitt 4
in derselben Datei). **R3, R4 und R6 sind offen** und in Abschnitt 2 dieses Dokuments
beschrieben. Der Nebenbefund N1 in Abschnitt 3 ist **offen und potenziell ergebnisrelevant**.

---

## 1. Behobene Befunde — Kurzfassung

Die vollständige Begründung und die Verifikationszahlen stehen in
`docs/review_round2_fixes.md`. Hier nur, was das Review gefunden hat.

### K1 (kritisch, behoben) — DCRNN-Evaluation stürzte immer ab

`get_test_results_dcrnn.py` lud keine topographischen Knotenfeatures, obwohl die Configs
`station_node_features: all` setzen und das Modell damit mit
`station_static_features = 4 + 9 = 13` gebaut wird. Empirischer Beleg:

```
model_cfg.station_static_features = 13
station.static aus einem 3-Spalten-Array: (60, 4)
forward RAISED: RuntimeError mat1 and mat2 shapes cannot be multiplied (60x135 and 144x64)
```

144 − 135 = 9 — exakt die neun Topo-Spalten. Das Skript war das einzige ohne Topo-Pfad;
`train_dcrnn.py`, `train_mtgnn.py`, `train_wavenet.py`, `get_test_results_mtgnn.py` und
`get_test_results_wavenet.py` hatten ihn alle. Betroffen waren alle sechs DCRNN-Eval-Jobs
der Hauptkampagne **und** alle sechs Ablations-Eval-Jobs. Der Fehler war laut, nicht
still — es sind keine falschen Zahlen entstanden.

### K2 (kritisch, behoben) — falsche Topo-Namensquelle im MTGNN-/WaveNet-Eval

Die Eval-Skripte nahmen die Topo-Namen aus `parse_edge_features`, die Trainingsskripte
aus `parse_station_node_features`. Belegte Diskrepanz:

```
config_wind_mtgnn.yaml    train: 9 -> static_dim 15   |   eval: 8 -> static_dim 14   (elev_std fehlte)
config_wind_wavenet.yaml  train: 9 -> static_dim 15   |   eval: 0 -> static_dim 6    (Topo fehlte ganz)
```

### K3 (kritisch, behoben) — Worker ohne `WEATHER_DB_URL` schrieb NWP-Höhen 0 in den Cache

Am 2026-08-03 hat ein HPO-Worker den geteilten GNNCache **zweimal** mit NWP-Höhen = 0
überschrieben und damit den B4-Fix aus Runde 1 für rund 15 Minuten ausgehebelt:

```
logs/hpo_dcrnn_wind_dcrnn_base_r1.log:62   16:10:47 [WARNING] DB URLs not set — NWP altitudes = 0
logs/hpo_dcrnn_wind_dcrnn_base_r1.log:69   16:10:50 [INFO] GNNCache — written to data_cache/gnns/d67d98241545ae6d
logs/hpo_dcrnn_wind_dcrnn_base_r1.log:123  16:19:39 [WARNING] DB URLs not set — NWP altitudes = 0
logs/hpo_dcrnn_wind_dcrnn_base_r1.log:130  16:19:42 [INFO] GNNCache — written to data_cache/gnns/d67d98241545ae6d
```

Der Cache legt `icond2_alts`, `ecmwf_alts` und die skalierten Statics im `derived`-Teil
ab; ein Cache-HIT holt die Nullen also zurück, ohne sie neu aufzulösen. **Kein Schaden
an der Kampagne:** der Probe-Lauf hat den Cache um 16:25:44 sauber überschrieben, und
der Ist-Zustand war auf beiden Hosts korrekt (`icond2_alts` 7…2650 m, `ecmwf_alts`
7…1733 m, jeweils `zeros=0`).

### K4 (kritisch, behoben) — `GNNCache.save()` war weder atomar noch gesperrt

`np.save` schrieb direkt an den Zielpfad, während andere Worker dieselbe 2,7-GB-Datei
per mmap lasen. Der Reproduktionstest ist der eindrücklichste Beleg dieser Runde:
**vorher wurden 6 von 6 Lesern per SIGBUS(-7) getötet** — ein mmap auf eine unter dem
Prozess gekürzte Datei, aus Python nicht abfangbar. Nachher: 3014 saubere Lesevorgänge,
0 inkonsistent, 0 getötet.

### R1 (behoben) — `train_dcrnn.py --eval` reichte die k-nächsten-Indexarrays nicht durch

`station_k_nearest_grid` / `station_k_nearest_ecmwf` fehlten im `run_evaluation`-Aufruf.
Im base-Arm hätte `station.x` dann nur den einen nächsten Gitterpunkt getragen statt
k·I2 + k_e·E2 Kanälen. Aktuell folgenlos, weil `launch_train_pipeline.py::build_cmd`
`--eval` nie setzt — die B1-Korrektur aus Runde 1 war an dieser Stelle aber wirkungslos.

### R2 (behoben) — `get_test_results_dcrnn.py` kannte `interpolate_history` nicht

Mit `interpolate_history: true` wäre das Modell mit `station_meas_features = M+1` gebaut
worden, der Eval-Batch hätte nur M Kanäle geliefert. Folgenlos, weil der Schlüssel
überall `false` ist und der Ablations-Guard die gefährliche Kombination abfängt.

### R5 (behoben) — Suchraum-Asymmetrie MTGNN vs. WaveNet

Siehe `docs/review_round2_fixes.md` Abschnitt 4.

---

## 2. Offene Befunde

### R3 — Die Evaluationsszenarien `excl_val` / `incl_val` existieren nicht

`implementation_plan_ablations.md` §5 und `ablations_verification_results.md` §5 sagen
zwei Auswertungsszenarien zu. `grep -rn "incl_val\|excl_val" geostatistics/ configs/`
findet sie **ausschließlich in den Notebooks** (`fold_evaluation.ipynb`,
`tft_vs_graph_evaluation.ipynb`), wo `scenario` mit dem Default `"excl_val"` ergänzt
wird, falls die Spalte fehlt. `get_test_results_dcrnn.py` kennt nur `--test-mode`
(train = files+val_files, val = test_files). Ein echtes Leave-One-Out über alle
Stationen **inklusive** der Validierungsstationen als reale Nachbarn — so beschreibt es
`research_summary` §6 — ist nirgends implementiert.

**Konsequenz:** Die Zusage ist ohne Erweiterung des Eval-Pfads nicht einlösbar.
Fällig vor der Auswertung.

### R4 — WaveNet hat keinen Retrain- und keinen Eval-Pfad

`launch_train_pipeline.py` enthält sechs Gruppen (`DCRNN_BASE/NWP/NWP_HIST`,
`MTGNN_BASE/NWP/NWP_HIST`) plus die zwei Ablationsgruppen — **keine WaveNet-Gruppe**.
`launch_eval_pipeline.py` enthält ebenfalls keinen WaveNet-Job, obwohl
`fold_evaluation.ipynb::MODEL_META` drei WaveNet-Einträge erwartet.

**Konsequenz:** Zwei der acht HPO-Studien (`wind_wavenet`, `wind_wavenet_nwp`) haben
keinen nachgelagerten Verbraucher. Die HPO läuft normal und liefert gültige
Hyperparameter — aber ohne die Launcher-Gruppen entstehen daraus keine Fold-Retrains
und keine Testzahlen, die WaveNet-Spalte der Ergebnistabelle bliebe leer. Vorbestehend,
nicht aus den Ablations-Commits. Fällig vor dem Retrain.

### R6 — Falsifikationen und Baselines aus `story_positioning` fehlen im Code

**Contribution (iv)** verspricht drei Falsifikationen:

- **(a) Attention durch feste Inverse-Distanz-Gewichte ersetzt.** Existiert als
  `aggregate_nwp: true` **nur im `HomoSampler`** (MTGNN/WaveNet) und in keiner Config.
  Im **DCRNN-Pfad gibt es sie gar nicht** —
  `grep -rn "aggregate_nwp\|idw\|IDW" geostatistics/dcrnn/ geostatistics/train_dcrnn.py geostatistics/hpo_dcrnn.py`
  ist leer. `nwp_injection: false` schaltet die Attention komplett ab, ist also keine
  IDW-Kontrolle, und steht in keiner DCRNN-Config.
- **(b) nur ICON-D2-Knoten.** Über `next_n_ecmwf: 0` erreichbar, wird aber von der HPO
  frei gewählt (Suchraum 0…4) statt als eigener Arm gefahren.
- **(c) nur ECMWF-Knoten.** Gar nicht konfigurierbar; es gibt keinen Pfad mit
  `next_n_icond2: 0`.

**Contribution (i)** nennt **Quantile Regression Forests** und **lokales MOS** als
Baselines. `grep -rln "QuantileRegress\|RandomForest\|quantile_forest\|\bMOS\b" geostatistics/`
ist leer. Vorhanden sind Regression-Kriging (`wind_interpol`) und TFT.

**Konsequenz:** Keine Bugfrage, sondern eine Scope-Entscheidung — entweder
implementieren oder die Claims im Paper zurücknehmen. Fällig vor dem Schreiben der
Contribution-Absätze.

---

## 3. Nebenbefunde aus der Fix-Runde — gemeldet, nicht repariert

### N1 (offen, potenziell ergebnisrelevant) — Geo-Statik-Scaler wird in Retrain und Eval auf verschiedenen Populationen gefittet

`hpo_dcrnn.py:660` und `train_dcrnn.py:865` fitten `lat`/`lon`/`alt` in der räumlichen CV
auf **allen 153 Stationen**. `get_test_results_dcrnn.py:415` fittet **immer** auf
`raw_static[:N_train]`, im Dev-Modus also nur auf den **102 Trainingsstationen** des
Folds. Ein Fold-Modell sieht bei der Evaluation damit andere Mittelwerte und Streuungen
für die drei Geo-Spalten als beim Training. Im `--test-mode` fällt beides zusammen.

Betrifft **nur** die drei Geo-Spalten, nicht die neun Topo-Spalten — der Topo-z-Score
wird an beiden Stellen korrekt auf den Trainingsstationen des Folds gefittet
(`n_train=N_train` in `train_dcrnn.py:879`, `train_idx=train_idx` in `hpo_dcrnn.py:750`).
Dass diese Unterscheidung tragend ist, ist gemessen: ein Fit über alle Stationen
verschiebt die Topo-Spalten um **max|Δ| = 0.4738**.

**Konsequenz:** Kann die Ablations- und Kampagnenzahlen verfälschen, weil die
Evaluation ein anderes Eingangsskalierungsschema verwendet als das Training.
Entscheidung nötig, welche der beiden Populationen die richtige ist.

### N2 (offen, latent) — `--broadcast-topo` fehlt in `get_test_results_mtgnn.py`

`train_mtgnn.py:247` und `hpo_mtgnn.py:320` haben das Flag, das MTGNN-Eval nicht (das
WaveNet-Eval schon). Latent, weil in der Kampagne nirgends gesetzt.

### N3 (offen, latent) — `ECMWF_WIND_SL_URL` bleibt ungeprüft

Der Zweig `elif weather_db_url:` in `hpo_dcrnn.py` greift auch, wenn nur die ECMWF-URL
fehlt; `ecmwf_alts` bleibt dann, was der Parquet-Loader lieferte. Der K3-Guard
`require_nwp_elevation_env` kann das über `need_ecmwf=True` abdecken — nicht gesetzt,
weil `max_next_n_ecmwf` an der Stelle noch nicht feststeht.

### N4 (offen, latent) — `train_stgnn2.py` hat die K3-Schwäche weiterhin

`train_stgnn2.py:1316` baut zwar eine `missing`-Liste, ruft dann aber `logger.warning`
und rechnet weiter — es ist kein harter Abbruch, entgegen der ursprünglichen Annahme.
Nicht angefasst, weil `train_stgnn2` nicht in den GNNCache schreibt.

### N5 (zurückgestellt) — lade-seitiger Teil der K4-Sperre

Umgesetzt ist die Schreibsperre. „Von N gleichzeitig MISSenden Workern lädt nur einer"
ist es nicht: dafür müsste die Sperre schon an der `exists()`-Prüfung genommen und
minutenlang quer durch die `main()`-Funktionen dreier Skripte gehalten werden. Ein
hängender Worker würde alle anderen unsichtbar blockieren, und der Nutzen ist gering —
Mehrfachladen ist verschwenderisch, aber korrekt; die Konsistenz sichert schon die
Schreibseite. Empfehlung: so lassen.

---

## 4. Was das Review geprüft und für korrekt befunden hat

Das ist ebenfalls ein Ergebnis — es sagt, worauf man sich verlassen kann.

### §4.1 Variante A unangetastet — unabhängig reproduziert

Der Reviewer hat `674a043` per `git archive` in ein frisches Verzeichnis ausgepackt,
dort **nur** die drei ablations-agnostischen Hilfsdateien aus HEAD eingespielt
(verifiziert: `grep -c neighbour_meas_available` = 0 in sampler/evaluation/config,
`grep -c '"none"'` = 0 in graph_builder) und den Fingerabdruck neu erzeugt:

> `IDENTICAL — 28 tensor fingerprints match bit for bit across 14 + 14 tensors.`

Die eigene Referenz war **byteidentisch** mit der des Implementierers. Die Dateizeiten
stützen es zusätzlich: Referenz 17:13:48, Hilfsdateien 17:13:31, sämtliche gepatchten
Produktivdateien **17:18:16 bis 17:18:23**.

*Erfasst der Fingerabdruck das Relevante?* Für die Behauptung „A ist unangetastet" ja:
14 Tensoren je Batch (`station.x` inkl. Messkanälen und NWP-Spalten, `station.static`,
`icond2/ecmwf .x/.static`, alle drei `edge_index`/`edge_attr`, `target_mask`,
`ground_truth`) plus Shapes, dtypes und die Dimensionen M=3, I2=4, E2=3,
`station_static_features`=13, s2s-Kanten 328, s2s-`edge_attr`-Breite 12. Nicht erfasst
sind `build_eval_batch`, der `nwp_nodes=false`-Pfad und der Kriging-Kanal — dort ist die
Änderung ein reines Voranstellen eines Zweigs hinter `if not neighbour_meas_available`
mit Default `True`, Zeile für Zeile gegen den Diff geprüft.

### §4.5 Permutationstest — Argumentation trägt

Der Test läuft auf `sample_val`, wo `all_global = neighbor_train + val_station_indices`
gilt und `target_mask[N_train:] = True` — die Indizes `< n_neigh` sind also **exakt** die
Nachbarn, die Zielstationen werden nachweislich nicht angefasst. Permutiert werden alle
drei Kanäle, über die ein Nachbar sprechen könnte: `station.x`, `station.static` und der
**Zielindex** der `icond2→station`- und `ecmwf→station`-Kanten, wodurch jeder Nachbar
Gitterpunkte *und* Kantengeometrie eines anderen erhält.

Zusätzlich geprüft, dass es keinen knotenübergreifenden Operator gibt, der in
`model.train()` leaken könnte, in `model.eval()` aber nicht: die einzigen
Normalisierungen sind `nn.LayerNorm` (`shared/nwp_gat.py:77`,
`dcrnn/model/nwp_attention.py:100`), die über die Feature-Achse je Knoten normieren;
**kein BatchNorm irgendwo**, Dropout ist elementweise.

| Variante | max\|Δpred\| an den Zielen |
|---|---|
| **C** (kein Graph, keine Messungen) | **0.000e+00** (exakt) |
| B (Graph, keine Messungen) | 5.856e-01 |
| A (Graph + Messungen) | 5.879e-01 |

Die B-Kontrolle ist das methodisch Entscheidende: B hat wie C überall genullte
Messkanäle, die Permutation wirkt dort **nur** über Geometrie, Statics und NWP-Kontext —
und sie wirkt. C's Null ist also kein Artefakt der Nullung, sondern Folge des fehlenden
Graphen.

### §7.1 A4-Zeitkonvention — empirisch am Datensatz nachgerechnet

Über die tatsächlich gebauten `all_run_pairs` (3297 Paare, 600 zufällig gezogen,
4 406 400 Wertepaare je Versatz), RMSE(rohes ICON-D2 am nächsten Gitterpunkt vs. Messung):

```
offset  RMSE
 -3    1.6719
 -2    1.5745
 -1    1.5040
 +0    1.4958   <- Minimum
 +1    1.5550
 +2    1.6475
 +3    1.7513
```

`timestamps[t_run_abs-1] != run_time` für **0 von 3297** Paaren. Für ECMWF
(zeitindiziert, ganze Reihe, 3,9 Mio Paare) liegt das Minimum ebenfalls bei Versatz 0
(1.7545 gegen 1.7666 bei −1 und 1.7904 bei +1). ICON-D2 und ECMWF sind untereinander
**und** mit dem Ziel zeitgleich.

Alle 118 Vorkommen von `t_run_abs` außerhalb der Verifikationsskripte wurden einzeln
durchgesehen: keine Stelle interpretiert es weiterhin als Laufzeit. Die
Persistenz-Referenz `meas_raw[t_run_abs - 1]` ist korrekt (letzte Beobachtung **vor**
dem Fenster), die Randbedingungen `t_run_abs >= H` und `t_run_abs + F_h <= T` sind
konsistent, und das entfernte handgesetzte `+1` in `evaluate_reference.py` war richtig
zu entfernen.

### §7.2 A2 im echten Trainingspfad — empirisch bestätigt

Mit einem echten `HomoSampler` und echten `MTGNNModel`/`GraphWaveNetModel`:

```
batch.x (24, 96, 32)  nwp_edge_attr (120, 4)  ecmwf_edge_attr (72, 4)
MTGNN:   k Gitterpunkte MIT Kantenattributen vertauscht  -> max|d| = 1.863e-07
MTGNN:   k Gitterpunkte OHNE Kantenattribute vertauscht  -> max|d| = 9.481e-02
MTGNN:   fehlende Kantenattribute -> ValueError OK
WaveNet: MIT -> 1.788e-07   OHNE -> 5.383e-02   ValueError OK
```

Die Attention ist äquivariant, nicht invariant — die Geometrie wird wirklich gelesen.
Analytisch nachgeprüft wurde außerdem die Ausrichtung der Kantenattribute über die
Zeitblöcke: `_expand_hetero_edge_index` erzeugt `src.reshape(-1)` aus einer
`(T, E)`-Matrix, also zeitmajor, und `edge_attr.repeat(T, 1)` liefert dieselbe
Blockstruktur — `repeat` ist hier korrekt, `tile` wäre falsch gewesen.

`in_ch_model` stimmt in allen Konstellationen: für `next_n_ecmwf_trial = 0` wird
`E2_trial = 0`, `ecmwf_out_dim = 0`, `k_ecmwf = 0`, `ecmwf_attn = None` und
`_aggregate_ecmwf` gibt `None` zurück; für `nwp_nodes = false` gilt
`in_ch_model = sampler.in_channels`. Teilbarkeit: `ecmwf_out_dim = nwp_out_dim` bei
denselben `nwp_heads`, MTGNN-Suchraum `low: 16, step: 4` bei `nwp_heads: 4`.
`get_test_results_{mtgnn,wavenet}.py` rufen die Modelle über `evaluate_homo_model` auf,
das die Attribute durchreicht — der neue `ValueError` fliegt dort nicht.

### Weiteres

- **Maskierungsreihenfolge** an allen drei Stellen korrekt: `sampler.py:259-262` (train),
  `sampler.py:356-359` (val), `evaluation.py:126-131` (build_eval_batch). Überall zuerst
  `if not neighbour_meas_available`, dann `elif not hist_wind_available`.
- **Kriging-Guard** sitzt in `train_dcrnn.py:409`, `get_test_results_dcrnn.py:193` und
  `hpo_dcrnn.py:287` — alle drei **vor** jedem Datenladen, alle drei feuern.
- **Leeres Kantenset:** `DiffConv.forward` mit `(2,0)` — `scatter_add_` über leeres `row`,
  `clamp(min=1e-8)` verhindert Division durch null, `propagate` liefert Nullen;
  `edge_weight_from_attr` auf `(0,12)` gibt `(0,)`. Endlicher Forward, `pred (15,48)`,
  NaN=0, Inf=0.
- **Fold-Configs** stimmen station-genau mit `spatial_folds.yaml` überein, Val-Mengen
  paarweise disjunkt, Vereinigung 153.
- **Hostkonsistenz:** l1 `git diff` außerhalb der Pfad-Rewrites = 0 Zeilen, alle Worker
  mit `WEATHER_DB_URL`, Cache-Höhen korrekt.

---

## 5. Die 6 FAIL-Trials vom 2026-08-03 — geklärt, kein Problem

Es sind Worker-Neustarts beim Kampagnen-Hochfahren, deren verwaiste RUNNING-Trials vom
nächsten Worker über Optunas Heartbeat-Sweep auf FAIL gesetzt wurden.

1. `optimize(..., catch=(Exception,))` (`hpo_dcrnn.py:1360`) würde eine echte
   In-Process-Exception fangen und sofort den nächsten Trial starten. In der Datenbank
   sieht man genau das: FAIL #0 `complete=16:19:43.313`, FAIL #1 `start=16:19:43.322`.
2. **Das Log widerlegt die Ketten-Hypothese:** `hpo_dcrnn_wind_dcrnn_base_r1.log`
   enthält drei HPO-Header — 16:06:46, **16:15:55**, **16:27:10** — also drei
   Prozessgenerationen. Optuna führt vor dem ersten Trial `fail_stale_trials` aus, was
   den verwaisten Trial in derselben Millisekunde auf FAIL setzt.
3. `n_intermediate_values = 0` bei allen sechs: keiner hat auch nur den ersten Fold
   beendet.
4. **Das `commit`-User-Attribut zeigt den Neustart direkt:** Trial 0 = `3c519ff`,
   Trials 1 und 2 = `674a043`. Derselbe Studienname über zwei HEADs kann nur durch einen
   Prozesswechsel entstehen. Die Provenance-Zeile hat genau ihren Zweck erfüllt.
5. **Keine Tracebacks, weil es keine Exception gab.**
   `optuna.logging.enable_propagation()` würde sie sonst ins Logfile schreiben.
6. Seit 16:33:54 keine weiteren FAILs.

**Wissenschaftliche Auswirkung: keine.** FAIL-Trials gehen weder in `study.best_*` noch
in den TPE-Sampler ein, und `remaining = max(n_trials - completed, 0)` zählt nur
COMPLETE — das Budget bleibt unangetastet.

Der eigentliche Fund dieser Untersuchung war nicht die FAIL-Serie, sondern **K3**: die
Neustarts haben zwei Cache-Schreibvorgänge ohne DB-Zugriff ausgelöst.

---

## 6. Abweichungen zwischen Code und Dokumenten

Noch **nicht** in die Dokumente eingearbeitet:

- `research_summary` §4.3 sagt, die NWP-Einbindung bei MTGNN sei „analog zu DCRNN".
  Das stimmt nicht: DCRNN benutzt eine Hidden-State-Query, MTGNN/WaveNet eine
  Zero-Query. (Die fehlenden Kantenattribute sind seit A2 behoben.)
- `research_summary` §6 beschreibt drei **zeitliche** Folds mit
  `min_train_date: 2024-03-31`; gefahren wird **räumliche** CV mit in allen Folds
  **identischem** Zeitfenster (Train bis 2024-08-01, 1473 Paare; Val 2024-08-01 →
  2025-08-01, 1460 Paare). Die Folds variieren also nur räumlich; die drei
  Fold-Ergebnisse sind zeitlich vollständig korreliert. **Für eine Wilcoxon-Auswertung
  über Folds ist das relevant.**
- `research_summary` §6 nennt „7 abgeschlossene HPO-Studien"; es sind 8 neu gestartete.
- `research_summary` §3.3 nennt 759 ECMWF-Gitterpunkte; geladen werden 553 für 153
  Stationen.
- `research_summary` §3.2 nennt Laufstunden „00/06/09/12/15"; die Configs fahren
  `[6, 9, 12, 15]`.
- `research_summary` §4.1 nennt „typ. 4" nächste Gitterpunkte; die HPO sucht 1…7
  (ICON-D2) und 0…4 (ECMWF).
- `story_positioning` Contribution (iv) verspricht „attention conditioned on the
  station's evolving forecast state" — das gilt weiterhin nur für DCRNN.
- Die drei Falsifikationen zu (iv) und die Baselines zu (i): siehe R6.
- `ablations_verification_results.md` §2 nennt den Arbeitsbaum vor dem Patch „sauber
  (0 Einträge)". Das kann nicht stimmen, weil `fixture.py` und `batch_fingerprint.py`
  als unversionierte Dateien vorhanden gewesen sein müssen. Am Ergebnis ändert es
  nichts — das Review hat es unabhängig reproduziert.

---

## 7. Gesamturteil des Reviews

- **Laufende Kampagne: tragfähig.** Die Runde-1-Fixes sind wirksam und durch die
  Ablations-Arbeit nachweislich nicht beschädigt. Die Zeitkonvention ist empirisch am
  Datensatz bestätigt, die NWP-Geometrie im echten Trainingspfad ebenso. Kein Grund
  anzuhalten.
- **Ablations-Varianten B und C: sauber, minimal und beweisbar korrekt implementiert.**
  Die HPO-Läufe und die sechs Fold-Trainings können unverändert starten.
- **Auswertung: war blockiert** durch K1/K2 — inzwischen behoben, aber der Eval-Pfad ist
  weiterhin **nie end-to-end gelaufen**.

### Verbleibende Verifikationslücke (Plan §4.7)

Ein kurzer Trainingslauf von Variante B steht aus; er braucht GPU. Ungeprüft bleiben
damit **Backward-Pass, Trainer-Loop und Checkpoint-Roundtrip**. Die Einschätzung des
Reviews: **vertretbar.** Der Vorwärtspfad ist über echte Sampler-Batches und das echte
DCRNN durchgemessen (NWP-Attention, DCGRU-Encoder, autoregressiver Decoder über alle 48
Schritte). Der Backward-Pass sieht dieselben Tensoren; ein leeres `edge_index` erzeugt
keinen zusätzlichen Gradientenpfad, und `out_deg.clamp(min=1e-8)` schützt auch die
Rückwärtsrichtung. Weder `neighbour_meas_available` noch `station_connectivity: "none"`
ändern eine einzige Parameterform — die Checkpoints von A, B und C haben identische
Shapes.

**Empfehlung:** Die ersten zwei bis drei Epochen von B nicht isoliert fahren, sondern
gemeinsam mit einem `get_test_results_dcrnn.py`-Durchlauf — dann schließt ein einziger
Kurzlauf beide Lücken (§4.7 und der nie ausgeführte Eval-Pfad).
