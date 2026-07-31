# Implementierungsauftrag: Räumliche statt zeitliche Kreuzvalidierung

**Wie dieses Dokument zu benutzen ist:** Das ist ein eigenständiger
Arbeitsauftrag für eine neue KI-Session ohne Vorwissen über diese
Konversation. Bitte komplett lesen, bevor Code geändert wird — Abschnitt 6
enthält offene Entscheidungen, die vor der Umsetzung mit dem Nutzer geklärt
werden müssen, nicht selbst getroffen werden sollten.

## 0. Kontext

Repo: `/home/viktor/Work/forecasting_framework` (mehrere Hosts, siehe
`docs/topo_rehpo_plan.md` Abschnitt 4 für die Host-Landschaft). Branch:
`fix/mtgnn-topo-static-dim`.

Drei Graph-Architekturen (DCRNN, MTGNN, Graph WaveNet, alle in
`geostatistics/`) prognostizieren Windgeschwindigkeit/-leistung an
Wetterstationen aus NWP-Gitterpunktdaten, mit einer IGNNK-artigen induktiven
Aufgabe: pro Trainings-Batch wird eine zufällige Teilmenge der
**Trainingsstationen** als Ziel maskiert, die übrigen Trainingsstationen
dienen als reale Nachbarn. Bei der Validierung (`HomoSampler.iter_val()`,
`geostatistics/homo_sampler.py:436`) sind **alle** Trainingsstationen
Nachbarn und **alle** Validierungsstationen Ziele — eine echte
Generalisierungsprobe auf nie gesehene Standorte.

Es gibt gerade eine parallele Code-Review-Session zu topografischen
Knoten-Features (siehe `docs/topo_features_review_brief.md`), die zwei
Blocker fand; einer davon (`load_topo_node_features` statt
`load_topo_station_features`) ist bereits gefixt (Commit `4f62937`). Der
andere (Code-Stand auf Host `l1`) ist unabhängig von dieser Aufgabe und muss
vor einem Produktionslauf ebenfalls behoben sein — betrifft aber nicht die
hier beschriebene Umsetzung.

## 1. Ausgangslage — wie die 3 Folds heute funktionieren

Jede Architektur hat 3 Fold-Configs (z. B.
`configs/mtgnn/config_wind_mtgnn_nwp_fold1/2/3.yaml`). Der **einzige**
Unterschied zwischen ihnen ist aktuell ein Zeitfenster:

```yaml
data:
  test_start: '2024-07-31'   # unterschiedlich pro Fold
  test_end:   '2024-11-30'   # unterschiedlich pro Fold
  files:      [... 103 Stations-IDs ...]   # IDENTISCH über alle 3 Folds
  val_files:  [... 50 Stations-IDs ...]    # IDENTISCH über alle 3 Folds
  test_files: [... 50 Stations-IDs ...]    # IDENTISCH über alle 3 Folds
```

`test_start` legt in `train_mtgnn.py`/`train_wavenet.py`/`train_dcrnn.py`
den Zeitpunkt fest, der `train_run_pairs` von `val_run_pairs` trennt
(`t_run < split_time` → train, sonst val — siehe `train_mtgnn.py:401-407` und
die Zuordnung bei `train_mtgnn.py:582`). `test_end` kappt den geladenen
Zeitraum insgesamt (`train_mtgnn.py:355-356`).

**Es gibt drei disjunkte Stationsgruppen, nicht zwei:**

```python
files:      103 Stationen  # Trainingsnachbarn
val_files:   50 Stationen  # HPO-/Dev-Validierung (iter_val Ziele)
test_files:  50 Stationen  # finale, zurückgehaltene Testmenge (--test-mode)
```

`--test-mode` (in allen `train_*.py`) schaltet um: dann werden `files +
val_files` (153) als Trainingsnachbarn und `test_files` (50) als Ziele
benutzt. Insgesamt **203 eindeutige Stationen**, nicht 153 — das ist wichtig
für Abschnitt 3.

Die Studiennamen-Auflösung (`hpo_stem = re.sub(r'_fold\d+$', '', config_stem)`,
identisch in `train_dcrnn.py:338`, `train_mtgnn.py:265`,
`train_wavenet.py:251`, den `get_test_results_*.py`- und `hpo_*.py`-Skripten)
strippt das `_foldN`-Suffix, um pro Architektur/Variante **eine** gemeinsame
Optuna-Studie zu finden, unabhängig davon, mit welchem Fold gerade retrainiert
wird. Diese Namenskonvention bleibt unverändert — solange die neuen Configs
weiterhin `..._fold1.yaml`/`_fold2.yaml`/`_fold3.yaml` heißen, funktioniert
das automatisch weiter.

## 2. Entscheidung des Nutzers

Statt 3 **zeitlichen** Folds (gleicher Stationssplit, unterschiedliches
Zeitfenster) werden 3 **räumliche** Folds gefahren (unterschiedlicher
Stationssplit, **volles Jahr** in jedem Fold — die zeitliche Kreuzvalidierung
entfällt ersatzlos). Begründung: die zentrale Aussage der Studie ist
Generalisierung auf ungesehene *Standorte*; die Streuung über Stationsgruppen
misst genau diese Fehlerquelle, während die bisherige Zeitfenster-Rotation nur
zeitliche Robustheit maß, die für den Kern-Claim zweitrangig ist.

**Wichtiger Kontext, warum die Standard-Empfehlung für räumliche CV
(zusammenhängende Blöcke zurückhalten, "spatial blocking") hier NICHT gilt:**
Nachbarstationen sind bei dieser Architektur ein bewusster Input, kein
Leckage-Kanal (siehe Abschnitt 0). Ein geblockter Split entfernt
Zielstationen ihre nahen Nachbarn und macht die Aufgabe künstlich hart und
architekturfeindlich — verifiziert an den echten 153 Stationen (Skript unten
in Abschnitt 3 reproduzierbar):

| Strategie | größte Nachbar-Lücke (Val→nächste Train-Station) | Terrain-Ungleichgewicht der Folds |
|---|---|---|
| aktueller fester Split (Referenz) | 55.7 km | — |
| Geblockt (Längengrad-Streifen) | **260.9 km** | 0.252 |
| Zufällig | 85.3 km | 0.197 |
| **Gestreut + terrain-balanciert** | **70.7 km** | **0.046** |

Verwendet wird **gestreut + terrain-balanciert**: räumlich nahe Tripel
bilden, aus jedem Tripel geht eine Station in einen Fold, die Zuordnung
innerhalb der Tripel wird so gewählt, dass die Terrain-Mittel der drei Folds
möglichst gleich sind (Nutzung der 9 z-skalierten Topo-Features aus
`load_topo_station_features`).

## 3. Bereits erledigt — die Fold-Zuordnung existiert

`configs/spatial_folds.yaml` enthält bereits eine geprüfte 51/51/51-Aufteilung
der **153** Stationen aus `files + val_files` (**nicht** `test_files`, siehe
Abschnitt 6.1 für die Begründung und eine offene Frage dazu):

```yaml
spatial_fold1:  {files: [...102 IDs...], val_files: [...51 IDs...]}
spatial_fold2:  {files: [...102 IDs...], val_files: [...51 IDs...]}
spatial_fold3:  {files: [...102 IDs...], val_files: [...51 IDs...]}
```

Jede der 153 Stationen ist genau einmal `val_files`, sonst `files` — eine
saubere Partition, verifiziert per Skript. Das Erzeugungsskript liegt unter
`geostatistics/make_spatial_folds.py` und enthält die `make_dispersed()`-
Funktion (Tripel-Bildung + lokale Suche über Permutationen für die
Terrain-Balance) sowie die Vergleichsrechnung aus der Tabelle in Abschnitt 2
— bei Bedarf neu ausführen (z. B. andere Fold-Anzahl, andere Architektur mit
anderem Stationspool), sonst reicht `configs/spatial_folds.yaml` als fertiges
Ergebnis.

**Zu tun:** die Stationslisten aus `spatial_folds.yaml` in die drei
Fold-Configs jeder Architektur/Variante übernehmen (`files`/`val_files`
ersetzen), `test_start`/`test_end` gemäß Abschnitt 4 entfernen bzw. anpassen.

## 4. Code-Änderungen: zeitliche Trennung entfernen

**Ziel:** in jedem Fold werden `train_run_pairs` und `val_run_pairs` aus dem
**gesamten verfügbaren Zeitraum** gebildet — die Trennung train/val entsteht
ausschließlich durch die Stationszugehörigkeit (welche IDs in `files` vs.
`val_files` stehen), nicht mehr durch `t_run < split_time`.

Betroffene Stellen (identisches Muster in allen sechs Skripten
`train_{dcrnn,mtgnn,wavenet}.py`, `hpo_{dcrnn,mtgnn,wavenet}.py`):

```python
# bisher (train_mtgnn.py:400-407):
test_start = data_cfg.get("test_start")
if test_start:
    split_t = int(np.searchsorted(timestamps, pd.Timestamp(test_start, tz="UTC"), side="left"))
else:
    split_t = int(T * (1 - data_cfg.get("val_frac", 0.2)))
split_time = timestamps[split_t]
...
(train_run_pairs if t_run < split_time else val_run_pairs).append(pair)  # Zeile ~582
```

Muss so geändert werden, dass **alle** validen Run-Paare in **beide** Listen
eingehen (`train_run_pairs = val_run_pairs = all_run_pairs`), oder — sauberer
— dass die Skripte gar keine getrennten Listen mehr bilden und `HomoSampler`
für `sample_train()` und `iter_val()` denselben Pool von Run-Paaren nutzt.
Bitte prüfen, ob `HomoSampler` das ohne Weiteres akzeptiert (Konstruktor
nimmt `train_run_pairs`/`val_run_pairs` separat entgegen,
`geostatistics/homo_sampler.py:104-105`) oder ob eine Kopie beider Listen auf
dieselbe zugrundeliegende Liste reicht.

**Was mit `test_end` und `run_cutoff` passiert:** aktuell kappt `test_end`
den geladenen Gesamtzeitraum, unabhängig vom Split (`train_mtgnn.py:355-356`,
`run_cutoff = pd.Timestamp(test_end) if test_end else None`, dann
`meas_raw = meas_raw[:cut_idx]`). Für die HPO-Folds (spatial_fold1-3) sollte
dies auf das Ende des insgesamt verfügbaren Datensatzes gesetzt werden (volles
Jahr, wie vom Nutzer gefordert), nicht mehr pro Fold verschieden. Für das
finale, zeitlich UND räumlich zurückgehaltene Testset (`test_files`,
`--test-mode`) ist unklar, ob weiterhin ein zeitlicher Cutoff sinnvoll ist —
**das ist eine offene Entscheidung, siehe Abschnitt 6.2, nicht selbst
festlegen.**

**Nicht vergessen:** `val_frac`-Fallback (`else`-Zweig oben) ist jetzt
irrelevant, wenn `test_start` überall entfernt wird — als toter Code-Pfad
markieren oder entfernen, je nachdem ob er noch anderswo referenziert wird
(`grep -rn val_frac`).

## 5. Topo-Skalierung pro Fold neu fitten (kritisch, sonst stilles Leck)

`load_topo_station_features(..., n_train=N_train)` (siehe
`geostatistics/stgnn/utils/topo_features.py`) fittet z-Score und
Median-Imputation auf `arr[:n_train]` — das setzt voraus, dass `station_ids`
**train-zuerst sortiert** ist (`all_ids = train_ids + val_ids`, siehe
`train_mtgnn.py:318-324` und die identische Konstruktion in den anderen fünf
Skripten). Das gilt für jeden der drei neuen räumlichen Folds automatisch,
**solange** `all_ids` weiterhin aus dem jeweiligen `train_ids + val_ids` der
Fold-Config gebaut wird (was der bestehende Code bereits so macht) — hier ist
vermutlich **kein Code-Fix nötig**, nur eine Verifikation:

- [ ] Für alle drei neuen Fold-Configs prüfen, dass `all_ids` tatsächlich in
      der Reihenfolge `files (train) + val_files (val)` aufgebaut wird (nicht
      z. B. alphabetisch sortiert irgendwo dazwischen) — sonst fittet die
      Skalierung auf eine zufällige Teilmenge statt auf die echten
      Trainingsstationen dieses Folds.
- [ ] Stichprobenartig für Fold 2 und Fold 3 die geloggte Meldung
      `"Loaded N topographic station features (z-score on first M stations)"`
      gegen die erwartete `N_train` des jeweiligen Folds gegenprüfen.

## 6. GNNCache — vermutlich kein Fix nötig, aber verifizieren

Der Cache-Key (`GNNCache.make_key(_key_cfg)`, z. B. `hpo_mtgnn.py:414-423`)
wird aus `cfg["data"]` gebildet, was `files`/`val_files`/`test_files`
einschließt. Da die drei neuen Fold-Configs unterschiedliche Stationslisten
haben, sollten sie **automatisch** unterschiedliche Cache-Keys erhalten —
anders als bei den bisherigen rein zeitlichen Folds, wo der Stationssplit
identisch war und deshalb (vermutlich) *ein* gemeinsamer Cache-Eintrag für
alle 3 Folds existierte. Zu prüfen:

- [ ] Nach der Umstellung: erzeugen die 3 Fold-Configs pro Architektur/Variante
      3 verschiedene `GNNCache`-Keys? (`grep "GNNCache key:" logs/...` nach
      einem `--preprocess-only`-Lauf pro Fold vergleichen)
- [ ] Alte Cache-Einträge unter `data_cache/gnns/` löschen oder ignorieren
      (sie gehören zum alten zeitlichen Split und sind für die neuen
      Stationslisten ohnehin ein Cache-Miss, aber sie belegen Speicherplatz).

## 7. Reihenfolge der Umsetzung

1. Stationslisten aus `configs/spatial_folds.yaml` in die drei Fold-Configs
   jeder Architektur/Variante übertragen (9 bzw. 11 Config-Familien × 3
   Folds, siehe `docs/topo_features_review_brief.md` Abschnitt 7 für die
   Liste der Studien — dort ggf. auf den aktuellen Stand prüfen, falls seither
   weitere Studien/Ablationen beschlossen wurden).
2. Zeitliche Trennung in den sechs `train_*.py`/`hpo_*.py`-Skripten entfernen
   (Abschnitt 4).
3. Smoke-Test mit `--preprocess-only` pro Architektur, mindestens für Fold 1
   und Fold 2 (unterschiedliche Stationslisten prüfen unterschiedliche
   Code-Pfade als Fold 1 allein).
4. Topo-Skalierung und Cache-Key-Verifikation (Abschnitte 5, 6).
5. Erst danach: die eigentliche HPO-Kampagne planen/starten (siehe
   `docs/topo_rehpo_plan.md` für die GPU-Verteilung — die dortige Tabelle
   müsste um die neue Fold-Struktur ergänzt werden, falls sich die Zahl der
   Studien durch die räumliche CV ändert, z. B. wenn pro Fold ein eigener
   HPO-Lauf statt eines gemeinsamen sinnvoll wird).

## 8. Entschiedene Fragen (Stand 2026-07-31)

### 8.1 `test_files` bleiben unangetastet — entschieden

Die 50 `test_files` sind ein gut verteilter, historisch gewachsener Rückhalt
und werden **nicht** neu zugeschnitten und **nicht** in die Fold-Rotation
aufgenommen. Es bleibt bei einer Ebene räumlicher CV (3 Folds über die 153),
keine geschachtelten äußeren Test-Folds.

Konsequenz, die dokumentiert werden muss: die Testauswertung ist dichter
vernetzt als jeder HPO-Fold. Geodätisch gemessen (Skript unten):

| Bedingung | Ziele | Nachbarn | median | p90 | max |
|---|---|---|---|---|---|
| HPO, alter fester Split | 50 | 103 | 38.7 | 48.6 | 55.7 |
| HPO, spatial fold1/2/3 | 51 | 102 | ~40.8 | 52–56 | 57.7–70.8 |
| **finaler Test** (`--test-mode`) | 50 | 153 | **23.9** | 30.0 | 34.5 |

Das ist keine Eigenschaft der Fold-Konstruktion, sondern des fixen Testsets:
die Untergrenze des 153er-Netzes (jede Station zur nächsten der 152 anderen)
liegt bei median 39.3 km — die Folds liegen mit 40.8 km praktisch auf dem
Optimum, tiefer geht es bei *keiner* Fold-Größe. Die Hyperparameter werden
also unter etwas dünnerer Vernetzung gewählt, als das finale Modell vorfindet
(konservativ, nicht optimistisch). In der Methodik so benennen.

### 8.2 Volles Jahr überall — entschieden

Alle drei Folds trainieren auf dem gesamten verfügbaren 12-Monats-Zeitraum.
`test_start`/`test_end` entfallen in den Fold-Configs, die zeitliche Trennung
verschwindet ersatzlos (Abschnitt 4). Auch für `--test-mode` gilt volles Jahr;
die Testauswertung ist räumlich out-of-sample, nicht zeitlich.

### 8.3 Ablationen — erledigt, kein Konflikt

Suchraum-Reduktion (`DCRNN_B`, `DCRNN_C`) und Datensplit sind orthogonal; die
Sorge aus der Vorversion dieses Dokuments war unbegründet. Einzige Auflage:
die Ablationen müssen dieselben drei räumlichen Fold-Configs verwenden, sonst
sind ihre Zahlen nicht gegen die Hauptstudien stellbar.

### 8.4 Fold-Größe 51/51/51 — bestätigt

Die Variante mit kleineren Val-Mengen (30/30/30, 123 Trainingsnachbarn, 63
Stationen dauerhaft Train) wurde durchgerechnet und bringt **nichts**:

| Variante | median | p90 | max | Terrain-Ungleichgewicht |
|---|---|---|---|---|
| 51/51/51 (volle Partition) | 40.8 | 52–56 | 70.8 | **0.046** |
| 30/30/30 | 40.7 | 48–56 | 70.8 | 0.120 |

Grund: `make_dispersed()` legt zu jeder Val-Station ihren unmittelbaren
räumlichen Partner in den Train-Satz, die bindende Größe ist damit der
Gruppenabstand (≈ die Netzdichte von 39.3 km), nicht die Anzahl der Nachbarn.
30/30/30 verschlechtert zusätzlich die Terrain-Balance und die statistische
Aussagekraft (30 statt 51 Zielstationen je Fold). Es bleibt bei 51/51/51.

### 8.5 Distanzen geodätisch — erledigt

`geostatistics/make_spatial_folds.py` rechnete Haversine (Kugel). Umgestellt
auf `pairwise_geodesic_km` (WGS-84), wie im übrigen Repo. Effekt gemessen:
max. 1.96 km / 0.33 % Abweichung, nächster Nachbar ändert sich für 2 von 153
Stationen — die Fold-Zuordnung in `configs/spatial_folds.yaml` ist danach
**identisch** geblieben. Das Skript nimmt jetzt `--n-val`, `--n-folds`,
`--compare` und `--write` und schreibt die YAML selbst (sortiert, für
lesbare Diffs).

### 8.6 Visualisierung

`geostatistics/fold_dashboard.py` (Streamlit): Karte mit Train/Val/Test je
Fold, optionale Verbindungslinien Val → nächste Train-Station, Hervorhebung
zu isolierter Val-Stationen, Kennzahlentabelle über alle Folds inkl. der
beiden Referenzzeilen aus 8.1, Terrain-Balance-Tabelle, Stationsliste mit
CSV-Export. Start: `streamlit run geostatistics/fold_dashboard.py`.
Alternativ weiterhin `geostatistics/gen_graph_html.py --config <fold-config>`
für die Folium-Karte inkl. Graphkanten und NWP-Gitterpunkten.

## 8.7 Umsetzungsstand (2026-07-31) — implementiert

Alles aus Abschnitt 9/10 ist umgesetzt. Betroffene Dateien:

| Datei | Änderung |
|---|---|
| `geostatistics/spatial_cv.py` | **neu** — lädt/validiert `spatial_folds.yaml`, bildet den Stationspool und die Fold-Indizes, löst `hpo.cv_mode` auf |
| `geostatistics/stgnn/utils/topo_features.py` | `load_topo_station_features(_dict)` nimmt jetzt wahlweise `n_train` (führende N) **oder** `train_idx` (explizite Indizes) |
| `geostatistics/hpo_{dcrnn,mtgnn,wavenet}.py` | `cv_mode`-Weiche, Fold-Pläne statt `fold_splits`, Scaler und Topo-z-Score pro Fold auf dessen Trainingsstationen, Cache-Key auf den Pool abgebildet |
| `geostatistics/train_{dcrnn,mtgnn,wavenet}.py` | neuer Schlüssel `data.val_start` — zweite Zeitgrenze zwischen Train und Val, `test_start` bleibt die Testgrenze |
| `configs/{dcrnn,mtgnn,wavenet}/*_fold[123].yaml` (33 Stück) | Stationslisten aus `spatial_folds.yaml`, `test_end`/`val_frac` entfernt, `val_start: '2024-08-01'` + `test_start: '2025-08-01'`, `hpo.cv_mode: spatial` + `hpo.spatial_folds`, `n_val_stations: null` |

**Rückwärtskompatibilität:** `hpo.cv_mode` ist per Default `temporal`, und ohne
`data.val_start` fällt die Val-Grenze wie bisher auf `test_start` — jede Config
ohne diese Schlüssel verhält sich exakt wie vorher. 15 Configs (u. a. alle
`_fold9`) laufen unverändert zeitlich weiter. Der `val_frac`-Pfad bleibt
bestehen und ist **nicht** toter Code (entgegen Abschnitt 4).

Fallstricke, die dabei auftauchten und jetzt abgesichert sind:

- `val_frac: 0.25` stand in allen Fold-Configs. Hätte man nur `test_start`
  entfernt, hätte der Fallback still 25 % des Zeitraums abgeschnitten. Deshalb
  ist der Stationssplit an einen **expliziten** Schlüssel gebunden und nicht an
  die Abwesenheit von Zeit-Keys.
- `test_start` und `test_end` kappen beide den geladenen/verwendeten Zeitraum und
  bleiben deshalb Teil des `GNNCache`-Keys; nur die Stationslisten werden im
  räumlichen Modus auf den gemeinsamen Pool abgebildet, damit die drei
  Fold-Configs nicht drei identische Cache-Einträge erzeugen.
- `hpo.n_val_stations` würde die Fold-Val-Menge beschneiden (51 → 50); in den
  Configs jetzt `null`, und `build_folds()` warnt, falls doch gesetzt.

## 8.8 Zeitfenster — entschieden

Der Datenbestand reicht von **2023-07-24 bis 2026-07-28**. Entscheidung des
Nutzers — drei Zeiträume, zwei Grenzen, **in allen drei Folds identisch**:

| Zeitraum | Fenster | Rolle |
|---|---|---|
| Training | 2023-07-24 → 2024-07-31 | ein Jahr |
| Validierung | 2024-08-01 → 2025-07-31 | ein Jahr |
| Test | ab 2025-08-01 | zurückgehalten, nur `--test-mode` |

Die räumlichen Folds rotieren **nur die Stationen**; das Zeitfenster ist in
jedem Fold dasselbe. Die Validierung ist damit räumlich *und* zeitlich
out-of-sample (51 ungesehene Stationen in einem ungesehenen Jahr). Das ersetzt
die alte Expanding-Window-CV, deren Folds sich ausschließlich zeitlich
unterschieden.

Dafür genügt **ein** neuer Schlüssel, weil die Skripte Run-Paare ohnehin an
einer Zeitgrenze teilen — es kam nur die zweite Grenze dazu:

```yaml
data:
  val_start:  '2024-08-01'   # Train davor, Validierung danach
  test_start: '2025-08-01'   # ab hier zurueckgehalten (finaler Test)
```

- `val_start` (Default: nicht gesetzt) trennt Train von Val. Fehlt der
  Schlüssel, fällt die Val-Grenze wie bisher auf `test_start` zusammen — altes
  Verhalten, alle Configs ohne den Schlüssel bleiben unverändert.
- `test_start` kappt nach oben: Run-Paare ab dieser Grenze werden verworfen.
- **`--test-mode` schaltet die Logik bewusst ab**: dort *ist* die Testperiode ab
  `test_start` die Val-Menge, mit `test_files` als Zielstationen. Die
  Headline-Zahl ist damit räumlich **und** zeitlich out-of-sample — Frage 6.2 der
  Vorversion ist damit erledigt.
- `val_start` wird **aus dem `GNNCache`-Key ausgenommen**: es teilt nur bereits
  geladene Paare auf und ändert nichts am Cache-Inhalt. Ohne diese Ausnahme lädt
  jede Verschiebung der Val-Grenze zwei Jahre Rohdaten neu.

`test_end` wird in den Fold-Configs nicht mehr gesetzt (die Kappung leistet
`test_start`), bleibt aber im Code und im Cache-Key wirksam, falls jemand den
geladenen Zeitraum zusätzlich beschneiden will.

**Verifiziert** (mtgnn/wavenet, 1-Trial-Lauf gegen die echten Fold-Configs):

```
Zeitfenster je Fold — Train bis 2024-08-01 (1473 Paare), Val 2024-08-01 bis 2025-08-01 (1460 Paare)
spatial_fold1/2/3 — je 102 Trainings-/51 Zielstationen
Loaded 8 topographic station features (z-score on 102 train stations, explicit indices)
```

## 9. Erledigt: die zweite CV-Ebene in den `hpo_*.py`

Abschnitt 4 übersieht, dass in den HPO-Skripten eine **eigene** zeitliche
3-fach-CV steckt (`hpo.n_folds: 3`, expanding window,
`hpo_mtgnn.py:642-654`): die Fold-Grenzen werden aus `test_start` abgeleitet
(`foldable_secs = test_start_dt - min_train_dt`), jeder Trial trainiert
`n_folds`-mal und Optuna bekommt den Mittelwert. Entfällt `test_start`
ersatzlos, bewertet jeder Trial nur noch einen einzigen Lauf.

**Entscheidung des Nutzers:** diese innere zeitliche CV wird durch die drei
räumlichen Folds ersetzt (ein Trial = drei Trainings auf spatial_fold1/2/3,
Optuna sieht den Mittelwert), aber **abwärtskompatibel** — der zeitliche
Expanding-Window-Modus muss erhalten bleiben und später wieder wählbar sein.

Umsetzung: neuer Config-Schlüssel `hpo.cv_mode: temporal | spatial`, Default
`temporal`, damit alle bestehenden Configs und laufenden Studien unverändert
weiterlaufen. Im Modus `spatial` treten `hpo.spatial_folds` (Pfad auf
`configs/spatial_folds.yaml`) an die Stelle von `fold_splits`.

Offen ist die Datenhaltung im Modus `spatial` — siehe Abschnitt 10.

## 10. Offene Entscheidung: Datenhaltung der räumlichen HPO-Folds

Die zeitlichen Folds waren billig, weil alle drei auf **derselben** geladenen
Stationsmenge arbeiten und nur das Zeitfenster verschieben. Räumliche Folds
ändern dagegen, welche Station Nachbar und welche Ziel ist — betroffen sind
Graphaufbau, Topo-z-Score, Scaler und der `GNNCache`-Key. Zwei Wege:

**(a) Superset einmal laden, pro Fold nur umindizieren.** Alle 153 Stationen
werden einmal geladen; pro Fold ändern sich nur `train_station_indices` /
`hpo_val_station_indices`, der Stationsgraph wird pro Fold neu gebaut (billig)
und Scaler/Topo-z-Score pro Fold auf dessen Trainingsstationen gefittet. Ein
einziger Cache-Eintrag für alle drei Folds. Erfordert, dass die
Stations-Reihenfolge nicht mehr implizit "train zuerst" ist — genau die
Annahme aus Abschnitt 5 (`n_train=len(...)` in
`load_topo_station_features`) muss dann auf eine explizite Indexliste
umgestellt werden. Empfohlen.

**(b) Pro Fold eine eigene Config und ein eigener Ladevorgang.** Näher am
bestehenden Code, aber dreifache I/O- und Cache-Kosten pro Trial und drei
`GNNCache`-Einträge.
