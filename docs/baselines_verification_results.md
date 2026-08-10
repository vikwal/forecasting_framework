# Baselines-Verifikation: MOS-Reparatur, Aufgabe-2-Befund, V1–V6/V9

**Erstellt:** 2026-08-10 · Nacharbeit zu Phase 2 (`docs/prompt_baselines_implementation.md`,
`docs/baselines_implementation_spec.md`). Basis: `forecasting_framework` auf `l2`, Branch
`fix/mtgnn-topo-static-dim`. Der Vorgänger hatte den MOS-Code implementiert, war aber an der
MOS-Gleichung (Rangdefekt) gescheitert und hatte weder Verifikation noch Dokumentation noch
Rollout geliefert. Dieses Dokument schließt Aufgaben 1–4 dieser Nacharbeit ab.

---

## 0. Dateiliste

| Datei | Änderung |
|---|---|
| `geostatistics/baselines/mos.py` | NEU implementiert: Stratifizierung nach (Laufstunde, Lead) statt Tagesgang-Harmonischen (Aufgabe 1), Null-Abschneidung |
| `geostatistics/baselines/dataset.py` | Bugfix: off-by-one in `time_idx` in `build_feature_matrix` und `build_mos_rows` entfernt (Aufgabe 2, siehe §2) |
| `geostatistics/baselines/qrf.py` | unverändert |
| `geostatistics/baselines/evaluate_baselines.py` | unverändert (mos-Aufrufe sind generisch über den Betas-Dict-Schlüsseltyp) |
| `geostatistics/hpo_qrf.py` | unverändert |
| `configs/baselines/config_wind_qrf_local_fold{1,2,3}.yaml` | unverändert |
| `archiv/baselines_verification/verify_baselines.py` | Bugfix: identisches off-by-one in `_offset_rmse` (V2) entfernt; V6 auf Spalten-Aufschlüsselung (nwp_ref/gt/pers_ref, je mit `n_nonzero`/`max\|Δ\|`) umgestellt |
| `archiv/baselines_verification/run_all_mos_scratch.py` | aus dem Repo-Wurzelverzeichnis nach `archiv/baselines_verification/` verschoben (Aufgabe 4.1) |
| `archiv/baselines_verification/{diag_05516_rows,nearest_dist,check_convention,check_imputation_overlap,check_imputation_broad,compute_filtered_mos}.py` | Wegwerf-Diagnoseskripte dieser Nacharbeit, im Repo belassen (nicht `/tmp`) für Reproduzierbarkeit |
| `docs/baselines_verification_results.md` | dieses Dokument |
| `docs/study_overview.md` | §9 R6-Zeile aktualisiert |

Gespiegelt lokal unter `/Users/viktorwalter/Latex/Graphs_Wind_Speed_Forecasting/docs/`.

---

## 1. Aufgabe 1 — MOS neu spezifiziert (der Blocker)

### 1.1 Der Rangdefekt (Befund des Vorgängers, hier reproduziert)

Spezifikation §3.5 sah je Lead `h` eine Gleichung mit sieben (`--nwp-sources both`) bzw. sechs
(`icond2`) Spalten vor: Achsenabschnitt, `ws_i2` (+`ws_e2`), plus vier Tagesgang-Spalten
(`sin/cos(2π·hour/24)`, `sin/cos(4π·hour/24)`). ICON-D2 läuft nur zu den Laufstunden
`{6, 9, 12, 15}`. Bei festem Lead `h` nimmt `valid_time.hour = (run_hour + h) mod 24` daher nur
**vier** Werte an. Fünf rein uhrzeitabhängige Spalten (Achsenabschnitt + 2×2 Harmonische) auf
vier Stützstellen sind linear abhängig — `matrix_rank(X) ≤ 4` bei 6 bzw. 7 Spalten. Die
Rangabsicherung der Spezifikation hat deshalb korrekt gefeuert und **jede** Vorhersage als NaN
markiert. Das ist ein Rangdefekt der alten Spezifikationsgleichung, kein Implementierungsfehler.

### 1.2 Neue Spezifikation (ersetzt §3.5 vollständig)

Belegt an `primo2024comparison` (DWD/KIT 2024,
`/Users/viktorwalter/Latex/Graphs_Wind_Speed_Forecasting/literatur/primo2024comparison.pdf`):
DWDs operationelles ModelMIX trainiert „each hourly time step separately and individually for
each location", der EMOS-Standard schätzt „locally and for each lead time separately", und die
Autoren stellen ausdrücklich fest, dass bei getrennter Schätzung je Lead und Lauf ein Tagesgang
„is not present in the data ... as the forecasts then all validate at the same time of the day".

- **Stratifizierung:** eine eigene OLS-Gleichung je `(Laufstunde r, Lead h)`, also
  48 × 4 = **192 Gleichungen je Fold und Arm**. Laufstunde = Stunde von `run_time`
  (`timestamps[t_run_abs-1]`). Verifiziert am Datensatz (nicht angenommen):
  `run_time.dt.hour` nimmt in **allen** 18 Läufen ausschließlich die Werte `{6, 9, 12, 15}` an
  (je 17 520 Zeilen pro Laufstunde bei 70 080 Zeilen/Station/Fold — exakt gleichverteilt, siehe
  `diag_05516_rows.py`-Ausgabe). Keine unerwartete Laufstunde in keinem Fold.
- **Prädiktoren**, unverändert aus `dataset.build_mos_rows`, namentlich bestimmter
  Windgeschwindigkeits-Feature-Index (Spezifikation §1.4):
  ```
  y = b0 + b1·ws_i2                      --nwp-sources icond2   (2 Parameter)
  y = b0 + b1·ws_i2 + b2·ws_e2           --nwp-sources both     (3 Parameter)
  ```
  KEIN Tagesgangterm, KEINE Harmonischen, KEIN Saisonterm — ausdrückliche
  Nutzerentscheidung (Begründung siehe §7(e) unten).
- **Nachbearbeitung:** `max(pred, 0)`, belegt an `primo2024comparison` („the forecast ... is
  truncated in zero"). Abgeschnittene Zeilen werden geloggt.
- **Rangabsicherung bleibt:** `matrix_rank(X) < X.shape[1]` → NaN + Log, keine Pseudoinverse.
  Sie hat in **keinem** der 6 Arme × 3 Folds gefeuert (192 Zellen je Regional-Fit, 102×192 bzw.
  51×192 je Per-Station-Fit, durchweg voller Rang) — die neue Gleichung ist rangstabil.
- **Dieselbe Gleichung für alle drei Varianten.** MOS-regional/-nearest/-local unterscheiden
  sich ausschließlich in der Fit-Menge (Spezifikation §4.1, unverändert):
  - MOS-regional: 102 Fold-Trainingsstationen gepoolt.
  - MOS-nearest: je Trainingsstation einzeln, Zielstation übernimmt die Koeffizienten der
    geodätisch nächsten Trainingsstation.
  - MOS-local: je **Zielstation** auf ihrer eigenen Trainingsfenster-Historie —
    **TRANSDUKTIVE OBERGRENZE**, in jeder Ausgabe/Tabellenzeile so zu kennzeichnen.

### 1.3 Zeilenzahlen je Fit-Zelle (geloggt, Größenordnung Fold 0)

| Variante | Zellen | Zeilen/Zelle | gegen Parameter |
|---|---|---|---|
| MOS-regional | 192 (= 48×4) | ≈ 102 Stationen × ~368/4 Paare je Laufstunde | 2 bzw. 3 |
| MOS-nearest (Fit) | 102 Stationen × 192 | ≈ 368 | 2 bzw. 3 |
| MOS-local (Fit) | 51 Stationen × 192 | ≈ 368 | 2 bzw. 3 |

(1473 Fit-Paare über 4 Laufstunden verteilt ≈ 368 Paare je Laufstunde und Station; regional
entsprechend ≈ 102× so viele Zeilen je Zelle — im Rahmen der Spezifikationserwartung „regional
etwa 102 × 368, nearest/local etwa 368".)

### 1.4 Lauf: alle sechs Arme × drei Folds, Val-Fenster

`archiv/baselines_verification/run_all_mos_scratch.py` (ein `ctx`-Load je Fold statt 18,
Gesamtrechenzeit rund 20 Minuten, davon der größte Teil ICON-D2/ECMWF-Ladephase, Fits selbst
unter einer Minute je Fold). Alle 18 Kombinationen (`{mos_regional, mos_nearest, mos_local} ×
{icond2, both} × {fold0, fold1, fold2}`) liefen ohne Fehler durch, `n_samples = 70 080` in
jeder erzeugten Zeile, keine Rangdefekt-Warnung.

---

## 2. Aufgabe 2 — Referenzbefund an Station 05516

### 2.1 Vier Kandidaten, geprüft am Code (nicht geraten)

| Kandidat | Befund |
|---|---|
| Gitterpunktwahl (Kandidatenmenge) | **Identisch.** Siehe §2.2 |
| Windgeschwindigkeitsindex namentlich vs. positionell | **Identisch, beide namentlich.** `dataset.py:189-192` und `evaluate_reference.py:370-374` verwenden dieselbe Fallback-Kette (`f == "wind_speed_10m"`, sonst erste Spalte, die `"wind_speed"` enthält) — beide finden Index 1 (`wind_speed_10m` nach `dir_in_deg`-Kodierung), bestätigt im Log beider Pfade |
| Zeitausrichtung `r_curr`/`h-1` | **Ein echter, unabhängiger Bug gefunden und behoben** — betraf aber NICHT die 05516-Diskrepanz (siehe §2.3/2.4) |
| `evaluate_reference.py` andere Kandidatenmenge als `geodesic_knn` | **Nein**, siehe §2.2. Spezifikation §1.4 („153/153 Übereinstimmung") ist bestätigt |

### 2.2 Gitterpunktwahl für Station 05516 — beide Pfade, konkret

Station 05516: lat 54.528301, lon 11.060600.

| Pfad | Code | Gitterpunkt | Koordinaten | Distanz |
|---|---|---|---|---|
| Loader „nächster" (`station_nearest_grid`, genutzt von `evaluate_reference.py`) | `train_stgnn2.py::_select_nearest_grid_files` (pyproj Geod, sortiert über die 22 Dateien in `ML/06/05516/`) | Stem `54_5356_11_0653` | lat 54.535600, lon 11.065300 | **0.8674 km** |
| `dataset.py`s `geodesic_knn` über das globale deduplizierte Gitter | `spatial.py::geodesic_knn` (`pairwise_geodesic_km`, ebenfalls pyproj Geod) | derselbe Stem, andere laufende Nummer je nach `k` (idx 592 bei k=4/N=612, idx 1037 bei k=7/N=1071) | lat 54.535600, lon 11.065300 | **0.8674 km** |
| Brute-Force-Kontrolle über alle 612 bzw. 1071 eindeutigen Gitterpunkte | — | derselbe Stem | dieselben Koordinaten | dieselbe Distanz |

**Beide Pfade wählen exakt denselben Gitterpunkt**, für `k=4` UND `k=7`, und für beide auf
`l1` und `l2` identischen Verzeichnisinhalt (22 Dateien in `ML/06/05516/`, byteidentische
Dateinamen auf beiden Hosts). Die Gitterpunktwahl ist **nicht** die Ursache.

### 2.3 Ein echter, unabhängig gefundener Bug (behoben, aber nicht die Ursache von 2.4)

Beim Nachvollziehen der Zeitausrichtung fiel auf: `dataset.py` berechnete
```python
time_idx = t_run_abs_arr[:, None] + np.arange(F_h)[None, :] - 1
```
in `build_feature_matrix` UND `build_mos_rows`. `np.arange(F_h)` an Spaltenposition `j` **ist
bereits** `h-1` (für Lead `h = j+1`), und `t_run_abs` zeigt bereits auf den ersten
Prognoseschritt (Spezifikation-Definition, `t_run + 1h`). Die korrekte absolute
Zeit-Position für Lead `h` ist `t_run_abs + (h-1) = t_run_abs + j`; der Code subtrahierte ein
zusätzliches, falsches `-1` und zeigte damit durchgehend **eine Stunde zu früh**. Das betraf
NICHT `nwp_ref`/`ws_i2` (die werden direkt über die Lead-Position aus `grid_icond2_runs`
gelesen, unabhängig von `time_idx`), sondern das `valid_time`-Label, `y`/`gt` und — bei
`--nwp-sources both` — `ws_e2`.

**Behoben** in beiden Vorkommen in `dataset.py` (Zeile ~400 und ~608) und in der identischen
Kopie in `archiv/baselines_verification/verify_baselines.py::_offset_rmse` (V2-Testhelfer, hätte
sonst ein Minimum bei Offset +1 statt bei 0 gezeigt und den eigenen Test unterlaufen).
Empirisch bestätigt: nach der Reparatur stimmen `nwp_ref` und `run_time` für Station 05516 über
alle 70 080 Val-Zeilen **exakt** (`max|Δ| = 0.0`) mit `l1:data/raw_preds/icon_d2_fold0_raw.parquet`
überein — vorher waren `valid_time`-Label und `y` konsistent um eine Stunde verschoben (row-level
verifiziert per Direktvergleich).

**Wichtig für die Einordnung:** dieser Bug ist real, unabhängig vom unten beschriebenen Befund,
und musste ohnehin behoben werden (er betrifft auch die QRF-Designmatrix — `y`-Target und
ECMWF-Spalten). Er ist aber **nicht** die Erklärung für die ursprünglich beobachtete
05516-Diskrepanz — die liegt an §2.4.

### 2.4 Die tatsächliche Ursache: veraltete, teilweise korrupte Referenzdaten auf `l1`

Nachvollzogen: `l1:data/raw_preds/icon_d2_fold0_raw.parquet` und `data/test_results/icon_d2_fold0.csv`
sind konsistent in sich (`pred == nwp_ref`, `max|Δ| = 0` über 70 080 Zeilen, beide ergeben für
05516 RMSE = 3.0826 — `evaluate_reference.py` hat **keinen** Fehler). Unser Baseline-Code stimmt
mit dieser Referenz bei `nwp_ref` bitgleich überein (0 von 70 080 Abweichungen) und bei
`run_time` (0 Abweichungen). Auseinander läuft ausschließlich die **Ground Truth**: `gt`
weicht auf 656 von 70 080 Zeilen ab (max`|Δ| = 114.90`), `pers_ref` auf 720 Zeilen
(max`|Δ| = 41.06`). Die Referenzdatei enthält an 05516 `gt`-Werte von **−102.08 bis +74.88 m/s**
(136 Zeilen `|gt| > 25`, 88 Zeilen `|gt| > 40`) — physikalisch unmöglich. Die aktuellen Daten
sind sauber: Rohmessung 05516 max 24.8 m/s (kein Wert > 25), aktuelles `rk_pred` max 25.16
m/s, aktuelles KNN max 22.75 m/s.

**Ursache:** die Referenzdateien auf `l1` sind vom **6. August, 11:58 Uhr**. Die
Kriging-Imputation unter `/mnt/lambda1/nvme1/synthetic/interpol/wind/` wurde am **10. August,
01:02 Uhr** neu erzeugt. Die Referenz stammt aus einer älteren, an einzelnen Stunden korrupten
Imputationsfassung, die inzwischen ersetzt wurde. Das ist ein **Befund am Datenbestand**
(l1-Referenzsnapshot), **kein Fehler** in `evaluate_reference.py` und **kein Fehler** im neuen
Baseline-Code. Nichts davon wird repariert (Auftrag).

**Vollständigkeitsnachweis — sind ALLE korrupten Zeilen imputierte Stunden?** Ja, ausnahmslos,
strukturell zwingend (bei unveränderten Rohmessungen kann `gt` nur dort abweichen, wo es aus der
Imputation kommt). Nachgerechnet mit `build_imputation_mask`/`_lookup_imputed`
(`geostatistics/stdrun/make_stdhp_figures.py`):

| | alle Zeilen | ohne imputierte Stunden |
|---|---|---|
| REFERENZ `pred` gegen `gt` (Station 05516) | 3.0826 | **1.6819** |
| BASELINE `nwp_ref` gegen `gt` (Station 05516) | 1.6782 | **1.6819** |

Auf der **gefilterten** Basis, auf der das Paper berichtet, stimmen Referenz und Baseline für
05516 **exakt** überein.

**Ausgeweitet auf alle 51 Zielstationen × 3 Folds (153 Kombinationen, `check_imputation_broad.py`,
ohne ICON-D2-Neuladung, nur Parquet- und Rohmessdateien):**

| Kennzahl | Wert |
|---|---|
| Stationen/Folds mit irgendeiner `gt`-Abweichung | 119 / 153 |
| davon Anteil der Abweichungen, der in imputierten Stunden liegt | **119 / 119 = 100 %**, ausnahmslos |
| Stationen/Folds mit Abweichung außerhalb imputierter Stunden | **0** |
| max`\|Δ nwp_ref\|` gefiltert, über alle 153 Kombinationen | **0.0** (exakt) |
| max`\|Δ gt\|` gefiltert, über alle 153 Kombinationen | **0.0** (exakt) |
| max`\|Δ(Stations-RMSE, nwp_ref gg. gt, gefiltert)\|`, Baseline gg. l1-Referenz | **3.933 × 10⁻⁷** |
| mittlerer Anteil abweichender `gt`-Zeilen (ungefiltert) je Kombination | 0.713 % |

Diese letzte Zahl (`3.933e-7`) ist die Zahl, die entscheidet, ob V6 auf der berichteten
(gefilterten) Basis bestanden ist — sie liegt weit unter der Spezifikationstoleranz `2e-6`
(siehe V6-Tabelle in §3).

**Nebenbefund (Datenqualität, kein Vergleichbarkeitsproblem):** das AKTUELLE `rk_pred`
(Kriging-Interpolation) liefert an Station 05516 weiterhin physikalisch unmögliche **negative**
Windgeschwindigkeiten bis **−14.766 m/s**, die über die Imputationskette in `gt` einfließen
können. Das betrifft alle Verfahren (Modelle wie Baselines) identisch, ist also kein
Vergleichbarkeitsproblem zwischen den hier verglichenen Armen — aber ein offener
Datenqualitätsbefund für den Bestand.

**Beobachtung, geteilt von beiden Pfaden:** Station 05426 hat sowohl in der Referenz als auch
in unserer Baseline ein ICON-D2-RMSE von 4.405–4.406 (Fold 0) — für ein 2.2-km-Modell
unplausibel hoch, aber IDENTISCH zwischen beiden Pfaden und damit kein Artefakt dieser
Untersuchung; nicht weiter verfolgt (außerhalb des Auftrags).

### 2.5 Zusammenfassung des Mechanismus

Es gab **zwei unabhängige Dinge**, die beide zur Klärung gehören:

1. Ein **echter Bug im neuen Baseline-Code** (`dataset.py`, off-by-one in `time_idx`) —
   **behoben**, betraf `y`/`gt`/`ws_e2`/`valid_time`-Label in Design-Matrix und MOS-Zeilen,
   NICHT `nwp_ref` selbst.
2. Ein **Befund am Datenbestand** (l1-Referenzsnapshot vom 6.8., stale Kriging-Imputation) —
   **gemeldet, nicht repariert**, erklärt die ursprünglich beobachtete 05516-Diskrepanz
   vollständig und ausschließlich in imputierten Stunden.

Keiner der vier ursprünglich zu prüfenden Kandidaten (Gitterpunktwahl, Kandidatenmenge,
Feature-Index-Bestimmung) war die Ursache.

---

## 3. Aufgabe 3 — Verifikation V1–V6, V9 (mit Zahlen)

Ausgeführt mit `archiv/baselines_verification/verify_baselines.py --tests V1,V2,V3,V4,V5,V6,V9
--folds 0,1,2`. V4/V5/V6 gegen `l1` (nicht `l2`). V7 bereits vom Vorgänger bestanden
(übernommen, nicht neu gerechnet): Vorhersagedelta 0.0 über 3 574 080 Zeilen, Spaltendelta
Topo-Fit 102 gg. 153 Stationen = 0.5978.

| Test | Ergebnis | Zahlen |
|---|---|---|
| **V1** Leckage | **PASS**, alle Folds | QRF: `X`/`y` bitidentisch (195 840×43), 20 Bäume `tree_.threshold`/`feature` bitidentisch. MOS-regional: 144 Leads verglichen, `max\|Δβ\|=0`. MOS-nearest: 102 Stationskoeffizientensätze verglichen. MOS-local ausdrücklich ausgenommen (transduktive Obergrenze) |
| **V2** Zeitausrichtung | **PASS**, alle Folds | Minimum bei Offset 0 für rohes ICON-D2 (Fold0/1: RMSE@0=1.4667, Anker 1.4958, `\|Δ\|=0.0291`; Fold2: RMSE@0=1.4981, Anker 1.4958, `\|Δ\|=0.0023`) und für MOS-regional-both (Fold0: `{-2:1.3949,-1:1.3031,0:1.2451,1:1.261,2:1.3386}`, Fold1/Fold2 analog Minimum bei 0) |
| **V3** Fold-Konsistenz | **PASS**, alle 18 Kombinationen × 3 Folds | je 51 IDs, exakt `spatial_fold{N+1}.val_files`, `overlap_with_train=0` |
| **V4** Formatgleichheit | **PASS**, alle 18×3 | Spalten exakt `station_id,mae,rmse,r2,skill,skill_nwp,n_samples`, 51 Zeilen, `n_samples=70080` in jeder Zeile |
| **V5** Zeilenschlüssel-Identität | **PASS**, alle 18×3 | `\|keys\|=3574080`, keine Duplikate, `equal_ref=True` gegen `l1:stdhp_dcrnn_wind_dcrnn_base_fold{N}_raw.parquet`, `horizon∈1..48` |
| **V6 ungefiltert** | **FAIL bei gt/pers_ref, PASS bei nwp_ref** — Befund am Bestand, siehe §2.4 | je Spalte, `n_joined=448443` (dedupliziert auf `(station_id,valid_time)`, spezifikationsgemäß): `nwp_ref` `max\|Δ\|=0.0` (0 nonzero) in ALLEN 18×3; `gt` `max\|Δ\|=114.9/27.8/22.5` (Fold0/1/2), 2445–3859 nonzero-Zeilen von 448443; `pers_ref` `max\|Δ\|=41.1/15.4/12.0`, 2445–3723 nonzero — durchgehend erklärt durch die stale Kriging-Imputation |
| **V6 gefiltert** (ohne imputierte Stunden, alle 153 Station-Fold-Kombinationen) | **PASS** | `max\|Δ nwp_ref\|=0.0`, `max\|Δ gt\|=0.0` exakt; daraus abgeleitet `max\|Δ(Stations-RMSE)\|=3.933×10⁻⁷ < 2×10⁻⁶` |
| **V7** Skalierungsinvarianz | **PASS** (übernommen vom Vorgänger) | `max\|Δpred\|=0.0` über alle Eval-Zeilen; Spaltendelta Topo-Fit 102 gg. 153 = 0.5978 |
| **V9** MOS-Varianten verschieden | **PASS**, alle Folds | 19 584 paarweise Koeffizientenvergleiche/Fold, `min\|Δβ\|≈0.0109`, `max\|Δβ\|=2.97–4.02`, alle `>0`. Geodätisch vs. euklidisch-in-Grad: 14/51 (27.5 %, Fold0/1) bzw. 15/51 (29.4 %, Fold2) Zielstationen erhalten bei euklidisch eine ANDERE nächste Trainingsstation |

**Gesamturteil: V1–V6 und V9 sind grün auf der code-relevanten Basis.** Die einzigen
V6-„Fehlschläge" (`gt`/`pers_ref` ungefiltert) sind ein dokumentierter, nicht-code-bezogener
Befund am `l1`-Datenbestand (§2.4) und blockieren den Commit nicht — sie werden als Befund
geführt, nicht als bestanden schöngerechnet.

### 3.1 Konventionsprüfung: `hpo_qrf.py` (Sortier-Pool) gegen `evaluate_baselines.py` (Training-zuerst)

`hpo_qrf.py` nutzt `station_pool(spatial_fold_defs)` (sortierte Vereinigung der 153 Stationen)
+ `build_folds()`, `evaluate_baselines.py`/`dataset.load_context` nutzt `data.files +
data.val_files` (Training zuerst) direkt aus der Fold-Config. Stationsweise nachgerechnet
(`check_convention.py`, reiner YAML-Vergleich, keine Annahme):

| Fold-Config | Trainingsstationen | Zielstationen |
|---|---|---|
| `config_wind_qrf_local_fold1.yaml` ↔ `spatial_fold1` | 102 = 102, symdiff = ∅ | 51 = 51, symdiff = ∅ |
| `config_wind_qrf_local_fold2.yaml` ↔ `spatial_fold2` | 102 = 102, symdiff = ∅ | 51 = 51, symdiff = ∅ |
| `config_wind_qrf_local_fold3.yaml` ↔ `spatial_fold3` | 102 = 102, symdiff = ∅ | 51 = 51, symdiff = ∅ |

**Beide Konventionen ergeben je Fold identische Trainings- und Zielstationsmengen.** Kein
blockierender Befund.

---

## 4. Die sechs MOS-Ergebnisse

### 4.1 Filtered (ohne imputierte Zielstunden, per Station gemittelt über 3 Folds) — vergleichbar mit dem Paper

| Arm | Quelle | Fold0 | Fold1 | Fold2 | **Mittel** | ggü. ICON-D2 (1.304) | ggü. Persistenz (2.240) |
|---|---|---|---|---|---|---|---|
| MOS-regional | icond2 | 1.2277 | 1.2129 | 1.2798 | **1.2401** | besser | besser |
| MOS-regional | both | 1.1416 | 1.1268 | 1.2125 | **1.1603** | besser | besser |
| MOS-nearest | icond2 | 1.3664 | 1.3356 | 1.4196 | **1.3739** | **SCHLECHTER** | besser |
| MOS-nearest | both | 1.2739 | 1.2556 | 1.3869 | **1.3055** | marginal schlechter (1.3055 > 1.304) | besser |
| MOS-local *(transduktiv)* | icond2 | 1.0058 | 1.0608 | 1.0874 | **1.0513** | besser | besser |
| MOS-local *(transduktiv)* | both | 0.9029 | 0.9512 | 0.9816 | **0.9452** | besser | besser |

Referenzen: rohes ICON-D2 **1.304**, Persistenz **2.240**, DCRNN GRID-NOGRAPH **1.138**,
TFT base **1.186** (alle gefiltert, per Station gemittelt).

### 4.2 Plausibilitätsbefund (kein Bug, ein Befund — Spezifikation V10)

**MOS-nearest/icond2 (1.3739) ist schlechter als rohes ICON-D2 (1.304).** Ungefiltert
verbessern sich nur 52.3 % der 51×3 Ziel-Fold-Kombinationen gegenüber ICON-D2 (Gegenprobe der
Session). MOS-nearest/both liegt mit 1.3055 ebenfalls knapp über der ICON-D2-Schwelle (nur
62.1 % Verbesserung, ungefiltert). Das ist **interpretierbar, kein Bug**: die geodätisch
nächste Trainingsstation liegt für die 51 Zielstationen je Fold typischerweise 30–60 km
entfernt. Vollständige Distanzverteilung (`nearest_dist.py`, alle 51 Zielstationen je Fold):

| Fold | Minimum | Median | Maximum |
|---|---|---|---|
| 0 | 9.03 km | 40.84 km | 57.70 km |
| 1 | 20.26 km | 40.60 km | 60.77 km |
| 2 | 9.03 km | 40.79 km | 70.81 km |

**Befund:** die Übertragung von MOS-Koeffizienten über 30–70 km hinweg (MOS-nearest)
verschlechtert die Vorhersage gegenüber gar keiner Nachbearbeitung des rohen NWP-Signals. Das
regionale Pooling (MOS-regional) und erst recht die transduktive Übertragung (MOS-local)
bleiben dagegen klar unter der ICON-D2-Schwelle. Kein Arm liegt unter der Persistenz-Schwelle.

---

## 5. Abweichungen von der Spezifikation, mit Begründung

| # | Abweichung | Begründung |
|---|---|---|
| 1 | MOS-Modellgleichung (§3.5) vollständig ersetzt | Rangdefekt, siehe §1.1/1.2 — Nutzerentscheidung 2026-08-10 |
| 2 | `n_params()` gibt jetzt 2/3 statt 6/7 zurück | Folge von #1 |
| 3 | Betas-Dicts sind jetzt über `(run_hour, lead)`-Tupel statt nur `lead` indiziert | Folge von #1; `evaluate_baselines.py` und die V1/V9-Tests iterieren generisch über die Dict-Schlüssel und brauchten deshalb KEINE Anpassung, `verify_baselines.py::_offset_rmse` und `v6_reference_identity` brauchten Anpassungen (siehe §2.3/§3) |
| 4 | Off-by-one in `dataset.py::time_idx` behoben | Aufgabe 2, siehe §2.3 — echter Bug im neuen Code |
| 5 | `run_all_mos_scratch.py` aus dem Repo-Wurzelverzeichnis nach `archiv/baselines_verification/` verschoben | Aufgabe 4.1 |
| 6 | Sechs zusätzliche Wegwerf-Diagnoseskripte im Repo unter `archiv/baselines_verification/` belassen statt gelöscht | Reproduzierbarkeit der in diesem Dokument berichteten Zahlen |

## 6. Offene Punkte

- **l1-Referenzdateien sind veraltet** (Kriging-Imputation vom 6.8., seit 10.8. 01:02 ersetzt).
  Alle in dieser Nacharbeit verwendeten `l1`-Referenzen (`icon_d2_fold{N}.csv`,
  `icon_d2_fold{N}_raw.parquet`, `stdhp_dcrnn_wind_dcrnn_base_fold{N}_raw.parquet`) tragen diese
  stale Imputation in ihren `gt`/`pers_ref`-Spalten. Für den gefilterten (Paper-)Vergleich
  folgenlos (§2.4) — aber ob/wann `l1` diese Referenzdateien mit der neuen Imputation neu
  erzeugt, ist offen und liegt außerhalb dieser Nacharbeit.
- **Aktuelles `rk_pred` liefert weiterhin negative Windgeschwindigkeiten** an mindestens einer
  Station (05516, bis −14.77 m/s) — Datenqualitätsbefund, kein Vergleichbarkeitsproblem,
  außerhalb dieser Nacharbeit zu beheben.
- **Station 05426** hat in beiden Pfaden ein ICON-D2-RMSE von 4.405–4.406 — für 2.2-km-NWP
  unplausibel, aber kein Artefakt dieser Untersuchung (identisch in Referenz und Baseline);
  nicht weiter verfolgt.
- **QRF-local, HPO, Retrain, QRF-IDW** bleiben unangetastet (nicht Teil dieser Nacharbeit). Der
  off-by-one-Fix aus §2.3 betrifft auch die QRF-Designmatrix (`y`-Target, ECMWF-Spalten) — die
  vom Vorgänger gemessene Skalierungskurve (§6.1) und die beiden Rauchtest-Trials wurden **vor**
  diesem Fix gemessen und sollten bei der nächsten QRF-Phase mit dem reparierten `dataset.py`
  neu gemessen werden — explizit NICHT Teil dieser Nacharbeit, aber ein Punkt für die nächste
  Phase.

### 6.1 Protokollpflicht (Spezifikation §5.3): Skalierungskurve

Vier gemessene Punkte (0.5 Mio. → 1.0860, 1 Mio. → 1.0858, 2 Mio. → 1.0892, 4 Mio. → 1.0852):
Spannweite 0.004, unter der Flachheitsschwelle 0.005 der Spezifikation — der fünfte Punkt (voller
Satz, 7.21 Mio. Zeilen) entfällt. Zwei Rauchtest-Trials mit `n_fit_rows=100000` erreichten
bereits 1.1591 und 1.1701. **Caveat (siehe §6):** diese vier Punkte und zwei Trials stammen aus
dem Lauf des Vorgängers, VOR dem in §2.3 behobenen off-by-one-Fix.

---

## 7. Für den Methodikteil zu berichten

(a) Unser MOS benutzt EINEN Prädiktor je NWP-Quelle. DWDs operationelles MOSMIX leitet laut
`primo2024comparison` §3.1 etwa 300 Prädiktoren aus 56 Modellvariablen ab, einschließlich
Flächenmitteln um den Standort. Unser MOS ist ein bewusst minimaler unterer Rand der
klassischen Familie und darf im Text nicht als „MOS" schlechthin auftreten.

(b) Zwei NWP-Quellen: operationell werden ICON und IFS GETRENNT post-prozessiert und danach
kombiniert (`primo2024comparison` §1, zweistufiges MOSMIX/WarnMOS). Unsere Zweiquellenvariante
ist eine gemeinsame Regression, also eine Vereinfachung, und ist so zu benennen.

(c) Klassisches lokales MOS ist an einer nie gemessenen Station undefiniert. Induktiv anwendbar
sind nur übertragene Formen, regional und nearest. MOS-local wird ausschließlich als
transduktive Obergrenze berichtet.

(d) DWDs operationelles Post-Processing benutzt Persistenzprädiktoren aus den zuletzt
beobachteten Werten AN DER STATION (`primo2024comparison` §2.2) und ist damit per Konstruktion
transduktiv. Belegt vom Betreiber selbst, dass der operationelle Stand der Technik an einer nie
gemessenen Station nicht anwendbar ist.

(e) Kein Saisonterm, obwohl die Literatur ihn kennt (Schulz und Lerch nutzen einen seasonal
training approach). Begründung: die Graphmodellpfade haben überhaupt keine Kalendermerkmale,
weder Tag im Jahr noch Tageszeit; ein Saisonterm nur für MOS würde dem Boden eine Information
geben, die kein anderer Vergleichspartner hat.

(f) Umgekehrte Asymmetrie, ergänzend zu Spezifikation §8.6: die Baselines BEKOMMEN die
Tageszeit explizit, QRF über `valid_hour_sin/cos`, MOS über die Stratifizierung nach
Laufstunde. Die Graphmodelle bekommen kein Kalendermerkmal und können die Tagesphase nur aus
ihren 48-Stunden-Eingangsfolgen erschließen.

(g) **Das binäre Etikett transduktiv/induktiv wird durch eine Anforderungsmatrix mit zwei
Spalten ersetzt:** „Zielstation im Fit" und „Beobachtungen der Zielstation zur Inferenz
verfügbar".

| Arm | Zielstation im Fit | Beobachtungen der Zielstation zur Inferenz |
|---|---|---|
| GRID+HIST (MTGNN/DCRNN), TFT hist | nein | **ja** |
| MOS-local | **ja** | nein |
| alle übrigen Arme (MOS-regional, MOS-nearest, QRF-local, GRID, BASE, NOGRAPH, NOMEAS, TFT base, ICON-D2, Persistenz) | nein | nein |

Grund: `hist_wind_available` steuert ausschließlich die IGNNK-Nullsetzung
(`meas_hist[:, target_mask_np, :] = 0.0` in `homo_sampler.py:392` und `evaluation.py:132`); bei
der Auswertung sind die Ziele die Val-Stationen, die Beobachter die Trainingsstationen — die
Zielstation ist in KEINEM Trainingsschritt eines GRID+HIST/TFT-hist-Modells enthalten, ihre
Historie ist ausschließlich zur Inferenzzeit sichtbar.

(h) **MOS-local ist damit der EINZIGE Arm im gesamten Vergleich, der die Zielstation im Fit
braucht** — der einzige echt transduktive Arm nach diesem Kriterium. **GRID+HIST ist bezüglich
der Stationsidentität induktiv** und derzeit in `stdhp_dryrun_results.md` §2 (Zeilen 91–93:
„MTGNN | GRID+HIST *(transduktiv)*", „DCRNN | GRID+HIST *(transduktiv)*", „TFT | hist
*(transduktiv, kein Graph)*") sowie in der begleitenden Diskussion (Zeilen 117–118, 145–146,
203–222, 294–326) falsch als transduktiv etikettiert. **Das ist ein Befund, der zu melden und
nicht in dieser Nacharbeit zu ändern ist** — `stdhp_dryrun_results.md` liegt außerhalb des
Umfangs dieser Baseline-Nacharbeit.

(i) **Es gibt zwei verschiedene Kosten, die nicht als eine berichtet werden dürfen:**

- TFT base − TFT hist = **+0.1049** — die Kosten des fehlenden Live-Messstroms an einer nie
  trainierten Station (Inferenzzugriff, Spalte 2 der Matrix in (g)).
- MOS-regional − MOS-local — die Kosten, an diesem Ort überhaupt nie gemessen zu haben
  (Trainingszugriff, Spalte 1 der Matrix in (g)), aus den Ergebnissen dieser Nacharbeit
  (gefiltert, Tabelle §4.1):
  - `both`: 1.1603 − 0.9452 = **+0.2151**
  - `icond2`: 1.2401 − 1.0513 = **+0.1888**

Die zweite Zahl ist 1.8–2.0× so groß wie die erste — nie an einem Ort gemessen zu haben, kostet
in diesem Vergleich MEHR als der fehlende Live-Messstrom an einer trainierten Modellarchitektur.
Das ist ein inhaltlicher Befund für den Methodikteil, keine Fußnote.

---

## 8. Verifikationstabelle (Zusammenfassung)

| # | Test | Status | Kernzahl |
|---|---|---|---|
| V1 | Leckage | PASS | MOS-Betas bitidentisch (`max\|Δ\|=0`), QRF-Bäume bitidentisch |
| V2 | Zeitausrichtung | PASS | Minimum bei Offset 0, alle 3 Folds, ICON-D2 UND MOS-regional-both |
| V3 | Fold-Konsistenz | PASS | 51/51 IDs je Kombination, `overlap=0` |
| V4 | Formatgleichheit | PASS | Spalten exakt, `n_samples=70080` überall |
| V5 | Zeilenschlüssel | PASS | `3574080` Schlüssel, keine Duplikate, `equal_ref=True` |
| V6 ungefiltert | Befund am Bestand | `nwp_ref` PASS (`Δ=0`), `gt`/`pers_ref` FAIL (stale l1-Imputation, §2.4) |
| V6 gefiltert | PASS | `max\|Δ\|=3.933e-7 < 2e-6`, über 153 Station-Fold-Kombinationen |
| V7 | Skalierungsinvarianz | PASS (übernommen) | `max\|Δpred\|=0.0` |
| V9 | MOS-Varianten verschieden | PASS | `max\|Δβ\|>0` in jedem Paar, 27.5–29.4 % geodätisch≠euklidisch |
| Konvention | HPO-Pool vs. Eval-Training-zuerst | PASS | 3/3 Folds identische Stationsmengen |

**Commit-Gate erfüllt:** V1–V6 (auf der code-relevanten/gefilterten Basis) und V9 sind grün.
