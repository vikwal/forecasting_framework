# Handoff — Solar-Kampagne, Stand 17.09.2026

> **Vorgänger ersetzt.** Hier stand bis zum 14.09.2026 die abgeschlossene
> Wind-Testauswertung (neun Läufe, Ergebnis als §19 in
> [evaluation_results.md](evaluation_results.md)). Die ist erledigt und
> dokumentiert; dieses Dokument beschreibt die laufende Solar-Kampagne.

Referenz zum Aufbau: [solar_tft_kampagne.md](solar_tft_kampagne.md).
Stationsaufteilung: [station_splits_solar.md](station_splits_solar.md).

---

## 1. Worum es geht

Solarstrahlungsprognose (GHI und DHI gemeinsam, 30-min-Raster, 48-h-Horizont)
aus ICON-D2-SL und ECMWF-HRES. Zwei Architekturen sollen gegeneinander
gehalten werden:

* **TFT** (`train_cl.py`, `configs/solar_tft/`) — fertig, Zahlen gültig
* **DCRNN** (`geostatistics/train_dcrnn.py`, `configs/solar_dcrnn/`) — sechs
  Arme neu trainiert (`v5`, 15.09.2026) und gegen den TFT gehalten, s. §2.1

Zeitachse: Training 2023-08-01…2024-07-31, Validierung 2024-08-01…2025-07-31,
Testjahr 2025-08-01…2026-07-31 **zurückgehalten**. Stationen: 21 Testsatz nie
im Training, 62 Pool in drei rotierenden Folds. Ablationen laufen auf Fold 1;
die drei Folds sind für die HPO vorgesehen.

## 2. Was gültig ist

| Lauf | Ergebnis |
|---|---|
| TFT Arm A, Fold 1–3 | GHI RMSE **63.56**, Skill_NWP 0.108 / DHI **36.08**, 0.114 |
| TFT Arm B (+9 Trainingsstationen) | GHI −0.05 (p = 0.66) / DHI −0.145 (p = 0.0094) |
| TFT Static-Ablation (6 statt 3 statische Features) | Null-Ergebnis, p = 0.66 / 0.56 |
| DCRNN v5, sechs Arme, Fold 1 | s. §2.1 — gegen den TFT gehalten |
| TFT HPO-Retrain, Fold 1–3 (17.09.) | GHI RMSE **64.16**, Skill_NWP 0.105 / DHI **36.69**, 0.110 — s. §5.1.2 |
| **TFT Schlussmessung, Testjahr (17.09.)** | GHI RMSE **61.53**, Skill_NWP **0.119** / DHI **33.89**, **0.141** — 21 Teststationen, s. §5.1.3 |

Dateien unter `results/solar/` (TFT Arm A/B), `results/solar_dcrnn_*_v5_*.pkl`
(DCRNN) und `data/{test_results,raw_preds}/tft_solar_tft_fold<N>*` (HPO-Retrain).
Die Auswertung der beiden TFT-Arme steht im Gesprächsverlauf, ein Skript dafür
gibt es noch nicht; der Architekturvergleich läuft über
`scripts/eval_solar_arch.py`.

Der HPO-Retrain liegt **nicht** besser als Arm A, obwohl er dessen Aufgabe mit
optimierten Hyperparametern löst. Das ist der Beleg für den HPO-Befund aus
§5.1.1 — an dieser Stelle ist das Modell datenlimitiert. Die Zahlen beider Zeilen
stammen aus verschiedenen Skripten und Laufmengen und sind auf ±0.5 W/m² genau zu
lesen, nicht schärfer.

### 2.1 Architekturvergleich DCRNN ↔ TFT (15.09.2026)

Gemeinsame Auswertung über `scripts/eval_solar_arch.py`: beide Seiten liefern
nur Rohvorhersagen, Filter und Aggregation liegen einmal darüber. 5 628 006
gepaarte Zeilen (21 Zielstationen × ~1 440 Läufe × 96 Leads × 2 Zielgrößen),
Validierungsjahr 2024-08…2025-07. NWP-Baseline für alle Quellen identisch:
72.50 (GHI) / 41.29 (DHI) W/m² — der Beleg, dass dieselbe Stichprobe gemessen
wird.

| Modell | GHI RMSE | DHI RMSE |
|---|---|---|
| **TFT** | **65.31** | **36.53** |
| DCRNN `idw_alt` | 65.56 | 37.53 |
| DCRNN `nomeas` | 66.32 | 38.14 |
| DCRNN `nograph` | 66.39 | 38.18 |
| DCRNN `a` | 66.55 | 37.65 |
| DCRNN `base` | 66.66 | 38.36 |
| DCRNN `nwp_hist` | 66.67 | 37.22 |

**Bei GHI sind die Architekturen nicht unterscheidbar.** TFT und `idw_alt`
trennen 0.24 W/m² im Stationsmittel; gepaart über die 21 Stationen liegt der
Median auf der DCRNN-Seite (−0.47 W/m², an 71 % der Stationen besser),
p_holm = 1.0. Bei DHI liegt der TFT vorn, nach Holm-Korrektur aber knapp nicht
signifikant (gegen `nwp_hist` +0.76 W/m², p_holm = 0.054).

**Die Leiter trägt nicht, wo sie sollte.** Die beiden als tragend angelegten
Differenzen sind null: `a − nomeas` (Wert der Nachbarmessungen) p_holm = 1.0
(GHI) / 0.17 (DHI), `nomeas − nograph` (Geometrie- und Kontextkanal)
p_holm = 1.0 in beiden Zielgrößen. Signifikant sind dafür zwei Sprossen, die
beide an der NWP-Aggregation hängen und sich je eine Zielgröße teilen:

* `a` gegen `base` bei **DHI**: −0.68 W/m² für die GATv2-Attention über
  NWP-Knoten, p_holm = 0.0037 — bei GHI nichts (p_holm = 1.0).
* `a` gegen `idw_alt` bei **GHI**: +1.27 W/m² für die Distanzgewichtung mit
  Höhenkorrektur, p_holm = 0.039 — bei DHI nichts (p_holm = 1.0).

**Vorbehalt, der über die HPO hinausgeht:** je Arm gibt es nur **einen** Lauf.
Die Solar-Ablationen vom August liefen mit 2–4 Wiederholungen, die
Lauf-zu-Lauf-Streuung lag dort bei rund 0.45 W/m² — in der Größenordnung der
Armunterschiede hier. Der Wilcoxon-Test ist über die 21 Stationen gepaart, er
trennt also Stationsrauschen ab, **nicht** Seed-Rauschen: ein Seed-Effekt, der
alle Stationen gleich trifft, sähe genauso aus. Die beiden signifikanten
Sprossen sind damit nicht gegen Wiederholungen abgesichert; die Nullbefunde
sind es eher (ein Nulleffekt wird durch Wiederholungen selten größer). Zwei
weitere Seeds je Arm kosten rund eine Stunde auf vier GPUs.

Tabellen: `data/test_results/solar_arch_v5_{metriken,je_station,wilcoxon}.csv`.

Die TFT-Zahl hier ist **nicht** die aus der Tabelle oben (63.56): die ist über
Fold 1–3 gemittelt, diese ist Fold 1 auf der gepaarten Laufmenge. Und es gibt
weiterhin keine Solar-HPO für das DCRNN (§6) — der Vergleich bleibt vorläufig.

**Befund aus der Static-Ablation:** `dist_coast`, `svf` und `horizon_solar`
bringen nichts. Das Screening gegen den per-Station-RMSE hatte das
vorhergesagt — nach Herausrechnen von `altitude` bleibt von allen
Topo-Größen partiell |r| ≤ 0.23, und `slope`/`aspect` liegen bei 0.00, genau
wie die Physik es für einen waagerecht liegenden Pyranometer vorhersagt.
Es bleibt bei `altitude`, `latitude`, `longitude`.

## 3. Der 30-min-Versatz — erledigt, aber der Testaufbau bleibt

> **Stand 15.09.2026:** die sechs `v4`-Arme waren davon betroffen und sind
> durch `v5` ersetzt. Der Abschnitt bleibt stehen, weil der Fehler die Art von
> Fehler ist, die dieser Pfad wiederholt produziert — und weil der Test, der
> ihn findet, jetzt in `scripts/eval_solar_arch.py` fest eingebaut ist.

Die sechs DCRNN-Arme (`results/solar_dcrnn_*v4*.pkl`, Modelle unter
`models/solar_dcrnn_*_v4_*.pt`) waren **mit einem um 30 Minuten verschobenen
NWP-Kanal trainiert**. Vier Prüfagenten haben das am 14./15.09. gefunden
und belegt; behoben in `dc8d295`.

Ursache: `t_run_abs = ts_lookup[t_run] + 1`. Für ICON-D2 **ML** (Wind) ist das
richtig — dort verwirft der Loader `forecasttime == 0` und setzt
`lead_idx = ft - 1`. Für **SL** (Solar) bildet `solar_preprocessing`
`acc_bin = ceil(ft/freq_h) - 1`; Lead 0 trägt die Akkumulation über
`[t_run, t_run+freq)` und ist linksbündig auf `t_run`.

Verschiebungstest, Station 00183, `ghi_nwp` gegen die Messung, 415 384 Paare:

| Offset | −60 min | −30 min | **0** | +30 min | +60 min |
|---|---|---|---|---|---|
| RMSE W/m² | 88.51 | 73.39 | **67.53** | 73.57 | 88.66 |

Der alte Code fuhr +30 min, also **9.0 % RMSE** über alle Läufe.

**Das betrifft das Training, nicht nur die Auswertung** — die sechs Arme
mussten neu trainiert werden, erneutes Auswerten reichte nicht. Erledigt am
15.09.2026 (`v5`, 08:52–09:52 auf l2, je 41–56 Epochen bis Early Stopping).

`scripts/eval_solar_arch.py` fährt denselben Test jetzt bei jeder Auswertung:
beide Seiten über ±2 Schritte gegen die Messreihe, Abbruch wenn das
RMSE-Minimum nicht bei 0 liegt. Gegen die alten `v4`-Parquets gehalten findet
er deren +30 min selbständig wieder — der Test ist also scharf, nicht
dekorativ. Die TFT-Seite lässt sich dabei nicht über `gt` prüfen (das entsteht
dort erst durch Nachschlagen in der Zielreihe, ein Vergleich gegen dieselbe
Reihe wäre zirkulär), sondern läuft über ihr Residuum gegen die
`nwp_ref`-Spalte des DCRNN: 0.0083 W/m² bei Offset 0 gegen 48 W/m² bei ±1
Schritt.

## 4. Was am 14./15.09. repariert wurde

| Commit | Inhalt |
|---|---|
| (15.09., s. §4.1) | `run_time` bei Solar einen Schritt zu früh, `eval.exclude_imputed` im DCRNN-Generator |
| `dc8d295` | Lead-0-Semantik (`shared/resolution.lead0_offset`), `freq_h` an drei Stellen tot, unbekanntes `ist_tag` galt als Nacht |
| `0ffb797` | `exclude_imputed` im GNN-Pfad, `valid_time` in Schritten, NaN-Filter der Auswertung, `hpo_dcrnn` nachgezogen, `target`-Spalte in drei Berichten |
| `21d76fc`, `935dbca`, `88eae9d` | Lead-Zahl nicht auf 48 festnageln (Sampler, `build_eval_batch`, Reshape) |
| `7b0201a` | akkumulierte ECMWF-Felder um ein Lead-Intervall versetzen |
| `293e6f0` | ECMWF als zweite NWP-Quelle im GNN-Pfad |
| `fc767e9` | `target_transform: nwp_residual` im GNN-Pfad |
| `7c88dbe` | Sonnengeometrie und Clear-Sky als Stationskanal |
| `ec3a1eb` | Multi-Target im DCRNN-Pfad |
| `1c4318c` | Solar-Lückenfüllung angebunden, imputierte Ziele aus der Auswertung |

Die Wind-Regression ist über alle Commits belegt: `station_df`/`raw_df`
byte-identisch, Laufpaarlisten auf echten Winddaten elementweise gleich,
Decoder-Forward und `state_dict` unverändert, 566 Wind-Configs geprüft.

### 4.1 Folgefehler von `dc8d295`: `run_time` bei Solar

`evaluation.py` bildete `run_ts = timestamps[t_run_abs - 1]`. Für Wind ist das
richtig (Lead 0 liegt eine Stunde nach dem Lauf, `lead0_offset` = 1); seit der
Lead-0-Reparatur ist der Offset bei Solar 0, und damit zeigte `run_ts` auf
einen Schritt **vor** den ICON-Lauf. In den Roh-Parquets stand `run_time`
also auf 05:30 statt 06:00 — `gt` und `valid_time` waren richtig, weil beide
denselben Versatz trugen und er sich heraushob.

Gefunden beim Aufbau der gemeinsamen Auswertung: ein Join TFT gegen DCRNN über
`(station_id, run_time, horizon)` hätte **kein einziges Paar** gefunden.
Behoben über einen `lead0_offset`-Parameter an `evaluate()` mit Default 1.
Wind-Neutralität: bei Offset 1 sind `run_time` und `valid_time` über ein reales
Jahresraster elementweise identisch zur alten Formel. Am `v5`-Parquet belegt:
Laufstunden 6/9/12/15 bei Minute 0, `valid_time = run_time + (horizon−1)·30 min`,
horizon 1 = Laufzeitpunkt — dieselbe Konvention wie beim TFT.

**Die Masken beider Pfade stimmen überein.** Das intern gefilterte
`v5`-Parquet hat 40 520 Zeilen weniger als das ungefilterte `v4`-Parquet —
exakt die Zahl, die der unabhängig aus der Rohmessung rekonstruierte Filter in
`eval_solar_arch.py` entfernt. Der Anteil echter Messungen ist auf beiden Wegen
93.76 %.

### 4.2 Der ECMWF-9999-Bug (15.09.2026)

Aufgefallen als 42 560 NaN-Vorhersagen im `v5`-Lauf, aus 16 Läufen Ende
August 2024. Die Kette von der Ursache zur Wirkung:

1. `/mnt/nas/ecmwf/write_db.py:274` verwarf beim GRIB-Import **jeden Wert, der
   exakt 9999 ist**, als Fehlwert: `bad = data == 9999`. Das war falsch — die
   echte Fehlstellenbehandlung passiert eine Zeile darüber über `vals.mask`,
   und 9999 ist bloß der eccodes-**Default** für den `missingValue`-Platzhalter.
   Belegt: `bitmapPresent = 0`, `numberOfMissing = 0`, und mit einem anderen
   Platzhalter (`grib_get -m -777`) kommt weiterhin 9999 zurück. Bei
   akkumulierter Strahlung (J/m²) liegt 9999 mitten im Wertebereich — die
   Nachricht reicht von 0 bis 55 812, die Nachbargitterpunkte tragen dort
   8753, 8803 und 12 737.
2. Ein genullter Akkumulationswert reißt über die Dekumulation
   (`x[t] = (x_acc[t] − x_acc[t−1]) / 3600`) **zwei** stündliche Werte auf.
3. `exclude_run_pairs_with_ecmwf_nan` prüft mit `any(axis=(1,2))` — ein NaN
   macht den ganzen Zeitschritt für alle 759 Gitterpunkte ungültig.
4. Über das ±48-h-Fenster fallen daraus 16 Laufpaare.
5. `get_test_results_dcrnn.py` spiegelte den Filter nicht, das Modell bekam die
   Läufe also trotzdem — mit NaN im ECMWF-Kanal.

Umfang: **8 genullte Zellen** (= 16 stündliche Werte) in 73.9 Mio
Parquet-Zeilen, 7 von 759 Gitterpunkten, ab 2024-05. Immer der erste
Sonnenaufgangsschritt, wo die Akkumulation den Bereich um 10 000 J/m²
durchläuft. **Der Wind-Pfad war nie betroffen** — dort kann keine Größe 9999
annehmen (Temperatur 270–304 K, Wind ±19 m/s, Dichte < 0.02).

Behoben: `write_db.py` korrigiert (Backup `write_db.py.bak_20260915`), die 16
Zellen aus den GRIB-Dateien in `ecmwf_solar` nachgetragen und in den Parquets
gesetzt, der Filter in `get_test_results_dcrnn.py` gespiegelt. Nachzählung über
alle 759 Gitterpunkte: 0 verbleibende NaN ab 2024-05. Die Auswertung aller
sechs Arme wurde danach neu gefahren — 1460 statt 1444 Läufe, keine
NaN-Vorhersage mehr, die Zahlen in §2.1 ändern sich erst in der dritten
Nachkommastelle.

**Nicht zu verwechseln** mit dem großen NaN-Block Juli 2023 – März 2024: dort
fehlt ausschließlich `ssrdc` (Clear-Sky-GHI), und zwar in allen Leads aller 550
Läufe. Das ist kein Defekt, sondern Bestandserweiterung — das Feld existiert in
den GRIB-Rohdaten vor April 2024 gar nicht (`aug23.grib`: 0 Nachrichten,
`may24.grib`: 3596) und wird in keiner Ableitung verwendet
(`utils/solar_ecmwf.py:344` begründet, warum `ecmwf_kt` bewusst gegen
`ghi_clearsky` statt gegen `ssrdc` rechnet). Es hat deshalb nie ein Laufpaar
gekostet. Wer `ssrdc` künftig als Feature will, hat dafür erst ab April 2024
Daten.

## 5. Was noch zu tun ist

### 5.1 Erledigt am 15.09.2026

Die drei blockierenden Punkte sind abgearbeitet:

1. **`eval.exclude_imputed: true`** steht in `scripts/make_solar_dcrnn_configs.py`,
   die sechs Configs sind neu erzeugt (Diff: genau zwei Zeilen je Datei).
2. **Die sechs Arme sind neu trainiert** (`v5`) und ausgewertet:
   `scripts/run_solar_dcrnn_arms.sh` startet das Training, die GPU hängt am
   Armnamen statt an der Aufrufreihenfolge; `scripts/run_solar_dcrnn_eval.sh`
   schreibt die Rohvorhersagen nach
   `data/raw_preds/solar_dcrnn_v5_<arm>_raw.parquet`.
3. **Die gemeinsame Auswertung** liegt als `scripts/eval_solar_arch.py` vor,
   Ergebnis in §2.1.

Was dabei über die Aufgabe hinaus anfiel und ebenfalls behoben ist: der
`run_time`-Folgefehler (§4.1) und der ECMWF-9999-Bug in der Ingest-Pipeline
(§4.2).

### 5.1.1 Solar-HPO für den TFT — ABGESCHLOSSEN am 17.09.2026, 12:47

**Status: beendet bei 98 abgeschlossenen Trials (Ziel waren 150), bewusst
abgebrochen wegen Konvergenz.** Ergebnis und Arbeitsauftrag stehen in §5.1.2.

`configs/solar_tft/config_solar_tft_hpo.yaml`, Studie
`cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo`, zuletzt 18 Worker über l2/l1/ws
(`scripts/run_solar_hpo.sh`). Räumliche 3-Fold-CV über `configs/solar_folds.yaml`,
Training bis `val_start` 2024-08-01, Validierung bis `test_start` 2025-08-01 —
das Testjahr blieb unberührt.

Endstand: 98 COMPLETE, 198 PRUNED, 52 FAIL (ausschließlich beim Umkonfigurieren
abgeräumte Trials, kein einziger inhaltlicher Fehlschlag). Volle Trial-Tabelle in
`archiv/solar_tft_hpo_trials_final.csv`, bester Trial zusätzlich als
`archiv/solar_tft_hpo_bester_trial.json`.

**Warum abgebrochen — die Suche war austrainiert.** Der beste Wert stand seit
Trial 111 (16.09., 23:59) unverändert bei 51.7188; 56 weitere abgeschlossene
Trials brachten keine Verbesserung. Die Größenordnungen erklären, warum:

| Größe | Wert |
|---|---|
| bester Trial | 51.719 W/m² |
| Median aller 97 ausgewerteten Trials | 51.955 W/m² |
| Abstand bester ↔ Median | **0.236 W/m²** (0.45 % des RMSE) |
| Spannweite über alle Trials | 1.049 W/m² |
| Vorsprung Top-10 vor dem Rest | 0.212 W/m² |
| Lauf-zu-Lauf-Streuung derselben Konfiguration (IDW-Test, §5.1.1 unten) | ±0.13 … ±0.21 W/m² |

Der Vorsprung der besten Trials entspricht damit etwa **einer Standardabweichung
der bloßen Wiederholung** — der „beste" Trial ist vom soliden Mittelfeld
statistisch kaum zu unterscheiden.

> **Korrektur vom 18.09.2026.** Hier stand zuerst, das Modell sei
> „datenlimitiert". Das ist nicht haltbar: 62 Stationen × 4 Läufe × 96 Leads ×
> 730 Tage sind rund 17 Mio Zielwerte für 508 134 Parameter, und die Stationen
> sind dabei **nicht** redundant. Die Korrelation des NWP-Fehlers zwischen zwei
> Stationen liegt bei 0.156 unter 50 km, 0.107 bei 100–200 km und im Mittel über
> alle 210 Paare bei **0.051** — das Netz liefert also fast 62 unabhängige
> Beobachtungen je Zeitpunkt. Richtig ist **informationslimitiert**: der
> NWP-Fehler ist aus den verfügbaren Eingangsgrößen überwiegend nicht
> rekonstruierbar. Der lernbare Anteil ist der bedingte Bias (§5.1.4), und der
> ist mit wenigen tausend Beispielen ausgeschöpft; der Rest ist irreduzibel —
> ICON-D2 hat die Wolke am falschen Ort, und keine Nachbearbeitung derselben
> Felder schiebt sie dorthin.

**Die Suche arbeitet unter der Rauschgrenze — in beiden Studien.** Das ist der
allgemeinere Befund und gilt über diesen Arm hinaus:

| | mit Historie (98 Trials) | ohne Historie (50 Trials) |
|---|---|---|
| Streuung über alle Trials (σ) | 0.181 | 0.266 |
| Abstand bester ↔ Median | 0.236 | 0.226 |
| Verbesserung in der zweiten Hälfte | **0.000** | — |

Zum Vergleich: dieselbe Konfiguration zweimal trainiert streut um ±0.13 … 0.21
(IDW-Test unten). Die Streuung *zwischen* Trials liegt damit in der Größenordnung
der Streuung *derselben* Konfiguration — was die Suche als besseren
Hyperparametersatz ausweist, ist zum großen Teil Trainingsrauschen. In der ersten
Studie stand der beste Wert bei Trial 38; die folgenden 60 Trials brachten exakt
null Verbesserung. **Für künftige Arme heißt das: 40–50 Trials genügen, und die
Wahl „bester statt typischer Trial" ist rund 0.23 W/m² wert — weniger als der
Unterschied, den die Featurewahl macht.**

**Keine Obergrenze des Suchraums bindet.** Geprüft über die zehn besten Trials:

* `hidden_dim` — Optimum bei 16–46, Median 27.5 bei einer Obergrenze von 128.
  Größere Werte sind sogar systematisch schlechter (Median 51.92 im Bereich
  8–32 gegen 52.15 im Bereich 96–128, Korrelation +0.39). Die Senkung von 256
  auf 128 am 16.09. hat also nichts abgeschnitten, sondern eine bereits
  schlechte Region gekappt.
* `batch_size` — **drückt gegen die UNTERE Schranke**: alle zehn besten liegen
  zwischen 33 und 75 bei einer unteren Grenze von 32, und der Effekt ist monoton
  (Median 51.94 bei 32–64 gegen 52.19 bei 256–512, Korrelation +0.62).
  Ausgerechnet der wichtigste Parameter (39 % der erklärten Varianz laut fANOVA,
  vor `lr` mit 27 % und `dropout` mit 17 %). Eine Öffnung nach unten (16 oder 8)
  wäre der einzige sachlich begründete Eingriff — der erwartbare Gewinn liegt
  aber unter der Wiederholungsstreuung und kostet Trainingszeit. **Bewusst
  nicht gemacht.**
* `num_lstm_layers` — alle zehn besten haben 1, also den kleinstmöglichen Wert.
  Nicht erweiterbar; die Aussage lautet schlicht: mehr LSTM-Schichten schaden.

Zur Vorsicht bei der Fold-Streuung: die beträgt 1.02 W/m², ist aber
**systematisch**, nicht zufällig (Fold 1 bei 53.14, Fold 2 bei 51.21, Fold 3 bei
51.65 — verschiedene Zielstationen). Da jeder Trial dieselben drei Folds sieht,
hebt sie sich im Vergleich zwischen Trials auf und taugt nicht als Rauschmaß.

**Der erste Anlauf vom Vormittag wurde verworfen und die Studie neu angelegt**
(23 Trials gesichert in `archiv/optuna_…_vor_reset_20260916.csv`). Drei Ursachen,
alle behoben in `a9ab2c6` und dem Folgecommit:

* **GPU-OOM.** Der Suchraum reichte bis `batch_size` 1024 und `hidden_dim` 256;
  ein Trial am oberen Ende belegte über 50 GB. Das sprengt die RTX 4090 auf ws
  (24 GB), und der CUDA-OOM beendet nicht nur den Trial, sondern den ganzen
  Worker — sein Trial bleibt als Zombie auf RUNNING stehen und blockiert den
  MedianPruner (`n_startup_trials: 10` zählt COMPLETE). Jetzt 512 / 128.
* **Doppelstart.** Ein zweiter Aufruf des Startskripts legte einen kompletten
  zweiten Workersatz neben den laufenden: zwei Trainings je GPU, worauf auch die
  A100 mit 80 GB an OOM starb. `run_solar_hpo.sh` überspringt belegte Slots jetzt.
* **Cache-Explosion.** Siehe unten.

Beim Neustart die Worker **gestaffelt** starten, wenn der Cache eines Hosts leer
ist: `DataCache.save_preprocessed_data` schreibt ohne Lock und ohne atomares
`os.replace` (den flock hat nur `GNNCache`). Mehrere Worker, die gleichzeitig mit
leerem Cache anlaufen, bauen denselben Eintrag mehrfach parallel, und ein Leser
kann eine halb geschriebene `prepared.pkl` sehen — ein plausiblerer Auslöser der
"empty split"-Fehlschläge als das Cache-Evicting, das zuerst verdächtigt wurde.
Erst einen Worker je Host, dann die übrigen.

Gemessene Laufzeit **mit warmem Cache** (16.09.2026, l2): ein ganzer Trial rund
23 min — 40 s Vorlauf für alle drei Fold-Einträge zusammen, dann 6–9 min je Fold
bei 19–29 Epochen bis Early Stopping. Die früher notierten "36 min je Fold,
davon 15 min Vorlauf" galten für Läufe, die den Cache noch bauen mussten; sie
beschreiben nicht den Dauerbetrieb. Fold 1 kostet je Trial ~27 s/Epoche, Fold 2
und 3 nur 12–15 s — der Unterschied ist der `torch.compile`-Aufwand, der einmal
je Trial anfällt. `max_epochs_per_trial: 100` ist eine nie erreichte Obergrenze;
`hpo_tft_bc.py:511` meldet ohnehin den besten Epochenwert an Optuna. Bei 150
Trials auf zehn Workern überschlägig 5–6 h, mit MedianPruner darunter.

GPU-Auslastung schwankt stark mit der Trial-Größe und ist kein Fehlerzeichen:
gemessen 25 % bei `batch=64, hidden=44` gegen 91 % bei `batch=382, hidden=111`.
Ein kleiner TFT lastet eine A100 nicht aus; auf A6000 und 4090 liegt der Wert
entsprechend höher.

**`u_10m` fest aufgenommen, `hpo.optional_features` leer.** Die drei Kandidaten
(`relhum_2m`, `t_2m`, `u_10m`) standen zunächst als binäre Hyperparameter im
Suchraum. Das kostete das Achtfache an Cache: die Flags ändern
`params.icond2_features`, das in `data_cache._get_config_hash` eingeht, also
bekam jede der 8 Kombinationen mal 3 spatialer Folds einen eigenen Eintrag zu
~16.5 GB — rund 400 GB je Host bei einem Budget von 500 GB. `enforce_cache_budget`
lief daraufhin im Dauerbetrieb (60 Evictions an einem Vormittag) und verwarf
Einträge, die ein anderer Worker kurz darauf neu bauen musste. Für 0.06–0.16 %
Restvarianz ist das nicht zu rechtfertigen. Aufgenommen ist `u_10m` als
stärkster Kandidat (DHI 0.161 %), verworfen `relhum_2m` (0.055 %) und `t_2m`
(< 0.01 %). Bedarf jetzt 3 Einträge à ~16.5 GB je Host, Budget 150 GB (l1 600 GB,
weil dort noch ~420 GB Wind-Cache im selben Manifest liegen — das Budget gilt
für das Manifest als Ganzes, nicht je Studie).

Zwei Fallstricke beim Cache-Schlüssel, beide am 16.09. aufgelaufen:

* `params.next_n_grid_ecmwf` muss **explizit** in der Config stehen.
  `hpo_tft_bc._range` fällt ohne `hpo`-Range auf `params` zurück und der Trial
  schreibt den Wert nach `config['params']`, wo der Hash ihn liest. Fehlt der
  Schlüssel, hasht die Config `None` und der Trial `0` — zwei Schlüssel für
  dieselben Daten.
* Der `model_name` im Hash ist **`tft`**, nicht `tft-bc`: `--model` hat den
  Default `tft`, das `-bc` in `cl_m-tft-bc_…` ist Teil des Studien-Namensmusters
  (`hpo_tft_bc.py:307`). Wer den Cache-Schlüssel von Hand nachrechnet, trifft
  mit `tft-bc` daneben.

**Wind-Cache auf l2 ist weg (16.09.2026).** Beim Aufräumen des Solar-Caches
wurde auf l2 auch der Cache der abgeschlossenen Wind-Studien `wind_tft_sp_base`
und `wind_tft_sp_hist` gelöscht (~310 GB) — `hpo_tft_bc.py` führt beide Use Cases
im selben `.tft_bc_cache_manifest.json`, die Löschgrenze lag am Manifest statt am
`data.use_case`. Die Optuna-Studien selbst sind unberührt, verloren ist nur
vorprozessierter Cache: ein Wind-Retrain oder Testlauf auf l2 rechnet sein
Preprocessing einmal neu. **Auf l1 liegt der Wind-Cache noch** (~420 GB,
`/mnt/nvme2/data_cache`) — wer die Wind-Kette nochmal anfasst, tut das dort
günstiger.

Grundlage der Featurewahl ist ein Leave-one-out-Screening auf den
Fold-1-Zielstationen im Validierungsjahr; die Begründung je Feature steht im Kopf
der Config. Kurzfassung der Befunde, die gegen weitere Features sprechen:

* Der Featuresatz ist gesättigt: die genutzten Features erklären bei GHI 85.0 %
  der Restvarianz gegen ICON, kein einzelnes trägt mehr als 0.21 % bei, und die
  Strahlungstripel (ghi/dhi/bhi, je ICON und ECMWF) sind exakt linear abhängig.
* Zur Windfrage: der **Betrag** trägt weniger als die **Richtung**
  (`wind_speed` 0.063 % gegen `wind_dir_sin` 0.155 % bei DHI). `u_10m` enthält
  beides und braucht kein abgeleitetes Feature.
* `td_2m` ist über die Magnus-Formel exakt aus `t_2m` und `relhum_2m`
  berechenbar — keine dritte Information.

**`next_n_grid_points` bleibt fest auf 1.** Zwei unabhängige Tests:

* Gittertest (ohne Training): die ICON-Strahlungsprognose ist räumlich extrem
  glatt — Korrelation zum Stationspunkt 0.998 auf 5 km und 0.982 auf 40 km,
  kein Punkt trägt nach Herausrechnen des Stationspunktes etwas zur Korrektur
  bei (alle |r| < 0.012). Das erklärt `ab_grid4` weitergehend als bisher: nicht
  die Nähe der vier Punkte ist der Grund, sondern dass ICON-D2 die Bewölkung
  auf dieser Skala nicht differenziert auflöst.
* IDW-Test (mit Training, `params.nwp_aggregation: idw`, neu in `utils/solar.py`):
  drei Varianten à drei Wiederholungen. `nearest` 53.331 ± 0.199,
  `idw4` 53.316 ± 0.126, `idw9` 53.258 ± 0.209 — die Unterschiede liegen unter
  der Lauf-zu-Lauf-Streuung. Gepaart über die Stationen zeigt `idw9` bei GHI
  einen schwachen Hinweis (Median −0.165 W/m², an 76 % der Stationen besser,
  p = 0.076), bei DHI dreht das Vorzeichen. Nicht in den Suchraum aufgenommen;
  die Option bleibt im Code, Default `nearest`.

**Offener Punkt: Station 05792 fällt aus dem CL-Pfad.** Seit `dc8d295`
(`ist_tag`-Fix) verwirft das Preprocessing die Alpenstation vollständig
("keine Daten übrig") — auch mit der unveränderten Arm-Config, mit der sie am
14.09. noch durchlief. Ursache ist die Korrektur selbst: wo vorher Nachtnullen
erfunden wurden, bleiben die Lücken offen, und bei dieser ohnehin dünnen
Station (18 % echte Messwerte im Testjahr) kippt das über `dropna()` den
gesamten Bestand. Die HPO läuft deshalb auf **61 statt 62 Pool-Stationen**, für
alle Trials gleich. Der **GNN-Pfad verliert sie nicht** — die `v5`-Parquets
haben 21 Zielstationen. Ein späterer Vergleich der HPO-Ergebnisse gegen die
DCRNN-Arme steht damit auf 20 gegen 21 Stationen; `eval_solar_arch.py` fängt
das über die gemeinsame Menge ab, es sollte aber bewusst entschieden werden.

**Offen bleibt:**

* **Solar-HPO für das DCRNN.** Ohne sie bleibt jeder Architekturvergleich
  vorläufig (§6): die DCRNN-Parameter stammen aus einer Stunden-Wind-Config,
  die TFT-Defaults immerhin aus einer Wind-HPO. Der Befund aus §2.1, dass die
  NWP-Aggregation die einzige tragende Sprosse ist, wäre der erste Kandidat
  für den Suchraum.
* **Die Folds 2 und 3** für das DCRNN — bisher läuft die Leiter nur auf Fold 1.
* **`kt_nwp`** fehlt weiterhin als einziges der 13 TFT-Features im
  DCRNN-Featuresatz.
* **Wiederholungsläufe je Arm** — bisher ein Seed je Arm, s. den Vorbehalt am
  Ende von §2.1.

### 5.1.2 Retrain und Schlussmessung — Folds erledigt am 17.09.2026, Testjahr läuft

**Die drei Fold-Modelle stehen und sind ausgewertet.** Trainiert mit den
Hyperparametern aus Trial 111 (`--hpo-study`, nicht abgetippt), je Fold auf den
41 bzw. 42 Trainingsstationen bis `val_start` 2024-08-01, Early Stopping auf den
Zielstationen desselben Folds im Validierungsjahr — dieselbe CV-Achse wie die HPO.

| Fold | GHI RMSE | Skill_NWP | DHI RMSE | Skill_NWP | Stationen | val_rmse | Epochen |
|---|---|---|---|---|---|---|---|
| 1 | 65.72 | 0.098 | 37.53 | 0.105 | 21 | 53.160 | 12 |
| 2 | 63.19 | 0.109 | 36.25 | 0.107 | 21 | 51.126 | 15 |
| 3 | 63.56 | 0.107 | 36.30 | 0.117 | 20 | 51.663 | 17 |
| **Mittel** | **64.16** | **0.105** | **36.69** | **0.110** | | | |

Dateien: `models/train_tft_bc_m-tft_c-solar_tft_fold<N>.pt`,
`data/test_results/tft_solar_tft_fold<N>.csv`,
`data/raw_preds/tft_solar_tft_fold<N>_raw.parquet` (Schema wie DCRNN, mit
`target`-Spalte).

**Der HPO-Retrain ist nicht besser als Arm A** (§2: GHI 63.56 / 0.108, DHI 36.08
/ 0.114). Fold 1 liegt sogar deutlich darüber. Das ist kein Widerspruch, sondern
die Bestätigung des HPO-Befunds aus §5.1.1: der Abstand zwischen bestem Trial und
Median lag bei 0.236 W/m², also in der Größenordnung der bloßen
Wiederholungsstreuung. An dieser Stelle ist das Modell datenlimitiert, nicht
hyperparameterlimitiert — **von der Schlussmessung ist entsprechend kein Sprung
zu erwarten.**

Zwei Belege, dass die Kette sauber sitzt:

* Die val_rmse der Retrains treffen die Fold-Werte der HPO auf zwei
  Nachkommastellen (53.160/51.126/51.663 gegen 53.14/51.21/51.65).
* Verschiebungstest gegen die DCRNN-Seite über 5 627 915 gepaarte Zeilen: die
  `nwp_ref`-Spalten beider Pfade stimmen bei Versatz 0 auf **0.0000 W/m²**
  überein, bei ±1 Schritt liegen 34.65 W/m² dazwischen. Zeitachse, Basisspalte
  und Gitterpunkt sind damit identisch; `horizon 1 = run_time` auf beiden Seiten.

#### Was dafür zu reparieren war

**Die vier Configs wären nach dem vollständigen Preprocessing abgebrochen.** Sie
erbten aus `configs/solar_final/config_solar_final_lag.yaml` `hpo.kfolds: 12`,
`train_cl_tft_bc.py` verlangt aber genau einen Fold; `cv_mode` setzte der
Generator nur für die HPO-Config. Behoben in `scripts/make_solar_tft_configs.py`
(Commit `d024146`) — die Configs sind **generiert und nicht von Hand zu pflegen**:

* `fold{1,2,3}`: `cv_mode: spatial`, `kfolds: 1`, Zeitachse der HPO-Studie
  (`val_start` 2024-08-01, `test_start` 2025-08-01). **`train_end` muss fehlen** —
  es begrenzt `df_train` (`preprocessing.py:412`), und der spatial-Pfad schneidet
  sein Val-Fenster genau daraus heraus; mit `train_end` bliebe es leer.
* `testyear`: gleiche Bauart, ein Jahr weiter — `val_start` 2025-08-01,
  `test_start` 2026-08-01. Training sind die Poolstationen davor (beide bisherigen
  Jahre), Validierung die 21 Teststationen **im Testjahr**, also die
  Auswertungsdaten selbst. Ausgewertet wird deshalb ebenfalls mit
  `--eval-split val`.

  **Achtung, hier lag der alte Handoff falsch.** Die frühere Stolperfalle 3
  behauptete, die 21 Stationen dienten „im Trainingszeitraum als Validierungsset
  … zeitlich getrennt". Das widerspricht
  [station_splits_solar.md](station_splits_solar.md) §6: dort wird das
  Auswertungsset ausdrücklich als Validierungsset übergeben, die Epoche also auf
  denselben Stationen **und demselben Zeitraum** gewählt, auf denen berichtet
  wird — Optimismus rund ein Prozent, am 18.08.2026 als vernachlässigbar
  entschieden. Die erste Fassung der Config folgte dem falschen Text, der Lauf
  vom 17.09. 15:07 wurde deshalb verworfen und neu gestartet. Ein Val-Chunk im
  Trainingszeitraum wäre die strengere Variante, wiche aber von Arm A und den
  Fold-Läufen ab und machte die Zahlen untereinander unvergleichbar.

**`hpo.val_split` ist ersatzlos entfallen** (Commit `fefd8f6`, 489 Configs und
`utils/hpo.py`/`utils/data_cache.py`/`hpo_fl.py`). Getrennt wird nach Datum: im
temporalen Pfad schneidet `_replace_val_with_val_files` die `val_files`-Stationen
zeitlich zu, im räumlichen trennt `val_start`. `val_split` schnitt daneben nur
noch Trainingsdaten ab, die anschließend verworfen wurden — die (damals noch
temporal aufgesetzte) Schlussmessung lief damit auf 162 993 statt 171 572
Fenstern, also ohne die letzten fünf Wochen vor dem Testjahr. **`kfolds: 1` ohne `val_files` bricht jetzt ab**,
das trifft `train_cl_tft_bc.py --test-mode` (leert `val_files`) und damit die
beiden Wind-Testyear-Configs; sie vermerken es in ihrem Kopf.

**Cache-Guard.** `data_cache.pruefe_cache_config` hält bei jedem Cache-Treffer 23
Schlüssel gegen die Config des Erzeugerlaufs, die das Ergebnis verändern, ohne in
`_get_config_hash` einzugehen (Zeitgrenzen, `val_start`, Skalierungsflags,
Imputationspfade, `t_0`, `cv_mode`) — Abbruch mit beiden Werten statt stiller
Weiterverwendung. Anlass: der korrigierte Testyear-Lauf bekam denselben Eintrag
mit dem alten 5-%-Schnitt zurück, ohne dass etwas gewarnt hätte. Den Hash zu
erweitern schied aus — er ist ein md5 über das ganze `hash_data`-Dict, jeder neue
Schlüssel entwertet **jede** bestehende `cache_id` und macht bereits trainierte
Modelle unauswertbar, weil `get_test_results_tft_bc.py` darüber `scaler_x`
zurückholt.

**`get_test_results_tft_bc.py` wertet jetzt je Zielgröße aus** — und hatte neben
dem bekannten Multi-Target-Punkt drei weitere, die für Solar still falsch
gerechnet hätten:

* `tools.get_y` lief mit `clip_negative=True`; bei `target_transform:
  nwp_residual` ist rund die Hälfte der Zielwerte negativ und wäre auf 0
  geschnitten worden.
* `valid_time` war auf die Wind-Konvention verdrahtet — jetzt über
  `shared.resolution.lead0_offset`, sonst läge das Parquet 30 min gegen die
  DCRNN-Seite versetzt (§3).
* `pers_ref` war **durchgängig NaN, auch im Wind-Pfad**: `reindex` mit einem
  `DatetimeIndex` trifft den MultiIndex nicht. Nachgeprüft an
  `data/raw_preds/retrain_tft_sp_base_fold1_raw.parquet` — 100 % NaN, `skill` in
  allen 51 Zeilen leer. Jetzt gefüllt (Bezugspunkt: ein Schritt vor
  Prognosestart, wie `homo_sampler`).

Dazu: `eval.exclude_imputed` elementweise über `<target>_observed`, `pred`/`gt`
im Parquet in W/m² zurückgerechnet über die abgezogene Basisspalte, und `n_values`
neben `n_samples` (Einzelwerte gegen Vorhersagefenster — was DCRNN `n_samples`
nennt, ist `n_values`).

#### Kommandos

```bash
# Fold-Retrain (erledigt)
frcst/bin/python train_cl_tft_bc.py -c configs/solar_tft/config_solar_tft_fold${N}.yaml \
    --hpo-study cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo \
    --gpu <G> --cache-dir /mnt/nvme2/data_cache

# Fold-Auswertung — --eval-split val ist PFLICHT, sonst misst sie im Testjahr
frcst/bin/python get_test_results_tft_bc.py -c configs/solar_tft/config_solar_tft_fold${N}.yaml \
    --hpo-study cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo \
    --model-tag train_tft_bc_m-tft_c-solar_tft_fold${N} \
    --raw-out-name tft_solar_tft_fold${N} --eval-split val \
    --cache-dir /mnt/nvme2/data_cache --gpu <G>
```

#### Schritt 2 — Schlussmessung auf dem Testjahr (erledigt, s. §5.1.3)

Training auf den 62 Poolstationen über beide bisherigen Jahre (alles vor
`val_start` 2025-08-01), gemessen auf den 21 zurückgehaltenen Teststationen im
dritten. Seit 17.09.2026, 15:55 auf l2, GPU 0 — der Lauf von 15:07 trug noch die
falsche Early-Stopping-Konstruktion (s. oben) und wurde samt Cache-Eintrag
verworfen.

```bash
frcst/bin/python train_cl_tft_bc.py -c configs/solar_tft/config_solar_tft_testyear.yaml \
    --hpo-study cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo \
    --gpu <G> --cache-dir /mnt/nvme2/data_cache

frcst/bin/python get_test_results_tft_bc.py -c configs/solar_tft/config_solar_tft_testyear.yaml \
    --hpo-study cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo \
    --model-tag train_tft_bc_m-tft_c-solar_tft_testyear \
    --raw-out-name tft_solar_tft_testyear --eval-split val --gpu <G>
```

Auch hier `--eval-split val`: das Auswertungsfenster ist `[val_start,
test_start)`, also das Testjahr. Ohne das Flag misst die Auswertung im Fenster
`[test_start, test_end]` — und das ist hier leer.

#### Stolperfallen, die weiter gelten

1. **`--test-mode` hier NICHT verwenden.** Das Flag mischt `val_files` in den
   Trainingspool; in `config_solar_tft_testyear.yaml` sind `val_files` und
   `test_files` **dieselben** 21 Teststationen, die Messung wäre wertlos. Seit der
   `val_split`-Entfernung bricht der Lauf in diesem Fall ohnehin ab.
2. **Dass `val_files == test_files` ist, ist Absicht** — und zwar im selben
   Zeitraum: Early Stopping läuft auf den Auswertungsdaten
   (`station_splits_solar.md` §6, Entscheidung Viktor 18.08.2026: rund ein Prozent
   Optimismus, bewusst akzeptiert). Dieselbe Konstruktion tragen die Fold-Läufe
   und Arm A; wer sie für einen Lauf ändert, macht dessen Zahlen mit allen
   übrigen unvergleichbar.
3. **`--hpo-study` immer explizit angeben**, sonst leiten beide Skripte den
   Studiennamen aus dem Config-Dateinamen ab und landen auf `…_solar_tft` statt
   `…_solar_tft_hpo`.
4. **Cache.** Die Retrain-Läufe haben eigene Einträge (~12–20 GB je Lauf, l2
   `/mnt/nvme2/data_cache`). Zwischen l1 und l2/ws sind sie nicht austauschbar,
   weil `DATA_ROOT` in den Schlüssel eingeht. Mehrere Läufe mit **leerem** Cache
   gleichzeitig auf einem Host vermeiden, solange sie denselben Eintrag bauen —
   `DataCache.save_preprocessed_data` schreibt ohne Lock; verschiedene Configs
   sind unproblematisch (die vier Läufe am 17.09. liefen parallel).
5. **Datenlage im Testjahr prüfen, bevor Zahlen interpretiert werden**: mittlerer
   Anteil echter Messwerte 0.85, Minimum 0.12, und `eval.exclude_imputed` nimmt
   den Rest heraus — je Station nachzählen (`n_values` in der CSV).
6. **Station 05792 fällt im CL-Pfad aus** (s. §5.1.1), die Läufe arbeiten auf 61
   statt 62 Poolstationen. Beim Vergleich gegen die DCRNN-Arme (21 Zielstationen)
   zu berücksichtigen; `eval_solar_arch.py` fängt es über die gemeinsame Menge ab.
7. **GPU-Wahl.** GPU 0 auf l2 ist oft fremdbelegt, auf l1 tragen 3 und 5–7
   dauerhaft Fremdlast. Vor dem Start `nvidia-smi`.

### 5.1.3 Schlussmessung auf dem Testjahr — erledigt am 17.09.2026, 18:39

Ein Modell mit den Hyperparametern aus Trial 111, trainiert auf allen 62
Poolstationen über beide vorangegangenen Jahre (alles vor `val_start`
2025-08-01, 171 807 Fenster), Early Stopping auf den 21 Teststationen im Testjahr
— 21 Epochen, bestes `val_rmse` 49.603. Gemessen wurde auf denselben 21
Stationen im Fenster 2025-08-01 … 2026-07-31, 2 156 607 bewertete Tagesschritte.

| | GHI RMSE | Skill_NWP | DHI RMSE | Skill_NWP |
|---|---|---|---|---|
| Folds, Validierungsjahr, 62 Stationen | 64.16 | 0.105 | 36.69 | 0.110 |
| **Testjahr, 21 Teststationen** | **61.53** | **0.119** | **33.89** | **0.141** |

**Die Schlussmessung fällt besser aus als die Fold-Läufe** — und zwar im Skill,
also unabhängig davon, dass das Testjahr andere absolute Fehlerniveaus hat. Das
passt zum Wind-Befund, dass von einem erweiterten Trainingsfenster nur die Arme
mit eigener Messhistorie profitieren: dieser hier ist einer.

**Alle Befunde des Validierungsjahres übertragen sich**, was die Kampagne
insgesamt trägt:

| GHI, Skill je Regime | Testjahr | Folds |
|---|---|---|
| bedeckt (kt < 0.3) | 0.041 | 0.006 |
| trüb (0.3–0.6) | 0.149 | 0.132 |
| heiter (0.6–0.85) | 0.206 | 0.206 |
| klar (kt > 0.85) | 0.105 | 0.101 |

21 von 21 Stationen liegen unter der ICON-D2-Referenz (Skill 0.085 … 0.181,
Median 0.119), und die Dreiteilung über die Vorlaufzeit ist dieselbe: 0.176 bei
Lead 0, 0.099 zwischen 1 und 6 h, 0.122 ab 36 h.

Dateien: `models/train_tft_bc_m-tft_c-solar_tft_testyear.pt`,
`data/test_results/tft_solar_tft_testyear.csv`,
`data/raw_preds/tft_solar_tft_testyear_raw.parquet`. Bericht:
`frcst/bin/python scripts/report_solar_folds.py --folds 0 --stem tft_solar_tft_testyear`.

### 5.1.4 Was die eigene Messhistorie beiträgt — und der Arm ohne sie

Die Solar-Configs führen beide Zielgrößen als `observed_features`, das Modell
sieht also 48 h eigene Vergangenheit; wegen `target_transform: nwp_residual`
genauer: 48 h NWP-Fehlerhistorie der Zielstation. Im Wind-Sprachgebrauch ist das
die **`hist`-Variante**. Gemessen am fertigen Modell
(`scripts/ablate_solar_observed.py`, 12 Stationen über alle drei Folds,
observed-Fenster über die Läufe permutiert):

| Lead | 0.0 h | 0.5 h | 1 h | 1.5–3 h | 3–6 h | 24–48 h |
|---|---|---|---|---|---|---|
| GHI, RMSE-Anstieg ohne Historie | **+29 %** | +9 % | +5 % | +2 % | +1 % | +0.3 % |
| DHI | **+28 %** | +12 % | +8 % | +4 % | +3 % | +1.0 % |

Die Gegenprobe stimmt: die Autokorrelation des Residuums zwischen dem letzten
Messzeitpunkt vor dem Lauf und dem Lead beträgt 0.55 (Lead 0), 0.33 (1 h),
0.12 (3 h) und ist ab 6 h weg. Über alle 96 Leads gewichtet bleiben +0.65 W/m²
(GHI) und +0.75 (DHI).

**Der Arm ohne Historie ist durchgerechnet** (`configs/solar_tft_nohist/`,
`observed_features: []`; da `next_n_stations` 0 ist, sieht das Modell überhaupt
keine Messung mehr, auch keine fremde — eine reine Nachbearbeitung der
NWP-Prognose). Studie `cl_m-tft-bc_out-96_freq-30min_solar_tft_nohist_hpo`,
50 Trials, bester Wert 52.114 (Trial 184) gegen 51.719 des Arms mit Historie.
Die Kette lief in der Nacht zum 18.09. unbeaufsichtigt durch
(`scripts/run_nohist_kette.sh`): 05:01 vier Trainings, 06:12 alle vier
ausgewertet.

| | Fold 1 | Fold 2 | Fold 3 | Testjahr |
|---|---|---|---|---|
| Skill_NWP mit Historie | 0.098 | 0.109 | 0.107 | **0.119** |
| Skill_NWP ohne Historie | 0.096 | 0.102 | 0.099 | **0.113** |

**Entscheidend ist der Lead-Verlauf**, gerechnet auf exakt gepaarten Zeilen
(8 349 545, GHI, nur Tagesschritte):

| Lead | 0.0 h | 0.5 h | 1 h | 1.5–3 h | 3–6 h | 12–24 h | 24–48 h | gesamt |
|---|---|---|---|---|---|---|---|---|
| mit Historie | **0.155** | 0.095 | 0.086 | 0.081 | 0.088 | 0.099 | 0.113 | 0.105 |
| ohne | **0.038** | 0.053 | 0.061 | 0.071 | 0.084 | 0.097 | 0.110 | 0.099 |
| Differenz | **0.117** | 0.042 | 0.024 | 0.011 | 0.004 | 0.001 | 0.003 | 0.005 |

Bei Lead 0 trennt die Messhistorie die beiden Arme um 0.117 Skill-Punkte (71.4
gegen 81.3 W/m²), ab drei Stunden ist der Unterschied nicht mehr vorhanden. Für
Nowcasting ist der Kanal also der wichtigste des Modells, für die Tagesplanung
bedeutungslos.

**Die Ablation überschätzt den Beitrag** — ein Punkt, der über diesen Fall
hinausgeht: sie ergab bei Lead 0 +29 % RMSE ohne Historie, der Vergleich zweier
getrennt trainierter Modelle nur +13.8 %. Ein Modell, das ohne den Kanal
trainiert wurde, stützt sich stärker auf die übrigen Signale; ein Modell, dem man
den Kanal zur Laufzeit wegnimmt, verliert mehr. Wer Featurebeiträge beziffert,
muss die beiden Verfahren auseinanderhalten.

Kennzahlen des Arms ohne Historie (Stationsmittel): Folds GHI 64.32 / R² 0.912,
DHI 36.99 / R² 0.831; Testjahr GHI 62.29 / R² 0.917, DHI 35.34 / R² 0.838.

### 5.2 Lead-0-Fehler in weiteren Solar-Pfaden

Dieselbe Zeile, dieselbe Reparatur (`lead0_offset(use_case)` aus
`geostatistics/shared/resolution.py`). Alle sechs haben einen
`if use_case == "solar"`-Zweig, sind also real betroffen, sobald MTGNN oder
WaveNet auf Solar laufen:

```
geostatistics/train_mtgnn.py:636
geostatistics/hpo_mtgnn.py:206
geostatistics/get_test_results_mtgnn.py:405
geostatistics/train_wavenet.py:606
geostatistics/hpo_wavenet.py:201
geostatistics/get_test_results_wavenet.py:404
```

Dazu `geostatistics/evaluate_reference.py:453` — kein Solar-Zweig, nagelt den
Offset aber fest; **vor jeder Solar-Referenzrechnung zu prüfen**.

`audit_data.py:200` und `get_test_results_stgnn2.py:421` sind wind-only und
dürfen so bleiben.

### 5.3 Kleinere Punkte aus der Prüfung

| Ort | Befund |
|---|---|
| `train_dcrnn.py`, `get_test_results_dcrnn.py` | Geo-Scaler wird über **alle** Stationen gefittet, der `stat_scaler` eine Zeile darüber im `--test-mode` bewusst nur über die Trainingsstationen. Sachlich vertretbar (Sonnenstand ist immer bekannt), aber unbegründet inkonsistent. |
| dieselben | Geo-Scaler wird nicht mitgespeichert, sondern in der Auswertung neu gerechnet — hält nur, solange beide Skripte dieselbe Stationsmenge sehen. Gilt ebenso für `meas_scaler`/`e2_scaler`, ist also vorbestehend. |
| `utils/imputation.py` | `impute_meas_raw_solar` hat keinen Raster-Guard; der DataFrame-Zwilling bricht bei feinerem Ziel laut ab, die Array-Variante reindexiert unbesehen. Bei `freq: 1h` nähme sie den :00-Stichpunkt statt des Stundenmittels. |
| `geostatistics/train_stgnn2.py` | `_ist_akkumuliertes_ecmwf_feature` hat `except Exception: return False` — schlägt der Import fehl, kommt der 1-h-Versatz stumm zurück. |
| `geostatistics/evaluation.py` | `gt_scaled` wird nur für das erste Ziel gebaut, die Residuumskorrektur nutzt hart `nwp_idx[0]`. Der Rückgabewert wird aktuell verworfen, ist also tot — aber eine Falle. |
| `get_test_results_dcrnn.py` | `residual_spec` ohne die `None`-Prüfung, die `train_dcrnn.py` hat: undurchsichtiger `TypeError` statt klarer Meldung. |
| ~~`get_test_results_dcrnn.py`, ECMWF-NaN~~ | **erledigt 15.09.2026**, s. §4.2 — Filter gespiegelt und die Datenursache behoben. |
| `train_fl.py:685,727` | reicht `exclude_imputed` nicht durch — FL-Solar misst auf gefüllten Zielen, CL-Solar nicht. |
| `utils/preprocessing.py:3853`, `utils/data_cache.py:519` | `Timedelta(hours=history_length)`, wobei `history_length` Schritte zählt. Vorbestehend; beide Stellen spiegeln einander, die Grenze wandert nur konservativ. |
| `geostatistics/solar_preprocessing.py:16-21` | Docstring behauptet, SL-Dateinamen seien lon-first und die Spalten vertauscht. Nachgemessen ist es lat-first ohne Vertauschung — der **Code ist richtig, der Docstring falsch**. |
| `scripts/eval_testyear.py` | Abbildungen 04/05/06 behalten Wind-Beschriftungen („m/s", „Windklasse"), laufen bei Multi-Target aber je Zielgröße. |
| ~~`get_test_results_tft_bc.py`, `pers_ref`~~ | **erledigt 17.09.2026** — war durchgängig NaN (auch beim Wind), `reindex` traf den MultiIndex nicht. Alte Wind-Parquets tragen die Lücke weiterhin, `skill` ist dort leer. |
| DCRNN-Featuresatz | `kt_nwp` fehlt als einziges der 13 TFT-Features. Aus `ghi_nwp` und `ghi_clearsky` ableitbar, beide im Modell. |

## 6. Vergleichbarkeit DCRNN ↔ TFT

Angeglichen: vier ICON-Läufe, ICON- und ECMWF-Features, Sonnengeometrie,
Residuumsziel, 30 min, Stationen, Zeitfenster, Trainingsmenge (1 404 × 41 ≈
57 564 gegen 55 624 TFT-Fenster).

Bewusst **nicht** angeglichen:

* `next_n_icond2: 4` (TFT nimmt einen Gitterpunkt) — die GATv2-Attention über
  NWP-Knoten braucht mehr als einen, und die Augustablation hat 1 gegen 4
  gemessen ohne Unterschied.
* `station_node_features` bleibt leer — mit `all` sähe das DCRNN
  Topo-Features, die der TFT nicht hat.
* Der GNN-Pfad kennt kein `train_start` und trainiert ab Datenbeginn
  (2023-07-24 statt 2023-08-01). Acht Tage mehr, über alle Arme gleich.

Offen bleibt `kt_nwp` und die fehlende Solar-HPO für das DCRNN. Solange keine
HPO existiert, ist jeder Architekturvergleich vorläufig: die DCRNN-Parameter
stammen aus einer Stunden-Wind-Konfiguration, die TFT-Defaults immerhin aus
einer Wind-HPO.

## 7. Betrieb

Drei Hosts, alle auf `0ffb797`. `DATA_ROOT` muss gesetzt sein; auf l1 bricht
die `.bashrc` in nicht-interaktiven Shells vor den Exports ab, dort also
explizit `DATA_ROOT=/mnt/nvme1` mitgeben. Läufe mit `setsid nohup … &`
starten, dann hängen sie an init (PPID 1) und überleben das Sessionende.
`/status` zeigt beide Hosts, `/sync` committet und zieht l1 und ws nach.

Auf l1 tragen GPU 2, 3 und 5 zeitweise Fremdlast eines anderen Nutzers.
