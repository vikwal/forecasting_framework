# Handoff — Solar-Kampagne, Stand 15.09.2026

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

Dateien unter `results/solar/` (TFT) und `results/solar_dcrnn_*_v5_*.pkl` (DCRNN).
Die Auswertung der beiden TFT-Arme steht im Gesprächsverlauf, ein Skript dafür
gibt es noch nicht; der Architekturvergleich läuft über
`scripts/eval_solar_arch.py`.

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
statistisch kaum zu unterscheiden. Das Modell ist an dieser Stelle
datenlimitiert, nicht hyperparameterlimitiert.

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

### 5.1.2 ARBEITSAUFTRAG: Retrain und Schlussmessung des Solar-TFT

**Ausgangslage.** Die HPO ist beendet (§5.1.1), die Hyperparameter stehen fest.
Was fehlt, ist die Modellkette wie beim Wind-Pfad: erst je Fold ein Modell mit
den besten Parametern nachtrainieren, speichern und auswerten, danach die
Schlussmessung auf dem zurückgehaltenen Testjahr. Nichts davon ist begonnen —
`results/solar_tft/` und `models/` enthalten noch keine Solar-TFT-Artefakte.

**Die besten Hyperparameter** (Trial 111, `val_rmse` 51.7188, gefunden auf `ws`,
Commit `3a0a6c8`). Sie müssen nicht abgetippt werden — beide Skripte lesen sie
über `--hpo-study` direkt aus der Optuna-Studie:

```
batch_size            59        num_lstm_layers        1
lr                    0.005740  static_embedding_dim  36
hidden_dim            33        clipnorm               2.1235
n_heads                3        dropout                0.52135
next_n_grid_points     1        next_n_grid_ecmwf      0      next_n_stations  0
```

#### Schritt 1 — drei Fold-Modelle auf dem Validierungsjahr

Je Fold ein Modell: Training auf den 41 bzw. 42 Trainingsstationen des Folds
(Zeitraum 2023-08-01 … 2024-07-31), ausgewertet auf den 21 bzw. 20 Zielstationen
desselben Folds im Validierungsjahr 2024-08-01 … 2025-07-31. Die Configs sind
fertig und liegen in `configs/solar_tft/`.

```bash
for N in 1 2 3; do
  frcst/bin/python train_cl_tft_bc.py \
      -c configs/solar_tft/config_solar_tft_fold${N}.yaml \
      --hpo-study cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo \
      --gpu <G> --cache-dir /mnt/nvme2/data_cache --max-cache-gb 150
done
```

Erzeugt je Fold `models/train_tft_bc_m-tft_c-solar_tft_fold<N>.pt` (state_dict),
`…_meta.pkl` und `…_history.pkl`. Danach die Auswertung:

```bash
for N in 1 2 3; do
  frcst/bin/python get_test_results_tft_bc.py \
      -c configs/solar_tft/config_solar_tft_fold${N}.yaml \
      --hpo-study cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo \
      --model-tag train_tft_bc_m-tft_c-solar_tft_fold${N} \
      --raw-out-name tft_solar_tft_fold${N} --gpu <G>
done
```

Erzeugt `data/test_results/tft_solar_tft_fold<N>.csv` (Metriken je Station) und
`data/raw_preds/tft_solar_tft_fold<N>_raw.parquet` (Rohprognosen) — dasselbe
Schema wie bei DCRNN/MTGNN/WaveNet, damit die Arme vergleichbar bleiben.

#### Schritt 2 — Schlussmessung auf dem Testjahr

**Erst starten, wenn Schritt 1 steht und geprüft ist.** Training auf allen 62
Poolstationen über **beide** Jahre (2023-08-01 … 2025-07-31, `train_end` ist in
der Config gesetzt), Test auf den 21 zurückgehaltenen Teststationen im dritten
Jahr 2025-08-01 … 2026-07-31.

```bash
frcst/bin/python train_cl_tft_bc.py \
    -c configs/solar_tft/config_solar_tft_testyear.yaml \
    --hpo-study cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo \
    --gpu <G> --cache-dir /mnt/nvme2/data_cache --max-cache-gb 150

frcst/bin/python get_test_results_tft_bc.py \
    -c configs/solar_tft/config_solar_tft_testyear.yaml \
    --hpo-study cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo \
    --model-tag train_tft_bc_m-tft_c-solar_tft_testyear \
    --raw-out-name tft_solar_tft_testyear --gpu <G>
```

**Trockentest am 17.09.2026, 12:58 (l2, GPU 1):** Das Retrain-Kommando für
Fold 1 läuft an — die Studie wird aufgelöst (`best_trial=111,
best_value=51.718800`), die Hyperparameter werden korrekt übernommen, das
Preprocessing lädt die 41 Trainingsstationen und fittet den globalen `scaler_x`
auf **35 Feature-Spalten** (30 known inklusive `u_10m`, 2 observed, 3 static).
Der Lauf wurde an dieser Stelle abgebrochen; Training und Auswertung sind noch
offen. Der Cache-Eintrag war zu diesem Zeitpunkt noch nicht geschrieben, der
erste echte Lauf baut ihn also neu.

#### Stolperfallen — vor dem Start lesen

1. **`get_test_results_tft_bc.py` ist noch nicht multi-target-fähig und muss
   angepasst werden.** Zeile 257 und 264 lesen `config['data']['target_col']`
   (Singular). Solar-Configs setzen `data.target_cols: [ghi, dhi]`, `target_col`
   ist dort **`None`** — verifiziert über `tools.load_config`. Richtig ist
   `preprocessing.get_target_cols(config)`, das `target_cols` bevorzugt und für
   Wind unverändert `['wind_speed']` liefert. Zweitens rechnet die Metrikschleife
   (ab Zeile 287) RMSE/MAE/R² über das ganze Array: bei zwei Zielgrößen hat
   `y_pred` die Form `(n, horizon, 2)`, und man erhielte **eine vermischte Zahl
   über GHI und DHI statt zweier getrennter Werte**. Beides muss vor der
   Auswertung behoben werden — sonst laufen die Skripte entweder auf einen
   Fehler oder, schlimmer, auf ein stilles Fehlergebnis. Vorbild für die
   Aufteilung je Zielgröße: `scripts/eval_testyear.py` (dort für Wind, aber
   Multi-Target-tauglich; die Wind-Beschriftungen „m/s"/„Windklasse" in den
   Abbildungen 04/05/06 sind bekannt und müssten für Solar angepasst werden).

2. **`--test-mode` hier NICHT verwenden** — anders als beim Wind-Pfad, wo
   `train_cl_tft_bc.py` es im Docstring ausdrücklich für den finalen Testlauf
   vorsieht. Grund: Das Flag mischt `val_files` in den Trainingspool. Bei Wind
   sind `val_files` und `test_files` disjunkt (50 gegen 50, Überschneidung 0),
   dort ist das korrekt. **In `config_solar_tft_testyear.yaml` sind `val_files`
   und `test_files` identisch** (dieselben 21 Teststationen) — `--test-mode`
   zöge sie ins Training und machte die Schlussmessung wertlos. Ohne das Flag
   trainiert die Config auf 62 Poolstationen und misst auf den 21 Teststationen,
   genau wie in `station_splits_solar.md` §4 festgelegt.

3. **Dass `val_files` gleich `test_files` ist, ist Absicht, kein Fehler.** Die
   21 Teststationen dienen im Trainingszeitraum als Validierungsset für Early
   Stopping und werden erst im Testjahr zur Messung herangezogen — zeitlich
   getrennt. Der dadurch entstehende Optimismus ist gemessen und bewusst
   akzeptiert (`station_splits_solar.md` §6, Entscheidung Viktor 18.08.2026:
   rund ein Prozent, kein Umbau).

4. **`--hpo-study` immer explizit angeben.** Beide Skripte leiten den
   Studiennamen sonst aus dem Config-Dateinamen ab und kämen bei
   `config_solar_tft_fold1.yaml` auf `…_solar_tft`, nicht auf das tatsächliche
   `…_solar_tft_hpo`. Ohne das Flag greifen sie ins Leere oder auf eine falsche
   Studie.

5. **Die Fold-Configs hatten kein `test_files` — am 17.09.2026 ergänzt.**
   `get_test_results_tft_bc.py` liest per Default `files_key='test_files'`; ohne
   den Schlüssel fände die Fold-Auswertung keine Station und liefe ins Leere.
   Die Zielstationen des Folds stehen jetzt in `val_files` **und** `test_files`
   (21/21/20), ausgewertet im Fenster `test_start`…`test_end` = 2024-08-01 …
   2025-08-01, also im Validierungsjahr. Der zurückgehaltene Testsatz bleibt
   unberührt. Das spiegelt, was der Generator bei der Schlussmessung ohnehin
   tut. Alternativ gäbe es `--eval-split val`, das dafür aber ein `val_start` in
   der Config braucht, das die Fold-Configs nicht setzen — bei
   `next_n_stations: 0` liefern beide Wege dasselbe, weil sich die Wege nur im
   `neighbor_pool` unterscheiden.

   Zum Kontrast, damit die Wind-Analogie nicht in die Irre führt: Wind-Folds
   haben drei getrennte Mengen (103 train / 50 val / 50 test), Solar-Folds nur
   zwei (41 train / 21 Ziel) — die 21 Teststationen liegen außerhalb des Pools
   und kommen erst in der Schlussmessung vor.

6. **Featuresatz ist am 17.09.2026 angeglichen worden.** Die Fold- und
   Testyear-Configs enthielten `u_10m` nicht, die HPO lief aber damit — sie
   hätten ein anderes Modell trainiert als das optimierte. `u_10m` steht jetzt
   am **Ende** von `icond2_features` und `known_features` (genau dort hängt
   `hpo_tft_bc.py` optionale Features an, damit bleibt der Cache-Schlüssel
   identisch), dazu `next_n_grid_ecmwf: 0` explizit. Erzeugt wurde das über
   `scripts/make_solar_tft_configs.py --force`; die Configs sind generiert und
   **nicht von Hand zu pflegen**.

7. **Cache.** Auf l2 liegen die drei HPO-Fold-Einträge unter
   `/mnt/nvme2/data_cache` (~50 GB, Budget 150 GB). Die Retrain-Läufe haben
   andere Zeitachsen als die HPO und bauen daher **eigene** Einträge — je Lauf
   rund 16 GB, beim ersten Start entsprechend Vorlaufzeit einplanen. Auf ws
   liegt der Cache unter `$HOME/data_cache` (dort existiert `/mnt/nvme2` nicht),
   auf l1 ist `DATA_ROOT=/mnt/nvme1` statt `/mnt/lambda1/nvme1`, was in den
   Cache-Schlüssel eingeht: **Einträge sind zwischen l1 und l2/ws nicht
   austauschbar**, ohne den Schlüssel neu zu rechnen.

8. **Mehrere Läufe gleichzeitig auf einem Host mit leerem Cache vermeiden.**
   `DataCache.save_preprocessed_data` schreibt ohne Lock und ohne atomares
   `os.replace` (den flock hat nur `GNNCache`). Gleichzeitig startende Prozesse
   bauen denselben Eintrag mehrfach parallel, und ein Leser kann eine halb
   geschriebene `prepared.pkl` sehen. Erst einen Lauf durchlassen, dann die
   übrigen.

9. **Datenlage im Testjahr prüfen, bevor Zahlen interpretiert werden.** Der
   mittlere Anteil echter Messwerte liegt dort bei 0.85, im Minimum bei 0.12.
   Mit `eval.exclude_imputed: true` (in allen Configs gesetzt) bleibt
   entsprechend weniger Auswertungsmasse übrig — je Station nachzählen.

10. **Station 05792 fällt im CL-Pfad aus** (siehe unten in §5.1.1), die Läufe
   arbeiten deshalb auf 61 statt 62 Poolstationen. Für alle Arme gleich, aber
   beim Vergleich gegen die DCRNN-Arme (21 Zielstationen) zu berücksichtigen.

11. **GPU-Wahl.** GPU 0 auf l2 ist oft fremdbelegt; auf l1 tragen die GPUs 3 und
    5–7 dauerhaft Fremdlast eines anderen Nutzers. Vor dem Start mit
    `nvidia-smi` prüfen. Ein Trainingslauf dieser Größe belegt 4–15 GB.

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
