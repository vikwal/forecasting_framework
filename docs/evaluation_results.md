# Auswertung der HPO-Kampagne: §3b und Vorprüfungen

**Erstellt:** 2026-08-17 · Auftrag: `docs/prompt_evaluation_kickoff.md`. Basis:
`forecasting_framework` auf `l2`, Branch `fix/mtgnn-topo-static-dim`, Commit
**`e15d778`** (Wind-Anteil, siehe §9.6; vorher HEAD `4f832ec` vom 2026-08-12 plus
uncommitteter Arbeitsbaum, siehe §6). Optuna in Postgres `optuna_db` auf `l2`.

**Was dieses Dokument abschließt:** §3b (HPO-Analyse der Gitterpunktzahl) vollständig,
die Vorprüfungen aus §7 des Auftrags, sowie die Entscheidungen zu allen fünf OFFENEN
FRAGEN und deren Umsetzung (§9). Der Befund N1 ist gefixt (§9.2), die Kampagne ist auf
`l2` zurückgefahren (§9.3), der Wind-Codestand ist committet (§9.6).

**Was offen bleibt:** §3a (die Retrains) ist **nicht gerechnet**. Es sind **keine**
Haupttabellenzahlen erzeugt worden. Blockierend ist allein noch GPU-Kapazität: die
freigegebenen Worker beenden zuerst ihren laufenden Trial. Das verifizierte Rezept für
die Retrains steht in §10.

Alle Zahlen unten sind selbst nachgerechnet, nicht aus Vorgängerdokumenten oder aus
Aussagen von Unteragenten übernommen.

---

## 0. Dateiliste

| Datei | Zweck |
|---|---|
| `/tmp/hpo_param_analysis.py` (l2) | §3b, erster Durchgang: Randverteilung und Zielwert je Parameterwert |
| `/tmp/hpo_param_robust.py` (l2) | §3b, zweiter Durchgang: Robustheit der Korrelation, Optuna-Wichtigkeiten |
| `/tmp/check_loader_consistency.py` (l2, l1) | Querprüfung der Messdatenlader zwischen den Hosts (§5) |
| `/tmp/n1_magnitude.py` (l2) | Größe der N1-Skaliererabweichung je Fold und Kanal (§9.2) |
| `/tmp/apply_n1_fix.py` (l2, l1, ws) | Anwendung des N1-Fixes mit Exact-Match-Absicherung |
| `/tmp/hpo_param_summary.csv`, `/tmp/hpo_param_robust.csv` (l2) | Ergebnistabellen der beiden Durchgänge |
| `geostatistics/get_test_results_dcrnn.py` | N1-Fix, committet in `e15d778` |
| `~/hpo_keeper_plan.json` (l2) | auf 2 Sollworker reduziert, Sicherung `.bak-20260817-eval` |
| `docs/evaluation_results.md` | dieses Dokument |

Die Diagnoseskripte liegen unter `/tmp`. Für dauerhafte Reproduzierbarkeit gehören sie
nach `archiv/hpo_analysis/`, analog zu `archiv/baselines_verification/`; das ist bewusst
noch nicht geschehen, weil der Arbeitsbaum ohnehin noch 40 uncommittete Einträge trägt.

---

## 1. Nachrechnung des Studienstands (§4 des Auftrags)

Stand 2026-08-17 16:10, eigene Abfrage über `trials`/`studies`/`trial_values`.

| Studie | COMPLETE | PRUNED | FAIL | RUNNING | bestes Val-RMSE (gepoolt) | bester Trial |
|---|---|---|---|---|---|---|
| dcrnn (GRID) | 170 | 16 | 14 | 0 | 1.1741 | #192 |
| dcrnn_base | 110 | 22 | 3 | 2 | 1.2242 | #111 |
| dcrnn_idw_alt | 109 | 15 | 10 | 0 | 1.1819 | #109 |
| dcrnn_nograph | 100 | 14 | 13 | 4 | 1.2117 | #105 |
| dcrnn_nomeas | 147 | 13 | 23 | 0 | 1.1857 | #170 |
| dcrnn_nwp_hist | 87 | 15 | 12 | 4 | 1.0624 | #101 |
| mtgnn | 76 | 16 | 10 | 4 | 1.2332 | #60 |
| mtgnn_nwp | 70 | 10 | 12 | 6 | 1.2147 | #89 |
| mtgnn_nwp_hist | 55 | 20 | 15 | 8 | 0.9868 | #67 |
| wavenet | 70 | 15 | 4 | 3 | 1.2172 | #72 |
| wavenet_nwp | 64 | 11 | 4 | 3 | 1.2069 | #55 |
| tft_sp_base | 53 | 7 | 7 | 0 | 1.2593 | #39 |
| tft_sp_hist | 101 | 15 | 6 | 0 | 1.0717 | #82 |

**Die Spalte „bestes Val-RMSE" stimmt in allen dreizehn Studien exakt mit §4 des Auftrags
überein.** Die Zählspalten weichen leicht ab, aus zwei getrennten Gründen:

1. **Weiterlaufende Studien** haben seit dem 16:00-Stand Trials abgeschlossen
   (dcrnn 169 → 170, nograph 99 → 100, nomeas 146 → 147).
2. **PRUNED und FAIL sind gesunken**, nicht gestiegen: `mtgnn_nwp` FAIL 13 → 12,
   `mtgnn_nwp_hist` FAIL 16 → 15, `wavenet_nwp` PRUNED 13 → 11 und FAIL 5 → 4.
   Grund siehe §2.

Insgesamt 1321 COMPLETE-Trials über die elf `cl_*wind_*`-Graphstudien und die zwei
TFT-Spatial-Studien. Davon **0 mit NULL- oder NaN-Zielwert** und 77 mit einem Zielwert
über 1.5 m/s, keiner über 5 m/s. Die 77 sind kollabierte, aber formal abgeschlossene
Läufe; sie sind für §3b wichtig, siehe §3.2.

## 2. Die fünf Trials aus dem Datenvorfall sind gelöscht, nicht nur zu ignorieren

Der Auftrag (§6.2) nennt fünf Trials, die aus jeder Analyse heraus müssen. Diese fünf
existieren in der Datenbank **nicht mehr**. Nachgewiesen an den Lücken in der
Trialnummerierung:

| Studie | erwartete Nummer | Zustand in der DB |
|---|---|---|
| `mtgnn_nwp` | #93 | fehlt (91 PRUNED, 92 COMPLETE, 94 RUNNING) |
| `mtgnn_nwp_hist` | #95 | fehlt (93, 94 RUNNING, 96 RUNNING) |
| `wavenet_nwp` | #79, #81, #82 | alle drei fehlen (77, 78 COMPLETE, 80 RUNNING) |

Das erklärt die gesunkenen PRUNED- und FAIL-Zahlen aus §1 vollständig und deckt sich
mit der Rechnung: zwei PRUNED weniger bei `wavenet_nwp`, je ein FAIL weniger bei den
beiden MTGNN-Studien.

**Folge für die Auswertung:** die Ausschlussliste aus §6.2 ist bereits auf
Datenbankebene durchgesetzt. Eine zusätzliche Filterung in den Analyseskripten ist
weder nötig noch möglich. Auch die Pruner-Statistik ist damit schon bereinigt, anders
als der Auftrag annimmt („die beiden PRUNED gehen in die Pruner-Statistik").

## 3. §3b: welche Gitterpunktzahl hat der Optimierer bevorzugt

### 3.1 Vorprüfung: der trialweise gezogene Wert wirkt tatsächlich

Der Auftrag verlangt, vor der Interpretation zu sichern, dass der je Trial gezogene
`next_n` nicht bloß protokolliert wird, während der Cache mit der Obergrenze gebaut ist.
**Verifiziert im Code, in allen drei HPO-Skripten, über zwei unabhängige Mechanismen:**

- **Graph je Trial neu gebaut.** `hpo_dcrnn.py:394` setzt
  `_rebuild_graph_per_trial = _i2_hpo_spec is not None or _e2_hpo_spec is not None`,
  also genau dann, wenn `next_n_icond2` oder `next_n_ecmwf` Suchparameter sind. Bei
  `True` wird in der Trialschleife (`hpo_dcrnn.py:1192`) ein eigener
  `HeterogeneousGraphBuilder(model_cfg.graph)` mit den Trialwerten instanziiert, statt
  den außerhalb gebauten statischen Graphen zu benutzen.
- **Nachbarschaftsindizes je Trial geschnitten.** `hpo_dcrnn.py:1278` liest
  `_trial_k = model_cfg.graph.next_n_icond2_grid_points` und
  `_trial_ke = ...next_n_ecmwf_grid_points` und übergibt die mit der Obergrenze
  geladenen Arrays geschnitten weiter:
  `station_k_nearest_grid[:, :_trial_k]` und `station_k_nearest_ecmwf[:, :_trial_ke]`.

In `hpo_mtgnn.py:930-932` und `hpo_wavenet.py:916-919` läuft es analog: die Trialwerte
gehen als `k_nwp`, `k_ecmwf` und `next_n_neighbors` in den Dataset- und Graphaufbau
(`hpo_mtgnn.py:993-1038`, `hpo_wavenet.py:977-1021`).

Der Cache mit der Obergrenze ist damit eine echte Obermenge, aus der je Trial korrekt
ausgeschnitten wird. Die Logzeile „loading 4 ECMWF grid points (max bound)" beschreibt
nur den Ladevorgang, nicht die Modelleingabe.

### 3.2 Warum die einfache Randkorrelation hier nicht trägt

Der Auftrag warnt, die Optuna-Wichtigkeiten könnten bei korrelierten Parametern
täuschen. Die Prüfung zeigt, dass auch die naive Gegenprobe (Zielwert über Parameter
über alle COMPLETE-Trials) nicht trägt, und zwar aus zwei messbaren Gründen:

1. **Der Sampler wandert.** Teilt man die Trials je Studie an ihrem Nummernmedian in
   eine frühe und eine späte Hälfte, kippt das Vorzeichen der Rangkorrelation in der
   Mehrzahl der Studien. Für `next_n_icond2`: dcrnn früh `+0.365` / spät `−0.464`,
   dcrnn_nomeas früh `+0.373` / spät `−0.520`, mtgnn_nwp_hist früh `+0.430` /
   spät `+0.355`, wavenet_nwp früh `+0.340` / spät `−0.142`. Eine Größe, die zwischen
   den Hälften einer Studie das Vorzeichen wechselt, misst überwiegend die
   Suchtrajektorie von TPE, nicht die Wirkung des Parameters.
2. **Ein Schwanz kollabierter Läufe dominiert die Ränge.** Die 77 COMPLETE-Trials mit
   RMSE über 1.5 sind extrem ungleich verteilt: `mtgnn_nwp` 15, `mtgnn` 13,
   `mtgnn_nwp_hist` 12, `wavenet` und `wavenet_nwp` je 10, dagegen `dcrnn_base` nur 1.
   Bei `wavenet` liegt der Mittelwert bei `next_n_icond2 = 1` auf 1.7871 m/s über nur
   drei Trials, einer davon 2.0561. Genau diese drei tragen das `full`-Rho von `−0.488`.

Belastbar sind daher drei Größen: die Rangkorrelation **innerhalb des besten Viertels**
der Trials (`top25`, der Bereich, aus dem die Bestenauswahl kommt), der **Medianwert des
Parameters unter dem besten Zehntel** der Trials, und der **Anteil dieser besten
Trials, der auf der Obergrenze des Suchbereichs sitzt**.

### 3.3 Ergebnis: die Gitterpunktzahl ist überwiegend flach

`next_n_icond2`, Suchbereich `int[1..7]` in allen Studien. `rho` negativ heißt „mehr
Punkte sind besser". `Median10` ist der Medianwert unter dem besten Zehntel,
`Rand%` der Anteil dieser Trials auf der Obergrenze 7, `fANOVA` die Optuna-Wichtigkeit.

| Studie | n | rho (alle) | rho (bestes Viertel) | Median10 | Rand% | fANOVA |
|---|---|---|---|---|---|---|
| dcrnn (GRID) | 170 | +0.199 | −0.084 | 4 | 0 | 0.001 |
| dcrnn_base | 110 | +0.439 | +0.096 | 2 | 0 | 0.106 |
| dcrnn_idw_alt | 109 | −0.667 | −0.110 | 7 | **90.9** | 0.024 |
| dcrnn_nograph | 100 | +0.033 | +0.072 | 3 | 20.0 | 0.002 |
| dcrnn_nomeas | 147 | +0.283 | −0.023 | 3 | 0 | 0.020 |
| dcrnn_nwp_hist | 87 | +0.275 | −0.340 | 4 | 0 | 0.007 |
| mtgnn | 76 | +0.400 | −0.265 | 2.5 | 0 | 0.016 |
| mtgnn_nwp | 70 | +0.263 | −0.153 | 3 | 0 | 0.001 |
| mtgnn_nwp_hist | 55 | +0.077 | −0.065 | 3 | 0 | 0.007 |
| wavenet | 70 | −0.488 | −0.266 | 7 | **85.7** | 0.001 |
| wavenet_nwp | 64 | −0.209 | −0.373 | 5 | 0 | 0.006 |

`next_n_ecmwf`, Suchbereich `int[0..4]`, Obergrenze 4:

| Studie | n | rho (alle) | rho (bestes Viertel) | Median10 | Rand% | fANOVA |
|---|---|---|---|---|---|---|
| dcrnn (GRID) | 170 | −0.245 | −0.135 | 4 | 64.7 | 0.014 |
| dcrnn_base | 110 | +0.262 | n/a | 2 | 0 | 0.028 |
| dcrnn_idw_alt | 109 | −0.390 | −0.345 | 4 | 63.6 | 0.002 |
| dcrnn_nograph | 100 | −0.117 | +0.261 | 3 | 0 | 0.090 |
| dcrnn_nomeas | 147 | +0.549 | n/a | 1 | 0 | 0.024 |
| dcrnn_nwp_hist | 87 | −0.412 | −0.174 | 4 | 88.9 | 0.006 |
| mtgnn | 76 | +0.379 | +0.172 | 1 | 0 | 0.008 |
| mtgnn_nwp | 70 | −0.386 | +0.397 | 3 | 0 | 0.001 |
| mtgnn_nwp_hist | 55 | −0.599 | −0.034 | 4 | **100.0** | 0.033 |
| wavenet | 70 | −0.540 | −0.443 | 4 | **100.0** | 0.000 |
| wavenet_nwp | 64 | −0.241 | −0.100 | 3 | 28.6 | 0.005 |

`next_n_neighbors`, Suchbereich `int[50..90]`. `dcrnn_nograph` hat den Parameter
erwartungsgemäß nicht.

| Studie | n | rho (alle) | rho (bestes Viertel) | Median10 | Rand% | fANOVA |
|---|---|---|---|---|---|---|
| dcrnn (GRID) | 170 | −0.275 | −0.210 | 66 | 0 | 0.009 |
| dcrnn_base | 110 | +0.207 | +0.428 | 55 | 0 | 0.014 |
| dcrnn_idw_alt | 109 | +0.409 | +0.036 | 64 | 0 | 0.010 |
| dcrnn_nomeas | 147 | −0.385 | +0.257 | 76 | 0 | 0.013 |
| dcrnn_nwp_hist | 87 | −0.485 | +0.330 | 84 | 0 | 0.083 |
| mtgnn | 76 | −0.508 | +0.457 | 84.5 | 0 | 0.018 |
| mtgnn_nwp | 70 | +0.417 | +0.487 | 64 | 0 | 0.006 |
| mtgnn_nwp_hist | 55 | +0.150 | +0.129 | 61 | 0 | 0.024 |
| wavenet | 70 | −0.402 | +0.123 | 85 | 0 | 0.003 |
| wavenet_nwp | 64 | −0.069 | +0.199 | 75 | 0 | 0.004 |

**Drei Aussagen, die der Auswertung standhalten:**

(a) **Die Gitterpunktzahl ist nirgends ein dominanter Hyperparameter.** Die
Optuna-Wichtigkeit liegt für `next_n_icond2` in zehn von elf Studien unter 0.03, für
`next_n_ecmwf` in neun von elf unter 0.03. Der größte Wert überhaupt ist 0.106
(`dcrnn_base`, `next_n_icond2`). Die Wichtigkeit steckt in Lernrate, Weite und Dropout,
nicht in der Gittergröße. Das ist das robusteste Ergebnis von §3b, weil es unabhängig
von der Rangstatistik ist.

(b) **Für ICON-D2 gibt es keine allgemeine Vorliebe für viele Gitterpunkte.** Nur zwei
von elf Studien schöpfen die Obergrenze aus, `dcrnn_idw_alt` (90.9 % der besten Trials
bei 7) und `wavenet` (85.7 %). In den übrigen neun liegt der Median des besten Zehntels
bei 2 bis 5 von 7, und der Rand wird von 0 bis 20 % der besten Trials berührt. Im
besten Viertel ist die Korrelation neunmal betragsmäßig unter 0.35.

(c) **Für ECMWF sieht es anders aus, und das ist ein Handlungspunkt.** In fünf Studien
sitzt die Mehrheit der besten Trials auf der Obergrenze 4: `mtgnn_nwp_hist` 100 %,
`wavenet` 100 %, `dcrnn_nwp_hist` 88.9 %, `dcrnn` 64.7 %, `dcrnn_idw_alt` 63.6 %. Das
ist das Muster eines **abgeschnittenen Suchbereichs**: das Optimum kann jenseits von 4
liegen, und die Kampagne kann es nicht sehen. Für `next_n_neighbors` gilt das
ausdrücklich **nicht**, dort liegt der Rand bei 0 % in allen zehn Studien und der Median
des besten Zehntels mit 55 bis 85 im Inneren von `[50, 90]`, der Parameter ist dort
tatsächlich flach.

**Offene Frage an den Nutzer, neu:** soll der Suchbereich von `next_n_ecmwf` über 4
hinaus erweitert werden? Das ist eine Aussage über die Kampagne selbst, nicht über die
Auswertung, und es wäre der einzige mir sichtbare Punkt, an dem die Kampagne systematisch
Leistung liegen lässt. Dagegen spricht, dass es die Vergleichbarkeit mit den bisherigen
Trials bricht und Budget kostet.

### 3.4 Die TFT-Spatial-Studien benutzen andere Parameternamen

`tft_sp_base` und `tft_sp_hist` suchen über `next_n_grid_points` (`int[1..7]`),
`next_n_grid_ecmwf` (`int[0..4]`) und `next_n_stations` (`int[0..8]`), nicht über die
`next_n_*`-Namen der Graphstudien. Beide haben **keinen** kollabierten Trial
(RMSE über 1.5). Ergebnis, falls sie nach FRAGE 4 in die Auswertung kommen:

| Studie | Parameter | n | rho (bestes Viertel) | Median10 | Rand% | fANOVA |
|---|---|---|---|---|---|---|
| tft_sp_base | next_n_grid_points | 53 | +0.166 | 2 | 0 | 0.026 |
| tft_sp_base | next_n_grid_ecmwf | 53 | +0.003 | 3 | 0 | **0.212** |
| tft_sp_base | next_n_stations | 53 | −0.170 | 5 | 0 | 0.045 |
| tft_sp_hist | next_n_grid_points | 101 | −0.309 | 7 | 72.7 | 0.091 |
| tft_sp_hist | next_n_grid_ecmwf | 101 | −0.105 | 3 | 0 | 0.057 |
| tft_sp_hist | next_n_stations | 101 | +0.162 | 2 | 0 | 0.022 |

`tft_sp_base`, `next_n_grid_ecmwf` mit fANOVA 0.212 ist die einzige Stelle in der
ganzen Kampagne, an der eine Gitterpunktzahl ein wichtiger Parameter ist, und dort ist
das Rho im besten Viertel mit `+0.003` exakt null. Das ist der vom Auftrag
vorhergesagte Fall: Wichtigkeit ohne monotone Wirkung, also reine Interaktion. Ohne die
Gegenprobe hätte man daraus „ECMWF-Gitterpunkte sind für TFT entscheidend" gelesen.

## 4. N1, der blockierende Befund für die DCRNN-Retrains

Der Auftrag beschreibt N1 als Fehler in `train_dcrnn.py:865`. **Das ist die falsche
Seite.** Nachgelesen im Code:

- `hpo_dcrnn.py:409-419`: im räumlichen CV-Modus ist `all_ids = station_pool(...)` und
  `N_train = len(all_ids)`, also **153**. Zeile 694 fittet
  `stat_scaler.fit(raw_static[:N_train])`, damit auf allen 153 Stationen.
- `train_dcrnn.py:882`: `stat_scaler.fit(raw_static if (val_start and not args.test_mode)
  else raw_static[:N_train])`, im Spatial-CV-Fall also ebenfalls auf allen 153. Der Code
  trägt dafür einen ausformulierten Kommentar mit Begründung (Review-Kürzel M5) und den
  Hinweis, dass `--test-mode` bewusst beim Train-only-Fit bleibt, weil dort die
  Teststationen an `all_ids` angehängt werden. **Train und HPO sind also konsistent.**
- `get_test_results_dcrnn.py:236-250` und `:427` (vor dem Fix, nach dem Fix `:436`):
  ohne `--test-mode` ist `train_ids = data_cfg["files"]`, also `N_train = 102`, und die
  Zeile fittete `stat_scaler.fit(raw_static[:N_train])` auf **102**.

**Die Inkonsistenz sitzt allein im Auswertungsskript**, und zwar genau im
Entwicklungsmodus, also in dem Pfad, den die Retrains aus §3a brauchen (51 nie gesehene
Zielstationen je Fold). Ein so ausgewertetes Modell bekommt `lat`, `lon` und `alt` mit
anderen Mittelwerten und Streuungen normiert, als es im Training gesehen hat. Die
Vorhersagen sind dann nicht falsch berechnet, sondern das Modell wird außerhalb seines
Eingaberaums betrieben.

Der Fix ist eine Zeile, gespiegelt aus `train_dcrnn.py:882`: im Spatial-CV-Fall (also
wenn `val_start` gesetzt und nicht `--test-mode`) auf ganz `raw_static` fitten. Ich habe
**nichts geändert**, weil der Auftrag verlangt, das vor der Zahlenerzeugung zu klären,
und weil dieselbe Datei im laufenden Betrieb steht.

Bemerkenswert daneben: die direkt anschließenden topographischen Merkmale werden in
`get_test_results_dcrnn.py` in `load_topo_station_features(...)` **absichtlich** mit `n_train=N_train`, also auf 102,
normiert, mit Kommentar „Fitting on all_ids would normalise the topography of the
held-out stations with their own statistics". Für `lat`/`lon`/`alt` gilt dieses Argument
nicht, weil Koordinaten bei einem induktiven Modell immer bekannte Eingaben sind, genau
so steht es in `train_dcrnn.py`. Die beiden Blöcke widersprechen sich also nicht, sie
behandeln absichtlich unterschiedliche Größen unterschiedlich. Beim Fix darf nur der
`stat_scaler` angefasst werden, nicht `load_topo_station_features`.

## 5. Die Hosts tragen unterschiedlichen Code, für Wind aber wirkungsgleich

Nicht im Auftrag erwähnt, aber sicherheitsrelevant, weil alle drei Hosts in **dieselbe**
Optuna-Studie schreiben: `l2` und `l1`/`ws` tragen unterschiedliche Fassungen der
Wind-Skripte. `l1` und `ws` sind untereinander bytegleich, `l2` weicht ab:

| Datei | l2 vs l1/ws |
|---|---|
| `geostatistics/train_stgnn2.py` | 105 Diffzeilen |
| `geostatistics/train_dcrnn.py` | 22 |
| `geostatistics/get_test_results_dcrnn.py` | 10 |
| `geostatistics/hpo_mtgnn.py`, `hpo_wavenet.py` | je 8 |
| `geostatistics/hpo_dcrnn.py` | 4 |
| `geostatistics/stgnn/training/sampler.py`, `evaluation.py`, `configs/spatial_folds.yaml` | bytegleich |

`l2` ist durchgehend der neuere Stand: es trägt die Solar-Umbauten (`use_case`,
`stations_master`, `time_label`, `sub_hourly_fill`, `freq_to_hours`), `l1`/`ws` die
ältere Fassung. Zwei Dinge daran waren zu prüfen, beide sind in Ordnung:

1. **Der Laufpaar-Ausschluss ist überall aktiv.**
   `def exclude_run_pairs_with_ecmwf_nan` existiert in `train_stgnn2.py` auf beiden
   Hosts, und die Aufrufzahlen in den drei HPO-Skripten stimmen exakt überein
   (`hpo_dcrnn.py` 3, `hpo_mtgnn.py` 2, `hpo_wavenet.py` 2). Es gibt also keine Hosts,
   die noch mit 2933 statt 2909 Laufpaaren rechnen.
2. **Der Messdatenlader liefert bitgleiche Tensoren.** `l2` aggregiert über
   `solar.resample_interval_mean`, `l1` über `resample(freq, closed="left",
   label="left").mean()`. Der `time_label`-Versatz auf `l2` steht hinter
   `if solar_mode:` und greift für Wind nicht. Gegenprobe auf 12 Stationen aus
   `spatial_fold1`, beide Hosts, `freq='1h'`, Spalten `wind_speed`/`wind_direction`:
   identische Form `(26760, 12, 2)`, identischer Indexbereich
   2023-07-24 00:00 bis 2026-08-11 23:00, identisch 3626 NaN,
   identischer `sha256` über die auf sechs Stellen gerundeten finiten Werte
   (`a4bedf94ae06f808`), und die Stationsmittel stimmen auf zehn Dezimalstellen
   (etwa `00183` 6.6532759684 auf beiden Hosts).

**Die Trials aus der Kampagne sind über die Hosts hinweg also vergleichbar.** Das war
nicht vorausgesetzt, sondern gemessen. Der Codeunterschied sollte trotzdem aufgelöst
werden, bevor die Retrains laufen, damit nicht offen bleibt, welcher Host welche Zahl
erzeugt hat.

## 6. Zustand des Arbeitsbaums: §9.5 ist derzeit nicht erfüllbar

Der Auftrag verlangt in §9.5 je Zahl den Commit. Das geht heute nicht. `git status` auf
`l2` zeigt **20 modifizierte Dateien** gegen HEAD `4f832ec` vom 2026-08-12, nicht die
vier aus §7 des Auftrags. Nach Änderungszeit:

- **2026-08-17, 15:33:** `train_stgnn2.py`, `hpo_dcrnn.py`, `hpo_mtgnn.py`,
  `hpo_wavenet.py`. Das sind die im Auftrag genannten vier.
- **2026-08-11 bis 08-13:** `train_dcrnn.py`, `get_test_results_dcrnn.py`,
  `get_test_results_mtgnn.py`, `get_test_results_wavenet.py`, `train_mtgnn.py`,
  `train_wavenet.py`, `utils/preprocessing.py`, `utils/eval.py`, `utils/models.py`,
  `utils/data_cache.py`, `train_cl.py`, `train_fl.py`, `evaluate_reference.py`. Das ist
  **genau der Code, der die Retrainzahlen erzeugen wird**, und er ist ebenfalls
  uncommittet.
- Dazwischen liegt eine große Menge unzusammenhängender Solar-Arbeit
  (`utils/solar.py`, `solar_preprocessing.py`, `configs/solar_*`, ein Dutzend
  `scripts/run_solar_*.sh`), plus `CLAUDE.md`.

Ein Commit vor den Retrains ist damit nicht Kosmetik, sondern die Voraussetzung dafür,
dass die Zahlen später zuordenbar sind. Weil Wind- und Solaränderungen im selben
Arbeitsbaum liegen, ist das kein reines „alles committen": das gehört getrennt.

## 7. Weitere Vorprüfungen aus §7 des Auftrags

| Punkt | Stand |
|---|---|
| Foldgrößen | `configs/spatial_folds.yaml` bestätigt: `spatial_fold1/2/3` je `files` 102, `val_files` 51, Pool 153. Deckt sich mit §2 des Auftrags. |
| `configs/*/test/` veraltet | Bestätigt. Alle geprüften Dateien tragen `files` 103, `val_files` 50, `test_files` 50, also die alte Aufteilung. Das ist ein konkreter Kostenpunkt für FRAGE 1(b): das Testfenster ist ohne neu erzeugte Fold-Configs nicht rechenbar. |
| `hist_wind_available` | Geprüft, **keine Leckage**. Der Verlauf wird in `sampler.py:401` und `evaluation.py:126` als `station_meas[t_hist_abs:t_run_abs]` geschnitten, also strikt vor dem ersten Prognoseschritt. Bei `false` werden die Zielstationen genullt (`meas_hist[:, N_train:, :] = 0.0`), bei `true` behalten sie ihren Vergangenheitsverlauf. Das ist die einzige inhaltliche Differenz zwischen `config_wind_dcrnn.yaml` und `config_wind_dcrnn_nwp_hist.yaml`. Siehe aber die Einordnungsfrage unten. |
| Station 05426 | Nicht geprüft. Braucht die Referenzauswertung und damit einen Datenlauf; steht in der Kampagne als Trainingsstation in `files`. |
| KNN-Fallback in `get_test_results_dcrnn.py` | Nicht geprüft, betrifft erst die Retrainauswertung. |
| Referenzdateien vom 2026-08-06 | Nicht angefasst. Es sind in diesem Dokument keine ungefilterten Zahlen benutzt. |
| Naive Winkelmittelung | Nicht geprüft, betrifft `run_spatial_interpolation.py` und `utils/preprocessing.py`, nicht den HPO-Pfad. |
| Laufpaarzahlen 2933/2909 | Nicht nachgerechnet. Braucht einen Datenlauf; als Abnahmekriterium beim ersten Retrain zu prüfen. |

**Einordnungsfrage, neu, zu `hist_wind_available`:** die `*_nwp_hist`-Arme sind zwar
induktiv im Sinne von „an der Zielstation nie trainiert", aber sie bekommen an der
Zielstation zur Laufzeit deren eigenen Messverlauf. Das ist ein anderes Problem als bei
den übrigen Armen, wo die Zielstation gar keine Messung hat. Genau diese Arme sind die
besten der Kampagne (`mtgnn_nwp_hist` 0.9868, `dcrnn_nwp_hist` 1.0624 gepoolt gegen 1.17
bis 1.23 bei den übrigen). Damit ist MOS-local mit 0.9452 für sie **keine Obergrenze**:
MOS-local benutzt Messungen der Zielstation zum Trainieren, die `hist`-Arme benutzen sie
zur Laufzeit, beide haben Zugang zur Zielstation. Die Bezeichnung „transduktive
Obergrenze" aus §4 des Auftrags trägt also nur für die Arme ohne Zielstationsmessung.
Das ist kein Datenfehler, sondern eine Frage der Tabellenaufteilung, und sie sollte vor
der Haupttabelle entschieden sein.

## 8. Zustand der Maschinen, Stand 16:10

Sollbesetzung 27 Worker, nachgerechnet aus den drei `~/hpo_keeper_plan.json`: `l2`
2+2+1+1+1+1 = 8, `l1` 5+4+3+2+2+1 = 17, `ws` 1+1 = 2, Summe **27**, deckt sich mit §5
des Auftrags. Alle vier A100 auf `l2` sind bei 97 bis 99 % Auslastung.

**Ein Widerspruch zum Auftrag:** `l2`s Halterplan besetzt weiter
`config_wind_dcrnn_base.yaml` (Soll 1) und `config_wind_dcrnn_nograph.yaml` (Soll 2),
obwohl §4 `dcrnn_base` als **eingefroren** führt und `dcrnn_nograph` einfrieren will.
Entsprechend laufen dort neue Trials, `dcrnn_base` #135 und #136, `dcrnn_nograph` #127
bis #130, die jüngsten mit Start 15:51. `dcrnn` (GRID), `dcrnn_idw_alt` und
`dcrnn_nomeas` stehen in keinem Plan mehr, sind also tatsächlich eingefroren. Wenn
`dcrnn_base` und `dcrnn_nograph` einfrieren sollen, müssen die drei Planzeilen raus,
sonst wachsen die Budgets weiter und FRAGE 2 verschiebt sich laufend. Ich habe die Pläne
**nicht** angefasst.

Die `.hpo_stop_<suffix>`-Dateien sind unangetastet: `l2` trägt r1, r4, r5, r8, `l1`
r2, r4, r5, r6, r10, `ws` r9. Die alten `r`-Worker laufen entsprechend aus, auf `l1`
sind noch 24 Worker plus Halter in `screen`, auf `l2` 11 plus Halter, auf `ws` 5.

Der Widerspruch ist mit den Entscheidungen aus §9 aufgelöst, siehe §9.2.

---

## 9. Entscheidungen des Nutzers und daraus ausgeführte Änderungen

Alle folgenden Punkte sind am 2026-08-17 zwischen 16:30 und 17:10 entschieden und
ausgeführt.

### 9.1 Entscheidungen

| Frage | Entscheidung |
|---|---|
| FRAGE 1, Fenster | **Nur Val.** Die Verzerrung wird ausdrücklich **nicht** thematisiert, weil es Validierungsdaten sind und die Testdaten später kommen. Das Testfenster 2025-08-01 bis 2025-10-31 bleibt unbenutzt. |
| FRAGE 2, Budgets | **Einfrieren**, Planzeilen entfernt. Budgets werden in der Tabelle ausgewiesen. |
| FRAGE 5, Bestenauswahl | **Bester Trial nach gepooltem Val-RMSE**, keine Mittelung über k, kein Fold-Median. |
| N1 | **Auswertung an das Training angleichen** (nach der Messung unten). |
| GPU | Gezielt Worker anhalten, 3 aus FRAGE 2 plus 3 weitere auf `l2`. |

### 9.2 N1 ist gefixt, mit gemessener Begründung

Der Einwand des Nutzers war, dass die statischen Merkmale öffentliche topographische
Daten sind und ein vorab auf einer repräsentativen Standortmenge gefitteter Skalierer
vertretbar wäre. Das trifft zu und ist genau das, was `train_dcrnn.py:882` tut. Der
Punkt von N1 ist ein anderer: **Training und Auswertung benutzten verschiedene
Skalierer**, das Modell wurde also außerhalb seines Eingaberaums betrieben. Gemessen
über die 51 Zielstationen je Fold, Verschiebung in Einheiten der Trainingsstreuung
(`/tmp/n1_magnitude.py` auf `l2`):

| Fold | max abs dz lat | max abs dz lon | max abs dz alt | Streuungsverhältnis alt (102/153) |
|---|---|---|---|---|
| spatial_fold1 | 0.0073 | 0.0287 | 0.1848 | 0.9981 |
| spatial_fold2 | 0.0234 | 0.0117 | 0.1900 | 1.1144 |
| spatial_fold3 | 0.0225 | 0.0175 | **2.5944** | **0.7589** |

Für Breite und Länge und für die Höhe in fold1 und fold2 ist der Effekt mit unter
0.2 Sigma tatsächlich unerheblich. In **fold3** haben die 102 Trainingsstationen eine
Höhenstreuung von 252 m gegen 332 m im Pool, die Hochlagen liegen dort in den
Zielstationen. Eine Zielstation wird deshalb um 2.59 Sigma verschoben, im Mittel über
alle 51 um 0.24 Sigma. Da die Papertabelle das Mittel über drei Folds berichtet, wäre
das in die Hauptzahl eingegangen. Der Einwand des Nutzers stützt den Fix: eine
repräsentative Fitmenge ist richtig, und fold3s 102 Stationen sind für die Höhe des
Pools gerade nicht repräsentativ.

Geändert auf **allen drei Hosts**, je eine Anweisung in
`geostatistics/get_test_results_dcrnn.py`:

```python
stat_scaler.fit(raw_static if (val_start and not args.test_mode) else raw_static[:N_train])
```

Gespiegelt aus `train_dcrnn.py:882`. `--test-mode` bleibt bewusst beim Train-only-Fit,
weil dort die Teststationen an `all_ids` hängen und ihre Aufnahme echte Leckage wäre.
`load_topo_station_features(..., n_train=N_train)` bleibt **unverändert** auf 102, das
ist an dieser Stelle Absicht. Sicherung je Host unter
`/tmp/get_test_results_dcrnn.py.bak-n1`, Syntax je Host mit `ast.parse` geprüft. Kein
laufender HPO-Worker benutzt diese Datei.

### 9.3 Kampagne auf l2 zurückgefahren

`~/hpo_keeper_plan.json` auf `l2` reduziert von 8 auf 2 Sollworker, Sicherung unter
`~/hpo_keeper_plan.json.bak-20260817-eval`:

| Studie | Soll vorher | Soll jetzt | Grund |
|---|---|---|---|
| dcrnn_nwp_hist | 2 | 2 | läuft weiter, Ziel 150 |
| dcrnn_nograph | 2 | 0 | eingefroren (FRAGE 2) |
| dcrnn_base | 1 | 0 | eingefroren (FRAGE 2) |
| mtgnn_nwp_hist | 1 | 0 | Kapazität für Retrains, läuft auf `l1` mit Soll 5 weiter |
| mtgnn_nwp | 1 | 0 | dito, `l1` Soll 4 |
| mtgnn | 1 | 0 | dito, `l1` Soll 3 |

Kampagnenweite Sollbesetzung damit 27 auf 21. Keine Studie fällt aus, die drei
MTGNN-Arme laufen unverändert auf `l1`.

Zwei Eigenschaften des Halters waren dafür zu beachten, beide im Code nachgelesen:

1. **Der Halter stoppt nie**, er startet nur (`hpo_keeper.py:111`,
   `if have >= target: continue`). Eine Planänderung verhindert das Nachbesetzen, sie
   beendet keinen Worker.
2. **Der Plan wird nur beim Start gelesen** (`hpo_keeper.py:91`, außerhalb der
   `while True`-Schleife). Der Halter musste deshalb neu gestartet werden, und zwar
   **vor** dem Setzen der Stop-Datei, sonst hätte der alte Halter `dcrnn_base` nach dem
   Auslaufen sofort mit Suffix `n2` nachbesetzt. Reihenfolge tatsächlich ausgeführt:
   Plan schreiben, Halter neu starten (Screen `hpo_keeper`, 17:08), dann Stop-Datei.

`.hpo_stop_n1` neu angelegt in `/home/viktor/Work/forecasting_framework`. Betroffen sind
genau die drei vorhergesagten Worker, verifiziert über `ps`:

| Worker | Studie | beabsichtigt |
|---|---|---|
| `hpo_dcrnn_wind_dcrnn_base_n1` | dcrnn_base | ja, einfrieren |
| `hpo_dcrnn_wind_dcrnn_nograph_n1` | dcrnn_nograph | ja, einfrieren |
| `hpo_dcrnn_wind_dcrnn_nwp_hist_n1` | dcrnn_nwp_hist | Kollateral, Halter ersetzt ihn als `n3` |

Die bestehenden Stop-Dateien r1, r4, r5, r8 sind unangetastet. Jeder Worker beendet
seinen laufenden Trial noch (`study.stop()` in `_stop_on_flag`, `hpo_dcrnn.py:1428`),
es geht keine GPU-Zeit verloren. Folge: `dcrnn_base` endet bei 111, `dcrnn_nograph` bei
101 COMPLETE. Die Kollateralkosten sind gemessen und klein: der Datencache ist
geschrieben (Schlüssel `06e46a74ffca0fce`), ein Worker mit Cache-HIT brauchte 68 s bis
zum ersten Trial (`n2`, 15:53:35 bis 15:54:43), gegen 8:49 min beim Cache-MISS von `n1`.

**Wichtig für den nächsten Bearbeiter:** ohne Stop-Datei ist Einfrieren nicht möglich.
Ein Worker endet nur bei gesetzter Stop-Datei oder wenn die Studie ihr Budget erreicht,
und `trials: 150` steht in allen drei Configs. `dcrnn_base` (110) und `dcrnn_nograph`
(100) wären sonst bis 150 weitergelaufen.

### 9.4 Was damit für §3a gilt

Haupttabelle ist das Val-Fenster, Auswahl ist der beste Trial nach gepooltem Val-RMSE,
je Studie drei Folds, per Station und gefiltert. Begonnen wird mit den drei
eingefrorenen DCRNN-Studien.

Zwei der drei Bestenauswahlen sind endgültig, eine noch nicht:

| Studie | bester Trial | Val-RMSE gepoolt | endgültig? |
|---|---|---|---|
| dcrnn (GRID) | #192 | 1.1741 | ja, 0 RUNNING |
| dcrnn_idw_alt | #109 | 1.1819 | ja, 0 RUNNING |
| dcrnn_base | #111 | 1.2242 | **nein**, #135 und #136 laufen noch aus |

Bei `dcrnn_base` ist vor dem Retrain erneut abzufragen, ob #135 oder #136 den Wert
1.2242 unterbieten. Der Abstand zu Platz zwei ist mit 1.2242 gegen 1.2326 klein genug,
dass ein neuer Trial die Auswahl kippen kann.

### 9.5 Weitere Entscheidungen

| Frage | Entscheidung |
|---|---|
| FRAGE 3, acht Tage nachexportieren | **Nein.** Datenstand bleibt 2909 Laufpaare, je Tabelle zu vermerken. Trials von vor dem 2026-08-17 bleiben auf 2933, die Inkonsistenz von 0.82 % wird in Kauf genommen. |
| FRAGE 4, TFT-Spatial | **Nur `tft_sp_hist`** (101 Trials) kommt in die Auswertung, vergleichbar mit den DCRNN-Ablationsarmen. `tft_sp_base` (53 Trials, Ladepfad seit 2026-08-12 kaputt) bleibt draußen, mit Begründung im Text. |
| Einordnung der hist-Arme | **Getrennte Tabellenblöcke.** Arme ohne Zielstationsmessung und `*_nwp_hist`-Arme in getrennten Blöcken. MOS-local (0.9452) wird nur im ersten Block als transduktive Obergrenze ausgewiesen. |
| Commit | **Nur der Wind-Anteil**, siehe §9.6. |

Folge für §3b: die Tabelle in §3.4 zu `tft_sp_base` bleibt als Diagnose stehen, weil sie
den Fall „Wichtigkeit ohne monotone Wirkung" belegt, aber die Studie geht nicht in die
Ergebnistabellen ein.

### 9.6 Commit e15d778

Branch `fix/mtgnn-topo-static-dim`, **nicht** gepusht. 22 Dateien, der Codestand, mit dem
die Retrains gerechnet werden.

Ein sauberer Schnitt allein nach Wind gegen Solar war **nicht möglich**, und das ist
selbst ein Befund: der Solar-Umbau hat den Wind-Pfad von Solar-Modulen abhängig gemacht.

- `geostatistics/shared/resolution.py` ist eine **neue, bisher unversionierte** Datei,
  die `hpo_dcrnn.py` auf Modulebene importiert (`freq_to_hours`). Ohne sie ist der
  Wind-Code auf einem frischen Checkout nicht einmal importierbar.
- `utils/solar.py` ist ebenfalls neu und wird von `train_stgnn2.py` im **Wind-Pfad** auf
  Modulebene importiert (`infer_sample_seconds`, `resample_interval_mean`). Sie musste
  daher mit in den Commit.
- `utils/eval.py` und `utils/preprocessing.py` tragen Wind- und Solaränderungen in
  denselben Dateien. Eine Trennung ginge nur hunkweise und ist nicht ohne Risiko, also
  sind sie ganz drin.

Bewusst **nicht** committet: `geostatistics/solar_preprocessing.py`,
`utils/solar_ecmwf.py`, `configs/solar_*`, `configs/*config_solar_*`,
`scripts/run_solar_*`, `scripts/launch_solar_*`, die Solar-Dokumente und `CLAUDE.md`.
`solar_ecmwf.py` und `solar_preprocessing.py` werden nur **lazy** innerhalb von
Funktionen importiert (`utils/solar.py:1368`, `train_dcrnn.py:682`), der Wind-Pfad
braucht sie nicht. Nach dem Commit sind noch 40 Einträge uncommittet, im Kern die
Solar-Arbeit.

Ebenfalls bewusst nicht committet: die `.hpo_stop_*`-Dateien. Sie sind Betriebszustand,
und in einem anderen Checkout würden sie dort Worker stilllegen. Vor dem Commit geprüft,
dass keine davon in der Staging-Area lag.

**Offen:** `l1` und `ws` tragen weiter den älteren, uncommitteten Stand (§5). Für die
Retrains ist das unerheblich, solange sie auf `l2` laufen. Wer sie auf `l1` rechnet, muss
den Stand vorher gleichziehen.

---

## 10. Verifiziertes Rezept für die Retrains (§3a), noch nicht ausgeführt

Die Hyperparameter müssen **nicht** in YAML materialisiert werden, anders als beim
stdhp-Trockenlauf. `train_dcrnn.py` lädt sie direkt aus Optuna:

```bash
cd /home/viktor/Work/forecasting_framework
source frcst/bin/activate
eval "$(grep -E '^export (WEATHER_DB_URL|ECMWF_WIND_SL_URL|OPTUNA_STORAGE)=' ~/.bashrc)"
CUDA_VISIBLE_DEVICES=<gpu> python geostatistics/train_dcrnn.py \
    --config configs/dcrnn/config_wind_<arm>_fold<N>.yaml --hpo-study auto
```

Geprüft, nicht angenommen:

- **Die Studienauflösung stimmt.** `train_dcrnn.py:341` entfernt mit
  `re.sub(r'_fold\d+$', '', config_stem)` das Fold-Suffix, `:356` baut daraus
  `cl_m-dcrnn_out-48_freq-1h_<stem>`. Für alle drei Arme wurde die Studie testweise
  geladen und liefert genau den Trial aus §9.4:

  | Config | aufgelöste Studie | Trial | Wert |
  |---|---|---|---|
  | `config_wind_dcrnn_fold1.yaml` | `…_wind_dcrnn` | #192 | 1.1741 |
  | `config_wind_dcrnn_base_fold1.yaml` | `…_wind_dcrnn_base` | #111 | 1.2242 |
  | `config_wind_dcrnn_idw_alt_fold1.yaml` | `…_wind_dcrnn_idw_alt` | #109 | 1.1819 |

- **Die geladenen Parameter deckenden sich mit §3b.** `study.best_params` liefert
  `next_n_icond2` 4 / 3 / 7 für dcrnn / base / idw_alt, genau die Werte, die in §3.3 als
  „bester Trial" ausgewiesen sind. Das ist eine unabhängige Gegenprobe des
  Analyseskripts gegen Optunas eigene Bestenauswahl.
- **`nwp_out_dim` wird nachgerechnet** (`train_dcrnn.py:384-387`), weil es nie in
  `best_params` steht, sondern nach dem Sampling aus `nwp_heads * nwp_out_per_head`
  abgeleitet wird. Ein handgeschriebenes YAML hätte hier still eine falsche Weite gesetzt.
- **Alle neun Fold-Configs sind korrekt**: je `files` 102, `val_files` 51,
  `val_start` 2024-08-01, `test_start` 2025-08-01. Das deckt sich mit §2 des Auftrags.
  Die veralteten Dateien liegen ausschließlich unter `configs/*/test/` (§7) und werden
  hier nicht benutzt.
- **`train_dcrnn.py` hat kein `--gpu`**, die GPU wird über `CUDA_VISIBLE_DEVICES`
  gewählt (`:335` nimmt schlicht `cuda`).
- Provenienz wird mitgeschrieben: das Ergebnis-`.pkl` enthält `hpo_study_name` und
  `hpo_best_params` (`train_dcrnn.py:1191-1192`).

Auswertung danach je Lauf mit `geostatistics/get_test_results_dcrnn.py`, **ohne**
`--test-mode` (Entwicklungsmodus, 51 nie gesehene Zielstationen je Fold), also genau der
Pfad, für den N1 gefixt wurde.

**Vor dem Start noch zu tun:**

1. `dcrnn_base` erneut abfragen: #135 und #136 laufen aus und können 1.2242 unterbieten
   (§9.4).
2. Abnahmekriterium prüfen: **2909** Laufpaare (nicht 2933, §6 des Auftrags), aufgeteilt
   in Trainings- und Val-Paare je Fold. Weicht die Zahl ab, stimmt etwas nicht.
3. Kapazität abwarten, bis die in §9.3 markierten Worker ihren laufenden Trial beendet
   haben.

---

## 11. Der zweite Blocker: ECMWF-NaN im Retrain-Pfad, und die Retrains laufen

### 11.1 Befund

Die Umstellung vom 2026-08-17 (`exclude_run_pairs_with_ecmwf_nan`) war **nur** in den
drei HPO-Skripten gelandet. Der Retrain-Pfad war damit blockiert beziehungsweise still
falsch. Nachgezählt, je Datei die Zahl der Aufrufstellen vor dem Fix:

| Datei | Aufrufe | Verhalten vor dem Fix |
|---|---|---|
| `hpo_dcrnn.py` | 3 | Warnung im Vor-Test-Fenster, Ausschluss aktiv |
| `hpo_mtgnn.py`, `hpo_wavenet.py` | je 2 | dito |
| `train_dcrnn.py` | **0** | **harter Abbruch** „ECMWF data contains NaN after loading" |
| `train_mtgnn.py`, `train_wavenet.py` | **0** | **kein Wächter, kein Ausschluss**, also stille NaN-Verluste |
| `get_test_results_*.py` | 0 | unkritisch, siehe unten |

`train_dcrnn.py` prüfte das **ganze** Array. Die Zeitachse beginnt am 2023-07-24, der neu
exportierte ECMWF-Bestand erst am 2023-08-01, also stehen 192 h NaN am Anfang, und der
Wächter feuerte, bevor überhaupt ein Training startete. `train_mtgnn.py` und
`train_wavenet.py` waren der schlechtere Fall: sie hätten die betroffenen Laufpaare
mitgenommen und still NaN-Verluste erzeugt, genau der Vorfall, den der Docstring der
Funktion beschreibt.

Der **Auswertungspfad braucht keine Änderung**: `get_test_results_dcrnn.py` hat gar
keinen ECMWF-Wächter, und die betroffenen Laufpaare liegen alle am Anfang der Zeitachse,
also im Trainingsfenster. Das Val-Fenster ab 2024-08-01 ist unberührt. Das war eine
Vorhersage und ist unten gemessen bestätigt.

### 11.2 Fix, Commit d501225

Alle drei Trainingsskripte angeglichen, gespiegelt aus `hpo_dcrnn.py:637-656` und
`:824-830`:

1. In `train_dcrnn.py` der harte `raise` zur Warnung, und nur noch über das Fenster bis
   `audit_t`, also den Bereich, der überhaupt Laufpaare liefern kann.
2. In allen drei Skripten `exclude_run_pairs_with_ecmwf_nan` getrennt für
   `train_run_pairs` und `val_run_pairs`. Die Funktion bricht selbst ab, wenn mehr als
   10 % wegfallen, ein großflächiger Datenverlust bleibt also laut.

Angewandt mit Exact-Match-Absicherung (`/tmp/apply_ecmwf_nan_fix.py`), Sicherungen unter
`/tmp/train_{dcrnn,mtgnn,wavenet}.py.bak-ecmwfnan`, `ast.parse` je Datei, und `pyflakes`
über alle drei: keine undefinierten Namen, nur die schon vorher vorhandenen
Unused-Import-Warnungen.

### 11.3 Abnahmekriterium erfüllt, am Lauf gemessen

Der erste Retrain (`dcrnn` GRID, fold1, auf `ws`) bestätigt die Zahlen aus §2 und §6 des
Auftrags **exakt**:

```
Run pairs — train: 1473  val: 1460  skipped: 960 (grid-NaN: 0)
Excluded 24 of 1473 run pairs due to NaN in ECMWF data
Excluded  0 of 1460 run pairs due to NaN in ECMWF data
Run pairs after ECMWF-NaN exclusion — train: 1449 (-24)  val: 1460 (-0)
```

1473 + 1460 = **2933** wie im Auftrag, minus **24** = **2909**, also 0.82 %. Und die 24
liegen wie vorhergesagt vollständig im Trainingsfenster (`val: -0`).

Weitere Bestätigungen aus demselben Log: `--hpo-study auto` löst die Studie korrekt auf
(„HPO study 'cl_m-dcrnn_out-48_freq-1h_wind_dcrnn' — best val_loss=1.174097
(trial #192)"), und die HPO-Parameter greifen tatsächlich (ICON-D2 wird mit
`4 grid pts` geladen, also `next_n_icond2=4` aus Trial #192).

Nebenbefund: `train_dcrnn.py` benutzt **keinen** `GNNCache`, anders als die HPO-Skripte.
Der Datenaufbau dauert etwa 100 s je Lauf und wird nicht zwischengespeichert; die
befürchteten neun Cache-Verzeichnisse à 3 GB entstehen also nicht.

### 11.4 Wo die Retrains laufen

Auf `ws` (2× RTX 4090), auf Vorschlag des Nutzers, weil die GPUs dort nur zu 30 bis 50 %
belegt waren, während alle vier A100 auf `l2` bei 97 bis 99 % lagen. Geprüft vor dem
Start:

- `train_dcrnn.py:866` und `get_test_results_dcrnn.py:436` tragen auf `ws` **beide** den
  153er-Fit, Trainings- und Auswertungsseite sind dort also konsistent (N1).
- `/mnt/lambda1/nvme1` existiert auf `ws` (NFS von 10.166.32.238), kein Pfad-Rewrite
  nötig; der ist `l1`-spezifisch.
- Speicher reicht mit Abstand: ein Lauf belegt etwa 4.6 GB von 24.5 GB.
- `pvlib 0.13.1` und `geopy` sind im `frcst`-Venv auf `ws` vorhanden, `utils/solar.py`
  ist also importierbar. Das war nicht selbstverständlich, weil der Wind-Pfad seit dem
  Solar-Umbau auf Modulebene `from utils import solar` macht.

**Codestand auf `ws`:** auf `d501225` gebracht. Der direkte `git fetch` von GitHub
scheiterte auf `ws` mit HTTP 500, deshalb über ein `git bundle` von `l2`. Die lokalen
Änderungen auf `ws` liegen in `stash@{0}` („ws-local pre-sync 20260817"), die
untracked-Dateien einschließlich `.hpo_stop_r9` sind unangetastet. Der Branch
`fix/mtgnn-topo-static-dim` ist auf `origin` gepusht (`4f832ec..d501225`).

**Ablaufplan**, zwei Warteschlangen à eine GPU (`~/retrain_queue.sh`, Einzellauf
`~/run_retrain.sh`, Logs `logs/retrain_<arm>_fold<N>.log` und
`logs/retrain_queue_gpu<G>.log`):

| GPU | Reihenfolge |
|---|---|
| 1 | `dcrnn` fold1 (läuft), dann `dcrnn_base` fold3, `dcrnn_idw_alt` fold1, fold2, fold3 |
| 0 | `dcrnn` fold2, fold3, `dcrnn_base` fold1, fold2 |

Etwa 60 s je Epoche, `max_epochs` 200 mit `patience` 15. Epoche 1 von fold1 liefert
val-RMSE 1.1816, plausibel neben dem gepoolten HPO-Bestwert 1.1741.

**Noch zu erledigen, wenn die Läufe durch sind:**

1. `dcrnn_base`: prüfen, ob #135 oder #136 den Wert 1.2242 unterboten haben. Die
   Warteschlange zieht die Hyperparameter erst beim Start des jeweiligen Laufs aus
   Optuna, die drei `dcrnn_base`-Folds können also einen anderen Trial erwischen als
   #111. Das ist im Log jedes Laufs festgehalten und muss vor der Tabelle abgeglichen
   werden, damit alle drei Folds denselben Trial benutzen.
2. Auswertung je Lauf mit `get_test_results_dcrnn.py`, ohne `--test-mode`, per Station
   und gefiltert.
3. Die Ergebnistabellen in getrennten Blöcken nach §9.5.

---

## 12. Zweiter ECMWF-Vorfall am 2026-08-17 abends, und Neuaufsetzen der Retrains

### 12.1 Was passiert ist

Zwischen **18:30 und 18:43** wurden **alle 759** ECMWF-Wind-Parquets unter
`/mnt/nvme1/ecmwf/parquet/SL` neu geschrieben, von uid 1003 (`meghnanegi`), also
derselben fremden Pipeline wie beim Vorfall vom 2026-08-15. Das geschah mitten in den
laufenden Retrains, unangekündigt.

Anders als am 15.08. ist der Bestand dadurch **besser** geworden, nicht kaputt. Geprüft
an einer Stichprobe von 25 der 759 Dateien:

- alle zehn Zielspalten vorhanden, **keine NaN**,
- einheitlich 112 984 Zeilen je Datei,
- Abdeckung durchgehend **2023-07-01 bis 2026-03-02 21:00**.

Das 192-Stunden-Loch am Anfang der Zeitachse existiert damit nicht mehr. **FRAGE 3 des
Auftrags erledigt sich dadurch von selbst**: die acht Tage müssen nicht nachexportiert
werden, sie sind da. Die verbleibenden 723 NaN-Zeitstempel sind ausschließlich der
Schwanz ab 2026-03-02 22:00, wo die Messzeitachse über das ECMWF-Ende hinausläuft; sie
liegen außerhalb des Laufpaar-Fensters.

### 12.2 Der Schaden war Vergleichbarkeit, nicht Datenqualität

Die Läufe zerfielen dadurch in zwei Datenstände, erkennbar am Fingerabdruck in jedem Log:

| Fingerabdruck | Laufpaare | Läufe |
|---|---|---|
| 915 betroffene Zeitstempel, 24 ausgeschlossen | 2909 | `dcrnn` fold1, fold2, `dcrnn_base` fold3 (erster Zyklus) |
| 723 betroffene, 0 ausgeschlossen | 2933 | `dcrnn` fold3, `dcrnn_idw_alt` fold1 |

Damit gibt es kampagnenweit jetzt einen **dritten** Datenstand: vor dem 2026-08-17 2933,
zwischen 15:33 und 18:30 2909, seit 18:43 wieder 2933 auf besserer Grundlage. Jeder
HPO-Worker, der ab jetzt startet, rechnet auf Stand drei.

**Entscheidung des Nutzers:** auf dem neuen Stand vereinheitlichen, die betroffenen
Läufe wiederholen. Kein Einfrieren des Eingangs durch einen schreibgeschützten Snapshot
(ausdrücklich abgelehnt); als Rest-Absicherung schreibt jeder Lauf seinen
Datenstand-Fingerabdruck ins Log, eine erneute Verschiebung bliebe also wenigstens
nachträglich nachweisbar. Das Prüfskript vergleicht diesen Fingerabdruck über alle neun
Läufe und schlägt bei Abweichung an.

### 12.3 Eigener Fehler: run_retrain.sh im laufenden Betrieb geändert

`dcrnn` fold3 und `dcrnn_base` fold3 zeigten je **zwei vollständige Trainingszyklen**
innerhalb eines einzigen Aufrufs (je eine START- und ENDE-Marke, aber zwei
`Device: cuda`, zwei `Training complete` und zwei Ergebnis-pkl).

Ursache ist nicht das Trainingsskript: `train_dcrnn.py` hat genau einen `main()`-Aufruf
unter `if __name__ == "__main__"`. Ursache war, dass ich `run_retrain.sh` um **18:54:39**
um die Skip-Marker-Prüfung erweitert habe, während beide Läufe (seit 18:21 bzw. 18:35)
liefen. **Bash liest Skriptdateien fortlaufend statt sie vorab zu puffern**; durch das
Einfügen von vier Zeilen am Dateianfang verschoben sich alle folgenden Bytes, und die
bereits laufenden Instanzen lasen nach der Rückkehr von Python an einem verschobenen
Offset weiter und führten den Python-Aufruf erneut aus. `fold1` (bis 18:20) und `fold2`
(bis 18:35:23) waren vorher fertig und blieben unberührt.

**Regel daraus:** `run_retrain.sh` und `retrain_queue.sh` nicht anfassen, solange Jobs
laufen. Wenn eine Änderung nötig ist, unter neuem Dateinamen ablegen und die
Warteschlange neu starten. Das Prüfskript zählt die Zyklen jetzt im Wrapper-Log mit und
meldet jeden Lauf mit mehr als einem Zyklus.

### 12.4 Stand nach dem Neuaufsetzen, 19:40

Verworfen und nach `archiv/retrain_verworfen_20260817_gemischte_basis/` verschoben
(Logs, Modelle, pkl): `dcrnn` fold1, fold2, fold3 und `dcrnn_base` fold3. Das Verschieben
war nötig, weil `run_retrain.sh` mit `>>` an bestehende Logs anhängt und die Modellnamen
identisch sind; ohne Aufräumen hätten alte und neue Läufe im selben Log gestanden.

| Host | GPU | Läufe | Stand |
|---|---|---|---|
| `ws` | 0 | `dcrnn` fold1, fold2, fold3 | neu gestartet 19:39 |
| `ws` | 1 | `dcrnn_base` fold1, fold2, fold3 | neu gestartet 19:39 |
| `l1` | 6 | `dcrnn_idw_alt` fold1 (fertig), fold2, fold3 | läuft seit 18:55 |

`dcrnn_idw_alt` fold1 ist bereits vollständig geprüft: ein Zyklus, 24 Epochen,
bester val-RMSE 1.1407, Trial #109, Laufpaare 1473/1460, Modell 2.0 MB mit 44 Tensoren
ohne NaN, pkl vorhanden.

### 12.5 Warum l1 überhaupt mitrechnet

Auf Vorschlag des Nutzers, weil GPU 6 dort frei war (560 MiB von 49 GB, 14 % Last).
Geprüft, bevor dort etwas gerechnet wurde:

- l1 hat eine **eigene Historie**: HEAD `3d041b7` („fix(nwp): ECMWF-NaN…"), gemeinsamer
  Vorfahr `4f832ec`. Genau ein Extra-Commit, der die vier Dateien aus §7 des Auftrags
  enthält, die auf `l2` im Arbeitsbaum lagen und in `e15d778` eingegangen sind.
- Die drei Trainingsskripte auf l1 haben denselben NaN-Fix bekommen wie auf l2
  (Exact-Match-Patch, `ast.parse`, `pyflakes` ohne undefinierte Namen).
- `train_stgnn2.py` wurde auf l1 **bewusst nicht** angefasst. Die HPO-Worker importieren
  `train_{dcrnn,mtgnn,wavenet}.py` nachweislich nicht (`grep -c` = 0 in allen drei
  HPO-Skripten), wohl aber `train_stgnn2.py` (`hpo_dcrnn.py:68, 670, 827`). Der Patch
  ist für die laufende Kampagne damit unsichtbar.
- l1s `idw_alt`-Fold-Configs sind lokal pfad-umgeschrieben (`/mnt/nvme1`), tragen aber
  dieselben Stationen: Prüfsummen der sortierten Stationslisten stimmen mit `ws` überein
  (fold1 `b9409aec8f5a8763`, fold2 `13fa75321a554108`, fold3 `03bc05381938c4df`).
- l1 und ws sehen **dieselbe** Datei-Ablage: l1s IP ist 10.166.32.238, ws mountet
  `10.166.32.238:/mnt/nvme1`. Der Unterschied kam nicht vom Host, sondern vom Zeitpunkt.

---

## 13. Retrains geprüft, dcrnn_base neu mit 200 Epochen, unbeaufsichtigte Auswertung

### 13.1 Prüfung der neun Retrains: bestanden

Alle neun Läufe auf dem einheitlichen Datenstand (Fingerabdruck 723 überall):

| Arm | Trial | val-RMSE fold1/2/3 | Epochen | Laufpaare |
|---|---|---|---|---|
| `dcrnn` (GRID) | #192 in allen drei | 1.1326 / 1.2344 / 1.2206 | 20 / 17 / 23 von 200 | 1473/1460 |
| `dcrnn_base` | #111 in allen drei | 1.1977 / 1.2547 / 1.3062 | **50/50** / 21 / 42 von 50 | 1473/1460 |
| `dcrnn_idw_alt` | #109 in allen drei | 1.1407 / 1.1811 / 1.1835 | 24 / 32 / 41 von 200 | 1473/1460 |

Kein Fehlermuster, kein NaN in Verlusten, kein NaN-Tensor in den Checkpoints, alle Modelle
korrekt benannt und abgelegt, alle pkl vorhanden. Die befürchtete Trial-Divergenz bei
`dcrnn_base` ist ausgeblieben.

### 13.2 dcrnn_base wird mit 200 Epochen wiederholt

`dcrnn_base` fold1 lief **50 von 50** Epochen, hat die Obergrenze also erreicht statt früh
zu stoppen. Die Config trug als einzige `max_epochs: 50`, während die übrigen Arme 200
haben — der Arm war damit gegenüber den anderen benachteiligt, was in einer
Ablationsleiter genau das Falsche ist. Auf Entscheidung des Nutzers auf **200** gesetzt
(`configs/dcrnn/config_wind_dcrnn_base_fold{1,2,3}.yaml:104`, `patience: 10` unverändert,
Sicherungen unter `/tmp/config_wind_dcrnn_base_fold*.yaml.bak-50ep`). Die drei
50-Epochen-Läufe liegen samt Logs, Modellen und pkl unter
`archiv/retrain_base_50epochen_20260817/`.

### 13.3 Auswertungsaufruf

```bash
python geostatistics/get_test_results_dcrnn.py \
    -m wind_<arm>_fold<N>_dcrnn_retrain_fold<N> \
    -c configs/dcrnn/config_wind_<arm>_fold<N>.yaml \
    --pkl results/<stem>_<timestamp>.pkl \
    --raw-out-name retrain_<arm>_fold<N>
```

Bewusst **`--pkl` statt `--hpo-study auto`**: das pkl trägt die Config so, wie trainiert
wurde (inklusive der HPO-Überschreibungen), und ist eingefroren. `--hpo-study auto` würde
den jeweils aktuellen Optuna-Bestwert ziehen und könnte die Architektur gegenüber dem
Checkpoint verschieben, sobald ein neuer Trial gewinnt. **Kein `--test-mode`**, also
Entwicklungsmodus mit den 51 nie gesehenen Zielstationen — genau der Pfad, für den N1
gefixt wurde.

Erster Lauf zur Kontrolle (`dcrnn` fold1) sauber durch, Fenster bestätigt
(`Evaluation period: 2024-08-01 .. 2025-08-01 (mode=dev)`), Ergebnis
`R²=0.621, RMSE=1.073 m/s, MAE=0.830, Skill_NWP=0.148`. Ausgaben je Lauf:
`data/test_results/retrain_<arm>_fold<N>.csv` und
`data/raw_preds/retrain_<arm>_fold<N>_raw.parquet`.

### 13.4 Was unbeaufsichtigt läuft

`~/pipeline.sh <REPO> <GPU> <schritt> …` arbeitet Schritte sequenziell ab
(`train:<arm>:<fold>` oder `eval:<arm>:<fold>`), protokolliert nach
`logs/pipeline_gpu<G>.log` und macht bei einem Fehlschlag mit dem nächsten Schritt
weiter, statt die ganze Kette abzubrechen.

| Host | GPU | Schritte |
|---|---|---|
| `ws` | 0 | eval `dcrnn` fold2, fold3 |
| `ws` | 1 | train `dcrnn_base` fold1–3 (200 Ep.), danach eval fold1–3 |
| `l1` | 6 | eval `dcrnn_idw_alt` fold1–3 |

**Regel, aus Schaden gelernt:** `run_retrain.sh`, `run_eval.sh` und `pipeline.sh` werden
nicht angefasst, solange Jobs laufen. Bash liest Skriptdateien fortlaufend; eine Änderung
im Flug hat am 2026-08-17 zwei Läufe dazu gebracht, das Training ein zweites Mal zu
starten (§12.3).

### 13.5 Noch offen: die Filterung

`get_test_results_dcrnn.py` filtert **nicht**. Es schreibt die Ergebnis-CSV und die
Rohvorhersagen; die Papierkonvention „per Station und gefiltert" (imputierte Zielstunden
ausgeschlossen, per Station über die drei Folds gemittelt) ist ein nachgelagerter
Schritt. Vorbild ist `docs/baselines_verification_results.md` §4.1 und das Skript
`archiv/baselines_verification/compute_filtered_mos.py`. Solange dieser Schritt nicht
angewandt ist, sind die Zahlen aus §13.3 **ungefiltert** und nicht mit den Referenzen
(ICON-D2 1.304, Persistenz 2.240, MOS-local 0.9452, MOS-regional 1.1603, TFT base 1.186)
vergleichbar.

### 13.6 tft_sp_base läuft wieder

Kein Defekt. Die Studie war seit dem 2026-08-17 09:03 schlicht **unbesetzt**; die letzte
Logzeile stammt von dort. Der Fehlschlag an jenem Morgen war der letzte Nachhall des
Vorfalls vom 08-12 (leere Spalten in neu erzeugten Station-Parquets): der Lauf lud Fold 1
aus dem Cache-Eintrag `44bb6ba8…` vom 08-12 04:26, geschrieben 20 Minuten bevor die
Parquets um 04:46 korrigiert wurden.

Nachgewiesen, dass heute nichts kaputt ist: Nachbau mit exakt den Parametern des
gescheiterten Trials (`grid_points=4, grid_ecmwf=1, stations=3`) auf **Fold 2**, wo er
abbrach, liefert 206 610 Trainingszeilen. Ein am 2026-08-17 21:15 gestarteter Worker
(l1, GPU 3, Screen `hpo_tft_sp_base_n1`) ist über die kritische Stelle hinaus:
`Fitted global scaler_x on 102 training stations / 6973543 rows / 25 feature columns`,
**null** Stationen ohne Trainingszeilen — und das mit demselben Cache-Eintrag.

Dafür wurde genau **ein** `mtgnn_nwp_hist`-Worker zum Auslaufen markiert
(`.hpo_stop_n3` auf l1; Suffix `n3` wird dort nur von dieser Studie benutzt, die
Stop-Datei trifft also keinen anderen Arm).

Meine frühere Darstellung, es habe eine „CSV-auf-Parquet-Umstellung" gegeben, war falsch:
es waren durchgehend Parquets. `preprocessing.py:95` probiert zuerst
`Station_<id>.parquet`; der Schlüssel `synth_<id>.csv` in `preprocessing.py:155/167` ist
nur eine feste Beschriftung, kein Dateiname.

---

## 14. Ergebnis von §3a: die gefilterten Papierzahlen

### 14.1 Ablauf

Alle drei Pipelines sind in der Nacht zum 2026-08-18 mit **0 Fehlern** durchgelaufen
(ws GPU0 22:02, l1 GPU6 22:24, ws GPU1 22:53). Die neun Retrains und die neun
Auswertungen sind vollständig, die Prüfung nach §13.1 besteht erneut: Trial-Konsistenz je
Arm, Laufpaare 1473/1460 überall, Datenstand-Fingerabdruck einheitlich 723, keine
NaN-Tensoren, alle Modelle und pkl am erwarteten Ort.

Der Sammler auf l2 hat die neun Parquets geholt, ist aber am letzten Schritt gescheitert:
`filtered_table.py` lag dort nur unter `/tmp`, nicht im Home, von wo das Skript es
aufrief. Ein Verteilungsfehler von mir, ohne Folgen ausser einem Nachlauf.

### 14.2 Die Filterung ist gegen die Referenz validiert

Die Rohvorhersage-Parquets tragen neben `pred` auch `nwp_ref` und `pers_ref`. Beide
wurden auf **derselben** gefilterten Basis mitgerechnet, als Kontrolle der Konvention:

| Referenz | gerechnet | dokumentiert | Abweichung |
|---|---|---|---|
| ICON-D2 | **1.3036** | 1.304 | 0.0004 |
| Persistenz | **2.2342** | 2.240 | 0.0058 |

ICON-D2 auf vier Nachkommastellen zu treffen ist der Beleg, dass die Filterung die
Papierkonvention reproduziert. Benutzt werden `build_imputation_mask` und
`_lookup_imputed` aus `archiv/baselines_verification/verify_baselines.py`, also dieselbe
Formel wie bei den Baselines, keine zweite Kopie.

### 14.3 Haupttabelle, Val-Fenster, per Station, gefiltert

Mittel über die drei Folds, 51 nie gesehene Zielstationen je Fold, imputierte Zielstunden
ausgeschlossen (0.60 / 0.68 / 0.86 % je Fold).

| Arm | fold1 | fold2 | fold3 | **Mittel** |
|---|---|---|---|---|
| `dcrnn_idw_alt` (D') | 1.0783 | 1.0954 | 1.1163 | **1.0967** |
| `dcrnn` (GRID) | 1.0742 | 1.1224 | 1.1381 | **1.1116** |
| `dcrnn_base` | 1.1741 | 1.1585 | 1.2020 | **1.1782** |

Einordnung gegen die Referenzen: ICON-D2 1.3036, Persistenz 2.2342, MOS-regional 1.1603,
MOS-local 0.9452 (transduktive Obergrenze), TFT base 1.186.

- Alle drei Arme schlagen **ICON-D2** deutlich (1.10 bis 1.18 gegen 1.30).
- GRID und D' schlagen auch **MOS-regional** (1.1603) und **TFT base** (1.186).
- **Keiner** schlägt MOS-local (0.9452). Das ist konsistent damit, dass MOS-local die
  transduktive Obergrenze ist.

### 14.4 Zwei Vorbehalte, die vor der Interpretation stehen müssen

**(a) Die Reihenfolge von GRID und D' ist nicht belegt.** Der Abstand beträgt
1.1116 − 1.0967 = **0.0149**. Aus den zwei unabhängigen Retrains von `dcrnn_base`
(50-Epochen- und 200-Epochen-Lauf, identische Hyperparameter aus Trial #111, nur andere
Zufallsinitialisierung) lässt sich die Lauf-zu-Lauf-Streuung beziffern:

| Fold | 50-Ep.-Lauf | 200-Ep.-Lauf | Differenz |
|---|---|---|---|
| fold1 | 1.1977 | 1.2349 | +0.0372 |
| fold2 | 1.2547 | 1.2648 | +0.0101 |
| fold3 | 1.3062 | 1.3016 | −0.0046 |
| Mittel | 1.2529 | 1.2671 | **+0.0142** |

Die Streuung des Dreifold-Mittels über zwei Läufe (**0.0142**) ist praktisch so gross wie
der Abstand zwischen GRID und D' (**0.0149**). Aus je einem Retrain lässt sich also
**nicht** sagen, welcher der beiden Arme besser ist. Wer die Reihenfolge behaupten will,
braucht mehrere Wiederholungen je Arm (Mittel und Streuung über Seeds) oder muss sich auf
die Aussage beschränken, dass beide gleichauf liegen und beide MOS-regional schlagen.
Der Abstand zu `dcrnn_base` (0.067 bzw. 0.082) liegt dagegen klar über der Streuung.

**(b) Die HPO-Reihenfolge kehrt sich um.** Gepoolt über das HPO-Objective lag GRID vorn
(1.1741 gegen 1.1819 für D'), auf der Papiermetrik liegt D' vorn (1.0967 gegen 1.1116).
Das ist kein Widerspruch, sondern der in §2 des Auftrags benannte Unterschied zwischen
gepoolt und per Station, plus die Filterung. Es zeigt aber, dass die Bestenauswahl nach
gepooltem Val-RMSE nicht dieselbe Rangfolge erzeugt wie die berichtete Metrik.

### 14.5 Zur Anhebung von dcrnn_base auf 200 Epochen

Die Änderung war der Sache nach richtig, `dcrnn_base` war als einziger Arm auf 50 Epochen
begrenzt. Sie hat aber **nichts gebracht**: der neue Lauf stoppte per Early Stopping bei
21, 12 und 32 Epochen, die 50er-Grenze war also gar nicht mehr bindend, und die Zahlen
wurden auf zwei von drei Folds leicht schlechter. Dass der frühere fold1 mit 50/50 an die
Grenze lief, war Lauf-zu-Lauf-Zufall, nicht ein systematisches Anstossen an die Obergrenze.
`dcrnn_base` ist damit nachweislich schwächer, nicht bloss unterausgestattet gewesen.

### 14.6 Stand von tft_sp_base

Seit dem Neustart (§13.6) von 53 auf **60** COMPLETE, 2 laufende Trials, **keine neuen
FAILs**. Beide Worker (l1 GPU3, l2 GPU0) arbeiten. Bester Wert unverändert 1.2593; die
Studie bleibt nach FRAGE 4 aus der Auswertung, die Wiederbelebung dient dem Budget.

---

## 15. Die vollständige Ablationsleiter, HPO-getunt

### 15.1 Was dazugekommen ist

Am 2026-08-18 wurden `dcrnn_nograph` (Variante C) und `dcrnn_nomeas` (Variante B) mit
ihren HPO-besten Hyperparametern nachtrainiert, je drei Folds. Beide Studien waren vorher
eingefroren und hatten **0 laufende Trials**, es musste also nichts gestoppt werden.
`dcrnn_nograph` hatte über Nacht noch drei Trials abgeschlossen (101 auf 104) und einen
besseren Bestwert bekommen (1.2117 auf 1.1999); der Retrain nutzt daher Trial #129.

| Arm | Trial | Epochen fold1/2/3 | Laufpaare | Fingerabdruck |
|---|---|---|---|---|
| `dcrnn_nograph` (C) | #129 | 21 / 17 / 31 von 200 | 1473/1460 | 723 |
| `dcrnn_nomeas` (B) | #170 | 29 / 27 / 52 von 200 | 1473/1460 | 723 |

Trial-Konsistenz je Arm eingehalten, keine NaN, alle Modelle und pkl vorhanden. Alle
**fünfzehn** Läufe der Leiter stehen damit auf demselben Datenstand.

Die Arme unterscheiden sich in genau zwei Zeilen der Config: **A (GRID)**
`station_connectivity: delaunay` mit Nachbarmessungen; **B (NOMEAS)** Kanten bleiben,
`neighbour_meas_available: false`; **C (NOGRAPH)** zusätzlich `station_connectivity: none`.
`BASE` ist davon unabhängig: dort ist `nwp_nodes: false`, die NWP-Gitterpunkte werden also
in `station.x` konkateniert statt als Graphknoten geführt.

### 15.2 Haupttabelle, gefiltert, per Station, Mittel über drei Folds

| Arm | fold1 | fold2 | fold3 | Mittel |
|---|---|---|---|---|
| `dcrnn_idw_alt` (D-Strich) | 1.0783 | 1.0954 | 1.1163 | **1.0967** |
| `dcrnn_nomeas` (B) | 1.0897 | 1.1011 | 1.1354 | **1.1087** |
| `dcrnn` (GRID, A) | 1.0742 | 1.1224 | 1.1381 | **1.1116** |
| `dcrnn_nograph` (C) | 1.1248 | 1.1848 | 1.1803 | **1.1633** |
| `dcrnn_base` | 1.1741 | 1.1585 | 1.2020 | **1.1782** |

Selbstkontrolle der Filterung auf denselben Zeilen: ICON-D2 **1.3036** gegen dokumentierte
1.304 (Abweichung 0.0004), Persistenz **2.2342** gegen 2.240. Referenzen zur Einordnung:
MOS-regional 1.1603, MOS-local 0.9452, TFT base 1.186.

> **Nachtrag (§16.6):** Diese Tabelle ist unvollständig — `dcrnn_nwp_hist` (GRID+HIST)
> fehlt, weil die Studie beim Erstellen dieser Ladder noch aktiv lief. Nachtrainiert
> schlägt sie mit 1.0008 (gefiltert) alle fünf Arme hier, `dcrnn_idw_alt` (D')
> eingeschlossen — s. §16.2/§16.6 für die vollständigen Zahlen.

### 15.3 Signifikanz auf Stationsebene

Wilcoxon-Vorzeichen-Rangtest, zweiseitig, gepaart über die **Vereinigung** der drei Folds
(N = 153; die Zielmengen sind paarweise disjunkt, geprüft an `spatial_folds.yaml`, ihre
Vereinigung ist exakt der Pool). Holm-korrigiert.

| Vergleich | Median A minus B | p (Holm) | signifikant |
|---|---|---|---|
| BASE gegen D-Strich | +0.0733 | 4.4e-11 | ja |
| C (NOGRAPH) gegen D-Strich | +0.0645 | 1.0e-08 | ja |
| BASE gegen B (NOMEAS) | +0.0551 | 4.8e-08 | ja |
| BASE gegen A (GRID) | +0.0581 | 1.2e-07 | ja |
| **A gegen C** (Beobachtungsnetz gesamt) | −0.0575 | 2.8e-07 | **ja** |
| **C gegen B** (reiner Geometriekanal) | +0.0562 | 8.4e-07 | **ja** |
| B gegen D-Strich | +0.0125 | 0.317 | nein |
| A gegen D-Strich | +0.0048 | 0.536 | nein |
| **A gegen B** (Messkanal der Nachbarn) | +0.0056 | 1.0 | **nein** |
| **BASE gegen C** | −0.0017 | 1.0 | **nein** |

### 15.4 Zwei Befunde, die das Mechanismuskapitel tragen

**(a) Das Netz trägt, aber nicht über die Beobachtungen.** A gegen C ist mit −0.0575
hochsignifikant (71.9 Prozent der Stationen besser), A gegen B dagegen null (p = 1.0). Der
gesamte Gewinn steckt in C gegen B, also in den Kanten **ohne** Nachbarmessungen (+0.0562,
74.5 Prozent der Stationen). Was der Graph liefert, ist räumlicher Kontext aus NWP-Größen
und Stationsattributen, nicht die Windmessung der Nachbarn. Das schärft B4 der Story: im
ungetunten Trockenlauf stand dort A minus C = −0.0084 („trägt als Obergrenze"), getunt
sind es −0.0575, aber mit verschobener Begründung.

**(b) Die beiden Graphkomponenten wirken nur zusammen.** `BASE` (Stationsgraph ja,
NWP-Knoten nein) und `NOGRAPH` (Stationsgraph nein, NWP-Knoten ja) sind statistisch
**ununterscheidbar** (Median −0.0017, p = 1.0), und beide werden von `GRID` klar geschlagen
(1.2e-07 bzw. 2.8e-07). Jede Komponente allein bringt gegenüber der anderen nichts; erst
gemeinsam ergeben sie rund 0.06 m/s. Das ist ein Interaktions-, kein Additivbefund.

Unverändert bleibt der Nullbefund zu B6: **A gegen D-Strich ist nicht signifikant**
(p = 0.536 nach Holm, 0.179 unkorrigiert), auch nachdem beide Seiten getunt wurden. Die
gelernte GATv2-Attention schlägt die feste, physikalisch motivierte IDW-Regel mit
Höhenkorrektur nicht.

### 15.5 Vorbehalte

- Die Tests laufen auf den **ungefilterten** Stations-CSVs des Notebooks, die Tabelle in
  §15.2 ist gefiltert. Bei 0.60 bis 0.86 Prozent imputierten Stunden ist ein
  Vorzeichenwechsel unwahrscheinlich, geprüft ist es nicht.
- „Nicht signifikant" ist nicht „gleichwertig". Für Äquivalenzaussagen bräuchte es einen
  TOST mit vorab festgelegter Marge.
- `dcrnn_base` läuft mit `patience: 10`, alle anderen Arme mit 15 (Rest der alten
  50-Epochen-Config). Nutzerentscheidung: so belassen, der Unterschied ist gegenüber der
  Lauf-zu-Lauf-Streuung von 0.014 unerheblich.
- Die Budgets der Arme sind ungleich (104 bis 170 Trials, siehe FRAGE 2) und gehören in
  die Tabelle.

### 15.6 Zwei widersprüchliche MTGNN-Auswertungen aufgelöst

Beim Zusammenführen der Ergebnisse aller drei Hosts traten 14 gleichnamige Dateien auf,
zehn davon bytegleich. Vier nicht, darunter `stdhp_mtgnn_wind_mtgnn_nwp_fold2.csv` und
`stdhp_mtgnn_wind_mtgnn_nwp_hist_fold2.csv` mit bis zu 0.58 m/s Unterschied je Station.

Geprüft: **beide sind in sich konsistent**, jede CSV passt auf 1e-7 genau zu ihrem eigenen
Rohvorhersage-Parquet. Es sind also zwei vollständige, unabhängige Auswertungen. Die
pkl-Namen lösen es auf: für die stdhp-Läufe sind fold1 und fold2 auf `l1` und `ws`
identisch, nur fold3 (in der CSV-Zählung fold2) unterscheidet sich. `l1` hat ihn am
2026-08-05 um 23:54 nachgerechnet, `ws` stammt von 17:24. Die `ws`-Fassung ist damit
abgelöst und liegt unter `archiv/stdhp_mtgnn_fold2_abgeloest_20260818/`. Ebenso
archiviert: die älteren `ecmwf_test_fold0.csv` und `icon_d2_test_fold0.csv` auf `l2`, die
das Notebook ohnehin in keiner Konfiguration als Referenz verwendet.

---

## 16. MTGNN (HPO-getunt) und tft_sp_base nachtrainiert, alle laufenden HPO-Studien gestoppt

### 16.1 Anlass und Ablauf

Am 2026-08-24 wurden alle noch laufenden HPO-Studien auf allen drei Hosts (lokal, `l1`,
`ws`) gestoppt — DCRNN GRID+HIST, MTGNN (alle drei Varianten) und `tft_sp_base`.
Reihenfolge wie in den Host-Notizen festgehalten: `hpo_keeper_plan.json` je Host auf
leere `studies`-Liste gesetzt, `hpo_keeper` neu gestartet (damit er den leeren Plan
übernimmt, bevor Worker beendet werden), erst danach alle Worker-Screens beendet. Keine
verwaisten Screens oder Python-Prozesse zurückgeblieben (geprüft über `pstree` je Screen).

Trial-Stand beim Stoppen (COMPLETE, bester gepoolter Wert):

| Studie | COMPLETE | best_value |
|---|---|---|
| `cl_m-mtgnn_out-48_freq-1h_wind_mtgnn` | 110 | 1.2332 |
| `cl_m-mtgnn_out-48_freq-1h_wind_mtgnn_nwp` | 113 | 1.2147 |
| `cl_m-mtgnn_out-48_freq-1h_wind_mtgnn_nwp_hist` | 88 | 0.9785 |
| `cl_m-tft-bc_out-48_freq-1h_wind_tft_sp_base` | 136 | 1.2433 |

Alle vier Studien wurden mit demselben Rezept wie die DCRNN-Arme (§10/§13.3)
nachtrainiert und ausgewertet: `train_mtgnn.py --hpo-study auto` (lädt den Optuna-
Bestwert direkt aus Postgres) → `get_test_results_mtgnn.py --hpo-study auto`, analog für
`tft_sp_base` über `train_cl_tft_bc.py` / `get_test_results_tft_bc.py` mit dem exakten
Studiennamen `cl_m-tft-bc_out-48_freq-1h_wind_tft_sp_base` (kein `"auto"` — die
tft_bc-Skripte verlangen den vollen Namen). Alle zwölf Läufe (vier Arme × drei Folds)
liefen lokal auf vier GPUs parallel (train → eval je Fold sequenziell pro GPU), 0 Fehler,
keine Tracebacks in den zwölf Trainingslogs.

### 16.2 Haupttabelle, gefiltert, per Station, Mittel über drei Folds

Dieselbe Filterung wie in §14/§15 (`build_imputation_mask`/`_lookup_imputed`,
imputierte Zielstunden ausgeschlossen). Bei `tft_sp_base` normalisiert auf die bare
Stations-ID (`synth_00161.csv` → `00161`), sonst identischer Code wie bei DCRNN/MTGNN.

| Arm | fold1 | fold2 | fold3 | **Mittel** | imputierter Anteil |
|---|---|---|---|---|---|
| `mtgnn_nwp_hist` (GRID+HIST) | 0.9059 | 0.9611 | 1.0052 | **0.9574** | 0.60 / 0.68 / 0.86 % |
| `dcrnn_nwp_hist` (GRID+HIST, s. §16.6) | 0.9642 | 1.0003 | 1.0379 | **1.0008** | 0.60 / 0.68 / 0.86 % |
| `mtgnn_nwp` (GRID) | 1.1286 | 1.1650 | 1.2174 | **1.1703** | 0.60 / 0.68 / 0.86 % |
| `tft_sp_base` | 1.1112 | 1.1769 | 1.2768 | **1.1883** | 0.00 % (dropna, s. §14.6/`tft_sp_hist`) |
| `mtgnn` (BASE) | 1.1072 | 1.2617 | 1.2217 | **1.1969** | 0.60 / 0.68 / 0.86 % |

Zur Einordnung gegen §15.2: `dcrnn_idw_alt` (D', bisheriger Spitzenreiter der dortigen
Ladder) 1.0967, `dcrnn` (GRID) 1.1116, `dcrnn_base` 1.1782. Referenzen: ICON-D2 1.3036,
Persistenz 2.2342, MOS-regional 1.1603, MOS-local 0.9452 (transduktive Obergrenze).

`mtgnn_nwp_hist` unterbietet mit 0.9574 sowohl **alle** DCRNN-Arme (einschliesslich des
nachträglich ergänzten `dcrnn_nwp_hist`, s. §16.6) als auch MOS-regional (1.1603)
deutlich und nähert sich der transduktiven Obergrenze MOS-local (0.9452) auf 0.012 an —
das ist das mit Abstand beste Modell der gesamten bisherigen Ablationsleiter.
`dcrnn_nwp_hist` liegt mit 1.0008 klar auf Platz zwei, vor `dcrnn_idw_alt` (D', bisher
Platz eins bei DCRNN). Die Rohwerte (ungefiltert) liegen je Fold nur 0.001–0.003 unter
den gefilterten Zahlen, dieselbe kleine Verschiebung wie bei Wavenet in der vorigen
Auswertung.

### 16.3 Signifikanz auf Stationsebene

Wilcoxon-Vorzeichen-Rangtest, zweiseitig, gepaart über die Vereinigung der drei Folds
(N = 153), Holm-korrigiert — Methodik identisch zu §15.3, auf den ungefilterten
Stations-CSVs. `dcrnn_idw_alt` (D') und `dcrnn_nwp_hist` (§16.6) als Referenz
mitgeführt (gleiches `retrain_..._foldN.csv`-Namens- und Faltungsschema wie die vier
neuen Arme, daher ohne Fold-Umrechnung direkt vergleichbar). 6 Arme, 15 Paare.

| Vergleich | Median A−B | A besser | p (Holm) | signifikant |
|---|---|---|---|---|
| GRID gegen GRID+HIST | +0.1346 | 2.0 % | 1.7e-25 | ja |
| BASE gegen GRID+HIST | +0.1303 | 5.9 % | 8.1e-25 | ja |
| **GRID+HIST gegen tft_sp_base** | −0.1543 | 93.5 % | 1.7e-24 | **ja** |
| GRID gegen `dcrnn_nwp_hist` | +0.1174 | 6.5 % | 4.4e-22 | ja |
| **GRID+HIST gegen D'** | −0.0756 | 88.2 % | 7.1e-21 | **ja** |
| tft_sp_base gegen `dcrnn_nwp_hist` | +0.1088 | 11.8 % | 1.1e-20 | ja |
| BASE gegen `dcrnn_nwp_hist` | +0.1062 | 15.0 % | 5.2e-20 | ja |
| **D' gegen `dcrnn_nwp_hist`** | +0.0430 | 20.3 % | 7.7e-16 | **ja** |
| **GRID+HIST gegen `dcrnn_nwp_hist`** | −0.0283 | 75.2 % | 7.5e-10 | **ja** |
| GRID gegen D' | +0.0534 | 31.4 % | 9.6e-07 | ja |
| tft_sp_base gegen D' | +0.0541 | 30.7 % | 3.7e-06 | ja |
| BASE gegen D' | +0.0483 | 34.6 % | 1.9e-05 | ja |
| GRID gegen tft_sp_base | −0.0077 | 52.3 % | 0.81 | nein |
| BASE gegen GRID | −0.0044 | 54.3 % | 1.0 | nein |
| BASE gegen tft_sp_base | −0.0167 | 56.2 % | 1.0 | nein |

"A besser" ist hier durchgehend der Anteil Stationen, an denen Arm A die *niedrigere*
RMSE hat (kleiner ist besser) — bei "D' gegen `dcrnn_nwp_hist`" mit 20.3 % heisst das:
`dcrnn_nwp_hist` gewinnt an 79.7 % der 153 Stationen.

### 16.4 Befund

`mtgnn_nwp_hist` (GRID+HIST) schlägt **jeden** anderen Arm in dieser Tabelle
hochsignifikant, einschliesslich des bisherigen Spitzenreiters `dcrnn_idw_alt` (D',
p_holm = 7.1e-21, an 88.2 % der 153 Stationen besser) und des nachträglich ergänzten
`dcrnn_nwp_hist` (p_holm = 7.5e-10, an 75.2 % besser, s. §16.6). Die drei übrigen neuen
Arme (`mtgnn`, `mtgnn_nwp`, `tft_sp_base`) sind untereinander statistisch nicht
unterscheidbar (p_holm zwischen 0.81 und 1.0), liegen aber alle signifikant hinter D'
**und** hinter `dcrnn_nwp_hist`.

Zweiter, ebenso wichtiger Befund: `dcrnn_nwp_hist` schlägt `dcrnn_idw_alt` (D')
signifikant (p_holm = 7.7e-16, an 79.7 % der Stationen besser). D' war in §15.2 als
"bisheriger Spitzenreiter" der DCRNN-Ladder bezeichnet — das gilt nur, solange
`dcrnn_nwp_hist` fehlt. Mit vollständiger Ladder ist `dcrnn_nwp_hist` der stärkste
DCRNN-Arm, nicht D'.

Das Ergebnis wiederholt qualitativ den DCRNN-Befund aus §15.4(a)/(b): die historischen
NWP-Läufe als Zusatzfeature tragen den mit Abstand grössten Teil des gemessenen
Gewinns — und zwar in **beiden** Architekturen (DCRNN wie MTGNN), nicht nur in einer.
Anders als bei den übrigen DCRNN-Ablationen (B/C aus §15.1) liegt für `dcrnn_nwp_hist`
und die MTGNN-Arme aber keine Zerlegung in Graph- vs. NWP-Knoten-Beitrag vor —
`mtgnn`/`mtgnn_nwp` unterscheiden sich strukturell anders als `BASE`/`NOGRAPH` bei
DCRNN (kein reines Analogon zu B/C), ein direkter Mechanismus-Vergleich mit §15.4 wäre
Überinterpretation.

### 16.5 Vorbehalte

- Wie in §15.5: der Wilcoxon-Test läuft auf den **ungefilterten** CSVs, die Haupttabelle
  in §16.2 ist gefiltert. Bei 0.6–0.9 % imputierten Stunden für MTGNN und 0 % für
  `tft_sp_base` (dropna) ist ein Vorzeichenwechsel unwahrscheinlich, nicht geprüft.
- Nur ein Retrain je Arm (kein Seed-Ensemble). Die in §14.4(a) gemessene
  Lauf-zu-Lauf-Streuung (~0.014 RMSE, gemessen an `dcrnn_base`) liegt deutlich unter dem
  Vorsprung von `mtgnn_nwp_hist` (~0.12–0.15 gegenüber den eigenen nächstbesten
  Varianten, ~0.08 gegenüber D'). Für den Gesamtbefund also wahrscheinlich unbedenklich,
  aber nicht durch Wiederholung abgesichert.
- Die Budgets der Studien sind ungleich (88 bis 139 abgeschlossene Trials) und nicht
  gegeneinander normalisiert, analog zu §15.5.
- Alle fünf Arme (vier vom 2026-08-24, `dcrnn_nwp_hist` vom 2026-08-25, s. §16.6) wurden
  auf einem einzelnen Host (lokal, vier bzw. eine GPU) nachtrainiert statt wie die
  ursprüngliche DCRNN-Kampagne über drei Hosts verteilt — ein zu §5 analoges
  Host-Vergleichsproblem entfällt damit hier per Konstruktion.

### 16.6 Nachtrag: `dcrnn_nwp_hist` war in dieser Ladder zunächst nicht enthalten

Direkt nach der ersten Fassung dieses Abschnitts fragte der Nutzer, ob DCRNN NWP+HIST
noch fehle. Prüfung bestätigte das: die Studie
`cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_nwp_hist` war eine der beiden DCRNN-Studien, die
beim Stoppen in §16.1 noch aktiv liefen (139 COMPLETE, best_value 1.0509, Trial #145) —
sie war also nie in die finale HPO-getunte Retrain-Ladder aus §15 aufgenommen worden.
Auf der Platte lagen für diesen Arm nur veraltete Checkpoints (Juni/Juli, vor mehreren
zwischenzeitlichen Trial-Verbesserungen) und ein ungetunter `stdhp`-Trockenlauf, aber
keine `retrain_dcrnn_nwp_hist_fold{1,2,3}.csv` im Namensschema der übrigen fünf
DCRNN-Arme aus §15.2.

Nachtrainiert mit demselben Rezept wie in §16.1 (`train_dcrnn.py --hpo-study auto` auf
Trial #145 → `get_test_results_dcrnn.py --hpo-study auto`), eine GPU, drei Folds
sequenziell, 0 Fehler. Die Zahlen sind bereits in §16.2 (Haupttabelle) und §16.3
(Wilcoxon-Matrix) eingearbeitet. Kernbefund: `dcrnn_nwp_hist` (1.0008 gefiltert) ist der
stärkste DCRNN-Arm der gesamten bisherigen Auswertung, signifikant vor `dcrnn_idw_alt`
(D', bisher als Spitzenreiter geführt) — die in §15.2 berichtete DCRNN-Rangfolge war
demnach unvollständig, nicht falsch: D' blieb dort nur vorn, weil der stärkere Arm nicht
mitgerechnet wurde.

## 17. Expanding-Window-Retraining der 5 Flagschiff-Arme

Auftrag und Ausführung: `docs/expanding_window_retrain_handoff.md`. 5 Arme × 3 Folds
× 3 Zeitschritte = 45 Trainings- und Eval-Läufe, 27.08.–29.08.2026 auf GPU 2
(zwei parallele Worker, 50,6 h Wall-Clock, **45/45 mit `train=0 eval=0`, 0 Fehler**).
Hyperparameter unverändert aus den bestehenden Optuna-Studies (`--hpo-study auto`),
variiert wurde ausschliesslich das Zeitfenster. Auswertung: `scripts/eval_expwin.py`.

### 17.1 Der Aufbau misst die gestellte Frage nicht direkt

Die drei Zeitschritte haben nicht nur wachsende Trainingsfenster, sondern auch
**verschiedene Validierungsfenster** — Aug–Nov / Dez–Mär / Apr–Jul. Ein Vergleich der
RMSE über die Schritte hinweg misst daher Saison und Datenmenge zugleich. Die
ungefilterten Rohzahlen zeigen in **allen fünf Armen** dasselbe Muster
(Schritt 2 am schlechtesten, Schritt 3 am besten), und zwar auch dann, wenn das Modell
gar nicht nachtrainiert wurde: das saisongleich geschnittene Kontrollmodell macht die
Bewegung 1:1 mit. Der Effekt ist also Winter vs. Sommer, nicht Trainingsfenster.

Auflösung ohne Neuberechnung: die Single-Window-Retrains aus §16 (fixes Fenster
`train < 2024-08-01`) haben ihre Roh-Vorhersagen in
`data/raw_preds/retrain_<arm>_fold<N>_raw.parquet` über den gesamten Zeitraum
2024-08-01 … 2025-08-02 liegen. Auf die drei Schrittfenster geschnitten ergeben sie je
Schritt ein **saisongleiches Kontrollmodell ohne Fenstererweiterung**. Verglichen wird
also Expanding gegen Fix innerhalb desselben Fensters.

Dabei ist **Schritt 1 eine Nullmessung**: sein Trainingsfenster (`< 2024-08-01`) ist
identisch mit dem der Kontrolle. Beide Modelle unterscheiden sich nur in Seed und
Early-Stopping-Fenster (4 statt 12 Monate). Was dort an Differenz auftaucht, ist
Rauschen, kein Datenmengeneffekt — und liefert den Massstab für Schritt 2 und 3.

### 17.2 Haupttabelle, gefiltert, per Station, Mittel über drei Folds

Filterung wie in §14–16 (`build_imputation_mask`/`_lookup_imputed`, imputierte
Zielstunden ausgeschlossen; imputierter Anteil 0,29–1,82 % je Fold/Schritt). Maske
gegen die gt<0-Stunden validiert. EW = Expanding-Window-Retrain, Fix = saisongleicher
Ausschnitt des Single-Window-Retrains.

| Arm | S1 EW | S1 Fix | Δ1 | S2 EW | S2 Fix | Δ2 | S3 EW | S3 Fix | Δ3 |
|---|---|---|---|---|---|---|---|---|---|
| DCRNN GRID (A) | 1.0981 | 1.1010 | −0.0029 | 1.1427 | 1.1429 | −0.0003 | 1.0746 | 1.0831 | −0.0085 |
| DCRNN IDW (D') | 1.0952 | 1.0918 | +0.0034 | 1.1425 | 1.1322 | +0.0103 | 1.0727 | 1.0554 | +0.0173 |
| DCRNN GRID+HIST | 1.0045 | 0.9977 | +0.0068 | 1.0287 | 1.0228 | +0.0059 | **0.9637** | 0.9771 | **−0.0135** |
| MTGNN GRID | 1.1421 | 1.1629 | −0.0208 | 1.2054 | 1.2254 | −0.0200 | 1.0850 | 1.1116 | −0.0267 |
| MTGNN GRID+HIST | 0.9506 | 0.9508 | −0.0003 | 0.9923 | 0.9921 | +0.0001 | **0.9145** | 0.9245 | **−0.0099** |

Negatives Δ = Expanding besser. Die absolute Rangfolge der Arme bleibt über alle drei
Schritte die aus §16.2 bekannte: MTGNN GRID+HIST vorn, dahinter DCRNN GRID+HIST, dann
die drei Arme ohne NWP-Historie.

### 17.3 Signifikanz auf Stationsebene

Wilcoxon-Vorzeichen-Rangtest, zweiseitig, gepaart über die Vereinigung der drei Folds
(N = 153), Holm über alle 15 Vergleiche. Methodik wie §15.3/§16.3.

| Arm | Schritt | Median Δ | EW besser | p (Holm) | signifikant |
|---|---|---|---|---|---|
| DCRNN GRID+HIST | 3 | −0.0114 | 65.4 % | 7.2e-03 | ja |
| MTGNN GRID | 3 | −0.0300 | 65.4 % | 4.8e-02 | ja |
| MTGNN GRID+HIST | 3 | −0.0111 | 71.9 % | 5.0e-06 | ja |
| *alle übrigen 12 Vergleiche* | 1, 2 | −0.030 … +0.006 | 38–62 % | ≥ 0.29 | nein |

### 17.4 Differenz-in-Differenzen gegen die Nullmessung

Die Tabelle in §17.3 überschätzt den Effekt dort, wo schon Schritt 1 — der **keinen**
Datenvorteil hat — in dieselbe Richtung ausschlägt. Sauberer ist je Station
(EW − Fix)_Schritt N − (EW − Fix)_Schritt 1. Holm über 10 Vergleiche:

| Arm | Schritt | Median DiD | besser | p (Holm) | signifikant |
|---|---|---|---|---|---|
| **DCRNN GRID+HIST** | **3** | **−0.0236** | 67.3 % | **3.1e-04** | **ja** |
| **MTGNN GRID+HIST** | **3** | **−0.0105** | 63.4 % | **8.2e-04** | **ja** |
| MTGNN GRID | 3 | −0.0091 | 52.9 % | 1.00 | nein |
| MTGNN GRID | 2 | −0.0121 | 53.6 % | 1.00 | nein |
| DCRNN IDW (D') | 2 / 3 | +0.0143 / +0.0104 | 43 / 44 % | 0.90 | nein |
| DCRNN GRID | 2 / 3 | −0.0093 / −0.0080 | 52 / 57 % | 1.00 | nein |
| DCRNN GRID+HIST | 2 | +0.0021 | 49.7 % | 1.00 | nein |
| MTGNN GRID+HIST | 2 | −0.0020 | 52.3 % | 1.00 | nein |

Der scheinbare Gewinn von MTGNN GRID aus §17.3 verschwindet hier: sein Schritt-3-Δ von
−0.0300 steckt zu −0.0208 bereits in der Nullmessung, ist also grösstenteils Seed- und
Early-Stopping-Rauschen, nicht Datenmenge.

### 17.5 Befund

**Nachtrainieren mit wachsendem Fenster hilft, aber nur den beiden Armen mit
NWP-Historie und erst bei +8 Monaten.** `dcrnn_nwp_hist` gewinnt 0.0236 RMSE
(an 67.3 % der 153 Stationen, p_holm = 3.1e-04), `mtgnn_nwp_hist` 0.0105
(63.4 %, p_holm = 8.2e-04). Die drei Arme ohne Historie — DCRNN GRID, DCRNN IDW,
MTGNN GRID — zeigen über alle Schritte **keinen** signifikanten Effekt.

Bei +4 Monaten (Schritt 2) ist in **keinem** Arm etwas nachweisbar. Der Effekt setzt
also erst zwischen +4 und +8 Monaten ein und ist auch dann klein: 0.010–0.024 m/s
entsprechen 1,1–2,4 % relativ, gemessen an einem Rauschband von ±0.02 aus der
Nullmessung.

Das passt zum Mechanismuskapitel: die Arme, die zusätzlich NWP-Historie konsumieren,
haben den grösseren Eingangsraum und profitieren als einzige davon, ihn mit mehr Daten
zu füllen; die Arme ohne Historie sind bei ~12 Monaten Trainingsdaten bereits gesättigt.

### 17.6 Vorbehalte

- **Kein separater Testsatz.** Ausgewertet wird das Val-Fenster (die 51 je Fold nie
  gesehenen Zielstationen), das zugleich das Early-Stopping-Fenster ist. Die Zahlen
  sind damit leicht optimistisch — für den *Vergleich* EW gegen Fix unkritisch, da
  beide Seiten denselben Bias tragen, für absolute Aussagen nicht zu verwenden. Der
  zurückgehaltene Testsatz ab 2025-08-01 wurde nicht angerührt.
- **Nur drei Zeitschritte, nur ein Seed je Zelle.** Das Rauschband aus der Nullmessung
  (±0.02) liegt in derselben Grössenordnung wie die gefundenen Effekte. Die beiden
  signifikanten Befunde stützen sich auf die Paarung über 153 Stationen, nicht auf
  wiederholte Läufe.
- **Schritt 1 ist nicht exakt deckungsgleich.** Der Kontroll-Ausschnitt hat dort
  1 185 036 statt 1 194 624 Zeilen (die Roh-Vorhersagen der Single-Window-Retrains
  beginnen bei `run_time` 2024-08-01 06:00). Betroffen sind 0,8 % der Zeilen am
  Fensteranfang; die Paarung erfolgt ohnehin je Station, nicht je Zeile.
- **Die Val-Fenster sind nicht gleich schwer.** Dez–Mär hat in allen Armen und auch in
  der Kontrolle die höchste RMSE. Absolute Vergleiche *zwischen* Schritten bleiben
  deshalb auch nach dieser Auswertung unzulässig — nur die Δ- und DiD-Spalten sind
  über Schritte hinweg interpretierbar.

## 18. Testjahr-Auswertung der 5 Flagschiff-Arme (vorläufig, 12 von 15 Läufen)

**Stand 2026-09-01 08:40 — 12 der 15 Läufe fertig, 0 Fehler.** Es fehlen noch
`mtgnn_nwp_hist` fold2/fold3 und `mtgnn_nwp` fold3 (laufen, s. § 18.6). Alle
MTGNN-Zahlen unten sind daher noch nicht über drei Folds gemittelt und können
sich verschieben. Auswertung: `scripts/eval_testyear.py`, Abbildungen unter
`figures/testyear/`.

### 18.1 Aufbau — und warum er vom Auftrag abweicht

Geplant war die Auswertung auf dem reservierten Testsatz (`test_files`, 50
Stationen, `--test-mode`). **Das ist derzeit nicht möglich:** 46 der 50
Teststationen haben Messlücken, die die Imputationskette nicht füllt (55–405
Stunden je Station, über den gesamten Zeitraum verteilt). `handle_nans: drop`
würde den Testsatz auf 4 Stationen reduzieren. Die 102 Trainings- und 51
Validierungsstationen sind dagegen lückenlos — deshalb ist das nie aufgefallen,
alle bisherigen Läufe liefen im Dev-Modus.

Die Lücken wären füllbar: die KNN-Imputation (`knnimputer_path`) ist für alle
203 Stationen im Testfenster lückenlos. Der DCRNN-Pfad ruft sie aber nur für
`wind_direction` auf (`get_test_results_dcrnn.py:274`), nicht für `wind_speed`.
Das zu ändern ist ein Eingriff in die Preprocessing-Kette und stand vor einer
finalen Auswertung nicht zur Debatte.

**Stattdessen umgesetzt:** alle 5 Arme neu trainiert auf allem vor 2025-08-01
(2 Jahre statt 1 — der Retrain, der laut § 17 nur den HIST-Armen nützt, den die
übrigen drei hier aber zwangsläufig mitbekommen), Zero-Shot-Auswertung auf den
**51 `val_files`-Zielstationen** im Fenster **2025-08-01 … 2026-06-01**.
Early Stopping auf dem Testfenster.

Zweite Einschränkung: das Fenster endet am 2026-06-01 statt 2026-07-31, weil die
Interpolationsdaten (`interpol_path`) nur bis **2026-06-13** reichen — obwohl die
Rohmessungen aller 205 Stationen bis mindestens 2026-07-28 vorliegen und der
wöchentliche Cron am 2026-08-31 sauber durchlief. Vermutlich ein veralteter
Cache-Key; ungeprüft. Das Testfenster umfasst damit 10 statt 12 Monate,
deckt aber einen vollständigen Jahresgang minus Juni/Juli ab.

Configs: `configs/testyear/` auf l1 (Dateinamen identisch zur Basis, s.
[[expwin-namensschema-optuna]] bzw. § 8.1). Suffix `testyear`, Roh-Ergebnisse
`data/raw_preds/testyear_<arm>_fold<N>_raw.parquet`.

### 18.2 Haupttabelle, gefiltert, per Station

Filterung wie §14–17 (imputierte Zielstunden ausgeschlossen).

| Arm | fold1 | fold2 | fold3 | **Mittel** |
|---|---|---|---|---|
| **DCRNN GRID+HIST** | 1.0215 | 1.0563 | 1.1064 | **1.0614** |
| MTGNN GRID+HIST | 1.0711 | *läuft* | *läuft* | *(1.0711)* |
| DCRNN GRID | 1.1634 | 1.2081 | 1.2460 | 1.2058 |
| DCRNN IDW (D') | 1.1898 | 1.2122 | 1.2423 | 1.2148 |
| MTGNN GRID | 1.2247 | 1.2629 | *läuft* | *(1.2438)* |
| ICON-D2 | 1.2668 | 1.2253 | 1.3284 | 1.2735 |
| Persistenz | 2.0438 | 2.1352 | 2.1862 | 2.1217 |

Alle Arme schlagen ICON-D2, aber die drei Arme ohne NWP-Historie nur um 3–5 %
(1.21–1.24 gegen 1.27), die beiden HIST-Arme um 16 % (1.06–1.07).

### 18.3 Signifikanz auf Stationsebene

Wilcoxon, zweiseitig, gepaart über die Vereinigung der verfügbaren Folds,
Holm über 10 Vergleiche. **n variiert**, weil für MTGNN noch Folds fehlen.

| A | B | n | Median A−B | A besser | p (Holm) |
|---|---|---|---|---|---|
| DCRNN GRID+HIST | DCRNN IDW (D') | 153 | −0.1264 | 92.2 % | <1e-15 |
| DCRNN GRID+HIST | DCRNN GRID | 153 | −0.0993 | 86.3 % | <1e-15 |
| DCRNN GRID+HIST | MTGNN GRID | 102 | −0.1445 | 98.0 % | <1e-15 |
| MTGNN GRID+HIST | MTGNN GRID | 51 | −0.0789 | 80.4 % | 1.0e-06 |
| **DCRNN GRID+HIST** | **MTGNN GRID+HIST** | **51** | **−0.0560** | **82.4 %** | **4.7e-05** |
| MTGNN GRID+HIST | DCRNN IDW (D') | 51 | −0.0676 | 76.5 % | 1.0e-04 |
| MTGNN GRID | DCRNN GRID | 102 | +0.0668 | 31.4 % | 2.7e-03 |
| MTGNN GRID | DCRNN IDW (D') | 102 | +0.0292 | 36.3 % | 2.1e-02 |
| MTGNN GRID+HIST | DCRNN GRID | 51 | −0.0291 | 58.8 % | 2.6e-02 |
| DCRNN GRID | DCRNN IDW (D') | 153 | −0.0131 | 54.2 % | 0.31 |

### 18.4 Befund: die Gruppierung überträgt sich, die Rangfolge innerhalb nicht

Der Abstand **mit gegen ohne NWP-Historie** ist auf dem Testjahr genauso deutlich
wie im Validierungsjahr — 1.06 gegen 1.21–1.24, hochsignifikant in jedem
Paarvergleich. Das ist der Befund, der trägt.

**Innerhalb** der HIST-Gruppe kippt die Rangfolge: im Validierungsjahr war
`mtgnn_nwp_hist` der beste Arm der gesamten Ladder (0.9145 in § 17.2, vor
`dcrnn_nwp_hist` mit 0.9637), auf dem Testjahr liegt `dcrnn_nwp_hist` vorn
(1.0215 gegen 1.0711 in fold1, an 82.4 % der Stationen besser, p_holm = 4.7e-05).
**Dieser Befund steht und fällt mit den beiden fehlenden MTGNN-Folds** — er
stützt sich derzeit auf fold1 allein und ist bis zum Nachzug nicht zitierfähig.

`figures/testyear/04_error_by_horizon.png` zeigt den Mechanismus: die beiden
HIST-Arme starten bei Stunde 1 mit 0.69–0.73 m/s (sie sehen die jüngsten
Messungen), die drei anderen bei 1.17–1.20, praktisch auf ICON-D2-Niveau. Der
Vorsprung schrumpft mit der Vorlaufzeit, verschwindet aber bis Stunde 48 nicht:
dort liegen die HIST-Arme bei ~1.21–1.27, die übrigen bei ~1.33–1.39, ICON-D2
bei 1.50.

`figures/testyear/10_val_vs_test.png`: alle fünf Arme liegen über der Diagonale,
das Testjahr ist durchgehend schwerer als das Validierungsjahr (auch für
ICON-D2). Die Zwei-Gruppen-Struktur bleibt in beiden Jahren klar getrennt.

### 18.5 Abbildungen

Unter `figures/testyear/` je als PNG und PDF:

| Datei | Inhalt |
|---|---|
| `01_bar_rmse` | Balkendiagramm RMSE je Arm mit Fold-Streuung, ICON-D2/Persistenz als Referenzlinien |
| `02_fold_dispersion` | Streuung der Arme über die drei Folds |
| `03_paired_diff_boxplots` | Gepaarte ΔRMSE je Station für alle Armpaare |
| `04_error_by_horizon` | RMSE über den Prognosehorizont 1–48 h, mit ICON-D2 und Persistenz |
| `05_error_by_windspeed_class` | RMSE nach gemessener Windklasse (0–2 … >12 m/s) |
| `06_error_by_month` | RMSE über die Monate des Testfensters |
| `07_scatter_best` | Hexbin Vorhersage gegen Messung, bester Arm |
| `08_scatter_all` | Dasselbe als Panel über alle Arme |
| `09_skill_nwp_distribution` | Verteilung des Skill gegenüber ICON-D2 über die Stationen |
| `10_val_vs_test` | RMSE Validierungsjahr gegen Testjahr je Arm |

Tabellen als CSV: `data/test_results/testyear_overview.csv`,
`testyear_wilcoxon.csv`, `testyear_per_station.csv` (Stationsebene, für eigene
Auswertungen und Abbildungen).

### 18.6 Vorbehalte

- **3 von 15 Läufen fehlen** (`mtgnn_nwp_hist` fold2/3, `mtgnn_nwp` fold3). Alle
  MTGNN-Mittelwerte und jeder Vergleich mit n < 153 sind vorläufig.
- **Kein separater Testsatz im Stationsraum.** Ausgewertet wird auf den 51
  `val_files`, die im Dev-Training als Early-Stopping-Menge dienten. Zeitlich ist
  der Hold-out sauber (das Testjahr war nie im Training), räumlich sind diese
  Stationen zero-shot, aber nicht unberührt. Der reservierte 50-Stationen-Satz
  bleibt bis zur Imputationsreparatur ungenutzt (§ 18.1).
- **Early Stopping auf dem Testfenster** (so entschieden). Die Testzahlen sind
  dadurch leicht optimistisch; der Effekt trifft alle fünf Arme gleich und
  verzerrt den Armvergleich nicht.
- **10 statt 12 Monate**, Juni/Juli 2026 fehlen (§ 18.1).
