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
