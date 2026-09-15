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
