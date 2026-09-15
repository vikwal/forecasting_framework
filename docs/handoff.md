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
* **DCRNN** (`geostatistics/train_dcrnn.py`, `configs/solar_dcrnn/`) — Läufe
  vorhanden, aber **ungültig**, s. §3

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

Dateien unter `results/solar/`. Die Auswertung der beiden Arme steht im
Gesprächsverlauf, ein Skript dafür gibt es noch nicht.

**Befund aus der Static-Ablation:** `dist_coast`, `svf` und `horizon_solar`
bringen nichts. Das Screening gegen den per-Station-RMSE hatte das
vorhergesagt — nach Herausrechnen von `altitude` bleibt von allen
Topo-Größen partiell |r| ≤ 0.23, und `slope`/`aspect` liegen bei 0.00, genau
wie die Physik es für einen waagerecht liegenden Pyranometer vorhersagt.
Es bleibt bei `altitude`, `latitude`, `longitude`.

## 3. Was ungültig ist und warum

Die sechs DCRNN-Arme (`results/solar_dcrnn_*v4*.pkl`, Modelle unter
`models/solar_dcrnn_*_v4_*.pt`) sind **mit einem um 30 Minuten verschobenen
NWP-Kanal trainiert** worden. Vier Prüfagenten haben das am 14./15.09. gefunden
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
müssen neu trainiert werden, erneutes Auswerten reicht nicht.

## 4. Was am 14./15.09. repariert wurde

Alles committet und auf allen drei Hosts (`0ffb797`).

| Commit | Inhalt |
|---|---|
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

## 5. Was noch zu tun ist

### 5.1 Blockierend für den Architekturvergleich

1. **`eval.exclude_imputed: true` in `scripts/make_solar_dcrnn_configs.py`
   aufnehmen**, Configs neu erzeugen. Ohne den Schlüssel misst das DCRNN auf
   8.73 % Nicht-Messungen (17 520 Modellfüllung + 14 598 Nachtnullen von
   367 920 Zielpositionen im Testjahr, Fold 1), während der TFT genau die
   entfernt. Der Code kann es seit `0ffb797`, die Configs setzen es nicht.
2. **Die sechs DCRNN-Arme neu trainieren** (`a`, `base`, `nomeas`, `nograph`,
   `idw_alt`, `nwp_hist`), Fold 1. Rund 45–75 min je Arm, sechs GPUs
   vorhanden. Danach auswerten — `get_test_results_dcrnn.py` lädt den
   Checkpoint und schreibt zusätzlich die Rohvorhersagen.
3. **Gemeinsame Auswertung schreiben.** Beide Seiten liefern ungefilterte
   Rohvorhersagen über dieselben 1 444 Läufe: DCRNN als
   `data/raw_preds/solar_dcrnn_*_raw.parquet`, TFT im Ergebnis-Pickle unter
   `predictions` (je Station und Zielgröße `pred`, `true`, Persistenz- und
   NWP-Baseline, 1 444 × 96). Eine Filterfunktion und eine Aggregation für
   beide, statt TFT-intern gegen GNN-intern. Vorbild: die Wind-Auswertung in
   `scripts/eval_testmode.py` (`build_imputation_mask`, dann elementweise
   filtern, dann RMSE je Station, dann Stationsmittel, Wilcoxon + Holm).

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
