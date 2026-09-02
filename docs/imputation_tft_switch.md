# Windgeschwindigkeits-Imputation auf die TFT-Werte umgestellt

**Status: UMGESETZT** am 2026-09-02. `l2` (`/home/viktor/Work/forecasting_framework`),
Branch `fix/mtgnn-topo-static-dim`. Die Umstellung aendert die Eingangsdaten aller
Modelle und loest deshalb GENAU EINE Erhoehung von `IMPUTATION_GUARD_VERSION`
(3 -> 4) aus, so wie beim ERA5-Umstieg (`docs/imputation_era5_switch.md`,
`docs/imputation_era5_only.md`).

Vorgeschichte in einer Zeile: Regression-Kriging (bis 2026-08-11) -> ERA5-OLS
(bis 2026-09-02) -> **TFT** (jetzt).

---

## 1. Was sich geaendert hat

Die Dateien unter `.../synthetic/interpol/wind` wurden am **2026-09-02 08:08** an
Ort und Stelle ersetzt. Sie tragen nicht mehr die Kriging-/IDW-Vorhersagen,
sondern die Vorhersage eines Temporal-Fusion-Transformers (Wind-Abschlussmodell,
trainiert auf 203 Stationen, fuellt Luecken aus ERA5 + Nachbarstationen +
Statik; Herkunft in `_herkunft.json`, Bilanz je Station in `_bericht.csv`).

Derselbe Baum, zwei Pfadkonventionen: `/mnt/nvme1/...` auf l1 (Besitzer),
`/mnt/lambda1/nvme1/...` auf l2 und ws (NFS). Der alte Bestand liegt unveraendert
daneben unter `.../synthetic/interpol/wind_vor_tft_20260902`.

### Dateiformat

| alt (Sicherung) | neu |
|---|---|
| `station_id`, `timestamp` | `station_id`, `timestamp` |
| `wind_speed_raw` — Messung, NaN in der Luecke | `wind_speed_raw` — **bitgleich** |
| `wind_speed_observed` — Messung, in der Luecke Kriging | `wind_speed_observed` — Messung, in der Luecke TFT |
| `rk_pred`, `idw_pred`, `ok_pred` | **entfallen** |
| — | `imputed` — die TFT-Vorhersage [m/s] |
| — | `n_fenster` — ueber wie viele ueberlappende Fenster gemittelt |
| — | `kontextfrei` — True = im ganzen 48-h-Fenster keine eigene Messung |

Zwei weitere Unterschiede:

1. **Zeitraum gewachsen**: neu 26 496 h, 2023-07-24 00:00 UTC bis 2026-07-31 23:00
   UTC, alle Stationen auf demselben Raster. Alt 25 326 h, 2023-07-24 07:00 bis
   2026-06-13 12:00. Interpol- und Messreihe sind also **nicht** deckungsgleich —
   die Messreihen laufen weiter (bis 2026-09-01 im Messtest unten).
2. **Emden (`05839`) fehlt bewusst** (94.4 % Fehlanteil). 203 Stationen — dieselbe
   Menge, die `interpol/wind` auch vorher fuehrte.

`imputed` ist genau dort nicht-NaN, wo `wind_speed_raw` NaN ist. Das Array hat
damit dieselbe Semantik wie `rk_pred`/`era5_pred` an der Fuellstelle und passt
ohne Umbau in den bestehenden „NaN in `meas_raw` fuellen"-Vertrag.

---

## 2. Was im Repo umgestellt wurde

### 2.1 `utils/imputation.py` — umbenannt statt heimlich umbelegt

Die alten Namen trugen „Kriging" im Namen und haetten nach dem Tausch etwas
anderes gemeint. Sie sind deshalb umbenannt, ohne Alias — eine uebersehene
Aufrufstelle soll einen `ImportError` werfen, nicht stillschweigend
weiterlaufen:

| alt | neu |
|---|---|
| `load_interpol_imputation` (las fest `rk_pred`) | `load_gap_imputation` (loest die Spalte aus dem Datei-Schema auf) |
| `apply_interpol_imputation` | `apply_imputation` |
| `impute_dfs_with_kriging` | `impute_dfs_from_interpol` |
| — | `impute_meas_raw_from_interpol` (neu: laden + anwenden + Diagnose in einem Aufruf) |
| — | `resolve_imputation_column` (neu) |

Die Spaltenaufloesung liest den Parquet-Footer und nimmt `imputed`, sonst
`rk_pred` (`IMPUTATION_VALUE_COLUMNS`). Damit bleibt **Solar unberuehrt**:
`interpol/solar` fuehrt weiterhin `rk_pred`, derselbe Code liest dort weiterhin
`rk_pred`. Findet sich keine der beiden Spalten, gibt es einen `KeyError` mit
Hinweis auf dieses Dokument — kein stiller Fallback.

`impute_meas_raw_from_interpol` loest die Spalte **einmal** an der ersten
vorhandenen Stationsdatei auf und benutzt sie fuer das ganze Verzeichnis; ein
gemischtes Verzeichnis faellt damit auf, statt gemittelt zu werden.

### 2.2 `utils/era5_imputation.py` — stillgelegt, nicht geloescht

Der OLS-Pfad wird von keiner Pipeline mehr aufgerufen. Das Modul bleibt
unveraendert als Beleg des vorigen Zustands liegen, der Kopfkommentar sagt das
jetzt in der ersten Zeile. Grund fuer die Ablosung: der OLS-Pfad deckte
153 Stationen bis 2026-06-30 ab, das TFT-Modell 203 Stationen bis 2026-07-31 —
die Umstellung gewinnt Abdeckung, nicht nur Guete (Messung in Abschnitt 3).

### 2.3 Aufrufstellen

Alle elf Aufrufer ersetzen die bisherige Dreierfolge
(`load_interpol_imputation` + `load_era5_imputation` + `apply_interpol_imputation`)
durch einen Aufruf:

```python
meas_raw, imput_diag = impute_meas_raw_from_interpol(
    meas_raw, all_ids, timestamps, measurement_cols, interpol_path, target_col,
)
```

`geostatistics/`: `train_dcrnn.py`, `hpo_dcrnn.py`, `get_test_results_dcrnn.py`,
`train_mtgnn.py`, `hpo_mtgnn.py`, `get_test_results_mtgnn.py`,
`train_wavenet.py`, `hpo_wavenet.py`, `get_test_results_wavenet.py`,
`evaluate_reference.py`, `baselines/dataset.py`; die Re-Exporte in
`train_stgnn2.py` entsprechend. `utils/preprocessing.py` (`get_data`, CL/FL-Pfad)
ruft `impute_dfs_from_interpol`.

**Kein Fallback**, unveraendert zur ERA5-Regel: Zellen, die das TFT-Modell nicht
abdeckt, bleiben NaN. Der KNN-Imputer bleibt fuer `wind_direction` und die
uebrigen Sekundaerspalten zustaendig — **`wind_direction` ist von dieser
Umstellung nicht betroffen** (rohes ERA5 verliert dort laut
`docs/imputation_era5_comparison.md` in jeder Windklasse gegen den KNN-Imputer,
31.5° gegen 9.1° mittlerer Fehler).

### 2.4 `dcrnn.interpolate_history` — offen, absichtlich

Das optionale Zusatzkanal-Feature fuetterte sich aus `rk_pred`: einer
Kriging-Schaetzung, die zu **jeder** Stunde definiert war, also eine
Nachbar-Sicht auf die Station *neben* der Messung. Dafuer gibt es in den neuen
Dateien keine Entsprechung — `imputed` existiert nur in den Luecken,
`wind_speed_observed` ist ausserhalb der Luecken die Messung selbst und damit ein
anderes Signal, kein Ersatzteil.

`interpolate_history` steht in allen 38 Configs, die den Schluessel setzen, auf
`false`. Statt eigenmaechtig ein Ersatzsignal zu waehlen, werfen
`train_dcrnn.py`, `hpo_dcrnn.py` und `get_test_results_dcrnn.py` bei `true` jetzt
einen `NotImplementedError` mit Verweis auf dieses Dokument — **direkt beim
Einlesen der Config**, bevor irgendetwas geladen wird, damit kein Lauf erst nach
Stunden abbricht. **Das ist eine offene Entscheidung, kein erledigter Punkt.**

Nebenbei behoben: `hpo_dcrnn.py` setzte `rk_pred` nur im Cache-MISS-Zweig, las es
aber im Fold-Block — bei einem Cache-Treffer waere `interpolate_history: true`
mit einem `NameError` gestorben. Unerreichbar, solange der Schluessel ueberall
`false` ist; mit dem Wegfall von `rk_pred` ist der Zweig ohnehin weg.

### 2.5 Schutz gegen versehentliches Ueberschreiben

`geostatistics/run_spatial_interpolation.py` schreibt mit `output.target_path` in
genau dieses Verzeichnis — ein Lauf haette die TFT-Dateien mit frischem Kriging
ueberschrieben. Vor dem Schreiben prueft `_guard_target_path()` jetzt, ob im
Zielverzeichnis bereits `Station_*.parquet` mit Spalte `imputed` liegen, und
bricht dann mit `SystemExit` ab (Hinweis auf die Sicherung). `--overwrite-imputed`
hebt die Sperre auf. Das Solar-Verzeichnis (`rk_pred`) loest die Sperre nicht aus.

### 2.6 Cache

`utils/data_cache.py`: `IMPUTATION_GUARD_VERSION` 3 -> 4. `interpol_fingerprint`
haette den Dateitausch allein schon bemerkt, **nicht** aber den Codewechsel weg
von ERA5 — der ERA5-Parquet-Cache wird von keinem Pfad beruehrt, den der
Cache-Key hasht. Ohne die Erhoehung wuerden unter Version 3 gebaute Eintraege
weiterhin ERA5-gefuellte Tensoren (153 Stationen, bis 2026-06-30) ausliefern.

---

## 3. Beleg: Zahl der gefuellten Zellen und Werteverteilung, vorher/nachher

Read-only-Messung auf **demselben** `meas_raw`, Datensatz
`configs/expwin/step1/config_wind_mtgnn_nwp_fold1.yaml`
(`files` + `val_files` + `test_files` = **203 Stationen**, `wind_speed` +
`wind_direction`, `freq: 1h`), `T = 27 264` (2023-07-24 00:00 … 2026-09-01 23:00
UTC), **5 534 592 Zellen, davon 48 226 fehlend (0.871 %)**.

Verglichen werden drei Fuellquellen an genau diesen 48 226 Zellen:
(a) Kriging `rk_pred` aus der Sicherung `wind_vor_tft_20260902`,
(b) ERA5-OLS (`utils/era5_imputation.py`, Stand bis heute),
(c) TFT `imputed` (neu).

| Quelle | gefuellt | Anteil | offen | Stationen mit Fuellung |
|---|---:|---:|---:|---:|
| (a) Kriging (Sicherung) | 42 966 | 89.1 % | 5 260 | 197 / 203 |
| (b) ERA5-OLS (bisher aktiv) | 33 673 | 69.8 % | 14 553 | **151 / 203** |
| (c) **TFT `imputed` (neu)** | **46 084** | **95.6 %** | **2 142** | **197 / 203** |

Verteilung der tatsaechlich **eingesetzten** Werte [m/s]:

| Quelle | mean | sd | min | p05 | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) Kriging | 3.421 | 2.282 | 0.000 | 0.952 | 1.854 | 2.868 | 4.344 | 7.811 | 20.760 |
| (b) ERA5-OLS | 3.578 | 2.544 | 0.000 | 1.028 | 1.845 | 2.912 | 4.424 | 9.115 | 21.761 |
| (c) TFT | 3.561 | 2.642 | **0.211** | 0.905 | 1.791 | 2.850 | 4.468 | 8.907 | 23.331 |

Weitere Kennzahlen der Messung:

- **Abdeckungsgewinn**: 12 411 Zellen fuellt nur das TFT-Modell, **0** Zellen nur
  ERA5. Der neue Pfad ist an dieser Fold eine echte Obermenge des alten.
- Auf den 33 673 gemeinsamen Zellen: mittlere absolute Abweichung TFT-ERA5
  **0.604 m/s**, Korrelation **0.9548**.
- Die 2 142 offen gebliebenen Zellen sind **vollstaendig** Stunden nach dem
  TFT-Ende 2026-07-31 23:00 UTC (0 durch fehlende Stationsdateien, 0 sonstige).
  Fuer diese Fold ist das folgenlos: `test_end` ist 2026-03-31, die Pipelines
  kappen auf `test_end + 2 d`, lange vor dem TFT-Ende.
- Die (a)-Zeile faellt gegen (c) auch deshalb ab, weil die Kriging-Reihen erst
  2023-07-24 **07:00** beginnen und 2026-06-13 enden — die 5 260 offenen Zellen
  sind ueberwiegend Randstunden.
- Untergrenze: das Kriging lieferte an dieser Stelle 0.000 m/s, weil der
  Plausibilitaets-Guard negative Werte auf 0 klemmte (der ERA5-Lauf der Messung
  klemmte 3 732 negative Werte). Das TFT-Minimum liegt bei 0.211 m/s — es gibt
  **keine** negativen Werte mehr zu klemmen.
- `n_cells_offered_unused` = **1**: genau eine Zelle, an der die Interpol-Datei
  eine Luecke sieht, die Messreihe des Frameworks aber einen Wert hat. Das Raster
  beider Quellen stimmt also praktisch exakt ueberein.

**36.3 % der Fuellungen (16 729 von 46 084) sind `kontextfrei`.**

Messskript: `scripts/measure_imputation_switch.py` (read-only, wiederholbar —
laedt die Messreihen neu und vergleicht alle drei Quellen an denselben Zellen).

---

## 4. Was ueber die neuen Werte gesagt werden darf — und was nicht

- **Zu `imputed` gibt es keine Guetezahl und kann es keine geben.** Die Guete des
  Abschlussmodells ist an *kuenstlich verdeckten, tatsaechlich beobachteten*
  Stunden gemessen (Skill gegen rohes ERA5: 0.557 bei 48-h-Luecke bis 0.709 bei
  1-h-Luecke). Ob das Modell an den **echten** Luecken genauso gut ist, weiss
  niemand: faellt ein Sensor bei Sturm aus, sind das systematisch andere Stunden.
  In keine Auswertung eine Zahl schreiben, die klingt, als sei die Imputation an
  den echten Luecken validiert.
- **`kontextfrei` nicht wegwerfen.** 36 % der gefuellten Stunden hatten im ganzen
  48-h-Fenster keine eigene Messung; die Vorhersage speist sich dort allein aus
  ERA5, Nachbarstationen und Statik. Die Spalte ist die ehrlichste Moeglichkeit,
  in einer Auswertung zwischen gut und schwach gestuetzten Luecken zu trennen —
  `impute_meas_raw_from_interpol` gibt sie als `n_cells_filled_kontextfrei` in
  der Diagnose zurueck, `load_gap_imputation(..., with_kontextfrei=True)` als
  volle `(T, N)`-Maske.
- Plausibilitaet der Quelle ist geprueft (Erzeugerseite): keine negativen Werte,
  Korrelation zu den alten `rk_pred` 0.905 bei 0.80 m/s mittlerer absoluter
  Abweichung, gemessene Stunden bitgleich zur Sicherung.
- Der Plausibilitaets-Guard aus `docs/imputation_plausibility_guard.md` greift
  auf diesem Pfad nicht mehr — er sass im Kriging-Erzeuger bzw. im ERA5-Modul.
  Bei den TFT-Werten gibt es an dieser Fold nichts zu klemmen (min 0.211,
  max 23.3 m/s, siehe oben); ein eigener Guard ist bewusst **nicht** eingebaut
  worden, weil er nur eine Fehlerquelle vortaeuschen wuerde, die die Quelle
  nicht hat.

## 5. Nicht betroffen

- **Solar.** `interpol/solar` fuehrt weiterhin `rk_pred`; die Spaltenaufloesung
  liest dort unveraendert weiter.
- **`wind_direction`.** Bleibt auf dem KNN-Pfad.
- **Bereits ausgewertete Laeufe.** Ergebnisdateien unter `results/` und die
  Auswertung in `geostatistics/stdrun/make_stdhp_figures.py` beziehen sich auf
  Laeufe, die unter dem ERA5-Pfad entstanden sind. Deren Imputationsmaske baut
  `make_stdhp_figures.py` aus den **Rohmessdateien**, nicht aus `interpol/` — sie
  ist von dieser Umstellung nicht kaputtgegangen. Die Kommentare dort, die von
  „regression kriging" sprechen, beschreiben den historischen Zustand jener
  Laeufe und sind bewusst nicht angefasst worden.
