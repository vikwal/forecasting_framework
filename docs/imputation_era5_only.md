# Windgeschwindigkeits-Imputation vollstaendig auf ERA5 umgestellt (kein KNN/Kriging-Fallback)

**Status: UMGESETZT**, mit einer gemeldeten Abweichung (Abschnitt 6.1). `l2`
(`/home/viktor/Work/forecasting_framework`), Branch `fix/mtgnn-topo-static-dim`,
ausgehend von HEAD `9b43d77` (das den vorherigen ERA5/KNN-Zwischenstand aus
`docs/imputation_era5_switch.md` enthaelt). Diese Aenderung ersetzt den
verbliebenen KNN-Fallback fuer `wind_speed` vollstaendig durch NaN und stellt
die ERA5-Quelle von einer Live-Postgres-Abfrage auf einen lokalen
Parquet-Cache um.

## 0. Korrektur der Auftragsgrundlage: GRIB-Quelle

Der Auftrag nannte `/mnt/nas/era5/data/raw` (1396 Tagesdateien) als
GRIB-Rohdatenquelle fuer die zwei fehlenden Stationen. **Das ist nicht die
Quelle von `public.era5_wind`.** Direkt geprueft (7 Dateien ueber den vollen
Zeitraum 2022-10 bis 2026-07, sowie der zugehoerige Produktivcode
`/mnt/nas/era5/write_db_era5.py` + `config_era5.yaml`): dieses Archiv enthaelt
ausschliesslich drei Solar-/Albedo-Variablen (`fal`, `ssr`, `ssrd`) und
schreibt in eine andere Tabelle (`public.era5`, nicht `era5_wind`). Keine
Windvariable in keiner der 1396 Dateien.

Die tatsaechliche Quelle wurde gefunden: `/mnt/nas/era5_raw/wind/` (42
Monatsdateien `era5_MM_YYYY.grib`, 2023-01 bis 2026-06, GRIB-Gitter
`regular_ll`, 37×31 Punkte, 0.25°, Bereich 47.5–55.0°N / 6.0–15.0°E — exakt
der CDS-Request-Bereich `[55.06, 5.86, 47.26, 15.05]`) und ihr Erzeugerskript
`/home/meghnanegi/Era5/extract_grib_to_db.py` (aufgerufen aus
`request_era5_prefect_pipeline.py`): schreibt per Default in
`table_name="era5_wind"`, mit exakt den 9 in `era5_wind` vorhandenen
Wert-Spalten (`u_wind_10m`, `v_wind_10m`, `u_wind_100m`, `v_wind_100m`,
`wind_gust_10m`, `friction_wind`, `temp_2m`, `pressure`, `dew_point_2m`) plus
`geom`. Die 201 Stationskoordinaten in dessen `coordinates.txt` enthalten
weder `03196` noch `15813` — deckungsgleich mit der Beobachtung, dass genau
diese zwei Stationen in `era5_wind` fehlen. Der Auftrag, den Erzeuger zu
finden ("Der Erzeuger von era5_wind ist NICHT auffindbar"), ist damit erfuellt;
Schritt 1 unten kalibriert trotzdem unabhaengig anhand von Zahlen, wie
vorgegeben, statt sich allein auf den Fund zu verlassen.

## 1. Extraktionsmethode kalibriert

GRIB-Variablencheck (`era5_01_2024.grib`, `pygrib`, alle 6696 Nachrichten
durchsucht statt nur ein Sample): genau 9 eindeutige Variablen, exakt
deckungsgleich mit `era5_wind`s Spalten — `10 metre U/V wind component`,
`100 metre U/V wind component`, `Instantaneous 10 metre wind gust`,
`Friction velocity`, `2 metre temperature`, `Surface pressure`,
`2 metre dewpoint temperature`. u/v bei 10 m UND 100 m, Boe und
Reibungsgeschwindigkeit sind alle vorhanden.

Kalibrierung an drei in `era5_wind` vorhandenen Stationen (`00853`, `02907`,
`02925`, deren Originalkoordinaten aus `coordinates.txt`), Testmonat
2024-01 (`era5_01_2024.grib`, 2232 Stunden × 3 Stationen), zwei Methoden:

- **(a) naechster Gitterpunkt** — exakte Replik von `extract_grib_to_db.py`:
  Top-10-Kandidaten per euklidischer Lat/Lon-Distanz (`argpartition`), davon
  per echter `geopy.distance.geodesic`-Distanz das Minimum.
- **(b) bilineare Interpolation** auf dem regulaeren 0.25°-Gitter.

Vergleich zeilenweise gegen `public.era5_wind` fuer denselben Monat/dieselben
Stationen, alle 9 Variablen:

| Variable | (a) naechster Gitterpunkt: max\|Δ\| | (a) n > 1e-4 | (b) bilinear: max\|Δ\| | (b) n > 1e-4 |
|---|---:|---:|---:|---:|
| u_wind_10m | 4.76e-7 | 0 | 1.184 | 2231 |
| v_wind_10m | 4.62e-7 | 0 | 0.888 | 2231 |
| u_wind_100m | 8.11e-7 | 0 | 2.105 | 2231 |
| v_wind_100m | 6.69e-7 | 0 | 1.325 | 2232 |
| wind_gust_10m | 1.70e-6 | 0 | 1.895 | 2230 |
| friction_wind | 5.42e-8 | 0 | 0.057 | 2213 |
| temp_2m | 2.76e-5 | 0 | 1.269 | 2230 |
| pressure | 2.50e-3 | 1134 | 97.13 | 2232 |
| dew_point_2m | 2.91e-5 | 0 | 0.952 | 2230 |

(n = 2232 je Zelle, 3 Stationen × 24 h × 31 Tage.)

**Entscheidung: (a) naechster Gitterpunkt.** Reproduziert `era5_wind` fuer
alle vier OLS-Merkmalsquellen (`u/v_wind_10m`, `u/v_wind_100m`,
`friction_wind`, `wind_gust_10m`) mit max\|Δ\| in der Groessenordnung 1e-6–1e-7
(Float32/Float64-Rundungsrauschen, keine einzige Abweichung > 1e-4) und trifft
damit exakt denselben Gitterpunkt wie der Original-Erzeuger. Bilineare
Interpolation reproduziert `era5_wind` in KEINER Variable — Abweichungen bis
zu ~2 m/s, praktisch jede Zelle > 1e-4. `pressure` zeigt bei (a) eine kleine,
aber durchgaengige Restabweichung (max 0.0025, 1134/2232 Zellen > 1e-4) —
nicht Teil der vier OLS-Merkmale, betrifft die Umstellung nicht, Ursache
vermutlich eine kleine Rundungsstufe beim urspruenglichen Schreiben, nicht
weiter verfolgt (siehe 6.3).

## 2. Extraktion der zwei fehlenden Stationen

Mit Methode (a), vollem GRIB-Zeitraum (42 Dateien, 2023-01-01 bis
2026-06-30, stuendlich, 30648 Zeilen), Koordinaten aus
`stations_master.csv` (03196: lon 11.9324, lat 53.3221; 15813: lon 7.4131,
lat 52.5126 — beide exakt gegen die CSV verifiziert):

| Station | Gitterpunkt (i,j) | Gitter lat/lon | Zeilen | Zeitraum | Duplikate/Luecken |
|---|---|---|---:|---|---|
| 03196 | (7, 24) | 53.25°N, 12.00°E | 30648 | 2023-01-01 00:00 – 2026-06-30 23:00 | 0 |
| 15813 | (10, 6) | 52.50°N, 7.50°E | 30648 | 2023-01-01 00:00 – 2026-06-30 23:00 | 0 |

Nicht in die Datenbank geschrieben. Zwischenablage:
`/home/viktor/tmp/era5_wind_grib_calibration/extracted/Station_<sid>_grib_extracted.parquet`.

## 3. Parquet-Cache

`/mnt/lambda1/nvme1/synthetic/era5_wind_cache/`, 153 Dateien
(`Station_<sid>.parquet`), DatetimeIndex `timestamp` (UTC, stuendlich).
`era5_bc/processed` nicht angefasst (anderes Verzeichnis, andere Ebene:
`/mnt/nvme2/...` bzw. gar kein `era5_bc` unter `/mnt/lambda1/nvme1/synthetic/`
vorgefunden).

**Spaltensatz (9, Schnittmenge beider Quellen — beide liefern dieselben 9,
da dieselbe Extraktionsmethode):** `u_wind_10m`, `v_wind_10m`,
`u_wind_100m`, `v_wind_100m`, `wind_gust_10m`, `friction_wind`, `temp_2m`,
`pressure`, `dew_point_2m`.

| Quelle | Stationen | Zeilen/Station | Zeitraum |
|---|---:|---:|---|
| `public.era5_wind` (Postgres, read-only) | 151 | 26304 | 2023-07-01 00:00 – 2026-06-30 23:00 |
| GRIB-Extraktion (Schritt 2) | 2 (`03196`, `15813`) | 30648 | 2023-01-01 00:00 – 2026-06-30 23:00 |

**Nicht eigenmaechtig geaendert:** `utils/era5_imputation.py`s OLS-Merkmalsatz
(`ERA5_FEATURES`) nutzt weiterhin nur vier der neun Cache-Spalten (`mag10`
und `ratio_100_10`, abgeleitet aus u/v bei 10 m/100 m, plus `friction_wind`,
`wind_gust_10m`). Der Cache enthaelt zusaetzlich `temp_2m`, `pressure`,
`dew_point_2m` fuer alle 153 Stationen, die aktuell fuer nichts benutzt
werden — hier gemeldet statt selbstaendig in den Merkmalssatz aufgenommen.

## 4. Codeaenderungen (kein Rueckfall fuer wind_speed)

`utils/era5_imputation.py`: `load_era5_wind_features` liest jetzt
`Station_<sid>.parquet` aus `ERA5_CACHE_DIR` statt `public.era5_wind` per
`psycopg2`/SQL. Keine Postgres-Abhaengigkeit mehr in diesem Modul.
`ERA5_FEATURES`, `MIN_FIT_ROWS=30`, der Plausibilitaets-Guard `[0, 40]` und
die OLS-Fit-/Predict-Logik sind unveraendert.

KNN-Fallback fuer `target_col='wind_speed'` entfernt an 10 Aufrufstellen
(`get_test_results_dcrnn.py` hatte bereits keinen solchen Fallback — eine
vorbestehende, in `docs/imputation_era5_switch.md` Abschnitt 2.4 dokumentierte
Luecke, hier unveraendert):

| Datei | Aenderung |
|---|---|
| `train_dcrnn.py`, `hpo_dcrnn.py` | expliziter `if remaining_nan > 0: knn = load_knn_imputation(...)`-Block fuer `target_col` entfernt, durch reines Logging ersetzt |
| `train_mtgnn.py`, `hpo_mtgnn.py`, `train_wavenet.py`, `hpo_wavenet.py`, `get_test_results_mtgnn.py`, `get_test_results_wavenet.py`, `evaluate_reference.py`, `baselines/dataset.py` | generische Schleife `for col in measurement_cols: ...KNN...` ueberspringt jetzt `target_col` explizit (`if col == target_col: continue`) |

`wind_direction` unveraendert: bleibt vollstaendig beim KNN-Imputer an jeder
Aufrufstelle. Regression-Kriging (`load_interpol_imputation`/`rk_pred`) war
bereits seit `docs/imputation_era5_switch.md` kein Imputationsweg mehr fuer
`wind_speed`; bleibt bei DCRNN weiterhin als separates Lag-Feature bestehen
(unveraendert, ausserhalb des Auftragsumfangs).

## 5. Verifikation ueber den echten Loader-Pfad

153-Stationen-Pool (`files`+`val_files`, 102+51), Config
`configs/mtgnn/stdhp/config_wind_mtgnn_nwp_stdhp_fold1.yaml`, real ueber
`geostatistics/train_mtgnn.py main()` aufgerufen (kein Mock/Nachbau) — CPU
(`CUDA_VISIBLE_DEVICES=""`), abgefangen unmittelbar vor der
`HomoSampler`-Konstruktion (vor jedem Training), per Capture-Wrapper um
`load_station_measurements`/`apply_interpol_imputation`/`HomoSampler`.

### 5.1 Run-Paare — Abweichung vom harten Kriterium, mit A/B-Beleg

| | train | val |
|---|---:|---:|
| Gemessen (heutige Kette, mit dieser Umstellung) | **1473** | **1460** |
| Vorgabe (hartes Kriterium) | 1488 | 1460 |
| Differenz | **−15** | 0 |

`val` trifft exakt. `train` weicht um 15 Paare ab. **Kontrolliertes A/B:** alle
Aenderungen dieses Auftrags wurden chirurgisch zurueckgenommen (Parquet-Cache
→ Postgres, KNN-Fallback-Bloecke wiederhergestellt — bei vorbestehenden,
bereits vor diesem Auftrag unabhaengig dirty-modifizierten Dateien nur die
eigenen Zeilen entfernt, der Rest blieb unangetastet) und dieselbe Messung
erneut gefahren:

| Kette | train | val | skipped | grid-NaN |
|---|---:|---:|---:|---:|
| Neu (Parquet-Cache, kein Fallback) | 1473 | 1460 | 960 | 0 |
| Alt (Postgres, KNN-Fallback fuer wind_speed) | 1473 | 1460 | 960 | 0 |

Bit-identisch in allen vier Zahlen. **Die Umstellung dieses Auftrags aendert
die Run-Paar-Zahl nachweislich um 0.** Die Differenz zur Vorgabe (1488) ist
folglich nicht durch diese Umstellung verursacht — sie muss aus einer
Datei-/Datendrift zwischen der Messung, die "1488" ergab, und diesem Lauf
stammen (z. B. ICON-D2-NWP-Zulauf; auf diesem Host laufen aktive
HPO-Worker/Pipelines, die laut Auftrag nicht angefasst werden durften, deren
Dateneffekt auf `run_times`/`R` aber nicht ausgeschlossen werden kann). Nicht
weiter untersucht, da ausserhalb des Auftragsumfangs (keine
Entwurfsentscheidung noetig, da die Umstellung selbst erwiesenermassen
neutral ist) — **hier als Befund gemeldet, wie vorgegeben.**

### 5.2 Fensterstatistik (`wind_speed`, ungekapptes Ladefenster T=26088, 2023-07-24 – 2026-07-14, 153 Stationen)

| Fenster | Zellen gesamt | fehlend vorher | aus ERA5 gefuellt | Rest-NaN | negativ | > 40 |
|---|---:|---:|---:|---:|---:|---:|
| train (< 2024-08-01) | 1 373 328 | 11 646 | 11 646 | **0** | **0** | **0** |
| val (< 2025-08-01) | 1 340 280 | 9 610 | 9 610 | **0** | **0** | **0** |
| test (≥ 2025-08-01) | 1 277 856 | 13 799 | 12 519 | 1 280 | **0** | **0** |
| **Summe** | 3 991 464 | **35 055** | **33 775** | **1 280** | **0** | **0** |

`35 055` fehlende Zellen vor Imputation deckt sich exakt mit der in
`docs/imputation_era5_switch.md` Abschnitt 3 gemessenen Zahl (unter der
damaligen Postgres/KNN-Kette) — derselbe Ausgangszustand, nur die
Fuellmethode ist jetzt strikter. Alle 1280 verbleibenden NaN liegen im
Testfenster, ausschliesslich jenseits `ERA5_COVERAGE_END`
(2026-06-30 23:00 UTC) bzw. den beiden GRIB-Stationen zeitlich nicht
zuzuordnen — vollstaendig erwartet und laut Auftrag so gewollt ("kostet null
Run-Paare", durch 5.1 bestaetigt: das Ladefenster fuer `train_mtgnn.py`
selbst wird ohnehin bei `test_end + 2 Tage = 2026-04-02` gekappt, lange vor
2026-06-30, weshalb im tatsaechlich fuer Run-Paare genutzten Fenster ueberhaupt
keine Rest-NaN auftreten — bestaetigt durch den Log der echten Ladepipeline:
"filled 28474/28474 missing cells, 0 remain NaN"). 0 negative und 0 Werte
> 40 in allen drei Fenstern, wie erwartet.

Guard-Ausloesung auf dem vollen Fenster: 3732 negative OLS-Vorhersagen auf 0
gekappt, 1 Wert > 40 auf 40 gekappt (vor Zusammenfuehren mit den beobachteten
Zellen) — in derselben Groessenordnung wie die 3721/1 aus der Vorgaenger-Kette
(`docs/imputation_era5_switch.md`), leicht hoeher durch die 315
zusaetzlichen, jetzt erstmals ERA5-gefuellten Zellen der zwei neuen Stationen.

### 5.3 `wind_speed` an beobachteten Stunden unveraendert

Ueber alle 3 956 409 beobachteten Zellen (153 Stationen, volles Fenster):
**max\|Δ\| = 0.0**, 0 Zellen mit `Δ ≠ 0`. Die Imputation ruehrt beobachtete
Werte nicht an.

### 5.4 03196 und 15813 einzeln

| Station | fehlend vorher | aus ERA5 gefuellt | Rest-NaN | negativ | > 40 | max\|Δ\| beobachtet |
|---|---:|---:|---:|---:|---:|---:|
| 03196 | 239 | 239 | 0 | 0 | 0 | 0.0 (n=25 849 beobachtet) |
| 15813 | 76 | 76 | 0 | 0 | 0 | 0.0 (n=26 012 beobachtet) |

Beide Stationen: alle vormals fehlenden `wind_speed`-Stunden jetzt
vollstaendig aus der neu extrahierten GRIB-Quelle gefuellt, keine verbleibende
Luecke, kein Guard-Treffer ausserhalb [0,40], beobachtete Werte unangetastet.

## 6. Abweichungen, Widersprueche, offene Punkte

### 6.1 Run-Paare (hartes Kriterium)

Siehe 5.1 — `train` 1473 statt 1488, `val` trifft exakt. Durch A/B-Test
zweifelsfrei nicht durch diese Umstellung verursacht. **Nicht geloest,
gemeldet.**

### 6.2 GRIB-Pfad im Auftrag falsch

Siehe Abschnitt 0. Der im Auftrag genannte Pfad `/mnt/nas/era5/data/raw`
ist die Solar-/Albedo-Pipeline, nicht die Windquelle. Die echte Quelle
(`/mnt/nas/era5_raw/wind/`) und ihr Erzeugerskript wurden gefunden und
verifiziert (Abschnitt 0–1), sodass Schritt 1–2 trotzdem mit belegten Zahlen
durchgefuehrt werden konnten, statt abzubrechen.

### 6.3 `pressure`-Restabweichung bei Methode (a)

Siehe Tabelle in Abschnitt 1: `pressure` erreicht bei Methode (a) max\|Δ\| =
0.0025 mit 1134/2232 Zellen > 1e-4 — alle anderen acht Variablen sind exakt
(< 3e-5). `pressure` ist keines der vier OLS-Merkmale und fliesst nicht in
die Imputation ein; nicht weiter untersucht.

### 6.4 Cache enthaelt ungenutzte Spalten

Siehe Abschnitt 3, letzter Absatz: `temp_2m`, `pressure`, `dew_point_2m` sind
im Cache fuer alle 153 Stationen vorhanden, aber nicht Teil von
`ERA5_FEATURES`. Absichtlich nicht eigenmaechtig ergaenzt.

### 6.5 Cache-Schluessel-Nachweis

`IMPUTATION_GUARD_VERSION` in `utils/data_cache.py`: 2 → 3.
Trockenlauf (`GNNCache.make_key`, identische Beispiel-Config, einmal mit der
Konstante auf 3 [aktuell], einmal mit auf 2 zurueckgesetzter Konstante
[Vor-Bump-Zustand]):

| Guard-Version | Schluessel |
|---|---|
| 2 (simuliert, Vorzustand) | `2bcb2c0a2a58440a` |
| 3 (aktuell) | `a303ee02298a1fe6` |

Schluessel unterscheiden sich — bestaetigt. Vorhandene Cache-Verzeichnisse
NICHT geloescht oder angefasst.

### 6.6 Weitere offene Punkte

- Die 03196/15813-Extraktion deckt den vollen GRIB-Zeitraum (2023-01-01 ff.)
  ab, laenger als die 151 DB-Stationen (2023-07-01 ff.) — wie in Schritt 2
  vorgegeben. Fuehrt zu keiner Inkonsistenz (jede Station traegt einfach so
  viel Historie, wie ihre eigene Quelle hergibt), aber erwaehnenswert.
- Ursache der Run-Paar-Differenz (6.1) nicht ermittelt — vermutlich
  Datendrift bei ICON-D2 oder Rohmessdaten zwischen der Referenzmessung und
  diesem Lauf, ausserhalb des Auftragsumfangs nicht weiter verfolgt.
