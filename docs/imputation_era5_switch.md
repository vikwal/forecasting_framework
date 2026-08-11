# Richtungsmittelung repariert + Windgeschwindigkeits-Imputation auf ERA5 umgestellt

**Status: UMGESETZT.** `l2` (`/home/viktor/Work/forecasting_framework`), Branch
`fix/mtgnn-topo-static-dim`, ausgehend von HEAD `309d420`. Beide Teile aendern die
Eingangsdaten aller Modelle und loesen deshalb gemeinsam GENAU EINE Erhoehung von
`IMPUTATION_GUARD_VERSION` (1 -> 2) aus.

---

## 1. Teil 1 — Richtungsmittelung repariert

### 1.1 Befund und Fix

`load_station_measurements` (`geostatistics/train_stgnn2.py`) resamplete 10-Minuten-Werte mit
`.resample(freq, closed="left", label="left").mean()` auf alle Spalten gleich, inklusive
`wind_direction` in Grad — ein arithmetisches Mittel aus z. B. 350° und 10° ergibt 180° statt der
korrekten Gegenrichtung 0°. `encode_circular_measurements` (`geostatistics/train_dcrnn.py`)
zerlegt danach in Sinus/Kosinus, der Schaden war zu diesem Zeitpunkt bereits eingetreten.

**Fix**: `wind_direction` wird jetzt separat behandelt — Sinus und Kosinus der 10-Minuten-Gradwerte
werden getrennt gemittelt (`.resample(...).mean()` auf `sin`/`cos`), per `arctan2`
zurueckgerechnet und modulo 360 genommen. Alle anderen Spalten (inkl. `wind_speed`) bleiben beim
arithmetischen Mittel, unveraendert. Das ist exakt die Konvention, die
`geostatistics/regen_knn_imputation.py` fuer dieselbe Groesse bereits verwendet (Zeilen 118-135) —
der Fix beseitigt damit auch die in `docs/imputation_era5_comparison.md` Abschnitt 6.3
dokumentierte Konventions-Inkonsistenz zwischen beiden Stellen.

NaN-Semantik erhalten: `np.sin`/`np.cos` von NaN ist NaN, `.mean()` ist skipna — eine Stunde ohne
jeden 10-Minuten-Messwert bleibt NaN in Sinus UND Kosinus und damit auch nach `arctan2` NaN,
exakt wie zuvor beim arithmetischen Mittel.

### 1.2 Beleg: `wind_speed` unveraendert

Vergleich altes/neues Verfahren, volle 153-Stationen-Poolliste (`files` + `val_files` aus
`configs/mtgnn/stdhp/config_wind_mtgnn_nwp_stdhp_fold1.yaml`), `T=26 088` Stunden,
`N=153` Stationen, `3 991 464` Zellen:

| Kennzahl | Wert |
|---|---:|
| `max\|wind_speed_alt − wind_speed_neu\|` | **0.0** |
| NaN-Muster-Abweichungen (alt NaN, neu nicht) | 0 |
| NaN-Muster-Abweichungen (neu NaN, alt nicht) | 0 |
| beidseitig-NaN-Zellen | 35 055 |

Der Fix aendert `wind_speed` an keiner einzigen Zelle.

### 1.3 Wirkungsabschaetzung (a): `|Δsin|` / `|Δcos|` — 153 Poolstationen

Gemessen an allen `3 956 361` Zellen, an denen sowohl altes als auch neues Verfahren einen Wert
liefern (beidseitig nicht NaN):

| Groesse | Mittel | Median | p95 | Maximum | Anteil `\|Δ\| > 0.1` |
|---|---:|---:|---:|---:|---:|
| `\|Δsin\|` | 0.0470 | 0.000062 | 0.4267 | 1.9932 | 8.82 % |
| `\|Δcos\|` | 0.1036 | 0.000066 | 0.9096 | 2.0000 | 9.37 % |

Das ist die tatsaechliche Stoerung der Modelleingaenge (nach `encode_circular_measurements`), nicht
der Gradfehler. Median nahe 0 (die meisten Stunden haben wenig Richtungsstreuung innerhalb der
6 Zehn-Minuten-Werte), aber ein Randbereich von 8.8–9.4 % der Zellen mit spuerbarer Verschiebung
(`> 0.1` auf einer `[-1, 1]`-Skala) und ein Maximum nahe dem theoretischen Extremwert 2 (voller
Vorzeichenwechsel).

Als Kreuzcheck der Gradabweichung (nicht Teil der geforderten Sinus/Kosinus-Messung, aber zur
Einordnung): Mittel 8.35°, Median 0.0047°, Anteil `> 5°`: 10.41 %, `> 45°`: 7.62 %, `> 90°`: 4.10 %,
Maximum 180°. Das liegt in derselben Groessenordnung wie der Sitzungsbefund auf der
15-Stationen-Stichprobe (8.23 % / 5.69 % / 2.96 % bei 5°/45°/90°, 389 282 Stundenwerte) — hier
volle 153-Stationen-Population statt Stichprobe, daher nicht identisch, aber konsistent.

### 1.4 Wirkungsabschaetzung (b): vorhandene Schranke aus der Ablationsleiter

Aus `stdhp_dryrun_results.md` Abschnitt 2 (siehe Abweichung 1.6 unten zur Fundstelle):

| Modell | Arm | RMSE (m/s) |
|---|---|---:|
| DCRNN | GRID *(induktiv, A)* | 1.129 ± 0.018 |
| DCRNN | GRID-NOMEAS *(B, Messkanal genullt)* | 1.133 ± 0.028 |
| MTGNN | GRID+HIST *(transduktiv)* | 1.024 ± 0.036 |
| MTGNN | GRID *(induktiv)* | 1.227 ± 0.067 |

Die Differenz DCRNN GRID vs. GRID-NOMEAS (0.004 m/s) ist der Wert des **gesamten**
Nachbarmesskanals (inkl. Richtung) im induktiven Setting. Ein Fehler in einem Teil dieses Kanals
kann nicht mehr kosten als der Kanal insgesamt wert ist — fuer die induktiven Arme (GRID,
GRID-NOMEAS; entsprechend WaveNet/MTGNN GRID ohne Zielstationshistorie) ist die Auswirkung des
Richtungsfehlers damit nach oben durch **0.004 m/s** beschraenkt, kleiner als die Fold-Streuung
(±0.018 bzw. ±0.028).

**Diese Schranke gilt NICHT** fuer die Arme mit Zielstationshistorie (GRID+HIST, TFT hist): dort
ist die eigene Historie 0.203 m/s wert (MTGNN GRID+HIST 1.024 vs. MTGNN GRID 1.227) — die
Groessenordnung, in der ein Richtungsfehler dort wirken koennte, ist folglich nicht durch die
gleiche Rechnung eingegrenzt. Keine eigenen Trainingslaeufe durchgefuehrt, wie vorgegeben.

### 1.5 Weitere Fundstellen (nur gemeldet, NICHT gefixt — ausserhalb des vorgegebenen Fix-Umfangs)

Systematische Suche nach `resample(...).mean()` in Verbindung mit Richtungs-/Winkelspalten im
gesamten Repository (ausser `frcst/lib/` Drittanbieter-Code, `runs/`, `checkpoints/`, `archiv/`):

1. **`geostatistics/regen_knn_imputation.py`** (Zeilen 118-135): bereits korrekt zirkulaer
   (Sinus/Kosinus-Mittel vor `arctan2`) — wie in der Vorabanalyse angegeben, kein Fund.
2. **`geostatistics/run_spatial_interpolation.py:553`**: `pivot_dir =
   pivot_dir.resample("1h", closed="left", label="left").mean()` — naives Gradmittel auf
   `wind_direction`, direkt danach Eingang in `wind_to_uv(pivot.values, pivot_dir.values)` fuer die
   Kriging-RK-Kovariaten (Windrichtungs-U/V-Matrizen). Aktiv nur, wenn `interpolate_uv` gesetzt ist.
   Kriging entfaellt mit Teil 2 dieses Auftrags als Imputations**weg** fuer die
   Windgeschwindigkeit, bleibt aber als Skript bestehen (u. a. fuer die weiterhin genutzte
   Kriging-Lag-Feature in `train_dcrnn.py`/`hpo_dcrnn.py`) — der Fehler ist dort weiterhin
   vorhanden, aber ausserhalb des fuer diesen Auftrag vorgegebenen Fix-Umfangs
   (`load_station_measurements`).
3. **`utils/preprocessing.py:2012,2139,2344`** (Funktion `preprocess_synth_wind_icond2`, die
   CL/TFT-ICON-D2-Pipeline): `df_synth.resample('1H', closed='left', label='left',
   origin='start').mean()` bzw. `_df_neighbor.resample(...).mean()` — operiert auf ALLEN
   numerischen Spalten des rohen Stations-Parquets (`select_dtypes(include='number')`), was
   `wind_direction` einschliesst (Spalte ist in `Station_<sid>.parquet` numerisch vorhanden).
   **Aktuell folgenlos**: `grep -rl "wind_direction" configs/` zeigt ausserhalb der
   GNN-Familie (DCRNN/MTGNN/WaveNet/STGCN, ueber `load_station_measurements` bereits gefixt) nur
   `configs/config_wind_160cl.yaml:57` — dort ist `wind_direction` **auskommentiert**
   (`#'wind_speed' #, 'wind_direction_next'`), wird also von keiner aktiven Config als Feature
   fuer diese Pipeline ausgewaehlt. Derselbe Fehlermuster liegt latent im Code, wuerde aber erst
   bei einer kuenftigen Config aktiv, die `wind_direction` fuer die CL/TFT-Pipeline waehlt.
4. **`archiv/train_stgcn.py`**: enthaelt denselben `apply_interpol_imputation`-Aufrufmuster wie die
   elf unten aufgefuehrten aktiven Aufrufstellen, liegt aber unter `archiv/` (bewusst archiviert,
   nicht Teil der aktiven Pipeline) — nicht angefasst.

Alle anderen Treffer aus der `resample(...).mean()`-Suche (`utils/imputation.py:103,252`,
`utils/preprocessing.py:139,762,3409,3438`, `geostatistics/run_spatial_interpolation.py:408,523,
533,582`, `geostatistics/run_solar_interpolation.py:*`, `geostatistics/stdrun/
make_stdhp_figures.py:283`, `geostatistics/verify_evaluate_reference_fix.py:32`) betreffen
`wind_speed` oder andere nicht-zirkulare Groessen (PV/Solar, `dhi`, generische KNN-Parquet-
Resamples bereits-stuendlicher Daten) — kein Winkel-Bezug, kein Fund.

---

## 2. Teil 2 — Windgeschwindigkeits-Imputation auf ERA5 umgestellt

### 2.1 Grundlage

`docs/imputation_era5_comparison.md` (read-only Vorabanalyse, 151 Poolstationen, 20 034
kuenstlich verdeckte aber tatsaechlich beobachtete Stationsstunden, Seed `20260811`): OLS je
Station 1.060 RMSE gegen Kriging 1.269 (kriging-vergleichbare Teilmenge: 1.027 gegen 1.269,
19 % besser). Das aufwendigste getestete Verfahren (globales RF mit Zeitfenstermerkmalen) erreicht
1.053 — kauft gegenueber OLS nur 0.007 m/s.

### 2.2 Messung: Zusammengesetzte Variante gegen reines OLS (PFLICHTMESSUNG vor der Entscheidung)

Getestet auf denselben 20 034 verdeckten Zellen (Seed `20260811`, Ziehverfahren wie in
`docs/imputation_era5_comparison.md` Abschnitt 1.2), unter Wiederverwendung der dort bereits ohne
Leck an den verdeckten Stunden gefitteten Vorhersagen (`predictions_speed.parquet`,
`pred_b_ols`/`pred_c_qmap_global`/`pred_c_qmap_monthly`, Fit-Mengen schliessen die verdeckten
Stunden ueberall aus).

**Aufbau**: OLS unterhalb einer Schwelle, Quantilabbildung oberhalb, weicher linearer Uebergang
ueber ein Intervall `[low, high]` auf einer zur Inferenzzeit verfuegbaren Schaltgroesse (nicht der
wahren Windgeschwindigkeit, die bei der Imputation ja gerade fehlt): `mag10` (rohe ERA5-Magnitude,
dieselbe Groesse, auf der die Quantilabbildung selbst fusst) bzw. testweise die OLS-eigene
Vorhersage. Gewicht `w = clip((x − low)/(high − low), 0, 1)`, `composite = (1−w)·OLS + w·Quantilabb.`

| Schaltgroesse | Intervall | Gesamt-RMSE | Δ ggue. OLS | 5–10 m/s RMSE | Δ ggue. OLS |
|---|---|---:|---:|---:|---:|
| `mag10` | [9, 11] | 1.0587 | −0.0009 | 1.4633 | +0.0216 |
| `mag10` | [9.5, 10.5] | 1.0580 | −0.0016 | 1.4625 | +0.0208 |
| `mag10` | [10, 12] | 1.0593 | −0.0003 | 1.4531 | +0.0114 |
| `pred_ols` | [9, 11] | 1.0576 | −0.0020 | 1.4602 | +0.0185 |
| `pred_ols` | [10, 12] | **1.0557** | **−0.0039** | 1.4463 | +0.0046 |
| — reines OLS je Station (Referenz) | — | 1.0596 | — | 1.4417 | — |

Ueber alle getesteten Konfigurationen: bestenfalls **0.4 % relative Verbesserung** im Gesamt-RMSE
(1.0596 → 1.0557), erkauft durch eine **durchgaengige Verschlechterung** der zweitgroessten
Windklasse (5–10 m/s, `n=3 537` von 20 034, 17.7 % der Zellen) in **jeder** getesteten
Konfiguration. Nur in den kleinen oberen Klassen (10–15 m/s, `n=398`; `>15` m/s, `n=26`) gewinnt
die Zusammensetzung klar (z. B. `>15` m/s: 3.390 → 2.548 im Basisaufbau) — aber diese beiden
Klassen zusammen sind `2.1 %` der Zellen und tragen den gesamten Effekt.

**Entscheidung gemaess Auftragsregel** ("Gewinnt die Zusammensetzung nicht klar, nimm reines OLS je
Station"): Der Gesamtgewinn ist mit `≤ 0.4 %` nicht klar (liegt in der Groessenordnung von
Rundungsrauschen, nicht von Fold-Streuung wie in Abschnitt 1.4) und geht auf Kosten der
zweitgroessten Klasse. **`utils/era5_imputation.py` implementiert deshalb ausschliesslich reines
per-Stations-OLS, keine Quantilabbildung.**

### 2.3 Implementierung: `utils/era5_imputation.py` (neu)

- `load_era5_wind_features(station_ids, start, end, db_url)`: laedt `public.era5_wind` ueber
  `WEATHER_DB_URL`, `station_id` per `.zfill(5)`. `timestamp` ist inhaltlich UTC, aber tz-naiv in
  der DB gespeichert — `tz_localize("UTC")` (nicht `tz_convert`), Query-Grenzen umgekehrt vor dem
  Senden auf tz-naiv zurueckgestutzt. Merkmale exakt wie in der Vergleichsanalyse: `mag10 =
  hypot(u_wind_10m, v_wind_10m)`, `mag100 = hypot(u_wind_100m, v_wind_100m)`, `ratio_100_10 =
  mag100/max(mag10, 0.1)`, `friction_wind`, `wind_gust_10m` (letztere zwei direkt aus der Tabelle).
- `fit_station_ols(era5_df, truth_long, min_fit_rows=30)`: ein `sklearn.linear_model.
  LinearRegression` je Station auf den Stunden mit vorliegender Messung (Fit-Menge kommt vom
  Aufrufer, siehe unten — schliesst fehlende/verdeckte Stunden korrekt aus). Stationen mit
  `< 30` Fit-Zeilen (derselbe Schwellenwert, den die Vergleichsanalyse bereits fuer den monatlichen
  Quantilabbildungs-Fallback verwendet, keine neue Zahl) bekommen kein Modell. Koeffizienten je
  Station werden als DataFrame zurueckgegeben und geloggt (`coefs`-Ergebnis von
  `load_era5_imputation`).
- `predict_station_ols` / `load_era5_imputation(station_ids, timestamps, meas_raw,
  measurement_cols, target_col="wind_speed", db_url=None)`: Anwendung NUR auf Stunden, in denen die
  Messung fehlt (Fit-Menge wird aus `meas_raw` VOR jeder Imputation abgeleitet — beobachtete
  Zellen sind die, an denen `wind_speed` nicht NaN ist). Rueckgabe ist ein `(T, N)`-Array,
  ausgerichtet auf `station_ids`/`timestamps`, NaN wo ERA5 nicht greift.
- Plausibilitaets-Guard `[0, 40]` (negative -> 0, `> 40` -> 40) wird innerhalb von
  `load_era5_imputation` angewendet, bevor das Array zurueckgegeben wird — dieselbe Schranke wie
  bei Kriging/KNN (`docs/imputation_plausibility_guard.md`).
- `wind_direction` wird von diesem Modul NICHT beruehrt — bleibt vollstaendig beim KNN-Imputer
  (rohes ERA5 verliert dort in jeder Windklasse, 31.5° gegen 9.1° Mittel, siehe
  `docs/imputation_era5_comparison.md` Abschnitt 5).

### 2.4 Integration / Aufrufstellen

Signaturen bestehender Funktionen wurden NICHT geaendert: `apply_interpol_imputation(meas_raw,
<array>, measurement_cols, target_col)` aus `utils/imputation.py` ist generisch genug (sie fuellt
NaN im Zielkanal mit einem beliebigen `(T, N)`-Array) und wird unveraendert wiederverwendet — nur
das uebergebene Array wechselt von `rk_pred` (Kriging) zu `era5_pred` (ERA5-OLS). Die
Aufrufstellen-Fallback-Logik (KNN fuer verbleibende NaN nach der primaeren Fuellung) existierte
bereits fuer den Kriging-Pfad und greift jetzt unveraendert fuer ERA5-Luecken.

**11 Aufrufstellen angepasst** (jeweils: `load_interpol_imputation`-Zeile fuer die Imputation
selbst durch `load_era5_imputation` + `apply_interpol_imputation(meas_raw, era5_pred, ...)`
ersetzt; `import` von `load_era5_imputation` ergaenzt):

| Datei | Rolle |
|---|---|
| `geostatistics/train_dcrnn.py` | Training DCRNN (Kriging-`rk_pred` bleibt fuer Lag-Feature) |
| `geostatistics/hpo_dcrnn.py` | HPO DCRNN (dito) |
| `geostatistics/get_test_results_dcrnn.py` | Test-Ergebnisse DCRNN (dito, `rk_pred` auch dort als Lag-Feature verwendet) |
| `geostatistics/train_mtgnn.py` | Training MTGNN |
| `geostatistics/hpo_mtgnn.py` | HPO MTGNN |
| `geostatistics/get_test_results_mtgnn.py` | Test-Ergebnisse MTGNN |
| `geostatistics/train_wavenet.py` | Training WaveNet |
| `geostatistics/hpo_wavenet.py` | HPO WaveNet |
| `geostatistics/get_test_results_wavenet.py` | Test-Ergebnisse WaveNet |
| `geostatistics/evaluate_reference.py` | Referenz-Evaluation |
| `geostatistics/baselines/dataset.py` | QRF/MOS-Baseline-Datensatz |

Bei DCRNN (`train_dcrnn.py`, `hpo_dcrnn.py`, `get_test_results_dcrnn.py`) bleibt
`rk_pred = load_interpol_imputation(...)` unveraendert bestehen — dort wird es NICHT mehr zur
Luecken-Fuellung verwendet, sondern ausschliesslich fuer das separate, von diesem Auftrag nicht
betroffene "Kriging-Lag-Feature" (ein eigener Modell-Input, keine Imputation).

**Nicht angepasst** (bewusst, siehe 1.5): `archiv/train_stgcn.py` (archiviert),
`geostatistics/train_stgnn2.py`s eigener `__main__`-Pfad (fuehrt ueberhaupt keine Imputation
durch, nur einen NaN-Audit — vermutlich ein aelterer, durch `train_dcrnn.py` abgeloester Pfad),
`geostatistics/get_test_results_stgnn2.py`, `geostatistics/audit_data.py` (beide rufen
`apply_interpol_imputation` nicht auf).

**Beobachtete Vorbestehende Luecke** (nicht durch diesen Auftrag verursacht, nur gemeldet):
`get_test_results_dcrnn.py`s KNN-Fallback-Block deckt nur `wind_direction` ab, nicht `target_col`
— d. h. in diesem einen Auswertungsskript gab es schon vor diesem Auftrag keinen KNN-Fallback fuer
`wind_speed`-Luecken nach Kriging (jetzt: nach ERA5). Nicht repariert, da ausserhalb des
vorgegebenen Umfangs (Verhaltensaenderung waere eine Entwurfsentscheidung).

### 2.5 Fallback-Kette

1. ERA5 + stationsweise OLS-Korrektur, wo `public.era5_wind` die Station/Stunde abdeckt.
2. Sonst: bestehender KNN-Imputer (unveraendert).

Kriging entfaellt damit als Imputationsweg fuer die Windgeschwindigkeit (bleibt nur als
Lag-Feature-Quelle bei DCRNN bestehen, s. o.). Windrichtung bleibt vollstaendig beim KNN-Imputer.

---

## 3. Verifikation ueber den echten Loader-Pfad

153-Stationen-Pool, Config `configs/mtgnn/stdhp/config_wind_mtgnn_nwp_stdhp_fold1.yaml`,
`T=26 088` Stunden (2023-01-01 … 2026-07-14), reale Aufrufkette
(`load_station_measurements` → `load_era5_imputation` → `apply_interpol_imputation` → KNN-Fallback
fuer `wind_speed` → KNN fuer `wind_direction`):

- Fehlende `wind_speed`-Zellen vor Imputation: **35 055** / 3 991 464.
- ERA5: 151/153 Stationen gefittet (`03196`, `15813` ohne `era5_wind`-Zeilen — die zwei aus der
  Vergleichsanalyse bekannten Luecken), 0 Stationen unter `min_fit_rows=30`.
- Guard beim Anwenden ausgeloest: 3 721 negative OLS-Vorhersagen auf 0 gekappt, 1 Wert `> 40` auf
  40 gekappt (beides VOR dem Zusammenfuehren mit dem KNN-Fallback).

| Fenster | Zellen gesamt | imputiert gesamt | davon ERA5 | davon KNN | negativ | `> 40` | Rest-NaN |
|---|---:|---:|---:|---:|---:|---:|---:|
| train (`< 2024-08-01`) | 1 373 328 | 11 646 | 11 601 | 45 | 0 | 0 | 0 |
| val (`< 2025-08-01`) | 1 340 280 | 9 610 | 9 370 | 240 | 0 | 0 | 0 |
| test (`≥ 2025-08-01`) | 1 277 856 | 13 799 | 12 489 | 1 310 | 0 | 0 | 0 |
| **Summe** | 3 991 464 | **35 055** | **33 460** | **1 595** | **0** | **0** | **0** |

Erwartung fuer die letzten drei Spalten (negativ, `>40`, Rest-NaN) war je 0 — erfuellt in allen
drei Fenstern.

**KNN-Fallback nach Ursache** (1 595 Zellen gesamt):

| Ursache | Zellen | Anteil |
|---|---:|---:|
| Station ohne `era5_wind`-Abdeckung (`03196`, `15813`) | 315 | 19.7 % |
| Stunde jenseits ERA5-Abdeckungsende (`2026-06-30 23:00` UTC) | 1 280 | 80.3 % |
| Sonstige (z. B. NaN in einem ERA5-Merkmal an sonst abgedeckter Stunde) | 0 | 0 % |

Je Fenster:

| Fenster | KNN gesamt | Stationsursache | Zeitursache |
|---|---:|---:|---:|
| train | 45 | 45 | 0 |
| val | 240 | 240 | 0 |
| test | 1 310 | 30 | 1 280 |

Die Zeitursache konzentriert sich vollstaendig auf das Testfenster, weil die Rohmessungen bis
2026-07-14 reichen, `public.era5_wind` aber am 2026-06-30 23:00 UTC endet. Anmerkung: dieses
volle Testfenster ist laenger als das von `train_dcrnn.py` tatsaechlich genutzte (dort wird
`meas_raw` vor der Imputation auf `test_end + 2 Tage` = 2026-04-02 gekappt, `test_end` in dieser
Fold-Config `2026-03-31`); die Tabelle oben verifiziert bewusst das VOLLE geladene Fenster ohne
diese Kappung, wie in der Aufgabenstellung ueber die drei Datumsgrenzen definiert.

Windrichtung bleibt vom Wechsel unberuehrt: 35 103 NaN vor KNN, 0 danach (KNN-Pfad unveraendert).

---

## 4. Cache-Invalidierung

`IMPUTATION_GUARD_VERSION` in `utils/data_cache.py`: **1 → 2**. Grund: beide Aenderungen wirken
auf `meas_raw`-Werte ueber Pfade, die der bestehende `_imputation_dir_fingerprint`-Mechanismus
NICHT erfasst — der Richtungsfix liest direkt aus `data_cfg["path"]` (im Cache-Schluessel nur als
String gehasht, nicht inhaltlich fingerprinted), die ERA5-Imputation liest aus einer laufenden
Postgres-Tabelle (kein Dateipfad, kein Fingerprint moeglich). Ohne die Versionserhoehung wuerde ein
Cache-Treffer mit identischem `interpol_path`/`knnimputer_path` weiterhin die alten,
vor-Fix-Tensoren liefern.

Trockenlauf-Beleg (`GNNCache.make_key`, identische Beispiel-Config, einmal mit
`IMPUTATION_GUARD_VERSION=2` [aktueller Code], einmal mit auf `1` zurueckgesetzter Konstante
[simuliert den Vor-Bump-Zustand]):

| Guard-Version | Schluessel |
|---|---|
| 1 (simuliert, Vorzustand) | `062cc24c2c316f42` |
| 2 (aktuell) | `4d0e9a9dd6974b48` |

Schluessel unterscheiden sich — bestaetigt. Vorhandene Cache-Verzeichnisse (`data_cache/gnns/`,
4 Eintraege, davon 2 mit aktivem `.write.lock` durch laufende HPO-Worker) wurden NICHT geloescht
oder angefasst.

---

## 5. Abweichungen vom Auftrag

1. **`docs/stdhp_dryrun_results.md` fehlt im l2-Repo** (`docs/` enthaelt die Datei nicht — direkt
   geprueft, `find` liefert keinen Treffer irgendwo unter `/home/viktor`). Die in Abschnitt 1.4
   zitierten Zahlen stammen aus der lokal gespiegelten Kopie
   (`/Users/viktorwalter/Latex/Graphs_Wind_Speed_Forecasting/docs/stdhp_dryrun_results.md`) und
   wurden dort Zeile fuer Zeile verifiziert (Zeilen 91-101: exakte Uebereinstimmung mit den im
   Auftrag genannten Werten 1.129/1.133/1.024/1.227). Kein Blocker, da die Zahlen bereits im
   Auftrag vorgegeben und hier unabhaengig bestaetigt wurden — aber die fehlende Datei im Repo
   selbst ist erwaehnenswert.
2. **Zusammengesetzte Variante**: die Wahl der Schaltgroesse (`mag10` vs. OLS-eigene Vorhersage)
   und des Uebergangsintervalls war im Auftrag nicht spezifiziert ("ueber ein Intervall" ohne
   Zahlenangabe). Mehrere plausible Konfigurationen wurden getestet (Abschnitt 2.2) statt nur
   einer einzelnen, um die Entscheidung "kein klarer Gewinn" nicht von einer zufaelligen
   Parameterwahl abhaengen zu lassen — das Ergebnis (kein Konfiguration gewinnt klar) ist ueber
   alle Varianten stabil.
3. **`min_fit_rows=30`** in `fit_station_ols` ist ein neu eingefuehrter Schwellenwert, nicht
   explizit im Auftrag genannt. Uebernommen von der bereits in der Vergleichsanalyse etablierten
   Schwelle fuer den monatlichen Quantilabbildungs-Fallback (`docs/imputation_era5_comparison.md`
   Abschnitt 2c), keine neu erfundene Zahl.
4. **`get_test_results_dcrnn.py`s fehlender KNN-Fallback fuer `wind_speed`** (Abschnitt 2.4) ist
   eine vorbestehende Luecke, unabhaengig von diesem Auftrag entstanden — gemeldet, nicht
   repariert (waere eine Verhaltensaenderung ausserhalb des Auftragsumfangs).
5. **Weitere Fundstellen naiver Winkelmittelung** (Abschnitt 1.5) wurden wie gefordert nur
   gemeldet, nicht gefixt — der Auftrag spezifiziert den Fix ausschliesslich fuer
   `load_station_measurements`.

## 6. Offene Punkte

- `geostatistics/run_spatial_interpolation.py:553` (naives Gradmittel vor `wind_to_uv`) bleibt
  ungefixt — relevant nur noch fuer die weiterhin genutzte Kriging-Lag-Feature bei DCRNN, nicht
  fuer die Windgeschwindigkeits-Imputation selbst.
- `utils/preprocessing.py:2012,2139,2344` (latenter, aktuell folgenloser Fund in der CL/TFT-
  Pipeline) bleibt ungefixt.
- Composite-vs-OLS-Messung nutzt die bereits vorhandenen Vorhersagen aus der Vergleichsanalyse
  (`predictions_speed.parquet`) statt eigener Neu-Fits — inhaltlich identisch (gleiche Fit-Logik,
  gleiche Daten, gleicher Seed), aber nicht als komplett unabhaengiger zweiter Lauf durchgefuehrt.
- Kein GPU-Training zur Bestaetigung der Wirkungsabschaetzung durchgefuehrt (wie vorgegeben).
