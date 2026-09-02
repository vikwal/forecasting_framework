# ERA5 als Ersatz fuer Regression-Kriging + KNN? Eine Datenanalyse

**Status: ABGESCHLOSSEN.** Read-only-Analyse auf `l2` (`/home/viktor/Work/forecasting_framework`,
HEAD `309d420`). Kein Produktivcode, keine Config und keine Produktionsdaten wurden veraendert.
Einziges Schreibprodukt ist dieses Dokument (nicht committet, nur abgelegt in `docs/` und lokal
gespiegelt).

## 0. Fragestellung

Soll die bisherige Windmessungs-Imputation (Regression-Kriging fuer Geschwindigkeit,
KNN-Imputer fuer Geschwindigkeit *und* Richtung) durch eine ERA5-Reanalyse-basierte Methode
ersetzt werden? Beurteilt anhand von 151 Poolstationen (153 aus
`configs/mtgnn/stdhp/config_wind_mtgnn_nwp_stdhp_fold1.yaml` minus den zwei ohne ERA5-Abdeckung,
`03196` und `15813`), an ca. 20 000 kuenstlich verdeckten, aber tatsaechlich beobachteten
Stationsstunden.

## 1. Kuenstliche Luecken

### 1.1 Empirische Laengenverteilung der echten Luecken

Pro Station: Lauflaengen aufeinanderfolgender NaN-Stunden im Stundenmittel
(`resample("1h", closed="left", label="left").mean()` auf `wind_speed`, aus den
10-Minuten-Rohmessungen `Station_<sid>.parquet`). Geschwindigkeits- und Richtungs-NaNs fallen an
allen geprueften Stichprobenstationen exakt zusammen (gleicher Sensorausfall) — die Verteilung
gilt daher fuer beide Groessen.

| Kennzahl | Wert |
|---|---|
| Anzahl Luecken (Bloecke) | 1 348 |
| Summe NaN-Stunden | 31 367 |
| Mittelwert | 23.27 h |
| Median | 5.0 h |
| 75. Perzentil | 18.0 h |
| 90. Perzentil | 56.3 h |
| 95. Perzentil | 89.65 h |
| 99. Perzentil | 270.12 h |
| Maximum | 1 595 h (≈ 66 Tage, eine Station) |

Stark rechtsschief: die Mehrzahl der Luecken ist kurz (Median 5 h), aber der Schwanz reicht bis
zu zweimonatigen Ausfaellen. 151 Stationen, `n_hours_total` je Station 23 232–26 088 (Median
26 088), `frac_nan` je Station 0.0–6.48 % (Median 0.46 %).

### 1.2 Ziehverfahren

Einzelne zufaellige Stunden zu ziehen wuerde Verfahren bevorteilen, die zeitlichen Kontext
nutzen (z. B. Interpolation zwischen Nachbarwerten), weil echte Ausfaelle in Bloecken kommen.
Daher: Bootstrap-Ziehung von Blocklaengen aus genau der 1 348 Werte grossen empirischen
Laengenliste aus 1.1, platziert an Stunden, die tatsaechlich beobachtet sind
(`wind_speed` nicht NaN), innerhalb der ERA5-Abdeckung (2023-07-01 00:00 – 2026-06-30 23:00
UTC), disjunkt von echten Luecken und von bereits gezogenen Bloecken (Freilisten-Allokator pro
Station, Start-Offset zufaellig innerhalb der jeweiligen freien Beobachtungsspanne). Ziehung
Round-Robin ueber eine einmal geshuffelte Stationsliste (`numpy.random.default_rng(20260811)`),
eine Runde = ein Block pro Station, bis Zielumfang ≈ 20 000 Stationsstunden erreicht ist. Seed
`20260811` fest, deterministisch reproduzierbar.

**Ergebnis der Ziehung:**

| Kennzahl | Wert |
|---|---|
| Gezogene Stationsstunden gesamt | **20 034** |
| Anzahl Bloecke | 938 |
| Passes (Runden) | 7 |
| Bloecke je Station | 6–7 (Median 6, Mittel 6.21) |
| Fehlgeschlagene Platzierungsversuche | 0 / 938 |
| Stunden je Station | Min 9, Median 82, Mittel 132.7, Max 1 664 |
| Blocklaenge gezogen: Median / p90 / p99 / Max | 5 / 51 / 183.9 / 1 595 h |
| Sanity-Check: gezogene Zellen, die in Wahrheit NaN sind | 0 |

Eine einzelne Station absorbierte durch den seltenen 1 595-Stunden-Block ≈ 8.3 % des gesamten
Budgets (1 664 von 20 034 Stunden) — eine unvermeidliche Konsequenz des Bootstraps aus einer
schwer-schwaenzigen Verteilung "nach genau dieser Laengenverteilung", kein Fehler. Jede der 151
Stationen erhaelt dennoch mindestens 6 Bloecke, keine Station fehlt in der Auswertung.

## 2. Verfahren Windgeschwindigkeit

Fuenf Verfahren plus eine optionale Variante, alle bewertet an denselben 20 034 verdeckten
Zellen. Fit-Mengen schliessen die verdeckten Stunden ueberall explizit aus (sonst Selbstbetrug);
gefittet wird je Verfahren auf allen uebrigen beobachteten Stunden der jeweiligen Station(en),
ueber das gesamte verfuegbare Zeitfenster (keine train/val/test-Restriktion beim Fit — das
entspricht der Offline-Imputationslogik: bei der Luecken-Fuellung ist die gesamte uebrige Serie
bekannt).

- **(a) roh ERA5**: `hypot(u_wind_10m, v_wind_10m)`. Keine Anpassung, Untergrenze.
- **(b) OLS je Station**: `wind_speed ~ mag10 + ratio_100_10 + friction_wind + wind_gust_10m`
  (`ratio_100_10 = mag100/max(mag10, 0.1)`), `sklearn.LinearRegression`, ein Fit pro Station.
- **(c) Quantilabbildung je Station** auf `mag10`, empirische Quantile (Rang-Interpolation).
  Zwei Varianten: **global** (ein Fit pro Station) und **monatlich stratifiziert** (12 Fits pro
  Station, Fallback auf global bei < 30 Fit-Punkten im Monat).
- **(d) global**: gepoolt ueber alle 151 Stationen, Merkmale = die vier ERA5-Merkmale aus (b) +
  neun Topo-Deskriptoren (`TOPO_FEATURE_ORDER`) + lat/lon/alt (16 Merkmale gesamt).
  `RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=10, n_jobs=16,
  random_state=20260811)` als Hauptvariante; `Ridge(alpha=1.0)` auf standardisierten Merkmalen
  zum Vergleich mitgefuehrt.
- **(e) Kriging**: `rk_pred` an den verdeckten Stunden, unveraendert aus
  `interpol/wind/Station_<sid>.parquet` gelesen (kein eigenes Clipping ergaenzt — negative Werte
  waren bereits auf 0 gekappt, siehe Abschnitt 6.3).
- **(f, optional)** ausgeloest, weil (b)–(d) (b)–(d) klar besser als (a) sind (≈ 19–26 % RMSE
  Reduktion, siehe Tabelle unten): globales RF wie (d), zusaetzlich `mag10` bei t−3 … t+3 h als
  sieben Zusatzmerkmale (ERA5 "Zukunft" ist bei Offline-Imputation bekannt).

Ein Datenluecke: Station `02961` hat in `topo_features.csv` keinen `tdi`-Wert (die ganze Spalte
ist fuer diese Station NaN). Fuer (d)/(f) mit dem stationsuebergreifenden Median (0.956) aufgefuellt
— betrifft nur diese eine Station, RF haette NaN ohnehin toleriert (sklearn ≥ 1.4), Ridge nicht.

## 3. Verfahren Windrichtung

ERA5-Richtung aus `arctan2` der Komponenten (`(270 − rad2deg(arctan2(v10, u10))) % 360`,
Standard-meteorologische Konvention "kommt aus"; an 20 000 zufaelligen beobachteten,
nicht-verdeckten Stunden gegen die Rohmessung geprueft: mittlerer Fehler faellt von 30.8° bei
allen Windstaerken auf 11.9° bei > 5 m/s — physikalisch plausibel, Formel korrekt), verglichen
mit der KNN-Richtungsdatei (`wind_direction_knn10_start_end_67558851.parquet`). Metrik:
mittlerer absoluter Winkelfehler mit Umbruch, `abs(((a−b+180) % 360) − 180)`, plus Median,
stratifiziert nach Windstaerkeklasse der Messung.

**Wichtiger Vorbehalt** (gehoert zwingend zur Interpretation): der KNN-Wert an einer kuenstlich
verdeckten Stunde ist **nicht ehrlich** bewertbar. Der `KNNImputer` hat beim Fit den wahren Wert
gesehen (die Stunde war ja nicht wirklich fehlend) und gibt ihn fuer ohnehin beobachtete Zellen
unveraendert zurueck. Der KNN-Vergleich ist eine Obergrenze fuer KNN. Wenn ERA5 trotzdem
schlechter bleibt, ist die Aussage "ERA5 ersetzt KNN fuer Richtung nicht" umso belastbarer — was
hier der Fall ist (Abschnitt 5).

## 4. Auswertung Windgeschwindigkeit

### 4.1 Gesamtvergleich (alle 20 034 verdeckten Zellen)

| Verfahren | RMSE (m/s) | MAE (m/s) | Bias (m/s) | n |
|---|---:|---:|---:|---:|
| (f) RF global + Zeitfenster t−3..t+3 | **1.0531** | 0.7798 | +0.0123 | 20 034 |
| (b) OLS je Station | 1.0596 | 0.7810 | +0.0089 | 20 034 |
| (d) RF global | 1.0610 | 0.7855 | +0.0102 | 20 034 |
| (c) Quantilabbildung, monatlich | 1.1728 | 0.8727 | +0.0334 | 20 034 |
| (c) Quantilabbildung, global | 1.1830 | 0.8833 | +0.0105 | 20 034 |
| (e) Kriging (`rk_pred`) | 1.2692 | 0.8999 | +0.1495 | 14 934 |
| (d) Ridge global | 1.3401 | 0.9915 | +0.0621 | 20 034 |
| (a) roh ERA5 | 1.4313 | 1.0387 | −0.0213 | 20 034 |

**Gesamtsieger: (f) globales RF mit ERA5-Zeitfenster**, hauchduenn vor (b) OLS je Station und
(d) RF ohne Zeitfenster — alle drei innerhalb 1.4 % RMSE voneinander, klar vor allen anderen.
Der Zeitfenster-Zusatznutzen ist real, aber klein (RMSE 1.0610 → 1.0531, ≈ 0.7 %); in der
Feature-Importance des RF traegt fast ausschliesslich `mag10_t+1` (0.008) etwas bei, alle
uebrigen sechs Zeitfenster-Merkmale liegen bei ≤ 0.003 zusammen. Dominante Merkmale insgesamt:
`mag10_t0` (0.63), `tpi5` (0.13, ein Topo-Deskriptor!), `wind_gust_10m` (0.11).

### 4.2 Kriging-vergleichbare Teilmenge (nur Stunden vor 2025-11-02 21:00)

Von den 20 034 verdeckten Stunden liegen 14 941 (74.6 %) vor dem Kriging-Abdeckungsende; davon
haben 14 934 einen gueltigen `rk_pred`-Wert (7 zusaetzliche Luecken direkt in der
Interpolationsdatei). Alle Verfahren auf genau dieser Teilmenge, fuer einen fairen Vergleich mit
Kriging:

| Verfahren | RMSE | MAE | Bias | n |
|---|---:|---:|---:|---:|
| (f) RF global + Zeitfenster | **1.0184** | 0.7591 | +0.0688 | 14 934 |
| (b) OLS je Station | 1.0266 | 0.7623 | +0.0620 | 14 934 |
| (d) RF global | 1.0278 | 0.7658 | +0.0661 | 14 934 |
| (c) Quantilabbildung, monatlich | 1.1532 | 0.8644 | +0.0562 | 14 934 |
| (c) Quantilabbildung, global | 1.1680 | 0.8763 | +0.0657 | 14 934 |
| (e) Kriging | 1.2692 | 0.8999 | +0.1495 | 14 934 |
| (d) Ridge global | 1.3269 | 0.9875 | +0.1453 | 14 934 |
| (a) roh ERA5 | 1.3925 | 1.0187 | +0.0447 | 14 934 |

Ranking unveraendert gegenueber 4.1: auch auf der fairen Teilmenge liegt Kriging klar hinter
(b)/(c)/(f)/(d), nur vor Ridge und rohem ERA5.

### 4.3 Stratifiziert nach Windstaerkeklasse der Messung

| Klasse | n (Kriging n) | Sieger (RMSE) | 2. | ... | Letzter |
|---|---|---|---|---|---|
| 0–5 m/s | 16 112 (12 271) | (f) 0.878 | (d) 0.883 | (b) 0.896 < qmap < ridge < kriging 1.083 | (a) 1.127 |
| 5–10 m/s | 3 500 (2 396) | (b) 1.444 | (f) 1.469 | (d) 1.481 < qmap < ridge < kriging 1.800 | (a) 1.938 |
| **10–15 m/s** | 396 (254) | **(c) global 2.158** | (c) monatlich 2.226 | (b) 2.232 < (f) 2.243 < (d) 2.289 < kriging 2.603 | (d) Ridge 3.918 vor (a) 4.106 |
| **> 15 m/s** | 26 (13) | **(c) monatlich 1.832** | (c) global 2.448 | (d) 3.032 < (f) 3.057 < (b) 3.390 < kriging 3.692 | (a) 5.343 vor (d) Ridge 6.257 |

Volle Zahlen (RMSE/MAE/Bias):

- **10–15 m/s**: c_global 2.158/1.677/−0.829; c_monatlich 2.226/1.685/−0.570; b_OLS
  2.232/1.820/−1.414; f_RF+window 2.243/1.814/−1.278; d_RF 2.289/1.851/−1.300; e_Kriging
  2.603/2.150/−1.933; d_Ridge 3.918/3.591/−3.530; a_ERA5 4.106/3.397/−3.189.
- **> 15 m/s** (n = 26, klein — mit Vorsicht lesen): c_monatlich 1.832/1.175/−0.906; c_global
  2.448/1.781/−1.652; d_RF 3.032/2.325/−2.199; f_RF+window 3.057/2.414/−2.337; b_OLS
  3.390/2.763/−2.746; e_Kriging 3.692/3.409/−3.409 (n=13); a_ERA5 5.343/5.031/−5.031; d_Ridge
  6.257/6.103/−6.103.

**Randverhalten**: bei 0–10 m/s gewinnen (b)/(d)/(f) klar. Ab 10 m/s **kippt die Rangfolge**:
Quantilabbildung gewinnt, und die Regressions-/RF-Verfahren fallen wegen systematischer
Unterschaetzung (grosser negativer Bias, bis −6.1 m/s bei Ridge) zurueck — Shrinkage-zum-Mittel
gegen die von Natur aus Verteilungs-treue Quantilabbildung. Kriging bleibt in beiden oberen
Klassen hinter Quantilabbildung UND Regression/RF zurueck, schlaegt aber weiterhin Ridge und
rohes ERA5. `n = 26` in der Extremklasse ist klein; das Muster ist aber in der 10–15-Klasse
(n = 396) bereits deutlich sichtbar und richtungskonsistent, also kein reines Zufallsprodukt.

### 4.4 Stratifiziert nach Gelaendekomplexitaet (Terzile von `elev_std`, 151 Stationen)

Terzil-Grenzen: 12.96 / 41.30 (elev_std, m).

| Terrain | n (Kriging n) | Sieger | ... | Kriging-Rang |
|---|---|---|---|---|
| komplex (T3) | 5 878 (4 919) | (f) 1.052 | (d) 1.064 < (b) 1.097 < qmap < ridge 1.376 | **(e) 1.510, vorletzt (nur vor ERA5-roh)** |
| mittel (T2) | 7 033 (5 396) | (b) 0.876 | (f) 0.907 < (d) 0.915 | **(e) 0.933, vor beiden Quantilabbildungen!** |
| flach (T1) | 7 123 (4 619) | (f) 1.180 | (d) 1.185 < (b) 1.187 < qmap | (e) 1.327, vorletzt |

**Keine Umkehrung der Spitzenrangfolge** in komplexem Gelaende — (f)/(d)/(b) bleiben vorn, wie
insgesamt. Aber: Kriging fällt in komplexem Gelaende relativ am weitesten zurueck (RMSE 1.510,
nur noch vor rohem ERA5), waehrend es im **mittleren** Terzil die einzige Klasse ist, in der
Kriging **beide Quantilabbildungs-Varianten schlaegt** (0.933 < 0.988/1.008) — eine echte,
wenn auch lokal begrenzte Rangfolgenumkehr.

### 4.5 Stratifiziert nach Zeitfenster

| Fenster | n (Kriging n) | Sieger | ... |
|---|---|---|---|
| train (< 2024-08-01) | 5 113 (5 106) | (f) 1.101 | (b) 1.103 < (d) 1.112 < qmap < **Ridge 1.348 < Kriging 1.383** < ERA5 1.472 |
| val (< 2025-08-01) | 6 439 (6 439) | (f) 1.004 | (d) 1.008 < (b) 1.028 < qmap < Ridge 1.155 < **ERA5 1.219 < Kriging 1.224** |
| test (≥ 2025-08-01) | 8 482 (3 389) | (b) 1.056 | (f) 1.060 < (d) 1.069 < qmap < Kriging 1.170 < Ridge 1.460 < ERA5 1.550 |

Zwei kleine Umkehrungen gegenueber der Gesamtrangfolge (dort: Kriging 1.269 < Ridge 1.340 <
ERA5-roh 1.431):

- **train**: Ridge (1.348) schlaegt Kriging (1.383) — im Trainingsfenster ist Kriging
  vergleichsweise schwaecher.
- **val**: rohes ERA5 (1.219) schlaegt Kriging (1.224) knapp — Differenz sehr klein (0.005), aber
  auffaellig ist der stark erhoehte Kriging-Bias in diesem Fenster (+0.203 m/s, gegenueber
  +0.149 m/s insgesamt und nur +0.153/+0.043 in train/test) — Kriging driftet im Val-Fenster
  spuerbar staerker positiv.

## 5. Auswertung Windrichtung

| | ERA5 (arctan2) | KNN-Datei (nicht ehrlich, s. o.) |
|---|---:|---:|
| **Gesamt** (n = 20 034) — Mittel / Median | 31.49° / 16.90° | 9.11° / 0.0047° |
| 0–5 m/s (n = 16 112) | 36.40° / 20.95° | 11.19° / 0.013° |
| 5–10 m/s (n = 3 500) | 11.75° / 8.19° | 0.64° / 0.0038° |
| 10–15 m/s (n = 396) | 8.06° / 6.09° | 0.0037° / 0.0038° |
| > 15 m/s (n = 26) | 6.20° / 6.07° | 0.016° / 0.0° |

ERA5-Richtungsfehler faellt monoton mit der Windstaerke (physikalisch erwartet: Richtung ist bei
Schwachwind schlecht definiert). Der KNN-"Fehler" bleibt erwartungsgemaess nahe 0 (Median in
allen Klassen ≤ 0.013°) — die dokumentierte Obergrenzen-Verzerrung. **ERA5 gewinnt in keiner
Windklasse gegen KNN**; selbst in der guenstigsten Klasse (> 15 m/s) bleibt ERA5 bei median
6.07° gegen praktisch 0° bei KNN. Die im Vorbehalt formulierte Bedingung ("wenn ERA5 selbst
dagegen gewinnt...") trifft nicht zu — die Empfehlung fuer Richtung ist entsprechend eindeutig
negativ (Abschnitt 7).

**Datenqualitaets-Befund zur KNN-Obergrenze**: bei 2 846 von 20 034 verdeckten Stunden (14.2 %)
weicht der KNN-Dateiwert um mehr als 1° von der hier verwendeten Wahrheit ab (bis zu 180° im
Extremfall), konzentriert in der 0–5-m/s-Klasse. Ursache identifiziert (Quellcode-Vergleich):
die hier fuer "Wahrheit" verwendete Konvention (`train_stgnn2.load_station_measurements`, exakt
wie im Auftrag vorgegeben: naives `.resample("1h", closed="left", label="left").mean()` auf rohe
Gradzahlen) unterscheidet sich von der Konvention, mit der `regen_knn_imputation.py` die
KNN-Cache-Datei aufgebaut hat: dort wird die Stundenrichtung **zirkulaer** ueber Sinus-/Kosinus-
Komponenten gemittelt (Zeilen 118–129 der Datei), bevor sie in Grad zurueckgewandelt wird. Bei
Stunden mit stark schwankender Richtung nahe Windstille (0/360°-Umbruch) liefern beide
Konventionen unterschiedliche "wahre" Werte. Das ist kein Fehler dieser Analyse — sie folgt
exakt der vorgegebenen Konvention — sondern eine reale, bisher unbemerkte Inkonsistenz zwischen
zwei Stundenmittel-Konventionen fuer dieselbe physikalische Groesse im Code. Auswirkung auf die
Kernaussage: keine (der Median-KNN-Fehler bleibt trotzdem ~0°, die "Obergrenze"-Interpretation
gilt weiterhin), aber die 14.2 % relativieren die Praezision der Obergrenzen-Aussage in der
Schwachwindklasse.

## 6. Widersprueche und Datenqualitaets-Befunde (melden, nicht ueberschrieben)

1. **ERA5-Zeilenzahl**: der Auftrag nennt 5 295 805 Zeilen in `public.era5_wind`. Direkt
   nachgezaehlt: **5 287 104** Zeilen (201 Stationen × exakt 26 304 stuendliche Zeilen,
   2023-07-01 00:00 – 2026-06-30 23:00, luecken- und duplikatfrei je Station). Differenz 8 701
   Zeilen, Ursache nicht ermittelt (evtl. ein frueherer Zaehlstand vor einer Bereinigung).
   Betrifft die Analyse nicht (Stationsabdeckung 151/151 und Voll-Gitter je Station unabhaengig
   verifiziert).
2. **`tdi` fehlt fuer Station `02961`** in `topo_features.csv` (gesamte Spalte NaN fuer diese
   Station). Fuer (d)/(f) mit dem Median der uebrigen 150 Stationen (0.956) aufgefuellt.
3. **Direktionskonventions-Inkonsistenz** zwischen `load_station_measurements` (naiver
   Grad-Mittelwert) und `regen_knn_imputation.py` (zirkulaerer Sinus/Kosinus-Mittelwert) — siehe
   Abschnitt 5. Beeinflusst nur die Richtungsauswertung, nicht die Geschwindigkeit (dort gibt es
   keine Umbruch-Problematik).
4. **Kriging-Nullkappung verifiziert**: unabhaengig von der Vorgabe direkt nachgeprueft — 0
   negative `rk_pred`-Werte in allen 203 Stationsdateien unter `interpol/wind/`
   (Datei-mtime 2026-08-11 10:02, nach dem im Repo dokumentierten abgebrochenen
   Voll-Regenerationsversuch von `run_spatial_interpolation.py` um 09:17–09:47, siehe
   `docs/imputation_plausibility_guard.md`). Kein Widerspruch — die Vorgabe war korrekt; die
   Guard-Dokumentation beschreibt offenbar nur den Stand vor einer zusaetzlichen, gezielten
   Nachkappung.
5. **Statistische Power in den oberen Windklassen** gering (n = 396 bzw. n = 26). Das Muster
   (Quantilabbildung gewinnt, Regression/RF unterschaetzt) ist in beiden Klassen
   richtungskonsistent, sollte aber nicht ueberinterpretiert werden.
6. **Ungleiche Stationsverteilung der gezogenen Stunden** (9–1 664 h je Station) durch den
   Bootstrap aus einer schwer-schwaenzigen Laengenverteilung — erwartete Konsequenz der Vorgabe
   "nach genau dieser Laengenverteilung", keine Verzerrung im Ziehverfahren selbst.

## 7. Empfehlung

**Windgeschwindigkeit**: die Regression-Kriging-Imputation sollte durch eine ERA5-basierte,
je-Station kalibrierte Korrektur ersetzt werden (OLS oder globales Random-Forest, beide ≈ 1.03–
1.06 m/s RMSE gegenueber 1.27 m/s bei Kriging auf der fairen Teilmenge, eine Verbesserung von
≈ 19 %), mit einer Ausnahme: bei Windgeschwindigkeiten ueber ca. 10 m/s liefert eine
monatlich stratifizierte Quantilabbildung auf ERA5 die genaueren Werte, weil Regressions-/
RF-Verfahren dort systematisch unterschaetzen — ein produktives Verfahren sollte also entweder
die Quantilabbildung fuer diesen Bereich vorhalten oder die Regressionsvorhersage fuer hohe
ERA5-Werte re-kalibrieren.

**Windrichtung**: die KNN-Imputation sollte **nicht** durch rohe ERA5-Richtung ersetzt werden;
der ERA5-Richtungsfehler (Median 6–21° je nach Windklasse, am groessten genau dort, wo Richtung
fuer Anwendungen wie Nachlaufmodellierung relevant waere) ist zu gross, und selbst gegen die
nur als Obergrenze zu lesende KNN-Bewertung verliert ERA5 in jeder Windklasse.

## 8. Offene Punkte

- Keine Kreuzvalidierung der (b)/(c)/(d)-Fits ueber mehrere Seeds/Ziehungen durchgefuehrt;
  Ergebnisse basieren auf einer einzigen Ziehung (Seed 20260811), wie im Auftrag vorgegeben.
- (d)/(f) nutzen `RandomForestRegressor`; `Ridge` wurde nur als Referenz mitgefuehrt und schneidet
  durchgehend schlechter ab als RF und OLS — ein Hinweis auf relevante Nichtlinearitaeten
  (bestaetigt durch `tpi5` als zweitwichtigstes RF-Merkmal, noch vor allen ERA5-Zusatzmerkmalen
  ausser `mag10`).
  Nicht getestet: Gradient-Boosting oder Interaktionsterme fuer Ridge — ausserhalb des
  vorgegebenen sklearn-Rahmens (RandomForestRegressor ODER Ridge).
- Direktionskorrektur (analog zu (b)–(d) fuer Geschwindigkeit) wurde nicht getestet — nur rohes
  ERA5 gegen KNN. Eine kalibrierte ERA5-Richtungskorrektur koennte die Luecke zu KNN verkleinern,
  ist aber aus dem Auftrag explizit nicht gefordert und hier nicht untersucht.
- Die Diskrepanz der ERA5-Zeilenzahl (Abschnitt 6.1) ist ungeklaert.
- Die Direktionskonventions-Inkonsistenz (Abschnitt 6.3) betrifft vermutlich auch andere Stellen
  im Code, die `load_station_measurements` fuer `wind_direction` nutzen — nicht im Rahmen dieses
  Auftrags weiterverfolgt.
- Fuer die Extremklasse > 15 m/s (n = 26) waere eine gezielte Nachziehung mit mehr Extremstunden
  sinnvoll, um die Quantilabbildungs-Empfehlung statistisch abzusichern.

## 9. Reproduzierbarkeit

Alle Zwischenergebnisse liegen auf `l2` unter `/home/viktor/tmp/era5_imputation_analysis/`
(Skripte `stage1`…`stage6`, Parquet-Zwischenstaende, CSV-Tabellen je Stratum). Seed durchgaengig
`20260811`. Keine Aenderung an `configs/`, an Produktivskripten oder an den Parquet-/DB-Quellen
(bis auf die bereits vor dieser Analyse abgeschlossene, in Abschnitt 6.4 verifizierte
Kriging-Nullkappung).
