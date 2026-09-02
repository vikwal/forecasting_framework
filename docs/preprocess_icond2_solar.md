# Solar-Preprocessing mit ICON-D2 (`utils/solar.py`)

Pendant zu [preprocess_icond2_wind.md](preprocess_icond2_wind.md). Verbindet die
DWD-Stationsmessungen der Global- und Diffusstrahlung mit den ICON-D2-Surface-Level-
Prognosen und liefert denselben Datenkontrakt wie die Wind-Pipeline.

Einstieg: `utils.solar.preprocess_solar_icond2(path, config, freq, features)`.
Geroutet wird über `data.use_case: solar` in
`utils/preprocessing.py::_get_data_from_config_files`.

---

## 1. Datenquellen

### 1.1 Messungen

`{data.path}/Station_<id>.parquet` — 204 Dateien unter
`/mnt/lambda1/nvme1/synthetic/raw/solar`.

| | |
|---|---|
| Auflösung | 10 min |
| Zeitraum | 2023-07-24 … 2026-08-04 |
| Index | `timestamp`, tz-aware UTC |
| Spalten | `station_id`, `ghi`, `dhi`, `temperature_2m`, `wind_speed`, `precipitation_rate`, `precipitation_duration` |
| **Stationen mit Strahlungsdaten** | **93 von 204** — bei den übrigen 111 sind `ghi`/`dhi` durchgehend NaN |

**Einheit:** `ghi`/`dhi` sind DWD-`GS_10`/`DS_10`, also **J/cm² je Messintervall**.
`load_station_measurements()` rechnet um:

```
W/m² = J/cm² · 10⁴ / Δt[s]        Δt aus dem tatsächlichen Abtastintervall
```

Bei 10 min ist der Faktor 16.667. Ohne diese Umrechnung stünde das Ziel um Faktor ~16.7
neben der ICON-D2-Prognose — die Pipeline würde trainieren, aber Unsinn lernen.

### 1.2 ICON-D2 Surface-Level (SL)

`{data.nwp_path}/SL/{forecast_hour}/<lon>_<lat>_SL.parquet`

| | |
|---|---|
| Auflösung | 15 min für die Strahlungsfelder, 1 h für alle übrigen — siehe [§ 2.5](#25-nur-die-strahlungsfelder-sind-viertelstündlich) |
| `forecasttime` | 0.0 … 48.0 in fraktionalen Stunden (193 Schritte je Lauf) |
| Läufe | 06, 09, 12, 15 UTC |
| Zeitraum | 2023-07-24 … 2026-06-27 |
| Gitterpunkte | 1218 |
| Spalten | `starttime`, `forecasttime`, `longitude`, `latitude`, `aswdifd_s_avg`, `aswdir_s_avg`, `aswdifd_s`, `aswdir_s`, `alb_rad`, `clct`, `t_2m`, `relhum_2m`, `td_2m`, `t_g`, `prr_gsp`, `prs_gsp`, `prg_gsp`, `h_snow`, `rho_snow`, `u_10m`, `v_10m`, `delivery_hour` |

---

## 2. Fallstricke der SL-Daten

Alle sind an den Daten verifiziert und in der Implementierung behandelt.

### 2.1 Dateinamen sind lon-first — anders als bei ML

| Layer | Beispiel | Bedeutung |
|---|---|---|
| ML (Wind) | `52_9057_12_9151_ML.parquet` | lat 52.9057, lon 12.9151 |
| **SL (Solar)** | `10_0000_47_8000_SL.parquet` | **lon 10.0, lat 47.8** |

Nachweis: Feld 0 hat über alle 1218 SL-Dateien den Wertebereich 6.01–14.98 (Längengrad
Deutschlands), Feld 1 den Bereich 47.38–55.03 (Breitengrad).

Zusätzlich sind die Spalten *innerhalb* der SL-Datei vertauscht: `longitude` enthält
54.7, `latitude` enthält 13.4. Deshalb liest `utils.solar.parse_sl_lonlat()` die
Koordinaten ausschließlich aus dem Dateinamen.

Mit der ML-Konvention gelesen läge der „nächste" Gitterpunkt Hunderte km entfernt.

### 2.2 SL ist flach, ML ist nach Station gruppiert

```
ML/06/00096/52_9057_12_9151_ML.parquet      ← Unterverzeichnis je Station
SL/06/10_0000_47_8000_SL.parquet            ← alle 1218 Gitterpunkte direkt hier
```

`_scan_sl_grid()` listet das Verzeichnis einmal pro `(nwp_path, forecast_hour)` und
cached es via `lru_cache` — sonst würden 1218 Dateien für jede der 93 Stationen erneut
gelistet.

### 2.3 Strahlungsfelder sind Intervallmittel, die auf `forecasttime` enden

`aswdir_s`/`aswdifd_s` mitteln das Intervall, das **auf** `forecasttime` endet; alle
übrigen Felder (`t_2m`, `clct`, `alb_rad`, `u_10m`, `h_snow`, …) sind Momentanwerte.

Nachweis am Datensatz:

```
mean(aswdifd_s über ft ∈ (1, 2])            = 127.66341
aswdifd_s_avg@2 · 2 − aswdifd_s_avg@1 · 1   = 127.66341
```

Die Messungen werden mit `closed='left', label='left'` gemittelt — der Wert bei
Zeitstempel *T* beschreibt also [T, T+freq). Daraus folgen zwei verschiedene
Lead-Zuordnungen:

| Feldtyp | Lead-Index | Begründung |
|---|---|---|
| akkumuliert (`aswdir_s`, `aswdifd_s`, `ghi_nwp`, …) | `ceil(ft/step) − 1` | Intervall (ft−step, ft] entspricht dem Label ft−step |
| Momentanwert (alles andere) | `floor(ft/step)` | Zustand zu Beginn des Intervalls |

Beide Zweige werden getrennt aggregiert und auf `(starttime, lead_idx)` gemerged;
einheitlich gilt `timestamp = starttime + lead_idx · step`. Resultat sind Leads
`0 … n_leads−1` — 48 Schritte je Lauf bei `freq: '1h'` (wie bei Wind), 192 bei
`'15min'`.

**Empirische Bestätigung** (Station 00183, Apr–Sep 2025, Messung vs. `ghi_nwp_1`):

| NWP-Verschiebung | Korrelation | RMSE |
|---|---|---|
| −2 h | 0.766 | 173.1 W/m² |
| −1 h | 0.899 | 114.8 W/m² |
| **0 h** | **0.960** | **73.9 W/m²** |
| +1 h | 0.936 | 92.4 W/m² |
| +2 h | 0.834 | 146.3 W/m² |

Ein gemeinsames `floor`-Mapping für alle Felder (wie in der Wind-Pipeline, wo alles
Momentanwerte sind) entspräche der +1-h-Zeile.

**Die `_avg`-Spalten sind etwas anderes und dürfen kein Feature sein.**
`aswdir_s_avg`/`aswdifd_s_avg` tragen das Mittel **seit Vorhersagebeginn** `[0, ft]`,
nicht das Intervall `(ft−step, ft]`. Verifiziert: das kumulative Mittel von
`aswdifd_s` bis `ft` ist gleich `aswdifd_s_avg` bei `ft`.

| `ft` | `cummean(aswdifd_s)` | `aswdifd_s_avg` |
|---|---|---|
| 1 | 42.1283 | 42.1283 |
| 2 | 61.0004 | 61.0004 |
| 6 | 99.1351 | 99.1351 |
| 24 | 49.1371 | 49.1385 |
| 48 | 62.1123 | 62.1122 |

Für ein Laufmittel passt **keine** der beiden Lead-Zuordnungen — weder `ceil`
(Intervallende) noch `floor` (Momentanwert). Als Feature durchgereicht wäre es ein
über den Lauf wanderndes Mittel mit dem Zeitstempel eines einzelnen Schritts, und
zwar eines, dessen Bedeutung sich mit dem Lead ändert. `_reject_running_means()`
bricht deshalb ab und nennt die dekumulierte Entsprechung (`aswdifd_s`), die in
derselben Datei steht.

**`forecasttime = 0` ist eine Schein-Null.** Bei allen vier Strahlungsspalten ist der
Wert dort in 99.3 % der Läufe **exakt 0.0** (Rest NaN) — es gibt kein vorangehendes
Intervall, aus dem dekumuliert werden könnte. Von „Nacht" ist das nicht zu
unterscheiden. Im Preprocessing ist der Fall ausgeschlossen, weil
`ceil(0/step) − 1 = −1` die Zeile verwirft und Lead 0 aus `ft = step` zieht
(Kontrolle: Lead 0 hat 0 von 24 Werten gleich null, Mittel 303.8 W/m²). Wer die
SL-Parquets **direkt** liest, muss selbst filtern.

### 2.4 Zeitzone von `starttime`

`starttime` ist mit **festem +02:00-Offset ganzjährig** gespeichert, ohne
Sommerzeitumstellung (2023-11-01 08:00+02:00, 2025-01-07 08:00+02:00).
`pd.to_datetime(..., utc=True)` liefert daher immer die Stunde aus `delivery_hour`.

### 2.5 Nur die Strahlungsfelder sind viertelstündlich

Die SL-Dateien haben 193 `forecasttime`-Schritte, aber **nicht jede Spalte ist auf
allen besetzt**. Anteil NaN, gemessen über alle Läufe ab 2024-01-01:

| Spalten | auf voller Stunde | auf Viertelstunde |
|---|---|---|
| `aswdir_s`, `aswdifd_s` (und die Laufmittel `*_avg`, s. § 2.3) | 0.6 % | **0.0 %** |
| `clct`, `t_2m`, `alb_rad`, `relhum_2m`, `td_2m`, `t_g`, `prr_gsp`, `prs_gsp`, `prg_gsp`, `h_snow`, `rho_snow`, `u_10m`, `v_10m` | 0.1 % | **100 %** |

Praktisch heißt das: `freq: '15min'` ist nur für die Strahlung eine echte
Verfeinerung. Ohne Behandlung wären drei von vier Leads in jeder Nicht-Strahlungs-
Spalte NaN, das nachgelagerte `dropna()` würde sie verwerfen, und `'15min'` fiele
stillschweigend auf ein Stundenraster zurück — mit dem Vierfachen an Ladezeit.

`params.sub_hourly_fill` steuert das (`solar._fill_sub_hourly`), gefüllt wird
**nur innerhalb eines Laufs** und höchstens über eine native Stunde:

| Wert | Verhalten |
|---|---|
| `ffill` (Default) | Stundenwert gilt bis zur nächsten vollen Stunde fort |
| `interpolate` | linear zwischen den Stundenwerten; nach dem letzten Anker fortgeschrieben, sonst verlöre jeder Lauf seine Endleads |
| `none` | nicht füllen — die Leads fallen weg, `'15min'` wird faktisch zu `'1h'` |

Beispiel `clct` (Station 00183, Lead 0–8 bei `'15min'`):

```
ffill        99.82  99.82  99.82  99.82  89.12  89.12  89.12  89.12  95.90
interpolate  99.82  97.14  94.47  91.79  89.12  90.81  92.51  94.21  95.90
```

Der Default ist `ffill`, weil er keine Werte erfindet und zur ohnehin verwendeten
Momentanwert-Semantik passt. `interpolate` ist für träge Größen wie `t_2m`
plausibler, für `clct` dagegen Fiktion — deshalb konfigurierbar statt verdrahtet.

### 2.6 Die DWD-Zeitstempel markieren das Intervallende

`params.measurement_time_label` legt fest, ob der Rohzeitstempel den Beginn (`left`)
oder das Ende (`right`) des 10-min-Intervalls meint. Der Verschiebungstest gegen
`ghi_nwp` beantwortet das eindeutig — **7 Stationen, Apr–Sep 2024, RMSE in W/m²**:

| | | −1 Schritt | 0 | +1 Schritt |
|---|---|---|---|---|
| Station 00183, `15min` | `left` | 84.43 | 79.71 | **78.82** |
| | `right` | 80.69 | **78.46** | 80.67 |
| Station 00183, `1h` | `left` | 107.43 | **69.02** | 91.55 |
| | `right` | 98.89 | **68.38** | 99.78 |

Entscheidend ist nicht der kleinere Absolutwert, sondern die **Symmetrie**: mit
`right` liegt das Minimum sauber auf 0 und fällt nach beiden Seiten gleich ab — die
Signatur korrekter Ausrichtung. Mit `left` ist die Kurve nach +1 gezogen, bei
`'15min'` sogar über das Minimum hinaus. Das Bild ist über alle sieben geprüften
Stationen (00183, 00232, 00591, 01346, 02115, 03032, 05426) identisch.

**Default ist deshalb seit Aug 2026 `'right'`** (vorher `'left'`). Bei stündlicher
Auflösung mittelt sich der 10-min-Versatz weitgehend heraus (69.0 → 68.4 W/m²,
≈ 1 %); erst `'15min'` macht ihn sichtbar. Der GNN-Zweig
(`train_stgnn2.load_station_measurements`) wendet dieselbe Verschiebung an, sonst
wären die beiden Pfade um 10 min gegeneinander versetzt.

---

## 2b. Zeitliche Auflösung (`data.freq`)

`solar.resolve_freq(freq)` prüft den Config-Wert und liefert `(step_h, n_leads)`:

| `data.freq` | `step_h` | Leads je Lauf |
|---|---|---|
| `'15min'` | 0.25 | 192 |
| `'30min'` | 0.5 | 96 |
| `'1h'` (Default) | 1.0 | 48 |
| `'2h'` | 2.0 | 24 |

Zulässig ist **jedes Vielfache von 15 min, das 48 h ganzzahlig teilt**. `'20min'`,
`'10min'` und `'5h'` werden mit einer expliziten Meldung abgelehnt, statt später als
leerer Datensatz aufzufallen.

**Wind kann das nicht.** Die ML-Dateien haben nur 49 `forecasttime`-Schritte
(stündlich). `geostatistics/shared/resolution.py::freq_to_hours(freq, use_case)`
kennt beide nativen Raster und lehnt `freq: '15min'` auf einer Wind-Config ab. Das
ersetzt acht kopierte Lookup-Tabellen mit stillem `.get(freq, 1.0)`-Fallback —
in `evaluate_reference.py` fehlte `'15min'` dort sogar ganz, ein Solar-Lauf mit
Viertelstundenraster wäre dort kommentarlos stündlich ausgewertet worden.

### Messseite: 10 min nestet nicht in 15 min

Die Rohmessungen liegen im 10-min-Raster. Für `'1h'` ist das arithmetische Mittel
exakt (sechs Werte je Stunde). Für `'15min'` **nicht**: das Intervall [00:00, 00:15)
enthält den Wert von 00:00 zu 10 min und den von 00:10 zu 5 min, [00:15, 00:30) nur
den von 00:20. Ein einfaches `.mean()` gewichtet beide gleich und bildet das zweite
Intervall aus einem einzigen Wert.

`solar.resample_interval_mean()` expandiert deshalb auf das gemeinsame Feinraster
(ggT von 10 und 15 min = 5 min) und mittelt erst dort — da alle Feinzellen gleich
lang sind, ist das exakt das flächengewichtete Intervallmittel:

```
Rohwerte 10 min:  v0=0  v1=1  v2=2  v3=3 …
korrekt (15 min): 0.333  1.667  3.333  4.667 …
naiv    (15 min): 0.5    2.0    3.5    5.0   …   ← systematisch zu hoch
```

Für `'1h'` greift der Ganzzahl-Zweig, die Ergebnisse sind bit-identisch zu vorher.
Lücken breiten sich nicht aus: ein Rohwert gilt nur über sein eigenes Intervall,
und ein Zielintervall wird verworfen, sobald weniger als `1 − max_nan_frac` seiner
Dauer gedeckt ist. Fehlende Zeilen und `NaN`-Zeilen verhalten sich identisch.

### Was bei `freq` sonst noch mitzuziehen ist

* **Modell-Horizont.** `model.horizon`/`output_dim`/`lookback` zählen in Schritten,
  nicht in Stunden. 48 h bedeuten bei `'15min'` **192**, nicht 48.
* **Frühe Läufe fallen weg.** Rund 2 % der SL-Läufe (bis ~2023-08-08) sind nur
  stündlich abgelegt und erfüllen die 192-Lead-Prüfung nicht. Das wird auf INFO
  geloggt, damit der Datensatz nicht grundlos kürzer wirkt.
* **Der Cache-Hash** enthält `freq` bereits — ein Wechsel erzwingt Neuberechnung.

---

## 3. Feature-Katalog

### 3.1 Abgeleitete NWP-Features

| Name | Berechnung |
|---|---|
| `ghi_nwp` | `aswdir_s + aswdifd_s` |
| `dhi_nwp` | `aswdifd_s` |
| `bhi_nwp` | `aswdir_s` |
| `dni_nwp` | `bhi_nwp / cos θz` (pvlib, gedeckelt gegen Clear-Sky-DNI) |
| `kt_nwp` | `ghi_nwp / ghi_clearsky` |
| `kd_nwp` | `dhi_nwp / ghi_nwp` |
| `wind_speed_nwp` | `sqrt(u_10m² + v_10m²)` |

Alle übrigen Namen in `params.icond2_features` werden direkt als SL-Spalte gelesen.
Spaltenbenennung wie bei Wind: `<feature>_<rang>`, Rang 1 = nächster Gitterpunkt.

### 3.2 Sonnengeometrie (stationsbezogen, ohne Gitterpunkt-Suffix)

`solar_zenith`, `solar_zenith_cos`, `solar_azimuth_sin`, `solar_azimuth_cos`,
`airmass`, `dni_extra`, `ghi_clearsky`, `dni_clearsky`, `dhi_clearsky`,
`hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`

Der Sonnenstand wird zur **Intervallmitte** ausgewertet (`label + freq/2`), weil die
Zeitstempel linksbündige Intervalllabels sind — sonst wäre der Zenitwinkel bei
stündlicher Auflösung systematisch um eine halbe Stunde versetzt.
Clear-Sky über `pvlib.clearsky.ineichen` mit Linke-Turbidity-Klimatologie.

### 3.3 Abgeleitete Zielgrößen

| Name | Berechnung |
|---|---|
| `bhi` | `clip(ghi − dhi, 0)` |
| `dni` | `pvlib.irradiance.dni(ghi, dhi, zenith, dni_clear=dni_clearsky)`, 0 bei θz > 88° |
| `kt` | `ghi / ghi_clearsky`, 0 unterhalb 20 W/m² Clear-Sky, gedeckelt bei 1.5 |
| `kd` | `dhi / ghi`, auf [0, 1] beschränkt |

---

## 4. Config-Schlüssel

| Schlüssel | Bedeutung |
|---|---|
| `data.use_case: solar` | schaltet das Routing auf `preprocess_solar_icond2` |
| `data.target_cols` | Liste von Zielspalten → Multi-Output; alternativ `data.target_col` |
| `data.forecast_hours` | ICON-D2-Läufe, Default `['06','09','12','15']` |
| `data.freq` | Zielauflösung, Vielfaches von 15 min das 48 h teilt — s. § 2b |
| `params.icond2_features` | zu ladende SL-Features (Namen s. o.) |
| `params.next_n_grid_points` | Zahl der nächstgelegenen SL-Gitterpunkte |
| `params.target_transform` | `none` (Default) \| `clearsky_index` |
| `params.measurement_time_label` | `right` (Default) \| `left` — markiert der Rohzeitstempel Intervallende oder -beginn? S. § 2.6 |
| `params.sub_hourly_fill` | `ffill` (Default) \| `interpolate` \| `none` — Auffüllen der nur stündlichen SL-Felder bei `freq` < 1 h, s. § 2.5 |
| `params.max_nan_frac` | ab diesem Anteil ungedeckter Intervalldauer wird das Zielintervall NaN |
| `params.nwp_baseline_col` | Referenzspalte für `Skill_NWP` (Default `ghi_nwp`) |
| `params.daytime_zenith_threshold` | Grenze für die Tagstunden-Metriken |
| `params.next_n_stations` | Nachbarstationen für `<feature>_next_<rang>` |
| `params.nwp_workers` | Threads für das SL-Laden |

---

## 5. Bekannte Limitierungen

### 5.1 Datenverfügbarkeit

Gemessen gegen ein gemeinsames 10-min-Raster 2023-07-24 … 2026-08-04
(159 552 Slots je Station); fehlende Zeilen zählen wie NaN-Werte.

#### Alle 204 Stationen, je Spalte

| Spalte | Ø fehlend | Median | Min | Max |
|---|---|---|---|---|
| `ghi` | 59.1 % | 100 % | 0.28 % | 100 % |
| `dhi` | 59.4 % | 100 % | 0.29 % | 100 % |
| `temperature_2m` | 1.0 % | 0.45 % | 0.00 % | 48.2 % |
| `wind_speed` | 1.2 % | 0.64 % | 0.00 % | 48.2 % |
| `precipitation_rate` | 53.0 % | 26.5 % | 24.6 % | 100 % |
| `precipitation_duration` | 55.6 % | 26.4 % | 24.6 % | 100 % |

Der Median von 100 % bei `ghi`/`dhi` kommt daher, dass **111 der 204 Stationen gar
keine Strahlungsmessung führen**. Für alles Weitere zählen nur die **93 Stationen mit
Strahlungsdaten**.

#### Die 93 Stationen mit Strahlungsdaten

Durchschnittlich **10.4 % fehlendes `ghi`**, Median aber nur **1.44 %** — die Verteilung
ist stark schief:

| Lücken-Anteil | Stationen |
|---|---|
| < 1 % | 35 |
| 1–2 % | 17 |
| 2–5 % | 12 |
| 5–10 % | 4 |
| 10–25 % | 9 |
| 25–50 % | 11 |
| > 50 % | 5 |

Zerlegt nach Ursache (Mittel über die 93):

| Ursache | Anteil |
|---|---|
| Vorlauf — Messung beginnt später | 0.71 % |
| **Nachlauf — Reihe endet vorzeitig** | **7.63 %** |
| Echte Lücken im aktiven Zeitraum | 2.11 % (Median 0.70 %) |

Das Problem ist also überwiegend **kein Rauschen, sondern abgebrochene Reihen**:
92 der 93 Stationen starten 2023-07/08 (einzige Ausnahme 04642 ab 2025-06), aber nur
55 laufen bis 2026-08 durch; 27 hören früher auf, 8 davon bereits zwischen 2024-12 und
2025-04. Die echten Messlücken innerhalb des aktiven Zeitraums liegen bei der Hälfte
der Stationen unter 0.7 %.

#### Auswirkung auf die beiden Pfade

* **CL/FL-Pfad (`train_cl.py`, `train_fl.py`) läuft.** Jede Station wird einzeln
  verarbeitet, `dropna()` entfernt nur ihre eigenen betroffenen Zeilen.
* **GNN-Pfad (DCRNN/MTGNN/WaveNet)** braucht einen dichten `(T, N, M)`-Block über
  *alle* Stationen gleichzeitig. Damit multiplizieren sich die Lücken auf:

| Stationsmenge | n | Std. gleichzeitig verfügbar | nutzbare 96-h-Fenster |
|---|---|---|---|
| alle 93 | 93 | 0.0 % | **0.0 %** |
| < 25 % Lücken | 78 | 45.0 % | 7.1 % |
| < 10 % Lücken | 70 | 55.1 % | 10.7 % |
| < 5 % Lücken | 66 | 61.0 % | 13.4 % |
| < 2 % Lücken | 56 | 77.6 % | 20.3 % |
| < 1 % Lücken | 42 | 90.9 % | **37.0 %** |

(96 h = 48 h Historie + 48 h Horizont, Fenster `2023-08-01 … 2026-06-27`.)

Verifiziert: ein MTGNN-Smoke-Lauf über 20 Stationen ohne Filterung erreicht
`Run pairs — train: 196  val: 0  skipped: 557`.

**Zwei Wege für den GNN-Pfad**, in dieser Reihenfolge sinnvoll:

1. **Stationsfilter** — Stationen oberhalb einer Lückenschwelle aus den Configs
   nehmen. Ohne jede Imputation bringt eine `< 1 %`-Auswahl bereits 42 Stationen und
   37 % nutzbare Fenster. Kostet Netzabdeckung, ist aber sofort verfügbar.
2. **Imputation** — Lücken füllen und über `data.interpol_path` /
   `data.knnimputer_path` einhängen. Bausteine vorhanden
   (`geostatistics/run_solar_interpolation.py`, `utils/imputation.py`,
   `utils/era5_imputation.py`), Ausgabeverzeichnisse noch nicht erzeugt.
   Erst damit werden alle 93 Stationen nutzbar.

### 5.2 Weitere Limitierungen

* **ECMWF fehlt.** `params.nwp_models: ['icon-d2']`; der Merge-Mechanismus der
  Wind-Pipeline ist übertragbar, sobald die Daten vorliegen.
* **Keine gemessene Direktstrahlung.** `bhi`/`dni` sind abgeleitet und erben damit die
  Fehler von `ghi` und `dhi`; bei tiefstehender Sonne ist `dni` numerisch heikel und
  wird oberhalb θz = 88° auf 0 gesetzt.
* **Nur `clct`, keine Schichtbedeckung.** Die SL-Dateien enthalten kein
  `clcl`/`clcm`/`clch`.
* **`precipitation_rate`/`precipitation_duration`** sind nur zu ~17 % befüllt und
  sollten nicht als Pflicht-Feature gesetzt werden.
* **Zeitstempel-Konvention der Rohdaten** ist nicht abschließend geklärt: DWD legt
  `MESS_DATUM` bei 10-min-Solardaten teils auf das Intervallende. Bei stündlicher
  Aggregation macht das höchstens 10 min aus; über `params.measurement_time_label`
  umschaltbar.
