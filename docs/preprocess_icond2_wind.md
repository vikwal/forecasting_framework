# Preprocessing Pipeline: ICON-D2 Wind (preprocess_icond2_wind)

Referenz-Datei: `utils/preprocessing.py`

---

## 1. Datenquellen

| Datei | Inhalt |
|---|---|
| `<path>/synth_<station_id>.csv` | Synthetische Einspeiseleistung pro Turbine (`power_t*`, `wind_speed_t*`) |
| `<path>/wind_parameter.csv` | Standort-Metadaten (Koordinaten, Altitude, Inbetriebnahme-Datum) |
| `<path>/turbine_parameter.csv` | Turbinentypen (Nabenhöhe, Durchmesser, cut-in/-out, rated) |
| `<path>/power_curves/turbine_power.csv` | Leistungskennlinien (kW je Windgeschwindigkeit) |
| `<data.nwp_path>/ML/<forecast_hour>/<station_id>/*.csv` | ICON-D2 NWP-Gitterpunkt-CSV-Dateien |
| `data/stations_master.csv` | Koordinaten aller Stationen (für Gitterpunkt-Abstand) |

---

## 2. ICON-D2 Rohdatenstruktur

Jede NWP-CSV enthält pro Zeitschritt und Modelllevel:

| Spalte | Beschreibung |
|---|---|
| `starttime` | Zeitpunkt des Prognosebeginns (UTC) |
| `forecasttime` | Prognosestunde relativ zu starttime (1..48, `forecasttime=0` wird gefiltert) |
| `timestamp` | `starttime + forecasttime` (abgeleitete absolute Zeit) |
| `toplevel` / `bottomlevel` | Ober-/Untergrenze der Modellschicht (m) |
| `u_wind`, `v_wind` | Horizontale Windkomponenten (m/s) |
| `temperature` | Temperatur (K) |
| `pressure` | Luftdruck (Pa) |
| `qs` | Spezifische Feuchte (kg/kg) |

Verfügbare Prognoseläufe (forecast_hour): `06`, `09`, `12`, `15` UTC.
Jeder Lauf deckt genau **48 Stunden** ab.

---

## 3. Verarbeitungsschritte in `_process_csv_file`

1. **Windgeschwindigkeit**: `wind_speed = sqrt(u_wind² + v_wind²)`
2. **Höhe**: Mittelpunkt zwischen `toplevel` und `bottomlevel` → `height = round((top+bottom)/2)`
3. **Relative Feuchte**: via MetPy `relative_humidity_from_specific_humidity(pressure, temperature, qs)`
4. **Luftdichte** (wenn `params.get_density: True`):
   - Sättigungsdampfdruck: Huang-Formel (standard) oder improved Magnus
   - Moist-air density:
     ```
     p_w = relhum * e_sat
     p_g = pressure - p_w
     rho = p_g / (R_dry * T) + p_w / (R_w * T)
     ```
     mit `R_dry=287.05 J/(kg·K)`, `R_w=461.5 J/(kg·K)`
5. **Pivot nach Höhe**: Jede Variable wird zu `<var>_h<height>` z.B. `wind_speed_h78`, `density_h127`
6. **Suffix für Gitterpunkt-Rang**: Nächste N Gitterpunkte (nach geodätischer Distanz) erhalten Suffix `_1`, `_2`, ... → `wind_speed_h78_1`, `wind_speed_h78_2`

---

## 4. Aggregation über Höhen: Rotor-äquivalente Größen

Wenn `params.aggregate_nwp_layers: 'weighted_mean'`:

### Rotor-äquivalente Windgeschwindigkeit (`wind_speed_rotor_eq`)

Physikalisch korrekte Aggregation über Rotorfläche:

1. Rotorkreis wird in Höhenschichten (entsprechend verfügbarer NWP-Level) aufgeteilt
2. Für jede Schicht: **Flächen-Anteil** des Rotors via `get_A_weights()` (Kreissegment-Geometrie mit `circle_cap_area`)
3. Energieäquivalente Mittelung:
   ```
   wind_speed_rotor_eq = ( sum_i(A_i * v_i^3) )^(1/3)
   ```
   → Erhält die kinetische Energie des Windes (Leistung ∝ v³)

### Rotor-äquivalente Luftdichte (`density_rotor_eq`)

Flächengewichteter Mittelwert:
```
density_rotor_eq = sum_i(A_i * rho_i)
```

`get_A_weights` berechnet für jeden Turbinentyp (Durchmesser + Nabenhöhe) den Flächen-Anteil jeder Schicht und mittelt über alle Turbinen des Parks.

---

## 5. Ausgabestruktur von `preprocess_synth_wind_icond2`

DataFrame mit **MultiIndex** `['starttime', 'forecasttime', 'timestamp']`:

- Jede Zeile ist ein Zeitschritt eines bestimmten Prognose-Starts
- Jede `starttime` enthält genau **48 Zeilen** (forecasttime 1..48)
- Unvollständige Prognoseläufe (< 48 Schritte) werden verworfen

Beispiel-Spalten (je nach Config):

| Spalte | Herkunft | Feature-Typ |
|---|---|---|
| `power` | synth CSV, normiert auf `installed_capacity` | observed |
| `wind_speed_h78` | ICON-D2, Höhe ~78m | known |
| `wind_speed_h127` | ICON-D2, Höhe ~127m | known |
| `wind_speed_h184` | ICON-D2, Höhe ~184m | known |
| `density_rotor_eq` | Flächengewichtete ICON-D2 Dichte | known |
| `park_age`, `hub_height`, ... | wind_parameter / turbine_parameter | static (optional) |

---

## 6. Feature-Zuweisung (known / observed / static)

Konfiguriert in `params.known_features`, `params.observed_features`, `params.static_features` in der YAML-Config.

- **observed**: Gemessene Vergangenheitswerte (primär `power`) — nur im Lookback-Fenster verfügbar
- **known**: NWP-Prognosen — sowohl im Lookback (aus altem Prognose-Lauf) als auch im Forecast-Horizont (aus aktuellem Lauf) verfügbar
- **static**: Stations-/Turbinen-Parameter, zeitlich konstant

`prepare_features_for_tft()` übersetzt die Feature-Namen der Config auf tatsächliche Spalten im DataFrame (Substring-Matching inkl. Gitterpunkt-Suffix).

---

## 7. Windowing in `create_tft_sequences` — NWP-aware

Da ICON-D2 Prognoseläufe á 48 Stunden liefert, ist das Windowing **nicht linear**, sondern lauf-basiert:

```
Zeitlinie:
  t-48h              t-0h             t+48h
  |--- Lauf (t-48h) ---|--- Lauf (t-0h) ---|
  |   forecasttime 1..48 |  forecasttime 1..48 |

known_window     = [Lauf(t-48h)] + [Lauf(t-0h)]   → 96 Schritte
                    (vergangene NWP)   (aktuelle NWP)
observed_window  = power[t-lookback ... t-1]         → lookback Schritte (Timestamp-Lookup)
target (y)       = power[t ... t+47]                 → horizon Schritte
```

Pro Fenster (= pro Prognose-Startzeit `current_start`):
- `X_known`: Immer genau `2 × horizon = 96` Schritte (vorheriger Lauf + aktueller Lauf)
- `X_observed`: Genau `lookback` Schritte, per Timestamp-Lookup aus dem MultiIndex
- `y`: Exakt `horizon=48` Schritte aus aktuellem Lauf

---

## 8. Lookback > 48: Unterstützung und Funktionsweise

`create_tft_sequences` berechnet die Anzahl benötigter vergangener Prognoseläufe dynamisch:

```python
n_past_runs = math.ceil(history_len / future_len)
```

Für jeden Forecast-Startzeitpunkt werden `n_past_runs` ältere Läufe (ältester zuerst) plus der aktuelle Lauf gestapelt:

```
Beispiel lookback=96, horizon=48, step_size=48:

  t-96h          t-48h           t-0h           t+48h
  |-- Lauf(t-96h) --|-- Lauf(t-48h) --|-- Lauf(t-0h) --|
     (past chunk 1)    (past chunk 2)    (future NWP)

known_window = [Lauf(t-96h)] + [Lauf(t-48h)] + [Lauf(t-0h)]
             → Shape (144, n_features)  = lookback + horizon  ✓
```

**Voraussetzung**: `lookback` sollte ein Vielfaches von `step_size` (=`horizon`) sein, damit alle past chunks vollständig befüllt werden. Bei nicht-ganzzahligem Vielfachen verwendet `ceil()` einen weiteren Lauf, der dann im `known_past`-Teil leicht übersteht — das passt aber zur Slicing-Logik in `get_y_chronos2`.

**Fenster-Verfügbarkeit**: Ein Fenster wird nur erzeugt, wenn alle `n_past_runs` Prognoseläufe vollständig (exakt `future_len` Schritte) im Datensatz vorhanden sind. Für `lookback=96` fallen die ersten 2 Startzeitpunkte eines Datensatzes weg (analog zu wie bisher der erste bei `lookback=48` wegfiel).

**Betroffene Modelle**: TFT und Chronos (beide über `prepare_data_for_tft` → `create_tft_sequences`).

---

## 9. Skalierung

- `X_known`, `X_observed` (außer `power`): `StandardScaler` — gefittet auf Trainingsdaten, gespeichert für Inverse-Transform
- `power` (observed): **Keine Skalierung** — normiert auf `[0,1]` durch Division durch `installed_capacity`
- Chronos-target (`y`): **Keine Skalierung** — Chronos-2 übernimmt interne Normalisierung

---

## 10. Parallelisierung

- **Forecast-Stunden** (06, 09, 12, 15): `ProcessPoolExecutor` (CPU-bound)
- **Gitterpunkt-CSV-Dateien**: `ThreadPoolExecutor` (I/O-bound)
