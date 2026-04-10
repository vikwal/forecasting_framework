# Data Sources — Structure & Conventions

Dokumentiert alle Rohdaten-Quellen, die im STGNN2-Training (und verwandten Pipelines) verwendet werden.

---

## 1. Stationsmessungen

**Pfad:** `/mnt/lambda1/nvme1/synthetic/wind/era5_fric_wind_hourly_age/synth_{station_id}.csv`

- **Trennzeichen:** Semikolon (`;`)
- **Zeitstempel-Spalte:** `timestamp` (Format: `2023-07-24 00:00:00+00:00`, UTC)
- **Frequenz:** stündlich
- **Station-ID:** 5-stellig, z.B. `00298`

**Wichtige Spalten:**

| Spalte | Beschreibung |
|---|---|
| `wind_speed` | Gemessene Windgeschwindigkeit [m/s] |
| `temperature_2m` | Temperatur in 2 m Höhe [°C] |
| `relative_humidity` | Relative Feuchte [%] |
| `pressure` | Luftdruck [hPa] |
| `wind_direction` | Windrichtung [°] |
| `friction_wind` | Reibungswindgeschwindigkeit |
| `density` | Luftdichte [kg/m³] |
| `power_t1` … `power_t6` | Synthetische Leistung für 6 Turbinen-Typen [W] |
| `wind_speed_t1` … `wind_speed_t6` | Nabenhöhen-extrapolierte Windgeschwindigkeit |

**Stationsmetadaten:** `{data_path}/wind_parameter.csv`
- Trennzeichen: `;`
- Spalten: `park_id` (str), `longitude`, `latitude`, `altitude`, `commissioning_date`

---

## 2. ICON-D2 NWP — ML (Mehrschicht, Hauptquelle für STGNN2)

**Pfad:** `/mnt/lambda1/nvme1/icon-d2/csv/ML/{run_hour}/{station_id}/{lat_lon}_ML.csv`

- **Run-Hours:** `06`, `09`, `12`, `15` (UTC)
- **Anzahl Runs:** ~957 pro Run-Hour (2023-07-24 – 2026-03-06)
- **Dateiformat:** CSV, Komma-getrennt
- **Grid-Punkt-ID im Dateinamen:** `{lat_int}_{lat_dec}_{lon_int}_{lon_dec}_ML.csv`
  - z.B. `54_3107_12_7354_ML.csv` → lat=54.3107, lon=12.7354
- **Trennzeichen:** Komma (`,`)

**Spalten:**

| Spalte | Typ | Beschreibung |
|---|---|---|
| `starttime` | datetime+tz | Run-Initialisierungszeitpunkt (UTC) |
| `forecasttime` | float | Lead-Time in Stunden (0–48) |
| `toplevel` | float | Obere Grenze der Atmosphärenschicht [m AGL] |
| `bottomlevel` | float | Untere Grenze der Atmosphärenschicht [m AGL] |
| `u_wind` | float | Zonalwind [m/s] |
| `v_wind` | float | Meridionalwind [m/s] |
| `temperature` | float | Temperatur [K] |
| `pressure` | float | Luftdruck [Pa] |
| `qs` | float | Spezifische Feuchte [kg/kg] |

**Höhenschichten (toplevel → Schichtmitte):**

| bottomlevel | toplevel | Schichtmitte (≈ Hub-Höhe) |
|---|---|---|
| 0 m | 20 m | **10 m** (`wind_speed_10m`) |
| 20 m | 55.212 m | **38 m** (`wind_speed_38m`) |
| 55.212 m | 100.277 m | **78 m** (`wind_speed_78m`) |
| 100.277 m | 153.438 m | **127 m** (`wind_speed_127m`) |
| 153.438 m | 213.746 m | **184 m** (`wind_speed_184m`) |
| 213.746 m | 280.598 m | **247 m** (`wind_speed_247m`) |

Die Feature-Namen (`wind_speed_38m`, `u_38m`, ...) im YAML werden auf den nächsten Schichtmittelpunkt gemappt.

**Rows pro (run, lead):** 6 (eine pro Höhenschicht)

---

## 3. ICON-D2 NWP — SL (Surface Layer, Sonderfall)

**Pfad:** `/mnt/lambda1/nvme1/icon-d2/csv/SL/{run_hour}/{station_id}/{lat_lon}_SL.csv`

- **Run-Hours:** `06`, `09` (UTC) — **nur 54 Runs** (2023-07-24 – 2023-09-15)
- Für STGNN2 **nicht verwendet** (ML ist die Hauptquelle)

**Spalten:** u_10m, v_10m, t_2m, relhum_2m, clct, aswdir_s_avg, ... (viele Surface-Variablen)

---

## 4. ECMWF NWP — PostgreSQL-Datenbank

**Connection:** Umgebungsvariable `ECMWF_WIND_SL_URL`

**Tabellen:**

### `ecmwf_wind_sl` — NWP-Zeitreihen

| Spalte | Typ | Beschreibung |
|---|---|---|
| `starttime` | timestamptz | Run-Startzeit (00 oder 12 UTC) |
| `forecasttime` | real | Lead-Time in Stunden (0–57) |
| `geom` | geometry(Point,4326) | Grid-Punkt-Koordinaten (PostGIS) |
| `u_wind_10m` | real | Zonalwind 10 m [m/s] |
| `v_wind_10m` | real | Meridionalwind 10 m [m/s] |
| `u_wind_100m` | real | Zonalwind 100 m [m/s] |
| `v_wind_100m` | real | Meridionalwind 100 m [m/s] |
| `u_wind_200m` | real | Zonalwind 200 m [m/s] |
| `v_wind_200m` | real | Meridionalwind 200 m [m/s] |
| `temp_2m` | real | Temperatur 2 m [K] |
| `dew_point_2m` | real | Taupunkt 2 m [K] |
| `specific_rho` | real | Spezifische Luftdichte |
| `friction_velocity` | real | Reibungsgeschwindigkeit [m/s] |

**Run-Hours:** 00 UTC und 12 UTC (= 02:00 und 14:00 CEST in der DB gespeichert)
**Zeitraum:** 2023-07-24 – 2025-10-31
**Anzahl Runs:** 1.662
**Max Lead:** 57 h

### `ecmwf_grid_points` — Grid-Topologie

| Spalte | Beschreibung |
|---|---|
| `geom` | geometry(Point,4326) — Grid-Punkt-Position |

**Anzahl Grid-Punkte:** 759

**KNN-Abfrage:** `geom <-> ST_SetSRID(ST_MakePoint(lon, lat), 4326)` (PostGIS-Operand)

**Spaltenname-Konvention in DB vs. Python:**
- DB: `u_wind_10m` (mit Unterstrich vor Zahl)
- Python-Feature-Name: `u_wind10m` (ohne Unterstrich vor Zahl)
- Konversion: `re.sub(r"(\D)(\d+m)", r"\1_\2", name)` für DB → Python

---

## 5. Feature-Naming-Konvention (STGNN2 YAML)

```yaml
icond2_features: ['u_10m', 'v_10m', 'wind_speed_38m']
ecmwf_features:  ['u_wind10m', 'v_wind10m', 'wind_speed_100m']
measurement_features: ['wind_speed']
target_col: 'wind_speed'
```

**Abgeleitete Features (on-the-fly berechnet):**
- `wind_speed_Xm` → `sqrt(u² + v²)` aus der Schicht mit Mittelpunkt ≈ X m
- Für ICON-D2 ML: Schichtauswahl via `toplevel`/`bottomlevel`-Mittelpunkt, nächste Schicht zu X
- Für ECMWF: `u_wind_Xm` + `v_wind_Xm` direkt als DB-Spalten verfügbar (10m, 100m, 200m)

---

## 6. Sampling-Logik (STGNN2)

Ein Trainings-Sample besteht aus **einem ICON-D2 Run** (`t_run`, `run_hour`):

```
t_run - 48h              t_run              t_run + 48h
     |                      |                      |
     [== hist NWP: Run(t_run-48h), Lead 1..48 ===][== curr NWP: Run(t_run), Lead 1..48 ==]
     [====== Messungen (verfügbar, H=48h) ========]         (Zukunft → nicht verfügbar)
                                                    [============ Target (F=48h) ===========]
```

- **NWP-Sequenz:** `concat(runs[t_run - 48h], runs[t_run])` → `(96, node, feat)`
- **Messungen:** nur `[t_run-48h, t_run)`, für Forecast-Periode auf 0 gesetzt
- **ECMWF:** für jeden ICON-D2 Run den neuesten ECMWF-Run `≤ t_run` nehmen, auf valid_time mergen
- **Training-Indices:** alle gültigen Runs wo auch der History-Run `(t_run - 48h)` existiert