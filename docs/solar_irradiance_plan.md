# Solar Irradiance Forecasting — Implementierungsplan

Status: **Entwurf** (Stand 2026-08-11). Analog zum Wind-Use-Case (`preprocess_synth_wind_icond2`)
soll eine vollständige, config-gesteuerte Pipeline für Globalstrahlung / Diffus- / Direktstrahlung
auf Basis von ICON-D2 (und später ECMWF) entstehen.

---

## 1. Datenlage (verifiziert)

### 1.1 Messdaten (DWD-Stationen)

`/mnt/lambda1/nvme1/synthetic/raw/solar/Station_<id>.parquet` — 204 Dateien.

| Eigenschaft | Wert |
|---|---|
| Auflösung | **10 min** |
| Zeitraum | 2023-07-24 … 2026-08-04 (159 552 Zeilen) |
| Index | `timestamp` (tz-aware UTC) |
| Spalten | `station_id`, `ghi`, `dhi`, `temperature_2m`, `wind_speed`, `precipitation_rate`, `precipitation_duration` |
| **Stationen mit Strahlungsdaten** | **93 von 204** — bei 111 Stationen sind `ghi`/`dhi` komplett `NaN` |
| Einheit `ghi`/`dhi` | **J/cm² je 10 min** (DWD `GS_10`/`DS_10`), Max ≈ 59.8 → 59.8·10⁴/600 ≈ 997 W/m² |
| Lücken | ~0.3 % NaN bei Strahlung, `precipitation_*` nur ~17 % befüllt |

Keine Direktstrahlung im Datensatz → muss abgeleitet werden:
- `bhi = ghi − dhi` (Direktstrahlung auf die Horizontale)
- `dni = bhi / cos(θ_z)` (via `pvlib.irradiance.dni`, robust gegen kleine cos θ_z)

Ein älterer Snapshot liegt unter `/mnt/lambda1/nvme1/synthetic/solar/solar_hourly/`
(trotz Name ebenfalls 10 min, endet 2026-05-05) — nicht verwenden.

**Aufgeräumt:** Die 204 `Station_*.csv` in `raw/solar/` waren fälschlich benannte Parquet-Dateien
(älterer Snapshot bis 2026-06-16), spaltenweise deckungsgleich mit den `.parquet`-Dateien auf dem
gemeinsamen Zeitbereich. Nach Verifikation aller 204 Paare gelöscht.

### 1.2 Stationsmetadaten

`data/stations_master.csv` — 203 Stationen mit `station_id`, `station_height`, `longitude`, `latitude`.
Deckt die Solar-Stations-IDs ab (identische ID-Systematik wie Wind).

### 1.3 ICON-D2 Surface-Level (SL)

`/mnt/lambda1/nvme1/icon-d2/parquet/SL/{06,09,12,15}/<lon>_<lat>_SL.parquet` — **flach**, 1218 Gitterpunkte.

| Eigenschaft | Wert |
|---|---|
| Auflösung | 15 min, `forecasttime` 0.0 … 48.0 (fraktionale Stunden) |
| Runs | 06, 09, 12, 15 UTC-Läufe; `starttime` in **lokaler Zeit (+02:00)** gespeichert |
| Zeitraum | 2023-07-24 … 2026-06-27 |
| Spalten | `starttime`, `forecasttime`, `longitude`, `latitude`, `aswdifd_s_avg`, `aswdir_s_avg`, `aswdifd_s`, `aswdir_s`, `alb_rad`, `clct`, `t_2m`, `relhum_2m`, `td_2m`, `t_g`, `prr_gsp`, `prs_gsp`, `prg_gsp`, `h_snow`, `rho_snow`, `u_10m`, `v_10m`, `delivery_hour` |

**Zwei verifizierte Fallstricke:**

1. **Koordinaten-Konvention ist gegenüber ML vertauscht.**
   ML-Dateien heißen `<lat>_<lon>_ML.parquet` (`52_9057_12_9151` → lat 52.9, lon 12.9).
   SL-Dateien heißen `<lon>_<lat>_SL.parquet` (`10_0000_47_8000` → **lon 10.0, lat 47.8**).
   Feld 0 hat den Wertebereich 6.01–14.98 (Längengrad), Feld 1 47.38–55.03 (Breitengrad).
   Zusätzlich sind die Spalten *innerhalb* der Parquet-Datei vertauscht
   (`longitude` = 47.8, `latitude` = 10.0).
2. **SL ist flach, nicht nach Station gruppiert.** Es gibt kein `SL/06/<station_id>/`.

→ Beides bricht das bestehende `geostatistics/solar_preprocessing.py`
(`_parse_latlon` erwartet lat-first, `_select_nearest_sl_files` erwartet `sid_dir`).
Der Solar-Pfad in `train_dcrnn.py` / `train_mtgnn.py` / `train_wavenet.py` ist damit aktuell
nicht lauffähig. Wird in Phase 4 mitgefixt.

### 1.4 ECMWF

Noch nicht beschafft. Die Pipeline bekommt denselben Merge-Mechanismus wie Wind
(`_fetch_ecmwf_data_from_split_parquets`, Run-Zuordnung 00/12 UTC), aber per Config abschaltbar
(`params.nwp_models: ['icon-d2']`). Benötigte ECMWF-Felder für Solar: `ssrd` (surface solar
radiation downwards, akkumuliert), `fdir` (direkte Komponente), `tcc`, `2t`, `10u`/`10v`.

---

## 2. Abbildung Wind → Solar

| Wind | Solar-Pendant |
|---|---|
| `preprocess_synth_wind_icond2()` | **`preprocess_solar_icond2()`** (neu) |
| ICON-D2 **ML** (Multilevel, per-Station-Ordner) | ICON-D2 **SL** (Surface, flacher Gitter-Ordner) |
| `wind_speed_h{78,127,184}` (Höhenlevel) | `ghi_nwp`, `dhi_nwp`, `bhi_nwp` (Strahlungskomponenten) |
| `density_rotor_eq` (abgeleitet, Rotorgeometrie) | `kt_nwp` / `kd_nwp` (Clear-Sky-Index, Diffusanteil — abgeleitet) |
| Power-Curve + `installed_capacity` (Normierung) | Clear-Sky-Modell (`pvlib.clearsky.ineichen`) als Normierung |
| `aggregate_nwp_layers` (Höhenlevel-Aggregation) | entfällt (SL ist einlagig) |
| Statisch: `hub_height`, `rotor_diameter`, `cut_in/out`, `rated_*`, `park_age` | Statisch: `altitude`, `latitude`, `longitude`, Topo-Features (Horizont/Verschattung, Sky-View-Faktor) |
| Statisch: `altitude` + Topo (`/mnt/lambda1/nvme1/topo_features`) | identisch übernehmbar |
| `extrapolate` (Power Law auf Nabenhöhe) | Zenith-/Airmass-Korrektur (`solar_zenith`, `airmass`, `dni_extra`) |
| NWP-Baseline `wind_speed_h10_1` → `Skill_NWP` | NWP-Baseline `ghi_nwp_1` → `Skill_NWP` |
| `next_n_stations` Nachbarstationen | identisch übernehmbar (Nachbar-`ghi` als Prädiktor) |

---

## 3. Implementierungsphasen

### Phase 1 — Kern-Preprocessing (`utils/preprocessing.py`)

**Neu: `preprocess_solar_icond2(path, config, freq, features)`**, strukturell parallel zur
Wind-Funktion, Rückgabe mit identischem Kontrakt: MultiIndex `['starttime','forecasttime','timestamp']`,
gefilterte Feature-Spalten, `df.attrs['nwp_nearest_label']`.

Schritte:

1. **Messdaten laden** — `Station_<id>.parquet`, Zeitfilter aus `train_start`/`test_end`
   (7 Tage Sicherheitsmarge wie bei Wind), Spalte `station_id` verwerfen.
2. **Einheiten-Konversion** — `ghi`, `dhi` von J/cm²/10 min → W/m² (`× 10000/600`).
   Faktor aus dem tatsächlichen Sampling-Intervall abgeleitet, nicht hart kodiert.
3. **Abgeleitete Zielgrößen** — `bhi = clip(ghi − dhi, 0)`, `dni` via pvlib,
   Sonnenstand via `pvlib.solarposition.get_solarposition` (Station-lat/lon/altitude).
4. **Resampling auf `freq`** (Default `1h`, `closed='left'`, `label='left'`) — Mittelwert.
   NaN-Anteil pro Stunde konfigurierbar begrenzen (`max_nan_frac`), sonst Stunde → NaN.
5. **Optional: Clear-Sky-Normierung** — `ghi_cs` per `pvlib.clearsky.ineichen` (Linke-Turbidity
   Klimatologie), `kt = ghi / ghi_cs`. Steuerung per `params.target_transform`.
6. **NWP-SL laden** — neue Hilfsfunktion `_load_icond2_sl_grid()`:
   - einmaliger, gecachter Scan von `SL/{fh}/` → `(lon, lat, pfad)` je Gitterpunkt
     (`@lru_cache`, ~1218 Einträge, korrekte **lon-first**-Parsung)
   - k nächste Gitterpunkte je Station (geodätisch), Spaltensuffixe `_1`, `_2`, …
     bzw. Kompass-Labels bei `get_next_grid_points_method: relative_position`
   - `starttime` → UTC normalisieren
   - 15 min → `freq` aggregieren (Mittel über `floor(forecasttime)`), Lead 0 verwerfen
   - abgeleitete NWP-Features (s. u.)
7. **Merge** auf `timestamp = starttime + forecasttime`, Filter auf vollständige 48-Schritt-Runs
   (identische Logik wie Wind, inkl. Nachfilterung nach `dropna()`).
8. **Nachbarstationen** (`next_n_stations`) + **`neighbor_pool`-Restriktion** — 1:1 aus Wind
   übernehmen (verhindert Leakage von Val/Test-Stationen in Trainings-Inputs).
9. **ECMWF-Merge** — vorbereitet, per Config aus.
10. **Statische Features** — `altitude`, `latitude`, `longitude`, Topo-Features.
11. **Zyklische Zeit-Encodings** — `hour_sin/cos`, `doy_sin/cos`, `solar_zenith_cos`, `azimuth_sin/cos`.

**NWP-Features (abgeleitet, konfigurierbar über Namen):**

| Name | Berechnung |
|---|---|
| `ghi_nwp` | `aswdir_s + aswdifd_s` |
| `dhi_nwp` | `aswdifd_s` |
| `bhi_nwp` | `aswdir_s` |
| `dni_nwp` | `bhi_nwp / cos(θ_z)` |
| `kt_nwp` | `ghi_nwp / ghi_clearsky` (Clear-Sky-Index der Prognose) |
| `kd_nwp` | `dhi_nwp / ghi_nwp` (Diffusanteil) |
| `wind_speed_nwp` | `sqrt(u_10m² + v_10m²)` |
| `solar_zenith`, `solar_azimuth`, `airmass`, `dni_extra` | pvlib, aus `timestamp` + Stationskoordinaten |

Alle übrigen Namen werden als direkte SL-Spalten durchgereicht
(`clct`, `alb_rad`, `t_2m`, `relhum_2m`, `td_2m`, `t_g`, `h_snow`, `prr_gsp`, …).

**Routing:** `_get_data_from_config_files()` unterscheidet aktuell über `'pv' in path or 'solar' in path`
→ `preprocess_synth_pv`. Wird auf ein explizites `data.use_case: solar` umgestellt
(gleicher Schlüssel wie im GNN-Verzeichnis), Pfad-Heuristik bleibt als Fallback.

### Phase 2 — Zielgrößen & Evaluation

- `data.target_col` akzeptiert `ghi` | `dhi` | `bhi` | `dni` | `kt`.
- Multi-Target über `data.target_cols: [ghi, dhi]` (siehe offene Frage 1).
- `utils/eval.py`: NWP-Baseline generalisieren — die Spalte `wind_speed_h10*` ist derzeit
  hart kodiert (`eval.py:276`). Stattdessen `params.nwp_baseline_col` aus der Config
  (Default `ghi_nwp_1` für Solar, `wind_speed_h10_1` für Wind).
- Zusätzliche Solar-Metriken: MAE/RMSE **nur über Tagstunden** (θ_z < 85°), da Nachtwerte
  trivial 0 sind und die Fehlermaße sonst optimistisch verzerrt werden. Beide Varianten werden
  ausgewiesen.
- Persistenz-Baseline: Clear-Sky-Persistenz (`kt` von t−24 h × `ghi_cs(t)`) statt naiver Persistenz.

### Phase 3 — Configs

- `configs/config_solar_50.yaml` — CL-Basiskonfiguration (analog `config_wind_50.yaml`)
- `configs/config_solar_fl.yaml` — FL-Variante
- Stationsliste: die **93 Stationen mit Strahlungsdaten**, per Train/Val/Test-Split
  analog `configs/dcrnn/config_solar_dcrnn.yaml` (dessen Listen enthalten aktuell auch
  Stationen ohne `ghi` → wird korrigiert).
- Neue Config-Schlüssel: `data.use_case`, `data.target_cols`, `params.target_transform`,
  `params.solar_features`, `params.nwp_baseline_col`, `params.daytime_zenith_threshold`.

### Phase 4 — GNN-Pendant (`geostatistics/`)

1. **`solar_preprocessing.py` fixen** — flache SL-Struktur, lon-first-Parsung, `starttime`→UTC.
   Neue Funktion `_parse_lonlat()`; `load_solar_sl_runs()` scannt `SL/{fh}/` einmal statt pro Station.
2. **`load_station_measurements`** (in `train_stgnn2.py`) um Solar-Zweig erweitern:
   Einheiten-Konversion, `bhi`/`dni`-Ableitung, `target_col` aus Config.
3. Configs: `configs/mtgnn/config_solar_mtgnn.yaml`, `configs/wavenet/config_solar_wavenet.yaml`
   (DCRNN existiert bereits, Stationslisten korrigieren).
4. Graph: Delaunay über Stationskoordinaten wie bei Wind; zusätzlich Höhen-/Richtungsfeatures.
   Für Solar ist der Wolkenzug-Advektionsvektor (`u_10m`,`v_10m` auf 700 hPa nicht in SL → 10 m
   als Proxy) ein sinnvolles gerichtetes Kantengewicht — als Ablation vorgesehen.
5. `hpo_dcrnn.py` / `hpo_mtgnn.py` / `hpo_wavenet.py` und die `get_test_results_*.py`
   auf `use_case: solar` prüfen.

### Phase 5 — Doku & Tests

- `docs/preprocess_icond2_solar.md` (analog `preprocess_icond2_wind.md`)
- `docs/predict_solar.md` (Zielgrößen, Transformationen, Baselines, Fallstricke)
- Smoke-Test: eine Station, ein Run-Hour, end-to-end bis Fenster-Erzeugung.

---

## 4. Bewusst zurückgestellt

- **Lückenfüllung** (Kriging/KNN/ERA5-Imputation für Solar) — vorhandene Bausteine
  (`utils/imputation.py`, `utils/era5_imputation.py`, `run_solar_interpolation.py`) sind
  anschlussfähig, aber laut Absprache ein separates Thema.
- **PV-Leistungsprognose** (POA-Transposition, Modultemperatur, Wechselrichter) — die
  Zielgröße ist zunächst Bestrahlungsstärke, nicht PV-Leistung. `utils/meteo.get_total_irradiance()`
  ist dafür bereits vorhanden.
- **ECMWF** — Schnittstelle vorbereitet, Daten fehlen.

---

## 5. Getroffene Entscheidungen (2026-08-11)

| Frage | Entscheidung |
|---|---|
| **Zielgrößen-Handling** | **Multi-Output-Modell.** `data.target_cols: [ghi, dhi]` → Modelle geben `(horizon × n_targets)` aus. `data.target_col` bleibt als Single-Target-Kurzform gültig. |
| **Zielrepräsentation** | **Beides implementiert, roh als Default.** `params.target_transform: none` (Default) \| `clearsky_index`. Baseline zuerst auf W/m². |
| **ICON-D2 SL-Featureset (Basis-Config)** | `ghi_nwp`, `dhi_nwp`, `bhi_nwp`, `clct`, `alb_rad`, `h_snow`, `rho_snow`, `prr_gsp`. Thermodynamik (`t_2m`, `relhum_2m`, `td_2m`, `t_g`) bleibt unterstützt, aber in der Basis-Config auskommentiert. |
| **Nachtstunden** | **Behalten.** Durchgehende 48-h-Sequenzen, keine Loss-Maske. Evaluation weist RMSE/MAE zusätzlich nur über θ_z < 85° aus (`params.daytime_zenith_threshold`). |

---

## 6. Umsetzungsstand (2026-08-11)

### Fertig und verifiziert

| Phase | Ergebnis |
|---|---|
| 1 — Preprocessing | `utils/solar.py` (neu). End-to-End getestet: 3 Stationen → 17 424 Zeilen, 18 Spalten. Gitterpunktwahl 1.08 km, Zeitausrichtung optimal bei Shift 0 (corr 0.960). |
| 2 — Multi-Target | `data.target_cols` durchgängig: `prepare_data`, `prepare_data_for_tft`, `create_tft_sequences`, TFT/TCN-TFT/CNNRNN, `get_y`, `evaluation_pipeline`. Single-Target-Formen unverändert (Regressionstest). |
| 2b — Evaluation | NWP-Baseline nicht mehr auf `wind_speed_h10` verdrahtet; `_NWP_BASELINE_BY_TARGET` + `params.nwp_baseline_col`. Im GNN-Zweig `geostatistics/shared/nwp_baseline.py` statt fünf Kopien. |
| 3 — Configs | `config_solar_93.yaml`, `config_solar_fl.yaml`, `dcrnn/mtgnn/wavenet`-Solar-Configs. Stationsaufteilung aus `data/station_split.csv`, auf die 93 Stationen mit ghi beschränkt. |
| 4 — GNN | `solar_preprocessing.py`: flache SL-Struktur + lon-first-Koordinaten + Lead-Semantik gefixt. `load_station_measurements(use_case='solar')` mit Einheiten und `bhi`/`dni`. |
| 5 — Doku | `docs/preprocess_icond2_solar.md`, `docs/predict_solar.md`, CLAUDE.md. |

### Offen

1. **Lückenfüllung — blockiert den GNN-Pfad.** Über alle 93 Stationen haben 100 % der
   Zeitschritte mindestens eine Lücke irgendwo, also 0 % nutzbare 96-h-Fenster.
   Ursache ist überwiegend **vorzeitiges Reihenende** (7.6 % im Mittel), nicht
   verstreutes Rauschen (2.1 %, Median 0.7 %).
   Zwei Wege: (a) Stationsfilter — eine `< 1 %`-Auswahl gibt 42 Stationen und 37 %
   nutzbare Fenster, sofort verfügbar; (b) Imputation über
   `geostatistics/run_solar_interpolation.py` → `data.interpol_path` /
   `data.knnimputer_path`, erst damit sind alle 93 nutzbar.
   Zahlen in [preprocess_icond2_solar.md](preprocess_icond2_solar.md#51-datenverfügbarkeit).
   (Der CL/FL-Pfad ist davon nicht betroffen und läuft.)
2. **Clear-Sky-Persistenz** als Baseline (`kt` von t−24 h × `ghi_cs(t)`) —
   derzeit läuft nur die naive Persistenz.
3. **Tagstunden-Metriken** sind als Config-Schlüssel vorgesehen
   (`params.daytime_zenith_threshold`), in `eval.get_metrics` aber noch nicht ausgewertet.
4. **ECMWF** beschaffen (`ssrd`, `fdir`, `tcc`, `2t`, `10u`/`10v`).
5. **HPO-Studies** für Solar anlegen (Study-Namen in `hpo.studies_path`).
