# ICON-D2 Database Integration

## Übersicht

Seit April 2026 können ICON-D2 NWP-Daten wahlweise aus einer PostgreSQL/PostGIS Datenbank geladen werden statt aus CSV-Dateien. Die CSV-Schnittstelle bleibt vollständig funktionsfähig als Fallback.

## Konfiguration

### Umgebungsvariable

Die Datenbankverbindung wird über die Umgebungsvariable `WEATHER_DB_URL` gesteuert:

```bash
export WEATHER_DB_URL="postgresql://user:password@host:port/WeatherDB"
```

Diese Variable sollte in der `.bashrc` oder `.bash_profile` gesetzt werden.

### Config-Parameter

In der YAML-Konfiguration wird die Datenquelle über `data.icond2_source` gesteuert:

```yaml
data:
  icond2_source: 'database'  # 'csv' oder 'database'
  nwp_path: '/mnt/nvme1/icon-d2/csv'  # Wird nur bei 'csv' verwendet
  stations_master: 'data/stations_master.csv'
  # ... weitere Parameter
```

**Werte:**
- `'database'`: Lädt ICON-D2 Daten aus PostgreSQL Datenbank (Standard für neue Experimente)
- `'csv'`: Lädt ICON-D2 Daten aus CSV-Dateien (Fallback, Legacy)

## Datenbankstruktur

### Tabellen

**`icon_d2_grid_points`** (ca. 1.200 Punkte)
- Lookup-Tabelle für alle ICON-D2 Gitterpunkte über Deutschland
- Spalten: `geom` (PostGIS POINT), indexiert für schnelle KNN-Suche
- Koordinaten als **nicht-standard** `POINT(lat, lon)` gespeichert (ST_X = lat, ST_Y = lon)

**`multilevelfields`** (ca. 1,27 Mrd. Zeilen)
- Zeitreihen-Daten für atmosphärische Schichten
- Primärschlüssel: `(starttime, forecasttime, geom, toplevel)`
- Index: `ml_geom_time_idx` auf `(geom, starttime, forecasttime)` — **Performance-kritisch!**

**Verfügbare Höhenstufen:**
- 20m, 55m, 100m, 153m, 214m, 281m (Oberkante der Schicht)

**Verfügbare Features:**
- `u_wind`, `v_wind`: Windkomponenten [m/s]
- `temperature`: Temperatur [K]
- `pressure`: Luftdruck [Pa]
- `qs`: Spezifische Feuchte [kg/kg]
- `relhum`: Relative Feuchte [%]

### Zeitliche Struktur

- **Runs**: 06, 09, 12, 15 UTC (in DB als `starttime`)
- **Forecast Horizon**: 0-48 Stunden (`forecasttime`)
- **Zeitliche Auflösung**: 1 Stunde (Multilevel)

## Implementation

### Module

**`utils/db_connector.py`**
- `WeatherDBConnector`: Singleton Connection Pool Manager
- `find_nearest_grid_points()`: KNN-Suche für Gitterpunkte
- `load_multilevel_data()`: Lädt Zeitreihen für gegebene Grid Points
- `test_connection()`: Testet Datenbankverbindung

**`utils/preprocessing.py`**
- `_load_icon_d2_from_database()`: Hauptfunktion zum Laden der DB-Daten
- Integration in `preprocess_synth_wind_icond2()` mit automatischem CSV-Fallback

### Workflow

1. **Grid Point Selection**
   - Nutzt `db_connector.find_nearest_grid_points()`
   - Unterstützt beide Methoden: `geodesic_next` und `relative_position`
   - KNN-Suche auf `icon_d2_grid_points` Tabelle (< 1ms)

2. **Feature Extraction**
   - Parst angeforderte Höhenstufen aus Config (`known_features`)
   - Beispiel: `'wind_speed_h78'` → Höhe 78m wird aus DB abgefragt

3. **Data Loading**
   - Query gegen `multilevelfields` für jeden Grid Point
   - Zeitbereich: `df_synth.index.min()` bis `.max()`
   - Parallele Abfragen für mehrere Grid Points (schnell dank Index)

4. **Post-Processing**
   - Berechnung von `wind_speed` aus `u_wind`, `v_wind`
   - Berechnung von `density` (falls `get_density: True`)
   - Rotor-äquivalente Windgeschwindigkeit (falls `aggregate_nwp_layers: 'weighted_mean'`)

5. **Merge mit Synthetic Data**
   - Identisches Format wie CSV-Loader
   - Merge auf `timestamp` Index

## Performance

### Vorteile Database

✅ **Schneller**: Index-basierte Abfragen (< 1s für 1 Jahr Daten, 1 Grid Point)  
✅ **Skalierbar**: In-Memory-Datenbank (~500 GB RAM), parallele Verbindungen  
✅ **Konsistent**: Zentrale Datenquelle, keine Duplikate, automatische Updates  
✅ **Flexibel**: Ad-hoc Queries, beliebige Zeitbereiche, Höhenstufen on-demand  
✅ **Fail-Fast**: Bei Fehlern crashed das Programm sofort (kein stiller Fallback)

### Nachteile

❌ **Netzwerkabhängig**: Erfordert DB-Server Erreichbarkeit (crashed bei Ausfall)  
❌ **Setup**: Umgebungsvariable `WEATHER_DB_URL` muss gesetzt sein  
❌ **Schema**: Nur `qs` verfügbar (kein `relhum`) — `relative_humidity` wird via `mpcalc` aus `qs` berechnet

### Performance-Regeln

1. **IMMER `geom` in WHERE-Klausel** — Index wird nur genutzt, wenn `geom` gefiltert wird
2. **Zeitbereich einschränken** — `starttime >= X AND starttime < Y`
3. **Nur benötigte Höhen** — `toplevel = ANY(ARRAY[10, 38, 78, ...])`
4. **Parallele Abfragen OK** — PostgreSQL ist Thread-safe

## Fehlerbehandlung

### Kein automatischer Fallback

**Wichtig:** Wenn `data.icond2_source: 'database'` gesetzt ist und die Datenbankverbindung fehlschlägt, **crashed das Programm**. Es gibt **keinen automatischen Fallback auf CSV**.

Dies ist gewolltes Verhalten, um sicherzustellen, dass:
- Fehler sofort sichtbar werden
- Keine unbeabsichtigte Vermischung von Datenquellen erfolgt
- Debugging einfacher ist (klare Fehlerursache)

### Häufige Fehler

**`WEATHER_DB_URL not set`**
→ Export WEATHER_DB_URL in `.bashrc` oder vor dem Skript-Aufruf

**`No grid points found near (lat, lon)`**
→ Station liegt außerhalb des ICON-D2 Gitter-Bereichs (Deutschland)

**`Database query returned no data`**
→ Zeitbereich außerhalb verfügbarer Daten, oder Grid Point hat Lücken

**Connection refused / Timeout**
→ Datenbankserver nicht erreichbar - prüfe Netzwerk, DB-Status

### Manueller Fallback auf CSV

Wenn die Datenbank nicht verfügbar ist, muss die Config manuell angepasst werden:

```yaml
data:
  icond2_source: 'csv'  # Manuell auf CSV umstellen
  nwp_path: '/mnt/nvme1/icon-d2/csv'
```

## Beispiel

### Config

```yaml
data:
  icond2_source: 'database'
  stations_master: 'data/stations_master.csv'

params:
  next_n_grid_points: 4
  get_next_grid_points_method: 'geodesic_next'
  known_features:
    - 'wind_speed_h10'
    - 'wind_speed_h38'
    - 'wind_speed_h78'
    - 'density_h78'
  get_density: True
  aggregate_nwp_layers: 'weighted_mean'
```

### Python

```python
import yaml
from utils.preprocessing import preprocess_synth_wind_icond2

# Load config
with open('configs/exps/my_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Preprocessing wird automatisch DB nutzen
df = preprocess_synth_wind_icond2(
    path='/path/to/synth_00298.csv',
    config=config,
    freq='1H',
    features={
        'known': config['params']['known_features'],
        'observed': [],
        'static': config['params']['static_features']
    }
)

# df enthält jetzt ICON-D2 Daten aus der Datenbank
```

## Migration CSV → Database

1. **Config anpassen**: `data.icond2_source: 'database'` setzen
2. **Umgebungsvariable**: `export WEATHER_DB_URL=...` in `.bashrc`
3. **Testen**: Einzelne Station mit kleinem Zeitbereich
4. **Batch**: Bestehende Skripte (z.B. `run_train_all_stations.sh`) funktionieren unverändert

## Siehe auch

- `~/Work/DWD/doc/DB_ACCESS_GUIDE.md`: Umfassende Dokumentation der WeatherDB
- `docs/preprocess_icond2_wind.md`: ICON-D2 Wind Preprocessing (allgemein)
- `utils/db_connector.py`: Source Code der DB-Integration
