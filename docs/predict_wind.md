# Wind Preprocessing — Zielgrößen & Besonderheiten

Dieses Dokument beschreibt die beiden möglichen Zielgrößen (`target_col: wind_speed` und
`target_col: power`) sowie alle zusätzlichen Preprocessing-Optionen, die wind-spezifisch sind.

---

## 1. Zielgröße: `target_col: wind_speed`

### Überblick

Neben der Standard-Aufgabe (Leistungsprognose, `target_col: power`) kann das Framework auch
**Windgeschwindigkeit direkt als Zielgröße** vorhersagen. Das ist sinnvoll um die Qualität der
NWP-Eingangsdaten zu beurteilen und den Mehrwert des ML-Modells gegenüber dem rohen NWP-Forecast
zu quantifizieren.

### Config

```yaml
data:
  target_col: wind_speed
```

Alle anderen Config-Parameter bleiben identisch zu einer Power-Prognose-Config.

### Preprocessing (`utils/preprocessing.py`)

#### Feature-Filterung — `preprocess_synth_wind_icond2()`

Am Ende der Funktion werden nur Spalten aus `features['known'] + features['observed']` behalten.
Für den `wind_speed`-Fall wird **zusätzlich** jede Spalte mit Präfix `wind_speed_h10*` im DataFrame
mitgeführt, auch wenn sie nicht in `known`/`observed` steht:

```python
# preprocessing.py ~L1352
if config['data'].get('target_col') == 'wind_speed':
    for col in df_merged.columns:
        if col.startswith('wind_speed_h10') and col not in available_cols:
            available_cols.append(col)
```

**Warum:** Die Spalte `wind_speed_h10_1` (nächster ICON-D2-Gitterpunkt) wird in `eval.py` als
NWP-Baseline benötigt. Ohne diesen Block würde sie beim Feature-Filtering entfernt und die
Baseline wäre nicht berechenbar.

#### Spaltennamen

ICON-D2 liefert `wind_speed_h10` mit Gitterpunkt-Suffix. Der nächste Gitterpunkt hat Suffix `_1`:

```
wind_speed_h10_1   ← nächster Gitterpunkt (wird für Baseline genutzt)
wind_speed_h10_2   ← 2. Gitterpunkt
...
```

### Evaluation (`utils/eval.py`)

#### NWP-Baseline: `NWP (wind_speed_h10)`

In `evaluation_pipeline()` wird — wenn `target_col == 'wind_speed'` — automatisch eine
**NWP-Baseline** berechnet, die `wind_speed_h10_1` direkt als Forecast verwendet (kein ML,
volle Abhängigkeit vom NWP-Modell):

```python
# eval.py ~L275
nwp_cols = [c for c in data.columns if c.startswith('wind_speed_h10')]
nwp_col = next((c for c in nwp_cols if c.endswith('_1')), nwp_cols[0] if nwp_cols else None)
if target_col == 'wind_speed' and nwp_col is not None:
    y_nwp = data[nwp_col]
    if isinstance(y_nwp.index, pd.MultiIndex):
        y_nwp = y_nwp.reset_index().groupby('timestamp').mean().iloc[:, -1]
    ...
    pers['NWP (wind_speed_h10)'] = df_nwp
```

**MultiIndex-Handling:** ICON-D2-Daten haben einen MultiIndex `(timestamp, step)`. Analog zur
`persistence()`-Funktion wird per `groupby('timestamp').mean()` auf einen einfachen DatetimeIndex
reduziert, bevor `make_windows()` aufgerufen wird.

#### Skill-Metriken

`evaluate_models()` berechnet zwei Skill-Scores:

| Spalte      | Bedeutung                                            |
|-------------|------------------------------------------------------|
| `Skill`     | `1 - RMSE_model / RMSE_Persistence` (zeitl. Persistence) |
| `Skill_NWP` | `1 - RMSE_model / RMSE_NWP` (roher ICON-D2-Forecast) |

`Skill_NWP` ist nur befüllt wenn `target_col == 'wind_speed'` und `wind_speed_h10_1` im DataFrame
vorhanden ist. Sonst bleibt es `NaN`.

---

## 2. Zielgröße: `target_col: power`

Standardfall. Das Modell prognostiziert die Einspeiseleistung direkt. Keine besonderen
Preprocessing-Abweichungen außer:

- `wind_speed_t1`–`wind_speed_t6` (Turbinen-Windgeschwindigkeiten auf Nabenhöhe) werden bei Bedarf
  zu `wind_speed_hub` zusammengefasst, wenn `wind_speed_hub` oder `wind_speed_t` in
  `known`/`observed` steht.
- `wind_speed` (generische Messreihe, 10 m Höhe) wird gedroppt, es sei denn es ist Target oder
  explizit als Feature gelistet — **oder `extrapolate: 'wind_speed'` ist gesetzt** (s. Abschnitt 3).

---

## 3. Windgeschwindigkeit auf Nabenhöhe extrapolieren (`extrapolate`)

### Motivation

NWP liefert Windgeschwindigkeiten auf fixen Höhenniveaus (10 m, 38 m, 78 m, 127 m, ...).
Windkraftanlagen arbeiten aber auf **Nabenhöhe** (typisch 80–150 m). Der Powerertrag hängt von der
Windgeschwindigkeit auf Nabenhöhe ab. Das Power Law extrapoliert von einer Referenzhöhe auf die
Nabenhöhe.

### Config-Parameter

```yaml
params:
  extrapolate: 'wind_speed_h10'   # oder 'wind_speed', 'wind_speed_h38', etc.
```

Wenn `null` oder nicht gesetzt → keine Extrapolation.

### Formel: Power Law

```
v2 = v1 * (h2 / h1)^alpha
```

- `v1`: Windgeschwindigkeit an Quellhöhe `h1`
- `v2`: Extrapolierte Windgeschwindigkeit an Nabenhöhe `h2`
- `alpha`: Hellmann-Exponent (geschätzt aus zwei NWP-Höhen)

### Alpha-Schätzung (per Zeitschritt)

```
alpha = ln(v_high / v_low) / ln(h_high / h_low)
```

- `v_low` = Werte der **Quellspalte** (z. B. `wind_speed_h10_1` für `extrapolate: 'wind_speed_h10'`)
- `v_high` = `wind_speed_h127_1` (obere Referenz, immer 127 m)
- Alpha wird **per Zeitschritt** berechnet → adaptiver Hellmann-Exponent

### Quellhöhe `h1`

Die Quellhöhe wird aus dem Spaltennamen extrahiert (Regex `_h(\d+)`):

| `extrapolate`-Wert | `h1` | Quelle |
|---|---|---|
| `wind_speed_h10`   | 10 m  | aus Spaltenname |
| `wind_speed_h38`   | 38 m  | aus Spaltenname |
| `wind_speed`       | 10 m  | **Fallback** (kein `_h{N}` im Namen, Annahme 10 m) |

> `wind_speed` ist die gemessene/synthetische Windgeschwindigkeit auf 10 m — sie hat keinen
> Höhensuffix im Namen. Die 10-m-Annahme ist explizit so kodiert und wird im Log angezeigt.

### Nabenhöhe `h2`

Wird aus `wind_parameter.csv` (Spalte `hub_height`) für die jeweilige Station gelesen.

### Output: Spalte `wind_speed_hub_extrap`

Das Ergebnis der Extrapolation wird als neue Spalte `wind_speed_hub_extrap` im DataFrame abgelegt.
Sie wird automatisch in `available_cols` aufgenommen und überlebt den Feature-Filter.

**Wichtig:** Um die Spalte als Modell-Input zu nutzen, muss sie explizit in der Config stehen:

```yaml
known_features:
  - 'wind_speed_hub_extrap'   # NWP-basiert → future known
```

`known_features` ist korrekt, weil NWP-Werte für den gesamten Forecast-Horizont im Voraus bekannt
sind.

Bei `extrapolate: 'wind_speed'` (Messung) ist `observed_features` semantisch korrekter, da der
zukünftige Messwert nicht bekannt ist:

```yaml
observed_features:
  - 'wind_speed_hub_extrap'
```

### Namenskonflikt mit `wind_speed_hub`

`wind_speed_hub` ist **bereits belegt** für die per-Turbine gemessenen Nabenhöhen-Windgeschwindigkeiten
aus den Synth-Daten (`wind_speed_t1`–`wind_speed_t6`). Die extrapolierte Variante heißt daher
bewusst `wind_speed_hub_extrap`.

### Implementierung

Separate Funktion `extrapolate_to_hub_height()` in `preprocessing.py`, die vom Haupt-Preprocessing
aufgerufen wird. Sie ist unabhängig von `target_col` und funktioniert für beide Zielgrößen.

---

## 4. Static Features: `latitude` / `longitude`

Wenn `latitude` oder `longitude` als `static_features` in der Config angegeben werden, werden sie
aus `stations_master.csv` gelesen und in `static_data` eingetragen:

```python
# preprocessing.py — nach dem Laden von station_lat / station_lon
if static_features:
    static_data['latitude'] = station_lat
    static_data['longitude'] = station_lon
```

---

## 5. Bekannte Fallstricke

| Problem | Ursache | Lösung |
|---|---|---|
| `Skill_NWP` ist NaN | `wind_speed_h10_1` fehlt im park_df | Block in `preprocess_synth_wind_icond2()` stellt sicher, dass die Spalte erhalten bleibt (s. Abschnitt 1) |
| `TypeError: 'tuple' - 'tuple'` in `make_windows` | MultiIndex nicht aufgelöst | `groupby('timestamp').mean()` vor `make_windows()` |
| `ValueError: X has N features, but StandardScaler expects M` | Scaler aus Cache mit anderer Feature-Anzahl | Data-Cache leeren (`data_cache/`) |
| `wind_speed` nicht in DataFrame bei `extrapolate: 'wind_speed'` + `target_col: power` | `wind_speed` wird gedroppt, da nicht Target und nicht in `rel_features` | Behoben: Drop-Logik prüft ob `extrapolate`-Quelle die Spalte benötigt |
| `wind_speed_hub_extrap` landet in falscher Feature-Gruppe | `wind_speed` ist Messung, `wind_speed_h10` ist NWP | Messung → `observed`, NWP-Höhe → `known` (s. Abschnitt 3) |

---

## 6. Empfehlungen

### Wann `target_col: wind_speed`?

- Um zu messen, wie viel das ML-Modell gegenüber dem rohen ICON-D2-Forecast gewinnt (`Skill_NWP`)
- Gibt Aufschluss über die Informationsausschöpfung des Modells
- **Nicht als End-Produkt:** Leistung (`power`) ist die betriebsrelevante Größe

### Wann `extrapolate`?

- Immer wenn NWP-Windgeschwindigkeit als Feature genutzt wird und die Turbinen-Nabenhöhe
  signifikant von den verfügbaren NWP-Höhen abweicht
- Physikalisch sinnvoller als rohe 10-m-Werte, da Leistung `P ∝ v³` gilt und der Höhenunterschied
  großen Einfluss hat
- Bei `target_col: power` typischerweise `extrapolate: 'wind_speed_h10'` (NWP → Nabenhöhe)
