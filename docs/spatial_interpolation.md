# Spatial Wind-Speed Interpolation (IDW / OK / RK)

Vergleich drei räumlicher Interpolationsverfahren für Windgeschwindigkeit an 160 DWD-Stationen via Leave-One-Out Cross-Validation (LOO-CV).

---

## Überblick

| Methode | Beschreibung |
|---|---|
| **IDW** | Inverse Distance Weighting — gewichteter Mittelwert der k nächsten Nachbarn (Gewicht ∝ 1/d²) |
| **OK** | Ordinary Kriging — stochastische Interpolation mit globaler, vorher gefitteter Variogramm-Funktion |
| **RK** | Regression Kriging — Trend (wind\_speed ~ altitude via OLS) plus gekrigter Residualterm |

---

## Datei-Übersicht

```
forecasting_framework/
├── configs/config_spatial_interpolation.yaml   # Experiment-Konfiguration
├── run_spatial_interpolation.py                 # Einstiegspunkt
└── utils/interpolation.py                       # Alle Hilfsfunktionen
```

**Outputs** werden in `data/test_results/` geschrieben (konfigurierbar via `output.path`).

---

## Konfiguration

`configs/config_spatial_interpolation.yaml` — relevante Sektionen:

```yaml
data:
  path: '/mnt/nvme1/synthetic/wind/...'   # Verzeichnis mit Station-CSVs + wind_parameter.csv
  files: [...]                             # Liste der 160 Stations-IDs
  test_start: '2025-08-01'                 # optional: Zeitfenster einschränken
  test_end:   '2025-10-31'

interpolation:
  k_neighbors: 8             # Anzahl nächster Nachbarn
  idw_power: 2.0             # IDW-Exponent
  n_variogram_lags: 20       # Anzahl Lag-Bins für empirisches Variogramm
  max_variogram_dist: null   # km; null = automatisch (halbe max. Paarweisedistanz)

output:
  path: 'data/test_results'
  prefix: 'spatial_interp'   # Präfix aller Output-Dateien
```

---

## Ausführung

```bash
cd forecasting_framework
source frcst/bin/activate
python run_spatial_interpolation.py --config configs/config_spatial_interpolation.yaml
```

Optional: `--log-level DEBUG` für detailliertere Ausgabe.

---

## Eingabedaten

### Stationsmetadaten
`{data.path}/wind_parameter.csv` — Spalten: `park_id`, `latitude`, `longitude`, `altitude`

### Zeitreihen
`{data.path}/{station_id}.csv` — enthält mindestens `timestamp` und `wind_speed`

---

## Pipeline-Schritte

### 1. Distanzmatrix (einmalig)

Alle paarweisen geodätischen Distanzen werden mit `geopy.distance.geodesic` als vollständige 160×160-Matrix berechnet und für die gesamte Laufzeit gecacht.

Die k=8 nächsten Nachbarn jeder Station werden einmalig aus dieser Matrix bestimmt.

### 2. Globales Variogramm (einmalig)

**Empirische Semivarianz:**
- Alle N(N-1)/2 Stationspaare werden ausgewertet
- Semivarianz pro Paar: γ(i,j) = 0.5 · E[(z_i − z_j)²], gemittelt über alle Timestamps
- Ergebnis wird in `n_variogram_lags` Distanz-Bins aggregiert

**Modell-Fit:**
- Sphärisches Variogramm-Modell via `scipy.optimize.curve_fit`
- Parameter: **nugget** (Messrauschen), **psill** (partielle Sill), **range** (Korrelationsreichweite in km)

Das gefittete Variogramm wird einmal gespeichert und für alle Kriging-Vorhersagen wiederverwendet.

### 3. Kriging-Gewichte (einmalig pro Station)

Da die Kriging-Gewichte nur von den Standorten der Nachbarn und dem Variogramm abhängen (nicht von den beobachteten Werten), werden sie **einmal pro Station** vorberechnet.

Das löst das (k+1)×(k+1) Ordinary-Kriging-System:

```
[Γ_nn  1] [λ]   [γ_t]
[1ᵀ   0] [μ] = [1  ]
```

- `Γ_nn`: Semivarianz-Matrix zwischen den k Nachbarn
- `γ_t`:  Semivarianz-Vektor von Nachbarn zum Zielpunkt
- `λ`:    Kriging-Gewichte (wiederverwendet für OK und RK)

**Performance-Vorteil:** O(k³) einmalig pro Station statt O(k³) pro (Station × Timestamp).

### 4. LOO-CV-Schleife

Für jeden Timestamp t und jede Station s:

| Schritt | IDW | OK | RK |
|---|---|---|---|
| Nachbarn | k=8 nächste (fest) | k=8 nächste (fest) | k=8 nächste (fest) |
| Vorhersage | ẑ = Σ(wᵢ·zᵢ) / Σwᵢ, w=1/d² | ẑ = Σλᵢ·zᵢ | Trend(alt) + Σλᵢ·eᵢ |
| Variogramm | — | global, vorher gefit | global, vorher gefit |

**Regression Kriging im Detail:**
1. OLS: `wind_speed ~ altitude` auf den k Nachbarn
2. Residuen: eᵢ = zᵢ − OLS(altᵢ)
3. OK auf Residuen: ê₀ = Σλᵢ·eᵢ (selbe Kriging-Gewichte wie OK)
4. Vorhersage: ẑ₀ = OLS(alt₀) + ê₀

---

## Outputs

| Datei | Inhalt |
|---|---|
| `{prefix}_loo_predictions.csv` | [station\_id, timestamp, wind\_speed\_observed, idw\_pred, ok\_pred, rk\_pred] |
| `{prefix}_results_per_station.csv` | [station\_id, method, rmse, mae, r2] |
| `{prefix}_results_summary.csv` | [method, rmse, mae, r2] — Mittelwert über alle Stationen |
| `{prefix}_scatter_idw.png` | Scatter-Plot Beobachtet vs. IDW-Vorhersage |
| `{prefix}_scatter_ok.png` | Scatter-Plot Beobachtet vs. OK-Vorhersage |
| `{prefix}_scatter_rk.png` | Scatter-Plot Beobachtet vs. RK-Vorhersage |

---

## Abhängigkeiten

```
geopy      # geodätische Distanzberechnung
scipy      # curve_fit für Variogramm-Fitting
scikit-learn  # LinearRegression (RK), Metriken
matplotlib # Scatter-Plots
pyyaml     # Config-Loading
```

Falls noch nicht installiert:
```bash
pip install geopy scipy scikit-learn matplotlib pyyaml
```

---

## Bekannte Limitierungen

- **Altitude-only Trend (RK):** Mit k=8 Nachbarn und nur einem Kovariate ist der OLS-Trend potenziell instabil. Bei schwachem Höhengradienten kann RK schlechter als OK abschneiden.
- **Globales Variogramm:** Ein einziges stationäres Variogramm über alle Zeiten ignoriert zeitliche Variabilität der räumlichen Korrelationsstruktur. Alternativ könnte man pro Windrichtungssektor oder Jahreszeit ein Variogramm fitten.
- **Gleichgewichtete Timestamps:** Timestamps mit hoher/niedriger Windvarianz tragen gleichgewichtet zur empirischen Semivarianz bei.
