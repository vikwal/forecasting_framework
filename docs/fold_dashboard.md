# Fold Dashboard (`geostatistics/fold_dashboard.py`)

Streamlit-Dashboard für die räumlichen Stationsaufteilungen — Wind und Solar.
Zeigt je Fold die Train-/Val-/Test-Stationen auf der Karte, die Distanz jeder
Val-Station zu ihrem nächsten Trainingsnachbarn und die Balance zwischen den
Folds.

Der Entwurf der Solar-Aufteilung steht in
[station_splits_solar.md](station_splits_solar.md).

## Deployment

Läuft als systemd-Service auf Port `8511` (8504 und 8510 sind von den
Optuna-Dashboards belegt, siehe `~/docs/services.md`).

Unit-Vorlage im Repo: `deploy/fold-dashboard.service`

```bash
sudo cp /home/viktor/Work/forecasting_framework/deploy/fold-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fold-dashboard
```

Der `cp` braucht den **absoluten Pfad** — aus `~` ausgeführt scheitert er sonst
mit `cannot stat 'deploy/fold-dashboard.service'`.

Service-Befehle:
```bash
sudo systemctl status fold-dashboard
sudo systemctl restart fold-dashboard
journalctl -u fold-dashboard -f
```

Läuft bereits eine Instanz von Hand auf 8511, belegt sie den Port und der
Dienst startet nicht: `pkill -f fold_dashboard` vorher.

Ohne Dienst, für einen einmaligen Blick:
```bash
frcst/bin/streamlit run geostatistics/fold_dashboard.py --server.port 8511
```

## Bedienung

Die Sidebar hat eine **Use-Case-Voreinstellung**; beide Pfade bleiben danach
frei editierbar.

| Voreinstellung | Fold-Definition | Basis-Config |
|---|---|---|
| Wind (153 Stationen, 3 Folds) | `configs/spatial_folds.yaml` | `configs/mtgnn/config_wind_mtgnn_nwp_fold1.yaml` |
| Solar (83 Stationen, 3 Folds) | `configs/solar_folds.yaml` | `configs/solar_baseline/config_solar_base_lag.yaml` |

Weitere Regler:

- **Fold** — welcher Fold gezeigt wird
- **Test-Stationen einblenden** — der zurückgehaltene Testsatz
- **Val → nächste Train-Station verbinden** — zeichnet die Linie zu jedem
  nächsten Trainingsnachbarn, macht Lücken sofort sichtbar
- **Hervorhebungsschwelle (km)** — Val-Stationen ab dieser Nachbardistanz
  werden markiert

## Woher der Testsatz kommt

Bevorzugt aus der Fold-Datei selbst (Top-Level-Schlüssel `test_files`, so legt
`scripts/make_solar_folds.py` ihn ab), sonst aus `data.test_files` der
Basis-Config. Damit bleibt die Wind-Ansicht unverändert, die ihn weiterhin aus
der Config bezieht.

`geostatistics/spatial_cv.py::load_spatial_folds` liest nur `spatial_fold*` und
ignoriert `test_files` — der zusätzliche Schlüssel bricht nichts.

## Koordinaten und Merkmale

Stationskoordinaten kommen aus `data/stations_master.csv`. Terrain-Features
werden über `geostatistics/stgnn/utils/topo_features.py` vom NAS geladen; fehlt
das Verzeichnis, blendet das Dashboard den Teil aus, statt abzubrechen — für die
Solar-Stationen existieren sie nicht.
