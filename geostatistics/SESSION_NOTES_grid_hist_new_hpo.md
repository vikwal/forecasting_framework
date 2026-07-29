# GRID+HIST — neue HPO: Val/Test-Läufe (Session-Notizen, Stand 2026-07-22)

Kontext & Hintergrund (Architektur, Feature-Namen, Sampling-Logik) steht in
`docs/data.md` und `docs/train_dcrnn.md` — hier **nicht** wiederholt, nur was
in dieser Session neu dazukam: zwei Daten-Bugs, zwei Regenerierungs-Scripts,
ein Code-Fix in beiden Trainingsskripten, und der aktuelle Lauf-Stand.

## Auslöser

Für die GRID+HIST-Variante (DCRNN + MTGNN, `nwp_hist`) wurde eine neue HPO
nachgezogen (Grund: die alte HPO validierte versehentlich auf demselben
Zeitfenster wie das spätere Test-Hold-out von Fold 1 → Leakage-Risiko).
Neue Optuna-Studies (Postgres `optuna_db`, `OPTUNA_STORAGE`):

- `cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_nwp_hist_new` (111/131 Trials, best 0.9592, Trial #117)
- `cl_m-mtgnn_out-48_freq-1h_wind_mtgnn_nwp_hist_new` (129/170 Trials, best 0.9244, Trial #93)

Fertig seit 2026-07-07. Fehlend: Retraining auf den 3 Val-Folds + 3
Test-Zeiträumen mit diesen neuen Best-Params.

## Konventions-Hinweis: `_new`-Stem für Study-Routing

`train_dcrnn.py`/`train_mtgnn.py` leiten den Optuna-Study-Namen **rein aus
dem Config-Dateinamen** ab (Stem minus `_fold\d+`), **nicht** aus `--hpo-study`
(das steuert bei Postgres-Storage nur ob überhaupt geladen wird). Deshalb
mussten neue Config-Dateien mit `_new` im Stem angelegt werden, sonst hätte
`--hpo-study auto` weiter die alte Study getroffen:

- `configs/{dcrnn,mtgnn}/config_wind_{dcrnn,mtgnn}_nwp_hist_new_fold{1,2,3}.yaml` — Val-Folds (inhaltlich Kopie der alten Fold-Configs, nur Stem geändert)
- `configs/{dcrnn,mtgnn}/test/config_wind_{dcrnn,mtgnn}_nwp_hist_new_fold{1,2,3}.yaml` — 3 Test-Zeiträume

## Test-Zeiträume (Stationen-Holdout, gleiche files/val_files/test_files wie Fold1)

| Zeitraum | Fenster |
|---|---|
| Val-Fold1 | 2024-07-31 → 2024-11-30 |
| Val-Fold2 | 2024-11-30 → 2025-04-01 |
| Val-Fold3 | 2025-04-01 → 2025-08-01 |
| Test-Zeitraum 1 | 2025-08-01 → 2025-11-30 |
| Test-Zeitraum 2 | 2025-12-01 → 2026-01-31 |
| Test-Zeitraum 3 | 2026-02-01 → 2026-04-30 |

## Bug 1: ECMWF-Pfad-Shadowing (nur Test-Zeitraum 3 betroffen)

`ecmwf_path` in den Configs zeigt auf `/mnt/lambda1/nvme1/ecmwf/parquet/`.
Der Loader (`train_stgnn2.py::load_ecmwf_parquet_at_stations_and_grid`)
bevorzugt automatisch einen `SL/`-Unterordner, falls vorhanden. Dieser
Unterordner existiert dort — ist aber **veraltet** (`starttime` max
2026-04-07), während die Dateien im Hauptordner selbst frischer sind
(`starttime` max 2026-04-30, geprüft per Stichprobe an
`55_0100_9_1100_wind_sl.parquet`, beide Kopien direkt verglichen).

**Workaround** (nicht die gemeinsamen L1-Daten angefasst): Symlink-Verzeichnis
`data/ecmwf_fresh_link/` zeigt auf alle `*_sl.parquet` im Hauptordner, ohne
`SL/`-Unterordner. Nur `configs/{dcrnn,mtgnn}/test/config_wind_*_nwp_hist_new_fold3.yaml`
wurden auf `ecmwf_path: '/home/viktor/Work/forecasting_framework/data/ecmwf_fresh_link'`
umgestellt. Andere Configs unverändert (ihr Zeitfenster liegt ohnehin
innerhalb der alten `SL/`-Abdeckung).

## Bug 2: ICON-D2-Grid-NaN wurde nie tatsächlich ausgeschlossen (Fix committed)

`load_icond2_ml_runs` (`train_stgnn2.py`) loggt bei NaN im Grid
("Affected runs will be excluded from training"), **filtert sie aber nie
tatsächlich** — das Masking existierte nur als Log-Text, nicht im Code. Die
Run-Pair-Schleife in `train_dcrnn.py`/`train_mtgnn.py` prüfte bislang nur
`_meas_nan_any` (Stations-Messwerte), nie das Grid selbst. Symptom: `val loss
= nan` sobald ein Val-Run auf eine Grid-Lücke fiel (reproduzierbar bei
MTGNN Fold3, zwei identische Läufe).

**Fix (bereits im Code, beide Dateien)**: neue Prüfung
`grid_nan_by_run = np.isnan(grid_icond2_runs).any(axis=(1,2,3))` in der
Run-Pair-Schleife, Runs mit NaN in `r_curr` oder `r_hist` werden jetzt
übersprungen (Log zeigt `skipped: N (grid-NaN: M)`). Betrifft potenziell
**jeden** Lauf, nicht nur Fold3 — einfach unauffällig, wenn keine Grid-Lücke
im jeweiligen Zeitfenster liegt.

### Warum gibt es Grid-Lücken — zwei getrennte Ursachen, nicht verwechseln

1. **DB-Lücken** (`WeatherDB.singlelevelfields`, `localhost:5432`): 97 Runs
   über die gesamte Historie mit nur 25.4 % Zeilen (59682 statt 235074) —
   nie 0 %, nie ein anderer Bruchteil → sieht nach systematischem
   Ingest-Fehler aus (z. B. 1 von 4 Domain-Kacheln). Volle Liste:
   `data/icond2_incomplete_runs.csv` (97 Zeilen, Spalten `starttime_utc`,
   `rows_present`, `pct_present`, `rows_expected`). Davon 8 im
   Test-Zeitraum-3-Fenster (2026-03-19, 2026-03-24, 2026-04-17, je
   06/09/12/15 UTC-Runs, s. CSV).
2. **Datei-Quelle "ML"** (`/mnt/lambda1/nvme1/icon-d2/parquet/ML/{run_hour}/{station_id}/*_ML.parquet`,
   von den Trainingsskripten tatsächlich gelesen, **nicht** die DB!) —
   deutlich patchiger: Stichprobe über mehrere Run-Hours/Stationen zeigt
   `starttime` max **2026-03-06**, manche Grid-Punkte weiter (Log zeigte in
   Aggregation macht bis 2026-03-31). **Das ist bereits in
   `docs/data.md` dokumentiert** ("2023-07-24 – 2026-03-06", Abschnitt 2) —
   war mir zu Beginn der Session nicht bewusst, weil ich nur die DB-Coverage
   geprüft hatte, nicht die tatsächlich genutzte Datei-Quelle. **Für
   zukünftige Sessions: bei ICON-D2-Verfügbarkeitsfragen zuerst
   `docs/data.md` lesen, nicht die DB pauschal für "die" Quelle halten.**

Die 8 DB-Lücken sind nur ein Teilbild; der Großteil der in Fold3
ausgeschlossenen ~47 Runs kommt vermutlich aus der ML-Datei-Quelle. Der
Grid-NaN-Fix behandelt beide Ursachen generisch (er prüft nur auf NaN im
geladenen Array, unabhängig vom Grund).

## Bug 3 / eigentliche Aufgabe: Stationsmessungs-Imputation zu alt

`knnimputer_path`/`interpol_path` (`/mnt/lambda1/nvme1/synthetic/{knnimputer,interpol}/wind`)
waren gecacht und veraltet (KNN bis 2026-04-07, Kriging bis 2025-11-02) →
NaN-Audit schlug für ~26-27 Stationen in Fold3 fehl (`handle_nans: break`).

**Regeneriert** (Reihenfolge auf Wunsch: erst KNN, dann Kriging):

- `geostatistics/regen_knn_imputation.py` — eigenständiges Script, **nicht**
  `run_spatial_interpolation.py` selbst nutzen für reines KNN-Update: dessen
  interner KNN-Schritt läuft nativ auf 10-Min-Auflösung (T≈156k) →
  O(T²)-Distanzberechnung, hat beim ersten Versuch **~5h** gebraucht (und
  landete wegen fehlendem `data.knn_cache_path` in der Config sogar am
  falschen Pfad `/mnt/nvme1/...` statt `/mnt/lambda1/nvme1/...`). Das
  eigenständige Script resampled vor dem Fit auf 1h (jeder Consumer
  resampled ohnehin auf 1h) → ~36x schneller, ~3 Minuten für beide Features.
  Schreibt nach `/mnt/lambda1/nvme1/synthetic/knnimputer/wind/wind_{speed,direction}_knn10_start_end_67558851.parquet`
  (Hash = md5 der sortierten 203-Stations-Liste, exkl. `14138`).
- `configs/config_spatial_interpolation_regen.yaml` + zwei Fixes in
  `geostatistics/run_spatial_interpolation.py` (jetzt permanent im Code):
  1. Rohdaten-Parquets haben `timestamp` als **Index**, nicht als Spalte —
     Loader kannte das nicht, `reset_index()` ergänzt.
  2. `relative_humidity` (RK-Kovariate) hatte NaN-Lücken → `LinearRegression.fit`
     crashte. Jetzt: `ffill().bfill()` pro Station + Cross-Station-Mean-Fallback
     für Rest-NaN.
  Voller Lauf (26088 Timestamps, kein Parallelisierung im Code): **~2h15**.
  Schreibt nach `/mnt/lambda1/nvme1/synthetic/interpol/wind/Station_{sid}.parquet`
  (Spalten: `rk_pred`, `idw_pred`, `ok_pred`, `wind_speed_raw`, jetzt bis 2026-07-14).

**Für künftige Regenerierung:** einfach `regen_knn_imputation.py` erneut
laufen lassen (kein Argument nötig, liest `raw/wind` neu ein), danach bei
Bedarf `run_spatial_interpolation.py --config configs/config_spatial_interpolation_regen.yaml`
für Kriging (lange Laufzeit einplanen, kein `--dry-run`/Fortschritts-Resume).

## Lauf-Status (Stand 2026-07-22 22:25 CEST)

Alle Läufe: Suffix `val_new` (Val-Folds) bzw. `test_new` (Test-Zeiträume),
Modelle in `models/`, Test-Eval-Parquets in `data/raw_preds/`.

| | Val-Fold1 | Val-Fold2 | Val-Fold3 | Test1 | Test2 | Test3 |
|---|---|---|---|---|---|---|
| DCRNN | ✅ fertig | ✅ fertig | ✅ fertig | 🔄 Epoche ~35/200 | 🔄 Epoche ~29/200 | 🔄 Epoche ~18/200 |
| MTGNN | ✅ fertig | ✅ fertig | ✅ fertig | ✅ fertig (Eval da) | ✅ fertig (Eval da) | ✅ fertig (Eval da) |

DCRNN ist pro Epoche deutlich langsamer als MTGNN (ca. 20-40 Min/Epoche unter
GPU-Konkurrenz vs. 2-5 Min bei MTGNN) und war noch nicht fertig, als diese
Notiz geschrieben wurde. `patience: 15` (Early Stopping) — sollte nicht
mehr sehr lange brauchen. Nach Abschluss von DCRNN Test1/Test2/Test3 fehlt
noch der jeweilige `get_test_results_dcrnn.py`-Eval-Schritt (in den Screen-
Sessions bereits verkettet, läuft automatisch nach Trainingsende).

**Nach Abschluss prüfen:** `data/raw_preds/dcrnn_wind_dcrnn_nwp_hist_new_test_fold{1,2,3}_raw.parquet`
sollten dann existieren (fehlten bei Erstellung dieser Notiz noch).
