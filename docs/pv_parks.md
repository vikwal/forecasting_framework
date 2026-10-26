# pv_parks — PV-Parks (synthetisches Release, Trianel-Zwillinge und -Zähler) im Solar-TFT-Pfad

Stand 2026-10-16. `data.use_case: 'pv_park'` → `utils/pv_parks.py::preprocess_pv_park`, Vertrag wie der Solar-Pfad
(MultiIndex `starttime / forecasttime / timestamp`, 96 Schritte je Lauf bei `freq: 30min`, `df.attrs['target_cols'] = ['power']`).

## Daten

| Quelle | Pfad | Inhalt |
|---|---|---|
| Release | `${DATA_ROOT}/synthetic/solar/parks_pv_v1_30min/` | `synth_<id>.parquet` (30-min-Kette), `parks.csv` (`capacity_kwp_dc`), `sites.csv`, `pv_arrays.csv` |
| Zwillinge | `${DATA_ROOT}/synthetic/solar/trianel_twins_30min/` | `synth_pv<k>.parquet`, `meter_pv<k>.parquet` (`power_w`), `manifest_pv<k>.json` (arrays), `parks.csv`, `sites_confidential.csv` |
| ICON-D2 SL | `${DATA_ROOT}/nwp_ready_pv/icon-d2/SL/<HH>/park_<id>/<lat>_<lon>_SL.parquet` | Strahlung 15 min (`aswdir_s`, `aswdifd_s`), Momentanfelder stündlich; `points.cli icon-sl-extract/assemble` (NWP-points) |
| ECMWF HRES Solar | `${DATA_ROOT}/nwp_ready_pv/ecmwf/SOLAR/<00|12>/park_<id>/<lat>_<lon>_solar_sl.parquet` | `ssrd`, `fdir`, … als Stundenmittel (W/m²); `points.cli ecmwf --kind SOLAR` |

Dateinamen der Gitterpunkte sind **lat-first** (anders als der DWD-Stations-Store). Je Park der nächste Gitterpunkt
(`params.next_n_grid_points`, Suffix `_1` wie im Solar-Pfad).

## Zielgröße und Quellen je Park

`power` = `power_park` (synthetisch) bzw. `power_w` (Zähler) / DC-Kapazität. Ein Park ist **ein** Sample, auch bei
mehreren Ausrichtungen; die Ziele werden nie aufgeteilt. Welche Quelle ein Park hat, entscheidet `pv_parks.park_source`:
`data.path` + `data.target_source` (Default), `data.twin_files` aus `data.twin_path` (synthetische Zwillinge),
`data.meter_files` aus `data.meter_path` (Zählerziel). Damit kann eine Config auf Release + Zwillingen vortrainieren
und die Zähler als `holdout_files` bewerten.

## Geometrieinvariante Strahlung

Für ein zentrales Modell wird die Strahlung auf die Modulebene des Parks gebracht (pvlib, Hay-Davies, anteils-
gewichtet über `pv_arrays.csv` bzw. die Manifest-Arrays; das ist die POA-Summe der Kette selbst):

| Feature | Bedeutung |
|---|---|
| `gti_nwp` | ICON-D2 Modulebenenstrahlung des Parks |
| `gti_clearsky` | Clear-Sky-Modulebenenstrahlung (Ineichen) |
| `kc_gti_nwp` | `gti_nwp / gti_clearsky` (Clear-Sky-Index, 0 bei Nacht, Kappung 1,5) |
| `kt_nwp` | `ghi_nwp / ghi_clearsky` |
| `ecmwf_gti`, `ecmwf_kc_gti` | dasselbe aus ECMWF `ssrd`/`fdir` |

Dazu die ICON-Rohfelder (`params.icond2_features`), ECMWF (`params.ecmwf_features`, z. B. `ecmwf_ghi`, `ecmwf_dhi`),
Sonnengeometrie und Zeitfeatures aus `utils.solar.solar_geometry`. Statik: `capacity_kwp`, `tilt_eff`,
`azimuth_eff_sin/cos` (mittlere Normale), `latitude`, `longitude`, `altitude`; `park_id` kategorisch wie bei parks_v1.

## Zeitliche Zuordnung

Wie im Solar-Pfad: linksbündige Labels, Strahlung = Intervallmittel, das auf dem Lead endet, Momentanfelder zum
Intervallbeginn (`utils.solar.load_sl_grid_point`). Geprüft am Park SEL900323191631: Korrelation `ghi_nwp` gegen das
korrigierte CAMS des Release 0,957 bei Versatz 0 gegen 0,941/0,949 bei ±1 Schritt. Läufe mit nur stündlicher Strahlung
(Juli/August 2023) fallen bei `30min` heraus (15 von 128 im Smoke).

## GFS (2026-10-26)

`data.gfs_solar_path` (`${DATA_ROOT}/nwp_ready_pv/gfs/SOLAR`), `'gfs'` in `params.nwp_models` und `gfs_*`-Features
(`utils/gfs.py`): `gfs_ghi` = `dswrf` (Stundenmittel, das auf dem Lead endet) und `gfs_<Spalte>` unverändert, z. B.
`gfs_tcc`, `gfs_lcc`, `gfs_t_2m`, `gfs_tp`, `gfs_albdo` (Wolken und Albedo in %). Zuordnung wie ECMWF: Strahlung
`lead = floor(h) + 1`, Momentanfelder `floor(h)`, im 30-min-Raster wird der Stundenwert gehalten; Lauf = jüngster GFS-Lauf
(06/12) ≤ ICON-Start. GFS hat keine Direkt-/Diffusaufteilung und keine Clear-Sky-Größe, deshalb (noch) kein `gfs_gti`;
Feature Engineering später. Ein fehlendes GFS-Verzeichnis bricht ab, statt die Features still wegzulassen.
Geprüft an SEL981208943378 (ICON 12 UTC, Aug 2023, tagsüber): Korrelation `gfs_ghi` mit `ghi_nwp` 0,92, mit
`ecmwf_ghi` 0,93, mit der Leistung 0,91 (ICON 0,91).

## Configs und Smoke

`configs/pv_parks/config_pv_v1_cl_smoke.yaml` (3 Release-Parks, 1 Epoche) und `config_trianel_meter_smoke.yaml`
(Zähler pv1/pv3). Lauf: `python train_cl.py -c configs/pv_parks/config_pv_v1_cl_smoke -m tft`. Smoke 2026-10-16:
249 Trainingsfenster, Ergebnis unter `results/parks_pv_v1_30min/`.

## Offen

Few-Shot-Feintuning eines vortrainierten zentralen Modells auf Zählerdaten vor dem Testfenster (1 d / 1 w / 1 m / 3 m)
ist im CL-Pfad noch nicht verdrahtet (FL hat `fine_tune`); die Experimentkonfigurationen für die zwei Schnitte
(Vortraining bis 2023-11-30 bzw. 2024-02-29, Test Dez–Feb bzw. Mär–Mai 2024, Schnitt 2 am 2026-10-17 von Feb–Apr auf Mär–Mai verschoben, weil die Zähler bis 2024-05-31 reichen) liegen unter `configs/pv_parks/` sobald die
ICON-Extraktion vollständig ist.

## Zero-Shot, Few-Shot und Real-Referenzen (2026-10-16)

Configs in `configs/pv_parks/`, alle mit denselben geometrieinvarianten Features und den Trianel-Zählern als `holdout_files`:

| Config | Trainingsdaten | Zweck |
|---|---|---|
| `config_pv_v1_zeroshot_cut{1,2}` | alle 90 synthetischen Release-Parks (ohne die Zwillinge: `twin_files` deklariert nur die Quelle, trainiert wird nur `files`), `train_end` am Schnitt (2023-11-30 / 2024-02-29) | Zero-Shot: weder echte Daten noch die Standorte gesehen; gespeichert als `models/cl_m-tft_out-96_freq-30min_pv_v1_zeroshot_cut{1,2}.pt` + Scaler |
| `config_pv_v1_zeroshot_nostatic_cut{1,2}` | wie Zero-Shot, `static_features: []` | Ablation: was die statischen Features beim Transfer auf unbekannte Standorte bringen |
| `config_pv_v1_zeroshot_twins_cut{1,2}` | wie Zero-Shot plus die 6 synthetischen Zwillinge in `files` | Zero-Shot mit Standortkenntnis: keine echten Daten, aber die Geometrie und der Standort der Trianel-Parks aus dem Generator |
| `config_pv_v1_twincentral_cut{1,2}`, `config_pv_v1_twinlocal_pv*_cut{1,2}` (+ `twincentral_nostatic`) | nur die 6 synthetischen Zwillinge bzw. ein Zwilling | Vortraining auf dem synthetischen Nachbau des echten Portfolios, das direkte Gegenstück zu `realcentral`/`reallocal` |
| `config_pv_v1_realcentral_nostatic_cut{1,2}` | die 6 Zähler, `static_features: []` | Statik-Ablation auf echten Daten |

**Protokoll (seit 2026-10-17, Vorgabe Viktor):** keine `val_files`. Early Stopping läuft wie im übrigen Framework auf der Testperiode
der Trainingsstationen (bei den synthetischen Configs also auf synthetischen Daten im Testfenster, bei `realcentral`/`reallocal`
auf den Zählern im Testfenster, d. h. dort wählt das Early Stopping den Checkpoint auf den Holdout-Daten selbst, die Real-Referenzen
sind damit eher zu optimistisch). Die Zähler werden immer als `holdout_files` ausgewertet. Die Läufe vom 2026-10-16 mit 4 synthetischen
Val-Parks (86 Trainingsparks) sind durch diese Läufe ersetzt.
| `config_pv_v1_fewshot_cut{1,2}` + `train_cl.py --init-model <Zero-Shot-.pt> --finetune-days N` | die Zähler über die N Tage vor dem Testfenster | Few-Shot (1 / 7 / 30 / 90 d) |
| `config_pv_v1_realcentral_cut{1,2}` | die 6 Zähler von 2023-07-24 bis zum Schnitt | was die Zwillinge gegen ein Modell kosten, das die echten Daten gesehen hat |
| `config_pv_v1_reallocal_pv*_cut{1,2}` | ein Zähler | lokale Referenz |

**Few-Shot-Mechanik** (`utils/finetune.py`): State Dict (Präfix von torch.compile entfernt) und der gespeicherte x-Scaler
werden geladen, der Scaler wird nicht neu angepasst. Das Fenster wird auf `data.train_start/train_end` gesetzt, der übliche
Split gilt: ein Fenster am Lauf t0 braucht den Lauf t0 − H für die beobachtete Vergangenheit und hat Ziele [t0, t0 + H).
Der letzte Feintuning-Lauf ist der, dessen Ziele am ersten Testlauf enden (keine Zielüberlappung); N Tage ergeben N Fenster
je Park und Laufstunde. `model.finetune_trainable: [Präfixe]` friert alles andere ein. Early Stopping ist in den Few-Shot-
Configs aus (die synthetischen Val-Parks protokollieren nur den Verlust); das Rezept (lr, Epochen, trainierbare Menge) wird
nur auf synthetischen Parks gewählt, nie auf den Zählern. Ergebnis- und Modelldateien tragen den Suffix `_ft<N>d`.

## HPO auf dem synthetischen Portfolio

`config_pv_v1_hpo.yaml`: die 90 synthetischen Parks (keine Zwillinge, keine Zähler), `hpo.fold_boundaries` (wachsendes
Fenster, `hpo.kfolds_by_dates`) kombiniert mit `hpo.station_folds`: eine disjunkte Stationsgruppe je Fold, Fold i validiert
nur die Parks der Gruppe i in [b_i, b_i+1) und trainiert auf den übrigen Parks mit Ausgabe + Horizont ≤ b_i. Die Gruppen
kommen aus `utils.pv_parks.station_folds` (stratifiziert über ein 3×2-Gitter aus Breite/Länge, geseedet, wie die
Stationsfolds des CAMS-Korrekturmodells) und stehen in der Config, damit die Studie reproduzierbar ist. Die Horizontlücke ist
`hpo.horizon_hours(config)` (Schritte × `data.freq`; der alte Aufruf nahm 96 Halbstundenschritte als 96 h). `data.val_files`
bleibt leer (mit Datumsfolds nicht erlaubt). Smoke: `config_pv_v1_hpo_smoke.yaml` (6 Parks, 1 Trial). Der Pfad `--no-cache`
von `hpo_cl.py` ist unabhängig davon kaputt (`preprocessing.preprocess_data` existiert nicht); mit Cache laufen lassen.

**Ergebnisse 2026-10-17 (Holdout-R², Mittel über die 6 Zähler; Tabelle je Park in FL_Contribution/reports/pv_pretraining_results_2026-10-17.md):**

| Training | Schnitt 1 (Dez–Feb) | Schnitt 2 (Mär–Mai) |
|---|---|---|
| 90 synthetische Parks (Zero-Shot) | 0,72 | 0,77 |
| dito ohne statische Features | 0,73 | 0,77 |
| 90 Parks + 6 Zwillinge | 0,72 | 0,77 |
| 6 Zwillinge zentral / ohne Statik | 0,71 / 0,73 | 0,75 / 0,76 |
| 1 Zwilling lokal | 0,52 | 0,75 |
| 6 Zähler zentral / ohne Statik | 0,62 / 0,61 | 0,70 / 0,75 |
| 1 Zähler lokal | 0,52 | 0,75 |
| Few-Shot (alle Varianten, 1–90 d) | wie das Ausgangsmodell ±0,01 | wie das Ausgangsmodell ±0,01 |

Lesart: Vortraining auf den Zwillingen kommt bis auf 0,01–0,02 an die 90 Parks heran und schlägt die echten Daten im Winterschnitt
deutlich; statische Features bringen nirgends etwas und schaden dem echten zentralen Modell (6 Parks, Statik wirkt als Park-ID);
Few-Shot mit Gradientenschritten (Head-only oder alle Gewichte, lr 1e-5) verändert kein Modell messbar.

**Entscheidung 2026-10-17 (Viktor):** statische Features ab jetzt weglassen, `static_features: []` in allen `configs/pv_parks/*.yaml`
(auch HPO). Die Modelle `*_nostatic_cut{1,2}.pt` sind damit die Referenzmodelle; die Modelle mit Statik bleiben nur als Ablation.

**Fine-Tuning pro Park (2026-10-17):** statt eines gemeinsamen Few-Shot-Schritts auf allen Zählern wird das vortrainierte Modell je Park
auf dessen 7/30/90 Tagen vor dem Testfenster nachtrainiert (`files`/`holdout_files` = ein Zähler, `--init-model`, `--finetune-days`),
mit Early Stopping nach Framework-Default (Testfenster des Parks, Patience 5, max. 30 Epochen) und drei Rezepten:
Head lr 1e-4, alle Gewichte lr 1e-5, alle Gewichte lr 3e-5. Ausgangsmodelle: `zeroshot_nostatic` (90 Parks) und
`twincentral_nostatic` (6 Zwillinge). Configs werden aus `config_pv_v1_fewshot_twin_nostatic_cut{1,2}.yaml` generiert
(Session-Scratchpad `ftpark/`, Logs `run_ftpark_*.log`).
Ergebnis Fine-Tuning pro Park (216 Läufe Batch 64 + 48 Läufe Batch 4 mit 90 d): in keiner Kombination aus Ausgangsmodell, Rezept,
Fenster und Park mehr als ±0,01 R² gegenüber dem Ausgangsmodell, obwohl das Early Stopping auf dem Testfenster selbst die Epoche wählt.
Gradienten-Fine-Tuning auf wenigen echten Tagen ist für diesen Aufbau damit abgehakt; der Restfehler liegt im NWP-Input, nicht in
parkspezifischen Abweichungen. Tabellen: FL_Contribution/reports/pv_pretraining_results_2026-10-17.md.

**Prognoseläufe (2026-10-17):** `data.forecast_hours: ['06','09','12','15']` mit/ohne `data.train_forecast_hours: ['09']`
(`config_pv_v1_{zeroshot,realcentral}_nostatic_runs4{,tr09}_cut{1,2}.yaml`). Training auf allen vier Läufen bringt gegenüber dem
09-Lauf nichts (R² ±0,01, Zähler-Modell Schnitt 1 −0,03); 15 UTC ist der schwächste Lauf (−0,01 bis −0,02). ECMWF-Zuordnung:
jüngster Lauf ≤ ICON-Start, also 00 für 06/09 und 12 für 12/15 (operativ wäre der 12er für 12/15 UTC noch nicht verfügbar).
Vorhersagen je Laufstunde: FL_Contribution/reports/pv_runs_2026-10-17/eval_runs_2026-10-17.txt.

## Review 2026-10-17 (vor der HPO)

Bericht: `FL_Contribution/reports/review_pv_pretraining_2026-10-17.md`. Im Framework gefixt (Tests `tests/test_pv_review.py`):

- **Testfenster** (`preprocessing.prepare_data_for_tft`): die Rückverlängerung des Testsplits um den Lookback rechnete
  `history_length` als Stunden, nicht als Schritte. Bei `freq: 30min` reichte der Testsplit 96 h statt 48 h zurück, und die
  zwei Läufe vor `test_start` (29./30.11. bzw. 28./29.02.) wurden Testfenster (93 statt 91 Läufe) – ausgegeben vor dem
  Testbeginn, Ziele im Trainingsfenster, in Early Stopping und in den gespeicherten Vorhersagen enthalten (in der
  Framework-Metrik fielen sie nur zufällig heraus, weil die Persistenz-Baseline dort NaN ist). Jetzt Schritte × Frequenz
  plus Filter `starttime >= test_start`; Cache-Schlüssel für sub-stündliche Daten geändert.
- **`wind_speed_nwp`** (`preprocessing.prepare_features_for_tft`): der Wind-Sonderfall suchte nur `wind_speed_h*`-Spalten und
  verwarf die SL-Spalte `wind_speed_nwp_1` stillschweigend – alle PV-Modelle bis 2026-10-17 liefen mit 20 statt 21
  bekannten Features. Jetzt Rückfall auf den exakten Namen. Alte `.pt`-Modelle passen deshalb nicht mehr als `--init-model`.
- **Cache-Schlüssel** (`data_cache._get_config_hash`): `target_source`, `twin_*`, `meter_*`, `nwp_path`, `forecast_hours`,
  `sub_hourly_fill` gehen für `pv_park`/`solar` in den Hash ein.
- Debug-Schreibzugriff auf `/tmp/ecmwf_debug.txt` in `preprocessing.pipeline` entfernt.

Protokollempfehlung (Entscheidung Viktor, siehe Bericht): `data.strict_split: True` in den Schnitt-Configs, damit das letzte
Trainingsziel vor dem ersten Testlauf liegt (bisher reichen die Ziele des Laufs vom 30.11. 09 UTC bis 02.12. 08:30, bei den
Real-Referenzen also Zählerziele des ersten Testtags im Training).
