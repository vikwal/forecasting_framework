# Implementierungsspezifikation: QRF- und MOS-Baselines

**Erstellt:** 2026-08-10 · **Phase 1 (Exploration) der dreiphasigen Aufgabe aus
`prompt_baselines_orchestration.md`** · **Verbindlich für den Implementierungsagenten
(Phase 2).**

Basis: `forecasting_framework` auf `l2`, Branch `fix/mtgnn-topo-static-dim`,
**HEAD `d49096f582ca4932bd10ae23afc46ec8c271c8d4`**, Arbeitsbaum clean (selbst geprüft,
2026-08-10 12:47 UTC). Alle Zeilennummern in diesem Dokument beziehen sich auf **genau
diesen Commit**.

> **Status der Aussagen.** Jede Behauptung über den Code ist an diesem Commit verifiziert,
> mit Datei und Zeile. Jede Zahl, die nicht aus einem Dokument zitiert ist, wurde für dieses
> Dokument auf `l2` nachgerechnet; die Rechnung steht jeweils dabei. Abschnitt 9 listet
> auf, was NICHT verifizierbar war.

---

## 0. Executive Summary für den Implementierungsagenten

Es entstehen **sechs neue Dateien** unter `geostatistics/baselines/` plus **ein**
HPO-Skript und **drei** Config-Dateien. Kein bestehender Produktivcode wird verändert.
`_station_metrics` und `_save` aus `evaluate_reference.py` werden **importiert**, nicht
nachgebaut.

Der Pflichtumfang ist:

| Arm | Getunt | Fit-Stationen | Zeitfenster Fit | Auswertung an |
|---|---|---|---|---|
| **QRF-local** | ja, eigene Optuna-Studie | 102 Fold-Trainingsstationen | Läufe `t_run < 2024-08-01` | 51 Fold-Zielstationen |
| **MOS-regional** | nein | 102 Fold-Trainingsstationen (gepoolt) | dito | 51 Fold-Zielstationen |
| **MOS-nearest** | nein | je Trainingsstation einzeln | dito | 51 Zielstationen, Koeffizienten der geodätisch nächsten Trainingsstation |
| **MOS-local** | nein | je **Zielstation** auf **ihrer eigenen** Historie | dito | dieselbe Zielstation (transduktive Obergrenze) |

**Vier Dinge, die den ganzen Auftrag kippen, wenn sie falsch gemacht werden:**

1. `t_run_abs` ist der Index des **ersten Prognoseschritts** (`t_run + 1 h`).
   Lead `h ∈ 1…48` liegt bei Array-Position `h-1`. Wer die Laufzeit braucht, nimmt
   `timestamps[t_run_abs - 1]`.
2. Die Run-Paar-Menge wird **nicht neu erfunden**. Sie wird mit exakt der Schleife aus
   `evaluate_reference.py:425-447` gebaut — **einschließlich des `r_hist`-Filters**, den
   die Baselines fachlich nicht brauchen (Abschnitt 4.3).
3. Der Fit sieht **niemals** eine der 51 Zielstationen — außer bei MOS-local, wo genau das
   der Zweck ist.
4. `nwp_ref` und `pers_ref` in den erzeugten Parquets müssen **bitgleich** mit den
   Modell-Parquets derselben `(station_id, valid_time)`-Zeilen sein.

---

## 1. Die ausgelesene Referenzpipeline

### 1.1 Stationsauswahl je Fold — wo sie steht und wie sie geladen wird

`configs/spatial_folds.yaml` (5151 Bytes, MD5-Präfix über `fold_hash()`,
`geostatistics/spatial_cv.py:55`) definiert drei Schlüssel `spatial_fold1/2/3`, je mit
`files` (102) und `val_files` (51).

Nachgerechnet auf `l2`:

```
spatial_fold1: train=102 val=51 overlap=0 union=153
spatial_fold2: train=102 val=51 overlap=0 union=153
spatial_fold3: train=102 val=51 overlap=0 union=153
val0 ∩ val1 = 0 · val0 ∩ val2 = 0 · val1 ∩ val2 = 0 · Vereinigung der Val-Mengen = 153
```

**Zwei verschiedene Ladewege existieren, und sie ergeben dieselben Mengen, aber eine
andere Reihenfolge:**

| Weg | Code | Stationsliste | Indexbedeutung |
|---|---|---|---|
| **HPO** (`cv_mode: spatial`) | `hpo_mtgnn.py:426-438`, `spatial_cv.py:96-98` | `station_pool()` = **sortierte Vereinigung** der 153 IDs; `files`/`val_files` der Config werden bewusst ignoriert | `train_idx`/`val_idx` aus `build_folds()` (`spatial_cv.py:101-134`) |
| **Retrain / Eval / Referenz** | `get_test_results_mtgnn.py:199-212`, `evaluate_reference.py:275-283` | `data_cfg["files"] + data_cfg["val_files"]`, also **train zuerst** | `val_indices = arange(N_train, N_train+N_val)` |

Nachgerechnet: für alle drei Folds und für `configs/mtgnn/config_wind_mtgnn_nwp_fold{1,2,3}.yaml`,
`configs/mtgnn/stdhp/…`, `configs/dcrnn/config_wind_dcrnn_fold{1,2,3}.yaml`,
`configs/dcrnn/stdhp/…` und `configs/tft_bc/config_wind_tft_sp_base_fold{1,2,3}.yaml` gilt
`set(files) == set(spatial_foldN.files)` und `set(val_files) == set(spatial_foldN.val_files)`.
**Kein Konflikt.** Die 50 `test_files` sind von den 153 disjunkt (Schnittmenge 0).

**Fold-Nummerierung — die häufigste Verwechslung:** Config `fold1` → Ausgabe-Index `0`.
Belegt in `launch_reference_eval.py:28-32` und im Kommentar
`launch_eval_pipeline.py` („Config fold1 → Notebook fold0"). Der TFT-Pfad ist 1-basiert,
`make_stdhp_figures.py:143` (`TFT_FOLD_OFFSET = 1`).

> **Für die Baselines verbindlich:** Stationsliste = `data_cfg["files"] + data_cfg["val_files"]`
> (train zuerst), also der Retrain/Eval-Weg. Grund: die Baselines müssen ihre Ausgabe
> zeilenweise gegen `icon_d2_fold{n}_raw.parquet` und die Modell-Parquets stellen können,
> und die sind so gebaut. Die HPO-Reihenfolge ist irrelevant, weil `station_id` in jeder
> Ausgabezeile steht.

### 1.2 Zeitfenster — was die Configs sagen, und wo sie dem Auftrag widersprechen

Nachgelesen in allen Fold-Configs:

| Config-Familie | `val_start` | `test_start` | `test_end` |
|---|---|---|---|
| `configs/mtgnn/config_wind_mtgnn*_fold{1,2,3}.yaml` | 2024-08-01 | 2025-08-01 | **2026-03-31** |
| `configs/mtgnn/stdhp/…`, `configs/dcrnn/stdhp/…` | 2024-08-01 | 2025-08-01 | **2026-03-31** |
| `configs/dcrnn/config_wind_dcrnn.yaml` | 2024-08-01 | 2025-08-01 | **2025-10-31** |
| `configs/tft_bc/config_wind_tft_sp_base_fold{1,2,3}.yaml` | 2024-08-01 | 2025-08-01 | **2025-10-31** |
| `configs/mtgnn/test/…`, `configs/dcrnn/test/…` | *fehlt* | 2025-08-01 | 2025-11-30 |

**Keine einzige Config trägt das laut Auftrag gültige `test_end: 2026-07-31.`**
`study_overview.md` §6 nennt 2025-10-31, `stdhp_dryrun_results.md` §1 nennt 2026-03-31, der
Auftrag nennt 2026-07-31. Der Kommentar in den Kampagnen-Configs
(`config_wind_mtgnn_nwp_fold1.yaml:32`) begründet 2026-03-31 mit „Grenze gegen NaN in
ECMWF-Parquets ab 2026-05-01".

**Das Val-Fenster ist dagegen eindeutig und in allen Folds identisch:**
Train bis `2024-08-01` = **1473 Run-Paare**, Val `2024-08-01 … 2025-08-01` = **1460
Run-Paare**. Nachgerechnet aus dem GNNCache `data_cache/gnns/07f8bea34c198f83/derived.pkl`:
`all_run_pairs` = 2933 = 1473 + 1460, erste/letzte Laufzeit
`2023-07-26 06:00` … `2024-07-31 15:00` (Train) und `2024-08-01 06:00` … `2025-07-31 15:00`
(Val). Deckungsgleich mit `verify_evaluate_reference_fix.py:19-21`
(`EXPECTED_ROWS = 3_574_080 = 1460 × 51 × 48`) und mit `stdhp_dryrun_results.md` §1.

**Das Testfenster ist datenseitig nicht auf zwölf Monate zu bringen.** Siehe Abschnitt 8.1;
das ist der wichtigste Befund dieses Laufs und keine Baseline-Frage.

### 1.3 Die Run-Paar-Schleife — Zeile für Zeile

Drei Implementierungen, funktional identisch bis auf einen Punkt:

| Ort | Zeilen |
|---|---|
| `evaluate_reference.py` | 420-456 |
| `get_test_results_mtgnn.py` | 375-408 |
| `hpo_mtgnn.py::_build_all_run_pairs` | 180-222 |

Die Schleife, wörtlich aus `evaluate_reference.py:425-447`:

```python
for r_curr in range(R):
    t_run = run_times[r_curr]
    if t_run < split_time:                 continue   # Fensteruntergrenze
    if eval_cutoff is not None and t_run >= eval_cutoff: continue   # Obergrenze
    if t_run not in ts_lookup.index:       continue
    t_run_abs = int(ts_lookup[t_run]) + 1              # ERSTER PROGNOSESCHRITT
    if t_run_abs < H or t_run_abs + F_h > T: continue
    t_hist_target = t_run - pd.Timedelta(hours=H * freq_h)
    diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
    r_hist  = int(np.argmin(diffs_s))
    if diffs_s[r_hist] > 3 * 3600:         continue    # r_hist-Filter
    if _meas_nan_any[t_run_abs - H : t_run_abs + F_h].any(): continue
    val_run_pairs.append((r_curr, r_hist, t_run_abs))
```

Es fallen also Paare weg, wenn (a) die Laufzeit nicht im Messzeitraster liegt, (b) das
96-Stunden-Fenster über den Datenrand ragt, (c) **kein Historienlauf innerhalb ±3 h** um
`t_run − 48 h` existiert, (d) **irgendeine** der 153 Stationen in irgendeiner der 96
Stunden nach der Imputationskette noch NaN hat.

**Der eine Unterschied:** `hpo_dcrnn.py`, `hpo_mtgnn.py` (dort Zeilen 666-682) und
`hpo_wavenet.py` schließen zusätzlich Läufe mit NaN im ICON-D2-Gitter aus. `train_dcrnn.py`,
`train_mtgnn.py`, alle vier `get_test_results_*.py` und `evaluate_reference.py`
haben diesen Filter **nicht** (mit `grep -c "_grid_nan_runs\|NaN in ICON-D2 grid"`
verifiziert: HPO-Skripte 5–6 Treffer, alle anderen 0).

Nachgerechnet, warum das derzeit folgenlos ist: im Cache `07f8bea34c198f83`
(`grid_icond2_runs` mit Form `(3303, 48, 1071, 4)`) haben **0 von 3303 Läufen** irgendein
NaN. Der Filter würde in Train- und Val-Fenster **0 Paare** verwerfen. HPO und
Evaluation rechnen dort also auf identischen Paarmengen. Für das Testfenster gilt das
nicht mehr — siehe 8.1.

### 1.4 Featurekonstruktion

**Kanäle laut Config** (`config_wind_mtgnn_nwp_fold1.yaml:41-44`):

```yaml
icond2_features: ['u_10m', 'v_10m', 'wind_speed_10m', 'wind_speed_38m']
ecmwf_features:  ['u_wind10m', 'v_wind10m', 'wind_speed_10m']
icond2_feature_mode: dir_in_deg
ecmwf_feature_mode:  dir_in_deg
```

`apply_dir_encoding` (`train_dcrnn.py:199-271`) ist **keine monotone Transformation je
Spalte**, sondern eine Reparametrisierung: gepaarte `(u, v)` werden durch
`(wind_speed, sin_dir, cos_dir)` ersetzt, nicht gepaarte Spalten wandern **nach vorn**.
Aufgelöst ergibt das:

| Modus | ICON-D2 (I2 = 4) | ECMWF (E2 = 3) |
|---|---|---|
| `absolute` | `u_10m, v_10m, wind_speed_10m, wind_speed_38m` | `u_wind10m, v_wind10m, wind_speed_10m` |
| `dir_in_deg` | `wind_speed_38m, wind_speed_10m, sin_dir_10m, cos_dir_10m` | `wind_speed_10m, sin_dir_10m, cos_dir_10m` |

Weil `wind_speed_38m` in `dir_in_deg` auf **Index 0** rutscht, suchen alle Skripte den
Windgeschwindigkeitsindex namentlich (`evaluate_reference.py:370-373`,
`evaluation.py::find_ws_feat_idx:36-50`). Das ist zu übernehmen, nicht zu hardcodieren.

**k nächste Gitterpunkte — geodätisch, auf zwei unabhängigen Wegen:**

| Weg | Code | Methode |
|---|---|---|
| Loader, „nächster" Punkt je Station | `train_stgnn2.py::_select_nearest_grid_files:295-324`, Rückgabe `station_nearest_grid` in `load_icond2_ml_runs:453-461` | `pyproj.Geod(ellps="WGS84").inv`, auf den ~22 Gitterdateien **im Stationsverzeichnis** |
| Sampler, k nächste | `homo_sampler.py::_init_grid_knn:193-243` → `spatial.py::geodesic_knn:88-123` | `pairwise_geodesic_km` über das **globale, deduplizierte** Gitter |
| ECMWF | `train_stgnn2.py::load_ecmwf_parquet_at_stations_and_grid:796-809` | `_GEOD.inv` über alle 553 (bzw. `unique_grids`) Punkte |

**Nachgerechnet, dass die beiden ICON-D2-Wege übereinstimmen:** für alle 153 Poolstationen,
`k = 7` (die HPO-Obergrenze), 1071 eindeutige Gitterknoten:
`nearest-grid agreement loader vs geodesic_knn: 153 / 153`. Da die Gitterknotenmenge mit
kleinerem `k` eine Teilmenge ist und der Loader-Nachbar immer darin liegt, gilt die
Übereinstimmung für jedes `k ≤ 7`. **Kein Rest-`cKDTree` mehr im Spiel** — der Befund B2
ist an beiden Stellen behoben (`homo_sampler.py:205-209` und `evaluate_reference.py:376-385`
dokumentieren die Reparatur), `cKDTree` wird in `evaluate_reference.py` **nicht mehr
importiert** (Abschnitt 10.1).

**Neun Topo-Deskriptoren**, kanonische Reihenfolge `topo_features.py:27-30`:
`slope, aspect_sin, aspect_cos, tpi5, tpi75, tdi, elev_std, z0, dist_coast`.
Geladen über `load_topo_station_features_dict` (`topo_features.py:219-239`) mit
varianzstabilisierenden Transformationen (`_TOPO_TRANSFORMS:53-61`) und z-Score **nur auf
den Fold-Trainingsstationen**; `aspect_sin/cos` bleiben ungeskaliert.

Nachgerechnet für Fold 1: keine fehlenden Werte für die 153 Stationen (nur Station `02961`
bekommt `aspect_*` und `tdi` = 0 wegen Nullrelief, das ist der dokumentierte
Sonderfall), `max|Δ|` zwischen Fit auf 102 und Fit auf 153 Stationen = **0.5978**, und —
entscheidend für QRF — **die spaltenweise Rangfolge der Stationen ist in beiden Fällen
identisch**. Ein Random Forest kann die beiden Varianten daher nicht unterscheiden
(Verifikationstest V7).

**Statische Knotenfeatures des Modells** (`homo_sampler.py::_build_static:343-364`):
`sin_lat, cos_lat, sin_lon, cos_lon, alt_norm, type_indicator` + 9 Topo = 15 Spalten.
`alt_norm` wird auf `alts[self.train_idx]` normiert (`_init_static:259-262`).
Über dem deutschen Ausschnitt (nachgerechnet: lat 47.398 … 55.011, lon-Bereich innerhalb
6…15 °E) sind `sin_lat`, `cos_lat`, `sin_lon`, `cos_lon` jeweils **strikt monoton** in
`lat` bzw. `lon` — für einen Baum also äquivalent zu `lat`, `lon` selbst.

### 1.5 Imputationskette und Auswertungsfilter

**Kette, in dieser Reihenfolge** (`evaluate_reference.py:301-317`,
`get_test_results_mtgnn.py:229-244`, identisch):

1. `load_station_measurements` (`train_stgnn2.py:85-119`): pro Station
   `Station_{sid}.parquet`, Spalten `wind_speed, wind_direction`, resample `1h`,
   `closed="left", label="left"`, Mittelwert.
2. ~~`load_interpol_imputation` + `apply_interpol_imputation`: Regression-Kriging
   `rk_pred` füllt NaN **nur im Zielkanal** `wind_speed`.~~ **Überholt.** Seit
   2026-08-11 füllte stattdessen die ERA5-OLS (`docs/imputation_era5_only.md`), seit
   2026-09-02 `impute_meas_raw_from_interpol` aus der TFT-Spalte `imputed`
   (`docs/imputation_tft_switch.md`); `rk_pred` existiert in `interpol/wind` nicht mehr.
   Der Rest dieses Abschnitts inklusive der Abdeckungstabelle beschreibt den Stand vor
   diesen beiden Umstellungen.
3. `load_knn_imputation` + `apply_knn_imputation` (`utils/imputation.py:74-118` / `146-170`):
   KNN-Parquet füllt den Rest, **je Kanal**, nur wo der KNN-Wert selbst nicht NaN ist.
   `matches[-1]` wählt die Datei — sortiert gewinnt `wind_speed_knn10_start_end_67558851.parquet`
   über `wind_speed_knn10_start_20251102_67558851.parquet`.
4. `encode_circular_measurements` (`train_dcrnn.py:168-196`): `wind_direction` →
   `sin_wind_direction, cos_wind_direction`. `wind_speed` behält Index 0.
5. `_meas_nan_any = np.isnan(meas_raw).any(axis=(1,2))` wird **vor** Schritt 4 gebildet
   (`evaluate_reference.py:316`) und steuert den Run-Paar-Filter.

**Abdeckung nachgerechnet** (`/tmp/testwin.py` auf `l2`, alle 153 Poolstationen,
2023-07-24 … 2026-07-14, 26088 Stunden):

```
vor  Imputation: 18564 von 26088 Stunden haben mindestens eine Station-NaN  (71.2 %)
nach Imputation:     0 von 26088 Stunden
KNN 'wind_speed'    : 153/153 Stationen, NaN remaining 0
KNN 'wind_direction': 153/153 Stationen, NaN remaining 0
```

Also: **kein Run-Paar fällt wegen Rest-NaN aus** — nicht im Train-, nicht im Val-, nicht im
Testfenster. Die 220 `meas_nan_any`-Stunden im Cache `d67d98241545ae6d` liegen jenseits des
Messdatenendes (dessen `timestamps` reichen bis 2026-07-28 23:00, die Rohmessungen und die
KNN-Datei bis 2026-07-14 23:00) und sind ein Cache-Artefakt, kein Datenproblem.

**Abdeckungsränder der Imputationsquellen** (nachgerechnet):

| Quelle | Bis |
|---|---|
| `interpol_path` (`rk_pred`, Regression-Kriging) | **2025-11-02 21:00 UTC** |
| `knnimputer_path` `wind_speed_knn10_start_end_…` | 2026-07-14 23:00 UTC |
| Rohmessungen `Station_*.parquet` (10-min) | 2026-07-14 23:50 UTC |

**Folge, die ins Paper gehört:** ab 2025-11-02 füllt **nur noch der KNN-Imputer**, das
Kriging liefert dort NaN und greift nicht. Die Imputationskette wechselt also **innerhalb
des Testfensters** ihren Charakter. Für die Baselines ist das folgenlos (dieselben Daten wie
für die Modelle), für die Methodenbeschreibung nicht.

**Auswertungsfilter: der Imputationsausschluss ist post-hoc, nicht in der Pipeline.**
Kein Produktivskript filtert imputierte Zielstunden. Das macht erst
`geostatistics/stdrun/make_stdhp_figures.py`:

| Funktion | Zeilen | Was sie tut |
|---|---|---|
| `build_imputation_mask` | 268-285 | pro Station 10-min-Rohmessung → `resample("1h").mean()`, `isna()` = imputiert |
| `_lookup_imputed` | 288-310 | Zeilenflag über `(station_id, valid_time)`; außerhalb der Rohdatei = imputiert |
| `verify_imputation_mask` | 323-348 | jede Stunde mit `gt < 0` **muss** als imputiert erkannt sein, sonst Abbruch |
| `_verify_uniform_filtering` | 373-394 | nach dem Filter **identische Zeilenzahl je Fold über alle Varianten**, sonst Abbruch |

> **Verbindlich für die Baselines:** imputierte Stunden bleiben **im Fit** und werden
> **in der Auswertung** über genau diese Maske entfernt. Damit das funktioniert, müssen die
> Baseline-Parquets `station_id` und `valid_time` tragen (tun sie, wenn die Record-Dicts aus
> `evaluate_reference.py:189-198` übernommen werden) **und dieselben
> `(station_id, valid_time)`-Schlüsselmengen haben wie die Modell-Parquets** — sonst bricht
> `_verify_uniform_filtering` die spätere Auswertung ab. Das ist Verifikationstest V5.

`stdhp_dryrun_results.md` §2 nennt die Imputationsanteile: 0.60 % (Fold 0), 0.68 % (Fold 1),
0.86 % (Fold 2), Streuung über Stationen 0 … 8.95 %, 34 von 153 Station-Fold-Kombinationen
ganz imputationsfrei.

**TFT-Abweichung (§5.6 dort), nur zur Kenntnis:** der TFT liest Rohmessungen
(`raw_station_source: true`, kein `interpol_path`/`knnimputer_path`) und verwirft Läufe mit
echter Messlücke — 2.99–3.05 M statt 3.54–3.55 M Zeilen (84–86 %), 1452 statt 1460 Paare.
**Die Baselines dürfen das nicht nachmachen.** Sie gehören auf die Graphen-Seite: dieselbe
Imputationskette, dieselben 1460 Paare, damit die Schnittmengenbildung in
`make_stdhp_figures.py` unverändert funktioniert.

### 1.6 Ausgabe- und Auswertungsformat

**`_station_metrics`** (`evaluate_reference.py:68-108`) — Signatur
`(preds_acc, gts_acc, nwp_acc, pers_acc, val_ids)`, jeweils Listen von Listen von
`(F_h,)`-Arrays, eine Liste je Station in der Reihenfolge von `val_ids`.

Spalten der Rückgabe, in dieser Reihenfolge:
`station_id, mae, rmse, r2, skill, skill_nwp, n_samples`.

Rechenregeln (Zeilen 84-102):
* `ok = ~(isnan(pred) | isnan(gt))`, mindestens 2 gültige Werte, sonst Station **entfällt**
* `rmse = sqrt(mean_squared_error(gt[ok], pred[ok]))`, `mae`, `r2` auf derselben Maske
* `pers_rmse` und `nwp_rmse` auf **eigenen** Masken (`ok_p`, `ok_n`)
* `skill = 1 − rmse/pers_rmse`, `skill_nwp = 1 − rmse/nwp_rmse`, sonst `NaN`
* `n_samples = int(ok.sum())`

Identisch in `evaluation.py:305-351` (DCRNN) und `homo_sampler.py:653-694` (MTGNN/WaveNet).

**`_save`** (`evaluate_reference.py:111-122`):
`data/test_results/{stem}.csv` (ohne Index) und `data/raw_preds/{stem}_raw.parquet`.

**Roh-Parquet-Spalten** (`evaluate_reference.py:189-198`, identisch
`evaluation.py:294-303` und `homo_sampler.py:642-651`):
`station_id, run_time, valid_time, horizon, pred, gt, nwp_ref, pers_ref`
mit `run_time = timestamps[t_run_abs - 1]`, `valid_time = run_time + (h+1) h`,
`horizon = h + 1`.

**Aggregationskonvention** (`stdhp_dryrun_results.md` §1, entschieden):
erst über die 51 Stationen eines Folds, dann über die drei Folds. Skill wird als
**Mittel der Stations-Skills** berichtet, nicht als `1 − Verhältnis der Mittel`.
Wichtige Größenordnung von dort: ICON-D2 per Station **1.325**, gepoolt **1.450** —
0.12 m/s Unterschied. Tabelle und Abbildung nie ohne Fußnote im selben Absatz.

**Bestehende Referenzdateien.** In `l2:data/test_results/` liegen die Fold-Referenzen
**nicht**; sie wurden am 2026-08-06 nach
`data/test_results/Archiv/veraltete_folds_2026-08-06/` verschoben (falsche, ältere
Fold-Definition, 50 statt 51 Stationen, Überschneidung 20–45 %). Die **gültigen**
Referenzen und alle 30 stdhp-Modellergebnisse liegen auf **`l1`**:
`l1:data/test_results/icon_d2_fold{0,1,2}.csv` (je 51 Zeilen + Kopf, `n_samples = 70080`
in jeder Zeile), `ecmwf_fold{0,1,2}.csv`, `stdhp_*_fold{0,1,2}.csv`, sowie die zugehörigen
`data/raw_preds/*.parquet` (je 50.4–50.5 MB). `l1` steht auf demselben Commit `d49096f`.

> **Verbindlich:** der Formatvergleich (V5) und der Referenzvergleich (V6) laufen gegen die
> Dateien auf **`l1`**. Wer sie auf `l2` sucht, findet nur den Archivstand.

### 1.7 Metriken und Referenzen

| Größe | Woher |
|---|---|
| `nwp_ref` | rohes ICON-D2 am geodätisch nächsten Gitterpunkt: `grid_icond2_runs[r_curr, :F_h, nearest_i2, ws_idx]` — `evaluate_reference.py:166-167`, `homo_sampler.py:627-628`, `evaluation.py:237-242` |
| `pers_ref` | `meas_raw[t_run_abs - 1, station, target_idx]`, über alle 48 Leads konstant — `evaluate_reference.py:170-171`, `homo_sampler.py:631-632`, `evaluation.py:244-246` |
| `skill` | `1 − RMSE / RMSE(Persistenz)` |
| `skill_nwp` | `1 − RMSE / RMSE(rohes ICON-D2)` |

Für die ICON-D2-Referenz selbst setzt `evaluate_reference.py:218` `nwp_acc` auf NaN, damit
`skill_nwp` dort NaN und nicht 0 ist. Deshalb ist die Spalte `skill_nwp` in
`l1:data/test_results/icon_d2_fold*.csv` leer.

Nachgeprüft laut `stdhp_dryrun_results.md` §1: die neuen `icon_d2_fold{0,1,2}.csv` stimmen
in allen drei Folds **bitgenau** mit dem aus `nwp_ref` gerechneten Wert überein.

**Bekannte Referenzwerte** (aus `stdhp_dryrun_results.md` §2/§5.4, Val-Fenster,
per Station gemittelt, Mittel über die drei Folds):

| Referenz | ungefiltert | ohne imputierte Zielstunden |
|---|---|---|
| rohes ICON-D2 | **1.325** ± 0.0288 (Folds 1.339 / 1.285 / 1.351) | **1.304** ± 0.024 |
| ECMWF HRES | 1.343 | 1.323 |
| Persistenz | 2.256 | 2.240 |
| DCRNN GRID-NOGRAPH (C, reines standortweises Downscaling) | 1.161 | **1.138** ± 0.028 |
| TFT base (induktiv, kein Graph) | 1.186 | 1.186 |

### 1.8 HPO-Mechanik

| Element | Wert / Code |
|---|---|
| Storage | PostgreSQL, `OPTUNA_STORAGE` (auf `l2` gesetzt, verifiziert). `optuna.storages.RDBStorage(url, heartbeat_interval=60, engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600})` — `hpo_mtgnn.py:1075-1082` |
| Studienname | `f"cl_m-{modell}_out-{F_h}_freq-{freq}_{hpo_stem}"` mit `hpo_stem = re.sub(r'_fold\d+$', '', Path(config).stem.replace("config_",""))` — `hpo_mtgnn.py:340-369`; identisch im Eval-Pfad `get_test_results_mtgnn.py:154-158` |
| Richtung | `minimize`, `TPESampler()`, `load_if_exists=True` — `hpo_mtgnn.py:1088-1095` |
| Pruner | `MedianPruner(n_startup_trials=20, n_warmup_steps=1)` aus `hpo.pruner_*` — `hpo_mtgnn.py:388-394` |
| Pruning-Punkt | `trial.report(mean(fold_rmses_bisher), step=fold_idx)` nach jedem Fold, dann `should_prune()` — `hpo_mtgnn.py:1058-1060` |
| Objective | `mean_rmse = float(np.mean(fold_rmses))`, ein Wert je Fold — `hpo_mtgnn.py:1065-1072` |
| Budget | `remaining = max(n_trials − completed, 0)`, **einmal beim Start** — `hpo_mtgnn.py:1097-1098`. N Worker holen sich je `n_trials`. |
| Provenienz | `trial.set_user_attr` für `host`, `commit`, `station_node_features`, `broadcast_topo`, `fold_hash` — `hpo_mtgnn.py:880-884` |
| Sampling | `sample_hyperparameters` (`hpo_mtgnn.py:172-173`) sampelt **generisch alles** unter `hpo.params` |

**Was `fold_rmses[k]` genau ist — verifiziert, und für die QRF-Studie entscheidend:**
`_fit_model` (`hpo_mtgnn.py:229-296`) gibt `best_val` zurück, und das stammt aus
`_val_epoch` (`train_mtgnn.py:177-211`), das `_metrics(preds_cat, gts_cat, target_scale,
target_mean)` auf den **konkatenierten** Vorhersagen aller Val-Läufe rechnet. Das ist ein
**über Stationen, Läufe und Leads gepooltes RMSE in physikalischen Einheiten**, nicht das
Mittel der Stations-RMSEs.

Der Curriculum-Vorbehalt löst sich auf: `cl_steps = min(epoch, F_h)` bei `cl_period: 1`,
und `best_val` wird bei jeder Erhöhung von `cl_steps` zurückgesetzt
(`hpo_mtgnn.py:263-267`), gleichzeitig auch `no_improve`. Vor Epoche 48 kann daher kein
Early Stopping greifen; `best_val` wird immer bei `cl_steps = 48`, also über den vollen
Horizont, gemessen.

> **Verbindlich für die QRF-Studie:** Objective = **Mittel über die drei räumlichen Folds
> des gepoolten, unskalierten Val-RMSE in m/s** über alle 51 Zielstationen × 1460 Val-Paare
> × 48 Leads. **Nicht** das Mittel der Stations-RMSEs. Das ist Verifikationstest V8.

**Studienbestand am 2026-08-10 12:56 UTC** (lesend über `get_all_study_summaries`):

```
cl_m-dcrnn_out-48_freq-1h_wind_dcrnn                56 trials  best 1.2506
cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_base          111        1.2539
cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_idw_alt        40        1.2361
cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_nograph        32        1.2423
cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_nomeas         24        1.2483
cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_nwp_hist       65        1.1157
cl_m-mtgnn_out-48_freq-1h_wind_mtgnn                41        1.2902
cl_m-mtgnn_out-48_freq-1h_wind_mtgnn_nwp            33        1.2696
cl_m-mtgnn_out-48_freq-1h_wind_mtgnn_nwp_hist       27        1.0279
cl_m-tft-bc_out-48_freq-1h_wind_tft_sp_base         24        1.2700
cl_m-tft-bc_out-48_freq-1h_wind_tft_sp_hist         29        1.1111
cl_m-wavenet_out-48_freq-1h_wind_wavenet            47        1.2481
cl_m-wavenet_out-48_freq-1h_wind_wavenet_nwp        45        1.2603
wind_interpol                                      131        1.1562   (Regression-Kriging)
```

Der QRF-Studienname `cl_m-qrf_out-48_freq-1h_wind_qrf_local` kollidiert mit keiner davon.

### 1.9 Umgebung, verifiziert auf `l2`

```
python 3.12.3 · sklearn 1.4.1.post1 · numpy 1.26.4 · pandas 2.3.3 · scipy 1.17.1
optuna 4.3.0 · joblib 1.3.2
quantile_forest ABSENT · statsmodels ABSENT
nproc 256 · load average 10.57 (13:00 UTC)
WEATHER_DB_URL / ECMWF_WIND_SL_URL / OPTUNA_STORAGE: alle in ~/.bashrc vorhanden
```

`RandomForestRegressor` und `LinearRegression`/`Ridge` sind also verfügbar; **es wird keine
neue Abhängigkeit gebraucht und keine installiert.**

---

## 2. Welche Dateien neu entstehen, und warum dort

```
geostatistics/baselines/__init__.py
geostatistics/baselines/dataset.py            # Datenladen, Run-Paare, Design-Matrix
geostatistics/baselines/qrf.py                # QRF-local: fit / predict
geostatistics/baselines/mos.py                # MOS regional / nearest / local
geostatistics/baselines/evaluate_baselines.py # CLI: fit -> predict -> _save
geostatistics/hpo_qrf.py                      # Optuna-Studie fuer QRF-local
configs/baselines/config_wind_qrf_local_fold{1,2,3}.yaml
archiv/baselines_verification/__init__.py
archiv/baselines_verification/verify_baselines.py
```

### 2.1 Begründung: Schwesterskript, nicht Einbau in `evaluate_reference.py`

Die Entscheidung ist **Schwesterskript plus geteiltes Modul**, aus vier Gründen:

1. **`evaluate_reference.py` ist ein publizierter Referenzproduzent.** Seine Ausgaben
   `icon_d2_fold*.csv` / `ecmwf_fold*.csv` tragen die Zahlen, auf denen
   `stdhp_dryrun_results.md` §1/§2/§5.4 beruht, und sie wurden am 2026-08-06 nach zwei
   Fehlerkorrekturen (Commit `bf13436`) neu erzeugt. Jede Änderung dort riskiert genau die
   Zahlen, gegen die die Baselines verglichen werden sollen.
2. **Die Baselines brauchen einen Trainingsdurchlauf, den das Skript nicht kennt.** Seine
   Schleife (`evaluate_nwp_baselines:159-215`) läuft ausschließlich über die
   Auswertungspaare. Ein Fit über 1473 Trainingspaare × 102 Stationen ist eine zweite,
   strukturell andere Phase.
3. **Die HPO braucht die Design-Matrix ohne die Auswertung.** Sie muss importierbar sein,
   also in ein Modul, nicht in ein `main()`.
4. **Aber das Format darf nicht driften.** Deshalb ist es kein „unabhängiges" Skript:
   `evaluate_baselines.py` **importiert** `_station_metrics` und `_save` aus
   `evaluate_reference.py`. Wenn dort etwas geändert wird, ändert es sich hier mit.

`geostatistics/baselines/` als Paket (nicht flache Dateien in `geostatistics/`), weil es
vier zusammengehörige Module sind und `geostatistics/ablations/` denselben Zuschnitt schon
vormacht (`gen_variant_configs.py`, `guard.py`, `__init__.py`).

`hpo_qrf.py` liegt bewusst **flach in `geostatistics/`**, neben `hpo_dcrnn.py`,
`hpo_mtgnn.py`, `hpo_wavenet.py`. Grund: die Namenskonvention der Studie leitet sich
sichtbar aus dem Dateinamen ab, und wer die Kampagne inspiziert, erwartet alle HPO-Treiber
an einer Stelle.

Die Verifikationssuite liegt in `archiv/baselines_verification/`, parallel zu
`archiv/ablations_verification/` (`verify.py`, `verify_review2.py`, `fixture.py`,
`batch_fingerprint.py`) — **nicht** nach `/tmp`, so wie es
`prompt_baselines_implementation.md` §6 verlangt.

### 2.2 Wiederverwendete Funktionen — Datei und Zeile, nichts davon nachbauen

| Zweck | Import |
|---|---|
| Config lesen | `geostatistics.train_stgnn2.load_yaml` — `train_stgnn2.py:76` |
| Messungen laden | `…load_station_measurements` — `train_stgnn2.py:85` |
| Stationsmetadaten (lat/lon/alt) | `…load_station_metadata` — `train_stgnn2.py:130` |
| ICON-D2-Läufe + geodätisch nächster Gitterpunkt | `…load_icond2_ml_runs` — `train_stgnn2.py:327` |
| ECMWF am Gitter und an der Station | `…load_ecmwf_parquet_at_stations_and_grid` — `train_stgnn2.py:733` |
| Env-Guard (K3) | `…require_nwp_elevation_env` — `train_stgnn2.py:1135` |
| Kriging-Imputation | `utils.imputation.load_interpol_imputation:48`, `apply_interpol_imputation:121` |
| KNN-Imputation | `utils.imputation.load_knn_imputation:74`, `apply_knn_imputation:146` |
| Richtungskodierung Messungen | `geostatistics.train_dcrnn.encode_circular_measurements` — `train_dcrnn.py:168` |
| `dir_in_deg`-Kodierung NWP | `geostatistics.train_dcrnn.apply_dir_encoding` — `train_dcrnn.py:199` |
| Topo-Deskriptoren | `geostatistics.stgnn.utils.topo_features.load_topo_station_features_dict:219`, `TOPO_FEATURE_ORDER:27` |
| k nächste Gitterpunkte, geodätisch | `geostatistics.stgnn.utils.spatial.geodesic_knn:88` |
| nächste Station, geodätisch | `geostatistics.stgnn.utils.spatial.pairwise_geodesic_km:66` |
| Fold-Definitionen | `geostatistics.spatial_cv.load_spatial_folds:69`, `station_pool:96`, `build_folds:101`, `fold_hash:55` |
| **Stationsmetriken** | `geostatistics.evaluate_reference._station_metrics` — `evaluate_reference.py:68` |
| **Ausgabe schreiben** | `geostatistics.evaluate_reference._save` — `evaluate_reference.py:111` |
| Imputationsmaske (nur Verifikation) | `geostatistics.stdrun.make_stdhp_figures.build_imputation_mask:268`, `_lookup_imputed:288` |

**Nicht** zu verwenden: `topo_features.load_topo_node_features` (`topo_features.py:242`) —
sie z-scored über **alle** Stationen und ohne Varianzstabilisierung; ihr eigener Docstring
warnt davor für absolute Knotenfeatures.

**Nicht** zu verwenden: `scipy.spatial.cKDTree` für irgendeine Nachbarschaft. Alles
geodätisch über `spatial.py`.

---

## 3. Die Design-Matrix, spaltenweise

### 3.1 Zeileneinheit

Eine Zeile ist ein Tupel **(Station `s`, Run-Paar `p = (r_curr, r_hist, t_run_abs)`,
Lead `h ∈ 1…48)`**.

| Größe | Definition |
|---|---|
| Ziel `y` | `meas_raw[t_run_abs + h − 1, s, target_idx]`, physikalisch in m/s, nach der Imputationskette, `target_idx = measurement_cols.index("wind_speed") = 0` |
| `run_time` | `timestamps[t_run_abs − 1]` |
| `valid_time` | `timestamps[t_run_abs + h − 1]` = `run_time + h h` |
| `horizon` | `h` |

Zeilenzahlen, nachgerechnet:

| Menge | Formel | Zeilen |
|---|---|---|
| Fit QRF-local / MOS-regional, je Fold | 102 × 1473 × 48 | **7 211 808** |
| Fit MOS-nearest, je Trainingsstation und Lead | 1473 | 1473 |
| Fit MOS-local, je Zielstation und Lead | 1473 | 1473 |
| Auswertung, je Fold | 51 × 1460 × 48 | **3 574 080** |

Die 3 574 080 sind exakt `EXPECTED_ROWS` aus `verify_evaluate_reference_fix.py:21` und die
Zeilenzahl der 30 stdhp-Parquets laut `stdhp_dryrun_results.md` §6.

### 3.2 QRF-local — 43 Spalten (Pflichtarm)

Gebaut wird die Matrix **einmal auf der HPO-Obergrenze** `k_i2 = 7`, `k_e2 = 4` (55
Spalten) und pro Konfiguration auf `(k_i2, k_e2)` gesliced — dasselbe Verfahren wie der
„Max-Bound-Trick" in `hpo_mtgnn.py:482-491`. Der **Pflicht-Betriebspunkt** ist
`k_i2 = 4`, `k_e2 = 4`, also 43 Spalten.

| # | Spaltenname | Herkunft (exakter Ausdruck) | Anzahl |
|---|---|---|---|
| 1 | `i2_k{j}_{feat}`, `j = 0…k_i2−1` | `grid_icond2_runs[r_curr, h−1, nearest_i2[s, j], f]`, `nearest_i2 = geodesic_knn(icond2_coords, station_ll, k=7)[1]` (`spatial.py:88`) | `k_i2 × I2` = 4 × 4 = **16** |
| 2 | `e2_k{j}_{feat}`, `j = 0…k_e2−1` | `grid_ecmwf_runs[t_run_abs + h − 1, nearest_e2[s, j], f]`, `nearest_e2` analog auf `ecmwf_coords` | `k_e2 × E2` = 4 × 3 = **12** |
| 3 | `slope, aspect_sin, aspect_cos, tpi5, tpi75, tdi, elev_std, z0, dist_coast` | `load_topo_station_features_dict(topo_path, all_ids, TOPO_FEATURE_ORDER, train_idx=fold_train_idx)[name][s]` | **9** |
| 4 | `lat, lon, alt` | `load_station_metadata(...)` — Rohwerte in Grad bzw. Metern, **keine** sin/cos-Kodierung, **keine** Normierung | **3** |
| 5 | `horizon` | `h` (Integer 1…48) | **1** |
| 6 | `valid_hour_sin, valid_hour_cos` | `sin/cos(2π · valid_time.hour / 24)` | **2** |
| | | **Summe** | **43** |

`I2` und `E2` werden **nach** `apply_dir_encoding` bestimmt, Feature-Namen aus dem
Rückgabewert, damit die Spaltenbenennung dem tatsächlichen Kanalinhalt folgt (siehe die
Index-0-Falle in 1.4).

**Was ausdrücklich NICHT hinein darf** (das ist der Kern des induktiven Anspruchs):

* keine Messung der Zielstation, zu keinem Zeitpunkt, auch nicht abgeleitet
  (kein Klimamittel, keine Bias-Statistik, kein `pers_ref` als Feature)
* keine Messung einer Nachbarstation (das wäre QRF-IDW, offene Frage F1)
* nichts aus dem Historienlauf `r_hist`
* keine Fold-übergreifende Statistik

**Drei Asymmetrien gegenüber den Graphmodellen, bewusst und zu berichten:**

| Was die Modelle zusätzlich sehen | Warum die Baseline es nicht bekommt |
|---|---|
| die NWP-Geometrie je Kante: `dist_norm, sin(bearing), cos(bearing), alt_diff` (`homo_sampler.py:240-242`) | nicht in der Prädiktorenliste des Auftrags. Die Rangordnung der k Punkte steckt implizit in der Spaltenposition, die **Entfernung** nicht. Optionaler Schalter `--nwp-geometry` (Default **aus**), erzeugt `d_i2_k{j}`, `dz_i2_k{j}`, `d_e2_k{j}`, `dz_e2_k{j}` — **nur auf ausdrückliche Freigabe einschalten** |
| die ICON-D2-Vorhersage des Historienlaufs über die letzten 48 h (`homo_sampler.py:305-307`) | nicht in der Prädiktorenliste. Optionaler Schalter `--i2-hist` (Default **aus**) |
| die Messungen der Nachbarstationen (Variante A) | ausdrücklich verboten ohne Freigabe (D2). QRF-local ist das Gegenstück zu Variante **C**, nicht zu A |

Diese drei Punkte benachteiligen die Baseline. Das ist die für uns unbequeme Richtung und
damit die richtige Voreinstellung, aber es **muss** im Methodikteil stehen, sonst ist der
Vergleich zu optimistisch für die Graphen.

### 3.3 Skalierung: für QRF gegenstandslos — festgehalten

Ein Entscheidungsbaum spaltet an Schwellen einer einzelnen Spalte. Jede **strikt monotone**
Transformation dieser Spalte bildet Schwellen auf Schwellen ab und lässt die
Baumstruktur, damit jede Vorhersage, unverändert. Folglich ist für QRF-local **irrelevant**:

* der z-Score der Topo-Spalten und ob er auf 102 oder 153 Stationen gefittet wird
  (numerisch belegt: spaltenweise Stationsrangfolge in beiden Fällen identisch, Abschnitt 1.4)
* die varianzstabilisierenden `log`/`log1p`/`sign·log1p`-Transformationen
  (`topo_features.py:53-61`)
* `sin/cos` von lat/lon gegen rohe Grad (strikt monoton über 47…55 °N, 6…15 °E)
* die Normierung von `alt` auf die Trainingsstationen (`homo_sampler.py:259-262`)
* jeder `StandardScaler` auf NWP-Kanälen

**Es wird deshalb keine Skalierung eingebaut.** Die ganze N1-Diskussion (Geo-Statik-Scaler)
betrifft QRF nicht. Verifikationstest V7 belegt das numerisch statt es zu behaupten.

**Nicht** irrelevant und daher explizit zu pinnen: `icond2_feature_mode` /
`ecmwf_feature_mode`. `dir_in_deg` ist eine **Reparametrisierung über Spalten hinweg**
(`(u,v) → (speed, sin_dir, cos_dir)`), keine monotone Spaltentransformation. Pflichtwert
`dir_in_deg` in beiden Fällen — das ist der harmonisierte Standard-Hyperparameter des
Trockenlaufs (`configs/mtgnn/stdhp/config_wind_mtgnn_nwp_stdhp_fold1.yaml:43-44`,
`configs/dcrnn/stdhp/config_wind_dcrnn_stdhp_fold1.yaml:327-328`).

### 3.4 Ein Wald für alle Lead-Zeiten — begründet

`prompt_baselines_implementation.md` §4.1 lässt die Wahl. Entschieden: **ein einziger
Random Forest mit `horizon` als Merkmal**, nicht 48 Wälder.

Gründe:

1. **Kosten.** 48 Wälder × 3 Folds × HPO-Trials ist um den Faktor 48 teurer. Bei den in
   5.3 gemessenen Zeiten wäre die HPO nicht durchführbar.
2. **Fairness in die richtige Richtung.** Die Graphmodelle sagen alle 48 Leads mit **einem**
   Parametersatz vorher. 48 unabhängig gefittete Wälder gäben der Baseline eine
   strukturelle Freiheit, die kein Vergleichspartner hat — der Vergleich wäre dann zugunsten
   der Baseline verzerrt, was ein Referee zu Recht angreifen würde. Umgekehrt: sollte die
   Baseline trotz dieser Einschränkung gewinnen, ist das Ergebnis stärker.
3. **Datenmenge.** 7.21 M Zeilen sind genug, damit der Wald auf `horizon` selbst splittet.

**Zu berichten:** Abweichung von `taillardat2016calibrated`, wo je Lead-Zeit gefittet wird.
Ein optionaler Arm `--per-lead` ist zu implementieren, aber **nicht** zu fahren.

### 3.5 MOS — Modellgleichung

Je Lead `h` eine eigene OLS-Gleichung („lead-time-spezifische Koeffizienten"):

```
y = β0(h)
  + β1(h) · ws_i2(h)                     ICON-D2 wind_speed_10m am naechsten Gitterpunkt
  + β2(h) · ws_e2(h)                     ECMWF   wind_speed_10m am naechsten Gitterpunkt   [nur --nwp-sources both]
  + β3(h) · sin(2π·hour/24) + β4(h) · cos(2π·hour/24)
  + β5(h) · sin(4π·hour/24) + β6(h) · cos(4π·hour/24)
```

`hour` = `valid_time.hour`. Die zweite Harmonische ist drin, weil der Tagesgang der
bodennahen Windgeschwindigkeit über Land ausgeprägt asymmetrisch ist; sie kostet zwei
Parameter gegen ≥1473 Zeilen.

`ws_i2(h) = grid_icond2_runs[r_curr, h−1, nearest_i2[s,0], ws_idx]`,
`ws_e2(h) = grid_ecmwf_runs[t_run_abs+h−1, nearest_e2[s,0], ws_idx_e2]`.
Beide Feature-Indizes **namentlich** bestimmen (1.4).

**Deviation, die begründet werden muss.** `prompt_baselines_implementation.md` §4.2 nennt
nur „NWP-Windgeschwindigkeit am nächsten Gitterpunkt", Singular. Ein MOS, das nur ICON-D2
sieht, ist gegen Modelle, die zwei Quellen sehen, ein Strohmann — und
`stdhp_dryrun_results.md` §5.4 zeigt, dass ECMWF an 74 von 153 Stationen besser ist als
ICON-D2. Deshalb: **`--nwp-sources` mit den Werten `icond2` und `both`, und beide Läufe
sind Pflicht.** Das kostet Sekunden und beantwortet die Frage, statt sie offen zu lassen.
Berichtet werden beide; die Haupttabelle nennt `both`, `icond2` steht als klassische
Einquellen-Variante daneben.

**Drei Varianten, exakt:**

| Variante | Fit-Menge | Anwendung | Stem |
|---|---|---|---|
| **MOS-regional** | alle Zeilen der **102** Fold-Trainingsstationen, Trainingsfenster, gepoolt; ein Koeffizientensatz je `h` | alle 51 Zielstationen | `mos_regional[_2nwp]` |
| **MOS-nearest** | je Trainingsstation `t` und `h` getrennt (1473 Zeilen) → 102 × 48 Koeffizientensätze | Zielstation `v` nimmt die Koeffizienten von `argmin_t pairwise_geodesic_km(v, t)` über die **102 Trainingsstationen** des Folds (`spatial.py:66`) | `mos_nearest[_2nwp]` |
| **MOS-local** | je **Zielstation** `v` und `h` auf **`v`s eigenen** 1473 Trainingsfenster-Zeilen | dieselbe Station `v`, Val-Fenster | `mos_local[_2nwp]` |

**MOS-local ist die transduktive Obergrenze und der einzige Arm, der die Historie der
Zielstation benutzt.** Das ist Absicht: er bemisst, was die Induktion kostet. Er ist
deshalb im Leckagetest ausdrücklich ausgenommen und in jeder Tabelle als *transduktiv* zu
kennzeichnen, genau wie GRID+HIST und TFT hist.

**Regularisierung: keine.** Reines OLS (`numpy.linalg.lstsq` oder
`sklearn.linear_model.LinearRegression`). Begründung: 150 246 Zeilen (regional) bzw. 1473
Zeilen (nearest/local) je Lead gegen 5 bzw. 7 Parameter; Ridge hat hier nichts zu tun, und
ohne Strafterm ist die Aussage „Skalierung ändert die Vorhersage nicht" exakt wahr statt
nur ungefähr. **Falls** der Nutzer Regularisierung verlangt: genau ein `alpha`, bestimmt
über `GroupKFold(n_splits=5, groups=station_id)` **auf den Trainingsstationen**, niemals
über Optuna, und dann mit `StandardScaler` derselben Konvention wie der Modellpfad
(Fit auf `[:val_start]` × Fold-Trainingsstationen). Nicht Teil des Pflichtumfangs.

**Rangdefekt-Absicherung:** wenn `numpy.linalg.matrix_rank(X) < X.shape[1]` für eine
`(Station, Lead)`-Kombination, ist das zu **loggen und die Zeile als NaN-Vorhersage zu
markieren**, nicht stillschweigend per Pseudoinverse zu füllen. Nach 1.5 sollte das nie
vorkommen (keine Rest-NaN); wenn es vorkommt, ist etwas anderes falsch.

---

## 4. Die Fit-Menge, exakt

### 4.1 Stationen

| Arm | Fit-Stationen | Auswertungsstationen |
|---|---|---|
| QRF-local | `fold.files` (102) | `fold.val_files` (51) |
| MOS-regional | `fold.files` (102), gepoolt | `fold.val_files` (51) |
| MOS-nearest | `fold.files` (102), je Station | `fold.val_files` (51) |
| MOS-local | **`fold.val_files` (51), je Station auf sich selbst** | `fold.val_files` (51) |

### 4.2 Zeitfenster

* **Fit:** Run-Paare mit `timestamps[t_run_abs − 1] < val_start` (2024-08-01) → **1473**.
  Die Grenze wird über die **Laufzeit** gezogen, nicht über `t_run_abs` — dieselbe Regel wie
  `hpo_mtgnn.py::_fold_pairs:799-805`.
* **Auswertung, Standard:** `val_start ≤ Laufzeit < test_start` → **1460**.
* **Auswertung, Testfenster:** `test_start ≤ Laufzeit < test_end`, Stationsmengen
  **unverändert** (51 Zielstationen). Dafür ist ein eigener Schalter nötig, siehe 6.2 und
  Befund 8.2.

### 4.3 Filter — alle, auch der unnötige

Der Fit läuft über **genau** die Run-Paare, die die Schleife aus 1.3 liefert, mit dem
Fenster `< val_start`. Das schließt den `r_hist`-Filter ein (`diffs_s[r_hist] > 3*3600`),
obwohl keine Baseline `r_hist` benutzt.

**Warum das so sein muss:** würde die Baseline auch die Paare ohne Historienlauf mitnehmen,
sähe sie mehr Trainingsdaten als jedes Graphmodell. Eine Baseline auf einer größeren
Stichprobe ist keine Baseline, sondern ein anderer Versuch. Der Filter ist explizit zu
implementieren und die Zahl der dadurch verworfenen Paare zu loggen.

Zusätzlich zu übernehmen: der **ICON-D2-Gitter-NaN-Filter** aus `hpo_mtgnn.py:666-682`.
Nachgerechnet verwirft er in Train und Val **0 von 2933** Paaren, ist dort also
folgenlos; im Testfenster wird er tragend (8.1). Er ist einzubauen **und seine Trefferzahl
zu loggen**, damit eine künftige Divergenz sichtbar wird, statt sich als stille
Stichprobendifferenz zu verstecken.

**Imputierte Zielstunden bleiben im Fit.** Entschieden in `stdhp_dryrun_results.md` §5.5.
Ausgeschlossen werden sie erst in der Auswertung, post hoc über die Maske aus 1.5.

---

## 5. HPO, Retrain, Evaluation für QRF-local

### 5.1 Pipeline

```
1. hpo_qrf.py --config configs/baselines/config_wind_qrf_local_fold1.yaml
   -> Studie cl_m-qrf_out-48_freq-1h_wind_qrf_local
   -> Objective: Mittel ueber 3 raeumliche Folds des GEPOOLTEN, unskalierten
      Val-RMSE in m/s  (identisch zu hpo_mtgnn.py, siehe 1.8)

2. evaluate_baselines.py --arm qrf_local --hpo-study auto -c <fold-config> --fold-idx N
   -> laedt study.best_params, refittet je Fold, wertet aus, schreibt
      data/test_results/qrf_local_fold{N}.csv + data/raw_preds/qrf_local_fold{N}_raw.parquet

3. evaluate_baselines.py --arm mos_regional|mos_nearest|mos_local  (ohne HPO)
```

Schritt 2 ersetzt „Retrain" und „Evaluation" in einem Aufruf, weil der Fit hier Sekunden bis
Minuten dauert und es keinen Checkpoint gibt, den man zwischenlagern müsste. Die Trennung im
Modellpfad existiert nur wegen der GPU-Kosten.

### 5.2 Suchraum — vier Parameter

```yaml
qrf:
  hpo:
    trials: 60
    pruner: median
    pruner_n_startup_trials: 20
    pruner_n_warmup_steps: 1
    n_folds: 3
    cv_mode: spatial
    spatial_folds: 'configs/spatial_folds.yaml'
    params:
      n_estimators:      {type: int,   low: 100,  high: 400, step: 50}
      min_samples_leaf:  {type: float, low: 1.0e-6, high: 1.0e-3, log: true}
      max_features:      {type: float, low: 0.15, high: 1.0}
      max_depth:         {type: int,   low: 8,    high: 40}
```

**Warum `min_samples_leaf` als Bruchteil und nicht als Anzahl:** `sklearn` interpretiert
einen `float` in `(0, 0.5]` als `ceil(min_samples_leaf · n_samples)`. Damit ist der
getunte Wert **unabhängig von der Zeilenzahl** und überträgt sich von einer
Trainings-Teilstichprobe auf den vollen Satz. Bei 7.21 M Zeilen entspricht der Bereich
`1e-6 … 1e-3` etwa **8 … 7212** Blattmindestgrößen, bei 1 M Zeilen `1 … 1000`. Ohne diesen
Trick wären HPO-Optimum und Retrain-Konfiguration nicht dieselbe Größe.

Feste, **nicht** gesuchte Werte, in der Config gepinnt und im Trial-`user_attr` protokolliert:
`k_i2 = 4`, `k_e2 = 4`, `icond2_feature_mode = dir_in_deg`, `ecmwf_feature_mode = dir_in_deg`,
`random_state = 20260810`, `n_jobs` aus CLI, `bootstrap = True`, `criterion = "squared_error"`.
Das hält den Suchraum bei den vom Auftrag erlaubten „drei bis fünf" Parametern.

### 5.3 Rechenbudget — gemessen, nicht geschätzt

Benchmark auf `l2`, `RandomForestRegressor`, 43 Merkmale, `max_features = 0.33`,
`nice -n 19`, `CUDA_VISIBLE_DEVICES=""`. **Achtung: zwei verschiedene
`min_samples_leaf`-Werte** — die Blattgröße bestimmt die Knotenzahl und damit den Speicher,
nicht die Fitzeit:

| Zeilen | `min_samples_leaf` | `n_jobs` | Bäume | Fit | s/Baum | Knoten/Baum | maxRSS |
|---|---|---|---|---|---|---|---|
| 200 000 | 5 | 16 | 10 | 5.4 s | 0.54 | 40 476 | — |
| 500 000 | 5 | 16 | 10 | 17.2 s | 1.72 | 101 375 | — |
| 1 000 000 | 5 | 16 | 10 | 47.5 s | 4.75 | 202 733 | — |
| 1 000 000 | 20 | 32 | 10 | 35.0 s | 3.50 | 48 795 | — |
| 2 000 000 | 20 | 32 | 10 | 71.9 s | 7.19 | 97 465 | 1.2 GB |
| **7 211 808** | **20** | 32 | 4 | **329.6 s** | **82.4** | 351 674 | 4.2 GB |

Zwei Ablesungen: die Knotenzahl je Baum ist **≈ n / min_samples_leaf** (bei leaf = 5 rund
`n/5`, bei leaf = 20 rund `n/20`), und die Fitzeit skaliert von 1 M auf 2 M noch
näherungsweise linear (Exponent 1.04), von 2 M auf 7.21 M dagegen mit Exponent **1.91** —
dort schlägt der Speicherdruck durch. Das ist der eigentliche Grund, warum eine HPO auf dem
vollen Satz nicht geht.

Vorhersage ist billig: 200 000 Zeilen × 10 Bäume in 0.11 s, also 3.57 M Val-Zeilen × 300
Bäume ≈ **60 s**.

Ableitungen:

* **Voller Satz, 300 Bäume, `n_jobs = 32`:** 300 × 82.4 s ≈ **6.9 h je Fold**, 20.6 h je
  Trial. Für eine HPO **nicht machbar**.
* **1 M Zeilen, 400 Bäume, `n_jobs = 64`:** ≈ 400 × 1.75 s ≈ 700 s je Fold ≈ **35 min je
  Trial**. 60 Trials ohne Pruning ≈ 35 h; mit `MedianPruner` (ab Trial 21, Abbruch nach
  Fold 0 oder 1) realistisch **20–25 h** auf CPU.
* Speicher: 300 × 48 795 Knoten ≈ 14.6 M Knoten ≈ **0.9 GB** bei 1 M Zeilen;
  300 × 351 674 ≈ 105 M Knoten ≈ **6.8 GB** beim vollen Satz.
* Design-Matrix: 7.21 M × 55 `float32` = **1.59 GB** (Fit, Max-Bound), 3.57 M × 55 = 0.79 GB
  (Auswertung).

**Verbindliches Vorgehen (Schritt 0 der Implementierung), damit die Teilstichprobe eine
Messung und keine Annahme ist:**

Fitte QRF-local auf Fold 0 mit festen Standardparametern (`n_estimators = 200`,
`min_samples_leaf = 2e-5`, `max_features = 0.33`, `max_depth = 30`) bei
`n ∈ {0.5, 1, 2, 4, 7.21} M` Zeilen und protokolliere gepooltes Val-RMSE und Laufzeit.

* Ist die RMSE-Kurve ab `N*` flach (Δ < 0.005 m/s), wird `N*` **überall** benutzt, in HPO
  **und** im Retrain. Die Kurve ist die Begründung und gehört ins Verifikationsdokument.
* Ist sie nicht flach, läuft die HPO auf 1 M Zeilen und der Retrain auf dem vollen Satz;
  wegen des Bruchteil-`min_samples_leaf` (5.2) überträgt sich das Optimum, und die
  Abweichung ist zu berichten.

Die Teilstichprobe wird **reproduzierbar** gezogen:
`rng = np.random.default_rng(20260810)`, `idx = rng.choice(n_rows, N, replace=False)`,
derselbe Seed in jedem Trial und jedem Fold, damit Trials vergleichbar sind. Die
tatsächliche Zeilenzahl und der Seed gehen als `trial.set_user_attr` mit.

### 5.4 Provenienz und Kampagnenschutz

`hpo_qrf.py` setzt dieselben `user_attrs` wie `hpo_mtgnn.py:880-884`:
`host`, `commit`, `fold_hash`, plus `n_fit_rows`, `subsample_seed`, `k_i2`, `k_e2`,
`icond2_feature_mode`, `ecmwf_feature_mode`.

Harte Auflagen:
* `CUDA_VISIBLE_DEVICES=""` und `nice -n 19` in jedem Aufruf. Kein Torch-Import.
* Nur die **eigene** Studie wird geschrieben. Jeder Zugriff auf die 13 Modellstudien ist
  lesend.
* `n_jobs` per CLI, Default **32**, harte Obergrenze **64** von 256 Kernen, damit die
  Datenlader der 18 GPU-Worker nicht ausgehungert werden.
* Env-Guard `require_nwp_elevation_env` (`train_stgnn2.py:1135`) am Anfang jedes Skripts —
  ein Lauf ohne `WEATHER_DB_URL` muss **hart abbrechen**, nicht warnen (Befund K3).
  QRF braucht die NWP-Knotenhöhen nicht, aber der Guard verhindert, dass ein
  halbkonfigurierter Lauf einen GNNCache-Eintrag mit Höhen = 0 schreibt, den die
  Modell-Worker erben.

---

## 6. CLI-Schnitt

### 6.1 `geostatistics/baselines/evaluate_baselines.py`

An `evaluate_reference.py:235-247` angelehnt, gleiche Flag-Namen wo es dieselbe Sache ist.

```
python geostatistics/baselines/evaluate_baselines.py
    -c, --config PATH            (Pflicht)  Fold-Config, wie evaluate_reference.py
    --fold-idx  {0,1,2}          (Pflicht)  Ausgabe-Index; Config fold1 -> 0
    --arm       NAME             (Pflicht)  qrf_local | mos_regional | mos_nearest | mos_local | all
    --eval-window {val,test}     Default val. Entkoppelt das ZEITfenster von den
                                 Stationsmengen (siehe 6.2)
    --test-mode                  wie evaluate_reference.py: train = files+val_files,
                                 Ziel = test_files. Schliesst --eval-window aus.
    --hpo-study {auto,none}      Default none. auto laedt best_params aus der QRF-Studie
    --nwp-sources {icond2,both}  Default both. Nur fuer die MOS-Arme
    --n-fit-rows INT             0 = alle. Default 0
    --subsample-seed INT         Default 20260810
    --n-jobs INT                 Default 32
    --ecmwf-features CSV         wie evaluate_reference.py:245
    --nwp-geometry               OPTIONAL, Default aus, nur mit Freigabe
    --i2-hist                    OPTIONAL, Default aus, nur mit Freigabe
    --per-lead                   OPTIONAL, Default aus, nur mit Freigabe
    --out-prefix STR             Default "" ; erlaubt z.B. stdhp-artige Praefixe
    --dry-run                    laedt, baut Matrizen, loggt alle Zahlen, fittet nicht
```

Ausgabe-Stems, `sfx = "_test"` wenn `--test-mode`, `"_tw"` wenn `--eval-window test`:

```
{out_prefix}{arm}{_2nwp}{sfx}_fold{N}.csv
{out_prefix}{arm}{_2nwp}{sfx}_fold{N}_raw.parquet
```

Beispiele: `qrf_local_fold0.csv`, `mos_regional_2nwp_fold1.csv`, `mos_local_tw_fold2.csv`.

Das Skript **muss** vor dem Fit eine Bannerzeile loggen, analog zu
`evaluate_reference.py:449-456` und `ablations/guard.py::check_ablation_flags`:

```
=== BASELINE fold=1 arm=qrf_local  fit_stations=102  eval_stations=51
    fit_pairs=1473  eval_pairs=1460  fit_rows=7211808 (subsample=0)  eval_rows=3574080
    k_i2=4 k_e2=4 i2_mode=dir_in_deg e2_mode=dir_in_deg  cols=43
    dropped: r_hist=0  grid_nan=0  meas_nan=0
    fold_hash=<md5:12>  commit=<sha>  seed=20260810 ===
```

und **hart abbrechen**, wenn `fit_pairs != 1473` oder `eval_pairs != 1460` im Standardmodus,
oder wenn die Schnittmenge `fit_stations ∩ eval_stations` nicht leer ist (außer bei
`--arm mos_local`).

### 6.2 `--eval-window` ist nötig, weil es kein Äquivalent gibt

`evaluate_reference.py:326-341` koppelt Zeitfenster und Stationsmenge:

```python
boundary = test_start if (args.test_mode or not val_start) else val_start
```

`--test-mode` tauscht **gleichzeitig** die Stationen auf `test_files` (Zeilen 275-277).
Es gibt also **keinen** Modus, der die **51 Fold-Zielstationen** auf dem **Testfenster**
auswertet — und genau das braucht das Paper für die induktive Aussage und für das
Retraining-Experiment. Die Configs unter `configs/*/test/` umgehen das, indem sie
`val_start` weglassen; sie tragen aber die **alten** Stationslisten (nachgerechnet:
103 `files`, 50 `val_files`, **nicht** die Fold-Mengen). Siehe Befund 8.2.

`--eval-window test` setzt daher direkt `boundary = test_start`,
`eval_cutoff = test_end`, Stationsmengen unverändert. Das ist eine reine Ergänzung im neuen
Skript und ändert nichts an `evaluate_reference.py`.

### 6.3 `geostatistics/hpo_qrf.py`

```
python geostatistics/hpo_qrf.py
    --config PATH        (Pflicht)  configs/baselines/config_wind_qrf_local_fold1.yaml
    --suffix STR         Log-Suffix, wie hpo_mtgnn.py:306
    --n-jobs INT         Default 32
    --n-fit-rows INT     Default 1000000, 0 = alle
    --subsample-seed INT Default 20260810
    --scaling-curve      Schritt 0 aus 5.3: fittet die n-Kurve und beendet
    --preprocess-only    wie hpo_mtgnn.py:308
```

Kein `--gpu`. Kein `--pin-inert`. Studienname strikt aus dem Config-Stem
(`re.sub(r'_fold\d+$','',stem)`), damit die drei Fold-Configs auf **eine** Studie zeigen —
dieselbe Mechanik wie `hpo_mtgnn.py:344`.

---

## 7. Verifikationssuite

Ablage: `archiv/baselines_verification/verify_baselines.py`, aufrufbar als
`CUDA_VISIBLE_DEVICES="" nice -n 19 python -m archiv.baselines_verification.verify_baselines`.
Jeder Test gibt **die Zahl** aus, nicht nur bestanden/durchgefallen. Die sieben Tests aus
`prompt_baselines_implementation.md` §6 sind die Untergrenze; V7–V11 kommen hinzu.

| # | Test | Erwartung |
|---|---|---|
| **V1** | **Leckage.** Fit von QRF-local, MOS-regional, MOS-nearest je Fold, einmal mit den 51 Zielstationen im Trainingssatz und einmal ohne. | Bei korrekter Maskierung **bitgleiche** Ausgaben: MOS-Koeffizienten identisch bis auf 0.0; QRF mit festem `random_state` **identische** `tree_.threshold`/`feature`-Arrays. MOS-local ist **ausgenommen** und muss im Testbericht ausdrücklich als Ausnahme geführt werden. |
| **V2** | **Zeitausrichtung.** RMSE(Vorhersage, Messung) bei Versatz −2 … +2 Stunden auf den tatsächlich gebauten Paaren. | Minimum **muss** bei 0 liegen, für jeden Arm. Referenz für rohes ICON-D2: **1.4958** bei 0, **1.5040** bei −1, **1.5550** bei +1 (`study_overview.md` §3). Die Stichprobe, über die diese drei Zahlen gelten, ist in der Doku nicht benannt — der Test muss sie **selbst benennen** und die drei Werte darauf reproduzieren, bevor er sie als Anker benutzt (siehe 9.2). |
| **V3** | **Fold-Konsistenz.** Stations-IDs jeder erzeugten CSV gegen `configs/spatial_folds.yaml`. | Je Fold **51** IDs, exakt `spatial_fold{N+1}.val_files`, Schnittmenge mit `files` = **0**. Für MOS-local zusätzlich: die Menge der Stationen mit gefitteten Koeffizienten ist **identisch** mit der Menge der ausgewerteten Stationen. |
| **V4** | **Formatgleichheit.** Spalten und Stationsmenge der erzeugten `data/test_results/*.csv` gegen `l1:data/test_results/icon_d2_fold{N}.csv`. | Spalten exakt `station_id, mae, rmse, r2, skill, skill_nwp, n_samples`; 51 Zeilen; `n_samples == 70080` in **jeder** Zeile (ungefilterter Lauf). |
| **V5** | **Zeilenschlüssel-Identität.** Für jeden Fold: `set((station_id, valid_time, horizon))` des Baseline-Parquets gegen das eines stdhp-Modell-Parquets desselben Folds auf `l1`. | **Gleich**, Größe **3 574 080**, keine Duplikate, `horizon ∈ 1…48`. Ohne das bricht `make_stdhp_figures.py::_verify_uniform_filtering:373` die spätere Auswertung ab. |
| **V6** | **Referenzidentität.** `nwp_ref` und `pers_ref` des Baseline-Parquets gegen die gleichnamigen Spalten des Modell-Parquets, gejoint auf `(station_id, valid_time)`. | `max|Δ| ≤ 2e-6` (Parquet-Rundung, derselbe Toleranzwert, den `stdhp_dryrun_results.md` §5.6 für den TFT-Abgleich über 18 079 296 Zeilen belegt). Zusätzlich `gt` identisch. |
| **V7** | **Skalierungsinvarianz (Beleg statt Behauptung).** QRF-local zweimal fitten: einmal mit Topo-z-Score auf `train_idx`, einmal auf allen 153 Stationen, sonst identisch. | **Identische** Vorhersagen, `max|Δpred| = 0.0`. Der numerische Vorbefund liegt vor: spaltenweise Stationsrangfolge in beiden Fällen identisch, `max|Δ|` der Spalten selbst = **0.5978** (Fold 1). Damit ist 3.3 belegt und N1 für QRF nachweisbar gegenstandslos. |
| **V8** | **HPO-Objective.** Der von `hpo_qrf.py` je Fold zurückgegebene Wert gegen das aus dem Roh-Parquet **gepoolt** nachgerechnete Val-RMSE. | `\|Δ\| ≤ 1e-6`. Zusätzlich der Kontrast zum Mittel der Stations-RMSEs, das um **rund 0.12 m/s** kleiner ausfällt (ICON-D2: 1.325 per Station gegen 1.450 gepoolt) — die Zahl gehört in den Bericht, damit niemand die beiden verwechselt. |
| **V9** | **MOS-Varianten sind verschieden.** Paarweise Koeffizientenvergleich regional / nearest / local, je Lead. | `max\|Δβ\| > 0` in jedem Paar; zusätzlich: die Zuordnung Ziel → nächste Trainingsstation ist **geodätisch**. Gegenprobe: Anteil der 51 Ziele, bei denen euklidisch-in-Grad eine **andere** Station gewählt würde — die Zahl ist zu berichten (bei 47–55 °N ist ein Längengrad nur 0.59-mal so lang wie ein Breitengrad). |
| **V10** | **Plausibilität.** Jeder Arm gegen die bekannten Referenzen, per Station gemittelt über die drei Folds, ohne imputierte Zielstunden. | Pflicht: besser als Persistenz (**2.240**) und mindestens so gut wie rohes ICON-D2 (**1.304**). Erwartungsband: QRF-local sollte in der Nähe von DCRNN GRID-NOGRAPH (**1.138**) und TFT base (**1.186**) landen; MOS-regional darüber, MOS-local deutlich darunter. Liegt ein Arm außerhalb, ist das ein Befund und keine Fußnote. |
| **V11** | **Kampagne unangetastet.** Prozessliste und Trial-Zustände der 13 Modellstudien vor und nach dem Lauf. | Gleiche Worker-PIDs; `COMPLETE`/`FAIL`/`PRUNED`-Zahlen je Studie ohne Sprung, der nicht durch die laufende Kampagne erklärt ist. Referenzstand vom 2026-08-10 12:56 in 1.8. `t.state.name` benutzen, **nicht** `str(t.state)` — unter Optuna 4.3 / Python 3.12 liefert `str()` die Zahl `'1'` statt `COMPLETE` (`study_overview.md` §9). |

Zusätzlich als reine Protokollpflicht, kein Test: die in 5.3 verlangte
**Skalierungskurve** mit fünf Punkten und die Trefferzahlen der drei Filter aus 4.3.

---

## 8. Vergleichbarkeitsrisiken

### 8.1 BLOCKIEREND — das Zwölf-Monats-Testfenster existiert datenseitig nicht

Vollständiger Scan aller ICON-D2-ML-Parquets der 153 Poolstationen (153 Stationen × 4
Laufstunden × 22 Gitterdateien = 13 464 Dateien, `max(starttime)` je Datei):

| Laufstunde | letzter Lauf | betroffene Stationen |
|---|---|---|
| 06 | **2026-06-11 06:00** bei 81 Stationen, **2026-04-29 06:00** bei **72** Stationen | gemischt |
| 09 | 2026-06-11 09:00 | alle 153 |
| 12 | 2026-06-11 12:00 | alle 153 |
| **15** | **2026-04-29 15:00** | **alle 153** |

Weitere Ränder: ECMWF-SL-Parquets enden **2026-06-18 21:00** (759 Dateien, stichprobenweise
über 8 Dateien identisch, **keine** NaN ab 2025-08 — die Config-Begründung „NaN in
ECMWF-Parquets ab 2026-05-01" trifft auf den aktuellen Stand nicht mehr zu);
Rohmessungen enden **2026-07-14 23:50**.

**Folgen:**

1. Nach **2026-04-29** gibt es die Laufstunde 15 nirgends mehr. Das Testfenster wechselt
   dort von 4 auf 3 Läufe pro Tag — die Stichprobendichte ändert sich mitten in der
   Auswertung.
2. Nach 2026-04-29 fehlen 72 von 153 Stationen die 06-Läufe. Deren Gitterknoten bleiben in
   `grid_icond2_runs` **NaN** (`load_icond2_ml_runs` initialisiert mit NaN und füllt nur
   vorhandene `(stem, run_hour)`-Kombinationen, `train_stgnn2.py:438-447`). Weil
   `evaluate_reference.py` und alle `get_test_results_*.py` den Gitter-NaN-Filter **nicht**
   haben (1.3), erzeugen sie dort NaN-Vorhersagen, `_station_metrics` maskiert sie pro
   Station weg, und `n_samples` unterscheidet sich zwischen Stationen. Dann bricht
   `make_stdhp_figures.py::_verify_uniform_filtering` die Auswertung ab — oder, schlimmer,
   die paarweisen Vergleiche laufen still über verschiedene Stichproben.
3. Der Retrain würde ebenfalls auf NaN-Läufen trainieren: `train_dcrnn.py` und
   `train_mtgnn.py` haben den Filter auch nicht.

**Das letzte Datum, für das alle vier Laufstunden mit vollständigem Gitter vorliegen, ist
2026-04-29.** Ein Testfenster `2025-08-01 … 2026-07-31` ist mit den Daten auf Platte nicht
sauber auswertbar; sauber möglich ist **2025-08-01 … 2026-04-29 (neun Monate)**.

Grober Rahmen der Paarzahl: bei 4 Läufen/Tag sind es 272 Tage × 4 = **1088** Läufe
theoretisch, gegen 1456 für zwölf Monate. Eine untere Schranke aus meiner Rekonstruktion
(die die Laufverfügbarkeit konservativ unterschätzt — sie reproduziert 1102 statt 1473
Trainingspaare, also ~75 %) liegt bei 850 Paaren bis 2026-04-30. Die genaue Zahl ist
**nur** durch einen echten Lauf mit gesetztem `test_end` zu bekommen.

**Das ist kein Baseline-Problem.** Es betrifft die Graphmodelle genauso und muss vom Nutzer
entschieden werden, bevor irgendetwas auf dem Testfenster rechnet. Empfehlung: `test_end`
auf **2026-04-29** setzen, das Fenster im Paper als neun Monate ausweisen, und das
Retraining-Szenario auf `2025-08-01 → 2025-12-01` (Fit-Verlängerung) und
`2025-12-01 → 2026-04-01` (Bewertung, beide Modelle) legen — das passt vollständig in den
verfügbaren Bereich.

### 8.2 HOCH — es gibt keine Config, die die 51 Zielstationen auf dem Testfenster auswertet

`configs/mtgnn/test/config_wind_mtgnn_nwp_fold1.yaml` hat **kein** `val_start` (damit greift
`boundary = test_start`), trägt aber die **alten** Stationslisten: nachgerechnet
**103** `files` und **50** `val_files`, und keine der beiden Mengen stimmt mit
`spatial_fold1` überein. Dieselbe Diskrepanz ist der Grund, aus dem am 2026-08-06 die alten
`icon_d2_fold*.csv` archiviert wurden (README dort: „enthalten 50 Stationen und stimmen
weder mit `val_files` (51) noch mit `test_files` (50) überein … Überschneidung 20 bis 45
Prozent").

Für die Baselines ist das über `--eval-window test` gelöst (6.2). Für die **Graphmodelle**
ist es offen: ihre Testfensterzahlen brauchen entweder neue Configs (Fold-Stationslisten,
kein `val_start`, `test_start`/`test_end` gesetzt) oder denselben Schalter in
`get_test_results_*.py`. **Melden, nicht reparieren.**

### 8.3 MITTEL — Zeitfensterangaben widersprechen sich in vier Dokumenten und drei Config-Familien

`test_end` steht auf 2025-10-31 (dcrnn-Basis, tft_bc), 2025-11-30 (test-Configs),
2026-03-31 (Kampagnen- und stdhp-Configs); `study_overview.md` §6 sagt 2025-10-31,
`stdhp_dryrun_results.md` §1 sagt 2026-03-31, der Auftrag sagt 2026-07-31. **Keine Config
trägt den Auftragswert.** Solange das nicht entschieden ist, kann kein Testfensterlauf
starten — auch keiner der Baselines.

Das Val-Fenster ist dagegen unstrittig und in allen Configs identisch. **Empfehlung: die
Baselines zuerst vollständig auf dem Val-Fenster fahren** (1460 Paare, 51 Stationen,
direkt gegen die stdhp-Zahlen und gegen `icon_d2_fold*.csv` stellbar) und das Testfenster
erst nach der Entscheidung aus 8.1.

### 8.4 MITTEL — HPO- und Evaluationspfad filtern nicht dieselben Run-Paare

Der ICON-D2-Gitter-NaN-Filter existiert nur in den drei HPO-Skripten (1.3). Derzeit
folgenlos (**0 von 3303 Läufen** betroffen, nachgerechnet), ab 2026-04-29 tragend.
Für die Baselines ist die Auflösung eindeutig: Filter einbauen **und** Trefferzahl loggen.
Für den Modellpfad ist es ein Befund (10.3).

### 8.5 MITTEL — Objective ist gepoolt, die Tabellen sind per Station

Das HPO-Objective ist ein **gepooltes** RMSE (1.8), die berichteten Tabellen mitteln **per
Station** (1.6). Die beiden unterscheiden sich bei ICON-D2 um **0.12 m/s** (1.450 gegen
1.325) und drehen laut `stdhp_dryrun_results.md` §1 beim Skill sogar eine Reihenfolge um.
Die QRF-Studie muss das **gepoolte** RMSE optimieren, weil die Modellstudien es tun; die
Ergebnistabelle nennt das **per-Station**-Mittel, weil die Modelltabellen es tun. Beides ist
richtig, aber die Zahlen dürfen nicht verwechselt werden. Verifikationstest V8 erzwingt,
dass beide Werte im Bericht stehen.

### 8.6 NIEDRIG — die Baseline sieht weniger als die Modelle, an drei Stellen

NWP-Kantengeometrie, Historienlauf-NWP, Nachbarmessungen (3.2). Alle drei benachteiligen die
Baseline. Richtung ist für uns unbequem, also die richtige Voreinstellung; sie müssen aber
im Methodikteil genannt werden, sonst überschätzt der Vergleich die Graphen. Die drei
optionalen Schalter existieren, damit die Frage bei Bedarf in Minuten beantwortbar ist.

### 8.7 NIEDRIG — die Imputationskette wechselt innerhalb des Testfensters

Kriging bis 2025-11-02, danach nur KNN (1.5). Betrifft Modelle und Baselines identisch,
also kein Vergleichbarkeitsproblem, aber eine Methodenbeschreibungspflicht.

### 8.8 NIEDRIG — die Fold-Ergebnisse sind zeitlich vollständig korreliert

Alle drei Folds nutzen dasselbe Zeitfenster (`study_overview.md` §6,
`stdhp_dryrun_results.md` §5.2). Die Fold-SD misst räumliche Stichprobenvariabilität, nicht
Jahr-zu-Jahr-Variabilität. Für die Baselines gilt das unverändert; keine Wilcoxon-Auswertung
„über Folds" ohne diesen Vorbehalt.

---

## 9. Was nicht verifizierbar war

### 9.1 Die genaue Testfenster-Paarzahl
Ohne einen echten Lauf mit gesetztem `test_end` nicht bestimmbar. Meine Rekonstruktion
(Schnittmenge der Laufzeiten über Stationen, eine Gitterdatei je Station) unterschätzt
systematisch — sie liefert 1102 statt 1473 Trainings- und 1095 statt 1460 Val-Paare, also
etwa 75 %. Die Ursache liegt in der Schlüsselstruktur `(stem, run_hour)` in
`load_icond2_ml_runs:364-378`: pro Schlüssel wird **eine** Datei geladen (`if key not in
unique_grid_paths`), und welche das ist, hängt von der Stationsreihenfolge ab. Die
Laufabdeckung ist damit nur durch den Loader selbst exakt zu bestimmen.

### 9.2 Der Zeitausrichtungsanker 1.4958 / 1.5040 / 1.5550
`study_overview.md` §3 nennt die drei Werte „über die tatsächlich gebauten Run-Paare", ohne
zu sagen, welche. Sie passen nicht zu den 1.450 (gepoolt, Val) und 1.325 (per Station, Val)
aus `stdhp_dryrun_results.md` §2, stammen also vermutlich aus Train **und** Val zusammen.
V2 muss die Stichprobe selbst benennen und die drei Zahlen darauf reproduzieren.

### 9.3 Die Topo-Verschiebung 0.4738 gegen 0.5978
`study_overview.md` §4 und `review_round2_findings.md` N1 nennen `max|Δ| = 0.4738` für den
Unterschied zwischen Topo-z-Score auf Trainings- gegen alle Stationen. Ich messe für
**Fold 1** **0.5978** (alle 9 Deskriptoren, 153 Stationen). Wahrscheinlich ein anderer Fold
oder eine andere Deskriptorenmenge. Kleine Doku-Abweichung, für QRF ohnehin folgenlos
(V7), aber zu melden.

### 9.4 Erwartungswerte der Baselines
Es gibt keine. Die Bänder in V10 sind aus benachbarten Varianten abgeleitet, nicht gemessen.

---

## 10. Gemeldete Codebefunde — nicht reparieren

### 10.1 `cKDTree` in `evaluate_reference.py` — **erledigt, Doku veraltet**
`prompt_baselines_implementation.md` §4.2 sagt, „`evaluate_reference.py` importiert weiterhin
`cKDTree`". **Das stimmt an HEAD `d49096f` nicht mehr.** Der Import fehlt vollständig
(Zeilen 26-53 enthalten ihn nicht), und Zeilen 376-385 dokumentieren die Reparatur samt
Zahlen für Station 05142. Verifiziert: der geodätische Nachbar aus dem Loader stimmt für
**alle 153** Stationen mit `geodesic_knn` überein. Kein Befund, aber die Anweisung im
Referenzprompt ist überholt.

### 10.2 N1 im Evaluationspfad — **weiterhin offen, DCRNN betroffen**
`train_dcrnn.py:865` fittet den Geo-Statik-Scaler auf `raw_static if (val_start and not
args.test_mode) else raw_static[:N_train]`, also im Dev-Modus auf **allen 153** Stationen.
`get_test_results_dcrnn.py:413-416` fittet **immer** `raw_static[:N_train]`, im Dev-Modus
also auf **102**. Die von `review_round2_findings.md` N1 (Vermerk 2026-08-10) verlangte
Angleichung ist an HEAD **nicht** umgesetzt. Ein DCRNN-Fold-Modell wird also weiterhin mit
einer anderen Normierung ausgewertet als trainiert.

**Nicht betroffen: MTGNN und WaveNet.** `HomoSampler._init_static:259-262` normiert `alt`
auf `alts[self.train_idx]` und kodiert lat/lon als `sin`/`cos` **ohne** Skalierung; in
`get_test_results_mtgnn.py:416-417` ist `train_station_indices = list(range(N_train))`, in
`hpo_mtgnn.py:840` `plan["train_idx"] = sf.train_idx` — beides die Fold-Trainingsstationen.
Beide Pfade stimmen also überein.

### 10.3 ICON-D2-Gitter-NaN-Filter nur in den HPO-Skripten
`grep -c "_grid_nan_runs\|NaN in ICON-D2 grid"`: `hpo_dcrnn.py` 6, `hpo_mtgnn.py` 5,
`hpo_wavenet.py` 5; `train_dcrnn.py`, `train_mtgnn.py`, alle vier `get_test_results_*.py`
und `evaluate_reference.py` je **0**. Derzeit folgenlos (0 von 3303 Läufen betroffen), ab
`test_end > 2026-04-29` tragend: Retrain würde auf NaN trainieren, Evaluation NaN-Zeilen
erzeugen und stationsabhängige `n_samples` produzieren. Der Fix wäre der Block aus
`hpo_mtgnn.py:666-682` an sechs weiteren Stellen. **Melden.**

### 10.4 `configs/*/test/` tragen veraltete Stationslisten
Siehe 8.2. `configs/mtgnn/test/config_wind_mtgnn_nwp_fold1.yaml`: 103 `files`, 50
`val_files`, keine Übereinstimmung mit `spatial_fold1`. Diese Configs sind vom Juni/Juli und
gehören zur archivierten Fold-Definition. Wer sie heute benutzt, erzeugt genau die Dateien,
die am 2026-08-06 als unbrauchbar archiviert wurden. **Löschen oder neu generieren — nicht
im Rahmen dieses Auftrags.**

### 10.5 ECMWF-Quellwahl: Elternverzeichnis hat keine `*_sl.parquet`
`load_ecmwf_parquet_at_stations_and_grid:754-760` bevorzugt `*_sl.parquet` **direkt** in
`ecmwf_path` und fällt nur sonst auf `SL/` zurück; der Kommentar dort warnt, dass die
`SL/`-Kopie im Mai aufgehört habe, aktualisiert zu werden. Aktueller Stand auf `l2`:
`/mnt/lambda1/nvme1/ecmwf/parquet` enthält **0** `*_sl.parquet`, `SL/` enthält **759**.
Es wird also `SL/` benutzt, und dessen Daten reichen bis **2026-06-18 21:00** ohne NaN ab
2025-08. Kein Befund, aber der Config-Kommentar „Grenze gegen NaN in ECMWF-Parquets ab
2026-05-01" (`config_wind_mtgnn_nwp_fold1.yaml:32`) beschreibt den heutigen Stand nicht mehr
und sollte niemanden zu einem zu frühen `test_end` verleiten.

### 10.6 Trial-Budget wird pro Worker gezählt
`hpo_mtgnn.py:1097-1098` berechnet `remaining` **einmal** beim Start. N Worker auf derselben
Studie holen sich je `n_trials`. Bekannt und in `study_overview.md` §7 dokumentiert. Für die
QRF-Studie ist die Konsequenz konkret: **genau einen Worker starten**, sonst laufen 2 × 60
Trials.

---

## 11. Aufwandsschätzung

Reine Implementierungs- und Rechenzeit, ohne Wartezeiten auf Entscheidungen.

| Teil | Datei | Arbeit | Rechenzeit |
|---|---|---|---|
| Datenmodul, Loader-Verdrahtung, Run-Paar-Schleife, Design-Matrix | `baselines/dataset.py` | **4–6 h** | Datenladen 15–30 min je Config-Variante (Cache-Miss); mit GNNCache-Treffer 2–4 min |
| QRF-Fit/Predict | `baselines/qrf.py` | 1–2 h | — |
| MOS drei Varianten × zwei Quellenmengen | `baselines/mos.py` | 2–3 h | 3 Folds × 6 Arme **< 10 min** gesamt |
| CLI, Banner, Ausgabe, Import von `_station_metrics`/`_save` | `baselines/evaluate_baselines.py` | 2–3 h | — |
| Skalierungskurve (Schritt 0, 5.3) | — | 0.5 h | 5 Fits auf Fold 0 ≈ **2.5–3 h** CPU |
| Optuna-Treiber | `hpo_qrf.py` | 2–3 h | — |
| Drei QRF-Configs | `configs/baselines/` | 0.5 h | — |
| **QRF-HPO, 60 Trials, 1 M Zeilen, `n_jobs=64`, MedianPruner** | — | — | **20–25 h** CPU, ein Worker |
| QRF-Retrain + Auswertung, 3 Folds | — | — | 3 × (Fit + 60 s Predict); 1 M Zeilen ≈ **1 h**, voller Satz ≈ **21 h** |
| Verifikationssuite V1–V11 | `archiv/baselines_verification/verify_baselines.py` | **5–7 h** | 1–3 h (V1 und V7 fitten je zweimal; auf 200 k Zeilen fahren, nicht auf 7 M) |
| Verifikationsdokument mit Zahlen | `docs/baselines_verification_results.md` | 1.5–2 h | — |
| Rollout l2 → l1 → ws inkl. Pfad-Rewrite-Kontrolle | — | 0.5 h | — |
| **Summe Implementierung** | | **21–29 h** | |
| **Summe Rechenzeit CPU** | | | **26–33 h** (1 M-Variante) bzw. **46–53 h** (voller Satz) |

Nicht enthalten, weil von Entscheidungen abhängig: QRF-IDW (F1, +1–2 h Arbeit, +HPO-Zeit
falls getunt), Retraining-Szenario (F2, +2–3 h Arbeit, +2–4 h Rechenzeit),
Regression-Kriging-Einreihung (F4, siehe dort), Testfensterläufe (blockiert durch 8.1/8.3).

**GPU: null.** Alles läuft mit `CUDA_VISIBLE_DEVICES="" nice -n 19` auf CPU.

---

## 12. Offene Fragen — entscheidet ausschließlich der Nutzer

### F0 (NEU, blockierend) — Wie lang ist das Testfenster wirklich?
Der Auftrag sagt 2025-08-01 … 2026-07-31. Die Daten geben das nicht her (8.1): Laufstunde 15
endet **überall** am 2026-04-29, Laufstunde 06 bei 72 von 153 Stationen ebenfalls; ECMWF
endet 2026-06-18.

| Option | Konsequenz |
|---|---|
| **(a) `test_end = 2026-04-29`** | Neun Monate, alle vier Laufstunden, vollständige Gitter, ~1088 Läufe. Kein Codefix nötig. Paper muss „neun Monate" schreiben. |
| (b) `test_end = 2026-06-11`, drei Laufstunden ab 2026-04-30 | Zehn Monate, aber wechselnde Stichprobendichte **und** der Gitter-NaN-Filter muss an sechs Stellen nachgerüstet werden (10.3), sonst stationsabhängige `n_samples`. |
| (c) ICON-D2 für Mai–Juli 2026 nachziehen | Zwölf Monate wie geplant. Kosten: Datenbeschaffung und Neuextraktion für 153 Stationen × 4 Laufstunden × 22 Gitterpunkte, plus GNNCache-Invalidierung. Unbekannter Zeitbedarf, nicht von mir abschätzbar. |

**Empfehlung: (a).** Es ist die einzige Option ohne Codeänderung an der laufenden
Kampagneninfrastruktur, und die Winterlastigkeit — der eigentliche Zweck des Testfensters
laut `story_positioning.md` §3.2 — ist mit August bis April vollständig abgedeckt. Das
Retraining-Szenario (4 + 4 Monate ab 2025-08-01) passt vollständig hinein.

### F1 — Zweiter QRF-Arm mit Nachbarmessungen (QRF-IDW)?
Er wäre die nicht-neuronale Gegenprobe zur Obergrenze aus Contribution (ii), also das
Gegenstück zu Variante **A** statt zu **C**.

* **Aufwand:** +1–2 h Arbeit (eine zusätzliche Spaltengruppe: inverse-distanzgewichtetes
  Mittel der `n` nächsten Nachbarmessungen zur Laufzeit `t_run`, plus die Distanz zum
  nächsten Nachbarn).
* **Rechenzeit:** eigene HPO wäre weitere 20–25 h CPU; ohne eigene HPO (Parameter von
  QRF-local übernehmen) nur +1 h.
* **Konsequenz bei Nein:** das Paper vergleicht die Graphmodelle mit Nachbarmessungen
  (Variante A, 1.129) gegen eine Baseline **ohne** sie. Ein Referee kann das als unfairen
  Vergleich lesen. Bei Ja fällt genau dieser Einwand weg.

**Empfehlung: ja, aber ohne eigene HPO** — Parameter von QRF-local übernehmen und das im
Text als konservativ zugunsten der Graphen ausweisen. Kostet ~2 h und schließt die Lücke.

### F2 — Fahren die Baselines das Retraining-Szenario mit?
* **Aufwand:** +2–3 h (zweite Fit-Fenstergrenze, zweiter Ausgabestem).
* **Rechenzeit:** +2–4 h CPU. MOS ist dabei kostenlos.
* **Konsequenz bei Nein:** die Frage „bringt Retraining etwas" ist nur für die Graphmodelle
  beantwortet. Da ein Random Forest auf vier zusätzliche Monate ganz anders reagiert als ein
  neuronales Modell, ist das die interessantere Hälfte des Experiments.

**Empfehlung: ja.** Auf der CPU billig, und ohne die Baseline ist der Retraining-Befund
nicht interpretierbar.

### F3 — Benennung im Paper
Bei deterministischer Ausgabe (D1) ist es ein **Random Forest**, kein Quantile Regression
Forest. Verglichen wird gegen `taillardat2016calibrated` und `schulz2022machine`, die
quantilbasiert sind.

| Option | Konsequenz |
|---|---|
| (a) „random forest (RF)" | Ehrlich, aber der Bezug zur zitierten Literatur muss im Text hergestellt werden. |
| **(b) „QRF, ausgewertet über den bedingten Mittelwert"** | Bezug bleibt sichtbar, Determinismus ist explizit. Erfordert einen Halbsatz Methodik. |
| (c) „QRF" ohne Zusatz | Irreführend. Nicht empfehlen. |

**Empfehlung: (b)** in der Methodik, Kurzform „RF" in Tabellenköpfen, mit Fußnote. Der
Dateiname bleibt `qrf_local`, damit Code und Studienname stabil sind.

### F4 — Regression-Kriging einreihen?
Es existiert als Optuna-Studie `wind_interpol` (**131 Trials, best 1.1562** — selbst
nachgeprüft). Der Wert liegt zwischen DCRNN GRID (1.129) und DCRNN BASE (1.162), ist aber
**nicht** vergleichbar: die Studie stammt aus der Zeit vor der räumlichen CV, und `1.1562`
ist ein Objective unbekannten Zuschnitts auf unbekannten Stations- und Zeitmengen.

| Option | Konsequenz |
|---|---|
| **(a) nicht einreihen** | Kostet nichts. Contribution (i) verliert eine der genannten klassischen Baselines. |
| (b) neu auswerten über `run_spatial_interpolation.py` auf den drei Folds | Aufwand **unklar, mindestens 6–10 h**: das Skript (50 kB) ist nicht auf die räumliche CV umgestellt, hat eigene Configs (`config_wind_interpol.yaml`) und eigene Ausgabeformate. Rechenzeit unbekannt. |
| (c) die Zahl 1.1562 zitieren wie sie ist | **Nicht empfehlen.** Genau der Fehler, den der Auftrag verbietet: vorgetäuschte Vergleichbarkeit. |

**Empfehlung: (a) für dieses Paper**, mit einem Satz in den Limitations. MOS-regional und
QRF-local decken die klassische Familie ab; Regression-Kriging ist ein drittes Verfahren
derselben Klasse und trägt keine neue Aussage, kostet aber den Umbau eines
50-kB-Skripts. Falls der Nutzer (b) will, ist es ein **eigener Auftrag**, nicht Teil dieses.

### F5 (NEU) — Widerspruch zu MOS-local auflösen
`prompt_baselines_orchestration.md` §4.2 sagt: MOS-local „klassisch je Station und
**ausschließlich an gehaltenen Stationen** als transduktive obere Schranke".
`prompt_baselines_implementation.md` §6 Test 5 sagt: „Beleg, dass es an Zielstationen
**nicht** ausgewertet wird — es ist dort per Definition undefiniert."

Das ist ein direkter Widerspruch. Ich lese §4.2 (neueres Dokument) als maßgeblich:
MOS-local wird **auf der eigenen Historie jeder Zielstation im Trainingsfenster gefittet
und an derselben Station im Val-/Testfenster ausgewertet**. Genau das macht es zur
transduktiven Obergrenze und beziffert, was die Induktion kostet. §6.5 stammt aus der Zeit,
als noch unklar war, ob „lokal" heißt „nur wo Historie existiert".

**Bitte bestätigen.** Bei der Gegenlesung („MOS-local nur an Trainingsstationen") wäre es
keine Obergrenze für die Induktion mehr, sondern eine In-Sample-Zahl auf einer anderen
Stationsmenge — und dann wertlos für den Vergleich.

### F6 (NEU) — Zwei NWP-Quellen für MOS?
`prompt_baselines_implementation.md` §4.2 nennt die NWP-Windgeschwindigkeit am nächsten
Gitterpunkt im Singular. Ich spezifiziere beide Läufe (`--nwp-sources icond2` und `both`,
3.5), weil ein Einquellen-MOS gegen Zweiquellen-Modelle ein Strohmann ist und
`stdhp_dryrun_results.md` §5.4 zeigt, dass ECMWF an 74 von 153 Stationen besser ist.
Kosten: Sekunden. **Bitte bestätigen, dass beide berichtet werden** — wenn nur einer in die
Haupttabelle darf, ist die Wahl eine inhaltliche.

---

## 13. Reihenfolge für Phase 2

1. Schritt 0: Skalierungskurve (5.3). Ohne sie ist jede Zeilenzahl geraten.
2. `dataset.py` + Bannerzeile + `--dry-run`. Erst wenn `fit_pairs = 1473`,
   `eval_pairs = 1460`, `fit_rows = 7 211 808`, `eval_rows = 3 574 080` und alle drei
   Filterzähler geloggt sind, geht es weiter.
3. MOS (alle sechs Arme). Sie sind in Minuten fertig und liefern die erste
   Plausibilitätsprobe gegen ICON-D2 und Persistenz.
4. V1–V6 und V9 auf den MOS-Ergebnissen. Format, Schlüssel, Referenzen und Leckage sind
   damit geklärt, bevor irgendein Wald gefittet wird.
5. `qrf.py` + V7 (Skalierungsinvarianz) auf 200 k Zeilen.
6. `hpo_qrf.py`, **ein** Worker (10.6), Studienname geprüft, V11 vorher und nachher.
7. QRF-Retrain, Auswertung, V8 und V10.
8. `docs/baselines_verification_results.md` mit allen Zahlen.
9. Rollout l2 → l1 → ws, Commit-Hash je Host belegen, auf `l1` die Pfad-Rewrites erneut
   anwenden und mit der Kontrollzeile aus `prompt_baselines_implementation.md` §2
   nachweisen, dass der Diff **nur** Pfade enthält.

**Testfensterläufe erst nach F0.**
