# KNN-Imputationscache neu gerechnet (2026-09-02)

**Status: UMGESETZT** am 2026-09-02, 09:19–09:27 UTC auf `l1`
(`/home/viktorwalter/Work/forecasting_framework`). Betrifft
`.../synthetic/knnimputer/wind` — den Cache, aus dem alle Pipelines die
**Sekundärspalten** füllen. `wind_speed` läuft seit dem 2026-09-02 ausschließlich
über die TFT-Dateien (`docs/imputation_tft_switch.md`); der KNN-Cache ist für
Wind damit praktisch der `wind_direction`-Pfad.

## 1. Anlass

Der Preflight zur finalen Testauswertung (`docs/handoff_testmode.md`) fand mit
`test_end: '2026-07-31'` **614 NaN in `wind_direction` an 26 Stationen**, alle
nach dem 2026-07-14 23:00 UTC. Ursache war nicht die Messreihe, sondern die
Reichweite des Caches:

| Quelle | Abdeckung vorher |
|---|---|
| Rohmessungen `.../synthetic/raw/wind` | bis 2026-09-01 23:50 UTC |
| TFT-Imputation `wind_speed` (`interpol/wind`, Spalte `imputed`) | bis 2026-07-31 23:00 UTC |
| **KNN-Cache `wind_direction`** | **bis 2026-07-14 23:00 UTC** |

Mit `handle_nans: break` (Standard in allen betroffenen Configs) hätten die fünf
Läufe mit `test_end: '2026-07-31'` abgebrochen. Die Alternative — die Testgrenze
auf den 2026-07-14 zurückzunehmen — hätte 16 Tage Testdaten gekostet; der Nutzer
hat sich am 2026-09-02 gegen diesen Verzicht und für die Neurechnung
entschieden.

Der Cache aktualisiert sich **nicht** von selbst: `load_knn_imputation` nimmt
`sorted(glob(...))[-1]`, prüft aber keine Aktualität (Kommentar dazu im Kopf von
`geostatistics/regen_knn_imputation.py`).

## 2. Was gerechnet wurde

`python geostatistics/regen_knn_imputation.py` — unverändert in der Sache,
`KNN_K = 10`, stündliches Resampling vor dem Fit, Plausibilitäts-Guard wie
gehabt. Laufzeit **7,5 min** (wind_speed 2,3 min, wind_direction 4,3 min).

Einzige Codeänderung: `DATA_PATH`/`CACHE_DIR` werden jetzt zur Laufzeit
aufgelöst (`/mnt/nvme1` auf l1, `/mnt/lambda1/nvme1` auf l2 und ws) — dasselbe
Muster wie `utils/era5_imputation.py:85-92`. Vorher waren die l2-Pfade fest
verdrahtet, das Skript war auf dem Besitzer-Host nicht lauffähig.

## 3. Was sich geändert hat

### 3.1 Zeitliche Abdeckung

26 088 → **27 264** stündliche Zeitschritte, 2023-07-24 00:00 →
**2026-09-01 23:00 UTC**. Nach der Neurechnung: **0 NaN** in `wind_direction`
über alle 203 Stationen im gesamten Auditfenster bis 2026-07-31.

### 3.2 Stationsmenge: 203 → 204

Der Stationssatz-Hash im Dateinamen wechselt von `67558851` auf `611c3831`. Neu
dabei ist **05839 (Emden)** — die Station, die die TFT-Imputation wegen 94,4 %
Fehlanteil bewusst auslässt. Sie steht in **keiner** Config-Stationsliste
(`files`/`val_files`/`test_files` bleiben bei 203) und wirkt nur als zusätzliche
Spalte im KNNImputer-Fit. Da der Imputer zeilenweise über die Zeitachse
arbeitet, ist ein zu 94 % leeres Merkmal für die `nan_euclidean`-Distanz
praktisch gewichtslos. Ihre Spalte im Cache liest niemand: `load_knn_imputation`
reindiziert auf die angeforderten Stations-IDs.

### 3.3 Werte

Der Fit läuft über einen längeren Zeitraum und eine Spalte mehr, deshalb
verschieben sich auch **historische** imputierte Richtungswerte leicht. Wer
Läufe von vor dem 2026-09-02 (§14–§18 in `docs/evaluation_results.md`:
stdhp-Ablationen, expwin, testyear) nachrechnet, arbeitet also nicht mehr auf
bitgleichen Eingangsdaten. Gemessene Stunden sind unberührt — imputiert wird nur
dort, wo keine Messung liegt.

Der Guard meldete beim Lauf: `wind_speed` 0 Werte < 0, 3 Werte > 40 m/s auf
40 m/s geklemmt; `wind_direction` 8 933 Werte ≥ 360° nach [0, 360) normalisiert.

### 3.4 Dateien

```
.../synthetic/knnimputer/wind/
    wind_speed_knn10_start_end_611c3831.parquet        NEU, in Benutzung
    wind_direction_knn10_start_end_611c3831.parquet    NEU, in Benutzung
    wind_speed_knn10_start_20251102_67558851.parquet   alt, 10-min, ungenutzt
    wind_direction_knn10_start_20251102_67558851.parquet
.../synthetic/knnimputer/wind_vor_regen_20260902/      abgelegter Vorstand
    wind_speed_knn10_start_end_67558851.parquet
    wind_direction_knn10_start_end_67558851.parquet
```

**Der alte Stand musste aus dem Verzeichnis heraus.** `load_knn_imputation`
wählt `sorted(...)[-1]`, und `611c3831` sortiert **vor** `67558851` — die neuen
Dateien wären danebengelegen, ohne je gelesen zu werden. Zweitkopie des
Vorstands: `~/backup_pre_testmode_20260902/knnimputer_wind/` auf l1.

Namenskonvention des abgelegten Verzeichnisses wie beim TFT-Umstieg
(`interpol/wind_vor_tft_20260902`).

### 3.5 Caches

`utils/data_cache.py` hasht `knnimputer_fingerprint` (mtime/size des
Verzeichnisses) in den Cache-Key. Bestehende `data_cache/`-Einträge werden durch
den Dateitausch automatisch ungültig, eine Erhöhung von
`IMPUTATION_GUARD_VERSION` ist **nicht** nötig — anders als beim TFT-Umstieg
ändert sich hier kein Codepfad, nur der Dateiinhalt, und genau den erfasst der
Fingerprint.

## 4. Nebenbefund: NaN-Audit von train_mtgnn/train_wavenet

`train_mtgnn.py` und `train_wavenet.py` prüften im `--test-mode` bis ans
Datenende statt bis `test_end` — entgegen ihrem eigenen Kommentar und anders als
`train_dcrnn.py`. `meas_raw` wird auf `test_end + 2 d` gekappt, damit der letzte
Lauf sein 48-h-Fenster füllen kann; in diesem Schwanz endet die TFT-Imputation
(2026-07-31 23:00 UTC), und der harte Audit schlug auf 26 Zellen an, die kein
Run-Paar als Ziel hat (`ValueError: 2 station(s) still have NaN … ['02961',
'02985']`). Die Reihenfolge ist jetzt in allen drei Trainern dieselbe:
`test_end` → `test_start` im Dev-Lauf → Datenende.

## 5. Nicht betroffen

- **`wind_speed` im GNN-Pfad.** Kommt aus `interpol/wind` (TFT), der KNN-Cache
  ist dort ausdrücklich kein Fallback (`docs/imputation_tft_switch.md` §2.3).
  Die neue `wind_speed`-Cachedatei bedient nur noch die CL/FL-Vorverarbeitung.
- **Solar.** Eigener Cache unter `knnimputer/solar`, nicht angefasst.
- **Bereits geschriebene Ergebnisdateien.** `results/`, `data/test_results/`,
  `data/raw_preds/` bleiben, wie sie sind; die Imputationsmaske der Auswertung
  baut `make_stdhp_figures.build_imputation_mask` aus den **Rohmessdateien**,
  nicht aus dem KNN-Cache.
