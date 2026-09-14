# Solar-TFT-Kampagne — Aufbau, Datenlage, Entscheidungen

Stand 14.09.2026. Löst [solar_irradiance_plan.md](solar_irradiance_plan.md) als
Referenz ab (der Plan beschreibt den Aufbau der Pipeline im August 2026 und ist
bis auf §6 „Offen" abgearbeitet).

---

## 1. Was vorher schon gemessen wurde

Alle Zahlen aus TFT-Läufen mit **untunten Default-Hyperparametern** — eine
Solar-HPO hat es bis zum 14.09.2026 nicht gegeben. Nachrechenbar mit
`scripts/compare_solar_ablation.py` und `scripts/bilanz_solar_final.py`.

### 1.1 Ablationsleiter (13.–17.08.2026)

Testfenster 2025-02-01…2025-08-01, 52 Trainings-/18 Auswertungsstationen,
30 min, Ziel `ghi`+`dhi`, RMSE_GHI in W/m² über die Stationen:

| Variante | RMSE_GHI | gegen `residual` |
|---|---|---|
| **ICON-D2 + ECMWF** (4 Spielarten, je 4 Wdh.) | **75.1–75.7** | **−3.9 %** |
| `ab_featgrid` (Featureset + 4 Gitterpunkte) | 77.8–78.3 | −0.4 % |
| `ab_features` (+`kt_nwp`/`airmass`/`dni_cs`/`dhi_cs`) | 78.1–78.7 | −0.4 % |
| `ab_grid4` (4 ICON-Gitterpunkte) | 78.1–78.3 | −0.2 % |
| `residual` — `target_transform: nwp_residual` | 78.25–78.41 | Referenz |
| `ab_clearsky` (Ziel = Clear-Sky-Index) | 78.5–79.0 | +0.3 % |
| `ab_single` (nur `ghi` statt `ghi`+`dhi`) | 78.57 | +0.3 % |
| `absolut` (Ziel in W/m² statt Residuum) | 78.6–79.1 | +0.4 % |
| `ab_noobs` (ohne eigene Messhistorie) | 79.0–79.3 | +0.9 % |
| `ab_neigh` (Nachbarstationen statt eigener Messung) | 79.3–79.5 | +1.4 % |
| `ab_raster15min` | 82.9 | +5.8 % |
| `ab_raster10min` | 88.1 | +12.5 % |

**Vier Festlegungen folgen daraus** und stehen seither in jeder Config:
30-min-Raster, `ghi`+`dhi` gemeinsam, ICON-D2 **und** ECMWF,
`target_transform: nwp_residual`. Alles außer ECMWF bewegt sich im Bereich der
Lauf-zu-Lauf-Streuung (rund 0.45 W/m², gemessen über je 2–4 Wiederholungen).

> Die `ab_clearsky`-DHI-Zeile aus `compare_solar_ablation.py` ist **nicht**
> vergleichbar: dort weicht die NWP-Baseline um 13 % ab (41.6 statt 47.8 W/m²),
> weil die Clear-Sky-Rücktransformation Nacht und tiefe Dämmerung auf 0 zwingt.

### 1.2 Abschlussbewertung Baseline (18.08.2026)

85 Stationen, **rein zeitlicher** Split (dieselben Stationen im Testfenster),
Training 2023-08…2024-07, Test 2024-08…2025-08, ICON-D2 + ECMWF:

| | RMSE_GHI | Skill_NWP | R² (absolute Skala) |
|---|---|---|---|
| mit Lag (`observed_features: [ghi, dhi]`) | **63.75** | 0.111 | 0.916 |
| ohne Lag | 64.21 | 0.105 | 0.915 |
| ICON-D2 roh | 71.69 | — | 0.894 |

Gepaart über 83 Stationen ist der Lag-Vorteil 0.72 %, p < 1e-5, und er trägt an
75 von 83 Stationen.

Die **stationsdisjunkte** zweite Stufe (`configs/solar_final/`, 52 train / 18
eval) wurde am 18.08. gestartet, brach im Training ab und hat nie Ergebnisse
geliefert — `results/solar_ecmwf` aus jener Config existiert nicht.

---

## 2. Datenlage (Stand 14.09.2026)

| Quelle | Umfang | Zeitraum |
|---|---|---|
| DWD-Messungen | 204 Stationen, 94 mit `ghi`/`dhi` | 2023-07-24 … 2026-08-04, 10 min |
| ICON-D2 SL | 1218 Gitterpunkte, Läufe 06/09/12/15 | 2023-07-24 … **2026-07-31** (15-UTC-Lauf erst ab 2023-08-08) |
| ECMWF HRES | 759 Gitterpunkte | 2023-07-01 … 2026-08-31 |
| Imputation `interpol/solar` | 94 Stationen, 30 min | 2023-07-24 … 2026-07-31 |

**Einen Juli 2023 gibt es für diesen Aufbau nicht** — der 15-UTC-Lauf beginnt
am 2023-08-08. Deshalb `train_start: 2023-08-01` in allen Configs.

### 2.1 Die Imputation

Erzeugt am 2026-09-03 von `~/Work/NWP/ERA5/scripts/impute_solar.py` aus dem
Solar-Abschlussmodell (Trial 135). Schema `ghi_imputed` / `dhi_imputed` /
`ist_tag` / `kontextfrei` — **nicht** das `imputed`/`rk_pred` des Wind-Baums.

Drei Eigenschaften, die man kennen muss:

1. **Der Zeitstempel meint das Intervall-Ende**, im Framework meint er den
   Anfang. Nachgemessen: das RMSE-Minimum von `ghi_observed` gegen die
   resamplete Rohmessung liegt bei Verschiebung **−1 Schritt** (Station 00183:
   exakt 0.000, also bitgleiche Werte; Verschiebung 0 ergäbe 76.8 W/m²).
   `utils/imputation._solar_imputation_frame` korrigiert das.
2. **Nachts wird nicht gefüllt.** `ist_tag` ist dabei keine Horizontgrenze,
   sondern `ghi_clearsky > Schwelle`; in den „Nacht"-Schritten stehen gemessen
   noch p99 = 10 W/m², maximal rund 33 W/m². Die offenen Nachtlücken sind
   genauso zahlreich wie die gefüllten Taglücken (Station 04887: 20 162 gegen
   19 748) und würden über das `dropna()` ganze 48-h-Läufe kosten. Sie werden
   deshalb mit **0** belegt (`params.impute_night_zero`, Default an) und zählen
   wie jede Füllung als nicht beobachtet.
3. **Zu den gefüllten Werten gibt es keine Gütezahl.** Die Kennzahlen des
   Abschlussmodells (GHI R² 0.954, RMSE 51.3 W/m²) stammen aus künstlich
   verdeckten, tatsächlich beobachteten Schritten — nicht aus den echten Lücken.
   `~/Work/NWP/ERA5/RESULTS.md` §5.2. In keine Auswertung gehört eine Zahl, die
   anders klingt.

### 2.2 Wie viel wirklich gemessen ist

Anteil echter Messwerte je Zeitfenster, `ghi`:

| Stationssatz | Train 23-08…24-07 | Val 24-08…25-07 | Test 25-08…26-07 |
|---|---|---|---|
| Testsatz (21) | 0.994 / min 0.952 | 0.994 / min 0.965 | 0.914 / min 0.181 |
| Pool (62) | 0.991 / min 0.901 | 0.989 / min 0.666 | 0.850 / min 0.120 |
| neu durch Imputation (11) | 0.847 / min 0.000 | 0.448 | 0.118 |

Der entscheidende Punkt: die 11 zusätzlichen Stationen sind **nicht verstreut
lückig, sie enden vorzeitig**. Neun von ihnen messen im Trainingsfenster zu
95–100 %; ihr Ausfall beginnt erst im Val-Jahr. Im Trainingsfenster sind sie
also schlicht neun weitere Messstationen, keine Imputationskonstrukte.

Ausgeschlossen bleiben **04642** (Reihe beginnt erst 2025-06, 0 % im Training)
und **04887** (41 %, unter der 50-%-Schwelle aus
[station_splits_solar.md](station_splits_solar.md) §5).

**Für das Testjahr gilt die Warnung umgekehrt:** dort fällt auch der Pool ab
(Mittel 0.85, Minimum 0.12). Mit `eval.exclude_imputed` bleibt entsprechend
weniger Auswertungsmasse — vor der Interpretation nachzählen.

---

## 3. Der Aufbau

```
94 Stationen mit Strahlungsdaten
├── 21 Testsatz      nie im Training, nie in der HPO — nur die Schlussmessung
├── 62 Pool          3 rotierende Folds, je 41/21 (Fold 3: 42/20)
└──  9 Zusatz        nur Trainingsrolle, nie Zielstation  → Arm B
     (+2 verworfen: 04642, 04887)
```

Stationsrollen aus `configs/solar_folds.yaml`, Begründung des Zuschnitts in
[station_splits_solar.md](station_splits_solar.md).

### 3.1 Zeitachse

| | Training | Auswertung |
|---|---|---|
| HPO und Modellwahl | 2023-08-01 … 2024-07-31 | 2024-08-01 … 2025-07-31 |
| Schlussmessung | 2023-08-01 … 2025-07-31 | 2025-08-01 … 2026-07-31 |

Das Testjahr wird zurückgehalten, bis die Modellwahl steht — wie beim Wind.

Kein Puffer zwischen Training und Auswertung: der Split greift auf `starttime`,
ein Puffer verwürfe die Läufe dazwischen ersatzlos. Der letzte Trainingslauf
reicht über seinen 48-h-Horizont in den Testzeitraum und überschneidet damit
2 von 366 Testtagen (0.55 %). Das ist der günstigere Tausch
(`preprocessing.py:407`, dazu der Memory-Eintrag zum Monatsende).

### 3.2 Zwei Arme

| Arm | Configs | Trainingspool |
|---|---|---|
| `solar_tft` | `configs/solar_tft/` | 62 |
| `solar_tft_plus` | `configs/solar_tft_plus/` | 62 + 9 |

**Die Zielstationen sind in beiden Armen identisch** — nur so ist der Vergleich
gepaart und beantwortet die gestellte Frage: bringen neun zusätzliche
Trainingsstationen den 62 etwas?

Erzeugt mit `scripts/make_solar_tft_configs.py` (idempotent, `--force` zum
Überschreiben). Kein Suffix hinter `_fold<N>` — die Optuna-Studienauflösung
leitet den Namen aus dem Dateinamen ab.

---

## 4. Was am Code geändert wurde (14.09.2026)

| Datei | Änderung |
|---|---|
| `utils/imputation.py` | `impute_solar_measurements` — Solar-Schema, Label-Korrektur, Nachtfüllung, `<target>_observed` |
| `utils/solar.py` | Imputation **vor** dem `dropna()` statt danach in `get_data` (sonst sind die Lücken schon verworfen); `_observed`-Spalten durch die Spaltenauswahl gerettet |
| `utils/eval.py` | `get_metrics(mask=…)` und `evaluate_models(observed=…)` — elementweise Filterung, zusätzliche Spalte `n_points` |
| `train_cl.py` | reicht `eval.exclude_imputed` durch |
| `geostatistics/spatial_cv.py` | `extra_train` in `station_pool`/`build_folds` |
| `hpo_tft_bc.py` | liest `hpo.extra_train_files` |
| `scripts/make_solar_tft_configs.py` | **neu** |
| `scripts/bilanz_solar_final.py`, `preflight_testmode.py`, `measure_dir.py` | `yaml.safe_load` → `load_config` (seit der `!ENV`-Umstellung kaputt) |

### 4.1 Warum elementweise gefiltert wird

`evaluate_models` verwirft bisher **ganze Läufe**, sobald eine Reihe eine Lücke
hat. Bei 96 Leads à 30 min kostete ein einziger imputierter Schritt damit 48
Stunden Auswertung. Die Filterung imputierter Ziele läuft deshalb elementweise
über eine `(Lauf × Lead)`-Maske — dieselbe Stichprobe für Modell und Baselines,
sodass `Skill` und `Skill_NWP` weiterhin Quotienten über derselben Menge sind.
Das entspricht dem Vorgehen beim Wind (`scripts/eval_testmode.py`, dort
nachträglich auf `(station_id, run_time, horizon)`).

Ohne `eval.exclude_imputed` ist das Verhalten unverändert — verifiziert:
`get_metrics` mit Vollmaske liefert bitgleiche Werte wie ohne Maske.

### 4.2 Neue Config-Schlüssel

| Schlüssel | Wirkung |
|---|---|
| `data.interpol_path` | Verzeichnis der Solar-Imputation |
| `params.impute_night_zero` | Nachtlücken mit 0 belegen (Default `true`) |
| `eval.exclude_imputed` | imputierte Zielpositionen aus allen Metriken nehmen |
| `hpo.extra_train_files` | Stationen mit Trainingsrolle in jedem Fold, nie Zielstation |

---

## 5. Offen

1. **Solar-HPO.** `hpo_tft_bc.py` ist mit `cv_mode: spatial` und
   `extra_train_files` vorbereitet, aber noch nicht gestartet. Suchraum und
   Trial-Budget stehen unentschieden.
2. **Clear-Sky-Persistenz** als Baseline (`kt` von t−24 h × `ghi_cs(t)`) — es
   läuft weiterhin nur die naive Persistenz.
3. **Tagstunden-Metrik.** `params.daytime_zenith_threshold` steht in den
   Configs, `eval.get_metrics` wertet ihn nicht aus.
4. **GNN-Pfad.** Mit der Imputation ist der Blocker aus
   `solar_irradiance_plan.md` §6 grundsätzlich gelöst; geprüft ist es nicht.
