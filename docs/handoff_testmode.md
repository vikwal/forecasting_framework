# Auftrag: finale Testauswertung im `--test-mode`

## Ziel

Je Arm **ein** Modell, trainiert auf den vollen **153 Trainingsknoten**
(`files` + `val_files`), zero-shot ausgewertet auf den **50 Testknoten**
(`test_files`).

- Trainingszeitraum: 2023-07-24 … 2025-08-01
- Testzeitraum: 2025-08-01 … 2026-07-31

Arme (Basis-Configs im Repo `~/Work/forecasting_framework`):

| Arm | Config (fold1 stellvertretend) | Skripte |
|---|---|---|
| `dcrnn` | `configs/dcrnn/config_wind_dcrnn_fold1.yaml` | `geostatistics/train_dcrnn.py` / `get_test_results_dcrnn.py` |
| `dcrnn_idw_alt` | `configs/dcrnn/config_wind_dcrnn_idw_alt_fold1.yaml` | dito |
| `dcrnn_nwp_hist` | `configs/dcrnn/config_wind_dcrnn_nwp_hist_fold1.yaml` | dito |
| `mtgnn_nwp` | `configs/mtgnn/config_wind_mtgnn_nwp_fold1.yaml` | `geostatistics/train_mtgnn.py` / `get_test_results_mtgnn.py` |
| `mtgnn_nwp_hist` | `configs/mtgnn/config_wind_mtgnn_nwp_hist_fold1.yaml` | dito |

## Warum das genau ein Modell je Arm ist

`--test-mode` setzt in Training und Eval `train_ids = files + val_files` und
`val_ids = test_files`. **Verifiziert:** `files + val_files` ergibt in allen drei
Fold-Configs dieselbe 153er-Menge, `test_files` dieselben 50, Überschneidung
null. Im `--test-mode` sind die Fold-Configs also austauschbar — drei Folds wären
nur drei Seeds, keine unterschiedlichen Splits. Ein Lauf je Arm genügt; nimm
`fold1` als Träger.

## Rezept

```bash
cd ~/Work/forecasting_framework && source frcst/bin/activate
export CUDA_VISIBLE_DEVICES=<gpu>

python geostatistics/train_<dcrnn|mtgnn>.py \
    --config configs/testmode/config_wind_<arm>_fold1.yaml \
    --suffix testmode --hpo-study auto --test-mode

python geostatistics/get_test_results_<dcrnn|mtgnn>.py \
    -m wind_<arm>_fold1_<dcrnn|mtgnn>_testmode \
    -c configs/testmode/config_wind_<arm>_fold1.yaml \
    --hpo-study auto --test-mode --raw-out-name testmode_<arm>
```

Ausgaben: `models/wind_<arm>_fold1_<m>_testmode.pt`,
`data/test_results/testmode_<arm>.csv`,
`data/raw_preds/testmode_<arm>_raw.parquet`.

## Configs anlegen

Basis-Config kopieren nach `configs/testmode/`, **Dateiname unverändert lassen**,
und nur setzen:

```yaml
  test_start: '2025-08-01'   # steht schon so drin
  test_end:   '2026-07-31'   # von '2026-03-31' hochsetzen
```

`val_start` ist im `--test-mode` wirkungslos (die Grenze ist `test_start`).

**Fallstrick:** Der Dateiname muss auf `_fold<N>` **enden**.
`train_dcrnn.py:338` / `train_mtgnn.py:277` leiten den Optuna-Study-Namen aus dem
Config-Stem ab und strippen `_fold\d+$` nur am Zeilenende. Ein Name wie
`config_wind_dcrnn_fold1_testmode.yaml` sucht die Study
`cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_fold1_testmode` und scheitert. Varianten
über das Verzeichnis unterscheiden, Läufe über `--suffix`.

## Datenlage — am 2026-09-02 geprüft, keine Blocker

| Quelle | Abdeckung |
|---|---|
| Stationsrohdaten (`/mnt/nvme1/synthetic/raw/wind`, auf l2 `/mnt/lambda1/...`) | bis 2026-07-28 … 2026-08-25 je Station |
| TFT-Imputation (`interpol/wind`, Spalte `imputed`) | alle 203 Stationen, bis 2026-07-31 23:00 |
| ICON-D2 (`multilevelfields`) | bis 2026-08-10 |
| ECMWF (`/mnt/nvme1/ecmwf/parquet/SL`) | bis 2026-07-31, **ohne NaN** |

Der NaN-Audit über alle 203 Stationen ergibt mit `test_end: '2026-07-31'`
**0 Stationen mit Restlücken**. Mit `'2026-08-01'` ist es genau **eine** Stunde
an Station 02961 (deren Rohdaten am 2026-07-28 enden) — deshalb 07-31.

> **Korrektur vom 2026-09-02:** Dieser Satz gilt nur für `wind_speed`. Für
> `wind_direction` endete der KNN-Cache am 2026-07-14 23:00 UTC und ließ 614 NaN
> an 26 Stationen stehen — Abbruch für 5 der 9 Läufe. Behoben durch Neurechnung
> des Caches, s. § Ausführungsstand und `docs/imputation_knn_regen_20260902.md`.

Zwei Punkte, die bis zum 2026-09-01 noch Blocker waren und es **nicht mehr sind**
(nicht erneut diagnostizieren): die alte Kriging-Imputation (`rk_pred`) endete am
2026-06-13 und ließ 46 der 50 Teststationen mit Lücken zurück. Die Umstellung auf
die TFT-Imputation (`imputed`, s. `docs/imputation_tft_switch.md`) hat beides
behoben. `wind_speed` wird **absichtlich nicht** per KNN gefüllt — der Kommentar
`NO KNN fallback` in `train_*.py` und `get_test_results_*.py` ist eine bewusste
Entscheidung, nicht ein Versehen. Nicht daran drehen.

## Hyperparameter

`--hpo-study auto` zieht den besten Optuna-Trial aus PostgreSQL
(`OPTUNA_STORAGE`). Alle fünf Studies existieren, die Bestwerte sind seit
2026-08-24 eingefroren, es läuft kein HPO-Prozess. Kein neues HPO — variiert wird
nur die Datenmenge.

## Rechnen

Host `l1` (8× RTX A6000). Vor dem Start `ssh l1 nvidia-smi` prüfen; GPUs 3–6
tragen üblicherweise Fremdlast. Vorlage für eine selbstbalancierende Queue
(mehrere Worker, `flock` auf eine gemeinsame Aufgabenliste):
`~/queue_scripts/testyear_worker.sh` und `testyear_tasks.txt` auf l1.

Anhaltswerte aus dem Lauf vom 2026-08-31 (2 Jahre Training, 10 Monate Eval,
5–6 Jobs parallel): `dcrnn_idw_alt` ~0,8 h, `dcrnn` ~1–2 h,
`dcrnn_nwp_hist` ~5–7 h, `mtgnn_nwp` ~7–11 h, `mtgnn_nwp_hist` ~9–16 h je Lauf.
Der Testzeitraum ist hier 12 statt 10 Monate, rechne entsprechend etwas mehr.
MTGNN ist der Engpass — zuerst starten.

Das ICON-D2-Laden (2448 Parquets je Lauf) dauert ~10–15 min vor der ersten
Epoche; parallele Läufe verlangsamen sich dabei gegenseitig. Nicht als Hänger
missdeuten.

## Auswertung

Filterung wie in `docs/evaluation_results.md` §14–18: imputierte Zielstunden
ausschließen, per Station RMSE, dann Stationsmittel. Bausteine in
`geostatistics/stdrun/make_stdhp_figures.py`: `norm_station`,
`build_imputation_mask`, `_lookup_imputed`.

Als Vorlage nutzbar (nicht ungeprüft übernehmen — beide sind auf die vorige,
fold-basierte Auswertung zugeschnitten):

- `scripts/eval_expwin.py` — Filterung, per-Station-RMSE, Wilcoxon + Holm
- `scripts/eval_testyear.py` — dieselbe Filterung plus zehn Abbildungen
  (Balken-RMSE, Fold-Streuung, gepaarte Differenz-Boxplots, Fehler über
  Prognosehorizont, über Windklasse, über Monat, Scatter, Skill-Verteilung,
  Val-gegen-Test). Läuft auf l1 (ältere pandas/matplotlib): kein
  `groupby.apply(include_groups=)`, Plot-Argumente als numpy-Arrays.

Ohne Folds entfällt die Fold-Streuung; die Signifikanz zwischen den Armen läuft
dann gepaart über die 50 Teststationen statt über 153.

Ergebnis als neuen Abschnitt in `docs/evaluation_results.md` dokumentieren:
Tabelle Arm × RMSE mit ICON-D2 und Persistenz als Referenz, Wilcoxon-Matrix
zwischen den Armen, Abbildungen unter `figures/`.

## Randbedingungen

- Der Nutzer will bei Blockern **gefragt** werden, nicht durch ein Ersatzdesign
  überrascht. Wenn etwas den Zuschnitt unmöglich macht: Optionen mit Kosten
  benennen und nachfragen, auch unter Termindruck.
- `sudo` verlangt auf beiden Hosts ein Passwort — solche Schritte dem Nutzer als
  Kommando geben.
- Hintergrundjobs in `screen` starten (überleben SSH-Abbruch und Session-Ende).

---

## Ausführungsstand (2026-09-02)

Ausgeführt auf `l1`, GPU 0/1/2 (3–7 tragen Fremdlast), Start 09:28 UTC.

### Zuschnitt — Abweichung vom Rezept oben

Nutzerauftrag vom 2026-09-02: **ein Modell je Arm**, aber die beiden HIST-Arme
mit **Retraining alle 4 Monate bei wachsendem Trainingsfenster**, jeweils
getestet auf den folgenden 4 Monaten. Damit sind es **9 statt 5 Läufe**:

| Arm | Läufe | Trainingsfenster | Testfenster |
|---|---|---|---|
| `dcrnn`, `dcrnn_idw_alt`, `mtgnn_nwp` | je 1 | alles vor 2025-08-01 | 2025-08-01 … 2026-07-31 |
| `dcrnn_nwp_hist`, `mtgnn_nwp_hist` | je 3 | < 2025-08-01 / < 2025-12-01 / < 2026-04-01 | Aug–Nov 25 / Dez 25–Mär 26 / Apr–Jul 26 |

Im `--test-mode` ist `test_start` die einzige Trennlinie: Training ist alles
davor, Eval ist `[test_start, test_end]`. Ein Retrain-Schritt ist deshalb nur
ein Paar `test_start`/`test_end` — kein Codeeingriff. Die drei Chunks sind über
`run_time` disjunkt und lückenlos (der Laufzeit-Cutoff ist `t_run > test_end`,
und `run_hours` enthält keine 00-Uhr-Läufe), lassen sich also zum vollen
Testjahr zusammenlegen und gegen die Ein-Modell-Arme stellen.

Warum nur die HIST-Arme nachtrainiert werden: § 17.1/§ 9 in
`docs/expanding_window_retrain_handoff.md` — mehr Trainingsdaten helfen
messbar nur `dcrnn_nwp_hist` (−0,0236 RMSE) und `mtgnn_nwp_hist` (−0,0105).
**Preis dieses Zuschnitts:** der Vergleich HIST gegen Nicht-HIST vermengt Arm
und Trainingsprotokoll. Die Tabelle je 4-Monats-Fenster
(`data/test_results/testmode_by_chunk.csv`) macht das sichtbar, hebt es aber
nicht auf.

### Configs

`configs/testmode/{full,step1,step2,step3}/config_wind_<arm>_fold1.yaml`,
erzeugt von `scripts/gen_testmode_configs.py` (idempotent, nur
`test_start`/`test_end` geändert). Dateinamen absichtlich identisch zur Basis —
die Optuna-Auflösung strippt `_fold\d+$` nur am Zeilenende. Geprüft: 153 Train-
und 50 Teststationen je Config, Überschneidung 0; alle fünf Studies lösen auf
(Bestwerte 200/134/174/147/137 Trials, seit 2026-08-24 eingefroren).

### Queue

`~/queue_scripts/testmode_worker.sh` + `testmode_tasks.txt` auf l1, drei Worker
in `screen` (`tm_w1..3`) auf GPU 0/1/2, gemeinsame Liste per `flock`.
Gegenüber `testyear_worker.sh` geändert: die Liste wird **in jeder Runde neu
gelesen** und ein Worker endet bei leerer Liste nicht, sondern wartet — Aufgaben
dürfen im Betrieb angehängt werden (nur anhängen, nie umsortieren, der Cursor
ist ein globaler Index). Stoppen: `touch ~/queue_scripts/.testmode_stop`.
Fortschritt: `~/queue_scripts/testmode_status.log`,
Joblogs `logs/testmode/<rawname>.log`.

| Artefakt | Muster |
|---|---|
| Config | `configs/testmode/<sub>/config_wind_<arm>_fold1.yaml` |
| Modell | `models/wind_<arm>_fold1_<dcrnn\|mtgnn>_testmode[_s<S>].pt` |
| Metriken | `data/test_results/testmode_<arm>[_s<S>].csv` |
| Rohvorhersagen | `data/raw_preds/testmode_<arm>[_s<S>]_raw.parquet` |

### Zwei Blocker, die im Rezept oben nicht standen

1. **`l1` trug den Code von vor der TFT-Umstellung.** Der dortige Baum stand auf
   `3d041b7`; 23 Python-Dateien waren älter, und fünf Module fehlten ganz —
   darunter `geostatistics/shared/resolution.py`, das `train_dcrnn.py`
   importiert. Abgeglichen per rsync aus dem l2-Arbeitsbaum (Backup:
   `~/backup_pre_testmode_20260902/py_backup.tgz`). Configs blieben unangetastet
   (l1 führt `/mnt/nvme1`, l2 `/mnt/lambda1/nvme1`).

2. **`wind_direction` reichte nicht bis ans Testjahresende.** Der Satz oben
   „0 Stationen mit Restlücken" gilt für `wind_speed`; der KNN-Cache für
   `wind_direction` endete am 2026-07-14 23:00 UTC und ließ mit
   `test_end: '2026-07-31'` 614 NaN an 26 Stationen stehen — mit
   `handle_nans: break` ein Abbruch für 5 der 9 Läufe. Der Nutzer hat sich gegen
   ein Zurücknehmen der Testgrenze und für die Neurechnung des Caches
   entschieden: `docs/imputation_knn_regen_20260902.md`. Danach 0 NaN in allen
   9 Auditfenstern.

   Nebenbefund aus demselben Vorgang: `train_mtgnn.py`/`train_wavenet.py`
   prüften im `--test-mode` bis ans Datenende statt bis `test_end` und schlugen
   auf dem 2-Tage-Schwanz an, den kein Run-Paar als Ziel hat. An
   `train_dcrnn.py` angeglichen (§ 4 des genannten Dokuments).

**Preflight vor dem Start:** `scripts/preflight_testmode.py` lädt für alle 9 Configs
Messreihen + Imputationen (ohne NWP, ohne Training) und meldet die NaN-Bilanz je
Spalte im jeweiligen Auditfenster. Wiederholbar; das ist der Schritt, der beide
`wind_direction`-Befunde vor dem ersten GPU-Lauf sichtbar gemacht hätte.

### Auswertung

`scripts/eval_testmode.py` — Filterung wie § 14–18, per-Station-RMSE,
Stationsmittel, Wilcoxon + Holm über die 50 Teststationen. Zwei
Tabellenvarianten (je Arm auf eigenen Zeilen / auf dem gemeinsamen Schnitt über
`station_id, run_time, horizon`), dazu die Tabelle je 4-Monats-Fenster und acht
Abbildungen unter `figures/testmode/`. Ergebnis kommt als § 19 in
`docs/evaluation_results.md`.

---

## Entscheidung 2026-09-05: Early Stopping auf den Teststationen bleibt

`--test-mode` setzt `val_ids = test_files`, das vorzeitige Abbrechen und die Wahl des
gespeicherten Checkpoints laufen also auf den 50 Teststationen ueber das Testfenster
(§ 19.5 in `evaluation_results.md`). Das ist formal eine Leckage der Teststationen in
die Epochenwahl eines einzelnen Laufs. **Nutzerentscheidung: so belassen, im Setting
vernachlaessigbar, muss im Paper nicht eigens genannt werden.**

Alternative, die dafuer verworfen wurde: `--fixed-epochs N` (seit Commit d19bed5 in
`train_dcrnn.py`/`train_mtgnn.py`; N = gespeicherte beste Epoche der Validierungs-Retrains,
A 5/2/8, D' 9/17/26, DCRNN+HIST 45/25/21, MTGNN 55/53/82, MTGNN+HIST 109/79/53). Neun
solche Laeufe wurden am 2026-09-05 gestartet und nach der Entscheidung abgebrochen; Reste
unter `archiv/testmode_fixed_epochs_abgebrochen_20260905/` auf l1 und l2. Die Option
bleibt im Code, wird aber fuer das Paper nicht benutzt.

Unabhaengig davon (und weiter gueltig): `configs/testmode/fullhist/` wertet die
Schritt-1-HIST-Checkpoints ueber das volle Testjahr aus (`testmode_*_nwp_hist_once`,
gerechnet 2026-09-05 auf l1), MOS und TFT laufen im Testjahr auf l2
(`configs/baselines/config_wind_mos_testyear_fold1.yaml`, `configs/tft_bc/*_testyear.yaml`),
Seed-Wiederholungen A/D' auf ws (`~/queue_scripts/seedrep_worker.sh`). Sammeln und
exportieren: `bash scripts/collect_testyear_20260905.sh && ./frcst/bin/python scripts/export_paper_metrics.py all`.
