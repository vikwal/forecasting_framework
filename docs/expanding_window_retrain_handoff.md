# Handoff: Expanding-Window-Retraining der 5 Flagschiff-Arme

**Gestartet am 2026-08-27 20:20** — 45 Configs erzeugt, 2 Worker laufen auf
GPU 2 (`w-lambdablade2`). Details siehe § 8 "Ausführungsstand" am Ende.

## 1. Ziel

Für 5 bereits bekannte, HPO-getunte Modelle prüfen, ob periodisches Nachtrainieren
mit wachsendem Trainingsfenster (statt einmaligem Training auf dem festen
`train < 2024-08-01`-Fenster) die Validierungsleistung verbessert. Für jeden der
3 bestehenden räumlichen Folds wird **3×** trainiert, jedes Mal mit einem länger
werdenden Trainingsfenster und einem neuen, direkt daran anschließenden
Validierungsfenster (Expanding-Window-Walk-Forward).

## 2. Die 5 Arme und ihre bestehenden Fold-Configs

| Arm | Bezeichnung im Auftrag | Basis-Configs (fold1/2/3) | Modellskript |
|---|---|---|---|
| DCRNN GRID | `dcrnn` (A) | `configs/dcrnn/config_wind_dcrnn_fold{1,2,3}.yaml` | `geostatistics/train_dcrnn.py` / `get_test_results_dcrnn.py` |
| DCRNN IDW | `dcrnn_idw_alt` (D') | `configs/dcrnn/config_wind_dcrnn_idw_alt_fold{1,2,3}.yaml` | dito |
| DCRNN GRID+HIST | `dcrnn_nwp_hist` | `configs/dcrnn/config_wind_dcrnn_nwp_hist_fold{1,2,3}.yaml` | dito |
| MTGNN GRID | `mtgnn_nwp` | `configs/mtgnn/config_wind_mtgnn_nwp_fold{1,2,3}.yaml` | `geostatistics/train_mtgnn.py` / `get_test_results_mtgnn.py` |
| MTGNN GRID+HIST | `mtgnn_nwp_hist` | `configs/mtgnn/config_wind_mtgnn_nwp_hist_fold{1,2,3}.yaml` | dito |

Alle 15 Basis-Configs haben identische Zeitgrenzen: `val_start: '2024-08-01'`,
`test_start: '2025-08-01'` (train = alles davor, kein separates `train_start`-Feld
— train ist implizit "alles vor `val_start`"). Stationsaufteilung (`files`/
`val_files`) bleibt für alle 3 Zeitschritte je Fold identisch — nur die beiden
Zeitgrenzen ändern sich.

## 3. Die 3 Zeitschritte (identisch für alle 3 Folds, alle 5 Arme)

| Schritt | `val_start` | `test_start` | Val-Fenster (informell) |
|---|---|---|---|
| 1 | `2024-08-01` | `2024-12-01` | Aug–Nov 2024 |
| 2 | `2024-12-01` | `2025-04-01` | Dez 2024–Mär 2025 |
| 3 | `2025-04-01` | `2025-08-01` | Apr–Jul 2025 |

Schritt 3 endet exakt an der ursprünglichen `test_start`-Grenze (`2025-08-01`) —
der zurückgehaltene Testsatz (alles ab da) wird **nicht** angerührt. Trainingsdaten
sind automatisch "alles vor `val_start`" (keine `train_start`-Angabe nötig), decken
sich mit der bereits verifizierten Datenverfügbarkeit (Stationsdaten ab 2023-07-24,
ICON-D2/ECMWF für diesen Zeitraum vollständig — geprüft in dieser Session, kein
Blocker).

**Vorgehen:** 15 Basis-Configs × 3 Zeitschritte = **45 neue Config-Dateien**
erzeugen (nur `val_start`/`test_start` ändern, Rest 1:1 kopieren), z. B.
`config_wind_dcrnn_fold1_step1.yaml`, `..._step2.yaml`, `..._step3.yaml` usw.
(Namensschema selbst wählen, Hauptsache konsistent und in `docs/evaluation_results.md`
dokumentiert.)

> **Umgesetzt anders — siehe § 8.1:** der hier vorgeschlagene `_stepN`-Suffix
> bricht die Optuna-Study-Auflösung (`_fold\d+$` wird nur am Zeilenende
> gestrippt). Die Schritte werden stattdessen über das Verzeichnis
> `configs/expwin/step<S>/` unterschieden.

## 4. Hyperparameter: kein neues HPO

Jeder Zeitschritt nutzt die **bereits vorhandenen** HPO-Bestwerte des jeweiligen
Arms — `--hpo-study auto` bei DCRNN/MTGNN lädt automatisch den besten Trial aus
Optuna (Postgres, `OPTUNA_STORAGE`). Es wird nur die Datenmenge variiert, nicht
die Architektursuche — das entspricht dem expliziten Nutzerauftrag ("wir müssen
jeden Fold 3× neu trainieren", nicht "9× HPO'en").

## 5. Rezept (analog zu den bereits abgeschlossenen Single-Window-Retrains, siehe
`docs/evaluation_results.md` §10/§13.3/§16)

```bash
# DCRNN-Arme:
python geostatistics/train_dcrnn.py --config <step_config> --suffix <suffix> --hpo-study auto
python geostatistics/get_test_results_dcrnn.py -m <model_name> -c <step_config> --hpo-study auto --raw-out-name <name>

# MTGNN-Arme:
python geostatistics/train_mtgnn.py --config <step_config> --suffix <suffix> --hpo-study auto
python geostatistics/get_test_results_mtgnn.py -m <model_name> -c <step_config> --hpo-study auto --raw-out-name <name>
```

`get_test_results_*.py` **ohne** `--test-mode` = Val-Modus (die 51 nie gesehenen
Zielstationen je Fold) — das ist das gewollte Verhalten hier, nicht der finale Test.

## 6. Umfang und Ressourcenverteilung

**5 Arme × 3 Folds × 3 Zeitschritte = 45 Trainings+Eval-Läufe.**

Auftrag: "Nutze alle verfügbaren Ressourcen, verteile auf allen verfügbaren GPUs,
l1, l2 (= dieser lokale Host `w-lambdablade2`) und ws." 14 GPUs insgesamt normalerweise
verfügbar (lokal 4× 80GB, `l1` 8× 49GB, `ws` 2× 24GB) — **GPU-Auslastung vor dem
Start neu prüfen**, beim Schreiben dieses Dokuments waren alle 3 Hosts bereits
ausgelastet (lokal 3/4, l1 8/8, ws 2/2 GPUs belegt, unabhängig von dieser Session).
`nvidia-smi` lokal, `ssh l1 nvidia-smi`, `ssh ws nvidia-smi`.

Empfehlung: pro GPU eine Warteschlange (Bash-Skript, `screen -dmS <name> ...`),
die die zugeteilten Läufe sequenziell abarbeitet (train → eval je Config), analog
zu den bereits in dieser Session verwendeten `queue_scripts/run_*_eval_queue_local.sh`
(liegen unter `~/queue_scripts/` auf diesem Host, als Vorlage nutzbar).

## 7. Nach Abschluss

- Gefilterte Stationsmittel-RMSE berechnen (`build_imputation_mask`/`_lookup_imputed`
  aus `geostatistics/stdrun/make_stdhp_figures.py`, Muster siehe §14–16 in
  `docs/evaluation_results.md`).
- Ergebnis als neuer Abschnitt (§17 o. ä.) in `docs/evaluation_results.md` dokumentieren:
  Tabelle Arm × Zeitschritt × Fold, dazu die Kernfrage beantworten — wird die
  Val-RMSE mit wachsendem Trainingsfenster über die 3 Zeitschritte hinweg
  systematisch besser?

---

## 8. Ausführungsstand (2026-08-27)

### 8.1 Configs

45 Configs unter `configs/expwin/step{1,2,3}/`. **Die Dateinamen sind absichtlich
identisch zu den Basis-Configs** (`config_wind_<arm>_fold<N>.yaml`) und nur über
das Verzeichnis unterschieden — nicht, wie in § 3 vorgeschlagen, über einen
`_stepN`-Suffix. Grund: `train_dcrnn.py:338` / `train_mtgnn.py:277` leiten den
Optuna-Study-Namen aus dem Config-Stem ab und strippen dabei `_fold\d+$` **nur am
Zeilenende**. Ein Name wie `config_wind_dcrnn_fold1_step1.yaml` würde nach
Study `cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_fold1_step1` suchen und scheitern.
Unterschieden werden die Läufe stattdessen über `--suffix ew_s<S>_f<N>`:

| Artefakt | Muster |
|---|---|
| Config | `configs/expwin/step<S>/config_wind_<arm>_fold<N>.yaml` |
| Modell | `models/wind_<arm>_fold<N>_<dcrnn\|mtgnn>_ew_s<S>_f<N>.pt` |
| Trainings-Log | `logs/train_<m>_config_wind_<arm>_fold<N>_ew_s<S>_f<N>.log` |
| Queue-Log (train+eval) | `logs/expwin/expwin_<arm>_s<S>_fold<N>.log` |
| Roh-Ergebnis | `--raw-out-name expwin_<arm>_s<S>_fold<N>` |

Geändert wurde je Config ausschließlich `val_start`/`test_start`; `test_end`
(`2026-03-31`) und die `hpo`-Sektion bleiben unangetastet. Alle 5 Optuna-Studies
wurden vor dem Start in Postgres verifiziert (200 / 134 / 174 / 147 / 137 Trials).

### 8.2 Ausführung

Nur **GPU 2** ist frei (GPU 0/1/3 laufen mit je 2 Fremdprozessen aus anderen
Baustellen). Statt fester Queues pro Worker teilen sich **2 Worker eine gemeinsame
Aufgabenliste** und reservieren per `flock` die nächste Zeile — das balanciert
sich selbst, was hier wichtig ist, weil `dcrnn_nwp_hist` allein 45 % der
Gesamtrechenzeit ausmacht.

```bash
screen -dmS expwin_w1 bash /home/viktor/queue_scripts/expwin_worker.sh 1 2
screen -dmS expwin_w2 bash /home/viktor/queue_scripts/expwin_worker.sh 2 2
```

- Aufgabenliste: `~/queue_scripts/expwin_tasks.txt` (45 Zeilen, **schritt-major** —
  erst alle Arme/Folds für Schritt 1, dann 2, dann 3; innerhalb eines Schritts
  die teuren Arme zuerst, damit sie nicht als langer Schwanz enden)
- Fortschritt: `tail -f ~/queue_scripts/expwin_status.log`
- Cursor: `~/queue_scripts/.expwin_cursor` (Index der nächsten Aufgabe)
- Sauber stoppen (nach dem jeweils laufenden Lauf): `touch ~/queue_scripts/.expwin_stop`
- Ein fehlgeschlagenes Training bricht die Queue **nicht** ab, sondern wird in
  `expwin_status.log` als `FEHLER #<idx>` protokolliert und übersprungen.

### 8.3 Laufzeitschätzung (aus den Logs der bisherigen Retrains)

Referenz sind die jüngsten Läufe mit den aktuellen HPO-Bestwerten (24./25.08.,
1 Job/GPU); die Juni-Läufe sind unbrauchbar (3 Folds parallel auf einer GPU, und
die DCRNN-Konfiguration hat sich seither stark verändert: `nwp_hist` ging von
0,9 auf 7,1 min/Epoche).

| Arm | min/Epoche | Epochen | 1 Lauf (Basisfenster) | 3 Folds × 3 Schritte |
|---|---|---|---|---|
| DCRNN GRID | 1,7–1,9 | 25–59 | ~1,4 h | 14,0 h |
| DCRNN IDW (D') | *geschätzt = GRID* | — | ~1,4 h | 14,0 h |
| DCRNN GRID+HIST | 6,8–7,3 | 36–60 | ~5,3 h | 52,3 h |
| MTGNN GRID | 1,08–1,10 | 68–97 | ~1,4 h | 14,3 h |
| MTGNN GRID+HIST | 1,40–1,43 | 68–124 | ~2,2 h | 22,0 h |
| **Summe** | | | | **117 GPU-h** |

Die 3 Zeitschritte kosten **nicht** 3× einen Basislauf: das Train-Fenster wächst
(374 → 496 → 617 Tage), das Val-Fenster schrumpft aber von 12 auf 4 Monate. Mit
Epochenkosten ∝ `train_tage + 0,4·val_tage` ergibt sich pro Fold Faktor
0,81 / 1,05 / 1,28, Summe **3,14**.

Bei 2 parallelen Jobs auf einer GPU und Contention-Faktor 1,3 / 1,6 / 2,0:
**76 h (3,2 d) — 93 h (3,9 d) — 117 h (4,9 d)**, realistisch **~4 Tage**.

Für `dcrnn_idw_alt` existiert kein einziges Trainingslog (nur HPO) — die 14 h
dort sind vom GRID-Arm übertragen und können danebenliegen, falls die
IDW-Adjazenz dichter ist als die Delaunay-Triangulation.

---

## Unabhängige Baustelle auf demselben Host — NICHT stören

Parallel zu diesem (noch nicht begonnenen) Retraining-Auftrag laufen auf diesem
Host mehrere **unabhängige** Hintergrundjobs zu einem separaten DWD-Datenbank-
Koordinaten-Bug (`ST_X`/`ST_Y` in `multilevelfields`/`singlelevelfields` waren
vertauscht gespeichert, Root Cause in `~/Work/NWP/DWD/write_db.py::insert_data()`
gefunden und gefixt). Diese haben mit dem Retraining-Auftrag nichts zu tun, sollten
aber nicht unterbrochen werden:

| Screen-Session | Zweck | Stand beim Schreiben |
|---|---|---|
| `fix_sl_swap` | Korrigiert `singlelevelfields`-Koordinaten, komplette Historie | Läuft, ETA ~6h ab Sessionsende |
| `fix_march_v3` | Füllt gelöschte `multilevelfields`-Zeilen für 2026-03-07..31 neu aus GRIBs | Läuft, ETA ~22h ab Sessionsende |

Status prüfen: `tail -f ~/fix_singlelevelfields_geom_swap.log` bzw.
`~/fix_march_collision_window_v3.log`. Falls diese beim Übernehmen bereits fertig
sind (Zeile `DONE.`), betrifft das nur die GNN-ICON-D2-Parquet-Pfade indirekt
(bessere Datenverfügbarkeit für spätere Testset-Auswertungen) — **kein
Blocker für dieses Retraining**, da alle benötigten Zeiträume (Juli 2023 bis
Juli 2025) bereits vor dem Bug-Fenster (~März–April 2026) liegen und laut
Session-Prüfung durchgehend sauber waren.

Offen auf DWD-Seite (nicht Teil dieses Auftrags, nur zur Info): Meghnanegis
`write_db.py`-Fix ist vom Nutzer bereits angewendet, ihr `prefect-dwd-sl-pipeline.service`
muss noch neugestartet werden (`sudo systemctl restart ...`, braucht Passwort,
nicht automatisierbar).

### 8.4 Gemessene Laufzeiten (Stand 2026-08-28 07:15, 8 von 45 Läufen fertig)

Alle bisherigen Läufe mit `train=0 eval=0`. Gemessen mit **2 Jobs parallel** auf
GPU 2, Wall-Clock je Job inkl. Eval:

| Arm | min/Epoche | Epochen | Wall-Clock je Lauf (3 Folds) |
|---|---|---|---|
| DCRNN GRID+HIST | 4,02–4,17 | 34 / 35 / 44 | 2,59 / 2,60 / 3,31 h |
| MTGNN GRID+HIST | 1,81–1,92 | 99 / 93 / 77 | 3,14 / 2,56 / 2,94 h |
| MTGNN GRID | 1,45–1,50 | 85 / 84 / … | 2,16 / 2,15 / … h |
| DCRNN GRID | 1,22 | läuft | — |

**Die Ex-ante-Schätzung in § 8.3 lag für DCRNN GRID+HIST deutlich zu hoch:**
4,0 statt 7,1 min/Epoche — und das *trotz* Contention durch den zweiten Job.
Ursache ist nicht ein geänderter Hyperparametersatz (Optuna-Trial #145 vom
2026-08-22 ist derselbe wie beim Referenzlauf am 24.08.), sondern das
Val-Fenster: es schrumpft von 12 auf 4 Monate, und bei DCRNN dominiert die
Validierung die Epochenkosten. Aus beiden Messpunkten folgt ein Val-Gewicht von
**w ≈ 1,9 … 5,7** statt der in § 8.3 angesetzten 0,4 — die Validierung kostet
pro Tag also ein Mehrfaches des Trainings. Plausibel, weil die Validierung die
51 zurückgehaltenen Stationen autoregressiv ohne Teacher Forcing dekodiert.
Bei MTGNN ist es umgekehrt (w ≈ 0,3, Contention-Faktor ≈ 1,6) — dort dominiert
das Training.

Konsequenz für die Hochrechnung: da das Val-Fenster über alle 3 Schritte
konstant 4 Monate bleibt und nur das Train-Fenster wächst, steigen die Kosten
über die Schritte **schwächer** als in § 8.3 angenommen — für DCRNN Faktor
1,00 / 1,20 / 1,40, für MTGNN 1,00 / 1,30 / 1,59.

**Revidierte Gesamtschätzung: ~113 Slot-Stunden = ~57 h Wall-Clock**, also
**~2,4 Tage statt der geschätzten ~4**. Voraussichtliches Ende: **30.08.2026,
Vormittag**. Unsicher bleiben die beiden bis dahin ungemessenen Arme
DCRNN GRID und DCRNN IDW (zusammen ~6,6 der 30 Slot-Stunden von Schritt 1).

## 9. Abgeschlossen (2026-08-29 22:57)

45/45 Läufe mit `train=0 eval=0`, 0 Fehler, Gesamtlaufzeit **50,6 h** (Schätzung
in § 8.4 war 57 h, die ursprüngliche in § 8.3 93 h). Artefakte vollständig:
45 Modelle, 45 Metrik-CSVs (`data/test_results/expwin_*.csv`), 45 Roh-Parquets
(`data/raw_preds/expwin_*_raw.parquet`).

**Auswertung: `scripts/eval_expwin.py` → `docs/evaluation_results.md` § 17.**

Wichtig für das Verständnis der Ergebnisse: der in § 3 spezifizierte Aufbau
beantwortet die Kernfrage aus § 7 **nicht direkt**, weil die drei Zeitschritte
verschiedene Val-Fenster (und damit Jahreszeiten) haben. Gelöst über einen
saisongleichen Ausschnitt der Single-Window-Retrains als Kontrolle plus
Differenz-in-Differenzen gegen Schritt 1, dessen Trainingsfenster mit dem der
Kontrolle identisch ist und der damit als Nullmessung dient. Details in § 17.1.

Kernbefund: mehr Trainingsdaten helfen nur `dcrnn_nwp_hist` (−0.0236 RMSE) und
`mtgnn_nwp_hist` (−0.0105), und erst bei +8 Monaten. Die drei Arme ohne
NWP-Historie zeigen keinen signifikanten Effekt.
