# Handoff — Stand 2026-09-04, 07:30 CEST

> **ERLEDIGT.** Alle neun Läufe sind am 2026-09-04 um 01:20 fertig geworden,
> **9/9 mit `train=0 eval=0`**, Gesamtlaufzeit 12 h 50 min. Ergebnisse
> zusammengeführt, ausgewertet und als **§19 in `evaluation_results.md`**
> dokumentiert. Kernbefund: die beiden Arme mit NWP-Historie und Retraining
> liegen bei RMSE 1,12 / 1,14 gegen 1,42–1,44 der drei übrigen und 1,51 für
> ICON-D2; DCRNN und MTGNN sind untereinander nicht unterscheidbar. Offen ist
> nur noch, ob die beiden HIST-Arme zusätzlich **ohne** Retraining über das
> volle Jahr gerechnet werden sollen (§19.5) — zwei Läufe.
>
> Der Rest dieses Dokuments beschreibt den Zustand *während* der Läufe und
> bleibt als Betriebsanleitung für einen Neustart stehen.

Detaildokumente: Zuschnitt und Vorgeschichte in
[handoff_testmode.md](handoff_testmode.md), die beiden Imputationsumstellungen in
[imputation_tft_switch.md](imputation_tft_switch.md),
[imputation_knn_regen_20260902.md](imputation_knn_regen_20260902.md) und
[imputation_richtung_tft_20260903.md](imputation_richtung_tft_20260903.md).

---

## 1. Was gerade läuft

Neun Trainings- und Auswertungsläufe, gestartet am **2026-09-03 um 12:31/12:32**,
verteilt auf drei Hosts. Je ein Lauf pro GPU, ein `screen` pro Lauf.

| Host | GPU | Arm | Testfenster | Roh-Ergebnis |
|---|---|---|---|---|
| l1 | 1 | `mtgnn_nwp_hist` step1 | Aug–Nov 2025 | `testmode_mtgnn_nwp_hist_s1` |
| l1 | 2 | `mtgnn_nwp_hist` step2 | Dez 25–Mär 26 | `testmode_mtgnn_nwp_hist_s2` |
| l1 | 3 | `mtgnn_nwp_hist` step3 | Apr–Jul 2026 | `testmode_mtgnn_nwp_hist_s3` |
| l1 | 4 | `mtgnn_nwp` | ganzes Testjahr | `testmode_mtgnn_nwp` |
| l1 | 5 | `dcrnn_nwp_hist` step1 | Aug–Nov 2025 | `testmode_dcrnn_nwp_hist_s1` |
| l1 | 7 | `dcrnn_nwp_hist` step2 | Dez 25–Mär 26 | `testmode_dcrnn_nwp_hist_s2` |
| l2 | 3 | `dcrnn_nwp_hist` step3 | Apr–Jul 2026 | `testmode_dcrnn_nwp_hist_s3` |
| ws | 0 | `dcrnn` | ganzes Testjahr | `testmode_dcrnn` |
| ws | 1 | `dcrnn_idw_alt` | ganzes Testjahr | `testmode_dcrnn_idw_alt` |

GPU-Auswahl nach der Vorgabe „frei oder über 40 % freie Kapazität": alle 14
Karten erfüllen das, für neun Läufe braucht es neun. Genommen wurden zuerst die
tatsächlich unbelasteten, aufgefüllt mit den am wenigsten belasteten. Unberührt
blieben l1 GPU 0 (100 % Fremdlast), l1 GPU 6 und l2 GPU 0–2 (81–88 % Fremdlast).

**Fortschritt um 14:00** — l1 bei Epoche 9–13 von 200, l2 bei Epoche 3, ws gerade
mit dem Datenladen fertig. Gemessene Epochendauer: MTGNN ~6 min, DCRNN GRID+HIST
~8,5 min (l1) bzw. ~8,8 min (l2, praktisch gleich schnell trotz Fremdlast).
Early Stopping mit `patience: 15`; vergleichbare Läufe endeten nach 34–124
Epochen. Daraus **grob 6–12 h je Lauf**, danach je ein Eval-Schritt.

> **ws lädt langsam, das ist kein Hänger.** Die 3248 bzw. 4872 ICON-D2-Parquets
> kommen dort über NFS von l1 — 1 h 24 min statt 10–15 min lokal, während l1
> dieselben Dateien für sechs eigene Läufe liest. Einmalig je Lauf, vor der
> ersten Epoche.

## 2. Nachsehen, stoppen, fortsetzen

```bash
# Fortschritt (je Host)
ssh l1 'cat ~/queue_scripts/testmode_status.log'      # gleiches auf l2 (lokal) und ws
ssh l1 'cd ~/Work/forecasting_framework && grep -aoE "Epoch +[0-9]+/[0-9]+" logs/testmode/<name>.log | tail -1'

# Was fertig ist
ls data/raw_preds/testmode_*_raw.parquet     # je Lauf eine Datei, 9 am Ende
ls data/test_results/testmode_*.csv

# Sauber stoppen (nach dem jeweils laufenden Job)
ssh l1 'touch ~/queue_scripts/.testmode_stop'         # je Host einzeln

# Sofort stoppen
ssh l1 'for s in $(screen -ls | grep -o "tm_g[0-9]*"); do screen -S $s -X quit; done'
```

Der Worker (`scripts/testmode_worker.sh`, jetzt im Repo) hängt einen
fehlgeschlagenen Lauf ans Listenende zurück statt ihn zu verwerfen und pausiert
dann 2 min; **drei Fehlschläge in Folge setzen die Stop-Datei** und beenden alle
Worker des Hosts. Das ist die Lehre aus dem 2026-09-02, siehe § 5.

Ein abgebrochener Lauf wird also von selbst wiederholt. Wenn ein Host komplett
steht, reicht ein Neustart der Worker — der Cursor
(`~/queue_scripts/.testmode_cursor`) merkt sich die Position, die Aufgabenliste
darf **nur angehängt, nie umsortiert** werden.

## 3. Wenn alle neun fertig sind

Ergebnisse liegen je Host unter `data/raw_preds/` und `data/test_results/`. Sie
müssen erst **auf einen Host zusammengeführt** werden (Vorlage:
`scripts/collect_and_aggregate.sh`), dann:

```bash
python scripts/eval_testmode.py
```

Das Skript filtert wie `docs/evaluation_results.md` §14–18 (imputierte
Zielstunden raus, RMSE je Station, dann Stationsmittel), rechnet Wilcoxon + Holm
über die 50 Teststationen und schreibt acht Abbildungen nach `figures/testmode/`.
Es liefert zwei Tabellenvarianten — je Arm auf seinen eigenen Zeilen und auf dem
gemeinsamen Schnitt über `(station_id, run_time, horizon)` —, weil die Arme
unterschiedlich viele Run-Paare behalten. Dazu eine Tabelle je 4-Monats-Fenster.
Ergebnis gehört als **§19** in `docs/evaluation_results.md`.

**Beim Interpretieren nicht vergessen:** die drei Arme ohne NWP-Historie laufen
mit *einem* Modell über das ganze Jahr, die beiden HIST-Arme mit Retraining alle
4 Monate. Der Vergleich HIST gegen Nicht-HIST vermengt damit Arm und
Trainingsprotokoll. Die Chunk-Tabelle macht das sichtbar, hebt es nicht auf. Wer
es sauber trennen will, braucht zusätzlich die beiden HIST-Arme *ohne*
Retraining über das volle Jahr (2 weitere Läufe).

## 4. Die Imputation — was sich in zwei Tagen geändert hat

Die Kette hat sich zweimal bewegt. Der Reihe nach, weil die Zwischenschritte
sonst verwirren:

**2026-09-02, Windgeschwindigkeit.** `interpol/wind` wurde auf die Vorhersage
eines TFT-Abschlussmodells umgestellt (Spalte `imputed` statt `rk_pred`), der
ERA5-OLS-Pfad damit abgelöst. → `imputation_tft_switch.md`

**2026-09-02, der KNN-Cache.** Die Windrichtung kam damals noch aus
`knnimputer/wind`, und dessen Abdeckung endete am **2026-07-14 23:00** — mit
`test_end: 2026-07-31` blieben 614 NaN an 26 Stationen, was den NaN-Audit
abbrechen ließ. Auf Entscheidung des Nutzers wurde der Cache mit
`geostatistics/regen_knn_imputation.py` **neu gerechnet** (7,5 min), reicht
seither bis 2026-09-01. Dabei:

- Der Stationssatz wuchs von 203 auf **204** — Emden (`05839`) hat inzwischen
  eine Rohdatei. Die Station steht in keiner Config-Stationsliste und wirkt nur
  als zusätzliche Spalte im Imputer-Fit.
- Der Dateiname trägt den Stationssatz-Hash: neu `611c3831`, alt `67558851`.
  **Der alte Stand musste aus dem Verzeichnis heraus** — `load_knn_imputation`
  nimmt `sorted(glob(...))[-1]`, und `611c3831` sortiert *vor* `67558851`; die
  neuen Dateien wären sonst danebengelegen, ohne je gelesen zu werden. Alter
  Stand: `knnimputer/wind_vor_regen_20260902`.
- Der Neu-Fit über einen längeren Zeitraum verschiebt auch **historische**
  Richtungswerte leicht. Läufe von vor dem 2026-09-02 sind nicht mehr auf
  bitgleichen Eingangsdaten nachrechenbar.

→ `imputation_knn_regen_20260902.md`

**2026-09-03, die Windrichtung — und damit ist der KNN-Cache für Wind raus.**
Es gibt jetzt `interpol/wind_richtung`: derselbe Baum, aber mit `imputed_dir`
(Richtung in Grad) neben `imputed`. Beide Läufe stammen aus einem gemeinsamen
Abschlussmodell. Die neun Configs zeigen darauf, und **`knnimputer_path` ist dort
auskommentiert** — auf ausdrückliche Vorgabe: kein Rückfall, lieber soll der Lauf
abbrechen. Der Code lässt zusätzlich in allen elf Aufrufern die Spalten aus, die
schon aus `interpol/` kamen, damit nie zwei Quellen in einer Spalte landen.

Zwei Zahlen, die man kennen sollte:

- **Der neue Baum ändert auch die Geschwindigkeit.** `imputed` weicht an 197 von
  203 Stationen von `interpol/wind` ab, maximal 6,44 m/s — ein anderer
  Trainingslauf, keine Ergänzung.
- **KNN und `imputed_dir` sind sich nicht einig**: wo beide füllen, im Mittel
  20,5° Winkeldifferenz (Median 11,6°, p95 73,7°).

`imputed_dir_guete` liegt bei Median 1,000, misst laut `_herkunft.json` aber die
Einigkeit der überlappenden Fenster, **nicht** den Fehler in Grad. Hohe Güte
heißt „das Modell ist sich sicher", nicht „das Modell hat recht". Wie bei
`imputed` gilt: zu diesen Werten gibt es keine Gütezahl an den echten Lücken, und
in keine Auswertung gehört eine Zahl, die so klingt.

→ `imputation_richtung_tft_20260903.md`

**Ist der KNN-Cache damit umsonst gewesen?** Nein. Er hat die Läufe vom
2026-09-02 überhaupt erst möglich gemacht, und er bleibt zuständig für alles
außer Wind — Solar und die CL/FL-Vorverarbeitung lesen ihn unverändert. Nur die
neun Wind-Läufe rühren ihn nicht mehr an. Gegenprobe: kein einziges der neun
Logs enthält die Zeile `KNN imputation`.

**Preflight vor dem Start** (`scripts/preflight_testmode.py`, read-only,
wiederholbar): lädt Messreihen und beide Imputationen ohne NWP und ohne GPU und
meldet die NaN-Bilanz **je Spalte** im jeweiligen Auditfenster. Ergebnis für alle
neun Configs: **0 NaN-Stationen**, Geschwindigkeit wie Richtung. Genau dieser
Spaltenschnitt hat am 2026-09-02 den Blocker gefunden, den die Gesamtzahl
verdeckt hätte. **Vor jedem Neustart der Queue laufen lassen.**

## 5. Offene Punkte und Stolperfallen

**Ein hostweiter SIGTERM auf l1 am 2026-09-02, 09:48.** Beendete alles unter dem
Benutzer: die damaligen Läufe, die Fremdlast auf GPU 3–7 und die seit dem
24.08. laufende `hpo_keeper`-Session. `KillUserProcesses` steht auf `no`,
`user@1001.service` lief durch, kein OOM — **Ursache ungeklärt**. Der HPO-Keeper
ist seitdem unten und wurde nicht wieder gestartet. Falls es erneut passiert:
der Worker hängt fehlgeschlagene Läufe zurück und stoppt nach drei Fehlschlägen
in Folge, es geht also nichts verloren.

**l1 und ws können nicht von GitHub ziehen.** Beide haben keinen
`credential.helper` und keine `~/.git-credentials`; `git ls-remote` geht
(Protokoll v1), `git fetch/pull` scheitert mit
`could not read Username for 'https://github.com'`, während `curl` auf
`info/refs` HTTP 200 liefert. Passt zum Rate-Limit für unauthentifizierte
Zugriffe — beide Hosts teilen sich die Ausgangs-IP. **Umgehung**, die benutzt
wurde:

```bash
# auf l2
git bundle create /tmp/upd.bundle <alterCommit>..main
scp /tmp/upd.bundle l1:/tmp/ && ssh l1 'cd ~/Work/forecasting_framework &&
  git fetch /tmp/upd.bundle main:refs/remotes/origin/main && git merge --ff-only origin/main'
```

Dauerhaft brauchen l1 und ws ein Token oder einen Deploy-Key mit SSH-Remote.

**l1 trägt einen dauerhaften Config-Unterschied.** Dort läuft über alle Configs
ein `sed` `/mnt/lambda1/nvme1` → `/mnt/nvme1` (238 Dateien, 865 Zeilen), weil l1
die Platte lokal sieht und l2/ws sie per NFS mounten. Das ist **kein**
versehentlicher Zustand: nach jedem `git checkout`/`merge` muss der Rewrite neu
angewendet werden.

```bash
ssh l1 'cd ~/Work/forecasting_framework &&
  grep -rl "/mnt/lambda1/nvme1" configs/ | xargs -r sed -i "s#/mnt/lambda1/nvme1#/mnt/nvme1#g"'
```

**Diesen Rewrite niemals auf l2 oder ws anwenden** — dort existiert `/mnt/nvme1`
nicht. Das ist am 2026-09-03 einmal versehentlich passiert und wurde korrigiert;
die laufenden Jobs waren nicht betroffen, weil sie ihre Config beim Start
gelesen hatten.

**Ein Arbeitsverzeichnis, mehrere Sitzungen.** In
`/home/viktor/Work/forecasting_framework` auf l2 arbeitet zeitweise mehr als eine
Sitzung. Vor `git checkout`/`stash` nachsehen, ob jemand anderes uncommittete
Änderungen liegen hat.

**`misc/migrate_optuna_study.py` wurde gelöscht** (Altlast mit hartkodiertem
Passwort, nie committet, auch nicht in der Historie).

## 6. Stand des Repos

`main` auf `d9ccea2`, gleicher Stand auf l2, l1 und ws; `main` ist der einzige
Branch (der alte `fix/mtgnn-topo-static-dim` wurde am 2026-09-02 nach dem
Fast-Forward gelöscht, wiederherstellbar mit
`git branch fix/mtgnn-topo-static-dim 44919e8`).

Erwartete lokale Abweichungen: auf l1 der Pfad-Rewrite (238 Configs), sonst
nichts. `configs/testfinal` und `configs/testyear` existieren nur auf l1 und sind
bewusst nicht im Repo.
