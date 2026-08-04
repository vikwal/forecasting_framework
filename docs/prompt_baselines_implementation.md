# Prompt: Klassische Baselines implementieren (Befund R6, Teil B)

*Stand 2026-08-04. Zum Einfügen in eine frische Session. Alles hier ist am Code verifiziert,
wo es als verifiziert ausgewiesen ist — der Rest ist ausdrücklich als offen markiert.*

---

## 1. Auftrag

Implementiere die klassischen Post-Processing-Baselines, die `docs/story_positioning.md` §4
für Contribution (i) benennt, aber die es im Code nicht gibt: **Quantile Regression Forests**
und **MOS**. Verifiziere sie ohne GPU-Zeit und rolle den Code auf allen drei Systemen aus.

Das ist **Teil B** des Review-Befunds **R6** (`docs/review_round2_findings.md` §2). Teil A
desselben Befunds — die drei Falsifikationen zur damaligen Contribution (iv) — ist **nicht**
Gegenstand dieses Auftrags; siehe Abschnitt 3.

### Warum das wichtiger ist, als „Baseline" klingt

Contribution (i) beansprucht den **ersten systematischen Benchmark** des induktiven
NWP-Post-Processing für bodennahen Wind. Ohne diese Baselines ist das eine leere Behauptung.
Und mehr noch:

> **QRF mit topographischen Standortkovariaten ist die entscheidende Nicht-Graph-Baseline
> im induktiven Setting.** Sie kann genau das, was die Studie beansprucht — an einem nie
> gemessenen Ort vorhersagen — nur ohne Graph und ohne Nachbarmessungen. Schlägt das
> Graphmodell sie nicht, hat Contribution (i) kein Ergebnis.

Das ist kein Glaubwürdigkeitsanhang, sondern der Test, der entscheidet, ob das Paper eine
Aussage hat. Arbeite entsprechend sorgfältig.

---

## 2. Systeme

| Host | Alias | Repo | GPUs | Besonderheit |
|---|---|---|---|---|
| `w-lambdablade2` | `l2` | `/home/viktor/Work/forecasting_framework` | 4× A100 80 GB | kanonische Kopie, hier committen; Postgres läuft hier |
| `w-lambdablade1` | `l1` | `/home/viktorwalter/Work/forecasting_framework` | 8× RTX A6000 48 GB | **lokale Pfad-Rewrites in `configs/`** (`/mnt/lambda1/nvme1/` → `/mnt/nvme1/`), nach jedem Pull neu anwenden |
| `w-lambda-vector` | `ws` | `/home/viktor/Work/forecasting_framework` | 2× RTX 4090 24 GB | **nur über l2 erreichbar** (`ssh l2 'ssh ws …'`) |

Branch `fix/mtgnn-topo-static-dim`. Aktuellen HEAD selbst feststellen, nicht raten.

**Rollout-Rezept:**

```bash
ssh l2 'cd ~/Work/forecasting_framework && git add -A geostatistics configs \
  && git commit -F /tmp/msg.txt && git push origin HEAD'

ssh l1 'cd /home/viktorwalter/Work/forecasting_framework && git checkout -- . \
  && git pull origin fix/mtgnn-topo-static-dim \
  && grep -rl "/mnt/lambda1/nvme1/" configs/ | xargs sed -i "s|/mnt/lambda1/nvme1/|/mnt/nvme1/|g"'
# danach pruefen, dass der Diff NUR Pfade enthaelt — muss 0 ergeben:
ssh l1 'cd … && git diff -U0 | grep "^[+-]" | grep -v "^[+-][+-]" | grep -cv "mnt/nvme1\|mnt/lambda1"'

ssh l2 "ssh ws bash -s" <<'EOF'
cd ~/Work/forecasting_framework && git pull origin fix/mtgnn-topo-static-dim
EOF
```

**Shell-Fallstricke:**
- Backslash-Zeilenfortsetzungen funktionieren innerhalb von `ssh host '…'` **nicht**.
  Skript per Heredoc durchpipen (`ssh host bash -s <<'EOF'`) oder alles in eine Zeile.
- `ssh host '…'` ist eine nicht-interaktive Shell, `.bashrc` bricht vor den `export`-Zeilen
  ab. Für `WEATHER_DB_URL` / `ECMWF_WIND_SL_URL` / `OPTUNA_STORAGE` explizit auslesen:
  `eval "$(grep -E '^export WEATHER_DB_URL=' ~/.bashrc)"`. Ohne sie fällt der Loader still
  auf **NWP-Höhen = 0** zurück. Seit Befund K3 bricht ein HPO-Worker ohne die Variable hart
  ab — dein Baseline-Skript sollte es genauso machen.
- Python-Umgebung: `source frcst/bin/activate` im Repo-Root.
- l2 gibt beim SSH harmlose Warnungen über belegte Ports (8504/8510) aus.

**HARTE GRENZE: Es läuft eine HPO-Kampagne.** 18 Worker auf l2 und l1 (8 Kampagnen-Worker je
Host plus die beiden Ablations-Worker auf l1 GPU 3 und 6). **Nicht abschießen, nicht neu
starten, Optuna nur lesend.** Deine Arbeit braucht keine GPU: fahre alles mit
`CUDA_VISIBLE_DEVICES="" nice -n 19`.

---

## 3. Kontext — was sich seit dem Review geändert hat

Der Befund R6 stammt aus Review-Runde 2 und bündelt **zwei Dinge, die inzwischen
auseinandergelaufen sind**. Lies `docs/review_round2_findings.md` §2 (R6) und
`docs/story_positioning.md` §4.

### 3.1 Teil A ist nicht mehr Gegenstand — nicht mitimplementieren

R6 nennt drei Falsifikationen zur **damaligen** Contribution (iv) („Feature-Level-Fusion
zweier NWP-Systeme über zustandskonditionierte Attention"):

- (a) Attention durch feste Inverse-Distanz-Gewichte ersetzt
- (b) nur ICON-D2-Knoten
- (c) nur ECMWF-Knoten

**Contribution (iv) existiert nicht mehr.** Der Contribution-Satz wurde am 2026-08-04 auf drei
Contributions umgebaut; die Multi-NWP-Fusion ist darin keine eigene Contribution mehr. Daraus
folgt:

- **(a) und (c): nicht implementieren.** (a) wäre ein Umbau des DCRNN-NWP-Pfads, (c) bräuchte
  einen `next_n_icond2: 0`-Pfad, den es nicht gibt. Beide zahlen auf nichts mehr ein.
- **(b): billig mitnehmen**, siehe Abschnitt 4.4. Ein gepinnter `next_n_ecmwf: 0` ist eine
  legitime Benchmark-Achse und kostet eine generierte Config.

Wenn dir beim Arbeiten auffällt, dass (a) oder (c) doch gebraucht würden: **melden, nicht
implementieren.**

### 3.2 Teil B ist aufgewertet

Contribution (i) heißt jetzt „Problem benennen, formalisieren und **vermessen**". Die
Baselines sind damit von einer Fleißaufgabe zur Voraussetzung geworden.

### 3.3 EMOS/NGR gehört ausdrücklich **nicht** dazu

`docs/critical_assessment_and_journals.md` §5.3 empfiehlt EMOS/NGR. Diese Empfehlung stammt
aus der Zeit **vor** der Ensemble-Einschränkung und ist überholt.
`docs/story_positioning.md` §3.4 hat entschieden: es gibt kein Ensemble, EMOS-Zahlen wären
nicht vergleichbar, *„do not put EMOS in the same table"*. **Implementiere kein EMOS.**

---

## 4. Was zu tun ist

### 4.0 Das Fundament: exakte Vergleichbarkeit

Die Baselines sind nur etwas wert, wenn sie **auf denselben Run-Paaren, denselben Folds,
denselben Stationen und mit denselben Metriken** ausgewertet werden wie die Graphmodelle.
Das ist die wichtigste Anforderung dieses Auftrags.

Vorhandene Infrastruktur, die du wiederverwenden sollst statt sie nachzubauen:

- **`geostatistics/evaluate_reference.py`** (~21 KB) wertet bereits rohes ICON-D2, ECMWF und
  Persistenz aus. Der Docstring sagt: *„Uses the exact same val-pair loop as
  get_test_results_mtgnn.py so results are directly comparable to trained model outputs."*
  CLI: `-c/--config`, `--fold-idx`, `--test-mode`, `--ecmwf-features`. Schreibt nach
  `data/raw_preds/{name}_fold{N}_raw.parquet` und `data/test_results/{name}_fold{N}.csv`.
  **Das ist der natürliche Ort für die neuen Baselines.** Entscheide begründet, ob du sie
  dort einhängst oder ein Schwesterskript baust, das dieselben Helfer importiert.
- `_station_metrics(...)` und `_save(...)` in derselben Datei liefern das Ausgabeformat.
- `configs/spatial_folds.yaml` definiert die 3 Folds (je 102 Trainings-/51 Zielstationen,
  Val-Mengen paarweise disjunkt, Vereinigung 153).

**Die Zeitkonvention ist die häufigste Fehlerquelle des Projekts.** `t_run_abs` ist der Index
des **ersten Prognoseschritts**, also `t_run + 1 h`, **nicht** der Laufzeit. Wer die Laufzeit
braucht, nimmt `t_run_abs - 1`. Details in `docs/study_overview.md` §3. Baue keine eigene
Run-Paar-Logik — übernimm die vorhandene.

**Leakage-Regeln, die für jede Baseline gelten:**
- Fit ausschließlich auf den **Trainingsstationen des Folds** und ausschließlich im
  **Trainingszeitfenster** (`[:val_start]`). Die 51 Zielstationen des Folds dürfen weder in
  Koeffizienten noch in Skalierern noch in Feature-Statistiken auftauchen.
- Das gilt auch für z-Scores von Topo-Features. Im Modellpfad wird `n_train=N_train` bzw.
  `train_idx=train_idx` benutzt; halte dich daran.

### 4.1 QRF — die entscheidende Baseline

Ein Random-Forest-Post-Prozessor nach `taillardat2016calibrated` (steht in
`literature_review_graph_nwp_bias_correction.md`), gefittet über **alle Trainingsstationen
gepoolt**, mit Standortkovariaten, sodass er an einer nie gemessenen Station anwendbar ist.

**Zwei Varianten, die die Ablationsleiter spiegeln** — das ist bewusst so gewählt:

| Variante | Prädiktoren | Fairer Vergleichspunkt |
|---|---|---|
| **QRF-local** | NWP-Features an den k nächsten Gitterpunkten (ICON-D2 + ECMWF), 9 Topo-Deskriptoren, lat/lon/alt, Lead-Zeit, Tageszeit | Variante **C** (rein standortweises Downscaling) |
| **QRF-IDW** | zusätzlich eine inverse-distanzgewichtete Nachbarbeobachtung | Variante **A** (mit Nachbarmessungen) |

QRF-IDW ist „klassisches räumliches Post-Processing, ordentlich gemacht" und damit die
stärkste Nicht-Graph-Baseline, die es gibt. Wenn das Graphmodell **die** nicht schlägt, ist
das ein Ergebnis, das ins Paper gehört.

Implementierungshinweise:
- `sklearn 1.4.1` ist installiert. **`quantile_forest` und `statsmodels` fehlen.**
- Ob du echte Quantile brauchst, ist eine **offene Entscheidung** — siehe Abschnitt 5.1.
  Für reine Punktvorhersagen genügt `sklearn.ensemble.RandomForestRegressor`.
- Trainiere **ein Modell je Lead-Zeit** oder nimm die Lead-Zeit als Feature auf. Begründe
  deine Wahl; die Literatur macht meist Ersteres, Letzteres ist billiger.
- Achte auf die Größe: 3 Folds × 48 Leads × 102 Stationen × ~1473 Run-Paare. Wenn das zu
  groß wird, subsample **reproduzierbar** und dokumentiere es.

### 4.2 MOS — und die Falle, die vorher zu klären ist

> **Lokales MOS ist an einer zurückgehaltenen Station nicht definiert.** Es braucht die
> Messhistorie des Zielorts, die es dort per Konstruktion nicht gibt.

Implementiere deshalb **drei Varianten**:

| Variante | Was sie ist | Wozu |
|---|---|---|
| **MOS-regional** | eine Regression, gepoolt über alle Trainingsstationen des Folds | überall anwendbar, der ehrliche klassische Boden |
| **MOS-nearest** | Koeffizienten der **nächstgelegenen Trainingsstation** auf das Ziel übertragen | die naive induktive Baseline; faktisch die Arme-Leute-Version von `baran2024clustering` |
| **MOS-local** | klassisch je Station auf deren eigener Historie, **nur an gehaltenen Stationen** | die **transduktive obere Schranke** (`parajka2005comparison`), die §4 ausdrücklich verlangt |

MOS-local ist besonders wertvoll: es beziffert, wie viel Skill die Induktion überhaupt kostet.

Prädiktoren (Standard, halte es einfach und begründe Abweichungen): NWP-Windgeschwindigkeit am
nächsten Gitterpunkt, Lead-Zeit-spezifische Koeffizienten, harmonische Terme für den
Tagesgang. „Nächste Station" **geodätisch** bestimmen (WGS-84), nicht euklidisch in Grad —
das war Befund B2 aus Runde 1, und `evaluate_reference.py` importiert weiterhin `cKDTree`.
Prüf beim Lesen, ob der Import dort noch in einem relevanten Pfad benutzt wird; falls ja:
**melden, nicht nebenbei reparieren.**

### 4.3 Regression-Kriging einreihen

Es existiert bereits als `wind_interpol` (Optuna-Studie, 127 COMPLETE, best 1.1562). Sorge
dafür, dass es im selben Ausgabeformat und über dieselben Folds vorliegt wie die neuen
Baselines, damit die Ergebnistabelle aus einer Quelle gebaut werden kann. Falls das teurer
ist als es aussieht: melden und begründen, nicht stillschweigend weglassen.

### 4.4 Billige Zugabe: der ICON-only-Arm

Generiere eine Varianten-Config mit gepinntem `next_n_ecmwf: 0`, analog zum Vorgehen in
`geostatistics/ablations/gen_variant_configs.py`. Damit wird die Achse „eine vs. zwei
NWP-Quellen" ein eigener Arm statt eines von der HPO frei gewählten Hyperparameters.

**Keine HPO starten.** Nur die Config erzeugen, verifizieren und dokumentieren.

---

## 5. Zwei Entscheidungen, die du vorher klären musst

Beide sind **Rückfragen an den Nutzer**, keine Annahmen. Formuliere sie präzise und
entscheidbar, mit deiner Empfehlung und den Konsequenzen.

### 5.1 Punktvorhersage oder Quantile?

`story_positioning.md` §3.4 hat das Paper auf **deterministisches Punkt-Post-Processing**
festgelegt. Dann genügt `RandomForestRegressor`, und „QRF" wäre streng genommen „RF".

- **(a)** Nur Punktvorhersage. Billig, konsistent mit §3.4, aber der Name QRF in der
  Ergebnistabelle wäre irreführend — dann ehrlich „Random Forest" nennen.
- **(b)** Echte Quantile über `quantile-forest` (muss installiert werden) oder über den
  Meinshausen-Blattindex-Trick auf `RandomForestRegressor` (~30 Zeilen, keine neue
  Abhängigkeit). Erlaubt später eine verteilungsbezogene Auswertung, ohne sich auf
  Ensemble-Vergleiche einzulassen.

Meine Einschätzung: (b) über den Blattindex-Trick, weil es keine Abhängigkeit kostet und die
Option offenhält. Aber es ist die Entscheidung des Nutzers.

### 5.2 Bekommt QRF Nachbarbeobachtungen?

Abschnitt 4.1 schlägt **beide** Varianten vor (QRF-local und QRF-IDW), weil sie die
Ablationsleiter spiegeln. Das verdoppelt den Rechenaufwand der Baseline. Falls nur eine
gefahren werden soll, ist es eine inhaltliche Entscheidung, welche — und sie ändert, wogegen
das Graphmodell antritt.

---

## 6. Verifikation

Alles ohne GPU, alles auf der CPU. Bau die Suite analog zu
`archiv/ablations_verification/verify.py` und lege sie **in dasselbe Verzeichnis**, nicht
nach `/tmp`.

1. **Leakage-Test.** Fit auf Fold *k*, dann prüfen, dass keine der 51 Zielstationen in die
   Koeffizienten oder Feature-Statistiken eingeht. Konkreter Beleg: derselbe Fit einmal mit
   und einmal ohne die Zielstationen im Trainingssatz muss **bit-identische** Koeffizienten
   liefern, wenn die Maskierung korrekt ist.
2. **Zeitausrichtung.** RMSE(Baseline vs. Messung) über die tatsächlich gebauten Run-Paare
   bei Versatz −2 … +2. Das Minimum **muss** bei 0 liegen. Für rohes ICON-D2 ist der
   Referenzwert bekannt: 1.4958 bei Versatz 0, gegen 1.5040 bei −1 und 1.5550 bei +1.
3. **Fold-Konsistenz.** Die von der Baseline ausgewerteten Stations-IDs je Fold müssen
   station-genau mit `configs/spatial_folds.yaml` übereinstimmen.
4. **Formatgleichheit.** Die erzeugten `data/test_results/*.csv` müssen dieselben Spalten und
   dieselbe Stationsmenge haben wie die vorhandenen `icon_d2_fold{N}.csv`, damit die
   Ergebnistabelle ohne Sonderfälle gebaut werden kann.
5. **MOS-local nur an gehaltenen Stationen.** Beleg, dass es an Zielstationen **nicht**
   ausgewertet wird — es ist dort per Definition undefiniert.
6. **Sanity gegen die bekannten Referenzen.** Jede Baseline muss besser sein als Persistenz
   und mindestens so gut wie rohes ICON-D2, sonst stimmt etwas nicht. Nenne die Zahlen.
7. **Die Kampagne ist unangetastet:** Prozessliste und FAIL-Zahlen vorher/nachher.

---

## 7. Wie du vorgehen sollst

1. Lies zuerst `docs/study_overview.md` — es ist das Einstiegsdokument und beschreibt Task,
   Datenfluss, Zeitkonvention, Graph, HPO-Mechanik und Pipeline. Dann `story_positioning.md`
   §4 (die drei Contributions) und `review_round2_findings.md` §2 (R6).
2. Verifiziere jede Aussage über den Code am Code. Zeilennummern in älteren Dokumenten sind
   veraltet.
3. Klär die beiden Entscheidungen aus Abschnitt 5, **bevor** du QRF implementierst.
4. Keine GPU, keine HPO-Starts, laufende Worker nicht anfassen.
5. Bei Unklarheiten über die **Absicht**: nachfragen, nicht annehmen. Mach alles fertig, was
   nicht von der Antwort abhängt, und beende deinen Lauf mit der präzise formulierten Frage.
6. Wenn dir etwas an bestehenden Fixes falsch vorkommt (Kandidaten: N1 Geo-Statik-Scaler,
   B2 `cKDTree` in `evaluate_reference.py`): **melden, nicht reparieren.**

---

## 8. Wenn du fertig bist

- Commit auf l2, Push, l1 und ws nachziehen, auf allen drei Hosts denselben Commit belegen.
- Die Verifikationsergebnisse mit **konkreten Zahlen** in ein Dokument
  `docs/baselines_verification_results.md` schreiben — nicht nur „bestanden". Lokal nach
  `/Users/viktorwalter/Latex/Graphs_Wind_Speed_Forecasting/docs/` spiegeln.
- In `docs/study_overview.md` Abschnitt 9 den Eintrag zu **R6** aktualisieren: Teil B erledigt,
  Teil A (a)/(c) bewusst gestrichen, (b) erledigt.
- Melden, was offen bleibt — insbesondere alles, was einen Trainingslauf gebraucht hätte.
