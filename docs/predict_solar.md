# Solar-Forecasting: Zeitraster und Akkumulationssemantik

Notizen zur Frage, wie ICON-D2 (15 min), die DWD-Messwerte (10 min) und die
globalen Modelle (stündlich) auf ein gemeinsames Raster kommen.

**Status:** Varianten A und C sind umgesetzt und über `data.freq` steuerbar
(Abschnitt 5). Default ist seit Aug 2026 `'30min'` — begründet über kgV(10, 15),
nicht über Messergebnisse: das A/B-Training konnte die Varianten nicht trennen
(Abschnitt 7).

Stand: 2026-08-12

---

## 1. Die Ausgangslage

| Quelle | Auflösung | Akkumulation | Umrechnung auf Intervallwerte |
|---|---|---|---|
| DWD-Messung | 10 min | Intervallsumme, Stempel = Ende | `J/cm² × 10000/600` → W/m² |
| ICON-D2 | 15 min | **Mittel seit Vorhersagebeginn** | `avg[i]·i − avg[i-1]·(i-1)` |
| ECMWF HRES | 1 h | **Summe seit Vorhersagebeginn** | `(F[i] − F[i-1]) / 3600` |
| ERA5 | 1 h | Summe über **genau eine** Stunde | `F / 3600` |

Drei verschiedene Akkumulationssemantiken bei vier Quellen. Die Formeln sind
nicht austauschbar: ICONs Index-Formel auf ECMWF angewandt liefert Unsinn und
umgekehrt.

Zwei Fallen, die dabei jeweils einmal zugeschlagen haben:

- **Der GRIB-Header hilft bei ECMWF nicht.** HRES meldet für `ssrd`, `tisr`,
  `tp` `stepType = instant`, obwohl die Felder eindeutig akkumuliert sind.
  Verlässlich ist nur die empirische Prüfung auf Monotonie entlang der Steps.
- **ICONs Index-Trick funktioniert nur bei äquidistanten Schritten.**
  `avg[i]·i − avg[i-1]·(i-1)` nutzt den Laufindex statt der Stunde; das ist
  korrekt, weil sich der Schrittabstand herauskürzt — und deshalb gilt dieselbe
  Formel für 49 stündliche wie für 193 viertelstündliche Schritte.

**Alle so gewonnenen Werte sind Intervallmittel, keine Momentanwerte.** Der Wert
zum Zeitpunkt `t` beschreibt `[t − Δ, t]`. Das ist für Ertragsrechnung genau
richtig und für alles, was am Sonnenstand hängt, eine Fehlerquelle.

---

## 2. 10 min → 15 min: warum der naheliegende Weg schief geht

Der intuitive Ansatz — die 10- und 20-Minuten-Werte mitteln, ebenso 40 und 50 —
ist **nicht flächentreu**. Die Intervalle überlappen versetzt:

```
10-min:  |--10--|--20--|--30--|--40--|--50--|--60--|
15-min:  |----15----|----30----|----45----|----60----|
```

`[0,15]` besteht aus dem *ganzen* ersten Zehnminutenwert plus der *Hälfte* des
zweiten. Gleichgewichtung verschiebt Energie über die Intervallgrenzen und
erzeugt am Tagesrand einen systematischen Bias.

### Variante A — flächentreue Umverteilung (Standardvorschlag)

Anteilig nach Überlappung gewichten:

```
Q[0,15]  = (w10 + 0.5·w20) / 1.5
Q[15,30] = (0.5·w20 + w30) / 1.5
Q[30,45] = (w40 + 0.5·w50) / 1.5
Q[45,60] = (0.5·w50 + w60) / 1.5
```

Praktisch in zwei Zeilen und exakt äquivalent — auf 5 min upsamplen, dann
zurück aggregieren:

```python
w5  = mess.resample('5min').ffill()           # jeder 10-min-Wert -> zwei 5-min-Werte
w15 = w5.resample('15min', label='right', closed='right').mean()
```

Die Stundensumme bleibt exakt erhalten. Prüfbar mit einer Zeile:
`w15.resample('1h').mean() ≈ mess.resample('1h').mean()`.

### Variante B — Clear-Sky-Index als Träger

Statt W/m² umzuverteilen, `k_c` auf 15 min interpolieren und mit einer
analytischen Clear-Sky-Kurve multiplizieren (dasselbe Prinzip wie in
`synthetic_re_data_generation/docs/era5_solar.md`, Weg B). Sauberer bei Sonnenauf-
und -untergang, wo sich die Strahlung innerhalb von 15 min stark nichtlinear
ändert und eine lineare Umverteilung den Tagesgang verschmiert.

Kostet eine Standortrechnung pro Station und die Annahme einer Trübung.

### Variante C — nicht die Messung anfassen, sondern ICON hochsampeln

ICONs 15-Minuten-Werte sind selbst Intervallmittel und lassen sich mit demselben
Verfahren auf 10 min bringen. Vorteil: die Messreihe — das eigentliche Ziel des
Imputationsmodells — bleibt unangetastet. Nachteil: man erzeugt in den NWP-Daten
eine Auflösung, die das Modell nicht hergibt.

---

## 3. Empfehlung

**Variante A als Standard.** Billig, in der Energiebilanz verlustfrei, in einer
Zeile prüfbar, keine zusätzlichen Annahmen.

**Variante B nur nachgeschaltet**, falls die Validierung zeigt, dass die
Randstunden (Sonnenauf-/-untergang) systematisch abweichen. Das ist die einzige
Stelle, an der A messbar schlechter sein sollte.

**Variante C nicht** für das Imputationsmodell — dort ist die 10-Minuten-Messung
das Zielraster und sollte nicht verändert werden. Für reines Forecasting, wo
ICON die Eingangsgröße ist, kann C dagegen die richtige Wahl sein: dann bestimmt
das Zielraster der Vorhersage, nicht das der Messung.

Daraus folgt: **die Wahl hängt am Anwendungsfall, nicht am Datensatz.** Lücken
füllen → auf 10 min bleiben. Vorhersage auf 15-min-Raster → alles auf 15 min.

---

## 4. Momentanwerte für die PV-Rechnung

Sobald `pvlib` ins Spiel kommt (Transposition auf geneigte Flächen), reichen
Intervallmittel nicht mehr — `pvlib` rechnet mit dem Sonnenstand zu einem
Zeitpunkt.

Kurzform: nicht die W/m² interpolieren, sondern die dimensionslosen Verhältnisse
(`k_c`, Direktanteil), und die absolute Höhe aus einer analytischen Clear-Sky-Kurve
holen. Die Rekonstruktion ist dann energieerhaltend und die Werte instantan gültig.

Ausführlich mit Code in
`synthetic_re_data_generation/docs/era5_solar.md`, Abschnitt 4.

Der schnelle Ersatz, wenn nur Stundenwerte gebraucht werden: Sonnenstand bei
`t − Δ/2` auswerten, Bestrahlungsstärken unverändert lassen.

---

## 5. Umsetzungsstand im Forecasting-Pfad (Aug 2026)

Variante A ist in `utils/solar.py` implementiert und über `data.freq` steuerbar.
Details in [preprocess_icond2_solar.md § 2b](preprocess_icond2_solar.md).

| Punkt aus diesem Dokument | Umsetzung |
|---|---|
| Variante A, flächentreue Umverteilung | `solar.resample_interval_mean()` — Feinraster = ggT(Messraster, Zielraster), also 5 min für 10↔15 |
| „Stempel = Ende" bei DWD | `params.measurement_time_label: 'right'`, seit Aug 2026 Default |
| Zielraster wählbar | `data.freq` — jedes Vielfache von 5 min (= ggT(10, 15)), das 48 h teilt; `solar.resolve_freq()` lehnt alles andere ab und loggt, welche Seite genähert wird |
| Wind darf das nicht | `geostatistics/shared/resolution.py::freq_to_hours(freq, use_case)` — ML ist stündlich |
| Variante B (Clear-Sky-Index als Träger) | nicht als Resampling, aber als Ziel-Transformation vorhanden: `params.target_transform: 'clearsky_index'` |
| Variante C (ICON hochsampeln) | `solar._redistribute_acc_leads()` — Umverteilung entlang der Lead-Achse, greift bei `freq` < 15 min (z. B. `'10min'`). Nur im CL-Pfad; `resolution.freq_to_hours()` weist es im GNN-Pfad ab |

Was beim Umsetzen dazukam:

* **Der `ffill`-Einzeiler aus Abschnitt 2 schüttet Lücken zu.**
  `mess.resample('5min').ffill()` füllt auch einen *fehlenden* 10-min-Wert mit
  seinem Vorgänger — die Lücke verschwindet still statt sichtbar zu bleiben. Die
  Implementierung ordnet stattdessen jeder Feinzelle den Rohwert ihres eigenen
  Intervalls zu (`reindex(method='ffill', tolerance=Δt)`) und verwirft ein
  Zielintervall, sobald weniger als `1 − max_nan_frac` seiner Dauer gedeckt ist.
* **Nur die Strahlungsfelder sind viertelstündlich.** `clct`, `t_2m`, `alb_rad`,
  `u_10m`, `h_snow`, `prr_gsp` liegen ausschließlich auf der vollen Stunde
  (100 % NaN auf den Viertelstunden). Ohne Behandlung fällt `'15min'` still auf
  ein Stundenraster zurück. Steuerung über `params.sub_hourly_fill`
  (`ffill` | `interpolate` | `none`).
* **Die Empirie stützt „Stempel = Ende".** Verschiebungstest gegen `ghi_nwp`,
  7 Stationen, Apr–Sep 2024: mit `right` liegt das RMSE-Minimum symmetrisch auf
  Verschiebung 0, mit `left` ist es nach +1 gezogen. Bei `'1h'` sind das nur
  ~1 % RMSE, bei `'15min'` wird es deutlich.
* **`aswdir_s_avg`/`aswdifd_s_avg` sind Laufmittel und werden abgelehnt.**
  Abschnitt 1 nennt für ICON-D2 „Mittel seit Vorhersagebeginn" — das gilt
  wörtlich nur für die `_avg`-Spalten; `aswdir_s`/`aswdifd_s` liegen bereits
  dekumuliert vor (das kumulative Mittel von `aswdifd_s` bis `ft` ist auf vier
  Nachkommastellen `aswdifd_s_avg` bei `ft`, geprüft für ft = 1, 2, 6, 12, 24, 48).
  Für ein Laufmittel passt **keine** der beiden Lead-Zuordnungen — weder `ceil`
  noch `floor` — es wäre ein über den Lauf wanderndes Mittel mit dem Zeitstempel
  eines einzelnen Schritts. `params.icond2_features` lehnt sie deshalb mit
  Verweis auf die dekumulierte Entsprechung ab, statt sie still umzudeuten.

---

## 6. Offen

- **Zielraster für das Forecasting festlegen.** Default steht seit Aug 2026 auf
  `'30min'` — begründet über kgV(10, 15), nicht über Messergebnisse. Das
  A/B-Training (Abschnitt 7) konnte die Varianten **nicht trennen**.
- **Randstunden quantifizieren.** Wie groß ist der Fehler von Variante A bei
  Sonnenauf-/-untergang wirklich? Eine Woche Sommerdaten gegen Variante B
  reicht zur Entscheidung. Unverändert offen.
- **ECMWF ist hier noch nicht berücksichtigt.** Dessen Stundenwerte müssten für
  ein 15-Minuten-Raster ebenfalls heruntergebrochen werden — laut aktueller
  Absprache passiert das preprocessing-seitig und nicht in der DB.
- **`forecasttime = 0` bei den dekumulierten Feldern** — bestätigt, und zwar in
  der schlimmeren Variante: nicht NULL, sondern **exakt 0.0** in 99.3 % der
  Läufe (der Rest NaN), bei allen vier Strahlungsspalten. Genau die Zeile, die
  von „Nacht" nicht unterscheidbar wäre.
  Im Forecasting-Pfad ist der Fall strukturell ausgeschlossen: durch
  `lead = ceil(ft/step) − 1` wird die `ft = 0`-Zeile verworfen, Lead 0 zieht aus
  `ft = step` (Intervall `(0, step]`). Nachgerechnet: Lead 0 hat 0 von 24 Werten
  gleich null, Mittel 303.8 W/m².
  **Offen bleibt es für jeden Konsumenten, der die SL-Parquets direkt liest** —
  dort ist die Null echt vorhanden und muss gefiltert werden.

---

## 7. Rasterexperiment (12.08.2026) — ergebnisoffen

Drei identische TFT-Läufe, nur `data.freq` und die daran gekoppelte Schrittzahl
verschieden (48 h Horizont in allen dreien). Station 02712, Training
2023-08 → 2024-08, Test 2024-08 → 2025-08, `lookup_hpo: False`.
Reproduzierbar über `scripts/launch_solar_raster.sh` und
`scripts/compare_solar_raster.py`, Configs in `configs/solar_raster/`.

| | | `ghi` | | | `dhi` | |
|---|---|---|---|---|---|---|
| `freq` | Variante | RMSE | RMSE_NWP | Skill_NWP | RMSE | Skill_NWP |
| `30min` | Referenz, exakt | 77.34 | 66.82 | −0.158 | 38.74 | **−0.078** |
| `15min` | A, Messung umverteilt | 80.20 | 70.44 | **−0.139** | 41.32 | −0.105 |
| `10min` | C, ICON umverteilt | 84.81 | 73.33 | −0.157 | 41.77 | −0.093 |

### Was das Experiment zeigt

**Rohe RMSE sind über Raster hinweg wertlos.** Sie steigen monoton von 30 → 15 →
10 min, aber die NWP-Baseline steigt mit (66.8 → 70.4 → 73.3). Ein feineres
Raster mittelt weniger weg, die Aufgabe wird also intrinsisch schwerer — das ist
kein Qualitätsunterschied zwischen den Varianten.

**Die Varianten sind nicht getrennt.** Auf `Skill_NWP` normiert liegt die
Spannweite bei 0.019 (`ghi`) bzw. 0.027 (`dhi`), und die beiden Zielgrößen
**ordnen die Raster gegensätzlich**: `ghi` sieht 15min vorn, `dhi` 30min. Das ist
das Muster von Seed-Rauschen, nicht von einem Effekt. Eine Station, ein
Trainingsjahr und ein Seed reichen nicht.

**Der eigentliche Befund liegt woanders:** `Skill_NWP` ist in **allen sechs**
Fällen negativ (−0.08 … −0.16). Das Modell ist durchgehend 8–16 % schlechter als
die rohe ICON-D2-Prognose. Ein Rastervergleich zwischen drei Modellen, die alle
die Baseline verfehlen, trägt ohnehin nicht weit.

### Warum der Default trotzdem auf `30min` steht

Nicht wegen dieser Zahlen, sondern weil 30 min das feinste Raster ist, in das
beide Quellen exakt nesten (§ 2b von
[preprocess_icond2_solar.md](preprocess_icond2_solar.md)). Es ist die Wahl, die
keine Rechtfertigung braucht.

### Bevor die Rasterfrage erneut angefasst wird

1. **Klären, warum das Modell die NWP-Baseline verfehlt.** Verdächtig sind: eine
   einzelne Station, ein Trainingsjahr, und Hyperparameter, die für Wind bei 1 h
   gewählt wurden. Ohne HPO auf Solar ist der Vergleich wenig aussagekräftig.
2. **Mehrere Stationen und Seeds**, sonst bleibt die Spannweite unter dem Rauschen.
3. Die drei Läufe unterscheiden sich zwangsläufig auch in der **Sequenzlänge**
   (96 / 192 / 288 Schritte). Gemessen wird „Raster inklusive Sequenzlänge" —
   isolieren ließe sich das nur über einen kürzeren Horizont bei feinem Raster.
