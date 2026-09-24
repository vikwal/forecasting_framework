# HRES lauf-indiziert laden (2026-09-24)

Bis zum 24.09.2026 las der ECMWF-Pfad Felder, die es zum Vorhersagezeitpunkt
noch nicht gab. Dieses Dokument haelt fest, was falsch war, wie es gemessen
wurde, was jetzt gilt und was davon geprueft ist.

Commits: `529d6da` (Loader), `c235a01` (Verdrahtung), `099fe5a` und `57f12d1`
(zwei Fehler aus dem Umbau selbst), `a42123e` (Pruefskripte).

---

## 1. Der Befund

`load_ecmwf_parquet_at_stations_and_grid` indiziert HRES **nur ueber die
Gueltigkeitszeit** und nimmt je Stunde den juengsten Lauf im Archiv:

```python
gdf.sort_values(by=["starttime", "forecasttime"])
gdf = gdf.drop_duplicates("valid_time", keep="last")
```

Die ICON-D2-Initialisierung kommt darin nicht vor. Fuer einen ICON-D2-Lauf um
09 UTC ergibt das:

| HRES-Lauf | deckt | HRES-Vorlauf | relativ zum ICON-Lauf |
|---|---|---|---|
| 00 UTC | 2 von 48 h | 10–11 h | −9 h, in Ordnung |
| 12 UTC | 12 h | 0–11 h | **+3 h** |
| 00 UTC Folgetag | 12 h | 0–11 h | **+15 h** |
| 12 UTC Folgetag | 12 h | 0–11 h | **+27 h** |
| 00 UTC uebernaechster | 10 h | 0–9 h | **+39 h** |

**46 von 48 Stunden stammen aus Laeufen, die es zum Vorhersagezeitpunkt noch
nicht gab**, und der genutzte HRES-Vorlauf uebersteigt nie 11 h. HRES ging also
nicht als 48-h-Prognose ein, sondern als rollende Quasi-Analyse.

Der Beleg steckte in den eigenen Zahlen. Stationsmittel RMSE im
Validierungsjahr, roh:

| Vorlauf | 6 h | 18 h | 30 h | 42 h | 48 h |
|---|---|---|---|---|---|
| ICON-D2 | 1.236 | 1.218 | 1.379 | 1.351 | **1.472** |
| HRES | 1.313 | 1.316 | 1.313 | 1.315 | **1.290** |

ICON-D2 verschlechtert sich ueber zwei Tage um 0.24 m s⁻¹, HRES ueberhaupt
nicht. Eine Prognose, deren Fehler nicht mit der Vorlaufzeit waechst, wird
nicht auf dieser Vorlaufzeit benutzt.

**Die Regel stand die ganze Zeit in der Spezifikation.** `docs/data.md`,
Abschnitt „Sample-Konstruktion": *„ECMWF: für jeden ICON-D2 Run den neuesten
ECMWF-Run `<= t_run` nehmen, auf valid_time mergen"*. Der Code hat davon
abgewichen, und die Abweichung ist niemandem aufgefallen, weil sie keine
Fehlermeldung erzeugt, sondern nur bessere Zahlen.

**Die Ursache ist strukturell:** ein nach Gueltigkeitszeit indiziertes Array
`(T, N, F)` kann gar nicht darstellen, dass derselbe Zeitpunkt fuer
verschiedene ICON-D2-Laeufe verschiedene HRES-Werte tragen muss. ICON-D2 war
von Anfang an lauf-indiziert `(R, 48, N, F)`, HRES nicht.

## 2. Die Regel, die jetzt gilt

Je ICON-D2-Lauf der **juengste HRES-Lauf mit `starttime <= icon_starttime`**.
An den Daten geprueft, alle vier ICON-D2-Laufstunden sind lueckenlos gedeckt:

| ICON-D2 | HRES-Lauf | benoetigter HRES-Vorlauf | Reichweite |
|---|---|---|---|
| 06 UTC | 00 UTC | bis 54 h | 57 h |
| 09 UTC | 00 UTC | bis 57 h | 57 h |
| 12 UTC | 12 UTC | bis 48 h | 57 h |
| 15 UTC | 12 UTC | bis 51 h | 57 h |

Im Archiv sind alle 2024 Laeufe des Zeitraums vorhanden, jeder reicht exakt
57 h, keiner fehlt. Ein Rueckfall auf einen aelteren Lauf ist nicht noetig.

**Offen und bewusst nicht entschieden:** die Verbreitungslatenz. Der
12-UTC-HRES-Lauf liegt real erst gegen 17:30 bis 19:00 UTC vor, bei ICON-D2 12
und 15 UTC waere er in Echtzeit noch nicht da. „Gleiche oder fruehere
Initialisierung" ist eine gaengige und vertretbare Konvention, aber eine
Konvention. Wer strenger sein will, nimmt fuer alle vier ICON-D2-Laeufe den
00-UTC-Lauf und verliert bei 12 und 15 UTC die letzten 3 bzw. 6 Stunden.

## 3. Was im Code geaendert wurde

`load_ecmwf_runs_at_stations_and_grid` in `train_stgnn2.py` gibt
`(R, horizon, N, F)` zurueck, an denselben `run_times` wie ICON-D2.
`load_ecmwf_parquet_at_stations_and_grid` bleibt stehen, wird aber von keiner
Stelle mehr gerufen.

| Ebene | Dateien |
|---|---|
| Sampler | `stgnn/training/sampler.py`, `homo_sampler.py`, `evaluation.py` |
| Training | `train_dcrnn`, `train_mtgnn`, `train_stgnn2`, `train_wavenet` |
| Evaluation | `get_test_results_{dcrnn,mtgnn,stgnn2,wavenet}`, `evaluate_reference` |
| HPO | `hpo_dcrnn`, `hpo_mtgnn`, `hpo_wavenet` |
| Baselines | `baselines/dataset.py` (MOS) |

Die Sampler lesen HRES jetzt wie ICON-D2: Historienblock aus `r_hist`,
Prognoseblock aus `r_curr`. Je Aufrufstelle zusaetzlich: Skaler auf
`train_r_mask` statt `[:split_t]`, NaN-Audit auf die Laufachse, Leer- und
Nullarrays auf `(R, F_h, …)`, Merkmalsdimension auf `shape[-1]`.

**MOS lag bewusst mit drin.** `baselines/dataset.py` las die HRES-Praediktoren
ueber `time_idx`. Waere nur der Graphpfad umgestellt worden, haette die
Baseline den Look-ahead behalten und der Vergleich waere schief gewesen.

**Solar:** `freq_h` ist Pflichtargument. Neun Solar-Configs laufen auf
`freq: 30min` mit `next_n_ecmwf > 0`; ohne die Zielschrittweite zoege der
Loader dort die falschen Vorlaufstunden. Die 30-Minuten-Achse haelt den
Stundenwert ueber die Teilschritte konstant (`floor`), akkumulierte Felder
werden um einen Quellschritt nach vorn geholt — beides gespiegelt aus
`_reindex_nwp_to_grid`.

### Drei Sicherungen gegen stille Fehler

Die Sampler werden **positional** aufgerufen, ein Umbenennen der Parameter
faengt eine vergessene Aufrufstelle also nicht.

1. `_assert_run_indexed` in `sampler.py` und die Pruefung in
   `homo_sampler.py` werfen bei einem 3-D-Array und nennen die wahrscheinliche
   Aufrufstelle.
2. `exclude_run_pairs_with_ecmwf_nan` uebersprang 4-D-Arrays vorher
   stillschweigend (`ndim != 3: continue`) und haette damit den NaN-Schutz
   verloren. Sie wirft jetzt bei 3-D und filtert ueber `r_curr`/`r_hist`.
3. `freq_h` hat keinen Vorgabewert.

## 4. Was geprueft ist

Alles Vorwaertslaeufe und Datenabgleiche, **kein Training**. Skripte unter
`scripts/`.

| Pruefung | Skript | Ergebnis |
|---|---|---|
| kein HRES-Lauf aus der Zukunft | `verify_hres_20260924.py` | 0 von 16 |
| Werte gegen die Quell-Parquets | `verify_hres2_20260924.py` | 360 von 360 exakt |
| MOS `ws_e2` gegen die Quelle | `verify_mos_20260924.py` | 192 von 192 exakt, NaN 0.000 |
| 30-Minuten-Achse gegen die 1-h-Achse | `verify_30min_20260924.py` | 0 Abweichungen |
| `get_test_results_dcrnn` | — | EXIT 0, 3905 von 3905 Laeufen gedeckt |
| `get_test_results_mtgnn` | — | EXIT 0 |
| Regression Windpfad | — | RMSE 1.112 vor und nach dem Frequenz-Umbau, bitgleich |

### Nicht geprueft

- Der **Solarpfad** ist nicht als Lauf gelaufen, nur statisch und ueber den
  gemeinsamen Sampler.
- Die drei **`hpo_*`** ebenfalls nicht.
- Der **akkumulierte Zweig** ist implementiert, aber von keiner Kampagne
  belegt: die Configs mit `ssrd`/`fdir` gehen ueber den TFT-Pfad, nicht ueber
  diesen Loader. Keine Config mit akkumulierten ECMWF-Feldern hat
  `next_n_ecmwf > 0`.

Alle drei fallen beim naechsten Start sofort auf, weil die Sicherungen werfen
statt still zu rechnen.

## 5. Zwei Fehler aus dem Umbau selbst

Beide kamen erst in den Vorwaertslaeufen hoch und waren an Syntax-, Import-
und Namenspruefung vorbeigelaufen:

1. `train_r_mask` wurde in `get_test_results_mtgnn` und
   `get_test_results_wavenet` **vor seiner Definition** benutzt, weil der
   ECMWF-Skaler dort vor der Maskenzeile sitzt (`099fe5a`).
2. `freq` war in `get_test_results_dcrnn` nur im `--hpo-study`-Zweig
   zugewiesen. `freq_h=freq` lief deshalb im `--pkl`-Pfad, den `run_eval.sh`
   benutzt, in einen `UnboundLocalError`. Alle elf Aufrufstellen ziehen die
   Frequenz jetzt direkt aus `data_cfg` (`57f12d1`).

## 6. Folgen fuer vorhandene Ergebnisse

**Jedes Modell und jede Metrik vor dem 24.09.2026 ist mit dem alten Verhalten
entstanden.** Der Methodenvergleich untereinander bleibt tragfaehig, weil alle
Arme dieselben HRES-Felder gelesen haben. Nicht tragfaehig sind die absoluten
Skill-Zahlen, zunehmend mit der Vorlaufzeit, sowie alle Aussagen ueber das
Verhalten ueber die Vorlaufzeit.

Ein Modell, das mit den alten Feldern trainiert wurde, darf nicht mit dem neuen
Loader ausgewertet werden — der Rauchtest oben tut genau das und liefert
deshalb RMSE 1.112 statt 1.081. Das ist ein Wiring-Test, keine Messung.
