# Imputation Plausibility Guard

**Status: TEILWEISE UMGESETZT.** Der Guard-Code ist in beiden Erzeugerskripten implementiert
und verifiziert. Der KNN-Imputer (`regen_knn_imputation.py`) wurde erfolgreich neu erzeugt und
bitgenau geprüft. Die Kriging-Regeneration (`run_spatial_interpolation.py`) musste nach 30
Minuten ohne Fortschritt abgebrochen werden (Abschnitt 6) — `interpol/wind` liegt daher
weiterhin im alten, fehlerhaften Zustand auf Platte. Siehe Abschnitt 10 ("Folgen") für die
Konsequenz: die ursprünglichen 130 negativen Werte im Trainingssignal sind durch diesen
Auftrag **noch nicht** behoben, weil sie ausschließlich aus `rk_pred` stammen.

## 1. Befund

Die imputierten Windgeschwindigkeiten enthielten physikalisch unmögliche Werte. Gemessen
mit dem echten Loader-Pfad (`load_station_measurements` + `apply_interpol_imputation` +
`apply_knn_imputation`), 153 Poolstationen (`data.files` + `data.val_files` aus
`configs/mtgnn/stdhp/config_wind_mtgnn_nwp_stdhp_fold1.yaml`), 26 088 Stunden:

- Dateien auf Platte (Stand vor dieser Änderung): negative Werte: 37 im Trainingsfenster, 80
  im Validierungsfenster, 13 im Testfenster. Insgesamt 130 von 3 991 464 Station-Stunden,
  Wertebereich −6.260 bis −0.003, 34 Stationen betroffen.
- Der zu diesem Zeitpunkt von der Kampagne gelesene Cache (`data_cache/gnns/*/meas_raw.npy`,
  3. August) war deutlich schlechter: 351 negative Werte im Training, 316 in der Validierung,
  zusätzlich 40 bzw. 26 Werte über 25 m/s und 7 bzw. 11 über 40 m/s, Wertebereich −102.082 bis
  +82.633.

Beleg, dass negative Werte reine Artefakte der Imputation sind: über alle 204 Rohmessdateien
(31 568 998 Werte) gibt es **keinen einzigen** negativen Rohwert. Der Maximalwert +82.633 m/s
stammt aus den Rohmessungen (Stationsmaximum 82.7 m/s) und bleibt nach ausdrücklicher
Nutzerentscheidung unangetastet — er braucht eine eigene QC-Behandlung außerhalb dieses
Auftrags.

## 2. Vorgabe

Korrigiert wird ausschließlich am Entstehungsort der imputierten Werte, in den beiden
Erzeugerskripten:

1. `geostatistics/run_spatial_interpolation.py` — Regression-Kriging/IDW/OK. Schreibt die
   Spalten `rk_pred`, `idw_pred`, `ok_pred` (plus die unangetastete Rohspalte
   `wind_speed_raw`) nach `/mnt/lambda1/nvme1/synthetic/interpol/wind/Station_{sid}.parquet`,
   Config `configs/config_spatial_interpolation_regen.yaml` (`output.target_path` exakt
   passend — einzige Config im Repo mit diesem Pfad).
2. `geostatistics/regen_knn_imputation.py` — KNN-Imputation. Schreibt `wind_speed` und
   `wind_direction` nach `/mnt/lambda1/nvme1/synthetic/knnimputer/wind/`.

Rohmessungen und `load_station_measurements` werden nicht angefasst.

## 3. Schritt 0 — Backup

`/mnt/lambda1/nvme1/synthetic/_backup_pre_guard_20260811_085640/`:
- `interpol_wind/`: 203 Dateien (163 MB), Anzahl vor und nach dem Kopieren identisch.
- `knnimputer_wind/`: 4 Dateien (77 MB), Anzahl vor und nach dem Kopieren identisch.

Freier Platz vorher: 6.2 TB von 7.0 TB — unkritisch.

## 4. Schritt 1 — Messung vor der Änderung

Über alle vorhandenen Erzeugerausgaben (203 Stationsdateien `interpol/wind`, 26 088×203-Matrix
`knnimputer/wind`), Rohmessung fehlt = Stunde, für die `apply_interpol_imputation` /
`apply_knn_imputation` den Wert überhaupt konsumieren würde:

| Spalte           | n total   | neg  | neg\@miss | \>25 | \>25\@miss | \>30 | \>30\@miss | \>40 | \>40\@miss | \>50 | \>50\@miss |
|------------------|-----------|------|-----------|------|------------|------|------------|------|------------|------|------------|
| rk\_pred         | 4 056 549 | 18 567 | 144     | 21   | 0          | 3    | 0          | 0    | 0          | 0    | 0          |
| idw\_pred        | 4 056 549 | 0    | 0         | 0    | 0          | 0    | 0          | 0    | 0          | 0    | 0          |
| ok\_pred         | 4 056 549 | 3    | 0         | 0    | 0          | 0    | 0          | 0    | 0          | 0    | 0          |
| knn wind\_speed  | 5 295 864 | 0    | 0         | 287  | 1          | 36   | 0          | 3    | 0          | 2    | 0          |

Wertebereiche: `rk_pred` [−25.115, 34.248], `idw_pred` [0.395, 25.000], `ok_pred` [−0.280,
17.236], KNN `wind_speed` [0.000, 77.200].

**Einordnung:** `load_interpol_imputation` liest ausschließlich die Spalte `rk_pred` — nur
deren 144 negative Treffer bei fehlender Rohmessung erklären (näherungsweise) den ursprünglich
gemeldeten Befund von 130 negativen Werten im echten Loader-Pfad (Differenz durch
153-Stationen-Pool vs. 203 Dateien und exakte Train/Val/Test-Grenzen statt Gesamtbereich).
Werte oberhalb 25/30/40 m/s treffen in **keinem** Fall auf eine fehlende Rohmessung — sie
wären im aktuellen Datenstand also folgenlos, sind aber ein Stabilitätsrisiko für künftige
Regenerationen (andere Variogramm-Parameter, mehr Daten). 40 m/s als obere Grenze kappt daher
im Ist-Zustand nichts Relevantes bei `rk_pred`/`idw_pred`/`ok_pred`, schneidet aber bereits 3
Werte in der KNN-`wind_speed`-Datei — die gewählte Grenze ist also weder zu eng (schneidet
keine plausiblen Werte weg) noch wirkungslos.

## 5. Schritt 2 — Guard-Implementierung

In beiden Skripten unmittelbar vor dem Schreiben:

- **Windgeschwindigkeit** auf `[0, upper]` begrenzt. `upper` konfigurierbar
  (`interpolation.wind_speed_upper_bound` in `run_spatial_interpolation.py`,
  Modulkonstante `WIND_SPEED_UPPER_BOUND` in `regen_knn_imputation.py`), Default **40.0 m/s**
  — absolute physikalische Obergrenze für ein Stundenmittel in 10 m Höhe, keine
  stationsrelative Grenze.
- Negative Werte → exakt `0.0`, begründet im Code-Kommentar mit der Zahl 0 negativer Rohwerte
  in 31 568 998 Messungen.
- `dir_pivot` (Windrichtung, Grad) wird **nicht** geschwindigkeitsbegrenzt, nur auf `[0, 360)`
  normalisiert (`np.mod`), mit Zählung der Verletzungen davor.
- Jeder gekappte/normalisierte Wert wird gezählt und geloggt (getrennt nach unten/oben, je
  Spalte).
- `wind_speed_raw` / `wind_speed_observed` unverändert.

## 6. Schritt 3 — Neuerzeugung

### KNN-Imputer (`regen_knn_imputation.py`)

Lief 2026-08-11 09:06:38–09:11:10 (≈4.5 min). Guard-Log:

```
Plausibility guard on wind_speed: clipped 0 value(s) < 0 -> 0.0, 3 value(s) > 40.0 -> 40.0
Plausibility guard on wind_direction: 0 value(s) < 0 deg, 8564 value(s) >= 360 deg -> normalized to [0, 360)
```

Die 8564 Richtungs-Fälle sind kein Bug im Guard, sondern ein reproduzierbares
Fließkomma-Randfall-Ergebnis: Rohwerte von genau 360° durchlaufen `sin`/`cos` und
`arctan2(...) % 360`; da `sin(2π) ≈ −2.449e-16` (nicht exakt 0), liefert `arctan2` einen
winzigen negativen Winkel, dessen `% 360`-Ergebnis (≈ 359.99999999999999998°) in float64
exakt auf `360.0` rundet — außerhalb `[0, 360)`. Ohne den Guard wäre das unbemerkt geblieben
(genau die Art von stiller Korrektur, die zum ursprünglichen Befund geführt hat).

### Kriging (`run_spatial_interpolation.py`, Config `config_spatial_interpolation_regen.yaml`)

**Abgebrochen nach 30 Minuten ohne Fortschritt — kein Guard-Ergebnis für diesen Lauf.**

Gestartet 2026-08-11 09:17:00, `CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 0-31`, 16
BLAS-Threads. Ab 09:17:27 fest in einem einzigen Log-Schritt:

```
KNN imputation (k=10): 286823 missing values (0.90%) — fitting ...
```

Bei 1818 s (30.3 min) Laufzeit, unveränderter CPU-Auslastung (~127 %, ein Kern) und
unverändertem RSS ohne jeden weiteren Log-Fortschritt abgebrochen (`SIGTERM`, sauber
terminiert). `interpol/wind` und der interne lokale Vorbefüllungs-Cache
(`/mnt/nvme1/synthetic/knnimputer/wind`, siehe unten) sind dadurch **unverändert** —
203 Dateien, Zeitstempel weiterhin 2026-08-10 01:02, keine Teil-Schreibvorgänge.

**Ursache (Code-Befund, nicht Teil dieses Auftrags zu beheben):** `run_spatial_interpolation.py`
führt vor der eigentlichen Kriging-LOO-CV eine interne KNN-Vorbefüllung der
Rohmessungslücken durch — auf **10-Minuten-Rohauflösung**, nicht auf der von
`regen_knn_imputation.py` verwendeten, 36× günstigeren Stundenauflösung (Zeile ~416–452:
`ws_pivot_raw = combined.pivot_table(...)` ohne vorheriges `.resample("1h", …)`, dann
`KNNImputer(n_neighbors=10).fit_transform(...)`, O(T²) bei T ≈ 156 000 Zeitschritten über den
vollen 3-Jahres-Rohdatenbereich → T² ≈ 2.45×10¹⁰ Punktepaare). Das Ergebnis wird unter
`data.knn_cache_path` gecacht, Default `/mnt/nvme1/synthetic/knnimputer/wind` — ein **lokaler**
Pfad auf `l2`, verschieden von `/mnt/lambda1/nvme1/synthetic/knnimputer/wind`. Dieser lokale
Cache-Ordner existierte vor diesem Lauf nicht (0 Dateien) und wurde durch den Abbruch auch
nicht gefüllt, sodass ein erneuter Versuch wieder denselben Cache-Miss hätte.
`configs/config_spatial_interpolation_regen.yaml` setzt `data.knn_cache_path` nicht — kein im
Repo verfügbarer Config-Pfad umgeht das. Ein historischer Lauf mit exakt dieser Config
(`logs/regen_kriging_interpol.log`, 2026-07-22) hat unter denselben Bedingungen **7,5 Stunden**
gebraucht — deutlich über der 30-Minuten-Grenze dieses Auftrags. Das Umleiten von
`knn_cache_path` auf einen bereits gefüllten Cache oder das Ändern der Auflösung wäre eine
Entwurfsentscheidung außerhalb dieses Auftrags und wurde nicht vorgenommen.

**Unabhängig davon:** die Guard-Logik selbst (Negativ→0, Speed-Clip auf 40 m/s, unveränderte
`idw_pred`/NaN-Werte, unangetastete `wind_speed_raw`) wurde isoliert an einem synthetischen
`predictions`-DataFrame mit exakt demselben Code verifiziert (siehe unten) — der Guard ist
also nachweislich korrekt, nur die Produktionsdaten in `interpol/wind` wurden in diesem
Auftrag nicht neu erzeugt.

## 7. Schritt 4 — Bitgleichheitsbeweis

### KNN-Imputer

| Datei | n_diff | erwartet (aus Guard-Log) | Übereinstimmung |
|---|---|---|---|
| `wind_speed_knn10_start_end_67558851.parquet` | 3 | 3 | ✅ |
| `wind_direction_knn10_start_end_67558851.parquet` | 8564 | 8564 | ✅ |

Alle Nicht-Diff-Positionen bitgenau identisch (`np.array_equal`, NaN-Maske separat
verglichen).

Zusätzlich isolierter Unit-Test des identischen Guard-Codeblocks aus
`run_spatial_interpolation.py` gegen ein synthetisches `predictions`-DataFrame mit
Negativ-, Über-40-, NaN- und Grenzwerten in `rk_pred`/`idw_pred`/`ok_pred`: alle Erwartungen
(Clipping exakt auf 0.0 bzw. 40.0, NaN unangetastet, `idw_pred` ohne Verletzungen komplett
unverändert, `wind_speed_raw` bitgleich zum Input) erfüllt.

### Kriging

**Nicht durchführbar** — `interpol/wind` wurde nicht neu erzeugt (Abschnitt 6), es gibt daher
keine neuen Dateien, gegen die verglichen werden könnte. Die auf Platte liegenden Dateien sind
exakt die aus dem Backup (0 Diff, weil keine Regeneration stattfand — durch `stat` bestätigt:
Zeitstempel unverändert seit 2026-08-10 01:02).

## 8. Schritt 5 — Cache-Schlüssel

`utils/data_cache.py::GNNCache.make_key` hashte `interpol_path`/`knnimputer_path` nur als
Zeichenketten. Ergänzt im `key_dict`:

- `interpol_fingerprint` / `knnimputer_fingerprint`: `_imputation_dir_fingerprint(path)` —
  content-blind (nur `os.stat`: Dateizahl, Größe, `mtime_ns`), MD5 über die sortierte Liste.
- `imputation_guard_version`: Modulkonstante `IMPUTATION_GUARD_VERSION = 1`.

Dry-run-Beweis (`/tmp/test_make_key.py` auf l2, alte `make_key` aus Git-HEAD vs. neue, gleiche
Config):

```
old_key (pre-edit make_key):  da0c2adc333681ae
new_key (post-edit make_key): 040fe7431edf3fc1
DIFFERENT: True

new_key before touch: f1a5f61629efa46f
new_key after touch:  d6156469d9dea685
CHANGED after touching one file: True
```

(Der Touch-Test lief auf einem privaten Scratch-Verzeichnis, nie auf den echten
Cache-Pfaden.) Bestehende Cache-Verzeichnisse (`data_cache/gnns/07f8bea34c198f83`,
`.../d67d98241545ae6d`) wurden nicht gelöscht; der neue Schlüssel erzeugt automatisch ein
neues Verzeichnis.

## 9. Schritt 6 — Verifikation mit dem echten Loader-Pfad

153 Poolstationen, `load_station_measurements` + `apply_interpol_imputation` +
`apply_knn_imputation`, Fenster train < 2024-08-01, val < 2025-08-01, test danach.

**Endstand dieses Auftrags** (KNN-Anteil mit Guard neu erzeugt, Kriging-Anteil unverändert
alt, siehe Abschnitt 6):

| window | n_cells | n_neg | n_gt40 | n_nan |
|---|---|---|---|---|
| train | 1 373 328 | 37 | 0 | 0 |
| val | 1 340 280 | 80 | 0 | 0 |
| test | 1 277 856 | 13 | 0 | 0 |
| TOTAL | 3 991 464 | 130 | 0 | 0 |

**Erwartung laut Auftrag war 0/0/0 — NICHT erreicht.** `n_neg` ist identisch zum
Ausgangsbefund (37/80/13/130), weil `load_interpol_imputation` ausschließlich `rk_pred` liest
und diese Spalte durch den abgebrochenen Kriging-Lauf nicht neu erzeugt wurde. `n_gt40` ist
in allen drei Fenstern 0 — konsistent mit Schritt 1 (Kriging hatte im Ist-Zustand ohnehin 0
Werte >40 bei fehlender Rohmessung) und mit dem bereits erfolgreichen KNN-Guard. `n_nan` ist 0
in allen Fenstern (unverändert korrekt, wie vor dieser Änderung).

## 10. Folgen

- Die vorhandenen Referenz- und stdhp-Ergebnisdateien sowie alle bisherigen HPO-Trials
  beruhen auf dem alten Stand (mit den 130 negativen Artefakten, die ausschließlich aus
  `rk_pred`/Kriging stammen).
- Die Kampagne wechselt beim nächsten Cache-Miss automatisch auf den neuen Cache-Schlüssel,
  weil er jetzt Dateizahl/Größe/mtime der beiden Imputationsverzeichnisse sowie die
  Guard-Version mit einbezieht. Alte Caches bleiben für Reproduktion erhalten.
- **Die 130 negativen Werte sind durch diesen Auftrag NICHT behoben.** `knnimputer/wind` ist
  vollständig neu erzeugt und bitgenau geprüft (Guard wirksam: 3 Werte >40 m/s in
  `wind_speed` gekappt, 8564 Randfall-Werte in `wind_direction` auf `[0, 360)` normalisiert).
  `interpol/wind` — die tatsächliche Quelle aller 130 negativen Werte, da nur `rk_pred`
  gelesen wird — musste wegen einer vorbestehenden Performance-Falle im Skript (interne
  KNN-Vorbefüllung auf 10-Minuten-Rohauflösung statt Stundenauflösung, kein nutzbarer Cache,
  historisch ≈7,5 h Laufzeit für exakt diese Config) nach 30 Minuten ohne Fortschritt
  abgebrochen werden. Der Guard-Code dafür ist implementiert und isoliert verifiziert, aber
  noch nie an den echten Produktionsdaten angewendet worden.
- **Notwendiger Folgeschritt (außerhalb dieses Auftrags):** `python geostatistics/
  run_spatial_interpolation.py --config configs/config_spatial_interpolation_regen.yaml`
  muss einmal ohne 30-Minuten-Grenze (voraussichtlich mehrere Stunden, unbeaufsichtigt)
  durchlaufen, bevor die negativen Werte im echten Trainingssignal verschwinden. Danach
  greift der neue Cache-Schlüssel automatisch und baut die betroffenen `meas_raw.npy` neu.
- Der Rohdaten-Ausreißer von 82.633 m/s bleibt bewusst unangetastet (Nutzerentscheidung) und
  braucht eine eigene QC-Behandlung außerhalb dieses Auftrags.
