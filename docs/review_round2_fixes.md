# Review Runde 2 — Behebung der Befunde K1–K4, R1 und R2

*Stand 2026-08-03, ausgeführt auf `l2` (`w-lambdablade2`), Branch
`fix/mtgnn-topo-static-dim`, Basis-Commit `9a0f3b6`, Ergebnis-Commit `7eeb42f`.
Alle Zahlen in diesem Dokument sind reproduzierbar, die Kommandos stehen in §8.*

Behoben wurden ausschließlich die sechs im Auftrag genannten Befunde. Die
zurückgestellten Punkte R3–R6 und die kosmetischen Punkte sind unangetastet.
Weitere Beobachtungen, die während der Arbeit auffielen, stehen in §7 — sie
wurden **gemeldet, nicht repariert**.

Die HPO-Kampagne lief während der gesamten Arbeit durch: 8 Worker auf `l2`,
8 auf `l1`. Kein Prozess wurde beendet, keine Studie beschrieben, keine GPU-Zeit
verbraucht. Beleg in §6.

---

## 1. Übersicht

| # | Befund | Datei(en) | Beleg |
|---|---|---|---|
| **K1** | `get_test_results_dcrnn.py` lud keine topographischen Knotenfeatures → `station.static` 3 statt 12 Spalten, Forward-Absturz | `get_test_results_dcrnn.py` | reproduzierter `RuntimeError`, danach sauberer Forward (§2.1) |
| **K2** | MTGNN-/WaveNet-Eval nahmen die Topo-Namen aus `parse_edge_features` statt aus `parse_station_node_features` | `get_test_results_mtgnn.py`, `get_test_results_wavenet.py` | `static_dim` train == eval, 15 == 15 für beide Configs (§2.2) |
| **K3** | Worker ohne `WEATHER_DB_URL` warnte nur und schrieb NWP-Höhen 0 in den geteilten Cache | `train_stgnn2.py`, `hpo_dcrnn.py`, `hpo_mtgnn.py`, `hpo_wavenet.py` | 6/6 Fälle, Abbruch vor jeder Cache-Berührung (§2.3) |
| **K4** | `GNNCache.save()` weder atomar noch gesperrt | `utils/data_cache.py` | vorher 6/6 Leser durch **SIGBUS(-7)** getötet, nachher 3014 saubere Lesevorgänge (§2.4) |
| **R1** | `train_dcrnn.py --eval` reichte `station_k_nearest_grid/_ecmwf` nicht durch | `train_dcrnn.py` | AST-Check (§2.5) |
| **R2** | `get_test_results_dcrnn.py` kannte `interpolate_history` nicht | `get_test_results_dcrnn.py` | AST-Check + Codeabgleich (§2.5) |

Gesamtergebnis der Verifikation:

```
geostatistics/ablations/verify_review2.py         34 passed, 0 failed
geostatistics/ablations/verify_review2_env.py      6 passed, 0 failed
geostatistics/ablations/verify_review2_cache.py   VERDICT: PASS
geostatistics/ablations/verify.py                 79 passed, 0 failed
geostatistics/ablations/batch_fingerprint.py      IDENTICAL — 28 tensor fingerprints
```

auf **allen drei Hosts** (`l2`, `l1`, `ws`).

---

## 2. Was geändert wurde, Befund für Befund

### 2.1 K1 — topographische Knotenfeatures im DCRNN-Eval

`geostatistics/get_test_results_dcrnn.py`

Der Block aus `train_dcrnn.py:868-899` ist jetzt wörtlich übernommen, direkt
nach dem Geo-Scaler:

```python
_node_feat_names = parse_station_node_features(dcrnn_cfg, args.station_node_features)
if _node_feat_names:
    _topo_path = dcrnn_cfg.get("topo_features_path")
    if not _topo_path:
        raise ValueError(
            "station_node_features is set but 'topo_features_path' is missing "
            "from the dcrnn config section."
        )
    # n_train=N_train: the z-score is fitted on the fold's train stations
    # only, exactly as in train_dcrnn.py. Fitting on all_ids would normalise
    # the topography of the held-out stations with their own statistics.
    _topo_cols, _ = load_topo_station_features(
        _topo_path, all_ids, _node_feat_names, n_train=N_train,
    )
    station_static_scaled = np.concatenate(
        [station_static_scaled, _topo_cols], axis=1,
    ).astype(np.float32)
```

Dazu:

* `station_node_features=args.station_node_features` wird an
  `DCRNNConfig.from_yaml` durchgereicht (vorher fiel die Auflösung still auf
  den Config-Wert `all` zurück);
* neues CLI-Argument `--station-node-features`, damit `train_dcrnn.py` und
  `get_test_results_dcrnn.py` dieselbe Schnittstelle haben;
* zwei neue Importe: `parse_station_node_features`, `load_topo_station_features`.

**Zur Fit-Population, die im Auftrag ausdrücklich zu prüfen war.** Der
z-Score wird mit `n_train=N_train` gefittet, also nur auf den Trainingsstationen
des Folds. Das ist exakt das, was `train_dcrnn.py:879` tut. `hpo_dcrnn.py:750`
benutzt die äquivalente Form `train_idx=train_idx` (explizite Fold-Indizes);
da `all_ids = train_ids + val_ids` train-first sortiert ist, sind beide Formen
identisch. Ein Leck ist damit ausgeschlossen: die 51 Zielstationen des Folds
gehen nicht in Mittelwert und Streuung ein.

Dass die Fit-Population tragend ist, ist gemessen und nicht nur behauptet — ein
Fit über alle 60 Fixture-Stationen statt über die 45 Trainingsstationen
verschiebt die Spalten um **max|Δ| = 0.4738** (in z-Score-Einheiten).

### 2.2 K2 — Quelle der Topo-Namen im MTGNN-/WaveNet-Eval

`geostatistics/get_test_results_mtgnn.py`, `geostatistics/get_test_results_wavenet.py`

Beide benutzen jetzt dieselbe Bedingung wie ihr Trainingsskript
(`train_mtgnn.py:467`, `train_wavenet.py:452`):

```python
if args.station_node_features is not None or "station_node_features" in mcfg:
    topo_feature_names = parse_station_node_features(mcfg, args.station_node_features)
else:
    _, _, _, topo_feature_names = parse_edge_features(mcfg)
```

`get_test_results_mtgnn.py` hatte das Argument `--station-node-features`
überhaupt nicht und importierte `parse_station_node_features` nicht — beides
ergänzt. `get_test_results_wavenet.py` hatte das Argument bereits, aber die
zweite Bedingung fehlte, sodass ein Lauf ohne Flag durchfiel.

### 2.3 K3 — `WEATHER_DB_URL` ist jetzt eine Vorbedingung

`geostatistics/train_stgnn2.py` — neuer gemeinsamer Helfer neben
`load_nwp_elevations`, den alle drei HPO-Skripte ohnehin schon importieren:

```python
class MissingNWPElevationEnvError(RuntimeError):
    """Raised when a run that needs NWP node elevations has no database URL."""


def require_nwp_elevation_env(*, need_icond2, need_ecmwf=False, context="") -> None:
    missing: list[str] = []
    if need_icond2 and not os.environ.get("WEATHER_DB_URL"):
        missing.append("WEATHER_DB_URL")
    if need_ecmwf and not os.environ.get("ECMWF_WIND_SL_URL"):
        missing.append("ECMWF_WIND_SL_URL")
    if not missing:
        return
    raise MissingNWPElevationEnvError(...)
```

Aufgerufen am Anfang von `main()`:

| Datei | Stelle | Bedingung |
|---|---|---|
| `hpo_dcrnn.py:296` | nach `check_ablation_flags`, vor `hpo_cfg` | `need_icond2=dcrnn_cfg["use_altitude_diff"]` |
| `hpo_mtgnn.py:356` | nach der `hpo`-Block-Prüfung | `need_icond2=True` (die NWP-Kantenattribute brauchen die Höhen immer) |
| `hpo_wavenet.py:342` | dito | `need_icond2=True` |

Das ist **vor** der Konstruktion von `GNNCache` und damit vor jeder Berührung
des Cache-Verzeichnisses. Die drei alten `else:`-Zweige, die vorher warnten und
weiterrechneten, sind jetzt `raise RuntimeError` — als Rückfallebene, damit eine
spätere Änderung den stillen 0-m-Pfad nicht wieder einführen kann. Sie sind
durch die Vorbedingung unerreichbar; das steht als Kommentar daneben.

> **Korrektur zum Auftrag:** `train_stgnn2.py:1316` wurde als Vorbild genannt
> („`missing.append("WEATHER_DB_URL")` → Fehler"). Tatsächlich baut die Stelle
> zwar eine `missing`-Liste, ruft dann aber `logger.warning` und rechnet weiter
> — sie ist kein harter Abbruch. Der neue Helfer setzt die *Absicht* um, nicht
> das vorgefundene Verhalten. `train_stgnn2.main()` selbst wurde nicht
> geändert, es gehört nicht zu den sechs Befunden.

### 2.4 K4 — `GNNCache.save()` ist atomar und gesperrt

`utils/data_cache.py`

Drei Eigenschaften, alle innerhalb von `save()`, ohne API-Änderung:

1. **Wechselseitiger Ausschluss.** `fcntl.flock(LOCK_EX)` auf
   `cache_dir/.{key}.write.lock` — bewusst **neben**, nicht **in** dem
   Cache-Verzeichnis, damit die Nutzlast dort unter sich bleibt.
2. **Doppelte Prüfung.** Sobald die Sperre gehalten wird, wird ein
   inzwischen von einem anderen Worker vollständig geschriebener Cache in Ruhe
   gelassen. Genau das hätte den Vorfall vom 03.08. verhindert: der Worker ohne
   `WEATHER_DB_URL` hätte den guten Cache nicht überschreiben können.
3. **Atomare Veröffentlichung.** Jede Datei wird nach `{name}.tmp.<pid>`
   geschrieben, mit `flush` + `os.fsync`, und dann per `os.replace` an ihren
   Platz gehoben. `derived.pkl` zuletzt, weil `exists()` daran hängt.

```python
with open(lock_path, "w") as lock_fh:
    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
    try:
        if self.exists(key):
            self.logger.warning(
                "GNNCache — %s already complete (written by another "
                "process); skipping the write.", p,
            )
            return
        ...
        with open(tmp, "wb") as fh:
            np.save(fh, arr)      # np.save haengt '.npy' an einen Pfad an,
            fh.flush()            # deshalb ein offenes Dateiobjekt
            os.fsync(fh.fileno())
        os.replace(tmp, final)
```

**Zur Frage nach `os.replace` auf ein bestehendes Verzeichnis:** ein
Verzeichnis-Rename wäre unter POSIX auf ein nicht leeres Ziel fehlgeschlagen,
und ein bestehendes Cache-Verzeichnis darf bei laufender Kampagne ohnehin nicht
verschoben werden. Deshalb wird **pro Datei** umbenannt, nicht pro Verzeichnis.
Kein bestehendes Cache-Verzeichnis wird angefasst, umbenannt oder gelöscht.
Leser, die bereits ein `mmap` halten, behalten den alten Inode und merken
nichts; neue Leser sehen entweder die alte oder die neue Datei, nie eine halbe.

**Der Lock-Teil ist nur zur Hälfte umgesetzt — siehe §5.**

### 2.5 R1 und R2

`geostatistics/train_dcrnn.py` (R1) — der `run_evaluation`-Aufruf reicht die
beiden Indexarrays jetzt durch:

```python
            interpol_meas=interpol_meas_scaled,
            # Without these two the base arm (nwp_nodes: false) would evaluate
            # against a station.x carrying only the single nearest grid point,
            # while the model was trained on k*I2 + k_e*E2 channels — the B1 fix
            # from round 1 would be inert on this path (review round 2, R1).
            station_k_nearest_grid=station_k_nearest_grid,
            station_k_nearest_ecmwf=station_k_nearest_ecmwf,
```

`geostatistics/get_test_results_dcrnn.py` (R2) — der Kriging-Lag-Kanal wird
aufgebaut und durchgereicht, wörtlich wie in `train_dcrnn.py:814-826`:

```python
interpolate_history = dcrnn_cfg.get("interpolate_history", False)
...
rk_pred = None   # kept for the optional Kriging lag feature below
...
interpol_meas_scaled = None
if interpolate_history:
    if rk_pred is None:
        raise ValueError(
            "dcrnn.interpolate_history: true requires data.interpol_path to be set "
            "and the Kriging parquet files to be present."
        )
    tidx = measurement_cols.index(target_col)
    rk_s = (rk_pred - meas_scaler.mean_[tidx]) / (meas_scaler.std_[tidx] + meas_scaler.eps)
    interpol_meas_scaled = np.nan_to_num(rk_s, nan=0.0).astype(np.float32)
...
    interpol_meas=interpol_meas_scaled,
```

Die Ablations-Assertion aus Runde 1 bleibt wirksam: bei
`neighbour_meas_available: false` **und** `interpolate_history: true` bricht
`check_ablation_flags` weiterhin ab, bevor dieser Kanal überhaupt gebaut wird.

---

## 3. Verifikation mit konkreten Zahlen

### 3.1 K1 — der Test, der den Befund gefunden hat

`geostatistics/ablations/verify_review2.py`, Abschnitt K1. Er baut die Fixture
aus der echten `config_wind_dcrnn.yaml`, stellt den Statik-Aufbau des alten
Eval-Skripts nach und führt einen echten `DCRNN`-Forward über
`evaluation.build_eval_batch` aus.

```
  model_cfg.station_static_features = 13
  [PASS] config resolves to station_static_features = 13 (4 geo/type + 9 topo)
  station.static from a 3-column array: (60, 3) (+1 type indicator = 4)
  forward (old, 3-column statics) RAISED: mat1 and mat2 shapes cannot be multiplied (60x135 and 144x64)
  [PASS] old 3-column path still reproduces the shape-mismatch RuntimeError
  station_node_features = 9 names -> topo columns (60, 9); station.static (60, 12) (+1 type indicator = 13)
  forward (fixed, 12-column statics): OK — no exception
  [PASS] fixed path runs a full DCRNN forward without error
  [PASS] static width matches the model: 12 + 1 type = 13
  [PASS] n_train=45 (train-only z-score) differs from an all-station fit: max|delta| = 0.4738 — the fit population is load-bearing
```

Der Fehler ist bitgenau der aus dem Auftrag: `(60x135 and 144x64)`,
144 − 135 = 9 = die neun Topo-Spalten.

### 3.2 K2 — `static_dim` train gegen eval

```
  config_wind_mtgnn.yaml
      train: 9 topo -> static_dim 15   ['slope', 'aspect_sin', 'aspect_cos', 'tpi5', 'tpi75', 'tdi', 'elev_std', 'z0', 'dist_coast']
      eval (old): 8 topo -> static_dim 14   ['slope', 'aspect_sin', 'aspect_cos', 'tpi5', 'tpi75', 'tdi', 'z0', 'dist_coast']
      eval (new): 9 topo -> static_dim 15
      old eval was missing: ['elev_std']
  [PASS] mtgnn: train static_dim 15 == eval static_dim 15
  config_wind_wavenet.yaml
      train: 9 topo -> static_dim 15   ['slope', 'aspect_sin', 'aspect_cos', 'tpi5', 'tpi75', 'tdi', 'elev_std', 'z0', 'dist_coast']
      eval (old): 0 topo -> static_dim 6   []
      eval (new): 9 topo -> static_dim 15
      old eval was missing: ['aspect_cos', 'aspect_sin', 'dist_coast', 'elev_std', 'slope', 'tdi', 'tpi5', 'tpi75', 'z0']
  [PASS] wavenet: train static_dim 15 == eval static_dim 15
```

Die Diskrepanz aus dem Auftrag (MTGNN 15 vs. 14, `elev_std` fehlt;
WaveNet 15 vs. 6, Topo fehlt ganz) ist damit vor dem Fix reproduziert und
nach dem Fix aufgehoben.

### 3.3 K3 — Abbruch vor jeder Cache-Berührung

`geostatistics/ablations/verify_review2_env.py` patcht `GNNCache.__init__` so,
dass es eine Sentinel-Ausnahme wirft. `GNNCache` ist in allen drei `main()` das
Erste, was das Cache-Verzeichnis anfasst. „REACHED_CACHE" heißt also: die
Prüfung hat den Lauf durchgelassen und es ist noch nichts geschrieben.

```
=== K3 — WEATHER_DB_URL is a precondition, not a warning ===
  [PASS] hpo_dcrnn   / WEATHER_DB_URL unset -> ABORTED: WEATHER_DB_URL not set (hpo_dcrnn.py, dcrnn.use_altitude_diff: true) …
  [PASS] hpo_dcrnn   / WEATHER_DB_URL set   -> REACHED_CACHE
  [PASS] hpo_mtgnn   / WEATHER_DB_URL unset -> ABORTED: WEATHER_DB_URL not set (hpo_mtgnn.py, NWP edge altitude difference) …
  [PASS] hpo_mtgnn   / WEATHER_DB_URL set   -> REACHED_CACHE
  [PASS] hpo_wavenet / WEATHER_DB_URL unset -> ABORTED: WEATHER_DB_URL not set (hpo_wavenet.py, NWP edge altitude difference) …
  [PASS] hpo_wavenet / WEATHER_DB_URL set   -> REACHED_CACHE

6 passed, 0 failed  (of 6 checks)
```

Zusätzlich ein Lauf über die echten Skripte, mit `env -u WEATHER_DB_URL`:
Exit ≠ 0 in allen drei Fällen, und die `mtime` beider produktiver
Cache-Verzeichnisse auf `l2` blieb dabei nanosekundengenau unverändert
(§4).

**Prüfung vor dem Commit, dass kein laufender Worker durch den Fix stirbt:**
alle 16 Worker haben `WEATHER_DB_URL` gesetzt, gelesen aus
`/proc/<pid>/environ`:

```
l2: 24 Prozesse (8 SCREEN + 8 bash + 8 python), alle WEATHER_DB_URL_set=1
l1: 20 Prozesse gelistet, alle WEATHER_DB_URL_set=1
```

Unabhängig davon greift die Änderung ohnehin erst beim nächsten Start: die
laufenden Worker haben das Modul längst geladen.

### 3.4 K4 — paralleles Schreiben und Lesen

`geostatistics/ablations/verify_review2_cache.py`, auf **Wegwerf**-Verzeichnissen
`/tmp/k4cache_old` und `/tmp/k4cache_new`. 4 schreibende und 6 lesende Prozesse
auf denselben Schlüssel, je 48 MB + 24 MB, 10 s Lesefenster.

```
  [A] pre-fix implementation
      6/10 child processes were killed: SIGBUS(-7)  — a reader holding an mmap of a file that
      is truncated under it takes SIGBUS, which Python cannot catch
  OLD (pre-fix): 4 writers, 0 successful reads, 0 inconsistent reads
      final state fill values: [101.0, 102.0] (INCONSISTENT)

  [B] fixed implementation
      GNNCache — … already complete (written by another process); skipping the write.   (3×)
  NEW (fixed): 4 writers, 3014 successful reads, 0 inconsistent reads
      final state fill values: [103.0] (consistent)

  [C] a complete cache is left alone
      fill before=103 after=103, derived.pkl mtime unchanged

  pre-fix: 0 inconsistent reads of 0, 6 reader process(es) killed, final state INCONSISTENT
  fixed  : 0 inconsistent reads of 3014, 0 reader process(es) killed, final state consistent
  VERDICT: PASS
```

Das ist schärfer als erwartet: mit der alten Implementierung überleben die
Leser den Vorgang **nicht**. Ein Prozess, der ein `mmap` auf eine Datei hält,
die unter ihm von `np.save` gekürzt wird, bekommt **SIGBUS**, und SIGBUS ist aus
Python nicht abfangbar — der Worker stirbt kommentarlos. Alle 6 Leser wurden so
getötet, und der zurückbleibende Cache mischte die Versionen zweier Schreiber
(`[101.0, 102.0]`: Arrays vom einen, `derived.pkl` vom anderen). Genau dieser
Zustand ist bei laufender Kampagne mit 2,7-GB-Arrays und 16 Workern möglich.

Mit dem Fix: 3014 erfolgreiche Lesevorgänge, **0** inkonsistent, **0** getötete
Prozesse, konsistenter Endzustand, drei der vier Schreiber übersprangen den
Schreibvorgang, und ein vollständiger Cache wird nicht überschrieben.

Ein früherer Lauf mit größeren Arrays (96 MB + 48 MB) lieferte für die
Vorher-Variante zusätzlich 2642 von 2642 inkonsistenten Lesevorgängen mit den
Beispielen `EOFError: No data left in file` und
`ValueError: mmap length is greater than file size`, bevor die Leser starben.

### 3.5 R1 und R2 — Codebeleg

**Ein empirischer Beleg wäre hier nur mit einem echten Trainingslauf zu haben**
(R1 sitzt hinter `train_dcrnn.py --eval`, R2 verlangt eine Config mit
`interpolate_history: true` plus einen passenden Checkpoint). Beides braucht
GPU-Zeit und war ausgeschlossen. Statt einer Behauptung steht deshalb ein
maschineller AST-Check im Repository, der die Aufrufstellen prüft:

```
  [PASS] R1: exactly one run_evaluation() call in geostatistics/train_dcrnn.py
  [PASS] R1: run_evaluation(... station_k_nearest_grid=...) is passed
  [PASS] R1: run_evaluation(... station_k_nearest_ecmwf=...) is passed
  [PASS] R1: run_evaluation(... interpol_meas=...) is passed
  [PASS] R1: run_evaluation(... hist_wind_available=...) is passed
  [PASS] R1: run_evaluation(... neighbour_meas_available=...) is passed
  [PASS] R2: exactly one evaluate() call in geostatistics/get_test_results_dcrnn.py
  [PASS] R2: evaluate(... interpol_meas=...) is passed
  [PASS] R2: evaluate(... station_k_nearest_grid=...) is passed
  [PASS] R2: evaluate(... station_k_nearest_ecmwf=...) is passed
  [PASS] R2: interpolate_history is read in the eval script
  [PASS] R2: the Kriging lag channel is scaled and NaN-filled like in training
  [PASS] K1: from_yaml(... station_node_features=args.station_node_features)
  [PASS] K1: topo z-score fitted on the fold's train stations (n_train=N_train), as in train_dcrnn.py
  [PASS] K2: get_test_results_mtgnn.py uses the same condition as its training script
  [PASS] K2: get_test_results_wavenet.py uses the same condition as its training script
  [PASS] K3: hpo_{dcrnn,mtgnn,wavenet}.py call require_nwp_elevation_env() once
  [PASS] K3: hpo_{dcrnn,mtgnn,wavenet}.py no longer warn-and-continue
  [PASS] K4: GNNCache.save takes an exclusive flock
  [PASS] K4: GNNCache.save publishes via os.replace
  [PASS] K4: no direct np.save to the destination path remains
```

Der Check ist bewusst strukturell (`ast`) und nicht textuell: er zählt die
Aufrufe und liest die Schlüsselwortargumente, statt nach Zeichenketten zu
suchen. `n_train=N_train` wird als Quelltext des Argumentwerts verglichen.

### 3.6 Die vollständige Verifikationssuite

Unverändert **79 Checks**, alle bestanden — die Zahl hat sich nicht geändert,
weil an den Ablations-Varianten nichts angefasst wurde:

```
79 passed, 0 failed  (of 79 checks)
```

Und der §4.1-Fingerabdruck gegen die vor dem Ablations-Patch erzeugte Referenz
`/tmp/fp_before.json`:

```
IDENTICAL — 28 tensor fingerprints match bit for bit across 14 + 14 tensors.
```

Der Fingerabdruck wurde **nach jedem Zwischenschritt** gelaufen, nicht nur am
Ende: vor Beginn der Arbeit (Ausgangslage), nach den Codeänderungen an
K1/K2/K3/K4, nach dem Ablegen der Verifikationsskripte, und nach dem Rollout auf
allen drei Hosts. Jedes Mal `IDENTICAL`.

| Lauf | Zeitpunkt / Host | Ergebnis |
|---|---|---|
| 0 | Ausgangslage, `l2`, Commit `9a0f3b6` | **IDENTICAL** |
| 1 | nach K1/K2/K3/K4/R1/R2, `l2` | **IDENTICAL** |
| 2 | Endstand vor dem Commit, `l2` | **IDENTICAL** |
| 3 | nach dem Rollout, `l1` (mit Pfad-Rewrites) | **IDENTICAL** |
| 4 | nach dem Rollout, `ws` | **IDENTICAL** |

---

## 4. Die produktiven Caches sind unangetastet

Gemessen vor Beginn der Arbeit und nach dem Rollout, jeweils Pfad, `mtime` in
Nanosekunden und die Nullzählung der NWP-Höhen.

**`l2` — `/home/viktor/Work/forecasting_framework/data_cache/gnns/`**

| Verzeichnis | Datei | `mtime_ns` vorher | `mtime_ns` nachher |
|---|---|---|---|
| `d67d98241545ae6d` (DCRNN) | `derived.pkl` | `1785767144.163177809` | `1785767144.163177809` |
| | `grid_icond2_runs.npy` | `1785767144.000175618` | `1785767144.000175618` |
| | `meas_raw.npy` | `1785767144.021175901` | `1785767144.021175901` |
| | `station_ecmwf_nwp.npy` | `1785767144.051176304` | `1785767144.051176304` |
| | `ecmwf_nwp.npy` | `1785767144.161177782` | `1785767144.161177782` |
| `07f8bea34c198f83` (MTGNN/WaveNet) | `derived.pkl` | `1785767477.569627714` | `1785767477.569627714` |
| | `grid_icond2_runs.npy` | `1785767477.406625551` | `1785767477.406625551` |
| | `meas_raw.npy` | `1785767477.432625896` | `1785767477.432625896` |
| | `grid_ecmwf_raw.npy` | `1785767477.567627688` | `1785767477.567627688` |

**`l1` — `/home/viktorwalter/Work/forecasting_framework/data_cache/gnns/`**

| Verzeichnis | `derived.pkl` `mtime_ns` vorher | nachher |
|---|---|---|
| `c9b1efe4ab1d88b6` (DCRNN) | `1785767674.293490568` | `1785767674.293490568` |
| `b68d18ffe5c9e46d` (MTGNN/WaveNet) | `1785767660.327626525` | `1785767660.327626525` |

**NWP-Höhen im `derived`-Teil, vorher und nachher identisch:**

| Host | Verzeichnis | `icond2_alts` | `ecmwf_alts` |
|---|---|---|---|
| `l2` | `d67d98241545ae6d` | shape (1071,), **zeros=0**, 7…2650 m | shape (553,), **zeros=0**, 7…1733 m |
| `l1` | `c9b1efe4ab1d88b6` | shape (1071,), **zeros=0**, 7…2650 m | shape (553,), **zeros=0**, 7…1733 m |

Die MTGNN/WaveNet-Caches enthalten keine `icond2_alts`/`ecmwf_alts` — dort
werden die Höhen erst **nach** dem Cache-Schreiben geladen und gehen nicht in
den `derived`-Teil ein. Der Korruptionspfad aus K3 betrifft also nur den
DCRNN-Cache; die Absicherung wurde trotzdem in allen drei Skripten gesetzt.

Es sind auch keine neuen Dateien in den Cache-Verzeichnissen entstanden:
`ls -A` liefert vor und nach der Arbeit dieselben zwei Einträge je Host, keine
`.lock`- und keine `.tmp.<pid>`-Reste. Die Sperrdatei entsteht erst beim ersten
Schreibvorgang eines neu gestarteten Workers.

---

## 5. Was offen bleibt

### 5.1 K4 — der Lade-seitige Teil der Sperre ist bewusst zurückgestellt

Umgesetzt ist die **Schreib**-Sperre: von N gleichzeitig schreibenden Workern
kommt nur einer durch, und wer einen bereits vollständigen Cache vorfindet,
lässt ihn in Ruhe.

**Nicht** umgesetzt ist der zweite Teil des Vorschlags: „damit von N gleichzeitig
MISSenden Workern nur einer **lädt** und schreibt". Dafür müsste die Sperre
bereits an der `exists()`-Prüfung genommen und über den gesamten Rohdaten-Ladevorgang
gehalten werden — also über mehrere Minuten und quer durch die `main()`-Funktionen
von `hpo_dcrnn.py`, `hpo_mtgnn.py` und `hpo_wavenet.py`, nicht mehr gekapselt
in `data_cache.py`. Gründe für die Zurückstellung:

* **Es ist keine Kapselung mehr.** Die Sperre müsste in drei Einstiegspunkten
  über einen langen, verzweigten Abschnitt gehalten werden. Ein Fehler dort
  blockiert alle anderen Worker statt nur einen.
* **Ein Absturz unter gehaltener Sperre ist teuer.** `flock` wird zwar beim
  Prozessende freigegeben, aber ein hängender Worker hält die anderen minutenlang
  fest, ohne dass das von außen sichtbar wäre.
* **Der Nutzen ist gering.** Die Verschwendung besteht darin, dass mehrere
  Worker dieselben Rohdaten laden. Das kostet Zeit und I/O, ist aber korrekt —
  die Konsistenz stellt schon die Schreibseite sicher.
* **Die laufende Kampagne würde nicht profitieren.** Die 16 Worker haben das
  Modul geladen; jede Änderung wirkt erst beim nächsten Start.

**Empfehlung:** so lassen. Falls der Ladevorgang doch serialisiert werden soll,
ist der richtige Zeitpunkt zwischen zwei Kampagnen, mit einem Timeout auf der
Sperre (`LOCK_NB` plus Warteschleife mit Obergrenze), damit ein hängender Worker
die anderen nicht blockiert.

### 5.2 Weiterhin offen aus Runde 1 / den Ablationen

Unverändert: Plan §4.7 (kurzer Trainingslauf von Variante B), die HPO-Läufe für
`wind_dcrnn_nomeas` / `wind_dcrnn_nograph`, die sechs Trainingsläufe, die
Evaluation in `excl_val` / `incl_val` (R3), sowie R4, R5 und R6.

---

## 6. Die laufende Kampagne ist unangetastet

Kein Prozess beendet, keine Studie beschrieben, keine GPU benutzt. Alle eigenen
Läufe mit `CUDA_VISIBLE_DEVICES="" nice -n 19`.

**Prozesse und Auslastung, vorher und nachher:**

| | vorher | nachher |
|---|---|---|
| `l2` HPO-Prozesse | 24 (8 SCREEN + 8 bash + 8 python) | 24 |
| `l2` GPU-Auslastung | 48 / 87 / 96 / 95 % | 96 / 93 / 98 / 95 % |
| `l1` HPO-Prozesse | 24 | 24 |
| `l1` GPU-Auslastung | 93 / 28 / 61 / 47 / 76 / 81 / 29 / 19 % | 39 / 34 / 20 / 48 / 76 / 81 / 28 / 20 % |

Die acht Screen-Sessions auf `l2` (`hpo_dcrnn_base_r1` … `hpo_wavenet_nwp_r1`)
laufen seit 16:27 durch, `etime` steigt monoton.

**Optuna, nur lesend abgefragt, vorher und nachher:**

| Studie | vorher | nachher |
|---|---|---|
| `…wind_dcrnn` | RUNNING 3, COMPLETE 1 | RUNNING 3, COMPLETE 1 |
| `…wind_dcrnn_base` | FAIL 3, RUNNING 3 | FAIL 3, RUNNING 3, **COMPLETE 1** |
| `…wind_dcrnn_nwp_hist` | RUNNING 2 | RUNNING 2, **COMPLETE 2** |
| `…wind_mtgnn` | FAIL 2, RUNNING 2 | FAIL 2, RUNNING 2 |
| `…wind_mtgnn_nwp` | RUNNING 2 | RUNNING 2 |
| `…wind_mtgnn_nwp_hist` | RUNNING 2 | RUNNING 2 |
| `…wind_wavenet` | FAIL 1, COMPLETE 1, RUNNING 2 | FAIL 1, COMPLETE 1, RUNNING 2 |
| `…wind_wavenet_nwp` | COMPLETE 1, RUNNING 2 | COMPLETE 1, RUNNING 2 |

**Keine einzige neue FAIL-Zahl.** Die drei FAIL in `wind_dcrnn_base`, die zwei in
`wind_mtgnn` und die eine in `wind_wavenet` sind dieselben wie zu Beginn. Was
sich geändert hat, sind ausschließlich COMPLETE-Zuwächse — die Kampagne kommt
voran.

---

## 7. Beobachtet, gemeldet, **nicht** repariert

Die folgenden Punkte fielen bei der Arbeit auf, gehören aber nicht zu den sechs
Befunden. Sie wurden nicht angefasst.

### 7.1 Der Geo-Statik-Scaler wird in Retrain und Eval auf verschiedenen Populationen gefittet

| Datei | Zeile | Fit-Population |
|---|---|---|
| `hpo_dcrnn.py` | 660 | `raw_static[:N_train]`, und in der räumlichen CV ist `N_train = 153`, also **alle** |
| `train_dcrnn.py` | 865 | `raw_static if (val_start and not args.test_mode) else raw_static[:N_train]` → in der räumlichen CV im Dev-Modus **alle 153** |
| `get_test_results_dcrnn.py` | 415 | **immer** `raw_static[:N_train]`, im Dev-Modus also nur die **102** Trainingsstationen des Folds |

Damit bekommt ein Fold-Modell im Dev-Modus bei der Evaluation andere
Mittelwerte und Streuungen für `lat`, `lon` und `alt` als beim Training. Im
`--test-mode` fällt es zusammen (dort ist `N_train = 153` auf beiden Seiten),
im Dev-Modus nicht. Das betrifft dieselbe Codestelle, die K1 fixt, ist aber ein
eigener Befund — die drei Geo-Spalten, nicht die neun Topo-Spalten. Der
Kommentar in `train_dcrnn.py:852-864` beschreibt die Absicht („Retrain must fit
on the same population"); `get_test_results_dcrnn.py` zieht sie nicht nach.

**Der Topo-z-Score ist davon nicht betroffen** und wurde bewusst so verdrahtet,
wie es der Auftrag verlangt: `n_train=N_train`, identisch zu `train_dcrnn.py`.

### 7.2 `--broadcast-topo` fehlt in `get_test_results_mtgnn.py`

`train_mtgnn.py:247` und `hpo_mtgnn.py:320` haben das Flag,
`get_test_results_mtgnn.py` nicht. Wurde ein MTGNN-Modell mit
`--broadcast-topo` trainiert, kann das Eval-Skript die Architektur nicht
nachbauen. `get_test_results_wavenet.py:125` hat das Flag. Dieselbe Klasse von
Befund wie K2, aber eine andere Achse; in der aktuellen Kampagne wird das Flag
nirgends gesetzt, der Punkt ist also latent.

### 7.3 `ECMWF_WIND_SL_URL` bleibt in `hpo_dcrnn.py` ungeprüft

Der Zweig `elif weather_db_url:` (Zeile 634) greift, wenn `ECMWF_WIND_SL_URL`
fehlt **oder** `max_next_n_ecmwf == 0`. Im ersten Fall werden die ICON-D2-Höhen
aus der Tabelle geladen, `ecmwf_alts` bleibt aber, was der Parquet-Loader
geliefert hat. Der neue `require_nwp_elevation_env`-Helfer kann das über
`need_ecmwf=True` abdecken; ich habe es nicht gesetzt, weil die Bedingung
(`max_next_n_ecmwf > 0`) an dieser Stelle im Ablauf noch nicht feststeht und K3
ausdrücklich `WEATHER_DB_URL` adressiert.

### 7.4 Der Vorbild-Verweis im Auftrag stimmt nicht

`train_stgnn2.py:1316` ist kein harter Abbruch, sondern eine Warnung (§2.3).
Umgesetzt wurde die Absicht, nicht der Ist-Zustand jener Stelle.
`train_stgnn2.main()` selbst hat dieselbe Schwäche weiterhin — es gehört nicht
zu den sechs Befunden, schreibt aber auch nicht in den GNNCache.

---

## 8. Rollout und Reproduktion

Commit `7eeb42f` auf Branch `fix/mtgnn-topo-static-dim`, Elternteil `9a0f3b6`.
12 Dateien, 918 Einfügungen, 21 Löschungen.

| Host | Repo | HEAD | Arbeitsbaum |
|---|---|---|---|
| `l2` (`w-lambdablade2`) | `/home/viktor/Work/forecasting_framework` | `7eeb42f` | sauber (0 Einträge) |
| `l1` (`w-lambdablade1`) | `/home/viktorwalter/Work/forecasting_framework` | `7eeb42f` | 87 modifizierte Configs, **ausschließlich** Pfad-Rewrites, plus die bekannte unversionierte `data_cache.py` vom 28.07. |
| `ws` (`w-lambda-vector`) | `/home/viktor/Work/forecasting_framework` | `7eeb42f` | sauber (0 Einträge) |

Auf `l1` wurden die Pfad-Rewrites nach dem Pull neu angewendet
(`/mnt/lambda1/nvme1/` → `/mnt/nvme1/`, 87 Dateien). Kontrolle:

```
git diff -U0 | grep "^[+-]" | grep -v "^[+-][+-]" | grep -cv "mnt/nvme1\|mnt/lambda1"
→ 0
```

Neue Dateien im Commit — die Belege dieser Runde, damit sie nicht in `/tmp`
verfallen:

| Datei | Zweck |
|---|---|
| `geostatistics/ablations/verify_review2.py` | K1-Reproduktion mit echtem DCRNN-Forward, K2-`static_dim`-Gegenüberstellung, AST-Checks für K1–K4, R1, R2 — 34 Checks |
| `geostatistics/ablations/verify_review2_env.py` | K3, 6 Checks über die drei echten HPO-Einstiegspunkte |
| `geostatistics/ablations/verify_review2_cache.py` | K4, Nebenläufigkeitstest gegen die alte **und** die neue `save()`, auf Wegwerf-Verzeichnissen |

### Reproduktion

```bash
ssh l2
cd ~/Work/forecasting_framework && source frcst/bin/activate

# Runde-2-Belege
CUDA_VISIBLE_DEVICES="" nice -n 19 python -m geostatistics.ablations.verify_review2
CUDA_VISIBLE_DEVICES="" nice -n 19 python -m geostatistics.ablations.verify_review2_env
CUDA_VISIBLE_DEVICES="" nice -n 19 python -m geostatistics.ablations.verify_review2_cache /tmp/k4cache

# Ablations-Suite und Fingerabdruck aus Runde 1 (unverändert)
CUDA_VISIBLE_DEVICES="" nice -n 19 python -m geostatistics.ablations.verify
CUDA_VISIBLE_DEVICES="" nice -n 19 python -m geostatistics.ablations.batch_fingerprint \
    --out /tmp/fp_after.json --compare /tmp/fp_before.json
```

Alle fünf laufen auf der CPU, ohne GPU und ohne Optuna. `verify_review2_cache.py`
legt ausschließlich `/tmp/k4cache_old` und `/tmp/k4cache_new` an und räumt sie
am Ende weg; `data_cache/gnns` wird nie angefasst.
