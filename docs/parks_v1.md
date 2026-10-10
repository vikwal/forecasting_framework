# parks_v1 — reale MaStR-Windparks im regulären Wind-Pfad

Stand 2026-10-06. Daten: 90 reale Windparks (MaStR-Lokationen, Selektion v1.1) aus
`~/Work/FL_Contribution` — Synthese `${DATA_ROOT}/synthetic/wind/parks_v1/`,
Prognosen `${DATA_ROOT}/nwp_ready/{icon-d2,ecmwf}/` (Punktextraktion `~/Work/NWP/points`).
Berichte: `FL_Contribution/reports/park_synthesis_v1.md`, `nwp_extraction_plan.md`.

## Ablauf

```bash
python scripts/build_parks_v1_inputs.py            # data/parks_v1/{stations_master,wind_parameter,turbine_parameter}.csv
python train_cl.py -c configs/parks_v1/config_parks_v1_cl_smoke -m tft   # 3 Parks, 1 Epoche
python train_cl.py -c configs/parks_v1/config_parks_v1_cl -m tft         # 90 Parks
```

## Erweiterungen in `preprocess_synth_wind_icond2` (nur aktiv, wenn gesetzt)

| Schlüssel | Wirkung |
|---|---|
| `data.power_col` | Ziel = diese Spalte des Synth-Parquets (parks_v1: `power_park`, also mit Wakes), statt Summe aller `power*`-Spalten |
| `data.capacity_col` | Normierung auf `wind_parameter[capacity_col]` kW (registrierte Leistung) |
| `data.wind_parameter_file`, `data.turbine_parameter_file` | Parametertabellen außerhalb von `data.path` |
| `data.nwp_site_prefix` | ICON/ECMWF-Ordner heißen `<prefix><station_id>` (nwp_ready: `park_`) |
| `params.static_categorical: [park_id]` | Park-ID als kategoriales statisches Feature: ganzzahliger Code (Position in `files`+`val_files`+`test_files`, `preprocessing.static_categories`), unskaliert, im TFT `nn.Embedding` statt `nn.Linear(1, d)` (`models._static_cardinalities`); nur `tft` |
| `data.ecmwf_layout: site_runs` | ECMWF aus `<ecmwf_path>/SL/{00,12}/<prefix><id>/<lat>_<lon>_wind_sl.parquet` (`_fetch_ecmwf_data_from_site_runs`) |

Die Laufzuordnung bleibt die des Wind-Pfads: ICON-Lauf < 12 UTC → ECMWF 00 UTC desselben Tages
(bei ICON 09 UTC ohne Blick in die Zukunft). Alle neuen Schlüssel gehen in den Cache-Hash ein,
aber nur wenn gesetzt — bestehende Cache-Einträge bleiben gültig.

## Konfiguration `configs/parks_v1/config_parks_v1_cl.yaml`

ICON-D2 09 UTC `wind_speed_h78/h127/h184` + ECMWF `wind_speed_h100/h200`, je 1 Gitterpunkt
(nächster zum Parkmittelpunkt = Mittel der Turbinenkoordinaten), Leistung der letzten 48 h als
observed, Horizont 48 h, TFT. Training Läufe 2023-07-24 … 2024-07-31, Validierung
2024-08-01 … 2025-07-31.

Variante `config_parks_v1_cl_parkid.yaml`: gleich, plus `static_features: [park_id]` als Embedding.
Numerische Park-Statiken (Nabenhöhe, Rotor, …) sind bewusst nicht drin: ein Park mischt Typen
und Nabenhöhen. Im `power_col`-Modus werden die alten Turbinen-Statiken übersprungen.
Tests: `python -m pytest tests/test_parks_v1.py`.

## Föderiert (FL, 80 Parks in 8 Clients)

Clients aus `FL_Contribution/data/mastr/wind_park_selection_v1_1.csv` (Spalte `client`: R0–R3
regional, N0–N3 überregional, je 10 Parks; `holdout` = 10 Parks). Strategie `fedgradient`
([fedgradient.md](fedgradient.md)), Daten, Features, Split und Modell wie CL.

| Config | Training | Bewertung |
|---|---|---|
| `config_parks_v1_fl_fedgradient.yaml` | 80 Client-Parks | 80 + Holdout (`val_files`, globales Modell) |
| `config_parks_v1_fl_fedgradient_parkid.yaml` | 80, Park-ID-Embedding (Codes global, zeilenweise Aggregation) | nur die 80 |
| `config_parks_v1_cl80{,_parkid}.yaml` | CL-Referenz auf denselben 80 Parks, Early Stopping auf ihnen | wie FL; Holdout über `data.holdout_files` |
| `…_smoke.yaml` | 2 Clients × 2 Parks, 2 Runden | |

```bash
python train_fl.py -c configs/parks_v1/config_parks_v1_fl_fedgradient -m tft   # 8 Clients, je 1 GPU
python train_cl.py -c configs/parks_v1/config_parks_v1_cl80 -m tft
```
Lokale Baseline (je Client ein eigenes Modell, [local_training.md](local_training.md)):
`python train_local.py -c configs/parks_v1/config_parks_v1_fl_fedgradient{,_parkid} -m tft --gpus 0-7`.
Vergleich FedAvg: `config_parks_v1_fl_fedavg{,_parkid}.yaml` (dieselbe Config, `fl.strategy: fedavg`,
`n_local_epochs: 1`).

`data.holdout_files` (nur `train_cl.py`): Stationen, die nach dem Training mit demselben Modell
und Scaler bewertet werden, ohne Training und ohne Early Stopping. `val_files` ersetzt in
`train_cl.py` dagegen die Validierungs- und Testmenge (Early Stopping und Bewertung auf ihnen).
In `train_fl.py` sind `val_files` reine Bewertungsparks.

Ergebnisse 2026-10-06 (R² je Park auf 2024-08 … 2025-07, 80 Client-Parks, Mittel / Median):
CL80 0,833 / 0,845, FedGradient 0,831 / 0,841, FedAvg 0,831 / 0,841; mit Park-ID CL80 0,861 /
0,868, FedGradient 0,857 / 0,866, FedAvg 0,843 / 0,850 (nach 100 Runden noch nicht konvergiert).
Holdout (10 Parks, ohne Park-ID): CL80 0,836, FedGradient 0,839, FedAvg 0,838.
Lokal je Client (`train_local.py`, 2026-10-07): 0,835 / 0,847, mit Park-ID 0,859 / 0,865 — so gut
wie FL und CL; Föderation bringt hier keinen messbaren Vorteil (Einordnung im Bericht).
Bericht: `FL_Contribution/reports/fl_fedgradient_parks_v1.md`.
Tests: `python -m pytest tests/test_fedgradient.py`.

## Abgeregelte Varianten (parks_v1_curt_v11, 2026-10-09)

Gleiche Parks, Clients, Features, Split und Modelle, aber `data.path` auf die abgeregelten Releases des Generators
(`${DATA_ROOT}/synthetic/wind/parks_v1_curt_v11` realistisch, `..._x4` Netzabregelung × 4; Bericht
`FL_Contribution/reports/curtailment_synthesis_v1_1.md`). `power_park` ist dort die abgeregelte Leistung (Label und
48-h-Lag); die verfügbare steht in `power_park_avail`, `curt_flag` markiert abgeregelte Stunden.

| Config | Lauf |
|---|---|
| `config_parks_v1_curt_v11[_x4]_fl_fedgradient[_parkid].yaml` | FedGradient (`train_fl.py`); lokal: `train_local.py -c` dieselbe Config |
| `config_parks_v1_curt_v11[_x4]_fl_fedavg[_parkid].yaml` | FedAvg |
| `config_parks_v1_curt_v11[_x4]_cl80[_parkid].yaml` | zentral auf den 80 Client-Parks |
| `config_parks_v1_curt_v11_cl_smoke.yaml` | 3 Parks, 1 Epoche |

```bash
python train_fl.py -c configs/parks_v1/config_parks_v1_curt_v11_fl_fedgradient -m tft --save-predictions   # l1, 8 GPUs
python train_local.py -c configs/parks_v1/config_parks_v1_curt_v11_fl_fedgradient -m tft --gpus 0-3 --per-gpu 2 --save-predictions
python train_cl.py -c configs/parks_v1/config_parks_v1_curt_v11_cl80 -m tft --save-predictions
```
`--save-predictions` (jetzt auch in `train_fl.py`, von `train_local.py` durchgereicht) legt die Vorhersagen ins
Ergebnis-Pickle. Damit bewertet `FL_Contribution/pipeline/summarize_fl_curtailment.py` zusätzlich gegen die
verfügbare Leistung und auf Stunden ohne Abregelung.

Ergebnis (R² je Park, Label / verfügbar, Mittel über 80 Client-Parks, ohne Park-ID): x1 lokal 0,771 / 0,827,
FedGradient 0,760 / 0,827, FedAvg 0,764 / 0,817, CL80 0,775 / 0,826; x4 lokal 0,658 / 0,773, FedGradient
0,648 / 0,765, FedAvg 0,646 / 0,769, CL80 0,660 / 0,752. Föderation verbessert die Label-Prognose in keiner
Variante; einziger FL-Vorteil gegen die verfügbare Leistung: FedGradient + Park-ID in x1 (+0,008).
Bericht: `FL_Contribution/reports/fl_curtailment_v11.md`.

### Statische Parkmerkmale und Variante ohne Leistungshistorie (2026-10-09)

Im `power_col`-Modus (reale Parks) setzt `preprocess_synth_wind_icond2` jetzt park-weite Statiken aus
`turbine_parameter.csv` (`utils.preprocessing.park_group_statics`): `cut_in`, `cut_out`, `rated_wind_speed`,
`hub_height` und `park_age`, je mit der installierten Leistung der Turbinengruppen (n × rated_kw) gewichtet;
`park_age` in Jahren zum Stichtag `data.train_start` (vorher: Inbetriebnahme der ersten Einheit bis heute, also vom
Laufdatum abhängig). `altitude` wie bisher aus `wind_parameter.csv`. Wirksam nur, wenn sie in
`params.static_features` stehen; die anderen Pfade (DWD-Stationen, Turbinenzuordnung) sind unverändert.

Configs `config_parks_v1_curt_v11_{fl_fedgradient,fl_fedavg,cl80}_static.yaml`
(`static_features: [park_age, altitude, hub_height, cut_in, cut_out, rated_wind_speed]`) und `…_static_nolag.yaml`
(zusätzlich `observed_features: []`, kein Leistungs-Lag). Ergebnis x1 gegen das Label (R² Mittel): lokal 0,795 /
0,789 (ohne Lag), CL80 0,797 / 0,791, FedGradient 0,779 / 0,779, FedAvg 0,782 / 0,761
(`FL_Contribution/reports/fl_curtailment_v11.md` §6).

### Szenario-Studie: wann lohnt sich Föderation? (2026-10-10)

Bericht `FL_Contribution/reports/fl_scenarios_v1.md`. Neue Schalter:

| Schlüssel | Wirkung |
|---|---|
| `data.target_mask: all \| market_env \| grid` | nur `power_col`-Modus auf den abgeregelten Releases: maskierte Trainingsstunden (aus `curt_flag` bzw. `loss_*`) bekommen `tools.TARGET_MASK_VALUE` (−1) und fallen aus dem Verlust (`tools.make_criterion`); der Bewertungszeitraum bleibt unverändert. Verlangt `observed_features` ohne `power`. Ohne den Schlüssel ist der Verlust exakt der bisherige. |
| `params.random_seed` | jetzt auch in `train_cl.py` wirksam (`tools.set_seed`) |
| `fl.client_holdout: {client: [stations]}` | Stationen eines Clients, die nie trainiert werden (neue Parks): lokal als `holdout_files` des Clients (`train_local.py`), in FL über `val_files` mit dem globalen Modell |
| `fl.fine_tune_eval_global: true` | mit `fl.fine_tune`: Clients zusätzlich mit dem globalen Modell bewerten (`evaluation_global`, `predictions_global` im Ergebnis) |

Der Cache-Schlüssel enthält jetzt `train_start` (nur reale Parks; der geladene Zeitraum und `park_age` hängen davon
ab) und `target_mask`.

Configs: `python scripts/make_fl_scenarios.py` → `configs/parks_v1/scenarios/` (81 Configs + `manifest.csv`);
Ausführen: `python scripts/run_fl_scenarios.py --scenarios scarce lopo mask --methods fl_fedgradient fl_fedavg cl80 local --gpus …`
(wiederaufnehmbar, Sperrdateien für parallele Warteschlangen, Logs `logs/scenarios/`). FL-Läufe vertragen sich
parallel: auf l1 liefen 6 gleichzeitig ohne längere Rundenzeiten.

### Szenario-Studie Teil 2 und Fine-Tuning-Fix (2026-10-10)

**Fix:** `fl.fine_tune` hat bis `e1fd65b` nie vom globalen Modell aus nachtrainiert — `tools.training_pipeline` baute
ein neues Modell, die übergebenen Gewichte wurden ignoriert. Jetzt: `training_pipeline(..., initial_weights=…,
trainable=[prefixes])`; `train_fl.py` übergibt die globalen Gewichte. Ältere `fine_tune`-Ergebnisse sind lokales
Training von null.

| Schlüssel | Wirkung |
|---|---|
| `fl.fine_tune_variants: [{name, epochs, lr_factor, trainable}]` | mehrere Nachtrainings desselben globalen Modells in einem Lauf (`results['fine_tune_variants'][name]` mit `evaluation`/`predictions`); `trainable` = Parameter-Präfixe, Rest eingefroren (TFT-Kopf: `positionwise_grn`, `output_gate`, `output_ln`, `output_layer`) |
| `data.station_history_start: {station: date}` | die Station hat im Trainingszeitraum erst ab diesem Datum Daten (neuer Park mit kurzer Historie); im Cache-Schlüssel |
| `data.target_mask: market_env_grid50` | Direktvermarkter-Maske plus die Hälfte der Netzereignisse (deterministisch je `grid_event_id`) |

Configs Teil 2: `python scripts/make_fl_scenarios.py --part 2` → `configs/parks_v1/scenarios2/`. Mehrere Rechner
arbeiten eine Liste ab, wenn `--log-dir` auf ein gemeinsames NFS-Verzeichnis zeigt (Sperrdateien
`*.running`, Fertig-Marken `*.done`), z. B. l2 `${DATA_ROOT}/runs/fl_scenarios2`, l1 `/mnt/nvme1/runs/fl_scenarios2`.
Bericht: `FL_Contribution/reports/fl_scenarios_v2.md`.

### Windgeschwindigkeit auf Nabenhöhe als Ziel (2026-10-10)

| Schlüssel | Wirkung |
|---|---|
| `data.target_kind: wind_speed_hub` | nur `power_col`-Modus: Ziel ist die Nabenhöhen-Windgeschwindigkeit des Parks statt der Leistung — Mittel der Gruppenspalten `wind_speed_hub_<turbine>` des Releases, gewichtet mit `n_turbines × rated_kw` (`preprocessing.park_hub_wind`), geteilt durch `WIND_TARGET_SCALE` = 25 m/s statt durch die Leistung. Der Lag (`observed_features: ['power']`) ist dann die vergangene Windgeschwindigkeit. Nicht mit `target_mask` kombinierbar; im Cache-Schlüssel. Standard `power` = bisheriges Verhalten. |

Die Abregelung wirkt nicht auf die Windgeschwindigkeit, das Ziel ist auf x1 und x4 identisch. Configs:
`python scripts/make_fl_scenarios.py --part wind` → `configs/parks_v1/scenarios_wind/` (FedGradient, FedAvg mit
Fine-Tune-Varianten, CL80; lokal = `train_local.py` auf der FedGradient-Config; mit/ohne Lag; Seeds 42–44).

### HPO-Vorbereitung und Review-Korrekturen (2026-10-11)

| Schlüssel / Werkzeug | Wirkung |
|---|---|
| `hpo.fold_boundaries: [b0, b1, …]` | Folds mit festen Datumsgrenzen und wachsendem Fenster (`hpo.kfolds_by_dates`): Fold i validiert die Läufe mit Ausgabe in [b_i, b_i+1) und trainiert auf allen Läufen mit Ausgabe + Horizont ≤ b_i (keine Ziel-Überlappung). Der x-Skalierer wird dann nur auf Daten vor b0 angepasst. Nicht mit `val_files` kombinierbar; `hpo_fl.py` lehnt den Schlüssel ab (FL-Folds noch positionsbasiert). Im Cache-Schlüssel. |
| `data.strict_split: true` | opt-in (`preprocessing.split_bounds`): Trainingsstichproben brauchen Ausgabe + Horizont ≤ `test_start` (kein Trainingsziel im Testzeitraum); ein reines Datum als `test_end` schließt den ganzen Tag ein. Ohne den Schlüssel unverändert. Im Cache-Schlüssel. |
| `data.train_forecast_hours: ['09']` | trainiert nur auf den Stichproben dieser ICON-Läufe (Ausgabestunde UTC); der Test behält alle Läufe aus `data.forecast_hours` (`preprocessing.filter_train_runs`). Im Cache-Schlüssel. |
| `hpo_local.py -c <FL-Config>` | eine Optuna-Studie je FL-Client (`hpo_cl.py` auf der Client-Config von `train_local.py`); danach `train_local.py --lookup-hpo`. Die Studiennamen stimmen mit dem Lookup überein (`cl_m-tft_out-48_freq-1h_<lokaler Name>_<Client>`). |
| `model.lookup_hpo` | bricht jetzt ab, wenn die Studie fehlt (vorher stiller Rückfall auf die Config-Werte). |

Korrekturen aus dem Review (Commit `130054c`):
- Eine fortgesetzte Studie behält den multivariaten TPE-Sampler; der Sampler hat einen Seed.
- HPO-Trials sind geseedet (`random_seed + trial.number`).
- Das Trial-Budget wird aus der Studie gezählt.
- Ein einzelner fehlgeschlagener Trial beendet die Studie nicht mehr; erst 5 Fehler in Folge (`hpo.max_consecutive_failures`).
- `restore_best_weights` greift auch, wenn die Epochengrenze erreicht wird (vorher nur bei ausgelöstem Early Stopping).
- Ein gesuchter `clipnorm` wird im Training verwendet; ohne gesuchten Wert gilt weiter `model.tft.clipnorm`.
- FedAvg-Clients und die Fine-Tuning-Clients in Ray sind geseedet.

Bekannte, bewusst belassene Punkte:
- Early Stopping im Endtraining läuft auf dem Testzeitraum (alle Methoden).
- Lokal skaliert je Client, FL und zentral global.
- FedGradient und FedAvg haben strategieeigene Optimierer und Clipping.
- ECMWF 12 UTC wird den ICON-Läufen 12 und 15 UTC zugeordnet (Annahme: verfügbar).
