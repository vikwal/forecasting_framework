# PV: NWP-Ausschnitte, Modellgröße, CNN-LSTM mit Personalisierung, Partnerkurve (2026-11-12)

Code-Referenz für die Studien vom 12.11.2026 (FL-Story PV, Auftrag Viktor). Ergebnisse und Einordnung im Paper-Repo:
`~/Work/pv_sim2real/docs/fl_experiments_2026-11-12.md` und die drei Berichte unter `~/Work/pv_sim2real/results/pv_{patch_cnn,capacity,partner_curve}_2026-11-12/`.

Gemeinsames Protokoll (wie [pv_quantile_finetuning.md](pv_quantile_finetuning.md)): sieben Quantile, Pinball tagsüber
(`gti_nwp > 10`), keine Leistungshistorie, Fenster 96 + 96 Schritte à 30 min, Lauf 09 UTC, lr 3e-4, Clipping 7, maximal
100 Epochen, Geduld 10, Early Stopping auf dem Auswertefenster (real Feb–Mai 2024, synthetisch Aug 2024–Jul 2025;
verbindliche Studienentscheidung). Zentral = FedGradient über die Parks (≡ gepoolt), lokal = Batch 64 je Park.
Keine HPO; die Kapazitätsreihe ist eine Ablation mit festen Werten (Go Viktor 12.11.).

## Module

| Datei | Inhalt |
|---|---|
| `utils/patch_models.py` | `PatchEncoder` (2D-CNN je Zeitschritt, optional erste Faltung über `t_kernel` Schritte als Conv3d), `PatchTFT` (Encoder-Einbettung 16-dim neben den Punktmerkmalen in den TFT), `tft_kwargs` (TFT-Argumente wie `models.get_model`) |
| `utils/models.py` (`CNNRNN`) | neu: `num_quantiles` (Ausgabe `(batch, horizon, Q)`; `get_model` übergibt die Zahl aus `model.tft.quantiles`, bei einem Quantil unverändert) und Hyperparameter `step_output` (Default aus): eine Ausgabe je Prognoseschritt aus dem RNN-Zustand dieses Schritts statt des ganzen Horizonts aus dem letzten Zustand. Für PV day-ahead ist der Zustandskopf rund 33 % schlechter als der TFT, der Schrittkopf 3–4 % besser |
| `archiv/cnn_lstm_q_2026-11-12.py` | erste, eigene CNN-LSTM-Variante (Residualblöcke, Dilatation, GroupNorm, Schrittkopf) der Läufe `cnnlstm_pers_2026-11-12_v1` und `boundary_synthetic_cnnlstm_2026-11-12_v1`; ersetzt durch `CNNRNN(step_output=True)`, gleich gut |
| `scripts/pv_patch_data.py` | ICON-D2-16×16-Ausschnitte laden (`$DATA_ROOT/nwp_ready_pv/icon-d2_patch16/SL/09/park_<id>/`, Dateien über `points.csv` → `point_key`), GTI je Gitterpunkt in der Modulebene des Parks (`pv_parks._add_park_irradiance`, Sonnenstand am Park) und `clct`; Spalten `pgti_<i>`, `pclct_<i>` (zeilenweise, Zeile 0 = Süden); `add_stats` (Mittel/Std über 3×3, 7×7, 16×16); `attach` (innerer Join auf `starttime, forecasttime`); `extended_scaler` (Punktmerkmale mit dem bisherigen Scaler, Patchkanäle mit einem Mittelwert/Std je Kanal) |
| `scripts/pv_quantile_finetune.py` | `train(..., factory=None)`: optionale Modell-Factory; ohne Angabe `models.get_model` wie bisher |

## Runner

| Skript | Studie | Ausgabe (`results/parks_pv_v4_30min/`) |
|---|---|---|
| `pv_patch_cnn.py` | Zähler, Arme `point`, `stats`, `cnn`, `cnn3`, je zentral und lokal; `--prepare` baut den Patch-Cache; `build_data()` wird von den beiden folgenden Skripten wiederverwendet | `patch_cnn_2026-11-12_v1/` |
| `pv_trainfit.py` | Pinball auf Trainings- und Testfenstern vorhandener Checkpoints (Patch-Studie point/stats, Grenzkarte P1 voll mit Vortraining), zentral gegen lokal | `trainfit_2026-11-12_v1/trainfit.csv` |
| `pv_capacity_synthetic.py` | synthetisch, TFT `hidden_dim` 32/64/128/256, 10 Zielparks lokal, global über 10, global über 100 Parks; misst auch die Trainingsfenster | `capacity_synthetic_2026-11-12_v1/` |
| `pv_cnnlstm_pers.py` | Zähler, `CNNRNN` mit Quantil-Ausgabe (`FrameworkCNNLSTM`: Encoder = Faltungsstapel, Kopf = LSTM + Ausgabeschicht, rechnet bitgleich wie `CNNRNN.forward`), 5/10 Faltungsschichten; `--arch framework_step` (Default, `step_output`) oder `framework` (letzter Zustand). Arme `local`, `global`, `global_head` (Encoder eingefroren, Kopf je Park nachtrainiert), `global_ft`, `fedper` (gemeinsamer Encoder, Kopf je Park, eigener Trainingsloop `train_fedper`) | `cnnlstm_pers_framework{,_step}_2026-11-12_v1/` (erste Variante: `cnnlstm_pers_2026-11-12_v1/`) |
| `pv_boundary_synthetic.py` | Grenzkarte synthetisch (seit 11.11.); neu `--arms`, `--model tft|cnnlstm` (`cnnlstm` = `CNNRNN(step_output)`, 5 Faltungsschichten; der Prior wird auf den 56 Quellparks vortrainiert und als `<output>/../cnnlstm_prior_seed<S>.pt` geteilt), Umgebungsvariable `CODE_PIN` für Läufe ohne `.git` | `boundary_synthetic_partners_v2_2026-11-12_v1/`, `boundary_synthetic_cnnlstm_framework_2026-11-12_v1/` (erste Variante: `boundary_synthetic_cnnlstm_2026-11-12_v1/`) |

## Ausführung

```bash
export DATA_ROOT=/mnt/lambda1/nvme1
O=results/parks_pv_v4_30min
frcst/bin/python scripts/pv_patch_cnn.py --prepare --output $O/patch_cnn_2026-11-12_v1          # einmal
frcst/bin/python scripts/pv_patch_cnn.py --seed 42 --gpu 0 --output $O/patch_cnn_2026-11-12_v1   # Seeds 43, 44 analog
frcst/bin/python scripts/pv_trainfit.py --gpu 0 --output $O/trainfit_2026-11-12_v1
frcst/bin/python scripts/pv_capacity_synthetic.py --seed 42 --gpu 0 --output $O/capacity_synthetic_2026-11-12_v1
frcst/bin/python scripts/pv_cnnlstm_pers.py --arch framework_step --seed 42 --gpu 0 --output $O/cnnlstm_pers_framework_step_2026-11-12_v1
bash $O/boundary_synthetic_partners_v2_2026-11-12_v1/run.sh        # Partnerkurve TFT (5/40 auf l2) + CNN-LSTM-Karte
# 80/130 Partner: l1, Codekopie ~/runs/fl_partner_curve_2026-11-12/run_l1.sh, Ergebnisse nach .../partners_v2.../l1/
```

Alle Runner setzen nach Abbruch fort (Fälle in `metrics.csv` werden übersprungen).

## Prüfungen

- Patch-Mittelpunkt (8, 8) bitgleich zum bisherigen `gti_nwp` an allen sechs Zählern; der Join verliert keinen Lauf, der
  Arm `point` reproduziert die Grenzkarte von Null (lokal 0,0241, zentral 0,0240).
- `pv_trainfit.py` reproduziert die Testwerte der Quellstudien exakt; die Tagesmaske der Trainingsfenster wird aus der
  unskalierten Known-Spalte `gti_nwp` gebildet (Abweichung zur gespeicherten Testmaske 1e-13).
- Zielparks der synthetischen Läufe in allen Partnerzahlen identisch (`clients.txt`).
- `FrameworkCNNLSTM` rechnet bitgleich wie `CNNRNN.forward` (mit und ohne `step_output`); `CNNRNN` ohne die neuen
  Optionen liefert unverändert `(batch, horizon)`.
- `tests/test_cnnrnn_quantiles.py` (5 Tests): Default-Form unverändert, Quantil- und Schrittausgabe, Ausgabe von Schritt t
  hängt nicht von späteren Eingaben ab (kausaler Stapel), `get_model` reicht die Quantile durch, `step_output` mit
  Attention wird abgelehnt.

## Stolperstellen

- Der TFT wächst mit der Zahl der Known-Variablen (Variable Selection je Variable): Punkt 0,32 M, +12 Gittermerkmale
  0,85 M, +CNN-Einbettung 1,06 M Parameter.
- In `pv_capacity_synthetic.py` erreichen bei `hidden_dim` 32 60 % der lokalen Läufe die Epochengrenze 100.
- `setsid nohup … &` aus dem Agenten heraus: Prozesse leben weiter, die PID des Wrappers ist aber nicht die des
  Python-Prozesses – vor einem Neustart mit `ps -eo pid,args | grep <skript>` prüfen.
