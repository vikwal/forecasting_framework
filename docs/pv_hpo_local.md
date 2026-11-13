# PV: lokales CNNRNN-Rezept, Pilot und HPO über Park-Folds (2026-11-13)

Code-Referenz. Plan, Entscheidungen und Ergebnisse: `~/Work/pv_sim2real/docs/plan_hpo_local_cnnrnn_2026-11-13.md`.

## Jahre und Daten

- J1 2023-07-24…2024-07-31 (Training), J2 Aug 2024–Jul 2025 (Validierung), J3 Aug 2025–Jul 2026 (Test, nur Testparks,
  nur Schlussmessung).
- Known-Merkmale wie in der Partnerkarte (ICON-D2 09 UTC, ECMWF 00 UTC, 30 min). Das CNNRNN sieht nur die 96
  Prognoseschritte, weil es keine Leistungshistorie gibt.
- Sieben Quantile, Pinball; Ausgabe je Schritt (`step_output`).
- `prepare` schreibt je Park eine Pickle-Datei: x-Scaler wie in der Transferstudie, J1/J2 bitgleich zu deren
  `prepared.pkl`.

## Module

| Datei | Inhalt |
|---|---|
| `scripts/pv_hpo_local.py` | Modi: `prepare`, `pilot` (A/B-Protokollvergleich ohne J3), `losscmp` (Pinball gegen MSE), `hpo` (ein Optuna-Worker einer Fold-Studie), `candidates` (Top 3 je Fold + bisheriges Rezept), `eval` (Konfigurationen × Seeds je Park; Protokoll `A` = J1 → J2 nur Pool-Parks, `final` = J1+J2 → J3 nur Testparks), `launch --stage pilot\|losscmp\|hpo\|eval` (GPU-Warteschlange `local_training.run_jobs`) |
| `scripts/pv_park_split.py` | 179 Parks mit vollem J1: 60 Testparks (KMeans 60, je ein Park), Pool in 3 Folds (KMeans 50, Cluster auf die Folds verteilt), Auswahl der ausgewogensten von 500 Ziehungen; Merkmale Breite, Länge, Höhe, Neigung, Azimut (sin/cos), DC/AC |
| `scripts/pv_hpo_local_pilot_report.py` | Auswertung Pilot und Verlustvergleich |
| `scripts/pv_quantile_finetune.py` `train` | neu: `lr`, `batch_size`, `weight_decay`, `save`, `val_fn` (Defaults wie bisher) |

## HPO

- Je Fold k eine Studie `pv_hpo_local_cnnrnn_2026-11-13_v1_fold<k>` (`OPTUNA_STORAGE`, sonst
  `hpo/studies.db`).
- Ein Trial trainiert das lokale Modell jedes Parks der beiden anderen Folds (J1 → J2). Zielwert ist der mittlere
  Tages-Pinball über diese Parks.
- Median-Pruning auf dem laufenden Mittel nach je 10 Parks, in fester Parkreihenfolge.
- 50 abgeschlossene Trials je Studie; Suchraum `space_v2`.

```bash
O=results/parks_pv_v4_30min/hpo_local_cnnrnn_2026-11-13_v1
python scripts/pv_hpo_local.py launch --stage hpo --output $O --gpus 0-7 --per-gpu 3 --workers 8
python scripts/pv_hpo_local.py candidates --output $O
python scripts/pv_hpo_local.py launch --stage eval --name candidates_pool --parks pool --protocol A \
    --configs-json $O/hpo/candidates.json --output $O --gpus 0-7 --per-gpu 3
python scripts/pv_hpo_local.py launch --stage eval --name test --parks test --protocol final \
    --configs-json <recipe + chosen>.json --output $O --gpus 0-7 --per-gpu 3
```

## Stolperstellen

- Ist `OPTUNA_STORAGE` in der Shell gesetzt, landen auch Smoke-Studien in Postgres. Smoke-Läufe brauchen deshalb einen
  eigenen Studiennamen oder müssen danach gelöscht werden.
- `cnn` mit geradem Kern verlängert die Folge je Schicht. Der Suchraum lässt nur ungerade Kerne zu.
