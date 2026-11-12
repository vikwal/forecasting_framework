# PV-Quantil-Fine-Tuning (2026-11-09)

Vergleich von Zero-shot, lokalem und zentralem Fine-Tuning sowie FedGradient auf den sechs realen PV-Clients. Kein Conformal-Postprocessing. Ziel: Nutzen gemeinsamer realer Daten für die Verteilungsprognose.

## Reproduktion

Im Framework mit gesetztem `DATA_ROOT`, vorhandenen Quantil-Checkpoints für Seeds 42/43/44:

```bash
frcst/bin/python scripts/pv_quantile_finetune.py --seed 42 --gpu 0 --output results/parks_pv_v4_30min/quantile_ft_<new_version>
```

Für Seeds 43 und 44 wiederholen. Das jeweilige Seed-Verzeichnis darf noch nicht existieren. Modelle und Vorhersagen bleiben im Framework. Auswertung nach Abschluss aller Seeds:

```bash
frcst/bin/python scripts/pv_quantile_report.py results/parks_pv_v4_30min/quantile_ft_<new_version> --export <new_csv_directory>
```

## Protokoll

- Je Seed eigener synthetisch vortrainierter Sieben-Quantil-Checkpoint samt ursprünglichem Scaler; alle Varianten eines Seeds starten identisch. Architektur und No-history-Datenconfig bleiben erhalten.
- Training 24.07.2023–31.01.2024; Early Stopping und Evaluation auf ganz Februar–Mai 2024. Lokale Modelle sehen dabei nur ihren Park, zentrale/föderierte alle sechs. Diese Zahlen sind Validierungsergebnisse mit Modellauswahl, keine unabhängige Testauswertung.
- Adam, lr 3e-4, maximal 100 Epochen, patience 10, min_delta 0.0001. Early Stopping anhand mittlerem Pinball-Loss aller sieben Quantile und aller Halb-Stunden-Zellen einschließlich Nacht. Startzustand ist Kandidat. Zusätzlich beste trainierte Epoche aus dem tatsächlich durchlaufenen Verlauf, keine separate Suche mit anderem Stoppkriterium.
- Lokal Batch 64. Zentral/FedGradient dieselben pro Client permutierten Batches mit je 11 Fenstern; zentral werden die Teilbatches zusammengeführt, föderiert Gradienten nach tatsächlicher Batchgröße gewichtet. Restbatches werden erhalten; erschöpfte Clients scheiden aus. Nominaler effektiver Batch 66.
- Gemeinsames Gradient Clipping mit Norm 7 **nach** der Aggregation. Anders als im älteren Punktmodell-Versuch wird nicht jeder Client separat geclippt. Dropout 0.1 beim Training; bei jeder Validation/Prediction ausdrücklich eval-Modus. Wegen Dropout und Gleitkommaarithmetik keine bitweise Gleichheit zwischen zentral und föderiert versprochen.
- Der Runner simuliert die Federation auf einer GPU. Clientgradienten und Server-Adam nutzen `utils/fedgradient.py`. Zentrale Variante darf Daten poolen, föderierte Worker behalten ihre lokalen Trainingstensoren. Validation kann durch Loss-Summen und Zählwerte erfolgen; Evaluation/Dateiexport läuft zentral zur wissenschaftlichen Analyse. Kein verteilter Netzwerktest, keine formale Datenschutzgarantie.

## Kennzahlen und Artefakte

Primär rohe Quantile: Pinball-Loss tagsüber und ganztägig. Tagmaske `gti_nwp > 10`. Intervallmetriken für 50/80/90 % aus sortierten, auf nichtnegative Werte begrenzten Quantilen, dazu Crossing-Rate und Pinball nach dieser Transformation. Median-R² und RMSE ergänzend. Keine Conformal-Korrektur.

`metrics.csv`: je Park, Seed, Variante und Auswahlregel. `*_history.csv`: Pinball-Verlauf. `*_best.pt` und `*_best_trained.pt`: Checkpoints. `*.npz`: sieben rohe Quantile, Ziel, NWP, Zeitindex. `metadata.json`: Schnitt, HP, Commit+dirty, Hashes von Modell/Configs/Implementierung. `training.json`: gewählte Epochen und numerisches Kommunikationsvolumen aus dem FedGradient-Engine; Transport- und Validation-Overhead nicht vollständig erfasst. `COMPLETE`: Abschlussmarker.

`pv_quantile_report.py` mittelt zuerst über die sechs Parks pro Seed, dann Mittelwert und Stichprobenstandardabweichung über die drei Seeds. Das ist keine Konfidenzintervallschätzung und keine Signifikanzprüfung.

## Prüfung

`tests/test_pv_quantile_finetune.py`: deterministischer Vergleich der Central-Union mit FedGradient inklusive ungleichen Clientgrößen, Restbatches, globalem Clipping und Adam-State; Quantilmetriken und Shapes; ausgeschaltetes Dropout bei Validation und gespeicherte Checkpoints. Ausführung mit `frcst/bin/python -m pytest tests/test_pv_quantile_finetune.py`.
