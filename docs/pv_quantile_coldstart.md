# PV-Quantil-Fine-Tuning mit wenigen eigenen Messtagen (2026-11-09)

Fortsetzung von [pv_quantile_finetuning.md](pv_quantile_finetuning.md). Nutzerauftrag: lokal, zentral und FedGradient bei wenigen eigenen Messtagen, drei Seeds. Kein Conformal-Postprocessing.

## Design

Feste Budgets 7, 14, 30, 60 und 90 Kalendertage bis exklusiv 01.02.2024 UTC. Jeder der sechs Parks wird einzeln zum datenarmen Zielpark. Dessen vollständige 48-h-Zielhorizonte müssen innerhalb des Budgets liegen; ausgewählt wird anhand der tatsächlichen `timestamp`-Werte des zugrunde liegenden NWP-Laufs. Historische NWP bleibt verfügbar; es gibt keine beobachtete Leistung als Input. Die fünf Donorparks behalten ihre kompletten Trainingsreihen Juli 2023–Januar 2024. Die Zahl nutzbarer Fenster wird separat gespeichert.

- Lokal: nur kurze Zielparkreihe, Batch64.
- Zentral: kurze Zielparkreihe plus fünf volle Donorreihen, identische Batch-Vereinigung wie FedGradient.
- FedGradient: dieselben sechs Datenmengen getrennt je Client; je Client Batch11, nach Batchgröße gewichtete Gradientenaggregation und gemeinsamer Server-Adam nach jedem Schritt. Restbatches bleiben erhalten; erschöpfte Clients scheiden aus. Dadurch tritt ein datenarmes Target im späteren Teil einer Epoche nicht mehr auf. Keine gezielte Wiederholung/Gewichtung der kurzen Reihe.

Alle Varianten starten je Seed aus demselben synthetisch vortrainierten Quantil-TFT samt Scaler. Standardarchitektur, lr3e-4, Clipping7 nach Aggregation, maximal100 Epochen, Geduld10, min_delta0.0001. Bisherigen Trainer unverändert wiederverwendet. Seeds42/43/44; 5 Budgets × 6 Zielparks × 3 Methoden × 3 Seeds = 270 Fine-Tuning-Läufe.

**Validation-Budget:** Wie beauftragt ES und Auswertung auf dem ganzen Zeitraum Februar–Mai2024. Lokal: ES am jeweiligen Zielpark. Gemeinsam: ES über alle sechs Parks, inklusive Zielpark. Der Startzustand bleibt Kandidat. „Wenige Messtage“ bezeichnet somit nur das Trainingslabelbudget; das Verfahren hat weiterhin Zugriff auf umfangreiche spätere Validierungslabels des Zielparks. Es ist eine retrospektive Few-shot-Trainingsstudie, kein labelbeschränkter operativer Deployment-Nachweis. Kein unabhängiger realer Testsatz.

Zusätzlich beste tatsächlich trainierte Epoche aus dem durchlaufenen Verlauf speichern. Metriken wie im vorigen Vergleich: rohe Pinball-Verluste, Tagesmaske gti_nwp>10, sortierte/nichtnegative Quantile für Intervallmetriken und Median, Crossing-Rate. Mehrere Tagebudgets sind keine HPO.

Die vorhandenen vollständigen Daten werden zunächst durch die bisherige Pipeline geführt und dann entlang vollständiger Zielhorizonte beschränkt; der Scaler bleibt eingefroren. Ein NaN-Guard verhindert, dass der frühere Targetverlauf beim vorgelagerten KNN-Imputer in die kurze Reihe einfließt. Vollständige X/y-, Testindex- und NWP-Identität zum bisherigen Generator wird je Seed geprüft.

## Ausführung

```bash
export DATA_ROOT=/mnt/lambda1/nvme1
frcst/bin/python scripts/pv_quantile_coldstart.py --seed 42 --gpu 0 --output results/parks_pv_v4_30min/quantile_coldstart_<new_version>
```

Seeds43/44 ebenfalls ausführen. Vorhandene Seed-Verzeichnisse werden nicht überschrieben. Nach vollständigem Abschluss:

```bash
frcst/bin/python scripts/pv_quantile_coldstart_report.py results/parks_pv_v4_30min/quantile_coldstart_<new_version> --export <new_export_directory>
```

Der Report prüft alle Budget/Park/Methoden/Selection-Kombinationen. Mittelwerte: zuerst gleich über Parks je Seed, dann über Seeds; Fehlerbalken sind Seed-Standardabweichungen, keine Konfidenzintervalle. Eigene vollständige Daten und Zero-shot dienen aus dem vorherigen Versuch als Kontext, werden nicht nochmals trainiert.

## Artefakte und Grenzen

Pro Seed `metadata.json` (Hash-Pins, Quellen, Labelgrenzen, tatsächliche Issue-Liste), `metrics.csv`, `training.json`. Pro Budget/Park Checkpoints und Epochenverläufe aller Varianten sowie rohe Quantile als NPZ, alles im Framework. Nur anonymisierte CSVs, Abbildungen und Bericht ins Paper-Repo. Keine Commits ohne Go.

Das synthetische Testjahr08/2025–07/2026 bleibt unbenutzt. Die zusätzliche Donorhistorie bringt sowohl andere Parks als auch andere Jahreszeiten; beides gehört zum gewählten Einsatzfall, ist hier aber nicht kausal getrennt. Mehr gemeinsame Daten bedeuten zudem mehr Optimierungsschritte pro Epoche als lokal. Aussagegegenstand ist der gesamte Transferprozess unter dem bisherigen ES-Protokoll.

Prüfungen: Fresh Review vor Start; `tests/test_pv_quantile_coldstart.py` prüft Grenzfälle, Targetzeitzuordnung mit zeitkodierten Labels und Pipeline-Identität. Die drei bestehenden Quantiltrainer-Regressionstests gelten weiterhin.
