# Synthetischer Quantiltransfer (2026-11-09)

Beauftragt: 60 bisherige Nicht-Client-Parks als Vortrainingsquelle, 166 Client-Parks in acht bestehenden Clients als Transferziele. Keine Überschneidung der Quellen und Ziele. Die 60 Parks haben in diesem Experiment die Rolle der Vortrainingsparks, nicht die eines räumlichen Holdouts.

Training Quelle und Fine-Tuning: 24.07.2023–31.07.2024. ES und Evaluation: 01.08.2024–31.07.2025 wie vereinbart. Zusätzliche Testjahr-Labels ab 01.08.2025 werden beim Laden ausgeschlossen. Parks ohne vollständige Trainings- oder Auswertungsfenster werden mit IDs und Fensterzahlen in metadata.json protokolliert; alle vier Varianten werden auf demselben verfügbaren Zielbestand verglichen.

TFT, sieben Quantile 0.05/0.1/0.25/0.5/0.75/0.9/0.95, keine Leistungshistorie; vier bekannte m4-Features. Scaler ausschließlich auf Quell-Trainingsdaten fitten und beim Transfer unverändert verwenden. Seeds 42/43/44 mit eigenem Vortraining. Ein gemeinsamer Ausgangscheckpoint je Seed.

Zero-shot ohne Anpassung. Lokal acht Modelle, eines je Client mit allen verfügbaren Trainingsdaten seiner Parks. Zentral und FedGradient derselbe Batchplan, bis zu 8×11 Fenster je Update, Gewichtung nach tatsächlicher Batchgröße, globales Gradient-Clipping 7 nach Aggregation. Quelle/lokal Batch64. Adam 3e-4 für Vortraining und Fine-Tuning, maximal100 Epochen, patience10, min_delta0.0001, Startmodell als ES-Kandidat. Auswahl nach Pinball über alle Zellen im jeweiligen Auswertungsbestand; bei Vortraining ausschließlich Quellparks, lokal eigener Client, zentral/föderiert alle Zielparks. Kein HPO.

Runner: scripts/pv_synthetic_transfer.py, train()-Wiederverwendung aus pv_quantile_finetune.py. Kampagne: scripts/pv_synthetic_transfer_campaign.py. Auswertung: scripts/pv_synthetic_transfer_report.py. Lauf: results/parks_pv_v4_30min/synthetic_transfer_2026-11-09_v1. Automatischer Bericht nach Abschluss aller Seeds im Paper-Repo: results/pv_synthetic_transfer_2026-11-09/bericht_2026-11-09.md.

Primär roher Tages-Pinball, gleiche Gewichte je Park und Mittel/SD über drei Seeds. Zusätzlich pro Client/Seed, Coverage, Interval Score, Median-R², Crossing. CSVs mit Datum und Code-Pin. Vorhersagen/Checkpoints bleiben im Framework. Keine Commits.
