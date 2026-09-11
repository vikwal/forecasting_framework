---
description: Trainings-Status auf l1, l2 und ws — GPUs, Sessions, Prozesse, Queue, Logs, Git-Stand
allowed-tools: Bash(bash scripts/status.sh:*)
---
!`bash scripts/status.sh`

Fasse den Status oben knapp zusammen — was läuft wo (Host, GPU, Modell, Laufzeit, letzte Epoche), was ist fertig, und was ist auffällig: idle Worker, die nur warten, STOP-Dateien, Repos mit ungleichem HEAD oder vielen geänderten Dateien, GPUs ohne Prozess bei hoher Belegung. Rohdaten nicht wiederholen. Wenn nichts läuft, sag das in einem Satz.
