---
description: Änderungen committen, pushen und l1/ws nachziehen — fragt nach, wenn dort etwas läuft
argument-hint: "[commit message]"
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git add:*), Bash(git commit:*), Bash(git push:*), Bash(bash scripts/sync.sh:*)
---
Stand der drei Hosts vor dem Sync:

!`bash scripts/sync.sh check`

Ablauf:

1. **Commit auf l2.** Zeige `git status --short` und `git diff --stat`. Untracked Dateien nur aufnehmen, wenn sie erkennbar zum Änderungssatz gehören — `figures/`, `paper_export/` und Ähnliches bleiben draußen. Commit-Message: `$ARGUMENTS`; ist das leer, formuliere sie selbst im Stil des Repos (`git log -5` als Vorlage: erste Zeile imperativ, dann das Warum, nicht das Was). Gibt es nichts zu committen, sag das und mach mit Schritt 2 weiter.
2. **Push** nach `origin main`.
3. **Pull auf l1 und ws** mit `bash scripts/sync.sh pull <host>` — aber nur für Hosts, die im Check oben mit `pull unbedenklich` markiert sind. Steht dort `NACHFRAGEN`, zeige die Begründung (welche Prozesse, welche lokalen Änderungen, welche Dateien der Pull ändern würde) und frage, bevor du pullst. `aktuell` braucht nichts.
4. **Abschluss:** eine Zeile pro Host mit HEAD. Wenn alle drei gleich sind, sag das in einem Satz.
