# Optuna Dashboard (`optuna_dashboard.py`)

Streamlit-basiertes Dashboard zur Visualisierung und Verwaltung von Optuna HPO-Studies.

## Deployment

Läuft als systemd-Service auf Port `8503`:

```
/etc/systemd/system/optuna-dashboard.service
```

Service-Befehle:
```bash
sudo systemctl start optuna-dashboard.service
sudo systemctl stop optuna-dashboard.service
sudo systemctl restart optuna-dashboard.service
sudo systemctl status optuna-dashboard.service
```

## Datenbank-Anbindung

Die Datenbank-URL wird über die Umgebungsvariable `OPTUNA_STORAGE` konfiguriert.

Da der Service unter systemd läuft (erbt keine Shell-Umgebung), muss die Variable direkt in der Service-Unit gesetzt werden:

```bash
sudo systemctl edit optuna-dashboard.service
```

Override-Datei (`/etc/systemd/system/optuna-dashboard.service.d/override.conf`):
```ini
[Service]
Environment="OPTUNA_STORAGE=postgresql://user:password@host:port/dbname"
```

Nach Änderungen:
```bash
sudo systemctl daemon-reload
sudo systemctl restart optuna-dashboard.service
```

**Hinweis:** `.bashrc`-Exports werden von systemd-Services nicht eingelesen — die Variable muss zwingend über die Unit-Datei gesetzt werden.

Fallback: Wenn `OPTUNA_STORAGE` nicht gesetzt ist, wird automatisch `studies/optuna_studies.db` (SQLite) verwendet.

## Study löschen

Das Löschen einer Study erfordert eine Passwort-Bestätigung:

1. Study im Dropdown auswählen
2. "Delete Study" klicken
3. Sudo-Passwort eingeben
4. "Yes, Delete" bestätigen

Das Passwort wird via `sudo -S -k /bin/true` verifiziert — der Service-User (`viktorwalter`) muss daher in der `sudoers`-Gruppe sein.
