# Stationsaufteilung Solar — Entwurf und Übertrag von Wind

Stand 18.08.2026.

## 1. Warum der geerbte Split nicht taugt

Die Solar-Configs erben ihre Rollen aus `data/station_split.csv` (13.04.2026,
nicht unter Versionskontrolle), erzeugt von `misc/get_trainvaltest_stations.py`
per Kennard-Stone für das **Wind**-Experiment: 103 train / 50 val / 50 test von
203 Stationen.

Für Solar bleiben davon 85 übrig (111 Stationen führen `ghi`/`dhi` gar nicht,
8 haben zu kurze Reihen) → 52 / 18 / 15. Der Filter lief **nach** dem
Kennard-Stone-Entwurf und war selbst nicht geografisch. Folge:

| mittlere Distanz zur nächsten Trainingsstation | Wind (203) | Solar (85) |
|---|---|---|
| val | 22.5 km | 50.3 km |
| test | 36.0 km | 51.5 km |

Dazu kommt: nur 18 Stationen werden je ausgewertet, 15 nie benutzt, und
dieselben 18 haben über rund 15 Experimente die Konfigurationsauswahl getragen.

## 2. Der entscheidende Unterschied zu Wind

| | Wind-GNN | Solar-TFT |
|---|---|---|
| `next_n_stations` | > 0 | **0** |
| Nachbarstation ist | **Modelleingang** | nichts |
| „Distanz zur nächsten Train-Station" misst | Datenverfügbarkeit | Schwierigkeit der Aufgabe |

`geostatistics/make_spatial_folds.py` optimiert die Distanz **klein**, weil das
Wind-Modell den Nachbar-Messkanal braucht. Für Solar ist eine kleine Distanz
kein Selbstzweck — sie macht die Aufgabe nur leichter. Der Übertrag ist deshalb
nicht mechanisch.

## 3. Die Netzdichte begrenzt, was möglich ist

Der **mediane Stationsabstand im Solar-Netz beträgt 48.1 km** (geodätisch,
WGS-84). Damit ist eine Val-Station im Mittel nicht näher als ~48 km an einer
Trainingsstation zu bekommen — unabhängig von der Strategie. Die Wind-Zahl von
22.5 km ist für Solar strukturell unerreichbar.

Gemessen, 83 Stationen, 4 Folds:

| Strategie | median Nachbardistanz | größte Lücke | Merkmals-Ungleichgewicht |
|---|---|---|---|
| geblockt (Längenstreifen) | 86.4 km | 250.9 km | 1.416 |
| zufällig | 53.6 km | 139.3 km | 0.529 |
| **gestreut (dispersed)** | **53.7 km** | **112.0 km** | **0.114** |

Gestreut und zufällig sind im Median praktisch gleich — das ist die Netzdichte,
nicht die Strategie. Der Gewinn von *dispersed* liegt woanders: die größte
Lücke ist 20 % kleiner, und das Ungleichgewicht der Fold-Mittel ist **um Faktor
4.6 geringer**. Das ist der Punkt, der zählt, wenn man Fold-Ergebnisse mitteln
oder gegeneinander halten will.

Balanciert wird auf dem, was das Modell wirklich als statische Eingänge sieht —
`altitude`/`latitude`/`longitude` — plus dem mittleren gemessenen GHI **im
Trainingszeitraum** als Regimevariable (Terrain-Features gibt es nur für die
Wind-Stationen). Es fließt nichts aus dem Testfenster ein.

## 4. Der Aufbau: Testsatz zurückhalten, HPO auf 3 rotierenden Folds

Zwei Stufen, weil die Pipeline am Ende gegen etwas gemessen werden muss, das an
keiner Stelle in die Auswahl eingeflossen ist.

```
83 brauchbare Stationen
├── 21 Testsatz          nie im Training, nie in der HPO — nur die Schlussmessung
└── 62 Pool              3 rotierende Folds, je 41 train / 21 val
    ├── spatial_fold1    41 / 21
    ├── spatial_fold2    41 / 21
    └── spatial_fold3    42 / 20
```

Jede der 62 Poolstationen ist in genau einem Fold Ziel — die HPO bewertet also
über den vollen Pool, nicht über eine feste Teilmenge. Danach wird auf allen 62
trainiert und auf den 21 Teststationen gemessen.

**Testauswahl.** Dieselbe gestreute Logik wie bei den Folds, nur mit
Gruppengröße K = round(N / n_test) = 4: `make_dispersed` bildet räumlich
benachbarte Vierergruppen und verteilt je einen Partner pro Gruppe; Gruppe 0
wird der Testsatz. Damit hat jede Teststation ihre unmittelbaren Nachbarn im
Pool, und der Testsatz deckt das Gebiet gleichmäßig ab — anders als bei reinem
Kennard-Stone, das die Randstationen zuerst zieht und den Test damit zur
Extrapolationsaufgabe machen würde.

Gemessen:

| | Wert |
|---|---|
| Test → nächste Pool-Station | median 57.4 km, p90 77.0 km, max 103.9 km |
| Merkmalsabweichung des Testsatzes vom Gesamtmittel | max **0.077 σ** |

Der Testsatz ist also weder systematisch schwerer noch leichter als die
Fold-Val-Mengen (53.1 km im Median) und in Höhe, Lage und Strahlungsregime
praktisch deckungsgleich mit der Gesamtmenge.

**Anzahl der Folds.** Die 3 sind gesetzt. Zum Preis: bei 3 Folds sieht die HPO
41 Trainingsstationen, das finale Modell dann 62 — die Hyperparameter werden
also auf einer um ein Drittel kleineren Menge gewählt, als sie am Ende bedienen.
Das ist bei k-facher CV normal, aber es ist der Grund, kein Modell mit stark
datenmengenabhängiger Kapazität zu erwarten. Mit 4 Folds wären es 46 statt 41.

Erzeugt mit:

```
frcst/bin/python scripts/make_solar_folds.py --n-folds 3 --n-test 21 \
    --write configs/solar_folds.yaml
```

Das Format ist dasselbe wie `configs/spatial_folds.yaml` (`spatial_fold1..3`
mit `files`/`val_files`), damit Dashboard und `geostatistics/spatial_cv.py` es
ohne Anpassung lesen. Der Testsatz steht als eigener Top-Level-Schlüssel
`test_files` daneben — `load_spatial_folds` liest nur `spatial_fold*` und
ignoriert ihn.

## 5. Zwei Stationen fallen raus

Nicht nach Dateigrenzen, sondern nach **Abdeckung mit nicht-NaN Messwerten je
Fenster** — alle 85 Parquets laufen von 2023-07 bis 2026-08 durch, `ghi` ist
darin aber teils fast vollständig NaN:

| Station | Training | Test | |
|---|---|---|---|
| 04642 | **0.0 %** | 13.8 % | Messungen beginnen erst 2025-06 |
| 04887 | **41.4 %** | 0.7 % | verstreut, nur 4 vollständige Testläufe |
| 00853 | 97.2 % | 66.6 % | knapp, aber brauchbar |
| übrige 80 | ≥ 95 % | ≥ 95 % | |

Die Schwelle liegt bei 50 % und schneidet sauber zwischen 41.4 % und 66.6 %.

Bis zum 18.08.2026 riss 04887 den gesamten Lauf mit: der `EmptySplitError`-
Wächter zählte Zeilen (384 ≠ 0), entscheidend sind aber Läufe, die für
`lookback + horizon` reichen. `utils/preprocessing.py:4018` prüft das jetzt
nach dem Fenstern und überspringt die Station, statt abzubrechen.

## 6. Early Stopping

`train_cl.py:524` übergibt die Auswertungsdaten als Validierungsset — die Epoche
wird also auf denselben Stationen gewählt, auf denen anschließend berichtet
wird. Gemessene Größenordnung: die Val-Kurve schwankt zwischen benachbarten
Epochen um rund 0.5 RMSE bei 63–75 RMSE Niveau, der Optimismus liegt damit
grob bei einem Prozent.

**Entschieden (Viktor, 18.08.2026): vernachlässigbar, kein Umbau.** Ein
zeitlicher Rückhalt auf den Trainingsstationen wäre die saubere Variante,
lohnt den Eingriff in `train_cl.py` aber nicht.

## 7. Dashboard

`geostatistics/fold_dashboard.py` — Karte je Fold, Distanz jeder Val-Station zu
ihrem nächsten Trainingsnachbarn, Balance zwischen den Folds. Die Sidebar hat
eine Use-Case-Voreinstellung (Wind / Solar); beide Pfade bleiben editierbar.

Als Dienst auf **Port 8511** (8504 = optuna-dashboard, 8510 = optuna-native):

```
sudo cp deploy/fold-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fold-dashboard
```
