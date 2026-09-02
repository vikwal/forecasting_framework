# Auftrag: Windgeschwindigkeits-Imputation auf die TFT-Werte umstellen

Geschrieben 2026-09-02 aus dem Schwesterprojekt `~/Work/NWP/ERA5` heraus, nachdem
dort die echten Messlücken mit dem Wind-Abschlussmodell gefüllt wurden. Du
arbeitest in `~/Work/forecasting_framework`.

## Was passiert ist

Die Dateien unter `interpol/wind` sind **heute, 2026-09-02 08:08, ersetzt worden**.
Sie tragen nicht mehr die Kriging-/IDW-/Kriging-Ordinary-Vorhersagen, sondern die
Vorhersage eines Temporal-Fusion-Transformers, der auf 203 Stationen trainiert
wurde und Lücken aus ERA5 plus Nachbarstationen füllt.

```
von l2/ws:  /mnt/lambda1/nvme1/synthetic/interpol/wind
von l1:     /mnt/nvme1/synthetic/interpol/wind
```

Beides ist derselbe Baum (NFS-Mount von L1s lokaler NVMe). 203 Stationsdateien
plus `_bericht.csv` und `_herkunft.json`.

**Der alte Bestand ist vollständig gesichert** und liegt unverändert daneben:

```
.../synthetic/interpol/wind_vor_tft_20260902
```

## Das ist die eigentliche Aufgabe

Alle Preprocessing-Pipelines sollen den TFT-Wert benutzen statt dessen, was sie
heute benutzen. Finde jede Stelle, die Lücken in `wind_speed` füllt, stelle sie
um, und belege die Umstellung mit einer Messung.

## Was sich am Dateiformat geändert hat — das ist der Kern

**Alt** (in der Sicherung noch so):

| Spalte | |
|---|---|
| `station_id`, `timestamp` | |
| `wind_speed_raw` | Messung, NaN in der Lücke |
| `wind_speed_observed` | Messung, in der Lücke der Kriging-Wert |
| `rk_pred` | Regression-Kriging |
| `idw_pred` | inverse Distanzwichtung |
| `ok_pred` | Ordinary Kriging |

**Neu**:

| Spalte | |
|---|---|
| `station_id`, `timestamp` | |
| `wind_speed_raw` | Messung, NaN in der Lücke — **unverändert**, nachgemessen |
| `wind_speed_observed` | Messung, in der Lücke der **TFT**-Wert |
| `imputed` | die TFT-Vorhersage [m/s] |
| `n_fenster` | über wie viele überlappende Fenster gemittelt wurde |
| `kontextfrei` | True = im ganzen 48-h-Fenster keine eigene Messung dieser Station |

**`rk_pred`, `idw_pred` und `ok_pred` gibt es in den neuen Dateien nicht mehr.**
Jede Stelle, die sie liest, wirft ab sofort einen `KeyError` bzw. bekommt von
`pd.read_parquet(..., columns=["timestamp", "rk_pred"])` einen Fehler. Das ist
kein Nebeneffekt, den man wegfangen soll — es ist der Anlass dieser Umstellung.

Zwei weitere Unterschiede, die leicht übersehen werden:

1. **Der Zeitraum ist gewachsen.** Neu 26 496 Stunden, 2023-07-24 00:00 UTC bis
   2026-07-31 23:00 UTC. Alt waren es 25 326 Stunden, 2023-07-24 07:00 bis
   2026-06-13 12:00. Alle Stationen haben dasselbe Raster. Wer bisher stillschweigend
   annahm, dass Interpol- und Messreihe deckungsgleich sind, prüft das neu.
2. **Emden (`05839`) fehlt bewusst.** 94.4 % Fehlanteil, ein Block von 24 393 h.
   203 Stationen statt 204 — dieselbe Menge, die `interpol/wind` schon vorher führte.

## Wo im Repo angefangen wird

- `utils/imputation.py` ist die zentrale Leserstelle: `load_interpol_imputation`
  liest fest `columns=["timestamp", "rk_pred"]` (Zeile ~65), `impute_dfs_with_kriging`
  ebenso (~203). Beide Namen tragen „Kriging" im Namen und meinen jetzt etwas anderes —
  überlege, ob umbenennen ehrlicher ist als den Inhalt zu tauschen.
- `utils/era5_imputation.py` ist der **aktive** Pfad — siehe den Absatz unten.
- `utils/preprocessing.py` liest ebenfalls `rk_pred`.
- Die Aufrufer sitzen in `geostatistics/` (elf Dateien): `train_dcrnn.py`, `hpo_dcrnn.py`,
  `train_mtgnn.py`, `hpo_mtgnn.py`, `train_wavenet.py`, `hpo_wavenet.py`,
  `get_test_results_{dcrnn,wavenet,mtgnn}.py`, `baselines/dataset.py`,
  `run_spatial_interpolation.py`, `evaluate_reference.py`.
- Pfade in den Konfigurationen: `configs/config_wind_interpol.yaml:45` und
  `configs/config_spatial_interpolation_regen.yaml:29` zeigen auf dasselbe
  Verzeichnis, einmal aus L1-, einmal aus L2-Sicht.

**Achte auf den heutigen Stand, nicht auf die Namen — es sind ZWEI Pfade.**

In mehreren Skripten steht schon `apply_interpol_imputation(meas_raw, era5_pred, ...)`
mit dem Kommentar „rk_pred no longer used for imputation itself". Der heute aktive
Pfad ist **`utils/era5_imputation.py`** — eine stationsweise OLS auf vier
ERA5-Merkmalen, laut eigenem Kopfkommentar „the SOLE imputation path for
wind_speed". Der Hintergrund steht in `docs/imputation_era5_switch.md`.

Zu ersetzen sind daher **beide**: der tote Kriging-Pfad in `utils/imputation.py`
(er bricht ab sofort an der fehlenden Spalte) und der lebende OLS-Pfad in
`utils/era5_imputation.py`. Dazu `utils/preprocessing.py` und die Aufrufer unter
`geostatistics/`.

Der OLS-Pfad deckt **153 Stationen** ab und endet **2026-06-30**. Das TFT-Modell
deckt 203 Stationen bis 2026-07-31 — die Umstellung gewinnt also nicht nur Güte,
sondern auch Abdeckung. Zur Einordnung der Güte: gegen rohes ERA5 erreicht das
Modell auf künstlich verdeckten Stunden einen Skill von 0.557 (48-h-Lücke) bis
0.709 (1-h-Lücke).

**`wind_direction` bleibt unberührt.** Das TFT-Modell sagt nur die
Windgeschwindigkeit vorher. Für die Richtung verliert rohes ERA5 laut der
Vergleichsanalyse dieses Repos gegen den KNN-Imputer in jeder Windklasse
(31.5° gegen 9.1° mittlerer Fehler) — an dieser Stelle nichts ändern.

## Zwingend zu beachten

**`IMPUTATION_GUARD_VERSION` erhöhen.** `utils/data_cache.py:1129`, steht auf 3.
Diese Umstellung ändert die Eingangsdaten aller Modelle, also muss sie auf 4 —
sonst arbeiten zwischengespeicherte Datensätze mit den alten Werten weiter und
niemand merkt es. Genau ein Schritt für die gesamte Umstellung, so wie beim
ERA5-Umstieg dokumentiert. Sieh dir vorher `docs/imputation_plausibility_guard.md`
an, dort steht die Konvention.

**Nicht neu interpolieren.** `geostatistics/run_spatial_interpolation.py` schreibt
mit `target_path` in genau dieses Verzeichnis. Ein Lauf überschreibt die
TFT-Dateien mit frischem Kriging. Wenn du das Skript anfasst, sichere den Ausgang
oder lenke ihn um.

## Was du über die neuen Werte wissen musst, bevor du sie verwendest

- **Zu `imputed` gibt es keine Kennzahl und kann es keine geben.** Die Güte ist an
  künstlich verdeckten, tatsächlich beobachteten Stunden gemessen. Ob das Modell an
  den echten Lücken genauso gut ist, weiß niemand: fiel der Sensor bei Sturm aus,
  sind das systematisch andere Stunden. Schreibe in keine Auswertung eine Zahl, die
  so klingt, als sei die Imputation validiert.
- **36 % der 46 085 gefüllten Stunden sind `kontextfrei`** — dort gab es im ganzen
  48-h-Fenster keine eigene Messung, die Vorhersage speist sich allein aus ERA5,
  Nachbarstationen und Statik. Diese Spalte ist die ehrlichste Möglichkeit, in einer
  Auswertung zwischen gut gestützten und schwach gestützten Lücken zu trennen; wirf
  sie nicht weg.
- Es gibt **keine offen gebliebenen Stunden**: alle 46 085 Fehlstunden haben einen
  Wert, `imputed` ist dort nirgends NaN.
- Plausibilität ist geprüft: keine negativen Werte, Korrelation zu den alten
  `rk_pred` 0.905 bei 0.80 m/s mittlerer absoluter Abweichung, und die **gemessenen**
  Stunden sind bitgleich zur Sicherung.

## Wie das Ergebnis aussehen soll

1. Jede Fundstelle umgestellt, keine verwaiste `rk_pred`-Referenz mehr, die auf
   `interpol/wind` zielt. (Referenzen auf die Sicherung oder auf `interpol/solar`
   sind etwas anderes — Solar ist von dieser Umstellung **nicht** betroffen.)
2. `IMPUTATION_GUARD_VERSION` auf 4.
3. Eine Messung, die belegt, dass die Umstellung greift: für einen Datensatz die
   Zahl der gefüllten Zellen und die Verteilung der eingesetzten Werte vorher/nachher.
   Die Sicherung `wind_vor_tft_20260902` erlaubt den direkten Vergleich.
4. Ein Eintrag in `docs/`, im Stil von `docs/imputation_era5_switch.md`: was
   umgestellt wurde, warum, welche Messung es belegt, und der Hinweis, dass die
   imputierten Werte keine eigene Gütezahl haben.

Frag nach, bevor du etwas löschst, überschreibst oder einen Interpolationslauf
startest.
