# Windrichtung auf die TFT-Werte umgestellt (2026-09-03)

**Status: UMGESETZT** am 2026-09-03. Die Windrichtung kommt nicht mehr aus dem
KNN-Cache, sondern aus `interpol/wind_richtung` — demselben Abschlussmodell, das
seit dem 2026-09-02 schon die Geschwindigkeit liefert
(`docs/imputation_tft_switch.md`). **Ohne Rückfall auf KNN**: eine Lücke, die das
Modell nicht deckt, bleibt NaN und löst den NaN-Audit aus.
Nutzerentscheidung vom 2026-09-03.

## 1. Der neue Baum

`.../synthetic/interpol/wind_richtung` (l1: `/mnt/nvme1/...`, l2/ws:
`/mnt/lambda1/nvme1/...`) — 203 Stationen, 26 496 Stunden,
2023-07-24 00:00 bis 2026-07-31 23:00 UTC, also dasselbe Raster wie
`interpol/wind`. Vorstand daneben unter `wind_richtung_vor_tft_20260903`.

Er ist eine **Obermenge** von `interpol/wind`:

| Spalte | |
|---|---|
| `wind_speed_raw`, `imputed`, `wind_speed_observed`, `n_fenster`, `kontextfrei` | wie gehabt |
| `wind_dir_raw` | Messung, NaN in der Lücke |
| `imputed_dir` | **die Vorhersage in Grad** (0–360, meteorologisch wie die Messung) |
| `wind_dir_observed` | Messung, in der Lücke `imputed_dir` |
| `imputed_dir_guete` | Länge des gemittelten Einheitsvektors über die überlappenden Fenster |
| `n_fenster_dir` | über wie viele Fenster gemittelt |

Erzeuger laut `_herkunft.json`: `wind_bc/final/richtung_vollmodell/modell_voll.pt`,
Trial 71, 67 Epochen, `schritt_h: 12`, 203 Stationen.

### Die Geschwindigkeit ändert sich mit

`imputed` ist in diesem Baum **nicht** bitgleich zu `interpol/wind`: an
**197 von 203** Stationen weicht die Spalte ab, maximal **6,44 m/s**. Es ist ein
anderer Trainingslauf (gemeinsames Modell für Geschwindigkeit und Richtung),
keine Ergänzung. Wer Läufe von vor dem 2026-09-03 nachrechnet, arbeitet also
nicht auf denselben Eingangsdaten — auch nicht bei der Geschwindigkeit.

## 2. Was im Repo umgestellt wurde

### 2.1 `utils/imputation.py`

Neu: `IMPUTATION_COLUMN_BY_FEATURE` — welche Spalte die Lücke welcher Messgröße
füllt.

```python
IMPUTATION_COLUMN_BY_FEATURE = {
    "wind_speed":     ("imputed", "rk_pred"),
    "wind_direction": ("imputed_dir",),
}
```

`resolve_imputation_column(fpath, feature=None)` beantwortet damit zwei Fragen:
ohne `feature` wie bisher die Zielspalte (`KeyError`, wenn keine passt — ein
Verzeichnis ohne brauchbare Spalte ist nicht das, was der Aufrufer meint); mit
`feature` die Spalte dieser Messgröße, und **`None` ist dort eine normale
Antwort**, kein Fehler. Sie heißt „dieser Baum führt nichts für diese
Messgröße", und der Aufrufer entscheidet, was folgt.

`impute_meas_raw_from_interpol(..., secondary_cols=True)` füllt jetzt neben der
Zielspalte auch jede weitere Messspalte, die die Karte kennt **und** der Baum
tatsächlich führt. Die Diagnose trägt zusätzlich `handled_cols` (alles, was aus
`interpol/` kam) und `secondary` (Zähler je Spalte).

**Solar bleibt unberührt:** `interpol/solar` führt weiter nur `rk_pred` und kein
`imputed_dir`; für dessen Sekundärspalten liefert die Auflösung `None`, und der
KNN-Pfad bleibt dort zuständig wie bisher.

### 2.2 Kein Rückfall — an beiden Enden

Alle elf Aufrufer lassen die Spalten in `handled_cols` beim KNN-Schritt aus:
`train_dcrnn`, `train_mtgnn`, `train_wavenet`, die drei `get_test_results_*`,
die drei `hpo_*` und `evaluate_reference`. Damit landen nie zwei Quellen in
einer Spalte, und eine im Interpol-Baum verbliebene Lücke erreicht den
NaN-Audit, statt still vom KNN-Imputer gefüllt zu werden.

In den neun Configs unter `configs/testmode/` ist `knnimputer_path` zusätzlich
**auskommentiert**. Das ist die stärkere Zusage: dort existiert überhaupt kein
Rückfallpfad mehr, auch nicht versehentlich.

### 2.3 Cache

`IMPUTATION_GUARD_VERSION` 4 → 5. Der Pfadwechsel allein hätte den Cache-Key
schon geändert (`interpol_path` steckt als Zeichenkette darin), aber der
Codewechsel — Richtung nicht mehr aus KNN — gehört sichtbar in den Schlüssel.

## 3. Beleg

Messung über die 203 Stationen der Testmode-Configs, Zeitachse des Frameworks
(2023-07-24 … 2026-09-01, 5 534 592 Zellen), Skript `scripts/measure_dir.py`
(read-only, wiederholbar).

**48 429 Richtungslücken (0,875 %).** Davon füllt:

| Quelle | gefüllt | Anteil | offen |
|---|---:|---:|---:|
| KNN-Cache (bisher) | 48 429 | 100 % | 0 |
| **`imputed_dir` (neu)** | **46 288** | **95,6 %** | **2 141** |

Die 2 141 offenen Zellen liegen **vollständig** jenseits des 2026-07-31 23:00 —
dort endet der Interpol-Baum, während die Messreihen bis 2026-09-01 laufen. In
jedem der neun Auditfenster (`test_end` 2025-12-01 / 2026-04-01 / 2026-07-31)
bleiben **0** Lücken, bei Geschwindigkeit wie bei Richtung. Genau deshalb trägt
der Verzicht auf den Rückfall.

Wo beide Quellen füllen (46 288 Zellen), sind sie sich **nicht** einig:
mittlere Winkeldifferenz **20,5°**, Median 11,6°, p95 73,7°. Die Umstellung ist
also eine inhaltliche Entscheidung, keine kosmetische.

`imputed_dir_guete` liegt an den Füllstellen bei Median 1,000 (p05 0,993, kein
Wert unter 0,5). **Das ist keine Fehlerangabe.** Laut `_herkunft.json` misst die
Zahl die Einigkeit der überlappenden Vorhersagefenster, nicht die Abweichung in
Grad — eine hohe Güte heißt „das Modell ist sich sicher", nicht „das Modell hat
recht". Wie schon bei `imputed`: zu diesen Werten gibt es keine Gütezahl an den
echten Lücken, und in keine Auswertung gehört eine Zahl, die so klingt.

### Ein überholter Zwischenstand

Eine erste Messung um 12:03 ergab 216 verbliebene Richtungslücken **innerhalb**
der Auditfenster. Der Bestand wurde um 12:12 ersetzt (Sicherung des Vorstands
unter `wind_richtung_vor_tft_20260903`); die Wiederholung um 12:24 ergab 0. Die
216 beziehen sich auf einen Stand, den es nicht mehr gibt — hier nur notiert,
damit die Zahl niemanden verwirrt, der die Sitzungsprotokolle liest.

## 4. Nicht betroffen

- **Solar** — eigener Baum, eigener Cache, unveränderte Logik (§ 2.1).
- **Der KNN-Cache selbst.** `knnimputer/wind` bleibt liegen, wie er am
  2026-09-02 neu gerechnet wurde (`docs/imputation_knn_regen_20260902.md`). Für
  Wind wird er von den Testmode-Läufen nicht mehr gelesen; andere Pipelines, die
  ihn noch benutzen, funktionieren unverändert.
- **Bereits geschriebene Ergebnisdateien.** Die Imputationsmaske der Auswertung
  baut `make_stdhp_figures.build_imputation_mask` aus den Rohmessdateien, nicht
  aus `interpol/`.
