"""
geostatistics/baselines/mos.py — MOS-regional / MOS-nearest / MOS-local.

Diese Datei ERSETZT Spezifikation §3.5 vollstaendig (Nutzerentscheidung
2026-08-10, siehe docs/baselines_verification_results.md "Aufgabe 1"). Der
Grund: ICON-D2 laeuft nur zu den Laufstunden {6, 9, 12, 15}. Bei FESTEM Lead
h nimmt ``valid_time.hour`` deshalb nur VIER Werte an. Die urspruengliche
Spezifikationsgleichung benutzte pro Lead FUeNF nur-von-der-Uhrzeit-abhaengige
Spalten (Achsenabschnitt + zwei Harmonische, je sin/cos) — fuenf linear
unabhaengige Funktionen auf vier Stuetzstellen sind unmoeglich (Rang <= 4 bei
6 bzw. 7 Spalten). Die Rangabsicherung hat deshalb korrekt gefeuert und ALLE
Vorhersagen als NaN markiert; das war kein Implementierungsfehler, sondern
ein Rangdefekt in der Spezifikation selbst.

NEUE Modellgleichung, nach der Literatur (primo2024comparison, DWD/KIT 2024):
DWDs operationelles ModelMIX trainiert "each hourly time step separately and
individually for each location", der EMOS-Standard schaetzt "locally and for
each lead time separately", und die Autoren stellen ausdruecklich fest, dass
bei getrennter Schaetzung je Lead und Lauf ein Tagesgang "is not present in
the data ... as the forecasts then all validate at the same time of the
day". Deshalb: STRATIFIZIERUNG nach (Laufstunde r, Lead h) statt eines
Tagesgangterms in der Gleichung selbst — 48 x 4 = 192 getrennte
OLS-Gleichungen je Fold und Arm. Die Laufstunde ist die Stunde von
``run_time`` (== ``timestamps[t_run_abs-1]``, Spezifikation 4.2) und wird bei
jedem Fit gegen die Daten geprueft/geloggt, nicht als {6,9,12,15} angenommen.

    y = b0 + b1 * ws_i2                     --nwp-sources icond2  (2 Parameter)
    y = b0 + b1 * ws_i2 + b2 * ws_e2        --nwp-sources both    (3 Parameter)

KEIN Tagesgangterm, KEINE Harmonischen, KEIN Saisonterm — das ist eine
ausdruckliche Nutzerentscheidung, kein Versehen: die Graphmodellpfade haben
ueberhaupt keine Kalendermerkmale (weder Tag im Jahr noch Tageszeit); ein
Saisonterm nur fuer MOS wuerde dem Boden eine Information geben, die kein
anderer Vergleichspartner hat (siehe "Fuer den Methodikteil" (e) im
Verifikationsbericht).

``ws_i2``/``ws_e2`` kommen unveraendert aus ``dataset.build_mos_rows``:
``ws_i2 = grid_icond2_runs[r_curr, h-1, nearest_i2[s,0], ws_idx]``,
``ws_e2 = grid_ecmwf_runs[t_run_abs+h-1, nearest_e2[s,0], ws_idx_e2]``, beide
Feature-Indizes namentlich bestimmt (Spezifikation 1.4).

Nachbearbeitung: Vorhersagen werden bei Null abgeschnitten
(``max(pred, 0)``), belegt an primo2024comparison ("the forecast ... is
truncated in zero"). Die Zahl der abgeschnittenen Zeilen wird geloggt.

Dieselbe Gleichung fuer ALLE DREI Varianten (MOS-regional, MOS-nearest,
MOS-local) — sie unterscheiden sich ausschliesslich in der Fit-Menge
(Spezifikation 4.1: regional gepoolt auf den 102 Trainingsstationen, nearest
je Trainingsstation einzeln mit Uebertragung auf die geodaetisch naechste
Zielstation, local je Zielstation auf ihrer eigenen Historie), niemals in der
Modellform.

Rangdefekt-Absicherung bleibt (Spezifikation 3.5, inhaltlich unveraendert):
bei ``matrix_rank(X) < X.shape[1]`` wird NICHT per Pseudoinverse gefuellt,
sondern NaN zurueckgegeben und geloggt. Sie sollte jetzt nirgends mehr
feuern (48 Leads x mind. ~368 Zeilen je (Laufstunde,Lead)-Zelle gegen 2/3
Parameter); feuert sie doch, ist DAS ein Befund.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("baselines.mos")

# ICON-D2 Laufstunden nach Spezifikation/Config (icond2_run_hours) — wird bei
# jedem Fit gegen die tatsaechlichen Daten geprueft, nicht angenommen.
EXPECTED_RUN_HOURS = (6, 9, 12, 15)


def _design(df: pd.DataFrame, nwp_sources: str) -> np.ndarray:
    cols = [np.ones(len(df)), df["ws_i2"].to_numpy(dtype=np.float64)]
    if nwp_sources == "both":
        cols.append(df["ws_e2"].to_numpy(dtype=np.float64))
    return np.stack(cols, axis=1)


def n_params(nwp_sources: str) -> int:
    return 3 if nwp_sources == "both" else 2


def _run_hour(df: pd.DataFrame) -> pd.Series:
    """Laufstunde = Stunde von ``run_time`` (== ``timestamps[t_run_abs-1]``,
    Spezifikation 4.2)."""
    return df["run_time"].dt.hour


def _log_run_hours(where: str, df: pd.DataFrame) -> None:
    hours = sorted(int(h) for h in df["run_time"].dt.hour.unique())
    unexpected = sorted(set(hours) - set(EXPECTED_RUN_HOURS))
    logger.info("%s: tatsaechliche Laufstunden = %s%s", where, hours,
                f"  UNERWARTET={unexpected}" if unexpected else "")


def fit_lead(df_cell: pd.DataFrame, nwp_sources: str) -> tuple[np.ndarray, bool]:
    """OLS fit for one (group, run_hour, lead) cell.

    Rangdefekt-Absicherung: bei ``matrix_rank(X) < X.shape[1]`` NICHT per
    Pseudoinverse fuellen, sondern NaN zurueckgeben und loggen.
    """
    X = _design(df_cell, nwp_sources)
    y = df_cell["y"].to_numpy(dtype=np.float64)
    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        return np.full(X.shape[1], np.nan), True
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, False


def predict_lead(df_cell: pd.DataFrame, beta: np.ndarray, nwp_sources: str) -> np.ndarray:
    if beta is None or np.isnan(beta).any():
        return np.full(len(df_cell), np.nan)
    X = _design(df_cell, nwp_sources)
    return X @ beta


def _clip_nonneg(preds: np.ndarray, where: str) -> np.ndarray:
    """Nachbearbeitung (Nutzerentscheidung 2026-08-10, belegt an
    primo2024comparison: "the forecast ... is truncated in zero"). NaN bleibt
    NaN -- ``np.maximum`` propagiert NaN unveraendert."""
    finite = ~np.isnan(preds)
    n_clipped = int(np.sum(finite & (preds < 0)))
    if n_clipped:
        logger.info("%s: zero-truncation -- %d/%d Vorhersagen waren negativ und wurden auf 0 abgeschnitten",
                    where, n_clipped, int(finite.sum()))
    return np.maximum(preds, 0.0)


def fit_regional(rows_train: pd.DataFrame, nwp_sources: str) -> dict[tuple[int, int], np.ndarray]:
    """One coefficient set per (run_hour, lead), pooled over ALL rows (all
    fold-train stations). Erwartete Zeilenzahl je Zelle: ~102 x 368
    (Spezifikation-Ersatz 2026-08-10)."""
    _log_run_hours("MOS-regional fit", rows_train)
    betas: dict[tuple[int, int], np.ndarray] = {}
    n_deficient = 0
    tmp = rows_train.assign(_run_hour=_run_hour(rows_train))
    row_counts = []
    for (r, h), grp in tmp.groupby(["_run_hour", "horizon"]):
        beta, deficient = fit_lead(grp, nwp_sources)
        betas[(int(r), int(h))] = beta
        n_deficient += int(deficient)
        row_counts.append(len(grp))
    logger.info("MOS-regional: %d (Laufstunde,Lead)-Zellen, Zeilen/Zelle mean=%.1f min=%d max=%d",
                len(betas), float(np.mean(row_counts)) if row_counts else float("nan"),
                min(row_counts) if row_counts else 0, max(row_counts) if row_counts else 0)
    if n_deficient:
        logger.warning("MOS-regional: %d/%d (Laufstunde,Lead)-Zellen rangdefekt", n_deficient, len(betas))
    return betas


def fit_per_station(rows_train: pd.DataFrame, nwp_sources: str) -> dict[str, dict[tuple[int, int], np.ndarray]]:
    """One coefficient set per (station, run_hour, lead) — MOS-nearest's
    train stations and MOS-local's target stations. Erwartete Zeilenzahl je
    Zelle: ~368 gegen 2/3 Parameter (Spezifikation-Ersatz 2026-08-10)."""
    _log_run_hours("MOS per-station fit", rows_train)
    out: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    n_deficient = 0
    n_total = 0
    row_counts = []
    tmp = rows_train.assign(_run_hour=_run_hour(rows_train))
    for sid, grp_s in tmp.groupby("station_id"):
        betas: dict[tuple[int, int], np.ndarray] = {}
        for (r, h), grp in grp_s.groupby(["_run_hour", "horizon"]):
            beta, deficient = fit_lead(grp, nwp_sources)
            betas[(int(r), int(h))] = beta
            n_deficient += int(deficient)
            n_total += 1
            row_counts.append(len(grp))
        out[str(sid)] = betas
    logger.info("MOS per-station: %d Stationen x %d Zellen/Station, Zeilen/Zelle mean=%.1f min=%d max=%d",
                len(out), (n_total // len(out)) if out else 0,
                float(np.mean(row_counts)) if row_counts else float("nan"),
                min(row_counts) if row_counts else 0, max(row_counts) if row_counts else 0)
    if n_deficient:
        logger.warning("MOS per-station: %d/%d (Station,Laufstunde,Lead)-Zellen rangdefekt", n_deficient, n_total)
    return out


def predict_with_regional(rows_eval: pd.DataFrame, betas: dict[tuple[int, int], np.ndarray],
                           nwp_sources: str) -> np.ndarray:
    """Same coefficients for every station — group by (run_hour, lead) only."""
    preds = np.full(len(rows_eval), np.nan, dtype=np.float64)
    tmp = rows_eval.assign(_run_hour=_run_hour(rows_eval))
    for (r, h), grp in tmp.groupby(["_run_hour", "horizon"]):
        beta = betas.get((int(r), int(h)))
        preds[grp.index.to_numpy()] = predict_lead(grp, beta, nwp_sources)
    return _clip_nonneg(preds, "MOS-regional")


def predict_with_per_station(rows_eval: pd.DataFrame,
                              station_betas: dict[str, dict[tuple[int, int], np.ndarray]],
                              nwp_sources: str) -> np.ndarray:
    """Per-(station, run_hour, lead) coefficients — MOS-local (own betas) or
    MOS-nearest (caller pre-resolves each target station's betas to its
    nearest train station's dict before calling this)."""
    preds = np.full(len(rows_eval), np.nan, dtype=np.float64)
    tmp = rows_eval.assign(_run_hour=_run_hour(rows_eval))
    for (sid, r, h), grp in tmp.groupby(["station_id", "_run_hour", "horizon"]):
        betas = station_betas.get(str(sid))
        beta = betas.get((int(r), int(h))) if betas else None
        preds[grp.index.to_numpy()] = predict_lead(grp, beta, nwp_sources)
    return _clip_nonneg(preds, "MOS-nearest/local")
