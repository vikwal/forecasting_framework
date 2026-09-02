#!/usr/bin/env python3
"""Ruestet die ICON-Ausschlusslogik fuer ECMWF nach.

  python3 patch_ecmwf_nan.py            Trockenlauf, prueft nur die Anker
  COMMIT=1 python3 patch_ecmwf_nan.py   schreibt

Idempotent: laeuft der Patch zweimal, meldet er "bereits gepatcht" und ruehrt
nichts an.
"""
import os
import sys
from pathlib import Path

REPO = Path.home() / "Work" / "forecasting_framework"
MARK = "exclude_run_pairs_with_ecmwf_nan"

HELPER = '''

def exclude_run_pairs_with_ecmwf_nan(
    all_run_pairs: list[tuple[int, int, int]],
    ecmwf_arrays: list,
    timestamps: pd.DatetimeIndex,
    H: int,
    F_h: int,
    max_drop_frac: float = 0.10,
) -> list[tuple[int, int, int]]:
    """Drop run pairs whose window touches a NaN in the ECMWF tensors.

    Counterpart to the ICON-D2 exclusion in the hpo_* scripts, which filters on
    the RUN axis. ECMWF is indexed by timestamp, so this filter runs over
    ``t_run_abs`` and uses the same window ``[t - H, t + F_h)`` that
    ``_build_all_run_pairs`` already applies to ``meas_nan_any``.

    Background: the ECMWF loading path has no NaN check of its own. On
    2026-08-15 a foreign pipeline overwrote the wind columns with NULL and the
    resulting tensor was NaN throughout; three workers trained on it without any
    error and only left NaN metrics behind (Optuna recorded them as pruned or
    failed). The 2026-08-17 re-export covers 2023-08-01..2026-02-28 while the
    time axis starts 2023-07-24, so 192 h of NaN remain at the front. Without
    this filter every batch touching them turns the loss into NaN.

    Raises if more than *max_drop_frac* of the pairs would be dropped, so a
    wholesale data loss stays loud instead of silently emptying the pool.
    """
    masks = []
    for arr in ecmwf_arrays:
        if arr is None or getattr(arr, "ndim", 0) != 3:
            continue
        if arr.shape[1] == 0 or arr.shape[2] == 0:
            continue
        masks.append(np.isnan(arr).any(axis=(1, 2)))
    if not masks:
        return all_run_pairs

    nan_any = np.logical_or.reduce(masks)
    if not nan_any.any():
        logger.info("ECMWF NaN audit: no missing values \\u2713")
        return all_run_pairs

    kept = [
        (rc, rh, t) for rc, rh, t in all_run_pairs
        if not nan_any[max(t - H, 0): t + F_h].any()
    ]
    dropped = len(all_run_pairs) - len(kept)
    idx = np.where(nan_any)[0]
    span = f"{timestamps[idx[0]]} .. {timestamps[idx[-1]]}"
    frac = dropped / len(all_run_pairs) if all_run_pairs else 0.0
    if frac > max_drop_frac:
        raise ValueError(
            f"ECMWF data contains NaN at {int(nan_any.sum())} of {len(nan_any)} "
            f"timestamps ({span}); that would drop {dropped} of "
            f"{len(all_run_pairs)} run pairs ({100 * frac:.1f} % > "
            f"{100 * max_drop_frac:.0f} %). Refusing to train on a partial ECMWF "
            f"archive \\u2014 check the parquet export."
        )
    logger.warning(
        "Excluded %d of %d run pairs due to NaN in ECMWF data "
        "(%d of %d timestamps affected, %s).",
        dropped, len(all_run_pairs), int(nan_any.sum()), len(nan_any), span,
    )
    return kept
'''

CALL_GRID = '''    # ── ECMWF-NaN: betroffene Run-Paare ausschliessen ────────────────────
    # Gegenstueck zum ICON-Block darueber, aber auf der Zeitachse statt auf der
    # Laufachse. Begruendung und Vorgeschichte stehen an der Funktion.
    from geostatistics.train_stgnn2 import exclude_run_pairs_with_ecmwf_nan
    all_run_pairs = exclude_run_pairs_with_ecmwf_nan(
        all_run_pairs, [grid_ecmwf_raw], timestamps, H, F_h,
    )

'''

CALL_DCRNN = '''    # ── ECMWF-NaN: betroffene Run-Paare ausschliessen ────────────────────
    # Gegenstueck zum ICON-Block darueber, aber auf der Zeitachse statt auf der
    # Laufachse. Begruendung und Vorgeschichte stehen an der Funktion.
    from geostatistics.train_stgnn2 import exclude_run_pairs_with_ecmwf_nan
    all_run_pairs = exclude_run_pairs_with_ecmwf_nan(
        all_run_pairs, [station_ecmwf_nwp, ecmwf_nwp], timestamps, H, F_h,
    )

'''

# Ohne den Kastenstrich, dessen Laenge zwischen den Dateien schwankt.
ANCHOR_NWP_ALT = "    # ── NWP-Knotenhoehen fuer die Kantenattribute "
ANCHOR_DCRNN = (
    '    logger.info("Total pre-test run pairs available for CV: %d", len(all_run_pairs))\n'
)

# Der harte Abbruch in hpo_dcrnn.py weicht dem Ausschluss: eine kleine Luecke am
# Rand der Zeitachse soll den Worker nicht am Starten hindern, ein Totalverlust
# schlaegt weiterhin zu, dann aber in der Ausschlussfunktion ueber max_drop_frac.
DCRNN_RAISE_OLD = '''            if ecmwf_nan_station > 0 or ecmwf_nan_grid > 0:
                raise ValueError(
                    f"ECMWF data contains NaN in training window — "
                    f"station array: {ecmwf_nan_station} NaN, "
                    f"grid array: {ecmwf_nan_grid} NaN."
                )
'''
DCRNN_RAISE_NEW = '''            if ecmwf_nan_station > 0 or ecmwf_nan_grid > 0:
                # Frueher ein harter Abbruch. Seit 2026-08-17 uebernimmt
                # exclude_run_pairs_with_ecmwf_nan die betroffenen Run-Paare und
                # bricht selbst ab, wenn zu viele wegfallen. Ein Rand-Loch in der
                # Zeitachse soll den Worker nicht mehr am Starten hindern.
                logger.warning(
                    "ECMWF data contains NaN in training window — "
                    "station array: %d NaN, grid array: %d NaN. "
                    "Affected run pairs will be excluded.",
                    ecmwf_nan_station, ecmwf_nan_grid,
                )
'''

TARGETS = [
    ("geostatistics/train_stgnn2.py", "append_helper", None, None),
    ("geostatistics/hpo_mtgnn.py",    "insert_before", ANCHOR_NWP_ALT, CALL_GRID),
    ("geostatistics/hpo_wavenet.py",  "insert_before", ANCHOR_NWP_ALT, CALL_GRID),
    ("geostatistics/hpo_dcrnn.py",    "insert_before", ANCHOR_DCRNN,   CALL_DCRNN),
]


def main():
    commit = bool(int(os.environ.get("COMMIT", "0")))
    plans = []

    for rel, mode, anchor, payload in TARGETS:
        path = REPO / rel
        src = path.read_text()
        if MARK in src:
            print(f"[bereits gepatcht] {rel}")
            continue
        if mode == "append_helper":
            if "\ndef load_ecmwf_parquet_at_stations_and_grid(" not in src:
                print(f"ABBRUCH {rel}: erwartete Funktion nicht gefunden"); sys.exit(1)
            new = src.rstrip("\n") + "\n" + HELPER
        else:
            n = src.count(anchor)
            if n != 1:
                print(f"ABBRUCH {rel}: Anker {n}x gefunden, erwartet genau 1")
                sys.exit(1)
            new = src.replace(anchor, payload + anchor, 1)
        plans.append((path, rel, new))
        print(f"[patch]  {rel}  ({len(new) - len(src):+d} Zeichen)")

    # dcrnn: harter Abbruch entschaerfen
    dp = REPO / "geostatistics/hpo_dcrnn.py"
    dsrc = next((n for p, _, n in plans if p == dp), None)
    if dsrc is None:
        dsrc = dp.read_text()
    if DCRNN_RAISE_NEW.strip() in dsrc:
        print("[bereits gepatcht] hpo_dcrnn.py raise")
    elif dsrc.count(DCRNN_RAISE_OLD) == 1:
        dsrc = dsrc.replace(DCRNN_RAISE_OLD, DCRNN_RAISE_NEW, 1)
        plans = [(p, r, n) for p, r, n in plans if p != dp] + [(dp, "geostatistics/hpo_dcrnn.py", dsrc)]
        print("[patch]  hpo_dcrnn.py: harter ECMWF-Abbruch -> Warnung + Ausschluss")
    else:
        print(f"ABBRUCH hpo_dcrnn.py: raise-Block {dsrc.count(DCRNN_RAISE_OLD)}x gefunden, erwartet 1")
        sys.exit(1)

    if not plans:
        print("nichts zu tun.")
        return

    if not commit:
        print("\nTrockenlauf, nichts geschrieben. COMMIT=1 setzen.")
        return

    import py_compile, tempfile
    for path, rel, new in plans:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(new)
            tmp = fh.name
        try:
            py_compile.compile(tmp, doraise=True, cfile=tmp + "c")
        except py_compile.PyCompileError as e:
            print(f"ABBRUCH {rel}: Syntaxfehler nach dem Patch:\n{e}")
            sys.exit(1)
        finally:
            os.unlink(tmp)
            if os.path.exists(tmp + "c"):
                os.unlink(tmp + "c")
        path.write_text(new)
        print(f"geschrieben: {rel}")


if __name__ == "__main__":
    main()
