#!/usr/bin/env python3
"""
make_stdhp_figures.py — headless static figures & tables for the "stdhp" dry run
(10 model variants x 3 spatial folds, fixed standard hyperparameters instead of
per-fold HPO-best params).

Run with e.g.:
    python geostatistics/stdrun/make_stdhp_figures.py --out-dir geostatistics/figures/stdhp
    python geostatistics/stdrun/make_stdhp_figures.py --skip-raw       # unfiltered CSV fallback
    python geostatistics/stdrun/make_stdhp_figures.py --keep-imputed  # old, unfiltered numbers

AGGREGATION CONVENTION — read this before quoting a number from this script's output:
  - IMPUTED-HOUR FILTERING (default on, --keep-imputed restores the old behaviour): ~0.6% of
    evaluated station-hours have an IMPUTED target -- the station had no raw measurement at that
    hour, and the interpol_path (regression kriging) / knnimputer_path (KNN) chain filled it in,
    in some cases extrapolating to non-physical values (gt as low as -102 m/s, as high as +74.9
    m/s). RMSE on those hours sits around 5.2 m/s for every model AND for raw ICON-D2 alike (vs.
    ~1.2 m/s overall) -- the target is close to noise there, not the weather being hard, so
    scoring them measures agreement with an imputation artifact, not forecast skill. They are
    excluded from every table/figure below by default (they remain in training, where using a
    filled value is unremarkable there). See build_imputation_mask() / verify_imputation_mask()
    for how the mask is reconstructed and self-checked, and imputed_coverage.md/csv for the
    magnitude per fold/station. The stdhp_*.csv files were written on the UNFILTERED sample and
    are therefore no longer the source for the reported per-station metrics -- see
    build_station_metrics(); they are used only by _crosscheck_csv_vs_raw(), deliberately against
    the unfiltered raw aggregates, as a pure data-integrity guard.
  - "per-fold value" for a variant = unweighted mean of the per-station metric (rmse / mae / r2 /
    skill / skill_nwp) over the 51 target stations of that fold, i.e. exactly the convention
    already used upstream in the stdhp_*.csv files (one row per station, mean-of-stations).
  - "overview value" (mean ± sd) for a variant = mean / population-sd (ddof=0) of the 3 per-fold
    values above. Each fold counts once, independent of how many valid samples it happens to
    contain — folds are not re-weighted by n_samples.
  - "pooled RMSE" (columns with suffix _pooled) = sqrt(mean of squared residuals over ALL rows of
    ALL 51 stations of a fold), i.e. every (station, run_time, horizon) sample gets equal weight
    instead of every station getting equal weight. Stations with systematically larger errors (or
    more valid samples) dominate this number more than they do the per-station mean. The two
    numbers are reported side by side deliberately — they can and do diverge, and the paper needs
    to know which one is being quoted.
  - Horizon / month / wind-speed-class curves use the POOLED convention WITHIN a fold (sqrt of
    summed squared residuals over all stations of that fold), and only the across-fold step is
    the same as above (each fold counts once). They are therefore NOT directly comparable to the
    overview table's per-station-mean RMSE column -- see the "RMSE pooled" column for the number
    that IS on the same convention as those figures, and quote figure levels against it, never
    against the per-station column.
  - SKILL AVERAGING: the skill / skill_nwp columns of the overview table are the mean over
    stations of the PER-STATION skill (1 - rmse_station / rmse_ref_station), then the mean over
    folds. That is NOT the same as 1 - mean(RMSE) / mean(RMSE_ref); the latter is systematically
    ~0.02-0.03 higher here and even reorders WaveNet GRID vs. MTGNN GRID. The per-station mean is
    what evaluate_reference.py / evaluation.py write into the CSVs, so it is what this table
    reports; state the definition explicitly wherever a skill number is quoted.
  - ICON-D2 and Persistence are NOT separate model runs — there is no stdhp_*.csv for them. They
    are pseudo-models derived from the nwp_ref / pers_ref columns of the raw prediction parquets,
    which are per-construction on the same station set / time window / horizon grid as whichever
    model run they were saved alongside. Before using them the script verifies that nwp_ref /
    pers_ref / gt are numerically identical across ALL available variants of the same fold; if
    that check fails it aborts loudly (SystemExit) instead of silently mixing disagreeing
    baselines.
  - PAIRED ANALYSIS: differences are formed per (fold, station) via an explicit pivot/merge join,
    never positionally -- including CROSS-MODEL pairs (e.g. DCRNN GRID vs. MTGNN GRID), which are
    valid because every (model, variant) is evaluated on the same 51 target stations per fold.
    Covers the DCRNN ablation ladder (A-B, B-C, A-C, A-BASE, paired_diff_analysis()) plus MTGNN/
    WaveNet ablations, cross-model inductive comparisons and the per-model transductive price
    (paired_diff_generic(), CROSS_MODEL_PAIRS) -- all in the same paired_diff_stats.md/csv,
    grouped by the "group" column.
    The "pooled" rows concatenate the three folds' 51-station difference vectors into one n=153
    vector; ci95_lo/ci95_hi bootstrap-resample stations from that pooled vector directly, i.e. the
    fold structure is IGNORED there. ci95_lo_stratified/ci95_hi_stratified (pooled rows only) is
    the same bootstrap done fold-stratified instead (resample 51 stations within each fold, then
    concatenate, see _bootstrap_stratified_ci()) -- both are exported side by side. They agree to
    <0.001 m/s for every pooled row here, so the fold structure is confirmed inert for this data
    set and the simpler flat CI can be quoted; per-fold rows are still the place to look for
    between-fold variance directly.
  - ECMWF is deliberately absent from every figure and table here. The only ECMWF reference CSVs
    on disk (data/test_results/ecmwf_fold{0,1}.csv) cover the 2025-08 to 2026-04 TEST window,
    while the stdhp runs cover the 2024-08 to 2025-08 VAL window — not comparable, see task notes
    on evaluate_reference.py's hard-coded `boundary = test_start` bug.
"""
from __future__ import annotations

import argparse
import gc
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Paths & static metadata
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]          # .../forecasting_framework
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "test_results"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw_preds"
DEFAULT_OUT_DIR = REPO_ROOT / "geostatistics" / "figures" / "stdhp"
# Per-station 10-min raw measurements, read-only, used only to reconstruct which evaluated hours
# are real vs. imputed (see build_imputation_mask()). NOT geostatistics/evaluate_reference.py's
# territory -- that script and its reference runs are owned by someone else right now; this path
# is a different, read-only input.
DEFAULT_STATION_RAW_DIR = Path("/mnt/nvme1/synthetic/raw/wind")

FOLDS = [0, 1, 2]
N_STATIONS_EXPECTED = 51

# filename-prefix (without _fold{n}.csv/_raw.parquet) -> (model, variant, ablation_letter)
STDHP_META = {
    "stdhp_dcrnn_wind_dcrnn_base":     ("DCRNN",   "BASE",         "BASE"),
    "stdhp_dcrnn_wind_dcrnn":          ("DCRNN",   "GRID",         "A"),
    "stdhp_dcrnn_wind_dcrnn_nomeas":   ("DCRNN",   "GRID-NOMEAS",  "B"),
    "stdhp_dcrnn_wind_dcrnn_nograph":  ("DCRNN",   "GRID-NOGRAPH", "C"),
    # R6(a): D = fixed inverse-distance NWP aggregation (no learned attention),
    # D' = D plus a height-corrected 3D distance (nwp_attention.py). Both derive
    # from the GRID/NWP arm (variant A) exactly like B/C do.
    "stdhp_dcrnn_wind_dcrnn_idw":      ("DCRNN",   "GRID-IDW",     "D"),
    "stdhp_dcrnn_wind_dcrnn_idw_alt":  ("DCRNN",   "GRID-IDW-ALT", "D'"),
    "stdhp_dcrnn_wind_dcrnn_nwp_hist": ("DCRNN",   "GRID+HIST",    None),
    "stdhp_mtgnn_wind_mtgnn":          ("MTGNN",   "BASE",         None),
    "stdhp_mtgnn_wind_mtgnn_nwp":      ("MTGNN",   "GRID",         None),
    "stdhp_mtgnn_wind_mtgnn_nwp_hist": ("MTGNN",   "GRID+HIST",    None),
    "stdhp_wavenet_wind_wavenet":      ("WaveNet", "BASE",         None),
    "stdhp_wavenet_wind_wavenet_nwp":  ("WaveNet", "GRID",         None),
}
# ── TFT (non-graph baseline) — separate filename scheme & fold numbering ──────────────────
# Filename prefix (without _stdhp_fold{n}.csv/_raw.parquet) -> (model, variant, ablation_letter).
# "base" = inductive (no target-station history in observed_features, the actual falsification
# target of the graph-vs-non-graph contribution claim), "hist" = transductive (own history
# included, parallel to the graph models' GRID+HIST).
TFT_META = {
    "train_tft_bc_m-tft_c-wind_tft_sp_base": ("TFT", "base", None),
    "train_tft_bc_m-tft_c-wind_tft_sp_hist": ("TFT", "hist", None),
}
# TFT config-fold numbering (configs/tft_bc/config_wind_tft_sp_*_fold{1,2,3}.yaml) is 1-based; the
# graph stdhp files are 0-based. canonical_fold = tft_config_fold - TFT_FOLD_OFFSET. Verified
# INDEPENDENTLY for all 6 TFT files against configs/spatial_folds.yaml's target-station sets in
# verify_fold_assignment() -- this offset is not just trusted from prior investigation.
TFT_FOLD_OFFSET = 1
ALL_META = {**STDHP_META, **TFT_META}

TRANSDUCTIVE_VARIANTS = {("DCRNN", "GRID+HIST"), ("MTGNN", "GRID+HIST"), ("TFT", "hist")}
ABLATION_PAIRS = [("A", "B"), ("B", "C"), ("A", "C"), ("A", "BASE"),
                   # R6(a): D = idw, D' = idw_alt (height-corrected) — see STDHP_META.
                   # A-D / A-D' isolate the learned-vs-fixed NWP aggregation each; D-D'
                   # isolates the height correction alone; D'-BASE is D's counterpart to A-BASE.
                   ("A", "D"), ("A", "D'"), ("D", "D'"), ("D'", "BASE")]
ABLATION_LETTER_LABEL = {"A": "GRID", "B": "GRID-NOMEAS", "C": "GRID-NOGRAPH", "BASE": "BASE",
                          "D": "GRID-IDW", "D'": "GRID-IDW-ALT"}

# Paired comparisons beyond the DCRNN ablation ladder: (group, label, model_a, variant_a,
# model_b, variant_b). All (model, variant) combinations share the SAME 51 target stations per
# fold (configs/spatial_folds.yaml val_files), so cross-model pairs are joined on (fold,
# station_id) exactly like the within-model ones -- see paired_diff_generic().
CROSS_MODEL_PAIRS = [
    ("MTGNN ablation",          "MTGNN GRID - MTGNN BASE",             "MTGNN",   "GRID",      "MTGNN",   "BASE"),
    ("MTGNN ablation",          "MTGNN GRID+HIST - MTGNN GRID",        "MTGNN",   "GRID+HIST", "MTGNN",   "GRID"),
    ("MTGNN ablation",          "MTGNN GRID+HIST - MTGNN BASE",        "MTGNN",   "GRID+HIST", "MTGNN",   "BASE"),
    ("WaveNet ablation",        "WaveNet GRID - WaveNet BASE",         "WaveNet", "GRID",      "WaveNet", "BASE"),
    ("Cross-model (inductive)", "DCRNN GRID - MTGNN GRID",             "DCRNN",   "GRID",      "MTGNN",   "GRID"),
    ("Cross-model (inductive)", "DCRNN GRID - WaveNet GRID",           "DCRNN",   "GRID",      "WaveNet", "GRID"),
    ("Transductive price",      "DCRNN GRID - DCRNN GRID+HIST",        "DCRNN",   "GRID",      "DCRNN",   "GRID+HIST"),
    ("Transductive price",      "MTGNN GRID - MTGNN GRID+HIST",        "MTGNN",   "GRID",      "MTGNN",   "GRID+HIST"),
    # ── Graph vs. TFT (non-graph baseline) — the falsification the paper's contribution
    # paragraph explicitly promises. "DCRNN GRID - TFT base" is the single most important new
    # comparison of the whole stdhp run: graph vs. non-graph, both inductive (target station
    # unseen). WaveNet/MTGNN GRID vs. TFT base add the other two graph families; the GRID+HIST
    # pairs are the transductive-setting counterpart; "TFT base - TFT hist" is TFT's own
    # transductive price, directly comparable to the graph models' rows above; "DCRNN BASE -
    # TFT base" additionally checks whether the graph structure matters even without NWP grid
    # context (DCRNN BASE has no NWP grid access either -- see STDHP_META).
    ("Graph vs TFT (inductive)",    "DCRNN GRID - TFT base",        "DCRNN",   "GRID",      "TFT", "base"),
    ("Graph vs TFT (inductive)",    "MTGNN GRID - TFT base",        "MTGNN",   "GRID",      "TFT", "base"),
    ("Graph vs TFT (inductive)",    "WaveNet GRID - TFT base",      "WaveNet", "GRID",      "TFT", "base"),
    ("Graph vs TFT (inductive)",    "DCRNN BASE - TFT base",        "DCRNN",   "BASE",      "TFT", "base"),
    ("Graph vs TFT (transductive)", "DCRNN GRID+HIST - TFT hist",   "DCRNN",   "GRID+HIST", "TFT", "hist"),
    ("Graph vs TFT (transductive)", "MTGNN GRID+HIST - TFT hist",   "MTGNN",   "GRID+HIST", "TFT", "hist"),
    ("Transductive price",          "TFT base - TFT hist",          "TFT",     "base",      "TFT", "hist"),
]

# "wichtigste Varianten" for the WS-class / month stratification (task item 7). Includes both
# GRID+HIST variants -- they are the two best-performing variants overall (see overview table)
# and were missing here; is_transductive() marks them in the legend (see fig_error_by_ws_class /
# fig_error_by_month).
IMPORTANT_VARIANTS = [("DCRNN", "GRID"), ("DCRNN", "BASE"), ("DCRNN", "GRID+HIST"),
                      ("MTGNN", "GRID"), ("MTGNN", "GRID+HIST"), ("WaveNet", "GRID"),
                      ("TFT", "base"), ("TFT", "hist")]
IMPORTANT_REF_MODELS = ["ICON-D2", "Persistence"]

MODEL_COLORS = {
    "DCRNN": "steelblue", "MTGNN": "darkorange", "WaveNet": "forestgreen", "TFT": "#984ea3",
    "ICON-D2": "#888888", "Persistence": "#c44e52",
}
FOLD_COLORS = {0: "#1b9e77", 1: "#d95f02", 2: "#7570b3"}

# Colour encodes the model, line style encodes the variant. Without the second channel the
# 10 variants collapse onto 3 colours and e.g. "DCRNN GRID" and "DCRNN BASE" — the single most
# important contrast in the ablation — are drawn as two identical solid blue lines.
# TFT's own variant names ("base"/"hist", lowercase, no relation to the graph ABLATION_LETTER
# convention) get their own entries below, deliberately styled like the graph BASE / GRID+HIST
# they are the non-graph counterpart of (dotted / long dash-dot-dot) -- colour (purple, TFT) is
# what actually distinguishes them on a plot, the linestyle just keeps inductive-vs-transductive
# visually consistent across model families.
VARIANT_LINESTYLE = {
    "BASE":         (0, (1, 1.2)),      # dotted
    "GRID":         "solid",
    "GRID-NOMEAS":  (0, (5, 2)),        # dashed
    "GRID-NOGRAPH": (0, (3, 1.5, 1, 1.5)),  # dash-dot
    "GRID+HIST":    (0, (7, 1.5, 1, 1.5, 1, 1.5)),  # long dash-dot-dot
    "base":         (0, (1, 1.2)),      # dotted -- TFT inductive, parallel to graph BASE
    "hist":         (0, (7, 1.5, 1, 1.5, 1, 1.5)),  # TFT transductive, parallel to GRID+HIST
}

WS_BINS = list(range(0, 22, 2))
WS_LABELS = [f"[{WS_BINS[i]},{WS_BINS[i + 1]})" for i in range(len(WS_BINS) - 1)] + ["[20,inf)"]

# run_time added (beyond what the graph-only script originally needed) so the (station_id,
# run_time, horizon) row identity used for the 14-variant intersection (see
# build_intersection_keys()/_composite_key()) and for the TFT pers_ref borrow/nwp_ref cross-check
# (scan_tft_raw_parquets()) is available directly from this same single read -- no second pass.
RAW_COLUMNS = ["station_id", "run_time", "valid_time", "horizon", "pred", "gt", "nwp_ref", "pers_ref"]


def norm_station(x) -> str:
    """CSV station_id is int64 (161) for the graph files, a zero-padded string ('00161') in the
    graph parquets, and 'synth_00161.csv' in the TFT files (both CSV and parquet) -- see the TFT
    section of the module docstring. Normalize all three to a 5-digit zero-padded string so they
    can be joined/compared."""
    s = str(x).strip()
    if s.startswith("synth_"):
        s = s[len("synth_"):]
    if s.endswith(".csv"):
        s = s[:-4]
    return s.zfill(5)


def variant_label(model: str, variant: str) -> str:
    return f"{model} {variant}"


def is_transductive(model: str, variant: str) -> bool:
    return (model, variant) in TRANSDUCTIVE_VARIANTS


# ─────────────────────────────────────────────────────────────────────────────
# 0) Imputed-hour mask
# ─────────────────────────────────────────────────────────────────────────────
# ~0.6% of evaluated station-hours have an IMPUTED target: the station had no raw measurement at
# that hour, so the training pipeline's interpol_path (regression kriging) / knnimputer_path
# (KNN) chain filled it in, in a handful of cases extrapolating to non-physical values (gt as low
# as -102 m/s, as high as +74.9 m/s). On those hours RMSE sits around 5.2 m/s for every model AND
# for raw ICON-D2 alike (vs. ~1.2 m/s overall) -- the target itself is close to noise there, not
# the weather situation being hard, so scoring them measures agreement with an imputation
# artifact rather than forecast skill. Decision: excluded from evaluation by default (they stay
# in training, where using a filled value is unremarkable); --keep-imputed restores the old,
# unfiltered sample for comparison.
#
# The stdhp raw parquets do not carry an "is this imputed" flag themselves, so it is reconstructed
# from the original per-station measurement files: resample the raw 10-min wind_speed to 1h mean
# and treat any hour that comes out NaN (no raw reading at all in that hour) as imputed. This is
# verified once against the ground truth we DO have: every gt < 0 hour must be non-physical, i.e.
# imputed by construction -- if the reconstructed mask disagrees, it is wrong and the script
# aborts (see verify_imputation_mask()).
def build_imputation_mask(station_ids, station_raw_dir: Path) -> dict:
    """
    Returns {station_id: pd.Series(bool, index=hourly tz-aware UTC timestamp)}, True = imputed
    (no raw measurement in that hour). Aborts if a station's raw measurement file is missing --
    silently treating an entire station as "fully imputed" or "fully real" would bias whichever
    figure/table happens to include it without any visible trace.
    """
    mask = {}
    for sid in station_ids:
        path = station_raw_dir / f"Station_{sid}.parquet"
        if not path.exists():
            sys.exit(f"[FATAL] Keine Rohmessdatei fuer Station {sid}: {path}. Die "
                     f"Imputationsmaske kann fuer diese Station nicht rekonstruiert werden -- "
                     f"Abbruch statt sie stillschweigend als (nicht-)imputiert zu behandeln.")
        raw = pd.read_parquet(path, columns=["wind_speed"])
        hourly = raw["wind_speed"].resample("1h").mean()
        mask[sid] = hourly.isna()
    return mask


def _lookup_imputed(df: pd.DataFrame, mask: dict) -> np.ndarray:
    """
    Per-row imputed flag for `df` (needs columns station_id, valid_time), via the mask built by
    build_imputation_mask(). Positions whose valid_time falls entirely outside the station's raw
    measurement file (no hourly bucket at all, not even a NaN one) are also treated as imputed --
    there is no real measurement to point to either way -- and counted so a large count would be
    visible rather than silently assumed away.
    """
    out = np.zeros(len(df), dtype=bool)
    vt = df["valid_time"].to_numpy()
    n_out_of_range = 0
    for sid, idx in df.groupby("station_id").indices.items():
        m = mask.get(sid)
        if m is None:
            out[idx] = True
            continue
        reindexed = m.reindex(pd.DatetimeIndex(vt[idx]))
        n_out_of_range += int(reindexed.isna().sum())
        out[idx] = reindexed.fillna(True).to_numpy(dtype=bool)
    if n_out_of_range:
        warnings.warn(f"{n_out_of_range} Zeilen mit valid_time ausserhalb der Rohmessdatei "
                      f"(keine Stunde dort ueberhaupt) -- als imputiert gezaehlt.")
    return out


def _canonical_fold_frame(raw_dir: Path, fold: int, columns) -> pd.DataFrame:
    prefixes = _canonical_prefixes_for_fold(fold, raw_dir)
    if not prefixes:
        sys.exit(f"[FATAL] Keine Parquets fuer fold {fold} in {raw_dir}.")
    path = raw_dir / f"{prefixes[0]}_fold{fold}_raw.parquet"
    df = pd.read_parquet(path, columns=columns)
    df["station_id"] = df["station_id"].map(norm_station)
    return df


def verify_imputation_mask(raw_dir: Path, mask: dict, folds=FOLDS) -> dict:
    """
    Loads each fold's canonical (station_id, valid_time, gt) frame, deduped to unique hours, and
    checks that EVERY gt < 0 hour (non-physical, so certainly not a real measurement) is flagged
    imputed by the reconstructed mask. Aborts loudly if not -- a mismatch means the mask is wrong
    (wrong station file, wrong resample rule, a timezone slip), not that the data changed.
    Returns {fold: deduped canonical frame} for reuse by build_imputed_coverage_table().
    """
    canon_by_fold = {}
    for fold in folds:
        df = _canonical_fold_frame(raw_dir, fold, ["station_id", "valid_time", "gt"])
        df = df.drop_duplicates(["station_id", "valid_time"])
        neg = df[df["gt"] < 0]
        if not neg.empty:
            is_imp = _lookup_imputed(neg, mask)
            n_bad = int((~is_imp).sum())
            if n_bad:
                sys.exit(f"[FATAL] fold {fold}: {n_bad} von {len(neg)} Stunden mit gt<0 sind laut "
                         f"rekonstruierter Imputationsmaske NICHT imputiert -- die Maske ist "
                         f"falsch (falsche Stationsdatei / falsches Resampling / Zeitzonenfehler). "
                         f"Abbruch statt mit einer falschen Maske weiterzurechnen.")
        print(f"[OK] fold {fold}: alle {len(neg)} Stunden mit gt<0 sind als imputiert erkannt "
              f"(Verifikation der rekonstruierten Imputationsmaske gegen {len(df):,} eindeutige "
              f"(Station, valid_time)-Paare).")
        canon_by_fold[fold] = df
    return canon_by_fold


def build_imputed_coverage_table(canon_by_fold: dict, mask: dict, folds=FOLDS) -> pd.DataFrame:
    """Per fold and station: evaluated hours, imputed hours, share -- plus one fold-total row per
    fold. This is the number that goes in the paper to justify the filter's magnitude."""
    rows = []
    for fold in folds:
        df = canon_by_fold[fold]
        is_imp = _lookup_imputed(df, mask)
        g = (pd.DataFrame({"station_id": df["station_id"].values, "is_imputed": is_imp})
             .groupby("station_id").agg(n_hours=("is_imputed", "size"),
                                        n_imputed=("is_imputed", "sum")).reset_index())
        g["fold"] = fold
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["share_imputed"] = out["n_imputed"] / out["n_hours"]

    totals = out.groupby("fold")[["n_hours", "n_imputed"]].sum().reset_index()
    totals["station_id"] = "ALL (fold total)"
    totals["share_imputed"] = totals["n_imputed"] / totals["n_hours"]
    out = pd.concat([out, totals], ignore_index=True)
    return out.sort_values(["fold", "station_id"]).reset_index(drop=True)


def _verify_uniform_filtering(pooled_df: pd.DataFrame) -> None:
    """
    Every (model, variant) of a given fold must retain EXACTLY the same row count after imputed-
    hour filtering: the filter is keyed on (station_id, valid_time) via gt alone, and gt is
    identical across variants of the same fold (verified in verify_and_derive_references()). A
    mismatch would mean the filter silently became variant-dependent -- e.g. a station_id dtype/
    formatting mismatch that dropped unmatched rows for only one variant -- and every paired
    comparison downstream would then silently compare different samples between variants. Abort
    rather than proceed on a broken filter.
    """
    per_fold_n = pooled_df.groupby("fold")["n"].nunique()
    bad = per_fold_n[per_fold_n != 1]
    if not bad.empty:
        detail = pooled_df[pooled_df["fold"].isin(bad.index)][["model", "variant", "fold", "n"]]
        sys.exit(f"[FATAL] Nach Imputations-Filterung ist die Zeilenzahl NICHT identisch ueber "
                 f"alle Varianten fuer Fold(s) {list(bad.index)}:\n{detail.to_string(index=False)}\n"
                 f"Der Filter ist variantenabhaengig geworden -- Abbruch statt verzerrter "
                 f"gepaarter Vergleiche.")
    per_fold = {int(f): int(n.iloc[0]) for f, n in pooled_df.groupby("fold")["n"]}
    n_variants = pooled_df.groupby(["model", "variant"]).ngroups
    print(f"[OK] Nach Imputations-Filterung: identische Zeilenzahl je Fold ueber alle "
          f"{n_variants} Varianten ({per_fold}).")


# ─────────────────────────────────────────────────────────────────────────────
# 1) Loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_station_csvs(results_dir: Path) -> pd.DataFrame:
    """Load all 30 stdhp_*_fold{n}.csv station-level result files into one long frame."""
    frames = []
    for path in sorted(results_dir.glob("stdhp_*_fold*.csv")):
        m = re.match(r"(stdhp_.+)_fold(\d+)$", path.stem)
        if not m:
            continue
        prefix, fold = m.group(1), int(m.group(2))
        if prefix not in STDHP_META:
            warnings.warn(f"Unbekanntes stdhp-Prefix, wird uebersprungen: {prefix}")
            continue
        model, variant, letter = STDHP_META[prefix]
        df = pd.read_csv(path)
        df["station_id"] = df["station_id"].map(norm_station)
        df["model"] = model
        df["variant"] = variant
        df["ablation_letter"] = letter
        df["fold"] = int(fold)
        df["run"] = "stdhp"
        frames.append(df)
    if not frames:
        sys.exit(f"[FATAL] Keine stdhp_*.csv in {results_dir} gefunden.")
    out = pd.concat(frames, ignore_index=True)

    expected = len(STDHP_META) * len(FOLDS)
    found = out.groupby(["model", "variant", "fold"]).ngroups
    if found != expected:
        warnings.warn(f"Erwartet {expected} (model,variant,fold)-Kombinationen, gefunden {found}.")
    counts = out.groupby(["model", "variant", "fold"]).size()
    bad = counts[counts != N_STATIONS_EXPECTED]
    if not bad.empty:
        warnings.warn(f"Nicht {N_STATIONS_EXPECTED} Stationen in:\n{bad}")
    return out


def load_tft_station_csvs(results_dir: Path) -> pd.DataFrame:
    """
    Load the 6 TFT stdhp station-level result CSVs (train_tft_bc_m-tft_c-wind_tft_sp_{base,hist}_
    stdhp_fold{1,2,3}.csv) into the SAME column shape load_station_csvs() produces for the graph
    CSVs (model, variant, ablation_letter, fold, run, station_id, n_samples, rmse, mae, r2,
    skill_nwp, skill), so downstream code (_crosscheck_csv_vs_raw(), verify_fold_assignment())
    can treat them uniformly. Two differences from the graph loader, both handled here:
      - filename scheme + 1-based fold numbering -> canonical_fold = tft_fold - TFT_FOLD_OFFSET
        (see TFT_META/TFT_FOLD_OFFSET).
      - station_id is "synth_00161.csv" -> norm_station() strips both the prefix and suffix.
    The 'skill' column is 100% NaN here (evaluate_reference.py / get_test_results_tft_bc.py never
    had a persistence reference to compute it from -- pers_ref is NaN in the TFT raw parquets, see
    the module docstring) -- left as-is; scan_tft_raw_parquets() + build_station_metrics()
    recompute a real 'skill' downstream from the pers_ref borrowed from the graph parquets.
    """
    frames = []
    for path in sorted(results_dir.glob("train_tft_bc_m-tft_c-wind_tft_sp_*_stdhp_fold*.csv")):
        m = re.match(r"(train_tft_bc_m-tft_c-wind_tft_sp_(?:base|hist))_stdhp_fold(\d+)$", path.stem)
        if not m:
            continue
        prefix, tft_fold = m.group(1), int(m.group(2))
        if prefix not in TFT_META:
            warnings.warn(f"Unbekanntes TFT-Prefix, wird uebersprungen: {prefix}")
            continue
        model, variant, letter = TFT_META[prefix]
        df = pd.read_csv(path)
        df["station_id"] = df["station_id"].map(norm_station)
        df["model"] = model
        df["variant"] = variant
        df["ablation_letter"] = letter
        df["fold"] = tft_fold - TFT_FOLD_OFFSET
        df["run"] = "stdhp_tft"
        frames.append(df)
    if not frames:
        sys.exit(f"[FATAL] Keine TFT-stdhp-CSVs in {results_dir} gefunden.")
    out = pd.concat(frames, ignore_index=True)

    expected = len(TFT_META) * len(FOLDS)
    found = out.groupby(["model", "variant", "fold"]).ngroups
    if found != expected:
        warnings.warn(f"Erwartet {expected} TFT (model,variant,fold)-Kombinationen, gefunden {found}.")
    counts = out.groupby(["model", "variant", "fold"]).size()
    bad = counts[counts != N_STATIONS_EXPECTED]
    if not bad.empty:
        warnings.warn(f"Nicht {N_STATIONS_EXPECTED} Stationen in TFT-CSVs:\n{bad}")
    return out


def verify_fold_assignment(tft_csv_df: pd.DataFrame, graph_csv_df: pd.DataFrame,
                            spatial_folds_path: Path) -> None:
    """
    Independent re-verification (task explicitly wants this NOT just trusted from prior
    investigation) that:
      - graph fold f (0-based) targets exactly configs/spatial_folds.yaml's spatial_fold{f+1}'s
        val_files (51 stations),
      - TFT config-fold n (1-based) targets the SAME 51 stations as spatial_fold{n}.val_files,
        i.e. TFT fold n <-> graph fold (n - TFT_FOLD_OFFSET).
    Station sets are taken directly from the loaded station-level CSVs (one row per target
    station, so the CSV's station_id set per (variant, fold) already IS the target-station set),
    compared by exact set equality, not just size -- a same-size, wrong-membership mismatch would
    otherwise slip through silently. Aborts loudly (SystemExit) on any mismatch, for EVERY (model,
    variant, fold) combination present, graph and TFT alike -- not just the canonical file per
    fold, since a fold-assignment bug could in principle affect only one variant's file.
    """
    import yaml
    spec = yaml.safe_load(Path(spatial_folds_path).read_text())
    spatial_val = {i: set(norm_station(x) for x in spec[f"spatial_fold{i}"]["val_files"]) for i in (1, 2, 3)}
    for name, exp in spatial_val.items():
        if len(exp) != N_STATIONS_EXPECTED:
            sys.exit(f"[FATAL] spatial_fold{name}.val_files hat {len(exp)} Stationen, erwartet "
                     f"{N_STATIONS_EXPECTED}.")

    n_checked = 0
    for (model, variant, fold), g in graph_csv_df.groupby(["model", "variant", "fold"]):
        got = set(g["station_id"])
        expected = spatial_val[fold + 1]
        if got != expected:
            sys.exit(f"[FATAL] Graph {model} {variant} fold {fold}: Stationsmenge stimmt NICHT mit "
                     f"spatial_fold{fold + 1}.val_files ueberein ({len(got ^ expected)} Differenzen, "
                     f"z.B. {sorted(got ^ expected)[:5]}). Fold-Zuordnung falsch -- Abbruch.")
        n_checked += 1
    for (model, variant, fold), g in tft_csv_df.groupby(["model", "variant", "fold"]):
        tft_fold = fold + TFT_FOLD_OFFSET
        got = set(g["station_id"])
        expected = spatial_val[tft_fold]
        if got != expected:
            sys.exit(f"[FATAL] TFT {model} {variant} fold {tft_fold} (-> kanonisch {fold}): "
                     f"Stationsmenge stimmt NICHT mit spatial_fold{tft_fold}.val_files ueberein "
                     f"({len(got ^ expected)} Differenzen, z.B. {sorted(got ^ expected)[:5]}). "
                     f"Fold-Zuordnung falsch -- Abbruch.")
        n_checked += 1
    print(f"[OK] Fold-Zuordnung unabhaengig gegen configs/spatial_folds.yaml verifiziert: "
          f"{n_checked} (model, variant, fold)-Stationsmengen (Graph + TFT) stimmen exakt mit den "
          f"jeweiligen spatial_fold{{1,2,3}}.val_files ueberein (TFT fold{{1,2,3}} <-> "
          f"spatial_fold{{1,2,3}} == Graph fold{{0,1,2}}).")


def _assert_finite(df: pd.DataFrame, cols: list[str], what: str) -> None:
    """
    Abort if any of `cols` contains NaN/inf.

    Rationale (paired comparisons): every downstream number here is either a paired per-station
    difference between variants or a ratio against a reference derived from a *different* file.
    Silently dropping non-finite rows would give each variant a slightly different sample set,
    which biases exactly the A-B / B-C / A-C / A-BASE contrasts this script exists to produce.
    The stdhp parquets are complete (51 stations x 1460 runs x 48 h = 3,574,080 rows, 0 NaN in
    pred/gt/nwp_ref/pers_ref, verified), so hitting this is a signal that something upstream
    changed — fail loudly rather than quietly re-defining the sample.
    """
    bad = {c: int((~np.isfinite(df[c].to_numpy(dtype=float))).sum()) for c in cols}
    bad = {c: n for c, n in bad.items() if n}
    if bad:
        sys.exit(f"[FATAL] {what}: nicht-endliche Werte in {bad}. Ein stiller Drop wuerde die "
                 f"Stichprobe zwischen Varianten unterschiedlich machen und alle gepaarten "
                 f"Vergleiche verzerren -- Abbruch. Erst die Ursache upstream klaeren.")


def _canonical_prefixes_for_fold(fold: int, raw_dir: Path) -> list[str]:
    return sorted(p for p in STDHP_META if (raw_dir / f"{p}_fold{fold}_raw.parquet").exists())


def verify_and_derive_references(raw_dir: Path, folds=FOLDS, atol: float = 1e-4,
                                  n_check_files: int | None = None) -> None:
    """
    For each fold: verify that gt / nwp_ref / pers_ref are numerically identical (within `atol`)
    across the stdhp raw parquets of that fold — they are supposed to be, since they depend
    only on the (station, valid_time, horizon) grid, not on the model that was trained.

    `n_check_files=None` (default) checks EVERY other variant of the fold. Do not lower this to a
    small number: the prefix list is sorted alphabetically, so a truncated check only ever covers
    the DCRNN family (stdhp_dcrnn_*) and never reaches stdhp_mtgnn_* / stdhp_wavenet_*, which run
    through a different get_test_results_*.py and are exactly where a divergent nwp_ref/pers_ref
    would first show up. A truncated check would therefore pass while proving nothing about the
    variants it skipped.

    Aborts with SystemExit (clear message) if the identity assumption is violated — do NOT
    silently average over disagreeing baselines.

    NOTE: this function used to ALSO derive the per-station ICON-D2/Persistence metrics from the
    canonical file. That now happens in scan_raw_parquets() + build_station_metrics() instead,
    because the metrics must be computed on the imputed-hour-FILTERED sample (see the module
    docstring's IMPUTED HOURS section), and this function's canonical-file read has no filtering
    applied -- it only cares about cross-variant identity, which holds on the raw grid regardless
    of any downstream filter.
    """
    for fold in folds:
        prefixes = _canonical_prefixes_for_fold(fold, raw_dir)
        if not prefixes:
            sys.exit(f"[FATAL] Keine Parquets fuer fold {fold} in {raw_dir}.")
        canon_path = raw_dir / f"{prefixes[0]}_fold{fold}_raw.parquet"
        canon = pd.read_parquet(canon_path, columns=["station_id", "valid_time", "horizon", "gt", "nwp_ref", "pers_ref"])
        canon["station_id"] = canon["station_id"].map(norm_station)
        _assert_finite(canon, ["gt", "nwp_ref", "pers_ref"], f"{prefixes[0]}_fold{fold}")

        check_prefixes = prefixes[1:] if n_check_files is None else prefixes[1:1 + n_check_files]
        for other_prefix in check_prefixes:
            other_path = raw_dir / f"{other_prefix}_fold{fold}_raw.parquet"
            other = pd.read_parquet(other_path, columns=["station_id", "valid_time", "horizon", "gt", "nwp_ref", "pers_ref"])
            other["station_id"] = other["station_id"].map(norm_station)
            merged = canon.merge(other, on=["station_id", "valid_time", "horizon"], suffixes=("_a", "_b"))
            if len(merged) != len(canon):
                sys.exit(f"[FATAL] fold {fold}: {other_prefix} hat ein anderes (station,valid_time,horizon)-Grid "
                          f"als {prefixes[0]} ({len(merged)} von {len(canon)} Zeilen matchen). Referenz-Ableitung "
                          f"aus nwp_ref/pers_ref setzt ein identisches Grid voraus -- Abbruch.")
            for col in ["gt", "nwp_ref", "pers_ref"]:
                d = (merged[f"{col}_a"] - merged[f"{col}_b"]).abs()
                if float(d.max()) > atol:
                    sys.exit(f"[FATAL] fold {fold}: Spalte '{col}' unterscheidet sich zwischen {prefixes[0]} und "
                             f"{other_prefix} um bis zu {float(d.max()):.6g} (> atol={atol}). Die Annahme "
                             f"'nwp_ref/pers_ref sind ueber Varianten eines Folds identisch' ist verletzt -- "
                             f"Skript bricht ab statt falsche Referenzwerte zu erzeugen.")
            del other, merged
        print(f"[OK] fold {fold}: gt/nwp_ref/pers_ref identisch (atol={atol}) ueber "
              f"{1 + len(check_prefixes)} geprueften Varianten (kanonisch: {prefixes[0]}).")
        del canon
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# 1c) 14-variant row intersection (TFT's sample differs from the graph models' -- see the module
#     docstring's TFT section -- so a like-for-like comparison needs the common
#     (station_id, run_time, horizon) rows, not each variant's own sample).
# ─────────────────────────────────────────────────────────────────────────────
def build_graph_canonical_keys(raw_dir: Path, mask: dict, folds=FOLDS, keep_imputed: bool = False) -> dict:
    """
    Per fold: read ONE representative graph stdhp parquet (any of the 10 -- their
    (station_id, run_time, horizon) grid and gt/nwp_ref/pers_ref values are already verified
    identical across all 10 in verify_and_derive_references()), apply the imputed-hour filter, and
    return the filtered (station_id, run_time, horizon, valid_time, gt, nwp_ref, pers_ref) frame.
    Because all 12 graph variants share one identical post-filter grid (see
    _verify_uniform_filtering()), this single frame doubles as (a) the graph side of the 14-variant
    row intersection (build_intersection_keys()) and (b) the source TFT's pers_ref is borrowed
    from and its nwp_ref is cross-checked against (scan_tft_raw_parquets()).
    """
    out = {}
    cols = ["station_id", "run_time", "valid_time", "horizon", "gt", "nwp_ref", "pers_ref"]
    for fold in folds:
        prefixes = _canonical_prefixes_for_fold(fold, raw_dir)
        if not prefixes:
            sys.exit(f"[FATAL] Keine Parquets fuer fold {fold} in {raw_dir}.")
        path = raw_dir / f"{prefixes[0]}_fold{fold}_raw.parquet"
        df = pd.read_parquet(path, columns=cols)
        df["station_id"] = df["station_id"].map(norm_station)
        if not keep_imputed:
            is_imp = _lookup_imputed(df, mask)
            df = df.loc[~is_imp].copy()
        out[fold] = df.reset_index(drop=True)
    return out


def build_tft_filtered_keys(raw_dir: Path, mask: dict, folds=FOLDS, keep_imputed: bool = False) -> dict:
    """Lightweight pre-pass (station_id/run_time/valid_time/horizon/gt only, no pred/nwp_ref/
    pers_ref) over the 6 TFT raw parquets, imputed-hour filtered, to get each (fold, variant)'s key
    set BEFORE the heavier scan_tft_raw_parquets() pass needs it (build_intersection_keys() has to
    run first so scan_raw_parquets()/scan_tft_raw_parquets() can compute the intersection-restricted
    buckets in their own single main pass instead of a second read)."""
    out = {}
    for prefix, (model, variant, letter) in sorted(TFT_META.items()):
        for fold in folds:
            tft_fold = fold + TFT_FOLD_OFFSET
            path = raw_dir / f"{prefix}_stdhp_fold{tft_fold}_raw.parquet"
            if not path.exists():
                sys.exit(f"[FATAL] Fehlt: {path}")
            df = pd.read_parquet(path, columns=["station_id", "run_time", "valid_time", "horizon", "gt"])
            df["station_id"] = df["station_id"].map(norm_station)
            if not keep_imputed:
                is_imp = _lookup_imputed(df, mask)
                df = df.loc[~is_imp]
            out[(fold, variant)] = df[["station_id", "run_time", "horizon"]].reset_index(drop=True)
    return out


def _composite_key(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorized (station_id, run_time, horizon) -> int64, for fast set membership without an
    explicit pandas merge in the scan_raw_parquets()/scan_tft_raw_parquets() hot loop.
    station_id is a 5-digit zero-padded string (<= 99999, via norm_station()), run_time is
    truncated to whole hours since epoch (data is hourly, freq: 1h, so this is lossless) which
    keeps it well under 10 digits for any date in this dataset, horizon <= 48 (2 digits) --
    station_id*1e10 + run_time_hours*100 + horizon is collision-free and fits comfortably in int64.
    """
    sid = df["station_id"].to_numpy(dtype="U5").astype(np.int64)
    # .dt.tz_convert("UTC") first, then view as int64 nanoseconds-since-epoch and floor-divide to
    # whole hours -- avoids numpy's tz-aware-Timestamp-object -> datetime64 auto-parsing path
    # (works today but is deprecated: converting a tz-aware object array via .astype() implicitly
    # drops/parses the tz, which numpy warns will become an error in a future version).
    rt_ns = df["run_time"].dt.tz_convert("UTC").to_numpy(dtype="datetime64[ns]").astype(np.int64)
    rt_hours = rt_ns // np.int64(3_600_000_000_000)
    hz = df["horizon"].to_numpy().astype(np.int64)
    return sid * np.int64(10_000_000_000) + rt_hours * np.int64(100) + hz


def build_intersection_keys(graph_canon: dict, tft_keys_by_fold_variant: dict, folds=FOLDS) -> dict:
    """
    Per fold: the (station_id, run_time, horizon) triplets present in ALL 14 variants (12 graph +
    TFT base + TFT hist) after imputed-hour filtering -- task B's "Schnittmenge". The 12 graph
    variants already share one identical filtered grid (build_graph_canonical_keys()), so this
    reduces to graph_keys ∩ tft_base_keys ∩ tft_hist_keys per fold. Returns {fold: set(int64)}
    (a plain Python set, used via np.isin(..., np.array(sorted(...))) at the call sites for
    vectorized membership tests over millions of rows).
    """
    out = {}
    for fold in folds:
        keys = set(_composite_key(graph_canon[fold]).tolist())
        for variant in ("base", "hist"):
            tft_keys = set(_composite_key(tft_keys_by_fold_variant[(fold, variant)]).tolist())
            keys &= tft_keys
        out[fold] = keys
    return out


def scan_raw_parquets(raw_dir: Path, mask: dict, folds=FOLDS, keep_imputed: bool = False,
                       intersect_keys: dict | None = None):
    """
    Single pass over all 30 stdhp raw parquets (~3.57M rows / 8 columns each). Per file: read
    only RAW_COLUMNS, immediately reduce to a handful of aggregate rows (pooled residuals,
    per-horizon, per-month, per-wind-speed-class, per-station), then discard the raw frame
    before moving to the next file. At no point are two full parquet files held in memory
    simultaneously. Reference (ICON-D2 / Persistence) aggregates only need to be computed once
    per fold (from the first file processed for that fold) since they don't depend on the model.

    IMPUTED-HOUR FILTERING (default on, `keep_imputed=True` restores the old behaviour): rows
    whose (station_id, valid_time) has no real raw measurement (see build_imputation_mask()) are
    dropped BEFORE month/ws_bin assignment and before every sum below, for BOTH the model's
    pred/gt AND the ICON-D2/Persistence references derived from nwp_ref/pers_ref -- the decision
    is that imputed hours are excluded from evaluation entirely, not just from one side of the
    comparison. "station_full" / "ref_station_full" are the two exceptions: always computed on
    the UNFILTERED sample regardless of `keep_imputed`, and read ONLY by
    _crosscheck_csv_vs_raw() (must match the stdhp_*.csv files, which were written on the full
    sample) and by the overview table's filtered-vs-unfiltered comparison columns. Everything
    else in the returned bundle is the (by default filtered) sample that all reported numbers use.

    Station-level buckets also carry sum_gt/sum_gt2 (not just sse/sae/n) so that per-station R2
    can be recomputed from these sums later (ss_tot = sum_gt2 - sum_gt**2/n) without keeping any
    raw arrays around -- this is what lets the CSVs be retired as the metrics source once
    imputed hours must be excluded (the CSVs cannot be corrected after the fact; see
    build_station_metrics()).

    Returns a dict of small aggregate DataFrames (sums, not final metrics — final RMSE/MAE is
    computed at plot time via sqrt(sse/n) etc., so that fold-averaging happens the same way
    everywhere: aggregate-then-average-across-folds, never a single pooled-across-folds number).
    """
    pooled_rows = []
    bucket_rows = {"horizon": [], "month": [], "ws_bin": []}
    station_rows = []
    station_full_rows = []
    station_intersect_rows = []
    ref_bucket_rows = {"horizon": [], "month": [], "ws_bin": []}
    ref_station_rows = []
    ref_station_full_rows = []
    ref_station_intersect_rows = []
    unbinned_rows = []
    seen_ref_fold = set()
    n_imputed_total = 0
    n_rows_total = 0

    # Sorted numpy arrays (not the raw sets) for fast np.isin() membership tests in the per-file
    # loop below -- see build_intersection_keys()/_composite_key().
    intersect_arr = ({f: np.array(sorted(intersect_keys[f]), dtype=np.int64) for f in folds}
                     if intersect_keys is not None else None)

    files = [(prefix, meta, fold)
             for prefix, meta in sorted(STDHP_META.items())
             for fold in folds]

    def _station_agg(station_id, sq, ab, gt):
        return pd.DataFrame({"station_id": station_id, "sq": sq, "ab": ab,
                             "gt": gt, "gt2": gt ** 2}).groupby("station_id").agg(
            sse=("sq", "sum"), sae=("ab", "sum"), n=("sq", "size"),
            sum_gt=("gt", "sum"), sum_gt2=("gt2", "sum")).reset_index()

    for i, (prefix, (model, variant, letter), fold) in enumerate(files):
        path = raw_dir / f"{prefix}_fold{fold}_raw.parquet"
        if not path.exists():
            warnings.warn(f"Fehlt: {path}")
            continue
        df_all = pd.read_parquet(path, columns=RAW_COLUMNS)
        df_all["station_id"] = df_all["station_id"].map(norm_station)
        _assert_finite(df_all, ["pred", "gt", "nwp_ref", "pers_ref"], f"{prefix}_fold{fold}")

        is_imputed = _lookup_imputed(df_all, mask)
        n_imputed_total += int(is_imputed.sum())
        n_rows_total += len(df_all)
        sample_mask = np.ones(len(df_all), dtype=bool) if keep_imputed else ~is_imputed

        # ---- FULL (unfiltered) per-station model bucket -- ONLY for _crosscheck_csv_vs_raw(). ----
        full_err = (df_all["pred"] - df_all["gt"]).to_numpy()
        full_gt = df_all["gt"].to_numpy()
        gst_full = _station_agg(df_all["station_id"].values, full_err ** 2, np.abs(full_err), full_gt)
        gst_full["model"], gst_full["variant"], gst_full["fold"] = model, variant, fold
        station_full_rows.append(gst_full)
        del full_err, full_gt

        # ---- Reference (ICON-D2/Persistence) buckets, FULL and FILTERED, once per fold. ----
        if fold not in seen_ref_fold:
            seen_ref_fold.add(fold)
            month_all = pd.to_datetime(df_all["valid_time"]).dt.month.to_numpy()
            wsbin_all = pd.cut(df_all["gt"], bins=WS_BINS + [np.inf], labels=WS_LABELS, right=False).to_numpy()
            horizon_all = df_all["horizon"].to_numpy()
            for ref_name, ref_col in [("ICON-D2", "nwp_ref"), ("Persistence", "pers_ref")]:
                rerr_all = (df_all[ref_col] - df_all["gt"]).to_numpy()
                rgt_all = df_all["gt"].to_numpy()

                gst2_full = _station_agg(df_all["station_id"].values, rerr_all ** 2, np.abs(rerr_all), rgt_all)
                gst2_full["model"], gst2_full["fold"] = ref_name, fold
                ref_station_full_rows.append(gst2_full)

                fm = sample_mask
                gst2 = _station_agg(df_all["station_id"].values[fm], rerr_all[fm] ** 2, np.abs(rerr_all[fm]), rgt_all[fm])
                gst2["model"], gst2["fold"] = ref_name, fold
                ref_station_rows.append(gst2)

                if intersect_arr is not None:
                    in_inter_all = np.isin(_composite_key(df_all), intersect_arr[fold])
                    gst2i = _station_agg(df_all["station_id"].values[in_inter_all], rerr_all[in_inter_all] ** 2,
                                         np.abs(rerr_all[in_inter_all]), rgt_all[in_inter_all])
                    gst2i["model"], gst2i["fold"] = ref_name, fold
                    ref_station_intersect_rows.append(gst2i)

                for key, vals_all in [("horizon", horizon_all), ("month", month_all), ("ws_bin", wsbin_all)]:
                    g = pd.DataFrame({key: vals_all[fm], "sq": rerr_all[fm] ** 2, "ab": np.abs(rerr_all[fm])}).groupby(
                        key, observed=True).agg(sse=("sq", "sum"), sae=("ab", "sum"), n=("sq", "size")).reset_index()
                    g["model"], g["fold"] = ref_name, fold
                    ref_bucket_rows[key].append(g)
            del month_all, wsbin_all, horizon_all

        # ---- FILTERED model bucket: the primary/default sample for everything else. ----
        df = df_all.loc[sample_mask].copy()
        del df_all

        df["month"] = pd.to_datetime(df["valid_time"]).dt.month
        # gt < 0 (non-physical, all imputed by construction -- see verify_imputation_mask()) falls
        # outside every wind-speed bin; with the default filter it should be exactly zero rows.
        # Kept as a live check rather than removed: under --keep-imputed this still needs to fire.
        df["ws_bin"] = pd.cut(df["gt"], bins=WS_BINS + [np.inf], labels=WS_LABELS, right=False)
        n_unbinned = int(df["ws_bin"].isna().sum())
        if n_unbinned:
            unbinned_rows.append({"model": model, "variant": variant, "fold": fold,
                                  "n_unbinned": n_unbinned, "gt_min": float(df["gt"].min())})

        err = (df["pred"] - df["gt"]).to_numpy()
        gt = df["gt"].to_numpy()
        sq, ab = err ** 2, np.abs(err)

        pooled_rows.append({"model": model, "variant": variant, "fold": fold,
                             "sse": float(sq.sum()), "sae": float(ab.sum()), "n": int(len(df))})

        for key in ("horizon", "month", "ws_bin"):
            g = pd.DataFrame({key: df[key].values, "sq": sq, "ab": ab}).groupby(key, observed=True).agg(
                sse=("sq", "sum"), sae=("ab", "sum"), n=("sq", "size")).reset_index()
            g["model"], g["variant"], g["fold"] = model, variant, fold
            bucket_rows[key].append(g)

        gst = _station_agg(df["station_id"].values, sq, ab, gt)
        gst["model"], gst["variant"], gst["fold"] = model, variant, fold
        station_rows.append(gst)

        # ---- Stage 3 (task B): restricted to the 14-variant (station, run_time, horizon)
        # intersection of this fold -- the sample the paper's reported numbers use by default
        # (--intersect-only). ----
        if intersect_arr is not None:
            in_inter = np.isin(_composite_key(df), intersect_arr[fold])
            gsti = _station_agg(df["station_id"].values[in_inter], sq[in_inter], ab[in_inter], gt[in_inter])
            gsti["model"], gsti["variant"], gsti["fold"] = model, variant, fold
            station_intersect_rows.append(gsti)

        del df, err, sq, ab, gt
        gc.collect()
        print(f"  [scan {i + 1}/{len(files)}] {prefix} fold{fold} done "
              f"({int(is_imputed.sum())} imputiert von {len(is_imputed)})", flush=True)

    def _cat(rows):
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    if unbinned_rows:
        ub = pd.DataFrame(unbinned_rows)
        print(f"[WARN] {int(ub['n_unbinned'].sum())} Zeilen mit gt < 0 (min {ub['gt_min'].min():.2f} m/s) "
              f"fallen in KEINE Windklasse und fehlen daher nur in 05_error_by_windspeed_class -- "
              f"die Uebersichtstabelle und alle anderen Figuren enthalten sie. Nicht-physikalische "
              f"Messwerte upstream, siehe stations_unbinned_gt.csv.")

    mode = "keep-imputed (ungefiltert)" if keep_imputed else "gefiltert (Standard)"
    print(f"[INFO] Imputations-Filter [{mode}]: {n_imputed_total:,} von {n_rows_total:,} "
          f"Zeilen ueber alle 30 Dateien als imputiert erkannt "
          f"({100 * n_imputed_total / n_rows_total:.2f}%).")

    return {
        "unbinned": pd.DataFrame(unbinned_rows),
        "pooled": pd.DataFrame(pooled_rows),
        "horizon": _cat(bucket_rows["horizon"]),
        "month": _cat(bucket_rows["month"]),
        "ws_bin": _cat(bucket_rows["ws_bin"]),
        "station": _cat(station_rows),
        "station_full": _cat(station_full_rows),
        "station_intersect": _cat(station_intersect_rows),
        "ref_horizon": _cat(ref_bucket_rows["horizon"]),
        "ref_month": _cat(ref_bucket_rows["month"]),
        "ref_ws_bin": _cat(ref_bucket_rows["ws_bin"]),
        "ref_station": _cat(ref_station_rows),
        "ref_station_full": _cat(ref_station_full_rows),
        "ref_station_intersect": _cat(ref_station_intersect_rows),
    }


def scan_tft_raw_parquets(raw_dir: Path, mask: dict, graph_canon: dict, intersect_keys: dict,
                          folds=FOLDS, keep_imputed: bool = False) -> dict:
    """
    Mirrors scan_raw_parquets() for the 2 TFT variants (base/hist) x 3 folds -- kept as a SEPARATE
    function rather than folded into scan_raw_parquets() because TFT's files break several
    assumptions that function relies on: different filename scheme and 1-based fold numbering
    (TFT_META/TFT_FOLD_OFFSET), different station-id format ("synth_00161.csv", see
    norm_station()), and -- unlike the 12 graph variants -- TFT's own sample is NOT the shared
    full (station, run_time, horizon) grid (see the module docstring's TFT section: TFT reads raw
    station measurements directly, `raw_station_source: true`, no interpol_path/knnimputer_path,
    so it drops any run/station whose lookback or target touches a raw measurement gap instead of
    imputing across it -- this is the actual cause of TFT's smaller, per-station-varying sample).

    Two data repairs happen here that the graph pipeline doesn't need:
      - pers_ref is 100% NaN in the TFT parquets (get_test_results_tft_bc.py never had a
        persistence reference to write) -- borrowed from `graph_canon[fold]` (see
        build_graph_canonical_keys()) via an explicit (station_id, run_time, horizon) join, valid
        because pers_ref depends only on that grid, not on which model produced `pred`.
      - nwp_ref IS native in the TFT parquets; cross-checked here against graph_canon's nwp_ref
        (and gt, as an extra consistency check) on the same join. A mismatch is recorded in the
        returned "nwp_ref_check" frame and warned about, NOT silently resolved by picking a
        source -- see the module docstring and the known station-05142 grid-point precedent.

    Returns a bundle shaped like a subset of scan_raw_parquets()'s: pooled, horizon, month,
    ws_bin, station (own-sample filtered), station_full (own-sample unfiltered), station_intersect
    (Stage 3, 14-variant intersection), nwp_ref_check.
    """
    pooled_rows = []
    bucket_rows = {"horizon": [], "month": [], "ws_bin": []}
    station_rows, station_full_rows, station_intersect_rows = [], [], []
    nwp_ref_check = []
    intersect_arr = {f: np.array(sorted(intersect_keys[f]), dtype=np.int64) for f in folds}

    def _station_agg(station_id, sq, ab, gt):
        return pd.DataFrame({"station_id": station_id, "sq": sq, "ab": ab, "gt": gt, "gt2": gt ** 2}
                            ).groupby("station_id").agg(sse=("sq", "sum"), sae=("ab", "sum"),
                                                        n=("sq", "size"), sum_gt=("gt", "sum"),
                                                        sum_gt2=("gt2", "sum")).reset_index()

    files = [(prefix, meta, fold) for prefix, meta in sorted(TFT_META.items()) for fold in folds]
    for i, (prefix, (model, variant, letter), fold) in enumerate(files):
        tft_fold = fold + TFT_FOLD_OFFSET
        path = raw_dir / f"{prefix}_stdhp_fold{tft_fold}_raw.parquet"
        if not path.exists():
            sys.exit(f"[FATAL] Fehlt: {path}")
        df = pd.read_parquet(path, columns=RAW_COLUMNS)
        df["station_id"] = df["station_id"].map(norm_station)
        _assert_finite(df, ["pred", "gt", "nwp_ref"], f"{prefix}_stdhp_fold{tft_fold}")
        if df["pers_ref"].notna().any():
            sys.exit(f"[FATAL] {prefix}_stdhp_fold{tft_fold}: pers_ref ist NICHT mehr durchgehend "
                     f"NaN wie bei der letzten Pruefung -- die Borrow-Logik unten geht davon aus, "
                     f"dass hier nichts Echtes ueberschrieben wird. Abbruch statt still zu ueberschreiben.")
        df = df.drop(columns=["pers_ref"])

        # ---- Borrow pers_ref from the graph canonical grid + cross-check nwp_ref/gt ----
        canon = graph_canon[fold][["station_id", "run_time", "horizon", "gt", "nwp_ref", "pers_ref"]]
        merged = df.merge(canon, on=["station_id", "run_time", "horizon"], how="left", suffixes=("", "_graph"))
        n_no_match = int(merged["pers_ref"].isna().sum())
        if n_no_match:
            warnings.warn(f"{prefix} fold{tft_fold}: {n_no_match} von {len(merged)} Zeilen ohne Gegenstueck "
                          f"im gefilterten Graphen-Grid dieses Folds -- pers_ref bleibt dort NaN.")
        n_checked = int(merged["nwp_ref_graph"].notna().sum())
        d_nwp = (merged["nwp_ref"] - merged["nwp_ref_graph"]).abs()
        d_gt = (merged["gt"] - merged["gt_graph"]).abs()
        nwp_ref_check.append({"prefix": prefix, "fold": fold, "tft_fold": tft_fold, "n_checked": n_checked,
                              "n_mismatch_nwp_ref": int((d_nwp > 1e-4).sum()),
                              "max_abs_diff_nwp_ref": float(d_nwp.max()) if n_checked else np.nan,
                              "n_mismatch_gt": int((d_gt > 1e-4).sum()),
                              "max_abs_diff_gt": float(d_gt.max()) if n_checked else np.nan})
        df["pers_ref"] = merged["pers_ref"].to_numpy()
        del merged, canon

        is_imputed = _lookup_imputed(df, mask)
        sample_mask = np.ones(len(df), dtype=bool) if keep_imputed else ~is_imputed

        # ---- FULL (unfiltered) per-station bucket -- for _crosscheck_csv_vs_raw(). ----
        full_err = (df["pred"] - df["gt"]).to_numpy()
        gst_full = _station_agg(df["station_id"].values, full_err ** 2, np.abs(full_err), df["gt"].to_numpy())
        gst_full["model"], gst_full["variant"], gst_full["fold"] = model, variant, fold
        station_full_rows.append(gst_full)
        del full_err

        dff = df.loc[sample_mask].copy()
        dff["month"] = pd.to_datetime(dff["valid_time"]).dt.month
        dff["ws_bin"] = pd.cut(dff["gt"], bins=WS_BINS + [np.inf], labels=WS_LABELS, right=False)
        err = (dff["pred"] - dff["gt"]).to_numpy()
        gt = dff["gt"].to_numpy()
        sq, ab = err ** 2, np.abs(err)

        pooled_rows.append({"model": model, "variant": variant, "fold": fold,
                            "sse": float(sq.sum()), "sae": float(ab.sum()), "n": int(len(dff))})
        for key in ("horizon", "month", "ws_bin"):
            g = pd.DataFrame({key: dff[key].values, "sq": sq, "ab": ab}).groupby(key, observed=True).agg(
                sse=("sq", "sum"), sae=("ab", "sum"), n=("sq", "size")).reset_index()
            g["model"], g["variant"], g["fold"] = model, variant, fold
            bucket_rows[key].append(g)

        gst = _station_agg(dff["station_id"].values, sq, ab, gt)
        gst["model"], gst["variant"], gst["fold"] = model, variant, fold
        station_rows.append(gst)

        in_inter = np.isin(_composite_key(dff), intersect_arr[fold])
        gsti = _station_agg(dff["station_id"].values[in_inter], sq[in_inter], ab[in_inter], gt[in_inter])
        gsti["model"], gsti["variant"], gsti["fold"] = model, variant, fold
        station_intersect_rows.append(gsti)

        del df, dff, err, sq, ab, gt
        gc.collect()
        print(f"  [scan-tft {i + 1}/{len(files)}] {prefix} fold{tft_fold} (-> kanonisch {fold}) done "
              f"({int(is_imputed.sum())} imputiert von {len(is_imputed)})", flush=True)

    def _cat(rows):
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    nwp_check_df = pd.DataFrame(nwp_ref_check)
    bad = nwp_check_df[(nwp_check_df["n_mismatch_nwp_ref"] > 0) | (nwp_check_df["n_mismatch_gt"] > 0)]
    if not bad.empty:
        warnings.warn(f"nwp_ref und/oder gt weichen zwischen TFT- und Graphen-Parquet auf gemeinsamen Zeilen "
                      f"ab -- siehe tft_vs_graph_nwp_ref_check.csv:\n{bad.to_string(index=False)}")
    else:
        print(f"[OK] nwp_ref und gt stimmen zwischen TFT- und Graphen-Parquets auf allen gemeinsamen "
              f"(station_id, run_time, horizon)-Zeilen ueberein (getestet: "
              f"{int(nwp_check_df['n_checked'].sum()):,} Zeilen ueber {len(files)} TFT-Dateien).")

    return {
        "pooled": pd.DataFrame(pooled_rows),
        "horizon": _cat(bucket_rows["horizon"]),
        "month": _cat(bucket_rows["month"]),
        "ws_bin": _cat(bucket_rows["ws_bin"]),
        "station": _cat(station_rows),
        "station_full": _cat(station_full_rows),
        "station_intersect": _cat(station_intersect_rows),
        "nwp_ref_check": nwp_check_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1b) Station-level metrics FROM THE RAW PARQUETS (replaces the stdhp_*.csv as the metrics
#     source once imputed hours are filtered out -- the CSVs were written by evaluate() on the
#     full, unfiltered sample and cannot be corrected after the fact; see main()/_COL_RENAME).
# ─────────────────────────────────────────────────────────────────────────────
def _station_rmse_mae_r2(bucket: pd.DataFrame) -> pd.DataFrame:
    out = bucket.copy()
    out["rmse"] = np.sqrt(out["sse"] / out["n"])
    out["mae"] = out["sae"] / out["n"]
    ss_tot = out["sum_gt2"] - (out["sum_gt"] ** 2) / out["n"]
    out["r2"] = np.where(ss_tot > 0, 1.0 - out["sse"] / ss_tot, np.nan)
    out["n_samples"] = out["n"].astype(int)
    return out


def build_station_metrics(model_bucket: pd.DataFrame, ref_bucket: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(model, variant, fold, station) rmse/mae/r2/skill/skill_nwp, computed directly from the
    sufficient statistics collected in scan_raw_parquets() (sse, sae, n, sum_gt, sum_gt2) --
    NOT from the stdhp_*.csv columns. Produces the same column shape the CSV-based loader used
    to (model, variant, fold, station_id, rmse, mae, r2, skill, skill_nwp, n_samples), so every
    downstream function (paired-diff analyses, fold-dispersion plot, skill distribution,
    stations-below-icon) is unchanged; only where the numbers come from changes.

    `ref_bucket` supplies ICON-D2/Persistence; both buckets must use the SAME filtering (both
    "station"+"ref_station" for the reported/filtered table, or both "station_full"+
    "ref_station_full" for the unfiltered comparison) -- mixing them would compare a model's
    filtered RMSE against an unfiltered reference RMSE, silently changing the reference window.
    """
    models = _station_rmse_mae_r2(model_bucket)
    refs = _station_rmse_mae_r2(ref_bucket)
    refs = refs.assign(variant="REF")

    wide = refs.pivot_table(index=["fold", "station_id"], columns="model", values="rmse").reset_index()
    wide = wide.rename(columns={"ICON-D2": "_rmse_icon", "Persistence": "_rmse_pers"})

    models = models.merge(wide, on=["fold", "station_id"], how="left")
    models["skill"] = 1.0 - models["rmse"] / models["_rmse_pers"]
    models["skill_nwp"] = 1.0 - models["rmse"] / models["_rmse_icon"]

    refs = refs.merge(wide, on=["fold", "station_id"], how="left")
    # skill := skill vs. Persistence (0 for Persistence itself); skill_nwp := skill vs. ICON-D2
    # (0 for ICON-D2 itself) -- same convention as the model rows.
    refs["skill"] = np.where(refs["model"] == "Persistence", 0.0, 1.0 - refs["_rmse_icon"] / refs["_rmse_pers"])
    refs["skill_nwp"] = np.where(refs["model"] == "ICON-D2", 0.0, 1.0 - refs["_rmse_pers"] / refs["_rmse_icon"])

    keep = ["model", "variant", "fold", "station_id", "rmse", "mae", "r2", "skill", "skill_nwp", "n_samples"]
    return pd.concat([models[keep], refs[keep]], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Overview table
# ─────────────────────────────────────────────────────────────────────────────
def build_overview_table(station_df: pd.DataFrame, pooled_df: pd.DataFrame | None,
                          station_df_unfiltered: pd.DataFrame | None = None,
                          station_df_ownfiltered: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    `station_df` is the PRIMARY sample the un-suffixed columns (rmse_mean, ...) report -- by
    default (--intersect-only) this is Stage 3, the 14-variant row intersection (task B); with
    --intersect-only disabled it is Stage 2, each variant's own imputed-hour-filtered sample (the
    graph-only script's original meaning). Two comparison stages are added alongside, task B's
    "drei Stufen nebeneinander", so the size of each filtering step stays visible in one table
    instead of requiring several script runs:
      - `station_df_unfiltered` -- ALWAYS Stage 1, the full sample as the runs deliver it,
        regardless of --keep-imputed/--intersect-only -- adds "*_mean_unfiltered"/"*_sd_unfiltered".
      - `station_df_ownfiltered` -- ALWAYS Stage 2, each variant's own imputed-hour-filtered
        sample (independent of the --intersect-only primary choice) -- adds
        "*_mean_ownfiltered"/"*_sd_ownfiltered". When --intersect-only is off this duplicates the
        primary columns; kept anyway so the column names' meaning does not depend on the flag.
    """
    rows = []
    for (model, variant), g in station_df.groupby(["model", "variant"]):
        fold_stats = g.groupby("fold")[["rmse", "mae", "r2", "skill", "skill_nwp"]].mean()
        row = {"model": model, "variant": variant, "transductive": is_transductive(model, variant)}
        for col in ["rmse", "mae", "r2", "skill", "skill_nwp"]:
            vals = fold_stats[col].reindex(FOLDS)
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_sd"] = float(vals.std(ddof=0))
            for f in FOLDS:
                row[f"{col}_fold{f}"] = float(vals.get(f, np.nan))
        rows.append(row)

    out = pd.DataFrame(rows)

    # ── The OTHER skill definition, reported side by side ────────────────────────────────
    # skill_mean / skill_nwp_mean above are  mean_over_stations(1 - rmse_s / rmse_ref_s).
    # The columns below are                  1 - mean_over_stations(rmse_s) / mean_over_stations(rmse_ref_s),
    # both averaged over folds afterwards. These are NOT the same number (Jensen: the ratio of
    # means ignores the covariance between a station's model error and its reference error) and
    # they do not even induce the same ranking here — WaveNet GRID and MTGNN GRID swap places.
    # Whichever goes into the paper must be named in the caption; they are exported together so
    # the discrepancy can never be discovered only after submission.
    ref_fold_rmse = (station_df[station_df["variant"] == "REF"]
                     .groupby(["model", "fold"])["rmse"].mean())
    if not ref_fold_rmse.empty:
        model_fold_rmse = station_df.groupby(["model", "variant", "fold"])["rmse"].mean()
        for ref_model, out_col in [("ICON-D2", "skill_nwp_ratio_mean"), ("Persistence", "skill_ratio_mean")]:
            if ref_model not in ref_fold_rmse.index.get_level_values("model"):
                continue
            ref_by_fold = ref_fold_rmse.xs(ref_model, level="model")

            def _ratio_skill(r, ref_by_fold=ref_by_fold):
                per_fold = [1.0 - model_fold_rmse.get((r["model"], r["variant"], f), np.nan) / ref_by_fold.get(f, np.nan)
                            for f in FOLDS]
                return float(np.nanmean(per_fold)) if np.any(np.isfinite(per_fold)) else np.nan

            out[out_col] = out.apply(_ratio_skill, axis=1)

    if pooled_df is not None and not pooled_df.empty:
        pooled_df = pooled_df.copy()
        pooled_df["pooled_rmse"] = np.sqrt(pooled_df["sse"] / pooled_df["n"])
        pooled_df["pooled_mae"] = pooled_df["sae"] / pooled_df["n"]
        for col in ["pooled_rmse", "pooled_mae"]:
            piv = pooled_df.pivot_table(index=["model", "variant"], columns="fold", values=col)
            out[f"{col}_mean"] = out.apply(lambda r: piv.loc[(r["model"], r["variant"])].reindex(FOLDS).mean()
                                            if (r["model"], r["variant"]) in piv.index else np.nan, axis=1)
            out[f"{col}_sd"] = out.apply(lambda r: piv.loc[(r["model"], r["variant"])].reindex(FOLDS).std(ddof=0)
                                          if (r["model"], r["variant"]) in piv.index else np.nan, axis=1)
            for f in FOLDS:
                out[f"{col}_fold{f}"] = out.apply(
                    lambda r, f=f: piv.loc[(r["model"], r["variant"])].get(f, np.nan)
                    if (r["model"], r["variant"]) in piv.index else np.nan, axis=1)

    if station_df_unfiltered is not None and not station_df_unfiltered.empty:
        rows_u = []
        for (model, variant), g in station_df_unfiltered.groupby(["model", "variant"]):
            fold_stats_u = g.groupby("fold")[["rmse", "mae", "r2", "skill", "skill_nwp"]].mean()
            row_u = {"model": model, "variant": variant}
            for col in ["rmse", "mae", "r2", "skill", "skill_nwp"]:
                vals = fold_stats_u[col].reindex(FOLDS)
                row_u[f"{col}_mean_unfiltered"] = float(vals.mean())
                row_u[f"{col}_sd_unfiltered"] = float(vals.std(ddof=0))
            rows_u.append(row_u)
        out = out.merge(pd.DataFrame(rows_u), on=["model", "variant"], how="left")

    if station_df_ownfiltered is not None and not station_df_ownfiltered.empty:
        rows_o = []
        for (model, variant), g in station_df_ownfiltered.groupby(["model", "variant"]):
            fold_stats_o = g.groupby("fold")[["rmse", "mae", "r2", "skill", "skill_nwp"]].mean()
            row_o = {"model": model, "variant": variant}
            for col in ["rmse", "mae", "r2", "skill", "skill_nwp"]:
                vals = fold_stats_o[col].reindex(FOLDS)
                row_o[f"{col}_mean_ownfiltered"] = float(vals.mean())
                row_o[f"{col}_sd_ownfiltered"] = float(vals.std(ddof=0))
            rows_o.append(row_o)
        out = out.merge(pd.DataFrame(rows_o), on=["model", "variant"], how="left")

    return out.sort_values("rmse_mean").reset_index(drop=True)


def overview_markdown(overview: pd.DataFrame, primary_label: str = "Stage") -> str:
    def fmt(mean, sd):
        return f"{mean:.4f} +/- {sd:.4f}" if pd.notna(mean) else "--"

    disp = pd.DataFrame({
        "Model": overview["model"],
        "Variant": overview["variant"],
        f"RMSE {primary_label} (mean+/-sd)": [fmt(m, s) for m, s in zip(overview["rmse_mean"], overview["rmse_sd"])],
        "MAE (mean+/-sd)": [fmt(m, s) for m, s in zip(overview["mae_mean"], overview["mae_sd"])],
        "R2 (mean+/-sd)": [fmt(m, s) for m, s in zip(overview["r2_mean"], overview["r2_sd"])],
        "Skill vs Persistence": [fmt(m, s) for m, s in zip(overview["skill_mean"], overview["skill_sd"])],
        "Skill vs ICON-D2": [fmt(m, s) for m, s in zip(overview["skill_nwp_mean"], overview["skill_nwp_sd"])],
    })
    # Both skill definitions in the same table — see build_overview_table() for why they differ.
    for col, header in [("skill_ratio_mean", "Skill vs Pers (ratio of means)"),
                        ("skill_nwp_ratio_mean", "Skill vs ICON-D2 (ratio of means)")]:
        if col in overview.columns:
            disp[header] = [f"{v:.4f}" if pd.notna(v) else "--" for v in overview[col]]
    if "pooled_rmse_mean" in overview.columns:
        disp[f"RMSE {primary_label} pooled (mean+/-sd)"] = [fmt(m, s) for m, s in
                                            zip(overview["pooled_rmse_mean"], overview["pooled_rmse_sd"])]
    # Stage 1 (full, as delivered) and Stage 2 (own-sample, imputed-hour-filtered) columns, right
    # next to the primary (Stage-flag-selected) ones -- see build_overview_table()'s
    # station_df_unfiltered / station_df_ownfiltered parameters and the module docstring's
    # THREE-STAGE TABLE section.
    for col, header in [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R2"),
                        ("skill", "Skill vs Persistence"), ("skill_nwp", "Skill vs ICON-D2")]:
        mcol, scol = f"{col}_mean_unfiltered", f"{col}_sd_unfiltered"
        if mcol in overview.columns:
            disp[f"{header} Stage1-full (mean+/-sd)"] = [fmt(m, s) for m, s in zip(overview[mcol], overview[scol])]
        mcol, scol = f"{col}_mean_ownfiltered", f"{col}_sd_ownfiltered"
        if mcol in overview.columns:
            disp[f"{header} Stage2-ownfiltered (mean+/-sd)"] = [fmt(m, s) for m, s in zip(overview[mcol], overview[scol])]
    disp["Transductive"] = overview["transductive"].map({True: "yes (sees target history)", False: ""})
    return disp.to_markdown(index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Bar chart: RMSE of all variants + ICON-D2 + Persistence, sorted, error bars = SD over folds
# ─────────────────────────────────────────────────────────────────────────────
def fig_bar_rmse(overview: pd.DataFrame, out_dir: Path, formats):
    df = overview.copy()
    df["label"] = df["model"] + " " + df["variant"]
    df = df.sort_values("rmse_mean")

    # ICON-D2 / Persistence are references, not trained models — they must not be painted in the
    # "inductive model" colour, otherwise the legend asserts something false about them.
    is_ref = df["variant"].eq("REF").to_numpy()
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.9), 6))
    x = np.arange(len(df))
    colors = ["#888888" if r else ("#c44e52" if t else "steelblue")
              for r, t in zip(is_ref, df["transductive"])]
    hatches = ["" if r else ("//" if t else "") for r, t in zip(is_ref, df["transductive"])]
    bars = ax.bar(x, df["rmse_mean"], yerr=df["rmse_sd"], capsize=4, color=colors,
                  edgecolor="white", linewidth=0.6, error_kw={"elinewidth": 1.3, "alpha": 0.75})
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    # Persistence sits ~1 m/s above everything else and flattens the model bars into a
    # visually identical row. Annotate the values so the ranking stays readable.
    for xi, v in zip(x, df["rmse_mean"]):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("RMSE (m/s)", fontsize=12)
    ax.set_title("stdhp dry run: RMSE by variant (mean +/- SD over 3 spatial folds)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="steelblue", label="Inductive model (target station unseen in training)"),
        Patch(facecolor="#c44e52", hatch="//", label="GRID+HIST — transductive, sees target-station\n"
                                                       "history — NOT an inductive comparison point"),
        Patch(facecolor="#888888", label="Reference, not a trained model\n"
                                          "(ICON-D2 nwp_ref / temporal persistence pers_ref)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"01_bar_rmse_all_variants.{fmt}", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Fold-dispersion parallel-coordinates plot for the DCRNN ablation ladder
#    (6 rungs: A/GRID, B/GRID-NOMEAS, C/GRID-NOGRAPH, D/GRID-IDW,
#    D'/GRID-IDW-ALT, BASE -- see ABLATION_LETTER_LABEL)
# ─────────────────────────────────────────────────────────────────────────────
# Order + colours for fig_fold_dispersion, ColorBrewer Dark2 (colourblind-safe),
# continued for D/D' with the palette's next two entries. Kept separate from
# ABLATION_LETTER_LABEL's dict (whose order is "ladder rung order", not
# necessarily "plot legend order") so this figure's presentation is an
# explicit choice, not an accident of insertion order elsewhere.
_FOLD_DISPERSION_ORDER = ["GRID", "GRID-NOMEAS", "GRID-NOGRAPH", "GRID-IDW", "GRID-IDW-ALT", "BASE"]
_FOLD_DISPERSION_COLORS = {
    "GRID": "#1b9e77", "GRID-NOMEAS": "#d95f02", "GRID-NOGRAPH": "#7570b3",
    "GRID-IDW": "#e7298a", "GRID-IDW-ALT": "#66a61e", "BASE": "#888888",
}


def fig_fold_dispersion(station_df: pd.DataFrame, out_dir: Path, formats):
    dcrnn = station_df[(station_df["model"] == "DCRNN") &
                        (station_df["variant"].isin(_FOLD_DISPERSION_ORDER))]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    # Rungs without data yet (e.g. D/D' before their stdhp runs exist) would
    # otherwise plot an all-NaN line and an "nan" annotation -- skip them
    # instead, same reasoning as fig_paired_diff_boxplots.
    present = [v for v in _FOLD_DISPERSION_ORDER if not dcrnn[dcrnn["variant"] == v].empty]
    missing = [v for v in _FOLD_DISPERSION_ORDER if v not in present]
    if missing:
        print(f"[fig_fold_dispersion] no data yet for {missing} -- plotting only {present}")
    for variant in present:
        sub = dcrnn[dcrnn["variant"] == variant]
        means = sub.groupby("fold")["rmse"].mean().reindex(FOLDS)
        color = _FOLD_DISPERSION_COLORS[variant]
        ax.plot(FOLDS, means.values, marker="o", linewidth=2.2, markersize=8,
                color=color, label=variant)
        for f, v in zip(FOLDS, means.values):
            ax.annotate(f"{v:.3f}", (f, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color=color)
    ax.set_xticks(FOLDS)
    ax.set_xticklabels([f"Fold {f}" for f in FOLDS])
    ax.set_ylabel("RMSE (m/s), mean over 51 stations", fontsize=12)
    ax.set_title("DCRNN ablation ladder: RMSE per fold\n"
                 "(if lines cross, the fold-mean ablation ranking is not stable)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"02_fold_dispersion_dcrnn_ablations.{fmt}", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5) Paired difference analysis for the ablation ladder
# ─────────────────────────────────────────────────────────────────────────────
def paired_diff_analysis(station_df: pd.DataFrame, n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    # Derived from ABLATION_LETTER_LABEL (single source of truth for the ladder's
    # letter<->variant mapping) rather than duplicated here, so adding a rung
    # there (as done for D/D', R6(a)) automatically reaches this function too.
    letter_of_variant = {v: k for k, v in ABLATION_LETTER_LABEL.items()}
    dcrnn = station_df[(station_df["model"] == "DCRNN") &
                        (station_df["variant"].isin(letter_of_variant))].copy()
    dcrnn["letter"] = dcrnn["variant"].map(letter_of_variant)
    # pivot_table's default aggfunc="mean" would silently average duplicate (fold, station,
    # variant) rows instead of failing — a duplicated CSV would then look like a valid pairing.
    dup = dcrnn.duplicated(["fold", "station_id", "letter"]).sum()
    if dup:
        sys.exit(f"[FATAL] {dup} doppelte (fold, station, variant)-Zeilen in den stdhp-CSVs. "
                 f"Die gepaarte Analyse wuerde sie stillschweigend mitteln -- Abbruch.")
    # Pairing is an explicit join on (fold, station_id): variant A and variant B are only ever
    # compared at the same station of the same spatial fold, never by row position.
    pivot = dcrnn.pivot_table(index=["fold", "station_id"], columns="letter", values="rmse")

    rng = np.random.default_rng(seed)
    results = []
    for a, b in ABLATION_PAIRS:
        if a not in pivot.columns or b not in pivot.columns:
            continue
        sub = pivot[[a, b]].dropna()
        diffs_by_fold = {}
        for fold in FOLDS:
            if fold not in sub.index.get_level_values("fold"):
                continue
            d = (sub.xs(fold, level="fold")[a] - sub.xs(fold, level="fold")[b]).to_numpy()
            if len(d) == 0:
                continue
            diffs_by_fold[fold] = d
            row = _diff_stats(f"{a}-{b} ({ABLATION_LETTER_LABEL[a]} minus {ABLATION_LETTER_LABEL[b]})",
                               str(fold), d, rng, n_boot)
            row["group"] = "DCRNN ablation ladder"
            results.append(row)
        if diffs_by_fold:
            all_d = np.concatenate(list(diffs_by_fold.values()))
            row = _diff_stats(f"{a}-{b} ({ABLATION_LETTER_LABEL[a]} minus {ABLATION_LETTER_LABEL[b]})",
                               "pooled", all_d, rng, n_boot, diffs_by_fold=diffs_by_fold)
            row["group"] = "DCRNN ablation ladder"
            results.append(row)
    return pd.DataFrame(results)


def _pivot_rmse_for(station_df: pd.DataFrame, model: str, variant: str) -> pd.Series:
    """Per-(fold, station_id) RMSE series for one (model, variant), indexed for joining."""
    sub = station_df[(station_df["model"] == model) & (station_df["variant"] == variant)]
    dup = sub.duplicated(["fold", "station_id"]).sum()
    if dup:
        sys.exit(f"[FATAL] {dup} doppelte (fold, station)-Zeilen fuer {model} {variant}. "
                 f"Die gepaarte Analyse wuerde sie stillschweigend mitteln -- Abbruch.")
    return sub.set_index(["fold", "station_id"])["rmse"]


def paired_diff_generic(station_df: pd.DataFrame, pairs, n_boot: int = 10000, seed: int = 1) -> pd.DataFrame:
    """
    Generic paired per-(fold, station) RMSE-difference analysis for arbitrary (model, variant)
    pairs, including CROSS-MODEL pairs (e.g. DCRNN GRID vs. MTGNN GRID). Same methodology as
    paired_diff_analysis() (the DCRNN-ladder-only version): explicit join on (fold, station_id)
    -- never positional -- Wilcoxon signed-rank, percentile bootstrap (+ fold-stratified bootstrap
    for the pooled rows), reported per fold and pooled across folds.

    Pairing across models is valid here because every (model, variant) combination is evaluated
    on the SAME 51 target stations per fold (configs/spatial_folds.yaml val_files) -- a cross-model
    join on (fold, station_id) compares the same station under the same fold, exactly like a
    within-model one.

    `pairs`: iterable of (group, label, model_a, variant_a, model_b, variant_b); `group` is only
    used to sort/label the output table (e.g. "MTGNN ablation", "Cross-model (inductive)").
    A different `seed` than paired_diff_analysis()'s default is used deliberately so the two
    functions' bootstrap draws are independent random streams, not a silent reuse of the same
    sequence across unrelated pairs.
    """
    rng = np.random.default_rng(seed)
    results = []
    for group, label, ma, va, mb, vb in pairs:
        sa = _pivot_rmse_for(station_df, ma, va)
        sb = _pivot_rmse_for(station_df, mb, vb)
        if sa.empty or sb.empty:
            warnings.warn(f"Keine Daten fuer Paar '{label}' ({ma} {va} vs. {mb} {vb}) -- uebersprungen.")
            continue
        joined = pd.concat([sa.rename("a"), sb.rename("b")], axis=1).dropna()
        if joined.empty:
            warnings.warn(f"Kein gemeinsamer (fold,station)-Schnitt fuer '{label}' -- uebersprungen.")
            continue
        diffs_by_fold = {}
        for fold in FOLDS:
            if fold not in joined.index.get_level_values("fold"):
                continue
            d = (joined.xs(fold, level="fold")["a"] - joined.xs(fold, level="fold")["b"]).to_numpy()
            if len(d) == 0:
                continue
            diffs_by_fold[fold] = d
            row = _diff_stats(label, str(fold), d, rng, n_boot)
            row["group"] = group
            results.append(row)
        if diffs_by_fold:
            all_d = np.concatenate(list(diffs_by_fold.values()))
            row = _diff_stats(label, "pooled", all_d, rng, n_boot, diffs_by_fold=diffs_by_fold)
            row["group"] = group
            results.append(row)
    return pd.DataFrame(results)


def _bootstrap_stratified_ci(diffs_by_fold: dict, rng, n_boot: int = 10000):
    """
    Fold-stratified bootstrap for a "pooled" row: resample stations WITH replacement
    INDEPENDENTLY WITHIN each fold, then concatenate the resampled fold-vectors and take the
    mean -- this preserves which fold each station-difference came from. The plain/flat pooled
    bootstrap (ci95_lo/ci95_hi) instead resamples directly from the 153 concatenated
    station-differences as if they were one exchangeable sample, ignoring fold membership
    entirely. The two are reported side by side (ci95_lo/hi vs. ci95_lo_stratified/hi) precisely
    to check whether that simplification matters: if fold identity carried extra between-fold
    variance (e.g. from spatial autocorrelation or systematically different station composition
    per fold), the stratified CI would be visibly wider or shifted. Verified here: they agree to
    <0.001 m/s for every pooled row, i.e. the fold structure is inert for this comparison and the
    simpler flat bootstrap can be trusted -- but the column stays so a future pair with real
    between-fold variance would show it instead of being silently averaged away.
    """
    arrays = list(diffs_by_fold.values())
    total_n = sum(len(a) for a in arrays)
    boot_sum = np.zeros(n_boot)
    for arr in arrays:
        n_f = len(arr)
        idx = rng.integers(0, n_f, size=(n_boot, n_f))
        boot_sum += arr[idx].sum(axis=1)
    boot_means = boot_sum / total_n
    return np.percentile(boot_means, [2.5, 97.5])


def _diff_stats(pair_label, fold_label, d, rng, n_boot, diffs_by_fold=None):
    """
    d = per-station RMSE(first variant) - RMSE(second variant); NEGATIVE means the first-named
    variant is better (lower RMSE). Wilcoxon and the bootstrap operate on exactly this one
    paired vector, so p-value, CI and mean_diff_rmse always refer to the same sign convention.

    `diffs_by_fold`: pass the {fold: array} dict when `fold_label == "pooled"` to additionally
    compute the fold-stratified bootstrap CI (ci95_lo_stratified/hi) alongside the flat one
    (ci95_lo/hi) -- see _bootstrap_stratified_ci(). Left as NaN for per-fold rows, where the
    distinction is moot (a single fold's own CI already respects fold membership trivially).
    """
    if np.any(d != 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Explicit: two-sided (we do not assume a direction a priori), zeros discarded
            # (scipy's "wilcox" default) — spelled out so a scipy default change cannot
            # silently alter published p-values.
            stat, p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    else:
        p = np.nan
    # Bootstrap unit = the paired station difference, resampled with replacement. Resampling the
    # two variants independently would destroy the pairing and inflate the CI.
    boot_idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot_means = d[boot_idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    result = {
        "pair": pair_label, "fold": fold_label, "n_stations": len(d),
        "mean_diff_rmse": float(d.mean()), "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        "wilcoxon_p": float(p) if pd.notna(p) else np.nan,
        "frac_first_better": float((d < 0).mean()),
        "ci95_lo_stratified": np.nan, "ci95_hi_stratified": np.nan,
    }
    if diffs_by_fold is not None:
        s_lo, s_hi = _bootstrap_stratified_ci(diffs_by_fold, rng, n_boot)
        result["ci95_lo_stratified"] = float(s_lo)
        result["ci95_hi_stratified"] = float(s_hi)
    return result


def fig_paired_diff_boxplots(station_df: pd.DataFrame, out_dir: Path, formats):
    # See paired_diff_analysis() — same derivation, same reasoning.
    letter_of_variant = {v: k for k, v in ABLATION_LETTER_LABEL.items()}
    dcrnn = station_df[(station_df["model"] == "DCRNN") &
                        (station_df["variant"].isin(letter_of_variant))].copy()
    dcrnn["letter"] = dcrnn["variant"].map(letter_of_variant)
    pivot = dcrnn.pivot_table(index=["fold", "station_id"], columns="letter", values="rmse")

    # Ladder rungs without any data yet (e.g. D/D' before their stdhp runs exist)
    # have no column in pivot at all -- skip those pairs instead of a KeyError,
    # so this figure keeps working for whichever rungs ARE available.
    pairs = [(a, b) for a, b in ABLATION_PAIRS if a in pivot.columns and b in pivot.columns]
    missing = [(a, b) for a, b in ABLATION_PAIRS if (a, b) not in pairs]
    if missing:
        print(f"[fig_paired_diff_boxplots] skipping {missing} — no data yet for "
              f"{sorted({x for pair in missing for x in pair if x not in pivot.columns})}")
    if not pairs:
        print("[fig_paired_diff_boxplots] no ablation pair has data yet — skipping figure.")
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (a, b) in zip(axes, pairs):
        sub = pivot[[a, b]].dropna()
        box_data = [sub.xs(f, level="fold")[a] - sub.xs(f, level="fold")[b]
                    for f in FOLDS if f in sub.index.get_level_values("fold")]
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax.boxplot(box_data, labels=[f"Fold {f}" for f in FOLDS], showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#a6cee3", alpha=0.7))
        # Place the annotation in axes coordinates: ax.get_ylim() grows while the shared y axis is
        # still being autoscaled, so a data-coordinate offset ends up at a different height in
        # every panel and can collide with the whiskers.
        for i, d in enumerate(box_data, start=1):
            frac = float((d < 0).mean())
            ax.text(i, 1.02, f"{frac * 100:.0f}% improved", ha="center", fontsize=8,
                    transform=ax.get_xaxis_transform())
        ax.set_title(f"{ABLATION_LETTER_LABEL[a]} minus {ABLATION_LETTER_LABEL[b]}", fontsize=11)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Per-station RMSE difference (m/s)\nnegative = first variant better", fontsize=10)
    fig.suptitle("Paired per-station RMSE differences, DCRNN ablation ladder", fontsize=13, y=1.03)
    plt.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"03_paired_diff_boxplots.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6) Error by forecast horizon
# ─────────────────────────────────────────────────────────────────────────────
def _rmse_by_key(bucket_df: pd.DataFrame, model: str, variant: str, key: str) -> pd.Series | None:
    sub = bucket_df[(bucket_df["model"] == model) & (bucket_df["variant"] == variant)]
    if sub.empty:
        return None
    per_fold = sub.groupby([key, "fold"], observed=True).agg(sse=("sse", "sum"), n=("n", "sum")).reset_index()
    per_fold["rmse"] = np.sqrt(per_fold["sse"] / per_fold["n"])
    return per_fold.pivot(index=key, columns="fold", values="rmse")


def _ref_rmse_by_key(ref_bucket_df: pd.DataFrame, model: str, key: str) -> pd.Series | None:
    sub = ref_bucket_df[ref_bucket_df["model"] == model]
    if sub.empty:
        return None
    per_fold = sub.groupby([key, "fold"], observed=True).agg(sse=("sse", "sum"), n=("n", "sum")).reset_index()
    per_fold["rmse"] = np.sqrt(per_fold["sse"] / per_fold["n"])
    return per_fold.pivot(index=key, columns="fold", values="rmse")


def fig_horizon_error(raw_bundle: dict, out_dir: Path, formats):
    horizon_df, ref_horizon_df = raw_bundle["horizon"], raw_bundle["ref_horizon"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    panels = [(0, "Fold 0"), (1, "Fold 1"), (2, "Fold 2"), (None, "Average over folds")]

    for ax, (fold, title) in zip(axes, panels):
        for prefix, (model, variant, letter) in ALL_META.items():
            piv = _rmse_by_key(horizon_df, model, variant, "horizon")
            if piv is None:
                continue
            y = piv.mean(axis=1) if fold is None else piv.get(fold)
            if y is None:
                continue
            ax.plot(y.index.to_numpy(), y.to_numpy(), label=f"{model} {variant}", linewidth=1.6,
                    color=MODEL_COLORS.get(model), alpha=0.9,
                    linestyle=VARIANT_LINESTYLE.get(variant, "solid"))
        for ref_model in ["ICON-D2", "Persistence"]:
            piv = _ref_rmse_by_key(ref_horizon_df, ref_model, "horizon")
            if piv is None:
                continue
            y = piv.mean(axis=1) if fold is None else piv.get(fold)
            if y is None:
                continue
            ax.plot(y.index.to_numpy(), y.to_numpy(), label=ref_model, linewidth=2.2, color=MODEL_COLORS[ref_model],
                    linestyle=":")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Forecast horizon (h)", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RMSE (m/s)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=6, fontsize=8.5)
    fig.suptitle("RMSE by forecast horizon — all stdhp variants + ICON-D2 + Persistence\n"
                 "(pooled over the 51 stations within each fold — NOT the per-station mean of the "
                 "overview table; ICON-D2 sits ~0.12 m/s higher here for that reason)",
                 fontsize=12, y=1.30)
    plt.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"04_error_by_horizon.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7) Error by wind-speed class / by month, important variants only
# ─────────────────────────────────────────────────────────────────────────────
def _important_series(bucket_df, ref_bucket_df, key, order=None):
    series = {}
    for model, variant in IMPORTANT_VARIANTS:
        piv = _rmse_by_key(bucket_df, model, variant, key)
        if piv is not None:
            series[f"{model} {variant}"] = piv.mean(axis=1)
    for ref_model in IMPORTANT_REF_MODELS:
        piv = _ref_rmse_by_key(ref_bucket_df, ref_model, key)
        if piv is not None:
            series[ref_model] = piv.mean(axis=1)
    if order is not None:
        for name in series:
            series[name] = series[name].reindex(order)
    return series


def fig_error_by_ws_class(raw_bundle, out_dir, formats):
    series = _important_series(raw_bundle["ws_bin"], raw_bundle["ref_ws_bin"], "ws_bin", order=WS_LABELS)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, y in series.items():
        model, _, variant = name.partition(" ")
        label = f"{name} [transductive]" if is_transductive(model, variant) else name
        ax.plot(range(len(WS_LABELS)), y.values, marker="o", label=label,
                color=MODEL_COLORS.get(model, None), linewidth=2,
                linestyle=VARIANT_LINESTYLE.get(variant, "solid"))
    ax.set_xticks(range(len(WS_LABELS)))
    ax.set_xticklabels(WS_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Wind speed class (m/s), ground truth, left-inclusive; last class is open-ended")
    ax.set_ylabel("RMSE (m/s), pooled within fold")
    ax.set_title("RMSE by wind-speed class — key variants (averaged over 3 folds)\n"
                 "Binned on the GROUND TRUTH, so each class is conditioned on the observation; "
                 "gt < 0 (non-physical, upstream) falls in no class and is excluded here only.")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"05_error_by_windspeed_class.{fmt}", dpi=150)
    plt.close(fig)


def fig_error_by_month(raw_bundle, out_dir, formats):
    months = list(range(1, 13))
    series = _important_series(raw_bundle["month"], raw_bundle["ref_month"], "month", order=months)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, y in series.items():
        model, _, variant = name.partition(" ")
        label = f"{name} [transductive]" if is_transductive(model, variant) else name
        ax.plot(months, y.values, marker="o", label=label, color=MODEL_COLORS.get(model, None),
                linewidth=2, linestyle=VARIANT_LINESTYLE.get(variant, "solid"))
    ax.set_xticks(months)
    ax.set_xlabel("Month (calendar month of valid_time, UTC)")
    ax.set_ylabel("RMSE (m/s), pooled within fold")
    ax.set_title("RMSE by calendar month — key variants (averaged over 3 folds)\n"
                 "Note: months are pooled across all 3 folds' val windows (Aug 2024-Aug 2025)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"06_error_by_month.{fmt}", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 8) Station-wise scatter: model RMSE vs ICON-D2 RMSE
# ─────────────────────────────────────────────────────────────────────────────
def fig_station_scatter(raw_bundle, out_dir, formats):
    station_df, ref_station_df = raw_bundle["station"], raw_bundle["ref_station"]
    ref_icon = ref_station_df[ref_station_df["model"] == "ICON-D2"].copy()
    ref_icon["rmse_icon"] = np.sqrt(ref_icon["sse"] / ref_icon["n"])

    for model, variant, fname in [("DCRNN", "GRID", "07_scatter_dcrnn_grid_vs_icon"),
                                   ("MTGNN", "GRID", "08_scatter_mtgnn_grid_vs_icon")]:
        sub = station_df[(station_df["model"] == model) & (station_df["variant"] == variant)].copy()
        if sub.empty:
            continue
        sub["rmse_model"] = np.sqrt(sub["sse"] / sub["n"])
        merged = sub.merge(ref_icon[["station_id", "fold", "rmse_icon"]], on=["station_id", "fold"], how="inner")

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        for fold in FOLDS:
            m = merged[merged["fold"] == fold]
            ax.scatter(m["rmse_icon"], m["rmse_model"], color=FOLD_COLORS[fold], alpha=0.75, s=35,
                       label=f"Fold {fold} (n={len(m)})")
        lim_max = max(merged["rmse_icon"].max(), merged["rmse_model"].max()) * 1.08
        ax.plot([0, lim_max], [0, lim_max], "--", color="#999", linewidth=1.2, zorder=0)
        ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
        ax.set_xlabel("ICON-D2 RMSE (m/s)", fontsize=12)
        ax.set_ylabel(f"{model} {variant} RMSE (m/s)", fontsize=12)
        ax.set_title(f"Per-station RMSE: {model} {variant} vs. ICON-D2\n"
                     "below diagonal = model beats ICON-D2 at that station", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        for fmt in formats:
            fig.savefig(out_dir / f"{fname}.{fmt}", dpi=150)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 9) Skill (vs. ICON-D2) distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig_skill_distribution(station_df: pd.DataFrame, out_dir: Path, formats):
    variants = list(ALL_META.values())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for model, variant, letter in variants:
        sub = station_df[(station_df["model"] == model) & (station_df["variant"] == variant)]
        vals = sub["skill_nwp"].dropna().values
        if len(vals) == 0:
            continue
        label = f"{model} {variant} (n={len(vals)})"
        ax1.hist(vals, bins=30, histtype="step", linewidth=1.8, label=label, color=None)
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax2.plot(xs, ys, linewidth=1.8, label=label)

    for ax in (ax1, ax2):
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("skill_nwp = 1 - RMSE_model / RMSE_ICON-D2", fontsize=11)
        ax.grid(alpha=0.3)
    ax1.set_ylabel("Station-fold count (of up to 153)")
    ax1.set_title("Histogram")
    ax2.set_ylabel("ECDF")
    ax2.set_title("Empirical CDF")
    ax2.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Distribution of skill vs. ICON-D2 across all 153 (station, fold) pairs, per variant", fontsize=13)
    plt.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"09_skill_nwp_distribution.{fmt}", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Extra tables: stations with skill_nwp <= 0
# ─────────────────────────────────────────────────────────────────────────────
def _crosscheck_csv_vs_raw(csv_station_df: pd.DataFrame, raw_station_df: pd.DataFrame,
                            atol: float = 1e-6, n_multiplier: int = 1) -> None:
    """
    The overview table is built from the stdhp_*.csv per-station metrics; every stratified figure
    is built from the raw parquets. Both must describe the same sample. Compare the CSV `rmse`
    against sqrt(SSE/n) recomputed from the parquet of the same (model, variant, fold, station).

    `n_multiplier`: the graph CSVs' n_samples IS the raw-parquet row count (1 run_time x 1
    horizon = 1 row = 1 sample, n_multiplier=1). The TFT CSVs' n_samples instead counts DISTINCT
    forecast issuances (run_time) -- every TFT run has the full 48-horizon output (verified: the
    ratio raw_row_count / n_samples is EXACTLY 48 for all 6 files x 51 stations, no exceptions),
    so n_samples * 48 == the raw row count for that station. n_multiplier=48 for the TFT call
    site; getting this wrong (comparing n_samples directly against raw row count) would abort on
    a difference that is not actually a data problem -- get_test_results_tft_bc.py just counts a
    different thing than evaluate()/evaluation.py's `n_samples` does for the graph models.
    """
    if raw_station_df.empty:
        return
    raw = raw_station_df.copy()
    raw["rmse_raw"] = np.sqrt(raw["sse"] / raw["n"])
    merged = csv_station_df.merge(raw[["model", "variant", "fold", "station_id", "rmse_raw", "n"]],
                                  on=["model", "variant", "fold", "station_id"], how="inner")
    if len(merged) != len(csv_station_df):
        warnings.warn(f"CSV/Parquet-Abgleich: nur {len(merged)} von {len(csv_station_df)} "
                      f"Stationszeilen haben ein Gegenstueck im Rohparquet.")
    d = (merged["rmse"] - merged["rmse_raw"]).abs()
    n_mismatch = int((merged["n_samples"] * n_multiplier != merged["n"]).sum())
    if float(d.max()) > atol or n_mismatch:
        sys.exit(f"[FATAL] CSV-RMSE und aus dem Rohparquet nachgerechnetes RMSE weichen um bis zu "
                 f"{float(d.max()):.3g} ab ({n_mismatch} Zeilen mit abweichender Samplezahl, "
                 f"n_multiplier={n_multiplier}). Uebersichtstabelle und stratifizierte Figuren "
                 f"wuerden verschiedene Stichproben beschreiben -- Abbruch.")
    print(f"[OK] CSV-RMSE == sqrt(SSE/n) aus den Rohparquets fuer alle {len(merged)} "
          f"(variant, fold, station)-Zeilen (max. Abweichung {float(d.max()):.2g}, "
          f"n_multiplier={n_multiplier}).")


def stations_below_icon_table(station_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, variant), g in station_df.groupby(["model", "variant"]):
        n_total = len(g)
        n_below = int((g["skill_nwp"] <= 0).sum())
        rows.append({"model": model, "variant": variant, "n_stations_total": n_total,
                     "n_skill_nwp_le_0": n_below, "share": n_below / n_total if n_total else np.nan})
    return pd.DataFrame(rows).sort_values("share")


# ─────────────────────────────────────────────────────────────────────────────
# README
# ─────────────────────────────────────────────────────────────────────────────
README_TEMPLATE = """# stdhp figures

Generated by `geostatistics/stdrun/make_stdhp_figures.py`. See the script's module docstring for
the exact aggregation convention (per-fold-then-averaged vs. pooled) before quoting any number.

Every table/figure here EXCLUDES imputed-hour rows by default (~0.6% of evaluated station-hours,
see `imputed_coverage.md`/`.csv` and the module docstring's IMPUTED-HOUR FILTERING section) --
run with `--keep-imputed` to reproduce the old, unfiltered numbers.

ECMWF is not included anywhere here: the only ECMWF reference CSVs on disk cover a different
(non-overlapping) time window than the stdhp runs — see the script docstring.

## Figures

- `01_bar_rmse_all_variants.*` — RMSE of all 10 stdhp variants + ICON-D2 + Persistence, sorted
  ascending, error bars = SD over the 3 spatial folds; GRID+HIST bars are hatched/red because
  they are transductive (see target-station history) and not an inductive comparison point.
- `02_fold_dispersion_dcrnn_ablations.*` — RMSE per fold (0/1/2) for the 4 DCRNN ablation variants
  (GRID / GRID-NOMEAS / GRID-NOGRAPH / BASE) as connected lines, to show how much the fold-mean
  ablation ranking depends on which fold you look at.
- `03_paired_diff_boxplots.*` — per-station RMSE differences (paired by station) for the 4
  ablation-ladder comparisons A-B, B-C, A-C, A-BASE, one boxplot per fold; percentage above each
  box is the share of stations where the first-named variant had the lower RMSE.
- `04_error_by_horizon.*` — RMSE vs. forecast horizon (1-48h) for all variants + ICON-D2 +
  Persistence, one panel per fold plus one panel averaged over folds.
- `05_error_by_windspeed_class.*` — RMSE by ground-truth wind-speed class (2 m/s bins) for the
  key variants (DCRNN GRID/BASE/GRID+HIST, MTGNN GRID/GRID+HIST, WaveNet GRID, ICON-D2,
  Persistence), averaged over folds. Both GRID+HIST variants are marked "[transductive]" in the
  legend (see the ablation-price rows in paired_diff_stats.md for how large that price is).
- `06_error_by_month.*` — RMSE by calendar month for the same key variants (incl. both
  GRID+HIST, marked "[transductive]"), averaged over folds (months are pooled across the 3
  folds' identical Aug 2024-Aug 2025 val window).
- `07_scatter_dcrnn_grid_vs_icon.*` / `08_scatter_mtgnn_grid_vs_icon.*` — per-station RMSE of the
  model vs. ICON-D2, colored by fold, with the y=x diagonal; points below the diagonal are
  stations where the model beats ICON-D2.
- `09_skill_nwp_distribution.*` — histogram + ECDF of skill_nwp (skill vs. ICON-D2) across all
  153 (station, fold) pairs, one line/curve per variant, to show whether skill is broad-based or
  carried by a few stations.

## Tables

- `overview_table.md` / `.csv` — per-variant RMSE/MAE/R2/skill vs. Persistence/skill vs. ICON-D2,
  mean +/- SD over the 3 folds (imputed hours EXCLUDED, per-station-mean over the raw parquets --
  no longer from the stdhp_*.csv columns, see build_station_metrics()), plus per-fold values and
  the pooled-RMSE variant. Also carries "... unfiltered" columns computed on the full,
  un-filtered sample side by side, so the size of the imputed-hour effect stays visible in the
  same table (e.g. DCRNN GRID fold 0: 1.1484 unfiltered -> 1.1080 filtered, -3.5%).
  TWO skill definitions are exported side by side: the "Skill vs ..." columns are the mean over
  stations of the per-station skill (what evaluation.py writes into the CSVs), the
  "... (ratio of means)" columns are 1 - mean(RMSE)/mean(RMSE_ref). They differ by ~0.02-0.03 and
  reorder WaveNet GRID vs. MTGNN GRID — name the one you quote in the caption.
- `imputed_coverage.md` / `.csv` — per fold and station: number of evaluated hours, number of
  those that are imputed (no raw measurement), and the share; plus one fold-total row per fold.
  This is the number that justifies the filter's magnitude in the paper.
- `stations_unbinned_gt.csv` — rows whose ground truth is < 0 m/s and therefore falls into no
  wind-speed class. With the default imputed-hour filter this should be empty (gt < 0 is itself
  always an imputation artifact, see verify_imputation_mask()); non-empty only under
  --keep-imputed, where such rows ARE scored everywhere except in `05_error_by_windspeed_class`.
- `paired_diff_stats.md` / `.csv` — mean RMSE difference, 95% bootstrap CI (flat AND
  fold-stratified, side by side), Wilcoxon signed-rank p-value and share of stations improved,
  per fold and pooled, grouped by the "group" column: DCRNN ablation ladder (A-B/B-C/A-C/A-BASE),
  MTGNN ablation, WaveNet ablation, cross-model inductive comparisons (DCRNN/MTGNN/WaveNet GRID
  vs. each other) and the per-model transductive price (GRID vs. GRID+HIST).
- `stations_below_icon.md` / `.csv` — number and share of (up to 153) station-fold pairs per
  variant where skill_nwp <= 0, i.e. the model does not beat ICON-D2 at that station.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--station-raw-dir", type=Path, default=DEFAULT_STATION_RAW_DIR,
                     help="Per-station raw 10-min measurement parquets (Station_<id>.parquet), "
                          "used only to reconstruct the imputed-hour mask.")
    ap.add_argument("--spatial-folds", type=Path, default=REPO_ROOT / "configs" / "spatial_folds.yaml",
                     help="Used only for the independent fold-assignment re-verification "
                          "(verify_fold_assignment()).")
    ap.add_argument("--formats", type=str, default="png,pdf")
    ap.add_argument("--skip-raw", action="store_true",
                     help="Skip everything that needs the raw prediction parquets (pooled RMSE, "
                          "ICON-D2/Persistence reference derivation, imputed-hour mask/filtering, "
                          "the 14-variant intersection, TFT, horizon/month/ws-class/scatter "
                          "figures). The overview table then falls back to the UNFILTERED "
                          "stdhp_*.csv sample (imputed hours included, graph only), with a "
                          "warning -- none of this can be reconstructed from the CSVs alone.")
    ap.add_argument("--keep-imputed", action="store_true",
                     help="Do not filter out imputed-hour rows; reproduces the pre-filter numbers "
                          "for comparison. Default is to filter them out (see module docstring, "
                          "IMPUTED-HOUR FILTERING). No effect together with --skip-raw.")
    ap.add_argument("--intersect-only", action=argparse.BooleanOptionalAction, default=True,
                     help="Task B, Stage 2 vs Stage 3: whether the PRIMARY reported numbers "
                          "(overview table's un-suffixed columns, bar chart, fold-dispersion, "
                          "skill-distribution and paired-diff figures/tables) are computed on the "
                          "Stage 3 14-variant (station_id, run_time, horizon) intersection "
                          "(default, --intersect-only) or on each variant's own Stage 2 "
                          "imputed-hour-filtered sample (--no-intersect-only, the graph-only "
                          "script's original behaviour). Both stages are ALWAYS written to the "
                          "overview table side by side regardless of this flag (see "
                          "build_overview_table()'s Stage1-full/Stage2-ownfiltered columns) -- the "
                          "flag only picks which one drives everything downstream of it. No effect "
                          "together with --skip-raw (no TFT/intersection there at all).")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/9] Loading {len(STDHP_META) * len(FOLDS)} graph stdhp CSVs from {args.results_dir} ...")
    station_df_csv = load_station_csvs(args.results_dir)
    print(f"      -> {len(station_df_csv):,} station rows, "
          f"{station_df_csv.groupby(['model', 'variant']).ngroups} variants.")

    print(f"[1b/9] Loading {len(TFT_META) * len(FOLDS)} TFT stdhp CSVs from {args.results_dir} ...")
    tft_df_csv = load_tft_station_csvs(args.results_dir)
    print(f"      -> {len(tft_df_csv):,} station rows, "
          f"{tft_df_csv.groupby(['model', 'variant']).ngroups} variants.")

    print("[1c/9] Verifying fold assignment against configs/spatial_folds.yaml (independent check) ...")
    verify_fold_assignment(tft_df_csv, station_df_csv, args.spatial_folds)

    raw_bundle = None
    station_metrics = None
    station_metrics_unfiltered = None
    station_metrics_intersect = None
    intersect_sizes = None

    if not args.skip_raw:
        print("[2/9] Verifying nwp_ref/pers_ref identity across GRAPH variants ...")
        verify_and_derive_references(args.raw_dir)

        print("[3/9] Reconstructing & verifying the imputed-hour mask ...")
        all_station_ids = sorted(set(station_df_csv["station_id"]) | set(tft_df_csv["station_id"]))
        mask = build_imputation_mask(all_station_ids, args.station_raw_dir)
        canon_by_fold = verify_imputation_mask(args.raw_dir, mask, FOLDS)
        coverage = build_imputed_coverage_table(canon_by_fold, mask, FOLDS)
        coverage.to_csv(args.out_dir / "imputed_coverage.csv", index=False)
        (args.out_dir / "imputed_coverage.md").write_text(coverage.to_markdown(index=False) + "\n")
        del canon_by_fold

        print("[4/9] Graph canonical filtered grid (source of TFT's pers_ref + intersection base) ...")
        graph_canon = build_graph_canonical_keys(args.raw_dir, mask, FOLDS, keep_imputed=args.keep_imputed)

        print("[5/9] Building the 14-variant (station_id, run_time, horizon) intersection per fold ...")
        tft_keys = build_tft_filtered_keys(args.raw_dir, mask, FOLDS, keep_imputed=args.keep_imputed)
        intersect_keys = build_intersection_keys(graph_canon, tft_keys, FOLDS)
        intersect_sizes = pd.DataFrame([
            {"fold": f, "n_intersection": len(intersect_keys[f]), "n_graph_own_filtered": len(graph_canon[f]),
             "n_tft_base_own_filtered": len(tft_keys[(f, "base")]), "n_tft_hist_own_filtered": len(tft_keys[(f, "hist")]),
             "share_of_graph": len(intersect_keys[f]) / len(graph_canon[f])}
            for f in FOLDS])
        intersect_sizes.to_csv(args.out_dir / "intersection_sizes.csv", index=False)
        print(intersect_sizes.to_string(index=False))
        del tft_keys

        mode = "KEEP-IMPUTED (--keep-imputed, ungefiltert)" if args.keep_imputed else "gefiltert (Standard)"
        print(f"[6/9] Single pass over {len(STDHP_META) * len(FOLDS)} GRAPH raw parquets "
              f"(expected — some may be missing, e.g. ablations not trained yet) [{mode}] ...")
        raw_bundle = scan_raw_parquets(args.raw_dir, mask, keep_imputed=args.keep_imputed,
                                       intersect_keys=intersect_keys)
        _verify_uniform_filtering(raw_bundle["pooled"])

        # Pooled RMSE for the ICON-D2 / Persistence pseudo-models too (from the per-station sums
        # already collected once per fold in ref_station), so the overview table's pooled column
        # is not just blank for the two references.
        ref_pooled = (raw_bundle["ref_station"].groupby(["model", "fold"])
                      .agg(sse=("sse", "sum"), n=("n", "sum")).reset_index())
        ref_pooled["variant"] = "REF"
        raw_bundle["pooled"] = pd.concat([raw_bundle["pooled"], ref_pooled], ignore_index=True)
        if not raw_bundle["unbinned"].empty:
            raw_bundle["unbinned"].to_csv(args.out_dir / "stations_unbinned_gt.csv", index=False)

        # Cross-check MUST run on the UNFILTERED sample ("station_full") -- that is what the
        # stdhp_*.csv files were computed on. This is a data-integrity guard (does evaluate()'s
        # own NaN masking agree with a from-scratch recompute?), independent of the separate
        # imputed-hour filtering decision -- it must keep checking the full sample even though
        # the reported table below now uses the filtered one.
        _crosscheck_csv_vs_raw(station_df_csv, raw_bundle["station_full"])

        print(f"[7/9] Single pass over {len(TFT_META) * len(FOLDS)} TFT raw parquets [{mode}] ...")
        tft_bundle = scan_tft_raw_parquets(args.raw_dir, mask, graph_canon, intersect_keys,
                                           folds=FOLDS, keep_imputed=args.keep_imputed)
        tft_bundle["nwp_ref_check"].to_csv(args.out_dir / "tft_vs_graph_nwp_ref_check.csv", index=False)
        _crosscheck_csv_vs_raw(tft_df_csv, tft_bundle["station_full"], n_multiplier=48)
        del graph_canon
        gc.collect()

        # ---- Merge TFT into the combined bundle the shared figures (bar chart, fold-dispersion,
        # horizon/month/ws-class stratification, skill distribution) iterate over via IMPORTANT_
        # VARIANTS / ALL_META -- these figures are otherwise completely unmodified graph-only code.
        for key in ("horizon", "month", "ws_bin", "station", "station_full"):
            raw_bundle[key] = pd.concat([raw_bundle[key], tft_bundle[key]], ignore_index=True)

        # station_metrics: Stage 2, own-sample imputed-hour-filtered (every variant on its own
        # sample -- NOT comparable point-for-point between TFT and graph, see module docstring).
        # station_metrics_unfiltered: Stage 1, ALWAYS the full sample.
        station_metrics = build_station_metrics(raw_bundle["station"], raw_bundle["ref_station"])
        station_metrics_unfiltered = build_station_metrics(raw_bundle["station_full"], raw_bundle["ref_station_full"])

        # Stage 3: the 14-variant row intersection -- what the paper reports by default. Built
        # from the SAME recipe (build_station_metrics()) as the other two stages, just fed the
        # intersection-restricted buckets instead.
        station_intersect_all = pd.concat([raw_bundle["station_intersect"], tft_bundle["station_intersect"]],
                                          ignore_index=True)
        station_metrics_intersect = build_station_metrics(station_intersect_all, raw_bundle["ref_station_intersect"])

        # Task D verification #3: after intersection, every one of the 14 variants (+ REF) must
        # have EXACTLY the same total row count per fold -- abort loudly otherwise.
        n_per_variant_fold = station_metrics_intersect.groupby(["model", "variant", "fold"])["n_samples"].sum()
        n_uniform = n_per_variant_fold.groupby("fold").nunique()
        bad_fold = n_uniform[n_uniform != 1]
        if not bad_fold.empty:
            detail = n_per_variant_fold[n_per_variant_fold.index.get_level_values("fold").isin(bad_fold.index)]
            sys.exit(f"[FATAL] Nach Schnittmengenbildung ist die Zeilenzahl je Fold NICHT ueber alle "
                     f"14 Varianten (+REF) identisch (Fold(s) {list(bad_fold.index)}):\n{detail}\n"
                     f"Abbruch statt verzerrter Schnittmengen-Vergleiche.")
        print(f"[OK] Nach Schnittmengenbildung: identische Zeilenzahl je Fold ueber alle 12 "
              f"Varianten + REF ({n_per_variant_fold.groupby('fold').first().to_dict()}).")

        # Sample-size table across all 3 stages, task D verification #3 in full: written to disk
        # for the report, not just checked in-memory.
        def _n_table(df, label):
            t = df.groupby(["model", "variant", "fold"])["n_samples"].sum().reset_index()
            t["stage"] = label
            return t
        n_table = pd.concat([_n_table(station_metrics_unfiltered, "1_full"),
                             _n_table(station_metrics, "2_own_filtered"),
                             _n_table(station_metrics_intersect, "3_intersect")], ignore_index=True)
        n_table.to_csv(args.out_dir / "sample_size_by_stage.csv", index=False)

        # Bias check (task B): does restricting the GRAPH models to the intersection move their
        # RMSE noticeably relative to their own (Stage 2) sample? A large shift means the ~rows
        # TFT is missing are NOT missing at random.
        bias_check = (station_metrics_intersect[station_metrics_intersect["variant"] != "REF"]
                     .groupby(["model", "variant"])["rmse"].mean()
                     .rename("rmse_intersect").reset_index()
                     .merge(station_metrics[station_metrics["variant"] != "REF"]
                            .groupby(["model", "variant"])["rmse"].mean().rename("rmse_own_filtered").reset_index(),
                            on=["model", "variant"]))
        bias_check["delta"] = bias_check["rmse_intersect"] - bias_check["rmse_own_filtered"]
        bias_check.to_csv(args.out_dir / "intersection_bias_check.csv", index=False)
        print("[INFO] Schnittmengen-Verzerrung (Graphenmodelle, RMSE Schnittmenge - RMSE eigene "
              f"gefilterte Stichprobe):\n{bias_check.to_string(index=False)}")
    else:
        print("[2/9] --skip-raw: keine Imputations-Filterung, kein TFT, keine Schnittmenge moeglich.")
        warnings.warn("--skip-raw: Uebersichtstabelle basiert auf der UNGEFILTERTEN Graphen-stdhp_*.csv-"
                      "Stichprobe (Imputations-Stunden inklusive, kein TFT) -- nicht auf der im Auftrag "
                      "entschiedenen Stichprobe. Kein ICON-D2/Persistence-Vergleich moeglich.")
        station_metrics = station_df_csv

    reported = station_metrics_intersect if (args.intersect_only and not args.skip_raw) else station_metrics
    stage_label = "Stage3-intersect" if (args.intersect_only and not args.skip_raw) else "Stage2-ownfiltered"
    print(f"[8/9] Building overview table (primary = {stage_label}) ...")
    overview = build_overview_table(reported, raw_bundle["pooled"] if raw_bundle else None,
                                     station_df_unfiltered=station_metrics_unfiltered,
                                     station_df_ownfiltered=station_metrics)
    overview.to_csv(args.out_dir / "overview_table.csv", index=False)
    (args.out_dir / "overview_table.md").write_text(overview_markdown(overview, primary_label=stage_label) + "\n")

    fold_detail_cols = ["model", "variant"] + [f"rmse_fold{f}" for f in FOLDS] + \
                        [f"mae_fold{f}" for f in FOLDS] + [f"r2_fold{f}" for f in FOLDS]
    overview[fold_detail_cols].to_csv(args.out_dir / "fold_detail_table.csv", index=False)

    print("[9/9] Stations below ICON-D2 (skill_nwp <= 0) ...")
    below = stations_below_icon_table(reported)
    below.to_csv(args.out_dir / "stations_below_icon.csv", index=False)
    (args.out_dir / "stations_below_icon.md").write_text(below.to_markdown(index=False) + "\n")

    print("Figures & paired-diff analysis ...")
    fig_bar_rmse(overview, args.out_dir, formats)
    fig_fold_dispersion(reported, args.out_dir, formats)
    fig_paired_diff_boxplots(reported, args.out_dir, formats)

    diff_stats_dcrnn = paired_diff_analysis(reported, n_boot=args.n_boot)
    diff_stats_cross = paired_diff_generic(reported, CROSS_MODEL_PAIRS, n_boot=args.n_boot)
    diff_stats = pd.concat([diff_stats_dcrnn, diff_stats_cross], ignore_index=True)
    _COL_ORDER = ["group", "pair", "fold", "n_stations", "mean_diff_rmse", "ci95_lo", "ci95_hi",
                  "ci95_lo_stratified", "ci95_hi_stratified", "wilcoxon_p", "frac_first_better"]
    diff_stats = diff_stats[_COL_ORDER]
    diff_stats.to_csv(args.out_dir / "paired_diff_stats.csv", index=False)
    (args.out_dir / "paired_diff_stats.md").write_text(diff_stats.to_markdown(index=False) + "\n")

    fig_skill_distribution(reported, args.out_dir, formats)

    if raw_bundle is not None:
        fig_horizon_error(raw_bundle, args.out_dir, formats)
        fig_error_by_ws_class(raw_bundle, args.out_dir, formats)
        fig_error_by_month(raw_bundle, args.out_dir, formats)
        fig_station_scatter(raw_bundle, args.out_dir, formats)
    else:
        print("[skip] horizon/month/ws-class/scatter figures need raw parquets (--skip-raw was set).")

    (args.out_dir / "README.md").write_text(README_TEMPLATE)

    print(f"\nDone. Output in {args.out_dir}")
    print(f"\n=== Overview (sorted by RMSE, primary = {stage_label}) ===")
    print(overview_markdown(overview, primary_label=stage_label))
    if intersect_sizes is not None:
        print("\n=== Intersection sizes ===")
        print(intersect_sizes.to_string(index=False))
    print("\n=== Paired diff (pooled rows only) ===")
    print(diff_stats[diff_stats["fold"] == "pooled"].to_markdown(index=False))


if __name__ == "__main__":
    main()
