#!/usr/bin/env python3
"""
make_stdhp_figures.py — headless static figures & tables for the "stdhp" dry run
(10 model variants x 3 spatial folds, fixed standard hyperparameters instead of
per-fold HPO-best params).

Run with e.g.:
    python geostatistics/stdrun/make_stdhp_figures.py --out-dir geostatistics/figures/stdhp
    python geostatistics/stdrun/make_stdhp_figures.py --skip-raw   # tables + CSV-only figures

AGGREGATION CONVENTION — read this before quoting a number from this script's output:
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
  - Horizon / month / wind-speed-class curves are aggregated the same way: per fold from raw
    residual sums, then averaged across the 3 folds for the "average" panel (fold counts equally,
    not sample-weighted).
  - ICON-D2 and Persistence are NOT separate model runs — there is no stdhp_*.csv for them. They
    are pseudo-models derived from the nwp_ref / pers_ref columns of the raw prediction parquets,
    which are per-construction on the same station set / time window / horizon grid as whichever
    model run they were saved alongside. Before using them the script verifies that nwp_ref /
    pers_ref / gt are numerically identical across several variants of the same fold; if that
    check fails it aborts loudly (SystemExit) instead of silently mixing disagreeing baselines.
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

FOLDS = [0, 1, 2]
N_STATIONS_EXPECTED = 51

# filename-prefix (without _fold{n}.csv/_raw.parquet) -> (model, variant, ablation_letter)
STDHP_META = {
    "stdhp_dcrnn_wind_dcrnn_base":     ("DCRNN",   "BASE",         "BASE"),
    "stdhp_dcrnn_wind_dcrnn":          ("DCRNN",   "GRID",         "A"),
    "stdhp_dcrnn_wind_dcrnn_nomeas":   ("DCRNN",   "GRID-NOMEAS",  "B"),
    "stdhp_dcrnn_wind_dcrnn_nograph":  ("DCRNN",   "GRID-NOGRAPH", "C"),
    "stdhp_dcrnn_wind_dcrnn_nwp_hist": ("DCRNN",   "GRID+HIST",    None),
    "stdhp_mtgnn_wind_mtgnn":          ("MTGNN",   "BASE",         None),
    "stdhp_mtgnn_wind_mtgnn_nwp":      ("MTGNN",   "GRID",         None),
    "stdhp_mtgnn_wind_mtgnn_nwp_hist": ("MTGNN",   "GRID+HIST",    None),
    "stdhp_wavenet_wind_wavenet":      ("WaveNet", "BASE",         None),
    "stdhp_wavenet_wind_wavenet_nwp":  ("WaveNet", "GRID",         None),
}
TRANSDUCTIVE_VARIANTS = {("DCRNN", "GRID+HIST"), ("MTGNN", "GRID+HIST")}
ABLATION_PAIRS = [("A", "B"), ("B", "C"), ("A", "C"), ("A", "BASE")]
ABLATION_LETTER_LABEL = {"A": "GRID", "B": "GRID-NOMEAS", "C": "GRID-NOGRAPH", "BASE": "BASE"}

# "wichtigste Varianten" for the WS-class / month stratification (task item 7)
IMPORTANT_VARIANTS = [("DCRNN", "GRID"), ("DCRNN", "BASE"), ("MTGNN", "GRID"), ("WaveNet", "GRID")]
IMPORTANT_REF_MODELS = ["ICON-D2", "Persistence"]

MODEL_COLORS = {
    "DCRNN": "steelblue", "MTGNN": "darkorange", "WaveNet": "forestgreen",
    "ICON-D2": "#888888", "Persistence": "#c44e52",
}
FOLD_COLORS = {0: "#1b9e77", 1: "#d95f02", 2: "#7570b3"}

WS_BINS = list(range(0, 22, 2))
WS_LABELS = [f"[{WS_BINS[i]},{WS_BINS[i + 1]})" for i in range(len(WS_BINS) - 1)] + ["[20,inf)"]

RAW_COLUMNS = ["station_id", "valid_time", "horizon", "pred", "gt", "nwp_ref", "pers_ref"]


def norm_station(x) -> str:
    """CSV station_id is int64 (161), parquet station_id is a zero-padded string ('00161').
    Normalize both to a 5-digit zero-padded string so they can be joined/compared."""
    return str(x).strip().zfill(5)


def variant_label(model: str, variant: str) -> str:
    return f"{model} {variant}"


def is_transductive(model: str, variant: str) -> bool:
    return (model, variant) in TRANSDUCTIVE_VARIANTS


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


def _canonical_prefixes_for_fold(fold: int, raw_dir: Path) -> list[str]:
    return sorted(p for p in STDHP_META if (raw_dir / f"{p}_fold{fold}_raw.parquet").exists())


def verify_and_derive_references(raw_dir: Path, folds=FOLDS, atol: float = 1e-4,
                                  n_check_files: int = 3) -> pd.DataFrame:
    """
    For each fold: verify that gt / nwp_ref / pers_ref are numerically identical (within `atol`)
    across several stdhp raw parquets of that fold — they are supposed to be, since they depend
    only on the (station, valid_time, horizon) grid, not on the model that was trained. Then
    derive per-station ICON-D2 and Persistence metrics from one canonical file per fold.

    Aborts with SystemExit (clear message) if the identity assumption is violated — per task
    instructions, do NOT silently average over disagreeing baselines.
    """
    ref_rows = []
    for fold in folds:
        prefixes = _canonical_prefixes_for_fold(fold, raw_dir)
        if not prefixes:
            sys.exit(f"[FATAL] Keine Parquets fuer fold {fold} in {raw_dir}.")
        canon_path = raw_dir / f"{prefixes[0]}_fold{fold}_raw.parquet"
        canon = pd.read_parquet(canon_path, columns=["station_id", "valid_time", "horizon", "gt", "nwp_ref", "pers_ref"])
        canon["station_id"] = canon["station_id"].map(norm_station)

        check_prefixes = prefixes[1:1 + n_check_files]
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

        for ref_col, model_name in [("nwp_ref", "ICON-D2"), ("pers_ref", "Persistence")]:
            err = (canon[ref_col] - canon["gt"]).values
            tmp = pd.DataFrame({"station_id": canon["station_id"], "err": err, "gt": canon["gt"].values})

            def _agg(g):
                sse = float((g["err"] ** 2).sum())
                n = len(g)
                mae = float(g["err"].abs().mean())
                rmse = float(np.sqrt(sse / n)) if n else np.nan
                ss_tot = float(((g["gt"] - g["gt"].mean()) ** 2).sum())
                r2 = 1.0 - sse / ss_tot if ss_tot > 0 else np.nan
                return pd.Series({"rmse": rmse, "mae": mae, "r2": r2, "n_samples": n})

            per_station = tmp.groupby("station_id").apply(_agg).reset_index()
            per_station["model"] = model_name
            per_station["variant"] = "REF"
            per_station["ablation_letter"] = None
            per_station["fold"] = fold
            per_station["run"] = "stdhp"
            ref_rows.append(per_station)
        del canon
        gc.collect()

    ref_df = pd.concat(ref_rows, ignore_index=True)
    wide = ref_df.pivot_table(index=["fold", "station_id"], columns="model", values="rmse").reset_index()
    wide = wide.rename(columns={"ICON-D2": "_rmse_icon", "Persistence": "_rmse_pers"})
    ref_df = ref_df.merge(wide[["fold", "station_id", "_rmse_icon", "_rmse_pers"]], on=["fold", "station_id"], how="left")
    # skill        := skill vs. Persistence (0 for the Persistence pseudo-model itself)
    # skill_nwp    := skill vs. ICON-D2     (0 for the ICON-D2 pseudo-model itself)
    ref_df["skill"] = np.where(ref_df["model"] == "Persistence", 0.0, 1.0 - ref_df["_rmse_icon"] / ref_df["_rmse_pers"])
    ref_df["skill_nwp"] = np.where(ref_df["model"] == "ICON-D2", 0.0, 1.0 - ref_df["_rmse_pers"] / ref_df["_rmse_icon"])
    ref_df = ref_df.drop(columns=["_rmse_icon", "_rmse_pers"])
    return ref_df


def scan_raw_parquets(raw_dir: Path, folds=FOLDS):
    """
    Single pass over all 30 stdhp raw parquets (~3.57M rows / 8 columns each). Per file: read
    only RAW_COLUMNS, immediately reduce to a handful of aggregate rows (pooled residuals,
    per-horizon, per-month, per-wind-speed-class, per-station), then discard the raw frame
    before moving to the next file. At no point are two full parquet files held in memory
    simultaneously. Reference (ICON-D2 / Persistence) aggregates only need to be computed once
    per fold (from the first file processed for that fold) since they don't depend on the model.

    Returns a dict of small aggregate DataFrames (sums, not final metrics — final RMSE/MAE is
    computed at plot time via sqrt(sse/n) etc., so that fold-averaging happens the same way
    everywhere: aggregate-then-average-across-folds, never a single pooled-across-folds number).
    """
    pooled_rows = []
    bucket_rows = {"horizon": [], "month": [], "ws_bin": []}
    station_rows = []
    ref_bucket_rows = {"horizon": [], "month": [], "ws_bin": []}
    ref_station_rows = []
    seen_ref_fold = set()

    files = [(prefix, meta, fold)
             for prefix, meta in sorted(STDHP_META.items())
             for fold in folds]

    for i, (prefix, (model, variant, letter), fold) in enumerate(files):
        path = raw_dir / f"{prefix}_fold{fold}_raw.parquet"
        if not path.exists():
            warnings.warn(f"Fehlt: {path}")
            continue
        df = pd.read_parquet(path, columns=RAW_COLUMNS)
        df["station_id"] = df["station_id"].map(norm_station)
        df["month"] = pd.to_datetime(df["valid_time"]).dt.month
        df["ws_bin"] = pd.cut(df["gt"], bins=WS_BINS + [np.inf], labels=WS_LABELS, right=False)

        err = (df["pred"] - df["gt"]).to_numpy()
        sq, ab = err ** 2, np.abs(err)

        pooled_rows.append({"model": model, "variant": variant, "fold": fold,
                             "sse": float(sq.sum()), "sae": float(ab.sum()), "n": int(len(df))})

        for key in ("horizon", "month", "ws_bin"):
            g = pd.DataFrame({key: df[key].values, "sq": sq, "ab": ab}).groupby(key, observed=True).agg(
                sse=("sq", "sum"), sae=("ab", "sum"), n=("sq", "size")).reset_index()
            g["model"], g["variant"], g["fold"] = model, variant, fold
            bucket_rows[key].append(g)

        gst = pd.DataFrame({"station_id": df["station_id"].values, "sq": sq, "ab": ab}).groupby(
            "station_id").agg(sse=("sq", "sum"), sae=("ab", "sum"), n=("sq", "size")).reset_index()
        gst["model"], gst["variant"], gst["fold"] = model, variant, fold
        station_rows.append(gst)

        if fold not in seen_ref_fold:
            seen_ref_fold.add(fold)
            for ref_name, ref_col in [("ICON-D2", "nwp_ref"), ("Persistence", "pers_ref")]:
                rerr = (df[ref_col] - df["gt"]).to_numpy()
                rsq, rab = rerr ** 2, np.abs(rerr)
                for key in ("horizon", "month", "ws_bin"):
                    g = pd.DataFrame({key: df[key].values, "sq": rsq, "ab": rab}).groupby(
                        key, observed=True).agg(sse=("sq", "sum"), sae=("ab", "sum"), n=("sq", "size")).reset_index()
                    g["model"], g["fold"] = ref_name, fold
                    ref_bucket_rows[key].append(g)
                gst2 = pd.DataFrame({"station_id": df["station_id"].values, "sq": rsq, "ab": rab}).groupby(
                    "station_id").agg(sse=("sq", "sum"), sae=("ab", "sum"), n=("sq", "size")).reset_index()
                gst2["model"], gst2["fold"] = ref_name, fold
                ref_station_rows.append(gst2)

        del df, err, sq, ab
        gc.collect()
        print(f"  [scan {i + 1}/{len(files)}] {prefix} fold{fold} done", flush=True)

    def _cat(rows):
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    return {
        "pooled": pd.DataFrame(pooled_rows),
        "horizon": _cat(bucket_rows["horizon"]),
        "month": _cat(bucket_rows["month"]),
        "ws_bin": _cat(bucket_rows["ws_bin"]),
        "station": _cat(station_rows),
        "ref_horizon": _cat(ref_bucket_rows["horizon"]),
        "ref_month": _cat(ref_bucket_rows["month"]),
        "ref_ws_bin": _cat(ref_bucket_rows["ws_bin"]),
        "ref_station": _cat(ref_station_rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2) Overview table
# ─────────────────────────────────────────────────────────────────────────────
def build_overview_table(station_df: pd.DataFrame, pooled_df: pd.DataFrame | None) -> pd.DataFrame:
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

    return out.sort_values("rmse_mean").reset_index(drop=True)


def overview_markdown(overview: pd.DataFrame) -> str:
    def fmt(mean, sd):
        return f"{mean:.4f} +/- {sd:.4f}" if pd.notna(mean) else "--"

    disp = pd.DataFrame({
        "Model": overview["model"],
        "Variant": overview["variant"],
        "RMSE (mean+/-sd)": [fmt(m, s) for m, s in zip(overview["rmse_mean"], overview["rmse_sd"])],
        "MAE (mean+/-sd)": [fmt(m, s) for m, s in zip(overview["mae_mean"], overview["mae_sd"])],
        "R2 (mean+/-sd)": [fmt(m, s) for m, s in zip(overview["r2_mean"], overview["r2_sd"])],
        "Skill vs Persistence": [fmt(m, s) for m, s in zip(overview["skill_mean"], overview["skill_sd"])],
        "Skill vs ICON-D2": [fmt(m, s) for m, s in zip(overview["skill_nwp_mean"], overview["skill_nwp_sd"])],
    })
    if "pooled_rmse_mean" in overview.columns:
        disp["RMSE pooled (mean+/-sd)"] = [fmt(m, s) for m, s in
                                            zip(overview["pooled_rmse_mean"], overview["pooled_rmse_sd"])]
    disp["Transductive"] = overview["transductive"].map({True: "yes (sees target history)", False: ""})
    return disp.to_markdown(index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Bar chart: RMSE of all variants + ICON-D2 + Persistence, sorted, error bars = SD over folds
# ─────────────────────────────────────────────────────────────────────────────
def fig_bar_rmse(overview: pd.DataFrame, out_dir: Path, formats):
    df = overview.copy()
    df["label"] = df["model"] + " " + df["variant"]
    df = df.sort_values("rmse_mean")

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.9), 6))
    x = np.arange(len(df))
    colors = ["#c44e52" if t else "steelblue" for t in df["transductive"]]
    hatches = ["//" if t else "" for t in df["transductive"]]
    bars = ax.bar(x, df["rmse_mean"], yerr=df["rmse_sd"], capsize=4, color=colors,
                  edgecolor="white", linewidth=0.6, error_kw={"elinewidth": 1.3, "alpha": 0.75})
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("RMSE (m/s)", fontsize=12)
    ax.set_title("stdhp dry run: RMSE by variant (mean +/- SD over 3 spatial folds)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="steelblue", label="Inductive (target station unseen in training)"),
        Patch(facecolor="#c44e52", hatch="//", label="GRID+HIST — transductive, sees target-station\n"
                                                       "history — NOT an inductive comparison point"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"01_bar_rmse_all_variants.{fmt}", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Fold-dispersion parallel-coordinates plot for the 4 DCRNN ablations
# ─────────────────────────────────────────────────────────────────────────────
def fig_fold_dispersion(station_df: pd.DataFrame, out_dir: Path, formats):
    dcrnn = station_df[(station_df["model"] == "DCRNN") &
                        (station_df["variant"].isin(["GRID", "GRID-NOMEAS", "GRID-NOGRAPH", "BASE"]))]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    order = ["GRID", "GRID-NOMEAS", "GRID-NOGRAPH", "BASE"]
    colors = {"GRID": "#1b9e77", "GRID-NOMEAS": "#d95f02", "GRID-NOGRAPH": "#7570b3", "BASE": "#888888"}
    for variant in order:
        sub = dcrnn[dcrnn["variant"] == variant]
        means = sub.groupby("fold")["rmse"].mean().reindex(FOLDS)
        ax.plot(FOLDS, means.values, marker="o", linewidth=2.2, markersize=8,
                color=colors[variant], label=variant)
        for f, v in zip(FOLDS, means.values):
            ax.annotate(f"{v:.3f}", (f, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color=colors[variant])
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
    dcrnn = station_df[(station_df["model"] == "DCRNN") &
                        (station_df["variant"].isin(["GRID", "GRID-NOMEAS", "GRID-NOGRAPH", "BASE"]))].copy()
    letter_of_variant = {"GRID": "A", "GRID-NOMEAS": "B", "GRID-NOGRAPH": "C", "BASE": "BASE"}
    dcrnn["letter"] = dcrnn["variant"].map(letter_of_variant)
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
            results.append(_diff_stats(f"{a}-{b} ({ABLATION_LETTER_LABEL[a]} minus {ABLATION_LETTER_LABEL[b]})",
                                        str(fold), d, rng, n_boot))
        if diffs_by_fold:
            all_d = np.concatenate(list(diffs_by_fold.values()))
            results.append(_diff_stats(f"{a}-{b} ({ABLATION_LETTER_LABEL[a]} minus {ABLATION_LETTER_LABEL[b]})",
                                        "pooled", all_d, rng, n_boot))
    return pd.DataFrame(results)


def _diff_stats(pair_label, fold_label, d, rng, n_boot):
    if np.any(d != 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = stats.wilcoxon(d)
    else:
        p = np.nan
    boot_idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot_means = d[boot_idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "pair": pair_label, "fold": fold_label, "n_stations": len(d),
        "mean_diff_rmse": float(d.mean()), "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        "wilcoxon_p": float(p) if pd.notna(p) else np.nan,
        "frac_first_better": float((d < 0).mean()),
    }


def fig_paired_diff_boxplots(station_df: pd.DataFrame, out_dir: Path, formats):
    dcrnn = station_df[(station_df["model"] == "DCRNN") &
                        (station_df["variant"].isin(["GRID", "GRID-NOMEAS", "GRID-NOGRAPH", "BASE"]))].copy()
    letter_of_variant = {"GRID": "A", "GRID-NOMEAS": "B", "GRID-NOGRAPH": "C", "BASE": "BASE"}
    dcrnn["letter"] = dcrnn["variant"].map(letter_of_variant)
    pivot = dcrnn.pivot_table(index=["fold", "station_id"], columns="letter", values="rmse")

    fig, axes = plt.subplots(1, len(ABLATION_PAIRS), figsize=(4.2 * len(ABLATION_PAIRS), 5), sharey=True)
    for ax, (a, b) in zip(axes, ABLATION_PAIRS):
        sub = pivot[[a, b]].dropna()
        box_data = [sub.xs(f, level="fold")[a] - sub.xs(f, level="fold")[b]
                    for f in FOLDS if f in sub.index.get_level_values("fold")]
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax.boxplot(box_data, labels=[f"Fold {f}" for f in FOLDS], showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#a6cee3", alpha=0.7))
        for i, d in enumerate(box_data, start=1):
            frac = float((d < 0).mean())
            ax.text(i, ax.get_ylim()[1] * 0.9, f"{frac * 100:.0f}% improved", ha="center", fontsize=8)
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
        for prefix, (model, variant, letter) in STDHP_META.items():
            piv = _rmse_by_key(horizon_df, model, variant, "horizon")
            if piv is None:
                continue
            y = piv.mean(axis=1) if fold is None else piv.get(fold)
            if y is None:
                continue
            ax.plot(y.index.to_numpy(), y.to_numpy(), label=f"{model} {variant}", linewidth=1.6,
                    color=MODEL_COLORS.get(model), alpha=0.5 if variant not in ("GRID", "BASE") else 0.95,
                    linestyle="--" if is_transductive(model, variant) else "-")
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
    fig.suptitle("RMSE by forecast horizon — all stdhp variants + ICON-D2 + Persistence", fontsize=13, y=1.25)
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
        model = name.split(" ")[0]
        ax.plot(range(len(WS_LABELS)), y.values, marker="o", label=name,
                color=MODEL_COLORS.get(model, None), linewidth=2)
    ax.set_xticks(range(len(WS_LABELS)))
    ax.set_xticklabels(WS_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Wind speed class (m/s), ground truth, left-inclusive")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("RMSE by wind-speed class — key variants (averaged over 3 folds)")
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
        model = name.split(" ")[0]
        ax.plot(months, y.values, marker="o", label=name, color=MODEL_COLORS.get(model, None), linewidth=2)
    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("RMSE (m/s)")
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
    variants = list(STDHP_META.values())
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
  key variants (DCRNN GRID/BASE, MTGNN GRID, WaveNet GRID, ICON-D2, Persistence), averaged over
  folds.
- `06_error_by_month.*` — RMSE by calendar month for the same key variants, averaged over folds
  (months are pooled across the 3 folds' identical Aug 2024-Aug 2025 val window).
- `07_scatter_dcrnn_grid_vs_icon.*` / `08_scatter_mtgnn_grid_vs_icon.*` — per-station RMSE of the
  model vs. ICON-D2, colored by fold, with the y=x diagonal; points below the diagonal are
  stations where the model beats ICON-D2.
- `09_skill_nwp_distribution.*` — histogram + ECDF of skill_nwp (skill vs. ICON-D2) across all
  153 (station, fold) pairs, one line/curve per variant, to show whether skill is broad-based or
  carried by a few stations.

## Tables

- `overview_table.md` / `.csv` — per-variant RMSE/MAE/R2/skill vs. Persistence/skill vs. ICON-D2,
  mean +/- SD over the 3 folds, plus per-fold values and the pooled-RMSE variant.
- `paired_diff_stats.md` / `.csv` — mean RMSE difference, 95% bootstrap CI, Wilcoxon signed-rank
  p-value and share of stations improved, per fold and pooled, for A-B / B-C / A-C / A-BASE.
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
    ap.add_argument("--formats", type=str, default="png,pdf")
    ap.add_argument("--skip-raw", action="store_true",
                     help="Skip everything that needs the 30 raw prediction parquets (pooled RMSE, "
                          "ICON-D2/Persistence reference derivation, horizon/month/ws-class/scatter figures).")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading {len(STDHP_META) * len(FOLDS)} stdhp CSVs from {args.results_dir} ...")
    station_df = load_station_csvs(args.results_dir)
    print(f"      -> {len(station_df):,} station rows, "
          f"{station_df.groupby(['model', 'variant']).ngroups} variants.")

    ref_df = None
    raw_bundle = None
    if not args.skip_raw:
        print("[2/6] Verifying nwp_ref/pers_ref identity across variants & deriving ICON-D2/Persistence ...")
        ref_df = verify_and_derive_references(args.raw_dir)
        station_df_full = pd.concat([station_df, ref_df], ignore_index=True)

        print("[3/6] Single pass over 30 raw parquets (pooled / horizon / month / ws-class / station) ...")
        raw_bundle = scan_raw_parquets(args.raw_dir)
        # Pooled RMSE for the ICON-D2 / Persistence pseudo-models too (from the per-station sums
        # already collected once per fold in ref_station), so the overview table's pooled column
        # is not just blank for the two references.
        ref_pooled = (raw_bundle["ref_station"].groupby(["model", "fold"])
                      .agg(sse=("sse", "sum"), n=("n", "sum")).reset_index())
        ref_pooled["variant"] = "REF"
        raw_bundle["pooled"] = pd.concat([raw_bundle["pooled"], ref_pooled], ignore_index=True)
    else:
        print("[2/6] --skip-raw: no ICON-D2/Persistence reference rows, no pooled RMSE, no raw-based figures.")
        station_df_full = station_df

    print("[4/6] Building overview table ...")
    overview = build_overview_table(station_df_full, raw_bundle["pooled"] if raw_bundle else None)
    overview.to_csv(args.out_dir / "overview_table.csv", index=False)
    (args.out_dir / "overview_table.md").write_text(overview_markdown(overview) + "\n")

    fold_detail_cols = ["model", "variant"] + [f"rmse_fold{f}" for f in FOLDS] + \
                        [f"mae_fold{f}" for f in FOLDS] + [f"r2_fold{f}" for f in FOLDS]
    overview[fold_detail_cols].to_csv(args.out_dir / "fold_detail_table.csv", index=False)

    print("[5/6] Stations below ICON-D2 (skill_nwp <= 0) ...")
    below = stations_below_icon_table(station_df)
    below.to_csv(args.out_dir / "stations_below_icon.csv", index=False)
    (args.out_dir / "stations_below_icon.md").write_text(below.to_markdown(index=False) + "\n")

    print("[6/6] Figures & paired-diff analysis ...")
    fig_bar_rmse(overview, args.out_dir, formats)
    fig_fold_dispersion(station_df, args.out_dir, formats)
    fig_paired_diff_boxplots(station_df, args.out_dir, formats)

    diff_stats = paired_diff_analysis(station_df, n_boot=args.n_boot)
    diff_stats.to_csv(args.out_dir / "paired_diff_stats.csv", index=False)
    (args.out_dir / "paired_diff_stats.md").write_text(diff_stats.to_markdown(index=False) + "\n")

    fig_skill_distribution(station_df, args.out_dir, formats)

    if raw_bundle is not None:
        fig_horizon_error(raw_bundle, args.out_dir, formats)
        fig_error_by_ws_class(raw_bundle, args.out_dir, formats)
        fig_error_by_month(raw_bundle, args.out_dir, formats)
        fig_station_scatter(raw_bundle, args.out_dir, formats)
    else:
        print("[skip] horizon/month/ws-class/scatter figures need raw parquets (--skip-raw was set).")

    (args.out_dir / "README.md").write_text(README_TEMPLATE)

    print(f"\nDone. Output in {args.out_dir}")
    print("\n=== Overview (sorted by RMSE) ===")
    print(overview_markdown(overview))
    print("\n=== Paired diff (pooled rows only) ===")
    print(diff_stats[diff_stats["fold"] == "pooled"].to_markdown(index=False))


if __name__ == "__main__":
    main()
