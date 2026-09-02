#!/usr/bin/env python3
"""
Finale Testauswertung (--test-mode), Testjahr 2025-08-01 .. 2026-07-31.

Aufbau (docs/handoff_testmode.md + Nutzerauftrag vom 2026-09-02):
  * 3 Arme ohne NWP-Historie  — EIN Modell, Training auf allem vor 2025-08-01,
    zero-shot auf den 50 test_files ueber das ganze Testjahr.
  * 2 HIST-Arme               — Expanding-Window-Retraining alle 4 Monate:
        step1  Training < 2025-08-01  →  Test Aug-Nov 2025
        step2  Training < 2025-12-01  →  Test Dez 2025 - Mrz 2026
        step3  Training < 2026-04-01  →  Test Apr-Jul 2026
    Die drei Roh-Parquets sind ueber die Laufzeit (run_time) disjunkt und
    werden zum vollen Testjahr zusammengelegt.

Filterung wie docs/evaluation_results.md §14-18: Zielstunden ohne Rohmessung
(also imputierte) werden ausgeschlossen, dann RMSE je Station, dann
Stationsmittel. Signifikanz gepaart ueber die 50 Teststationen (Wilcoxon,
Holm-korrigiert).

Zwei Tabellenvarianten, weil die Arme unterschiedlich viele Run-Paare
behalten (ECMWF-/Grid-NaN-Ausschluss haengt am Arm):
  * "eigen"   — je Arm auf seinen eigenen Zeilen (Konvention der bisherigen
                Auswertungen, vergleichbar mit §14-18)
  * "gepaart" — auf dem Schnitt aller Arme ueber (station_id, run_time, horizon)
Weichen beide stark voneinander ab, ist die Abdeckung das Problem, nicht das
Modell — deshalb stehen beide nebeneinander.

Ausgaben: data/test_results/testmode_*.csv, figures/testmode/*.png|pdf
"""
import sys
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geostatistics.stdrun.make_stdhp_figures import (  # noqa: E402
    norm_station, build_imputation_mask, _lookup_imputed,
)

# Auf l1 liegen die Rohmessungen lokal, auf l2/ws ueber den NFS-Mount.
_CANDIDATES = [Path("/mnt/nvme1/synthetic/raw/wind"),
               Path("/mnt/lambda1/nvme1/synthetic/raw/wind")]
STATION_RAW_DIR = next((c for c in _CANDIDATES if c.is_dir()), _CANDIDATES[-1])

RAW = Path("data/raw_preds")
OUT = Path("data/test_results")
FIGS = Path("figures/testmode")

# arm -> (Anzeigename, Liste der Roh-Parquet-Stems)
ARMS = {
    "dcrnn_nwp_hist": ("DCRNN GRID+HIST", [f"testmode_dcrnn_nwp_hist_s{s}" for s in (1, 2, 3)]),
    "mtgnn_nwp_hist": ("MTGNN GRID+HIST", [f"testmode_mtgnn_nwp_hist_s{s}" for s in (1, 2, 3)]),
    "mtgnn_nwp":      ("MTGNN GRID",      ["testmode_mtgnn_nwp"]),
    "dcrnn":          ("DCRNN GRID",      ["testmode_dcrnn"]),
    "dcrnn_idw_alt":  ("DCRNN IDW (D')",  ["testmode_dcrnn_idw_alt"]),
}
RETRAINED = {"DCRNN GRID+HIST", "MTGNN GRID+HIST"}
COLORS = {"DCRNN GRID+HIST": "#1f77b4", "MTGNN GRID+HIST": "#d62728",
          "MTGNN GRID": "#ff7f0e", "DCRNN GRID": "#2ca02c",
          "DCRNN IDW (D')": "#9467bd", "ICON-D2": "#555555", "Persistenz": "#999999"}
COLS = ["station_id", "run_time", "valid_time", "horizon", "pred", "gt", "nwp_ref", "pers_ref"]

# Grenzen der drei Retrain-Fenster, angewendet auf run_time.
CHUNKS = [("Aug–Nov 2025", "2025-08-01", "2025-12-01"),
          ("Dez 2025–Mär 2026", "2025-12-01", "2026-04-01"),
          ("Apr–Jul 2026", "2026-04-01", "2026-08-01")]


def grouped_rmse(df, by, pred_col="pred", obs_col="gt"):
    """RMSE je Gruppe ohne groupby.apply — laeuft auch auf pandas < 2.2."""
    e2 = (df[pred_col].to_numpy() - df[obs_col].to_numpy()) ** 2
    key = df[by] if isinstance(by, str) else by
    return np.sqrt(pd.Series(e2, index=df.index).groupby(key, observed=True).mean())


def chunk_of(run_time: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=run_time.index, dtype="object")
    for name, lo, hi in CHUNKS:
        sel = (run_time >= pd.Timestamp(lo, tz="UTC")) & (run_time < pd.Timestamp(hi, tz="UTC"))
        out[sel] = name
    return out


def load_arm(arm, mask):
    """Alle Roh-Parquets eines Arms laden, filtern, zusammenlegen."""
    label, stems = ARMS[arm]
    parts, missing = [], []
    for stem in stems:
        p = RAW / f"{stem}_raw.parquet"
        if not p.exists():
            missing.append(stem)
            continue
        d = pd.read_parquet(p, columns=COLS)
        d["chunk_src"] = stem
        parts.append(d)
    if not parts:
        return None, stems
    df = pd.concat(parts, ignore_index=True)
    df["station_id"] = df["station_id"].map(norm_station)
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    n_before = len(df)
    df = df.loc[~_lookup_imputed(df, mask)].copy()
    df["arm"] = label
    df["chunk"] = chunk_of(df["run_time"])
    df.attrs["n_before_filter"] = n_before
    return df, missing


def holm(t: pd.DataFrame) -> pd.DataFrame:
    t = t.sort_values("p_raw").reset_index(drop=True)
    m = len(t)
    t["p_holm"] = np.maximum.accumulate(
        [(m - i) * p for i, p in enumerate(t["p_raw"])]).clip(max=1.0)
    return t


def paired_wilcoxon(per_station: dict, labels: list) -> pd.DataFrame:
    rows = []
    for A, B in itertools.combinations(labels, 2):
        d = (per_station[A] - per_station[B]).dropna()
        if len(d) < 10:
            continue
        _, p = wilcoxon(d)
        rows.append(dict(A=A, B=B, n=len(d), median_diff=float(d.median()),
                         share_A_better=float((d < 0).mean()), p_raw=float(p)))
    return holm(pd.DataFrame(rows)) if rows else pd.DataFrame()


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    available = {a: [s for s in ARMS[a][1] if (RAW / f"{s}_raw.parquet").exists()] for a in ARMS}
    n_have = sum(len(v) for v in available.values())
    n_want = sum(len(v[1]) for v in ARMS.values())
    print(f"[i] {n_have}/{n_want} Roh-Parquets vorhanden")
    for a, got in available.items():
        miss = [s for s in ARMS[a][1] if s not in got]
        if miss:
            print(f"    fehlt für {ARMS[a][0]}: {', '.join(miss)}")
    if not n_have:
        sys.exit("[FATAL] Noch keine testmode_*_raw.parquet vorhanden.")

    sids = set()
    for a, stems in available.items():
        for s in stems:
            sids |= set(pd.read_parquet(RAW / f"{s}_raw.parquet", columns=["station_id"])
                        ["station_id"].map(norm_station).unique())
    print(f"[i] {len(sids)} Stationen — baue Imputationsmaske aus {STATION_RAW_DIR} …", flush=True)
    mask = build_imputation_mask(sorted(sids), STATION_RAW_DIR)

    frames, per_station, cover = {}, {}, []
    for a in ARMS:
        if not available[a]:
            continue
        df, _ = load_arm(a, mask)
        label = ARMS[a][0]
        frames[label] = df
        per_station[label] = grouped_rmse(df, "station_id")
        cover.append(dict(arm=label, rows_raw=df.attrs["n_before_filter"], rows_kept=len(df),
                          share_kept=len(df) / df.attrs["n_before_filter"],
                          stations=df["station_id"].nunique(),
                          run_times=df["run_time"].nunique(),
                          first_run=df["run_time"].min(), last_run=df["run_time"].max()))
        print(f"  {label:18s} RMSE {per_station[label].mean():.4f}  "
              f"({df['station_id'].nunique()} Stationen, {len(df):,} Zeilen)", flush=True)

    labels = list(frames)
    big = pd.concat(frames.values(), ignore_index=True)
    t0 = pd.DataFrame(cover)
    t0.to_csv(OUT / "testmode_coverage.csv", index=False)
    print("\n=== Abdeckung je Arm ===")
    print(t0.to_string(index=False))

    # ── Referenzen: auf den Zeilen des am besten abgedeckten Arms ───────
    ref_label = max(labels, key=lambda l: len(frames[l]))
    ref_df = frames[ref_label]
    refs = {"ICON-D2": grouped_rmse(ref_df, "station_id", "nwp_ref"),
            "Persistenz": grouped_rmse(ref_df, "station_id", "pers_ref")}

    # ── T1: Haupttabelle, je Arm auf eigenen Zeilen ─────────────────────
    rows = [dict(arm=l, rmse=float(per_station[l].mean()),
                 rmse_sd_station=float(per_station[l].std()),
                 retrain="ja" if l in RETRAINED else "nein",
                 n_stationen=int(per_station[l].size)) for l in labels]
    rows += [dict(arm=r, rmse=float(v.mean()), rmse_sd_station=float(v.std()),
                  retrain="—", n_stationen=int(v.size)) for r, v in refs.items()]
    t1 = pd.DataFrame(rows).sort_values("rmse")
    t1.to_csv(OUT / "testmode_overview.csv", index=False)
    print("\n=== Gefilterte Stationsmittel-RMSE, Testjahr (eigene Zeilen je Arm) ===")
    print(t1.round(4).to_string(index=False))

    # ── T1b: strikt gepaart auf dem gemeinsamen Zeilenschnitt ───────────
    key = ["station_id", "run_time", "horizon"]
    common = None
    for l in labels:
        idx = pd.MultiIndex.from_frame(frames[l][key])
        common = idx if common is None else common.intersection(idx)
    per_station_p, rows_p = {}, []
    for l in labels:
        d = frames[l].set_index(key)
        d = d.loc[d.index.isin(common)].reset_index()
        per_station_p[l] = grouped_rmse(d, "station_id")
        rows_p.append(dict(arm=l, rmse=float(per_station_p[l].mean()),
                           retrain="ja" if l in RETRAINED else "nein", n_zeilen=len(d)))
    dref = frames[ref_label].set_index(key)
    dref = dref.loc[dref.index.isin(common)].reset_index()
    refs_p = {"ICON-D2": grouped_rmse(dref, "station_id", "nwp_ref"),
              "Persistenz": grouped_rmse(dref, "station_id", "pers_ref")}
    rows_p += [dict(arm=r, rmse=float(v.mean()), retrain="—", n_zeilen=len(dref))
               for r, v in refs_p.items()]
    t1b = pd.DataFrame(rows_p).sort_values("rmse")
    t1b.to_csv(OUT / "testmode_overview_paired.csv", index=False)
    print(f"\n=== Dasselbe auf dem gemeinsamen Zeilenschnitt ({len(common):,} Zeilen) ===")
    print(t1b.round(4).to_string(index=False))

    # ── T2: Wilcoxon zwischen den Armen (gepaarter Schnitt) ─────────────
    t2 = paired_wilcoxon(per_station_p, labels)
    if len(t2):
        t2.to_csv(OUT / "testmode_wilcoxon.csv", index=False)
        print("\n=== Wilcoxon über die Teststationen (Holm, gemeinsamer Schnitt) ===")
        print(t2.round(6).to_string(index=False))

    # ── T3: RMSE je 4-Monats-Chunk ─────────────────────────────────────
    rows3 = []
    for l in labels:
        d = frames[l]
        for name, _, _ in CHUNKS:
            s = d[d["chunk"] == name]
            if not len(s):
                continue
            g = grouped_rmse(s, "station_id")
            rows3.append(dict(arm=l, chunk=name, rmse=float(g.mean()), n_zeilen=len(s),
                              retrain="ja" if l in RETRAINED else "nein"))
    for rn, cn in (("ICON-D2", "nwp_ref"), ("Persistenz", "pers_ref")):
        for name, _, _ in CHUNKS:
            s = ref_df[ref_df["chunk"] == name]
            if not len(s):
                continue
            g = grouped_rmse(s, "station_id", cn)
            rows3.append(dict(arm=rn, chunk=name, rmse=float(g.mean()), n_zeilen=len(s),
                              retrain="—"))
    t3 = pd.DataFrame(rows3)
    t3.to_csv(OUT / "testmode_by_chunk.csv", index=False)
    print("\n=== RMSE je 4-Monats-Fenster ===")
    print(t3.pivot(index="arm", columns="chunk", values="rmse")
          .reindex(columns=[c[0] for c in CHUNKS]).round(4).to_string())

    pd.concat([pd.DataFrame({"arm": l, "station_id": v.index, "rmse": v.values,
                             "rmse_paired": per_station_p[l].reindex(v.index).values})
               for l, v in per_station.items()], ignore_index=True).to_csv(
        OUT / "testmode_per_station.csv", index=False)

    make_figures(big, t1, t2, per_station_p, refs, refs_p, labels, t3)
    print(f"\n[i] Abbildungen → {FIGS}/")


def _save(fig, name):
    for fmt in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_figures(big, t1, t2, per_station_p, refs, refs_p, labels, t3):
    # 01 Balken-RMSE mit Stationsstreuung
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = t1[t1["arm"].isin(labels)].sort_values("rmse")
    ax.barh(np.asarray(sub["arm"]), np.asarray(sub["rmse"]),
            color=[COLORS.get(a, "#777") for a in sub["arm"]], alpha=.85)
    for r in t1[~t1["arm"].isin(labels)].itertuples():
        ax.axvline(r.rmse, ls=":", color=COLORS.get(r.arm, "#999"), lw=1.5)
        ax.text(r.rmse, -.6, r.arm, rotation=90, va="bottom", ha="center",
                fontsize=8, color=COLORS.get(r.arm, "#999"))
    ax.set_xlabel("RMSE (m/s), Mittel über die 50 Teststationen")
    ax.grid(axis="x", alpha=.3)
    ax.set_title("Testjahr 2025-08 … 2026-07 — gefiltert, zero-shot")
    _save(fig, "01_bar_rmse")

    # 02 RMSE je 4-Monats-Fenster
    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs = np.arange(len(CHUNKS))
    for l in labels:
        d = t3[t3["arm"] == l].set_index("chunk").reindex([c[0] for c in CHUNKS])
        ax.plot(xs, np.asarray(d["rmse"]), marker="o", color=COLORS.get(l),
                lw=2, label=l + (" (Retrain)" if l in RETRAINED else ""))
    for rn in ("ICON-D2", "Persistenz"):
        d = t3[t3["arm"] == rn].set_index("chunk").reindex([c[0] for c in CHUNKS])
        ax.plot(xs, np.asarray(d["rmse"]), ls=":", lw=2, color=COLORS[rn], label=rn)
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in CHUNKS])
    ax.set_ylabel("RMSE (m/s)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("RMSE je 4-Monats-Fenster — HIST-Arme mit Retraining, übrige mit einem Modell")
    _save(fig, "02_rmse_by_chunk")

    # 03 gepaarte Differenz-Boxplots
    if len(t2):
        fig, ax = plt.subplots(figsize=(9, max(3, .55 * len(t2))))
        data, names = [], []
        for r in t2.itertuples():
            data.append((per_station_p[r.A] - per_station_p[r.B]).dropna().values)
            names.append(f"{r.A}\n− {r.B}")
        ax.boxplot(data, vert=False, labels=names, showfliers=False)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("ΔRMSE je Station (m/s), negativ = A besser")
        ax.grid(axis="x", alpha=.3)
        _save(fig, "03_paired_diff_boxplots")

    # 04 Fehler über Prognosehorizont
    fig, ax = plt.subplots(figsize=(9, 5))
    for l in labels:
        h = grouped_rmse(big[big["arm"] == l], "horizon")
        ax.plot(np.asarray(h.index), np.asarray(h.values), label=l, color=COLORS.get(l), lw=1.8)
    d0 = big[big["arm"] == labels[0]]
    for rn, col in (("nwp_ref", "ICON-D2"), ("pers_ref", "Persistenz")):
        h = grouped_rmse(d0, "horizon", rn)
        ax.plot(np.asarray(h.index), np.asarray(h.values), label=col,
                color=COLORS[col], ls=":", lw=2)
    ax.set_xlabel("Prognosehorizont (h)"); ax.set_ylabel("RMSE (m/s)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Fehler über den Prognosehorizont, Testjahr")
    _save(fig, "04_error_by_horizon")

    # 05 Fehler über Windklasse
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = [0, 2, 4, 6, 8, 10, 12, 100]
    lab = ["0–2", "2–4", "4–6", "6–8", "8–10", "10–12", ">12"]
    big = big.copy()
    big["ws_class"] = pd.cut(big["gt"], bins=bins, labels=lab, right=False)
    for l in labels:
        h = grouped_rmse(big[big["arm"] == l], "ws_class")
        ax.plot(range(len(h)), np.asarray(h.values), marker="o", label=l, color=COLORS.get(l))
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab)
    ax.set_xlabel("Gemessene Windgeschwindigkeit (m/s)"); ax.set_ylabel("RMSE (m/s)")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("Fehler nach Windklasse, Testjahr")
    _save(fig, "05_error_by_windspeed_class")

    # 06 Fehler über Monat
    fig, ax = plt.subplots(figsize=(10, 5))
    big["month"] = pd.DatetimeIndex(big["valid_time"]).to_period("M").astype(str)
    idx = None
    for l in labels:
        h = grouped_rmse(big[big["arm"] == l], "month")
        ax.plot(range(len(h)), np.asarray(h.values), marker="o", label=l, color=COLORS.get(l))
        idx = h.index
    for name, lo, hi in CHUNKS[1:]:
        pos = list(idx).index(pd.Timestamp(lo).strftime("%Y-%m")) if \
            pd.Timestamp(lo).strftime("%Y-%m") in list(idx) else None
        if pos is not None:
            ax.axvline(pos - .5, color="k", lw=1, ls="--", alpha=.5)
    ax.set_xticks(range(len(idx))); ax.set_xticklabels(list(idx), rotation=45, ha="right")
    ax.set_ylabel("RMSE (m/s)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Fehler über die Monate — gestrichelt: Retrain-Grenzen der HIST-Arme")
    _save(fig, "06_error_by_month")

    # 07 Scatter aller Arme
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4),
                             sharex=True, sharey=True)
    for ax, ll in zip(np.atleast_1d(axes), labels):
        d = big[big["arm"] == ll]
        d = d.sample(min(60000, len(d)), random_state=0)
        ax.hexbin(d["gt"], d["pred"], gridsize=60, bins="log", cmap="viridis", mincnt=1)
        ax.plot([0, 25], [0, 25], "r--", lw=1)
        ax.set_title(ll, fontsize=9); ax.set_xlabel("Messung (m/s)")
    np.atleast_1d(axes)[0].set_ylabel("Vorhersage (m/s)")
    _save(fig, "07_scatter_all")

    # 08 Skill-Verteilung gegen ICON-D2
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in labels:
        s = (1 - per_station_p[l] / refs_p["ICON-D2"].reindex(per_station_p[l].index)).dropna()
        ax.hist(s, bins=25, histtype="step", lw=1.8, color=COLORS.get(l),
                label=f"{l} (Median {s.median():.3f})")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Skill gegenüber ICON-D2 je Station"); ax.set_ylabel("Stationen")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Verteilung des Skill_NWP über die 50 Teststationen")
    _save(fig, "08_skill_nwp_distribution")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
