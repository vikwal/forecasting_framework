#!/usr/bin/env python3
"""
Testjahr-Auswertung der 5 Flagschiff-Arme (2025-08-01 .. 2026-06-01).

Alle 5 Arme wurden mit erweitertem Trainingsfenster (alles vor 2025-08-01, also
2 Jahre statt 1) neu trainiert und zero-shot auf den 51 val_files-Zielstationen
im Testjahr ausgewertet. Filterung identisch zu docs/evaluation_results.md
§14-17: imputierte Zielstunden (keine Rohmessung in der Stunde) ausgeschlossen.

Erzeugt Tabellen (data/test_results/testyear_*.csv) und die Abbildungen
(figures/testyear/) fuer das Paper.
"""
import sys, warnings, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geostatistics.stdrun.make_stdhp_figures import norm_station, build_imputation_mask, _lookup_imputed

# Auf l1 liegen die Rohmessungen lokal, auf w-lambdablade2 ueber den l1-Mount.
_CANDIDATES = [Path("/mnt/nvme1/synthetic/raw/wind"), Path("/mnt/lambda1/nvme1/synthetic/raw/wind")]
STATION_RAW_DIR = next((c for c in _CANDIDATES if c.is_dir()), _CANDIDATES[-1])
RAW  = Path("data/raw_preds")
OUT  = Path("data/test_results")
FIGS = Path("figures/testyear")
FOLDS = (1, 2, 3)
ARMS = {
    "dcrnn_nwp_hist": "DCRNN GRID+HIST",
    "mtgnn_nwp_hist": "MTGNN GRID+HIST",
    "mtgnn_nwp":      "MTGNN GRID",
    "dcrnn":          "DCRNN GRID",
    "dcrnn_idw_alt":  "DCRNN IDW (D')",
}
COLORS = {"DCRNN GRID+HIST": "#1f77b4", "MTGNN GRID+HIST": "#d62728", "MTGNN GRID": "#ff7f0e",
          "DCRNN GRID": "#2ca02c", "DCRNN IDW (D')": "#9467bd",
          "ICON-D2": "#555555", "Persistenz": "#999999"}
COLS = ["station_id", "run_time", "valid_time", "horizon", "pred", "gt", "nwp_ref", "pers_ref"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def grouped_rmse(df, by, pred_col="pred", obs_col="gt"):
    """RMSE je Gruppe, ohne groupby.apply — laeuft auf allen pandas-Versionen
    (include_groups gibt es erst ab 2.2) und ist deutlich schneller."""
    e2 = (df[pred_col].to_numpy() - df[obs_col].to_numpy()) ** 2
    key = df[by] if isinstance(by, str) else by
    return np.sqrt(pd.Series(e2, index=df.index).groupby(key, observed=True).mean())


def load(arm, fold, mask):
    """Gefilterter Roh-Frame eines Laufs; None wenn (noch) nicht vorhanden."""
    p = RAW / f"testyear_{arm}_fold{fold}_raw.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=COLS)
    df["station_id"] = df["station_id"].map(norm_station)
    df = df.loc[~_lookup_imputed(df, mask)].copy()
    df["arm"] = ARMS[arm]
    df["fold"] = fold
    return df


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    have = [(a, f) for a in ARMS for f in FOLDS if (RAW / f"testyear_{a}_fold{f}_raw.parquet").exists()]
    if not have:
        sys.exit("[FATAL] Noch keine testyear_*_raw.parquet vorhanden.")
    print(f"[i] {len(have)}/{len(ARMS)*len(FOLDS)} Läufe vorhanden")

    sids = set()
    for a, f in have:
        sids |= set(pd.read_parquet(RAW / f"testyear_{a}_fold{f}_raw.parquet",
                                    columns=["station_id"])["station_id"].map(norm_station).unique())
    print(f"[i] {len(sids)} Stationen, baue Imputationsmaske …", flush=True)
    mask = build_imputation_mask(sorted(sids), STATION_RAW_DIR)

    frames, per_station, refs = [], {}, {}
    for a, f in have:
        d = load(a, f, mask)
        g = grouped_rmse(d, "station_id")
        per_station[(ARMS[a], f)] = g
        frames.append(d)
        if f not in refs:
            refs[f] = {
                "ICON-D2":    grouped_rmse(d, "station_id", "nwp_ref"),
                "Persistenz": grouped_rmse(d, "station_id", "pers_ref"),
            }
        print(f"  {ARMS[a]:18s} fold{f}: RMSE {g.mean():.4f}  ({len(g)} Stationen)", flush=True)
    big = pd.concat(frames, ignore_index=True)

    # ── T1: Haupttabelle ────────────────────────────────────────────────
    rows = []
    for label in ARMS.values():
        vals = [per_station[(label, f)].mean() for f in FOLDS if (label, f) in per_station]
        if vals:
            rows.append(dict(arm=label, **{f"fold{f}": per_station[(label, f)].mean()
                                           for f in FOLDS if (label, f) in per_station},
                             mean=float(np.mean(vals))))
    for rname in ("ICON-D2", "Persistenz"):
        vals = [refs[f][rname].mean() for f in refs]
        rows.append(dict(arm=rname, **{f"fold{f}": refs[f][rname].mean() for f in refs},
                         mean=float(np.mean(vals))))
    t1 = pd.DataFrame(rows).sort_values("mean")
    t1.to_csv(OUT / "testyear_overview.csv", index=False)
    print("\n=== Gefilterte Stationsmittel-RMSE, Testjahr ===")
    print(t1.round(4).to_string(index=False))

    # ── T2: Wilcoxon-Matrix zwischen den Armen ──────────────────────────
    labels = [l for l in ARMS.values() if any((l, f) in per_station for f in FOLDS)]
    stats = []
    for A, B in itertools.combinations(labels, 2):
        a = pd.concat([per_station[(A, f)].rename(lambda s, f=f: f"{f}_{s}") for f in FOLDS if (A, f) in per_station])
        b = pd.concat([per_station[(B, f)].rename(lambda s, f=f: f"{f}_{s}") for f in FOLDS if (B, f) in per_station])
        d = (a - b).dropna()
        if len(d) < 10:
            continue
        _, p = wilcoxon(d)
        stats.append(dict(A=A, B=B, n=len(d), median_diff=float(d.median()),
                          share_A_better=float((d < 0).mean()), p_raw=p))
    t2 = pd.DataFrame(stats)
    if len(t2):
        t2 = t2.sort_values("p_raw").reset_index(drop=True)
        m = len(t2)
        t2["p_holm"] = np.maximum.accumulate([(m - i) * p for i, p in enumerate(t2["p_raw"])]).clip(max=1.0)
        t2.to_csv(OUT / "testyear_wilcoxon.csv", index=False)
        print("\n=== Wilcoxon zwischen den Armen (Holm) ===")
        print(t2.round(6).to_string(index=False))

    # per-Station persistieren
    pd.concat([pd.DataFrame({"arm": k[0], "fold": k[1], "station_id": v.index, "rmse": v.values})
               for k, v in per_station.items()], ignore_index=True).to_csv(
        OUT / "testyear_per_station.csv", index=False)

    make_figures(big, t1, t2, per_station, refs, labels)
    print(f"\n[i] Abbildungen → {FIGS}/")


def _save(fig, name):
    for fmt in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_figures(big, t1, t2, per_station, refs, labels):
    # 01 Balken-RMSE
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = t1[t1["arm"].isin(labels)].sort_values("mean")
    fcols = [c for c in t1.columns if c.startswith("fold")]
    err = sub[fcols].std(axis=1)
    ax.barh(np.asarray(sub["arm"]), np.asarray(sub["mean"]), xerr=np.asarray(err), color=[COLORS.get(a, "#777") for a in sub["arm"]],
            alpha=.85, capsize=3)
    for r in t1[~t1["arm"].isin(labels)].itertuples():
        ax.axvline(r.mean, ls=":", color=COLORS.get(r.arm, "#999"), lw=1.5)
        ax.text(r.mean, -.6, r.arm, rotation=90, va="bottom", ha="center", fontsize=8,
                color=COLORS.get(r.arm, "#999"))
    ax.set_xlabel("RMSE (m/s), Stationsmittel über 3 Folds"); ax.grid(axis="x", alpha=.3)
    ax.set_title("Testjahr 2025-08 … 2026-06 — gefiltert, per Station")
    _save(fig, "01_bar_rmse")

    # 02 Fold-Streuung
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, l in enumerate(labels):
        for f in FOLDS:
            if (l, f) in per_station:
                ax.scatter([i], [per_station[(l, f)].mean()], s=60, color=COLORS.get(l),
                           marker="o$123"[f] if False else "o", alpha=.5 + .15 * f)
                ax.annotate(f"f{f}", (i, per_station[(l, f)].mean()), fontsize=7,
                            xytext=(6, 0), textcoords="offset points")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("RMSE (m/s)"); ax.grid(alpha=.3); ax.set_title("Streuung über die 3 Folds")
    _save(fig, "02_fold_dispersion")

    # 03 gepaarte Differenz-Boxplots
    if len(t2):
        fig, ax = plt.subplots(figsize=(9, max(3, .5 * len(t2))))
        data, names = [], []
        for r in t2.itertuples():
            a = pd.concat([per_station[(r.A, f)].rename(lambda s, f=f: f"{f}_{s}") for f in FOLDS if (r.A, f) in per_station])
            b = pd.concat([per_station[(r.B, f)].rename(lambda s, f=f: f"{f}_{s}") for f in FOLDS if (r.B, f) in per_station])
            data.append((a - b).dropna().values); names.append(f"{r.A}\n− {r.B}")
        ax.boxplot(data, vert=False, labels=names, showfliers=False)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("ΔRMSE je Station (m/s), negativ = A besser"); ax.grid(axis="x", alpha=.3)
        _save(fig, "03_paired_diff_boxplots")

    # 04 Fehler über Prognosehorizont
    fig, ax = plt.subplots(figsize=(9, 5))
    for l in labels:
        d = big[big["arm"] == l]
        h = grouped_rmse(d, "horizon")
        ax.plot(np.asarray(h.index), np.asarray(h.values), label=l, color=COLORS.get(l), lw=1.8)
    d0 = big[big["arm"] == labels[0]]
    for rn, col in (("nwp_ref", "ICON-D2"), ("pers_ref", "Persistenz")):
        h = grouped_rmse(d0, "horizon", rn)
        ax.plot(np.asarray(h.index), np.asarray(h.values), label=col, color=COLORS[col], ls=":", lw=2)
    ax.set_xlabel("Prognosehorizont (h)"); ax.set_ylabel("RMSE (m/s)")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("Fehler über den Prognosehorizont, Testjahr")
    _save(fig, "04_error_by_horizon")

    # 05 Fehler über Windklasse
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = [0, 2, 4, 6, 8, 10, 12, 100]
    lab = ["0–2", "2–4", "4–6", "6–8", "8–10", "10–12", ">12"]
    big["ws_class"] = pd.cut(big["gt"], bins=bins, labels=lab, right=False)
    for l in labels:
        d = big[big["arm"] == l]
        h = grouped_rmse(d, "ws_class")
        ax.plot(range(len(h)), np.asarray(h.values), marker="o", label=l, color=COLORS.get(l))
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab)
    ax.set_xlabel("Gemessene Windgeschwindigkeit (m/s)"); ax.set_ylabel("RMSE (m/s)")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("Fehler nach Windklasse, Testjahr")
    _save(fig, "05_error_by_windspeed_class")

    # 06 Fehler über Monat
    fig, ax = plt.subplots(figsize=(9, 5))
    big["month"] = pd.DatetimeIndex(big["valid_time"]).to_period("M").astype(str)
    for l in labels:
        d = big[big["arm"] == l]
        h = grouped_rmse(d, "month")
        ax.plot(range(len(h)), np.asarray(h.values), marker="o", label=l, color=COLORS.get(l))
        idx = h.index
    ax.set_xticks(range(len(idx))); ax.set_xticklabels(list(idx), rotation=45, ha="right")
    ax.set_ylabel("RMSE (m/s)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Fehler über die Monate des Testjahres")
    _save(fig, "06_error_by_month")

    # 07/08 Scatter
    best = t1[t1["arm"].isin(labels)].iloc[0]["arm"]
    for name, l in (("07_scatter_best", best), ("08_scatter_all", None)):
        if l is None:
            fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), sharex=True, sharey=True)
            for ax, ll in zip(np.atleast_1d(axes), labels):
                d = big[big["arm"] == ll].sample(min(60000, (big["arm"] == ll).sum()), random_state=0)
                ax.hexbin(d["gt"], d["pred"], gridsize=60, bins="log", cmap="viridis", mincnt=1)
                ax.plot([0, 25], [0, 25], "r--", lw=1)
                ax.set_title(ll, fontsize=9); ax.set_xlabel("Messung (m/s)")
            np.atleast_1d(axes)[0].set_ylabel("Vorhersage (m/s)")
        else:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            d = big[big["arm"] == l].sample(min(150000, (big["arm"] == l).sum()), random_state=0)
            hb = ax.hexbin(d["gt"], d["pred"], gridsize=70, bins="log", cmap="viridis", mincnt=1)
            ax.plot([0, 25], [0, 25], "r--", lw=1)
            fig.colorbar(hb, ax=ax, label="Anzahl")
            ax.set_xlabel("Messung (m/s)"); ax.set_ylabel("Vorhersage (m/s)"); ax.set_title(l)
        _save(fig, name)

    # 09 Skill-Verteilung
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in labels:
        sk = []
        for f in FOLDS:
            if (l, f) in per_station:
                sk.append(1 - per_station[(l, f)] / refs[f]["ICON-D2"].reindex(per_station[(l, f)].index))
        s = pd.concat(sk).dropna()
        ax.hist(s, bins=30, histtype="step", lw=1.8, label=f"{l} (Median {s.median():.3f})",
                color=COLORS.get(l))
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Skill gegenüber ICON-D2 je Station"); ax.set_ylabel("Stationen")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("Verteilung des Skill_NWP, Testjahr")
    _save(fig, "09_skill_nwp_distribution")

    # 10 Validierungsjahr gegen Testjahr
    vp = OUT / "expwin_per_station.csv"
    if vp.exists():
        v = pd.read_csv(vp, dtype={"station_id": str})
        vmap = {"dcrnn_nwp_hist": "DCRNN GRID+HIST", "mtgnn_nwp_hist": "MTGNN GRID+HIST",
                "mtgnn_nwp": "MTGNN GRID", "dcrnn": "DCRNN GRID", "dcrnn_idw_alt": "DCRNN IDW (D')"}
        v["label"] = v["arm"].map(vmap)
        val = v[v["step"] == 3].groupby("label")["rmse_ew"].mean()
        fig, ax = plt.subplots(figsize=(7, 5))
        for l in labels:
            if l in val.index:
                t = t1.loc[t1["arm"] == l, "mean"].iloc[0]
                ax.scatter(val[l], t, s=110, color=COLORS.get(l), label=l)
                ax.annotate(l, (val[l], t), fontsize=8, xytext=(7, -3), textcoords="offset points")
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=.5)
        ax.set_xlabel("RMSE Validierungsjahr (Expanding, Schritt 3)")
        ax.set_ylabel("RMSE Testjahr")
        ax.grid(alpha=.3); ax.set_title("Übertragen die Validierungsbefunde auf den Testsatz?")
        _save(fig, "10_val_vs_test")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
