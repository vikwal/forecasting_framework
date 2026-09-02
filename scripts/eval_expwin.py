#!/usr/bin/env python3
"""
Auswertung des Expanding-Window-Retrainings (docs/expanding_window_retrain_handoff.md § 7).

Kernproblem des Aufbaus: die drei Zeitschritte haben unterschiedliche Val-FENSTER
(Aug-Nov / Dez-Mae / Apr-Jul). Ein direkter Vergleich der RMSE ueber die Schritte
misst daher Saison und Fenstergroesse zugleich. Deshalb wird jeder Schritt gegen ein
saisongleiches Kontrollmodell gestellt: den bereits vorhandenen Single-Window-Retrain
(fixes Trainingsfenster train < 2024-08-01), dessen Roh-Vorhersagen ueber das gesamte
Jahr 2024-08 .. 2025-08 vorliegen und hier auf dasselbe Fenster geschnitten werden.

Schritt 1 traniert auf exakt demselben Fenster wie die Kontrolle und dient damit als
Nullmessung (Seed-/Early-Stopping-Rauschen), Schritt 2/3 haben +4 bzw. +8 Monate.

Filterung wie in §14-16 von docs/evaluation_results.md: imputierte Zielstunden
(keine Rohmessung in der Stunde) werden ausgeschlossen.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geostatistics.stdrun.make_stdhp_figures import (
    norm_station, build_imputation_mask, _lookup_imputed,
)

STATION_RAW_DIR = Path("/mnt/lambda1/nvme1/synthetic/raw/wind")  # auf l1: /mnt/nvme1/...
RAW = Path("data/raw_preds")
ARMS = {
    "dcrnn":          "DCRNN GRID (A)",
    "dcrnn_idw_alt":  "DCRNN IDW (D')",
    "dcrnn_nwp_hist": "DCRNN GRID+HIST",
    "mtgnn_nwp":      "MTGNN GRID",
    "mtgnn_nwp_hist": "MTGNN GRID+HIST",
}
STEPS = {1: ("2024-08-01", "2024-12-01"), 2: ("2024-12-01", "2025-04-01"), 3: ("2025-04-01", "2025-08-01")}
FOLDS = (1, 2, 3)
COLS = ["station_id", "valid_time", "pred", "gt"]


def per_station_rmse(df, mask):
    """Gefilterte RMSE je Station + imputierter Anteil."""
    df = df.copy()
    df["station_id"] = df["station_id"].map(norm_station)
    imp = _lookup_imputed(df, mask)
    frac = float(imp.mean())
    df = df.loc[~imp]
    g = df.groupby("station_id").apply(
        lambda d: float(np.sqrt(np.mean((d["pred"] - d["gt"]) ** 2))), include_groups=False)
    return g, frac


def main():
    # ── Stationsmenge und Maske ──────────────────────────────────────────
    sids = set()
    for arm in ARMS:
        for f in FOLDS:
            p = RAW / f"expwin_{arm}_s1_fold{f}_raw.parquet"
            sids |= set(pd.read_parquet(p, columns=["station_id"])["station_id"].map(norm_station).unique())
    sids = sorted(sids)
    print(f"[i] {len(sids)} Stationen, baue Imputationsmaske …", flush=True)
    mask = build_imputation_mask(sids, STATION_RAW_DIR)

    # Verifikation: jede Stunde mit gt<0 ist nicht-physikalisch, muss also imputiert sein
    chk = pd.read_parquet(RAW / "expwin_dcrnn_nwp_hist_s1_fold1_raw.parquet", columns=COLS)
    chk["station_id"] = chk["station_id"].map(norm_station)
    neg = chk[chk["gt"] < 0].drop_duplicates(["station_id", "valid_time"])
    if len(neg):
        bad = int((~_lookup_imputed(neg, mask)).sum())
        if bad:
            sys.exit(f"[FATAL] Maske falsch: {bad} von {len(neg)} Stunden mit gt<0 gelten als nicht imputiert.")
        print(f"[i] Maske validiert: alle {len(neg)} Stunden mit gt<0 sind als imputiert markiert.")
    else:
        print("[i] Keine gt<0-Stunden im Pruefdatensatz — Maske nicht gegenpruefbar, aber Filter greift.")

    # ── Expanding-Window-Läufe und saisongleiche Kontrolle ───────────────
    rows, per_st = [], {}
    for arm in ARMS:
        for step, (a, b) in STEPS.items():
            for fold in FOLDS:
                ew = pd.read_parquet(RAW / f"expwin_{arm}_s{step}_fold{fold}_raw.parquet", columns=COLS)
                r_ew, f_ew = per_station_rmse(ew, mask)

                ct = pd.read_parquet(RAW / f"retrain_{arm}_fold{fold}_raw.parquet", columns=COLS)
                lo, hi = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
                ct = ct[(ct["valid_time"] >= lo) & (ct["valid_time"] < hi)]
                r_ct, f_ct = per_station_rmse(ct, mask)

                common = r_ew.index.intersection(r_ct.index)
                rows.append(dict(arm=arm, step=step, fold=fold,
                                 rmse_ew=r_ew.mean(), rmse_ct=r_ct.mean(),
                                 n_st_ew=len(r_ew), n_st_common=len(common),
                                 imp_ew=f_ew, imp_ct=f_ct,
                                 n_rows_ew=len(ew), n_rows_ct=len(ct)))
                per_st[(arm, step, fold)] = (r_ew.loc[common], r_ct.loc[common])
                print(f"  {arm:15s} s{step} fold{fold}: EW {r_ew.mean():.4f}  Fix {r_ct.mean():.4f}  "
                      f"({len(common)} St., imp {f_ew*100:.2f}/{f_ct*100:.2f} %)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("data/test_results/expwin_summary.csv", index=False)

    tidy = pd.concat([
        pd.DataFrame({"arm": arm, "step": step, "fold": fold, "station_id": ew.index,
                      "rmse_ew": ew.values, "rmse_ct": ct.values})
        for (arm, step, fold), (ew, ct) in per_st.items()
    ], ignore_index=True)
    tidy.to_csv("data/test_results/expwin_per_station.csv", index=False)

    # ── Wilcoxon je (Arm, Schritt) über die Vereinigung der 3 Folds ──────
    stats = []
    for arm in ARMS:
        for step in STEPS:
            ew = pd.concat([per_st[(arm, step, f)][0].rename(lambda s, f=f: f"{f}_{s}") for f in FOLDS])
            ct = pd.concat([per_st[(arm, step, f)][1].rename(lambda s, f=f: f"{f}_{s}") for f in FOLDS])
            d = (ew - ct).dropna()
            st, p = wilcoxon(d)
            stats.append(dict(arm=arm, step=step, n=len(d), median_delta=float(d.median()),
                              mean_delta=float(d.mean()), share_ew_better=float((d < 0).mean()), p_raw=p))
    sdf = pd.DataFrame(stats)
    # Holm über alle 15 Vergleiche
    order = sdf["p_raw"].rank(method="first").astype(int)
    m = len(sdf)
    sdf = sdf.sort_values("p_raw").reset_index(drop=True)
    sdf["p_holm"] = np.maximum.accumulate([(m - i) * p for i, p in enumerate(sdf["p_raw"])]).clip(max=1.0)
    sdf.to_csv("data/test_results/expwin_wilcoxon.csv", index=False)

    # ── Differenz-in-Differenzen: Schritt 2/3 gegen die Nullmessung Schritt 1 ──
    # Schritt 1 trainiert auf demselben Fenster wie die Kontrolle, sein Delta ist also
    # reines Seed-/Early-Stopping-Rauschen. Der Zugewinn durch mehr Trainingsdaten ist
    # (EW-Fix)_sN - (EW-Fix)_s1, gepaart je Station.
    did = []
    for arm in ARMS:
        base = {f: (per_st[(arm, 1, f)][0] - per_st[(arm, 1, f)][1]) for f in FOLDS}
        for step in (2, 3):
            d = pd.concat([
                (per_st[(arm, step, f)][0] - per_st[(arm, step, f)][1] - base[f]).rename(
                    lambda x, f=f: f"{f}_{x}") for f in FOLDS]).dropna()
            st, p = wilcoxon(d)
            did.append(dict(arm=arm, step=step, n=len(d), median_did=float(d.median()),
                            mean_did=float(d.mean()), share_better=float((d < 0).mean()), p_raw=p))
    ddf = pd.DataFrame(did).sort_values("p_raw").reset_index(drop=True)
    md = len(ddf)
    ddf["p_holm"] = np.maximum.accumulate([(md - i) * p for i, p in enumerate(ddf["p_raw"])]).clip(max=1.0)
    ddf.to_csv("data/test_results/expwin_did.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Gefilterte Stationsmittel-RMSE: Expanding vs. saisongleiche Kontrolle ===")
    piv = df.pivot_table(index="arm", columns="step", values=["rmse_ew", "rmse_ct"])
    print(piv.round(4).to_string())
    print("\n=== Wilcoxon (Expanding − Fix), gepaart über 153 Stationen, Holm über 15 Vergleiche ===")
    print(sdf[["arm", "step", "n", "median_delta", "share_ew_better", "p_raw", "p_holm"]]
          .sort_values(["arm", "step"]).round(6).to_string(index=False))
    print("\n=== Differenz-in-Differenzen gegen Schritt 1 (Nullmessung), Holm über 10 Vergleiche ===")
    print(ddf[["arm", "step", "n", "median_did", "share_better", "p_raw", "p_holm"]]
          .sort_values(["arm", "step"]).round(6).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
