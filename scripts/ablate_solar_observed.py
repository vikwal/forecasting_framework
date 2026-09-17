#!/usr/bin/env python3
"""Was traegt die eigene Messhistorie im observed-Fenster bei?

Die Fold-Modelle sehen 48 h eigene Vergangenheit (``observed_features:
[ghi, dhi]``) — wegen ``target_transform: nwp_residual`` genauer: 48 h
NWP-Fehlerhistorie der Zielstation. Ob das Modell diesen Kanal nutzt und ueber
welche Vorlaufzeit, laesst sich am fertigen Modell direkt messen: dieselben
Eingaben zweimal durchrechnen, einmal mit echtem observed-Fenster und einmal
mit genullten bzw. ueber die Laeufe permutierten Werten.

Permutation ist der ehrlichere der beiden Tests — sie erhaelt die Verteilung des
Kanals und zerstoert nur den Zusammenhang zum jeweiligen Lauf. Nullen sind fuer
das Modell dagegen ein Eingabewert, den es so nie gesehen hat.

Gegenprobe zur Frage, ob ueberhaupt etwas zu holen waere: die Autokorrelation des
Residuums zwischen dem letzten Messzeitpunkt vor dem Lauf und dem Lead
(``--autokorrelation``).

    frcst/bin/python scripts/ablate_solar_observed.py --stationen 5
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import tools, preprocessing, models  # noqa: E402
from utils.solar import solar_geometry  # noqa: E402

OUT = REPO / "data/test_results"
#: Leadbloecke der Auswertung. Fein am Anfang, weil sich dort alles entscheidet:
#: ein Block 0-6 h mittelt ueber 12 Leads und verdeckt den Effekt der ersten
#: beiden (genau das ist dem ersten Berichtsstand passiert).
BLOECKE = [("0.0 h", 0, 1), ("0.5 h", 1, 2), ("1 h", 2, 3), ("1.5–3 h", 3, 6),
           ("3–6 h", 6, 12), ("6–12 h", 12, 24), ("12–24 h", 24, 48), ("24–48 h", 48, 96)]


def lade_modell(fold: int, arm: str = "solar_tft"):
    tag = f"train_tft_bc_m-tft_c-{arm}_fold{fold}"
    cfg = tools.load_config(f"configs/{arm}/config_{arm}_fold{fold}.yaml")
    cfg["model"]["name"] = "tft"
    meta = pickle.load(open(REPO / f"models/{tag}_meta.pkl", "rb"))
    cfg["model"]["feature_dim"] = meta["feature_dim"]
    modell = models.get_model(config=cfg, hyperparameters=meta["hyperparameters"])
    modell.load_state_dict(torch.load(REPO / f"models/{tag}.pt", map_location="cpu"))
    modell.eval()
    return modell, cfg, meta


def _station_daten(cfg: dict, station: str):
    """Eine Station im Auswertungsfenster vorbereiten — wie get_test_results_tft_bc."""
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    c["data"] = dict(cfg["data"])
    c["data"]["files"] = [station]
    c["data"]["val_files"] = []
    c["data"]["test_files"] = [station]
    c["data"]["neighbor_pool"] = [station]
    # --eval-split val: gemessen wird im Validierungsjahr [val_start, test_start)
    c["data"]["test_start"] = cfg["data"]["val_start"]
    c["data"]["test_end"] = cfg["data"]["test_start"]
    feat = preprocessing.get_features(c)
    dfs = preprocessing.get_data(data_dir=c["data"]["path"], config=c, freq=c["data"]["freq"],
                                 features=feat, files_key="test_files")
    if not dfs:
        return None, None
    sid, df = next(iter(dfs.items()))
    prep, _ = preprocessing.pipeline(data=df, config=c, known_cols=feat["known"],
                                     observed_cols=feat["observed"],
                                     static_cols=feat["static"], target_col="ghi")
    return prep, df


def ablation(folds=(1, 2, 3), n_stationen: int = 5, arm: str = "solar_tft") -> pd.DataFrame:
    zeilen = []
    for fold in folds:
        modell, cfg, _ = lade_modell(fold, arm)
        stationen = list(cfg["data"]["val_files"])[:n_stationen]
        for station in stationen:
            prep, _ = _station_daten(cfg, station)
            if prep is None or prep.get("X_test") is None or not len(prep.get("y_test", [])):
                continue
            X, y = prep["X_test"], prep["y_test"]
            obs = torch.tensor(X["observed"], dtype=torch.float32)
            kno = torch.tensor(X["known"], dtype=torch.float32)
            sta = torch.tensor(X["static"], dtype=torch.float32)

            def lauf(o):
                with torch.no_grad():
                    return modell(o, kno, sta).numpy()

            p_echt = lauf(obs)
            p_null = lauf(torch.zeros_like(obs))
            g = torch.Generator().manual_seed(0)
            p_perm = lauf(obs[torch.randperm(len(obs), generator=g)])

            for j, ziel in enumerate(preprocessing.get_target_cols(cfg)):
                yt = y[:, :, j] if y.ndim == 3 else y
                for name, a, b in BLOECKE:
                    def rmse(p):
                        d = p[:, a:b, j] - yt[:, a:b]
                        return float(np.sqrt(np.nanmean(d ** 2)))
                    zeilen.append(dict(fold=fold, station=station, target=ziel, block=name,
                                       n=len(yt), rmse_echt=rmse(p_echt),
                                       rmse_null=rmse(p_null), rmse_perm=rmse(p_perm)))
            print(f"  fold {fold} {station}: {len(yt)} Laeufe", flush=True)
    return pd.DataFrame(zeilen)


def autokorrelation(fold: int = 1, n_stationen: int = 4, arm: str = "solar_tft") -> pd.DataFrame:
    """Residuum am letzten Messzeitpunkt vor dem Lauf gegen das Residuum je Lead."""
    cfg = tools.load_config(f"configs/{arm}/config_{arm}_fold{fold}.yaml")
    cfg["model"]["name"] = "tft"
    meta = pd.read_csv(REPO / "data/stations_master.csv", dtype={"station_id": str}).set_index("station_id")
    schritt = pd.Timedelta("30min")
    zeilen = []
    for station in list(cfg["data"]["val_files"])[:n_stationen]:
        _, df = _station_daten(cfg, station)
        if df is None:
            continue
        kurz = station.replace("synth_", "").replace(".csv", "")
        res = df["ghi"].droplevel(["starttime", "forecasttime"])
        res = res[~res.index.duplicated(keep="first")].sort_index()
        z = meta.loc[kurz]
        geo = solar_geometry(pd.DatetimeIndex(res.index), float(z["latitude"]),
                             float(z["longitude"]), float(z["station_height"]), freq="30min")
        cs = pd.Series(geo["ghi_clearsky"].values, index=res.index)
        laeufe = pd.DatetimeIndex(sorted(df.index.get_level_values("starttime").unique()))
        vor = res.reindex(laeufe - schritt).to_numpy()
        for lead_schritte in range(0, 96, 2):
            ziel = res.reindex(laeufe + lead_schritte * schritt).to_numpy()
            hell = cs.reindex(laeufe + lead_schritte * schritt).to_numpy() > 50
            g = np.isfinite(vor) & np.isfinite(ziel) & hell
            if g.sum() > 30:
                zeilen.append(dict(station=kurz, lead_h=lead_schritte / 2,
                                   n=int(g.sum()), r=float(np.corrcoef(vor[g], ziel[g])[0, 1])))
    return pd.DataFrame(zeilen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--stationen", type=int, default=5, help="Stationen je Fold")
    ap.add_argument("--arm", default="solar_tft")
    ap.add_argument("--autokorrelation", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.autokorrelation:
        a = autokorrelation(args.folds[0], args.stationen, args.arm)
        p = OUT / "solar_tft_residuum_autokorrelation.csv"
        a.to_csv(p, index=False)
        print(a.groupby("lead_h")["r"].mean().head(14).round(3).to_string())
        print(f"[ok] {p}")
        return

    print(f"[i] Ablation des observed-Fensters, {args.stationen} Stationen je Fold …")
    d = ablation(tuple(args.folds), args.stationen, args.arm)
    p = OUT / "solar_tft_ablation_observed.csv"
    d.to_csv(p, index=False)
    z = (d.groupby(["target", "block"], sort=False)[["rmse_echt", "rmse_null", "rmse_perm"]]
           .mean().reset_index())
    z["delta_null"] = z["rmse_null"] - z["rmse_echt"]
    z["delta_perm"] = z["rmse_perm"] - z["rmse_echt"]
    print(z.round(2).to_string(index=False))
    print(f"[ok] {p}")


if __name__ == "__main__":
    main()
