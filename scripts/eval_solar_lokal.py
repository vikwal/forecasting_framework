#!/usr/bin/env python3
"""Lokale gegen globale Solar-Modelle: bringt Pooling ueber Stationen etwas?

Drei Arme, erzeugt von ``scripts/make_solar_lokal_configs.py``:

===================  ==================  ====================  ==============
Arm                  Training            Auswertung            Laeufe
===================  ==================  ====================  ==============
``lokal``            je 1 Station        dieselbe Station      62
``global_trans``     alle 62 Stationen   dieselben 62          1
``global_induk``     41 Pool-Stationen   21 andere, zero-shot  1 (vorhanden)
===================  ==================  ====================  ==============

Die ersten beiden stehen auf derselben Stationsmenge und derselben Zeitachse —
ihr Abstand misst allein den Pooling-Effekt. Der dritte ist kein direkter
Vergleichspartner (andere Stationsmenge), sondern der Bezugspunkt dafuer, was
die Generalisierung auf unbekannte Stationen kostet; er wird deshalb nur auf
seinen eigenen 21 Zielstationen gegen die beiden anderen gehalten.

Alle Bausteine — Beobachtungsmaske, Rueckrechnung aus dem Residuumsraum,
RMSE/R2 je Station, Holm-Korrektur — kommen aus ``eval_solar_arch.py``, damit
beide Auswertungen nachweislich dasselbe messen.

**Transduktiv, und das ist der Punkt:** in beiden neuen Armen ist die
Auswertungsstation im Training bekannt, getrennt allein durch die Zeitgrenze.
Ein Schluss auf unbekannte Stationen ist daraus nicht moeglich.

Aufruf:
    frcst/bin/python scripts/eval_solar_lokal.py --suffix v1
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from eval_solar_arch import (  # noqa: E402
    SCHLUESSEL, ZIELE, lade_messreihen, lade_tft, r2_je_station,
    rmse_je_station, spalte_aus_reihen, holm,
)
from geostatistics.stdrun.make_stdhp_figures import norm_station  # noqa: E402

ERG = REPO / "results/solar"
OUT = REPO / "data/test_results"


def juengstes(muster: str) -> Path | None:
    """Neuestes Pickle zu einem Namensmuster (mehrere Laeufe je Station moeglich)."""
    treffer = sorted(glob.glob(str(ERG / muster)))
    return Path(treffer[-1]) if treffer else None


def lade_arm_lokal(stationen: list[str], suffix: str, ziel: dict) -> pd.DataFrame:
    """Die 62 Einzel-Pickles zu einem long-Frame zusammenlegen."""
    teile, fehlend = [], []
    for sid in stationen:
        p = juengstes(f"cl_m-tft_*_solar_lokal_{sid}_{suffix}_lokal_{sid}_*.pkl")
        if p is None:
            fehlend.append(sid)
            continue
        teile.append(lade_tft(p, ziel, {sid}))
    if fehlend:
        print(f"  !! {len(fehlend)} Station(en) ohne Ergebnis: {', '.join(fehlend)}")
    if not teile:
        raise SystemExit("[FATAL] Kein einziges lokales Ergebnis gefunden.")
    return pd.concat(teile, ignore_index=True)


def gepaart(a: pd.Series, b: pd.Series, name_a: str, name_b: str) -> dict:
    """Wilcoxon ueber die Stationen; negative Differenz = A besser."""
    d = (a - b).dropna()
    if len(d) < 6:
        return {}
    _, p = wilcoxon(d)
    return dict(A=name_a, B=name_b, n=len(d), median_diff=float(d.median()),
                mittel_diff=float(d.mean()),
                anteil_A_besser=float((d < 0).mean()), p_raw=float(p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="v1")
    ap.add_argument("--config", default="configs/solar_lokal/config_solar_global_trans.yaml")
    ap.add_argument("--induktiv", default=None,
                    help="Pickle des Fold-1-Laufs (Default: juengstes solar_tft_fold1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from utils.tools import load_config  # noqa: E402  (braucht DATA_ROOT)
    cfg = load_config(str(REPO / args.config))
    d = cfg["data"]
    stationen = [str(s).zfill(5) for s in d["files"]]
    print(f"{len(stationen)} Stationen, Auswertung {d['test_start']} … {d['test_end']}\n")

    print("Messreihen und Beobachtungsmaske …")
    beobachtet, ziel = lade_messreihen(
        stationen, Path(d["path"]), Path(d["interpol_path"]), d["stations_master"])

    print("Ergebnisse laden …")
    quellen: dict[str, pd.DataFrame] = {}
    quellen["lokal"] = lade_arm_lokal(stationen, args.suffix, ziel)
    print(f"  lokal          {len(quellen['lokal']):>9,} Zeilen")

    p_glob = juengstes(f"cl_m-tft_*_solar_global_trans_{args.suffix}_global_trans_*.pkl")
    if p_glob is None:
        raise SystemExit("[FATAL] Kein Ergebnis des global-transduktiven Laufs gefunden.")
    quellen["global"] = lade_tft(p_glob, ziel, set(stationen))
    print(f"  global         {len(quellen['global']):>9,} Zeilen")

    # ── Filter und gemeinsame Menge ──────────────────────────────────────
    print("\nFilter (imputierte Zielpositionen elementweise):")
    for name, v in quellen.items():
        maske = spalte_aus_reihen(v, beobachtet, "valid_time")
        echt = np.nan_to_num(maske, nan=0.0).astype(bool)
        brauchbar = echt & ~np.isnan(v["pred"].to_numpy()) & ~np.isnan(v["gt"].to_numpy())
        print(f"  {name:14s} {brauchbar.sum():>9,} von {len(v):>9,} "
              f"({100 * brauchbar.mean():5.2f} %)")
        quellen[name] = v.loc[brauchbar].copy()

    gemeinsam = None
    for v in quellen.values():
        s = v.set_index(SCHLUESSEL).index
        gemeinsam = s if gemeinsam is None else gemeinsam.intersection(s)
    print(f"\nGemeinsame Menge: {len(gemeinsam):,} Zeilen")

    je_st, je_st_r2, zeilen = {}, {}, []
    for name, v in quellen.items():
        g = v.set_index(SCHLUESSEL).loc[gemeinsam].reset_index()
        je_st[name] = rmse_je_station(g)
        je_st_r2[name] = r2_je_station(g)
        nwp = rmse_je_station(g, "nwp_ref")
        for target in ZIELE:
            r = je_st[name].loc[target]
            zeilen.append(dict(arm=name, target=target, n_stationen=len(r),
                               rmse=float(r.mean()),
                               r2=float(je_st_r2[name].loc[target].mean()),
                               rmse_nwp=float(nwp.loc[target].mean()),
                               skill_nwp=float(1 - (r / nwp.loc[target]).mean())))
    tab = pd.DataFrame(zeilen).sort_values(["target", "rmse"])
    print("\nStationsmittel auf der gemeinsamen Menge (W/m²):")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    # ── Pooling-Effekt, gepaart ueber die Stationen ──────────────────────
    print("\nPooling-Effekt (lokal gegen global), gepaart ueber die Stationen:")
    sig = []
    for target in ZIELE:
        r = gepaart(je_st["lokal"].loc[target], je_st["global"].loc[target],
                    "lokal", "global")
        if r:
            r["target"] = target
            r["groesse"] = "RMSE"
            sig.append(r)
    if sig:
        st = holm(pd.DataFrame(sig))
        print(st.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
        print("  negative Differenz = lokal besser")

    # ── Haengt der Vorteil an der Datenlage der Station? ─────────────────
    anteil = pd.Series({sid: float(np.mean([beobachtet[sid][z].mean() for z in ZIELE]))
                        for sid in stationen})
    print("\nHaengt der Unterschied an der Datenlage? "
          "(Anteil echter Messwerte je Station gegen die RMSE-Differenz)")
    for target in ZIELE:
        diff = (je_st["lokal"].loc[target] - je_st["global"].loc[target]).dropna()
        gem = anteil.reindex(diff.index).dropna()
        d2 = diff.reindex(gem.index)
        if len(gem) >= 6:
            rho = float(np.corrcoef(gem.to_numpy(), d2.to_numpy())[0, 1])
            drittel = gem.quantile([0.0, 1/3, 2/3, 1.0]).to_numpy()
            print(f"  {target}: Korrelation(Messanteil, RMSE_lokal-RMSE_global) = {rho:+.3f}")
            for lo, hi, lab in ((drittel[0], drittel[1], "schlechteste Datenlage"),
                                (drittel[1], drittel[2], "mittlere"),
                                (drittel[2], drittel[3] + 1e-9, "beste")):
                sel = d2[(gem >= lo) & (gem <= hi)]
                if len(sel):
                    print(f"      {lab:22s} n={len(sel):2d}  "
                          f"Messanteil {gem[sel.index].mean():.3f}  "
                          f"ΔRMSE {sel.mean():+7.3f} W/m²")

    # ── Was kostet die Generalisierung? (nur auf den 21 Fold-1-Zielen) ───
    p_ind = (Path(args.induktiv) if args.induktiv else
             juengstes("cl_m-tft_*_solar_tft_fold1_*.pkl"))
    if p_ind is not None:
        import yaml
        with open(REPO / "configs/solar_folds.yaml") as fh:
            folds = yaml.safe_load(fh)
        ziel21 = [str(s).zfill(5) for s in folds["spatial_fold1"]["val_files"]]
        print(f"\nPreis der Generalisierung — nur die {len(ziel21)} zero-shot-Stationen "
              f"des Fold-1-Laufs ({p_ind.name}):")
        ind = lade_tft(p_ind, ziel, set(ziel21))
        maske = spalte_aus_reihen(ind, beobachtet, "valid_time")
        echt = np.nan_to_num(maske, nan=0.0).astype(bool)
        ind = ind.loc[echt & ~np.isnan(ind["pred"].to_numpy())
                      & ~np.isnan(ind["gt"].to_numpy())].copy()
        idx_ind = ind.set_index(SCHLUESSEL).index
        gem21 = gemeinsam.intersection(idx_ind)
        print(f"  gemeinsame Menge auf diesen Stationen: {len(gem21):,} Zeilen")
        rows = []
        for name, v in list(quellen.items()) + [("induktiv", ind)]:
            g = v.set_index(SCHLUESSEL).loc[gem21].reset_index()
            r = rmse_je_station(g)
            r2 = r2_je_station(g)
            for target in ZIELE:
                rows.append(dict(arm=name, target=target,
                                 rmse=float(r.loc[target].mean()),
                                 r2=float(r2.loc[target].mean())))
        print(pd.DataFrame(rows).sort_values(["target", "rmse"]).to_string(
            index=False, float_format=lambda x: f"{x:9.4f}"))

    praefix = args.out or f"solar_lokal_{args.suffix}"
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / f"{praefix}_metriken.csv", index=False)
    pd.DataFrame(je_st).to_csv(OUT / f"{praefix}_rmse_je_station.csv")
    pd.DataFrame(je_st_r2).to_csv(OUT / f"{praefix}_r2_je_station.csv")
    print(f"\nGeschrieben: {OUT}/{praefix}_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
