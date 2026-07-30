#!/usr/bin/env python3
"""
compare_topo_screening.py — Arm A (ohne Topo) gegen Arm B (alle Topo-Features).

Vergleicht die Result-Pickles der --station-node-features none/all Laeufe und
prueft dabei, dass die beiden Arme wirklich nur in den Topo-Features
differieren: gleiche Stations-Splits, gleiche Hyperparameter. Ohne diese
Kontrolle ist der RMSE-Vergleich wertlos (siehe die verworfenen _topo_test-
Laeufe vom Juli, die sich in 18 Config-Werten und im Split unterschieden).

    python geostatistics/compare_topo_screening.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
IGNORE = {"station_node_features"}   # der beabsichtigte Unterschied


def newest(stem: str):
    hits = sorted(RESULTS.glob(f"{stem}_*.pkl"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def main() -> None:
    pairs = [
        ("DCRNN   GRID fold1", "wind_dcrnn_fold1_dcrnn_topoA",      "wind_dcrnn_fold1_dcrnn_topoB"),
        ("MTGNN   GRID fold1", "wind_mtgnn_nwp_fold1_mtgnn_topoA",  "wind_mtgnn_nwp_fold1_mtgnn_topoB"),
        ("WaveNet GRID fold1", "wind_wavenet_nwp_fold1_wavenet_topoA", "wind_wavenet_nwp_fold1_wavenet_topoB"),
    ]
    print(f"{'Modell':20s} {'RMSE A':>8s} {'RMSE B':>8s} {'Delta':>8s}  {'Ep A/B':>9s}  Kontrolle")
    print("-" * 78)
    for label, sa, sb in pairs:
        fa, fb = newest(sa), newest(sb)
        if not fa or not fb:
            miss = " / ".join(x for x, f in ((sa, fa), (sb, fb)) if not f)
            print(f"{label:20s} {'—':>8s} {'—':>8s} {'—':>8s}  {'—':>9s}  laeuft noch ({miss})")
            continue
        a, b = pickle.load(open(fa, "rb")), pickle.load(open(fb, "rb"))
        ra, rb = a.get("best_val_rmse"), b.get("best_val_rmse")

        notes = []
        if a.get("train_ids") != b.get("train_ids") or a.get("val_ids") != b.get("val_ids"):
            notes.append("SPLIT WEICHT AB")
        ca, cb = a.get("config", {}), b.get("config", {})
        diff = [k for k in set(ca) | set(cb)
                if k not in IGNORE and ca.get(k, "<>") != cb.get(k, "<>")]
        if diff:
            notes.append(f"CONFIG-DIFF {sorted(diff)}")
        ok = "ok" if not notes else " | ".join(notes)

        d = f"{(rb - ra) / ra * 100:+.2f}%" if (ra and rb) else "—"
        print(f"{label:20s} {ra:8.4f} {rb:8.4f} {d:>8s}  "
              f"{str(a.get('stopped_epoch'))+'/'+str(b.get('stopped_epoch')):>9s}  {ok}")
    print("\nNegatives Delta = Arm B (mit Topo) ist besser.")
    print("Bei 'SPLIT WEICHT AB' oder 'CONFIG-DIFF' ist der Vergleich nicht verwertbar.")


if __name__ == "__main__":
    sys.exit(main())
