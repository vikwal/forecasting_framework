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


# Arme: A = ohne Topo, B = Topo nur in der Adjazenz (emb_mlp / edge_fc),
# C = Topo zusaetzlich als zeitkonstante Input-Kanaele (--broadcast-topo).
# DCRNN hat kein C: dort laeuft station.static ohnehin durch jeden Zeitschritt,
# B ist dort also bereits der Feature-Strom-Arm.
MODELS = [
    ("DCRNN   GRID fold1", "wind_dcrnn_fold1_dcrnn",       False),
    ("MTGNN   GRID fold1", "wind_mtgnn_nwp_fold1_mtgnn",   True),
    ("WaveNet GRID fold1", "wind_wavenet_nwp_fold1_wavenet", True),
]


def check(ref: dict, other: dict) -> str:
    """Sind zwei Arme ausser den Topo-Features wirklich identisch konfiguriert?"""
    notes = []
    if ref.get("train_ids") != other.get("train_ids") or ref.get("val_ids") != other.get("val_ids"):
        notes.append("SPLIT WEICHT AB")
    ca, cb = ref.get("config", {}), other.get("config", {})
    diff = [k for k in set(ca) | set(cb)
            if k not in IGNORE and ca.get(k, "<>") != cb.get(k, "<>")]
    if diff:
        notes.append(f"CONFIG-DIFF {sorted(diff)}")
    return " | ".join(notes)


def main() -> None:
    print(f"{'Modell':20s} {'RMSE A':>8s} {'RMSE B':>8s} {'RMSE C':>8s} "
          f"{'B vs A':>8s} {'C vs A':>8s}  Kontrolle")
    print("-" * 92)
    for label, stem, has_c in MODELS:
        arms = {}
        for arm in ("topoA", "topoB", "topoC"):
            if arm == "topoC" and not has_c:
                continue
            f = newest(f"{stem}_{arm}")
            arms[arm] = pickle.load(open(f, "rb")) if f else None

        missing = [a for a, d in arms.items() if d is None]
        a = arms.get("topoA")
        cells, deltas = [], []
        for arm in ("topoA", "topoB", "topoC"):
            d = arms.get(arm)
            r = d.get("best_val_rmse") if d else None
            cells.append(f"{r:8.4f}" if r else f"{'—':>8s}")
            if arm != "topoA":
                if r and a and a.get("best_val_rmse"):
                    ra = a["best_val_rmse"]
                    deltas.append(f"{(r - ra) / ra * 100:+7.2f}%")
                else:
                    deltas.append(f"{'—':>8s}")

        notes = [check(a, d) for arm, d in arms.items()
                 if arm != "topoA" and a and d] if a else []
        notes = [n for n in notes if n]
        status = " | ".join(notes) if notes else ("ok" if not missing else "")
        if missing:
            status = (status + " | " if status else "") + f"laeuft noch: {','.join(missing)}"

        print(f"{label:20s} {cells[0]} {cells[1]} {cells[2]} "
              f"{deltas[0]:>8s} {deltas[1]:>8s}  {status}")

    print("\nArme:  A = ohne Topo | B = Topo nur in der Adjazenz | "
          "C = Topo zusaetzlich im Feature-Strom")
    print("Negatives Delta = besser als Arm A.")
    print("DCRNN hat kein C — dort ist station.static schon Teil des Feature-Stroms.")
    print("Bei 'SPLIT WEICHT AB' oder 'CONFIG-DIFF' ist der Vergleich nicht verwertbar.")


if __name__ == "__main__":
    sys.exit(main())
