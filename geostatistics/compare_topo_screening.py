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


# Die Suffixe A/B/C bedeuten NICHT pro Architektur dasselbe, deshalb wird hier
# nach Wirkungskanal aufgeschluesselt statt nach Buchstabe:
#
#   Kanal                     DCRNN   MTGNN   WaveNet
#   kein Topo                 topoA   topoA   topoA
#   Adjazenz/Graphstruktur    --      topoB   topoB
#   Feature-Strom             topoB   topoC   topoC
#
# DCRNN hat keinen Adjazenz-Arm: sein Stationsgraph reduziert edge_attr auf
# exp(-d^2/sigma^2) aus Spalte 0, es gibt dort also keinen Mechanismus fuer
# Kanten-Features. Das ist Li et al. 2018 wie publiziert und bleibt so; ein
# gelernter Kanten-Bias waere eine eigene Ablation.
# DCRNNs Feature-Strom-Arm heisst topoB, weil station.static ohnehin an jedem
# Zeitschritt in den DCGRU-Input konkateniert wird.
MODELS = [
    #  Label                 Stem                             kein Topo  Adjazenz  Feature-Strom
    ("DCRNN   GRID fold1", "wind_dcrnn_fold1_dcrnn",         "topoA",   None,     "topoB"),
    ("MTGNN   GRID fold1", "wind_mtgnn_nwp_fold1_mtgnn",     "topoA",   "topoB",  "topoC"),
    ("WaveNet GRID fold1", "wind_wavenet_nwp_fold1_wavenet", "topoA",   "topoB",  "topoC"),
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


def load(stem: str, arm: str | None):
    """Neuestes Pickle fuer einen Arm; (None, 'n/a') wenn der Arm nicht existiert."""
    if arm is None:
        return None, "n/a"
    f = newest(f"{stem}_{arm}")
    if f is None:
        return None, f"laeuft ({arm})"
    return pickle.load(open(f, "rb")), None


def main() -> None:
    hdr = (f"{'Modell':20s} {'ohne Topo':>10s} | {'Adjazenz':>10s} {'vs A':>8s} "
           f"| {'Feat.-Strom':>11s} {'vs A':>8s}  Kontrolle")
    print(hdr)
    print("-" * len(hdr))

    for label, stem, arm_none, arm_adj, arm_feat in MODELS:
        base, base_note = load(stem, arm_none)
        r_base = base.get("best_val_rmse") if base else None
        notes = []

        cells = [f"{r_base:10.4f}" if r_base else f"{'—':>10s}"]
        for arm, width in ((arm_adj, 10), (arm_feat, 11)):
            d, note = load(stem, arm)
            if note:
                notes.append(note)
            r = d.get("best_val_rmse") if d else None
            cells.append(f"{r:{width}.4f}" if r else f"{'—':>{width}}")
            if r and r_base:
                cells.append(f"{(r - r_base) / r_base * 100:+7.2f}%")
            else:
                cells.append(f"{'—':>8s}")
            # Kontrolle: nur die Topo-Features duerfen sich unterscheiden
            if d and base:
                w = check(base, d)
                if w:
                    notes.append(f"{arm}: {w}")

        if base_note:
            notes.insert(0, base_note)
        status = " | ".join(n for n in notes if n and n != "n/a") or "ok"
        print(f"{label:20s} {cells[0]} | {cells[1]} {cells[2]} | {cells[3]} {cells[4]}  {status}")

    print("\nSpalten sind Wirkungskanaele, nicht Suffixe — die Suffixe A/B/C bedeuten")
    print("pro Architektur Verschiedenes (siehe MODELS im Quelltext):")
    print("  Adjazenz     = Topo formt die gelernte Graphstruktur (MTGNN/WaveNet topoB)")
    print("  Feat.-Strom  = Topo als Praediktor im Zeitreihen-Input (DCRNN topoB, sonst topoC)")
    print("DCRNN hat keinen Adjazenz-Arm: sein Stationsgraph reduziert edge_attr auf")
    print("die Distanzspalte, ein Kanten-Feature-Mechanismus existiert dort nicht.")
    print("Negatives Delta = besser als ohne Topo.")
    print("Bei 'SPLIT WEICHT AB' oder 'CONFIG-DIFF' ist der Vergleich nicht verwertbar.")


if __name__ == "__main__":
    sys.exit(main())
