#!/usr/bin/env python3
"""
get_trainvaltest_stations.py
----------------------------
Splits all stations from stations_master.csv into train / val / test
using a two-pass sequential Kennard-Stone algorithm.

Strategy: Sequential Kennard-Stone
------------------------------------
  Pass 1 — Test set (from all N stations):
    Run Kennard-Stone on all 203 stations → select the 50 that achieve
    maximum geographic coverage of the full station network.

  Pass 2 — Val set (from remaining N - n_test stations):
    Run Kennard-Stone again on the leftover 153 stations → select the
    50 that achieve maximum coverage of the *remaining* space.

  Train set — the leftover 103 stations:
    These naturally fill the geographic gaps between the test and val
    stations, ensuring excellent spatial interleaving.

Why this works:
  - Test stations cover the entire domain evenly (no regional blind spots).
  - Val stations cover the remaining space evenly.
  - Train stations are scattered *between* test and val → every test/val
    station has a training neighbour nearby. No same-type clusters.

Usage:
    python data/get_trainvaltest_stations.py
    python data/get_trainvaltest_stations.py --n-test 50 --n-val 50
    python data/get_trainvaltest_stations.py --plot    # requires matplotlib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Kennard-Stone
# ---------------------------------------------------------------------------

def geodetic_distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Full N×N geodetic distance matrix (km) using geopy.distance.geodesic
    (Vincenty formula on the WGS-84 ellipsoid).
    """
    from geopy.distance import geodesic
    N = len(lats)
    dist = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = geodesic((lats[i], lons[i]), (lats[j], lons[j])).km
            dist[i, j] = d
            dist[j, i] = d
    return dist


def convex_hull_indices(lats: np.ndarray, lons: np.ndarray) -> list[int]:
    """
    Return indices of the stations that form the convex hull (outermost boundary).
    These are reserved for training so the model never has to extrapolate.
    """
    from scipy.spatial import ConvexHull
    coords = np.stack([lons, lats], axis=1)   # (N, 2)
    hull = ConvexHull(coords)
    return sorted(hull.vertices.tolist())


def kennard_stone(dist_matrix: np.ndarray, n_select: int,
                  candidate_indices: list[int] | None = None) -> list[int]:
    """
    Kennard-Stone algorithm on a sub-set of the full distance matrix.

    Parameters
    ----------
    dist_matrix      : full N×N distance matrix (km)
    n_select         : how many points to select
    candidate_indices: indices (into dist_matrix) to consider as candidates.
                       If None, all N points are candidates.

    Returns
    -------
    List of selected indices (into dist_matrix), in selection order.
    """
    if candidate_indices is None:
        candidate_indices = list(range(dist_matrix.shape[0]))

    cands = list(candidate_indices)
    assert n_select <= len(cands), \
        f"Cannot select {n_select} from only {len(cands)} candidates"

    # Seed: pair with maximum pairwise distance among candidates
    sub = dist_matrix[np.ix_(cands, cands)]
    r, c = np.unravel_index(np.argmax(sub), sub.shape)
    selected = [cands[r], cands[c]]
    remaining = [x for x in cands if x not in selected]

    while len(selected) < n_select:
        # min distance from each remaining point to any already-selected point
        d_to_sel = dist_matrix[remaining][:, selected]   # (n_rem, n_sel)
        min_d    = d_to_sel.min(axis=1)                   # (n_rem,)
        best_loc = int(np.argmax(min_d))
        best_idx = remaining[best_loc]
        selected.append(best_idx)
        remaining.pop(best_loc)

    return selected


def simultaneous_ks(
    dist_matrix: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    candidate_indices: list[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Multi-set Kennard-Stone: assigns all stations to train/val/test simultaneously.

    Every new station maximises its minimum distance to ALL already-assigned
    stations (regardless of which set they belong to).  Assignment to a
    specific set follows a proportional schedule: the set that is furthest
    behind its target fraction receives the next station.

    This guarantees maximum pairwise separation *across* all sets — no two
    stations of any type cluster together unless forced by the data geometry.

    Parameters
    ----------
    dist_matrix       : full N x N geodetic distance matrix
    n_train/val/test  : target counts for each set
    candidate_indices : pool to draw from (hull stations already excluded)

    Returns
    -------
    (train_indices, val_indices, test_indices)  -- indices into dist_matrix
    """
    if candidate_indices is None:
        candidate_indices = list(range(dist_matrix.shape[0]))

    cands = list(candidate_indices)
    total = n_train + n_val + n_test
    assert total == len(cands), \
        f"n_train+n_val+n_test ({total}) must equal len(candidates) ({len(cands)})"

    targets  = {"tr": n_train, "va": n_val, "te": n_test}
    assigned = {"tr": [], "va": [], "te": []}
    all_assigned: list[int] = []

    # Seed: the pair of candidates with maximum distance.
    # Assign one to train (largest set) and one to test so the two most
    # extreme points anchor different sets.
    sub = dist_matrix[np.ix_(cands, cands)]
    r, c = np.unravel_index(np.argmax(sub), sub.shape)
    seed_a, seed_b = cands[r], cands[c]
    assigned["tr"].append(seed_a)
    assigned["te"].append(seed_b)
    all_assigned.extend([seed_a, seed_b])
    remaining = [x for x in cands if x not in {seed_a, seed_b}]

    # min-distance vector: for each remaining point its distance to the
    # nearest already-assigned point (updated incrementally).
    min_d = dist_matrix[remaining][:, all_assigned].min(axis=1).copy()

    while remaining:
        # Which set gets the next station? (proportional fill)
        # "debt" = how far behind a set is relative to its target fraction.
        n_done = len(all_assigned)
        debt = {
            s: targets[s] / total - len(assigned[s]) / n_done
            for s in targets
            if len(assigned[s]) < targets[s]
        }
        if not debt:
            break
        next_set = max(debt, key=debt.__getitem__)

        # Pick the remaining station furthest from ALL assigned stations
        best_loc = int(np.argmax(min_d))
        best_idx = remaining[best_loc]

        assigned[next_set].append(best_idx)
        all_assigned.append(best_idx)
        remaining.pop(best_loc)
        min_d = np.delete(min_d, best_loc)

        # Incrementally update min-distances
        if remaining:
            d_new = dist_matrix[remaining, best_idx]
            min_d = np.minimum(min_d, d_new)

    return assigned["tr"], assigned["va"], assigned["te"]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def format_yaml_list(ids: list[str], items_per_row: int = 10) -> str:
    rows = []
    for i in range(0, len(ids), items_per_row):
        chunk = ids[i:i + items_per_row]
        rows.append("          " + ", ".join(f"'{s}'" for s in chunk) + ",")
    rows[-1] = rows[-1].rstrip(",")
    return "[\n" + "\n".join(rows) + "]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential Kennard-Stone geographic station split"
    )
    parser.add_argument("--stations-master", default="data/stations_master.csv",
                        help="Path to stations_master.csv (all rows are used)")
    parser.add_argument("--n-test", type=int, default=50,
                        help="Number of test stations (default: 50)")
    parser.add_argument("--n-val",  type=int, default=50,
                        help="Number of val stations (default: 50)")
    parser.add_argument("--mode", choices=["test-val-train", "test-train-val",
                                            "train-val-test", "simultaneous"],
                        default="simultaneous",
                        help=(
                            "KS selection order:\n"
                            "  simultaneous   : all 3 sets grow at once — max distance across all sets\n"
                            "  test-val-train : KS test -> KS val -> train=leftover\n"
                            "  test-train-val : KS test -> KS train -> val=leftover\n"
                            "  train-val-test : KS train+val pool -> KS val -> test=leftover"
                        ))
    parser.add_argument("--plot", action="store_true",
                        help="Save map to data/station_split_map.png (needs matplotlib)")
    args = parser.parse_args()

    # ── Load stations ────────────────────────────────────────────────────────
    meta = (
        pd.read_csv(args.stations_master, dtype={"station_id": str})
        .assign(station_id=lambda d: d["station_id"].str.zfill(5))
        .set_index("station_id")
    )
    ids  = list(meta.index)
    lats = meta["latitude"].values.astype(float)
    lons = meta["longitude"].values.astype(float)
    N    = len(ids)

    assert args.n_test + args.n_val < N, \
        f"n_test ({args.n_test}) + n_val ({args.n_val}) must be < total stations ({N})"

    n_train = N - args.n_test - args.n_val
    print(f"Loaded {N} stations → test={args.n_test}  val={args.n_val}  train={n_train}")

    # ── Distance matrix (computed once) ──────────────────────────────────────
    print("Computing geodetic distance matrix (WGS84) …", end=" ", flush=True)
    dist = geodetic_distance_matrix(lats, lons)
    print("done")

    # ── Convex hull — outermost stations are reserved for training ────────────
    # This ensures the model is never asked to extrapolate beyond its training
    # domain: every val/test station has training stations on all sides.
    hull_idx = convex_hull_indices(lats, lons)
    hull_set = set(hull_idx)
    interior = [i for i in range(N) if i not in hull_set]
    print(f"Convex hull: {len(hull_idx)} boundary stations → reserved for training")
    print(f"Interior pool: {len(interior)} stations available for val/test selection")

    assert len(interior) >= args.n_test + args.n_val, (
        f"Not enough interior stations ({len(interior)}) for "
        f"n_test={args.n_test} + n_val={args.n_val}. "
        f"Reduce targets or use --no-hull-constraint."
    )

    if args.mode == "simultaneous":
        # All 3 sets grow at once: every pick goes to the globally furthest
        # remaining station; the set that is most behind its target gets it.
        # Hull stations (boundary) are always added to train after.
        n_interior_train = n_train - len(hull_idx)
        print(f"Mode simultaneous: assigning {len(interior)} interior stations "
              f"({n_interior_train} train / {args.n_val} val / {args.n_test} test) "
              f"via multi-set KS ...")
        tr_int, val_global, test_global = simultaneous_ks(
            dist, n_interior_train, args.n_val, args.n_test,
            candidate_indices=interior,
        )
        train_global = hull_idx + tr_int
        val_ids      = [ids[i] for i in val_global]
        test_ids     = [ids[i] for i in test_global]
        test_set     = set(test_global)
        val_set      = set(val_global)
        train_ids    = [ids[i] for i in train_global]

    elif args.mode == "train-val-test":
        # Pass 1: KS selects the train+val POOL from interior.
        #   pool_size = (n_train + n_val) - len(hull)  [hull already in train]
        #   test      = interior stations NOT in the pool (the "leftover" 50)
        # Pass 2: KS selects val from that pool → remaining pool = train-interior.
        n_pool = (n_train + args.n_val) - len(hull_idx)
        print(f"Mode train-val-test: KS pass 1 — selecting {n_pool} train+val pool "
              f"from {len(interior)} interior stations …")
        pool_global = kennard_stone(dist, n_select=n_pool,
                                    candidate_indices=interior)
        pool_set    = set(pool_global)
        test_global = [i for i in interior if i not in pool_set]
        test_ids    = [ids[i] for i in test_global]
        test_set    = set(test_global)

        print(f"Mode train-val-test: KS pass 2 — selecting {args.n_val} val stations "
              f"from {n_pool} pool stations …")
        val_global   = kennard_stone(dist, n_select=args.n_val,
                                     candidate_indices=pool_global)
        val_set      = set(val_global)
        train_global = hull_idx + [i for i in pool_global if i not in val_set]

    else:
        # All other modes: pass 1 always selects test from interior.
        print(f"Kennard-Stone pass 1 — selecting {args.n_test} test stations "
              f"from {len(interior)} interior stations …")
        test_global = kennard_stone(dist, n_select=args.n_test,
                                    candidate_indices=interior)
        test_ids    = [ids[i] for i in test_global]
        test_set    = set(test_global)
        after_test  = [i for i in interior if i not in test_set]

        if args.mode == "test-val-train":
            # Pass 2: KS val → train = leftover
            print(f"Mode test-val-train: KS pass 2 — selecting {args.n_val} val stations "
                  f"from {len(after_test)} remaining interior stations …")
            val_global   = kennard_stone(dist, n_select=args.n_val,
                                         candidate_indices=after_test)
            val_set      = set(val_global)
            train_global = [i for i in range(N) if i not in test_set and i not in val_set]

        else:  # test-train-val
            # Pass 2: KS train-interior → val = leftover
            n_train_interior = n_train - len(hull_idx)
            print(f"Mode test-train-val: KS pass 2 — selecting {n_train_interior} "
                  f"additional train stations from {len(after_test)} remaining interior …")
            train_interior = kennard_stone(dist, n_select=n_train_interior,
                                           candidate_indices=after_test)
            train_extra_set = set(train_interior)
            val_global   = [i for i in after_test if i not in train_extra_set]
            val_set      = set(val_global)
            train_global = hull_idx + train_interior

    val_ids   = [ids[i] for i in val_global]
    val_set   = set(val_global)
    train_ids = [ids[i] for i in train_global]

    assert len(train_ids) == n_train
    assert len(set(train_ids) & set(val_ids) & set(test_ids)) == 0

    # ── Statistics ───────────────────────────────────────────────────────────
    def _geo_stats(global_idx, label):
        sub_lats = lats[global_idx]
        sub_lons = lons[global_idx]
        print(f"  {label:6s}: n={len(global_idx):3d}  "
              f"lat=[{sub_lats.min():.2f}, {sub_lats.max():.2f}]  "
              f"lon=[{sub_lons.min():.2f}, {sub_lons.max():.2f}]  "
              f"centroid=({sub_lats.mean():.2f}°N, {sub_lons.mean():.2f}°E)")

    def _avg_nn_km(check_idx, ref_idx):
        """Average nearest-neighbour distance (km) from check → ref."""
        d_sub = dist[np.ix_(check_idx, ref_idx)]
        return float(d_sub.min(axis=1).mean())

    print(f"\nFinal split: {n_train} train | {len(val_ids)} val | {len(test_ids)} test  "
          f"(total: {N})")
    print("Geographic statistics:")
    _geo_stats(train_global, "train")
    _geo_stats(val_global,   "val")
    _geo_stats(test_global,  "test")

    avg_val_nn  = _avg_nn_km(val_global,  train_global)
    avg_test_nn = _avg_nn_km(test_global, train_global)
    print(f"\n  Avg nearest-train distance — val: {avg_val_nn:.1f} km  | "
          f" test: {avg_test_nn:.1f} km")
    print(f"  (lower = val/test stations well interleaved with training stations)")

    # ── YAML output ──────────────────────────────────────────────────────────
    mode_desc = {
        "simultaneous":   "multi-set KS (all 3 sets grow at once)",
        "test-val-train":  "KS test -> KS val -> train=leftover",
        "test-train-val":  "KS test -> KS train -> val=leftover",
        "train-val-test":  "KS train+val pool -> KS val -> test=leftover",
    }[args.mode]
    yaml_out = f"""
# ── Station split: Sequential KS ({n_train} train / {len(val_ids)} val / {len(test_ids)} test) ──
# Generated by data/get_trainvaltest_stations.py  (N={N} stations total)
# Mode: {mode_desc}
# Hull: {len(hull_idx)} boundary stations reserved for training
# Avg nearest-train distance: val={avg_val_nn:.0f} km  test={avg_test_nn:.0f} km

files: {format_yaml_list(train_ids)}

val_files: {format_yaml_list(val_ids)}

test_files: {format_yaml_list(test_ids)}
"""
    print("\n" + "=" * 70)
    print("YAML output (paste into your config):")
    print("=" * 70)
    print(yaml_out)

    # ── Optional plot ────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, ax = plt.subplots(figsize=(10, 8))

            styles = {
                "train": dict(c="#2196F3", marker="o",  s=55,  label=f"Train ({n_train})",        zorder=3),
                "val":   dict(c="#FF9800", marker="^",  s=80,  label=f"Val ({len(val_ids)})",     zorder=4),
                "test":  dict(c="#F44336", marker="s",  s=80,  label=f"Test ({len(test_ids)})",   zorder=5),
            }
            for role, idx_list in [("train", train_global), ("val", val_global), ("test", test_global)]:
                ax.scatter(lons[idx_list], lats[idx_list], **styles[role])

            # Annotate first 5 K-S selections for test (shows the algorithm's order)
            for rank, idx in enumerate(test_global[:5]):
                ax.annotate(f"T{rank+1}", (lons[idx], lats[idx]),
                            fontsize=7, ha="center", va="bottom", color="#F44336")

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"Sequential Kennard-Stone Split  "
                         f"({n_train} train / {len(val_ids)} val / {len(test_ids)} test)")
            ax.legend(loc="upper left")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            out = Path("data/station_split_map.png")
            plt.savefig(out, dpi=150)
            print(f"Map saved to {out}")
            plt.show()
        except ImportError:
            print("[WARN] matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
