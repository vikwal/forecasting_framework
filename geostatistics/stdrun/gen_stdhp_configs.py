#!/usr/bin/env python3
"""
gen_stdhp_configs.py — Generate harmonized "standard hyperparameter" (stdhp)
configs for the end-to-end dry-run of DCRNN / MTGNN / GraphWaveNet without HPO.

Context
-------
The running HPO campaign has 0 COMPLETE trials for 9 of 10 studies, so
--hpo-study auto is not usable yet. Meanwhile the *static* YAML values in the
existing fold configs come from different, historical HPO rounds and disagree
across exactly the axes this dry-run is meant to compare (DCRNN base:
max_epochs 50/patience 10/hidden 64/K_hop 1/grad_accum(batch_size) 64 vs. DCRNN
nwp: 200/15/hidden 128/K_hop 2/grad_accum 8, etc). This script writes new,
harmonized copies of the 30 fold configs so a single set of DCRNN
hyperparameters is used across all 5 DCRNN variants, while the parts of the
config that carry the actual scientific comparison — data splits, time
windows, feature lists, and the variant switches themselves — are inherited
byte-for-byte from the source config.

MTGNN and WaveNet are NOT harmonized (their variants are already identically
parametrized on all shared hyperparameter keys — verified below) — they are
copied unchanged.

Usage
-----
    python geostatistics/stdrun/gen_stdhp_configs.py           # generate + report
    python geostatistics/stdrun/gen_stdhp_configs.py --check   # report only, no writes
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

import yaml

WORK_DIR = Path(__file__).resolve().parent.parent.parent  # forecasting_framework/

# ---------------------------------------------------------------------------
# Harmonized DCRNN hyperparameter set (user-decided; see task spec).
# nwp_out_dim is NOT hardcoded here — it is read from the Variant-A config
# (config_wind_dcrnn_fold1.yaml) at runtime and applied to all 15 configs.
# ---------------------------------------------------------------------------
DCRNN_HARMONIZED: dict = {
    "hidden": 64,
    "num_layers": 1,
    "K_hop": 2,
    "dropout": 0.1,
    "lr": 3.0e-4,
    "weight_decay": 1.0e-3,
    "gradient_clip": 1.0,
    "teacher_forcing_ratio": 0.3,
    "horizon_decay": 0.85,
    "grad_accum": 32,
    "next_n_icond2": 4,
    "next_n_ecmwf": 4,
    "next_n_neighbors": 30,
    "edge_weight_sigma": 0.2,
    "nwp_heads": 4,
    "icond2_feature_mode": "dir_in_deg",
    "ecmwf_feature_mode": "dir_in_deg",
    "max_epochs": 200,
    "patience": 15,
}

# Keys that are explicitly allowed to change (or be added) in the `dcrnn:`
# section of a stdhp config, relative to its source. Used only for the
# self-check that nothing else moved.
DCRNN_ALLOWED_CHANGED_KEYS = set(DCRNN_HARMONIZED) | {"nwp_out_dim", "batch_size"}

# Keys that must NEVER be touched even though they live inside sections we
# otherwise modify (`dcrnn:`) — asserted explicitly as a belt-and-braces check
# in addition to "everything not in DCRNN_ALLOWED_CHANGED_KEYS is untouched".
DCRNN_PROTECTED_KEYS = {
    "files", "val_files", "test_files",
    "path", "nwp_path", "ecmwf_path", "interpol_path", "knnimputer_path",
    "topo_features_path",
    "test_start", "test_end", "val_start",
    "icond2_run_hours", "icond2_features", "ecmwf_features",
    "measurement_features", "target_col",
    "nwp_nodes", "hist_wind_available", "neighbour_meas_available",
    "station_connectivity", "station_graph_mode", "station_node_features",
    "interpolate_history", "edge_features",
}

VARIANT_A_SOURCE = "configs/dcrnn/config_wind_dcrnn_fold1.yaml"


def new_stem(source_stem: str) -> str:
    """config_wind_dcrnn_fold1 -> config_wind_dcrnn_stdhp_fold1 (insert _stdhp
    right before the trailing _fold<N>)."""
    m = re.search(r"_fold(\d+)$", source_stem)
    if not m:
        raise ValueError(f"Expected a config stem ending in _fold<N>, got: {source_stem!r}")
    return re.sub(r"_fold(\d+)$", r"_stdhp_fold\1", source_stem)


def build_source_list() -> list[dict]:
    """Enumerate the 30 (source, target, kind) triples."""
    entries = []

    dcrnn_variants = ["", "_base", "_nwp_hist", "_nomeas", "_nograph"]
    for variant in dcrnn_variants:
        for fold in (1, 2, 3):
            src = f"configs/dcrnn/config_wind_dcrnn{variant}_fold{fold}.yaml"
            stem = f"wind_dcrnn{variant}_fold{fold}"
            tgt = f"configs/dcrnn/stdhp/config_{new_stem(stem)}.yaml"
            entries.append({"source": src, "target": tgt, "kind": "dcrnn"})

    mtgnn_variants = ["", "_nwp", "_nwp_hist"]
    for variant in mtgnn_variants:
        for fold in (1, 2, 3):
            src = f"configs/mtgnn/config_wind_mtgnn{variant}_fold{fold}.yaml"
            stem = f"wind_mtgnn{variant}_fold{fold}"
            tgt = f"configs/mtgnn/stdhp/config_{new_stem(stem)}.yaml"
            entries.append({"source": src, "target": tgt, "kind": "copy"})

    wavenet_variants = ["", "_nwp"]
    for variant in wavenet_variants:
        for fold in (1, 2, 3):
            src = f"configs/wavenet/config_wind_wavenet{variant}_fold{fold}.yaml"
            stem = f"wind_wavenet{variant}_fold{fold}"
            tgt = f"configs/wavenet/stdhp/config_{new_stem(stem)}.yaml"
            entries.append({"source": src, "target": tgt, "kind": "copy"})

    return entries


def deep_diff_keys(a: dict, b: dict, path: str = "") -> list[str]:
    """Return dotted-path keys that differ (added, removed, or changed) between
    two nested dicts a (old) and b (new). Order-insensitive for lists of
    scalars is NOT applied — lists must match exactly (station id lists etc.
    must be byte-identical, order included)."""
    diffs = []
    keys = set(a) | set(b)
    for k in keys:
        p = f"{path}.{k}" if path else k
        if k not in a:
            diffs.append(f"{p} [ADDED]")
        elif k not in b:
            diffs.append(f"{p} [REMOVED]")
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            diffs.extend(deep_diff_keys(a[k], b[k], p))
        elif a[k] != b[k]:
            diffs.append(f"{p} [CHANGED] {a[k]!r} -> {b[k]!r}")
    return diffs


def harmonize_dcrnn(doc: dict, nwp_out_dim: int) -> dict:
    """Return a deep-copied doc with the harmonized keys written into
    doc['dcrnn']. All other sections/keys are left byte-identical.

    The legacy `batch_size` key (if present in the source) is dropped: it is
    dead -- geostatistics/dcrnn/config.py:208 reads
    d.get("grad_accum", d.get("batch_size", 4)), so grad_accum always shadows
    it -- but it is a trap for later readers who don't know that. For DCRNN,
    grad_accum IS the batch size (DCRNNTrainer batches for real, it does not
    accumulate); for MTGNN/WaveNet grad_accum is true gradient accumulation.
    See the header this function's caller writes for the persisted version
    of this note.
    """
    new_doc = copy.deepcopy(doc)
    dcrnn_sec = new_doc.setdefault("dcrnn", {})
    dcrnn_sec.update(DCRNN_HARMONIZED)
    dcrnn_sec["nwp_out_dim"] = nwp_out_dim
    dcrnn_sec.pop("batch_size", None)
    return new_doc


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(doc: dict, path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False, width=100, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                         help="Only report the diff/verification, do not write any files.")
    args = parser.parse_args()

    entries = build_source_list()

    # ---- sanity: all 30 sources exist -----------------------------------
    missing = [e["source"] for e in entries if not (WORK_DIR / e["source"]).exists()]
    if missing:
        print("MISSING SOURCE CONFIGS:")
        for m in missing:
            print(" ", m)
        sys.exit(1)
    print(f"All {len(entries)} source configs found.")

    # ---- read nwp_out_dim from variant-A fold1 ---------------------------
    variant_a_doc = load_yaml(WORK_DIR / VARIANT_A_SOURCE)
    nwp_out_dim = variant_a_doc["dcrnn"]["nwp_out_dim"]
    print(f"nwp_out_dim taken from {VARIANT_A_SOURCE}: {nwp_out_dim}")
    if "nwp_out_per_head" in variant_a_doc["dcrnn"]:
        print("NOTE: nwp_out_per_head IS present in the source — unexpected, check manually.")
    else:
        print("NOTE: nwp_out_per_head is absent from all source dcrnn configs (as expected) "
              "and is not added to the stdhp configs — it is only meaningful as an "
              "Optuna-HPO-derived key (nwp_out_dim = nwp_heads * nwp_out_per_head, see "
              "train_dcrnn.py:382-385) and this run never sets --hpo-study.")

    diff_rows = []  # (source, target, kind, diff_summary)

    for e in entries:
        src_path = WORK_DIR / e["source"]
        tgt_path = WORK_DIR / e["target"]
        src_doc = load_yaml(src_path)

        if e["kind"] == "dcrnn":
            new_doc = harmonize_dcrnn(src_doc, nwp_out_dim)
            diffs = deep_diff_keys(src_doc, new_doc)
            # Verify every diff is within dcrnn.<allowed key>
            bad = []
            for d in diffs:
                key_path = d.split(" [")[0]
                parts = key_path.split(".")
                if len(parts) != 2 or parts[0] != "dcrnn" or parts[1] not in DCRNN_ALLOWED_CHANGED_KEYS:
                    bad.append(d)
                if len(parts) == 2 and parts[1] in DCRNN_PROTECTED_KEYS:
                    bad.append(d + " [PROTECTED KEY TOUCHED]")
            if bad:
                print(f"ERROR: unexpected diff in {e['source']} -> {e['target']}:")
                for b in bad:
                    print("   ", b)
                sys.exit(2)
            diff_rows.append((e["source"], e["target"], "dcrnn-harmonized", len(diffs), diffs))
            if not args.check:
                header = (
                    f"# AUTO-GENERATED by geostatistics/stdrun/gen_stdhp_configs.py\n"
                    f"# Source: {e['source']}\n"
                    f"# Harmonized DCRNN hyperparameters written; all data paths, time windows,\n"
                    f"# feature lists and variant switches are inherited unchanged from the source.\n"
                    f"# NOTE: dcrnn.grad_accum is this model's BATCH SIZE (DCRNNTrainer batches\n"
                    f"# for real, nothing is accumulated) -- see geostatistics/dcrnn/config.py:208.\n"
                    f"# For MTGNN/WaveNet, grad_accum IS true gradient accumulation -- do not\n"
                    f"# confuse the two. The legacy 'batch_size' key is removed here (it was\n"
                    f"# always dead: grad_accum shadowed it) to avoid misleading future readers.\n"
                    f"# Do not hand-edit — regenerate from the source config instead.\n"
                )
                dump_yaml(new_doc, tgt_path, header)
        else:
            # MTGNN / WaveNet: unchanged byte-for-byte copy.
            diff_rows.append((e["source"], e["target"], "copy", 0, []))
            if not args.check:
                tgt_path.parent.mkdir(parents=True, exist_ok=True)
                tgt_path.write_bytes(src_path.read_bytes())

    # ---- report -----------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"{'source':55s} {'kind':18s} {'#diffs':7s}")
    print("-" * 100)
    for src, tgt, kind, ndiff, diffs in diff_rows:
        print(f"{src:55s} {kind:18s} {ndiff:7d}")
    print("=" * 100)

    print("\nFull key-level diffs for the 15 DCRNN pairs:")
    for src, tgt, kind, ndiff, diffs in diff_rows:
        if kind != "dcrnn-harmonized":
            continue
        print(f"\n  {src} -> {tgt}")
        for d in diffs:
            print(f"    {d}")

    # ---- MTGNN / WaveNet cross-variant hyperparameter identity check -----
    print(f"\n{'='*100}")
    print("Cross-variant hyperparameter identity check (MTGNN, WaveNet)")
    print("=" * 100)

    def hp_section(path: Path, section: str) -> dict:
        d = load_yaml(path)
        sec = dict(d.get(section, {}))
        sec.pop("hpo", None)
        return sec

    def report_identity(label: str, section: str, variant_stems: list[str], config_dir: str):
        for fold in (1, 2, 3):
            secs = {}
            for v in variant_stems:
                p = WORK_DIR / config_dir / f"config_wind_{label}{v}_fold{fold}.yaml"
                secs[v] = hp_section(p, section)
            base_v = variant_stems[0]
            base_keys = set(secs[base_v])
            for v in variant_stems[1:]:
                keys = set(secs[v])
                only_in_base = base_keys - keys
                only_in_v = keys - base_keys
                common = base_keys & keys
                changed = {k: (secs[base_v][k], secs[v][k]) for k in common if secs[base_v][k] != secs[v][k]}
                print(f"  fold{fold} {base_v or '(base)'} vs {v}: "
                      f"only-in-{base_v or 'base'}={sorted(only_in_base)} "
                      f"only-in-{v}={sorted(only_in_v)} "
                      f"changed={changed}")

    report_identity("mtgnn", "mtgnn", ["", "_nwp", "_nwp_hist"], "configs/mtgnn")
    report_identity("wavenet", "wavenet", ["", "_nwp"], "configs/wavenet")

    # ---- substring uniqueness of the 30 stdhp model names -----------------
    print(f"\n{'='*100}")
    print("Substring uniqueness of the 30 future stdhp model checkpoint names against models/*.pt")
    print("=" * 100)
    models_dir = WORK_DIR / "models"
    existing = [p.name for p in models_dir.glob("*.pt")]
    print(f"Existing checkpoints scanned: {len(existing)}")

    suffix_map = {"dcrnn": "dcrnn", "mtgnn": "mtgnn", "wavenet": "wavenet"}
    all_ok = True
    for e in entries:
        tgt_stem = Path(e["target"]).stem.replace("config_", "")
        kind_dir = Path(e["target"]).parts[1]  # dcrnn/mtgnn/wavenet
        script_type = suffix_map[kind_dir]
        model_name = f"{tgt_stem}_{script_type}_stdhp.pt"
        matches = [m for m in existing if model_name in m]
        status = "OK" if len(matches) == 0 else f"COLLISION ({len(matches)} matches: {matches})"
        if len(matches) != 0:
            all_ok = False
        print(f"  {model_name:55s} {status}")
    print(f"\nAll 30 future model names are currently absent / non-colliding as substrings: {all_ok}")

    print(f"\n{'DRY (--check, nothing written)' if args.check else 'Configs written.'}")


if __name__ == "__main__":
    main()
