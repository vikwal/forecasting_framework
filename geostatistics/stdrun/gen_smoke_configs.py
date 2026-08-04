#!/usr/bin/env python3
"""
gen_smoke_configs.py -- Generate three throw-away "smoke test" configs, each
derived from an already-generated stdhp config, with ONLY max_epochs and
patience reduced to 2 (~15 min run instead of ~2h to first eval). Purpose:
verify the eval scripts reconstruct the architecture and load the checkpoint
correctly for the three DCRNN/MTGNN/WaveNet paths that have never run
end-to-end, before committing to the full 30-job stdhp dry run.

Deliberately a separate sibling script next to gen_stdhp_configs.py rather
than a --smoke flag bolted onto it:
  - the transformation is different in kind (source is an stdhp config, not
    the original fold config; the allowed-changed-key set is {max_epochs,
    patience} uniformly for whichever of dcrnn/mtgnn/wavenet is being
    touched, not the 19-key DCRNN-only harmonization set), so reusing
    gen_stdhp_configs.py's DCRNN-specific self-check machinery would need
    as much new special-casing as a fresh ~150-line script does anyway;
  - gen_stdhp_configs.py already went through review for the 30-job path;
    keeping it untouched (beyond the batch_size fix) avoids re-opening that
    surface for a one-off, 3-config throwaway generator.

Usage
-----
    python geostatistics/stdrun/gen_smoke_configs.py           # generate + report
    python geostatistics/stdrun/gen_smoke_configs.py --check   # report only, no writes
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

WORK_DIR = Path(__file__).resolve().parent.parent.parent  # forecasting_framework/

# Keys allowed to change in the model section, relative to the stdhp source.
SMOKE_ALLOWED_CHANGED_KEYS = {"max_epochs", "patience"}
SMOKE_MAX_EPOCHS = 2
SMOKE_PATIENCE = 2

# (stdhp source path, model section/type key, target dir name)
SMOKE_SOURCES = [
    ("configs/wavenet/stdhp/config_wind_wavenet_stdhp_fold1.yaml", "wavenet"),
    ("configs/dcrnn/stdhp/config_wind_dcrnn_nograph_stdhp_fold1.yaml", "dcrnn"),
    ("configs/mtgnn/stdhp/config_wind_mtgnn_nwp_stdhp_fold1.yaml", "mtgnn"),
]


def smoke_target(source: str, model_type: str) -> str:
    """configs/wavenet/stdhp/config_wind_wavenet_stdhp_fold1.yaml ->
    configs/wavenet/smoke/config_wind_wavenet_smoke_fold1.yaml"""
    src_path = Path(source)
    if "_stdhp_" not in src_path.stem:
        raise ValueError(f"Expected an stdhp config stem (contains '_stdhp_'), got: {src_path.stem!r}")
    smoke_stem = src_path.stem.replace("_stdhp_", "_smoke_")
    return f"configs/{model_type}/smoke/{smoke_stem}.yaml"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(doc: dict, path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False, width=100, allow_unicode=True)


def deep_diff_keys(a: dict, b: dict, path: str = "") -> list[str]:
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


def make_smoke(doc: dict, model_type: str) -> dict:
    new_doc = copy.deepcopy(doc)
    sec = new_doc[model_type]
    sec["max_epochs"] = SMOKE_MAX_EPOCHS
    sec["patience"] = SMOKE_PATIENCE
    return new_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                         help="Only report the diff/verification, do not write any files.")
    args = parser.parse_args()

    missing = [s for s, _ in SMOKE_SOURCES if not (WORK_DIR / s).exists()]
    if missing:
        print("MISSING STDHP SOURCE CONFIGS (generate stdhp configs first):")
        for m in missing:
            print(" ", m)
        sys.exit(1)
    print(f"All {len(SMOKE_SOURCES)} stdhp source configs found.")

    rows = []
    for source, model_type in SMOKE_SOURCES:
        src_path = WORK_DIR / source
        tgt = smoke_target(source, model_type)
        tgt_path = WORK_DIR / tgt
        src_doc = load_yaml(src_path)
        new_doc = make_smoke(src_doc, model_type)
        diffs = deep_diff_keys(src_doc, new_doc)

        bad = []
        for d in diffs:
            key_path = d.split(" [")[0]
            parts = key_path.split(".")
            if len(parts) != 2 or parts[0] != model_type or parts[1] not in SMOKE_ALLOWED_CHANGED_KEYS:
                bad.append(d)
        if bad:
            print(f"ERROR: unexpected diff in {source} -> {tgt}:")
            for b in bad:
                print("   ", b)
            sys.exit(2)
        if len(diffs) != 2:
            print(f"ERROR: expected exactly 2 diffs (max_epochs, patience) for {source} -> {tgt}, got {len(diffs)}:")
            for d in diffs:
                print("   ", d)
            sys.exit(2)

        rows.append((source, tgt, model_type, diffs))

        if not args.check:
            header = (
                f"# AUTO-GENERATED by geostatistics/stdrun/gen_smoke_configs.py\n"
                f"# Source (stdhp): {source}\n"
                f"# THROW-AWAY SMOKE-TEST CONFIG -- max_epochs/patience reduced to "
                f"{SMOKE_MAX_EPOCHS}/{SMOKE_PATIENCE}.\n"
                f"# Purpose: verify the eval script reconstructs this architecture and loads the\n"
                f"# checkpoint correctly (this path never ran end-to-end before). Not a real result.\n"
                f"# All data paths, time windows, feature lists, variant switches and every other\n"
                f"# hyperparameter are inherited unchanged from the stdhp source.\n"
                f"# Do not hand-edit -- regenerate from the stdhp source config instead.\n"
            )
            dump_yaml(new_doc, tgt_path, header)

    print(f"\n{'='*100}")
    for source, tgt, model_type, diffs in rows:
        print(f"\n  {source} -> {tgt}")
        for d in diffs:
            print(f"    {d}")
    print("=" * 100)

    # ---- substring uniqueness of the 3 smoke checkpoint names -------------
    print("\nSubstring uniqueness of the 3 future smoke checkpoint names:")
    print("  against (a) existing models/*.pt, (b) 30 future stdhp checkpoint names, (c) each other")

    models_dir = WORK_DIR / "models"
    existing = [p.name for p in models_dir.glob("*.pt")]
    print(f"  existing checkpoints scanned: {len(existing)}")

    # (b) the 30 future stdhp checkpoint names, computed the same way
    # run_stdhp_pipeline.py computes StdhpJob.model_name (stdhp_stem + "_" + model_type + "_stdhp").
    import re

    def new_stem(source_stem: str) -> str:
        m = re.search(r"_fold(\d+)$", source_stem)
        if not m:
            raise ValueError(source_stem)
        return re.sub(r"_fold(\d+)$", r"_stdhp_fold\1", source_stem)

    stdhp_names = []
    dcrnn_variants = ["", "_base", "_nwp_hist", "_nomeas", "_nograph"]
    for v in dcrnn_variants:
        for fold in (1, 2, 3):
            stem = new_stem(f"wind_dcrnn{v}_fold{fold}")
            stdhp_names.append(f"{stem}_dcrnn_stdhp")
    mtgnn_variants = ["", "_nwp", "_nwp_hist"]
    for v in mtgnn_variants:
        for fold in (1, 2, 3):
            stem = new_stem(f"wind_mtgnn{v}_fold{fold}")
            stdhp_names.append(f"{stem}_mtgnn_stdhp")
    wavenet_variants = ["", "_nwp"]
    for v in wavenet_variants:
        for fold in (1, 2, 3):
            stem = new_stem(f"wind_wavenet{v}_fold{fold}")
            stdhp_names.append(f"{stem}_wavenet_stdhp")
    assert len(stdhp_names) == 30, len(stdhp_names)
    print(f"  30 future stdhp checkpoint names computed.")

    smoke_names = []
    for source, model_type in SMOKE_SOURCES:
        tgt = smoke_target(source, model_type)
        tgt_stem = Path(tgt).stem  # e.g. config_wind_wavenet_smoke_fold1
        tgt_stem = tgt_stem.replace("config_", "")
        model_name = f"{tgt_stem}_{model_type}_smoke"
        smoke_names.append((source, model_type, model_name))

    all_ok = True
    for source, model_type, model_name in smoke_names:
        against_existing = [m for m in existing if model_name in m]
        against_stdhp = [m for m in stdhp_names if model_name in m or m in model_name]
        against_self = [n for _, _, n in smoke_names if n != model_name and (model_name in n or n in model_name)]
        ok = not against_existing and not against_stdhp and not against_self
        all_ok = all_ok and ok
        status = "OK" if ok else "COLLISION"
        print(f"  {model_name:55s} {status}"
              f"{'' if ok else f'  existing={against_existing} stdhp={against_stdhp} self={against_self}'}")
    print(f"\nAll 3 smoke checkpoint names uniquely resolvable: {all_ok}")
    if not all_ok:
        sys.exit(3)

    print(f"\n{'DRY (--check, nothing written)' if args.check else 'Smoke configs written.'}")


if __name__ == "__main__":
    main()
