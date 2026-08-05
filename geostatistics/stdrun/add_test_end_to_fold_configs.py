#!/usr/bin/env python3
"""
Add `test_end: '2026-03-31'` to the `data:` block of the original
(non-stdhp, non-smoke, non-test/, non-fold9) fold configs for dcrnn,
mtgnn, and wavenet.

Background: train_dcrnn.py (and hpo_dcrnn.py) truncate the loaded
measurement/ECMWF time series to `test_end + 2 days`, but only when
`test_end` is present in the config's `data:` block. The *_fold1/2/3.yaml
configs never set it, so they load all the way to the end of available
data -- which contains NaN in the ECMWF parquets from 2026-05-01 onward
(74/153 stations), causing a hard ValueError at load time. Setting
test_end restores the truncation guard so that region is never loaded.

test_end='2026-03-31' covers both planned test periods (2025-08 to
2025-11 and 2025-12 to 2026-03) and sits safely before the 2026-05-01
NaN onset.

Implementation note: this is a plain text-line insertion right after the
existing `test_start` line (matching its indentation), NOT a YAML
load/dump round-trip. That guarantees every untouched line in the file
stays byte-identical -- comments, quoting style, key order, blank lines,
etc. A dump-based rewrite could silently reformat things far away from
the one line that's supposed to change.

Usage:
    python geostatistics/stdrun/add_test_end_to_fold_configs.py [--dry-run]
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_END_VALUE = "2026-03-31"
TEST_END_COMMENT = "# Grenze gegen NaN in ECMWF-Parquets ab 2026-05-01"

# Families whose top-level (non-fold9) config_wind_<family>*_fold{1,2,3}.yaml
# files are in scope. Discovery is via glob on each configs/<family>/ dir
# (non-recursive), so it does NOT walk into stdhp/, smoke/, or test/
# subdirectories, and [123] excludes any *_fold9*.yaml.
FAMILIES = ["dcrnn", "mtgnn", "wavenet"]


def target_files():
    files = []
    for family in FAMILIES:
        d = REPO_ROOT / "configs" / family
        matched = sorted(d.glob(f"config_wind_{family}*_fold[123].yaml"))
        files.extend(matched)
    return files


def process(path: Path, dry_run: bool):
    if not path.exists():
        print(f"MISSING: {path}")
        return "missing"

    text = path.read_text()
    lines = text.split("\n")

    test_end_idx = None
    test_start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("  test_end:"):
            test_end_idx = i
        if line.startswith("  test_start:"):
            test_start_idx = i

    if test_end_idx is not None:
        print(f"SKIP (already has test_end): {path.relative_to(REPO_ROOT)}")
        return "skip"

    if test_start_idx is None:
        print(f"ERROR: no top-level 'test_start:' line found in {path}")
        return "error"

    test_start_line = lines[test_start_idx]
    indent = test_start_line[: len(test_start_line) - len(test_start_line.lstrip(" "))]
    new_line = f"{indent}test_end: '{TEST_END_VALUE}'   {TEST_END_COMMENT}"

    if dry_run:
        print(f"WOULD PATCH: {path.relative_to(REPO_ROOT)}  (+1 line after test_start)")
        return "patched"

    lines.insert(test_start_idx + 1, new_line)
    path.write_text("\n".join(lines))
    print(f"PATCHED: {path.relative_to(REPO_ROOT)}")
    return "patched"


def main():
    dry_run = "--dry-run" in sys.argv
    files = target_files()
    counts = {"patched": 0, "skip": 0, "missing": 0, "error": 0}
    for f in files:
        result = process(f, dry_run)
        counts[result] += 1

    print()
    print(f"Total targeted: {len(files)}")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    if counts["missing"] or counts["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
