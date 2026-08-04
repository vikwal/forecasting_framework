#!/usr/bin/env python3
"""
gen_variant_configs.py — derive the ablation-variant DCRNN configs from the
variant-A configs.

The derivation is text surgery **inside the top-level ``dcrnn:`` section only**,
so every comment and every unrelated setting is preserved byte for byte and the
resulting diff against the source is provably confined to the ablated axis.

    Variant B  (``nomeas``)  neighbour_meas_available: false
    Variant C  (``nograph``) + station_connectivity: "none"
                             + direction_to_adj: false

Sources (4 per variant):
    config_wind_dcrnn.yaml          → the HPO config, one Optuna study per variant
    config_wind_dcrnn_fold{1,2,3}.yaml → the three spatial-CV retrain configs

Study resolution
----------------
``train_dcrnn.py`` and ``hpo_dcrnn.py`` both derive the Optuna study name from
the config stem with any ``_fold<N>`` suffix stripped:

    hpo_stem   = re.sub(r'_fold\\d+$', '', config_stem)
    study_name = f"cl_m-dcrnn_out-{H}_freq-{freq}_{hpo_stem}"

so ``config_wind_dcrnn_nomeas{,_fold1..3}.yaml`` all resolve to
``cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_nomeas``.  Each variant therefore gets its
own HPO study automatically — no CLI plumbing needed.  That is decision (b) from
``docs/prompt_ablations_implementation.md`` §5.1.

Trial budget
------------
``--trials`` rewrites ``dcrnn.hpo.trials`` in the non-fold (HPO) config only.
The fold configs never run HPO, so their ``hpo:`` blocks are left alone.

Inert parameters in variant C
-----------------------------
With no station edges, ``K_hop`` and ``next_n_neighbors`` provably cannot change
the prediction (verified: max|Δpred| = 0.0 under a full neighbour permutation).
``--pin-inert`` drops them from C's HPO search space, mirroring what was done for
``config_wind_dcrnn_base.yaml`` (which dropped ``nwp_heads`` /
``nwp_out_per_head``).  The static values ``K_hop: 2`` / ``next_n_neighbors: 90``
stay, so C's batch composition still matches A and B.  Applies to the non-fold
config only, since the fold configs never run HPO.  See
docs/implementation_plan_ablations.md §9.2 — the campaign uses ``--pin-inert``.

Usage
-----
    python -m geostatistics.ablations.gen_variant_configs \\
        --dir configs/dcrnn --variant nomeas --trials 60 [--dry-run] [--force]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

SRC = ["config_wind_dcrnn.yaml"] + [f"config_wind_dcrnn_fold{i}.yaml" for i in (1, 2, 3)]

VARIANTS: dict[str, tuple[list[tuple[str, str, str]], str]] = {
    # variant: ([(key, value, trailing comment)], header comment block)
    "nomeas": (
        [("neighbour_meas_available", "false",
          "  # Ablation B: keine Station traegt Messungen")],
        "Ablation B (nomeas): KEINE Station traegt Messwerte — auch die Nachbarn\n"
        "  # nicht. Der Stationsgraph transportiert weiterhin NWP-Features und\n"
        "  # Statics, deshalb isoliert A minus B genau den Wert der gemessenen\n"
        "  # Nachbarwerte. target_mask bleibt unveraendert, A und B werden also an\n"
        "  # denselben Knoten bewertet.\n"
        "  # Erzeugt aus {src} von geostatistics/ablations/gen_variant_configs.py.",
    ),
    "nograph": (
        [("neighbour_meas_available", "false",
          "  # Ablation C: keine Station traegt Messungen"),
         ("station_connectivity", '"none"',
          "  # Ablation C: leeres station<->station Kantenset"),
         ("direction_to_adj", "false",
          "  # ohne Stationskanten ohnehin wirkungslos, explizit aus")],
        "Ablation C (nograph): zusaetzlich zu B gibt es KEINE station<->station\n"
        "  # Kanten. Das DCGRU degeneriert damit zu einer gewoehnlichen GRU je\n"
        "  # Station; uebrig bleibt der reine standortweise Downscaling-Boden ueber\n"
        "  # die eigenen k NWP-Gitterpunkte. B minus C misst den Geometrie- und\n"
        "  # NWP-Kontextkanal.\n"
        "  # Erzeugt aus {src} von geostatistics/ablations/gen_variant_configs.py.",
    ),
}

# HPO params that provably cannot influence variant C (no station edges).
INERT_IN_C = ["K_hop", "next_n_neighbors", "edge_weight_sigma"]


def dcrnn_region(lines: list[str]) -> tuple[int, int]:
    """Return [start, end) line indices of the top-level ``dcrnn:`` mapping."""
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^dcrnn:\s*$", ln):
            start = i + 1
            break
    if start is None:
        raise SystemExit("ABORT: no top-level 'dcrnn:' key found")
    for j in range(start, len(lines)):
        if re.match(r"^[A-Za-z_]", lines[j]):       # next top-level key
            return start, j
    return start, len(lines)


def set_key(lines: list[str], lo: int, hi: int, key: str, value: str,
            comment: str) -> tuple[list[str], str]:
    """Replace ``  key: ...`` inside [lo, hi) if present. Returns a note."""
    pat = re.compile(rf"^(  {re.escape(key)}:)(\s*)(\S.*)$")
    for i in range(lo, hi):
        m = pat.match(lines[i].rstrip("\n"))
        if m:
            old = m.group(3)
            lines[i] = f"  {key}: {value}{comment}\n"
            return lines, f"replaced (was: {old})"
    return lines, "absent"


def drop_hpo_params(lines: list[str], lo: int, hi: int, keys: list[str]) -> tuple[list[str], list[str]]:
    """Remove ``      <key>:`` entries (and their block) from the hpo params list."""
    dropped = []
    out = list(lines)
    for key in keys:
        pat = re.compile(rf"^      {re.escape(key)}:\s*$")
        idx = None
        for i in range(lo, min(hi, len(out))):
            if pat.match(out[i].rstrip("\n")):
                idx = i
                break
        if idx is None:
            continue
        # block = the key line plus every following line indented deeper than 6,
        # plus any immediately preceding comment lines at indent 6
        end = idx + 1
        while end < len(out) and (out[end].strip() == "" or out[end].startswith("        ")):
            if out[end].strip() == "":
                # keep looking: a blank line may separate the block from the next key
                nxt = end + 1
                if nxt < len(out) and out[nxt].startswith("        "):
                    end += 1
                    continue
                break
            end += 1
        begin = idx
        while begin - 1 >= lo and out[begin - 1].lstrip().startswith("#") \
                and out[begin - 1].startswith("      #"):
            begin -= 1
        note = f"{key} (lines {begin - lo}..{end - lo} of dcrnn region)"
        out = out[:begin] + [
            f"      # {key} entfaellt: ohne station<->station Kanten wirkungslos "
            f"(Permutationstest: max|dpred| = 0.0).\n"
        ] + out[end:]
        dropped.append(note)
        hi -= (end - begin) - 1
    return out, dropped


def set_trials(lines: list[str], lo: int, hi: int, trials: int) -> str:
    pat = re.compile(r"^(    trials:)(\s*)(\S+)(.*)$")
    for i in range(lo, hi):
        m = pat.match(lines[i].rstrip("\n"))
        if m:
            old = m.group(3)
            lines[i] = (
                f"    trials: {trials}"
                f"    # reduziertes Budget fuer die Ablation (A: {old}) — "
                f"siehe docs/ablations_verification_results.md\n"
            )
            return f"replaced (was: {old})"
    return "absent"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="configs/dcrnn")
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--trials", type=int, default=None,
                    help="rewrite dcrnn.hpo.trials in the non-fold config")
    ap.add_argument("--pin-inert", action="store_true",
                    help="variant C only: drop K_hop / next_n_neighbors from the "
                         "HPO search space (they cannot influence C)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    d = Path(args.dir)
    kvs, header = VARIANTS[args.variant]

    for src_name in SRC:
        src = d / src_name
        if not src.exists():
            raise SystemExit(f"ABORT: missing {src}")
        stem = src.stem
        m = re.match(r"^(config_wind_dcrnn)(_fold\d+)?$", stem)
        if not m:
            raise SystemExit(f"ABORT: unexpected stem {stem}")
        is_fold = bool(m.group(2))
        dst = d / f"{m.group(1)}_{args.variant}{m.group(2) or ''}.yaml"

        lines = src.read_text().splitlines(keepends=True)
        lo, hi = dcrnn_region(lines)

        notes: list[str] = []
        pending: list[tuple[str, str, str]] = []
        for key, value, comment in kvs:
            lines, note = set_key(lines, lo, hi, key, value, comment)
            notes.append(f"{key}: {value}  [{note}]")
            if note == "absent":
                pending.append((key, value, comment))

        if pending:
            block = [f"\n  # ── {header.format(src=src_name)}\n"]
            for key, value, comment in pending:
                block.append(f"  {key}: {value}{comment}\n")
            ins = None
            for i in range(lo, hi):
                if re.match(r"^  station_connectivity:", lines[i]):
                    ins = i + 1
                    break
            if ins is None:
                raise SystemExit(f"ABORT: no station_connectivity anchor in {src_name}")
            lines = lines[:ins] + block + lines[ins:]
            hi += len(block)
        else:
            # every key already existed → still record provenance
            ins = None
            for i in range(lo, hi):
                if re.match(r"^  station_connectivity:", lines[i]):
                    ins = i + 1
                    break
            block = [f"\n  # ── {header.format(src=src_name)}\n"]
            lines = lines[:ins] + block + lines[ins:]
            hi += len(block)

        if args.trials is not None and not is_fold:
            notes.append(f"hpo.trials: {args.trials}  [{set_trials(lines, lo, hi, args.trials)}]")

        # Only the non-fold config ever runs HPO, so only its search space is
        # touched — exactly like --trials above. The fold configs keep their
        # (unused) hpo blocks byte for byte, which keeps their diff against the
        # variant-A fold configs confined to the ablated axis.
        if args.pin_inert and args.variant == "nograph" and not is_fold:
            lines, dropped = drop_hpo_params(lines, lo, hi, INERT_IN_C)
            notes.append(f"dropped from HPO search space: {dropped or 'none found'}")

        out = "".join(lines)
        print(f"{src_name}  ->  {dst.name}")
        for n in notes:
            print(f"      {n}")
        if not args.dry_run:
            if dst.exists() and not args.force:
                raise SystemExit(f"ABORT: {dst} already exists (use --force)")
            dst.write_text(out)

    print("DRY RUN — nothing written." if args.dry_run else "WRITTEN.")


if __name__ == "__main__":
    main()
