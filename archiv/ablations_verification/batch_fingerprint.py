#!/usr/bin/env python3
"""
batch_fingerprint.py — Plan §4.1: prove the ablation patch is a no-op for variant A.

Draws one training batch and one validation batch from the *unmodified* variant-A
config with fixed seeds and writes a byte-level fingerprint (SHA-256 over the raw
tensor buffers) of every tensor the sampler produces.

Run it once **before** the ablation patch and once **after**:

    ./frcst/bin/python -m archiv.ablations_verification.batch_fingerprint \\
        --out /tmp/fp_before.json
    # … apply the patch …
    ./frcst/bin/python -m archiv.ablations_verification.batch_fingerprint \\
        --out /tmp/fp_after.json --compare /tmp/fp_before.json

The script deliberately uses **no** ablation-specific API, so it is byte-for-byte
runnable on both sides of the change.  If any fingerprint differs, variant A was
touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from archiv.ablations_verification.fixture import build_fixture  # noqa: E402


def _hash(t: torch.Tensor) -> dict:
    a = t.detach().cpu().numpy()
    a = np.ascontiguousarray(a)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def _fingerprint_batch(batch) -> dict:
    d = batch.data
    out: dict = {}
    for ntype in ("station", "icond2", "ecmwf"):
        for attr in ("x", "static"):
            if attr in d[ntype]:
                out[f"{ntype}.{attr}"] = _hash(d[ntype][attr])
    for ekey in (
        ("station", "near", "station"),
        ("icond2", "informs", "station"),
        ("ecmwf", "informs", "station"),
    ):
        name = "__".join(ekey)
        out[f"{name}.edge_index"] = _hash(d[ekey].edge_index)
        out[f"{name}.edge_attr"] = _hash(d[ekey].edge_attr)
    out["target_mask"] = _hash(batch.target_mask)
    out["ground_truth"] = _hash(batch.ground_truth)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dcrnn/config_wind_dcrnn.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None,
                    help="Path to an earlier fingerprint JSON; exits 1 on any difference.")
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    fx = build_fixture(args.config, seed=args.seed)

    fp: dict = {
        "config": args.config,
        "seed": args.seed,
        "dims": {"M": fx.M, "I2": fx.I2, "E2": fx.E2,
                 "N_all": fx.N_all, "N_train": fx.N_train, "N_val": fx.N_val,
                 "station_static_features": fx.model_cfg.station_static_features,
                 "s2s_edges": int(fx.base_graph["station", "near", "station"].edge_index.shape[1]),
                 "s2s_edge_attr_dim": int(fx.base_graph["station", "near", "station"].edge_attr.shape[1])},
    }

    # --- training batch (uses the `random` module for station selection) ---
    random.seed(4711)
    np.random.seed(4711)
    torch.manual_seed(4711)
    b_tr = fx.sampler.sample_train(
        r_curr=2, r_hist=1, t_run_abs=fx.H_hist + 3, **fx.sample_train_kwargs(),
    )
    fp["train_batch"] = _fingerprint_batch(b_tr)

    # --- validation batch (deterministic layout) ---
    random.seed(4711)
    np.random.seed(4711)
    torch.manual_seed(4711)
    b_va = fx.sampler.sample_val(
        r_curr=2, r_hist=1, t_run_abs=fx.H_hist + 3, **fx.sample_val_kwargs(),
    )
    fp["val_batch"] = _fingerprint_batch(b_va)

    Path(args.out).write_text(json.dumps(fp, indent=2, sort_keys=True))
    print(f"fingerprint written → {args.out}")
    print(json.dumps(fp["dims"], indent=2, sort_keys=True))

    if args.compare:
        old = json.loads(Path(args.compare).read_text())
        new = json.loads(Path(args.out).read_text())
        diffs = []

        def walk(a, b, path=""):
            if isinstance(a, dict) and isinstance(b, dict):
                for k in sorted(set(a) | set(b)):
                    walk(a.get(k), b.get(k), f"{path}/{k}")
            elif a != b:
                diffs.append((path, a, b))

        walk(old, new)
        if diffs:
            print(f"\n*** {len(diffs)} DIFFERENCE(S) — variant A CHANGED ***")
            for p, a, b in diffs:
                print(f"  {p}\n    before: {a}\n    after : {b}")
            sys.exit(1)
        n_hashes = sum(
            1 for sec in ("train_batch", "val_batch") for _ in new[sec]
        )
        print(f"\nIDENTICAL — {n_hashes} tensor fingerprints match bit for bit "
              f"across {len(new['train_batch'])} + {len(new['val_batch'])} tensors.")


if __name__ == "__main__":
    main()
