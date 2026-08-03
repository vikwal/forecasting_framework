#!/usr/bin/env python3
"""
verify_review2_env.py — K3: an HPO worker without ``WEATHER_DB_URL`` must die
before it can touch the shared GNNCache, and a worker with the variable set must
be unaffected.

Method
------
``GNNCache.__init__`` is monkeypatched to raise a sentinel. It is the first
thing in each ``hpo_*.py`` ``main()`` that touches the cache directory, so

  * reaching the sentinel      == the env check let the run through and nothing
                                  has been written yet;
  * MissingNWPElevationEnvError == the run aborted, and it aborted before
                                  GNNCache was even constructed.

No GPU, no Optuna, no data loading, no cache directory created in either case.
Each case runs in its own subprocess because ``main()`` mutates global state.

Run from the repository root::

    CUDA_VISIBLE_DEVICES="" nice -n 19 \
        python -m geostatistics.ablations.verify_review2_env
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

CASES = [
    ("hpo_dcrnn", "geostatistics.hpo_dcrnn",
     "configs/dcrnn/config_wind_dcrnn_base.yaml"),
    ("hpo_mtgnn", "geostatistics.hpo_mtgnn",
     "configs/mtgnn/config_wind_mtgnn.yaml"),
    ("hpo_wavenet", "geostatistics.hpo_wavenet",
     "configs/wavenet/config_wind_wavenet.yaml"),
]


class ReachedCache(BaseException):
    """Raised instead of building a GNNCache."""


def _child(which: str, mode: str) -> int:
    import utils.data_cache as dc
    from geostatistics.train_stgnn2 import MissingNWPElevationEnvError

    def _boom(self, *a, **kw):
        raise ReachedCache
    dc.GNNCache.__init__ = _boom

    name, modname, cfg = next(c for c in CASES if c[0] == which)
    if mode == "unset":
        os.environ.pop("WEATHER_DB_URL", None)
    else:
        os.environ.setdefault("WEATHER_DB_URL", "postgresql://probe/not-used")

    import importlib
    mod = importlib.import_module(modname)
    sys.argv = [modname, "--config", str(_ROOT / cfg), "--suffix", "k3probe"]
    try:
        mod.main()
        result = "RETURNED"
    except ReachedCache:
        result = "REACHED_CACHE"
    except MissingNWPElevationEnvError as exc:
        result = f"ABORTED: {exc}"
    except SystemExit as exc:
        result = f"SystemExit({exc.code})"

    ok = ((mode == "unset" and result.startswith("ABORTED"))
          or (mode == "set" and result == "REACHED_CACHE"))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} / WEATHER_DB_URL "
          f"{'unset' if mode == 'unset' else 'set'} -> {result[:150]}")
    return 0 if ok else 1


def main() -> int:
    print("=== K3 — WEATHER_DB_URL is a precondition, not a warning ===")
    n_ok = 0
    n = 0
    for name, _, _ in CASES:
        for mode in ("unset", "set"):
            n += 1
            env = dict(os.environ)
            env.pop("WEATHER_DB_URL", None)
            env["CUDA_VISIBLE_DEVICES"] = ""
            proc = subprocess.run(
                [sys.executable, __file__, "--child", name, mode],
                cwd=str(_ROOT), env=env, capture_output=True, text=True,
            )
            for line in proc.stdout.splitlines():
                if line.startswith("  ["):
                    print(line)
            if proc.returncode == 0:
                n_ok += 1
            elif not any(l.startswith("  [") for l in proc.stdout.splitlines()):
                print(f"  [FAIL] {name} / {mode}: child crashed\n"
                      f"{proc.stderr[-800:]}")
    # the probe logs are a side effect of hpo_*.py's file handler
    for f in (_ROOT / "logs").glob("*_k3probe.log"):
        f.unlink(missing_ok=True)
    print(f"\n{n_ok} passed, {n - n_ok} failed  (of {n} checks)")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        sys.exit(_child(sys.argv[2], sys.argv[3]))
    sys.exit(main())
