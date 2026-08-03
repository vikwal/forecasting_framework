#!/usr/bin/env python3
"""
verify_review2_cache.py — K4: concurrency test for GNNCache.save() on a *throwaway* cache
directory. Never touches data_cache/gnns.

Harness: 4 writer processes save the same key at the same time, each with a
distinguishable fill value; 6 reader processes poll the cache and, whenever it
reports exists(), load derived.pkl and mmap the arrays and check that

  * every element of every array carries the same fill value  (no torn write)
  * that fill value equals the one recorded in derived.pkl     (no mixed version)
  * both arrays carry the same fill value                      (no split version)

The same harness is run twice: once against the pre-fix save() (restored here
verbatim from git) and once against the fixed one.

Run from anywhere (uses a throwaway directory, default /tmp/k4cache)::

    CUDA_VISIBLE_DEVICES="" nice -n 19 \
        python -m archiv.ablations_verification.verify_review2_cache [<cache_root>]
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from utils.data_cache import GNNCache

KEY = "k4testkey0000001"
N_ELEM = 12_000_000          # 48 MB + 24 MB per writer
N_WRITERS = 4
N_READERS = 6
READ_SECONDS = 10.0


# --- the implementation as it was before the fix (git 9a0f3b6) --------------
def old_save(self, key, arrays, derived):
    p = self._dir(key)
    p.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        out = p / f"{name}.npy"
        np.save(out, arr)
    with open(p / "derived.pkl", "wb") as fh:
        pickle.dump(derived, fh, protocol=4)


def _writer(cache_root: str, fill: int, use_old: bool, barrier, q):
    if use_old:
        GNNCache.save = old_save
    c = GNNCache(cache_root)
    a = np.full(N_ELEM, fill, dtype=np.float32)
    b = np.full(N_ELEM // 2, fill, dtype=np.float32)
    barrier.wait()
    t0 = time.time()
    c.save(KEY, arrays={"grid_icond2_runs": a, "meas_raw": b},
           derived={"fill": fill, "pid": os.getpid()})
    q.put(("write", fill, round(time.time() - t0, 2)))


def _reader(cache_root: str, barrier, q):
    c = GNNCache(cache_root)
    barrier.wait()
    t_end = time.time() + READ_SECONDS
    n_reads = n_bad = 0
    bad_examples = []
    while time.time() < t_end:
        if not c.exists(KEY):
            time.sleep(0.005)
            continue
        try:
            der = c.load_derived(KEY)
            arrs = c.load_arrays(KEY, names=["grid_icond2_runs", "meas_raw"],
                                 mmap=True)
            n_reads += 1
            seen = set()
            for name, arr in arrs.items():
                seen.update({float(arr[0]), float(arr[len(arr) // 2]),
                             float(arr[-1])})
                if not bool((arr == arr[0]).all()):
                    seen.add(-1.0)          # torn array
            seen.add(float(der["fill"]))
            if len(seen) != 1:
                n_bad += 1
                if len(bad_examples) < 3:
                    bad_examples.append(sorted(seen))
        except Exception as exc:            # truncated header, EOF in pickle …
            n_reads += 1
            n_bad += 1
            if len(bad_examples) < 3:
                bad_examples.append(f"{type(exc).__name__}: {exc}")
        time.sleep(0.003)
    q.put(("read", n_reads, n_bad, bad_examples))


def run(cache_root: str, use_old: bool) -> tuple[int, int, int]:
    shutil.rmtree(cache_root, ignore_errors=True)
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(N_WRITERS + N_READERS)
    q = ctx.Queue()
    procs = [ctx.Process(target=_writer,
                         args=(cache_root, 100 + i, use_old, barrier, q))
             for i in range(N_WRITERS)]
    procs += [ctx.Process(target=_reader, args=(cache_root, barrier, q))
              for _ in range(N_READERS)]
    for p in procs:
        p.start()
    reads = bad = 0
    writes = 0
    died = 0
    examples = []
    deadline = time.time() + 180
    got = 0
    while got < len(procs) and time.time() < deadline:
        try:
            item = q.get(timeout=5)
        except Exception:
            if not any(p.is_alive() for p in procs):
                break
            continue
        got += 1
        if item[0] == "read":
            reads += item[1]
            bad += item[2]
            examples.extend(item[3])
        else:
            writes += 1
    import signal as _sig
    codes: list = []
    for p in procs:
        p.join(timeout=10)
        if p.exitcode not in (0, None):
            died += 1
            codes.append(p.exitcode)
        if p.is_alive():
            p.terminate()
    if died:
        names = []
        for c in sorted(set(codes)):
            if c < 0:
                try:
                    names.append(f"{_sig.Signals(-c).name}({c})")
                except ValueError:
                    names.append(str(c))
            else:
                names.append(str(c))
        print(f"      {died}/{len(procs)} child processes were killed: "
              f"{', '.join(names)}  — a reader holding an mmap of a file that "
              f"is truncated under it takes SIGBUS, which Python cannot catch")
    label = "OLD (pre-fix)" if use_old else "NEW (fixed)"
    print(f"  {label}: {writes} writers, {reads} successful reads, "
          f"{bad} inconsistent reads")
    if examples:
        print(f"      examples: {examples[:3]}")
    # final state
    c = GNNCache(cache_root)
    der = c.load_derived(KEY)
    arrs = c.load_arrays(KEY, names=["grid_icond2_runs", "meas_raw"], mmap=True)
    finals = {float(a[0]) for a in arrs.values()} | {float(der["fill"])}
    print(f"      final state fill values: {sorted(finals)} "
          f"({'consistent' if len(finals) == 1 else 'INCONSISTENT'})")
    return reads, bad, len(finals), died


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/k4cache"
    print("=== K4 — concurrent GNNCache.save() on a throwaway directory ===")
    print(f"  cache root: {root}  writers={N_WRITERS} readers={N_READERS} "
          f"array={N_ELEM * 4 / 1e6:.0f} MB + {N_ELEM * 2 / 1e6:.0f} MB")

    print("\n  [A] pre-fix implementation")
    r_old, b_old, f_old, d_old = run(root + "_old", use_old=True)

    print("\n  [B] fixed implementation")
    r_new, b_new, f_new, d_new = run(root + "_new", use_old=False)

    # -- extra: a complete cache is never overwritten -----------------------
    print("\n  [C] a complete cache is left alone")
    c = GNNCache(root + "_new")
    before = (Path(root + "_new") / KEY / "derived.pkl").stat().st_mtime_ns
    keep = c.load_derived(KEY)["fill"]
    c.save(KEY, arrays={"grid_icond2_runs": np.full(16, 999, dtype=np.float32),
                        "meas_raw": np.full(16, 999, dtype=np.float32)},
           derived={"fill": 999, "pid": os.getpid()})
    after = (Path(root + "_new") / KEY / "derived.pkl").stat().st_mtime_ns
    now = c.load_derived(KEY)["fill"]
    print(f"      fill before={keep} after={now}, derived.pkl mtime "
          f"{'unchanged' if before == after else 'CHANGED'}")

    print("\n" + "=" * 74)
    ok = (b_new == 0 and f_new == 1 and d_new == 0
          and now == keep and before == after)
    print(f"  pre-fix: {b_old} inconsistent reads of {r_old}, "
          f"{d_old} reader process(es) killed, final state "
          f"{'consistent' if f_old == 1 else 'INCONSISTENT'}")
    print(f"  fixed  : {b_new} inconsistent reads of {r_new}, "
          f"{d_new} reader process(es) killed, final state "
          f"{'consistent' if f_new == 1 else 'INCONSISTENT'}")
    print("  VERDICT:", "PASS" if ok else "FAIL")
    print("=" * 74)
    for d in (root + "_old", root + "_new"):
        shutil.rmtree(d, ignore_errors=True)
    sys.exit(0 if ok else 1)
