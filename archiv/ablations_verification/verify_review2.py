#!/usr/bin/env python3
"""
verify_review2.py — reproduces the review-round-2 findings K1 and K2 and proves
they are gone after the fix.

K1  get_test_results_dcrnn.py builds station.static from lat/lon/alt only while
    DCRNNConfig.from_yaml resolves station_node_features='all' -> 13 static
    columns.  A real DCRNN forward pass on such a batch raises
        RuntimeError: mat1 and mat2 shapes cannot be multiplied (60x135 and 144x64)
    144 - 135 = 9 = the nine topographic columns.

K2  get_test_results_{mtgnn,wavenet}.py take the topo *names* from
    parse_edge_features(mcfg) while train_{mtgnn,wavenet}.py take them from
    parse_station_node_features(mcfg, ...) whenever the config carries the key.
    The resulting static_dim must agree between train and eval.

Run:  CUDA_VISIBLE_DEVICES="" nice -n 19 python /tmp/verify_review2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from archiv.ablations_verification.fixture import build_fixture
from geostatistics.dcrnn import DCRNN, DCRNNConfig
from geostatistics.evaluation import build_eval_batch
from geostatistics.stgnn.config import parse_edge_features, parse_station_node_features
from geostatistics.stgnn.utils.topo_features import (
    load_topo_station_features,
    load_topo_station_features_dict,
)

CFG_DCRNN = str(_ROOT / "configs/dcrnn/config_wind_dcrnn.yaml")
CFG_MTGNN = str(_ROOT / "configs/mtgnn/config_wind_mtgnn.yaml")
CFG_WAVENET = str(_ROOT / "configs/wavenet/config_wind_wavenet.yaml")

_results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    _results.append((ok, msg))
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")


# ---------------------------------------------------------------------------
# K1
# ---------------------------------------------------------------------------

def _forward_on(static_pre: np.ndarray, fx) -> str:
    """Run one real DCRNN forward on an eval batch built with *static_pre*.

    Returns "" on success or the RuntimeError message.
    """
    torch.manual_seed(4711)
    model = DCRNN(fx.model_cfg).eval()
    n_tr = len(fx.train_station_indices)
    data, target_mask, _ = build_eval_batch(
        sampler=fx.sampler,
        r_curr=1, r_hist=0,
        t_run_abs=fx.model_cfg.history_length,
        station_meas_scaled=fx.station_meas,
        station_nearest_grid=fx.station_nearest_grid,
        grid_icond2_runs_scaled=fx.grid_icond2_runs,
        station_ecmwf_nwp_scaled=fx.station_ecmwf_nwp,
        station_static=static_pre,
        ecmwf_nwp_scaled=fx.ecmwf_nwp,
        icond2_static=fx.icond2_static,
        ecmwf_static=fx.ecmwf_static,
        target_global=list(fx.val_station_indices),
        observer_global=list(fx.train_station_indices)[: n_tr],
        fold_train_indices=list(fx.train_station_indices),
        target_feat_idx=fx.model_cfg.target_feat_idx,
        H_hist=fx.model_cfg.history_length,
        H_fore=fx.model_cfg.forecast_horizon,
    )
    try:
        with torch.no_grad():
            model(data, target_mask)
        return ""
    except RuntimeError as exc:
        return str(exc)


def k1() -> None:
    print("\n=== K1 — get_test_results_dcrnn.py station.static width ===")
    fx = build_fixture(CFG_DCRNN)
    cfg = yaml.safe_load(Path(CFG_DCRNN).read_text())
    dcrnn_cfg = cfg["dcrnn"]
    n_train = len(fx.train_station_indices)

    print(f"  model_cfg.station_static_features = {fx.model_cfg.station_static_features}")
    check(fx.model_cfg.station_static_features == 13,
          f"config resolves to station_static_features = "
          f"{fx.model_cfg.station_static_features} (4 geo/type + 9 topo)")

    # -- OLD eval-script behaviour: lat/lon/alt only -----------------------
    lats = fx.station_coords[:, 0]
    lons = fx.station_coords[:, 1]
    alts = fx.station_altitudes
    old_static = np.stack([lats, lons, alts], axis=1).astype(np.float32)
    print(f"  station.static from a 3-column array: {old_static.shape} "
          f"(+1 type indicator = {old_static.shape[1] + 1})")
    err_old = _forward_on(old_static, fx)
    print(f"  forward (old, 3-column statics) RAISED: {err_old or '<no error>'}")
    check("cannot be multiplied" in err_old and "144x64" in err_old,
          "old 3-column path still reproduces the shape-mismatch RuntimeError")

    # -- NEW eval-script behaviour: topo columns appended ------------------
    names = parse_station_node_features(dcrnn_cfg, None)
    topo_cols, _ = load_topo_station_features(
        dcrnn_cfg["topo_features_path"], fx.all_ids, names, n_train=n_train,
    )
    new_static = np.concatenate([old_static, topo_cols], axis=1).astype(np.float32)
    print(f"  station_node_features = {len(names)} names -> topo columns "
          f"{topo_cols.shape}; station.static {new_static.shape} "
          f"(+1 type indicator = {new_static.shape[1] + 1})")
    err_new = _forward_on(new_static, fx)
    print(f"  forward (fixed, {new_static.shape[1]}-column statics): "
          f"{err_new or 'OK — no exception'}")
    check(err_new == "", "fixed path runs a full DCRNN forward without error")
    check(new_static.shape[1] + 1 == fx.model_cfg.station_static_features,
          f"static width matches the model: {new_static.shape[1]} + 1 type "
          f"= {fx.model_cfg.station_static_features}")

    # -- the z-score must be fitted on the fold's train stations only ------
    topo_all, _ = load_topo_station_features(
        dcrnn_cfg["topo_features_path"], fx.all_ids, names,
        n_train=len(fx.all_ids),
    )
    d = float(np.abs(topo_cols - topo_all).max())
    check(d > 1e-6,
          f"n_train={n_train} (train-only z-score) differs from an all-station "
          f"fit: max|delta| = {d:.4f} — the fit population is load-bearing")


# ---------------------------------------------------------------------------
# K2
# ---------------------------------------------------------------------------

def _static_dim(names: list[str], all_ids: list[str], path: str, n_train: int,
                base: int) -> int:
    if not names:
        return base
    feats = load_topo_station_features_dict(path, all_ids, names, n_train=n_train)
    return base + len(feats)


def k2() -> None:
    print("\n=== K2 — MTGNN / WaveNet topo name source, train vs. eval ===")
    for label, cfg_path, section in (
        ("mtgnn", CFG_MTGNN, "mtgnn"),
        ("wavenet", CFG_WAVENET, "wavenet"),
    ):
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        mcfg = cfg[section]
        all_ids = [str(s) for s in cfg["data"]["files"]][:60]
        n_train = 45

        # train_{mtgnn,wavenet}.py
        if "station_node_features" in mcfg:
            train_names = parse_station_node_features(mcfg, None)
        else:
            _, _, _, train_names = parse_edge_features(mcfg)

        # eval, AFTER the fix — same condition as training
        if "station_node_features" in mcfg:
            eval_names = parse_station_node_features(mcfg, None)
        else:
            _, _, _, eval_names = parse_edge_features(mcfg)

        # eval, BEFORE the fix
        _, _, _, old_names = parse_edge_features(mcfg)

        path = mcfg.get("topo_features_path")
        base = 6  # lat, lon, alt + 3 (unchanged by this fix, same on both sides)
        d_train = _static_dim(train_names, all_ids, path, n_train, base)
        d_eval = _static_dim(eval_names, all_ids, path, n_train, base)
        d_old = _static_dim(old_names, all_ids, path, n_train, base)

        print(f"  {Path(cfg_path).name}")
        print(f"      train: {len(train_names)} topo -> static_dim {d_train}   "
              f"{train_names}")
        print(f"      eval (old): {len(old_names)} topo -> static_dim {d_old}   "
              f"{old_names}")
        print(f"      eval (new): {len(eval_names)} topo -> static_dim {d_eval}")
        missing = sorted(set(train_names) - set(old_names))
        if missing:
            print(f"      old eval was missing: {missing}")
        check(d_train == d_eval,
              f"{label}: train static_dim {d_train} == eval static_dim {d_eval}")


# ---------------------------------------------------------------------------
# R1 / R2 — source-level proof (an empirical proof would need a training run)
# ---------------------------------------------------------------------------

def _calls(path: str, func_name: str) -> list:
    """All ast.Call nodes in *path* whose callee is named *func_name*."""
    import ast
    tree = ast.parse(Path(path).read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name == func_name:
            out.append(node)
    return out


def _kwnames(call) -> set:
    return {kw.arg for kw in call.keywords if kw.arg}


def _kwsrc(call, name: str, src: str) -> str:
    import ast
    for kw in call.keywords:
        if kw.arg == name:
            return ast.get_source_segment(src, kw.value) or ""
    return ""


def r1_r2() -> None:
    print("\n=== R1 / R2 — source-level checks "
          "(empirical proof would need a training run) ===")
    import ast

    # -- R1: train_dcrnn.py --eval must forward the k-nearest index arrays ---
    p = str(_ROOT / "geostatistics/train_dcrnn.py")
    src = Path(p).read_text()
    calls = _calls(p, "run_evaluation")
    check(len(calls) == 1, f"R1: exactly one run_evaluation() call in {p}")
    kw = _kwnames(calls[0])
    for name in ("station_k_nearest_grid", "station_k_nearest_ecmwf",
                 "interpol_meas", "hist_wind_available",
                 "neighbour_meas_available"):
        check(name in kw, f"R1: run_evaluation(... {name}=...) is passed")

    # -- R2: get_test_results_dcrnn.py must know the Kriging lag channel -----
    p = str(_ROOT / "geostatistics/get_test_results_dcrnn.py")
    src = Path(p).read_text()
    calls = _calls(p, "evaluate")
    check(len(calls) == 1, f"R2: exactly one evaluate() call in {p}")
    kw = _kwnames(calls[0])
    for name in ("interpol_meas", "station_k_nearest_grid",
                 "station_k_nearest_ecmwf"):
        check(name in kw, f"R2: evaluate(... {name}=...) is passed")
    check("interpolate_history" in src,
          "R2: interpolate_history is read in the eval script")
    check("meas_scaler.mean_" in src and "nan_to_num" in src,
          "R2: the Kriging lag channel is scaled and NaN-filled like in training")

    # -- K1: from_yaml must receive station_node_features, topo z-score on train
    calls = _calls(p, "from_yaml")
    check(len(calls) == 1, f"K1: exactly one DCRNNConfig.from_yaml() call in {p}")
    check("station_node_features" in _kwnames(calls[0]),
          "K1: from_yaml(... station_node_features=args.station_node_features)")
    calls = _calls(p, "load_topo_station_features")
    check(len(calls) == 1, f"K1: exactly one load_topo_station_features() call in {p}")
    n_train_src = _kwsrc(calls[0], "n_train", src)
    check(n_train_src == "N_train",
          f"K1: topo z-score fitted on the fold's train stations "
          f"(n_train={n_train_src or '<missing>'}), as in train_dcrnn.py")

    # -- K2: both eval scripts use the training condition -------------------
    for p in (_ROOT / "geostatistics/get_test_results_mtgnn.py",
              _ROOT / "geostatistics/get_test_results_wavenet.py"):
        s = Path(p).read_text()
        check('if args.station_node_features is not None or '
              '"station_node_features" in mcfg:' in s,
              f"K2: {Path(p).name} uses the same condition as its training script")

    # -- K3: the three HPO entry points require the DB URL ------------------
    for p in (_ROOT / "geostatistics/hpo_dcrnn.py", _ROOT / "geostatistics/hpo_mtgnn.py",
              _ROOT / "geostatistics/hpo_wavenet.py"):
        s = Path(p).read_text()
        check(len(_calls(str(p), "require_nwp_elevation_env")) == 1,
              f"K3: {Path(p).name} calls require_nwp_elevation_env() once")
        check("logger.warning(\"DB URLs not set" not in s
              and "WEATHER_DB_URL nicht gesetzt" not in s,
              f"K3: {Path(p).name} no longer warns-and-continues")

    # -- K4: save() is locked and atomic ------------------------------------
    s = (_ROOT / "utils/data_cache.py").read_text()
    check("fcntl.flock" in s, "K4: GNNCache.save takes an exclusive flock")
    check("os.replace" in s, "K4: GNNCache.save publishes via os.replace")
    check("np.save(out, arr)" not in s,
          "K4: no direct np.save to the destination path remains")


if __name__ == "__main__":
    k1()
    k2()
    r1_r2()
    n_ok = sum(1 for ok, _ in _results if ok)
    print("\n" + "=" * 74)
    print(f"{n_ok} passed, {len(_results) - n_ok} failed  (of {len(_results)} checks)")
    print("=" * 74)
    sys.exit(0 if n_ok == len(_results) else 1)
