#!/usr/bin/env python3
"""
verify.py — verification suite for ablation variants B and C
(``docs/implementation_plan_ablations.md`` §4, checks 2–6).

Check 1 (variant A untouched) lives in ``batch_fingerprint.py`` because it has to
run on both sides of the code change and therefore must not reference any
ablation API.

None of the checks below needs data files or a GPU; the whole suite runs on CPU
in a few seconds.

    cd ~/Work/forecasting_framework
    ./frcst/bin/python -m archiv.ablations_verification.verify
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from archiv.ablations_verification.fixture import build_fixture          # noqa: E402
from geostatistics.ablations.guard import (                        # noqa: E402
    AblationConfigError, check_ablation_flags,
)
from geostatistics.dcrnn import DCRNN                              # noqa: E402
from geostatistics.stgnn.training.sampler import TrainingSampler   # noqa: E402

CONFIG_A = "configs/dcrnn/config_wind_dcrnn.yaml"

VARIANT_B = {"neighbour_meas_available": False}
VARIANT_C = {"neighbour_meas_available": False,
             "station_connectivity": "none",
             "direction_to_adj": False}

SEED = 20260803
BATCH_SEED = 4711

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _sampler(fx, **flags) -> TrainingSampler:
    return TrainingSampler(
        fx.model_cfg, fx.builder, fx.base_graph,
        target_feat_idx=fx.model_cfg.target_feat_idx,
        station_coords=fx.station_coords,
        **flags,
    )


def _seed(n: int = BATCH_SEED) -> None:
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)


# ---------------------------------------------------------------------------
# Check 2 — variant B zeroes every measurement channel
# ---------------------------------------------------------------------------

def check_2_b_zeroes(fx_a) -> None:
    print("\n§4.2  B actually zeroes everything")
    M = fx_a.M
    t = fx_a.H_hist + 3

    s_a = _sampler(fx_a, hist_wind_available=False, neighbour_meas_available=True)
    s_b = _sampler(fx_a, hist_wind_available=False, neighbour_meas_available=False)
    # hist_wind_available=True is the nwp_hist arm; B must still win over it.
    s_bh = _sampler(fx_a, hist_wind_available=True, neighbour_meas_available=False)

    _seed(); b_a = s_a.sample_train(r_curr=2, r_hist=1, t_run_abs=t, **fx_a.sample_train_kwargs())
    _seed(); b_b = s_b.sample_train(r_curr=2, r_hist=1, t_run_abs=t, **fx_a.sample_train_kwargs())
    _seed(); b_bh = s_bh.sample_train(r_curr=2, r_hist=1, t_run_abs=t, **fx_a.sample_train_kwargs())

    meas_a = b_a.data["station"].x[:, :fx_a.H_hist, :M]
    meas_b = b_b.data["station"].x[:, :fx_a.H_hist, :M]
    meas_bh = b_bh.data["station"].x[:, :fx_a.H_hist, :M]

    n_nonzero_a = int((meas_a != 0).sum())
    n_nonzero_b = int((meas_b != 0).sum())
    n_nonzero_bh = int((meas_bh != 0).sum())
    record(
        "train: B zeroes all measurement cells",
        n_nonzero_b == 0,
        f"A has {n_nonzero_a} non-zero of {meas_a.numel()} meas cells, "
        f"B has {n_nonzero_b}",
    )
    record(
        "train: B dominates hist_wind_available=True (masking order)",
        n_nonzero_bh == 0,
        f"B+hist_wind_available=True → {n_nonzero_bh} non-zero meas cells "
        f"(would be {n_nonzero_a} if the branches were swapped)",
    )

    n_tgt_a = int(b_a.target_mask.sum())
    n_tgt_b = int(b_b.target_mask.sum())
    record(
        "train: target_mask unchanged",
        n_tgt_a == n_tgt_b and b_a.target_mask.equal(b_b.target_mask),
        f"|target| A={n_tgt_a}  B={n_tgt_b}, masks identical="
        f"{bool(b_a.target_mask.equal(b_b.target_mask))}, "
        f"n_nodes A={b_a.target_mask.numel()} B={b_b.target_mask.numel()}",
    )
    record(
        "train: ground_truth unchanged (identical scoring)",
        bool(torch.equal(b_a.ground_truth, b_b.ground_truth)),
        f"gt shape {tuple(b_a.ground_truth.shape)}, "
        f"max|Δ|={float((b_a.ground_truth - b_b.ground_truth).abs().max()):.3e}",
    )
    record(
        "train: NWP channels of station.x untouched by B",
        bool(torch.equal(b_a.data["station"].x[:, :, M:],
                         b_b.data["station"].x[:, :, M:])),
        f"{b_a.data['station'].x[:, :, M:].numel()} NWP cells bit-identical",
    )

    # -- validation twin -------------------------------------------------
    _seed(); v_a = s_a.sample_val(r_curr=2, r_hist=1, t_run_abs=t, **fx_a.sample_val_kwargs())
    _seed(); v_b = s_b.sample_val(r_curr=2, r_hist=1, t_run_abs=t, **fx_a.sample_val_kwargs())
    _seed(); v_bh = s_bh.sample_val(r_curr=2, r_hist=1, t_run_abs=t, **fx_a.sample_val_kwargs())
    vm_a = v_a.data["station"].x[:, :fx_a.H_hist, :M]
    vm_b = v_b.data["station"].x[:, :fx_a.H_hist, :M]
    vm_bh = v_bh.data["station"].x[:, :fx_a.H_hist, :M]
    record(
        "val: B zeroes all measurement cells",
        int((vm_b != 0).sum()) == 0,
        f"A has {int((vm_a != 0).sum())} non-zero of {vm_a.numel()}, "
        f"B has {int((vm_b != 0).sum())}",
    )
    record(
        "val: B dominates hist_wind_available=True (masking order)",
        int((vm_bh != 0).sum()) == 0,
        f"{int((vm_bh != 0).sum())} non-zero meas cells",
    )

    # -- evaluation.py twin ----------------------------------------------
    from geostatistics.evaluation import build_eval_batch

    common = dict(
        sampler=s_a,
        r_curr=2, r_hist=1, t_run_abs=t,
        station_meas_scaled=fx_a.station_meas,
        station_nearest_grid=fx_a.station_nearest_grid,
        grid_icond2_runs_scaled=fx_a.grid_icond2_runs,
        station_ecmwf_nwp_scaled=fx_a.station_ecmwf_nwp,
        station_static=fx_a.station_static,
        ecmwf_nwp_scaled=fx_a.ecmwf_nwp,
        icond2_static=fx_a.icond2_static,
        ecmwf_static=fx_a.ecmwf_static,
        target_global=fx_a.val_station_indices,
        observer_global=fx_a.train_station_indices,
        fold_train_indices=fx_a.train_station_indices,
        target_feat_idx=fx_a.model_cfg.target_feat_idx,
        H_hist=fx_a.H_hist, H_fore=fx_a.H_fore,
    )
    d_a, _, _ = build_eval_batch(**common, hist_wind_available=False)
    d_b, _, _ = build_eval_batch(**common, hist_wind_available=False,
                                 neighbour_meas_available=False)
    d_bh, _, _ = build_eval_batch(**common, hist_wind_available=True,
                                  neighbour_meas_available=False)
    em_a = d_a["station"].x[:, :fx_a.H_hist, :M]
    em_b = d_b["station"].x[:, :fx_a.H_hist, :M]
    em_bh = d_bh["station"].x[:, :fx_a.H_hist, :M]
    record(
        "evaluation.build_eval_batch: B zeroes all measurement cells",
        int((em_b != 0).sum()) == 0,
        f"A has {int((em_a != 0).sum())} non-zero of {em_a.numel()}, "
        f"B has {int((em_b != 0).sum())}",
    )
    record(
        "evaluation.build_eval_batch: masking order",
        int((em_bh != 0).sum()) == 0,
        f"B+hist_wind_available=True → {int((em_bh != 0).sum())} non-zero",
    )


# ---------------------------------------------------------------------------
# Check 3 — variant C builds an empty station edge set
# ---------------------------------------------------------------------------

def check_3_empty_edges(fx_a, fx_c) -> None:
    print("\n§4.3  C builds an empty station edge set")
    ei_a = fx_a.base_graph["station", "near", "station"].edge_index
    ea_a = fx_a.base_graph["station", "near", "station"].edge_attr
    ei_c = fx_c.base_graph["station", "near", "station"].edge_index
    ea_c = fx_c.base_graph["station", "near", "station"].edge_attr

    record(
        "graph: edge_index is (2, 0)",
        tuple(ei_c.shape) == (2, 0) and ei_c.dtype == torch.long,
        f"A={tuple(ei_a.shape)}  C={tuple(ei_c.shape)} dtype={ei_c.dtype}",
    )
    record(
        "graph: edge_attr is (0, F) with the *probed* F",
        ea_c.shape[0] == 0 and int(ea_c.shape[1]) == int(ea_a.shape[1]),
        f"A={tuple(ea_a.shape)}  C={tuple(ea_c.shape)} — "
        f"F={int(ea_c.shape[1])} matches A's edge feature width "
        f"(1 dist + 2 dir + 1 alt + {int(ea_a.shape[1]) - 4} topo)",
    )
    record(
        "graph: NWP→station edges unaffected",
        (fx_c.base_graph["icond2", "informs", "station"].edge_index.shape
         == fx_a.base_graph["icond2", "informs", "station"].edge_index.shape)
        and (fx_c.base_graph["ecmwf", "informs", "station"].edge_index.shape
             == fx_a.base_graph["ecmwf", "informs", "station"].edge_index.shape),
        f"i2s {tuple(fx_c.base_graph['icond2', 'informs', 'station'].edge_index.shape)}, "
        f"e2s {tuple(fx_c.base_graph['ecmwf', 'informs', 'station'].edge_index.shape)}",
    )

    subset = list(range(0, fx_c.N_all, 2))
    try:
        sub_ei, sub_ea = fx_c.builder.subgraph_station_edges(fx_c.base_graph, subset)
        ok = tuple(sub_ei.shape) == (2, 0) and sub_ea.shape[0] == 0
        record(
            "subgraph_station_edges on the empty graph",
            ok,
            f"returns edge_index={tuple(sub_ei.shape)} dtype={sub_ei.dtype}, "
            f"edge_attr={tuple(sub_ea.shape)} for a {len(subset)}-node subset "
            "(no exception)",
        )
    except Exception as exc:                                  # noqa: BLE001
        record("subgraph_station_edges on the empty graph", False, f"raised {exc!r}")


# ---------------------------------------------------------------------------
# Check 4 — variant C produces finite output
# ---------------------------------------------------------------------------

def _forward(fx, sampler, seed_model: int = 7) -> tuple[torch.Tensor, object]:
    torch.manual_seed(seed_model)
    model = DCRNN(fx.model_cfg)
    model.eval()
    _seed()
    batch = sampler.sample_val(
        r_curr=2, r_hist=1, t_run_abs=fx.H_hist + 3, **fx.sample_val_kwargs(),
    )
    with torch.no_grad():
        pred = model(batch.data, batch.target_mask)
    return pred, (model, batch)


def check_4_finite(fx_c) -> None:
    print("\n§4.4  C produces finite output")
    s_c = _sampler(fx_c, hist_wind_available=False, neighbour_meas_available=False)
    pred, _ = _forward(fx_c, s_c)
    finite = bool(torch.isfinite(pred).all())
    record(
        "DCRNN forward pass on an empty station graph",
        finite,
        f"pred shape {tuple(pred.shape)}, finite={finite}, "
        f"min={float(pred.min()):.6f} max={float(pred.max()):.6f} "
        f"mean={float(pred.mean()):.6f}, NaN={int(torch.isnan(pred).sum())}, "
        f"Inf={int(torch.isinf(pred).sum())}",
    )


# ---------------------------------------------------------------------------
# Check 5 — variant C is genuinely graph-free (permutation test)
# ---------------------------------------------------------------------------

def _permute_neighbours(batch, n_neigh: int, rng: np.random.Generator):
    """Return a deep copy of the batch with the NEIGHBOUR stations permuted.

    Permuted are all three channels a neighbour could speak through:
      * ``station.x``      — measurements + the station's own NWP columns
      * ``station.static`` — lat/lon/alt + the nine topographic node features
      * the destination index of the icond2→station and ecmwf→station edges,
        i.e. each neighbour is handed a different neighbour's grid points and
        edge geometry.
    Target stations (indices >= n_neigh) are never touched, and the type
    indicator in ``static`` is constant across neighbours, so the permutation
    cannot leak through it either.
    """
    import copy
    data = copy.deepcopy(batch.data)

    perm = np.arange(n_neigh)
    while True:
        rng.shuffle(perm)
        if not np.array_equal(perm, np.arange(n_neigh)):
            break
    perm_t = torch.from_numpy(perm.copy()).long()

    data["station"].x[:n_neigh] = data["station"].x[perm_t]
    data["station"].static[:n_neigh] = data["station"].static[perm_t]

    full_perm = torch.arange(data["station"].x.size(0))
    full_perm[:n_neigh] = perm_t
    for ekey in (("icond2", "informs", "station"), ("ecmwf", "informs", "station")):
        ei = data[ekey].edge_index
        if ei.numel() == 0:
            continue
        data[ekey].edge_index = torch.stack([ei[0], full_perm[ei[1]]], dim=0)

    return data, perm


def _permutation_delta(fx, sampler, label: str, seed_model: int = 7) -> float:
    pred0, (model, batch) = _forward(fx, sampler, seed_model)
    n_neigh = int((~batch.target_mask).sum())
    data_p, perm = _permute_neighbours(batch, n_neigh, np.random.default_rng(99))
    with torch.no_grad():
        pred1 = model(data_p, batch.target_mask)
    delta = float((pred1 - pred0).abs().max())
    n_moved = int((perm != np.arange(n_neigh)).sum())
    print(f"      {label}: {n_neigh} neighbours ({n_moved} moved), "
          f"{int(batch.target_mask.sum())} targets, max|Δpred| = {delta:.3e}")
    return delta


def check_5_graph_free(fx_a, fx_b, fx_c) -> None:
    print("\n§4.5  C is genuinely graph-free (neighbour permutation)")
    s_a = _sampler(fx_a, hist_wind_available=False, neighbour_meas_available=True)
    s_b = _sampler(fx_b, hist_wind_available=False, neighbour_meas_available=False)
    s_c = _sampler(fx_c, hist_wind_available=False, neighbour_meas_available=False)

    d_c = _permutation_delta(fx_c, s_c, "C (no graph, no meas)")
    d_b = _permutation_delta(fx_b, s_b, "B (graph, no meas)   ")
    d_a = _permutation_delta(fx_a, s_a, "A (graph + meas)     ")

    tol = 1e-6
    record(
        "C: target predictions invariant under neighbour permutation",
        d_c <= tol,
        f"max|Δpred| = {d_c:.3e} (tolerance {tol:.0e})",
    )
    record(
        "control B: the same permutation DOES move the prediction",
        d_b > 1e-4,
        f"max|Δpred| = {d_b:.3e} — proves the test is sensitive to the "
        "geometry/NWP-context channel alone, so C's 0 is not an artefact of "
        "the zeroed measurements",
    )
    record(
        "control A: the same permutation DOES move the prediction",
        d_a > 1e-4,
        f"max|Δpred| = {d_a:.3e}",
    )


# ---------------------------------------------------------------------------
# Check 6 — the Kriging-lag assertion fires
# ---------------------------------------------------------------------------

def check_6_assertion() -> None:
    print("\n§4.6  Kriging-lag assertion")
    import logging
    log = logging.getLogger("verify_ablations")

    bad = {"neighbour_meas_available": False, "interpolate_history": True}
    try:
        check_ablation_flags(bad, log)
        record("assertion fires on B + interpolate_history", False, "no exception raised")
    except AblationConfigError as exc:
        record("assertion fires on B + interpolate_history", True,
               f"AblationConfigError: {str(exc)[:90]}…")

    good_a = {"neighbour_meas_available": True, "interpolate_history": True}
    try:
        f = check_ablation_flags(good_a, log)
        record("variant A with interpolate_history still allowed", True,
               f"variant detected = {f['variant']}")
    except AblationConfigError as exc:
        record("variant A with interpolate_history still allowed", False, repr(exc))

    good_b = {"neighbour_meas_available": False, "interpolate_history": False}
    f = check_ablation_flags(good_b, log)
    record("variant detection B", f["variant"].startswith("B"), f["variant"])
    good_c = {"neighbour_meas_available": False, "station_connectivity": "none"}
    f = check_ablation_flags(good_c, log)
    record("variant detection C", f["variant"].startswith("C"), f["variant"])

    # The three production entry points must all call the guard.
    root = Path(__file__).resolve().parents[2]
    for rel in ("geostatistics/train_dcrnn.py",
                "geostatistics/get_test_results_dcrnn.py",
                "geostatistics/hpo_dcrnn.py"):
        src = (root / rel).read_text()
        record(f"guard wired into {Path(rel).name}",
               "check_ablation_flags(dcrnn_cfg" in src,
               "check_ablation_flags(dcrnn_cfg, logger) present")


# ---------------------------------------------------------------------------
# Check 0 — the config parser actually reads the flag
# ---------------------------------------------------------------------------

def check_0_config(fx_a, fx_b, fx_c) -> None:
    print("\n§0  config plumbing")
    record("DCRNNConfig default (variant A)",
           fx_a.model_cfg.neighbour_meas_available is True,
           f"neighbour_meas_available={fx_a.model_cfg.neighbour_meas_available}, "
           f"station_connectivity={fx_a.model_cfg.graph.station_connectivity!r}")
    record("DCRNNConfig variant B",
           fx_b.model_cfg.neighbour_meas_available is False
           and fx_b.model_cfg.graph.station_connectivity == "delaunay",
           f"neighbour_meas_available={fx_b.model_cfg.neighbour_meas_available}, "
           f"station_connectivity={fx_b.model_cfg.graph.station_connectivity!r}")
    record("DCRNNConfig variant C",
           fx_c.model_cfg.neighbour_meas_available is False
           and fx_c.model_cfg.graph.station_connectivity == "none",
           f"neighbour_meas_available={fx_c.model_cfg.neighbour_meas_available}, "
           f"station_connectivity={fx_c.model_cfg.graph.station_connectivity!r}, "
           f"direction_to_adj={fx_c.model_cfg.direction_to_adj}")
    record("interpolate_history off in variant A config",
           fx_a.model_cfg.interpolate_history is False,
           f"interpolate_history={fx_a.model_cfg.interpolate_history} "
           "(the Kriging channel that would invalidate B)")


# ---------------------------------------------------------------------------
# Check 7 — the generated variant configs differ only on the ablated axis
# ---------------------------------------------------------------------------

def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _study_name(config_path: str, dcrnn_cfg: dict, data_cfg: dict) -> str:
    """Reproduce train_dcrnn.py / hpo_dcrnn.py study-name derivation."""
    import re as _re
    stem = Path(config_path).stem.replace("config_", "")
    hpo_stem = _re.sub(r"_fold\d+$", "", stem)
    freq = data_cfg.get("freq", "1h")
    h = dcrnn_cfg.get("forecast_horizon", 48)
    return f"cl_m-dcrnn_out-{h}_freq-{freq}_{hpo_stem}"


def check_7_configs(cfg_dir: str) -> None:
    print("\n§4.7a  generated variant configs (semantic diff against variant A)")
    import yaml

    d = Path(cfg_dir)
    expected = {
        "nomeas": {"neighbour_meas_available": False},
        "nograph": {"neighbour_meas_available": False,
                    "station_connectivity": "none",
                    "direction_to_adj": False},
    }
    # Deliberate, documented deviations on the *study* (non-fold) configs:
    #   * reduced HPO budget, 150 → 60
    #   * variant C only: K_hop / next_n_neighbors / edge_weight_sigma pinned,
    #     i.e. removed from the HPO search space because the permutation test
    #     proves they cannot influence C (max|Δpred| = 0.0) — edge_weight_sigma
    #     scales the station-edge kernel, of which C has no edges at all. Same
    #     move as config_wind_dcrnn_base.yaml, which dropped nwp_heads /
    #     nwp_out_per_head for the same reason.
    budget_key = "hpo.trials"
    PINNED_IN_C = ("K_hop", "next_n_neighbors", "edge_weight_sigma")
    pinned_keys = {f"dcrnn.hpo.params.{p}.{f}"
                   for p in PINNED_IN_C for f in ("type", "low", "high", "step", "log")}

    for variant, want in expected.items():
        for fold in ("", "_fold1", "_fold2", "_fold3"):
            src = d / f"config_wind_dcrnn{fold}.yaml"
            dst = d / f"config_wind_dcrnn_{variant}{fold}.yaml"
            if not dst.exists():
                record(f"{dst.name} exists", False, "missing")
                continue
            a = yaml.safe_load(src.read_text())
            b = yaml.safe_load(dst.read_text())

            fa, fb = _flat(a), _flat(b)
            changed = {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}
            allowed = {f"dcrnn.{k}" for k in want}
            if fold == "":
                allowed.add(f"dcrnn.{budget_key}")
                if variant == "nograph":
                    allowed |= pinned_keys
            unexpected = changed - allowed
            record(
                f"{dst.name}: only the ablated axis differs",
                not unexpected,
                f"{len(changed)} key(s) differ: "
                + ", ".join(f"{k}: {fa.get(k, '<absent>')!r}→{fb.get(k, '<absent>')!r}"
                            for k in sorted(changed))
                + (f"   UNEXPECTED: {sorted(unexpected)}" if unexpected else ""),
            )

            vals_ok = all(b["dcrnn"].get(k) == v for k, v in want.items())
            record(f"{dst.name}: flag values",
                   vals_ok,
                   ", ".join(f"{k}={b['dcrnn'].get(k)!r}" for k in want))

            inherited = {k: b["dcrnn"].get(k) for k in ("station_node_features",)}
            inherited["hpo.trials"] = b["dcrnn"].get("hpo", {}).get("trials")
            record(f"{dst.name}: inherited campaign settings",
                   b["dcrnn"].get("station_node_features") == a["dcrnn"].get("station_node_features"),
                   f"{inherited} (A: station_node_features="
                   f"{a['dcrnn'].get('station_node_features')!r}, "
                   f"hpo.trials={a['dcrnn'].get('hpo', {}).get('trials')})")

            record(f"{dst.name}: Kriging channel off",
                   b["dcrnn"].get("interpolate_history", False) is False,
                   f"interpolate_history={b['dcrnn'].get('interpolate_history', False)}")

            sn = _study_name(str(dst), b["dcrnn"], b["data"])
            want_sn = f"cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_{variant}"
            record(f"{dst.name}: Optuna study resolution",
                   sn == want_sn, sn)

            # Inert parameters in C (plan §9.2, decided: pin them).
            params = b["dcrnn"].get("hpo", {}).get("params", {})
            a_params = a["dcrnn"].get("hpo", {}).get("params", {})
            if variant == "nograph" and fold == "":
                gone = [p for p in PINNED_IN_C if p not in params]
                statics_kept = {p: b["dcrnn"].get(p) for p in PINNED_IN_C}
                record(
                    f"{dst.name}: inert params removed from the search space",
                    len(gone) == len(PINNED_IN_C),
                    f"absent from hpo.params: {gone} (A had "
                    f"{[p for p in PINNED_IN_C if p in a_params]}); "
                    f"search space {len(a_params)} → {len(params)} parameters",
                )
                record(
                    f"{dst.name}: static values of the pinned params kept",
                    all(statics_kept[p] == a["dcrnn"].get(p) for p in PINNED_IN_C),
                    f"{statics_kept} — identical to A, so C's batch composition "
                    "still matches A and B",
                )
            elif variant == "nomeas" and fold == "":
                record(
                    f"{dst.name}: search space untouched (B still uses the graph)",
                    all(p in params for p in PINNED_IN_C)
                    and len(params) == len(a_params),
                    f"{len(params)} parameters, same as A; "
                    f"{list(PINNED_IN_C)} still searched",
                )

            # parse through the production config parser
            try:
                from archiv.ablations_verification.fixture import build_fixture as _bf
                fx = _bf(str(dst), seed=SEED, n_stations=20,
                         n_grid_lat=6, n_grid_lon=6, n_ecmwf_lat=4, n_ecmwf_lon=4)
                n_edges = int(fx.base_graph["station", "near", "station"].edge_index.shape[1])
                ok = (fx.model_cfg.neighbour_meas_available is False
                      and (n_edges == 0) == (variant == "nograph"))
                record(f"{dst.name}: parses and builds the intended graph",
                       ok,
                       f"neighbour_meas_available="
                       f"{fx.model_cfg.neighbour_meas_available}, s2s edges={n_edges}")
            except Exception as exc:                          # noqa: BLE001
                record(f"{dst.name}: parses and builds the intended graph", False, repr(exc))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_A)
    ap.add_argument("--config-dir", default="configs/dcrnn")
    ap.add_argument("--skip-configs", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("Ablation verification suite — plan §4.2 … §4.6  (no data, no GPU)")
    print(f"config: {args.config}   seed: {SEED}")
    print("=" * 78)

    fx_a = build_fixture(args.config, seed=SEED)
    fx_b = build_fixture(args.config, seed=SEED, overrides=VARIANT_B)
    fx_c = build_fixture(args.config, seed=SEED, overrides=VARIANT_C)

    check_0_config(fx_a, fx_b, fx_c)
    check_2_b_zeroes(fx_a)
    check_3_empty_edges(fx_a, fx_c)
    check_4_finite(fx_c)
    check_5_graph_free(fx_a, fx_b, fx_c)
    check_6_assertion()
    if not args.skip_configs:
        check_7_configs(args.config_dir)

    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_fail = len(_results) - n_pass
    print("\n" + "=" * 78)
    print(f"{n_pass} passed, {n_fail} failed  (of {len(_results)} checks)")
    print("=" * 78)
    if n_fail:
        for name, ok, detail in _results:
            if not ok:
                print(f"  FAILED: {name} — {detail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
