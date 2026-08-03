# Implementation Plan: Ablation Variants B (no neighbour measurements) and C (no station graph)

**Goal.** Two additional trained variants of DCRNN that decompose the model's spatial machinery:

| Variant | Own k grid points | Neighbour **measurements** | Neighbour **geometry / NWP context** |
|---|---|---|---|
| **A** = existing `config_wind_dcrnn.yaml` | yes | yes | yes |
| **B** = no neighbour measurements | yes | **no** | yes |
| **C** = no station graph | yes | no | **no** |

A minus B = value of the neighbour measurements. B minus C = value of the geometry and context channel.
C = the pure per-site downscaling floor.

**Status, 3 August 2026 — current.** Everything except the short training run (§4.7) is
**implemented, committed and rolled out to all three hosts** as commit `3f8d6b8` on branch
`fix/mtgnn-topo-static-dim` (parent `674a043`). Full evidence with concrete numbers:
[`docs/ablations_verification_results.md`](ablations_verification_results.md).

| Item | State |
|---|---|
| `neighbour_meas_available` flag threaded through 6 files | **done** (`dcrnn/config.py`, `stgnn/training/sampler.py` x2, `evaluation.py` x2, `train_dcrnn.py` x2, `get_test_results_dcrnn.py`, `hpo_dcrnn.py`) |
| `station_connectivity: "none"` branch in the graph builder | **done**, edge-feature width probed (F = 12, incl. the 8 topo columns; a flags-only count would have given 4) |
| Assertion + startup banner (`geostatistics/ablations/guard.py`) | **done**, wired into `train_dcrnn.py`, `get_test_results_dcrnn.py` **and** `hpo_dcrnn.py` |
| Variant B configs, 4 files | **done**, generated; semantic diff against A = only `neighbour_meas_available` (+ `hpo.trials` on the study config) |
| Variant C configs, 4 files | **done**, generated; semantic diff against A = only `neighbour_meas_available`, `station_connectivity`, `direction_to_adj` (+ `hpo.trials`) |
| Groups `DCRNN_NOMEAS` / `DCRNN_NOGRAPH` | **done** in `launch_train_pipeline.py` and `launch_eval_pipeline.py` |
| §4.1 variant A untouched | **bit-identical** — 28 SHA-256 tensor fingerprints, before vs. after the patch |
| §4.2–§4.6 verification suite (no data, no GPU) | **76 passed, 0 failed**; §4.5 gives max\|Δpred\| = **0.000e+00** for C against 5.9e-1 for the B and A controls |
| §4.7 short training run | **open** — needs GPU; the A campaign occupies all 14 GPUs and has 0 completed trials |
| Inert parameters in C (§9.2) | **decision still open**; `gen_variant_configs.py --pin-inert` applies it in one command, recommendation in the results doc §4 |
| HPO runs and trainings for B and C | **not started** |

Author decisions incorporated: **own HPO study per variant** (route (b) of
`prompt_ablations_implementation.md` §5.1, i.e. option 2 in §6 below), with a reduced trial budget
of 60 instead of 150; committed and rolled out to `l2`, `l1` and `ws`. Because each variant is now
tuned on its own study, B and C no longer inherit A's hyperparameters — the fairness objection in §6
is largely defused. What remains is the smaller search budget, which still favours A slightly and
makes the measured channel contributions upper bounds.

The helper scripts that used to live in `/tmp` on `l1` are now in the repository under
`geostatistics/ablations/`. `/tmp/patch_ablations.py` was used as a **reference only**: its edits
were re-derived against the current code, and its edge-feature-width probe was corrected.

> **Warning about the rest of this document.** It was written on 30 July 2026. All **line numbers**
> below are stale. §2.3's list of configs carrying `interpolate_history: false` was not re-checked
> file by file — the assertion it motivates now lives in the production code. §9.1 applies only to
> the `_fold1..3` files.

---

## 1. Access and code map, verified

Both hosts reachable over ssh without a password prompt: `l1` = `w-lambdablade1`, `l2` = `w-lambdablade2`.
Code root `~/Work/forecasting_framework/`.

| Concern | File | Detail |
|---|---|---|
| Training masking | `geostatistics/stgnn/training/sampler.py:220-222` | `if not self.hist_wind_available: meas_hist[:, target_mask_np, :] = 0.0` |
| Eval masking | `geostatistics/evaluation.py:118-119` | `if not hist_wind_available: meas_hist[:, N_obs:, :] = 0.0` |
| Station edge construction | `geostatistics/stgnn/graph_builder.py:136-188` | `_build_station_edges`, branches on `cfg.station_connectivity` in `{delaunay, knn}`, else raises |
| Subgraph edge filter | `geostatistics/stgnn/graph_builder.py:~252` | boolean mask over `full_ei[1]` |
| Diffusion convolution | `geostatistics/dcrnn/model/dcgru_cell.py:96-124` | row-normalised `DiffConv`, wrapped bidirectionally at `:174-182` |
| Adjacency hand-off | `geostatistics/dcrnn/model/dcrnn.py:151` | `s2s_ei = data[s2s_key].edge_index`, passed to encoder and decoder |
| Config parsing | `geostatistics/dcrnn/config.py:148,154-158,186` | `station_connectivity`, `min/max_target_stations`, `next_n_neighbors`, `interpolate_history` |
| Existing configs | `configs/dcrnn/config_wind_dcrnn{,_base,_nwp_hist_new}.yaml` plus `_fold1..3` | A = `nwp_nodes: true`, `hist_wind_available: false` |
| Training launcher | `geostatistics/launch_train_pipeline.py` | groups `DCRNN_BASE`, `DCRNN_NWP`, `DCRNN_NWP_HIST`, `MTGNN_*`; `--gpus`, `--groups`, `--dry-run` |
| Eval launcher | `geostatistics/launch_eval_pipeline.py`, `geostatistics/get_test_results_dcrnn.py` | per-variant eval, `hist_wind_available` read at `get_test_results_dcrnn.py:437` |

**Scope decision.** DCRNN only. Ablating the headline architecture is standard practice, and MTGNN carries a
complication that makes it the wrong choice here: its graph-learning module
`A = ReLU(tanh(α(M1M2ᵀ − M2M1ᵀ)))` would rebuild an adjacency from the static-derived node embeddings even
after the predefined graph is removed, so variant C would require disabling two mechanisms and would no
longer be a single-variable change.

---

## 2. Design decisions, and why

### 2.1 Variant B is "zero every measurement channel", not "make every station a target"

The tempting implementation is to set `max_target_stations = N`. Reject it. That would simultaneously change
the subgraph selection, the number of scored nodes per batch, the effective batch size and the loss
composition, so B would differ from A in four ways instead of one.

The correct implementation zeroes the measurement channels for **all** stations while leaving `target_mask`
untouched. Consequences, all of them desirable:

* the loss is still computed only at the designated target stations, so A and B are scored identically;
* the `type_ind` feature appended at `sampler.py:231` stays informative;
* the graph still carries NWP features and statics along station edges, which is exactly the channel B is
  designed to preserve.

### 2.2 Variant C uses a new `station_connectivity` value, not a model-side switch

`_build_station_edges` already dispatches on a string, so adding `"none"` keeps the change in one function
and leaves the model untouched.

**An empty `edge_index` is provably safe in this diffusion convolution.** From `dcgru_cell.py:96-124`:

* `out = self.lins[0](x)` is an unconditional k = 0 self-transform, so the layer always has a path that does
  not depend on any edge;
* `out_deg.clamp(min=1e-8)` removes the division-by-zero risk when no edge writes into `out_deg`;
* `propagate` over an edge set of size zero returns a zero tensor, so every k ≥ 1 term contributes
  `self.lins[k](0)`, which is zero up to the bias.

The DCGRU therefore degenerates to a plain GRU with a linear input transform, which is precisely the intent.

`direction_to_adj` must be set to `false` in the C config. Leaving it on would recompute edge weights over an
empty edge set, which is harmless but would be a second changed variable.

**Fallback if any zero-size tensor misbehaves downstream:** add `"self"` instead, emitting self-loops only
(`src == dst`, distance 0). Then `out_deg` equals the edge weight, the normalisation is exactly 1, `propagate`
returns `x` itself, and the result is still algebraically equivalent to having no graph while avoiding
empty tensors everywhere. Keep this in reserve; do not implement both.

### 2.3 The trap that would silently invalidate variant B

`sampler.py:224-227` appends a **regression-kriging lag channel** after the zeroing step, with the comment
*"append as extra channel after zeroing so target nodes still carry an external prior estimate (not zeroed,
always available)"*. That estimate is interpolated from other stations' measurements. If it is ever enabled,
variant B receives neighbour measurement information through a channel that bypasses the graph entirely, and
the A minus B difference stops measuring what it claims to measure.

Verified current state: `interpolate_history: false` in `config_wind_dcrnn.yaml`, `config_wind_dcrnn_base.yaml`,
`config_wind_dcrnn_nwp_hist_new.yaml` and `config_wind_dcrnn_base_fold1..3.yaml`. The key is absent from
`config_wind_dcrnn_fold1..3.yaml`, which falls back to the default `False` at `config.py:186`.

So the channel is off today. The plan adds a hard assertion rather than trusting that, because this is the one
failure mode that would produce a plausible but meaningless number.

---

## 3. Implementation steps

### Step 1: thread a new flag `neighbour_meas_available`

Copy the wiring of `hist_wind_available` literally. It already passes through every call site that the new
flag needs, which makes this mechanical rather than inventive.

| File | Change |
|---|---|
| `geostatistics/dcrnn/config.py` | add field, `d.get("neighbour_meas_available", True)`, default preserves current behaviour |
| `geostatistics/stgnn/training/sampler.py` | constructor arg + `self.` assignment (next to `:47,56`); at `:220-222` and at the eval-sampler twin `:312` apply the new branch |
| `geostatistics/evaluation.py` | signature arg next to `:76`; at `:118-119` apply the new branch |
| `geostatistics/train_dcrnn.py:911,1001` | pass `dcrnn_cfg.get("neighbour_meas_available", True)` |
| `geostatistics/get_test_results_dcrnn.py:437` | same |
| `geostatistics/hpo_dcrnn.py:872` | same, so a later HPO run inherits the flag |

Masking logic at each of the three sites:

```
if not neighbour_meas_available:
    meas_hist[:, :, :] = 0.0          # variant B: nobody has measurements
elif not hist_wind_available:
    meas_hist[:, target_mask_np, :] = 0.0    # current behaviour
```

Note the order. B subsumes the existing zeroing, so the new branch must come first.

### Step 2: add `station_connectivity: "none"`

In `graph_builder.py:_build_station_edges`, before the existing `delaunay` and `knn` branches:

```
if self.cfg.station_connectivity == "none":
    F = <edge feature dim for the configured flags>
    return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, F), dtype=torch.float32)
```

`F` must match what `edge_features` produces for the configured `use_distance_features`,
`use_direction_features` and `use_altitude_diff`. Derive it by calling `edge_features` once on a synthetic
single-edge input and reading `.shape[1]`, rather than hard-coding a number that will drift.

### Step 3: assertions

* in `train_dcrnn.py`, immediately after config parsing: if `neighbour_meas_available` is `False`, assert
  `interpolate_history is False` and abort with a clear message otherwise;
* same assertion in `get_test_results_dcrnn.py`, so an eval run cannot silently reintroduce the channel;
* log the resolved values of `neighbour_meas_available`, `hist_wind_available`, `interpolate_history` and
  `station_connectivity` in one line at startup, so every log file records which variant it is.

### Step 4: configs

Derive from `config_wind_dcrnn.yaml` and its three fold files so that everything except the ablated axis is
identical.

`config_wind_dcrnn_nomeas{,_fold1,_fold2,_fold3}.yaml`:
```
neighbour_meas_available: false
# everything else copied verbatim from config_wind_dcrnn*.yaml
```

`config_wind_dcrnn_nograph{,_fold1,_fold2,_fold3}.yaml`:
```
neighbour_meas_available: false
station_connectivity: none
direction_to_adj: false
# everything else copied verbatim
```

Two notes on C. It inherits `K_hop` and `next_n_neighbors`, both of which become meaningless once there are no
station edges; that is harmless but should be stated in the paper. And C still samples a station subgraph, so
the batch composition matches A and B even though no message passing occurs.

### Step 5: register the new groups

Add `DCRNN_NOMEAS` and `DCRNN_NOGRAPH` to the group table in `launch_train_pipeline.py`, following the
existing `DCRNN_NWP` entry, and the matching eval entries in `launch_eval_pipeline.py`.

---

## 4. Verification before any GPU time is spent

Run in this order. Each step is cheap and each one catches a distinct failure mode.

1. **Flag default is a no-op.** Instantiate the sampler from the unchanged `config_wind_dcrnn.yaml`, draw one
   batch with a fixed seed before and after the code change, assert the tensors are bit-identical. This proves
   variant A is untouched.
2. **B actually zeroes everything.** Draw one batch with `neighbour_meas_available: false`, assert
   `meas_hist` is all zeros across every station and every measurement channel, and assert that `target_mask`
   still has the same number of `True` entries as in A under the same seed.
3. **C builds an empty edge set.** Build the graph with `station_connectivity: none`, assert
   `edge_index.shape == (2, 0)` and `edge_attr.shape[0] == 0`, then call `_subgraph_station_edges` on it and
   assert it returns empty tensors rather than raising.
4. **C produces finite output.** One forward pass of the full DCRNN on a single C batch, assert
   `torch.isfinite(pred).all()`. This is the empty-tensor NaN check that §2.2 argues should pass.
5. **C is genuinely graph-free.** Take one C batch, permute the measurement and NWP features across
   neighbour stations while leaving the target stations untouched, and assert the prediction at the target
   stations is unchanged to numerical tolerance. This is the strongest available proof that no information
   crosses station boundaries, and it would catch an unnoticed second path.
6. **Assertion fires.** Set `interpolate_history: true` together with `neighbour_meas_available: false` and
   confirm the run aborts.
7. **One short training run.** Two or three epochs of B on a single GPU, confirm the loss decreases and the
   checkpoint round-trips through the eval path.

Only after 7 passes should the full training start.

---

## 5. Training, evaluation, analysis

* 2 variants × 3 folds = **6 training runs**. Wall-clock per DCRNN fold is recoverable from the existing
  `logs/train_dcrnn_*.log` timestamps; measure it before scheduling rather than estimating.
* Evaluate in both existing scenarios, `excl_val` and `incl_val`, so the new rows are directly comparable with
  the existing result tables.
* Report per station: RMSE, MAE, R², `skill_icond2`, `skill_ecmwf`, skill against persistence.
* Pair A, B and C **per station** and test with a paired two-sided Wilcoxon signed-rank test plus Holm
  correction across the comparison family. Assert the per-station index is unique before pairing. This is the
  exact bug that inflated `n` fourfold in the iAIMS26 analysis, and the fold-level result tables here have the
  same shape, so the same trap exists.
* Headline figure: skill contribution of the two channels, either as a waterfall from raw ICON-D2 through C
  and B to A, or as skill against lead time with the three variants as three lines. The second is more
  informative if the channels behave differently at short and long lead times.

---

## 6. The fairness question, stated openly

B and C inherit A's hyperparameters, which were tuned by Optuna for A. That biases the comparison **in favour
of A**, and a referee can legitimately raise it. Three positions, pick one before running:

1. **Inherit and disclose.** Cheapest. State in the paper that ablation variants reuse the full model's
   hyperparameters and that this is conservative with respect to the ablation. Acceptable for an ablation
   table, weak if an ablation result becomes a headline number.
2. **Short HPO per variant**, reduced budget in the style of the iAIMS26 campaign, roughly 40 epochs and
   patience 5. Defensible and affordable.
3. **Re-tune only the parameters whose meaning changes.** For C that is `hidden`, `lr` and `K_hop`, since
   `K_hop` and `next_n_neighbors` no longer do anything. Middle ground.

Recommendation: option 1 for the first pass, so the direction of the effect is known quickly, and option 2
afterwards for whichever variant turns out to carry a load-bearing number.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Kriging lag channel silently enabled in some config or in the HPO path | **High.** Would invalidate B without any visible symptom | Step 3 assertion plus the startup log line |
| Zero-size tensors break something outside the DiffConv, for example in batching or `_subgraph_station_edges` | Medium | Verification steps 3 and 4; fallback to `station_connectivity: "self"` |
| A second information path between stations that I have not found, so C is not truly graph-free | Medium | Verification step 5 is designed exactly for this |
| Inherited hyperparameters make the ablation look worse than it is | Medium | §6 |
| Non-unique per-station index when pairing the statistics | Medium | Explicit `index.is_unique` assertion, per the iAIMS26 lesson |
| B collapses because the `type_ind` and target-mask semantics interact unexpectedly when no station has data | Low | Verification step 2 checks the mask is unchanged |

---

## 8. How the HPO study is resolved, and what that implies

`train_dcrnn.py:315-333` derives the study name from the config file name:

```
config_stem = Path(config).stem.replace("config_", "")
hpo_stem    = re.sub(r'_fold\d+$', '', config_stem)
study_name  = f"cl_m-dcrnn_out-{H_fore}_freq-{freq}_{hpo_stem}"
```

Two consequences that shape the workflow:

1. **The naming is automatic.** `config_wind_dcrnn_nomeas.yaml` and its three fold files all resolve to
   `cl_m-dcrnn_out-48_freq-1h_wind_dcrnn_nomeas`. So each variant needs **one HPO run on its base config,
   then three fold trainings** that inherit those parameters. No plumbing required.
2. **`dcrnn_cfg.update(hpo_best_params)` runs after the config is loaded**, so every key inside the search
   space is overridden by Optuna and the static value in the YAML is decoration. Keys outside the search
   space do come from the file. `station_connectivity`, `nwp_nodes`, `hist_wind_available` and the new
   `neighbour_meas_available` are all outside the search space, so the ablation flags survive the update.
   This was checked, not assumed.

Order of operations per variant: `hpo_dcrnn.py` on the base config, then `launch_train_pipeline.py` for the
three folds, then `get_test_results_dcrnn.py` in both `excl_val` and `incl_val`.

---

## 9. Two findings from generating the configs

### 9.1 The fold configs are out of sync with the base config

`config_wind_dcrnn_fold1..3.yaml` are structurally different from `config_wind_dcrnn.yaml`. They carry an
extra top-level `gnn:` section, their `dcrnn:` block starts at a different line, and they **omit**
`nwp_nodes`, `hist_wind_available` and `direction_to_adj` entirely, relying on the parser defaults, which
happen to be the values variant A wants. They also carry different static values for keys that are
**outside** the HPO search space, `batch_size`, `max_epochs`, `patience` and `min/max_target_stations` among
them, so those genuinely differ between the base run and the fold runs.

For the ablation this is harmless and in fact correct: each variant config was generated by copying its
corresponding source file, so variant B fold 1 matches variant A fold 1 in everything except the ablated
flag, which is exactly the comparability that matters. But it is worth knowing, because the fold configs do
not mirror the base config the way one would assume, and if the HPO is being redone anyway this is the
moment to regenerate them from a single source.

### 9.2 Variant C's search space contains three provably inert parameters

With no station edges, `K_hop`, `next_n_neighbors` and `direction_to_adj` cannot influence the model. The
permutation check in the verification suite proves this directly: with `station_connectivity: none` the
output at a node is bit-identical when every other node's features are permuted. Leaving them in the search
space wastes trials and would read poorly in an appendix that lists the searched ranges.

`next_n_neighbors` deserves special attention because it is not merely inert, it is **expensive**. It
controls how many neighbour station nodes enter the subgraph, and in variant C those nodes consume memory
and compute while contributing nothing. Pinning it to its minimum makes variant C substantially cheaper to
train than A or B with no effect whatsoever on the result, since each node is independent.

**The one decision left before variant C can be generated:** pin `K_hop`, `direction_to_adj` and
`next_n_neighbors` and remove them from C's HPO `params` block, with `next_n_neighbors` at its minimum. My
recommendation is yes to all three. What stays in C's search space: `next_n_icond2`, `next_n_ecmwf`,
`nwp_heads`, `nwp_out_per_head`, `hidden`, `num_layers`, `dropout`, `lr`, `weight_decay`, `grad_accum`,
`horizon_decay`, `teacher_forcing_ratio`, `gradient_clip`. The NWP attention is untouched by this ablation,
so all of its parameters remain meaningful.

### 9.3 Still open from the previous discussion, independent of this plan

Whether variant BASE stays in the paper or is replaced by the fixed-inverse-distance ablation. That decision
does not block B or C, but it changes how many HPO studies the campaign needs, and the free check that
informs it (the `next_n_ecmwf` marginal over all completed trials, lower bound 0 verified in both search
spaces) has not been run yet.

---

## 10. Reproduction commands

```
# revert everything
python3 /tmp/patch_ablations.py --root ~/Work/forecasting_framework --revert

# re-verify (no data, no GPU, ~2 s)
cd ~/Work/forecasting_framework && ./frcst/bin/python /tmp/verify_ablations.py

# generate variant C configs once §9.2 is decided
python3 /tmp/gen_variant_configs.py --dir configs/dcrnn --variant nograph --dry-run
```

Scripts live in `/tmp` on `l1` and in the session scratchpad locally. They should be moved into the
repository before anything is committed.
