#!/usr/bin/env python3
"""Configs of the FL scenario study on the curtailed parks (FL_Contribution/reports/fl_scenarios_v1.md).

Base: configs/parks_v1/config_parks_v1_curt_v11_{fl_fedgradient,fl_fedavg,cl80}_static[_nolag].yaml
(x1 = parks_v1_curt_v11, capacity-weighted park statics). Written to configs/parks_v1/scenarios/,
results to results/fl_scenarios/. FL configs run with fine-tuning and evaluate the clients with
the global and the fine-tuned model (fl.fine_tune_eval_global); the local baseline is
train_local.py on the FedGradient config.

  A  scarce   training history 1/3/6/12 months before 2024-08-01, seeds 42/43/44; epochs/rounds,
              patience and fine-tune epochs scaled by 12/months (same number of optimizer steps)
  B  lopo     10 folds: park k of every client is a new park (never trained on), evaluated by
              its client's local model (fl.client_holdout), the global FL model and CL72
  C  mask     x1 and x4, no power lag; data.target_mask all | market_env (direct-marketer view:
              grid curtailment unknown) | none (x4 reference; x1 reference = *_static_nolag)

Part 2 (--part 2, configs/parks_v1/scenarios2/, after the fine-tuning fix of 2026-10-10): FL runs
evaluate three fine-tune variants of the global model (ft_full, ft_gentle: lr x 0.1 and few epochs,
ft_head: only the output head trainable).
  lopo2   new parks, seeds 43/44, folds 0-4
  warm    the new park is trained with only 2 weeks of own history (data.station_history_start
          2024-07-18); folds 0-9 seed 42, folds 0-4 seeds 43/44
  mask2   x4, no lag: none / market_env (seeds 43, 44) and market_env_grid50 (half of the grid
          events known to the direct marketer; seeds 42-44)
  scarce2 1 and 3 months, seeds 42-44, FL only (local/central of part 1 are the references)

Hub wind (--part wind, configs/parks_v1/scenarios_wind/): target = capacity-weighted park hub wind
speed (data.target_kind wind_speed_hub, unaffected by curtailment) instead of power; base comparison
FedGradient / FedAvg (with fine-tune variants) / CL80 / local, with and without the target lag
(static / static_nolag), seeds 42-44.

ICON only (--part noecmwf, configs/parks_v1/scenarios_noecmwf/): base comparison without the ECMWF features
(nwp_models [icon-d2], known features = ICON-D2 h78/h127/h184 only), statics, with lag, seeds 42-44.

Pool size (--part poolsize, configs/parks_v1/scenarios_poolsize/): models trained on k parks, run with
train_local.py (method local) on a redefined fl.clients: k = 1 / 2 / 5 split every client (and the 10 holdout
parks) into groups of k parks (k = 1: one model per park, 90 models), k = 20 / 40 merge clients of the same kind
(R0+R1, R2+R3, N0+N1, N2+N3 / R0-R3, N0-N3); seeds 42-44. k = 10 (local) and 80 (central) are the 12-month runs
of part 1, the holdout operator's own model (k = 10) is scenarios_holdout.

Split 2 (--part split2, configs/parks_v1/split2/): base configs (x1 statics, with lag) for HPO and final
test on the later data: training data 2023-07-24..2025-07-31, test 2025-08..2026-07; HPO with three
expanding folds validating Aug-Nov 2024, Dec 2024-Mar 2025, Apr-Jul 2025 (hpo.fold_boundaries).

Forecast runs (--part runs, configs/parks_v1/scenarios_runs/): local models (train_local, method local) on
split 1 with the ICON-D2 runs 06/09/12/15 loaded (data.forecast_hours) and trained on one run or on all
(data.train_forecast_hours); the test split keeps all four runs, so every model is scored on every run.
With ECMWF (06/09 -> 00 UTC, 12/15 -> 12 UTC run, assumed available) and ICON-D2 only; strict_split;
seeds 42-44.

Features (--part features, configs/parks_v1/scenarios_features/): split 1, 09 UTC run, strict_split, seeds 42-44,
eval.vsn_weights (mean TFT selection weights in the result). base = x1 statics, ICON-D2 h78/h127/h184 + ECMWF;
h247 = + ICON-D2 layer 213-281 m (wind_speed_h247); coast = + static dist_coast_km (park centre, Natural Earth
coastline, scripts/parks_v1_dist_coast.py). base/h247/coast with the power lag for local and CL80; base/h247 without
the lag for local.

  python scripts/make_fl_scenarios.py              # part 1: configs and scenarios/manifest.csv
  python scripts/make_fl_scenarios.py --part 2     # part 2: scenarios2/
Holdout parks (--part holdout, configs/parks_v1/scenarios_holdout/): the 10 holdout parks scored by every
client's local model (fl.client_holdout = the 10 parks for each client; run with method local) and by the own
model of the holdout operator (CL on the 10 parks, run with method cl80); seeds 42-44. The global models of the
same setup are the 12-month runs of part 1 (scen_scarce_m12_*).

Drift (--part drift, configs/parks_v1/scenarios_drift/, fl.drift_metrics on, x1 statics with lag, seeds 42-44):
  clients  k = 10 / 5 / 2 / 1 parks per client (8 / 16 / 40 / 80 clients, consecutive split of every client's
           list as in poolsize; local references = poolsize k < 10, part-1 scen_scarce_m12 for k = 10), FedGradient
           and FedAvg, rounds/patience scaled by 10/k (same number of optimizer steps); FedAvg also with
           5 local epochs per round (local_epochs 5) for k = 10 and k = 1
  skew     one partition with 25/15/12/10/8/5/3/2 parks per client (seeded random assignment of the 80 parks,
           clients S0_25 .. S7_2), FedGradient / FedAvg; local = train_local on the FedGradient config
No lag (--part nolag, configs/parks_v1/scenarios_nolag/): the pure NWP-driven setting (observed_features [],
static_nolag bases) for the scenarios where federation mattered with the lag:
  scarce_nolag  1 and 3 months of history, seeds 42-44, FL with fine-tune variants / CL80 / local
  lopo_nolag    new parks (folds 0-4, seed 42), FL with fine-tune variants / CL72 / local

  python scripts/make_fl_scenarios.py              # part 1: configs and scenarios/manifest.csv
  python scripts/make_fl_scenarios.py --part 2     # part 2: scenarios2/
  python scripts/make_fl_scenarios.py --part wind  # hub wind target: scenarios_wind/
  python scripts/make_fl_scenarios.py --part holdout  # holdout parks: scenarios_holdout/
  python scripts/make_fl_scenarios.py --part noecmwf  # ICON-D2 only: scenarios_noecmwf/
  python scripts/make_fl_scenarios.py --part poolsize  # parks per model: scenarios_poolsize/
  python scripts/make_fl_scenarios.py --part split2    # split 2 base configs: split2/
  python scripts/make_fl_scenarios.py --part runs      # forecast runs: scenarios_runs/
  python scripts/make_fl_scenarios.py --part features  # feature test: scenarios_features/
  python scripts/make_fl_scenarios.py --part drift     # client count, skew, drift metrics: scenarios_drift/
  python scripts/make_fl_scenarios.py --part nolag     # scarce / new parks without the power lag: scenarios_nolag/
"""

import copy
import os
import sys

import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(REPO, 'configs', 'parks_v1')
OUT = os.path.join(BASE, 'scenarios')
HEAD = ['positionwise_grn', 'output_gate', 'output_ln', 'output_layer']
WARM_START = '2024-07-18'
RESULTS = 'results/fl_scenarios'
DATASETS = {'x1': 'parks_v1_curt_v11', 'x4': 'parks_v1_curt_v11_x4'}
MONTHS = {1: '2024-07-01', 3: '2024-05-01', 6: '2024-02-01', 12: '2023-07-24'}
SEEDS = (42, 43, 44)


class _Loader(yaml.SafeLoader):
    """Keeps !ENV tags verbatim (configs are copied, not resolved)."""


class _Env(str):
    pass


_Loader.add_constructor('!ENV', lambda loader, node: _Env(loader.construct_scalar(node)))
yaml.SafeDumper.add_representer(_Env, lambda d, v: d.represent_scalar('!ENV', str(v)))


def load(name: str) -> dict:
    with open(os.path.join(BASE, name)) as f:
        return yaml.load(f, Loader=_Loader)


def write(cfg: dict, name: str, header: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    cfg['eval']['results_path'] = RESULTS
    path = os.path.join(OUT, f'config_{name}.yaml')
    with open(path, 'w') as f:
        f.write(f'# {header}\n# generated by scripts/make_fl_scenarios.py - do not edit\n')
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, width=120)
    return os.path.relpath(path, REPO)[:-5]


def dataset(cfg: dict, ds: str) -> dict:
    cfg['data']['path'] = _Env(f"${{DATA_ROOT}}/synthetic/wind/{DATASETS[ds]}")
    return cfg


def fine_tune(cfg: dict, factor: int = 1, variants: bool = False) -> dict:
    cfg['fl'].update(fine_tune=True, fine_tune_eval_global=True, fine_tune_epochs=50 * factor)
    if variants:
        cfg['fl']['fine_tune_variants'] = [{'name': 'ft_full'},
                                           {'name': 'ft_gentle', 'epochs': 5 * factor, 'lr_factor': 0.1},
                                           {'name': 'ft_head', 'trainable': HEAD}]
    return cfg


def scale(cfg: dict, factor: int) -> dict:
    cfg['model']['epochs'] = 100 * factor
    cfg['model']['early_stopping']['patience'] = 10 * factor
    if cfg['model'].get('fl'):
        cfg['fl']['n_rounds'] = 100 * factor
        cfg['fl']['global_early_stopping']['patience'] = 10 * factor
    return cfg


def bases(suffix: str = 'static') -> dict:
    return {m: load(f'config_parks_v1_curt_v11_{m}_{suffix}.yaml') for m in ('fl_fedgradient', 'fl_fedavg', 'cl80')}


def scarce(rows: list) -> None:
    b = bases('static')
    for m, start in MONTHS.items():
        f = 12 // m
        for seed in SEEDS:
            for meth, cfg0 in b.items():
                cfg = scale(copy.deepcopy(cfg0), f)
                cfg['data']['train_start'] = start
                cfg['params']['random_seed'] = seed
                if meth.startswith('fl_'):
                    fine_tune(cfg, f)
                name = f'scen_scarce_m{m}_s{seed}_{meth}'
                rows.append({'scenario': 'scarce', 'months': m, 'seed': seed, 'method': meth,
                             'config': write(cfg, name, f'A scarce: {m} months training history, seed {seed}, {meth}')})


def lopo(rows: list) -> None:
    b = bases('static')
    clients = b['fl_fedgradient']['fl']['clients']
    holdout = list(b['fl_fedgradient']['data']['val_files'])
    for k in range(10):
        new = {c: [p[k]] for c, p in clients.items()}
        keep = {c: [x for i, x in enumerate(p) if i != k] for c, p in clients.items()}
        files = [x for c in keep for x in keep[c]]
        newparks = [new[c][0] for c in clients]
        for meth, cfg0 in b.items():
            cfg = copy.deepcopy(cfg0)
            cfg['data']['files'] = files
            if meth.startswith('fl_'):
                cfg['fl']['clients'] = keep
                cfg['fl']['client_holdout'] = new
                cfg['data']['val_files'] = newparks + holdout
                fine_tune(cfg)
            else:
                cfg['data']['holdout_files'] = newparks + holdout
            name = f'scen_lopo_k{k}_{meth.replace("cl80", "cl72")}'
            rows.append({'scenario': 'lopo', 'fold': k, 'seed': 42, 'method': meth,
                         'config': write(cfg, name, f'B lopo fold {k}: park {k} of every client is new, {meth}')})


def mask(rows: list) -> None:
    b = bases('static_nolag')
    for ds in ('x1', 'x4'):
        for spec in ('all', 'market_env', None):
            if ds == 'x1' and spec is None:
                continue            # reference: config_parks_v1_curt_v11_*_static_nolag (already run)
            for meth, cfg0 in b.items():
                cfg = dataset(copy.deepcopy(cfg0), ds)
                if spec:
                    cfg['data']['target_mask'] = spec
                if meth.startswith('fl_'):
                    fine_tune(cfg)
                name = f'scen_mask_{ds}_{spec or "none"}_{meth}'
                rows.append({'scenario': 'mask', 'dataset': ds, 'mask': spec or 'none', 'seed': 42, 'method': meth,
                             'config': write(cfg, name, f'C mask {spec or "none"} on {DATASETS[ds]} (no power lag), {meth}')})


def part2(rows: list) -> None:
    global OUT
    OUT = os.path.join(BASE, 'scenarios2')
    b = bases('static')
    clients = b['fl_fedgradient']['fl']['clients']
    holdout = list(b['fl_fedgradient']['data']['val_files'])
    runs = [(k, s) for s in (43, 44) for k in range(5)]
    for k, seed in runs:                                            # lopo2: cold start with seeds
        new = {c: [p[k]] for c, p in clients.items()}
        keep = {c: [x for i, x in enumerate(p) if i != k] for c, p in clients.items()}
        files = [x for c in keep for x in keep[c]]
        newparks = [new[c][0] for c in clients]
        for meth, cfg0 in b.items():
            cfg = copy.deepcopy(cfg0)
            cfg['params']['random_seed'] = seed
            cfg['data']['files'] = files
            if meth.startswith('fl_'):
                cfg['fl']['clients'] = keep
                cfg['fl']['client_holdout'] = new
                cfg['data']['val_files'] = newparks + holdout
                fine_tune(cfg, variants=True)
            else:
                cfg['data']['holdout_files'] = newparks + holdout
            name = f'scen2_lopo_k{k}_s{seed}_{meth.replace("cl80", "cl72")}'
            rows.append({'scenario': 'lopo', 'fold': k, 'seed': seed, 'method': meth,
                         'config': write(cfg, name, f'B lopo fold {k}, seed {seed}, {meth}')})
    for k, seed in [(k, 42) for k in range(10)] + runs:             # warm: 2 weeks own history
        newparks = [p[k] for p in clients.values()]
        for meth, cfg0 in b.items():
            cfg = copy.deepcopy(cfg0)
            cfg['params']['random_seed'] = seed
            cfg['data']['station_history_start'] = {p: WARM_START for p in newparks}
            if meth.startswith('fl_'):
                fine_tune(cfg, variants=True)
            name = f'scen2_warm_k{k}_s{seed}_{meth}'
            rows.append({'scenario': 'warm', 'fold': k, 'seed': seed, 'method': meth,
                         'config': write(cfg, name, f'B warm fold {k}: new parks with 2 weeks history, seed {seed}, {meth}')})
    bn = bases('static_nolag')
    for spec, seeds in (('none', (43, 44)), ('market_env', (43, 44)), ('market_env_grid50', (42, 43, 44))):
        for seed in seeds:
            for meth, cfg0 in bn.items():
                cfg = dataset(copy.deepcopy(cfg0), 'x4')
                cfg['params']['random_seed'] = seed
                if spec != 'none':
                    cfg['data']['target_mask'] = spec
                if meth.startswith('fl_'):
                    fine_tune(cfg, variants=True)
                name = f'scen2_mask_x4_{spec}_s{seed}_{meth}'
                rows.append({'scenario': 'mask', 'dataset': 'x4', 'mask': spec, 'seed': seed, 'method': meth,
                             'config': write(cfg, name, f'C mask {spec} on x4 (no lag), seed {seed}, {meth}')})
    for m in (1, 3):
        f = 12 // m
        for seed in SEEDS:
            for meth in ('fl_fedgradient', 'fl_fedavg'):
                cfg = scale(copy.deepcopy(b[meth]), f)
                cfg['data']['train_start'] = MONTHS[m]
                cfg['params']['random_seed'] = seed
                fine_tune(cfg, f, variants=True)
                name = f'scen2_scarce_m{m}_s{seed}_{meth}'
                rows.append({'scenario': 'scarce', 'months': m, 'seed': seed, 'method': meth,
                             'config': write(cfg, name, f'A scarce {m} months, seed {seed}, {meth} (fine-tune variants)')})


def hub_wind(rows: list) -> None:
    global OUT
    OUT = os.path.join(BASE, 'scenarios_wind')
    for lag, suffix in ((True, 'static'), (False, 'static_nolag')):
        for seed in SEEDS:
            for meth, cfg0 in bases(suffix).items():
                cfg = copy.deepcopy(cfg0)
                cfg['data']['target_kind'] = 'wind_speed_hub'
                cfg['params']['random_seed'] = seed
                if meth.startswith('fl_'):
                    fine_tune(cfg, variants=True)
                name = f'scen_wind_{"lag" if lag else "nolag"}_s{seed}_{meth}'
                rows.append({'scenario': 'wind', 'lag': lag, 'seed': seed, 'method': meth,
                             'config': write(cfg, name, f'hub wind target, {"with" if lag else "without"} '
                                                        f'target lag, seed {seed}, {meth}')})


def holdout_parks(rows: list) -> None:
    global OUT
    OUT = os.path.join(BASE, 'scenarios_holdout')
    b = bases('static')
    hold = [str(p) for p in b['fl_fedgradient']['data']['val_files']]
    for seed in SEEDS:
        cfg = copy.deepcopy(b['fl_fedgradient'])                    # local models of the 8 clients
        cfg['params']['random_seed'] = seed
        cfg['fl']['client_holdout'] = {c: list(hold) for c in cfg['fl']['clients']}
        rows.append({'scenario': 'holdout', 'model': 'foreign_local', 'seed': seed, 'method': 'fl_fedgradient',
                     'config': write(cfg, f'scen_hold_foreign_s{seed}_fl_fedgradient',
                                     f'holdout parks scored by every client local model, seed {seed} (run: local)')})
        cfg = copy.deepcopy(b['cl80'])                              # own model of the holdout operator
        cfg['params']['random_seed'] = seed
        cfg['data']['files'] = list(hold)
        cfg['data'].pop('holdout_files', None)
        cfg['data'].pop('val_files', None)
        rows.append({'scenario': 'holdout', 'model': 'own', 'seed': seed, 'method': 'cl80',
                     'config': write(cfg, f'scen_hold_own_s{seed}_cl10',
                                     f'own model of the holdout operator (CL on its 10 parks), seed {seed}')})


def no_ecmwf(rows: list) -> None:
    global OUT
    OUT = os.path.join(BASE, 'scenarios_noecmwf')
    for seed in SEEDS:
        for meth, cfg0 in bases('static').items():
            cfg = copy.deepcopy(cfg0)
            cfg['params']['nwp_models'] = ['icon-d2']
            cfg['params']['known_features'] = [f for f in cfg['params']['known_features'] if not f.startswith('ecmwf_')]
            cfg['params'].pop('ecmwf_features', None)
            cfg['params']['random_seed'] = seed
            if meth.startswith('fl_'):
                fine_tune(cfg, variants=True)
            rows.append({'scenario': 'noecmwf', 'seed': seed, 'method': meth,
                         'config': write(cfg, f'scen_noecmwf_s{seed}_{meth}',
                                         f'ICON-D2 only (no ECMWF features), seed {seed}, {meth}')})


POOL_MERGE = {20: [('R0', 'R1'), ('R2', 'R3'), ('N0', 'N1'), ('N2', 'N3')],
              40: [('R0', 'R1', 'R2', 'R3'), ('N0', 'N1', 'N2', 'N3')]}


def pool_clients(clients: dict, holdout: list, k: int) -> dict:
    """fl.clients with k parks per model: split (k < 10, holdout parks included) or merge (k > 10)."""
    if k in POOL_MERGE:
        return {'_'.join(g): [p for c in g for p in clients[c]] for g in POOL_MERGE[k]}
    if 10 % k:
        raise ValueError(f'k must divide 10, got {k}')
    out = {}
    for cid, parks in list(clients.items()) + [('H', holdout)]:
        for i in range(0, len(parks), k):
            out[f'{cid}_g{i // k}'] = list(parks[i:i + k])
    return out


def pool_size(rows: list) -> None:
    global OUT
    OUT = os.path.join(BASE, 'scenarios_poolsize')
    b = bases('static')['fl_fedgradient']
    hold = [str(p) for p in b['data']['val_files']]
    for k in (1, 2, 5, 20, 40):
        for seed in SEEDS:
            cfg = copy.deepcopy(b)
            cfg['params']['random_seed'] = seed
            cfg['fl']['clients'] = pool_clients(b['fl']['clients'], hold, k)
            rows.append({'scenario': 'poolsize', 'k': k, 'seed': seed, 'method': 'fl_fedgradient',
                         'config': write(cfg, f'scen_pool_k{k}_s{seed}_fl_fedgradient',
                                         f'{k} parks per model (train_local), seed {seed}')})


SPLIT2 = {'train_start': '2023-07-24', 'train_end': '2025-07-31 23:00', 'test_start': '2025-08-01',
          'test_end': '2026-07-31', 'data_cutoff': '2026-08-03'}
SPLIT2_FOLDS = ['2024-08-01', '2024-12-01', '2025-04-01', '2025-08-01']


def split2(rows: list) -> None:
    global OUT
    OUT = os.path.join(BASE, 'split2')
    for meth, cfg in bases('static').items():
        cfg['data'].update(SPLIT2, strict_split=True)        # no training target in the test period
        cfg['hpo'].update(fold_boundaries=list(SPLIT2_FOLDS), objective_reduction='best', kfolds=3,
                          weight_decay=[1e-6, 1e-3])                      # float, log
        # MedianPruner after fold 2 (step 1 of 3): compares the same fold across trials (Viktor 2026-10-13)
        cfg['hpo']['pruning'] = dict(use_median_pruner=True, n_startup_trials=10, n_warmup_steps=1,
                                     interval_steps=1, disable_pruning=False)
        cfg['hpo']['tft'].update(n_lstm_layers=[1, 2], static_embedding_dim=[8, 64],   # int
                                 clipnorm=[1.0, 10.0])                    # float, linear
        if meth != 'cl80':    # FL-HPO (hpo_fl.py, date folds): FedGradient server step; FedAvg keeps 1 local epoch
            cfg['hpo']['fl']['strategy'] = cfg['fl']['strategy']
            cfg['hpo']['fl'].setdefault('fedgradient', {}).update(server_optimizer=['adam', 'sgd_momentum'],
                                                                  server_lr=[1e-4, 1e-1],   # log
                                                                  clipnorm=[0.3, 10.0])     # log, client clipping
            cfg['hpo']['fl'].pop('n_local_epochs', None)   # Viktor 2026-10-13: local epochs stay at 1
            # Viktor 2026-10-14: PercentilePruner(75) instead of the median - the median of completed trials
            # tightened with every prune (10 of 12 FedAvg trials pruned at < 1 % distance), and pruned
            # trials do not count toward the budget while still costing 2 of 3 folds
            cfg['hpo']['pruning'].update(percentile=75.0, use_median_pruner=False)
        cfg['eval']['results_path'] = 'results/parks_v1_split2'
        name = f'parks_v1_curt_v11_{meth}_static_s2'
        path = write(cfg, name, f'split 2 (train data 2023-07-24..2025-07, test 2025-08..2026-07, HPO folds '
                                f'{SPLIT2_FOLDS}), x1 statics, {meth}')
        rows.append({'scenario': 'split2', 'method': meth, 'config': path})
    # write() sets eval.results_path to the scenario folder; keep the split-2 folder
    for r in rows:
        p = os.path.join(REPO, r['config'] + '.yaml')
        txt = open(p).read().replace(f'results_path: {RESULTS}', 'results_path: results/parks_v1_split2')
        open(p, 'w').write(txt)


RUNS = ['06', '09', '12', '15']


def forecast_runs(rows: list) -> None:
    global OUT
    OUT = os.path.join(BASE, 'scenarios_runs')
    b = bases('static')['fl_fedgradient']
    for feat in ('ecmwf', 'icon'):
        for train in RUNS + ['all']:
            for seed in SEEDS:
                cfg = copy.deepcopy(b)
                cfg['params']['random_seed'] = seed
                cfg['data']['forecast_hours'] = list(RUNS)
                cfg['data']['train_forecast_hours'] = list(RUNS) if train == 'all' else [train]
                cfg['data']['strict_split'] = True
                if feat == 'icon':
                    cfg['params']['nwp_models'] = ['icon-d2']
                    cfg['params']['known_features'] = [f for f in cfg['params']['known_features']
                                                       if not f.startswith('ecmwf_')]
                    cfg['params'].pop('ecmwf_features', None)
                rows.append({'scenario': 'runs', 'features': feat, 'train_runs': train, 'seed': seed,
                             'method': 'fl_fedgradient',
                             'config': write(cfg, f'scen_runs_{feat}_tr{train}_s{seed}_fl_fedgradient',
                                             f'local models, NWP {feat}, trained on run(s) {train}, all runs '
                                             f'tested, seed {seed}')})


FEATURE_VARIANTS = {'base': ([], []), 'h247': (['wind_speed_h247'], []), 'coast': ([], ['dist_coast_km'])}


def features(rows: list) -> None:
    global OUT
    OUT = os.path.join(BASE, 'scenarios_features')
    for lag, suffix in ((True, 'static'), (False, 'static_nolag')):
        b = bases(suffix)
        for var, (known, static) in FEATURE_VARIANTS.items():
            for meth in ('fl_fedgradient', 'cl80'):                  # fl_fedgradient config -> train_local
                if not lag and (var == 'coast' or meth == 'cl80'):
                    continue
                for seed in SEEDS:
                    cfg = copy.deepcopy(b[meth])
                    cfg['params']['random_seed'] = seed
                    icon = [f for f in cfg['params']['known_features'] if not f.startswith('ecmwf_')]
                    ecmwf = [f for f in cfg['params']['known_features'] if f.startswith('ecmwf_')]
                    cfg['params']['known_features'] = icon + known + ecmwf
                    cfg['params']['static_features'] = cfg['params']['static_features'] + static
                    cfg['data']['strict_split'] = True
                    cfg['eval']['vsn_weights'] = True
                    tag = f'{var}_{"lag" if lag else "nolag"}'
                    rows.append({'scenario': 'features', 'variant': var, 'lag': lag, 'seed': seed,
                                 'method': meth,
                                 'config': write(cfg, f'scen_feat_{tag}_s{seed}_{meth}',
                                                 f'feature test {var}, {"with" if lag else "without"} power lag, '
                                                 f'seed {seed}, {meth}')})


DRIFT_K = (10, 5, 2, 1)
SKEW_SIZES = (25, 15, 12, 10, 8, 5, 3, 2)


def fl_pool_clients(clients: dict, k: int) -> dict:
    """fl.clients with k parks per client: consecutive split of every client's list (poolsize split
    without the holdout groups, so the local poolsize models are the references)."""
    if k == 10:
        return {c: list(p) for c, p in clients.items()}
    if 10 % k:
        raise ValueError(f'k must divide 10, got {k}')
    return {f'{cid}_g{i // k}': list(parks[i:i + k])
            for cid, parks in clients.items() for i in range(0, len(parks), k)}


def skew_clients(clients: dict, sizes=SKEW_SIZES, seed: int = 0) -> dict:
    """Quantity-skewed partition of the 80 parks (sizes sum to 80), seeded random assignment."""
    import numpy as np
    parks = [p for c in clients for p in clients[c]]
    if sum(sizes) != len(parks):
        raise ValueError(f'sizes sum to {sum(sizes)}, {len(parks)} parks')
    order = np.random.default_rng(seed).permutation(len(parks))
    out, pos = {}, 0
    for i, n in enumerate(sizes):
        out[f'S{i}_{n}'] = [parks[j] for j in order[pos:pos + n]]
        pos += n
    return out


def drift(rows: list) -> None:
    global OUT
    OUT = os.path.join(BASE, 'scenarios_drift')
    b = bases('static')
    clients = b['fl_fedgradient']['fl']['clients']
    for k in DRIFT_K:
        f = 10 // k
        for seed in SEEDS:
            for meth, local_epochs in (('fl_fedgradient', 1), ('fl_fedavg', 1), ('fl_fedavg', 5)):
                if local_epochs > 1 and k not in (10, 1):
                    continue
                cfg = scale(copy.deepcopy(b[meth]), f)
                cfg['params']['random_seed'] = seed
                cfg['fl']['clients'] = fl_pool_clients(clients, k)
                cfg['fl']['drift_metrics'] = True
                cfg['fl']['n_local_epochs'] = local_epochs
                cfg['fl']['max_concurrent_actors'] = max(8, len(cfg['fl']['clients']))
                tag = f'k{k}_e{local_epochs}' if local_epochs > 1 else f'k{k}'
                rows.append({'scenario': 'clients', 'k': k, 'n_clients': len(cfg['fl']['clients']),
                             'local_epochs': local_epochs, 'seed': seed, 'method': meth,
                             'config': write(cfg, f'scen_drift_{tag}_s{seed}_{meth}',
                                             f'{k} parks per client ({len(cfg["fl"]["clients"])} clients), '
                                             f'{local_epochs} local epoch(s), drift metrics, seed {seed}, {meth}')})
    for seed in SEEDS:
        for meth in ('fl_fedgradient', 'fl_fedavg'):
            cfg = copy.deepcopy(b[meth])
            cfg['params']['random_seed'] = seed
            cfg['fl']['clients'] = skew_clients(clients)
            cfg['fl']['drift_metrics'] = True
            rows.append({'scenario': 'skew', 'k': None, 'n_clients': 8, 'local_epochs': 1, 'seed': seed,
                         'method': meth,
                         'config': write(cfg, f'scen_drift_skew_s{seed}_{meth}',
                                         f'quantity skew {SKEW_SIZES} parks per client, drift metrics, '
                                         f'seed {seed}, {meth} (local: train_local on the FedGradient config)')})


def no_lag(rows: list) -> None:
    global OUT
    OUT = os.path.join(BASE, 'scenarios_nolag')
    b = bases('static_nolag')
    for m in (1, 3):
        f = 12 // m
        for seed in SEEDS:
            for meth, cfg0 in b.items():
                cfg = scale(copy.deepcopy(cfg0), f)
                cfg['data']['train_start'] = MONTHS[m]
                cfg['params']['random_seed'] = seed
                if meth.startswith('fl_'):
                    fine_tune(cfg, f, variants=True)
                rows.append({'scenario': 'scarce_nolag', 'months': m, 'seed': seed, 'method': meth,
                             'config': write(cfg, f'scen_nolag_scarce_m{m}_s{seed}_{meth}',
                                             f'{m} months history, no power lag, seed {seed}, {meth}')})
    clients = b['fl_fedgradient']['fl']['clients']
    holdout = list(b['fl_fedgradient']['data']['val_files'])
    for k in range(5):
        new = {c: [p[k]] for c, p in clients.items()}
        keep = {c: [x for i, x in enumerate(p) if i != k] for c, p in clients.items()}
        files = [x for c in keep for x in keep[c]]
        newparks = [new[c][0] for c in clients]
        for meth, cfg0 in b.items():
            cfg = copy.deepcopy(cfg0)
            cfg['data']['files'] = files
            if meth.startswith('fl_'):
                cfg['fl']['clients'] = keep
                cfg['fl']['client_holdout'] = new
                cfg['data']['val_files'] = newparks + holdout
                fine_tune(cfg, variants=True)
            else:
                cfg['data']['holdout_files'] = newparks + holdout
            rows.append({'scenario': 'lopo_nolag', 'fold': k, 'seed': 42, 'method': meth,
                         'config': write(cfg, f'scen_nolag_lopo_k{k}_{meth.replace("cl80", "cl72")}',
                                         f'lopo fold {k}: park {k} of every client is new, no power lag, {meth}')})


TRIANEL_PARKS = ['eisleben', 'trendelburg', 'mattenberg', 'twv_windpark', 'eggstedt', 'lintern', 'kengen',
                 'barlt_west', 'jettebruch', 'rheurdt', 'jessin_grimmen', 'hoeheinoed']   # vogelsberg: 51 % gaps, no ECMWF
TRIANEL_ROOT = '${DATA_ROOT}/../nvme2'          # l1 nvme2 (l2: /mnt/lambda1/nvme2, l1: /mnt/nvme2)


def trianel_base(meth: str, lag: bool) -> dict:
    """parks_v1 model/fl/hpo sections with the Trianel data section (real SCADA, 12 parks, 09 UTC)."""
    cfg = bases('static' if lag else 'static_nolag')[meth]
    cfg['data'] = {
        'use_case': 'trianel',
        'path': _Env(f'{TRIANEL_ROOT}/synthetic/trianel_parks'),
        'real_data_path': _Env(f'{TRIANEL_ROOT}/trianel_hourly'),
        'real_synthetic_data_path': _Env(f'{TRIANEL_ROOT}/synthetic/trianel_parks'),
        'wind_parameter_file': _Env(f'{TRIANEL_ROOT}/synthetic/trianel_parks/wind_parameter.csv'),
        'turbine_parameter_file': _Env(f'{TRIANEL_ROOT}/synthetic/trianel_parks/turbine_parameter.csv'),
        'nwp_path': _Env(f'{TRIANEL_ROOT}/icon-d2/trianel_parks'),
        'ecmwf_path': _Env(f'{TRIANEL_ROOT}/ecmwf/parquet/SL'),
        'ecmwf_features': ['wind_speed_h100', 'wind_speed_h200'],
        'files': list(TRIANEL_PARKS), 'icond2_source': 'csv', 'forecast_hours': ['09'],
        'freq': '1h', 'target_col': 'power', 'train_frac': 1, 'val_frac': 0.0,
        'train_start': '2023-07-24', 'train_end': '2024-02-29 23:00', 'test_start': '2024-03-01',
        'test_end': '2024-05-31', 'data_cutoff': '2024-06-01', 'strict_split': True,
        'retrain_interval': 1, 'eval_interval': 1, 'scale_x': True, 'scale_y': False, 'use_pca': False,
        'pca_components': 3, 'lag_in_col': False, 'n_neighbors': 5}
    cfg['params'].update(known_features=['wind_speed_h78', 'wind_speed_h127', 'wind_speed_h184',
                                         'wind_speed_h100', 'wind_speed_h200'],
                         static_features=['altitude', 'park_age', 'hub_height', 'rotor_diameter', 'cut_in',
                                          'cut_out', 'rated_wind_speed'],
                         nwp_models=['icon-d2'])
    cfg['params'].pop('ecmwf_features', None)
    if meth.startswith('fl_'):
        cfg['fl']['clients'] = {p: [p] for p in TRIANEL_PARKS}
        cfg['fl']['drift_metrics'] = True
        cfg['fl']['max_concurrent_actors'] = len(TRIANEL_PARKS)
    return cfg


def trianel(rows: list) -> None:
    """Real parks: base comparison (lag / no lag, seeds) and new parks (leave-one-park-out, seed 42)."""
    global OUT
    OUT = os.path.join(BASE, '..', 'trianel')
    for lag in (True, False):
        for seed in SEEDS:
            for meth in ('fl_fedgradient', 'fl_fedavg', 'cl80'):
                cfg = scale(trianel_base(meth, lag), 2 if meth == 'cl80' else 10)   # ~220 samples per park
                cfg['params']['random_seed'] = seed
                if meth.startswith('fl_'):
                    fine_tune(cfg, 10, variants=True)
                tag = 'lag' if lag else 'nolag'
                rows.append({'scenario': 'trianel_base', 'lag': lag, 'seed': seed, 'method': meth,
                             'config': write(cfg, f'trianel_{tag}_s{seed}_{meth.replace("cl80", "cl12")}',
                                             f'Trianel real parks, {"with" if lag else "without"} power lag, '
                                             f'seed {seed}, {meth} (local: train_local on the FedGradient config)')})
    for seed in SEEDS:                        # park-id embedding: the pooled model knows which park it predicts
        for meth in ('fl_fedgradient', 'fl_fedavg', 'cl80'):
            cfg = scale(trianel_base(meth, True), 2 if meth == 'cl80' else 10)
            cfg['params']['random_seed'] = seed
            cfg['params']['static_features'] = cfg['params']['static_features'] + ['park_id']
            cfg['params']['static_categorical'] = ['park_id']
            if meth.startswith('fl_'):
                fine_tune(cfg, 10, variants=True)
            rows.append({'scenario': 'trianel_parkid', 'lag': True, 'seed': seed, 'method': meth,
                         'config': write(cfg, f'trianel_lag_parkid_s{seed}_{meth.replace("cl80", "cl12")}',
                                         f'Trianel real parks, with lag and park-id embedding, seed {seed}, {meth}')})
    for k, park in enumerate(TRIANEL_PARKS):
        keep = [p for p in TRIANEL_PARKS if p != park]
        for meth in ('fl_fedgradient', 'fl_fedavg', 'cl80'):
            cfg = scale(trianel_base(meth, True), 2 if meth == 'cl80' else 10)
            cfg['data']['files'] = keep
            if meth.startswith('fl_'):
                cfg['fl']['clients'] = {p: [p] for p in keep}
                cfg['fl']['max_concurrent_actors'] = len(keep)
                cfg['data']['val_files'] = [park]
            else:
                cfg['data']['holdout_files'] = [park]
            rows.append({'scenario': 'trianel_lopo', 'fold': k, 'seed': 42, 'method': meth,
                         'config': write(cfg, f'trianel_lopo_k{k}_{meth.replace("cl80", "cl11")}',
                                         f'Trianel new park {park} (never trained), with lag, {meth}')})
    # write() sets eval.results_path to the scenario folder; keep a Trianel folder
    for r in rows:
        p = os.path.join(REPO, r['config'] + '.yaml')
        txt = open(p).read().replace(f'results_path: {RESULTS}', 'results_path: results/trianel')
        open(p, 'w').write(txt)


TRIANEL_NEWCOMER_HISTORY = {'2w': '2024-02-15', '1m': '2024-02-01', '3m': '2023-12-01'}   # train_end 2024-02-29
TRIANEL_NEWCOMER_PARKS = ('barlt_west', 'jettebruch', 'eisleben', 'kengen', 'trendelburg', 'mattenberg')


def trianel_newcomer(rows: list) -> None:
    """Onboarding curve on the real parks: one park is a newcomer with 2 weeks / 1 month / 3 months of own history
    before the test period (data.station_history_start), the other 11 keep their 7 months. Parks spread over the
    local R² range (0.50 .. 0.76). local = train_local (the newcomer's own model on its short history), FedGradient
    global + fine-tuned, central CL12 with the same data; seeds 42-44 (seed 42 config names carry no seed tag,
    they were generated first), with lag. Same question as --part newcomer."""
    global OUT
    OUT = os.path.join(BASE, '..', 'trianel_newcomer')
    for seed in SEEDS:
        for park in TRIANEL_NEWCOMER_PARKS:
            for tag, start in TRIANEL_NEWCOMER_HISTORY.items():
                for meth in ('fl_fedgradient', 'cl80'):
                    cfg = scale(trianel_base(meth, True), 2 if meth == 'cl80' else 10)
                    cfg['params']['random_seed'] = seed
                    cfg['data']['station_history_start'] = {park: start}
                    if meth.startswith('fl_'):
                        fine_tune(cfg, 10, variants=True)
                    stag = '' if seed == 42 else f'_s{seed}'
                    rows.append({'scenario': 'trianel_newcomer', 'park': park, 'history': tag, 'seed': seed,
                                 'method': meth,
                                 'config': write(cfg, f'trianel_new_{park}_h{tag}{stag}_{meth.replace("cl80", "cl12")}',
                                                 f'Trianel newcomer {park} with {tag} own history, with lag, seed '
                                                 f'{seed}, {meth} (local: train_local on the FedGradient config)')})
    for r in rows:
        p = os.path.join(REPO, r['config'] + '.yaml')
        txt = open(p).read().replace(f'results_path: {RESULTS}', 'results_path: results/trianel')
        open(p, 'w').write(txt)


def newcomer_pool(rows: list) -> None:
    """Control for the onboarding advantage (--part newcomer): federation or just a pretrained pool? The newcomer
    (park 0 of client c, short history) trains with only its own operator's other 9 parks as the second client.
    FedGradient weights every step by batch samples and a client leaves the round after its own epoch, so the
    newcomer carries < 1 % of the aggregate: the global model is in effect the operator's 9-park model.
    zero-shot = that model on the newcomer (fine_tune_eval_global), few-shot = the FT variants on its own weeks.
    Compare with the same newcomer inside the 16-client federation and with its own short-history model.
    Histories 2w / 1m, seeds 42-44 in seed-major order (seed 42 runs first)."""
    global OUT
    OUT = os.path.join(BASE, 'scenarios_newcomer_pool')
    b = bases('static')
    clients = b['fl_fedgradient']['fl']['clients']
    allp = [p for ps in clients.values() for p in ps]
    holdout = list(b['fl_fedgradient']['data']['val_files'])
    for seed in SEEDS:
        for tag in ('2w', '1m'):
            for c, parks in clients.items():
                cfg = copy.deepcopy(b['fl_fedgradient'])
                cfg['params']['random_seed'] = seed
                cfg['data']['station_history_start'] = {parks[0]: NEWCOMER_HISTORY[tag]}
                # the 9-park model is also scored on the 70 foreign client parks (val_files: end-of-run only)
                cfg['data']['val_files'] = holdout + [p for p in allp if p not in parks]
                cfg['model']['epochs'] = 1000
                cfg['model']['early_stopping']['patience'] = 100
                cfg['fl']['clients'] = {c: list(parks[1:]), f'{c}_new': [parks[0]]}
                cfg['fl']['n_rounds'] = 200
                cfg['fl']['global_early_stopping']['patience'] = 20
                cfg['fl']['max_concurrent_actors'] = 2
                cfg['fl']['drift_metrics'] = True
                fine_tune(cfg, 10, variants=True)
                rows.append({'scenario': 'newcomer_pool', 'history': tag, 'seed': seed, 'client': c,
                             'newcomer': parks[0], 'method': 'fl_fedgradient',
                             'config': write(cfg, f'scen_newpool_h{tag}_s{seed}_{c}_fl_fedgradient',
                                             f'newcomer {parks[0]} ({tag} history) with only its operator {c} '
                                             f'(9 parks) as pool, seed {seed}, fl_fedgradient')})


TRANSFER_K = (10, 8, 6, 4, 2, 1)


def transfer_ladder(rows: list) -> None:
    """K2 transfer ladder (Viktor, 2026-10-12): every operator trains a central model on its first k parks
    (k = 10 .. 1) and that model is scored zero-shot on all other parks (data.holdout_files: the operator's own
    unused parks, the 70+ foreign client parks and the 10 holdout parks). Reference per unseen park: its own
    full-history one-park model (drift block k = 1) / the holdout operator's own 10-park model (T12).
    Seeds 42-44 in seed-major order."""
    global OUT
    OUT = os.path.join(BASE, 'scenarios_transfer')
    b = bases('static')
    clients = b['fl_fedgradient']['fl']['clients']
    allp = [p for ps in clients.values() for p in ps]
    holdout = list(b['cl80']['data']['holdout_files'])
    for seed in SEEDS:
        for k in TRANSFER_K:
            for c, parks in clients.items():
                cfg = copy.deepcopy(b['cl80'])
                cfg['params']['random_seed'] = seed
                cfg['data']['files'] = list(parks[:k])
                cfg['data']['holdout_files'] = [p for p in allp if p not in parks[:k]] + holdout
                rows.append({'scenario': 'transfer', 'k': k, 'client': c, 'seed': seed, 'method': 'cl80',
                             'config': write(cfg, f'scen_transfer_k{k}_{c}_s{seed}_cl80',
                                             f'K2 transfer ladder: operator {c} model from its first {k} parks, '
                                             f'zero-shot on the other {90 - k} parks, seed {seed}')})


POOLS = {'mix48': lambda cl: [p for ps in cl.values() for p in ps[:6]],
         'mix24': lambda cl: [p for ps in cl.values() for p in ps[:3]],
         'mix8': lambda cl: [ps[0] for ps in cl.values()],
         'reg24': lambda cl: [p for c, ps in cl.items() if c.startswith('R') for p in ps[:6]],
         'nat24': lambda cl: [p for c, ps in cl.items() if c.startswith('N') for p in ps[:6]]}


def pool_global(rows: list) -> None:
    """K3 global zero-/few-shot model (Viktor, 2026-10-12): a pool of 8 / 24 / 48 parks (mixed over all clients)
    or 24 parks of the regional / nationwide clients only, scored on the same 42 unseen parks (parks 6-9 of every
    client + the 10 holdout parks). zero-shot = central model on the pool (data.holdout_files = unseen);
    few-shot = FedGradient with the pool as ONE client and the 42 unseen parks as one-park newcomers with 2w / 1m
    history + FT variants (the newcomers' weeks enter pretraining at ~3 % weight). Reference: the unseen parks'
    own short-history models (scenario poolglobal_ref, run `local` only) and their full-history models.
    Seeds 42-44 in seed-major order."""
    global OUT
    OUT = os.path.join(BASE, 'scenarios_poolglobal')
    b = bases('static')
    clients = b['fl_fedgradient']['fl']['clients']
    holdout = list(b['fl_fedgradient']['data']['val_files'])
    unseen = [p for ps in clients.values() for p in ps[6:]] + holdout

    def few_shot(pool_parks, tag, seed):
        cfg = copy.deepcopy(b['fl_fedgradient'])
        cfg['params']['random_seed'] = seed
        cfg['data']['files'] = list(pool_parks) + list(unseen)
        cfg['data']['val_files'] = []
        cfg['data']['station_history_start'] = {p: NEWCOMER_HISTORY[tag] for p in unseen}
        cfg['model']['epochs'] = 1000
        cfg['model']['early_stopping']['patience'] = 100
        cfg['fl']['clients'] = ({'pool': list(pool_parks)} if pool_parks else {}) | {f'{p}_new': [p] for p in unseen}
        cfg['fl']['n_rounds'] = 200
        cfg['fl']['global_early_stopping']['patience'] = 20
        cfg['fl']['max_concurrent_actors'] = 22
        cfg['fl']['drift_metrics'] = True
        return fine_tune(cfg, 10, variants=True)

    for seed in SEEDS:
        for name, fn in POOLS.items():
            pool = fn(clients)
            cfg = copy.deepcopy(b['cl80'])
            cfg['params']['random_seed'] = seed
            cfg['data']['files'] = list(pool)
            cfg['data']['holdout_files'] = list(unseen)
            rows.append({'scenario': 'poolglobal', 'pool': name, 'history': 'none', 'seed': seed, 'method': 'cl80',
                         'config': write(cfg, f'scen_poolglobal_{name}_s{seed}_cl80',
                                         f'K3 pool {name} ({len(pool)} parks) central, zero-shot on 42 unseen parks, '
                                         f'seed {seed}')})
            for tag in ('2w', '1m'):
                rows.append({'scenario': 'poolglobal', 'pool': name, 'history': tag, 'seed': seed,
                             'method': 'fl_fedgradient',
                             'config': write(few_shot(pool, tag, seed), f'scen_poolglobal_{name}_h{tag}_s{seed}_fl_fedgradient',
                                             f'K3 pool {name} ({len(pool)} parks) as one client + 42 unseen parks as '
                                             f'newcomers with {tag} history, few-shot FT, seed {seed}')})
        for tag in ('2w', '1m'):
            rows.append({'scenario': 'poolglobal_ref', 'pool': 'ref', 'history': tag, 'seed': seed,
                         'method': 'fl_fedgradient',
                         'config': write(few_shot([], tag, seed), f'scen_poolglobal_ref_h{tag}_s{seed}_fl_fedgradient',
                                         f'K3 reference: the 42 unseen parks as one-park clients with {tag} history '
                                         f'(run method local only), seed {seed}')})


def trianel_newcomer_pool(rows: list) -> None:
    """Real-park control for --part trianel_newcomer: the 11 established parks form ONE client (the operator's
    pool, in effect CL11) and the newcomer with its short history the second; zero-shot / few-shot as in
    newcomer_pool. Seeds 42-44 in seed-major order (seed 42 first)."""
    global OUT
    OUT = os.path.join(BASE, '..', 'trianel_newcomer_pool')
    for seed in SEEDS:
        for park in TRIANEL_NEWCOMER_PARKS:
            for tag, start in TRIANEL_NEWCOMER_HISTORY.items():
                cfg = scale(trianel_base('fl_fedgradient', True), 10)
                cfg['params']['random_seed'] = seed
                cfg['data']['station_history_start'] = {park: start}
                cfg['fl']['clients'] = {'pool': [p for p in TRIANEL_PARKS if p != park], park: [park]}
                cfg['fl']['max_concurrent_actors'] = 2
                fine_tune(cfg, 10, variants=True)
                stag = '' if seed == 42 else f'_s{seed}'
                rows.append({'scenario': 'trianel_newcomer_pool', 'park': park, 'history': tag, 'seed': seed,
                             'method': 'fl_fedgradient',
                             'config': write(cfg, f'trianel_newpool_{park}_h{tag}{stag}_fl_fedgradient',
                                             f'Trianel newcomer {park} ({tag} history) with the 11 other parks as '
                                             f'one pool client, seed {seed}, fl_fedgradient')})
    for r in rows:
        p = os.path.join(REPO, r['config'] + '.yaml')
        txt = open(p).read().replace(f'results_path: {RESULTS}', 'results_path: results/trianel')
        open(p, 'w').write(txt)


def large_batch(rows: list) -> None:
    """Central x4 runs with the FedGradient optimisation regime (effective batch 8 x 64 = 512, Adam lr 0.01,
    clipnorm 1.0, no lag): is the FL gain against the available power (fl_scenarios_v2 C) an aggregation effect
    or a large-batch / few-steps optimisation effect reproducible centrally? masks none / market_env, seeds 42-44."""
    global OUT
    OUT = os.path.join(BASE, 'scenarios_regular')
    b = bases('static_nolag')['cl80']
    for spec in ('none', 'market_env'):
        for seed in SEEDS:
            for bs, lr in ((512, 0.01), (64, 0.0005)):
                cfg = dataset(copy.deepcopy(b), 'x4')
                cfg['params']['random_seed'] = seed
                if spec != 'none':
                    cfg['data']['target_mask'] = spec
                cfg['model']['batch_size'] = bs
                cfg['model']['lr'] = lr
                cfg['model']['tft']['clipnorm'] = 1.0 if bs == 512 else cfg['model']['tft']['clipnorm']
                if bs == 512:
                    cfg['model']['epochs'] = 300
                    cfg['model']['early_stopping']['patience'] = 30     # ~58 steps/epoch as in FedGradient
                rows.append({'scenario': 'largebatch', 'mask': spec, 'batch': bs, 'seed': seed, 'method': 'cl80',
                             'config': write(cfg, f'scen_reg_x4_{spec}_b{bs}_s{seed}_cl80',
                                             f'central x4 no lag, mask {spec}, batch {bs}, lr {lr}, seed {seed}')})


NEWCOMER_HISTORY = {'2w': '2024-07-18', '1m': '2024-07-01', '2m': '2024-06-01', '3m': '2024-05-01', '6m': '2024-02-01'}


def newcomer(rows: list) -> None:
    """Onboarding curve of a small operator: park 0 of every client becomes its own one-park client with a
    short history (data.station_history_start), the 8 established clients keep their other 9 parks with the
    full year. 16 clients; local = train_local (the newcomer's own model on its few weeks), FL global and
    fine-tuned (newcomer fine-tunes on its own data), central with the same data. Histories 2 weeks .. 6 months,
    seeds 42-44, with lag. Question: how much own history does a one-park operator need before federation
    stops paying off?"""
    global OUT
    OUT = os.path.join(BASE, 'scenarios_newcomer')
    b = bases('static')
    clients = b['fl_fedgradient']['fl']['clients']
    new = {f'{c}_new': [p[0]] for c, p in clients.items()}
    keep = {c: list(p[1:]) for c, p in clients.items()}
    for tag, start in NEWCOMER_HISTORY.items():
        for seed in SEEDS:
            for meth, cfg0 in b.items():
                cfg = copy.deepcopy(cfg0)
                cfg['params']['random_seed'] = seed
                cfg['data']['station_history_start'] = {p[0]: start for p in new.values()}
                cfg['model']['epochs'] = 1000                      # local newcomer models / fine-tuning: few samples
                cfg['model']['early_stopping']['patience'] = 100
                if meth.startswith('fl_'):
                    cfg['fl']['clients'] = {**keep, **new}
                    cfg['fl']['n_rounds'] = 200
                    cfg['fl']['global_early_stopping']['patience'] = 20
                    cfg['fl']['max_concurrent_actors'] = 16
                    cfg['fl']['drift_metrics'] = True
                    fine_tune(cfg, 10, variants=True)
                rows.append({'scenario': 'newcomer', 'history': tag, 'seed': seed, 'method': meth,
                             'config': write(cfg, f'scen_new_h{tag}_s{seed}_{meth}',
                                             f'newcomer one-park clients with {tag} history, seed {seed}, {meth}')})


PHYSICS = ['nwp_power_curve', 'nwp_wind_hub']


def physics(rows: list) -> None:
    """Physics-harmonised inputs as a heterogeneity treatment (G5): the NWP layer winds interpolated to the
    turbine groups' hub heights and the library power curve applied (preprocessing.add_physics_features).
    add = base ICON layers + physics + ECMWF; only = physics + ECMWF (no raw ICON layers). x1 statics with lag,
    FedGradient / FedAvg (fine-tune variants, drift metrics) / CL80 / local, seeds 42-44."""
    global OUT
    OUT = os.path.join(BASE, 'scenarios_physics')
    b = bases('static')
    for var in ('add', 'only'):
        for seed in SEEDS:
            for meth, cfg0 in b.items():
                cfg = copy.deepcopy(cfg0)
                cfg['params']['random_seed'] = seed
                icon = [f for f in cfg['params']['known_features'] if not f.startswith('ecmwf_')]
                ecmwf = [f for f in cfg['params']['known_features'] if f.startswith('ecmwf_')]
                cfg['params']['known_features'] = (icon if var == 'add' else []) + PHYSICS + ecmwf
                if meth.startswith('fl_'):
                    cfg['fl']['drift_metrics'] = True
                    fine_tune(cfg, variants=True)
                rows.append({'scenario': 'physics', 'variant': var, 'seed': seed, 'method': meth,
                             'config': write(cfg, f'scen_phys_{var}_s{seed}_{meth}',
                                             f'physics features ({var}: {cfg["params"]["known_features"]}), '
                                             f'seed {seed}, {meth}')})


def main():
    rows = []
    part = sys.argv[sys.argv.index('--part') + 1] if '--part' in sys.argv else '1'
    if part == 'physics':
        physics(rows)
    elif part == 'newcomer':
        newcomer(rows)
    elif part == 'largebatch':
        large_batch(rows)
    elif part == 'trianel':
        trianel(rows)
    elif part == 'trianel_newcomer':
        trianel_newcomer(rows)
    elif part == 'newcomer_pool':
        newcomer_pool(rows)
    elif part == 'transfer':
        transfer_ladder(rows)
    elif part == 'poolglobal':
        pool_global(rows)
    elif part == 'trianel_newcomer_pool':
        trianel_newcomer_pool(rows)
    elif part == 'drift':
        drift(rows)
    elif part == 'nolag':
        no_lag(rows)
    elif part == '2':
        part2(rows)
    elif part == 'wind':
        hub_wind(rows)
    elif part == 'holdout':
        holdout_parks(rows)
    elif part == 'noecmwf':
        no_ecmwf(rows)
    elif part == 'poolsize':
        pool_size(rows)
    elif part == 'split2':
        split2(rows)
    elif part == 'runs':
        forecast_runs(rows)
    elif part == 'features':
        features(rows)
    else:
        scarce(rows)
        lopo(rows)
        mask(rows)
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(OUT, 'manifest.csv'), index=False)
    print(man.groupby(['scenario', 'method']).size().to_string())
    print(f'{len(man)} configs -> {OUT}')


if __name__ == '__main__':
    sys.exit(main())
