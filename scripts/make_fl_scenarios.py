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

  python scripts/make_fl_scenarios.py              # part 1: configs and scenarios/manifest.csv
  python scripts/make_fl_scenarios.py --part 2     # part 2: scenarios2/
Holdout parks (--part holdout, configs/parks_v1/scenarios_holdout/): the 10 holdout parks scored by every
client's local model (fl.client_holdout = the 10 parks for each client; run with method local) and by the own
model of the holdout operator (CL on the 10 parks, run with method cl80); seeds 42-44. The global models of the
same setup are the 12-month runs of part 1 (scen_scarce_m12_*).

  python scripts/make_fl_scenarios.py              # part 1: configs and scenarios/manifest.csv
  python scripts/make_fl_scenarios.py --part 2     # part 2: scenarios2/
  python scripts/make_fl_scenarios.py --part wind  # hub wind target: scenarios_wind/
  python scripts/make_fl_scenarios.py --part holdout  # holdout parks: scenarios_holdout/
  python scripts/make_fl_scenarios.py --part noecmwf  # ICON-D2 only: scenarios_noecmwf/
  python scripts/make_fl_scenarios.py --part poolsize  # parks per model: scenarios_poolsize/
  python scripts/make_fl_scenarios.py --part split2    # split 2 base configs: split2/
  python scripts/make_fl_scenarios.py --part runs      # forecast runs: scenarios_runs/
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
        cfg['hpo']['tft'].update(n_lstm_layers=[1, 2], static_embedding_dim=[8, 64],   # int
                                 clipnorm=[1.0, 10.0])                    # float, linear
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


def main():
    rows = []
    part = sys.argv[sys.argv.index('--part') + 1] if '--part' in sys.argv else '1'
    if part == '2':
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
