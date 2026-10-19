"""Create the deterministic 5-fold site split for the iAIMS26 experiments.

Writes iaims26/folds.json (committed artifact):
  - seed and source config for reproducibility
  - folds: fold index -> list of 20 held-out station ids
  - station_index: station id -> integer index (sorted order; used by the
    site-ID-embedding model)
Run once; every downstream script reads the file instead of recomputing.
"""

import argparse
import json
from datetime import date

import numpy as np

import common

N_FOLDS = 5
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Overwrite an existing folds.json')
    args = parser.parse_args()

    if common.FOLDS_PATH.exists() and not args.force:
        raise SystemExit(f'{common.FOLDS_PATH} exists — refusing to overwrite '
                         f'(use --force only if no experiments depend on it yet).')

    config = common.load_base_config()
    stations = sorted(str(s) for s in config['data']['files'])
    assert len(stations) == 100, f'expected 100 stations, got {len(stations)}'
    assert len(set(stations)) == 100, 'duplicate station ids in base config'

    rng = np.random.default_rng(SEED)
    permuted = list(rng.permutation(stations))
    folds = {str(k): sorted(permuted[k * 20:(k + 1) * 20]) for k in range(N_FOLDS)}

    all_ids = [s for f in folds.values() for s in f]
    assert sorted(all_ids) == stations, 'folds do not cover all stations exactly once'

    payload = {
        'seed': SEED,
        'created': str(date.today()),
        'source_config': 'iaims26/configs/base_iaims26.yaml',
        'n_folds': N_FOLDS,
        'folds': folds,
        'station_index': {s: i for i, s in enumerate(stations)},
    }
    with open(common.FOLDS_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Wrote {common.FOLDS_PATH}')
    for k, f_stations in folds.items():
        print(f'  fold {k}: {len(f_stations)} stations, e.g. {f_stations[:4]}')


if __name__ == '__main__':
    main()
