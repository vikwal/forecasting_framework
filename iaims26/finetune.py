"""Fine-tune a fold checkpoint on one held-out farm across the
(budget x freezing-variant) grid and evaluate on the test window.

One farm = 1 zero-shot reference + len(BUDGETS) x len(VARIANTS) fine-tune
cells. The farm's data is loaded once; each budget re-runs the preprocessing
pipeline with a different train_start (the repo's built-in few-shot hook).
Early stopping validates on the farm's test window — same convention as the
repo's local baselines.

Usage (single farm, mostly for debugging — use run_finetune_grid.py for all):
  python iaims26/finetune.py --station 15813 [--smoke] [--save-models]

Output: iaims26/results/finetune/ft_{station}.csv
"""

import argparse
import copy
import gc
import logging
import time

import pandas as pd
import torch

import common
import freezing
import trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

COLUMNS = ['station_id', 'fold', 'budget', 'variant', 'period',
           'R2', 'RMSE', 'MAE', 'n_windows',
           'n_train_windows', 'trainable_params', 'epochs_run', 'seconds']

FT_LR_FACTOR = 0.1       # precedent: fl.pretrained_weights_lr = 0.1 * base lr
FT_MAX_EPOCHS = 100
FT_PATIENCE = 10
LOCAL_MAX_EPOCHS = 150   # from-scratch trainings (local/scratch) -> more epochs

# 'scratch' = pretraining control: random-init model on the SAME budget data
# with the SAME (fold) scaler, full base lr. Isolates the contribution of the
# pretrained weights at every budget (vs. the data budget alone).
GRID_VARIANTS = list(freezing.VARIANTS) + ['scratch']


def _metric_rows(y_true, y_pred, index_test, base: dict):
    from utils import eval as eval_utils
    rows = []
    for period, mask in common.period_masks(index_test).items():
        if not mask.any():
            continue
        m = eval_utils.get_metrics(y_pred=y_pred[mask], y_true=y_true[mask])
        rows.append([base['station_id'], base['fold'], base['budget'], base['variant'],
                     period, m['R^2'][0], m['RMSE'][0], m['MAE'][0], int(mask.sum()),
                     base['n_train_windows'], base['trainable_params'],
                     base['epochs_run'], base['seconds']])
    return rows


def _done_cells(csv_path):
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df['budget'], df['variant']))


def run_farm(station: str, smoke: bool = False, save_models: bool = False,
             device: str = None, budgets=None, variants=None):
    from utils import tools, preprocessing

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    budgets = budgets or list(common.BUDGETS)
    variants = variants or list(GRID_VARIANTS)

    folds = common.load_folds()
    if smoke:
        import gen_configs
        fold = next(k for k, held in enumerate(gen_configs.SMOKE_HELDOUT)
                    if station in held)
    else:
        fold = common.fold_of_station(folds, station)
    name = common.config_name_for_fold(fold, smoke=smoke)
    config = common.load_generated_config(name)
    assert station in config['data']['val_files'], \
        f'{station} is not held out in fold {fold} — refusing to fine-tune on a training farm'

    scaler = common.load_scaler(common.scaler_path(name))
    sd = common.load_state_dict_clean(common.checkpoint_path(name))
    dims = common.infer_feature_dims(sd)
    hyperparameters = common.resolve_hyperparameters(config)
    base_lr = hyperparameters.get('lr', config['model']['lr'])
    clipnorm = config['model']['tft'].get('clipnorm')

    out_csv = common.RESULTS_DIR / 'finetune' / f'ft_{"smoke_" if smoke else ""}{station}.csv'
    done = _done_cells(out_csv)

    dfs = common.get_station_dfs(config, [station])
    features = preprocessing.get_features(config=config)

    def fresh_model():
        model = common.build_tft(config, hyperparameters, dims)
        model.load_state_dict(sd, strict=True)
        return model.to(device)

    for budget in budgets:
        cells = [v for v in variants if (budget, v) not in done]
        need_zeroshot = ('zeroshot', 'none') not in done and budget == budgets[0]
        if not cells and not need_zeroshot:
            continue

        eval_cfg = common.make_eval_config(
            config, config['data']['test_start'], config['data']['test_end'],
            train_start=common.budget_train_start(budget))
        generator = tools.create_data_generator(copy.deepcopy(dfs), eval_cfg,
                                                features, scaler_x=scaler)
        data_dict = next(iter(generator))
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        index_test = data_dict['index_test']
        scaler_y = data_dict['scalers']['y']
        n_train_windows = int(len(y_train))
        logging.info(f'{station} fold={fold} budget={budget}: '
                     f'{n_train_windows} train windows, {len(y_test)} test windows')

        # Zero-shot reference (no training), once per farm
        if need_zeroshot:
            model = fresh_model()
            model.eval()
            y_true, y_pred = tools.get_y(X_test=X_test, y_test=y_test, model=model,
                                         scaler_y=scaler_y, device=device)
            base = dict(station_id=station, fold=fold, budget='zeroshot', variant='none',
                        n_train_windows=0, trainable_params=0, epochs_run=0, seconds=0.0)
            common.append_rows_atomic(out_csv, _metric_rows(y_true, y_pred, index_test, base),
                                      COLUMNS)
            del model

        if n_train_windows == 0:
            logging.warning(f'{station} budget={budget}: no training windows — '
                            f'recording empty cells')
            for variant in cells:
                base = dict(station_id=station, fold=fold, budget=budget, variant=variant,
                            n_train_windows=0, trainable_params=0, epochs_run=0, seconds=0.0)
                rows = [[station, fold, budget, variant, 'no_data',
                         float('nan'), float('nan'), float('nan'), 0, 0, 0, 0, 0.0]]
                common.append_rows_atomic(out_csv, rows, COLUMNS)
            continue

        for variant in cells:
            t0 = time.time()
            if variant == 'scratch':
                model = common.build_tft(config, hyperparameters, dims).to(device)
                info = {'trainable_params': sum(p.numel() for p in model.parameters())}
                lr, max_epochs = base_lr, LOCAL_MAX_EPOCHS
            else:
                model = fresh_model()
                info = freezing.apply_freeze(model, variant)
                lr, max_epochs = base_lr * FT_LR_FACTOR, FT_MAX_EPOCHS
            history, model, epochs_run = trainer.train_model(
                model, X_train, y_train, X_test, y_test,
                batch_size=hyperparameters['batch_size'],
                lr=lr,
                device=device, epochs=max_epochs, patience=FT_PATIENCE,
                clipnorm=clipnorm)
            model.eval()
            y_true, y_pred = tools.get_y(X_test=X_test, y_test=y_test, model=model,
                                         scaler_y=scaler_y, device=device)
            seconds = round(time.time() - t0, 1)
            base = dict(station_id=station, fold=fold, budget=budget, variant=variant,
                        n_train_windows=n_train_windows,
                        trainable_params=info['trainable_params'],
                        epochs_run=epochs_run, seconds=seconds)
            common.append_rows_atomic(out_csv, _metric_rows(y_true, y_pred, index_test, base),
                                      COLUMNS)
            logging.info(f'{station} budget={budget} variant={variant}: '
                         f'{epochs_run} epochs in {seconds}s')
            if save_models:
                ft_dir = common.MODELS_DIR / 'ft'
                ft_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),
                           ft_dir / f'ft_f{fold}_{station}_{budget}_{variant}.pt')
            del model
            torch.cuda.empty_cache()

        del data_dict, X_train, y_train, X_test, y_test
        gc.collect()

    return out_csv


def run_local_baseline(station: str, smoke: bool = False, device: str = None):
    """Protocol-identical local baseline: fresh (random-init) TFT trained only
    on this farm's full history with its OWN per-farm scaler, validated on the
    farm's test window — the local counterpart to every grid cell.

    Appends rows with budget='local', variant='local' to the same per-farm CSV.
    """
    from utils import tools, preprocessing
    import hpo_zeroshot

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    folds = common.load_folds()
    if smoke:
        import gen_configs
        fold = next(k for k, held in enumerate(gen_configs.SMOKE_HELDOUT)
                    if station in held)
    else:
        fold = common.fold_of_station(folds, station)
    name = common.config_name_for_fold(fold, smoke=smoke)
    config = common.load_generated_config(name)
    hyperparameters = common.resolve_hyperparameters(config)

    out_csv = common.RESULTS_DIR / 'finetune' / f'ft_{"smoke_" if smoke else ""}{station}.csv'
    if ('local', 'local') in _done_cells(out_csv):
        return out_csv

    dfs = common.get_station_dfs(config, [station])
    features = preprocessing.get_features(config=config)
    eval_cfg = common.make_eval_config(config, config['data']['test_start'],
                                       config['data']['test_end'])
    own_scaler = hpo_zeroshot.fit_fold_scaler(dfs, eval_cfg)
    generator = tools.create_data_generator(dfs, eval_cfg, features,
                                            scaler_x=own_scaler)
    data_dict = next(iter(generator))

    t0 = time.time()
    sd_dims = {'observed_dim': data_dict['X_train']['observed'].shape[-1],
               'known_dim': data_dict['X_train']['known'].shape[-1],
               'static_dim': (data_dict['X_train']['static'].shape[-1]
                              if 'static' in data_dict['X_train'] else 0)}
    model = common.build_tft(config, hyperparameters, sd_dims).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    history, model, epochs_run = trainer.train_model(
        model, data_dict['X_train'], data_dict['y_train'],
        data_dict['X_test'], data_dict['y_test'],
        batch_size=hyperparameters['batch_size'],
        lr=hyperparameters.get('lr', config['model']['lr']),
        device=device, epochs=LOCAL_MAX_EPOCHS, patience=FT_PATIENCE,
        clipnorm=config['model']['tft'].get('clipnorm'))
    model.eval()
    y_true, y_pred = tools.get_y(X_test=data_dict['X_test'], y_test=data_dict['y_test'],
                                 model=model, scaler_y=data_dict['scalers']['y'],
                                 device=device)
    seconds = round(time.time() - t0, 1)
    base = dict(station_id=station, fold=fold, budget='local', variant='local',
                n_train_windows=int(len(data_dict['y_train'])),
                trainable_params=n_params, epochs_run=epochs_run, seconds=seconds)
    common.append_rows_atomic(out_csv, _metric_rows(y_true, y_pred,
                                                    data_dict['index_test'], base),
                              COLUMNS)
    logging.info(f'{station} local baseline: {epochs_run} epochs in {seconds}s')
    del model, data_dict, dfs
    torch.cuda.empty_cache()
    gc.collect()
    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--station', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--budgets', nargs='+', default=None)
    parser.add_argument('--variants', nargs='+', default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    run_farm(args.station, smoke=args.smoke, save_models=args.save_models,
             device=args.device, budgets=args.budgets, variants=args.variants)


if __name__ == '__main__':
    main()
