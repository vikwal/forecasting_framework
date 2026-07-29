"""
Smoke test: one 'base' run vs one 'hist' run with IDENTICAL hyperparameters, identical
seed, identical folds — the only difference is params.observed_features.

Mirrors hpo_cl_tft_bc.py's trial body (same data_cache entry point, same
tools.training_pipeline call, same 'best epoch of val_rmse' metric), but with fixed
hyperparameters instead of an Optuna sample, so the two variants are directly
comparable. Writes nothing to Optuna.

Usage: python smoke_test_base_vs_hist.py <base|hist> --gpu N [--folds K] [--epochs E]
"""
import sys, os, json, time, argparse, logging, random
import numpy as np
import torch

sys.path.insert(0, '.')
from utils import tools, preprocessing, hpo, data_cache

p = argparse.ArgumentParser()
p.add_argument('variant', choices=['base', 'hist'])
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--folds', type=int, default=3)
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--patience', type=int, default=8)
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Full determinism so the two variants differ only in their features
random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

config = tools.load_config(f'configs/tft_bc/config_wind_tft_{args.variant}.yaml')
features = preprocessing.get_features(config=config)

# Fixed preprocessing params (config defaults) — identical for both variants
config['params']['next_n_grid_points'] = 4
config['params']['next_n_grid_ecmwf'] = 4
config['params']['next_n_stations'] = 5
config['model']['epochs'] = args.epochs
config['model']['early_stopping']['patience'] = args.patience
config['model']['force_retrain'] = True

hyperparameters = hpo.get_hyperparameters(config=config, hpo=False)
hyperparameters['epochs'] = args.epochs

device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
logging.info(f"=== SMOKE TEST variant={args.variant} device={device} seed={args.seed}")
logging.info(f"observed_features: {features['observed']}")
logging.info(f"hyperparameters:   {json.dumps(hyperparameters, default=str)}")

t0 = time.time()
lazy_fold_loader, cache_id = data_cache.create_or_load_preprocessed_data(
    config=config, features=features, model_name='tft',
    force_reprocess=False, use_cache=True)
logging.info(f"cache_id={cache_id}  folds={len(lazy_fold_loader)}  "
             f"(preprocessing took {time.time()-t0:.0f}s)")

fold_rmses = []
n_folds = min(args.folds, len(lazy_fold_loader))
for fold_idx in range(n_folds):
    train, val = lazy_fold_loader[fold_idx]
    X_train, y_train = train
    X_val, y_val = val
    logging.info(f"--- fold {fold_idx}: train={y_train.shape} val={y_val.shape} "
                 f"observed_dim={X_train['observed'].shape[-1]} "
                 f"known_dim={X_train['known'].shape[-1]} static_dim={X_train['static'].shape[-1]}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    config['model_name'] = f'smoke_{args.variant}_fold{fold_idx}'
    history, model = tools.training_pipeline(
        train=train, val=val, hyperparameters=hyperparameters,
        config=config, device=device)

    best = min(history['val_rmse'])
    fold_rmses.append(best)
    logging.info(f"--- fold {fold_idx}: best val_rmse = {best:.4f} m/s "
                 f"(epochs run: {len(history['val_rmse'])})")
    del model, history
    torch.cuda.empty_cache()

mean_rmse = float(np.mean(fold_rmses))
print("\n" + "=" * 64)
print(f"RESULT  variant={args.variant}")
print(f"  per-fold best val_rmse : {[round(v, 4) for v in fold_rmses]}")
print(f"  MEAN val_rmse          : {mean_rmse:.4f} m/s")
print("=" * 64)

os.makedirs('reports/smoke_test', exist_ok=True)
with open(f'reports/smoke_test/{args.variant}.json', 'w') as f:
    json.dump({'variant': args.variant, 'cache_id': cache_id, 'seed': args.seed,
               'hyperparameters': hyperparameters, 'fold_rmses': fold_rmses,
               'mean_val_rmse': mean_rmse}, f, indent=1, default=str)
