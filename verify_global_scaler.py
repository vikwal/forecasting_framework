"""
Verification for the global-scaler change. Uses a REDUCED station list so it runs in
minutes; writes its own cache entry (own hash) and removes it again at the end, so the
production cache is untouched.

Checks:
  1. the global scaler_x branch is actually taken (statics no longer raw)
  2. the observed history keeps a consistent physical scale across stations
  3. y stays in raw m/s  -> val_rmse remains comparable to the GNN studies
"""
import sys, os, glob, shutil, logging
import numpy as np

sys.path.insert(0, '.')
from utils import tools, preprocessing, data_cache

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

STATIC_NAMES = ['altitude', 'latitude', 'longitude', 'elevation', 'slope', 'aspect_sin',
                'aspect_cos', 'tpi5', 'tpi75', 'tdi', 'elev_std', 'z0', 'dist_coast']

variant = sys.argv[1] if len(sys.argv) > 1 else 'hist'
cfg = tools.load_config(f'configs/tft_bc/config_wind_tft_{variant}.yaml')
cfg['data']['files'] = ['00183', '00460', '00856', '05404', '01443', '01684']
cfg['data']['val_files'] = ['01550', '02907', '07393']
cfg['hpo']['kfolds'] = 2
features = preprocessing.get_features(cfg)

cache = data_cache.DataCache()
_, cache_id = cache.is_cached(cfg, features, 'tft')
print(f"\n### variant={variant}  cache_id={cache_id}")

loader, cache_id = data_cache.create_or_load_preprocessed_data(
    config=cfg, features=features, model_name='tft', force_reprocess=True)

(Xtr, ytr), (Xva, yva) = loader[0]
print(f"\nfold 0: train {ytr.shape}  val {yva.shape}")
print(f"  observed {Xtr['observed'].shape}   known {Xtr['known'].shape}   static {Xtr['static'].shape}")

print("\n--- 1) STATIC features (must no longer be raw) ---")
st = Xtr['static']
raw_like = 0
for i in range(st.shape[-1]):
    c = st[:, i]
    nm = STATIC_NAMES[i] if i < len(STATIC_NAMES) else str(i)
    flag = ''
    if np.nanmax(np.abs(c)) > 25:
        flag = '  <-- STILL RAW'; raw_like += 1
    print(f"    {nm:12s} min={np.nanmin(c):9.4f} max={np.nanmax(c):9.4f} mean={np.nanmean(c):8.4f}{flag}")
print(f"  columns still on a raw scale: {raw_like}  ->  {'FAIL' if raw_like else 'OK (global scaler applied)'}")

print("\n--- 2) OBSERVED tensor scale ---")
obs = Xtr['observed']
for i in range(obs.shape[-1]):
    c = obs[:, :, i]
    print(f"    obs[{i}] min={np.nanmin(c):8.3f} max={np.nanmax(c):8.3f} "
          f"mean={np.nanmean(c):7.4f} std={np.nanstd(c):7.4f}")
print("  (per-station scaling would force EVERY column to mean 0 / std 1 per station;")
print("   a global scaler leaves station-to-station level differences visible)")

print("\n--- 3) TARGET must stay in raw m/s ---")
print(f"    y_train min={ytr.min():.3f} max={ytr.max():.3f} mean={ytr.mean():.3f} m/s")
print(f"    y_val   min={yva.min():.3f} max={yva.max():.3f} mean={yva.mean():.3f} m/s")
ok_y = 0.0 <= ytr.min() and 1.0 < ytr.mean() < 15.0 and ytr.max() > 15
print(f"    plausible physical wind speeds: {'OK' if ok_y else 'CHECK'}  "
      f"-> val_rmse stays in m/s, comparable to the GNN studies")

# clean up this verification-only cache entry
for p in glob.glob(os.path.join(data_cache.DEFAULT_CACHE_DIR, f"{cache_id}*")):
    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
print(f"\ncleaned up verification cache entry {cache_id}")
