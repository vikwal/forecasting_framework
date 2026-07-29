"""
Repro / diagnostic ONLY — no config, code or DB changes.

Runs the real preprocessing path (get_data -> knn_imputer -> prepare_data_for_tft)
for ONE station under configs/tft_bc/config_wind_tft_{base,hist}.yaml and answers:

  1. which columns actually end up in known_future_cols / observed_past_cols
  2. shapes of X_train['observed'] / X_train['known']
  3. does the 'wind_speed' column in the observed tensor carry the station's own
     measured history, unscaled-comparable to the raw dataframe values?
"""
import sys, logging
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from utils import tools, preprocessing

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

STATION = sys.argv[1] if len(sys.argv) > 1 else '00183'
out = {}

for variant in ('base', 'hist'):
    print(f"\n{'='*78}\n===  VARIANT: {variant}   station {STATION}\n{'='*78}")
    cfg = tools.load_config(f'configs/tft_bc/config_wind_tft_{variant}.yaml')
    cfg['data']['files'] = [STATION]
    cfg['data']['val_files'] = []
    target_col = cfg['data']['target_col']
    feats = preprocessing.get_features(cfg)
    print(f"config observed_features : {feats['observed']}")
    print(f"config known_features    : {feats['known']}")
    print(f"config target_col        : {target_col}")

    dfs = preprocessing.get_data(data_dir=cfg['data']['path'],
                                 freq=cfg['data']['freq'],
                                 config=cfg,
                                 features=feats,
                                 target_col=target_col)
    key, df_raw = list(dfs.items())[0]

    # mirror pipeline()'s first step so the resolved column names match exactly
    df_imp = preprocessing.knn_imputer(data=df_raw.copy(),
                                       n_neighbors=cfg['data']['n_neighbors'])
    known_res, obs_res = preprocessing.prepare_features_for_tft(
        cols=df_imp.columns.tolist(),
        known_future_cols=feats['known'],
        observed_past_cols=feats['observed'])

    print(f"\nRESOLVED observed_past_cols ({len(obs_res)}): {obs_res}")
    print(f"RESOLVED known_future_cols  ({len(known_res)}): {known_res}")
    overlap = sorted(set(obs_res) & set(known_res))
    print(f"OVERLAP known/observed      : {overlap if overlap else 'none'}")

    # raw, pre-scaling reference: timestamp -> own wind_speed (first occurrence,
    # exactly how create_tft_sequences resolves the observed window)
    ts_all = df_imp.index.get_level_values('timestamp')
    raw_ws = None
    if 'wind_speed' in df_imp.columns:
        raw_ws = pd.Series(df_imp['wind_speed'].values, index=ts_all)
        raw_ws = raw_ws[~raw_ws.index.duplicated(keep='first')]

    prepared, _ = preprocessing.pipeline(data=df_imp.copy(),
                                         config=cfg,
                                         known_cols=feats['known'],
                                         observed_cols=feats['observed'],
                                         static_cols=feats['static'],
                                         target_col=target_col)

    Xtr, ytr = prepared['X_train'], prepared['y_train']
    idx_tr = prepared['index_train']
    print(f"\nX_train['observed'] shape : {Xtr['observed'].shape}")
    print(f"X_train['known']    shape : {Xtr['known'].shape}")
    print(f"y_train             shape : {ytr.shape}")
    if 'static' in Xtr:
        print(f"X_train['static']   shape : {Xtr['static'].shape}")

    assert Xtr['observed'].shape[-1] == len(obs_res), \
        f"observed tensor width {Xtr['observed'].shape[-1]} != len(obs_res) {len(obs_res)}"

    obs_scaler = prepared['scalers'].get('x_observed')
    print(f"observed scaler           : {type(obs_scaler).__name__ if obs_scaler else None}")
    if obs_scaler is not None and hasattr(obs_scaler, 'mean_'):
        for c, m, s in zip(obs_res, obs_scaler.mean_, obs_scaler.scale_):
            print(f"    {c:24s} mean={m:8.4f} scale={s:8.4f}")

    # ---- the actual check: own wind_speed inside the observed tensor ----------
    if 'wind_speed' in obs_res:
        col = obs_res.index('wind_speed')
        print(f"\n'wind_speed' is observed column index {col}")
        j = min(5, len(ytr) - 1)
        t0 = pd.Timestamp(idx_tr[j])
        win_scaled = Xtr['observed'][j, :, :]
        win_raw = obs_scaler.inverse_transform(win_scaled) if obs_scaler is not None else win_scaled
        got = win_raw[:, col]
        # replicate create_tft_sequences' anchor exactly: the observed window ends
        # at the FIRST timestamp of the current forecast run, not at its starttime.
        st_all = df_imp.index.get_level_values('starttime')
        cur = np.where(st_all == t0)[0]
        fst = ts_all[cur[0]]
        expect_ts = pd.date_range(start=fst - pd.Timedelta(hours=len(got)),
                                  end=fst, freq='1h', inclusive='left')
        expect = raw_ws.reindex(expect_ts).values
        tgt_ts = ts_all[cur]
        print(f"window j={j}  starttime={t0}  first run timestamp={fst}")
        print(f"  observed window : {expect_ts[0]} .. {expect_ts[-1]}")
        print(f"  target window   : {tgt_ts[0]} .. {tgt_ts[-1]}")
        gap = (tgt_ts[0] - expect_ts[-1]) / pd.Timedelta(hours=1)
        print(f"  gap last-observed -> first-target: {gap:.0f} h  (1 h = contiguous, no leakage)")
        print(f"  observed[:, wind_speed] first 6 (inverse-scaled): {np.round(got[:6], 4)}")
        print(f"  raw df wind_speed        first 6 (from df_imp)  : {np.round(expect[:6], 4)}")
        print(f"  observed[:, wind_speed] last 6                  : {np.round(got[-6:], 4)}")
        print(f"  raw df wind_speed        last 6                 : {np.round(expect[-6:], 4)}")
        print(f"  max |diff| over the 48h window : {np.nanmax(np.abs(got - expect)):.3e}")
        print(f"  MATCH: {np.allclose(got, expect, atol=1e-6, equal_nan=True)}")
        print(f"  y_train[{j}] (target, 48h after t0) first 6     : {np.round(ytr[j][:6], 4)}")
        # continuity: last observed hour vs first target hour
        print(f"  last observed hour = {got[-1]:.4f}   first target hour = {ytr[j][0]:.4f}")

        # how redundant is own history with the neighbour columns?
        flat = obs_scaler.inverse_transform(
            Xtr['observed'].reshape(-1, Xtr['observed'].shape[-1])) if obs_scaler is not None \
            else Xtr['observed'].reshape(-1, Xtr['observed'].shape[-1])
        own = flat[:, col]
        print("\n  correlation own wind_speed vs. other observed cols:")
        for i, c in enumerate(obs_res):
            if i == col:
                continue
            m = np.isfinite(own) & np.isfinite(flat[:, i])
            print(f"    r(wind_speed, {c:22s}) = {np.corrcoef(own[m], flat[m, i])[0,1]: .4f}")

        # persistence value: corr(last observed hour, target at each lead)
        last_obs = obs_scaler.inverse_transform(
            Xtr['observed'][:, -1, :])[:, col] if obs_scaler is not None else Xtr['observed'][:, -1, col]
        print("\n  corr(last observed wind_speed, y at lead h):")
        for h in (0, 1, 2, 5, 11, 23, 47):
            m = np.isfinite(last_obs) & np.isfinite(ytr[:, h])
            print(f"    lead {h+1:2d}h : r = {np.corrcoef(last_obs[m], ytr[m, h])[0,1]: .4f}")
    else:
        print("\n'wind_speed' NOT in observed columns (expected for 'base').")

    out[variant] = dict(obs=obs_res, known=known_res,
                        obs_shape=Xtr['observed'].shape,
                        known_shape=Xtr['known'].shape,
                        n=len(ytr))

print(f"\n{'='*78}\n===  BASE vs HIST\n{'='*78}")
print(f"known cols identical : {out['base']['known'] == out['hist']['known']}")
print(f"observed base        : {out['base']['obs']}")
print(f"observed hist        : {out['hist']['obs']}")
print(f"observed delta       : {sorted(set(out['hist']['obs']) - set(out['base']['obs']))}")
print(f"observed shapes      : base={out['base']['obs_shape']}  hist={out['hist']['obs_shape']}")
print(f"n train windows      : base={out['base']['n']}  hist={out['hist']['n']}")
