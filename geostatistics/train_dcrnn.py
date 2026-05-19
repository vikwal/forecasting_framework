"""
train_dcrnn.py — Training script for the DCRNN Seq2Seq model.

Reuses all data loading from train_stgnn2.py (ICON-D2, ECMWF, station
measurements, scaler fitting, graph building, run-pair construction).
Only the model, config, sampler wiring, and trainer differ.

DCRNN vs STGNN key differences
--------------------------------
  Model    : Seq2Seq DCGRUCell instead of ST-blocks with GATv2Conv
  Decoder  : autoregressive loop with NWP injection at each step
  TF ratio : teacher_forcing_ratio linearly decayed over training
  Config   : ``dcrnn`` section in the YAML (not ``stgnn2``)

Add a ``dcrnn`` section to your config YAML, e.g.::

    dcrnn:
      icond2_features: ['u_10m', 'v_10m', 'wind_speed_10m']
      ecmwf_features:  ['u_wind10m', 'v_wind10m', 'wind_speed_10m']
      measurement_features: ['wind_speed', 'wind_direction']
      target_col: 'wind_speed'
      n_workers: 16                  # ICON-D2 data loader workers
      n_ahead_prefetch: 2            # batch prefetcher queue depth (default: 2)
      prefetch_workers: 1            # batch prefetcher worker threads (default: 1)

      history_length: 48
      forecast_horizon: 48
      temporal_encoding: gru    # sampler output format

      hidden: 128
      num_layers: 2
      K_hop: 2
      dropout: 0.1
      teacher_forcing_ratio: 0.5  # starting ratio (decays to 0)

      station_connectivity: delaunay
      next_n_icond2: 4
      next_n_ecmwf: 4
      use_altitude_diff: true
      neighbor_radius_km: 500

      min_target_stations: 1
      max_target_stations: 10
      max_neighbor_stations: 60
      loss_fn: mse
      loss_weights_by_horizon: true
      horizon_decay: 0.95
      lr: 3.0e-4
      weight_decay: 1.0e-5
      scheduler: cosine
      batch_size: 8
      max_epochs: 200
      patience: 15
      gradient_clip: 1.0
      icond2_run_hours: [6, 9, 12, 15]

Usage
-----
    python geostatistics/train_dcrnn.py \\
        --config configs/config_wind_stgcn.yaml \\
        --suffix v1
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Reuse ALL data-loading functions from train_stgnn2 ──────────────────────
from geostatistics.train_stgnn2 import (
    load_yaml,
    load_station_measurements,
    load_station_metadata,
    load_icond2_ml_runs,
    load_ecmwf_at_stations_and_grid,
    load_ecmwf_parquet_at_stations_and_grid,
    load_nwp_elevations,
    load_interpol_imputation,
    apply_interpol_imputation,
    load_knn_imputation,
    apply_knn_imputation,
)

# ── DCRNN-specific imports ───────────────────────────────────────────────────
from geostatistics.dcrnn import DCRNNConfig, DCRNN
from geostatistics.dcrnn.training import DCRNNTrainer

# ── Shared graph and sampling infrastructure ─────────────────────────────────
from geostatistics.stgnn import HeterogeneousGraphBuilder
from geostatistics.stgnn.training.sampler import TrainingSampler
from geostatistics.stgnn.utils.normalization import StandardScaler
from geostatistics.evaluation import evaluate as run_evaluation, find_ws_feat_idx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_dcrnn")


# ---------------------------------------------------------------------------
# Feature mode helpers
# ---------------------------------------------------------------------------

import re as _re

def resolve_feature_mode(features: list[str], mode: str) -> list[str]:
    """Filter a NWP feature list by mode.

    Modes
    -----
    absolute   : wind_speed_* only
    components : u_*/v_* only
    both       : all features (no filtering)
    dir_in_deg : load all (same as both); call apply_dir_encoding afterwards
                 to convert (u, v, speed) → (speed, sin_dir, cos_dir)

    Non-wind features (temperature, pressure, …) always pass through.
    """
    if mode in ("both", "dir_in_deg"):
        return list(features)
    result = []
    for f in features:
        is_speed = bool(_re.match(r"wind_speed_", f))
        is_comp  = bool(_re.match(r"[uv]_", f))  # u_10m, v_10m, u_wind10m, v_wind10m
        if is_speed or is_comp:
            if mode == "absolute" and is_speed:
                result.append(f)
            elif mode == "components" and is_comp:
                result.append(f)
        else:
            result.append(f)
    if not result:
        raise ValueError(
            f"resolve_feature_mode(mode={mode!r}) produced an empty feature list "
            f"from {features}. Check your config."
        )
    return result


def encode_circular_measurements(
    meas_raw: "np.ndarray",
    measurement_cols: list[str],
) -> "tuple[np.ndarray, list[str]]":
    """Replace ``wind_direction`` (degrees) with (``sin_wind_direction``, ``cos_wind_direction``).

    The two output channels lie in [-1, 1] and preserve the circular topology so a
    StandardScaler won't see the 0/360 discontinuity.  All other columns pass through.
    Applied once after imputation, before the scaler is fit.
    """
    if "wind_direction" not in measurement_cols:
        return meas_raw, measurement_cols

    idx    = measurement_cols.index("wind_direction")
    wd_rad = np.deg2rad(meas_raw[:, :, idx])        # (T, N)
    sin_wd = np.sin(wd_rad).astype(np.float32)
    cos_wd = np.cos(wd_rad).astype(np.float32)

    parts: list["np.ndarray"] = []
    if idx > 0:
        parts.append(meas_raw[:, :, :idx])
    parts.append(sin_wd[:, :, None])
    parts.append(cos_wd[:, :, None])
    if idx + 1 < meas_raw.shape[2]:
        parts.append(meas_raw[:, :, idx + 1:])

    new_cols = list(measurement_cols)
    new_cols[idx:idx + 1] = ["sin_wind_direction", "cos_wind_direction"]
    return np.concatenate(parts, axis=2), new_cols


def apply_dir_encoding(
    data: "np.ndarray",
    features: list[str],
) -> "tuple[np.ndarray, list[str]]":
    """Transform matched (u, v) pairs to (wind_speed, sin_dir, cos_dir).

    For each height level where both u_Xm and v_Xm (or u_windXm / v_windXm) are
    present: the pair is replaced by three channels:
      - wind_speed_Xm (kept if already present, else computed as sqrt(u²+v²))
      - sin_dir_Xm = -u / speed   (met. convention: sin of FROM-direction)
      - cos_dir_Xm = -v / speed

    Standalone wind_speed_* features without a matching u/v pair pass through
    unchanged.  Non-wind features always pass through.
    """
    import re

    # Locate matched u/v pairs (supports ICON-D2 and ECMWF naming)
    pair_info: dict[str, dict] = {}
    for i, f in enumerate(features):
        for pattern, key in [
            (r"^u_(\d+m)$",       "u"),
            (r"^v_(\d+m)$",       "v"),
            (r"^u_wind(\d+m)$",   "u"),
            (r"^v_wind(\d+m)$",   "v"),
            (r"^wind_speed_(\d+m)$", "speed"),
        ]:
            m = re.match(pattern, f)
            if m:
                h = m.group(1)
                pair_info.setdefault(h, {})
                if key == "speed":
                    pair_info[h].setdefault("speeds", []).append((i, f))
                else:
                    pair_info[h][key] = (i, f)
                break

    complete = {h: p for h, p in pair_info.items() if "u" in p and "v" in p}
    if not complete:
        return data, features

    consumed: set[int] = set()
    for h, p in complete.items():
        consumed.add(p["u"][0])
        consumed.add(p["v"][0])
        for idx_f, _ in p.get("speeds", []):
            consumed.add(idx_f)

    out_parts: list["np.ndarray"] = []
    out_cols: list[str] = []

    # Non-pair features first (preserve original order)
    for i, f in enumerate(features):
        if i not in consumed:
            out_parts.append(data[..., i:i + 1])
            out_cols.append(f)

    # Encoded pairs (appended after non-pair columns)
    for h in sorted(complete.keys()):
        p = complete[h]
        u_data  = data[..., p["u"][0]]
        v_data  = data[..., p["v"][0]]
        speed   = np.sqrt(u_data ** 2 + v_data ** 2)
        speed_s = np.where(speed < 1e-6, 1e-6, speed)
        sin_dir = (-u_data / speed_s).astype(np.float32)
        cos_dir = (-v_data / speed_s).astype(np.float32)

        speed_name = p["speeds"][0][1] if "speeds" in p else f"wind_speed_{h}_uv"
        out_parts += [speed[..., None].astype(np.float32),
                      sin_dir[..., None], cos_dir[..., None]]
        out_cols  += [speed_name, f"sin_dir_{h}", f"cos_dir_{h}"]

    return np.concatenate(out_parts, axis=-1), out_cols


def feature_indices(full_features: list[str], subset: list[str]) -> list[int]:
    """Return integer indices of *subset* within *full_features*."""
    idx_map = {f: i for i, f in enumerate(full_features)}
    return [idx_map[f] for f in subset]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DCRNN Seq2Seq model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument(
        "--eval", action="store_true",
        help="Run per-station evaluation on val run pairs after training and save results in the pkl.",
    )
    parser.add_argument(
        "--hpo-study", default=None, metavar="DB_PATH",
        help=(
            "Load best hyperparameters from an Optuna study and override YAML values. "
            "Pass the path to the SQLite .db file, or 'auto' to derive the path from "
            "the config stem (studies/hpo_dcrnn_<config_stem>.db). "
            "PostgreSQL is used automatically when OPTUNA_STORAGE env var is set."
        ),
    )
    parser.add_argument(
        "--test-mode", action="store_true",
        help=(
            "Final-training mode: train on files + val_files (all 'known' stations), "
            "use test_files for early stopping. "
            "Default (no flag): train on files, validate on val_files."
        ),
    )
    args = parser.parse_args()

    cfg      = load_yaml(args.config)
    data_cfg = cfg["data"]
    dcrnn_cfg = cfg.get("dcrnn", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config_stem = Path(args.config).stem.replace("config_", "")
    # Strip _fold<N> suffix for HPO study lookup so fold-specific configs
    # resolve to the same study as the base config.
    hpo_stem = _re.sub(r'_fold\d+$', '', config_stem)

    # ------------------------------------------------------------------
    # Optional: load best hyperparameters from an Optuna HPO study
    # ------------------------------------------------------------------
    hpo_best_params: dict | None = None
    hpo_study_name:  str  | None = None
    hpo_best_val_loss: float | None = None

    if args.hpo_study:
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is not installed — cannot load HPO study.")

        freq       = data_cfg.get("freq", "1h")
        H_fore_tmp = dcrnn_cfg.get("forecast_horizon", 48)
        hpo_study_name = f"cl_m-dcrnn_out-{H_fore_tmp}_freq-{freq}_{hpo_stem}"

        storage_url = os.environ.get("OPTUNA_STORAGE")
        if storage_url:
            storage = storage_url
            logger.info("Loading HPO study from PostgreSQL (OPTUNA_STORAGE) …")
        else:
            db_path = (
                args.hpo_study
                if args.hpo_study != "auto"
                else f"studies/hpo_dcrnn_{hpo_stem}.db"
            )
            storage = f"sqlite:///{db_path}"
            logger.info("Loading HPO study from SQLite: %s", db_path)

        study = optuna.load_study(study_name=hpo_study_name, storage=storage)
        hpo_best_params   = study.best_params
        hpo_best_val_loss = study.best_value
        logger.info(
            "HPO study '%s' — best val_loss=%.6f  (trial #%d)",
            hpo_study_name, hpo_best_val_loss, study.best_trial.number,
        )
        logger.info("Overriding dcrnn_cfg with HPO best params:")
        for k, v in hpo_best_params.items():
            logger.info("  %-30s %s → %s", k, dcrnn_cfg.get(k, "<unset>"), v)
        dcrnn_cfg.update(hpo_best_params)
        # Mirror hpo_dcrnn.py: nwp_out_dim is never stored in Optuna best_params
        # (it's derived post-sampling). Recompute here to guarantee divisibility.
        if "nwp_heads" in dcrnn_cfg and "nwp_out_per_head" in dcrnn_cfg:
            dcrnn_cfg["nwp_out_dim"] = dcrnn_cfg.pop("nwp_out_per_head") * dcrnn_cfg["nwp_heads"]
            logger.info("Derived nwp_out_dim = %d (nwp_heads=%d × nwp_out_per_head)",
                        dcrnn_cfg["nwp_out_dim"], dcrnn_cfg["nwp_heads"])

    model_name  = (
        f"{config_stem}_dcrnn_{args.suffix}.pt"
        if args.suffix else
        f"{config_stem}_dcrnn.pt"
    )
    model_path = Path("models") / model_name

    # Set up dynamic log file
    log_stem = Path(args.config).stem
    log_name = f"train_dcrnn_{log_stem}_{args.suffix}.log" if args.suffix else f"train_dcrnn_{log_stem}.log"
    log_path = Path("logs") / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    logging.getLogger("geostatistics").addHandler(fh)

    # ------------------------------------------------------------------
    # Station IDs
    # ------------------------------------------------------------------
    if args.test_mode:
        if not data_cfg.get("test_files"):
            raise ValueError("--test-mode requires 'test_files' to be set in the data config.")
        train_ids = [str(s) for s in data_cfg["files"]] + [str(s) for s in data_cfg["val_files"]]
        val_ids   = [str(s) for s in data_cfg["test_files"]]
        logger.info("Mode: FINAL TRAINING — train: %d (files+val_files)  early-stop val: %d (test_files)",
                    len(train_ids), len(val_ids))
    else:
        train_ids = [str(s) for s in data_cfg["files"]]
        val_ids   = [str(s) for s in data_cfg["val_files"]]
        logger.info("Mode: DEVELOPMENT — train: %d (files)  val: %d (val_files)",
                    len(train_ids), len(val_ids))
    all_ids = train_ids + val_ids
    N_train = len(train_ids)
    N_val   = len(val_ids)

    # ------------------------------------------------------------------
    # Feature config
    # ------------------------------------------------------------------
    use_case = data_cfg.get("use_case", "wind")

    icond2_features_all = dcrnn_cfg.get("icond2_features") or []
    ecmwf_features_all  = dcrnn_cfg.get("ecmwf_features")  or []
    if use_case == "wind":
        i2_mode = dcrnn_cfg.get("icond2_feature_mode", "both")
        e2_mode = dcrnn_cfg.get("ecmwf_feature_mode",  "both")
        icond2_features = resolve_feature_mode(icond2_features_all, i2_mode)
        ecmwf_features  = resolve_feature_mode(ecmwf_features_all,  e2_mode)
        if i2_mode != "both":
            logger.info("icond2_feature_mode=%s → %s", i2_mode, icond2_features)
        if e2_mode != "both":
            logger.info("ecmwf_feature_mode=%s  → %s", e2_mode, ecmwf_features)
        # direction_to_adj: u/v for decoder wind direction are loaded separately and
        # stored as data["icond2"].wind_uv — they are NOT added to model input I2.
        features_to_load = list(icond2_features)  # may gain extra u/v columns below
        uv_load_indices: list[int] | None = None
        if dcrnn_cfg.get("direction_to_adj", False):
            u_feat = dcrnn_cfg.get("wind_u_icond2_feature", "u_10m")
            v_feat = dcrnn_cfg.get("wind_v_icond2_feature", "v_10m")
            for feat in [u_feat, v_feat]:
                if feat not in icond2_features_all:
                    raise ValueError(
                        f"direction_to_adj=True requires '{feat}' in dcrnn.icond2_features "
                        f"(not found in icond2_features_all={icond2_features_all}). "
                        f"Override with wind_u_icond2_feature / wind_v_icond2_feature."
                    )
                if feat not in features_to_load:
                    features_to_load.append(feat)
            uv_load_indices = [features_to_load.index(u_feat), features_to_load.index(v_feat)]
            extra = [f for f in [u_feat, v_feat] if f not in icond2_features]
            if extra:
                logger.info(
                    "direction_to_adj: loading %s separately for decoder wind direction "
                    "(NOT model input features)", extra,
                )
    else:
        # Solar and other use cases: features are direct column / derived names,
        # mode filtering is wind-specific and not applied.
        icond2_features  = list(icond2_features_all)
        ecmwf_features   = list(ecmwf_features_all)
        features_to_load = list(icond2_features)
        uv_load_indices  = None
    measurement_cols = dcrnn_cfg.get("measurement_features")
    target_col       = dcrnn_cfg.get("target_col")
    if target_col not in measurement_cols:
        raise ValueError(f"target_col '{target_col}' must be in measurement_features")

    run_hours     = tuple(dcrnn_cfg.get("icond2_run_hours", [6, 9, 12, 15]))
    next_n_icond2 = dcrnn_cfg.get("next_n_icond2")
    n_workers     = dcrnn_cfg.get("n_workers", 8)
    nwp_path      = data_cfg.get("nwp_path")
    data_path     = data_cfg["path"]

    H   = dcrnn_cfg.get("history_length")
    F_h = dcrnn_cfg.get("forecast_horizon")

    freq   = data_cfg.get("freq", "1h")
    _freq_h_map = {"1h": 1.0, "1H": 1.0, "30min": 0.5, "30T": 0.5, "15min": 0.25, "15T": 0.25}
    freq_h = _freq_h_map.get(freq, 1.0)

    # ------------------------------------------------------------------
    # Station measurements  (T, N_all, M)  — identical to stgnn2
    # ------------------------------------------------------------------
    test_start = data_cfg.get("test_start")
    test_end   = data_cfg.get("test_end")
    run_cutoff = pd.Timestamp(test_end, tz="UTC") if test_end else None

    logger.info("Loading station measurements …")
    meas_raw, timestamps = load_station_measurements(
        data_path, all_ids, cols=measurement_cols, freq=freq
    )

    # Truncate to test_end + 2 days before imputation so NaN counts reflect
    # only the relevant window and imputation doesn't waste time on unused data.
    if run_cutoff is not None:
        meas_cutoff = run_cutoff + pd.Timedelta(days=2)
        cut_idx = int(np.searchsorted(timestamps, meas_cutoff, side="right"))
        meas_raw   = meas_raw[:cut_idx]
        timestamps = timestamps[:cut_idx]

    T = len(timestamps)
    logger.info("Timestamps: %d  (%s … %s)", T, timestamps[0], timestamps[-1])

    interpolate_history = dcrnn_cfg.get("interpolate_history", False)

    # Wind speed imputation via regression-kriging predictions
    rk_pred = None   # kept for optional Kriging lag feature below
    interpol_path = data_cfg.get("interpol_path")
    if interpol_path:
        logger.info("Loading interpolation (rk_pred) for imputation from %s …", interpol_path)
        rk_pred = load_interpol_imputation(interpol_path, all_ids, timestamps)  # noqa: kept for Kriging lag feature
        nan_before = int(np.isnan(meas_raw[:, :, measurement_cols.index(target_col)]).sum())
        meas_raw = apply_interpol_imputation(meas_raw, rk_pred, measurement_cols, target_col)
        nan_after = int(np.isnan(meas_raw[:, :, measurement_cols.index(target_col)]).sum())
        logger.info("Imputation: %d NaN → %d NaN in '%s'", nan_before, nan_after, target_col)

        # Fallback: fill remaining NaN in target_col via KNN imputation
        remaining_nan = int(np.isnan(meas_raw[:, :, measurement_cols.index(target_col)]).sum())
        if remaining_nan > 0:
            knnimputer_path = data_cfg.get("knnimputer_path")
            if knnimputer_path:
                logger.info(
                    "Kriging left %d NaN in '%s' (e.g. before NWP start time). "
                    "Attempting KNN fallback from %s …",
                    remaining_nan, target_col, knnimputer_path,
                )
                knn_ws = load_knn_imputation(knnimputer_path, target_col, all_ids, timestamps, freq=freq)
                nan_before_knn = int(np.isnan(meas_raw[:, :, measurement_cols.index(target_col)]).sum())
                meas_raw = apply_knn_imputation(meas_raw, knn_ws, measurement_cols, target_col)
                nan_after_knn = int(np.isnan(meas_raw[:, :, measurement_cols.index(target_col)]).sum())
                logger.info(
                    "KNN fallback imputation: %d NaN → %d NaN in '%s'",
                    nan_before_knn, nan_after_knn, target_col,
                )

    # Secondary-column KNN imputation (e.g. wind_direction, dhi, …)
    knnimputer_path = data_cfg.get("knnimputer_path")
    if knnimputer_path:
        secondary_cols = [c for c in measurement_cols if c != target_col]
        for sec_col in secondary_cols:
            if int(np.isnan(meas_raw[:, :, measurement_cols.index(sec_col)]).sum()) == 0:
                continue
            knn_sec = load_knn_imputation(knnimputer_path, sec_col, all_ids, timestamps, freq=freq)
            nan_before = int(np.isnan(meas_raw[:, :, measurement_cols.index(sec_col)]).sum())
            meas_raw = apply_knn_imputation(meas_raw, knn_sec, measurement_cols, sec_col)
            nan_after = int(np.isnan(meas_raw[:, :, measurement_cols.index(sec_col)]).sum())
            logger.info("KNN imputation: %d NaN → %d NaN in '%s'", nan_before, nan_after, sec_col)

    _meas_nan_any = np.isnan(meas_raw).any(axis=(1, 2))

    if test_start:
        ts_cutoff = pd.Timestamp(test_start, tz="UTC")
        split_t   = int(np.searchsorted(timestamps, ts_cutoff, side="left"))
    else:
        split_t = int(T * (1 - data_cfg.get("val_frac", 0.2)))
    split_time = timestamps[split_t]

    # NaN audit — restricted to [0, test_end].
    handle_nans = dcrnn_cfg.get("handle_nans", "break")
    audit_t = int(np.searchsorted(timestamps, run_cutoff, side="right")) if run_cutoff else T
    nan_counts = np.isnan(meas_raw[:audit_t]).any(axis=2).sum(axis=0)
    bad_ids = [all_ids[i] for i in np.where(nan_counts > 0)[0]]
    audit_end = timestamps[min(audit_t, T) - 1].date()

    if bad_ids:
        for sid in bad_ids:
            si = all_ids.index(sid)
            nan_mask = np.isnan(meas_raw[:audit_t, si, :]).any(axis=1)
            bad_ts = timestamps[:audit_t][nan_mask]
            cols_affected = [
                measurement_cols[c]
                for c in range(len(measurement_cols))
                if np.isnan(meas_raw[:audit_t, si, c]).any()
            ]
            logger.error(
                "Station %s: %d NaN timestamp(s) in [%s]: %s",
                sid, len(bad_ts), ", ".join(cols_affected),
                ", ".join(str(ts) for ts in bad_ts),
            )

        if handle_nans == "drop":
            bad_set  = set(bad_ids)
            keep_idx = [i for i, sid in enumerate(all_ids) if sid not in bad_set]
            meas_raw = meas_raw[:, keep_idx, :]
            all_ids  = [all_ids[i] for i in keep_idx]
            train_ids = [sid for sid in train_ids if sid not in bad_set]
            val_ids   = [sid for sid in val_ids   if sid not in bad_set]
            N_train   = len(train_ids)
            N_val     = len(val_ids)
            logger.warning(
                "NaN audit (up to %s): dropped %d station(s) with gaps — "
                "continuing with %d train + %d val stations: %s",
                audit_end, len(bad_ids), N_train, N_val, bad_ids,
            )
        else:
            raise ValueError(
                f"NaN audit failed (up to {audit_end}): {len(bad_ids)} station(s) still have "
                f"gaps after imputation — fix source data or set handle_nans: drop: {bad_ids}"
            )
    else:
        logger.info("NaN audit (up to %s): no gaps ✓", audit_end)
    logger.info("Temporal split — train_T: %d  val_T: %d  (split at %s)",
                split_t, T - split_t, split_time.date())

    # Sin/cos encoding of circular measurement features (replaces raw degrees)
    meas_raw, measurement_cols = encode_circular_measurements(meas_raw, measurement_cols)
    if len(measurement_cols) != len(dcrnn_cfg.get("measurement_features", [])):
        logger.info(
            "Circular encoding: measurement_features → %s  (M=%d)",
            measurement_cols, len(measurement_cols),
        )

    # ------------------------------------------------------------------
    # Station metadata
    # ------------------------------------------------------------------
    meta_path = data_cfg.get("stations_master")
    lats, lons, alts = load_station_metadata(data_path, all_ids, meta_path=meta_path)
    station_coords = np.stack([lats, lons], axis=1)

    # ------------------------------------------------------------------
    # NWP runs  →  (R, 48, N_grid, I2)
    # ------------------------------------------------------------------
    if use_case == "solar":
        from geostatistics.solar_preprocessing import load_solar_sl_runs
        logger.info("Loading ICON-D2 SL runs (solar, hours %s) …", list(run_hours))
        run_times, icond2_coords, grid_icond2_runs, station_nearest_grid = \
            load_solar_sl_runs(
                nwp_path=nwp_path,
                station_ids=all_ids,
                station_coords=station_coords,
                features=icond2_features,
                run_hours=run_hours,
                next_n_grid=next_n_icond2,
                n_workers=n_workers,
                cutoff=run_cutoff,
                freq_h=freq_h,
            )
    else:
        logger.info("Loading ICON-D2 ML runs (hours %s) …", list(run_hours))
        run_times, icond2_coords, grid_icond2_runs, station_nearest_grid = \
            load_icond2_ml_runs(
                nwp_path=nwp_path,
                station_ids=all_ids,
                station_coords=station_coords,
                features=features_to_load,
                run_hours=run_hours,
                next_n_grid=next_n_icond2,
                n_workers=n_workers,
                cutoff=run_cutoff,
            )
    R = len(run_times)
    I2 = len(icond2_features)  # model features only (excludes extra u/v if any)
    N_igrid = len(icond2_coords)

    # Split: model features and raw u/v for directional edge weights
    grid_icond2_uv_runs = None
    if uv_load_indices is not None:
        grid_icond2_uv_runs = grid_icond2_runs[:, :, :, uv_load_indices]  # (R, 48, N_grid, 2) raw
    if grid_icond2_runs.shape[3] > I2:
        grid_icond2_runs = grid_icond2_runs[:, :, :, :I2]  # drop extra u/v columns

    if i2_mode == "dir_in_deg":
        grid_icond2_runs, icond2_features = apply_dir_encoding(grid_icond2_runs, icond2_features)
        I2 = len(icond2_features)
        logger.info("dir_in_deg (ICON-D2): encoded → %s  (I2=%d)", icond2_features, I2)

    logger.info("ICON-D2 grid nodes: %d  runs: %d", N_igrid, R)

    # ------------------------------------------------------------------
    # ECMWF NWP
    # ------------------------------------------------------------------
    next_n_ecmwf = dcrnn_cfg.get("next_n_ecmwf", 4)
    db_url = os.environ.get("ECMWF_WIND_SL_URL")

    if next_n_ecmwf == 0:
        logger.info("next_n_ecmwf=0 — ECMWF nodes disabled, skipping ECMWF loading")
        station_ecmwf_nwp = np.empty((T, len(all_ids), 0), dtype=np.float32)
        ecmwf_coords      = np.empty((0, 2), dtype=np.float32)
        ecmwf_nwp         = np.empty((T, 0, 0), dtype=np.float32)
        ecmwf_alts        = np.empty(0, dtype=np.float32)
    else:
        ecmwf_parquet_file = data_cfg.get("ecmwf_path")

        if os.path.exists(ecmwf_parquet_file):
            station_ecmwf_nwp, ecmwf_coords, ecmwf_nwp, ecmwf_alts = \
                load_ecmwf_parquet_at_stations_and_grid(
                    parquet_path=ecmwf_parquet_file,
                    station_lats=lats, station_lons=lons,
                    features=ecmwf_features, timestamps=timestamps,
                    next_n_grid_per_station=next_n_ecmwf,
                )
            logger.info("ECMWF grid nodes: %d", len(ecmwf_coords))

            ecmwf_nan_station = int(np.isnan(station_ecmwf_nwp).sum())
            ecmwf_nan_grid    = int(np.isnan(ecmwf_nwp).sum())
            if ecmwf_nan_station > 0 or ecmwf_nan_grid > 0:
                raise ValueError(
                    f"ECMWF data contains NaN after loading — "
                    f"station array: {ecmwf_nan_station} NaN, "
                    f"grid array: {ecmwf_nan_grid} NaN. "
                    f"Fix the ECMWF data pipeline before training."
                )
        else:
            logger.warning(f"Parquet file {ecmwf_parquet_file} not found. Using fallbacks...")
            # if db_url:
            #     logger.info("Loading ECMWF NWP from database …")
            #     station_ecmwf_nwp, ecmwf_coords, ecmwf_nwp, ecmwf_alts = \
            #         load_ecmwf_at_stations_and_grid(
            #             db_url=db_url,
            #             station_lats=lats, station_lons=lons,
            #             features=ecmwf_features, timestamps=timestamps,
            #             next_n_grid_per_station=next_n_ecmwf,
            #         )
            #     logger.info("ECMWF grid nodes: %d", len(ecmwf_coords))
            # else:
            logger.warning("ECMWF_WIND_SL_URL not set — using zero ECMWF features")
            station_ecmwf_nwp = np.zeros(
                (T, len(all_ids), len(ecmwf_features)), dtype=np.float32
            )
            ec_lats = np.arange(47.5, 55.0, 0.5)
            ec_lons = np.arange(6.0, 15.5, 0.5)
            eg, lg  = np.meshgrid(ec_lats, ec_lons)
            ecmwf_coords = np.stack([eg.ravel(), lg.ravel()], axis=1).astype(np.float32)
            ecmwf_nwp    = np.zeros((T, len(ecmwf_coords), len(ecmwf_features)), dtype=np.float32)
            ecmwf_alts   = np.zeros(len(ecmwf_coords), dtype=np.float32)

    # dir_in_deg ECMWF encoding (applied after loading, before scaling)
    if e2_mode == "dir_in_deg" and station_ecmwf_nwp.shape[2] > 0:
        ecmwf_features_pre = list(ecmwf_features)
        station_ecmwf_nwp, ecmwf_features = apply_dir_encoding(station_ecmwf_nwp, ecmwf_features_pre)
        ecmwf_nwp, _                       = apply_dir_encoding(ecmwf_nwp, ecmwf_features_pre)
        logger.info("dir_in_deg (ECMWF): encoded → %s  (E2=%d)", ecmwf_features, len(ecmwf_features))

    # ------------------------------------------------------------------
    # NWP elevations
    # ------------------------------------------------------------------
    weather_db_url = os.environ.get("WEATHER_DB_URL")
    if dcrnn_cfg.get("use_altitude_diff", False):
        if weather_db_url and db_url and next_n_ecmwf > 0:
            icond2_alts, ecmwf_alts = load_nwp_elevations(
                weather_db_url=weather_db_url,
                ecmwf_db_url=db_url,
                icond2_coords=icond2_coords,
                ecmwf_coords=ecmwf_coords,
            )
            # Offset grid elevations by +10.0m because grid coords are ground-level and NWP features are 10m wind
            icond2_alts += 10.0
            ecmwf_alts += 10.0
        elif weather_db_url:
            from geostatistics.train_stgnn2 import _load_elevations_from_table
            icond2_alts = _load_elevations_from_table(
                weather_db_url, "icon_d2_grid_points", icond2_coords
            ) + 10.0
        else:
            logger.warning("DB URL missing — NWP altitudes set to 0 m")
            icond2_alts = np.zeros(N_igrid, dtype=np.float32)
    else:
        icond2_alts = np.zeros(N_igrid, dtype=np.float32)

    # ------------------------------------------------------------------
    # Scalers
    # ------------------------------------------------------------------
    M_meas = len(measurement_cols)

    meas_scaler = StandardScaler()
    meas_scaler.fit(meas_raw[:split_t, :N_train].reshape(-1, M_meas))
    meas_scaled = meas_scaler.transform(
        meas_raw.reshape(-1, M_meas)
    ).reshape(T, len(all_ids), M_meas)

    # Kriging lag feature: scale rk_pred with the target column's mean/std from meas_scaler
    interpol_meas_scaled = None
    if interpolate_history:
        if rk_pred is None:
            raise ValueError(
                "dcrnn.interpolate_history: true requires data.interpol_path to be set "
                "and the Kriging parquet files to be present."
            )
        tidx = measurement_cols.index(target_col)
        rk_s = (rk_pred - meas_scaler.mean_[tidx]) / (meas_scaler.std_[tidx] + meas_scaler.eps)
        interpol_meas_scaled = np.nan_to_num(rk_s, nan=0.0).astype(np.float32)
        logger.info(
            "Kriging lag feature (interpolate_history=True): shape=%s, NaN→0 filled",
            interpol_meas_scaled.shape,
        )

    train_r_mask = run_times < split_time
    i2_scaler = StandardScaler()
    i2_scaler.fit(grid_icond2_runs[train_r_mask].reshape(-1, I2))
    n_leads = grid_icond2_runs.shape[1]
    grid_icond2_runs_scaled = i2_scaler.transform(
        grid_icond2_runs.reshape(-1, I2)
    ).reshape(R, n_leads, N_igrid, I2)

    E2 = station_ecmwf_nwp.shape[2]   # 0 when next_n_ecmwf == 0
    if E2 > 0:
        e2_scaler = StandardScaler()
        e2_scaler.fit(station_ecmwf_nwp[:split_t, :N_train].reshape(-1, E2))
        station_ecmwf_scaled = e2_scaler.transform(
            station_ecmwf_nwp.reshape(-1, E2)
        ).reshape(T, len(all_ids), E2)
        ecmwf_nwp_scaled = e2_scaler.transform(
            ecmwf_nwp.reshape(-1, E2)
        ).reshape(T, len(ecmwf_coords), E2)
    else:
        station_ecmwf_scaled = np.empty((T, len(all_ids),  0), dtype=np.float32)
        ecmwf_nwp_scaled     = np.empty((T, 0,             0), dtype=np.float32)

    stat_scaler = StandardScaler()
    raw_static  = np.stack([lats, lons, alts], axis=1)
    stat_scaler.fit(raw_static[:N_train])
    station_static_scaled = stat_scaler.transform(raw_static)

    icond2_static_scaled = StandardScaler().fit_transform(
        np.concatenate([icond2_coords, icond2_alts[:, None]], axis=1)
    ).astype(np.float32)
    if len(ecmwf_coords) > 0:
        ecmwf_static_scaled = StandardScaler().fit_transform(
            np.concatenate([ecmwf_coords, ecmwf_alts[:, None]], axis=1)
        ).astype(np.float32)
    else:
        ecmwf_static_scaled = np.empty((0, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # Run pairs
    # ------------------------------------------------------------------
    ts_lookup = pd.Series(np.arange(T), index=timestamps)

    train_run_pairs: list[tuple[int, int, int]] = []
    val_run_pairs:   list[tuple[int, int, int]] = []
    skipped = 0

    for r_curr in range(R):
        t_run = run_times[r_curr]
        if t_run not in ts_lookup.index:
            skipped += 1; continue
        t_run_abs = int(ts_lookup[t_run])
        if t_run_abs < H or t_run_abs + F_h > T:
            skipped += 1; continue

        t_hist_target = t_run - pd.Timedelta(hours=H)
        diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
        r_hist  = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600:
            skipped += 1; continue
        if _meas_nan_any[t_run_abs - H : t_run_abs + F_h].any():
            skipped += 1; continue

        pair = (r_curr, r_hist, t_run_abs)
        (train_run_pairs if t_run < split_time else val_run_pairs).append(pair)

    logger.info(
        "Run pairs — train: %d  val: %d  skipped: %d",
        len(train_run_pairs), len(val_run_pairs), skipped,
    )

    # ------------------------------------------------------------------
    # DCRNN config
    # ------------------------------------------------------------------
    nwp_injection = dcrnn_cfg.get("nwp_injection", True)
    if not nwp_injection:
        logger.info(
            "nwp_injection=false — NWPAttentionLayer disabled. "
            "NWP remains available as nearest-grid station features in x_station."
        )

    model_cfg = DCRNNConfig.from_yaml(
        dcrnn_cfg,
        icond2_features=icond2_features,
        ecmwf_features=ecmwf_features,
        measurement_features=measurement_cols,
        target_col=target_col,
        n_train=N_train,
        n_val=N_val,
        checkpoint_path=str(model_path),
    )

    # ------------------------------------------------------------------
    # k-nearest grid indices for nwp_nodes=False
    # ------------------------------------------------------------------
    _cfg_nwp_nodes = dcrnn_cfg.get("nwp_nodes", True)
    if _cfg_nwp_nodes:
        station_k_nearest_grid = None
    else:
        from sklearn.neighbors import BallTree as _BallTree
        _k = model_cfg.graph.next_n_icond2_grid_points
        _bt = _BallTree(np.radians(icond2_coords), metric="haversine")
        station_k_nearest_grid = _bt.query(
            np.radians(station_coords), k=_k, return_distance=False,
        ).astype(np.int64)  # (N_stations, k)
        logger.info(
            "nwp_nodes=False — station_k_nearest_grid: %s  (k=%d)",
            station_k_nearest_grid.shape, _k,
        )

    # ------------------------------------------------------------------
    # Graph  (shared builder and sampler from stgnn)
    # ------------------------------------------------------------------
    builder    = HeterogeneousGraphBuilder(model_cfg.graph)
    base_graph = builder.build(
        station_coords=station_coords,
        station_altitudes=alts,
        icond2_grid_coords=icond2_coords,
        ecmwf_grid_coords=ecmwf_coords,
        icond2_altitudes=icond2_alts,
        ecmwf_altitudes=ecmwf_alts,
    )
    logger.info(
        "Graph — s2s: %d  i2s: %d  e2s: %d",
        base_graph["station", "near", "station"].edge_index.shape[1] // 2,
        base_graph["icond2", "informs", "station"].edge_index.shape[1],
        base_graph["ecmwf",  "informs", "station"].edge_index.shape[1],
    )

    # ------------------------------------------------------------------
    # Sampler  — reused unchanged from stgnn
    # The sampler reads model_cfg.temporal_encoding to decide output shape.
    # DCRNNConfig exposes .training, .history_length, .forecast_horizon,
    # .temporal_encoding — exactly what TrainingSampler needs.
    # ------------------------------------------------------------------
    target_feat_idx = model_cfg.target_feat_idx
    sampler = TrainingSampler(
        model_cfg, builder, base_graph,
        target_feat_idx=target_feat_idx,
        station_coords=station_coords,
    )

    # ------------------------------------------------------------------
    # Model + Trainer
    # ------------------------------------------------------------------
    model = DCRNN(model_cfg)
    logger.info("DCRNN parameters: %s", f"{model.count_parameters():,}")

    tb_dir = Path("runs") / Path(model_name).stem
    writer = SummaryWriter(log_dir=str(tb_dir)) if SummaryWriter is not None else None
    if writer is not None:
        logger.info("TensorBoard log dir: %s", tb_dir)

    target_scale = float(meas_scaler.std_[target_feat_idx] + meas_scaler.eps)
    target_mean  = float(meas_scaler.mean_[target_feat_idx])

    trainer = DCRNNTrainer(
        model=model,
        sampler=sampler,
        config=model_cfg,
        device=device,
        teacher_forcing_start=dcrnn_cfg.get("teacher_forcing_ratio", 1.0),
        teacher_forcing_end=0.0,
        writer=writer,
        target_scale=target_scale,
        target_mean=target_mean,
    )

    train_station_indices = list(range(N_train))
    val_station_indices   = list(range(N_train, N_train + N_val))

    logger.info("Hyperparameters:\n%s", yaml.dump(dcrnn_cfg, default_flow_style=False))
    fit_result = trainer.fit(
        station_meas=meas_scaled,
        station_nearest_grid=station_nearest_grid,
        grid_icond2_runs=grid_icond2_runs_scaled,
        station_ecmwf_nwp=station_ecmwf_scaled,
        station_static=station_static_scaled,
        ecmwf_nwp=ecmwf_nwp_scaled,
        icond2_static=icond2_static_scaled,
        ecmwf_static=ecmwf_static_scaled,
        train_run_pairs=train_run_pairs,
        val_run_pairs=val_run_pairs,
        train_station_indices=train_station_indices,
        val_station_indices=val_station_indices,
        interpol_meas=interpol_meas_scaled,
        grid_icond2_uv_runs=grid_icond2_uv_runs,
        station_k_nearest_grid=station_k_nearest_grid,
    )

    if writer is not None:
        writer.close()
    logger.info("Training complete. Best model: %s", model_path)

    # ------------------------------------------------------------------
    # Optional post-training evaluation
    # ------------------------------------------------------------------
    eval_df: pd.DataFrame | None = None
    if args.eval:
        logger.info("Running per-station evaluation on val run pairs …")
        # Load best checkpoint
        ckpt = torch.load(model_path, map_location=device)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(sd)
        ws_feat_idx_i2 = find_ws_feat_idx(icond2_features)
        eval_df = run_evaluation(
            model=model,
            sampler=sampler,
            device=device,
            meas_raw=meas_raw,
            meas_scaled=meas_scaled,
            station_nearest_grid=station_nearest_grid,
            grid_icond2_runs_raw=grid_icond2_runs,
            grid_icond2_runs_scaled=grid_icond2_runs_scaled,
            station_ecmwf_nwp_scaled=station_ecmwf_scaled,
            station_static=station_static_scaled,
            ecmwf_nwp_scaled=ecmwf_nwp_scaled,
            icond2_static=icond2_static_scaled,
            ecmwf_static=ecmwf_static_scaled,
            meas_scaler=meas_scaler,
            target_feat_idx=target_feat_idx,
            ws_feat_idx_i2=ws_feat_idx_i2,
            H_hist=H,
            H_fore=F_h,
            train_station_indices=train_station_indices,
            val_station_indices=val_station_indices,
            all_ids=all_ids,
            test_run_pairs=val_run_pairs,
            interpol_meas=interpol_meas_scaled,
        )
        cols = ["mae", "rmse", "r2", "skill", "skill_nwp"]
        tbl  = eval_df.set_index("station_id")[cols]
        mean_row = tbl.mean().to_frame().T
        mean_row.index = ["MEAN"]
        tbl = pd.concat([tbl, mean_row])
        logger.info("Per-station evaluation:\n%s", tbl.to_string(float_format="%.4f"))

    # ------------------------------------------------------------------
    # Save training pkl
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_stem = Path(model_name).stem   # e.g. "wind_stgcn_dcrnn_v1"
    pkl_path = Path("results") / f"{pkl_stem}_{timestamp}.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    pkl_data = {
        "model_name":         pkl_stem,
        "model_path":         str(model_path),
        "architecture":       "dcrnn",
        "mode":               "test" if args.test_mode else "dev",
        "config":             dcrnn_cfg,
        "train_ids":          train_ids,
        "val_ids":            val_ids,
        "n_train":            N_train,
        "n_val":              N_val,
        "history":            fit_result["history"],
        "best_val_rmse":      fit_result["best_val_rmse"],
        "stopped_epoch":      fit_result["stopped_epoch"],
        "evaluation":         eval_df,
        "hpo_study_name":     hpo_study_name,
        "hpo_best_params":    hpo_best_params,
        "hpo_best_val_loss":  hpo_best_val_loss,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)
    logger.info("Training results saved → %s", pkl_path)


if __name__ == "__main__":
    main()
