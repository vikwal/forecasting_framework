#!/usr/bin/env python3
"""
Evaluate trained STGNN models on test data and save metrics to CSV.

Provides exactly identical CLI arguments to get_test_results.py natively.
It restricts evaluation dynamically only to models with 'stgnn' or 'stgcn' in filename.
"""

import os
import sys
import warnings
import yaml

warnings.filterwarnings('ignore', message='X does not have valid feature names', category=UserWarning)
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import gc
import pickle
import logging
from datetime import datetime
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geostatistics.run_spatial_interpolation import load_config, load_data
from geostatistics.train_stgnn import (
    load_nwp_feature_matrices,
    load_measurement_features,
    load_static_features,
    SpatioTemporalGNN,
    predict_val_stations_with_neighbors,
    build_radius_graph
)
from geostatistics.train_gnn import build_edge_attr
from utils.interpolation import compute_distance_matrix

def setup_logging():
    """Setup logging to file and console."""
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'test_results_stgnn_{timestamp}.log'

    logger = logging.getLogger('test_results_stgnn')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

def extract_model_name(model_name):
    parts = model_name.replace('.pt', '').split('_')
    return '_'.join(parts)

def get_config_path_from_model_name(model_name):
    # Try guessing the config path from the typical stgnn syntax
    base = model_name.replace('_stgnn_model.pt', '').replace('_stgcn_model.pt', '').replace('_model.pt', '').replace('.pt', '')
    if base.startswith("cl_m-stgnn_"):
        base = base.replace("cl_m-stgnn_", "")
    config_name = f"config_{base}"
    return Path(f'configs/{config_name}.yaml')

def evaluate_stgnn_model(model_path, config_path, device, overlap_mode="shortest_lead"):
    logger.info(f"Loading config {config_path}")
    config = load_config(str(config_path))
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    
    node_dim = checkpoint.get("node_dim", 0)
    edge_dim = checkpoint.get("edge_dim", 0)
    hidden = checkpoint.get("hidden", config.get("stgnn", {}).get("hidden", 128))
    heads = checkpoint.get("heads", config.get("stgnn", {}).get("heads", 4))
    num_layers = checkpoint.get("num_layers", config.get("stgnn", {}).get("num_layers", 3))
    temporal_kernel_size = checkpoint.get("temporal_kernel_size", config.get("stgnn", {}).get("temporal_kernel_size", 3))
    dropout = checkpoint.get("dropout", config.get("stgnn", {}).get("dropout", 0.1))
    seq_len = checkpoint.get("seq_len", config.get("stgnn", {}).get("seq_len", 96))
    forecast_horizon = checkpoint.get("forecast_horizon", config.get("stgnn", {}).get("forecast_horizon", 48))
    
    stgnn_cfg = config.get("stgnn", {})
    batch_size = int(stgnn_cfg.get("batch_size", 32))
    val_fraction = float(stgnn_cfg.get("val_fraction", 0.2))
    radius_km = float(stgnn_cfg.get("radius_km", 300.0))
    max_neighbors = stgnn_cfg.get("max_neighbors")
    if max_neighbors is not None:
        max_neighbors = int(max_neighbors)

    _load_cfg = {**config, "interpolation": {**config.get("interpolation", {}), "rk_features": None}}
    logger.info("Loading training data...")
    (pivot, lats, lons, alts, station_ids, _u, _v, _rk_names, _rks, _rkd) = load_data(_load_cfg)
    timestamps = pivot.index
    T, N_train = pivot.shape

    params_cfg = config.get("params", {})
    meas_feats = list(params_cfg.get("measurement_features", ["wind_speed"]))
    nwp_feats = list(params_cfg.get("nwp_features", []))
    stat_feats = list(params_cfg.get("static_features", ["altitude", "latitude", "longitude"]))
    
    data_path = config["data"]["path"]
    meas_raw = load_measurement_features(data_path, station_ids, meas_feats, timestamps)
    nwp_tr_raw = load_nwp_feature_matrices(config, station_ids, nwp_feats, timestamps)

    val_config = {**_load_cfg, "data": {**config["data"], "files": config["data"]["val_files"]}}
    logger.info("Loading test/val data...")
    (val_pivot, val_lats, val_lons, val_alts, val_station_ids, _vu, _vv, _vrk, _vrks, _vrkd) = load_data(val_config)
    val_ws_full = val_pivot.values.astype(np.float64)
    val_timestamps = val_pivot.index
    N_val = len(val_station_ids)
    T_val = val_ws_full.shape[0]

    nwp_val_raw = load_nwp_feature_matrices(config, val_station_ids, nwp_feats, val_timestamps)
    val_meas_raw = load_measurement_features(data_path, val_station_ids, meas_feats, val_timestamps)
    
    split_t = int(T * (1 - val_fraction))

    logger.info("Dynamically fitting scalers based on local dataset config ratios...")
    meas_scalers: dict = {}
    meas_scaled_list: list = []
    meas_valid_list: list = []
    for fname in meas_feats:
        raw = meas_raw[fname].copy()
        valid = ~np.isnan(raw)
        raw[~valid] = 0.0
        sc = StandardScaler()
        sc.fit(raw[:split_t][valid[:split_t]].reshape(-1, 1))
        scaled = sc.transform(raw.reshape(-1, 1)).reshape(T, N_train)
        scaled[~valid] = 0.0
        meas_scalers[fname] = sc
        meas_scaled_list.append(scaled)
        meas_valid_list.append(valid)
        
    ws_sc = meas_scalers[meas_feats[0]]

    nwp_scalers = {}
    nwp_scaled_tr = []
    for fname in nwp_feats:
        raw = nwp_tr_raw[fname].copy()
        valid = ~np.isnan(raw)
        raw[~valid] = 0.0
        sc = StandardScaler()
        sc.fit(raw[:split_t].reshape(-1, 1))
        scaled = sc.transform(raw.reshape(-1, 1)).reshape(T, N_train)
        scaled[~valid] = 0.0
        nwp_scalers[fname] = sc
        nwp_scaled_tr.append(scaled)

    static_raw = load_static_features(data_path, station_ids, stat_feats, lats, lons, alts)
    static_sc = StandardScaler()
    static_scaled = static_sc.fit_transform(static_raw)

    nwp_scaled_val_list = []
    for fname in nwp_feats:
        raw = nwp_val_raw[fname].copy()
        sc_arr = nwp_scalers[fname].transform(raw.reshape(-1, 1)).reshape(T_val, N_val)
        nwp_scaled_val_list.append(sc_arr)

    meas_scaled_val_list = []
    meas_valid_val_list = []
    for fname in meas_feats:
        raw = val_meas_raw[fname].copy()
        valid = ~np.isnan(raw)
        sc_arr = meas_scalers[fname].transform(raw.reshape(-1, 1)).reshape(T_val, N_val)
        meas_scaled_val_list.append(sc_arr)
        meas_valid_val_list.append(valid)

    val_static_raw = load_static_features(data_path, val_station_ids, stat_feats, val_lats, val_lons, val_alts)
    val_static_scaled = static_sc.transform(val_static_raw)

    all_meas_scaled = [np.concatenate([tr, va], axis=1) for tr, va in zip(meas_scaled_list, meas_scaled_val_list)]
    all_meas_valid = [np.concatenate([tr, va], axis=1) for tr, va in zip(meas_valid_list, meas_valid_val_list)]
    all_nwp_scaled = [np.concatenate([tr, va], axis=1) for tr, va in zip(nwp_scaled_tr, nwp_scaled_val_list)]
    all_static = np.concatenate([static_scaled, val_static_scaled], axis=0)

    # Build Graph
    all_lats = np.concatenate([lats, val_lats])
    all_lons = np.concatenate([lons, val_lons])
    dist_full = compute_distance_matrix(all_lats, all_lons)
    full_edge_index = build_radius_graph(dist_full, radius_km, max_neighbors)
    full_edge_attr, _ = build_edge_attr(dist_full, all_lats, all_lons, full_edge_index)

    computed_node_dim = len(meas_feats) + len(nwp_feats) + 1 + static_scaled.shape[1]
    
    model = SpatioTemporalGNN(
        node_dim=node_dim if node_dim > 0 else computed_node_dim,
        edge_dim=edge_dim if edge_dim > 0 else full_edge_attr.shape[1],
        hidden=hidden, heads=heads, num_layers=num_layers,
        temporal_kernel_size=temporal_kernel_size, dropout=dropout,
        seq_len=seq_len, forecast_horizon=forecast_horizon
    ).to(device)
    
    cleaned_state_dict = {k.replace('_orig_mod.', ''): v for k,v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict)
    
    val_indices = list(range(N_train, N_train + N_val))
    
    logger.info("Evaluating Spatial GNN graph via LOO context...")
    preds_all = predict_val_stations_with_neighbors(
        model=model,
        all_meas_scaled=all_meas_scaled,
        all_meas_valid=all_meas_valid,
        all_nwp_scaled=all_nwp_scaled,
        all_static_scaled=all_static,
        full_edge_index=full_edge_index,
        full_edge_attr=full_edge_attr,
        val_indices=val_indices,
        ws_scaler=ws_sc,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
        overlap_mode=overlap_mode
    )

    results = []

    for i, sid in enumerate(val_station_ids):
        preds = preds_all[:, i]
        obs = val_ws_full[:, i]
        m = ~(np.isnan(obs) | np.isnan(preds))
        if m.sum() > 0:
            rmse = float(np.sqrt(mean_squared_error(obs[m], preds[m])))
            mae = float(mean_absolute_error(obs[m], preds[m]))
            r2 = float(r2_score(obs[m], preds[m]))
            
            skill_nwp = np.nan
            if 'wind_speed_h10' in nwp_feats:
                nwp_h10_raw = nwp_val_raw['wind_speed_h10'][:, i]
                m_nwp = ~(np.isnan(obs) | np.isnan(nwp_h10_raw))
                if m_nwp.sum() > 0:
                    rmse_nwp = float(np.sqrt(mean_squared_error(obs[m_nwp], nwp_h10_raw[m_nwp])))
                    # Skill = 1 - (RMSE_model / RMSE_nwp)
                    if rmse_nwp != 0:
                        skill_nwp = 1.0 - (rmse / rmse_nwp)
            
            results.append({
                "station_id": sid,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "skill_nwp": skill_nwp
            })

    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate STGNN models on test data')
    parser.add_argument('-g', '--global', dest='global_models', action='store_true',
                        help='Evaluate global models instead of station-specific models')
    parser.add_argument('-e', '--evallocal', dest='eval_local', action='store_true',
                        help='Evaluate global models on local datasets')
    parser.add_argument('--evaldaily', dest='eval_daily', action='store_true',
                        help='Evaluate global models on local datasets with daily R² per station (CSV format)')
    parser.add_argument('--localperformance', dest='local_performance', action='store_true',
                        help='Evaluate local station-specific models with daily R² per station (CSV format)')
    parser.add_argument('--model', dest='model', default='tft',
                        help='Model type to evaluate: "tft" (default) or "chronos"')
    parser.add_argument('--config', dest='config_path', default=None,
                        help='Path to config YAML (required for --model chronos)')
    parser.add_argument('--zero-shot', dest='zero_shot', action='store_true',
                        help='For Chronos: load from HuggingFace instead of local fine-tuned model')
    parser.add_argument('--model-name', dest='model_name', default=None,
                        help='Filter: only evaluate the global model whose filename contains this string')
    
    args = parser.parse_args()

    # Search domains matching behavior of get_test_results.py + geostatistics results
    gnn_models_dir = Path('results/geostatistics')
    base_models_dir = Path('models')
    output_dir = Path('data/test_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate STGNN / STGCN models
    all_model_files = []
    if gnn_models_dir.exists():
        all_model_files.extend(list(gnn_models_dir.glob('*.pt')))
    if base_models_dir.exists():
        all_model_files.extend(list(base_models_dir.glob('*.pt')))
    
    # Filter explicitly for stgnn or stgcn as requested by user
    model_files = []
    seen = set()
    for f in all_model_files:
        if ('stgnn' in f.name.lower() or 'stgcn' in f.name.lower()) and f.name not in seen:
            model_files.append(f)
            seen.add(f.name)
            
    if args.model_name:
        model_files = [f for f in model_files if args.model_name in f.name]
        
    logger.info(f"Found {len(model_files)} STGNN/STGCN model(s).")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # For STGNN, graph evaluation is holistic across testing stations.
    # Therefore, args mapping defaults to iterating and pushing local-level details per model uniformly.

    for i, model_path in enumerate(model_files):
        model_id = extract_model_name(model_path.name)
        
        logger.info(f"\n[{i+1}/{len(model_files)}] Processing: {model_id}")
        
        # Determine CSV destination
        if args.global_models:
            output_file = output_dir / 'test_results_global.csv'
            identifier_key = 'model'
            identifier_value = model_id
        else:
            output_file = output_dir / f'test_results_{model_id}.csv'
            identifier_key = 'station_id'
            
        if args.config_path and Path(args.config_path).exists():
            config_path = Path(args.config_path)
            logger.info(f"Using explicitly provided config path: {config_path}")
        else:
            config_path = get_config_path_from_model_name(model_path.name)
            if not config_path.exists():
                # try stripping _gpuX
                import re
                base_no_gpu = re.sub(r'_gpu\d+', '', config_path.stem.replace('config_', ''))
                fallback_path = Path(f'configs/config_{base_no_gpu}.yaml')
                if fallback_path.exists():
                    config_path = fallback_path
                else:
                    logger.warning(f"Config not found at {config_path}. Searching in configs/...")
                    search_str = model_id.replace('model_', '')
                    search_str = re.sub(r'_gpu\d+', '', search_str)
                    search_str = search_str.replace('_stgnn', '').replace('_stgcn', '')
                    config_glob = list(Path('configs').glob(f"*{search_str}*.yaml"))
                    if config_glob:
                        config_path = config_glob[0]
                    else:
                        logger.error(f"Cannot deduce config for {model_path.name}, skipping.")
                        continue

        try:
            results = evaluate_stgnn_model(model_path, config_path, device)
            
            if args.global_models:
                if results:
                    df = pd.DataFrame(results)
                    avg_metrics = {
                        identifier_key: identifier_value,
                        'rmse': df['rmse'].mean(),
                        'mae': df['mae'].mean(),
                        'r2': df['r2'].mean(),
                        'skill_nwp': df['skill_nwp'].mean() if not df['skill_nwp'].isna().all() else np.nan
                    }
                    new_df = pd.DataFrame([avg_metrics])
                    if not output_file.exists():
                        new_df.to_csv(output_file, index=False, mode='w')
                    else:
                        new_df.to_csv(output_file, index=False, mode='a', header=False)
                    logger.info(f"Appended global avg results to {output_file}")
            else:
                if results:
                    new_df = pd.DataFrame(results)
                    # Follow get_test_results.py format
                    if not output_file.exists():
                        new_df.to_csv(output_file, index=False, mode='w')
                    else:
                        new_df.to_csv(output_file, index=False, mode='a', header=False)
                    logger.info(f"Appended station-wise results to {output_file}")
                        
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Error evaluating {model_id}: {e}", exc_info=False)


if __name__ == '__main__':
    main()
