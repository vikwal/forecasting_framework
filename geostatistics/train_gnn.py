#!/usr/bin/env python3
"""GNN-based spatial wind speed interpolation — 80/80 station split.

Workflow:
  1. Split N stations into train (80) and test (80) using a fixed seed.
  2. Train GATv2 on the 80-node training graph (LOO-style masking).
  3. Infer at each test station: insert it as a new masked node into the
     inference graph (80 train stations as context) and predict its wind
     speed for all timestamps.
  4. Save predictions, per-station metrics, and the station split so that
     the downstream TFT pipeline uses exactly the same 80/80 split.

The station split is written to {output_dir}/{prefix}_station_split.csv so
that train_cl.py / the federated pipeline can load the same split.

Usage:
    python geostatistics/train_gnn.py --config configs/config_spatial_interpolation.yaml
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATv2Conv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geostatistics.run_spatial_interpolation import load_config, load_data, load_nwp_wind_speed
from utils.interpolation import compute_distance_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def build_knn_graph(dist_matrix: np.ndarray, k: int) -> torch.Tensor:
    """Directed K-NN: edges FROM neighbour TO node."""
    N = dist_matrix.shape[0]
    src, dst = [], []
    for i in range(N):
        d = dist_matrix[i].copy()
        d[i] = np.inf
        for j in np.argsort(d)[:k]:
            src.append(j)
            dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)


def build_edge_attr(
    dist_matrix: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Edge features: [dist_norm, sin(bearing), cos(bearing)].

    Returns (edge_attr, max_dist) — max_dist needed to normalise new edges.
    """
    s = edge_index[0].numpy()
    d = edge_index[1].numpy()
    dists = dist_matrix[s, d]
    max_dist = float(dists.max()) + 1e-8

    lat1, lat2 = np.radians(lats[s]), np.radians(lats[d])
    dlon = np.radians(lons[d] - lons[s])
    bx = np.sin(dlon) * np.cos(lat2)
    by = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    return (
        torch.tensor(
            np.stack([dists / max_dist, np.sin(np.arctan2(bx, by)),
                      np.cos(np.arctan2(bx, by))], axis=1),
            dtype=torch.float32,
        ),
        max_dist,
    )


def build_test_edges(
    test_lat: float, test_lon: float,
    train_lats: np.ndarray, train_lons: np.ndarray,
    train_dist_to_test: np.ndarray,
    max_dist: float,
    test_node_idx: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K edges FROM the k nearest training nodes TO the test node."""
    k_nn = np.argsort(train_dist_to_test)[:k]
    dists = train_dist_to_test[k_nn]

    lat1 = np.radians(train_lats[k_nn])
    lat2 = np.radians(test_lat)
    dlon = np.radians(test_lon - train_lons[k_nn])
    bx = np.sin(dlon) * np.cos(lat2)
    by = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(bx, by)

    src = torch.tensor(k_nn, dtype=torch.long)
    dst = torch.full((k,), test_node_idx, dtype=torch.long)
    attr = torch.tensor(
        np.stack([dists / max_dist, np.sin(bearing), np.cos(bearing)], axis=1),
        dtype=torch.float32,
    )
    return torch.stack([src, dst]), attr


def make_mega_batch(
    x_batch: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack B identical-topology graphs into one disconnected mega-graph."""
    B, N, F = x_batch.shape
    E = edge_index.shape[1]
    offsets = torch.arange(B, device=edge_index.device).repeat_interleave(E) * N
    return (
        x_batch.reshape(B * N, F),
        edge_index.repeat(1, B) + offsets.unsqueeze(0),
        edge_attr.repeat(B, 1),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpatialDataset(torch.utils.data.Dataset):
    """(timestamp, target_station) pairs for training."""

    def __init__(self, ws: np.ndarray, nwp: np.ndarray, static: np.ndarray,
                 t_idxs: np.ndarray, s_idxs: np.ndarray):
        self.ws = torch.tensor(ws, dtype=torch.float32)
        self.nwp = torch.tensor(nwp, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.float32)
        self.t_idxs = t_idxs
        self.s_idxs = s_idxs

    def __len__(self) -> int:
        return len(self.t_idxs)

    def __getitem__(self, idx):
        t, s = int(self.t_idxs[idx]), int(self.s_idxs[idx])
        N = self.ws.shape[1]
        ws_t = self.ws[t].clone()
        y = ws_t[s].clone()
        ws_t[s] = 0.0
        is_target = torch.zeros(N, 1)
        is_target[s] = 1.0
        x = torch.cat([ws_t.unsqueeze(1), self.nwp[t].unsqueeze(1),
                        self.static, is_target], dim=1)
        return x, y, s


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpatialGNN(nn.Module):
    """Inductive GATv2 for spatial interpolation."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int,
                 heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden // heads, heads=heads,
                      edge_dim=edge_dim, dropout=dropout, concat=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + self.dropout(conv(x, edge_index, edge_attr)))
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training / validation epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, edge_index, edge_attr, device,
              optimizer=None) -> float:
    training = optimizer is not None
    model.train() if training else model.eval()
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    total_loss, n = 0.0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x_batch, y_batch, s_batch in loader:
            B, N, _ = x_batch.shape
            x_batch, y_batch, s_batch = (
                x_batch.to(device), y_batch.to(device), s_batch.to(device))
            bx, bei, bea = make_mega_batch(x_batch, edge_index, edge_attr)
            out = model(bx, bei, bea).view(B, N)
            preds = out[torch.arange(B, device=device), s_batch]
            loss = nn.functional.mse_loss(preds, y_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * B
            n += B

    return total_loss / n


# ---------------------------------------------------------------------------
# Inference at a test station
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_test_station(
    model,
    ws_scaled: np.ndarray,           # (T, N_train)
    nwp_scaled: np.ndarray,          # (T, N_train)
    static_scaled: np.ndarray,       # (N_train, S)
    train_edge_index: torch.Tensor,
    train_edge_attr: torch.Tensor,
    test_static_scaled: np.ndarray,  # (S,)
    test_nwp_scaled: np.ndarray,     # (T,)
    test_lat: float, test_lon: float,
    train_lats: np.ndarray, train_lons: np.ndarray,
    train_dist_to_test: np.ndarray,
    max_dist: float, k: int,
    ws_scaler: StandardScaler,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict one test station for all T timestamps (original scale)."""
    model.eval()
    T, N_tr = ws_scaled.shape
    test_idx = N_tr  # appended as last node

    new_ei, new_ea = build_test_edges(
        test_lat, test_lon, train_lats, train_lons,
        train_dist_to_test, max_dist, test_idx, k,
    )
    eval_ei = torch.cat([train_edge_index, new_ei], dim=1).to(device)
    eval_ea = torch.cat([train_edge_attr, new_ea], dim=0).to(device)
    N_eval = N_tr + 1

    ws_t = torch.tensor(ws_scaled, dtype=torch.float32)
    nwp_t = torch.tensor(nwp_scaled, dtype=torch.float32)
    static_t = torch.tensor(static_scaled, dtype=torch.float32)
    test_nwp_t = torch.tensor(test_nwp_scaled, dtype=torch.float32)
    test_static_t = torch.tensor(test_static_scaled, dtype=torch.float32)

    preds_sc = []
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        B = end - start
        train_x = torch.cat([
            ws_t[start:end].unsqueeze(2),
            nwp_t[start:end].unsqueeze(2),
            static_t.unsqueeze(0).expand(B, -1, -1),
            torch.zeros(B, N_tr, 1),
        ], dim=2)
        test_x = torch.cat([
            torch.zeros(B, 1, 1),
            test_nwp_t[start:end].view(B, 1, 1),
            test_static_t.view(1, 1, -1).expand(B, 1, -1),
            torch.ones(B, 1, 1),
        ], dim=2)
        x_batch = torch.cat([train_x, test_x], dim=1).to(device)
        bx, bei, bea = make_mega_batch(x_batch, eval_ei, eval_ea)
        out = model(bx, bei, bea).view(B, N_eval)
        preds_sc.append(out[:, test_idx].cpu().numpy())

    preds_sc = np.concatenate(preds_sc)
    return ws_scaler.inverse_transform(preds_sc.reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GNN spatial interpolation — 80/80 station split")
    parser.add_argument("--config", required=True)
    parser.add_argument("-s", "--suffix", default="")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    config = load_config(args.config)
    interp_cfg = config.get("interpolation", {})
    gnn_cfg = config.get("gnn", {})
    out_cfg = config.get("output", {})

    output_dir = out_cfg.get("path", "results/geostatistics")
    config_stem = (
        os.path.splitext(os.path.basename(args.config))[0].removeprefix("config_")
    )
    prefix = config_stem + (f"_{args.suffix}" if args.suffix else "") + "_gnn"

    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(logs_dir, f"{prefix}.log"), mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("=== Output prefix: %s | dir: %s ===", prefix, output_dir)

    # ---- Hyperparameters ---------------------------------------------------
    k = int(interp_cfg.get("k_neighbors", 15))
    split_seed = int(gnn_cfg.get("station_split_seed", 42))
    test_fraction = float(gnn_cfg.get("station_test_fraction", 0.5))
    val_fraction = float(gnn_cfg.get("val_fraction", 0.2))
    hidden = int(gnn_cfg.get("hidden", 128))
    heads = int(gnn_cfg.get("heads", 4))
    num_layers = int(gnn_cfg.get("num_layers", 3))
    dropout = float(gnn_cfg.get("dropout", 0.1))
    lr = float(gnn_cfg.get("lr", 1e-3))
    weight_decay = float(gnn_cfg.get("weight_decay", 1e-5))
    batch_size = int(gnn_cfg.get("batch_size", 512))
    max_epochs = int(gnn_cfg.get("max_epochs", 100))
    patience = int(gnn_cfg.get("patience", 15))
    num_workers = int(gnn_cfg.get("num_workers", 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Data loading ------------------------------------------------------
    logger.info("=== Loading data ===")
    (pivot, lats, lons, alts, station_ids,
     u_matrix, v_matrix,
     rk_feature_names, rk_static_features, rk_dynamic_features) = load_data(config)

    timestamps = pivot.index
    ws_full = pivot.values.astype(np.float64)
    T, N = ws_full.shape

    nwp_matrix = rk_dynamic_features.get("nwp_wind_speed")
    if nwp_matrix is None:
        nwp_path = config["data"].get("nwp_path")
        hub_height = float(interp_cfg.get("nwp_hub_height", 10.0))
        if nwp_path:
            logger.info("=== Loading ICON-D2 NWP ===")
            nwp_matrix = load_nwp_wind_speed(
                nwp_path=nwp_path, station_ids=station_ids,
                station_lats=lats, station_lons=lons,
                timestamps=timestamps, hub_height=hub_height,
            )
        else:
            logger.warning("No NWP data — using zeros.")
            nwp_matrix = np.zeros_like(ws_full)

    # ---- Station split -----------------------------------------------------
    rng = np.random.default_rng(split_seed)
    all_idxs = np.arange(N)
    n_test = int(N * test_fraction)
    test_idxs = rng.choice(all_idxs, size=n_test, replace=False)
    test_idxs = np.sort(test_idxs)
    train_idxs = np.setdiff1d(all_idxs, test_idxs)
    N_train = len(train_idxs)

    logger.info("Station split (seed=%d): %d train / %d test",
                split_seed, N_train, len(test_idxs))

    # Save split so TFT pipeline uses the same stations
    split_df = pd.DataFrame({
        "station_id": station_ids,
        "split": ["test" if i in set(test_idxs.tolist()) else "train"
                  for i in range(N)],
    })
    split_path = os.path.join(output_dir, f"{prefix}_station_split.csv")
    split_df.to_csv(split_path, index=False)
    logger.info("Saved station split → %s", split_path)

    # ---- Training data (train stations only) -------------------------------
    lats_tr = lats[train_idxs]
    lons_tr = lons[train_idxs]
    alts_tr = alts[train_idxs]
    ws_tr = ws_full[:, train_idxs]
    nwp_tr = nwp_matrix[:, train_idxs]

    # Temporal train/val split for early stopping
    split_t = int(T * (1 - val_fraction))

    ws_sc = StandardScaler()
    ws_sc.fit(ws_tr[:split_t].reshape(-1, 1))
    ws_scaled = ws_sc.transform(ws_tr.reshape(-1, 1)).reshape(T, N_train)

    nwp_sc = StandardScaler()
    nwp_sc.fit(nwp_tr[:split_t].reshape(-1, 1))
    nwp_scaled = nwp_sc.transform(nwp_tr.reshape(-1, 1)).reshape(T, N_train)

    static_raw = np.stack([alts_tr, lats_tr, lons_tr], axis=1).astype(np.float32)
    static_sc = StandardScaler()
    static_scaled = static_sc.fit_transform(static_raw)

    # ---- Build training graph ----------------------------------------------
    logger.info("=== Building K=%d NN graph on %d training stations ===",
                k, N_train)
    dist_tr = compute_distance_matrix(lats_tr, lons_tr)
    edge_index = build_knn_graph(dist_tr, k)
    edge_attr, max_dist = build_edge_attr(dist_tr, lats_tr, lons_tr, edge_index)
    logger.info("  Nodes: %d  |  Edges: %d", N_train, edge_index.shape[1])

    # ---- DataLoaders -------------------------------------------------------
    ws_t, ws_v = ws_scaled[:split_t], ws_scaled[split_t:]
    nwp_t, nwp_v = nwp_scaled[:split_t], nwp_scaled[split_t:]

    t_t = np.repeat(np.arange(split_t), N_train)
    s_t = np.tile(np.arange(N_train), split_t)
    t_v = np.repeat(np.arange(T - split_t), N_train)
    s_v = np.tile(np.arange(N_train), T - split_t)

    tr_loader = torch.utils.data.DataLoader(
        SpatialDataset(ws_t, nwp_t, static_scaled, t_t, s_t),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        SpatialDataset(ws_v, nwp_v, static_scaled, t_v, s_v),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ---- Model -------------------------------------------------------------
    node_dim = 2 + static_scaled.shape[1] + 1
    model = SpatialGNN(node_dim, edge_attr.shape[1], hidden, heads,
                       num_layers, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d params  node_dim=%d  hidden=%d  heads=%d  layers=%d",
                n_params, node_dim, hidden, heads, num_layers)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=patience // 3, factor=0.5, min_lr=1e-6)

    # ---- Training loop -----------------------------------------------------
    logger.info("=== Training (train_T=%d  val_T=%d) ===", split_t, T - split_t)
    best_val, best_state, no_imp = np.inf, None, 0

    for epoch in range(1, max_epochs + 1):
        tl = run_epoch(model, tr_loader, edge_index, edge_attr, device, opt)
        vl = run_epoch(model, val_loader, edge_index, edge_attr, device)
        sched.step(vl)
        logger.info("Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e",
                    epoch, max_epochs, tl, vl, opt.param_groups[0]["lr"])

        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    logger.info("Best val loss: %.4f", best_val)

    # Save model checkpoint
    ckpt_path = os.path.join(output_dir, f"{prefix}_model.pt")
    torch.save({"model_state": best_state, "node_dim": node_dim,
                "edge_dim": edge_attr.shape[1], "hidden": hidden,
                "heads": heads, "num_layers": num_layers, "dropout": dropout},
               ckpt_path)
    logger.info("Saved model checkpoint → %s", ckpt_path)

    # ---- Inference at test stations ----------------------------------------
    logger.info("=== Inference at %d test stations ===", len(test_idxs))

    # Precompute distance matrix including test station lats/lons
    combined_lats = np.append(lats_tr, lats[test_idxs])
    combined_lons = np.append(lons_tr, lons[test_idxs])
    dist_combined = compute_distance_matrix(combined_lats, combined_lons)

    pred_records = []
    per_station_rows = []

    for i, test_idx in enumerate(tqdm(test_idxs, desc="Test stations", unit="station")):
        sid = station_ids[test_idx]

        # Scale test station data with training scalers
        test_nwp_sc = nwp_sc.transform(
            nwp_matrix[:, test_idx].reshape(-1, 1)).flatten()
        test_static_sc = static_sc.transform(
            np.array([[alts[test_idx], lats[test_idx], lons[test_idx]]],
                     dtype=np.float32)).flatten()

        # Distances from each training station to this test station
        train_dist_to_test = dist_combined[:N_train, N_train + i]

        preds = predict_test_station(
            model=model,
            ws_scaled=ws_scaled, nwp_scaled=nwp_scaled,
            static_scaled=static_scaled,
            train_edge_index=edge_index, train_edge_attr=edge_attr,
            test_static_scaled=test_static_sc,
            test_nwp_scaled=test_nwp_sc,
            test_lat=float(lats[test_idx]), test_lon=float(lons[test_idx]),
            train_lats=lats_tr, train_lons=lons_tr,
            train_dist_to_test=train_dist_to_test,
            max_dist=max_dist, k=k, ws_scaler=ws_sc,
            device=device, batch_size=batch_size,
        )

        obs = ws_full[:, test_idx]
        m = ~(np.isnan(obs) | np.isnan(preds))
        rmse = float(np.sqrt(mean_squared_error(obs[m], preds[m])))
        mae = float(mean_absolute_error(obs[m], preds[m]))
        r2 = float(r2_score(obs[m], preds[m]))

        logger.info("  %s  R²=%.4f  RMSE=%.4f", sid, r2, rmse)
        per_station_rows.append({"station_id": sid, "method": "gnn",
                                  "rmse": rmse, "mae": mae, "r2": r2})

        for t_idx in range(T):
            pred_records.append({
                "station_id": sid,
                "timestamp": timestamps[t_idx],
                "wind_speed_observed": float(obs[t_idx]),
                "gnn_pred": float(preds[t_idx]),
            })

    # ---- Save outputs ------------------------------------------------------
    per_station = pd.DataFrame(per_station_rows)
    ps_path = os.path.join(output_dir, f"{prefix}_results_per_station.csv")
    per_station.to_csv(ps_path, index=False)
    logger.info("Saved per-station metrics → %s", ps_path)

    preds_df = pd.DataFrame(pred_records)
    pred_path = os.path.join(output_dir, f"{prefix}_predictions.csv")
    preds_df.to_csv(pred_path, index=False)
    logger.info("Saved predictions → %s", pred_path)

    summary = per_station[["rmse", "mae", "r2"]].mean().to_frame().T
    summary.insert(0, "method", "gnn")
    sm_path = os.path.join(output_dir, f"{prefix}_results_summary.csv")
    summary.to_csv(sm_path, index=False)
    logger.info("Saved summary → %s", sm_path)

    print("\nGNN summary (mean over test stations):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
