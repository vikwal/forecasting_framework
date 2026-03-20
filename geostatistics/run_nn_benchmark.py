#!/usr/bin/env python3
"""Neural-network benchmark for spatial wind-speed interpolation.

Trains a simple MLP that maps k-nearest-neighbour wind speeds (plus their
distances and altitude differences relative to the target station) to the
wind speed at that target station.  Architecture and data split are kept
minimal so the result serves as a fair data-driven baseline alongside the
geostatistics methods (IDW / OK / RK).

Data split
----------
Training : all data **before** ``data.test_start``
Test     : ``data.test_start`` – ``data.test_end`` (LOO structure identical
           to the geostatistics pipeline — k neighbours, target excluded)

The trained model is applied to the test period: at every timestamp it
receives the k neighbours' *current* wind speeds as input and predicts the
wind speed at the held-out target station.

Usage (from forecasting_framework/):
    python geostatistics/run_nn_benchmark.py \\
        --config configs/config_spatial_interpolation.yaml
"""

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.interpolation import compute_distance_matrix, get_k_nearest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpatialMLP(nn.Module):
    """MLP that maps k-NN inputs to a scalar wind-speed prediction.

    Input features per sample:
        For each of the k neighbours (ordered by ascending distance):
            [wind_speed, normalised_distance, altitude_diff_km]
        Plus the target station's altitude in km.
    Total input dimension: k * 3 + 1.
    """

    def __init__(self, input_dim: int, hidden_dims: tuple = (128, 64, 32)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_all_data(config: dict) -> tuple:
    """Load metadata + complete wind-speed time series (no time filter).

    Returns:
        pivot:       DataFrame (T_all × N) — all available timestamps.
        lats:        (N,) latitudes.
        lons:        (N,) longitudes.
        alts:        (N,) altitudes in metres.
        station_ids: List[str] of station IDs.
        test_mask:   Boolean Series aligned with pivot.index — True = test.
    """
    data_path = config["data"]["path"]
    station_ids = [str(s) for s in config["data"]["files"]]

    meta_path = os.path.join(data_path, "wind_parameter.csv")
    meta = pd.read_csv(meta_path, sep=";", dtype={"park_id": str})
    meta = meta.set_index("park_id")

    lats = meta.loc[station_ids, "latitude"].values.astype(np.float64)
    lons = meta.loc[station_ids, "longitude"].values.astype(np.float64)
    alts = meta.loc[station_ids, "altitude"].values.astype(np.float64)

    all_dfs = []
    for sid in station_ids:
        fpath = os.path.join(data_path, f"synth_{sid}.csv")
        df = pd.read_csv(fpath, sep=";", parse_dates=["timestamp"])
        df["station_id"] = sid
        all_dfs.append(df[["station_id", "timestamp", "wind_speed"]])

    combined = pd.concat(all_dfs, ignore_index=True)
    if combined["timestamp"].dt.tz is None:
        combined["timestamp"] = combined["timestamp"].dt.tz_localize("UTC")

    pivot = combined.pivot_table(
        index="timestamp", columns="station_id", values="wind_speed", aggfunc="first"
    )
    pivot = pivot.reindex(columns=station_ids).sort_index()

    test_start = config["data"].get("test_start")
    test_end = config["data"].get("test_end")
    test_mask = pd.Series(False, index=pivot.index)
    if test_start:
        test_mask |= pivot.index >= pd.Timestamp(test_start, tz="UTC")
    if test_end:
        test_mask &= pivot.index <= pd.Timestamp(test_end, tz="UTC")
    # If neither bound given, all data is test — warn
    if not test_start and not test_end:
        logger.warning("No test_start/test_end set — using all data as test, no training.")
        test_mask[:] = True

    n_train = int((~test_mask).sum())
    n_test = int(test_mask.sum())
    logger.info("Timestamps  total=%d  train=%d  test=%d", len(pivot), n_train, n_test)
    return pivot, lats, lons, alts, station_ids, test_mask


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_samples(
    values_matrix: np.ndarray,
    neighbor_sets: dict,
    dist_matrix: np.ndarray,
    alts: np.ndarray,
) -> tuple:
    """Vectorise all valid (timestamp, station) pairs into (X, y, meta).

    Input vector per sample:
        [ws_n0, dist_n0, alt_diff_n0,  ws_n1, dist_n1, alt_diff_n1, ...,  alt_target]

    Distances are normalised by the maximum pairwise distance; altitude
    differences by 1000 m so all features are on a similar scale before the
    optional StandardScaler step.

    Args:
        values_matrix: (T, N) wind speeds for the relevant time slice.
        neighbor_sets: {s → (k,) neighbour indices} pre-computed from dist_matrix.
        dist_matrix:   (N, N) geodesic distances in km.
        alts:          (N,) altitudes in metres.

    Returns:
        X:    (M, input_dim) float32 feature matrix.
        y:    (M,) float32 target wind speeds.
        meta: list of (t_local, s_idx) tuples — used to recover timestamps.
    """
    T, N = values_matrix.shape
    max_dist = dist_matrix[dist_matrix > 0].max()

    X_rows, y_rows, meta = [], [], []

    for t in range(T):
        vals_t = values_matrix[t]
        if np.all(np.isnan(vals_t)):
            continue
        for s in range(N):
            obs = vals_t[s]
            if np.isnan(obs):
                continue
            n_idxs = neighbor_sets[s]
            n_vals = vals_t[n_idxs]
            if np.any(np.isnan(n_vals)):
                continue

            n_dists    = dist_matrix[s, n_idxs] / max_dist
            n_alt_diff = (alts[n_idxs] - alts[s]) / 1000.0

            row = np.concatenate([
                np.stack([n_vals, n_dists, n_alt_diff], axis=1).ravel(),
                [alts[s] / 1000.0],
            ])
            X_rows.append(row)
            y_rows.append(obs)
            meta.append((t, s))

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.float32),
        meta,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_dims: tuple = (128, 64, 32),
    lr: float = 1e-3,
    batch_size: int = 512,
    max_epochs: int = 50,
    patience: int = 5,
    val_fraction: float = 0.1,
    device: str = "cpu",
) -> SpatialMLP:
    """Train an MLP with early stopping on a held-out validation split."""
    input_dim = X_train.shape[1]
    model = SpatialMLP(input_dim, hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_val = max(1, int(len(X_train) * val_fraction))
    perm = np.random.permutation(len(X_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr  = torch.tensor(X_train[tr_idx]).to(device)
    y_tr  = torch.tensor(y_train[tr_idx]).to(device)
    X_val = torch.tensor(X_train[val_idx]).to(device)
    y_val = torch.tensor(y_train[val_idx]).to(device)

    dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for Xb, yb in dl:
            optimizer.zero_grad()
            loss_fn(model(Xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        logger.info("  Epoch %3d/%d  val_loss=%.4f  best=%.4f",
                    epoch, max_epochs, val_loss, best_val)

        if no_improve >= patience:
            logger.info("  Early stopping after epoch %d.", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLP benchmark for spatial wind-speed interpolation (LOO-CV)"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"]
    )
    parser.add_argument(
        "--hidden-dims", default="128,64,32",
        help="Comma-separated hidden layer sizes  (default: 128,64,32)",
    )
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--patience", type=int,   default=5)
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device (e.g. 'cpu', 'cuda:0')",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config      = load_config(args.config)
    interp_cfg  = config.get("interpolation", {})
    out_cfg     = config.get("output", {})
    k           = int(interp_cfg.get("k_neighbors", 8))
    output_dir  = out_cfg.get("path", "results/geostatistics")
    prefix      = out_cfg.get("prefix", "spatial_interp")
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data (full time range — no filter yet)
    logger.info("=== Loading data ===")
    pivot, lats, lons, alts, station_ids, test_mask = load_all_data(config)
    values_all    = pivot.values          # (T_all, N)
    timestamps_all = pivot.index
    N = len(station_ids)

    train_idx = np.where(~test_mask.values)[0]
    test_idx  = np.where( test_mask.values)[0]

    if len(train_idx) == 0:
        raise ValueError(
            "No training data found before test_start. "
            "Ensure the CSVs cover a period before data.test_start."
        )

    # 2. Distance matrix + k-NN neighbour sets
    logger.info("=== Computing %d×%d distance matrix ===", N, N)
    dist_matrix  = compute_distance_matrix(lats, lons)
    neighbor_sets = {s: get_k_nearest(dist_matrix, s, k) for s in range(N)}

    # 3. Build feature matrices
    logger.info("=== Building training samples ===")
    X_train, y_train, _ = build_samples(
        values_all[train_idx], neighbor_sets, dist_matrix, alts
    )
    logger.info("Training samples: %d  input_dim: %d", len(X_train), X_train.shape[1])

    logger.info("=== Building test samples ===")
    X_test, y_test, meta_test = build_samples(
        values_all[test_idx], neighbor_sets, dist_matrix, alts
    )
    logger.info("Test samples: %d", len(X_test))

    # 4. Normalise features (fit scaler on training data only)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 5. Train
    logger.info("=== Training MLP %s  lr=%.4f  epochs=%d ===",
                hidden_dims, args.lr, args.epochs)
    model = train_model(
        X_train, y_train,
        hidden_dims=hidden_dims,
        lr=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        device=args.device,
    )
    model.eval()

    # 6. Predict on test set
    logger.info("=== Predicting on test period ===")
    with torch.no_grad():
        preds = (
            model(torch.tensor(X_test, dtype=torch.float32).to(args.device))
            .cpu().numpy()
        )

    # 7. Reconstruct DataFrame (same format as geostatistics pipeline)
    records = []
    for i, (t_local, s_idx) in enumerate(meta_test):
        records.append({
            "station_id":         station_ids[s_idx],
            "timestamp":          timestamps_all[test_idx[t_local]],
            "wind_speed_observed": float(y_test[i]),
            "nn_pred":            float(preds[i]),
        })
    predictions = pd.DataFrame(records)

    pred_path = os.path.join(output_dir, f"{prefix}_nn_predictions.csv")
    predictions.to_csv(pred_path, index=False)
    logger.info("Saved predictions → %s", pred_path)

    # 8. Metrics — per station + global summary
    rows = []
    for sid, grp in predictions.groupby("station_id"):
        obs  = grp["wind_speed_observed"].values
        pred = grp["nn_pred"].values
        rows.append({
            "station_id": sid,
            "method":     "nn",
            "rmse": float(np.sqrt(mean_squared_error(obs, pred))),
            "mae":  float(mean_absolute_error(obs, pred)),
            "r2":   float(r2_score(obs, pred)),
        })
    per_station_df = pd.DataFrame(rows)
    summary = {
        "method": "nn",
        "rmse":   per_station_df["rmse"].mean(),
        "mae":    per_station_df["mae"].mean(),
        "r2":     per_station_df["r2"].mean(),
    }

    per_station_df.to_csv(
        os.path.join(output_dir, f"{prefix}_nn_results_per_station.csv"), index=False
    )
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, f"{prefix}_nn_results_summary.csv"), index=False
    )
    logger.info("NN  rmse=%.4f  mae=%.4f  r2=%.4f",
                summary["rmse"], summary["mae"], summary["r2"])

    # 9. Scatter plot
    obs_all  = predictions["wind_speed_observed"].values
    pred_all = predictions["nn_pred"].values
    fig, ax  = plt.subplots(figsize=(6, 6))
    ax.scatter(obs_all, pred_all, alpha=0.15, s=4, color="steelblue", rasterized=True)
    lim = [min(obs_all.min(), pred_all.min()) - 0.5,
           max(obs_all.max(), pred_all.max()) + 0.5]
    ax.plot(lim, lim, "r--", linewidth=1.0, label="1:1 line")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed wind speed (m/s)")
    ax.set_ylabel("NN predicted (m/s)")
    ax.set_title(
        f"Neural Network — Observed vs. Predicted\n"
        f"RMSE={summary['rmse']:.3f}  MAE={summary['mae']:.3f}  R²={summary['r2']:.3f}"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_scatter_nn.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved scatter plot.")

    print("\nNN benchmark summary:")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
