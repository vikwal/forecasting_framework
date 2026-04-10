#!/usr/bin/env python3
"""Spatiotemporal GNN (STGNN) for wind speed interpolation — 96-step sequences.

Extends train_gnn.py with temporal modelling via 1D convolutions interleaved
with GATv2 spatial message passing.  Each training sample is a 96-step sliding
window (48 past + 48 future).  The model learns to propagate neighbour
measurements observed in the past to forecast the target node in the future.

Tensor shapes throughout:
    Input per sample:  (N, seq_len, F)     — node × time × feature
    ST-block input:    (B, N, T, H)        — batch × node × time × hidden
    Temporal conv:     (B*N, H, T) → Conv1d → (B*N, H, T)
    Spatial conv:      (B*T, N, H) → mega-batch → GATv2 → (B*T*N, H)
    Head input:        (B, forecast_horizon, H)  — target node, future steps only
    Output:            (B, forecast_horizon)

Usage (run from forecasting_framework/):
    python geostatistics/train_stgnn.py --config configs/config_spatial_interpolation.yaml
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

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
from geostatistics.train_gnn import (
    build_edge_attr,
    build_test_edges,
    make_mega_batch,
)
from utils.interpolation import compute_distance_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph helper — radius-based (geodesic distances)
# ---------------------------------------------------------------------------

def build_radius_graph(
    dist_matrix: np.ndarray,
    radius_km: float,
    max_neighbors: int = None,
) -> torch.Tensor:
    """Directed graph: edges FROM neighbour TO node, only within *radius_km*.

    Uses pre-computed geodesic distances (km), so no projection error.
    ``max_neighbors`` optionally caps the number of incoming edges per node
    (keeps the nearest ones if the cap is exceeded).
    """
    N = dist_matrix.shape[0]
    src, dst = [], []
    for i in range(N):
        d = dist_matrix[i].copy()
        d[i] = np.inf
        neighbors = np.where(d <= radius_km)[0]
        if max_neighbors is not None and len(neighbors) > max_neighbors:
            neighbors = neighbors[np.argsort(d[neighbors])[:max_neighbors]]
        for j in neighbors:
            src.append(j)
            dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------

def load_nwp_feature_matrices(
    config: dict,
    station_ids: list,
    feature_names: list,
    timestamps: pd.DatetimeIndex,
) -> dict:
    """Load NWP feature columns from ICON-D2/ECMWF data as (T, N) matrices.

    This function processes NWP data using the same logic as preprocess_synth_wind_icond2,
    extracting features like wind_speed_h10, wind_speed_h38, ecmwf_wind_speed_h10, etc.

    Falls back to load_feature_matrices for features that exist in synth CSVs.

    Parameters
    ----------
    config : dict
        Configuration dictionary with data paths and params
    station_ids : list
        List of station IDs
    feature_names : list
        List of feature names to load (e.g., ['wind_speed_h10', 'wind_speed_h38'])
    timestamps : pd.DatetimeIndex
        Target timestamps to align the data to

    Returns
    -------
    dict mapping feature_name → np.ndarray of shape (T, N).
    """
    if not feature_names:
        return {}

    from utils.preprocessing import preprocess_synth_wind_icond2

    # Separate NWP features that need processing vs. synth CSV features
    nwp_like_features = []
    synth_csv_features = []

    for fname in feature_names:
        # Check if this looks like an NWP-derived feature
        if ('_h' in fname or fname.startswith('ecmwf_') or
            fname.endswith('_rotor_eq') or fname == 'density'):
            nwp_like_features.append(fname)
        else:
            synth_csv_features.append(fname)

    result = {}

    # Load synth CSV features using the original function
    if synth_csv_features:
        result.update(load_feature_matrices(
            config['data']['path'], station_ids, synth_csv_features, timestamps
        ))

    # Load NWP features by preprocessing ICON-D2 data
    if nwp_like_features:
        logger.info("Processing NWP features from ICON-D2 data (using nearest grid point): %s",
                    nwp_like_features)

        # Force next_n_grid_points to 1 for GNN (we only want the nearest grid point)
        gnn_config = {**config}
        gnn_config['params'] = {**config.get('params', {}), 'next_n_grid_points': 1}

        # Also set next_n_grid_ecmwf to 1 if not specified
        if 'next_n_grid_ecmwf' not in gnn_config['params']:
            gnn_config['params']['next_n_grid_ecmwf'] = 1

        # Create a minimal features dict for preprocessing
        preprocess_features = {
            'known': nwp_like_features,
            'observed': [],
            'static': []
        }

        all_station_dfs = []
        for sid in tqdm(station_ids, desc="Loading NWP features"):
            try:
                # Process NWP data for this station
                df_preprocessed = preprocess_synth_wind_icond2(
                    path=os.path.join(gnn_config['data']['path'], f"synth_{sid}.csv"),
                    config=gnn_config,
                    freq='1h',
                    features=preprocess_features
                )

                # Extract only the NWP features we need
                if isinstance(df_preprocessed.index, pd.MultiIndex):
                    # Use 'timestamp' level from MultiIndex
                    df_preprocessed = df_preprocessed.reset_index()
                    if 'timestamp' not in df_preprocessed.columns:
                        logger.warning(
                            "Station %s: no 'timestamp' column after reset_index, skipping.", sid
                        )
                        continue
                    df_preprocessed.set_index('timestamp', inplace=True)

                # Create a DataFrame to collect features for this station
                df_station = pd.DataFrame(index=df_preprocessed.index)

                # Extract requested features
                # Since we set next_n_grid_points=1, features will have _1 suffix
                for fname in nwp_like_features:
                    # Direct match (unlikely for NWP features)
                    if fname in df_preprocessed.columns:
                        df_station[fname] = df_preprocessed[fname]
                    else:
                        # Look for grid-point suffix _1 (nearest grid point)
                        fname_with_suffix = f"{fname}_1"
                        if fname_with_suffix in df_preprocessed.columns:
                            df_station[fname] = df_preprocessed[fname_with_suffix]
                        else:
                            logger.debug(
                                "Station %s: feature '%s' not found (checked '%s' and '%s')",
                                sid, fname, fname, fname_with_suffix
                            )

                if df_station.empty or len(df_station.columns) == 0:
                    logger.warning(
                        "Station %s: No NWP features could be extracted.", sid
                    )
                    continue

                df_station['station_id'] = sid
                all_station_dfs.append(df_station)

            except Exception as e:
                logger.warning("Failed to process NWP data for station %s: %s", sid, e)
                continue

        if not all_station_dfs:
            logger.error("No NWP data could be loaded for any station!")
            for fname in nwp_like_features:
                result[fname] = np.full((len(timestamps), len(station_ids)), np.nan)
            return result

        # Combine all stations
        combined = pd.concat(all_station_dfs, axis=0)
        if combined.index.tz is None:
            combined.index = combined.index.tz_localize("UTC")

        ts_idx = timestamps.tz_localize("UTC") if timestamps.tz is None else timestamps

        # Pivot each feature
        for fname in nwp_like_features:
            if fname not in combined.columns:
                logger.warning(
                    "Feature '%s' not found after NWP processing — using NaN.", fname
                )
                result[fname] = np.full((len(timestamps), len(station_ids)), np.nan)
                continue

            piv = (
                combined.reset_index()
                .pivot_table(
                    index="timestamp", columns="station_id",
                    values=fname, aggfunc="first",
                )
                .reindex(columns=station_ids)
            )
            piv.index = pd.DatetimeIndex(piv.index)
            result[fname] = piv.reindex(ts_idx).values.astype(np.float64)

    return result


def load_feature_matrices(
    data_path: str,
    station_ids: list,
    feature_names: list,
    timestamps: pd.DatetimeIndex,
) -> dict:
    """Load feature columns from station CSVs as (T, N) matrices.

    Reads each ``synth_{sid}.csv`` once and extracts all requested columns.
    The result is reindexed to *timestamps* (already filtered by load_data).
    Missing columns produce an all-NaN matrix with a warning.

    Returns
    -------
    dict mapping feature_name → np.ndarray of shape (T, N).
    """
    if not feature_names:
        return {}

    cols_needed = ["timestamp", "station_id"] + list(feature_names)
    all_dfs = []
    for sid in station_ids:
        fpath = os.path.join(data_path, f"synth_{sid}.csv")
        df = pd.read_csv(fpath, sep=";", parse_dates=["timestamp"])
        df["station_id"] = sid
        available = [c for c in cols_needed if c in df.columns]
        all_dfs.append(df[available])

    combined = pd.concat(all_dfs, ignore_index=True)
    if combined["timestamp"].dt.tz is None:
        combined["timestamp"] = combined["timestamp"].dt.tz_localize("UTC")

    ts_idx = timestamps.tz_localize("UTC") if timestamps.tz is None else timestamps

    result = {}
    for fname in feature_names:
        if fname not in combined.columns:
            logger.warning(
                "Feature '%s' not found in station CSVs — using NaN.", fname
            )
            result[fname] = np.full((len(timestamps), len(station_ids)), np.nan)
            continue
        piv = (
            combined.pivot_table(
                index="timestamp", columns="station_id",
                values=fname, aggfunc="first",
            )
            .reindex(columns=station_ids)
        )
        piv.index = pd.DatetimeIndex(piv.index)
        result[fname] = piv.reindex(ts_idx).values.astype(np.float64)

    return result


def load_measurement_features(
    data_path: str,
    station_ids: list,
    feature_names: list,
    timestamps: pd.DatetimeIndex,
) -> dict:
    """Load multiple measurement features from station CSVs.

    Returns
    -------
    dict: {feature_name: (T, N) array}
        Each feature is a (T, N) matrix aligned to timestamps × station_ids.
        Missing values are NaN.
    """
    result = {}
    for fname in feature_names:
        all_dfs = []
        for sid in station_ids:
            fpath = os.path.join(data_path, f"synth_{sid}.csv")
            if not os.path.exists(fpath):
                logger.warning("Station CSV not found: %s — skipping.", fpath)
                continue
            df = pd.read_csv(fpath, sep=";", parse_dates=["timestamp"])
            df["station_id"] = sid
            if fname in df.columns:
                all_dfs.append(df[["timestamp", "station_id", fname]])
            else:
                logger.warning("Feature '%s' not in %s — will be NaN.", fname, fpath)

        if not all_dfs:
            logger.warning("Feature '%s' not found in any station CSV — using NaN.", fname)
            result[fname] = np.full((len(timestamps), len(station_ids)), np.nan, dtype=np.float64)
            continue

        combined = pd.concat(all_dfs, ignore_index=True)
        piv = (
            combined.pivot_table(
                index="timestamp", columns="station_id",
                values=fname, aggfunc="first",
            )
            .reindex(columns=station_ids)
        )
        piv.index = pd.DatetimeIndex(piv.index)
        result[fname] = piv.reindex(timestamps).values.astype(np.float64)

    return result


def load_static_features(
    data_path: str,
    station_ids: list,
    feature_names: list,
    lats: np.ndarray,
    lons: np.ndarray,
    alts: np.ndarray,
) -> np.ndarray:
    """Build (N, S) static feature matrix from config-specified feature names.

    Handles the three standard names directly from already-loaded arrays:
    ``latitude``, ``longitude``, ``altitude``.  Any other names are looked up
    in ``wind_parameter.csv``; missing columns produce zeros with a warning.
    """
    _builtin = {"latitude": lats, "longitude": lons, "altitude": alts}
    extra_names = [f for f in feature_names if f not in _builtin]

    extra_data: dict = {}
    if extra_names:
        meta_path = os.path.join(data_path, "wind_parameter.csv")
        meta = pd.read_csv(meta_path, sep=";", dtype={"park_id": str}).set_index("park_id")
        for fname in extra_names:
            if fname in meta.columns:
                extra_data[fname] = meta.loc[station_ids, fname].values.astype(np.float64)
            else:
                logger.warning(
                    "Static feature '%s' not in wind_parameter.csv — using zeros.", fname
                )
                extra_data[fname] = np.zeros(len(station_ids))

    cols = []
    for fname in feature_names:
        cols.append(_builtin[fname] if fname in _builtin else extra_data[fname])

    return np.stack(cols, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpatioTemporalDataset(torch.utils.data.Dataset):
    """Sliding-window (N, seq_len, F) samples with LOO-style target masking.

    Feature vector per node per timestep::

        [meas_1 * mask, ..., meas_M * mask, nwp_1, ..., nwp_K, mask, static_1, ..., static_S]

    Measurement features (M), NWP features (K), and static features (S) are
    specified via config lists (``params.measurement_features``,
    ``params.nwp_features``, ``params.static_features``).

    Masking logic
    -------------
    - Neighbour nodes, steps 0..past_len-1 (past):        mask = 1 if observed
    - Neighbour nodes, steps past_len..seq_len-1 (future): mask = 0 (measurements not available)
    - Target node: mask = 0 for ALL steps (simulates unseen station)

    The target ``y`` is the first measurement feature (typically wind_speed)
    at the target node for steps past_len..seq_len-1 (the forecast horizon).
    """

    def __init__(
        self,
        measurement_matrices: list,  # M × (T, N) arrays, each a scaled measurement feature
        measurement_valid: list,     # M × (T, N) bool arrays: True where observed
        nwp_matrices: list,          # K × (T, N) arrays, each a scaled NWP feature
        static: np.ndarray,          # (N, S) static features, scaled
        seq_len: int = 96,
        forecast_horizon: int = 48,
        station_idxs: np.ndarray = None,
    ):
        if not measurement_matrices:
            raise ValueError("At least one measurement feature required.")
        T, N = measurement_matrices[0].shape

        # Store measurement features: list of (T, N) tensors
        self.measurements = [torch.tensor(m, dtype=torch.float32) for m in measurement_matrices]
        self.meas_valid = [torch.tensor(v, dtype=torch.float32) for v in measurement_valid]
        self.n_meas = len(self.measurements)

        # Stack NWP features along last dim: (T, N, K)
        if nwp_matrices:
            self.nwp = torch.stack(
                [torch.tensor(m, dtype=torch.float32) for m in nwp_matrices], dim=2
            )
        else:
            self.nwp = torch.zeros((T, N, 0), dtype=torch.float32)

        self.static = torch.tensor(static, dtype=torch.float32)
        self.seq_len = seq_len
        self.past_len = seq_len - forecast_horizon
        self.forecast_horizon = forecast_horizon
        self.N = N

        if station_idxs is None:
            station_idxs = np.arange(N)
        self.station_idxs = station_idxs
        self.n_stations = len(station_idxs)
        self.n_windows = T - seq_len + 1
        if self.n_windows <= 0:
            raise ValueError(
                f"Not enough timesteps for seq_len={seq_len}: T={T}"
            )

    def __len__(self) -> int:
        return self.n_windows * self.n_stations

    def __getitem__(self, idx: int):
        w = idx // self.n_stations
        s = int(self.station_idxs[idx % self.n_stations])

        # Extract measurement features for this window
        meas_tensors = []
        for i in range(self.n_meas):
            meas_w = self.measurements[i][w : w + self.seq_len].clone()  # (T, N)
            valid_w = self.meas_valid[i][w : w + self.seq_len]           # (T, N)

            # Mask: past steps & neighbors & valid observations
            mask = valid_w.clone()
            mask[self.past_len :] = 0.0  # Future: measurements not available
            mask[:, s] = 0.0             # Target node: never revealed

            meas_w = meas_w * mask       # Zero-out masked positions
            meas_tensors.append(meas_w.unsqueeze(2))  # (T, N, 1)

        # NWP features (available for all timesteps)
        nwp_w = self.nwp[w : w + self.seq_len]  # (T, N, K)

        # Combined mask (use first measurement feature's validity as reference)
        mask = self.meas_valid[0][w : w + self.seq_len].clone()
        mask[self.past_len :] = 0.0
        mask[:, s] = 0.0

        # Ground truth: first measurement feature (typically wind_speed) at target node
        y = self.measurements[0][w + self.past_len : w + self.seq_len, s].clone()

        # Static features broadcast over time: (N, S) → (T, N, S)
        static_exp = self.static.unsqueeze(0).expand(self.seq_len, -1, -1)

        # Build feature tensor: (T, N, F)  where F = M + K + 1 + S
        x = torch.cat(
            [
                *meas_tensors,          # M × (T, N, 1) measurement features
                nwp_w,                  # (T, N, K) NWP features
                mask.unsqueeze(2),      # (T, N, 1) mask
                static_exp,             # (T, N, S) static features
            ],
            dim=2,
        )  # (T, N, F)

        return x.permute(1, 0, 2), y, s  # (N, T, F), (forecast_horizon,), int


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class STBlock(nn.Module):
    """One spatiotemporal block: Temporal Conv1d then Spatial GATv2Conv.

    Temporal sub-block
    ------------------
    Applies the same Conv1d independently to every node across the time axis.
    Reshape: (B, N, T, H) → (B*N, H, T) → Conv1d → back.

    Spatial sub-block
    -----------------
    Applies GATv2Conv at every timestep across all nodes.
    Reshape: (B, N, T, H) → (B*T, N, H) → make_mega_batch → GATv2 → back.

    Both sub-blocks use residual connections + LayerNorm.
    """

    def __init__(
        self,
        hidden: int,
        heads: int,
        edge_dim: int,
        temporal_kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.temp_conv = nn.Conv1d(
            hidden, hidden, kernel_size=temporal_kernel_size, padding="same"
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.spatial_conv = GATv2Conv(
            hidden,
            hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (B, N, T, H)
            edge_index: (2, E) — static graph topology
            edge_attr:  (E, edge_dim)
        Returns:
            (B, N, T, H)
        """
        B, N, T, H = x.shape

        # ---- Temporal -------------------------------------------------------
        x_t = x.contiguous().reshape(B * N, T, H).permute(0, 2, 1)  # (B*N, H, T)
        x_t = self.temp_conv(x_t)                                     # (B*N, H, T)
        x_t = x_t.permute(0, 2, 1).contiguous().reshape(B, N, T, H) # (B, N, T, H)
        x = self.norm1(x + self.dropout(x_t))

        # ---- Spatial --------------------------------------------------------
        # Each of the B*T (batch, timestep) pairs becomes an independent graph.
        x_s = x.permute(0, 2, 1, 3).contiguous().reshape(B * T, N, H)  # (B*T, N, H)
        bx, bei, bea = make_mega_batch(x_s, edge_index, edge_attr)
        x_s_out = self.spatial_conv(bx, bei, bea)                       # (B*T*N, H)
        x_s_out = x_s_out.view(B, T, N, H).permute(0, 2, 1, 3)         # (B, N, T, H)
        x = self.norm2(x + self.dropout(x_s_out))

        return x


class SpatioTemporalGNN(nn.Module):
    """Inductive STGNN: input_proj → ST-blocks → per-horizon forecast head.

    The head operates on the hidden state of the *target node* at the
    ``forecast_horizon`` future timesteps only.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden: int,
        heads: int,
        num_layers: int,
        temporal_kernel_size: int,
        dropout: float,
        seq_len: int = 96,
        forecast_horizon: int = 48,
    ):
        super().__init__()
        self.past_len = seq_len - forecast_horizon
        self.forecast_horizon = forecast_horizon

        self.input_proj = nn.Linear(node_dim, hidden)
        self.st_blocks = nn.ModuleList(
            [
                STBlock(hidden, heads, edge_dim, temporal_kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (B, N, T, F_in)
            edge_index: (2, E)
            edge_attr:  (E, edge_dim)
            target_idx: (B,) long — which node is the target per sample
        Returns:
            (B, forecast_horizon) — predicted wind speed at the target node
        """
        B = x.shape[0]
        x = self.input_proj(x)  # (B, N, T, H)

        for block in self.st_blocks:
            x = block(x, edge_index, edge_attr)

        # Extract the target node's hidden state at forecast timesteps only
        x_target = x[
            torch.arange(B, device=x.device), target_idx, self.past_len :, :
        ]  # (B, forecast_horizon, H)

        return self.head(x_target).squeeze(-1)  # (B, forecast_horizon)


# ---------------------------------------------------------------------------
# Training / validation epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
    optimizer=None,
) -> tuple[float, float, float]:
    """Run one epoch and return (avg_loss, rmse, r2)."""
    training = optimizer is not None
    model.train() if training else model.eval()
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    total_loss, n = 0.0, 0

    all_preds = []
    all_targets = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x_batch, y_batch, s_batch in loader:
            # x_batch: (B, N, T, F)
            # y_batch: (B, forecast_horizon)
            # s_batch: (B,)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            s_batch = s_batch.to(device)

            preds = model(x_batch, edge_index, edge_attr, s_batch)
            loss = nn.functional.mse_loss(preds, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            B = x_batch.shape[0]
            total_loss += loss.item() * B
            n += B

            # Collect predictions and targets for metrics
            all_preds.append(preds.detach().cpu())
            all_targets.append(y_batch.detach().cpu())

    avg_loss = total_loss / n

    # Calculate RMSE and R²
    all_preds = torch.cat(all_preds, dim=0).numpy()  # (N_samples, forecast_horizon)
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Flatten for metrics calculation
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()

    rmse = np.sqrt(mean_squared_error(targets_flat, preds_flat))
    r2 = r2_score(targets_flat, preds_flat)

    return avg_loss, rmse, r2


# ---------------------------------------------------------------------------
# Inference at a test station
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_test_station(
    model: nn.Module,
    meas_scaled: list,               # M × (T, N_train) arrays, scaled measurement features
    meas_valid: list,                # M × (T, N_train) bool arrays
    nwp_scaled: list,                # K × (T, N_train) arrays, each a scaled NWP feature
    static_scaled: np.ndarray,       # (N_train, S)
    train_edge_index: torch.Tensor,
    train_edge_attr: torch.Tensor,
    test_static_scaled: np.ndarray,  # (S,)
    test_nwp_scaled: list,           # K × (T,) arrays, scaled NWP for the test station
    test_lat: float,
    test_lon: float,
    train_lats: np.ndarray,
    train_lons: np.ndarray,
    train_dist_to_test: np.ndarray,
    max_dist: float,
    k: int,
    ws_scaler: StandardScaler,
    device: torch.device,
    seq_len: int = 96,
    forecast_horizon: int = 48,
    batch_size: int = 32,
    overlap_mode: str = "shortest_lead",
) -> np.ndarray:
    """Predict full T-step wind speed at one unseen test station.

    Processes all valid 96-step windows.  When multiple windows produce a
    prediction for the same absolute timestep, ``overlap_mode`` controls how
    they are combined:
      - ``"shortest_lead"``: keep the prediction with the smallest forecast
        step index (i.e. from the most recent window).
      - ``"mean"``: average all predictions.

    Returns an (T,) array of wind speed in the original (unscaled) units.
    Timesteps with no window coverage are NaN.
    """
    model.eval()
    n_meas = len(meas_scaled)
    T, N_tr = meas_scaled[0].shape
    past_len = seq_len - forecast_horizon
    test_node_idx = N_tr  # appended as the last node

    new_ei, new_ea = build_test_edges(
        test_lat, test_lon, train_lats, train_lons,
        train_dist_to_test, max_dist, test_node_idx, k,
    )
    eval_ei = torch.cat([train_edge_index, new_ei], dim=1).to(device)
    eval_ea = torch.cat([train_edge_attr, new_ea], dim=0).to(device)

    # Pre-build CPU tensors once; individual windows are sliced later.
    meas_ts = [torch.tensor(m, dtype=torch.float32) for m in meas_scaled]
    meas_valid_ts = [torch.tensor(v, dtype=torch.float32) for v in meas_valid]
    # NWP for training nodes: list of (T, N_tr) → stacked (T, N_tr, K)
    nwp_tr_t = torch.stack(
        [torch.tensor(m, dtype=torch.float32) for m in nwp_scaled], dim=2
    )
    static_t = torch.tensor(static_scaled, dtype=torch.float32)   # (N_tr, S)
    # NWP for test node: list of (T,) → stacked (T, K)
    test_nwp_t = torch.stack(
        [torch.tensor(m, dtype=torch.float32) for m in test_nwp_scaled], dim=1
    )  # (T, K)
    test_static_t = torch.tensor(
        test_static_scaled, dtype=torch.float32
    ).unsqueeze(0)  # (1, S)

    n_windows = T - seq_len + 1
    preds_acc: dict[int, list] = defaultdict(list)  # abs_t → [(h, value), ...]

    for start in range(0, n_windows, batch_size):
        window_starts = list(range(start, min(start + batch_size, n_windows)))
        B = len(window_starts)
        x_list = []

        for w in window_starts:
            # Build measurement feature tensors for train nodes
            meas_tensors_tr = []
            for mi in range(n_meas):
                mw = meas_ts[mi][w : w + seq_len].clone()        # (T, N_tr)
                vw = meas_valid_ts[mi][w : w + seq_len].clone()  # (T, N_tr)
                msk = vw.clone()
                msk[past_len:] = 0.0  # future: no measurements
                mw = mw * msk
                meas_tensors_tr.append(mw.unsqueeze(2))  # (T, N_tr, 1)

            nwp_w = nwp_tr_t[w : w + seq_len]  # (T, N_tr, K)

            # Combined mask from first measurement feature
            mask_tr = meas_valid_ts[0][w : w + seq_len].clone()
            mask_tr[past_len:] = 0.0

            static_tr_exp = static_t.unsqueeze(0).expand(seq_len, -1, -1)  # (T, N_tr, S)
            x_tr = torch.cat(
                [
                    *meas_tensors_tr,       # M × (T, N_tr, 1)
                    nwp_w,                  # (T, N_tr, K)
                    mask_tr.unsqueeze(2),   # (T, N_tr, 1)
                    static_tr_exp,          # (T, N_tr, S)
                ],
                dim=2,
            )  # (T, N_tr, F)

            # Test node: all measurements = 0, mask = 0
            t_nwp_w = test_nwp_t[w : w + seq_len].unsqueeze(1)  # (T, 1, K)
            t_static_exp = test_static_t.unsqueeze(0).expand(seq_len, -1, -1)  # (T, 1, S)
            x_test = torch.cat(
                [
                    torch.zeros(seq_len, 1, n_meas),  # M measurement features = 0
                    t_nwp_w,                           # (T, 1, K)
                    torch.zeros(seq_len, 1, 1),        # mask = 0
                    t_static_exp,                      # (T, 1, S)
                ],
                dim=2,
            )  # (T, 1, F)

            x_full = torch.cat([x_tr, x_test], dim=1)  # (T, N_eval, F)
            x_list.append(x_full.permute(1, 0, 2))      # (N_eval, T, F)

        x_batch = torch.stack(x_list, dim=0).to(device)  # (B, N_eval, T, F)
        tgt = torch.full(
            (B,), test_node_idx, dtype=torch.long, device=device
        )
        out = model(x_batch, eval_ei, eval_ea, tgt)  # (B, forecast_horizon)
        out_np = out.cpu().numpy()

        for i, w in enumerate(window_starts):
            for h in range(forecast_horizon):
                abs_t = w + past_len + h
                if abs_t < T:
                    preds_acc[abs_t].append((h, float(out_np[i, h])))

    # Combine overlapping predictions
    preds_scaled = np.full(T, np.nan)
    for abs_t, preds_list in preds_acc.items():
        if overlap_mode == "shortest_lead":
            # Minimum h → latest window → shortest forecast lead time
            val = min(preds_list, key=lambda p: p[0])[1]
        else:
            val = float(np.mean([p[1] for p in preds_list]))
        preds_scaled[abs_t] = val

    valid_mask = ~np.isnan(preds_scaled)
    result = np.full(T, np.nan)
    result[valid_mask] = ws_scaler.inverse_transform(
        preds_scaled[valid_mask].reshape(-1, 1)
    ).flatten()
    return result


@torch.no_grad()
def predict_val_stations_with_neighbors(
    model: nn.Module,
    all_meas_scaled: list,          # M × (T, N_all) arrays (train + val concatenated)
    all_meas_valid: list,           # M × (T, N_all) bool arrays
    all_nwp_scaled: list,           # K × (T, N_all) arrays
    all_static_scaled: np.ndarray,  # (N_all, S)
    full_edge_index: torch.Tensor,  # (2, E_full)
    full_edge_attr: torch.Tensor,   # (E_full, edge_dim)
    val_indices: list,              # indices of val nodes in N_all space
    ws_scaler: StandardScaler,
    device: torch.device,
    seq_len: int,
    batch_size: int = 32,
    overlap_mode: str = "shortest_lead",
) -> np.ndarray:
    """LOO inference for all val stations simultaneously.

    Each val station is predicted with all other nodes (train + remaining val)
    visible as neighbours. Only the current target's measurements are masked.
    This means val stations provide spatial context for each other.

    Returns
    -------
    np.ndarray of shape (T, N_val) — wind speed in original (unscaled) units.
    """
    model.eval()
    n_meas = len(all_meas_scaled)
    T, N_all = all_meas_scaled[0].shape
    N_val = len(val_indices)

    meas_ts = [torch.tensor(m, dtype=torch.float32) for m in all_meas_scaled]
    meas_valid_ts = [torch.tensor(v, dtype=torch.float32) for v in all_meas_valid]
    nwp_t = torch.stack(
        [torch.tensor(m, dtype=torch.float32) for m in all_nwp_scaled], dim=2
    )  # (T, N_all, K)
    static_t = torch.tensor(all_static_scaled, dtype=torch.float32)  # (N_all, S)

    n_windows = T - seq_len + 1
    full_edge_index = full_edge_index.to(device)
    full_edge_attr = full_edge_attr.to(device)

    result = np.full((T, N_val), np.nan)

    for vi, target_node in enumerate(
        tqdm(val_indices, desc="Predicting val stations", unit="station")
    ):
        preds_acc: dict[int, list] = defaultdict(list)

        for start in range(0, n_windows, batch_size):
            window_starts = list(range(start, min(start + batch_size, n_windows)))
            B = len(window_starts)
            x_list = []

            for w in window_starts:
                meas_tensors = []
                for mi in range(n_meas):
                    mw = meas_ts[mi][w : w + seq_len].clone()        # (seq_len, N_all)
                    vw = meas_valid_ts[mi][w : w + seq_len].clone()  # (seq_len, N_all)
                    msk = vw.clone()
                    msk[:, target_node] = 0.0  # mask target node
                    mw = mw * msk
                    meas_tensors.append(mw.unsqueeze(2))              # (seq_len, N_all, 1)

                nwp_w = nwp_t[w : w + seq_len]                       # (seq_len, N_all, K)
                mask = meas_valid_ts[0][w : w + seq_len].clone()
                mask[:, target_node] = 0.0
                static_exp = static_t.unsqueeze(0).expand(seq_len, -1, -1)

                x = torch.cat(
                    [*meas_tensors, nwp_w, mask.unsqueeze(2), static_exp],
                    dim=2,
                )  # (seq_len, N_all, F)
                x_list.append(x.permute(1, 0, 2))  # (N_all, seq_len, F)

            x_batch = torch.stack(x_list, dim=0).to(device)  # (B, N_all, seq_len, F)
            tgt = torch.full((B,), target_node, dtype=torch.long, device=device)
            out = model(x_batch, full_edge_index, full_edge_attr, tgt)  # (B, seq_len)
            out_np = out.cpu().numpy()

            for bi, w in enumerate(window_starts):
                for h in range(seq_len):
                    abs_t = w + h
                    if abs_t < T:
                        preds_acc[abs_t].append((h, float(out_np[bi, h])))

        preds_scaled = np.full(T, np.nan)
        for abs_t, preds_list in preds_acc.items():
            if overlap_mode == "shortest_lead":
                val_pred = min(preds_list, key=lambda p: p[0])[1]
            else:
                val_pred = float(np.mean([p[1] for p in preds_list]))
            preds_scaled[abs_t] = val_pred

        valid_mask = ~np.isnan(preds_scaled)
        result[valid_mask, vi] = ws_scaler.inverse_transform(
            preds_scaled[valid_mask].reshape(-1, 1)
        ).flatten()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="STGNN spatial interpolation — 96-step sequences"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("-s", "--suffix", default="")
    parser.add_argument(
        "--overlap-mode",
        default="shortest_lead",
        choices=["shortest_lead", "mean"],
        help="How to combine multiple window predictions for the same timestep",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"]
    )
    args = parser.parse_args()
    suffix = ''
    if args.suffix:
        suffix = f'_{args.suffix}'
    if '.yaml' in args.config:
        args.config = args.config.split('.')[0]
    if '/' in args.config:
        config_name = args.config.split('/')[-1]
    else:
        config_name = args.config
    log_file = f'logs/train_cl_m-stgnn_{("_").join(config_name.split("_")[1:])}{suffix}.log'
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    config = load_config(f'{args.config}.yaml')
    interp_cfg = config.get("interpolation", {})
    stgnn_cfg = config.get("stgnn", {})
    out_cfg = config.get("output", {})

    output_dir = out_cfg.get("path", "results/geostatistics")
    config_stem = os.path.splitext(os.path.basename(args.config))[0].removeprefix(
        "config_"
    )
    prefix = config_stem + "_stgnn" + (f"_{args.suffix}" if args.suffix else "")

    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
    )
    os.makedirs(logs_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(logs_dir, f"{prefix}.log"), mode="w"
    )
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
        )
    )
    logging.getLogger().addHandler(fh)
    logger.info("=== Output prefix: %s | dir: %s ===", prefix, output_dir)

    # ---- Hyperparameters ---------------------------------------------------
    radius_km = float(stgnn_cfg.get("radius_km", 300.0))
    max_neighbors = stgnn_cfg.get("max_neighbors")   # None = no cap
    if max_neighbors is not None:
        max_neighbors = int(max_neighbors)
    k = max_neighbors if max_neighbors is not None else 15  # for build_test_edges
    val_fraction = float(stgnn_cfg.get("val_fraction", 0.2))
    seq_len = int(stgnn_cfg.get("seq_len", 96))
    forecast_horizon = int(stgnn_cfg.get("forecast_horizon", 48))
    hidden = int(stgnn_cfg.get("hidden", 128))
    heads = int(stgnn_cfg.get("heads", 4))
    num_layers = int(stgnn_cfg.get("num_layers", 3))
    temporal_kernel_size = int(stgnn_cfg.get("temporal_kernel_size", 3))
    dropout = float(stgnn_cfg.get("dropout", 0.1))
    lr = float(stgnn_cfg.get("lr", 1e-3))
    weight_decay = float(stgnn_cfg.get("weight_decay", 1e-5))
    batch_size = int(stgnn_cfg.get("batch_size", 32))
    max_epochs = int(stgnn_cfg.get("max_epochs", 100))
    patience = int(stgnn_cfg.get("patience", 15))
    num_workers = int(stgnn_cfg.get("num_workers", 4))

    past_len = seq_len - forecast_horizon
    assert past_len > 0 and forecast_horizon > 0, (
        f"seq_len={seq_len} must be > forecast_horizon={forecast_horizon} > 0"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Data loading ------------------------------------------------------
    # Strip interpolation.rk_features so load_data only loads wind_speed and
    # metadata — no RK feature loading, no RK-driven timestamp dropping.
    # The STGNN loads its own NWP features via load_feature_matrices below.
    # Strip test_start/test_end so load_data loads ALL historical data.
    # The split into train/val windows is done below based on test_start.
    _load_cfg = {**config, "data": {
        **config["data"], "test_start": None, "test_end": None
    }, "interpolation": {
        **config.get("interpolation", {}), "rk_features": None
    }}
    logger.info("=== Loading data (train stations) ===")
    (
        pivot, lats, lons, alts, station_ids,
        _u, _v,
        _rk_names, _rk_static, _rk_dynamic,
    ) = load_data(_load_cfg)

    timestamps = pivot.index
    T, N = pivot.shape

    # ---- Feature lists from config -----------------------------------------
    params_cfg = config.get("params", {})
    measurement_feature_names = list(params_cfg.get("measurement_features", ["wind_speed"]))
    nwp_feature_names = list(params_cfg.get("nwp_features", []))
    static_feature_names = list(
        params_cfg.get("static_features", ["altitude", "latitude", "longitude"])
    )

    if not measurement_feature_names:
        raise ValueError(
            "params.measurement_features must contain at least one feature (e.g., ['wind_speed'])."
        )

    if not nwp_feature_names:
        logger.warning(
            "params.nwp_features not set — no NWP features will be used. "
            "Set params.nwp_features in the config."
        )

    # Load measurement features from station CSVs
    logger.info("=== Loading measurement features: %s ===", measurement_feature_names)
    data_path = config["data"]["path"]
    meas_raw = load_measurement_features(
        data_path, station_ids, measurement_feature_names, timestamps
    )

    # Validate shapes
    for fname, arr in meas_raw.items():
        if arr.shape != (T, N):
            raise ValueError(
                f"Measurement feature '{fname}' has shape {arr.shape}, expected ({T}, {N})."
            )
        logger.info("  Feature '%s': %.1f%% NaN", fname, 100.0 * np.mean(np.isnan(arr)))

    # ---- Station split: explicit from config (files = train, val_files = test) ----
    if not config["data"].get("val_files"):
        raise ValueError(
            "data.val_files not set in config — required for STGNN evaluation."
        )

    N_train = N
    logger.info("Station split from config: %d train / %d test (val_files)",
                N_train, len(config["data"]["val_files"]))

    split_df = pd.DataFrame(
        [{"station_id": sid, "split": "train"} for sid in station_ids]
        + [{"station_id": str(sid), "split": "test"}
           for sid in config["data"]["val_files"]]
    )
    split_path = os.path.join(output_dir, f"{prefix}_station_split.csv")
    split_df.to_csv(split_path, index=False)
    logger.info("Saved station split → %s", split_path)

    # ---- Load NWP feature matrices for training stations -------------------
    data_path = config["data"]["path"]
    logger.info("=== Loading NWP features for training stations: %s ===",
                nwp_feature_names)
    nwp_tr_raw = load_nwp_feature_matrices(
        config, station_ids, nwp_feature_names, timestamps
    )  # {name: (T, N_train)}

    # ---- Load val/test station data ----------------------------------------
    logger.info("=== Loading val/test station data ===")
    val_config = {**_load_cfg, "data": {**config["data"], "files": config["data"]["val_files"]}}
    (
        val_pivot, val_lats, val_lons, val_alts, val_station_ids,
        _vu, _vv, _val_rk_names, _val_rk_static, _val_rk_dynamic,
    ) = load_data(val_config)
    val_ws_full = val_pivot.values.astype(np.float64)   # (T, N_val)
    val_timestamps = val_pivot.index

    logger.info("=== Loading NWP features for val/test stations: %s ===",
                nwp_feature_names)
    nwp_val_raw = load_nwp_feature_matrices(
        config, val_station_ids, nwp_feature_names, val_timestamps
    )  # {name: (T, N_val)}

    logger.info("=== Loading measurement features for val/test stations: %s ===",
                measurement_feature_names)
    val_meas_raw = load_measurement_features(
        data_path, val_station_ids, measurement_feature_names, val_timestamps
    )  # {name: (T, N_val)}

    N_val = len(val_station_ids)
    T_val = val_ws_full.shape[0]
    assert T_val == T, (
        f"Val timestamps ({T_val}) differ from training timestamps ({T}). "
        "Check that test_start/test_end are stripped from val_config data loading."
    )

    # ---- Training data assignment ------------------------------------------
    lats_tr = lats
    lons_tr = lons
    alts_tr = alts

    # Temporal train / val split: windows starting before test_start → train,
    # windows starting at or after test_start → val (early stopping).
    test_start_cfg = config["data"].get("test_start")
    if test_start_cfg:
        ts_cutoff = pd.Timestamp(test_start_cfg, tz="UTC")
        if timestamps.tz is None:
            ts_idx = timestamps.tz_localize("UTC")
        else:
            ts_idx = timestamps
        split_t = int(np.searchsorted(ts_idx, ts_cutoff, side="left"))
        logger.info("train/val split at test_start=%s → split_t=%d", test_start_cfg, split_t)
    else:
        split_t = int(T * (1 - val_fraction))
        logger.info("train/val split by val_fraction=%.2f → split_t=%d", val_fraction, split_t)
    n_val_t = T - split_t
    has_val = n_val_t >= seq_len
    if not has_val:
        logger.warning(
            "Validation period (%d timesteps) < seq_len (%d) — "
            "early stopping disabled, training for max_epochs.",
            n_val_t, seq_len,
        )

    # ---- Scaling: measurement features -------------------------------------
    meas_scalers: dict = {}        # {name: StandardScaler}
    meas_scaled_list: list = []    # M × (T, N_train) scaled arrays
    meas_valid_list: list = []     # M × (T, N_train) bool arrays
    for fname in measurement_feature_names:
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
        logger.info("  Measurement '%s': scaler mean=%.4f std=%.4f",
                    fname, float(sc.mean_[0]), float(sc.scale_[0]))

    # First measurement feature scaler for inverse transform of predictions
    ws_sc = meas_scalers[measurement_feature_names[0]]

    # NWP features: one StandardScaler per feature, fit on training portion only
    nwp_scalers: dict = {}    # {name: StandardScaler}
    nwp_scaled_tr: list = []  # K × (T, N_train) scaled arrays (same order as nwp_feature_names)
    for fname in nwp_feature_names:
        raw = nwp_tr_raw[fname].copy()
        valid = ~np.isnan(raw)
        raw[~valid] = 0.0
        sc = StandardScaler()
        sc.fit(raw[:split_t].reshape(-1, 1))
        scaled = sc.transform(raw.reshape(-1, 1)).reshape(T, N_train)
        scaled[~valid] = 0.0
        nwp_scalers[fname] = sc
        nwp_scaled_tr.append(scaled)
        logger.info("  NWP feature '%s': %.1f%% NaN",
                    fname, 100.0 * np.mean(~valid))

    # Static features (fit scaler on training stations only)
    static_raw = load_static_features(
        data_path, station_ids, static_feature_names, lats_tr, lons_tr, alts_tr
    )
    static_sc = StandardScaler()
    static_scaled = static_sc.fit_transform(static_raw)
    n_static = static_scaled.shape[1]

    # ---- Build training graph ----------------------------------------------
    logger.info(
        "=== Building radius graph (r=%.0f km, max_neighbors=%s) on %d training stations ===",
        radius_km, str(max_neighbors) if max_neighbors else "unlimited", N_train,
    )
    dist_tr = compute_distance_matrix(lats_tr, lons_tr)
    edge_index = build_radius_graph(dist_tr, radius_km, max_neighbors)
    if edge_index.shape[1] == 0:
        raise ValueError(
            f"No edges created with radius_km={radius_km}. Increase radius_km."
        )
    edge_attr, max_dist = build_edge_attr(dist_tr, lats_tr, lons_tr, edge_index)
    avg_deg = edge_index.shape[1] / N_train
    logger.info("  Nodes: %d  |  Edges: %d  |  Avg degree: %.1f",
                N_train, edge_index.shape[1], avg_deg)

    # ---- DataLoaders -------------------------------------------------------
    nwp_t_list = [m[:split_t] for m in nwp_scaled_tr]
    nwp_v_list = [m[split_t:] for m in nwp_scaled_tr]
    meas_t_list = [m[:split_t] for m in meas_scaled_list]
    meas_valid_t_list = [v[:split_t] for v in meas_valid_list]

    tr_dataset = SpatioTemporalDataset(
        meas_t_list, meas_valid_t_list,
        nwp_t_list, static_scaled,
        seq_len=seq_len, forecast_horizon=forecast_horizon,
    )
    tr_loader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    val_loader = None
    if has_val:
        meas_v_list = [m[split_t:] for m in meas_scaled_list]
        meas_valid_v_list = [v[split_t:] for v in meas_valid_list]
        val_dataset = SpatioTemporalDataset(
            meas_v_list, meas_valid_v_list,
            nwp_v_list, static_scaled,
            seq_len=seq_len, forecast_horizon=forecast_horizon,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    n_meas = len(measurement_feature_names)
    logger.info(
        "Dataset: %d train samples (%d windows × %d stations)  "
        "node_dim=%d+%d+1+%d",
        len(tr_dataset), tr_dataset.n_windows, N_train,
        n_meas, len(nwp_feature_names), n_static,
    )

    # ---- Model -------------------------------------------------------------
    node_dim = n_meas + len(nwp_feature_names) + 1 + n_static  # meas + NWP + mask + static
    model = SpatioTemporalGNN(
        node_dim=node_dim,
        edge_dim=edge_attr.shape[1],
        hidden=hidden,
        heads=heads,
        num_layers=num_layers,
        temporal_kernel_size=temporal_kernel_size,
        dropout=dropout,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
    ).to(device)

    n_params_input  = sum(p.numel() for p in model.input_proj.parameters() if p.requires_grad)
    n_params_blocks = sum(p.numel() for p in model.st_blocks.parameters() if p.requires_grad)
    n_params_head   = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    n_params        = n_params_input + n_params_blocks + n_params_head
    logger.info(
        "Model parameters: %s total  (input_proj=%s  st_blocks=%s  head=%s)",
        f"{n_params:,}", f"{n_params_input:,}", f"{n_params_blocks:,}", f"{n_params_head:,}",
    )
    logger.info(
        "  node_dim=%d  hidden=%d  heads=%d  layers=%d  kernel=%d  seq_len=%d  horizon=%d",
        node_dim, hidden, heads, num_layers, temporal_kernel_size, seq_len, forecast_horizon,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=patience // 3, factor=0.5, min_lr=1e-6
    )

    # ---- Log all hyperparameters -------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("HYPERPARAMETERS")
    logger.info("=" * 80)
    logger.info("Data:")
    logger.info("  Training stations: %d", N_train)
    logger.info("  Test stations: %d", len(val_station_ids))
    logger.info("  Training timesteps: %d", split_t)
    logger.info("  Validation timesteps: %d", T - split_t)
    logger.info("  Test window: %s to %s",
                config["data"].get("test_start", "N/A"),
                config["data"].get("test_end", "N/A"))
    logger.info("")
    logger.info("Features:")
    logger.info("  Measurement features (%d): %s", n_meas, measurement_feature_names)
    logger.info("  NWP features (%d): %s", len(nwp_feature_names), nwp_feature_names)
    logger.info("  Static features (%d): %s", n_static, static_feature_names)
    logger.info("  Node feature dim: %d (%d meas + %d nwp + 1 mask + %d static)",
                node_dim, n_meas, len(nwp_feature_names), n_static)
    logger.info("")
    logger.info("Model Architecture:")
    logger.info("  Model type: SpatioTemporalGNN")
    logger.info("  Hidden dimension: %d", hidden)
    logger.info("  Attention heads: %d", heads)
    logger.info("  ST-Blocks: %d", num_layers)
    logger.info("  Temporal kernel size: %d", temporal_kernel_size)
    logger.info("  Dropout: %.2f", dropout)
    logger.info("  Total parameters: %d", n_params)
    logger.info("")
    logger.info("Graph:")
    logger.info("  k-NN neighbors: %d", k)
    logger.info("  Nodes: %d", N_train)
    logger.info("  Edges: %d", edge_index.shape[1])
    logger.info("  Edge feature dim: %d", edge_attr.shape[1])
    logger.info("")
    logger.info("Sequence Settings:")
    logger.info("  Sequence length: %d steps", seq_len)
    logger.info("  Past context: %d steps", seq_len - forecast_horizon)
    logger.info("  Forecast horizon: %d steps", forecast_horizon)
    logger.info("")
    logger.info("Training:")
    logger.info("  Batch size: %d", batch_size)
    logger.info("  Learning rate: %.2e", lr)
    logger.info("  Weight decay: %.2e", weight_decay)
    logger.info("  Max epochs: %d", max_epochs)
    logger.info("  Early stopping patience: %d", patience)
    logger.info("  LR scheduler patience: %d", patience // 3)
    logger.info("  Gradient clipping: 1.0")
    logger.info("  Optimizer: AdamW")
    logger.info("  LR scheduler: ReduceLROnPlateau (factor=0.5, min_lr=1e-6)")
    logger.info("")
    logger.info("Data Scaling:")
    logger.info("  Measurement features: StandardScaler per feature (fit on train)")
    logger.info("  NWP features: StandardScaler per feature (fit on train)")
    logger.info("  Static features: StandardScaler (fit on train stations)")
    logger.info("=" * 80 + "\n")

    # ---- Training loop -----------------------------------------------------
    logger.info(
        "=== Training (train_T=%d  val_T=%d) ===", split_t, T - split_t
    )
    best_val, best_state, no_imp = np.inf, None, 0

    for epoch in range(1, max_epochs + 1):
        tl, tr_rmse, tr_r2 = run_epoch(model, tr_loader, edge_index, edge_attr, device, opt)

        if val_loader is not None:
            vl, val_rmse, val_r2 = run_epoch(model, val_loader, edge_index, edge_attr, device)
            sched.step(vl)
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  train_rmse=%.4f  train_r2=%.3f  "
                "val_loss=%.4f  val_rmse=%.4f  val_r2=%.3f  lr=%.2e",
                epoch, max_epochs, tl, tr_rmse, tr_r2, vl, val_rmse, val_r2,
                opt.param_groups[0]["lr"],
            )

            if vl < best_val - 1e-6:
                best_val = vl
                best_state = {
                    k_: v.cpu().clone() for k_, v in model.state_dict().items()
                }
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        else:
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  train_rmse=%.4f  train_r2=%.3f  lr=%.2e",
                epoch, max_epochs, tl, tr_rmse, tr_r2, opt.param_groups[0]["lr"],
            )
            best_state = {
                k_: v.cpu().clone() for k_, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    if val_loader is not None:
        logger.info("Best val loss: %.4f", best_val)

    ckpt_path = os.path.join(output_dir, f"{prefix}_model.pt")
    torch.save(
        {
            "model_state": best_state,
            "node_dim": node_dim,
            "edge_dim": edge_attr.shape[1],
            "hidden": hidden,
            "heads": heads,
            "num_layers": num_layers,
            "temporal_kernel_size": temporal_kernel_size,
            "dropout": dropout,
            "seq_len": seq_len,
            "forecast_horizon": forecast_horizon,
        },
        ckpt_path,
    )
    logger.info("Saved model checkpoint → %s", ckpt_path)

    # ---- Inference at val/test stations ------------------------------------
    logger.info("=== Inference at %d val/test stations (LOO with val-as-neighbours) ===", N_val)

    # Build full graph over all stations (train + val)
    all_lats = np.concatenate([lats_tr, val_lats])
    all_lons = np.concatenate([lons_tr, val_lons])
    dist_full = compute_distance_matrix(all_lats, all_lons)
    full_edge_index = build_radius_graph(dist_full, radius_km, max_neighbors)
    full_edge_attr, _ = build_edge_attr(dist_full, all_lats, all_lons, full_edge_index)
    logger.info(
        "Full graph (train+val): %d nodes | %d edges | avg degree %.1f",
        N_train + N_val, full_edge_index.shape[1],
        full_edge_index.shape[1] / (N_train + N_val),
    )

    # Scale val NWP features with training scalers
    nwp_scaled_val_list = []
    for fname in nwp_feature_names:
        raw = nwp_val_raw[fname].copy()
        sc_arr = nwp_scalers[fname].transform(raw.reshape(-1, 1)).reshape(T_val, N_val)
        nwp_scaled_val_list.append(sc_arr)

    # Scale val measurement features with training scalers
    meas_scaled_val_list = []
    meas_valid_val_list = []
    for fname in measurement_feature_names:
        raw = val_meas_raw[fname].copy()
        valid = ~np.isnan(raw)
        sc_arr = meas_scalers[fname].transform(raw.reshape(-1, 1)).reshape(T_val, N_val)
        meas_scaled_val_list.append(sc_arr)
        meas_valid_val_list.append(valid)

    # Scale val static features with training scaler
    val_static_raw = load_static_features(
        data_path, val_station_ids, static_feature_names, val_lats, val_lons, val_alts
    )
    val_static_scaled = static_sc.transform(val_static_raw)

    # Concatenate train + val along node axis
    all_meas_scaled = [
        np.concatenate([tr, va], axis=1)
        for tr, va in zip(meas_scaled_list, meas_scaled_val_list)
    ]
    all_meas_valid = [
        np.concatenate([tr, va], axis=1)
        for tr, va in zip(meas_valid_list, meas_valid_val_list)
    ]
    all_nwp_scaled = [
        np.concatenate([tr, va], axis=1)
        for tr, va in zip(nwp_scaled_tr, nwp_scaled_val_list)
    ]
    all_static = np.concatenate([static_scaled, val_static_scaled], axis=0)

    # Val station indices in the full (train+val) graph
    val_indices = list(range(N_train, N_train + N_val))

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
        overlap_mode=args.overlap_mode,
    )  # (T, N_val)

    pred_records = []
    per_station_rows = []

    for i, sid in enumerate(val_station_ids):
        preds = preds_all[:, i]

        obs = val_ws_full[:, i]
        m = ~(np.isnan(obs) | np.isnan(preds))
        rmse = float(np.sqrt(mean_squared_error(obs[m], preds[m])))
        mae = float(mean_absolute_error(obs[m], preds[m]))
        r2 = float(r2_score(obs[m], preds[m]))

        # Calculate NWP baseline RMSE (wind_speed_h10) if available
        skill_nwp = np.nan
        if 'wind_speed_h10' in nwp_feature_names:
            idx_h10 = nwp_feature_names.index('wind_speed_h10')
            # Get raw (unscaled) NWP predictions
            nwp_h10_raw = nwp_val_raw['wind_speed_h10'][:, i]
            # Calculate RMSE for NWP baseline
            m_nwp = ~(np.isnan(obs) | np.isnan(nwp_h10_raw))
            if m_nwp.sum() > 0:
                rmse_nwp = float(np.sqrt(mean_squared_error(obs[m_nwp], nwp_h10_raw[m_nwp])))
                # Skill = 1 - (RMSE_model / RMSE_nwp)
                skill_nwp = 1.0 - (rmse / rmse_nwp)

        logger.info("  %s  R²=%.4f  RMSE=%.4f  Skill_NWP=%.4f", sid, r2, rmse, skill_nwp if not np.isnan(skill_nwp) else 0.0)
        per_station_rows.append(
            {"station_id": sid, "method": "stgnn", "rmse": rmse, "mae": mae, "r2": r2, "skill_nwp": skill_nwp}
        )

        for t_idx in range(T_val):
            pred_records.append(
                {
                    "station_id": sid,
                    "timestamp": val_timestamps[t_idx],
                    "wind_speed_observed": float(obs[t_idx]),
                    "stgnn_pred": float(preds[t_idx]),
                }
            )

    # ---- Save outputs ------------------------------------------------------
    per_station = pd.DataFrame(per_station_rows)
    ps_path = os.path.join(output_dir, f"{prefix}_results_per_station.csv")
    per_station.to_csv(ps_path, index=False)
    logger.info("Saved per-station metrics → %s", ps_path)

    preds_df = pd.DataFrame(pred_records)
    pred_path = os.path.join(output_dir, f"{prefix}_predictions.csv")
    preds_df.to_csv(pred_path, index=False)
    logger.info("Saved predictions → %s", pred_path)

    summary = per_station[["rmse", "mae", "r2", "skill_nwp"]].mean().to_frame().T
    summary.insert(0, "method", "stgnn")
    sm_path = os.path.join(output_dir, f"{prefix}_results_summary.csv")
    summary.to_csv(sm_path, index=False)
    logger.info("Saved summary → %s", sm_path)

    # ---- Print results table -----------------------------------------------
    print("\n" + "=" * 80)
    print("STGNN Results per Test Station")
    print("=" * 80)

    # Create display table with station_id and metrics
    display_df = per_station[["station_id", "rmse", "mae", "r2", "skill_nwp"]].copy()
    display_df.columns = ["Station ID", "RMSE", "MAE", "R²", "Skill_NWP"]

    # Add mean row
    mean_row = pd.DataFrame({
        "Station ID": ["MEAN"],
        "RMSE": [display_df["RMSE"].mean()],
        "MAE": [display_df["MAE"].mean()],
        "R²": [display_df["R²"].mean()],
        "Skill_NWP": [display_df["Skill_NWP"].mean()],
    })
    display_with_mean = pd.concat([display_df, mean_row], ignore_index=True)

    # Format numbers for better readability
    display_with_mean["RMSE"] = display_with_mean["RMSE"].map("{:.4f}".format)
    display_with_mean["MAE"] = display_with_mean["MAE"].map("{:.4f}".format)
    display_with_mean["R²"] = display_with_mean["R²"].map("{:.4f}".format)
    # Handle NaN values in Skill_NWP
    display_with_mean["Skill_NWP"] = display_with_mean["Skill_NWP"].apply(
        lambda x: "{:.4f}".format(x) if not pd.isna(x) else "N/A"
    )

    # Print the table
    print(display_with_mean.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
