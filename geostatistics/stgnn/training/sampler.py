"""
Training task sampler: IGNNK-style random node masking, run-based temporal structure.

Shape conventions used throughout
-----------------------------------
meas_hist      : (H,     N_sub,  M)   — measurements, history only
i2_full        : (N_sub, 96,    I2)   — ICON-D2 NWP for station subset, full 96 steps
e2_full        : (N_sub, 96,    E2)   — ECMWF NWP for station subset, full 96 steps
icond2_nwp     : (96,    N_grid, I2)  — ICON-D2 NWP for all grid nodes
ecmwf_nwp      : (96,    N_ecmwf,E2) — ECMWF NWP for all ECMWF nodes

NOTE on numpy fancy indexing:
  grid_icond2_runs[r, :, nearest_array, :] — because the advanced index (nearest_array)
  is surrounded by basic slices, numpy puts the advanced-index dimension FIRST:
  result shape = (N_sub, 48, F), NOT (48, N_sub, F).
  We therefore concatenate along axis=1 to get (N_sub, 96, F).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from ..config import ModelConfig
from ..graph_builder import HeterogeneousGraphBuilder


@dataclass
class SampleBatch:
    data: HeteroData
    target_mask: Tensor    # (N_stations_in_sample,) bool
    ground_truth: Tensor   # (N_target, forecast_horizon)


class TrainingSampler:
    def __init__(
        self,
        model_config: ModelConfig,
        graph_builder: HeterogeneousGraphBuilder,
        base_graph: HeteroData,
        target_feat_idx: int = 0,
        station_coords: np.ndarray | None = None,  # (N_all, 2) [lat, lon] raw degrees
    ) -> None:
        self.cfg = model_config
        self.tc  = model_config.training
        self.builder    = graph_builder
        self.base_graph = base_graph
        self.te = model_config.temporal_encoding
        self.target_feat_idx = target_feat_idx
        self.station_coords  = station_coords  # needed for radius filtering

    # ------------------------------------------------------------------

    def _neighbors_within_radius(
        self,
        target_global_indices: list[int],
        candidate_global_indices: list[int],
    ) -> list[int]:
        """
        Filter candidates to those within neighbor_radius_km of ANY target station.
        If radius is None or station_coords are not set, return candidates unchanged.
        """
        radius = self.tc.neighbor_radius_km
        if radius is None or self.station_coords is None:
            return candidate_global_indices

        from ..utils.spatial import geodesic_km

        t_lats = self.station_coords[target_global_indices, 0]
        t_lons = self.station_coords[target_global_indices, 1]

        kept = []
        for ci in candidate_global_indices:
            c_lat = self.station_coords[ci, 0]
            c_lon = self.station_coords[ci, 1]
            dists = geodesic_km(t_lats, t_lons,
                                np.full(len(t_lats), c_lat),
                                np.full(len(t_lats), c_lon))
            if dists.min() <= radius:
                kept.append(ci)
        return kept

    # ------------------------------------------------------------------

    def sample_train(
        self,
        r_curr: int,
        r_hist: int,
        t_run_abs: int,
        station_meas: np.ndarray,          # (T, N_stations, M)
        station_nearest_grid: np.ndarray,  # (N_stations,) int
        grid_icond2_runs: np.ndarray,      # (R, 48, N_grid, I2)
        station_ecmwf_nwp: np.ndarray,    # (T, N_stations, E2)
        station_static: np.ndarray,        # (N_stations, S-1)
        ecmwf_nwp: np.ndarray,            # (T, N_ecmwf, E2)
        icond2_static: np.ndarray,
        ecmwf_static: np.ndarray,
        train_station_indices: list[int],
    ) -> SampleBatch:
        H_hist  = self.cfg.history_length
        H_fore  = self.cfg.forecast_horizon
        t_hist_abs = t_run_abs - H_hist
        N_train = len(train_station_indices)

        # Random station selection
        n_target = random.randint(
            min(self.tc.min_target_stations, N_train),
            min(self.tc.max_target_stations, N_train),
        )
        target_local    = sorted(random.sample(range(N_train), n_target))
        target_global_list = [train_station_indices[i] for i in target_local]

        neighbour_local = [i for i in range(N_train) if i not in set(target_local)]
        neighbour_global = [train_station_indices[i] for i in neighbour_local]

        # Radius filter: keep only neighbours within neighbor_radius_km of any target
        neighbour_global = self._neighbors_within_radius(target_global_list, neighbour_global)
        neighbour_local  = [i for i in neighbour_local
                            if train_station_indices[i] in set(neighbour_global)]

        if self.tc.subsample_neighbors and len(neighbour_local) > self.tc.max_neighbor_stations:
            neighbour_local  = sorted(random.sample(neighbour_local, self.tc.max_neighbor_stations))
            neighbour_global = [train_station_indices[i] for i in neighbour_local]

        all_local  = sorted(neighbour_local + target_local)
        all_global = [train_station_indices[i] for i in all_local]
        target_global = set(target_global_list)
        target_mask_bool = torch.tensor([g in target_global for g in all_global], dtype=torch.bool)
        target_mask_np   = target_mask_bool.numpy()

        # ICON-D2: fancy indexing puts advanced dim first → (N_sub, 48, I2)
        nearest_for_sub = station_nearest_grid[all_global]
        i2_hist = grid_icond2_runs[r_hist, :, nearest_for_sub, :]   # (N_sub, 48, I2)
        i2_curr = grid_icond2_runs[r_curr, :, nearest_for_sub, :]   # (N_sub, 48, I2)
        i2_full = np.concatenate([i2_hist, i2_curr], axis=1)        # (N_sub, 96, I2)

        # Grid NWP: plain indexing → (48, N_grid, I2), concat → (96, N_grid, I2)
        i2_grid_full = np.concatenate([
            grid_icond2_runs[r_hist],
            grid_icond2_runs[r_curr],
        ], axis=0)

        # ECMWF
        e2_full      = station_ecmwf_nwp[t_hist_abs:t_run_abs + H_fore, :, :][:, all_global, :]
        e2_full      = e2_full.transpose(1, 0, 2)                    # (N_sub, 96, E2)
        e2_grid_full = ecmwf_nwp[t_hist_abs:t_run_abs + H_fore]      # (96, N_ecmwf, E2)

        # Measurements: history only, (H, N_sub, M)
        meas_hist = station_meas[t_hist_abs:t_run_abs, :, :][:, all_global, :].copy()

        # Ground truth
        gt = station_meas[t_run_abs:t_run_abs + H_fore, :, self.target_feat_idx][:, all_global]
        gt_target    = gt[:, target_mask_np].T
        ground_truth = torch.from_numpy(gt_target.copy().astype(np.float32))

        # Zero target measurements
        meas_hist[:, target_mask_np, :] = 0.0

        stat_sub  = station_static[all_global, :]
        type_ind  = (~target_mask_bool).float().unsqueeze(1).numpy()
        stat_full = np.concatenate([stat_sub, type_ind], axis=1)

        data = self._make_data(
            all_global=all_global,
            meas_hist=meas_hist,
            i2_full=i2_full,
            e2_full=e2_full,
            stat_full=stat_full,
            icond2_nwp=i2_grid_full,
            ecmwf_nwp=e2_grid_full,
            icond2_static=icond2_static,
            ecmwf_static=ecmwf_static,
        )
        return SampleBatch(data=data, target_mask=target_mask_bool, ground_truth=ground_truth)

    def sample_val(
        self,
        r_curr: int,
        r_hist: int,
        t_run_abs: int,
        station_meas: np.ndarray,
        station_nearest_grid: np.ndarray,
        grid_icond2_runs: np.ndarray,
        station_ecmwf_nwp: np.ndarray,
        station_static: np.ndarray,
        ecmwf_nwp: np.ndarray,
        icond2_static: np.ndarray,
        ecmwf_static: np.ndarray,
        train_station_indices: list[int],
        val_station_indices: list[int],
    ) -> SampleBatch:
        H_hist = self.cfg.history_length
        H_fore = self.cfg.forecast_horizon
        t_hist_abs = t_run_abs - H_hist

        # Radius filter: only include training stations within radius of any val station
        neighbor_train = self._neighbors_within_radius(val_station_indices, train_station_indices)
        all_global = neighbor_train + val_station_indices
        N_train    = len(neighbor_train)
        N_all      = len(all_global)

        target_mask_bool = torch.zeros(N_all, dtype=torch.bool)
        target_mask_bool[N_train:] = True

        nearest_for_all = station_nearest_grid[all_global]
        i2_hist = grid_icond2_runs[r_hist, :, nearest_for_all, :]   # (N_all, 48, I2)
        i2_curr = grid_icond2_runs[r_curr, :, nearest_for_all, :]   # (N_all, 48, I2)
        i2_full = np.concatenate([i2_hist, i2_curr], axis=1)        # (N_all, 96, I2)

        i2_grid_full = np.concatenate([
            grid_icond2_runs[r_hist],
            grid_icond2_runs[r_curr],
        ], axis=0)

        e2_full      = station_ecmwf_nwp[t_hist_abs:t_run_abs + H_fore, :, :][:, all_global, :]
        e2_full      = e2_full.transpose(1, 0, 2)                    # (N_all, 96, E2)
        e2_grid_full = ecmwf_nwp[t_hist_abs:t_run_abs + H_fore]

        meas_hist = station_meas[t_hist_abs:t_run_abs, :, :][:, all_global, :].copy()
        meas_hist[:, N_train:, :] = 0.0

        gt = station_meas[t_run_abs:t_run_abs + H_fore, :, self.target_feat_idx][:, all_global]
        gt_val       = gt[:, N_train:].T
        ground_truth = torch.from_numpy(gt_val.copy().astype(np.float32))

        stat_sub  = station_static[all_global, :]
        type_ind  = (~target_mask_bool).float().unsqueeze(1).numpy()
        stat_full = np.concatenate([stat_sub, type_ind], axis=1)

        data = self._make_data(
            all_global=all_global,
            meas_hist=meas_hist,
            i2_full=i2_full,
            e2_full=e2_full,
            stat_full=stat_full,
            icond2_nwp=i2_grid_full,
            ecmwf_nwp=e2_grid_full,
            icond2_static=icond2_static,
            ecmwf_static=ecmwf_static,
        )
        return SampleBatch(data=data, target_mask=target_mask_bool, ground_truth=ground_truth)

    # ------------------------------------------------------------------

    def _make_data(
        self,
        all_global: list[int],
        meas_hist: np.ndarray,      # (H,     N_sub,  M)
        i2_full: np.ndarray,        # (N_sub, 96,    I2)
        e2_full: np.ndarray,        # (N_sub, 96,    E2)
        stat_full: np.ndarray,      # (N_sub, S)
        icond2_nwp: np.ndarray,     # (96,    N_grid, I2)
        ecmwf_nwp: np.ndarray,      # (96,    N_ecmwf,E2)
        icond2_static: np.ndarray,
        ecmwf_static: np.ndarray,
    ) -> HeteroData:
        H_hist = meas_hist.shape[0]
        N_sub  = i2_full.shape[0]
        T_total = i2_full.shape[1]   # 96
        M  = meas_hist.shape[2]
        I2 = i2_full.shape[2]
        E2 = e2_full.shape[2]

        data = HeteroData()

        # Station–station edges
        sub_s2s_ei, sub_s2s_ea = self.builder.subgraph_station_edges(self.base_graph, all_global)
        data["station", "near", "station"].edge_index = sub_s2s_ei
        data["station", "near", "station"].edge_attr  = sub_s2s_ea

        # NWP → station edges
        remap = {g: l for l, g in enumerate(all_global)}
        for _, edge_key in [
            ("icond2", ("icond2", "informs", "station")),
            ("ecmwf",  ("ecmwf",  "informs", "station")),
        ]:
            full_ei = self.base_graph[edge_key].edge_index
            full_ea = self.base_graph[edge_key].edge_attr
            mask    = torch.tensor([d.item() in remap for d in full_ei[1]], dtype=torch.bool)
            sub_ei  = full_ei[:, mask]
            new_dst = torch.tensor([remap[d.item()] for d in sub_ei[1]], dtype=torch.long)
            data[edge_key].edge_index = torch.stack([sub_ei[0], new_dst], dim=0)
            data[edge_key].edge_attr  = full_ea[mask]

        # Station features
        # i2_full: (N_sub, 96, I2) — already in (N, T, F) layout
        # e2_full: (N_sub, 96, E2) — already in (N, T, F) layout
        # meas_hist: (H, N_sub, M) → transpose to (N_sub, H, M)
        meas_nt = meas_hist.transpose(1, 0, 2)   # (N_sub, H, M)

        te = self.te
        if te == "flat":
            mh  = meas_nt.reshape(N_sub, -1)
            i2f = i2_full.reshape(N_sub, -1)
            e2f = e2_full.reshape(N_sub, -1)
            x_station = np.concatenate([mh, i2f, e2f], axis=1)
        elif te in ("gru", "cnn"):
            # (N_sub, 96, M+I2+E2) — meas zeroed for forecast steps
            meas_padded = np.zeros((N_sub, T_total, M), dtype=np.float32)
            meas_padded[:, :H_hist, :] = meas_nt
            x_station = np.concatenate([meas_padded, i2_full, e2_full], axis=2)
        else:
            raise ValueError(f"Unknown temporal_encoding: {te!r}")

        data["station"].x      = torch.from_numpy(x_station.astype(np.float32))
        data["station"].static = torch.from_numpy(stat_full.astype(np.float32))

        # ICON-D2 node features: (96, N_grid, I2) → transpose to (N_grid, 96, I2)
        N_i = icond2_nwp.shape[1]
        if te == "flat":
            icond2_x = icond2_nwp.transpose(1, 0, 2).reshape(N_i, -1)
        else:
            icond2_x = icond2_nwp.transpose(1, 0, 2)
        data["icond2"].x      = torch.from_numpy(icond2_x.astype(np.float32))
        data["icond2"].static = torch.from_numpy(icond2_static.astype(np.float32))

        # ECMWF node features: (96, N_ecmwf, E2) → transpose to (N_ecmwf, 96, E2)
        N_e = ecmwf_nwp.shape[1]
        if te == "flat":
            ecmwf_x = ecmwf_nwp.transpose(1, 0, 2).reshape(N_e, -1)
        else:
            ecmwf_x = ecmwf_nwp.transpose(1, 0, 2)
        data["ecmwf"].x      = torch.from_numpy(ecmwf_x.astype(np.float32))
        data["ecmwf"].static = torch.from_numpy(ecmwf_static.astype(np.float32))

        return data