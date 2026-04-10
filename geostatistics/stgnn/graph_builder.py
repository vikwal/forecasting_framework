"""
HeterogeneousGraphBuilder — constructs the static PyG HeteroData graph topology.

The graph is built **once** at the start of training from station and NWP grid
point locations.  Node feature tensors are populated per-sample by the data
loader / sampler; only the edge indices and edge attributes live here.

Node types
----------
  "station"  — weather stations (neighbours + targets)
  "icond2"   — ICON-D2 NWP grid point nodes
  "ecmwf"    — ECMWF HRES NWP grid point nodes

Edge types (directed)
---------------------
  ("station", "near",    "station")   bidirectional station ↔ station
  ("icond2",  "informs", "station")   unidirectional NWP → station
  ("ecmwf",   "informs", "station")   unidirectional NWP → station
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from .config import GraphConfig
from .utils.spatial import (
    delaunay_edges,
    edge_features,
    geodesic_knn,
    pairwise_geodesic_km,
)


class HeterogeneousGraphBuilder:
    """
    Builds a PyG HeteroData graph from station and NWP grid point locations.

    This class is called **once** at startup to build the static graph topology.
    Node feature tensors are populated per-sample by the training sampler.

    Parameters
    ----------
    config : GraphConfig
        Graph construction hyperparameters.
    """

    def __init__(self, config: GraphConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        station_coords: np.ndarray,
        station_altitudes: np.ndarray,
        icond2_grid_coords: np.ndarray,
        ecmwf_grid_coords: np.ndarray,
        icond2_altitudes: np.ndarray | None = None,
        ecmwf_altitudes: np.ndarray | None = None,
    ) -> HeteroData:
        """
        Build and return the static HeteroData graph.

        Parameters
        ----------
        station_coords :     (N_stations, 2)  [lat, lon] degrees
        station_altitudes :  (N_stations,)    metres a.s.l.
        icond2_grid_coords : (N_icond2, 2)    [lat, lon] degrees
        ecmwf_grid_coords :  (N_ecmwf, 2)    [lat, lon] degrees
        icond2_altitudes :   (N_icond2,)      optional, metres a.s.l.
        ecmwf_altitudes :    (N_ecmwf,)       optional, metres a.s.l.

        Returns
        -------
        HeteroData
            Graph with edge_index and edge_attr populated for all edge types.
            Node feature tensors are **not** set here (done by the sampler).
        """
        data = HeteroData()

        # Store coordinates as node metadata (not used by the model but handy)
        data["station"].coords = torch.from_numpy(station_coords.astype(np.float32))
        data["station"].altitude = torch.from_numpy(station_altitudes.astype(np.float32))
        data["icond2"].coords = torch.from_numpy(icond2_grid_coords.astype(np.float32))
        data["ecmwf"].coords = torch.from_numpy(ecmwf_grid_coords.astype(np.float32))

        if icond2_altitudes is not None:
            data["icond2"].altitude = torch.from_numpy(icond2_altitudes.astype(np.float32))
        if ecmwf_altitudes is not None:
            data["ecmwf"].altitude = torch.from_numpy(ecmwf_altitudes.astype(np.float32))

        # --- station ↔ station edges ---
        s2s_ei, s2s_ea = self._build_station_edges(station_coords, station_altitudes)
        data["station", "near", "station"].edge_index = s2s_ei
        data["station", "near", "station"].edge_attr = s2s_ea

        # --- ICON-D2 → station edges ---
        i2s_ei, i2s_ea = self._build_nwp_to_station_edges(
            nwp_coords=icond2_grid_coords,
            station_coords=station_coords,
            nwp_altitudes=icond2_altitudes,
            station_altitudes=station_altitudes,
            k=self.cfg.next_n_icond2_grid_points,
        )
        data["icond2", "informs", "station"].edge_index = i2s_ei
        data["icond2", "informs", "station"].edge_attr = i2s_ea

        # --- ECMWF → station edges ---
        e2s_ei, e2s_ea = self._build_nwp_to_station_edges(
            nwp_coords=ecmwf_grid_coords,
            station_coords=station_coords,
            nwp_altitudes=ecmwf_altitudes,
            station_altitudes=station_altitudes,
            k=self.cfg.next_n_ecmwf_grid_points,
        )
        data["ecmwf", "informs", "station"].edge_index = e2s_ei
        data["ecmwf", "informs", "station"].edge_attr = e2s_ea

        return data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_station_edges(
        self,
        coords: np.ndarray,
        altitudes: np.ndarray,
    ) -> tuple[Tensor, Tensor]:
        """
        Build bidirectional station ↔ station edges.

        Returns
        -------
        edge_index : (2, 2E) long tensor — both directions included
        edge_attr  : (2E, F) float32 tensor
        """
        if self.cfg.station_connectivity == "delaunay":
            undirected = delaunay_edges(coords)          # (E, 2) i < j
        elif self.cfg.station_connectivity == "knn":
            # k+1 because the point itself is the closest neighbour
            _, idx = geodesic_knn(coords, coords, k=self.cfg.station_k + 1)
            pairs: set[tuple[int, int]] = set()
            for i, row in enumerate(idx):
                for j in row:
                    if j != i:
                        pairs.add((min(i, int(j)), max(i, int(j))))
            undirected = np.array(sorted(pairs), dtype=np.int64)
        else:
            raise ValueError(f"Unknown station_connectivity: {self.cfg.station_connectivity!r}")

        # Build directed edge pairs: both (i→j) and (j→i)
        src = np.concatenate([undirected[:, 0], undirected[:, 1]])
        dst = np.concatenate([undirected[:, 1], undirected[:, 0]])

        src_alt = altitudes[src] if self.cfg.use_altitude_diff else None
        dst_alt = altitudes[dst] if self.cfg.use_altitude_diff else None

        # Compute global max distance for consistent normalisation (geodesic, per-edge)
        from .utils.spatial import geodesic_km
        all_dists_flat = geodesic_km(
            coords[undirected[:, 0], 0], coords[undirected[:, 0], 1],
            coords[undirected[:, 1], 0], coords[undirected[:, 1], 1],
        )
        max_dist = float(all_dists_flat.max()) if len(all_dists_flat) > 0 else 1.0

        ea = edge_features(
            src_coords=coords[src],
            dst_coords=coords[dst],
            src_alt=src_alt,
            dst_alt=dst_alt,
            max_dist_km=max_dist,
            use_distance=self.cfg.use_distance_features,
            use_direction=self.cfg.use_direction_features,
            use_altitude_diff=self.cfg.use_altitude_diff,
        )

        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        edge_attr = torch.from_numpy(ea)
        return edge_index, edge_attr

    def _build_nwp_to_station_edges(
        self,
        nwp_coords: np.ndarray,
        station_coords: np.ndarray,
        nwp_altitudes: np.ndarray | None,
        station_altitudes: np.ndarray,
        k: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Build directed nwp_node → station edges using k-nearest-neighbour lookup.

        Each station gets connected to its k nearest NWP grid points.
        Direction: NWP grid point → station (source=NWP, target=station).

        Returns
        -------
        edge_index : (2, N_stations * k) — row 0 = NWP indices, row 1 = station indices
        edge_attr  : (N_stations * k, F)
        """
        dist_km, nwp_idx = geodesic_knn(nwp_coords, station_coords, k=k)
        # dist_km : (N_stations, k), nwp_idx : (N_stations, k)

        N_stations = len(station_coords)
        station_idx = np.repeat(np.arange(N_stations), k)   # (N_stations*k,)
        nwp_idx_flat = nwp_idx.reshape(-1)                   # (N_stations*k,)

        src_alt = nwp_altitudes[nwp_idx_flat] if (
            self.cfg.use_altitude_diff and nwp_altitudes is not None
        ) else None
        dst_alt = station_altitudes[station_idx] if (
            self.cfg.use_altitude_diff and nwp_altitudes is not None
        ) else None

        max_dist = float(dist_km.max()) + 1e-8  # geodesic km

        ea = edge_features(
            src_coords=nwp_coords[nwp_idx_flat],
            dst_coords=station_coords[station_idx],
            src_alt=src_alt,
            dst_alt=dst_alt,
            max_dist_km=max_dist,
            use_distance=self.cfg.use_distance_features,
            use_direction=self.cfg.use_direction_features,
            use_altitude_diff=self.cfg.use_altitude_diff,
        )

        edge_index = torch.tensor(
            np.stack([nwp_idx_flat, station_idx], axis=0), dtype=torch.long
        )
        edge_attr = torch.from_numpy(ea)
        return edge_index, edge_attr

    # ------------------------------------------------------------------
    # Utility: rebuild station subgraph for a node subset
    # ------------------------------------------------------------------

    def subgraph_station_edges(
        self,
        data: HeteroData,
        station_subset: list[int] | np.ndarray,
    ) -> tuple[Tensor, Tensor]:
        """
        Extract station↔station edges restricted to a subset of station indices.

        Useful for building subgraph samples during training.

        Parameters
        ----------
        data :           Full HeteroData returned by build()
        station_subset : List of station indices to keep

        Returns
        -------
        edge_index : remapped (2, E') long tensor
        edge_attr  : (E', F) float32 tensor
        """
        subset = torch.tensor(station_subset, dtype=torch.long)
        full_ei = data["station", "near", "station"].edge_index
        full_ea = data["station", "near", "station"].edge_attr

        # Build a set of kept nodes for fast membership test
        keep = set(station_subset)
        mask = torch.tensor(
            [(s.item() in keep and d.item() in keep)
             for s, d in zip(full_ei[0], full_ei[1])],
            dtype=torch.bool,
        )
        sub_ei = full_ei[:, mask]
        sub_ea = full_ea[mask]

        # Remap node indices to 0..len(subset)-1
        remap = {old: new for new, old in enumerate(station_subset)}
        remapped = torch.tensor(
            [[remap[i.item()] for i in sub_ei[0]],
             [remap[j.item()] for j in sub_ei[1]]],
            dtype=torch.long,
        )
        return remapped, sub_ea
