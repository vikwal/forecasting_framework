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

from dataclasses import dataclass

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
from .utils.topo_features import load_topo_node_features


@dataclass
class FoldTopology:
    """
    Station-graph topology for one train/target split.

    Delaunay triangulation and node subsetting do not commute: removing a point
    can only ever destroy edges, never create them, so the subgraph induced by a
    node subset is a strict subset of the triangulation over that subset.  Taking
    the induced subgraph of a triangulation over *all* stations therefore both
    thins the training graph and lets the held-out stations' positions leak into
    its topology.  This structure avoids that by triangulating the training
    stations on their own and wiring every other station in the way it would be
    wired at inference time.

    backbone : (E, 2) global index pairs, i < j — Delaunay over the training
               stations only.  Independent of where the held-out stations lie.
    attach   : per station, the training stations it connects to when inserted
               into that network.  For a training station this is its own
               backbone neighbourhood; for a held-out station it is the set of
               edges it gains in Delaunay(train ∪ {station}) — exactly the wiring
               a single new site would receive in deployment.
    """
    backbone: np.ndarray
    attach: dict[int, np.ndarray]


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
        # Populated by build(); consumed by build_fold_topology / sample_station_edges.
        self.station_coords: np.ndarray | None = None
        self.station_pair_attr: np.ndarray | None = None   # (N, N, F) directed
        self._max_dist_km: float = 1.0
        self._s2s_feat_dim: int = 1

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
        station_ids: list[str] | None = None,
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
        station_ids :        (N_stations,) DWD station IDs, same order as
                              station_coords — required when
                              ``self.cfg.topo_feature_names`` is non-empty, to
                              join topographic node features onto s2s edges.

        Returns
        -------
        HeteroData
            Graph with edge_index and edge_attr populated for all edge types.
            Node feature tensors are **not** set here (done by the sampler).
        """
        data = HeteroData()

        topo_node_feats: dict[str, np.ndarray] | None = None
        if self.cfg.topo_feature_names:
            if station_ids is None:
                raise ValueError(
                    "GraphConfig.topo_feature_names is set but station_ids was not "
                    "passed to HeterogeneousGraphBuilder.build()."
                )
            topo_node_feats = load_topo_node_features(
                self.cfg.topo_features_path, station_ids, self.cfg.topo_feature_names,
            )

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
        s2s_ei, s2s_ea = self._build_station_edges(
            station_coords, station_altitudes, topo_node_feats,
        )
        data["station", "near", "station"].edge_index = s2s_ei
        data["station", "near", "station"].edge_attr = s2s_ea

        # --- pairwise edge-attribute table, for per-sample edge assembly ---
        # station_graph_mode="attach" rebuilds the station edges per sample, so the
        # attributes of every possible pair have to be available at that point.
        # The table is small (N² × F floats) and computing it once here keeps the
        # distance normaliser — and with it the Gaussian edge kernel — identical
        # for an edge no matter which sample it turns up in.
        self.station_coords = station_coords
        if self.cfg.station_connectivity != "none":
            self.station_pair_attr = self._build_pair_attr_table(
                station_coords, station_altitudes, topo_node_feats,
            )

        # --- ICON-D2 → station edges ---
        i2s_ei, i2s_ea, i2s_max_dist_km = self._build_nwp_to_station_edges(
            nwp_coords=icond2_grid_coords,
            station_coords=station_coords,
            nwp_altitudes=icond2_altitudes,
            station_altitudes=station_altitudes,
            k=self.cfg.next_n_icond2_grid_points,
        )
        data["icond2", "informs", "station"].edge_index = i2s_ei
        data["icond2", "informs", "station"].edge_attr = i2s_ea
        # Distance-column normaliser (see _build_nwp_to_station_edges), persisted
        # so callers can recover physical km from edge_attr[:, 0] without
        # re-deriving it from raw coordinates. Only consumer today: DCRNN's
        # nwp_aggregation="idw_alt" (geostatistics/dcrnn/model/nwp_attention.py),
        # which combines this with the (fixed-normalisation) altitude-diff column
        # and therefore cannot rely on the distance normaliser cancelling out the
        # way plain "idw" does. Harmless for every other consumer (MTGNN/WaveNet
        # ignore unrecognised HeteroData attributes).
        data["icond2", "informs", "station"].max_dist_km = i2s_max_dist_km

        # --- ECMWF → station edges ---
        if self.cfg.next_n_ecmwf_grid_points > 0:
            e2s_ei, e2s_ea, e2s_max_dist_km = self._build_nwp_to_station_edges(
                nwp_coords=ecmwf_grid_coords,
                station_coords=station_coords,
                nwp_altitudes=ecmwf_altitudes,
                station_altitudes=station_altitudes,
                k=self.cfg.next_n_ecmwf_grid_points,
            )
        else:
            e2s_ei = torch.zeros((2, 0), dtype=torch.long)
            e2s_ea = torch.zeros((0, 1), dtype=torch.float32)
            e2s_max_dist_km = 0.0   # unused: no ecmwf edges to normalise
        data["ecmwf", "informs", "station"].edge_index = e2s_ei
        data["ecmwf", "informs", "station"].edge_attr = e2s_ea
        data["ecmwf", "informs", "station"].max_dist_km = e2s_max_dist_km

        return data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_station_edges(
        self,
        coords: np.ndarray,
        altitudes: np.ndarray,
        topo_node_feats: dict[str, np.ndarray] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Build bidirectional station ↔ station edges.

        Returns
        -------
        edge_index : (2, 2E) long tensor — both directions included
        edge_attr  : (2E, F) float32 tensor
        """
        if self.cfg.station_connectivity == "none":
            # Ablation C: no station↔station message passing at all.
            #
            # An empty edge set is provably safe in DiffConv (dcgru_cell.py):
            #   * out = self.lins[0](x) is an unconditional k=0 self-transform,
            #     so there is always a path that does not touch any edge;
            #   * out_deg.clamp(min=1e-8) removes the division by zero when no
            #     edge writes into out_deg;
            #   * propagate() over zero edges returns a zero tensor, so every
            #     k >= 1 term contributes lins[k](0).
            # The DCGRU therefore degenerates to a plain GRU with a linear input
            # transform, which is exactly the intent of variant C.
            #
            # The edge-feature width is *probed* rather than hard-coded so it
            # tracks use_distance/use_direction/use_altitude_diff **and** the
            # topographic edge columns — with the campaign's edge_features list
            # that is 1 + 2 + 1 + 8 = 12, not the 4 a flags-only count gives.
            _topo_names = self.cfg.topo_feature_names if topo_node_feats else None
            _zero_idx = np.zeros(1, dtype=np.int64)
            probe = edge_features(
                src_coords=coords[:1],
                dst_coords=coords[:1],
                src_alt=(altitudes[:1] if self.cfg.use_altitude_diff else None),
                dst_alt=(altitudes[:1] if self.cfg.use_altitude_diff else None),
                max_dist_km=1.0,
                use_distance=self.cfg.use_distance_features,
                use_direction=self.cfg.use_direction_features,
                use_altitude_diff=self.cfg.use_altitude_diff,
                topo_node_feats=topo_node_feats,
                topo_feature_names=_topo_names,
                src_idx=_zero_idx,
                dst_idx=_zero_idx,
            )
            n_feat = int(probe.shape[1])
            self._s2s_feat_dim = n_feat
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, n_feat), dtype=torch.float32),
            )

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
        self._max_dist_km = max_dist

        ea = edge_features(
            src_coords=coords[src],
            dst_coords=coords[dst],
            src_alt=src_alt,
            dst_alt=dst_alt,
            max_dist_km=max_dist,
            use_distance=self.cfg.use_distance_features,
            use_direction=self.cfg.use_direction_features,
            use_altitude_diff=self.cfg.use_altitude_diff,
            topo_node_feats=topo_node_feats,
            topo_feature_names=self.cfg.topo_feature_names,
            src_idx=src,
            dst_idx=dst,
        )

        self._s2s_feat_dim = int(ea.shape[1])
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        edge_attr = torch.from_numpy(ea)
        return edge_index, edge_attr

    def _build_pair_attr_table(
        self,
        coords: np.ndarray,
        altitudes: np.ndarray,
        topo_node_feats: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        """
        Edge attributes for every directed station pair, shaped (N, N, F).

        Uses the same distance normaliser as ``_build_station_edges``, so an edge
        present in both carries bit-identical attributes.
        """
        n = len(coords)
        src = np.repeat(np.arange(n), n)
        dst = np.tile(np.arange(n), n)
        attr = edge_features(
            src_coords=coords[src],
            dst_coords=coords[dst],
            src_alt=(altitudes[src] if self.cfg.use_altitude_diff else None),
            dst_alt=(altitudes[dst] if self.cfg.use_altitude_diff else None),
            max_dist_km=self._max_dist_km,
            use_distance=self.cfg.use_distance_features,
            use_direction=self.cfg.use_direction_features,
            use_altitude_diff=self.cfg.use_altitude_diff,
            topo_node_feats=topo_node_feats,
            topo_feature_names=self.cfg.topo_feature_names if topo_node_feats else None,
            src_idx=src,
            dst_idx=dst,
        )
        return attr.reshape(n, n, -1)

    # ------------------------------------------------------------------
    # Per-fold topology and per-sample edge assembly
    # ------------------------------------------------------------------

    def build_fold_topology(self, train_indices) -> FoldTopology:
        """
        Triangulate the training stations and record how every station attaches.

        Call once per train/target split; the result is independent of the
        individual sample and is cached by the sampler.  Cost is one Delaunay over
        the training stations plus one per held-out station, a few tens of ms for
        the sizes used here.
        """
        if self.station_coords is None:
            raise RuntimeError("build() must run before build_fold_topology().")

        train = np.asarray(sorted(int(i) for i in train_indices), dtype=np.int64)
        if self.cfg.station_connectivity == "none" or len(train) < 3:
            return FoldTopology(np.zeros((0, 2), dtype=np.int64), {})

        # train is sorted, so local i < j maps to global i < j
        backbone = train[delaunay_edges(self.station_coords[train])]

        adj: dict[int, set[int]] = {int(t): set() for t in train}
        for a, b in backbone:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
        attach = {t: np.array(sorted(nbrs), dtype=np.int64) for t, nbrs in adj.items()}

        train_set = set(int(t) for t in train)
        for v in range(len(self.station_coords)):
            if v in train_set:
                continue
            sub = np.append(train, v)
            local = delaunay_edges(self.station_coords[sub])
            v_local = len(train)
            nbrs = {int(sub[j]) for i, j in local if i == v_local}
            nbrs |= {int(sub[i]) for i, j in local if j == v_local}
            attach[v] = np.array(sorted(nbrs), dtype=np.int64)

        return FoldTopology(backbone.astype(np.int64), attach)

    def sample_station_edges(
        self,
        topo: FoldTopology,
        all_global: list[int],
        target_global,
    ) -> tuple[Tensor, Tensor]:
        """
        Assemble the station ↔ station edges for one sample.

        Neighbour ↔ neighbour edges come from the fold backbone restricted to the
        neighbours present; every target is wired to its attachment neighbours
        among them.  Target ↔ target edges are deliberately absent: a single new
        site has no co-targets to talk to at inference, and training on edges that
        will not exist there is what made the validation graph denser than the
        training one.
        """
        n_feat = (self.station_pair_attr.shape[-1]
                  if self.station_pair_attr is not None else self._s2s_feat_dim)
        empty = (torch.zeros((2, 0), dtype=torch.long),
                 torch.zeros((0, n_feat), dtype=torch.float32))
        if self.cfg.station_connectivity == "none" or not all_global:
            return empty

        n_all = len(self.station_coords)
        targets = np.asarray(sorted(int(g) for g in target_global), dtype=np.int64)
        is_target = np.zeros(n_all, dtype=bool)
        is_target[targets] = True

        in_nb = np.zeros(n_all, dtype=bool)
        in_nb[np.asarray([int(g) for g in all_global], dtype=np.int64)] = True
        in_nb &= ~is_target

        blocks = []
        if len(topo.backbone):
            keep = in_nb[topo.backbone[:, 0]] & in_nb[topo.backbone[:, 1]]
            if keep.any():
                blocks.append(topo.backbone[keep])
        for t in targets:
            nbrs = topo.attach.get(int(t))
            if nbrs is None or not len(nbrs):
                continue
            nbrs = nbrs[in_nb[nbrs]]
            if len(nbrs):
                blocks.append(np.stack([np.full(len(nbrs), t, dtype=np.int64), nbrs], axis=1))

        if not blocks:
            return empty
        undirected = np.concatenate(blocks, axis=0)

        src = np.concatenate([undirected[:, 0], undirected[:, 1]])
        dst = np.concatenate([undirected[:, 1], undirected[:, 0]])

        local = np.full(n_all, -1, dtype=np.int64)
        local[np.asarray([int(g) for g in all_global], dtype=np.int64)] = np.arange(len(all_global))

        edge_index = torch.from_numpy(np.stack([local[src], local[dst]], axis=0))
        edge_attr = torch.from_numpy(np.ascontiguousarray(self.station_pair_attr[src, dst]))
        return edge_index, edge_attr

    def _build_nwp_to_station_edges(
        self,
        nwp_coords: np.ndarray,
        station_coords: np.ndarray,
        nwp_altitudes: np.ndarray | None,
        station_altitudes: np.ndarray,
        k: int,
    ) -> tuple[Tensor, Tensor, float]:
        """
        Build directed nwp_node → station edges using k-nearest-neighbour lookup.

        Each station gets connected to its k nearest NWP grid points.
        Direction: NWP grid point → station (source=NWP, target=station).

        Returns
        -------
        edge_index  : (2, N_stations * k) — row 0 = NWP indices, row 1 = station indices
        edge_attr   : (N_stations * k, F)
        max_dist_km : the distance-column normaliser used in edge_attr[:, 0]
                      (edge_attr[:, 0] == dist_km / max_dist_km) — the caller
                      persists this on the returned HeteroData so it can be
                      recovered later without re-deriving it from coordinates.
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
        return edge_index, edge_attr, max_dist

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
