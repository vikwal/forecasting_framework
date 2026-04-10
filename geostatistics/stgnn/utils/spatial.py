"""
Spatial utility functions: geodesic distance, bearing, KNN helpers,
and Delaunay triangulation for station connectivity.

Distances are computed with pyproj.Geod (WGS-84 ellipsoid), which is more
accurate than the spherical haversine approximation — relevant for edge
features and NWP grid-point lookup over regional domains.
"""
from __future__ import annotations

import numpy as np
from pyproj import Geod
from scipy.spatial import Delaunay

_GEOD = Geod(ellps="WGS84")


def geodesic_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorised geodesic distance in kilometres (WGS-84 ellipsoid).

    All inputs must be broadcastable to the same shape.

    Args:
        lat1, lon1: source coordinates in degrees
        lat2, lon2: destination coordinates in degrees

    Returns:
        Distance array in km, same shape as inputs after broadcasting.
    """
    lat1, lon1, lat2, lon2 = np.broadcast_arrays(lat1, lon1, lat2, lon2)
    shape = lat1.shape
    _, _, dist_m = _GEOD.inv(
        lon1.ravel(), lat1.ravel(),
        lon2.ravel(), lat2.ravel(),
    )
    return (dist_m / 1000.0).reshape(shape)


def bearing_deg(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Initial bearing from (lat1, lon1) to (lat2, lon2) in degrees [0, 360).

    Uses the WGS-84 geodesic (pyproj.Geod.inv).
    0° = North, 90° = East.
    """
    lat1, lon1, lat2, lon2 = np.broadcast_arrays(lat1, lon1, lat2, lon2)
    shape = lat1.shape
    az, _, _ = _GEOD.inv(
        lon1.ravel(), lat1.ravel(),
        lon2.ravel(), lat2.ravel(),
    )
    return (az % 360).reshape(shape)


def pairwise_geodesic_km(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise geodesic distances between two sets of (lat, lon) points.

    Args:
        coords_a: (N, 2) array of [lat, lon] in degrees
        coords_b: (M, 2) array of [lat, lon] in degrees

    Returns:
        (N, M) distance matrix in km
    """
    N = len(coords_a)
    M = len(coords_b)
    # Repeat/tile to get all pairs
    lat1 = np.repeat(coords_a[:, 0], M)          # (N*M,)
    lon1 = np.repeat(coords_a[:, 1], M)
    lat2 = np.tile(coords_b[:, 0], N)             # (N*M,)
    lon2 = np.tile(coords_b[:, 1], N)
    _, _, dist_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    return (dist_m / 1000.0).reshape(N, M)


def geodesic_knn(
    ref_coords: np.ndarray,
    query_coords: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the k nearest neighbours in ``ref_coords`` for each point in ``query_coords``
    using geodesic (WGS-84) distances.

    For large grids this is O(N*M) — acceptable for one-time graph construction
    with a few hundred stations and a few thousand grid points.

    Args:
        ref_coords:   (N, 2) [lat, lon] degrees — the reference set (e.g. NWP grid)
        query_coords: (M, 2) [lat, lon] degrees — the query set (e.g. stations)
        k:            number of nearest neighbours

    Returns:
        distances_km: (M, k) geodesic distances in km
        indices:      (M, k) integer indices into ref_coords
    """
    dist_matrix = pairwise_geodesic_km(query_coords, ref_coords)  # (M, N)
    idx = np.argpartition(dist_matrix, k, axis=1)[:, :k]          # (M, k) unsorted
    # Sort by distance within each row
    row_idx = np.arange(len(query_coords))[:, None]
    dists = dist_matrix[row_idx, idx]
    order = np.argsort(dists, axis=1)
    idx = idx[row_idx, order]
    dists = dists[row_idx, order]
    return dists, idx


def delaunay_edges(coords: np.ndarray) -> np.ndarray:
    """
    Compute undirected edge pairs from a 2-D Delaunay triangulation.

    The triangulation is performed in a local projected space (lat/lon treated
    as Euclidean) which introduces small distortions but is adequate for
    regional networks spanning a few hundred kilometres.

    Args:
        coords: (N, 2) array of [lat, lon] (or any 2-D coordinates)

    Returns:
        (E, 2) array of undirected edge pairs (i, j) with i < j
    """
    tri = Delaunay(coords)
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            i, j = int(simplex[a]), int(simplex[b])
            edges.add((min(i, j), max(i, j)))
    return np.array(sorted(edges), dtype=np.int64)


def edge_features(
    src_coords: np.ndarray,
    dst_coords: np.ndarray,
    src_alt: np.ndarray | None = None,
    dst_alt: np.ndarray | None = None,
    max_dist_km: float | None = None,
    use_distance: bool = True,
    use_direction: bool = True,
    use_altitude_diff: bool = False,
) -> np.ndarray:
    """
    Compute edge feature vectors for directed edges src → dst.

    Args:
        src_coords:      (E, 2) [lat, lon] in degrees for edge sources
        dst_coords:      (E, 2) [lat, lon] in degrees for edge destinations
        src_alt:         (E,) altitude in metres for sources (optional)
        dst_alt:         (E,) altitude in metres for destinations (optional)
        max_dist_km:     normalisation factor for distance; if None uses max over edges
        use_distance:    include normalised distance feature
        use_direction:   include sin/cos of bearing feature
        use_altitude_diff: include normalised altitude difference feature

    Returns:
        (E, F) float32 feature matrix
    """
    lat1, lon1 = src_coords[:, 0], src_coords[:, 1]
    lat2, lon2 = dst_coords[:, 0], dst_coords[:, 1]

    parts: list[np.ndarray] = []

    if use_distance:
        dist = geodesic_km(lat1, lon1, lat2, lon2)
        norm = max_dist_km if max_dist_km is not None else (dist.max() + 1e-8)
        parts.append((dist / norm).reshape(-1, 1))

    if use_direction:
        brg = bearing_deg(lat1, lon1, lat2, lon2)
        parts.append(np.sin(np.radians(brg)).reshape(-1, 1))
        parts.append(np.cos(np.radians(brg)).reshape(-1, 1))

    if use_altitude_diff and src_alt is not None and dst_alt is not None:
        diff = (dst_alt - src_alt).reshape(-1, 1)
        # rough normalisation: ±3000 m range → ±1
        parts.append(np.clip(diff / 3000.0, -1.0, 1.0))

    if not parts:
        # Fallback: constant 1.0 so edge_dim >= 1
        parts.append(np.ones((len(src_coords), 1)))

    return np.concatenate(parts, axis=1).astype(np.float32)
