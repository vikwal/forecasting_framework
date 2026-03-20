"""Spatial interpolation utilities: IDW, Ordinary Kriging, Regression Kriging.

All functions are stateless except KrigingPredictor, which pre-computes and
caches the OK weight vector for a fixed set of neighbour locations so that the
same weights can be reused across many timestamps without re-solving the
kriging system each time.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from pyproj import Transformer as _ProjTransformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def compute_distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Compute the full N×N geodesic distance matrix in km.

    Args:
        lats: Latitudes of N stations.
        lons: Longitudes of N stations.

    Returns:
        Symmetric (N, N) float array; diagonal is zero.
    """
    n = len(lats)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = geodesic((lats[i], lons[i]), (lats[j], lons[j])).km
            dist[i, j] = d
            dist[j, i] = d
    return dist


def compute_utm_coords(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Project lat/lon to UTM EPSG:25832 (meters, Central Europe).

    Returns:
        (N, 2) array of [Easting, Northing] in metres.
    """
    if not _HAS_PYPROJ:
        raise ImportError("pyproj is required for anisotropic kriging. Install with: pip install pyproj")
    transformer = _ProjTransformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    east, north = transformer.transform(lons, lats)
    return np.column_stack([east, north])


def compute_anisotropic_distance_matrix(
    lats: np.ndarray,
    lons: np.ndarray,
    angle_deg: float,
    ratio: float,
) -> np.ndarray:
    """Compute N×N anisotropic distance matrix using geometric anisotropy.

    Applies a coordinate transformation that stretches space along the minor
    axis so that the variogram becomes isotropic in the transformed space:

      1. Project (lat, lon) to UTM (metres).
      2. Decompose each pair's displacement into the major-axis component u
         and minor-axis component v (perpendicular to major).
      3. Scale: v_scaled = v / ratio   (ratio = range_minor / range_major).
      4. Anisotropic distance = sqrt(u² + v_scaled²), converted to km.

    The resulting distances can be fed directly into any isotropic variogram
    model to obtain the anisotropic covariance.

    Args:
        lats:      Latitudes  (N,).
        lons:      Longitudes (N,).
        angle_deg: Direction of the *major* axis (geographic degrees, clockwise
                   from North, e.g. 135 = SE–NW).
        ratio:     range_minor / range_major  (0 < ratio ≤ 1).

    Returns:
        Symmetric (N, N) float array of anisotropic distances in km.
    """
    xy = compute_utm_coords(lats, lons)   # (N, 2): [East, North] in metres
    N = len(lats)
    theta = np.deg2rad(angle_deg)
    sin_t, cos_t = np.sin(theta), np.cos(theta)

    dist = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        dx = xy[:, 0] - xy[i, 0]   # East differences to all stations
        dy = xy[:, 1] - xy[i, 1]   # North differences
        # Project onto major axis (angle_deg from North, clockwise)
        u = dx * sin_t + dy * cos_t
        # Project onto minor axis (perpendicular, 90° clockwise)
        v = dx * cos_t - dy * sin_t
        d = np.sqrt(u ** 2 + (v / ratio) ** 2) / 1000.0   # → km
        dist[i, :] = d
    np.fill_diagonal(dist, 0.0)
    return dist


def get_k_nearest(dist_matrix: np.ndarray, target_idx: int, k: int) -> np.ndarray:
    """Return indices of the k nearest neighbours, excluding the target itself.

    Args:
        dist_matrix: Full (N, N) pairwise distance matrix.
        target_idx:  Index of the station to predict.
        k:           Number of neighbours to select.

    Returns:
        Array of length k with neighbour indices sorted by ascending distance.
    """
    row = dist_matrix[target_idx].copy()
    row[target_idx] = np.inf
    return np.argsort(row)[:k]


# ---------------------------------------------------------------------------
# Variogram models
# ---------------------------------------------------------------------------

def _spherical(h: np.ndarray, nugget: float, psill: float, range_: float) -> np.ndarray:
    """Spherical variogram model γ(h) = nugget + psill * f(h/range)."""
    h = np.asarray(h, dtype=np.float64)
    out = np.where(
        h <= range_,
        nugget + psill * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
        nugget + psill,
    )
    return np.where(h == 0.0, 0.0, out)


def _gaussian(h: np.ndarray, nugget: float, psill: float, range_: float) -> np.ndarray:
    """Gaussian variogram model γ(h) = nugget + psill * (1 - exp(-(h/range)²)).

    The practical range (where γ reaches 95 % of the sill) is range * sqrt(3).
    """
    h = np.asarray(h, dtype=np.float64)
    out = nugget + psill * (1.0 - np.exp(-((h / range_) ** 2)))
    return np.where(h == 0.0, 0.0, out)


def _exponential(h: np.ndarray, nugget: float, psill: float, range_: float) -> np.ndarray:
    """Exponential variogram model γ(h) = nugget + psill * (1 - exp(-h/range)).

    The practical range (95 % of sill) is 3 * range.
    """
    h = np.asarray(h, dtype=np.float64)
    out = nugget + psill * (1.0 - np.exp(-h / range_))
    return np.where(h == 0.0, 0.0, out)


VARIOGRAM_MODELS = {
    "spherical": _spherical,
    "gaussian": _gaussian,
    "exponential": _exponential,
}


def _get_variogram_fn(model: str):
    """Return the variogram function for *model*, falling back to spherical."""
    fn = VARIOGRAM_MODELS.get(model)
    if fn is None:
        logger.warning("Unknown variogram model '%s'. Falling back to 'spherical'.", model)
        fn = _spherical
    return fn


# ---------------------------------------------------------------------------
# Empirical semivariance + fitting
# ---------------------------------------------------------------------------

def compute_empirical_semivariance(
    values_matrix: np.ndarray,
    dist_matrix: np.ndarray,
    n_lags: int = 20,
    max_dist: Optional[float] = None,
) -> tuple:
    """Compute empirical semivariances across all station pairs and timestamps.

    The semivariance for each pair is averaged over all timestamps first, then
    binned by lag distance. This is fully vectorised over pairs and timestamps.

    Args:
        values_matrix: (T, N) array of observed wind speeds; NaN allowed.
        dist_matrix:   (N, N) pairwise distance matrix in km.
        n_lags:        Number of equally-spaced lag bins.
        max_dist:      Upper distance cutoff in km.
                       Defaults to half of the maximum pairwise distance.

    Returns:
        lags:          Bin-centre distances in km (only bins with ≥1 pair).
        semivariances: Mean semivariance per bin.
    """
    N = dist_matrix.shape[0]
    if max_dist is None:
        max_dist = dist_matrix[dist_matrix > 0].max() / 2.0

    bin_edges = np.linspace(0.0, max_dist, n_lags + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dist_matrix[i_idx, j_idx]

    # Restrict to pairs within max_dist
    within = pair_dists <= max_dist
    i_idx, j_idx, pair_dists = i_idx[within], j_idx[within], pair_dists[within]

    # Semivariance per pair, averaged over timestamps (ignore NaN)
    vals_i = values_matrix[:, i_idx]   # (T, M)
    vals_j = values_matrix[:, j_idx]   # (T, M)
    sv_pairs = 0.5 * np.nanmean((vals_i - vals_j) ** 2, axis=0)  # (M,)

    # Bin by distance
    bin_idx = np.searchsorted(bin_edges[1:], pair_dists).clip(0, n_lags - 1)
    sv_sums = np.zeros(n_lags)
    sv_counts = np.zeros(n_lags)
    np.add.at(sv_sums, bin_idx, sv_pairs)
    np.add.at(sv_counts, bin_idx, 1)

    valid = sv_counts > 0
    return bin_centers[valid], sv_sums[valid] / sv_counts[valid]


def fit_global_variogram(
    lags: np.ndarray,
    semivariances: np.ndarray,
    model: str = "spherical",
) -> dict:
    """Fit a variogram model to empirical semivariances using non-linear LS.

    Args:
        lags:          Lag distances in km.
        semivariances: Empirical semivariance values.
        model:         Variogram model name: 'spherical', 'gaussian', 'exponential'.

    Returns:
        dict with keys 'nugget', 'psill', 'range', 'model'.
    """
    model_fn = _get_variogram_fn(model)

    sill_init = float(np.max(semivariances))
    idx80 = np.searchsorted(semivariances, 0.8 * sill_init)
    range_init = float(lags[min(idx80, len(lags) - 1)])
    nugget_init = float(semivariances[0]) * 0.5
    psill_init = sill_init - nugget_init

    p0 = [nugget_init, psill_init, range_init]
    bounds = ([0.0, 1e-6, 1e-6], [sill_init * 3, sill_init * 3, lags[-1] * 2])

    try:
        popt, _ = curve_fit(model_fn, lags, semivariances, p0=p0, bounds=bounds, maxfev=10_000)
        nugget, psill, range_ = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception as exc:
        logger.warning("Variogram curve_fit failed (%s). Using initial parameters.", exc)
        nugget, psill, range_ = nugget_init, psill_init, range_init

    params = {"nugget": nugget, "psill": psill, "range": range_, "model": model}
    logger.info("Global variogram parameters: %s", params)
    return params


# ---------------------------------------------------------------------------
# Kriging weight pre-computation
# ---------------------------------------------------------------------------

class KrigingPredictor:
    """Pre-compute and cache OK weights for a fixed neighbour configuration.

    Because the kriging system depends only on the locations and the variogram
    (not on the observed values), the weight vector is identical for every
    timestamp.  Solving it once per station and reusing it reduces the per-
    prediction cost from O(k³) to O(k).

    The same weight vector is also valid for Regression Kriging because the
    residuals are kriged at the same locations.
    """

    def __init__(
        self,
        neighbor_lats: np.ndarray,
        neighbor_lons: np.ndarray,
        target_lat: float,
        target_lon: float,
        variogram_params: dict,
        nn_dist_precomputed: Optional[np.ndarray] = None,
        target_dist_precomputed: Optional[np.ndarray] = None,
    ) -> None:
        """Build the kriging system and solve for weights.

        Args:
            neighbor_lats/lons:      Coordinates of the k neighbours.
            target_lat/lon:          Coordinates of the target station.
            variogram_params:        Fitted variogram parameters dict.
            nn_dist_precomputed:     (k, k) distance matrix between neighbours.
                                     If None, geodesic distances are computed.
            target_dist_precomputed: (k,) distances from each neighbour to target.
                                     If None, geodesic distances are computed.
        """
        n = len(neighbor_lats)
        nugget = variogram_params["nugget"]
        psill = variogram_params["psill"]
        range_ = variogram_params["range"]
        model_fn = _get_variogram_fn(variogram_params.get("model", "spherical"))

        # Distances between all pairs of neighbours
        if nn_dist_precomputed is not None:
            nn_dist = nn_dist_precomputed
        else:
            nn_dist = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = geodesic(
                        (neighbor_lats[i], neighbor_lons[i]),
                        (neighbor_lats[j], neighbor_lons[j]),
                    ).km
                    nn_dist[i, j] = d
                    nn_dist[j, i] = d

        # Distances from each neighbour to the target
        if target_dist_precomputed is not None:
            dist_to_target = target_dist_precomputed
        else:
            dist_to_target = np.array(
                [
                    geodesic(
                        (neighbor_lats[i], neighbor_lons[i]), (target_lat, target_lon)
                    ).km
                    for i in range(n)
                ]
            )

        # Build the (n+1)×(n+1) ordinary kriging system
        gamma_nn = model_fn(nn_dist, nugget, psill, range_)
        np.fill_diagonal(gamma_nn, 0.0)
        gamma_t = model_fn(dist_to_target, nugget, psill, range_)

        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = gamma_nn
        A[:n, n] = 1.0
        A[n, :n] = 1.0

        b = np.zeros(n + 1)
        b[:n] = gamma_t
        b[n] = 1.0

        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)

        self.weights = sol[:n]   # Lagrange multiplier is sol[n], not needed

    def predict(self, values: np.ndarray) -> float:
        """Apply pre-computed weights to observed neighbour values."""
        return float(np.dot(self.weights, values))


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------

def predict_idw(values: np.ndarray, distances: np.ndarray, power: float = 2.0) -> float:
    """Inverse Distance Weighting prediction.

    Args:
        values:    Observed wind speeds at k neighbour stations.
        distances: Geodesic distances to those neighbours in km.
        power:     IDW exponent (weight = 1 / dist^power).

    Returns:
        Weighted estimate at the target location.
    """
    weights = 1.0 / np.power(distances, power)
    return float(np.dot(weights, values) / weights.sum())


def predict_rk(
    values: np.ndarray,
    neighbor_features: np.ndarray,
    target_features: np.ndarray,
    kriging_predictor: KrigingPredictor,
) -> float:
    """Regression Kriging prediction.

    Steps:
      1. Fit OLS trend  wind_speed ~ features  on the k neighbours.
      2. Compute residuals at the neighbours.
      3. Krige the residuals using the pre-computed OK weights.
      4. Return  trend(target_features) + kriged_residual.

    Args:
        values:             Observed wind speeds at k neighbours (k,).
        neighbor_features:  Feature matrix at k neighbours (k, F).
        target_features:    Feature vector at the target station (F,).
        kriging_predictor:  Pre-built KrigingPredictor for this station.

    Returns:
        RK wind-speed estimate.
    """
    X_n = np.atleast_2d(neighbor_features)
    X_t = np.atleast_2d(target_features)
    reg = LinearRegression().fit(X_n, values)
    residuals = values - reg.predict(X_n)
    kriged_residual = kriging_predictor.predict(residuals)
    trend_target = float(reg.predict(X_t)[0])
    return trend_target + kriged_residual


# ---------------------------------------------------------------------------
# Wind component helpers
# ---------------------------------------------------------------------------

def wind_to_uv(speed: np.ndarray, direction_deg: np.ndarray) -> tuple:
    """Convert wind speed + meteorological direction to u/v components.

    Meteorological convention: direction is the bearing the wind comes FROM,
    measured clockwise from North.
      u = −|ws| · sin(dir_rad)   (positive = westerly → eastward)
      v = −|ws| · cos(dir_rad)   (positive = southerly → northward)

    Args:
        speed:         Wind speed (m/s), any shape.
        direction_deg: Wind direction (degrees, met. convention), same shape.

    Returns:
        u: Zonal component (m/s).
        v: Meridional component (m/s).
    """
    rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


# ---------------------------------------------------------------------------
# LOO-CV loop
# ---------------------------------------------------------------------------

def run_loo_cv(
    values_matrix: np.ndarray,
    timestamps: pd.DatetimeIndex,
    lats: np.ndarray,
    lons: np.ndarray,
    alts: np.ndarray,
    station_ids: list,
    dist_matrix: np.ndarray,
    variogram_params_list: list,
    segment_indices: np.ndarray,
    k: int = 8,
    idw_power: float = 2.0,
    u_matrix: Optional[np.ndarray] = None,
    v_matrix: Optional[np.ndarray] = None,
    rk_feature_names: Optional[list] = None,
    rk_static_features: Optional[dict] = None,
    rk_dynamic_features: Optional[dict] = None,
    aniso_dist_matrix: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Run leave-one-out cross-validation for IDW, OK, and RK over all
    stations and timestamps.

    Kriging weights are rebuilt whenever the active variogram segment changes.
    For ``variogram_segments=1`` this happens once; for ``variogram_segments=N``
    at N−1 segment boundaries; for ``variogram_segments=0`` (per-timestamp) at
    every step.

    Anisotropic kriging:
      If *aniso_dist_matrix* is provided it replaces geodesic distances in the
      kriging system (but NOT for IDW or k-nearest-neighbour selection, which
      always use geodesic distances).  Use
      ``compute_anisotropic_distance_matrix`` to build it.

    RK feature selection:
      When *rk_feature_names* is provided, Regression Kriging uses a
      multi-feature OLS trend.  Features are split into:
        - *rk_static_features*:  dict of name → (N,) array  (e.g. altitude)
        - *rk_dynamic_features*: dict of name → (T, N) array (e.g. temperature)
      If *rk_feature_names* is None, falls back to altitude-only behaviour.

    Args:
        values_matrix:        (T, N) observed wind speeds; NaN allowed.
        timestamps:           DatetimeIndex of length T.
        lats/lons/alts:       Station coordinates and altitudes (N,).
        station_ids:          List of station ID strings (N,).
        dist_matrix:          (N, N) geodesic distance matrix in km.
        variogram_params_list: List of variogram parameter dicts, one per segment.
        segment_indices:      (T,) int array mapping each timestamp to a segment.
        k:                    Number of nearest neighbours.
        idw_power:            IDW exponent.
        u_matrix:             (T, N) zonal wind component; optional.
        v_matrix:             (T, N) meridional wind component; optional.
        rk_feature_names:     Ordered list of feature names for RK trend.
        rk_static_features:   Per-station static feature arrays.
        rk_dynamic_features:  Per-timestamp-per-station feature arrays.
        aniso_dist_matrix:    (N, N) anisotropic distance matrix in km; optional.

    Returns:
        DataFrame with columns:
        [station_id, timestamp, wind_speed_observed, idw_pred, ok_pred, rk_pred]
        plus, when u/v matrices are provided:
        [u_observed, u_idw_pred, u_ok_pred, u_rk_pred,
         v_observed, v_idw_pred, v_ok_pred, v_rk_pred]
    """
    N = len(station_ids)
    T = len(timestamps)
    do_uv = u_matrix is not None and v_matrix is not None
    use_multi_features = rk_feature_names is not None
    use_aniso = aniso_dist_matrix is not None

    if use_multi_features:
        logger.info("RK features: %s", rk_feature_names)
    else:
        logger.info("RK features: altitude (default)")
    if use_aniso:
        logger.info("Anisotropic kriging enabled.")

    # Pre-compute fixed neighbour indices for every station (geodesic, unchanged)
    neighbor_sets = {s: get_k_nearest(dist_matrix, s, k) for s in range(N)}

    def _build_kriging_predictors(vp: dict) -> list:
        """Build one KrigingPredictor per station for variogram params *vp*."""
        preds = []
        for s_idx in range(N):
            n_idxs = neighbor_sets[s_idx]
            if use_aniso:
                nn_d = aniso_dist_matrix[np.ix_(n_idxs, n_idxs)]
                td   = aniso_dist_matrix[n_idxs, s_idx]
            else:
                nn_d = td = None
            preds.append(KrigingPredictor(
                neighbor_lats=lats[n_idxs],
                neighbor_lons=lons[n_idxs],
                target_lat=float(lats[s_idx]),
                target_lon=float(lons[s_idx]),
                variogram_params=vp,
                nn_dist_precomputed=nn_d,
                target_dist_precomputed=td,
            ))
        return preds

    # For segments > 0, precompute all segment predictors upfront.
    # For per-timestamp (many segments), we rebuild lazily inside the loop.
    n_segments = len(variogram_params_list)
    per_timestamp = (n_segments == T)

    if not per_timestamp:
        logger.info("Pre-computing kriging weights for %d segment(s) × %d stations ...",
                    n_segments, N)
        segment_predictors = [_build_kriging_predictors(vp) for vp in variogram_params_list]
        logger.info("Kriging weights ready.")
    else:
        logger.info("Per-timestamp mode: kriging weights will be rebuilt for each timestamp.")
        segment_predictors = None  # built lazily

    def _build_features(t_idx: int, n_idxs: np.ndarray, s_idx: int):
        """Return (neighbor_features (k,F), target_features (F,)) for this sample."""
        cols_n, cols_t = [], []
        for name in rk_feature_names:
            if name in rk_static_features:
                arr = rk_static_features[name]
                cols_n.append(arr[n_idxs])
                cols_t.append(arr[s_idx])
            elif name in rk_dynamic_features:
                arr = rk_dynamic_features[name]
                cols_n.append(arr[t_idx, n_idxs])
                cols_t.append(arr[t_idx, s_idx])
        return np.column_stack(cols_n), np.array(cols_t)

    records = []
    current_seg = -1
    active_predictors = None

    for t_idx in range(T):
        if t_idx % 200 == 0:
            logger.info("  Timestamp %d / %d", t_idx, T)

        seg = int(segment_indices[t_idx])

        # Rebuild kriging predictors when entering a new segment
        if seg != current_seg:
            if per_timestamp:
                active_predictors = _build_kriging_predictors(variogram_params_list[seg])
            else:
                active_predictors = segment_predictors[seg]
            current_seg = seg

        vals_t = values_matrix[t_idx]
        if np.all(np.isnan(vals_t)):
            continue

        u_t = u_matrix[t_idx] if do_uv else None
        v_t = v_matrix[t_idx] if do_uv else None

        for s_idx in range(N):
            obs = vals_t[s_idx]
            if np.isnan(obs):
                continue

            n_idxs = neighbor_sets[s_idx]
            n_vals = vals_t[n_idxs]

            # Skip if any neighbour is missing
            if np.any(np.isnan(n_vals)):
                continue

            n_dists = dist_matrix[s_idx, n_idxs]   # geodesic for IDW
            kp = active_predictors[s_idx]

            idw_pred = predict_idw(n_vals, n_dists, idw_power)
            ok_pred = kp.predict(n_vals)

            if use_multi_features:
                n_feats, t_feats = _build_features(t_idx, n_idxs, s_idx)
                rk_pred = predict_rk(n_vals, n_feats, t_feats, kp)
            else:
                n_alts = alts[n_idxs]
                rk_pred = predict_rk(n_vals, n_alts.reshape(-1, 1), np.array([alts[s_idx]]), kp)

            rec = {
                "station_id": station_ids[s_idx],
                "timestamp": timestamps[t_idx],
                "wind_speed_observed": float(obs),
                "idw_pred": idw_pred,
                "ok_pred": ok_pred,
                "rk_pred": rk_pred,
            }

            if do_uv:
                u_obs = u_t[s_idx]
                v_obs = v_t[s_idx]

                if not (np.isnan(u_obs) or np.isnan(v_obs)):
                    nu_vals = u_t[n_idxs]
                    nv_vals = v_t[n_idxs]

                    if not (np.any(np.isnan(nu_vals)) or np.any(np.isnan(nv_vals))):
                        rec["u_observed"] = float(u_obs)
                        rec["u_idw_pred"] = predict_idw(nu_vals, n_dists, idw_power)
                        rec["u_ok_pred"] = kp.predict(nu_vals)

                        rec["v_observed"] = float(v_obs)
                        rec["v_idw_pred"] = predict_idw(nv_vals, n_dists, idw_power)
                        rec["v_ok_pred"] = kp.predict(nv_vals)

                        if use_multi_features:
                            rec["u_rk_pred"] = predict_rk(nu_vals, n_feats, t_feats, kp)
                            rec["v_rk_pred"] = predict_rk(nv_vals, n_feats, t_feats, kp)
                        else:
                            n_alts = alts[n_idxs]
                            rec["u_rk_pred"] = predict_rk(nu_vals, n_alts.reshape(-1, 1), np.array([alts[s_idx]]), kp)
                            rec["v_rk_pred"] = predict_rk(nv_vals, n_alts.reshape(-1, 1), np.array([alts[s_idx]]), kp)

            records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> tuple:
    """Compute per-station and global aggregated metrics.

    Handles both scalar wind-speed predictions and, if present, u/v component
    predictions. For u/v, also derives the vector wind-speed error.

    Args:
        df: LOO-CV output DataFrame with columns
            [station_id, wind_speed_observed, idw_pred, ok_pred, rk_pred]
            and optionally [u_observed, u_*_pred, v_observed, v_*_pred].

    Returns:
        per_station_df: DataFrame [station_id, method, rmse, mae, r2].
        summary_df:     DataFrame [method, rmse, mae, r2] (mean over stations).
    """
    has_uv = "u_observed" in df.columns

    # Scalar wind-speed methods
    methods: dict[str, tuple[str, str]] = {
        "idw": ("wind_speed_observed", "idw_pred"),
        "ok": ("wind_speed_observed", "ok_pred"),
        "rk": ("wind_speed_observed", "rk_pred"),
    }

    if has_uv:
        uv_df = df.dropna(subset=["u_observed", "v_observed"])
        for comp in ("u", "v"):
            for m in ("idw", "ok", "rk"):
                methods[f"{comp}_{m}"] = (f"{comp}_observed", f"{comp}_{m}_pred")

    rows = []
    for sid, grp in df.groupby("station_id"):
        for method, (obs_col, pred_col) in methods.items():
            sub = grp.dropna(subset=[obs_col, pred_col]) if has_uv else grp
            obs = sub[obs_col].values
            pred = sub[pred_col].values
            if len(obs) < 2:
                continue
            rows.append(
                {
                    "station_id": sid,
                    "method": method,
                    "rmse": float(np.sqrt(mean_squared_error(obs, pred))),
                    "mae": float(mean_absolute_error(obs, pred)),
                    "r2": float(r2_score(obs, pred)),
                }
            )

    # Derived vector wind-speed from predicted u/v
    if has_uv:
        uv_df = df.dropna(subset=["u_observed", "v_observed"])
        ws_obs = np.sqrt(uv_df["u_observed"] ** 2 + uv_df["v_observed"] ** 2).values
        for m in ("idw", "ok", "rk"):
            ws_pred = np.sqrt(uv_df[f"u_{m}_pred"] ** 2 + uv_df[f"v_{m}_pred"] ** 2).values
            for sid, grp_idx in uv_df.groupby("station_id").groups.items():
                sub_obs = np.sqrt(
                    uv_df.loc[grp_idx, "u_observed"] ** 2
                    + uv_df.loc[grp_idx, "v_observed"] ** 2
                ).values
                sub_pred = np.sqrt(
                    uv_df.loc[grp_idx, f"u_{m}_pred"] ** 2
                    + uv_df.loc[grp_idx, f"v_{m}_pred"] ** 2
                ).values
                if len(sub_obs) < 2:
                    continue
                rows.append(
                    {
                        "station_id": sid,
                        "method": f"ws_from_uv_{m}",
                        "rmse": float(np.sqrt(mean_squared_error(sub_obs, sub_pred))),
                        "mae": float(mean_absolute_error(sub_obs, sub_pred)),
                        "r2": float(r2_score(sub_obs, sub_pred)),
                    }
                )

    per_station_df = pd.DataFrame(rows)

    all_methods = list(methods.keys())
    if has_uv:
        all_methods += [f"ws_from_uv_{m}" for m in ("idw", "ok", "rk")]

    summary_rows = []
    for method in all_methods:
        sub = per_station_df[per_station_df["method"] == method]
        if sub.empty:
            continue
        summary_rows.append(
            {
                "method": method,
                "rmse": sub["rmse"].mean(),
                "mae": sub["mae"].mean(),
                "r2": sub["r2"].mean(),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    return per_station_df, summary_df
