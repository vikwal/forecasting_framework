"""
geostatistics/baselines/dataset.py — shared data loading, run-pair construction
and design-matrix building for the QRF and MOS baselines.

Verbindliche Referenz: docs/baselines_implementation_spec.md (Abschnitte 1, 3, 4).
Dieses Modul baut NICHTS an Zeitkonvention oder Run-Paar-Logik neu: die Schleife
in ``build_run_pairs`` ist Zeile für Zeile die aus ``evaluate_reference.py:425-447``
(plus dem Gitter-NaN-Filter aus ``hpo_mtgnn.py:666-682``, siehe Spezifikation 4.3).

Stationsliste je Fold-Config, train zuerst (Spezifikation 1.1, "verbindlich"):
``all_ids = data.files + data.val_files``, ``train_idx = range(N_train)``,
``val_idx = range(N_train, N_train + N_val)``.

Ein Random Forest ist invariant gegen jede streng monotone Transformation seiner
Merkmale (Spezifikation 3.3) — deshalb baut dieses Modul KEINE Skalierung ein.
Nur die Topo-z-Scores kommen bereits skaliert aus
``load_topo_station_features_dict`` (train-only, wie im Modellpfad).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from geostatistics.train_stgnn2 import (          # noqa: E402
    load_yaml,
    load_station_measurements,
    load_station_metadata,
    load_icond2_ml_runs,
    load_ecmwf_parquet_at_stations_and_grid,
    load_interpol_imputation,
    apply_interpol_imputation,
    load_knn_imputation,
    apply_knn_imputation,
    require_nwp_elevation_env,
)
from geostatistics.train_dcrnn import encode_circular_measurements, apply_dir_encoding  # noqa: E402
from geostatistics.stgnn.utils.topo_features import (      # noqa: E402
    load_topo_station_features_dict, TOPO_FEATURE_ORDER,
)
from geostatistics.stgnn.utils.spatial import geodesic_knn, pairwise_geodesic_km  # noqa: E402

logger = logging.getLogger("baselines.dataset")

# HPO-Obergrenze fuer next_n_icond2 ueber alle Graphmodelle
# (configs/mtgnn/config_wind_mtgnn_nwp_fold1.yaml: hpo.params.next_n_icond2.high).
# Geladen wird auf dieser Obergrenze, der Pflicht-Betriebspunkt k_i2=4 wird
# anschliessend aus den ersten 4 Spalten der (nearest-first) k=7-Nachbarschaft
# gesliced (Spezifikation 3.2, "Max-Bound-Trick").
LOAD_K_I2 = 7
# ECMWF-Obergrenze == Pflichtwert (next_n_ecmwf HPO-Bereich 0..4).
LOAD_K_E2 = 4

SUBSAMPLE_SEED_DEFAULT = 20260810

_FREQ_H = {"1h": 1.0, "1H": 1.0, "30min": 0.5, "30T": 0.5, "15min": 0.25, "15T": 0.25}


# ---------------------------------------------------------------------------
# Context loading — one fold-config, train-first station ordering
# ---------------------------------------------------------------------------

def load_context(
    config_path: str,
    ecmwf_features_override: Optional[list] = None,
    files_override: Optional[list] = None,
    val_files_override: Optional[list] = None,
    test_mode: bool = False,
) -> dict:
    """Load everything needed from ONE fold config.

    Station ordering is ``data.files + data.val_files`` (train first) — the
    "Retrain/Eval" convention that Spezifikation 1.1 declares binding for the
    baselines. Returns a dict ("ctx") consumed by ``build_run_pairs`` and
    ``build_feature_matrix`` / ``build_mos_rows``.

    ``files_override``/``val_files_override`` let ``hpo_qrf.py`` load the
    sorted union pool (``spatial_cv.station_pool``) once instead of one
    fold-config's train/val split — used only for the 3-fold spatial-CV HPO
    objective (Spezifikation 5.1/1.8), which needs all three folds' arrays
    from a single shared load. Station ROLES for a given fold are then
    resolved via ``spatial_cv.build_folds`` on this same pool, not by this
    function; Spezifikation 1.1 already establishes that both conventions
    assign identical station sets per fold.
    """
    cfg = load_yaml(config_path)
    data_cfg = cfg["data"]
    qcfg = cfg.get("qrf", {})

    if test_mode:
        # evaluate_reference.py Konvention (Spezifikation 6.1): train = files +
        # val_files (153 Poolstationen), Ziel = test_files (die 50 alten,
        # bekannt-veralteten Stationen — Befund 10.4, hier NICHT reparieren).
        train_ids = [str(s) for s in data_cfg["files"]] + [str(s) for s in data_cfg["val_files"]]
        val_ids = [str(s) for s in data_cfg["test_files"]]
    else:
        train_ids = [str(s) for s in (files_override if files_override is not None else data_cfg["files"])]
        val_ids = [str(s) for s in (val_files_override if val_files_override is not None else data_cfg["val_files"])]
    all_ids = train_ids + val_ids
    N_train, N_val = len(train_ids), len(val_ids)
    if set(train_ids) & set(val_ids):
        raise ValueError(f"{config_path}: files und val_files ueberschneiden sich")

    icond2_features = list(qcfg.get("icond2_features") or
                            ["u_10m", "v_10m", "wind_speed_10m", "wind_speed_38m"])
    i2_mode = qcfg.get("icond2_feature_mode", "dir_in_deg")
    e2_mode = qcfg.get("ecmwf_feature_mode", "dir_in_deg")
    measurement_cols = list(qcfg.get("measurement_features") or ["wind_speed", "wind_direction"])
    target_col = qcfg.get("target_col", "wind_speed")
    run_hours = tuple(qcfg.get("icond2_run_hours", [6, 9, 12, 15]))
    n_workers = qcfg.get("n_workers", 8)
    nwp_path = data_cfg.get("nwp_path")
    data_path = data_cfg["path"]
    H = qcfg.get("history_length", 48)
    F_h = qcfg.get("forecast_horizon", 48)
    freq = data_cfg.get("freq", "1h")
    freq_h = _FREQ_H.get(freq, 1.0)

    ecmwf_features = list(
        ecmwf_features_override if ecmwf_features_override is not None
        else (qcfg.get("ecmwf_features") or ["wind_speed_10m"])
    )

    # K3-Guard: eine nicht-interaktive Shell ohne WEATHER_DB_URL/ECMWF_WIND_SL_URL
    # darf nicht stillschweigend weiterlaufen (Spezifikation 5.4). QRF/MOS
    # brauchen die NWP-Knotenhoehen selbst nicht, aber der Guard verhindert,
    # dass ein halbkonfigurierter Lauf unbemerkt bleibt.
    require_nwp_elevation_env(
        need_icond2=True, need_ecmwf=True,
        context=f"baselines/dataset.py ({config_path})",
    )

    test_end = data_cfg.get("test_end")
    run_cutoff = pd.Timestamp(test_end, tz="UTC") if test_end else None

    logger.info("Loading station measurements for %d stations …", len(all_ids))
    meas_raw, timestamps = load_station_measurements(
        data_path, all_ids, cols=measurement_cols, freq=freq,
    )
    if run_cutoff is not None:
        cut_idx = int(np.searchsorted(timestamps, run_cutoff + pd.Timedelta(days=2), side="right"))
        meas_raw = meas_raw[:cut_idx]
        timestamps = timestamps[:cut_idx]
    T = len(timestamps)

    interpol_path = data_cfg.get("interpol_path")
    if interpol_path:
        rk_pred = load_interpol_imputation(interpol_path, all_ids, timestamps)
        meas_raw = apply_interpol_imputation(meas_raw, rk_pred, measurement_cols, target_col)

    knnimputer_path = data_cfg.get("knnimputer_path")
    if knnimputer_path:
        for col in measurement_cols:
            feat_idx = measurement_cols.index(col)
            if not np.isnan(meas_raw[:, :, feat_idx]).any():
                continue
            knn_arr = load_knn_imputation(knnimputer_path, col, all_ids, timestamps, freq=freq)
            meas_raw = apply_knn_imputation(meas_raw, knn_arr, measurement_cols, col)

    meas_nan_any = np.isnan(meas_raw).any(axis=(1, 2))
    meas_raw, measurement_cols = encode_circular_measurements(meas_raw, measurement_cols)
    target_feat_idx = measurement_cols.index(target_col)

    meta_path = data_cfg.get("stations_master")
    lats, lons, alts = load_station_metadata(data_path, all_ids, meta_path=meta_path)
    station_coords = np.stack([lats, lons], axis=1)

    logger.info("Loading ICON-D2 runs (next_n_grid=%d) …", LOAD_K_I2)
    run_times, icond2_coords, grid_icond2_runs, _station_nearest_i2 = load_icond2_ml_runs(
        nwp_path=nwp_path, station_ids=all_ids, station_coords=station_coords,
        features=icond2_features, run_hours=run_hours, next_n_grid=LOAD_K_I2,
        n_workers=n_workers, cutoff=run_cutoff,
    )
    if i2_mode == "dir_in_deg":
        grid_icond2_runs, icond2_features = apply_dir_encoding(grid_icond2_runs, icond2_features)
    I2 = len(icond2_features)
    R = len(run_times)
    n_leads = grid_icond2_runs.shape[1]

    nwp_ws_feat_idx = next(
        (i for i, f in enumerate(icond2_features) if f == "wind_speed_10m"),
        next((i for i, f in enumerate(icond2_features) if "wind_speed" in f), 0),
    )

    # k naechste Gitterpunkte, geodaetisch, ueber das GLOBALE (dedupliziertem)
    # Gitter — dieselbe Methode wie homo_sampler.py::_init_grid_knn (Spezifikation
    # 1.4/2.2), NICHT die einzelne "nearest"-Zuordnung des Loaders.
    k_load_i2 = min(LOAD_K_I2, len(icond2_coords))
    _dists_i2, nearest_i2_idx = geodesic_knn(icond2_coords, station_coords, k=k_load_i2)

    # ECMWF — Pflichtbestandteil von QRF-local (43 Spalten), unabhaengig von
    # --nwp-sources (das steuert nur MOS).
    ecmwf_path = data_cfg.get("ecmwf_path")
    ecmwf_coords = None
    grid_ecmwf_raw = None
    ecmwf_ws_feat_idx = 0
    nearest_e2_idx = None
    E2 = 0
    if ecmwf_path and os.path.exists(ecmwf_path):
        logger.info("Loading ECMWF (next_n_grid=%d) …", LOAD_K_E2)
        _station_ecmwf_nwp, ecmwf_coords, grid_ecmwf_raw, _grid_alts = \
            load_ecmwf_parquet_at_stations_and_grid(
                parquet_path=ecmwf_path, station_lats=lats, station_lons=lons,
                features=ecmwf_features, timestamps=timestamps,
                next_n_grid_per_station=LOAD_K_E2,
            )
        if e2_mode == "dir_in_deg":
            grid_ecmwf_raw, ecmwf_features = apply_dir_encoding(grid_ecmwf_raw, ecmwf_features)
        E2 = grid_ecmwf_raw.shape[2]
        ecmwf_ws_feat_idx = next(
            (i for i, f in enumerate(ecmwf_features) if f == "wind_speed_10m"),
            next((i for i, f in enumerate(ecmwf_features) if "wind_speed" in f), 0),
        )
        k_load_e2 = min(LOAD_K_E2, len(ecmwf_coords))
        _dists_e2, nearest_e2_idx = geodesic_knn(ecmwf_coords, station_coords, k=k_load_e2)
    else:
        logger.warning("ecmwf_path %s missing/unset — QRF-local ECMWF block disabled", ecmwf_path)

    # Topo-Deskriptoren, z-Score NUR auf den Fold-Trainingsstationen (train_idx =
    # range(N_train), weil all_ids train-first ist — Spezifikation 3.2 Punkt 3).
    topo_features_path = qcfg.get("topo_features_path")
    topo_train: dict[str, np.ndarray] = {}
    if topo_features_path:
        topo_train = load_topo_station_features_dict(
            topo_features_path, all_ids, TOPO_FEATURE_ORDER, train_idx=list(range(N_train)),
        )

    val_start = pd.Timestamp(data_cfg["val_start"], tz="UTC")
    test_start = pd.Timestamp(data_cfg["test_start"], tz="UTC") if data_cfg.get("test_start") else None
    test_end_ts = pd.Timestamp(data_cfg["test_end"], tz="UTC") if data_cfg.get("test_end") else None

    ctx = dict(
        config_path=str(config_path), cfg=cfg, data_cfg=data_cfg, qcfg=qcfg,
        train_ids=train_ids, val_ids=val_ids, all_ids=all_ids,
        N_train=N_train, N_val=N_val,
        lats=lats, lons=lons, alts=alts,
        timestamps=timestamps, T=T,
        meas_raw=meas_raw, measurement_cols=measurement_cols,
        target_feat_idx=target_feat_idx, meas_nan_any=meas_nan_any,
        run_times=run_times, R=R, n_leads=n_leads,
        icond2_coords=icond2_coords, grid_icond2_runs=grid_icond2_runs,
        icond2_features=icond2_features, I2=I2,
        nwp_ws_feat_idx=nwp_ws_feat_idx, nearest_i2_idx=nearest_i2_idx,
        ecmwf_coords=ecmwf_coords, grid_ecmwf_raw=grid_ecmwf_raw,
        ecmwf_features=ecmwf_features, E2=E2,
        ecmwf_ws_feat_idx=ecmwf_ws_feat_idx, nearest_e2_idx=nearest_e2_idx,
        topo_train=topo_train,
        H=H, F_h=F_h, freq_h=freq_h,
        val_start=val_start, test_start=test_start, test_end=test_end_ts,
    )
    return ctx


# ---------------------------------------------------------------------------
# Run-pair construction — evaluate_reference.py:425-447 + grid-NaN filter
# ---------------------------------------------------------------------------

def build_run_pairs(
    ctx: dict,
    lo: Optional[pd.Timestamp],
    hi: Optional[pd.Timestamp],
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    """Run-Paare mit ``lo <= Laufzeit < hi`` (Grenzen ueber die Laufzeit, nicht
    ueber ``t_run_abs`` — Spezifikation 4.2). ``lo=None`` heisst keine
    Untergrenze (Fit-Fenster), ``hi=None`` heisst keine Obergrenze.

    Identisch zu ``evaluate_reference.py:425-447``, danach zusaetzlich der
    ICON-D2-Gitter-NaN-Filter aus ``hpo_mtgnn.py:666-682`` als Post-Filter
    (zwei Phasen, genau wie dort). Gibt (pairs, counters) zurueck, counters =
    {"r_hist": int, "grid_nan": int, "meas_nan": int}.
    """
    run_times = ctx["run_times"]
    timestamps = ctx["timestamps"]
    meas_nan_any = ctx["meas_nan_any"]
    H, F_h, freq_h = ctx["H"], ctx["F_h"], ctx["freq_h"]
    T = len(timestamps)
    R = len(run_times)
    ts_lookup = pd.Series(np.arange(T), index=timestamps)

    pairs: list[tuple[int, int, int]] = []
    n_rhist_drop = 0
    n_measnan_drop = 0

    for r_curr in range(R):
        t_run = run_times[r_curr]
        if lo is not None and t_run < lo:
            continue
        if hi is not None and t_run >= hi:
            continue
        if t_run not in ts_lookup.index:
            continue
        t_run_abs = int(ts_lookup[t_run]) + 1
        if t_run_abs < H or t_run_abs + F_h > T:
            continue
        t_hist_target = t_run - pd.Timedelta(hours=H * freq_h)
        diffs_s = np.abs((run_times - t_hist_target).total_seconds().values)
        r_hist = int(np.argmin(diffs_s))
        if diffs_s[r_hist] > 3 * 3600:
            n_rhist_drop += 1
            continue
        if meas_nan_any[t_run_abs - H: t_run_abs + F_h].any():
            n_measnan_drop += 1
            continue
        pairs.append((r_curr, r_hist, t_run_abs))

    grid_icond2_runs = ctx["grid_icond2_runs"]
    grid_nan_runs = set(
        int(i) for i in np.where(np.isnan(grid_icond2_runs).any(axis=(1, 2, 3)))[0]
    )
    n_before = len(pairs)
    if grid_nan_runs:
        pairs = [(rc, rh, t) for rc, rh, t in pairs
                 if rc not in grid_nan_runs and rh not in grid_nan_runs]
    n_gridnan_drop = n_before - len(pairs)

    counters = {"r_hist": n_rhist_drop, "grid_nan": n_gridnan_drop, "meas_nan": n_measnan_drop}
    return pairs, counters


# ---------------------------------------------------------------------------
# QRF-local design matrix — vectorised, station-major / pair / lead row order
# ---------------------------------------------------------------------------

def feature_columns(ctx: dict, k_i2: int, k_e2: int, nwp_geometry: bool = False,
                     i2_hist: bool = False, idw_n: int = 0) -> list[str]:
    """Column names, in construction order (Spezifikation 3.2 Tabelle)."""
    icond2_features = ctx["icond2_features"]
    ecmwf_features = ctx["ecmwf_features"] if ctx["grid_ecmwf_raw"] is not None else []
    cols: list[str] = []
    for j in range(k_i2):
        cols += [f"i2_k{j}_{f}" for f in icond2_features]
    for j in range(k_e2):
        cols += [f"e2_k{j}_{f}" for f in ecmwf_features]
    cols += list(TOPO_FEATURE_ORDER)
    cols += ["lat", "lon", "alt"]
    cols += ["horizon"]
    cols += ["valid_hour_sin", "valid_hour_cos"]
    if nwp_geometry:
        for j in range(k_i2):
            cols += [f"d_i2_k{j}", f"dz_i2_k{j}"]
        for j in range(k_e2):
            cols += [f"d_e2_k{j}", f"dz_e2_k{j}"]
    if i2_hist:
        cols += [f"i2hist_k{j}_{f}" for j in range(k_i2) for f in icond2_features]
    if idw_n > 0:
        cols += ["idw_mean_neighbour_ws", "dist_nearest_train_km"]
    return cols


def build_feature_matrix(
    ctx: dict,
    station_pos: np.ndarray,
    pairs: list[tuple[int, int, int]],
    k_i2: int = 4,
    k_e2: int = 4,
    need_meta: bool = True,
    nwp_geometry: bool = False,
    i2_hist: bool = False,
    idw_neighbour_pos: Optional[np.ndarray] = None,
    idw_n: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], Optional[pd.DataFrame]]:
    """Vectorised QRF-local design matrix.

    Row order: station (S, from ``station_pos``) major, pair (P) middle, lead
    (F_h) minor — i.e. row index ``s_i * P * F_h + p_i * F_h + (h - 1)``.

    Returns
    -------
    X    : (S*P*F_h, n_cols) float32
    y    : (S*P*F_h,) float32 — ``meas_raw[valid_time, station, target]``
    cols : column names, matching X's column order
    meta : DataFrame with station_id, run_time, valid_time, horizon, gt,
           nwp_ref, pers_ref (None if ``need_meta`` is False)
    """
    all_ids = ctx["all_ids"]
    F_h = ctx["F_h"]
    S = len(station_pos)
    P = len(pairs)
    station_pos = np.asarray(station_pos, dtype=np.int64)

    if S == 0 or P == 0:
        cols = feature_columns(ctx, k_i2, k_e2, nwp_geometry, i2_hist, idw_n)
        return (np.zeros((0, len(cols)), dtype=np.float32),
                np.zeros((0,), dtype=np.float32), cols, (pd.DataFrame() if need_meta else None))

    r_curr_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    r_hist_arr = np.array([p[1] for p in pairs], dtype=np.int64)
    t_run_abs_arr = np.array([p[2] for p in pairs], dtype=np.int64)

    # Absolute time index of each (pair, lead): valid_time = timestamps[t_run_abs + h - 1]
    # (h in 1..F_h, array position j=h-1 -> np.arange(F_h) IS h-1 already; adding a
    # further "-1" here double-subtracted and pointed one hour BEFORE the true valid
    # time -- Aufgabe 2 der Nacharbeit vom 2026-08-10, gefunden per Row-Level-Diff
    # gegen l1:data/raw_preds/icon_d2_fold0_raw.parquet fuer Station 05516: identische
    # nwp_ref-Werte, aber unsere valid_time lag durchgaengig 1h vor der Referenz.
    # nwp_ref/ws_i2 sind NICHT betroffen (die werden ueber die Lead-Position direkt aus
    # grid_icond2_runs[..., :F_h, ...] gelesen, nicht ueber time_idx) -- betroffen waren
    # y/gt, ws_e2 und das valid_time-Label in build_feature_matrix UND build_mos_rows.
    time_idx = t_run_abs_arr[:, None] + np.arange(F_h)[None, :]  # (P, F_h)

    grid_icond2_runs = ctx["grid_icond2_runs"]
    nearest_i2_idx = ctx["nearest_i2_idx"]
    I2 = ctx["I2"]
    idx_i2 = nearest_i2_idx[station_pos][:, :k_i2]  # (S, k_i2)

    sub_i2 = grid_icond2_runs[r_curr_arr][:, :F_h, :, :]      # (P, F_h, N_grid, I2)
    sub_i2 = sub_i2[:, :, idx_i2, :]                          # (P, F_h, S, k_i2, I2)
    blk_i2 = sub_i2.transpose(2, 0, 1, 3, 4).reshape(S * P * F_h, k_i2 * I2).astype(np.float32)

    blocks = [blk_i2]

    grid_ecmwf_raw = ctx["grid_ecmwf_raw"]
    if grid_ecmwf_raw is not None and k_e2 > 0:
        nearest_e2_idx = ctx["nearest_e2_idx"]
        E2 = ctx["E2"]
        idx_e2 = nearest_e2_idx[station_pos][:, :k_e2]           # (S, k_e2)
        sub_e2 = grid_ecmwf_raw[time_idx]                        # (P, F_h, N_grid_e2, E2)
        sub_e2 = sub_e2[:, :, idx_e2, :]                          # (P, F_h, S, k_e2, E2)
        blk_e2 = sub_e2.transpose(2, 0, 1, 3, 4).reshape(S * P * F_h, k_e2 * E2).astype(np.float32)
        blocks.append(blk_e2)

    # Topo (9), constant per station -> broadcast (S, P, F_h)
    topo_train = ctx["topo_train"]
    topo_cols = []
    for name in TOPO_FEATURE_ORDER:
        arr = topo_train.get(name)
        if arr is None:
            topo_cols.append(np.zeros(S * P * F_h, dtype=np.float32))
            continue
        v = arr[station_pos].astype(np.float32)                  # (S,)
        topo_cols.append(np.broadcast_to(v[:, None, None], (S, P, F_h)).reshape(-1))
    blocks.append(np.stack(topo_cols, axis=1))

    # lat, lon, alt — raw degrees / metres, no encoding, no scaling (3.2 #4)
    lats, lons, alts = ctx["lats"], ctx["lons"], ctx["alts"]
    lat_v = np.broadcast_to(lats[station_pos][:, None, None], (S, P, F_h)).reshape(-1)
    lon_v = np.broadcast_to(lons[station_pos][:, None, None], (S, P, F_h)).reshape(-1)
    alt_v = np.broadcast_to(alts[station_pos][:, None, None], (S, P, F_h)).reshape(-1)
    blocks.append(np.stack([lat_v, lon_v, alt_v], axis=1).astype(np.float32))

    # horizon (1..F_h), broadcast over (S, P)
    h_arr = np.arange(1, F_h + 1, dtype=np.float32)
    horizon_v = np.broadcast_to(h_arr[None, None, :], (S, P, F_h)).reshape(-1)
    blocks.append(horizon_v[:, None])

    # valid_hour_sin/cos
    timestamps = ctx["timestamps"]
    valid_time_pf = timestamps[time_idx.reshape(-1)].values.reshape(P, F_h)  # (P, F_h) datetime64
    hours_pf = pd.DatetimeIndex(valid_time_pf.reshape(-1)).hour.values.reshape(P, F_h).astype(np.float32)
    hour_rad = 2 * np.pi * hours_pf / 24.0
    sin_h = np.broadcast_to(np.sin(hour_rad)[None, :, :], (S, P, F_h)).reshape(-1)
    cos_h = np.broadcast_to(np.cos(hour_rad)[None, :, :], (S, P, F_h)).reshape(-1)
    blocks.append(np.stack([sin_h, cos_h], axis=1).astype(np.float32))

    if nwp_geometry:
        from geostatistics.stgnn.utils.spatial import geodesic_km
        s_lat = lats[station_pos]
        s_lon = lons[station_pos]
        s_alt = alts[station_pos]
        g_lat_i2 = ctx["icond2_coords"][idx_i2, 0]   # (S, k_i2)
        g_lon_i2 = ctx["icond2_coords"][idx_i2, 1]
        d_i2 = geodesic_km(np.repeat(s_lat[:, None], k_i2, axis=1), np.repeat(s_lon[:, None], k_i2, axis=1),
                            g_lat_i2, g_lon_i2)                       # (S, k_i2)
        dz_i2 = np.zeros_like(d_i2)  # no NWP-node elevations loaded here (not needed for QRF)
        geo_blk = []
        for j in range(k_i2):
            v_d = np.broadcast_to(d_i2[:, j][:, None, None], (S, P, F_h)).reshape(-1)
            v_z = np.broadcast_to(dz_i2[:, j][:, None, None], (S, P, F_h)).reshape(-1)
            geo_blk += [v_d, v_z]
        if grid_ecmwf_raw is not None and k_e2 > 0:
            g_lat_e2 = ctx["ecmwf_coords"][idx_e2, 0]
            g_lon_e2 = ctx["ecmwf_coords"][idx_e2, 1]
            d_e2 = geodesic_km(np.repeat(s_lat[:, None], k_e2, axis=1), np.repeat(s_lon[:, None], k_e2, axis=1),
                                g_lat_e2, g_lon_e2)
            dz_e2 = np.zeros_like(d_e2)
            for j in range(k_e2):
                v_d = np.broadcast_to(d_e2[:, j][:, None, None], (S, P, F_h)).reshape(-1)
                v_z = np.broadcast_to(dz_e2[:, j][:, None, None], (S, P, F_h)).reshape(-1)
                geo_blk += [v_d, v_z]
        blocks.append(np.stack(geo_blk, axis=1).astype(np.float32))

    if i2_hist:
        sub_i2h = grid_icond2_runs[r_hist_arr][:, :F_h, :, :]
        sub_i2h = sub_i2h[:, :, idx_i2, :]
        blk_i2h = sub_i2h.transpose(2, 0, 1, 3, 4).reshape(S * P * F_h, k_i2 * I2).astype(np.float32)
        blocks.append(blk_i2h)

    if idw_n > 0:
        idw_mean, dist_nearest = _idw_neighbour_block(
            ctx, station_pos, t_run_abs_arr, idw_neighbour_pos, idw_n,
        )  # both (S, P)
        idw_mean_v = np.broadcast_to(idw_mean[:, :, None], (S, P, F_h)).reshape(-1)
        dist_v = np.broadcast_to(dist_nearest[:, :, None], (S, P, F_h)).reshape(-1)
        blocks.append(np.stack([idw_mean_v, dist_v], axis=1).astype(np.float32))

    X = np.concatenate(blocks, axis=1).astype(np.float32)

    meas_raw = ctx["meas_raw"]
    target_feat_idx = ctx["target_feat_idx"]
    gt_pf = meas_raw[time_idx.reshape(-1), :, target_feat_idx].reshape(P, F_h, -1)[:, :, station_pos]  # (P,F_h,S)
    y = gt_pf.transpose(2, 0, 1).reshape(-1).astype(np.float32)

    cols = feature_columns(ctx, k_i2, k_e2, nwp_geometry, i2_hist, idw_n)
    assert X.shape[1] == len(cols), f"{X.shape[1]} != {len(cols)} ({cols})"

    meta = None
    if need_meta:
        station_ids_v = np.broadcast_to(
            np.array(all_ids, dtype=object)[station_pos][:, None, None], (S, P, F_h),
        ).reshape(-1)
        run_time_pf = timestamps[t_run_abs_arr - 1]                      # (P,)
        run_time_v = np.broadcast_to(
            np.asarray(run_time_pf.values)[None, :, None], (S, P, F_h),
        ).reshape(-1)
        valid_time_v = np.broadcast_to(valid_time_pf[None, :, :], (S, P, F_h)).reshape(-1)
        horizon_v_i = np.broadcast_to(
            np.arange(1, F_h + 1, dtype=np.int16)[None, None, :], (S, P, F_h),
        ).reshape(-1)

        pers_vals = meas_raw[t_run_abs_arr - 1][:, station_pos, target_feat_idx]  # (P, S)
        pers_ref_v = np.broadcast_to(pers_vals.T[:, :, None], (S, P, F_h)).reshape(-1)

        idx0 = nearest_i2_idx[station_pos][:, 0]                          # (S,)
        nwp_ws = ctx["nwp_ws_feat_idx"]
        sub0 = grid_icond2_runs[r_curr_arr][:, :F_h, :, :][:, :, idx0, nwp_ws]  # (P, F_h, S)
        nwp_ref_v = sub0.transpose(2, 0, 1).reshape(-1)

        meta = pd.DataFrame({
            "station_id": station_ids_v,
            "run_time": pd.to_datetime(run_time_v, utc=True),
            "valid_time": pd.to_datetime(valid_time_v, utc=True),
            "horizon": horizon_v_i,
            "gt": y,
            "nwp_ref": nwp_ref_v.astype(np.float32),
            "pers_ref": pers_ref_v.astype(np.float32),
        })

    return X, y, cols, meta


def _idw_neighbour_block(
    ctx: dict,
    station_pos: np.ndarray,
    t_run_abs_arr: np.ndarray,
    neighbour_pos: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """QRF-IDW extra columns (F1, Nutzerentscheidung 2026-08-10).

    Neighbours are EXCLUSIVELY the fold's ``neighbour_pos`` training stations
    (the target station is never among them, by construction of the caller).
    Measurement channel: ``wind_speed`` at the SAME row ``pers_ref`` is built
    from (``meas_raw[t_run_abs-1, neighbour, 0]``), no window, no history mean.
    Weighting: classic inverse-distance, exponent p=1, geodesic distance via
    ``pairwise_geodesic_km`` (spatial.py:66).

    Returns (idw_mean, dist_nearest_km), each (S, P).
    """
    lats, lons = ctx["lats"], ctx["lons"]
    target_coords = np.stack([lats[station_pos], lons[station_pos]], axis=1)     # (S, 2)
    nb_coords = np.stack([lats[neighbour_pos], lons[neighbour_pos]], axis=1)     # (Nnb, 2)
    dist_km = pairwise_geodesic_km(target_coords, nb_coords)                     # (S, Nnb)
    order = np.argsort(dist_km, axis=1)[:, :n]                                   # (S, n)
    d_n = np.take_along_axis(dist_km, order, axis=1)                             # (S, n)
    nb_n = neighbour_pos[order]                                                  # (S, n)

    meas_raw = ctx["meas_raw"]
    target_feat_idx = ctx["target_feat_idx"]
    vals = meas_raw[t_run_abs_arr - 1][:, :, target_feat_idx]                    # (P, N_all)
    vals_n = vals[:, nb_n.reshape(-1)].reshape(len(t_run_abs_arr), *nb_n.shape)   # (P, S, n)
    vals_n = vals_n.transpose(1, 0, 2)                                           # (S, P, n)

    eps = 1e-6
    w = 1.0 / np.maximum(d_n, eps)                                              # (S, n)
    w = w / w.sum(axis=1, keepdims=True)
    idw_mean = np.einsum("spn,sn->sp", vals_n, w)                               # (S, P)
    dist_nearest = np.broadcast_to(d_n[:, 0][:, None], (len(station_pos), len(t_run_abs_arr)))
    return idw_mean.astype(np.float32), dist_nearest.astype(np.float32)


# ---------------------------------------------------------------------------
# MOS design rows — one small tidy table, grouped per (station, lead) by caller
# ---------------------------------------------------------------------------

def build_mos_rows(
    ctx: dict,
    station_pos: np.ndarray,
    pairs: list[tuple[int, int, int]],
    nwp_sources: str = "both",
) -> pd.DataFrame:
    """Tidy (station, pair, lead) rows with the MOS predictors (Spezifikation 3.5).

    Columns: station_id, run_time, valid_time, horizon, ws_i2, ws_e2 (NaN if
    ``nwp_sources == 'icond2'`` or ECMWF unavailable), y, pers_ref, nwp_ref.
    """
    all_ids = ctx["all_ids"]
    F_h = ctx["F_h"]
    S = len(station_pos)
    P = len(pairs)
    station_pos = np.asarray(station_pos, dtype=np.int64)
    if S == 0 or P == 0:
        return pd.DataFrame(columns=["station_id", "run_time", "valid_time", "horizon",
                                      "ws_i2", "ws_e2", "y", "pers_ref", "nwp_ref"])

    r_curr_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    t_run_abs_arr = np.array([p[2] for p in pairs], dtype=np.int64)
    # Siehe die identische Korrektur in build_feature_matrix (oben) — dieselbe
    # off-by-one-Reparatur, Aufgabe 2 der Nacharbeit vom 2026-08-10.
    time_idx = t_run_abs_arr[:, None] + np.arange(F_h)[None, :]   # (P, F_h)

    grid_icond2_runs = ctx["grid_icond2_runs"]
    nearest_i2_idx = ctx["nearest_i2_idx"]
    nwp_ws = ctx["nwp_ws_feat_idx"]
    idx0_i2 = nearest_i2_idx[station_pos][:, 0]                                  # (S,)
    sub0_i2 = grid_icond2_runs[r_curr_arr][:, :F_h, :, :][:, :, idx0_i2, nwp_ws]  # (P, F_h, S)
    ws_i2 = sub0_i2.transpose(2, 0, 1).reshape(-1)                               # (S,P,F_h)->flat

    grid_ecmwf_raw = ctx["grid_ecmwf_raw"]
    if nwp_sources == "both" and grid_ecmwf_raw is not None:
        nearest_e2_idx = ctx["nearest_e2_idx"]
        ecmwf_ws = ctx["ecmwf_ws_feat_idx"]
        idx0_e2 = nearest_e2_idx[station_pos][:, 0]
        sub0_e2 = grid_ecmwf_raw[time_idx][:, :, idx0_e2, ecmwf_ws]              # (P, F_h, S)
        ws_e2 = sub0_e2.transpose(2, 0, 1).reshape(-1)
    else:
        ws_e2 = np.full(S * P * F_h, np.nan, dtype=np.float32)

    meas_raw = ctx["meas_raw"]
    target_feat_idx = ctx["target_feat_idx"]
    gt_pf = meas_raw[time_idx.reshape(-1), :, target_feat_idx].reshape(P, F_h, -1)[:, :, station_pos]
    y = gt_pf.transpose(2, 0, 1).reshape(-1).astype(np.float32)

    pers_vals = meas_raw[t_run_abs_arr - 1][:, station_pos, target_feat_idx]     # (P, S)
    pers_ref_v = np.broadcast_to(pers_vals.T[:, :, None], (S, P, F_h)).reshape(-1)

    nwp_ref_v = ws_i2   # ICON-D2 nearest single point — same reference for every arm

    timestamps = ctx["timestamps"]
    valid_time_pf = timestamps[time_idx.reshape(-1)].values.reshape(P, F_h)
    valid_time_v = np.broadcast_to(valid_time_pf[None, :, :], (S, P, F_h)).reshape(-1)
    run_time_v = np.broadcast_to(
        np.asarray(timestamps[t_run_abs_arr - 1].values)[None, :, None], (S, P, F_h),
    ).reshape(-1)
    horizon_v = np.broadcast_to(
        np.arange(1, F_h + 1, dtype=np.int16)[None, None, :], (S, P, F_h),
    ).reshape(-1)
    station_ids_v = np.broadcast_to(
        np.array(all_ids, dtype=object)[station_pos][:, None, None], (S, P, F_h),
    ).reshape(-1)

    return pd.DataFrame({
        "station_id": station_ids_v,
        "run_time": pd.to_datetime(run_time_v, utc=True),
        "valid_time": pd.to_datetime(valid_time_v, utc=True),
        "horizon": horizon_v,
        "ws_i2": ws_i2.astype(np.float32),
        "ws_e2": ws_e2.astype(np.float32),
        "y": y,
        "pers_ref": pers_ref_v.astype(np.float32),
        "nwp_ref": nwp_ref_v.astype(np.float32),
    })


# ---------------------------------------------------------------------------
# Fold-consistency helper
# ---------------------------------------------------------------------------

def nearest_train_station(
    ctx: dict,
    target_pos: int,
    train_pos: np.ndarray,
) -> tuple[int, float]:
    """Geodaetisch naechste Trainingsstation zu ``target_pos`` — MOS-nearest
    (Spezifikation 3.5) — nutzt ``pairwise_geodesic_km`` (spatial.py:66), NICHT
    ``cKDTree``/euklidisch in Grad."""
    lats, lons = ctx["lats"], ctx["lons"]
    t_coord = np.array([[lats[target_pos], lons[target_pos]]])
    tr_coords = np.stack([lats[train_pos], lons[train_pos]], axis=1)
    d = pairwise_geodesic_km(t_coord, tr_coords)[0]
    j = int(np.argmin(d))
    return int(train_pos[j]), float(d[j])
