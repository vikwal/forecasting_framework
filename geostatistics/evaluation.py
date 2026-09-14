"""
geostatistics/evaluation.py — Single-pass evaluation for STGNN2 / DCRNN.

Imported by:
  - get_test_results_stgnn2.py  (standalone evaluation script)
  - train_stgnn2.py             (optional post-training eval via --eval)
  - train_dcrnn.py              (same)

Evaluation design
-----------------
  observer=train,  target=val  (zero-shot: train context, all val stations as targets)

All metrics are computed in physical units (inverse-transformed).
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from geostatistics.stgnn.training.sampler import TrainingSampler
from geostatistics.stgnn.utils.normalization import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature index helper
# ---------------------------------------------------------------------------

def find_ws_feat_idx(features: list[str]) -> int | None:
    """Return index of wind_speed_10m in features, or None if not found.

    Must match exactly — apply_dir_encoding reorders columns (non-consumed
    features first) so wind_speed_38m ends up at index 0 in dir_in_deg mode,
    which would be wrong for the NWP baseline.
    """
    for i, f in enumerate(features):
        if f == "wind_speed_10m":
            return i
    # Fallback: first feature starting with wind_speed (logs a warning at call site)
    for i, f in enumerate(features):
        if f.startswith("wind_speed"):
            return i
    return None


# ---------------------------------------------------------------------------
# Build a HeteroData eval batch for an arbitrary observer / target split
# ---------------------------------------------------------------------------

def build_eval_batch(
    sampler: TrainingSampler,
    r_curr: int,
    r_hist: int,
    t_run_abs: int,
    station_meas_scaled: np.ndarray,        # (T, N_all, M)
    station_nearest_grid: np.ndarray,       # (N_all,)
    grid_icond2_runs_scaled: np.ndarray,    # (R, 48, N_grid, I2)
    station_ecmwf_nwp_scaled: np.ndarray,  # (T, N_all, E2)
    station_static: np.ndarray,             # (N_all, S-1)  without type indicator
    ecmwf_nwp_scaled: np.ndarray,          # (T, N_ecmwf, E2)
    icond2_static: np.ndarray,
    ecmwf_static: np.ndarray,
    target_global: list[int],
    observer_global: list[int],
    fold_train_indices: list[int],
    target_feat_idx: int,
    H_hist: int,
    H_fore: int,
    interpol_meas: np.ndarray | None = None,  # (T, N_all) Kriging lag, pre-scaled
    hist_wind_available: bool = False,
    neighbour_meas_available: bool = True,   # ablation B/C: False → no station has measurements
    station_k_nearest_grid: np.ndarray | None = None,  # (N_all, k) — k nearest for nwp_nodes=False
    station_k_nearest_ecmwf: np.ndarray | None = None, # (N_all, k_e) — k nearest ECMWF, nwp_nodes=False
    station_geo: np.ndarray | None = None,             # (T, N_all, G) Sonnengeometrie
) -> tuple:
    """
    Build a HeteroData evaluation batch for the given station split.

    Returns
    -------
    data        : HeteroData (not yet on GPU)
    target_mask : (N_all,) bool tensor
    gt_scaled   : (N_target, H_fore) numpy array — scaled ground truth
    """
    all_global = observer_global + target_global
    N_obs = len(observer_global)
    N_all = len(all_global)

    target_mask = torch.zeros(N_all, dtype=torch.bool)
    target_mask[N_obs:] = True

    t_hist_abs = t_run_abs - H_hist

    if station_k_nearest_grid is not None:
        # k nearest: (N_all, k) → features (N_all, 48, k*I2) matching training
        # L = Leads je Lauf, nicht fest 48 — dieselbe Falle wie im Sampler
        # (stuendlich 48, 30 min 96), s. 88eae9d.
        L_i2    = grid_icond2_runs_scaled.shape[1]
        k_idx   = station_k_nearest_grid[all_global]             # (N_all, k)
        i2_hist = grid_icond2_runs_scaled[r_hist, :, k_idx, :].transpose(0, 2, 1, 3).reshape(N_all, L_i2, -1)
        i2_curr = grid_icond2_runs_scaled[r_curr, :, k_idx, :].transpose(0, 2, 1, 3).reshape(N_all, L_i2, -1)
    else:
        nearest = station_nearest_grid[all_global]
        i2_hist = grid_icond2_runs_scaled[r_hist, :, nearest, :]    # (N_all, 48, I2)
        i2_curr = grid_icond2_runs_scaled[r_curr, :, nearest, :]    # (N_all, 48, I2)
    i2_full = np.concatenate([i2_hist, i2_curr], axis=1)        # (N_all, 96, [k*]I2)

    i2_grid_full = np.concatenate([
        grid_icond2_runs_scaled[r_hist],
        grid_icond2_runs_scaled[r_curr],
    ], axis=0)                                                   # (96, N_grid, I2)

    e2_grid_full = ecmwf_nwp_scaled[t_hist_abs:t_run_abs + H_fore]   # (96, N_ecmwf, E2)
    if station_k_nearest_ecmwf is not None:
        # k naechste ECMWF-Punkte konkateniert, spiegelbildlich zu ICON-D2 oben
        ke_idx  = station_k_nearest_ecmwf[all_global]            # (N_all, k_e)
        e2_full = e2_grid_full[:, ke_idx, :].transpose(1, 0, 2, 3).reshape(
            N_all, e2_grid_full.shape[0], -1)                    # (N_all, 96, k_e*E2)
    else:
        e2_full = station_ecmwf_nwp_scaled[t_hist_abs:t_run_abs + H_fore, :, :][:, all_global, :]
        e2_full = e2_full.transpose(1, 0, 2)                     # (N_all, 96, E2)

    meas_hist = station_meas_scaled[t_hist_abs:t_run_abs, :, :][:, all_global, :].copy()

    geo_full = None
    if station_geo is not None:
        geo_full = station_geo[t_hist_abs:t_run_abs + H_fore, :, :][:, all_global, :]
        geo_full = geo_full.transpose(1, 0, 2)            # (N_all, T_total, G)

    # Residuum wie im Sampler: Historie gegen r_hist. VOR dem Nullen, sonst
    # liesse die Ablation B/C einen Offset stehen.
    _rs = getattr(sampler, "residual_spec", None)
    if _rs is not None:
        _ti_e = sampler.target_feat_idx
        _tidx_e = list(_ti_e) if isinstance(_ti_e, (list, tuple)) else [_ti_e]
        _nfs = station_nearest_grid[all_global]
        _H = t_run_abs - t_hist_abs
        for k, (nwp_idx, mcol) in enumerate(zip(_rs["nwp_idx"], _tidx_e)):
            if nwp_idx is None:
                continue
            ref_h = sampler._nwp_in_meas_scale(
                grid_icond2_runs_scaled, r_hist, _nfs, nwp_idx, k)
            meas_hist[:, :, mcol] -= ref_h[:, :_H].T

    # Order matters: ablation B subsumes the IGNNK zeroing and must come first,
    # otherwise only the target stations would lose their measurements. Same rule
    # as in TrainingSampler.sample_train / sample_val.
    if not neighbour_meas_available:
        meas_hist[:, :, :] = 0.0                # ablation B/C: nobody has measurements
    elif not hist_wind_available:
        meas_hist[:, N_obs:, :] = 0.0           # IGNNK masking (variant A)

    if interpol_meas is not None:
        rk_slice = interpol_meas[t_hist_abs:t_run_abs, :][:, all_global, np.newaxis]
        meas_hist = np.concatenate([meas_hist, rk_slice], axis=2)

    gt_raw = station_meas_scaled[t_run_abs:t_run_abs + H_fore, :, target_feat_idx]
    gt_scaled = gt_raw[:, all_global][:, N_obs:].T.copy()      # (N_target, H_fore)
    # gt_scaled dient nur als Rueckgabe fuer Aufrufer, die es brauchen; die
    # Metriken zieht evaluate() aus meas_raw. Im Residuumsraum entsprechend
    # gegen r_curr, damit beide dasselbe meinen.
    if _rs is not None and _rs["nwp_idx"] and _rs["nwp_idx"][0] is not None:
        _ref_c = sampler._nwp_in_meas_scale(
            grid_icond2_runs_scaled, r_curr, station_nearest_grid[all_global],
            _rs["nwp_idx"][0], 0)[:, :H_fore]
        gt_scaled = gt_scaled - _ref_c[N_obs:, :]

    stat_sub  = station_static[all_global, :]
    type_ind  = (~target_mask).float().unsqueeze(1).numpy()
    stat_full = np.concatenate([stat_sub, type_ind], axis=1)   # (N_all, S)

    data = sampler._make_data(
        all_global=all_global,
        geo_full=geo_full,
        meas_hist=meas_hist,
        i2_full=i2_full,
        e2_full=e2_full,
        stat_full=stat_full,
        icond2_nwp=i2_grid_full,
        ecmwf_nwp=e2_grid_full,
        icond2_static=icond2_static,
        ecmwf_static=ecmwf_static,
        fold_train_indices=fold_train_indices,
        target_global=target_global,
    )
    return data, target_mask, gt_scaled


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    sampler: TrainingSampler,
    device: torch.device,
    meas_raw: np.ndarray,                    # (T, N_all, M) — physical units
    meas_scaled: np.ndarray,                 # (T, N_all, M) — scaled
    station_nearest_grid: np.ndarray,        # (N_all,)
    grid_icond2_runs_raw: np.ndarray,        # (R, 48, N_grid, I2) — physical
    grid_icond2_runs_scaled: np.ndarray,
    station_ecmwf_nwp_scaled: np.ndarray,
    station_static: np.ndarray,
    ecmwf_nwp_scaled: np.ndarray,
    icond2_static: np.ndarray,
    ecmwf_static: np.ndarray,
    meas_scaler: StandardScaler,
    target_feat_idx: int,
    ws_feat_idx_i2: int | None,
    H_hist: int,
    H_fore: int,
    train_station_indices: list[int],
    val_station_indices: list[int],
    all_ids: list[str],
    test_run_pairs: list[tuple[int, int, int]],
    interpol_meas: np.ndarray | None = None,  # (T, N_all) Kriging lag, pre-scaled
    hist_wind_available: bool = False,
    neighbour_meas_available: bool = True,   # ablation B/C: False → no station has measurements
    timestamps: "pd.DatetimeIndex | None" = None,
    station_k_nearest_grid: np.ndarray | None = None,  # (N_all, k) — k nearest for nwp_nodes=False
    station_k_nearest_ecmwf: np.ndarray | None = None, # (N_all, k_e) — k nearest ECMWF, nwp_nodes=False
    station_geo: np.ndarray | None = None,   # (T, N_all, G) Sonnengeometrie
    target_feat_idxs: tuple | None = None,   # alle Zielspalten; None = Einziel
    target_names: list[str] | None = None,   # Namen dazu, fuer die target-Spalte
    nwp_ref_idxs: list | None = None,        # NWP-Referenzspalte je Ziel (None = keine)
    step_hours: float = 1.0,                 # Schrittweite (data.freq) in Stunden
    meas_observed: np.ndarray | None = None, # (T, N_all, K) True = echte Messung
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Single-pass evaluation over all test run pairs.

    All train stations serve as context; all val stations are predicted simultaneously.
    Returns (station_df, raw_df):
      station_df — per-station aggregate metrics: station_id, mae, rmse, r2, skill, skill_nwp, n_samples
                   Bei mehreren Zielgroessen kommt eine Spalte 'target' dazu und
                   es gibt eine Zeile je (Station, Zielgroesse). Der Einziel-Fall
                   ist unveraendert, ohne die Spalte.
      raw_df     — per-prediction rows: station_id, run_time, valid_time, horizon, pred, gt, nwp_ref, pers_ref
                   (run_time / valid_time are NaT when timestamps=None)

    step_hours
        Schrittweite von ``data.freq`` in Stunden. ``horizon`` zaehlt SCHRITTE,
        ``valid_time`` muss deshalb ``run_ts + (h+1) * step_hours`` sein. Mit
        dem alten festen ``hours=h+1`` behaupteten die Solar-Records (30 min,
        96 Leads) Gueltigkeitszeiten bis run+96 h statt run+48 h. Die Metriken
        laufen ueber Array-Positionen und waren nie betroffen, jeder Join auf
        ``valid_time`` und jede Tagesgang-Auswertung schon.

    meas_observed
        ``(T, N_all, K)``-Bool-Array, K in der Reihenfolge von
        ``target_feat_idxs``: True, wo an dieser Zielposition eine echte
        Messung stand. Muss VOR der Imputation gebildet werden — danach ist die
        Information weg. Ist es gesetzt, fallen imputierte Zielpositionen
        ELEMENTWEISE aus Metriken und ``raw_df``, nicht laufweise: bei 96 Leads
        kostete ein einziger imputierter Schritt sonst 48 h Auswertung.
        Modell, Persistenz und NWP sehen dieselbe Maske, sonst waeren die
        Skill-Quotienten ueber verschiedene Mengen gerechnet. Gegenstueck zu
        ``eval.exclude_imputed`` im CL-Pfad (``utils/eval.py``), das dort ueber
        die Spalte ``<target>_observed`` genau dieselben Positionen entfernt —
        erst damit werden DCRNN und TFT auf derselben Stichprobe gemessen.
    """
    preds_acc: dict[int, list[np.ndarray]] = defaultdict(list)
    gt_acc:    dict[int, list[np.ndarray]] = defaultdict(list)
    nwp_acc:   dict[int, list[np.ndarray]] = defaultdict(list)
    pers_acc:  dict[int, list[np.ndarray]] = defaultdict(list)
    obs_acc:   dict[int, list[np.ndarray]] = defaultdict(list)
    raw_records: list[dict] = []
    n_pos_total = 0      # Zielpositionen insgesamt
    n_pos_kept  = 0      # davon nach der Imputationsmaske uebrig

    # Einziel bleibt der Normalfall: dann ist _idxs einelementig, die
    # Schleifen laufen einmal und die Ausgabe traegt keine target-Spalte.
    _idxs = tuple(target_feat_idxs) if target_feat_idxs else (target_feat_idx,)
    _multi = len(_idxs) > 1
    _names = list(target_names) if target_names else [None] * len(_idxs)
    # NWP-Referenzspalte je Zielgroesse. Ohne Angabe traegt nur die erste eine
    # (ws_feat_idx_i2, der Wind-Fall); die uebrigen bekommen NaN statt still die
    # Referenz der ersten Zielgroesse zu erben.
    _nwp_idx = (list(nwp_ref_idxs) if nwp_ref_idxs is not None
                else [ws_feat_idx_i2] + [None] * (len(_idxs) - 1))
    _mean = [float(meas_scaler.mean_[i]) for i in _idxs]
    _std  = [float(meas_scaler.std_[i] + meas_scaler.eps) for i in _idxs]
    mean_ws, std_ws = _mean[0], _std[0]

    # Die Maske ist nach Zielgroesse geordnet (K-Achse == Reihenfolge von
    # _idxs), nicht nach Messspaltenindex: die Spaltenindizes verschieben sich
    # durch encode_circular_measurements, die Reihenfolge der Ziele nicht.
    if meas_observed is not None and meas_observed.shape[2] != len(_idxs):
        raise ValueError(
            f"meas_observed hat {meas_observed.shape[2]} Zielkanaele, erwartet "
            f"{len(_idxs)} (eine je Zielgroesse, in derselben Reihenfolge).")

    def _to_phys(arr: np.ndarray, k: int = 0) -> np.ndarray:
        return arr * _std[k] + _mean[k]

    common = dict(
        sampler=sampler,
        station_meas_scaled=meas_scaled,
        station_nearest_grid=station_nearest_grid,
        station_k_nearest_grid=station_k_nearest_grid,
        station_k_nearest_ecmwf=station_k_nearest_ecmwf,
        grid_icond2_runs_scaled=grid_icond2_runs_scaled,
        station_ecmwf_nwp_scaled=station_ecmwf_nwp_scaled,
        station_static=station_static,
        ecmwf_nwp_scaled=ecmwf_nwp_scaled,
        icond2_static=icond2_static,
        ecmwf_static=ecmwf_static,
        target_feat_idx=target_feat_idx,
        station_geo=station_geo,
        H_hist=H_hist,
        H_fore=H_fore,
        interpol_meas=interpol_meas,
        hist_wind_available=hist_wind_available,
        neighbour_meas_available=neighbour_meas_available,
    )

    def _nwp_ref(gidx: int, r_curr: int, k: int = 0) -> np.ndarray:
        idx = _nwp_idx[k] if k < len(_nwp_idx) else None
        if idx is None:
            return np.full(H_fore, np.nan, dtype=np.float32)
        return grid_icond2_runs_raw[
            r_curr, :H_fore, station_nearest_grid[gidx], idx
        ]

    def _pers_ref(gidx: int, t_run_abs: int, k: int = 0) -> np.ndarray:
        val = float(meas_raw[t_run_abs - 1, gidx, _idxs[k]])
        return np.full(H_fore, val, dtype=np.float32)

    # Observer (context) selection must MATCH training: the model was trained
    # seeing only the next_n_neighbors nearest train stations per target, not all
    # train stations. Using all of them at eval changes the station graph topology
    # and degrades models that rely on the neighbour context (esp. nwp_nodes=true
    # + hist_wind_available=false). Shared with sample_val so the two cannot drift.
    observer_global = sampler.select_val_neighbours(
        val_station_indices, train_station_indices,
    )
    logger.info(
        "Observer context: %d / %d train stations (next_n_neighbors=%s)",
        len(observer_global), len(train_station_indices), sampler.tc.next_n_neighbors,
    )

    model.eval()
    with torch.no_grad():
        for step, (r_curr, r_hist, t_run_abs) in enumerate(test_run_pairs):
            if step % 10 == 0:
                logger.info("  Pair %d / %d", step + 1, len(test_run_pairs))

            if not val_station_indices:
                continue

            data_a, mask_a, _ = build_eval_batch(
                **common,
                r_curr=r_curr, r_hist=r_hist, t_run_abs=t_run_abs,
                target_global=val_station_indices,
                observer_global=observer_global,
                fold_train_indices=train_station_indices,
            )
            raw_out = model(data_a.to(device), mask_a.to(device)).cpu().numpy()
            if raw_out.ndim == 2:
                raw_out = raw_out[:, :, None]       # (N_val, H_fore, 1)

            run_ts = timestamps[t_run_abs - 1] if timestamps is not None else None
            _resid = getattr(sampler, "residual_spec", None) is not None
            for k, fidx in enumerate(_idxs):
                # Im Residuumsraum sagt das Modell (Messung - NWP) in Einheiten
                # der Messwertstreuung vorher. Zurueck also nur mit *std und
                # anschliessend + NWP-Prognose — nicht mit + mean, das steckt
                # schon in der NWP-Referenz. Danach stehen alle Metriken wieder
                # im physikalischen Absolutraum und sind direkt mit den
                # TFT-Zahlen und den Absolutlaeufen vergleichbar.
                preds_a = (raw_out[:, :, k] * _std[k] if _resid
                           else _to_phys(raw_out[:, :, k], k))    # (N_val, H_fore)
                gt_a = meas_raw[
                    t_run_abs:t_run_abs + H_fore, :, fidx
                ][:, val_station_indices].T                      # (N_val, H_fore)
                obs_a = (
                    meas_observed[t_run_abs:t_run_abs + H_fore, :, k
                                  ][:, val_station_indices].T
                    if meas_observed is not None else None
                )                                                # (N_val, H_fore) bool
                # Die NWP-Referenz zeigt auf EINE Spalte des Gitters
                # (ws_feat_idx_i2). Fuer weitere Zielgroessen gibt es sie nicht,
                # ohne dass der Aufrufer sie benennt — dann bleibt skill_nwp NaN,
                # statt still die Referenz der ersten Zielgroesse zu verwenden.
                for i, gidx in enumerate(val_station_indices):
                    nwp_h  = _nwp_ref(gidx, r_curr, k)
                    pers_h = _pers_ref(gidx, t_run_abs, k)
                    pred_i = preds_a[i] + nwp_h if _resid else preds_a[i]
                    key = (gidx, k)
                    preds_acc[key].append(pred_i)
                    gt_acc[key].append(gt_a[i])
                    nwp_acc[key].append(nwp_h)
                    pers_acc[key].append(pers_h)
                    if obs_a is not None:
                        obs_acc[key].append(obs_a[i])
                    sid = all_ids[gidx]
                    n_pos_total += H_fore
                    for h in range(H_fore):
                        # Imputierte Zielposition: weder in die Metriken noch in
                        # raw_df. Die Zeile ganz wegzulassen ist hier die sichere
                        # Variante — ein nachgelagertes Skript, das ein Flag nicht
                        # kennt, wuerde sonst wieder ueber die volle Menge mitteln.
                        if obs_a is not None and not obs_a[i, h]:
                            continue
                        n_pos_kept += 1
                        rec = {
                            "station_id": sid,
                            "run_time":   run_ts,
                            # horizon zaehlt SCHRITTE, nicht Stunden — bei 30 min
                            # ist Schritt 96 der Zeitpunkt run+48 h, nicht run+96 h.
                            "valid_time": (run_ts + pd.Timedelta(hours=(h + 1) * step_hours)) if run_ts is not None else None,
                            "horizon":    h + 1,
                            "pred":       float(pred_i[h]),
                            "gt":         float(gt_a[i, h]),
                            "nwp_ref":    float(nwp_h[h]),
                            "pers_ref":   float(pers_h[h]),
                        }
                        if _multi:
                            rec["target"] = _names[k] or f"target_{k}"
                        raw_records.append(rec)

    if meas_observed is not None:
        logger.info(
            "eval.exclude_imputed: %d von %d Zielpositionen bleiben (%.2f %%), "
            "%d imputierte entfernt — elementweise, Modell und Baselines ueber "
            "derselben Menge.",
            n_pos_kept, n_pos_total,
            100.0 * n_pos_kept / max(n_pos_total, 1), n_pos_total - n_pos_kept,
        )

    logger.info("Computing per-station metrics …")
    records = []

    for gidx, k in [(g, kk) for kk in range(len(_idxs)) for g in val_station_indices]:
        key = (gidx, k)
        p_all  = np.concatenate(preds_acc[key])
        g_all  = np.concatenate(gt_acc[key])
        n_all  = np.concatenate(nwp_acc[key])
        ps_all = np.concatenate(pers_acc[key])

        # Eine gemeinsame Imputationsmaske fuer Modell, Persistenz und NWP.
        # Wuerde sie nur auf das Modell wirken, stuenden im Skill-Quotienten
        # Zaehler und Nenner ueber verschiedenen Stichproben.
        obs_all = (np.concatenate(obs_acc[key]) if obs_acc.get(key)
                   else np.ones_like(g_all, dtype=bool))

        valid = ~(np.isnan(p_all) | np.isnan(g_all)) & obs_all
        if valid.sum() < 2:
            logger.warning(
                "Station %s: too few valid samples (%d), skipping",
                all_ids[gidx], int(valid.sum()),
            )
            continue

        p_v, g_v = p_all[valid], g_all[valid]
        r2   = float(r2_score(g_v, p_v))
        rmse = float(math.sqrt(mean_squared_error(g_v, p_v)))
        mae  = float(mean_absolute_error(g_v, p_v))

        valid_pers = ~(np.isnan(ps_all) | np.isnan(g_all)) & obs_all
        if valid_pers.sum() >= 2:
            rmse_pers = float(math.sqrt(mean_squared_error(g_all[valid_pers], ps_all[valid_pers])))
            skill     = (1.0 - rmse / rmse_pers) if rmse_pers > 0 else float("nan")
        else:
            skill = float("nan")

        valid_nwp = ~(np.isnan(n_all) | np.isnan(g_all)) & obs_all
        if valid_nwp.sum() >= 2:
            rmse_nwp  = float(math.sqrt(mean_squared_error(g_all[valid_nwp], n_all[valid_nwp])))
            skill_nwp = (1.0 - rmse / rmse_nwp) if rmse_nwp > 0 else float("nan")
        else:
            skill_nwp = float("nan")

        rec = {
            "station_id": all_ids[gidx],
            "mae":        mae,
            "rmse":       rmse,
            "r2":         r2,
            "skill":      skill,
            "skill_nwp":  skill_nwp,
            "n_samples":  int(valid.sum()),
        }
        if _multi:
            rec["target"] = _names[k] or f"target_{k}"
        records.append(rec)

    return pd.DataFrame(records), pd.DataFrame(raw_records)
