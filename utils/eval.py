"""
utilities for model evaluation.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from . import tools
from . import preprocessing


# Framework-agnostic functions (identical to eval.py)

def persistence(y: pd.Series,
                horizon: int,
                from_date=None) -> pd.Series:
    """Persistence model - framework agnostic"""
    shifted = y.shift(horizon)
    y_pers = pd.Series(data=shifted, index=y.index)
    if type(y.index) == pd.core.indexes.multi.MultiIndex:
        y_pers = y_pers.reset_index().groupby('timestamp').mean().iloc[:,-1]
    if from_date:
        y_pers = y_pers[from_date:]
    return y_pers


def lin_reg(data: pd.DataFrame,
            train_end: str,
            test_start: str,
            target_col: str):
    """Linear regression persistence - framework agnostic"""
    df = data.copy()
    df.dropna(inplace=True)
    y_train = df[:train_end][target_col].values
    df.drop([target_col], axis=1, inplace=True)
    X_train = df[:train_end].values
    X_test = df[test_start:].values
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pers = model.predict(X_test)
    return y_pers


def get_synth_wind(synth_dir: str,
                   park_id: str,
                   from_date: str = None,
                   to_date: str = None,
                   params: dict = None):
    """Get synthetic wind data - framework agnostic"""
    file_path = os.path.join(synth_dir, f'synth_{park_id}.csv')
    df = pd.read_csv(file_path, sep=';')
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    turbines = params['turbines']
    turbine_params_path = os.path.join(synth_dir, 'turbine_parameter.csv')
    turbine_params = pd.read_csv(turbine_params_path, sep=';', dtype={'park_id': str})
    y_synth = pd.Series(data=np.zeros(len(df)), index=df.index)
    installed_capacity = 0
    for turbine in turbines:
        turbine_row = turbine_params.loc[turbine_params.turbine_name == turbine]
        installed_capacity += turbine_row['rated'].iloc[0]
        turbine_id = turbine_row['turbine'].iloc[0]
        y_synth += df[f'power_{turbine_id}']
    y_synth /= (installed_capacity * 1000 * 1000) # MW -> W
    if from_date and to_date:
        y_synth = y_synth[from_date:to_date]
    return y_synth


def get_synth_pv(synth_dir: str,
                 park_id: str,
                 from_date: str = None):
    """Get synthetic PV data - framework agnostic"""
    file_path = os.path.join(synth_dir, f'synth_{park_id}.csv')
    df = pd.read_csv(file_path, sep=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    y_synth = df['power']
    if from_date:
        y_synth = y_synth[from_date:]
    return y_synth


def benchmark_models(data: pd.DataFrame,
                     horizon: int,
                     train_end: str,
                     test_start: str,
                     output_dim: int,
                     index_test: np.ndarray,
                     target_col='power',
                     t_0=None):
    """Pipeline for applying benchmark models - framework agnostic"""
    results = {}
    # Persistence
    y_pers = persistence(y=data[target_col],
                        horizon=horizon,
                        from_date=test_start)
    y_pers = preprocessing.make_windows(data=y_pers,
                                        seq_len=output_dim)
    df_pers = tools.y_to_df(y=y_pers,
                      output_dim=output_dim,
                      horizon=horizon,
                      index=index_test,
                      t_0=t_0)
    results['Persistence'] = df_pers
    # linear regression persistence
    y_pers = lin_reg(data=data,
                     train_end=train_end,
                     test_start=test_start,
                     target_col=target_col)
    y_pers = preprocessing.make_windows(data=y_pers,
                                        seq_len=output_dim)
    df_pers = tools.y_to_df(y=y_pers,
                      output_dim=output_dim,
                      horizon=horizon,
                      index=index_test,
                      t_0=t_0)
    results['LinearRegression'] = df_pers
    return results


#: Default-Präfix der NWP-Baselinespalte je Zielgröße. Wind vergleicht gegen die
#: 10-m-Windgeschwindigkeit, Solar gegen die entsprechende ICON-D2-Strahlungsgröße.
#: Über params.nwp_baseline_col in der Config überschreibbar.
_NWP_BASELINE_BY_TARGET = {
    'wind_speed': 'wind_speed_h10',
    'ghi': 'ghi_nwp',
    'dhi': 'dhi_nwp',
    'bhi': 'bhi_nwp',
    'dni': 'dni_nwp',
    'kt': 'kt_nwp',
    'kd': 'kd_nwp',
}


def _default_nwp_baseline_prefix(target_col: str) -> str | None:
    """Baselinespalte für eine Zielgröße; None, wenn es keine sinnvolle NWP-Entsprechung
    gibt (z. B. target_col='power', das ICON-D2 nicht direkt prognostiziert)."""
    return _NWP_BASELINE_BY_TARGET.get(target_col)


def _resolve_nwp_baseline(nwp_baseline_col, target_col: str, n_targets: int) -> str | None:
    """Baselinespalte für *diese* Zielgröße auflösen.

    ``params.nwp_baseline_col`` darf sein:

    * ``None``  — Per-Ziel-Default aus :data:`_NWP_BASELINE_BY_TARGET`
    * ``dict``  — explizite Zuordnung ``{Zielgröße: Spalte}``, fehlende Einträge
      fallen auf den Default zurück
    * ``str``   — gilt nur bei **einer** Zielgröße

    Ein Skalar bei mehreren Zielgrößen wäre still falsch: er würde jedes Ziel gegen
    dieselbe Spalte messen. Mit ``nwp_baseline_col: 'ghi_nwp'`` und
    ``target_cols: ['ghi', 'dhi']`` wurde ``dhi`` gegen die *Global*-Prognose
    gehalten — Baseline-RMSE 165 statt 41 W/m² und damit ein Skill_NWP von +0.77,
    das nur besagt, dass GHI eine schlechte DHI-Vorhersage ist.
    """
    if isinstance(nwp_baseline_col, dict):
        return nwp_baseline_col.get(target_col) or _default_nwp_baseline_prefix(target_col)
    if nwp_baseline_col:
        if n_targets > 1:
            logging.warning(
                "params.nwp_baseline_col='%s' ist ein einzelner Spaltenname, es werden "
                "aber %d Zielgrößen ausgewertet. Für '%s' wird stattdessen der "
                "Per-Ziel-Default verwendet. Für eine explizite Zuordnung ein Dict "
                "angeben: {ghi: ghi_nwp, dhi: dhi_nwp}.",
                nwp_baseline_col, n_targets, target_col,
            )
            return _default_nwp_baseline_prefix(target_col)
        return nwp_baseline_col
    return _default_nwp_baseline_prefix(target_col)



def _can_align_nwp_by_run(data: pd.DataFrame, index_test: np.ndarray, output_dim: int) -> bool:
    """Laesst sich die NWP-Baseline laufweise ausrichten?

    Nur bei MultiIndex-NWP-Daten mit starttime/forecasttime und wenn y_to_df die
    Laufstruktur beibehaelt (output_dim > 1). Ob die Ausrichtung dann wirklich
    trifft, entscheidet ``_column_by_run`` anhand der Abdeckung — hier laesst
    sich das nicht sehen, weil ``index_test`` je nach Pipeline entweder ein
    1D-Array von starttimes oder ein (n, 3)-Array ist.
    """
    return (
        isinstance(data.index, pd.MultiIndex)
        and {'starttime', 'forecasttime'}.issubset(set(data.index.names or []))
        and output_dim != 1
    )


#: Clear-Sky-Bezugsgroesse je Zielgroesse — dieselbe Zuordnung wie der ``cs_map``
#: in solar._apply_target_transform. Waeren es verschiedene, wuerde mit einer anderen
#: Groesse zurueckgerechnet als hingerechnet wurde.
_CLEARSKY_BY_TARGET = {
    'ghi': 'ghi_clearsky', 'dhi': 'dhi_clearsky',
    'bhi': 'dni_clearsky', 'dni': 'dni_clearsky',
}


def _column_by_run(data: pd.DataFrame,
                   col: str,
                   df_true: pd.DataFrame,
                   min_coverage: float = 0.5) -> pd.DataFrame | None:
    """Spalte ``col`` als (Lauf x Lead)-Frame, deckungsgleich mit ``df_true``.

    ``None``, wenn ``df_true.index`` keine starttimes sind — dann trifft der Pivot
    nicht und der Aufrufer faellt auf den alten Pfad zurueck.
    """
    if col not in data.columns:
        return None
    piv = (data[col]
           .reset_index()
           .pivot_table(index='starttime', columns='forecasttime',
                        values=col, aggfunc='first')
           .sort_index(axis=1))
    try:
        piv = piv.reindex(df_true.index)
    except (TypeError, ValueError):
        return None

    coverage = float(piv.notna().any(axis=1).mean()) if len(piv) else 0.0
    if coverage < min_coverage:
        logging.warning(
            "Spalte laufweise nicht ausrichtbar: nur %.1f %% der %d "
            "Vorhersagefenster haben einen passenden starttime. Fallback auf den "
            "gemittelten Pfad — Skill_NWP ist dann zu optimistisch.",
            100 * coverage, len(piv)
        )
        return None

    n_leads = df_true.shape[1]
    if piv.shape[1] < n_leads:
        logging.warning(
            "NWP-Baseline hat nur %d Leads, die Vorhersage aber %d — fehlende Leads "
            "bleiben NaN.", piv.shape[1], n_leads
        )
    piv = piv.iloc[:, :n_leads]
    piv.columns = [f't+{i + 1}' for i in range(piv.shape[1])]
    return piv


def evaluate_models(pred: pd.DataFrame,
                    true: pd.DataFrame,
                    persistence: dict,
                    main_model_name='Main',
                    drop_except_main=False) -> pd.DataFrame:
    """Evaluate models against benchmarks - framework agnostic

    Alle Modelle werden auf **derselben** Stichprobe gemessen: Zeilen, in denen
    irgendeine Reihe NaN ist, fallen fuer alle weg. Sonst waeren Skill und Skill_NWP
    Quotienten zweier RMSE ueber verschiedene Teilmengen — die Baseline kann
    ausgerechnet dort fehlen, wo die Vorhersage leicht oder schwer faellt.
    Luecken entstehen real: die Persistenz braucht ein volles Fenster in der
    Messreihe, das am Ende des Testzeitraums nicht mehr passt.
    """
    valid = np.isfinite(pred.to_numpy()).all(axis=1) & np.isfinite(true.to_numpy()).all(axis=1)
    for _m, _y in persistence.items():
        valid &= np.isfinite(_y.to_numpy()).all(axis=1)
    n_drop = int((~valid).sum())
    if n_drop:
        logging.info(
            "%d von %d Vorhersagelaeufen aus der Bewertung genommen (Luecke in "
            "Vorhersage, Messung oder einer Baseline). Alle Modelle werden auf den "
            "verbleibenden %d bewertet.", n_drop, len(valid), int(valid.sum()))
    if not valid.any():
        raise ValueError(
            "Kein einziger Vorhersagelauf ist in allen Reihen vollstaendig — "
            f"Vorhersage {pred.shape}, Baselines {[k for k in persistence]}.")
    pred, true = pred[valid], true[valid]
    persistence = {m: y[valid] for m, y in persistence.items()}

    evaluation = get_metrics(y_pred=pred.values,
                             y_true=true.values)
    evaluation['Models'] = [main_model_name]
    for model, y_pred in persistence.items():
        evaluation['Models'].append(model)
        metrics = get_metrics(y_pred=y_pred.values,
                              y_true=true.values)
        for metric, value in metrics.items():
            evaluation[metric].append(value[0])
    results = pd.DataFrame(data=evaluation)
    results['n_runs'] = int(valid.sum())
    results.set_index('Models', inplace=True)
    # skill factor
    results['Skill'] = 0.0
    results['Skill_NWP'] = np.nan
    if 'Persistence' in results.index:
        for model in evaluation['Models']:
            results.loc[model, 'Skill'] = 1 - results.loc[model].RMSE / results.loc['Persistence'].RMSE
    # Die NWP-Baseline heißt je Use-Case anders ('NWP (wind_speed_h10)',
    # 'NWP (ghi_nwp)', …) — über das Präfix suchen statt den Namen zu verdrahten.
    nwp_key = next((m for m in results.index if isinstance(m, str) and m.startswith('NWP (')), None)
    if nwp_key is not None:
        for model in evaluation['Models']:
            results.loc[model, 'Skill_NWP'] = 1 - results.loc[model].RMSE / results.loc[nwp_key].RMSE
    # drop all models except main model
    if drop_except_main:
        results = results.loc[[main_model_name]]
    return results


def get_metrics(y_pred: np.ndarray, # shape (n_samples, n_horizon)
                y_true: np.ndarray,
                detailed=False) -> dict:
    """Calculate metrics - framework agnostic"""
    error = y_pred - y_true
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(np.square(error).mean())
    mae = np.abs(error).mean()
    if detailed:
        rmse = np.mean(np.sqrt(np.mean(np.square(error), axis=0)))
        r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
    metrics = {'R^2': [r2],
               'RMSE': [rmse],
               'MAE': [mae]}
    return metrics

def get_metrics_per_forecast_run(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """
    y_pred, y_true shape: (n_samples, 48) -> z.B. (1000 Tage, 48 Stunden)
    """
    error = y_pred - y_true
    global_rmse = np.sqrt(np.mean(np.square(error)))
    global_mae = np.mean(np.abs(error))
    global_r2 = r2_score(y_true.flatten(), y_pred.flatten())

    raw_rmse_per_step = np.sqrt(np.mean(np.square(error), axis=0))
    raw_mae_per_step = np.mean(np.abs(error), axis=0)

    raw_r2_per_step = r2_score(y_true, y_pred, multioutput='raw_values')

    return {
        # Globale Zahlen (Scalar)
        'Global_RMSE': global_rmse,
        'Global_R2': global_r2,
        'RMSE_per_hour': raw_rmse_per_step,
        'MAE_per_hour': raw_mae_per_step,
        'R2_per_hour': raw_r2_per_step
    }


def evaluation_pipeline(data: pd.DataFrame,
                        model: nn.Module,
                        model_name: str,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        scaler_y: StandardScaler,
                        output_dim: int,
                        horizon: int,
                        index_test: np.ndarray,
                        test_start: str,
                        t_0: int,
                        park_id: str = None,
                        synth_dir: str = None,
                        get_physical_persistence: bool = False,
                        timestamp_col: str ='timestamp',
                        target_col: str ='power',
                        evaluate_on_all_test_data: bool = True,
                        device: str = 'cpu',
                        target_cols: list = None,
                        nwp_baseline_col=None,
                        nwp_residual: bool = False,
                        target_transform: str = 'none',
                        collect: dict | None = None) -> pd.DataFrame:
    """
    PyTorch evaluation pipeline.
    Main difference: uses PyTorch model instead of Keras model.

    Multi-Target: ist ``y_test`` dreidimensional (n, horizon, n_targets), wird die
    Vorhersage einmal berechnet und anschließend je Zielspalte ausgewertet; das
    Ergebnis erhält einen zusätzlichen Index-Level ``target``. Der Single-Target-Pfad
    bleibt davon unberührt.
    """
    # Get predictions using PyTorch
    # Beim Residuum-Ziel darf NICHT bei 0 abgeschnitten werden — rund die Haelfte
    # der Zielwerte ist dort negativ.
    y_true, y_pred = tools.get_y(X_test=X_test,
                                 y_test=y_test,
                                 scaler_y=scaler_y,
                                 model=model,
                                 device=device,
                                 clip_negative=not nwp_residual)

    if y_true.ndim == 3:
        cols = target_cols or [f'target_{i}' for i in range(y_true.shape[-1])]
        if len(cols) != y_true.shape[-1]:
            raise ValueError(
                f"target_cols hat {len(cols)} Einträge, die Vorhersage aber "
                f"{y_true.shape[-1]} Zielgrößen."
            )
        frames = []
        for i, tgt in enumerate(cols):
            part = _evaluate_single_target(
                data=data, model_name=model_name,
                y_true=y_true[:, :, i], y_pred=y_pred[:, :, i],
                output_dim=output_dim, horizon=horizon, index_test=index_test,
                test_start=test_start, t_0=t_0, park_id=park_id, synth_dir=synth_dir,
                get_physical_persistence=get_physical_persistence,
                timestamp_col=timestamp_col, target_col=tgt,
                evaluate_on_all_test_data=evaluate_on_all_test_data,
                nwp_baseline_col=nwp_baseline_col,
                _n_targets_for_baseline=len(cols),
                nwp_residual=nwp_residual,
                target_transform=target_transform,
                collect=collect,
            )
            frames.append(pd.concat({tgt: part}, names=['target']))
        return pd.concat(frames)

    return _evaluate_single_target(
        data=data, model_name=model_name, y_true=y_true, y_pred=y_pred,
        output_dim=output_dim, horizon=horizon, index_test=index_test,
        test_start=test_start, t_0=t_0, park_id=park_id, synth_dir=synth_dir,
        get_physical_persistence=get_physical_persistence,
        timestamp_col=timestamp_col,
        target_col=(target_cols[0] if target_cols else target_col),
        evaluate_on_all_test_data=evaluate_on_all_test_data,
        nwp_baseline_col=nwp_baseline_col,
        nwp_residual=nwp_residual,
        target_transform=target_transform,
        collect=collect,
    )


def _evaluate_single_target(data: pd.DataFrame,
                            model_name: str,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            output_dim: int,
                            horizon: int,
                            index_test: np.ndarray,
                            test_start: str,
                            t_0: int,
                            park_id: str = None,
                            synth_dir: str = None,
                            get_physical_persistence: bool = False,
                            timestamp_col: str = 'timestamp',
                            target_col: str = 'power',
                            evaluate_on_all_test_data: bool = True,
                            nwp_baseline_col=None,
                            _n_targets_for_baseline: int = 1,
                            nwp_residual: bool = False,
                            target_transform: str = 'none',
                            collect: dict | None = None) -> pd.DataFrame:
    """Auswertung einer einzelnen Zielgröße aus bereits berechneten Vorhersagen."""
    df_pred = tools.y_to_df(y=y_pred,
                            output_dim=output_dim,
                            horizon=horizon,
                            index=index_test,
                            t_0=None if evaluate_on_all_test_data else t_0)
    df_true = tools.y_to_df(y=y_true,
                            output_dim=output_dim,
                            horizon=horizon,
                            index=index_test,
                            t_0=None if evaluate_on_all_test_data else t_0)

    # Clear-Sky-Index zurueckrechnen: das Ziel ist k = ghi / ghi_clearsky, die
    # NWP-Baseline steht aber in W/m². Ohne Ruecktransformation waere Skill_NWP ein
    # Vergleich zweier verschiedener Einheiten. Faktor laufweise ausrichten, damit
    # Spalte 't+j' denselben Lead meint wie in der Vorhersage.
    #
    # Grenze der Umkehrung: solar.clearsky_index setzt k unterhalb von 20 W/m²
    # Clear-Sky-Einstrahlung hart auf 0. Nacht und tiefe Daemmerung lassen sich
    # deshalb nicht rekonstruieren — dort werden Vorhersage UND Wahrheit zu 0,
    # waehrend die NWP-Baseline an echten Werten gemessen wird. Das schoent
    # Skill_NWP zugunsten des Clear-Sky-Laufs. Der Betrag ist klein (nachts ist
    # ghi_nwp selbst nahe 0), aber er ist nicht null.
    cs_factor = None
    if target_transform == 'clearsky_index':
        cs_col = _CLEARSKY_BY_TARGET.get(target_col)
        if cs_col is None:
            logging.warning(
                "target_transform='clearsky_index', aber fuer Zielgroesse '%s' ist "
                "keine Clear-Sky-Bezugsgroesse bekannt (%s). Es wird NICHT "
                "zurueckgerechnet — die Metriken stehen dann in k-Einheiten.",
                target_col, sorted(_CLEARSKY_BY_TARGET))
        elif _can_align_nwp_by_run(data, index_test, output_dim):
            cs_factor = _column_by_run(data, cs_col, df_true)
        if target_transform == 'clearsky_index' and cs_col is not None and cs_factor is None:
            raise ValueError(
                f"Clear-Sky-Ruecktransformation fuer '{target_col}' nicht moeglich: "
                f"Spalte '{cs_col}' fehlt im Datensatz oder laesst sich nicht laufweise "
                "ausrichten. Ohne sie waeren RMSE (in k) und die NWP-Baseline (in W/m²) "
                "nicht vergleichbar und Skill_NWP waere bedeutungslos.")
        if cs_factor is not None:
            df_pred = df_pred * cs_factor.to_numpy()
            df_true = df_true * cs_factor.to_numpy()

    test_indices = None
    if len(index_test.shape) != 1:
        test_indices = index_test[:,0]
        if output_dim == 1:
            test_indices = np.array(list(set(index_test[:,2])))
            test_indices.sort()
            y_pers = y_pers[test_indices]
            y_pers_raw = pd.DataFrame(index_test, columns=list(data.index.names))
            y_pers = y_pers_raw.merge(y_pers.to_frame(), how='left', on=timestamp_col)[target_col].values
            test_indices = None
    else:
        test_indices = index_test

    pers = {}
    if isinstance(data.index, pd.MultiIndex):
        tz = data.index.get_level_values('timestamp').tz
    else:
        tz = data.index.tz
    test_start_ts = pd.Timestamp(test_start, tz=tz)
    y_pers = persistence(y=data[target_col],
                        horizon=horizon,
                        from_date=test_start_ts)
    # get persistence (most recent value)
    _seq_len = y_pred.shape[-1]
    _pers_index = None
    if test_indices is not None and isinstance(y_pers.index, pd.DatetimeIndex):
        # Dieselbe Auswahl, die make_windows intern trifft: nur Fensterstarts, fuer
        # die noch ein volles Fenster in die Reihe passt. Am Ende des Testzeitraums
        # fallen dadurch Laeufe weg — frueher lieferte y_to_df dann stillschweigend
        # "Shape of passed values is (198, 96), indices imply (206, 96)".
        _starts = y_pers.index[: max(len(y_pers.index) - _seq_len + 1, 0)]
        _pers_index = _starts[_starts.isin(pd.Index(test_indices))]
    y_pers = preprocessing.make_windows(data=y_pers,
                                        seq_len=_seq_len,
                                        step_size=1,
                                        indices=test_indices)
    if _pers_index is not None and len(_pers_index) == len(y_pers):
        df_pers = pd.DataFrame(
            y_pers, index=_pers_index,
            columns=[f't+{i + 1}' for i in range(y_pers.shape[-1])],
        ).reindex(df_true.index)
    else:
        df_pers = tools.y_to_df(y=y_pers,
                                output_dim=output_dim,
                                horizon=horizon,
                                index=index_test,
                                t_0=None if evaluate_on_all_test_data else t_0)
    if cs_factor is not None:
        # Persistenz stammt aus data[target_col], also ebenfalls aus dem k-Raum.
        df_pers = df_pers * cs_factor.to_numpy()
    pers['Persistence'] = df_pers

    # NWP-Baseline: die rohe Prognose derselben Größe am nächsten Gitterpunkt.
    # Default je Use-Case — Wind: 'wind_speed_h10', Solar: 'ghi_nwp'/'dhi_nwp'/…;
    # per params.nwp_baseline_col in der Config überschreibbar.
    nwp_prefix = _resolve_nwp_baseline(nwp_baseline_col, target_col, _n_targets_for_baseline)
    nwp_col = None
    if nwp_prefix:
        nwp_cols = [c for c in data.columns if c.startswith(nwp_prefix)]
        nwp_col = next((c for c in nwp_cols if c.endswith('_1')),
                       nwp_cols[0] if nwp_cols else None)
    if nwp_residual and nwp_col is not None:
        # Im Residuumsraum IST die rohe NWP-Prognose die Nullreihe: das Ziel ist
        # bereits 'Messung - NWP'. Deren RMSE gegen y_true ist damit exakt der
        # NWP-Fehler in W/m², und Skill_NWP bleibt Zahl fuer Zahl vergleichbar mit
        # einem Lauf ohne Residuum. Die NWP-Spalte selbst zu nehmen waere hier
        # doppelt gezaehlt.
        pers[f'NWP ({nwp_prefix})'] = df_pred * 0.0
    elif nwp_col is not None and _can_align_nwp_by_run(data, index_test, output_dim) \
            and (df_nwp := _column_by_run(data, nwp_col, df_true)) is not None:
        # Baseline exakt so aufbauen wie die Vorhersage: bei NWP-Daten bildet
        # create_tft_sequences EIN Fenster je Vorhersagelauf, Spalte 't+j' ist also
        # 'forecasttime j-1' desselben starttime. Ein Pivot auf (starttime x
        # forecasttime) trifft genau diese Zuordnung.
        #
        # Der frühere Pfad (weiter unten) hat die NWP-Spalte vorher per
        # groupby('timestamp').mean() über ALLE überlappenden Läufe gemittelt. Damit
        # bekam die Baseline eine Glättung, die die Modellvorhersage nicht bekommt —
        # ihr RMSE fiel zu niedrig aus und Skill_NWP entsprechend zu gut.
        # Nachgerechnet an Station 00853 (Solar, ghi, 145 Testläufe): der gemittelte
        # Pfad liefert 84.61 W/m², die laufweise Zuordnung 92.84 — und 92.84 ist
        # exakt der Wert, den der Residuum-Lauf als Nullreihen-RMSE meldet.
        pers[f'NWP ({nwp_prefix})'] = df_nwp
    elif nwp_col is not None:
        y_nwp = data[nwp_col]
        if isinstance(y_nwp.index, pd.MultiIndex):
            y_nwp = y_nwp.reset_index().groupby('timestamp').mean().iloc[:, -1]
        y_nwp = y_nwp[test_start_ts:]
        y_nwp_windows = preprocessing.make_windows(data=y_nwp,
                                                   seq_len=y_pred.shape[-1],
                                                   step_size=1,
                                                   indices=test_indices)
        df_nwp = tools.y_to_df(y=y_nwp_windows,
                                output_dim=output_dim,
                                horizon=horizon,
                                index=index_test,
                                t_0=None if evaluate_on_all_test_data else t_0)
        pers[f'NWP ({nwp_prefix})'] = df_nwp

    # get physical persistence
    if get_physical_persistence:
        if 'wind' in synth_dir:
            y_synth = get_synth_wind(synth_dir=synth_dir,
                                    park_id=park_id,
                                    from_date=index_test[0],
                                    to_date=data.index[-1])
        elif 'solar' in synth_dir:
            y_synth = get_synth_pv(synth_dir=synth_dir,
                                    park_id=park_id,
                                    from_date=index_test[0],
                                    to_date=data.index[-1])
        y_synth = preprocessing.make_windows(data=y_synth,
                                             seq_len=y_pred.shape[-1],
                                             step_size=1,
                                             indices=test_indices)
        df_synth = tools.y_to_df(y=y_pers,
                                output_dim=output_dim,
                                horizon=horizon,
                                index=index_test,
                                t_0=None if evaluate_on_all_test_data else t_0)
        pers['Synth'] = df_synth

    # Vorhersagen je (Station, Zielgroesse) herausreichen, wenn angefordert. Alles
    # in W/m² und laufweise ausgerichtet — damit laesst sich spaeter auf ein
    # gemeinsames Zeitraster aggregieren und ueber Aufloesungen hinweg vergleichen,
    # was Skill_NWP allein nicht kann (die NWP-Baseline aendert sich mit dem Raster).
    if collect is not None:
        collect[(park_id, target_col)] = {
            'pred': df_pred.copy(), 'true': df_true.copy(),
            **{f'baseline::{k}': v.copy() for k, v in pers.items()},
        }

    evaluation = evaluate_models(pred=df_pred,
                                 true=df_true,
                                 persistence=pers,
                                 main_model_name=model_name,
                                 drop_except_main=True)
    return evaluation


def evaluate_retrain(config,
                     data,
                     index_test,
                     model,
                     hyperparameters,
                     cols,
                     scaler_y=None,
                     target_col='power',
                     device='cpu'):
    """
    PyTorch retraining evaluation.
    Main difference: uses PyTorch training instead of Keras fit().
    """
    t_0 = None if config['eval']['eval_on_all_test_data'] else config['eval']['t_0']
    retrain_interval = config['eval']['retrain_interval']
    output_dim = config['model']['output_dim']
    horizon = config['model']['horizon']
    freq = config['data']['freq']
    known, observed, static = cols

    if freq == '15min':
        retrain_interval /= 4

    full_days = len(index_test) // retrain_interval - 1
    y_true, y_pred, y_pers, index = None, None, None, None

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.get('lr', 0.001))
    quantiles = config['model'].get('tft', {}).get('quantiles', None)
    if quantiles:
        criterion = lambda pred, tgt: tools._pinball_loss(pred, tgt, quantiles)
    else:
        criterion = nn.MSELoss()

    for day in range(full_days):
        from_index = day * retrain_interval
        if output_dim == 1:
            adj_horizon = horizon
            to_index = day * retrain_interval + horizon
        else:
            adj_horizon = retrain_interval
            to_index = retrain_interval * (1 + day)

        index_day = index_test[from_index:to_index]
        prepared_data, df = preprocessing.pipeline(data=data,
                                                   config=config,
                                                   known_cols=known,
                                                   observed_cols=observed,
                                                   static_cols=static,
                                                   test_start=index_day[0])

        X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
        X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
        X_test, y_test = X_test[:adj_horizon], y_test[:adj_horizon]

        y_true_new, y_pred_new = tools.get_y(X_test=X_test,
                                             y_test=y_test,
                                             model=model,
                                             scaler_y=scaler_y,
                                             device=device)

        y_pers_raw = persistence(y=df[target_col],
                                 horizon=horizon,
                                 from_date=str(index_test[0].date()))
        y_pers_new = preprocessing.make_windows(y_pers_raw, y_pred_new.shape[-1], step_size=1)
        y_pers_new = y_pers_new[from_index:to_index]

        if y_pred is None:
            y_true, y_pred, y_pers, index = y_true_new, y_pred_new, y_pers_new, index_day
        else:
            y_true = np.concatenate((y_true, y_true_new))
            y_pred = np.concatenate((y_pred, y_pred_new))
            y_pers = np.concatenate((y_pers, y_pers_new))
            index = np.concatenate((index, index_day))

        # PyTorch retraining
        model.train()
        train_loader = tools.create_pytorch_dataloader(
            X_train, y_train,
            batch_size=hyperparameters['batch_size'],
            shuffle=config['model']['shuffle'],
            device=device
        )

        for epoch in range(2):  # Quick retrain
            for batch in train_loader:
                if isinstance(X_train, dict):
                    # TFT case
                    if len(batch) == 4:
                        obs, known, static, targets = [b.to(device) for b in batch]
                    else:
                        obs, known, targets = [b.to(device) for b in batch]
                        static = None
                    predictions = model(obs, known, static)
                else:
                    inputs, targets = [b.to(device) for b in batch]
                    predictions = model(inputs)

                loss = criterion(predictions, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    df_pred = tools.y_to_df(y_pred, output_dim, horizon, index, t_0)
    df_pers = tools.y_to_df(y_pers, output_dim, horizon, index, t_0)
    df_true = tools.y_to_df(y_true, output_dim, horizon, index, t_0)
    pers = {}
    pers['Persistence'] = df_pers
    evaluation = evaluate_models(pred=df_pred,
                                 true=df_true,
                                 persistence=pers,
                                 main_model_name=config['model']['name'],
                                 drop_except_main=True)
    return evaluation
