"""
guard.py — one-line variant banner plus the hard assertion that protects
ablation variant B.

Used identically by ``train_dcrnn.py``, ``get_test_results_dcrnn.py`` and
``hpo_dcrnn.py`` so no entry point can quietly disagree with the others.

The assertion
-------------
``interpolate_history: true`` appends a **regression-kriging lag channel** to
``meas_hist`` *after* the zeroing step (sampler.py).  That estimate is
interpolated from the other stations' measurements.  With
``neighbour_meas_available: false`` it would therefore hand variant B exactly the
information the ablation is supposed to remove, over a path that bypasses the
station graph completely — and it would do so without any visible symptom.  That
is the one failure mode that yields a plausible but meaningless number, so it is
a hard abort rather than a warning.

Ablation D guard (R6(a), docs/review_round2_findings.md)
----------------------------------------------------------
``nwp_aggregation: idw`` (geostatistics/dcrnn/model/nwp_attention.py) replaces
the learned GATv2 attention weights with fixed inverse-distance weights read
from column 0 of the icond2/ecmwf -> station edge_attr. That column only
exists, and only means "distance", under three preconditions — each a silent
failure mode, not a crash, if left unchecked:

  * ``nwp_nodes: true``      — otherwise there is no NWPAttentionLayer at all
    (NWP features are concatenated into station.x instead) and 'idw' has
    nothing to attach to.
  * ``nwp_injection: true``  — otherwise nwp_out_dim is forced to 0 and the
    layer is never constructed, same failure mode as above from the other flag.
  * a distance edge feature enabled (``use_distance_features: true``, the
    default, or ``distance`` present in ``edge_features``) — otherwise column 0
    of edge_attr is NOT a distance (it would be bearing sin, or altitude diff,
    or whatever feature happens to sit first), and ``d**-p`` would silently
    weight stations by the wrong quantity.

``idw_p`` is range-checked here too (``validate_idw_p``): it is a free config
field outside the HPO search space, and p <= 0 fails silently rather than
loudly (inverts or flattens the weighting instead of crashing). It no longer
has an upper bound — see nwp_attention.py's "Min-distance renormalisation"
section for why the previous float32-overflow cap was removed.

Ablation D' guard (idw_alt, height-corrected IDW)
----------------------------------------------------
Same three preconditions as D, plus a fourth: ``use_altitude_diff: true`` (or
``altitude_diff`` present in ``edge_features``) — idw_alt reads the
altitude-diff column to build its 3D distance, so without it there is no
height information to correct with (and, absent the earlier checks, the
column position derived by DCRNNConfig.altitude_diff_col() would not even
exist). ``alpha_alt`` is range-checked the same way ``idw_p`` is
(``validate_alpha_alt``).
"""
from __future__ import annotations

import logging

from geostatistics.dcrnn.model.nwp_attention import validate_alpha_alt, validate_idw_p
from geostatistics.stgnn.config import parse_edge_features


class AblationConfigError(ValueError):
    """Raised when an ablation flag combination would silently invalidate a run."""


def check_ablation_flags(dcrnn_cfg: dict, logger: logging.Logger | None = None) -> dict:
    """Log the resolved variant flags and abort on an invalidating combination.

    Call this **after** any HPO best-params override has been applied, so the
    banner shows the values the run actually uses.

    Returns the resolved flags as a dict (handy for tests).
    """
    flags = {
        "neighbour_meas_available": bool(dcrnn_cfg.get("neighbour_meas_available", True)),
        "hist_wind_available": bool(dcrnn_cfg.get("hist_wind_available", False)),
        "interpolate_history": bool(dcrnn_cfg.get("interpolate_history", False)),
        "station_connectivity": dcrnn_cfg.get("station_connectivity", "delaunay"),
        "direction_to_adj": bool(dcrnn_cfg.get("direction_to_adj", False)),
        "nwp_nodes": bool(dcrnn_cfg.get("nwp_nodes", True)),
        "nwp_injection": bool(dcrnn_cfg.get("nwp_injection", True)),
        "nwp_aggregation": dcrnn_cfg.get("nwp_aggregation", "attention"),
    }

    if flags["neighbour_meas_available"] and flags["station_connectivity"] != "none":
        variant = "A (full model)"
    elif flags["station_connectivity"] == "none":
        variant = "C (no station graph)"
    else:
        variant = "B (no neighbour measurements)"
    flags["variant"] = variant

    log = logger or logging.getLogger(__name__)
    log.info(
        "ABLATION VARIANT %s — neighbour_meas_available=%s  hist_wind_available=%s  "
        "interpolate_history=%s  station_connectivity=%s  direction_to_adj=%s  nwp_nodes=%s  "
        "nwp_injection=%s  nwp_aggregation=%s",
        variant,
        flags["neighbour_meas_available"], flags["hist_wind_available"],
        flags["interpolate_history"], flags["station_connectivity"],
        flags["direction_to_adj"], flags["nwp_nodes"],
        flags["nwp_injection"], flags["nwp_aggregation"],
    )

    if not flags["neighbour_meas_available"] and flags["interpolate_history"]:
        raise AblationConfigError(
            "neighbour_meas_available=False together with interpolate_history=True is "
            "forbidden: the regression-kriging lag channel is appended AFTER the "
            "measurement channels are zeroed and is interpolated from the other "
            "stations' measurements, so ablation variant B would receive neighbour "
            "measurement information on a path that bypasses the station graph. "
            "Set interpolate_history: false in the variant config."
        )

    if flags["nwp_aggregation"] not in ("attention", "idw", "idw_alt"):
        raise AblationConfigError(
            f"nwp_aggregation must be 'attention', 'idw' or 'idw_alt', got "
            f"{flags['nwp_aggregation']!r}."
        )

    if flags["nwp_aggregation"] in ("idw", "idw_alt"):
        agg = flags["nwp_aggregation"]
        if not flags["nwp_nodes"]:
            raise AblationConfigError(
                f"nwp_aggregation={agg!r} requires nwp_nodes=true: with nwp_nodes=false "
                "there is no NWPAttentionLayer to switch — NWP features go straight "
                f"into station.x instead. Set nwp_nodes: true, or drop nwp_aggregation: {agg}."
            )
        if not flags["nwp_injection"]:
            raise AblationConfigError(
                f"nwp_aggregation={agg!r} requires nwp_injection=true: with "
                "nwp_injection=false, nwp_out_dim is forced to 0 and the "
                "NWPAttentionLayer is never constructed. Set nwp_injection: true, "
                f"or drop nwp_aggregation: {agg}."
            )
        try:
            validate_idw_p(dcrnn_cfg.get("idw_p", 2.0))
        except (ValueError, TypeError) as exc:
            raise AblationConfigError(f"nwp_aggregation={agg!r}: {exc}") from exc

        use_distance, _, use_altitude_diff, _ = parse_edge_features(dcrnn_cfg)
        if not use_distance:
            raise AblationConfigError(
                f"nwp_aggregation={agg!r} requires a distance edge feature (the default "
                "use_distance_features=true, or 'distance' present in edge_features): "
                "the IDW weights are read from column 0 of edge_attr, which is only a "
                "distance when that feature is enabled — otherwise column 0 holds "
                "whatever feature (bearing sin, altitude diff, …) happens to sit first, "
                "and d**-p would silently weight stations by the wrong quantity."
            )

        if agg == "idw_alt":
            if not use_altitude_diff:
                raise AblationConfigError(
                    "nwp_aggregation='idw_alt' requires the altitude_diff edge feature "
                    "(use_altitude_diff: true, or 'altitude_diff' present in edge_features): "
                    "idw_alt's 3D distance needs a height difference to correct with. Set "
                    "use_altitude_diff: true, or use nwp_aggregation: idw (no height term)."
                )
            try:
                validate_alpha_alt(dcrnn_cfg.get("alpha_alt", 10.0))
            except (ValueError, TypeError) as exc:
                raise AblationConfigError(f"nwp_aggregation='idw_alt': {exc}") from exc

    return flags
