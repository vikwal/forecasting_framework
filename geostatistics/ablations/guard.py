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
"""
from __future__ import annotations

import logging


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
        "interpolate_history=%s  station_connectivity=%s  direction_to_adj=%s  nwp_nodes=%s",
        variant,
        flags["neighbour_meas_available"], flags["hist_wind_available"],
        flags["interpolate_history"], flags["station_connectivity"],
        flags["direction_to_adj"], flags["nwp_nodes"],
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

    return flags
