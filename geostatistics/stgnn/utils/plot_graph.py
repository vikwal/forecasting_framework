"""
Visualise a HeterogeneousGraphBuilder result on an interactive folium map.

Usage
-----
::

    from stgnn import GraphConfig, HeterogeneousGraphBuilder
    from stgnn.utils.plot_graph import plot_hetero_graph

    builder = HeterogeneousGraphBuilder(cfg)
    graph = builder.build(station_coords, station_alt, ic_coords, ec_coords)

    plot_hetero_graph(
        graph=graph,
        station_coords=station_coords,
        icond2_coords=ic_coords,
        ecmwf_coords=ec_coords,
        station_labels=station_ids,   # optional list of strings
        output_path="graph.html",
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import folium
import numpy as np
from torch_geometric.data import HeteroData


# Colour palette
_COLORS = {
    "station": "#2196F3",   # blue
    "icond2":  "#F44336",   # red
    "ecmwf":   "#4CAF50",   # green
    "s2s":     "#2196F3",   # station–station edges
    "i2s":     "#F44336",   # icond2→station edges
    "e2s":     "#4CAF50",   # ecmwf→station edges
}


def plot_hetero_graph(
    graph: HeteroData,
    station_coords: np.ndarray,
    icond2_coords: np.ndarray,
    ecmwf_coords: np.ndarray,
    station_labels: Sequence[str] | None = None,
    icond2_labels: Sequence[str] | None = None,
    ecmwf_labels: Sequence[str] | None = None,
    show_s2s_edges: bool = True,
    show_nwp_edges: bool = True,
    output_path: str | Path = "graph.html",
) -> folium.Map:
    """
    Plot the heterogeneous graph on an interactive folium map.

    Parameters
    ----------
    graph :            HeteroData returned by HeterogeneousGraphBuilder.build()
    station_coords :   (N_s, 2) [lat, lon] degrees
    icond2_coords :    (N_i, 2) [lat, lon] degrees
    ecmwf_coords :     (N_e, 2) [lat, lon] degrees
    station_labels :   optional list of station ID strings (shown on hover)
    icond2_labels :    optional list of ICON-D2 grid point labels
    ecmwf_labels :     optional list of ECMWF grid point labels
    show_s2s_edges :   draw station ↔ station edges
    show_nwp_edges :   draw NWP → station edges
    output_path :      where to save the HTML file

    Returns
    -------
    The folium.Map object (also saved to disk).
    """
    # --- Centre map on mean station position ---
    center_lat = float(station_coords[:, 0].mean())
    center_lon = float(station_coords[:, 1].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron",
    )

    # --- Layer groups (toggleable in the layer control) ---
    lg_stations = folium.FeatureGroup(name="Stationen", show=True)
    lg_icond2   = folium.FeatureGroup(name="ICON-D2 Gitterpunkte", show=True)
    lg_ecmwf    = folium.FeatureGroup(name="ECMWF Gitterpunkte", show=True)
    lg_s2s      = folium.FeatureGroup(name="Station–Station Kanten", show=True)
    lg_i2s      = folium.FeatureGroup(name="ICON-D2 → Station Kanten", show=False)
    lg_e2s      = folium.FeatureGroup(name="ECMWF → Station Kanten", show=False)

    # ----------------------------------------------------------------
    # Edges (draw first so nodes appear on top)
    # ----------------------------------------------------------------

    if show_s2s_edges:
        ei = graph["station", "near", "station"].edge_index.numpy()
        # Undirected: only draw i < j to avoid double lines
        drawn: set[tuple[int, int]] = set()
        for src, dst in zip(ei[0], ei[1]):
            key = (min(int(src), int(dst)), max(int(src), int(dst)))
            if key in drawn:
                continue
            drawn.add(key)
            folium.PolyLine(
                locations=[
                    [station_coords[src, 0], station_coords[src, 1]],
                    [station_coords[dst, 0], station_coords[dst, 1]],
                ],
                color=_COLORS["s2s"],
                weight=1.2,
                opacity=0.5,
            ).add_to(lg_s2s)

    if show_nwp_edges:
        # ICON-D2 → station
        ei = graph["icond2", "informs", "station"].edge_index.numpy()
        for src, dst in zip(ei[0], ei[1]):
            folium.PolyLine(
                locations=[
                    [icond2_coords[src, 0], icond2_coords[src, 1]],
                    [station_coords[dst, 0], station_coords[dst, 1]],
                ],
                color=_COLORS["i2s"],
                weight=0.8,
                opacity=0.3,
                dash_array="4 4",
            ).add_to(lg_i2s)

        # ECMWF → station
        ei = graph["ecmwf", "informs", "station"].edge_index.numpy()
        for src, dst in zip(ei[0], ei[1]):
            folium.PolyLine(
                locations=[
                    [ecmwf_coords[src, 0], ecmwf_coords[src, 1]],
                    [station_coords[dst, 0], station_coords[dst, 1]],
                ],
                color=_COLORS["e2s"],
                weight=0.8,
                opacity=0.3,
                dash_array="4 4",
            ).add_to(lg_e2s)

    # ----------------------------------------------------------------
    # Nodes
    # ----------------------------------------------------------------

    # ICON-D2 grid points
    for i, (lat, lon) in enumerate(icond2_coords):
        label = icond2_labels[i] if icond2_labels else f"ICON-D2 #{i}"
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=_COLORS["icond2"],
            fill=True,
            fill_color=_COLORS["icond2"],
            fill_opacity=0.7,
            tooltip=label,
        ).add_to(lg_icond2)

    # ECMWF grid points
    for i, (lat, lon) in enumerate(ecmwf_coords):
        label = ecmwf_labels[i] if ecmwf_labels else f"ECMWF #{i}"
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=_COLORS["ecmwf"],
            fill=True,
            fill_color=_COLORS["ecmwf"],
            fill_opacity=0.7,
            tooltip=label,
        ).add_to(lg_ecmwf)

    # Stations (largest, drawn last → on top)
    for i, (lat, lon) in enumerate(station_coords):
        label = station_labels[i] if station_labels else f"Station #{i}"
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=_COLORS["station"],
            fill_opacity=0.9,
            tooltip=label,
        ).add_to(lg_stations)

    # ----------------------------------------------------------------
    # Legend (custom HTML)
    # ----------------------------------------------------------------
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; padding: 12px 16px; border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-family: sans-serif;
        font-size: 13px; line-height: 1.8;
    ">
        <b>Legende</b><br>
        <span style="color:{s};">&#11044;</span> Wetterstation ({ns})<br>
        <span style="color:{i};">&#11044;</span> ICON-D2 Gitterpunkt ({ni})<br>
        <span style="color:{e};">&#11044;</span> ECMWF Gitterpunkt ({ne})<br>
        <hr style="margin:6px 0;">
        <span style="color:{s};">&#9135;</span> Station–Station Kante ({es})<br>
        <span style="color:{i}; opacity:0.6;">&#9135; &#9135;</span> ICON-D2 → Station ({ei})<br>
        <span style="color:{e}; opacity:0.6;">&#9135; &#9135;</span> ECMWF → Station ({ee})<br>
    </div>
    """.format(
        s=_COLORS["station"], i=_COLORS["icond2"], e=_COLORS["ecmwf"],
        ns=len(station_coords), ni=len(icond2_coords), ne=len(ecmwf_coords),
        es=graph["station", "near", "station"].edge_index.shape[1] // 2,
        ei=graph["icond2", "informs", "station"].edge_index.shape[1],
        ee=graph["ecmwf", "informs", "station"].edge_index.shape[1],
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    # ----------------------------------------------------------------
    # Add layers and controls
    # ----------------------------------------------------------------
    for lg in [lg_s2s, lg_i2s, lg_e2s, lg_icond2, lg_ecmwf, lg_stations]:
        lg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"Graph saved to {output_path.resolve()}")

    return m
