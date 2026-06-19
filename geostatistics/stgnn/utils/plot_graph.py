"""
Visualise weather station graphs on an interactive folium map.

Usage (mit Kategorien und NWP-Gitterpunkten)
---------------------------------------------
::

    from stgnn.utils.plot_graph import plot_hetero_graph

    plot_hetero_graph(
        station_coords=station_coords,      # (N_s, 2) [lat, lon]
        icond2_coords=ic_coords,            # (N_i, 2) [lat, lon], optional
        ecmwf_coords=ec_coords,             # (N_e, 2) [lat, lon], optional
        s2s_edge_index=s2s_ei,             # (2, E) int
        i2s_edge_index=i2s_ei,             # (2, E) int
        e2s_edge_index=e2s_ei,             # (2, E) int
        station_categories=categories,      # list of 'train'/'val'/'test'
        station_labels=station_ids,
        output_path="stgnn_graph.html",
    )

Layer-Struktur (9 togglebare Gruppen bei Kategorie-Modus):
  Train-Nodes  — Train-Stationsmarker + zugehörige s2s-Kanten
  Val-Nodes    — Val-Stationsmarker + zugehörige s2s-Kanten
  Test-Nodes   — Test-Stationsmarker + zugehörige s2s-Kanten
  Train-ICON   — ICON-D2 Gitterpunkte + Kanten für Train-Stationen
  Val-ICON     — ...
  Test-ICON    — ...
  Train-ECMWF  — ECMWF Gitterpunkte + Kanten für Train-Stationen
  Val-ECMWF    — ...
  Test-ECMWF   — ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import folium
import numpy as np


_COLORS = {
    "train":   "#27F5F2",  # hellblau
    "val":     "#2731F5",  # mittelblau
    "test":    "#9C27F5",  # dunkelblau
    #"station": "#2196F3",  # fallback (ohne Kategorie)
    "icond2":  "#00C853",  # grün
    "ecmwf":   "#FF6D00",  # orange
    "s2s":     "#000000",  # schwarz (Station–Station-Kanten)
}

_CATS = ["train", "val", "test"]
_CAT_PRIORITY = {"train": 0, "val": 1, "test": 2}
_CAT_LABEL = {"train": "Train", "val": "Val", "test": "Test"}


def _higher_cat(a: str, b: str) -> str:
    return a if _CAT_PRIORITY[a] >= _CAT_PRIORITY[b] else b


def _nwp_node_categories(
    edge_index: np.ndarray,
    station_categories: Sequence[str],
    n_nodes: int,
) -> list[str]:
    """
    Assign each NWP node to the HIGHEST-priority connected station category.

    Ein Gitterpunkt der sowohl eine Train- als auch eine Test-Station bedient,
    landet in Test-ICON. So sieht jede Test-Station immer alle k Gitterpunkte
    in ihrem Layer — der k-NN-Radius wird pro Station vollständig abgebildet.
    """
    node_cat = ["train"] * n_nodes  # Startwert: niedrigste Priorität
    for src, dst in zip(edge_index[0], edge_index[1]):
        sc = station_categories[int(dst)]
        if _CAT_PRIORITY[sc] > _CAT_PRIORITY[node_cat[int(src)]]:
            node_cat[int(src)] = sc
    return node_cat


def plot_hetero_graph(
    station_coords: np.ndarray,
    icond2_coords: np.ndarray | None = None,
    ecmwf_coords: np.ndarray | None = None,
    s2s_edge_index: np.ndarray | None = None,
    s2s_edge_by_cat: dict[str, np.ndarray] | None = None,
    i2s_edge_index: np.ndarray | None = None,
    e2s_edge_index: np.ndarray | None = None,
    station_categories: Sequence[str] | None = None,
    station_labels: Sequence[str] | None = None,
    icond2_labels: Sequence[str] | None = None,
    ecmwf_labels: Sequence[str] | None = None,
    output_path: str | Path = "graph.html",
) -> folium.Map:
    """
    Plot weather stations (und optional NWP-Gitterpunkte) auf einer folium-Karte.

    Parameters
    ----------
    station_coords :     (N_s, 2) [lat, lon] in degrees
    icond2_coords :      (N_i, 2) — None = kein ICON-D2
    ecmwf_coords :       (N_e, 2) — None = kein ECMWF
    s2s_edge_index :     (2, E) int — Station↔Station-Kanten
    i2s_edge_index :     (2, E) int — ICON-D2→Station-Kanten
    e2s_edge_index :     (2, E) int — ECMWF→Station-Kanten
    station_categories : pro Station eine von 'train'/'val'/'test'.
                         Mit Kategorien: 9 togglebare Layer-Gruppen.
                         Ohne (None): 4 Gruppen, Einheitsfarbe.
    s2s_edge_by_cat :    Vor-berechnete Kanten pro Kategorie, z.B. aus hierarchischem Delaunay.
                         {'train': (2, E_t), 'val': (2, E_v), 'test': (2, E_te)}.
                         Hat Vorrang vor s2s_edge_index wenn beide gesetzt.
    station_labels :     Tooltip pro Station
    icond2_labels :      Tooltip pro ICON-D2-Knoten
    ecmwf_labels :       Tooltip pro ECMWF-Knoten
    output_path :        Speicherpfad für die HTML-Datei

    Returns
    -------
    folium.Map (wird auch auf Disk gespeichert)
    """
    center_lat = float(station_coords[:, 0].mean())
    center_lon = float(station_coords[:, 1].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron",
    )

    use_cats = station_categories is not None

    # ----------------------------------------------------------------
    # FeatureGroups
    # ----------------------------------------------------------------
    if use_cats:
        # 3 Stationsgruppen (Marker + s2s-Kanten) + 3 ICON + 3 ECMWF
        lg_nodes = {c: folium.FeatureGroup(name=f"{_CAT_LABEL[c]}-Nodes",  show=True)  for c in _CATS}
        lg_icon  = {c: folium.FeatureGroup(name=f"{_CAT_LABEL[c]}-ICON",   show=False) for c in _CATS}
        lg_ecmwf = {c: folium.FeatureGroup(name=f"{_CAT_LABEL[c]}-ECMWF",  show=False) for c in _CATS}
    else:
        lg_stations_all = folium.FeatureGroup(name="Stationen",                show=True)
        lg_s2s_all      = folium.FeatureGroup(name="Station–Station Kanten",   show=True)
        lg_icon_all     = folium.FeatureGroup(name="ICON-D2",                  show=False)
        lg_ecmwf_all    = folium.FeatureGroup(name="ECMWF",                    show=False)

    # ----------------------------------------------------------------
    # NWP-Knoten → Kategorie (höchste Priorität der angebundenen Station)
    # ----------------------------------------------------------------
    if use_cats:
        n_icond2 = len(icond2_coords) if icond2_coords is not None else 0
        n_ecmwf  = len(ecmwf_coords)  if ecmwf_coords  is not None else 0

        icond2_node_cat: list[str] = (
            _nwp_node_categories(i2s_edge_index, station_categories, n_icond2)
            if (i2s_edge_index is not None and n_icond2 > 0) else ["train"] * n_icond2
        )
        ecmwf_node_cat: list[str] = (
            _nwp_node_categories(e2s_edge_index, station_categories, n_ecmwf)
            if (e2s_edge_index is not None and n_ecmwf > 0) else ["train"] * n_ecmwf
        )

    # ================================================================
    # Kanten zuerst zeichnen (Knoten erscheinen darüber)
    # ================================================================

    # --- Station–Station-Kanten ---
    n_s2s_drawn = 0

    def _draw_s2s_edges(ei: np.ndarray, target_group) -> int:
        """Zeichnet Kanten aus (2,E) edge_index in target_group. Gibt Anzahl zurück."""
        drawn_set: set[tuple[int, int]] = set()
        count = 0
        for src, dst in zip(ei[0], ei[1]):
            key = (min(int(src), int(dst)), max(int(src), int(dst)))
            if key in drawn_set:
                continue
            drawn_set.add(key)
            count += 1
            folium.PolyLine(
                locations=[
                    [station_coords[src, 0], station_coords[src, 1]],
                    [station_coords[dst, 0], station_coords[dst, 1]],
                ],
                color=_COLORS["s2s"],
                weight=1.2,
                opacity=0.45,
            ).add_to(target_group)
        return count

    if use_cats and s2s_edge_by_cat is not None:
        # Hierarchisch vorberechnete Kanten: jede Kategorie hat eigene Edge-Liste
        for cat, ei in s2s_edge_by_cat.items():
            if ei.shape[1] > 0:
                n_s2s_drawn += _draw_s2s_edges(ei, lg_nodes[cat])
    elif s2s_edge_index is not None:
        if use_cats:
            # Fallback: automatische Zuweisung nach höchster Endpunkt-Priorität
            drawn_all: set[tuple[int, int]] = set()
            for src, dst in zip(s2s_edge_index[0], s2s_edge_index[1]):
                key = (min(int(src), int(dst)), max(int(src), int(dst)))
                if key in drawn_all:
                    continue
                drawn_all.add(key)
                n_s2s_drawn += 1
                cat = _higher_cat(station_categories[int(src)], station_categories[int(dst)])
                folium.PolyLine(
                    locations=[
                        [station_coords[src, 0], station_coords[src, 1]],
                        [station_coords[dst, 0], station_coords[dst, 1]],
                    ],
                    color=_COLORS["s2s"],
                    weight=1.2,
                    opacity=0.45,
                ).add_to(lg_nodes[cat])
        else:
            n_s2s_drawn += _draw_s2s_edges(s2s_edge_index, lg_s2s_all)

    # --- ICON-D2 → Station-Kanten ---
    n_i2s = 0
    if i2s_edge_index is not None and icond2_coords is not None:
        n_i2s = i2s_edge_index.shape[1]
        for src, dst in zip(i2s_edge_index[0], i2s_edge_index[1]):
            line = folium.PolyLine(
                locations=[
                    [icond2_coords[src, 0], icond2_coords[src, 1]],
                    [station_coords[dst, 0], station_coords[dst, 1]],
                ],
                color=_COLORS["icond2"],
                weight=0.8,
                opacity=0.3,
                dash_array="4 4",
            )
            if use_cats:
                line.add_to(lg_icon[icond2_node_cat[int(src)]])
            else:
                line.add_to(lg_icon_all)

    # --- ECMWF → Station-Kanten ---
    n_e2s = 0
    if e2s_edge_index is not None and ecmwf_coords is not None:
        n_e2s = e2s_edge_index.shape[1]
        for src, dst in zip(e2s_edge_index[0], e2s_edge_index[1]):
            line = folium.PolyLine(
                locations=[
                    [ecmwf_coords[src, 0], ecmwf_coords[src, 1]],
                    [station_coords[dst, 0], station_coords[dst, 1]],
                ],
                color=_COLORS["ecmwf"],
                weight=0.8,
                opacity=0.3,
                dash_array="4 4",
            )
            if use_cats:
                line.add_to(lg_ecmwf[ecmwf_node_cat[int(src)]])
            else:
                line.add_to(lg_ecmwf_all)

    # ================================================================
    # Knoten zeichnen
    # ================================================================

    # --- ICON-D2-Knoten ---
    if icond2_coords is not None:
        for i, (lat, lon) in enumerate(icond2_coords):
            label = icond2_labels[i] if icond2_labels else f"ICON-D2 #{i}"
            marker = folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=_COLORS["icond2"],
                fill=True,
                fill_color=_COLORS["icond2"],
                fill_opacity=0.7,
                tooltip=label,
            )
            if use_cats:
                marker.add_to(lg_icon[icond2_node_cat[i]])
            else:
                marker.add_to(lg_icon_all)

    # --- ECMWF-Knoten ---
    if ecmwf_coords is not None:
        for i, (lat, lon) in enumerate(ecmwf_coords):
            label = ecmwf_labels[i] if ecmwf_labels else f"ECMWF #{i}"
            marker = folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=_COLORS["ecmwf"],
                fill=True,
                fill_color=_COLORS["ecmwf"],
                fill_opacity=0.7,
                tooltip=label,
            )
            if use_cats:
                marker.add_to(lg_ecmwf[ecmwf_node_cat[i]])
            else:
                marker.add_to(lg_ecmwf_all)

    # --- Stationsknoten (zuletzt → obendrauf) ---
    for i, (lat, lon) in enumerate(station_coords):
        label = station_labels[i] if station_labels else f"Station #{i}"
        cat   = station_categories[i] if use_cats else None
        color = _COLORS[cat] if cat else _COLORS["station"]

        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=label,
        )
        if use_cats:
            marker.add_to(lg_nodes[cat])
        else:
            marker.add_to(lg_stations_all)

    # ================================================================
    # Legende
    # ================================================================
    if use_cats:
        legend_html = """
        <div style="
            position: fixed; bottom: 30px; left: 30px; z-index: 1000;
            background: white; padding: 12px 16px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-family: sans-serif;
            font-size: 13px; line-height: 1.8;
        ">
            <b>Legende</b><br>
            <span style="color:{train};">&#11044;</span> Train-Station ({n_train})<br>
            <span style="color:{val};">&#11044;</span> Val-Station ({n_val})<br>
            <span style="color:{test};">&#11044;</span> Test-Station ({n_test})<br>
            <hr style="margin:6px 0;">
            <span style="color:#000;">&#9135;</span> Station–Station-Kante ({n_s2s})<br>
            <span style="color:{icond2}; opacity:0.7;">&#9135; &#9135;</span> ICON-D2 → Station ({n_i2s})<br>
            <span style="color:{ecmwf}; opacity:0.7;">&#9135; &#9135;</span> ECMWF → Station ({n_e2s})<br>
        </div>
        """.format(
            train=_COLORS["train"], val=_COLORS["val"], test=_COLORS["test"],
            icond2=_COLORS["icond2"], ecmwf=_COLORS["ecmwf"],
            n_train=sum(1 for c in station_categories if c == "train"),
            n_val=sum(1 for c in station_categories if c == "val"),
            n_test=sum(1 for c in station_categories if c == "test"),
            n_s2s=n_s2s_drawn, n_i2s=n_i2s, n_e2s=n_e2s,
        )
    else:
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
            <span style="color:{s};">&#9135;</span> Station–Station ({es})<br>
            <span style="color:{i}; opacity:0.6;">&#9135; &#9135;</span> ICON-D2 → Station ({ei})<br>
            <span style="color:{e}; opacity:0.6;">&#9135; &#9135;</span> ECMWF → Station ({ee})<br>
        </div>
        """.format(
            s=_COLORS["station"], i=_COLORS["icond2"], e=_COLORS["ecmwf"],
            ns=len(station_coords),
            ni=len(icond2_coords) if icond2_coords is not None else 0,
            ne=len(ecmwf_coords)  if ecmwf_coords  is not None else 0,
            es=n_s2s_drawn, ei=n_i2s, ee=n_e2s,
        )

    m.get_root().html.add_child(folium.Element(legend_html))

    # ================================================================
    # Layer zur Karte hinzufügen
    # ================================================================
    if use_cats:
        # Kanten zuerst (unterste Ebene), dann NWP, dann Stationen
        for c in _CATS:
            lg_nodes[c].add_to(m)
        for c in _CATS:
            lg_icon[c].add_to(m)
        for c in _CATS:
            lg_ecmwf[c].add_to(m)
    else:
        for lg in [lg_s2s_all, lg_icon_all, lg_ecmwf_all, lg_stations_all]:
            lg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ================================================================
    # Speichern
    # ================================================================
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"Graph saved to {output_path.resolve()}")

    return m
