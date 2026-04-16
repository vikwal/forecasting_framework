"""
plot_interpolation_map.py — Interactive station map for RK interpolation quality.

Reads:
  results/geostatistics/wind_interpol_results_per_station.csv
  data/stations_master.csv

Produces:
  results/geostatistics/wind_interpol_map.html

Both RMSE and R² layers are shown; toggle between them via the layer control
in the top-right corner.  Green = better, red = worse.

Usage (run from forecasting_framework/):
    python geostatistics/plot_interpolation_map.py
    python geostatistics/plot_interpolation_map.py \\
        --results  results/geostatistics/wind_interpol_results_per_station.csv \\
        --stations data/stations_master.csv \\
        --out      results/geostatistics/wind_interpol_map.html
"""
from __future__ import annotations

import argparse
import colorsys
import os

import folium
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _value_to_hex(value: float, vmin: float, vmax: float, *, invert: bool = False) -> str:
    """
    Map *value* in [vmin, vmax] to a hex colour.
    Green (#2ecc71) = best end, Red (#e74c3c) = worst end.
    Set *invert=True* when lower is better (e.g. RMSE).
    """
    if vmax == vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)   # 0 = low, 1 = high
    if invert:
        t = 1.0 - t                           # flip: low RMSE → green

    # Hue: 0.0 = red (worst), 0.33 = green (best)
    hue = t * 0.33
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.88)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _add_colorbar_legend(m: folium.Map, vmin: float, vmax: float,
                         label: str, *, invert: bool, layer_name: str) -> None:
    """Inject an SVG colour-bar into the map as a FloatImage-style HTML macro."""
    n = 200
    stops = []
    for i in range(n + 1):
        t = i / n
        hex_col = _value_to_hex(t * (vmax - vmin) + vmin, vmin, vmax, invert=invert)
        stops.append(f'<stop offset="{t:.3f}" stop-color="{hex_col}"/>')
    gradient = "\n".join(stops)

    lo_label = f"{vmin:.2f}"
    hi_label = f"{vmax:.2f}"

    # If invert, the visual left is green (best) → annotate accordingly
    left_txt  = f"best ({lo_label})"  if invert else lo_label
    right_txt = f"worst ({hi_label})" if invert else hi_label
    if not invert:
        left_txt  = f"worst ({lo_label})"
        right_txt = f"best ({hi_label})"

    html = f"""
    <div id="legend_{layer_name}"
         style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:8px 12px;border-radius:6px;
                box-shadow:0 2px 6px rgba(0,0,0,.3);font-family:sans-serif;
                font-size:12px;display:none;">
      <b>{label}</b><br>
      <span style="font-size:10px">{left_txt}</span>
      <svg width="160" height="14" style="display:block;margin:4px 0">
        <defs>
          <linearGradient id="grad_{layer_name}" x1="0" y1="0" x2="1" y2="0">
            {gradient}
          </linearGradient>
        </defs>
        <rect width="160" height="14" fill="url(#grad_{layer_name})"/>
      </svg>
      <span style="font-size:10px">{right_txt}</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def _layer_toggle_js(layer_names: list[str]) -> str:
    """
    JS that shows the matching legend div whenever its layer is toggled on,
    and hides all others.
    """
    listeners = []
    for name in layer_names:
        js_name = name.replace(" ", "_").replace("²", "2").replace("^", "")
        listeners.append(f"""
        map.on('overlayadd', function(e) {{
          if (e.name === '{name}') {{
            document.getElementById('legend_{js_name}').style.display = 'block';
          }}
        }});
        map.on('overlayremove', function(e) {{
          if (e.name === '{name}') {{
            document.getElementById('legend_{js_name}').style.display = 'none';
          }}
        }});
        """)
    # Show the first layer's legend by default
    first_js = layer_names[0].replace(" ", "_").replace("²", "2").replace("^", "")
    listeners.append(f"""
        document.getElementById('legend_{first_js}').style.display = 'block';
    """)
    return "<script>" + "\n".join(listeners) + "</script>"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RK interpolation quality map")
    parser.add_argument("--results",  default="results/geostatistics/wind_interpol_results_per_station.csv")
    parser.add_argument("--stations", default="data/stations_master.csv")
    parser.add_argument("--out",      default="results/geostatistics/wind_interpol_map.html")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    results = pd.read_csv(args.results)
    rk = results[results["method"] == "rk"].copy()
    rk["station_id"] = rk["station_id"].astype(str).str.zfill(5)

    stations = pd.read_csv(args.stations)
    stations.columns = stations.columns.str.strip().str.replace('"', '')
    stations["station_id"] = stations["station_id"].astype(str).str.zfill(5)

    df = rk.merge(stations[["station_id", "latitude", "longitude", "station_height"]],
                  on="station_id", how="left")
    missing = df["latitude"].isna().sum()
    if missing:
        print(f"Warning: {missing} stations have no coordinates — skipped")
    df = df.dropna(subset=["latitude", "longitude"])

    # ── Colour scales ────────────────────────────────────────────────────────
    rmse_min, rmse_max = df["rmse"].min(), df["rmse"].max()
    r2_min,   r2_max   = df["r2"].min(),   df["r2"].max()

    # ── Folium map ───────────────────────────────────────────────────────────
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6,
                   tiles="CartoDB positron")

    layer_rmse = folium.FeatureGroup(name="RMSE (m/s)",   show=True)
    layer_r2   = folium.FeatureGroup(name="R² ",          show=False)

    for _, row in df.iterrows():
        sid   = row["station_id"]
        lat   = float(row["latitude"])
        lon   = float(row["longitude"])
        rmse  = float(row["rmse"])
        r2    = float(row["r2"])
        alt   = int(row["station_height"]) if pd.notna(row["station_height"]) else "?"

        tooltip_rmse = (f"<b>Station {sid}</b><br>"
                        f"RMSE: <b>{rmse:.3f} m/s</b><br>"
                        f"R²: {r2:.3f}<br>"
                        f"Höhe: {alt} m")
        tooltip_r2   = (f"<b>Station {sid}</b><br>"
                        f"R²: <b>{r2:.3f}</b><br>"
                        f"RMSE: {rmse:.3f} m/s<br>"
                        f"Höhe: {alt} m")

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="white", weight=0.8,
            fill=True,
            fill_color=_value_to_hex(rmse, rmse_min, rmse_max, invert=True),
            fill_opacity=0.88,
            tooltip=folium.Tooltip(tooltip_rmse, sticky=False),
        ).add_to(layer_rmse)

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="white", weight=0.8,
            fill=True,
            fill_color=_value_to_hex(r2, r2_min, r2_max, invert=False),
            fill_opacity=0.88,
            tooltip=folium.Tooltip(tooltip_r2, sticky=False),
        ).add_to(layer_r2)

    layer_rmse.add_to(m)
    layer_r2.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # ── Legends (one per layer) ───────────────────────────────────────────────
    layer_names = ["RMSE (m/s)", "R² "]
    _add_colorbar_legend(m, rmse_min, rmse_max, "RMSE (m/s) — niedriger = besser",
                         invert=True, layer_name="RMSE_(m/s)")
    _add_colorbar_legend(m, r2_min, r2_max, "R² — höher = besser",
                         invert=False, layer_name="R2_")

    # JS: show/hide legend when layer is toggled
    m.get_root().html.add_child(folium.Element(_layer_toggle_js(layer_names)))

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    m.save(args.out)
    print(f"Saved → {args.out}")
    print(f"  Stations plotted : {len(df)}")
    print(f"  RMSE range       : {rmse_min:.3f} – {rmse_max:.3f} m/s")
    print(f"  R²   range       : {r2_min:.3f} – {r2_max:.3f}")


if __name__ == "__main__":
    main()