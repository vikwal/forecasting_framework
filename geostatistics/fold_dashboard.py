"""Streamlit-Dashboard fuer die raeumlichen Fold-Splits.

Zeigt fuer jeden Fold aus ``configs/spatial_folds.yaml`` (oder einer beliebigen
Fold-Config) die Train-/Val-/Test-Stationen auf der Karte und die Kennzahlen,
die fuer diese Architektur zaehlen: Distanz jeder Val-Station zu ihrer
naechsten Train-Station (= existiert der Nachbar-Messkanal noch?) und die
Terrain-Balance zwischen den Folds.

Start:
    source frcst/bin/activate
    streamlit run geostatistics/fold_dashboard.py
    # oder auf einem Remote-Host:
    streamlit run geostatistics/fold_dashboard.py --server.port 8504 \
        --server.address 0.0.0.0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "geostatistics" / "stgnn" / "utils"))

from spatial import pairwise_geodesic_km  # noqa: E402

st.set_page_config(page_title="Spatial Fold Dashboard", page_icon="🗺️", layout="wide")

COLORS = {"train": "#27F5F2", "val": "#2731F5", "test": "#9C27F5", "fix": "#B0BEC5"}
LABELS = {
    "train": "Train (Nachbar)",
    "val": "Val (Ziel)",
    "test": "Test (zurueckgehalten)",
    "fix": "immer Train",
}


# ──────────────────────────────────────────────────────────────────────────────
# Laden
# ──────────────────────────────────────────────────────────────────────────────

def norm_id(x) -> str:
    return str(x).zfill(5)


@st.cache_data(show_spinner=False)
def load_master(meta_path: str) -> pd.DataFrame:
    m = pd.read_csv(meta_path, dtype={"station_id": str})
    m["station_id"] = m["station_id"].str.zfill(5)
    return m.set_index("station_id")


@st.cache_data(show_spinner=False)
def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


@st.cache_data(show_spinner=False)
def dist_matrix(coords_a: tuple, coords_b: tuple) -> np.ndarray:
    return pairwise_geodesic_km(np.array(coords_a), np.array(coords_b))


@st.cache_data(show_spinner=False)
def load_topo(topo_dir: str, ids: tuple[str, ...]) -> pd.DataFrame | None:
    """Topo-Features als DataFrame — None, wenn das NAS/Verzeichnis fehlt."""
    try:
        from geostatistics.stgnn.utils.topo_features import (
            load_topo_station_features, TOPO_FEATURE_ORDER,
        )
        arr, names = load_topo_station_features(
            topo_dir, list(ids), TOPO_FEATURE_ORDER, n_train=len(ids),
        )
        return pd.DataFrame(arr, index=list(ids), columns=names)
    except Exception:
        return None


def coords_of(master: pd.DataFrame, ids: list[str]) -> np.ndarray:
    return np.c_[master.loc[ids, "latitude"].values, master.loc[ids, "longitude"].values]


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Quellen")
folds_path = st.sidebar.text_input(
    "Fold-Definition (YAML)", str(REPO / "configs/spatial_folds.yaml"),
    help="Datei mit spatial_fold1/2/3 → {files, val_files}",
)
base_cfg_path = st.sidebar.text_input(
    "Basis-Config (fuer test_files & Pfade)",
    str(REPO / "configs/mtgnn/config_wind_mtgnn_nwp_fold1.yaml"),
)

try:
    folds = load_yaml(folds_path)
    base = load_yaml(base_cfg_path)
except Exception as e:
    st.error(f"Konnte Konfiguration nicht laden: {e}")
    st.stop()

data_cfg = base.get("data", {})
meta_path = REPO / data_cfg.get("stations_master", "data/stations_master.csv")
master = load_master(str(meta_path))

fold_names = [k for k in folds if k.startswith("spatial_fold")]
if not fold_names:
    st.error(f"Keine `spatial_fold*`-Eintraege in {folds_path}")
    st.stop()

sel_fold = st.sidebar.selectbox("Fold", fold_names, index=0)
show_test = st.sidebar.checkbox("Test-Stationen einblenden", True)
show_links = st.sidebar.checkbox(
    "Val → naechste Train-Station verbinden", False,
    help="Zeichnet fuer jede Val-Station die Linie zu ihrem naechsten "
         "Trainingsnachbarn — macht Luecken sofort sichtbar.",
)
mark_far = st.sidebar.slider(
    "Val-Stationen ab dieser Nachbar-Distanz hervorheben (km)", 0, 150, 60, 5,
)

# ──────────────────────────────────────────────────────────────────────────────
# Aufbereitung
# ──────────────────────────────────────────────────────────────────────────────

all_ids = sorted({norm_id(x) for f in fold_names
                  for x in folds[f]["files"] + folds[f]["val_files"]})
test_ids = [norm_id(x) for x in data_cfg.get("test_files", [])]

# Fold-Zugehoerigkeit: -1 = in jedem Fold Train
assign = pd.Series(-1, index=all_ids, dtype=int)
for i, f in enumerate(fold_names):
    for s in folds[f]["val_files"]:
        assign[norm_id(s)] = i

val_ids = [norm_id(x) for x in folds[sel_fold]["val_files"]]
train_ids = [norm_id(x) for x in folds[sel_fold]["files"]]
fix_ids = [s for s in train_ids if assign[s] == -1]
rot_train_ids = [s for s in train_ids if assign[s] != -1]

D = dist_matrix(tuple(map(tuple, coords_of(master, val_ids))),
                tuple(map(tuple, coords_of(master, train_ids))))
nn_dist = D.min(axis=1)
nn_idx = D.argmin(axis=1)

st.title("Raeumliche Fold-Aufteilung")
st.caption(
    f"{len(all_ids)} rotierende Stationen aus `{Path(folds_path).name}` · "
    f"{len(test_ids)} Test-Stationen aus `{Path(base_cfg_path).name}` · "
    "Distanzen geodaetisch (WGS-84)"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Train (Nachbarn)", len(train_ids))
c2.metric("Val (Ziele)", len(val_ids))
c3.metric("Val → naechste Train, Median", f"{np.median(nn_dist):.1f} km")
c4.metric("groesste Luecke", f"{nn_dist.max():.1f} km")

# ──────────────────────────────────────────────────────────────────────────────
# Karte
# ──────────────────────────────────────────────────────────────────────────────

rows = []
for s in fix_ids:
    rows.append((s, "fix"))
for s in rot_train_ids:
    rows.append((s, "train"))
for s in val_ids:
    rows.append((s, "val"))
if show_test:
    for s in test_ids:
        rows.append((s, "test"))

df = pd.DataFrame(rows, columns=["station_id", "rolle"])
df = df.join(master[["latitude", "longitude", "station_height"]], on="station_id")
df["Rolle"] = df["rolle"].map(LABELS)
df["Nachbar-Distanz (km)"] = df["station_id"].map(
    dict(zip(val_ids, np.round(nn_dist, 1)))
).fillna(0)
df["Val-Fold"] = df["station_id"].map(
    # Test-Stationen rotieren nicht und stehen deshalb nicht in `assign`.
    {s: (f"Fold {assign.get(s, -1)+1}" if assign.get(s, -1) >= 0 else "—")
     for s in df["station_id"]}
)

fig = px.scatter_geo(
    df, lat="latitude", lon="longitude", color="Rolle",
    color_discrete_map={LABELS[k]: v for k, v in COLORS.items()},
    hover_name="station_id",
    hover_data={"Val-Fold": True, "station_height": True,
                "Nachbar-Distanz (km)": True, "latitude": False, "longitude": False},
    scope="europe",
)
fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))

if show_links:
    lats, lons = [], []
    for k, s in enumerate(val_ids):
        t = train_ids[nn_idx[k]]
        lats += [master.at[s, "latitude"], master.at[t, "latitude"], None]
        lons += [master.at[s, "longitude"], master.at[t, "longitude"], None]
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons, mode="lines", name="Val → naechste Train",
        line=dict(width=1, color="#888"), hoverinfo="skip",
    ))

far = [s for s, d in zip(val_ids, nn_dist) if d >= mark_far]
if far:
    fig.add_trace(go.Scattergeo(
        lat=master.loc[far, "latitude"], lon=master.loc[far, "longitude"],
        mode="markers", name=f"Val ≥ {mark_far} km",
        marker=dict(size=16, symbol="circle-open", color="#D32F2F", line=dict(width=2)),
        text=far, hoverinfo="text",
    ))

fig.update_geos(
    fitbounds="locations", resolution=50,
    showcountries=True, countrycolor="#999",
    showsubunits=True, subunitcolor="#ccc",
    showland=True, landcolor="#f5f5f5", showlakes=True, lakecolor="#e3f2fd",
)
fig.update_layout(height=720, margin=dict(l=0, r=0, t=10, b=0),
                  legend=dict(orientation="h", y=-0.02))
st.plotly_chart(fig, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# Kennzahlen ueber alle Folds
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("Nachbar-Abdeckung je Fold")

stats = []
for i, f in enumerate(fold_names):
    v = [norm_id(x) for x in folds[f]["val_files"]]
    t = [norm_id(x) for x in folds[f]["files"]]
    d = dist_matrix(tuple(map(tuple, coords_of(master, v))),
                    tuple(map(tuple, coords_of(master, t)))).min(axis=1)
    stats.append({
        "Fold": f.replace("spatial_", ""),
        "n_val": len(v), "n_train": len(t),
        "median (km)": round(float(np.median(d)), 1),
        "p90 (km)": round(float(np.percentile(d, 90)), 1),
        "max (km)": round(float(d.max()), 1),
        f"Val ≥ {mark_far} km": int((d >= mark_far).sum()),
    })

# Referenzzeilen: der feste Split und die finale Testauswertung
if data_cfg.get("files") and data_cfg.get("val_files"):
    ref_t = [norm_id(x) for x in data_cfg["files"]]
    ref_v = [norm_id(x) for x in data_cfg["val_files"]]
    d = dist_matrix(tuple(map(tuple, coords_of(master, ref_v))),
                    tuple(map(tuple, coords_of(master, ref_t)))).min(axis=1)
    stats.append({"Fold": "Referenz: fester Split", "n_val": len(ref_v),
                  "n_train": len(ref_t), "median (km)": round(float(np.median(d)), 1),
                  "p90 (km)": round(float(np.percentile(d, 90)), 1),
                  "max (km)": round(float(d.max()), 1),
                  f"Val ≥ {mark_far} km": int((d >= mark_far).sum())})
if test_ids:
    d = dist_matrix(tuple(map(tuple, coords_of(master, test_ids))),
                    tuple(map(tuple, coords_of(master, all_ids)))).min(axis=1)
    stats.append({"Fold": "Referenz: finaler Test", "n_val": len(test_ids),
                  "n_train": len(all_ids), "median (km)": round(float(np.median(d)), 1),
                  "p90 (km)": round(float(np.percentile(d, 90)), 1),
                  "max (km)": round(float(d.max()), 1),
                  f"Val ≥ {mark_far} km": int((d >= mark_far).sum())})

st.dataframe(pd.DataFrame(stats), width="stretch", hide_index=True)

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("**Verteilung Val → naechste Train-Station**")
    hist = px.histogram(
        pd.DataFrame({"km": nn_dist}), x="km", nbins=25,
        labels={"km": "Distanz zur naechsten Train-Station (km)"},
    )
    hist.update_traces(marker_color=COLORS["val"])
    hist.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), bargap=0.05)
    st.plotly_chart(hist, width="stretch")

with col_r:
    st.markdown("**Terrain-Balance der Folds**")
    topo_dir = None
    for arch in ("mtgnn", "dcrnn", "wavenet", "stgnn"):
        if arch in base and base[arch].get("topo_features_path"):
            topo_dir = base[arch]["topo_features_path"]
            break
    topo = load_topo(topo_dir, tuple(all_ids)) if topo_dir else None
    if topo is None:
        st.info(
            "Topo-Features nicht verfuegbar "
            f"(`{topo_dir}` nicht gemountet?) — Terrain-Balance uebersprungen."
        )
    else:
        gm, gs = topo.mean(), topo.std() + 1e-9
        bal = pd.DataFrame({
            f.replace("spatial_", ""): (
                (topo.loc[[norm_id(x) for x in folds[f]["val_files"]]].mean() - gm) / gs
            )
            for f in fold_names
        })
        st.caption("(Fold-Mittel − Gesamtmittel) / σ je Topo-Feature — "
                   "je naeher an 0, desto vergleichbarer die Folds.")
        st.dataframe(bal.round(3), width="stretch")
        st.metric("groesste Abweichung", f"{bal.abs().max().max():.3f} σ")

# ──────────────────────────────────────────────────────────────────────────────
# Stationsliste
# ──────────────────────────────────────────────────────────────────────────────

with st.expander("Stationsliste (aktueller Fold)"):
    lst = df[["station_id", "Rolle", "Val-Fold", "latitude", "longitude",
              "station_height", "Nachbar-Distanz (km)"]].sort_values(
        ["Rolle", "Nachbar-Distanz (km)"], ascending=[True, False])
    st.dataframe(lst, width="stretch", hide_index=True)
    st.download_button("Als CSV", lst.to_csv(index=False).encode(),
                       f"{sel_fold}_stations.csv", "text/csv")
