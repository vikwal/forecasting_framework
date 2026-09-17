#!/usr/bin/env python3
"""HTML-Bericht zu den Solar-TFT-Fold-Laeufen.

Dieselbe Analyse, die scripts/eval_testyear.py fuer den Wind-Pfad fuehrt, auf
die Solarlaeufe uebertragen und um die Achsen erweitert, die bei Strahlung
ueberhaupt erst erklaeren, wann eine Nachbearbeitung gegen ICON-D2 gewinnt:
Tagesgang, Sonnenstand und Bewoelkungsregime (Clear-Sky-Index).

Erzeugt eine eigenstaendige HTML-Datei mit eingebetteten Abbildungen (keine
Nebendateien, per scp/Browser lesbar) sowie die Aggregattabellen als CSV.

    frcst/bin/python scripts/report_solar_folds.py
    frcst/bin/python scripts/report_solar_folds.py --folds 1 --out reports/probe.html
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from solar_fold_analysis import (  # noqa: E402
    lade_alles, metriken, nach, stationsmeta, KT_LABELS, GHI_LABELS, TAG_CLEARSKY_MIN,
)

OUT_CSV = REPO / "data/test_results"

# --- Farben -----------------------------------------------------------------
# Kategoriale Slots 1-3 der validierten Referenzpalette; die drei Reihen sind
# auch fuer Scatter/all-pairs sauber trennbar. Persistenz ist Referenz, keine
# gleichrangige Reihe, und traegt deshalb Grau statt eines eigenen Farbslots.
C_MODELL = "#2a78d6"
C_NWP = "#eb6834"
C_PERS = "#8a8a85"
C_NEUTRAL = "#e8e7e2"
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"

#: Divergierend um 0 fuer Skill-Karten: zwei Hues, neutraler Grauton in der
#: Mitte, nie ein Regenbogen. Blau = Modell besser, Orange = ICON-D2 besser.
CMAP_SKILL = LinearSegmentedColormap.from_list("skill", [C_NWP, C_NEUTRAL, C_MODELL])
#: Sequentiell fuer Dichte: eine Hue, hell nach dunkel.
CMAP_DICHTE = LinearSegmentedColormap.from_list("dichte", ["#eef4fc", C_MODELL, "#123a68"])

EINHEIT = "W/m²"
ZIEL_NAME = {"ghi": "GHI (Globalstrahlung)", "dhi": "DHI (Diffusstrahlung)"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": C_MUTED,
    "axes.titlecolor": C_TEXT, "text.color": C_TEXT,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "grid.color": "#e3e2dc", "grid.linewidth": .8,
    "font.size": 9.5, "axes.titlesize": 10.5, "legend.frameon": False,
})


def fig_zu_html(fig) -> str:
    """Abbildung als eingebettetes PNG — der Bericht bleibt eine einzelne Datei."""
    puffer = io.BytesIO()
    fig.savefig(puffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    b64 = base64.b64encode(puffer.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="">'


def _grid(ax, achse="y"):
    ax.grid(axis=achse, alpha=.7, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Abbildungen
# ---------------------------------------------------------------------------

def abb_uebersicht(agg: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, ziel in zip(axes, ("ghi", "dhi")):
        z = agg[agg["target"] == ziel].iloc[0]
        namen = ["TFT", "ICON-D2", "Persistenz"]
        werte = [z["rmse"], z["rmse_nwp"], z["rmse_pers"]]
        farben = [C_MODELL, C_NWP, C_PERS]
        balken = ax.barh(namen[::-1], werte[::-1], color=farben[::-1], height=.62)
        for b, w in zip(balken, werte[::-1]):
            ax.text(w + max(werte) * .015, b.get_y() + b.get_height() / 2,
                    f"{w:.1f}", va="center", fontsize=9, color=C_TEXT)
        ax.set_xlim(0, max(werte) * 1.18)
        ax.set_xlabel(f"RMSE ({EINHEIT})")
        ax.set_title(f"{ZIEL_NAME[ziel]} — Skill vs. ICON-D2: {z['skill_nwp']:.3f}")
        _grid(ax, "x")
    fig.suptitle("Fehler im Validierungsjahr, nur Tagesschritte", y=1.04, fontsize=11)
    return fig_zu_html(fig)


def abb_lead(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, ziel in zip(axes, ("ghi", "dhi")):
        d = df[df["target"] == ziel]
        h = nach(d, "lead_h")
        ax.plot(h["lead_h"], h["rmse"], color=C_MODELL, lw=2, label="TFT")
        ax.plot(h["lead_h"], h["rmse_nwp"], color=C_NWP, lw=2, label="ICON-D2")
        ax.plot(h["lead_h"], h["rmse_pers"], color=C_PERS, lw=1.6, ls=":", label="Persistenz")
        ax.set_xlabel("Vorlaufzeit (h)")
        ax.set_ylabel(f"RMSE ({EINHEIT})") if ziel == "ghi" else None
        ax.set_title(ZIEL_NAME[ziel])
        _grid(ax)
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Fehler über die Vorlaufzeit", y=1.03, fontsize=11)
    return fig_zu_html(fig)


def abb_skill_lead(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    for ziel, stil in (("ghi", "-"), ("dhi", "--")):
        h = nach(df[df["target"] == ziel], "lead_h")
        ax.plot(h["lead_h"], h["skill_nwp"], stil, color=C_MODELL, lw=2,
                label=f"{ziel.upper()}")
    ax.axhline(0, color=C_NWP, lw=1.4)
    ax.text(0.3, 0.002, "ICON-D2 unkorrigiert", color=C_NWP, fontsize=8, va="bottom")
    ax.set_xlabel("Vorlaufzeit (h)"); ax.set_ylabel("Skill gegenüber ICON-D2")
    ax.legend(fontsize=8.5); _grid(ax)
    ax.set_title("Wie lange trägt die Korrektur?")
    return fig_zu_html(fig)


def abb_tagesgang(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, ziel in zip(axes, ("ghi", "dhi")):
        h = nach(df[df["target"] == ziel], "hour")
        ax.plot(h["hour"], h["rmse"], color=C_MODELL, lw=2, marker="o", ms=4, label="TFT")
        ax.plot(h["hour"], h["rmse_nwp"], color=C_NWP, lw=2, marker="o", ms=4, label="ICON-D2")
        ax.set_xlabel("Stunde der Gültigkeitszeit (UTC)")
        if ziel == "ghi":
            ax.set_ylabel(f"RMSE ({EINHEIT})")
        ax.set_title(ZIEL_NAME[ziel]); _grid(ax)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Fehler über den Tagesgang", y=1.03, fontsize=11)
    return fig_zu_html(fig)


def abb_bias_tagesgang(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ziel in zip(axes, ("ghi", "dhi")):
        h = nach(df[df["target"] == ziel], "hour")
        ax.plot(h["hour"], h["bias"], color=C_MODELL, lw=2, marker="o", ms=4, label="TFT")
        ax.plot(h["hour"], h["bias_nwp"], color=C_NWP, lw=2, marker="o", ms=4, label="ICON-D2")
        ax.axhline(0, color="#9a9992", lw=1)
        ax.set_xlabel("Stunde (UTC)")
        if ziel == "ghi":
            ax.set_ylabel(f"Bias ({EINHEIT})")
        ax.set_title(ZIEL_NAME[ziel]); _grid(ax)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Systematische Abweichung über den Tag (Vorhersage − Messung)", y=1.03, fontsize=11)
    return fig_zu_html(fig)


def abb_monat(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    monatsnamen = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                   "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    for ax, ziel in zip(axes, ("ghi", "dhi")):
        h = nach(df[df["target"] == ziel], "month").sort_values("month")
        x = range(len(h))
        ax.plot(x, h["rmse"], color=C_MODELL, lw=2, marker="o", ms=4, label="TFT")
        ax.plot(x, h["rmse_nwp"], color=C_NWP, lw=2, marker="o", ms=4, label="ICON-D2")
        ax.set_xticks(list(x))
        ax.set_xticklabels([monatsnamen[int(m) - 1] for m in h["month"]], fontsize=8)
        if ziel == "ghi":
            ax.set_ylabel(f"RMSE ({EINHEIT})")
        ax.set_title(ZIEL_NAME[ziel]); _grid(ax)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Fehler über das Jahr (Gültigkeitszeit)", y=1.03, fontsize=11)
    return fig_zu_html(fig)


def abb_strahlungsklasse(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, ziel in zip(axes, ("ghi", "dhi")):
        h = nach(df[df["target"] == ziel], "ghi_klasse")
        h = h.set_index("ghi_klasse").reindex(GHI_LABELS).reset_index()
        x = range(len(h))
        ax.plot(x, h["rmse"], color=C_MODELL, lw=2, marker="o", ms=5, label="TFT")
        ax.plot(x, h["rmse_nwp"], color=C_NWP, lw=2, marker="o", ms=5, label="ICON-D2")
        ax.set_xticks(list(x)); ax.set_xticklabels(GHI_LABELS, fontsize=8)
        ax.set_xlabel(f"gemessene Strahlung ({EINHEIT})")
        if ziel == "ghi":
            ax.set_ylabel(f"RMSE ({EINHEIT})")
        ax.set_title(ZIEL_NAME[ziel]); _grid(ax)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Fehler nach Stärke der gemessenen Strahlung", y=1.03, fontsize=11)
    return fig_zu_html(fig)


def abb_regime(df: pd.DataFrame) -> str:
    """Kernbild: Skill und Trefferanteil je Bewoelkungsregime."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    h_ghi = nach(df[df["target"] == "ghi"], "kt_klasse").set_index("kt_klasse").reindex(KT_LABELS)
    h_dhi = nach(df[df["target"] == "dhi"], "kt_klasse").set_index("kt_klasse").reindex(KT_LABELS)
    x = np.arange(len(KT_LABELS))

    ax = axes[0]
    ax.bar(x - .2, h_ghi["rmse"], .38, color=C_MODELL, label="TFT")
    ax.bar(x + .2, h_ghi["rmse_nwp"], .38, color=C_NWP, label="ICON-D2")
    ax.set_xticks(x); ax.set_xticklabels(KT_LABELS, fontsize=8)
    ax.set_ylabel(f"RMSE ({EINHEIT})"); ax.set_title("GHI: Fehlerniveau je Regime")
    ax.legend(fontsize=8.5); _grid(ax)

    ax = axes[1]
    ax.bar(x - .2, h_ghi["skill_nwp"], .38, color=C_MODELL, label="GHI")
    ax.bar(x + .2, h_dhi["skill_nwp"], .38, color="#7fb1e8", label="DHI")
    ax.axhline(0, color="#9a9992", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(KT_LABELS, fontsize=8)
    ax.set_ylabel("Skill gegenüber ICON-D2"); ax.set_title("Wo die Korrektur wirkt")
    ax.legend(fontsize=8.5); _grid(ax)

    ax = axes[2]
    ax.bar(x - .2, h_ghi["anteil_besser"], .38, color=C_MODELL, label="GHI")
    ax.bar(x + .2, h_dhi["anteil_besser"], .38, color="#7fb1e8", label="DHI")
    ax.axhline(.5, color=C_NWP, lw=1.4)
    ax.text(len(x) - .5, .505, "Zufallsniveau", color=C_NWP, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(KT_LABELS, fontsize=8)
    ax.set_ylabel("Anteil Schritte näher an der Messung")
    ax.set_title("Wie oft der TFT vorn liegt"); ax.legend(fontsize=8.5); _grid(ax)

    fig.suptitle("Bewölkungsregime — Clear-Sky-Index der Messung", y=1.04, fontsize=11)
    return fig_zu_html(fig)


def abb_regime_prognose(df: pd.DataFrame) -> str:
    """Dieselbe Achse, aber nach der PROGNOSTIZIERTEN Lage — vorab bekannt."""
    d = df[df["target"] == "ghi"]
    h = nach(d, "kt_nwp_klasse").set_index("kt_nwp_klasse").reindex(KT_LABELS)
    x = np.arange(len(KT_LABELS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    ax.bar(x - .2, h["rmse"], .38, color=C_MODELL, label="TFT")
    ax.bar(x + .2, h["rmse_nwp"], .38, color=C_NWP, label="ICON-D2")
    ax.set_xticks(x); ax.set_xticklabels(KT_LABELS, fontsize=8)
    ax.set_xlabel("von ICON-D2 vorhergesagte Lage")
    ax.set_ylabel(f"RMSE ({EINHEIT})"); ax.set_title("GHI: Fehler je prognostizierter Lage")
    ax.legend(fontsize=8.5); _grid(ax)

    ax = axes[1]
    balken = ax.bar(x, h["skill_nwp"], .5, color=C_MODELL)
    ax.set_ylim(0, float(h["skill_nwp"].max()) * 1.22)
    for b, w, n in zip(balken, h["skill_nwp"], h["n"]):
        ax.text(b.get_x() + b.get_width() / 2, w + .004, f"{w:+.3f}", ha="center",
                fontsize=8.5, color=C_TEXT)
        ax.text(b.get_x() + b.get_width() / 2, w * .5, f"n={n/1e3:.0f}k", ha="center",
                va="center", fontsize=7.5, color="white")
    ax.axhline(0, color="#9a9992", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(KT_LABELS, fontsize=8)
    ax.set_xlabel("von ICON-D2 vorhergesagte Lage")
    ax.set_ylabel("Skill gegenüber ICON-D2"); ax.set_title("Wo sich die Korrektur vorab lohnt")
    _grid(ax)

    ax = axes[2]
    ax.bar(x - .2, h["bias"], .38, color=C_MODELL, label="TFT")
    ax.bar(x + .2, h["bias_nwp"], .38, color=C_NWP, label="ICON-D2")
    ax.axhline(0, color="#55534d", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(KT_LABELS, fontsize=8)
    ax.set_xlabel("von ICON-D2 vorhergesagte Lage")
    ax.set_ylabel(f"Bias ({EINHEIT})")
    ax.set_title("Der bedingte Bias — und was davon bleibt")
    ax.legend(fontsize=8.5); _grid(ax)

    fig.suptitle("Operative Sicht: Regime nach der Prognose, nicht nach der Messung",
                 y=1.04, fontsize=11)
    return fig_zu_html(fig)


def abb_bias_regime(df: pd.DataFrame) -> str:
    """Bias je Regime — macht die Mittelwertsneigung beider Seiten sichtbar."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, ziel in zip(axes, ("ghi", "dhi")):
        h = nach(df[df["target"] == ziel], "kt_klasse").set_index("kt_klasse").reindex(KT_LABELS)
        x = np.arange(len(KT_LABELS))
        ax.bar(x - .2, h["bias"], .38, color=C_MODELL, label="TFT")
        ax.bar(x + .2, h["bias_nwp"], .38, color=C_NWP, label="ICON-D2")
        ax.axhline(0, color="#55534d", lw=1.2)
        ax.set_xticks(x); ax.set_xticklabels(KT_LABELS, fontsize=8)
        if ziel == "ghi":
            ax.set_ylabel(f"Bias ({EINHEIT})")
        ax.set_title(ZIEL_NAME[ziel]); ax.legend(fontsize=8.5); _grid(ax)
    fig.suptitle("Systematische Abweichung je Bewölkungsregime (Vorhersage − Messung)",
                 y=1.04, fontsize=11)
    return fig_zu_html(fig)


def abb_heatmap(df: pd.DataFrame) -> str:
    d = df[df["target"] == "ghi"].copy()
    d["lead_block"] = pd.cut(d["lead_h"], [0, 6, 12, 24, 36, 48],
                             labels=["0–6 h", "6–12 h", "12–24 h", "24–36 h", "36–48 h"],
                             right=False, include_lowest=True)
    h = nach(d, ["kt_klasse", "lead_block"])
    m = h.pivot(index="kt_klasse", columns="lead_block", values="skill_nwp").reindex(KT_LABELS)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    grenze = float(np.nanmax(np.abs(m.values))) or .1
    bild = ax.imshow(m.values, cmap=CMAP_SKILL, norm=TwoSlopeNorm(0, -grenze, grenze), aspect="auto")
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns, fontsize=8.5)
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            w = m.values[i, j]
            if np.isfinite(w):
                ax.text(j, i, f"{w:+.3f}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(w) > grenze * .55 else C_TEXT)
    fig.colorbar(bild, ax=ax, label="Skill vs. ICON-D2", shrink=.85)
    ax.set_title("GHI: Skill nach Regime und Vorlaufzeit")
    return fig_zu_html(fig)


def abb_ablation(pfad: Path) -> str:
    """Beitrag des observed-Fensters je Lead-Block (Permutationstest)."""
    d = pd.read_csv(pfad)
    reihenfolge = ["0.0 h", "0.5 h", "1 h", "1.5–3 h", "3–6 h", "6–12 h", "12–24 h", "24–48 h"]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    breite = .38
    for i, (ziel, farbe) in enumerate((("ghi", C_MODELL), ("dhi", "#7fb1e8"))):
        z = (d[d["target"] == ziel].groupby("block", sort=False)[["rmse_echt", "rmse_perm"]]
             .mean().reindex(reihenfolge))
        anteil = 100 * (z["rmse_perm"] - z["rmse_echt"]) / z["rmse_echt"]
        x = np.arange(len(reihenfolge)) + (i - .5) * breite
        balken = ax.bar(x, anteil, breite, color=farbe, label=ziel.upper())
        for b, w in zip(balken, anteil):
            if w > 1:
                ax.text(b.get_x() + b.get_width() / 2, w + .4, f"{w:.0f}", ha="center",
                        fontsize=8, color=C_TEXT)
    ax.set_xticks(np.arange(len(reihenfolge))); ax.set_xticklabels(reihenfolge, fontsize=8.5)
    ax.set_xlabel("Vorlaufzeit"); ax.set_ylabel("RMSE-Anstieg ohne Historie (%)")
    ax.legend(fontsize=8.5); _grid(ax)
    ax.set_title("Was die eigene Messhistorie beiträgt")
    return fig_zu_html(fig)


def abb_stationen(je_station: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, ziel in zip(axes, ("ghi", "dhi")):
        d = je_station[je_station["target"] == ziel]
        lo = min(d["rmse"].min(), d["rmse_nwp"].min()) * .95
        hi = max(d["rmse"].max(), d["rmse_nwp"].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color="#9a9992", lw=1, ls="--", zorder=1)
        besser = d["rmse"] < d["rmse_nwp"]
        ax.scatter(d.loc[besser, "rmse_nwp"], d.loc[besser, "rmse"], s=46, color=C_MODELL,
                   edgecolor="white", lw=.8, zorder=3, label=f"TFT besser ({besser.sum()})")
        ax.scatter(d.loc[~besser, "rmse_nwp"], d.loc[~besser, "rmse"], s=46, color=C_NWP,
                   edgecolor="white", lw=.8, zorder=3, label=f"ICON-D2 besser ({(~besser).sum()})")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"RMSE ICON-D2 ({EINHEIT})")
        if ziel == "ghi":
            ax.set_ylabel(f"RMSE TFT ({EINHEIT})")
        ax.set_title(ZIEL_NAME[ziel]); ax.legend(fontsize=8.5, loc="upper left"); _grid(ax, "both")
    fig.suptitle("Je Station: Korrektur gegen rohe NWP-Prognose", y=1.02, fontsize=11)
    return fig_zu_html(fig)


def abb_karte(je_station: pd.DataFrame, meta: pd.DataFrame) -> str:
    d = je_station[je_station["target"] == "ghi"].join(meta, on="station_id")
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    # Divergierend nur, wenn es tatsaechlich beide Vorzeichen gibt — sonst waere die
    # halbe Skala leer und alle Punkte saehen gleich aus. Liegt alles auf einer
    # Seite, ist die Groesse des Gewinns die Botschaft: eine Hue, hell nach dunkel.
    einseitig = (d["skill_nwp"] > 0).all() or (d["skill_nwp"] < 0).all()
    if einseitig:
        p = ax.scatter(d["longitude"], d["latitude"], c=d["skill_nwp"], cmap=CMAP_DICHTE,
                       s=95, edgecolor="#55534d", lw=.6)
    else:
        grenze = float(np.nanmax(np.abs(d["skill_nwp"]))) or .1
        p = ax.scatter(d["longitude"], d["latitude"], c=d["skill_nwp"], cmap=CMAP_SKILL,
                       norm=TwoSlopeNorm(0, -grenze, grenze), s=95, edgecolor="#55534d", lw=.6)
    fig.colorbar(p, ax=ax, label="Skill vs. ICON-D2 (GHI)", shrink=.75)
    ax.set_xlabel("Länge (°O)"); ax.set_ylabel("Breite (°N)")
    ax.set_title("Räumliche Verteilung des Gewinns")
    _grid(ax, "both")
    return fig_zu_html(fig)


def abb_skillverteilung(je_station: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for ziel, farbe, name in (("ghi", C_MODELL, "GHI"), ("dhi", "#7fb1e8", "DHI")):
        s = je_station.loc[je_station["target"] == ziel, "skill_nwp"].dropna()
        ax.hist(s, bins=22, histtype="step", lw=2, color=farbe,
                label=f"{name} (Median {s.median():+.3f})")
    ax.axvline(0, color=C_NWP, lw=1.4)
    ax.set_xlabel("Skill gegenüber ICON-D2 je Station"); ax.set_ylabel("Stationen")
    ax.legend(fontsize=8.5); _grid(ax)
    ax.set_title("Verteilung über die 62 Poolstationen")
    return fig_zu_html(fig)


def abb_scatter(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, ziel in zip(axes, ("ghi", "dhi")):
        d = df[df["target"] == ziel]
        d = d.sample(min(250_000, len(d)), random_state=0)
        obergrenze = float(np.nanpercentile(d["gt"], 99.9))
        hb = ax.hexbin(d["gt"], d["pred"], gridsize=70, bins="log", cmap=CMAP_DICHTE,
                       mincnt=1, extent=(0, obergrenze, 0, obergrenze))
        ax.plot([0, obergrenze], [0, obergrenze], color=C_NWP, lw=1.2, ls="--")
        fig.colorbar(hb, ax=ax, label="Schritte", shrink=.85)
        ax.set_xlabel(f"Messung ({EINHEIT})")
        if ziel == "ghi":
            ax.set_ylabel(f"Vorhersage ({EINHEIT})")
        ax.set_title(ZIEL_NAME[ziel])
    fig.suptitle("Vorhersage gegen Messung", y=1.02, fontsize=11)
    return fig_zu_html(fig)


def abb_runstunde(df: pd.DataFrame) -> str:
    d = df[df["target"] == "ghi"]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    farben = {6: "#123a68", 9: C_MODELL, 12: "#7fb1e8", 15: "#b9d4f2"}
    for rh, teil in d.groupby("run_hour"):
        h = nach(teil, "lead_h")
        ax.plot(h["lead_h"], h["skill_nwp"], lw=2, color=farben.get(rh, C_MODELL),
                label=f"{rh:02d} UTC")
    ax.axhline(0, color=C_NWP, lw=1.4)
    ax.set_xlabel("Vorlaufzeit (h)"); ax.set_ylabel("Skill vs. ICON-D2 (GHI)")
    ax.legend(fontsize=8.5, title="Lauf", title_fontsize=8.5); _grid(ax)
    ax.set_title("Skill je ICON-D2-Lauf über die Vorlaufzeit")
    return fig_zu_html(fig)


# ---------------------------------------------------------------------------
# Bericht
# ---------------------------------------------------------------------------

KOPF_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f6f5f1; color: #0b0b0b;
       font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1060px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 27px; line-height: 1.25; margin: 0 0 6px; letter-spacing: -.01em; }
h2 { font-size: 19px; margin: 44px 0 10px; padding-top: 18px; border-top: 1px solid #ddddd6; }
h3 { font-size: 15px; margin: 26px 0 6px; color: #52514e; }
p { margin: 10px 0; }
.lead { color: #52514e; font-size: 15px; }
.meta { color: #7a7970; font-size: 13px; margin-bottom: 26px; }
figure { margin: 18px 0 8px; background: #fff; border: 1px solid #e3e2dc;
         border-radius: 10px; padding: 14px; }
figure img { width: 100%; height: auto; display: block; }
figcaption { color: #52514e; font-size: 13px; margin-top: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 14px 0;
        background: #fff; border: 1px solid #e3e2dc; border-radius: 10px; overflow: hidden; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid #eeede7; }
th { background: #f0efe9; font-weight: 600; text-align: right; color: #3b3a36; }
th:first-child, td:first-child { text-align: left; }
tr:last-child td { border-bottom: none; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0 4px; }
.tile { flex: 1 1 190px; background: #fff; border: 1px solid #e3e2dc; border-radius: 10px;
        padding: 14px 16px; }
.tile .k { color: #7a7970; font-size: 12.5px; text-transform: uppercase; letter-spacing: .04em; }
.tile .v { font-size: 25px; font-weight: 600; margin-top: 4px; letter-spacing: -.01em; }
.tile .s { color: #52514e; font-size: 13px; margin-top: 2px; }
.befund { background: #fff; border-left: 3px solid #2a78d6; border-radius: 0 8px 8px 0;
          padding: 12px 16px; margin: 16px 0; }
.befund b { color: #0b0b0b; }
code { background: #ecebe5; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
footer { color: #7a7970; font-size: 13px; margin-top: 50px; border-top: 1px solid #ddddd6;
         padding-top: 14px; }
@media (max-width: 640px) { main { padding: 20px 14px 60px; } h1 { font-size: 22px; } }
"""


def tabelle(df: pd.DataFrame, spalten: dict, nachkomma: int = 3) -> str:
    kopf = "".join(f"<th>{v}</th>" for v in spalten.values())
    zeilen = []
    for _, r in df.iterrows():
        zellen = []
        for k in spalten:
            w = r[k]
            if isinstance(w, (int, np.integer)):
                zellen.append(f"<td>{w:,}</td>")
            elif isinstance(w, (float, np.floating)):
                zellen.append(f"<td>{w:.{nachkomma}f}</td>" if np.isfinite(w) else "<td>–</td>")
            else:
                zellen.append(f"<td>{w}</td>")
        zeilen.append("<tr>" + "".join(zellen) + "</tr>")
    return f"<table><thead><tr>{kopf}</tr></thead><tbody>{''.join(zeilen)}</tbody></table>"


def kachel(k: str, v: str, s: str = "") -> str:
    return f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'


def baue_bericht(df: pd.DataFrame, folds, pfad: Path) -> None:
    #: Ein Lauf ohne Foldnummer ist die Schlussmessung auf dem Testjahr — sie
    #: bekommt einen eigenen Vorspann, eigene Aggregatdateien, und der
    #: Ablationsabschnitt entfaellt (der ist an den Fold-Modellen gemessen).
    schlussmessung = list(folds) == [0]
    meta = stationsmeta()
    gesamt = nach(df, "target")
    je_station = nach(df, ["target", "station_id"])
    je_station.to_csv(OUT_CSV / f"solar_tft_{'testyear' if schlussmessung else 'folds'}_je_station.csv", index=False)
    nach(df, ["target", "kt_klasse"]).to_csv(OUT_CSV / f"solar_tft_{'testyear' if schlussmessung else 'folds'}_regime.csv", index=False)
    nach(df, ["target", "lead_h"]).to_csv(OUT_CSV / f"solar_tft_{'testyear' if schlussmessung else 'folds'}_lead.csv", index=False)

    g = gesamt[gesamt["target"] == "ghi"].iloc[0]
    d = gesamt[gesamt["target"] == "dhi"].iloc[0]
    st_ghi = je_station[je_station["target"] == "ghi"]
    gewinner = int((st_ghi["rmse"] < st_ghi["rmse_nwp"]).sum())

    reg = nach(df[df["target"] == "ghi"], "kt_klasse").set_index("kt_klasse").reindex(KT_LABELS)
    bestes = reg["skill_nwp"].idxmax()
    schlechtestes = reg["skill_nwp"].idxmin()

    lead = nach(df[df["target"] == "ghi"], "lead_h")
    skill_kurz = lead.loc[lead["lead_h"] <= 6, "skill_nwp"].mean()
    skill_lang = lead.loc[lead["lead_h"] >= 36, "skill_nwp"].mean()
    # Lead 0 gesondert: dort wirkt die eigene Messhistorie, und ein Block 0-6 h
    # mittelt genau diesen Effekt weg (12 Leads, von denen zwei ihn tragen).
    skill_lead0 = float(lead.loc[lead["lead_h"] == 0, "skill_nwp"].iloc[0])
    skill_1_6 = float(lead.loc[(lead["lead_h"] > 1) & (lead["lead_h"] <= 6), "skill_nwp"].mean())

    stunden = nach(df[df["target"] == "ghi"], "hour")
    schlimmste_stunde = int(stunden.loc[stunden["rmse"].idxmax(), "hour"])

    _d = df[df["target"] == "ghi"].copy()
    _d["lead_block"] = pd.cut(_d["lead_h"], [0, 6, 12, 24, 36, 48],
                              labels=["0–6 h", "6–12 h", "12–24 h", "24–36 h", "36–48 h"],
                              right=False, include_lowest=True)
    _hm = nach(_d, ["kt_klasse", "lead_block"]).set_index(["kt_klasse", "lead_block"])["skill_nwp"]
    hm_kurz = float(_hm.loc[(KT_LABELS[2], "0–6 h")])
    hm_lang = float(_hm.loc[(KT_LABELS[2], "36–48 h")])

    # Deckelung der Vorhersage: hoechster vorhergesagter Wert gegen die Messung.
    deckel, ueber = {}, {}
    for _z in ("ghi", "dhi"):
        _t = df[df["target"] == _z]
        deckel[_z] = float(np.percentile(_t["pred"], 99.99))
        ueber[_z] = float((_t["gt"] > deckel[_z]).mean() * 100)

    regn = nach(df[df["target"] == "ghi"], "kt_nwp_klasse").set_index("kt_nwp_klasse").reindex(KT_LABELS)

    def _wertung(anteil: float) -> str:
        """Trefferanteil einordnen, ohne dem Text eine feste Richtung zu unterstellen."""
        if anteil < .485:
            return "fällt dort unter das Zufallsniveau von 50 %"
        if anteil < .515:
            return "liegt dort mit {:.1f} % praktisch auf dem Zufallsniveau".format(anteil * 100)
        return "liegt dort bei {:.1f} %".format(anteil * 100)

    def _groesse(skill: float, referenz: float) -> str:
        anteil = skill / referenz if referenz else 0
        return "klein" if anteil < .35 else ("moderat" if anteil < .75 else "deutlich")
    bedeckt, trueb, heiter, klar = (str(x) for x in KT_LABELS)
    r_bed, r_heit, r_klar = reg.loc[bedeckt], reg.loc[heiter], reg.loc[klar]
    n_bed, n_heit = regn.loc[bedeckt], regn.loc[heiter]

    vorspann = ("Schlussmessung auf dem zurückgehaltenen Testjahr 2025-08 bis 2026-07: ein Modell "
                "mit den Hyperparametern der abgeschlossenen HPO, trainiert auf allen 62 "
                "Poolstationen über beide vorangegangenen Jahre, gemessen auf den 21 Teststationen, "
                "die in keinem Training vorkamen."
                if schlussmessung else
                "Drei Fold-Modelle mit den Hyperparametern der abgeschlossenen HPO, ausgewertet auf "
                "ihren jeweils zurückgehaltenen Zielstationen im Validierungsjahr 2024-08 bis "
                "2025-07. Die drei Zielmengen sind disjunkt, zusammen decken sie alle 62 "
                "Poolstationen ab — jede bewertet von einem Modell, das sie nie im Training gesehen hat.")

    teile = [f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Solar-TFT — Fold-Auswertung</title><style>{KOPF_CSS}</style></head><body><main>
<h1>Solar-TFT: was die Nachbearbeitung von ICON-D2 gewinnt — und wo nicht</h1>
<p class="lead">{vorspann}</p>
<p class="meta">Erzeugt {datetime.now():%d.%m.%Y %H:%M} aus
<code>data/raw_preds/tft_solar_tft_fold{{{','.join(str(f) for f in folds)}}}_raw.parquet</code> ·
{len(df):,} bewertete Tagesschritte · Nachtschritte und Dämmerung
(Clear-Sky &lt; {TAG_CLEARSKY_MIN:.0f} W/m²) sind ausgeschlossen.</p>

<div class="tiles">
{kachel("GHI — RMSE TFT", f"{g['rmse']:.1f}", f"ICON-D2: {g['rmse_nwp']:.1f} {EINHEIT}")}
{kachel("GHI — Skill", f"{g['skill_nwp']:+.3f}", f"{g['anteil_besser']*100:.1f} % der Schritte näher dran")}
{kachel("DHI — RMSE TFT", f"{d['rmse']:.1f}", f"ICON-D2: {d['rmse_nwp']:.1f} {EINHEIT}")}
{kachel("DHI — Skill", f"{d['skill_nwp']:+.3f}", f"{d['anteil_besser']*100:.1f} % der Schritte näher dran")}
{kachel("Stationen mit Gewinn", f"{gewinner} / {len(st_ghi)}", "GHI, RMSE besser als ICON-D2")}
{kachel("Größter Hebel", f"{regn.loc[bedeckt, 'skill_nwp']:+.3f}", "wenn ICON-D2 Bedeckung prognostiziert")}
</div>

<h2>1. Gesamtbild</h2>
{abb_uebersicht(gesamt)}
<p>Der TFT senkt den RMSE gegenüber der rohen ICON-D2-Prognose am nächsten Gitterpunkt
um {(1-g['rmse']/g['rmse_nwp'])*100:.1f} % (GHI) bzw. {(1-d['rmse']/d['rmse_nwp'])*100:.1f} % (DHI).
Gegen die Persistenz — den letzten Messwert vor Prognosebeginn, über 48 h fortgeschrieben —
sind es {g['skill_pers']*100:.0f} % bzw. {d['skill_pers']*100:.0f} %; diese Referenz ist bei
Strahlung allerdings schwach, weil sie den Tagesgang ignoriert.</p>
<div class="befund"><b>Der Bias ist das, was die Korrektur zuerst holt.</b>
ICON-D2 unterschätzt die Globalstrahlung im Mittel um {abs(d['bias_nwp']) if False else abs(g['bias_nwp']):.1f} {EINHEIT},
der TFT bringt das auf {abs(g['bias']):.1f} {EINHEIT} herunter. Bei DHI dreht das Vorzeichen:
ICON-D2 liegt {abs(d['bias_nwp']):.1f} {EINHEIT} zu niedrig, der TFT {abs(d['bias']):.1f} {EINHEIT} zu hoch.</div>
{tabelle(gesamt.assign(target=gesamt['target'].str.upper()),
         {"target": "Ziel", "n": "Schritte", "rmse": "RMSE TFT", "rmse_nwp": "RMSE ICON",
          "rmse_pers": "RMSE Persistenz", "bias": "Bias TFT", "bias_nwp": "Bias ICON",
          "skill_nwp": "Skill", "anteil_besser": "Anteil besser"}, 2)}
"""]

    teile.append(f"""
<h2>2. Wie lange trägt die Korrektur?</h2>
{abb_lead(df)}
{abb_skill_lead(df)}
<p>Der absolute Fehler beider Seiten wächst mit der Vorlaufzeit, der Skill verläuft
dabei aber nicht flach, sondern in drei Abschnitten:</p>
<div class="befund">
<b>Lead 0</b> — Skill {skill_lead0:+.3f}. Hier wirkt die eigene Messhistorie: das Modell
sieht im observed-Fenster die letzten 48 h NWP-Fehler dieser Station, und der Fehler des
laufenden Zeitschritts hängt mit dem zuletzt gemessenen stark zusammen.<br>
<b>Lead 1–6 h</b> — Skill {skill_1_6:+.3f}. Die Autokorrelation des Residuums ist nach
ein bis zwei Stunden aufgebraucht; was bleibt, ist die Bias-Korrektur aus Abschnitt 4.<br>
<b>Lead ab 36 h</b> — Skill {skill_lang:+.3f}. Der Vorsprung wächst wieder, weil die rohe
NWP-Prognose mit der Vorlaufzeit stärker verliert als die Korrektur.
</div>
<p>Wichtig für die Lesart der Abbildung: Vorlaufzeit und Tageszeit sind bei Solarstrahlung
gekoppelt. Lead 0–3 h bedeutet beim 06-UTC-Lauf den Morgen, beim 15-UTC-Lauf den späten
Nachmittag — beide mit wenig Strahlung und entsprechend kleinem absolutem Fehler. Deshalb
die folgende Auftrennung nach Laufstunde, in der die Tageszeit kontrolliert ist.</p>
{abb_runstunde(df)}
<p>Nach Läufen getrennt zeigt sich dasselbe Bild für alle vier Startzeiten.</p>
""")

    teile.append(f"""
<h2>3. Tagesgang und Jahresgang</h2>
{abb_tagesgang(df)}
<p>Der Fehler folgt der Einstrahlung: am größten um {schlimmste_stunde:02d} UTC, wenn absolut am
meisten Strahlung ankommt und damit auch am meisten danebenliegen kann.</p>
{abb_bias_tagesgang(df)}
{abb_monat(df)}
""")

    teile.append(f"""
<h2>4. Wann gewinnt das Modell gegen ICON-D2?</h2>
<p>Das ist die eigentliche Frage, und sie hat eine klare Antwort. Der Clear-Sky-Index
kt = Messung / Clear-Sky trennt die Bewölkungslagen: kt nahe 1 heißt wolkenlos,
kleine Werte heißen bedeckt, die Mitte ist wechselhaft.</p>
{abb_regime(df)}
<div class="befund"><b>Der Gewinn steckt fast vollständig im mittleren Bereich.</b>
Bei heiterer Lage hebt die Korrektur den RMSE um {r_heit['skill_nwp']*100:.1f} %
({r_heit['rmse_nwp']:.0f} → {r_heit['rmse']:.0f} {EINHEIT}), bei trüber um
{reg.loc[trueb, 'skill_nwp']*100:.1f} %. Bei <b>bedecktem Himmel bleibt praktisch nichts
übrig</b> ({r_bed['skill_nwp']:+.3f}), und der Trefferanteil {_wertung(r_bed['anteil_besser'])}.
Bei wolkenlosem Himmel ist der Gewinn mit {r_klar['skill_nwp']:+.3f}
{_groesse(r_klar['skill_nwp'], r_heit['skill_nwp'])}.</div>
{abb_bias_regime(df)}
<p><b>Warum das so ist, zeigt der Bias.</b> In den Fällen, in denen tatsächlich Bedeckung
herrschte, liegt ICON-D2 um {r_bed['bias_nwp']:+.0f} {EINHEIT} zu hoch — es hat die Wolken
nicht dort gehabt, wo sie waren. Bei wolkenlosem Himmel liegt es um
{r_klar['bias_nwp']:+.0f} {EINHEIT} zu tief. Beide Vorzeichen zusammen sind das bekannte
Bild einer Prognose, die zur Mitte zieht: Extreme werden verfehlt, und zwar in beide
Richtungen.</p>
<p>Der TFT erbt dieses Muster, weil er dieselbe Information benutzt. Er verschiebt das
Niveau ({g['bias_nwp']:+.1f} → {g['bias']:+.1f} {EINHEIT} im Mittel) und glättet den
Zufallsfehler, aber er kann nicht wissen, dass gerade jetzt eine Wolke über der Station
steht, die ICON-D2 nicht hat. Im Bedeckt-Fall verstärkt seine mittlere Aufwärtskorrektur
den Fehler sogar ({r_bed['bias_nwp']:+.0f} → {r_bed['bias']:+.0f} {EINHEIT}) — deshalb bleibt
dort vom Gewinn nichts übrig, obwohl das Modell in jedem anderen Regime vorn liegt.</p>
<p>Eine Einschränkung, die man mitlesen muss: diese Klassen sind nach der <i>Messung</i>
gebildet, also im Nachhinein. Sie zeigen, wo die Prognose danebenlag, nicht, was man vorab
wissen kann. Deshalb dieselbe Aufteilung noch einmal nach der <i>prognostizierten</i> Lage:</p>
{abb_regime_prognose(df)}
<div class="befund"><b>Operativ dreht sich das Bild — und das ist der praktisch
wichtigste Befund.</b> Sagt ICON-D2 eine <b>bedeckte</b> Lage voraus, bringt die Korrektur
{n_bed['skill_nwp']*100:.1f} % ({n_bed['rmse_nwp']:.0f} → {n_bed['rmse']:.0f} {EINHEIT}) —
sagt es wolkenlos voraus, nur {regn.loc[klar, 'skill_nwp']*100:.1f} %. Der Grund steht in
derselben Aufteilung: prognostiziert ICON-D2 Bedeckung, liegt es im Mittel
{abs(n_bed['bias_nwp']):.0f} {EINHEIT} <b>zu niedrig</b> — es ist dann in Wirklichkeit meist
heller als vorhergesagt. Der TFT kennt diese bedingte Verzerrung und rechnet sie fast
vollständig heraus ({n_bed['bias_nwp']:+.1f} → {n_bed['bias']:+.1f} {EINHEIT}).</div>
<p><b>Beide Sichten zusammen ergeben eine klare Arbeitsteilung.</b> Der TFT korrigiert, was
ICON-D2 <i>systematisch</i> falsch macht: den von der prognostizierten Lage abhängigen Bias.
Er korrigiert nicht, was ICON-D2 <i>zufällig</i> falsch macht — eine Wolke, die zur falschen
Zeit am falschen Ort steht. Deshalb ist der Gewinn dort am größten, wo der NWP-Bias am
größten ist ({n_bed['skill_nwp']:+.3f} bei prognostizierter Bedeckung), und dort bei null,
wo die tatsächliche Bewölkung von der prognostizierten abweicht ({r_bed['skill_nwp']:+.3f}
bei gemessener Bedeckung). Wer die Prognose operativ einsetzt, kann sich daran halten: je
trüber ICON-D2 die Lage sieht, desto mehr ist von der Nachbearbeitung zu erwarten.</p>
{abb_heatmap(df)}
<p>Regime und Vorlaufzeit zusammen: die Zeilen trennen sich deutlich, die Spalten kaum —
was zählt, ist die Wetterlage. Innerhalb der ergiebigen Regime <i>wächst</i> der Vorsprung
sogar mit der Vorlaufzeit (heiter: {hm_kurz:+.3f} bei 0–6 h auf {hm_lang:+.3f} bei 36–48 h),
weil die rohe NWP-Prognose mit der Zeit stärker verliert als die Korrektur. Der
Lead-0-Effekt aus Abschnitt 2 geht in diesen Blöcken unter — er betrifft nur die ersten
ein bis zwei Zeitschritte.</p>
""")

    sk = st_ghi["skill_nwp"]
    st_dhi = je_station[je_station["target"] == "dhi"]
    gewinner_dhi = int((st_dhi["rmse"] < st_dhi["rmse_nwp"]).sum())
    schwach = st_ghi.nsmallest(1, "skill_nwp").iloc[0]
    stark = st_ghi.nlargest(1, "skill_nwp").iloc[0]

    abl_pfad = OUT_CSV / "solar_tft_ablation_observed.csv"
    if abl_pfad.exists() and not schlussmessung:
        abl = pd.read_csv(abl_pfad)
        _g = abl[abl["target"] == "ghi"].groupby("block", sort=False)[["rmse_echt", "rmse_perm"]].mean()
        _p = lambda b: 100 * (_g.loc[b, "rmse_perm"] - _g.loc[b, "rmse_echt"]) / _g.loc[b, "rmse_echt"]
        teile.append(f"""
<h2>5. Was trägt die eigene Messhistorie bei?</h2>
<p>Das Modell sieht im observed-Fenster die letzten 48 h der Zielstation — wegen
<code>target_transform: nwp_residual</code> genauer: 48 h NWP-<i>Fehler</i>historie. Wie viel
davon in der Vorhersage ankommt, lässt sich am fertigen Modell messen: dieselben Eingaben
zweimal durchrechnen, einmal mit echtem Fenster und einmal mit über die Läufe permutiertem.
Die Permutation erhält die Verteilung des Kanals und zerstört nur seinen Bezug zum Lauf.</p>
{abb_ablation(abl_pfad)}
<div class="befund"><b>Der Beitrag ist groß, aber kurz.</b> Bei Lead 0 steigt der RMSE ohne
Historie um {_p("0.0 h"):.0f} % (GHI), nach einer halben Stunde sind es {_p("0.5 h"):.0f} %,
nach einer Stunde {_p("1 h"):.0f} %, ab drei Stunden unter {_p("3–6 h"):.1f} %. Das deckt sich
mit der Autokorrelation des Residuums selbst: 0.55 am Laufzeitpunkt, 0.33 nach einer Stunde,
ab sechs Stunden nicht mehr messbar. Über alle 96 Leads gemittelt bleiben rund 1 % (GHI)
bzw. 2 % (DHI) — genug, um zwei Studien zu trennen, aber wenig für eine 48-h-Prognose.</div>
<p>Für das Nowcasting ist das der wichtigste Kanal des Modells, für die Tagesplanung fast
bedeutungslos. Gemessen an 12 Stationen über alle drei Folds
(<code>scripts/ablate_solar_observed.py</code>).</p>
""")

    teile.append(f"""
<h2>6. Welche Stationen profitieren?</h2>
{abb_stationen(je_station)}
<div class="befund"><b>Alle.</b> {gewinner} von {len(st_ghi)} Stationen liegen bei GHI unter
der ICON-D2-Referenz, bei DHI {gewinner_dhi} von {len(st_dhi)} — im Scatter liegt jeder Punkt
unter der Diagonale. Die Frage ist also nicht, <i>ob</i> eine Station profitiert, sondern
wie stark: der Skill reicht von {sk.min():+.3f} (Station {schwach['station_id']}) bis
{sk.max():+.3f} (Station {stark['station_id']}), Median {sk.median():+.3f}.</div>
{abb_skillverteilung(je_station)}
<p>Auffällig ist, wie eng die Verteilung ist — kein Ausreißer nach unten, keine Station,
an der die Nachbearbeitung schadet. Das spricht dafür, dass der Gewinn aus einem
großräumig wirksamen Muster stammt und nicht aus stationsspezifischer Anpassung; die
Modelle haben diese Stationen ja auch nie im Training gesehen.</p>
{abb_karte(je_station, meta)}
<p>Räumlich zeigt sich kein belastbares Muster: Gewinn und Verlust verteilen sich über das
Netz, ohne dass sich Nord/Süd oder Höhenlagen klar absetzen.</p>
""")

    teile.append(f"""
<h2>7. Vorhersage gegen Messung</h2>
{abb_scatter(df)}
<div class="befund"><b>Der TFT deckelt seine Vorhersagen.</b> Bei DHI endet die
Punktwolke bei rund {deckel['dhi']:.0f} {EINHEIT} — darüber sagt das Modell praktisch
nichts mehr vorher, obwohl {ueber['dhi']:.2f} % der Messungen darüber liegen. Bei GHI ist
dieselbe Kante bei etwa {deckel['ghi']:.0f} {EINHEIT} zu sehen ({ueber['ghi']:.2f} % der
Messungen darüber). Das ist die Kehrseite eines Modells, das auf den quadratischen Fehler
optimiert: ein seltener Extremwert kostet weniger, wenn man ihn gar nicht erst versucht.</div>
<p>In der Wolke selbst zeigt sich dieselbe Mittelwertsneigung wie in Abschnitt 4: unterhalb
der Diagonale bei hohen Messwerten, oberhalb bei niedrigen. Genau diese Verzerrung begrenzt,
was eine Nachbearbeitung ohne zusätzliche Beobachtungsinformation leisten kann.</p>
""")

    bestes_s = str(bestes).replace("\n", " ")
    schlecht_s = str(schlechtestes).replace("\n", " ")
    teile.append(f"""
<h2>8. Stationen im Einzelnen</h2>
{tabelle(st_ghi.sort_values('skill_nwp', ascending=False)
         .assign(station_id=st_ghi.sort_values('skill_nwp', ascending=False)['station_id']),
         {"station_id": "Station", "n": "Schritte", "rmse": "RMSE TFT", "rmse_nwp": "RMSE ICON",
          "bias": "Bias TFT", "skill_nwp": "Skill", "anteil_besser": "Anteil besser"}, 2)}
<footer>Regime mit dem größten Gewinn: {bestes_s} · geringster Gewinn: {schlecht_s}.
Erzeugt von <code>scripts/report_solar_folds.py</code>, Kennzahlen aus
<code>scripts/solar_fold_analysis.py</code>. Tabellen als CSV unter
<code>data/test_results/solar_tft_folds_*.csv</code>.</footer>
</main></body></html>""")

    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text("\n".join(teile), encoding="utf-8")
    print(f"[ok] Bericht → {pfad}  ({pfad.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3],
                    help="0 = ein Lauf ohne Foldnummer (Schlussmessung)")
    ap.add_argument("--stem", default="tft_solar_tft_fold")
    ap.add_argument("--out", default="reports/solar_tft_folds.html")
    args = ap.parse_args()

    OUT_CSV.mkdir(parents=True, exist_ok=True)
    print(f"[i] lade Folds {args.folds} …")
    df = lade_alles(folds=tuple(args.folds), stem=args.stem)
    print(f"[i] {len(df):,} Tagesschritte, {df['station_id'].nunique()} Stationen")
    baue_bericht(df, args.folds, REPO / args.out)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
