import os
from pathlib import Path
import json
import urllib.request

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
DATA = ROOT / "q_results_with_ecosystem_precip_swc_latlon_pet_ai.csv"
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTBASE = OUTDIR / "global_study_area_sites"
CACHE = OUTDIR / "ne_110m_admin_0_countries.geojson"
NE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_110m_admin_0_countries.geojson"
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.85,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "figure.facecolor": "white",
    }
)


def boxed_axes(ax):
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.85)
    ax.tick_params(top=False, right=False)
    ax.grid(False)


def load_world():
    if not CACHE.exists():
        with urllib.request.urlopen(NE_URL, timeout=30) as response:
            CACHE.write_bytes(response.read())
    return json.loads(CACHE.read_text(encoding="utf-8"))


def iter_rings(geometry):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        for ring in coords:
            yield ring
    elif gtype == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                yield ring


def load_sites():
    df = pd.read_csv(DATA, usecols=["SITE_ID", "lat", "lon"], low_memory=False)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    return df.drop_duplicates(["lat", "lon"]).reset_index(drop=True)


world = load_world()
sites = load_sites()

fig, ax = plt.subplots(figsize=(9.2, 5.25))
fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.135)

for feature in world["features"]:
    geom = feature.get("geometry") or {}
    for ring in iter_rings(geom):
        arr = np.asarray(ring, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        ax.plot(arr[:, 0], arr[:, 1], color="#b6b6b6", lw=0.55, zorder=1)

ax.scatter(
    sites["lon"],
    sites["lat"],
    s=17,
    facecolor="#b9dd82",
    edgecolor="#1f2a22",
    linewidth=0.45,
    alpha=0.92,
    zorder=3,
)

ax.set_xlim(-180, 180)
ax.set_ylim(-85, 85)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Longitude", fontsize=15)
ax.set_ylabel("Latitude", fontsize=15)
ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
ax.set_yticks([-50, 0, 50])
ax.tick_params(labelsize=10, length=4, width=0.85)
boxed_axes(ax)

for ext in [".png", ".pdf", ".svg"]:
    fig.savefig(OUTBASE.with_suffix(ext), dpi=600 if ext == ".png" else None, bbox_inches="tight")

print(OUTBASE.with_suffix(".png"))
print(f"sites={len(sites)}")
