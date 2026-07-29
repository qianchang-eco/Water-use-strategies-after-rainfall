import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTBASE = OUTDIR / "theta_peak_minus_first_vs_dq_bivariate_ecosystems"

SOURCES = {
    "forest": ROOT / "rewet_forest_process" / "forest_rewet_merged_firstdrought_state.csv",
    "grassland": ROOT / "rewet_grassland_process" / "grassland_rewet_merged_firstdrought_state.csv",
    "savanna": ROOT / "rewet_savanna_process" / "savanna_rewet_merged_firstdrought_state.csv",
}

PALETTE = {
    "forest": "#0b5b48",
    "grassland": "#87b93d",
    "savanna": "#1aa69a",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "figure.facecolor": "white",
    }
)


def load_ecosystem(name, path):
    usecols = ["theta_peak_after", "SWC_first", "dq"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in usecols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=usecols).copy()
    df["ecosystem"] = name
    df["theta_peak_minus_first"] = df["theta_peak_after"] - df["SWC_first"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["theta_peak_minus_first", "dq"])
    return df[["ecosystem", "theta_peak_minus_first", "dq"]]


def trimmed(df, xcol, ycol):
    out = df.copy()
    xlo, xhi = out[xcol].quantile([0.005, 0.995])
    ylo, yhi = out[ycol].quantile([0.005, 0.995])
    return out[(out[xcol].between(xlo, xhi)) & (out[ycol].between(ylo, yhi))].copy()


def density_contour(ax, x, y, color, xlim, ylim, bins=120):
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    smooth = gaussian_filter(hist.T, sigma=4.0)
    if smooth.max() <= 0:
        return
    xx = (xedges[:-1] + xedges[1:]) / 2
    yy = (yedges[:-1] + yedges[1:]) / 2
    vals = smooth[smooth > 0]
    levels = np.quantile(vals, [0.965])
    levels = np.unique(levels)
    if len(levels):
        ax.contour(xx, yy, smooth, levels=levels, colors=[color], linewidths=[1.25], alpha=0.72)


def sample(df, n, seed):
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


frames = [load_ecosystem(k, v) for k, v in SOURCES.items()]
data = pd.concat(frames, ignore_index=True)

plot_data = trimmed(data, "theta_peak_minus_first", "dq")
xlim = (
    float(plot_data["theta_peak_minus_first"].quantile(0.002)),
    float(plot_data["theta_peak_minus_first"].quantile(0.998)),
)
ylim = (
    float(plot_data["dq"].quantile(0.002)),
    float(plot_data["dq"].quantile(0.998)),
)
xlim = (min(xlim[0], -0.16), max(xlim[1], 0.22))
ylim = (min(ylim[0], -3.0), max(ylim[1], 3.0))

summary = []
for eco, df_eco in data.groupby("ecosystem"):
    valid = df_eco.dropna(subset=["theta_peak_minus_first", "dq"])
    summary.append(
        {
            "ecosystem": eco,
            "n": len(valid),
            "median_theta_peak_minus_first": valid["theta_peak_minus_first"].median(),
            "median_dq21": valid["dq"].median(),
            "frac_more_moisture_more_aggressive": ((valid["theta_peak_minus_first"] > 0) & (valid["dq"] < 0)).mean(),
            "frac_dq_negative": (valid["dq"] < 0).mean(),
            "frac_theta_peak_minus_first_positive": (valid["theta_peak_minus_first"] > 0).mean(),
        }
    )
summary_df = pd.DataFrame(summary).sort_values("ecosystem")
summary_df.to_csv(OUTDIR / "theta_peak_minus_first_vs_dq_bivariate_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(5.4, 4.25))

ax.axvspan(0, xlim[1], ymin=0, ymax=(0 - ylim[0]) / (ylim[1] - ylim[0]), color="#dcefe3", alpha=0.58, zorder=0)
ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.62, zorder=1)
ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.62, zorder=1)

for i, eco in enumerate(["forest", "grassland", "savanna"]):
    sub = plot_data[plot_data["ecosystem"].eq(eco)].copy()
    s = sample(sub, 5200 if eco != "savanna" else 4200, 20260626 + i)
    ax.scatter(
        s["theta_peak_minus_first"],
        s["dq"],
        s=5.0,
        color=PALETTE[eco],
        alpha=0.065 if eco != "savanna" else 0.15,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    density_contour(
        ax,
        sub["theta_peak_minus_first"].to_numpy(),
        sub["dq"].to_numpy(),
        PALETTE[eco],
        xlim=xlim,
        ylim=ylim,
    )

ax.text(
    0.965,
    0.06,
    "more moisture\nbut more aggressive",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.8,
    fontweight="bold",
    color="#315f3b",
    bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="none", alpha=0.72),
)

legend_handles = []
for eco in ["forest", "grassland", "savanna"]:
    row = summary_df[summary_df["ecosystem"].eq(eco)].iloc[0]
    label = f"{eco} ({row['frac_more_moisture_more_aggressive'] * 100:.1f}% paradox)"
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color=PALETTE[eco],
            markerfacecolor=PALETTE[eco],
            markersize=4.8,
            lw=1.1,
            label=label,
        )
    )
ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.985),
    fontsize=6.8,
    frameon=False,
    handlelength=1.8,
)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xlabel(r"$\theta_{peak} - \theta_{first}$ (m$^3$ m$^{-3}$)", fontsize=8.7, fontweight="bold")
ax.set_ylabel(r"$dq_{21}$", fontsize=8.7, fontweight="bold")
ax.tick_params(labelsize=7.4, length=3.3, width=0.75)
ax.grid(True, color="#e7e7e7", lw=0.55, alpha=0.75)

fig.savefig(OUTBASE.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUTBASE.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUTBASE.with_suffix(".svg"), bbox_inches="tight")

print(OUTBASE.with_suffix(".png"))
print(OUTBASE.with_suffix(".pdf"))
print(OUTBASE.with_suffix(".svg"))
print(OUTDIR / "theta_peak_minus_first_vs_dq_bivariate_summary.csv")
