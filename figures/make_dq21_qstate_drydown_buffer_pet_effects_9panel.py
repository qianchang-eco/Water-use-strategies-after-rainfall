import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
OUTDIR = ROOT / "plots"
STATS = OUTDIR / "dq21_qstate_drydown_buffer_pet_linear_effects.csv"
OUTBASE = OUTDIR / "dq21_qstate_drydown_buffer_pet_effects_9panel"

ECOS = ["savanna", "forest", "grassland"]
ECO_LABELS = {"savanna": "savanna", "forest": "forest", "grassland": "grassland"}
VARS = ["Drydown rate", "State buffer", "PET90"]
DISPLAY_VARS = {"Drydown rate": "Drydown rate", "State buffer": "moisture buffering", "PET90": r"$PET_{90}$"}
STATE_ORDER = ["Q1_low", "Q2_midlow", "Q3_midhigh", "Q4_high"]
STATE_LABELS = ["Q1", "Q2", "Q3", "Q4"]
STATE_LEGEND = ["Q1 low", "Q2 midlow", "Q3 midhigh", "Q4 high"]
COLORS = ["#d9ee9b", "#88c76e", "#36aaa5", "#1f734e"]
EDGE = "#1b2530"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "figure.facecolor": "white",
    }
)


def boxed_axes_no_grid(ax):
    ax.grid(False)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(top=False, right=False)


def panel_label(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, fontweight="bold", color="black")


def fmt_slope(v):
    av = abs(v)
    if av >= 10:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


stats = pd.read_csv(STATS)

fig, axes = plt.subplots(3, 3, figsize=(10.4, 7.15), sharex=True, sharey=True)
plt.subplots_adjust(left=0.095, right=0.985, top=0.89, bottom=0.14, wspace=0.16, hspace=0.33)

letters = list("abcdefghi")
for r, eco in enumerate(ECOS):
    for c, var in enumerate(VARS):
        ax = axes[r, c]
        sub = (
            stats[(stats["ecosystem"].eq(eco)) & (stats["variable"].eq(var))]
            .set_index("q_state")
            .reindex(STATE_ORDER)
        )
        vals = sub["std_beta"].to_numpy(float)
        slopes = sub["slope"].to_numpy(float)
        stars = sub["stars"].fillna("n.s.").to_numpy(str)
        x = np.arange(4)
        absmax = np.nanargmax(np.abs(vals))
        for i, (xi, val) in enumerate(zip(x, vals)):
            lw = 1.0
            ec = COLORS[i]
            if i == absmax:
                lw = 1.8
                ec = EDGE
            ax.bar(xi, val, width=0.62, color=COLORS[i], edgecolor=ec, linewidth=lw)
            va = "bottom" if val >= 0 else "top"
            yoff = 0.018 if val >= 0 else -0.018
            ax.text(
                xi,
                val + yoff,
                f"{fmt_slope(slopes[i])} {stars[i]}",
                ha="center",
                va=va,
                fontsize=8.0,
                color="black",
            )
        ax.axhline(0, color="#666666", lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.49, 0.49)
        ax.set_xticks(x)
        ax.set_xticklabels(STATE_LABELS, fontsize=8.2)
        if r == 0:
            ax.set_title(DISPLAY_VARS[var], fontsize=10.2, fontweight="bold", pad=7)
        if c == 0:
            ax.set_ylabel(f"{ECO_LABELS[eco]}\nstandardized beta", fontsize=8.3)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7.5, length=3)
        boxed_axes_no_grid(ax)
        panel_label(ax, letters[r * 3 + c])

handles = [Patch(facecolor=COLORS[i], edgecolor=COLORS[i], label=STATE_LEGEND[i]) for i in range(4)]
handles.append(Patch(facecolor="white", edgecolor=EDGE, linewidth=1.8, label="largest within state"))
legend = fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9.0,
                    bbox_to_anchor=(0.53, 0.965), handlelength=1.0, columnspacing=1.45)
for text in legend.get_texts():
    text.set_color("black")
fig.text(0.045, 0.51, "Standardized effect on Δq$_{21}$", rotation=90,
         va="center", ha="center", fontsize=10.0, color="black")
fig.text(0.50, 0.075, r"$q_{\mathrm{original}}$ quartile within each ecosystem",
         ha="center", va="center", fontsize=9.2, color="black")
fig.text(0.965, 0.075, "Text labels show raw slope and significance",
         ha="right", va="center", fontsize=8.0, color="black")

for ext in [".png", ".pdf", ".svg"]:
    fig.savefig(OUTBASE.with_suffix(ext), dpi=600 if ext == ".png" else None, bbox_inches="tight")

print(OUTBASE.with_suffix(".png"))
