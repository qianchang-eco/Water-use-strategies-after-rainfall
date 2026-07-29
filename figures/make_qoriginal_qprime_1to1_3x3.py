import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
DATA = ROOT / "rewet_q_results_time_pet.csv"
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTBASE = OUTDIR / "qoriginal_qprime_1to1_ecosystem_windows_3x3"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "figure.facecolor": "white",
    }
)

ECOSYSTEMS = ["forest", "grassland", "savanna"]
WINDOWS = [("7d", "q_7"), ("14d", "q_14"), ("21d", "q_21")]
USECOLS = ["ecosystem", "q_original", "q_7", "q_14", "q_21"]

PALETTE = {
    "forest": "#126a4d",
    "grassland": "#6eaf3e",
    "savanna": "#1f9e99",
}


def boxed_axes_no_grid(ax):
    ax.grid(False)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(top=False, right=False)


def load_data():
    frames = []
    for chunk in pd.read_csv(DATA, usecols=USECOLS, chunksize=700_000, low_memory=False, on_bad_lines="skip"):
        chunk["ecosystem"] = chunk["ecosystem"].astype(str).str.lower()
        chunk = chunk[chunk["ecosystem"].isin(ECOSYSTEMS)]
        if chunk.empty:
            continue
        for col in USECOLS[1:]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        frames.append(chunk)
    if not frames:
        raise RuntimeError("No valid rows found.")
    return pd.concat(frames, ignore_index=True)


def deterministic_sample(df, n=18000, seed=20260625):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def density_colors(x, y):
    if len(x) < 20:
        return np.ones_like(x)
    try:
        xy = np.vstack([x, y])
        dens = gaussian_kde(xy)(xy)
        return dens
    except Exception:
        return np.ones_like(x)


df = load_data()
summary_rows = []

fig, axes = plt.subplots(3, 3, figsize=(8.3, 8.35), sharex=True, sharey=True)
plt.subplots_adjust(left=0.09, right=0.985, top=0.935, bottom=0.075, wspace=0.12, hspace=0.19)

letters = list("abcdefghi")

for r, eco in enumerate(ECOSYSTEMS):
    for c, (win_label, qcol) in enumerate(WINDOWS):
        ax = axes[r, c]
        sub = df.loc[df["ecosystem"].eq(eco), ["q_original", qcol]].dropna().copy()
        sub = sub[(sub["q_original"].between(0, 3)) & (sub[qcol].between(0, 3))]
        if sub.empty:
            continue

        n_total = len(sub)
        frac_below = float((sub[qcol] < sub["q_original"]).mean())
        median_dq = float((sub[qcol] - sub["q_original"]).median())
        mean_dq = float((sub[qcol] - sub["q_original"]).mean())
        summary_rows.append(
            {
                "ecosystem": eco,
                "window": win_label,
                "n": n_total,
                "frac_qprime_below_qoriginal": frac_below,
                "median_dq": median_dq,
                "mean_dq": mean_dq,
            }
        )

        samp = deterministic_sample(sub, n=16000, seed=1000 + r * 10 + c)
        x = samp["q_original"].to_numpy()
        y = samp[qcol].to_numpy()
        dens = density_colors(x, y)
        order = np.argsort(dens)
        x, y, dens = x[order], y[order], dens[order]

        base_color = PALETTE[eco]
        ax.scatter(
            x,
            y,
            s=4.5,
            c=base_color,
            alpha=0.18,
            linewidths=0,
            rasterized=True,
        )
        ax.plot([0, 3], [0, 3], color="black", lw=1.0, ls="--", alpha=0.75)
        ax.fill_between([0, 3], [0, 3], [0, 0], color="#7fbf7b", alpha=0.09, linewidth=0)

        ax.text(
            0.04,
            0.94,
            letters[r * 3 + c],
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="top",
            color="black",
        )
        ax.text(
            0.96,
            0.06,
            f"{frac_below * 100:.1f}% below\nmedian $\\Delta q$={median_dq:.2f}",
            transform=ax.transAxes,
            fontsize=7.1,
            ha="right",
            va="bottom",
            color="black",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.70),
        )

        if r == 0:
            ax.set_title(win_label, fontsize=10, fontweight="bold", pad=6)
        if c == 0:
            ax.set_ylabel(f"{eco}\n" + r"$q_{\mathrm{post}}$", fontsize=9.5, fontweight="bold", color="black")
            ax.set_ylabel(f"{eco}\nq′", fontsize=9.5, fontweight="bold", color="black")
        if c == 0:
            ax.set_ylabel(f"{eco}\n" + r"$q_{\mathrm{post}}$", fontsize=9.5, fontweight="bold", color="black")
        if r == 2:
            ax.set_xlabel(r"$q_{\mathrm{original}}$", fontsize=9.5, fontweight="bold", color="black")

        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2, 3])
        ax.tick_params(labelsize=8, colors="black", length=3.2, width=0.7)
        boxed_axes_no_grid(ax)

fig.savefig(OUTBASE.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUTBASE.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUTBASE.with_suffix(".svg"), bbox_inches="tight")

pd.DataFrame(summary_rows).to_csv(OUTDIR / "qoriginal_qprime_1to1_summary.csv", index=False)

print(OUTBASE.with_suffix(".png"))
print(OUTBASE.with_suffix(".pdf"))
print(OUTBASE.with_suffix(".svg"))
print(OUTDIR / "qoriginal_qprime_1to1_summary.csv")
