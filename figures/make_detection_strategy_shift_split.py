"""Build the main rewetting figure and Extended Data Fig. 2.

This retains the source datasets and K-W results used by
make_detection_strategy_shift_4panel.py and writes the submission filenames.
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
PLOTS = ROOT / "plots"
SOURCE_IMAGE = PLOTS / "QQ截图20260722111755.png"
SUMMARY_FILE = PLOTS / "rewet_window_ecosystem_dq_summary.csv"
EVENT_FILE = ROOT / "rewet_q_results_time_pet.csv"
PROCESS_FILES = {
    "forest": ROOT / "rewet_forest_process" / "forest_rewet_merged_firstdrought_state.csv",
    "grassland": ROOT / "rewet_grassland_process" / "grassland_rewet_merged_firstdrought_state.csv",
    "savanna": ROOT / "rewet_savanna_process" / "savanna_rewet_merged_firstdrought_state.csv",
}
ECOS = ("forest", "grassland", "savanna")
WINDOWS = ("7d", "14d", "21d")
QCOL = {"7d": "q_7", "14d": "q_14", "21d": "q_21"}
KW_P = {"7d": 0.886, "14d": 0.007, "21d": 2.63e-7}
COLOR = {"forest": "#1b7258", "grassland": "#9fbd45", "savanna": "#2aa7a1"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def crop_white(image, threshold=0.988, pad=5):
    rgb = image[..., :3]
    mask = np.any(rgb < threshold, axis=2)
    ys, xs = np.where(mask)
    return image[max(0, ys.min()-pad):min(len(image), ys.max()+pad+1),
                 max(0, xs.min()-pad):min(image.shape[1], xs.max()+pad+1)]


def clean_schematic_y_label(image):
    """Remove the old compound y-axis label from the raster schematic."""
    cleaned = image.copy()
    cleaned[:, :110, :3] = 1.0
    if cleaned.shape[2] == 4:
        cleaned[:, :110, 3] = 1.0
    return cleaned


def panel(ax, letter, fontsize=16):
    ax.text(-0.10, 1.025, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=fontsize, fontweight="bold")


def frame(ax, tick_fontsize=10):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(top=False, right=False, labelsize=tick_fontsize)


def lookup(summary, eco, window):
    return summary.loc[(summary.ecosystem == eco) & (summary.window == window)].iloc[0]


def load_dq_sample(summary, max_per_group=1400):
    """Repeat the fixed-seed sampling in the original four-panel source."""
    n = {(r.ecosystem, r.window): int(r.N_detected) for r in summary.itertuples()}
    probabilities = {k: min(1.0, max_per_group / max(v, 1)) for k, v in n.items()}
    random = np.random.default_rng(20260626)
    samples = []
    cols = ["ecosystem", "q_original", "q_7", "q_14", "q_21"]
    for chunk in pd.read_csv(EVENT_FILE, usecols=cols, chunksize=450_000,
                             low_memory=False, on_bad_lines="skip"):
        chunk["ecosystem"] = chunk["ecosystem"].astype(str).str.lower()
        chunk = chunk[chunk.ecosystem.isin(ECOS)]
        if chunk.empty:
            continue
        for col in cols[1:]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        for eco in ECOS:
            subset = chunk[chunk.ecosystem == eco]
            if subset.empty:
                continue
            for window, qcol in QCOL.items():
                pair = subset[["q_original", qcol]].dropna()
                if pair.empty:
                    continue
                keep = random.random(len(pair)) < probabilities.get((eco, window), 0.002)
                pair = pair.loc[keep]
                if pair.empty:
                    continue
                samples.append(pd.DataFrame({
                    "ecosystem": eco, "window": window,
                    "dq": pair[qcol].to_numpy() - pair.q_original.to_numpy(),
                }))
    sample = pd.concat(samples, ignore_index=True)
    return pd.concat((group.sample(n=min(max_per_group, len(group)), random_state=17)
                      for _, group in sample.groupby(["ecosystem", "window"])),
                     ignore_index=True)


def load_paradox():
    frames = []
    for eco, path in PROCESS_FILES.items():
        data = pd.read_csv(path, usecols=["theta_peak_after", "SWC_first", "dq"],
                           low_memory=False)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna()
        frames.append(pd.DataFrame({
            "ecosystem": eco,
            "dtheta": data.theta_peak_after - data.SWC_first,
            "dq": data.dq,
        }))
    return pd.concat(frames, ignore_index=True)


def plot_bars(ax, summary, windows, big=False):
    x = np.arange(len(windows))
    width = 0.23 if len(windows) > 1 else 0.22
    offsets = np.array([-1, 0, 1]) * width
    values = []
    for i, eco in enumerate(ECOS):
        rows = [lookup(summary, eco, w) for w in windows]
        heights = [r.occurrence_frequency * 100 for r in rows]
        values.extend(heights)
        ax.bar(x + offsets[i], heights, width=width*0.9,
               color=COLOR[eco], label=eco, edgecolor="white", linewidth=0.6)
        for pos, height, row in zip(x + offsets[i], heights, rows):
            ax.text(pos, height + (0.75 if big else 0.33),
                    f"n=\n{int(row.N_detected):,}",
                    ha="center", va="bottom", fontsize=17 if big else 8.5,
                    linespacing=0.95, color="#333333")
    ax.set_xlim(-0.52, len(windows)-0.48)
    ax.set_ylim(0, max(values) + (7.0 if big else 3.3))
    ax.set_ylabel("Occurrence frequency (%)", fontsize=20 if big else 10)
    ax.set_xticks(x, windows, fontsize=19 if big else 10)
    ax.legend(loc="upper left", ncol=3, frameon=False,
              fontsize=17 if big else 8.5, handlelength=0.9,
              columnspacing=0.7, bbox_to_anchor=(0, 1.01))
    frame(ax, tick_fontsize=18 if big else 10)


def draw_boxes(ax, groups, positions, colors, width, dot_limit, seed=5):
    clipped = [np.clip(group, -3, 3) for group in groups]
    box = ax.boxplot(clipped, positions=positions, widths=width,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "black", "linewidth": 1.15},
                     whiskerprops={"color": "#666666", "linewidth": 0.9},
                     capprops={"color": "#666666", "linewidth": 0.9})
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.25)
        patch.set_linewidth(1)
    random = np.random.default_rng(seed)
    for pos, group, color in zip(positions, clipped, colors):
        draw = group if len(group) <= dot_limit else random.choice(group, size=dot_limit, replace=False)
        jitter = random.normal(0, width*0.14, len(draw))
        ax.scatter(np.full(len(draw), pos)+jitter, draw, s=7,
                   color=color, alpha=0.25, linewidths=0, rasterized=True)


def plot_dq_21(ax, dq):
    positions = np.arange(3)
    groups = [dq.loc[(dq.window == "21d") & (dq.ecosystem == eco), "dq"].to_numpy()
              for eco in ECOS]
    draw_boxes(ax, groups, positions, [COLOR[e] for e in ECOS], 0.48, 260)
    ax.axhline(0, color="#555555", ls="--", lw=0.9)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-3.45, 4.0)
    ax.set_xticks(positions, [e.capitalize() for e in ECOS], fontsize=18)
    ax.set_ylabel(r"$\Delta q$ after rewetting", fontsize=20)
    ax.text(0.5, 0.97, r"K–W $p$ = 2.63 $\times$ 10$^{-7}$ (***)", transform=ax.transAxes,
            ha="center", va="top", fontsize=18, color="#315f3b")
    frame(ax, tick_fontsize=18)


def plot_dq_all(ax, dq):
    positions, groups, colors, labels = [], [], [], []
    for wi, window in enumerate(WINDOWS):
        for ei, eco in enumerate(ECOS):
            positions.append(wi*1.33 + (ei-1)*0.31)
            groups.append(dq.loc[(dq.window == window) & (dq.ecosystem == eco), "dq"].to_numpy())
            colors.append(COLOR[eco])
            labels.append(eco[0].upper())
    draw_boxes(ax, groups, positions, colors, 0.245, 220)
    ax.axhline(0, color="#555555", ls="--", lw=0.9)
    ax.set_ylim(-3.45, 4.0)
    ax.set_xlim(-0.55, 3.21)
    ax.set_ylabel(r"$\Delta q$ after rewetting", fontsize=11)
    ax.set_xticks(positions, labels, fontsize=10, fontweight="bold")
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
    for wi, window in enumerate(WINDOWS):
        ax.text(wi*1.33, -0.115, window, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10.5)
        p = KW_P[window]
        p_label = f"{p:.3f}" if p >= 0.001 else r"2.63 $\times$ 10$^{-7}$"
        stars = "" if p >= 0.05 else ("*" if p >= 0.001 else "***")
        annotation = f"K–W $p$ = {p_label}" + (f"\n{stars}" if stars else "")
        ax.text(wi*1.33, 3.86, annotation,
                ha="center", va="top", fontsize=9.2, linespacing=1.0)
    frame(ax)


def plot_paradox(ax, paradox):
    xlim = (min(paradox.dtheta.quantile(0.003), -0.16),
            max(paradox.dtheta.quantile(0.997), 0.22))
    ylim = (-3.0, 3.25)
    ax.axvspan(0, xlim[1], ymin=0,
               ymax=(0-ylim[0])/(ylim[1]-ylim[0]),
               color="#dcefe3", alpha=0.62, zorder=0)
    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.55)
    ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.55)
    for i, eco in enumerate(ECOS):
        subset = paradox[paradox.ecosystem == eco]
        subset = subset[subset.dtheta.between(*xlim) & subset.dq.between(*ylim)]
        n = 3300 if eco != "savanna" else 2600
        draw = subset.sample(n=min(n, len(subset)), random_state=310+i)
        ax.scatter(draw.dtheta, draw.dq, s=6, color=COLOR[eco],
                   alpha=0.075 if eco != "savanna" else 0.17,
                   linewidths=0, rasterized=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\theta_{peak}-\theta_{first}$ (m$^3$ m$^{-3}$)", fontsize=11)
    ax.set_ylabel(r"$\Delta q_{21}$", fontsize=11)
    ax.text(0.97, 0.065, "more moisture\nbut more aggressive",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold", color="#315f3b",
            bbox={"boxstyle":"round,pad=0.2", "facecolor":"white",
                  "edgecolor":"none", "alpha":0.78})
    handles = [Line2D([0], [0], marker="o", linestyle="none",
                      color=COLOR[e], label=e, markersize=7) for e in ECOS]
    ax.legend(handles=handles, loc="upper left", fontsize=9,
              frameon=False, ncol=3, columnspacing=0.8)
    frame(ax)


def save(fig, basename):
    for suffix in ("png", "pdf", "svg"):
        path = PLOTS / f"{basename}.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        print(path)
    plt.close(fig)


def main():
    summary = pd.read_csv(SUMMARY_FILE)
    dq = load_dq_sample(summary)
    paradox = load_paradox()

    fig = plt.figure(figsize=(14.2, 8.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.57, 1],
                            left=0.035, right=0.985, top=0.96, bottom=0.09,
                            hspace=0.43, wspace=0.27)
    ax_a = fig.add_subplot(grid[:, 0])
    schematic = clean_schematic_y_label(plt.imread(SOURCE_IMAGE))
    ax_a.imshow(schematic, aspect="equal")
    ax_a.text(0.038, 0.50, r"Soil moisture, $\theta$",
              transform=ax_a.transAxes, rotation=90,
              ha="center", va="center", fontsize=15,
              fontfamily="serif", color="black")
    ax_a.axis("off")
    panel(ax_a, "a", fontsize=22)
    ax_b = fig.add_subplot(grid[0, 1])
    plot_bars(ax_b, summary, ("21d",), big=True)
    panel(ax_b, "b", fontsize=22)
    ax_c = fig.add_subplot(grid[1, 1])
    plot_dq_21(ax_c, dq)
    panel(ax_c, "c", fontsize=22)
    save(fig, "Fig. 1")

    fig = plt.figure(figsize=(13.2, 10.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.08],
                            left=0.085, right=0.98, top=0.96, bottom=0.085,
                            hspace=0.37, wspace=0.24)
    ax_a = fig.add_subplot(grid[0, 0])
    plot_bars(ax_a, summary, WINDOWS)
    panel(ax_a, "a")
    ax_b = fig.add_subplot(grid[0, 1])
    plot_dq_all(ax_b, dq)
    panel(ax_b, "b")
    ax_c = fig.add_subplot(grid[1, :])
    plot_paradox(ax_c, paradox)
    panel(ax_c, "c")
    save(fig, "Extended Data Fig. 2")


if __name__ == "__main__":
    main()
