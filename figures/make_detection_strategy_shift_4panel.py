import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTBASE = OUTDIR / "detection_rewetting_strategy_shift_4panel"

CONCEPT = Path(os.environ.get("WATER_USE_CONCEPT_IMAGE", "concept_diagram.png"))
REWET = ROOT / "rewet_q_results_time_pet.csv"
SUMMARY = OUTDIR / "rewet_window_ecosystem_dq_summary.csv"
PROCESS = {
    "forest": ROOT / "rewet_forest_process" / "forest_rewet_merged_firstdrought_state.csv",
    "grassland": ROOT / "rewet_grassland_process" / "grassland_rewet_merged_firstdrought_state.csv",
    "savanna": ROOT / "rewet_savanna_process" / "savanna_rewet_merged_firstdrought_state.csv",
}

ECOS = ["forest", "grassland", "savanna"]
WINDOWS = ["7d", "14d", "21d"]
QCOL = {"7d": "q_7", "14d": "q_14", "21d": "q_21"}
KW_PVALUES = {"7d": 0.886, "14d": 0.007, "21d": 2.63e-7}
LABEL = {"forest": "forest", "grassland": "grassland", "savanna": "savanna"}
PALETTE = {"forest": "#1b7258", "grassland": "#9fbd45", "savanna": "#2aa7a1"}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
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


def panel_label(ax, letter, x=-0.10, y=1.03):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold", color="black")


def boxed_axes_no_grid(ax):
    ax.grid(False)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(top=False, right=False)


def crop_white_margin(img, threshold=0.985, pad=8):
    arr = img[..., :3]
    mask = np.any(arr < threshold, axis=2)
    if not mask.any():
        return img
    yy, xx = np.where(mask)
    y0, y1 = max(0, yy.min() - pad), min(img.shape[0], yy.max() + pad + 1)
    x0, x1 = max(0, xx.min() - pad), min(img.shape[1], xx.max() + pad + 1)
    return img[y0:y1, x0:x1]


def load_dq_sample(max_per_group=1400):
    summary = pd.read_csv(SUMMARY)
    n_lookup = {(r.ecosystem, r.window): int(r.N_detected) for r in summary.itertuples()}
    probs = {k: min(1.0, max_per_group / max(v, 1)) for k, v in n_lookup.items()}

    rng = np.random.default_rng(20260626)
    samples = []
    usecols = ["ecosystem", "q_original", "q_7", "q_14", "q_21"]
    for chunk in pd.read_csv(REWET, usecols=usecols, chunksize=450_000, low_memory=False, on_bad_lines="skip"):
        chunk["ecosystem"] = chunk["ecosystem"].astype(str).str.lower()
        chunk = chunk[chunk["ecosystem"].isin(ECOS)]
        if chunk.empty:
            continue
        for col in usecols[1:]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        for eco in ECOS:
            ce = chunk[chunk["ecosystem"].eq(eco)]
            if ce.empty:
                continue
            for window, qcol in QCOL.items():
                tmp = ce[["q_original", qcol]].dropna()
                if tmp.empty:
                    continue
                p = probs.get((eco, window), 0.002)
                keep = rng.random(len(tmp)) < p
                tmp = tmp.loc[keep].copy()
                if tmp.empty:
                    continue
                tmp["dq"] = tmp[qcol] - tmp["q_original"]
                tmp["ecosystem"] = eco
                tmp["window"] = window
                samples.append(tmp[["ecosystem", "window", "dq"]])
    sample = pd.concat(samples, ignore_index=True)
    out = []
    for (eco, window), sub in sample.groupby(["ecosystem", "window"]):
        out.append(sub.sample(n=min(max_per_group, len(sub)), random_state=17))
    return pd.concat(out, ignore_index=True)


def load_paradox_data():
    frames = []
    for eco, path in PROCESS.items():
        df = pd.read_csv(path, usecols=["theta_peak_after", "SWC_first", "dq"], low_memory=False)
        for col in ["theta_peak_after", "SWC_first", "dq"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().copy()
        df["ecosystem"] = eco
        df["dtheta"] = df["theta_peak_after"] - df["SWC_first"]
        frames.append(df[["ecosystem", "dtheta", "dq"]])
    return pd.concat(frames, ignore_index=True)


summary = pd.read_csv(SUMMARY)
dq_sample = load_dq_sample()
paradox = load_paradox_data()
significance_rows = []

fig = plt.figure(figsize=(10.8, 7.8))
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[1.0, 1.22],
    height_ratios=[1.0, 1.0],
    left=0.055,
    right=0.985,
    top=0.965,
    bottom=0.085,
    wspace=0.22,
    hspace=0.32,
)

# a. Conceptual schematic
ax_a = fig.add_subplot(gs[0, 0])
img = plt.imread(CONCEPT)
img = crop_white_margin(img, threshold=0.988, pad=4)
ax_a.imshow(img, aspect="auto")
ax_a.axis("off")
panel_label(ax_a, "a", x=-0.04, y=1.02)

# b. Detection frequency
ax_b = fig.add_subplot(gs[0, 1])
x = np.arange(len(WINDOWS))
bar_w = 0.22
offsets = [-bar_w, 0, bar_w]
for i, eco in enumerate(ECOS):
    vals = []
    nums = []
    for window in WINDOWS:
        row = summary[(summary["ecosystem"].eq(eco)) & (summary["window"].eq(window))].iloc[0]
        vals.append(row["occurrence_frequency"] * 100)
        nums.append(int(row["N_detected"]))
    ax_b.bar(x + offsets[i], vals, width=bar_w * 0.92, color=PALETTE[eco], edgecolor="white", linewidth=0.6, label=LABEL[eco])
    for xi, yi, n in zip(x + offsets[i], vals, nums):
        ax_b.text(xi, yi + 0.35, f"n={n:,}", ha="center", va="bottom", fontsize=6.2, rotation=0, color="#555555")
ax_b.set_xticks(x)
ax_b.set_xticklabels(WINDOWS, fontsize=8.5)
ax_b.set_ylabel("Occurrence frequency (%)", fontsize=9.5)
ax_b.set_ylim(0, max(summary["occurrence_frequency"] * 100) + 1.35)
ax_b.tick_params(labelsize=8.2)
boxed_axes_no_grid(ax_b)
ax_b.legend(loc="upper left", ncol=3, fontsize=7.2, frameon=False, handlelength=0.9, columnspacing=0.9)
panel_label(ax_b, "b", x=-0.08, y=1.03)

# c. dq distributions by window and ecosystem
ax_c = fig.add_subplot(gs[1, 0])
positions = []
data_groups = []
colors = []
labels = []
base_gap = 1.28
within = 0.27
for wi, window in enumerate(WINDOWS):
    base = wi * base_gap
    for ei, eco in enumerate(ECOS):
        pos = base + (ei - 1) * within
        positions.append(pos)
        sub = dq_sample[(dq_sample["window"].eq(window)) & (dq_sample["ecosystem"].eq(eco))]["dq"].to_numpy()
        data_groups.append(np.clip(sub, -3, 3))
        colors.append(PALETTE[eco])
        labels.append(eco[0].upper())

bp = ax_c.boxplot(
    data_groups,
    positions=positions,
    widths=0.22,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", lw=0.8),
    whiskerprops=dict(color="#777777", lw=0.6),
    capprops=dict(color="#777777", lw=0.6),
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
    patch.set_alpha(0.22)
    patch.set_linewidth(0.8)

rng = np.random.default_rng(5)
for pos, vals, color in zip(positions, data_groups, colors):
    if len(vals) == 0:
        continue
    draw = vals if len(vals) < 220 else rng.choice(vals, size=220, replace=False)
    jitter = rng.normal(0, 0.035, len(draw))
    ax_c.scatter(np.full(len(draw), pos) + jitter, draw, s=4.5, color=color, alpha=0.22, linewidths=0, rasterized=True)
ax_c.axhline(0, color="black", ls="--", lw=0.8, alpha=0.45)
ax_c.set_ylim(-3.45, 3.65)
ax_c.set_ylabel("Δq after rewetting", fontsize=8)
ax_c.set_xticks(positions)
ax_c.set_xticklabels(labels, fontsize=7.8, fontweight="bold")
for tick, color in zip(ax_c.get_xticklabels(), colors):
    tick.set_color(color)
ax_c.tick_params(labelsize=8.2)
boxed_axes_no_grid(ax_c)
for wi, window in enumerate(WINDOWS):
    ax_c.text(
        wi * base_gap,
        -0.125,
        window,
        transform=ax_c.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.8,
        color="black",
        clip_on=False,
    )
    kw_p = KW_PVALUES[window]
    significance_rows.append({
        "window": window,
        "test": "Kruskal-Wallis",
        "comparison": "forest vs grassland vs savanna",
        "p_value": kw_p,
    })
    if window == "7d":
        kw_label = "K-W p = 0.886\nn.s."
    elif window == "14d":
        kw_label = "K-W p = 0.007\n*"
    else:
        kw_label = "K-W p = 2.63 × 10$^{-7}$\n***"
    ax_c.text(
        wi * base_gap,
        3.16,
        kw_label,
        ha="center",
        va="bottom",
        fontsize=6.2,
        fontweight="bold" if window == "21d" else "normal",
        color="#315f3b" if window == "21d" else "#555555",
        linespacing=1.15,
    )
for sep in [(WINDOWS.index("7d") + 0.5) * base_gap, (WINDOWS.index("14d") + 0.5) * base_gap]:
    pass
panel_label(ax_c, "c", x=-0.10, y=1.03)

# d. 21d moisture-gain paradox
ax_d = fig.add_subplot(gs[1, 1])
plot = paradox.copy()
xlim = (
    min(plot["dtheta"].quantile(0.003), -0.16),
    max(plot["dtheta"].quantile(0.997), 0.22),
)
ylim = (-3.0, 3.25)
ax_d.axvspan(0, xlim[1], ymin=0, ymax=(0 - ylim[0]) / (ylim[1] - ylim[0]), color="#dcefe3", alpha=0.58, zorder=0)
ax_d.axhline(0, color="black", lw=0.8, ls="--", alpha=0.55)
ax_d.axvline(0, color="black", lw=0.8, ls="--", alpha=0.55)
for i, eco in enumerate(ECOS):
    sub = plot[plot["ecosystem"].eq(eco)].copy()
    sub = sub[(sub["dtheta"].between(*xlim)) & (sub["dq"].between(*ylim))]
    draw_n = 3300 if eco != "savanna" else 2600
    draw = sub.sample(n=min(draw_n, len(sub)), random_state=310 + i)
    ax_d.scatter(draw["dtheta"], draw["dq"], s=4.8, color=PALETTE[eco], alpha=0.07 if eco != "savanna" else 0.16, linewidths=0, rasterized=True)
ax_d.set_xlim(*xlim)
ax_d.set_ylim(*ylim)
ax_d.set_xlabel(r"$\theta_{peak} - \theta_{first}$ (m$^3$ m$^{-3}$)", fontsize=9.5)
ax_d.set_ylabel(r"$\Delta q_{21}$", fontsize=9.5)
ax_d.tick_params(labelsize=8.2)
boxed_axes_no_grid(ax_d)
ax_d.text(
    0.965,
    0.055,
    "more moisture\nbut more aggressive",
    transform=ax_d.transAxes,
    ha="right",
    va="bottom",
    fontsize=8.0,
    fontweight="bold",
    color="#315f3b",
    bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.72),
)
legend_handles = [Line2D([0], [0], marker="o", color=PALETTE[e], label=LABEL[e], markersize=4.5, lw=1.0) for e in ECOS]
ax_d.legend(handles=legend_handles, loc="upper left", fontsize=7.2, frameon=False, handlelength=1.4)
panel_label(ax_d, "d", x=-0.08, y=1.03)

fig.savefig(OUTBASE.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUTBASE.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUTBASE.with_suffix(".svg"), bbox_inches="tight")
pd.DataFrame(significance_rows).to_csv(
    OUTDIR / "rewet_window_ecosystem_dq_significance.csv", index=False
)

print(OUTBASE.with_suffix(".png"))
print(OUTBASE.with_suffix(".pdf"))
print(OUTBASE.with_suffix(".svg"))
