import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTBASE = OUTDIR / "soil_moisture_rewetting_change_ecosystems"

REWET_CSV = ROOT / "rewet_q_results_time_pet.csv"
FIRST_CSV = ROOT / "q_results_with_ecosystem_precip_swc_latlon_pet_ai.csv"
PROCESS_SOURCES = {
    "forest": ROOT / "rewet_forest_process" / "forest_rewet_merged_firstdrought_state.csv",
    "savanna": ROOT / "rewet_savanna_process" / "savanna_rewet_merged_firstdrought_state.csv",
    "grassland": ROOT / "rewet_grassland_process" / "grassland_rewet_merged_firstdrought_state.csv",
}

ECOS = ["forest", "savanna", "grassland"]
ECO_LABELS = {"forest": "Forest", "savanna": "Savanna", "grassland": "Grassland"}
COLORS = {"forest": "#1b7258", "savanna": "#2aa7a1", "grassland": "#9fbd45"}
LIGHT = {"forest": "#d9eee5", "savanna": "#d7f1ee", "grassland": "#edf5c9"}
RNG = np.random.default_rng(20260702)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.9,
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "figure.facecolor": "white",
})


def boxed_axes(ax):
    ax.grid(False)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(top=False, right=False, labelsize=8.5)


def panel_label(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, fontweight="bold", color="black")


def load_joined_data():
    # Reproduce the original ED2 statistical level: use the already merged
    # process tables, retaining their record weighting and site identifiers.
    # These are the tables behind the earlier 9-site Savanna summary.
    frames = []
    for eco, path in PROCESS_SOURCES.items():
        usecols = [
            "file", "q_21", "theta_peak_after", "theta_jump_after", "SITE_ID",
            "SWC_first", "theta_star_first",
        ]
        src = pd.read_csv(path, usecols=usecols, low_memory=False)
        src["ecosystem"] = eco
        for col in ["q_21", "theta_peak_after", "theta_jump_after", "SWC_first", "theta_star_first"]:
            src[col] = pd.to_numeric(src[col], errors="coerce")
        src["dtheta"] = src["theta_peak_after"] - src["SWC_first"]
        # Post-rewetting moisture buffering: peak soil moisture during the
        # 21 d response window relative to the fitted threshold.
        src["moisture_buffering"] = src["theta_peak_after"] - src["theta_star_first"]
        frames.append(src)
    data = pd.concat(frames, ignore_index=True)

    # Add drydown duration from the original rewetting table. The process
    # tables retain the same q21 values but not the start/end dates.
    raw_cols = ["file", "ecosystem", "q_21", "q_21_theta_peak", "q_21_theta_jump",
                "q_21_start", "q_21_end"]
    raw = pd.read_csv(REWET_CSV, usecols=raw_cols, low_memory=False)
    for col in ["q_21", "q_21_theta_peak", "q_21_theta_jump"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").round(10)
    raw["duration_21"] = (
        pd.to_datetime(raw["q_21_end"], errors="coerce")
        - pd.to_datetime(raw["q_21_start"], errors="coerce")
    ).dt.days
    raw = raw.dropna(subset=["q_21", "q_21_theta_peak", "q_21_theta_jump", "duration_21"])
    key_cols = ["file", "q_21", "q_21_theta_peak", "q_21_theta_jump"]
    raw_key = raw[key_cols + ["duration_21"]].drop_duplicates(key_cols)
    raw_key = raw_key.rename(columns={
        "q_21": "q_21_key", "q_21_theta_peak": "theta_peak_key",
        "q_21_theta_jump": "theta_jump_key",
    })
    for col in ["q_21", "theta_peak_after", "theta_jump_after"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").round(10)
    data = data.rename(columns={
        "q_21": "q_21_key", "theta_peak_after": "theta_peak_key",
        "theta_jump_after": "theta_jump_key",
    })
    data = data.merge(raw_key, on=["file", "q_21_key", "theta_peak_key", "theta_jump_key"], how="left")
    data["drydown_rate"] = data["theta_jump_key"] / data["duration_21"]
    return data[["ecosystem", "SITE_ID", "dtheta", "drydown_rate", "moisture_buffering"]].replace(
        [np.inf, -np.inf], np.nan
    )


def site_cluster_ci(df, col, n_boot=1000):
    df = df.dropna(subset=[col])
    median = float(df[col].median())
    groups = {site: g[col].to_numpy(float) for site, g in df.groupby("SITE_ID", observed=True)}
    sites = np.array(list(groups), dtype=object)
    if len(sites) < 2:
        return median, np.nan, np.nan
    vals = []
    for _ in range(n_boot):
        sampled = RNG.choice(sites, size=len(sites), replace=True)
        vals.append(np.median(np.concatenate([groups[s] for s in sampled])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return median, float(lo), float(hi)


data = load_joined_data()
metrics = [
    ("dtheta", r"$Δ\theta$", r"$m^3$ $m^{-3}$", r"Event-level $Δ\theta$"),
    ("drydown_rate", "Drydown rate", r"$m^3$ $m^{-3}$ $d^{-1}$", "Event-level drydown rate"),
    ("moisture_buffering", "Moisture buffering", r"$m^3$ $m^{-3}$", "Event-level moisture buffering"),
]

summary_rows = []
for eco in ECOS:
    sub = data[data["ecosystem"].eq(eco)]
    row = {"ecosystem": eco, "N_events": len(sub), "N_sites": sub["SITE_ID"].nunique()}
    for col, *_ in metrics:
        med, lo, hi = site_cluster_ci(sub, col)
        row.update({f"{col}_median": med, f"{col}_lo": lo, f"{col}_hi": hi})
    summary_rows.append(row)
pd.DataFrame(summary_rows).to_csv(OUTDIR / "soil_moisture_rewetting_change_summary.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.6), squeeze=False)
fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.13, wspace=0.34, hspace=0.42)
positions = np.arange(len(ECOS))

for col_idx, (metric, label, unit, title) in enumerate(metrics):
    # Top row: event-level distributions. No raw points or site counts are shown.
    ax = axes[0, col_idx]
    arrays = []
    for eco in ECOS:
        vals = data.loc[data["ecosystem"].eq(eco), metric].dropna().to_numpy(float)
        lo, hi = np.quantile(vals, [0.005, 0.995])
        arrays.append(vals[(vals >= lo) & (vals <= hi)])
    parts = ax.violinplot(arrays, positions=positions, widths=0.62,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, eco in zip(parts["bodies"], ECOS):
        body.set_facecolor(LIGHT[eco])
        body.set_edgecolor(COLORS[eco])
        body.set_linewidth(1.15)
        body.set_alpha(0.95)
    top_means = []
    for i, eco in enumerate(ECOS):
        sub = data[data["ecosystem"].eq(eco)]
        med, lo, hi = site_cluster_ci(sub, metric)
        mean = float(sub[metric].mean())
        top_means.append(mean)
        ax.plot([i - 0.23, i + 0.23], [med, med], color="black", lw=1.25)
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], fmt="none",
                    ecolor="black", lw=1.0, capsize=3)
        ax.scatter(i, mean, s=18, marker="o", facecolor="white", edgecolor="black",
                   linewidth=0.9, zorder=4)
    data_lo = min(float(np.min(arr)) for arr in arrays)
    data_hi = max(float(np.max(arr)) for arr in arrays)
    data_span = max(data_hi - data_lo, 1e-6)
    top_pad = 0.10 * data_span
    ax.set_ylim(data_lo - 0.04 * data_span, data_hi + top_pad)
    for i, mean in enumerate(top_means):
        label_y = float(np.max(arrays[i])) + 0.035 * data_span
        ax.text(i, label_y, f"{mean:.3g}", ha="center", va="bottom",
                fontsize=7.2, color="black", zorder=5)
    ax.axhline(0, color="#666666", lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(positions)
    ax.set_xticklabels([ECO_LABELS[e] for e in ECOS], fontsize=8.5)
    ax.set_ylabel(f"{label} ({unit})", fontsize=9)
    ax.set_title(title, fontsize=10, pad=7)
    boxed_axes(ax)
    panel_label(ax, chr(ord("a") + col_idx))

    # Bottom row: site-level distributions, with no site labels or points.
    ax = axes[1, col_idx]
    site_arrays = []
    for eco in ECOS:
        site_arrays.append(
            data.loc[data["ecosystem"].eq(eco)].groupby("SITE_ID", observed=True)[metric].median().dropna().to_numpy(float)
        )
    box = ax.boxplot(site_arrays, positions=positions, widths=0.48, patch_artist=True,
                     showfliers=False,
                     medianprops=dict(color="black", linewidth=1.25),
                     whiskerprops=dict(color="black", linewidth=0.9),
                     capprops=dict(color="black", linewidth=0.9),
                     boxprops=dict(linewidth=1.05))
    for patch, eco in zip(box["boxes"], ECOS):
        patch.set_facecolor(LIGHT[eco])
        patch.set_edgecolor(COLORS[eco])
        patch.set_alpha(0.95)
    site_means = []
    for i, eco in enumerate(ECOS):
        site_values = data.loc[data["ecosystem"].eq(eco)].groupby("SITE_ID", observed=True)[metric].median()
        site_mean = float(site_values.mean())
        site_means.append(site_mean)
        ax.scatter(i, site_mean, s=18, marker="o", facecolor="white", edgecolor="black",
                   linewidth=0.9, zorder=4)
    site_lo = min(0.0, min(float(np.min(v)) for v in site_arrays))
    site_hi = max(0.0, max(float(np.max(v)) for v in site_arrays))
    site_span = max(site_hi - site_lo, 1e-6)
    site_pad = 0.10 * site_span
    ax.set_ylim(site_lo - 0.04 * site_span, site_hi + site_pad)
    for i, site_mean in enumerate(site_means):
        label_y = float(np.max(site_arrays[i])) + 0.035 * site_span
        ax.text(i, label_y, f"{site_mean:.3g}", ha="center", va="bottom",
                fontsize=7.2, color="black", zorder=5)
    ax.axhline(0, color="#666666", lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(positions)
    ax.set_xticklabels([ECO_LABELS[e] for e in ECOS], fontsize=8.5)
    ax.set_ylabel(f"Site-median {label} ({unit})", fontsize=9)
    ax.set_title("Site-level distributions", fontsize=10, pad=7)
    boxed_axes(ax)
    panel_label(ax, chr(ord("d") + col_idx))

fig.savefig(OUTBASE.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUTBASE.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUTBASE.with_suffix(".svg"), bbox_inches="tight")
print(OUTBASE.with_suffix(".png"))
print(OUTDIR / "soil_moisture_rewetting_change_summary.csv")
