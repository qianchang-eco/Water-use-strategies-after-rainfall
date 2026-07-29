import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
DATA = ROOT / "rewet_savanna_process" / "savanna_rewet_process_table_21d.csv"
TABLE_DIR = ROOT / "rewet_savanna_process" / "results" / "phase_3x2_pet_tables"
SMOOTH_DIR = TABLE_DIR / "shape_data_smooth"
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTBASE = OUTDIR / "extended_data_fig5_savanna_site_support"

SEGMENTS = [
    ("S1", 0.50, 1.49, "Low q$_{original}$"),
    ("S2", 1.49, 2.58, "Intermediate q$_{original}$"),
    ("S3", 2.58, 3.00, "High q$_{original}$"),
]
SEGMENT_LABELS = {"S1": "S1\nlow", "S2": "S2\nmid", "S3": "S3\nhigh"}
SEGMENT_COLORS = {"S1": "#b7cf3d", "S2": "#35aa88", "S3": "#075f55"}
PREDICTORS = ["drydown_rate_21", "state_buffer_first", "PET_p90_21"]
PRED_LABELS = {"drydown_rate_21": "drydown rate", "state_buffer_first": "state buffer", "PET_p90_21": "PET"}
RNG = np.random.default_rng(20260626)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "figure.facecolor": "white",
    }
)


def panel_label(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, fontweight="bold")


def load_data():
    cols = ["SITE_ID", "q_original", "dq_21", "drydown_rate_21", "PET_p90_21", "SWC_first", "theta_star_first"]
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    for c in cols:
        if c != "SITE_ID":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["state_buffer_first"] = df["SWC_first"] - df["theta_star_first"]
    df = df.dropna(subset=["SITE_ID", "q_original", "dq_21", "drydown_rate_21", "PET_p90_21", "state_buffer_first"])
    df = df[df["q_original"].between(0.50, 3.00)].copy()
    bins = [0.50, 1.49, 2.58, 3.000001]
    df["q_state"] = pd.cut(df["q_original"], bins=bins, labels=["S1", "S2", "S3"], include_lowest=False)
    df = df.dropna(subset=["q_state"]).copy()
    return df


def site_balanced_beta(df, predictors=PREDICTORS):
    d = df.dropna(subset=["dq_21"] + predictors + ["SITE_ID"]).copy()
    # Site-level medians keep this panel aligned with the robustness question:
    # whether the result is controlled by individual sites rather than repeated events.
    d = (
        d.groupby("SITE_ID", observed=True)[["dq_21"] + predictors]
        .median()
        .reset_index()
    )
    if len(d) < 4:
        return np.full(len(predictors), np.nan)
    y = d["dq_21"].clip(-3, 3).to_numpy(float)
    y = (y - y.mean()) / (y.std(ddof=0) or 1)
    x = d[predictors].to_numpy(float)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0, ddof=0) == 0, 1, x.std(axis=0, ddof=0))
    X = np.column_stack([np.ones(len(x)), x])
    alpha = 0.35
    penalty = np.eye(X.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.solve(X.T @ X + alpha * penalty, X.T @ y)
    return beta[1:]


def loo_effects(df):
    rows = []
    full_by = {}
    for state in ["S1", "S2", "S3"]:
        ds = df[df["q_state"].eq(state)]
        full_by[state] = site_balanced_beta(ds)
        for site in sorted(df["SITE_ID"].unique()):
            beta = site_balanced_beta(ds[ds["SITE_ID"].ne(site)])
            for pred, val in zip(PREDICTORS, beta):
                rows.append({"state": state, "left_out_site": site, "predictor": pred, "beta": val})
    full = []
    for state, beta in full_by.items():
        for pred, val in zip(PREDICTORS, beta):
            full.append({"state": state, "predictor": pred, "beta": val})
    return pd.DataFrame(rows), pd.DataFrame(full)


def bootstrap_core_effects(df, n_boot=800):
    sites = df["SITE_ID"].drop_duplicates().to_numpy()
    groups = {s: g for s, g in df.groupby("SITE_ID", observed=True)}
    base = site_balanced_beta(df)
    boots = []
    for _ in range(n_boot):
        sampled = RNG.choice(sites, size=len(sites), replace=True)
        dd = pd.concat([groups[s] for s in sampled], ignore_index=True)
        boots.append(site_balanced_beta(dd))
    arr = np.vstack(boots)
    rows = []
    for j, pred in enumerate(PREDICTORS):
        beta = base[j]
        lo, hi = np.nanpercentile(arr[:, j], [2.5, 97.5])
        if pred == "drydown_rate_21":
            pdir = np.nanmean(arr[:, j] < 0)
            direction = "P(beta < 0)"
        elif pred == "state_buffer_first":
            pdir = np.nanmean(arr[:, j] > 0)
            direction = "P(beta > 0)"
        else:
            pdir = max(np.nanmean(arr[:, j] < 0), np.nanmean(arr[:, j] > 0))
            direction = "P(direction)"
        rows.append({"predictor": pred, "beta": beta, "lo": lo, "hi": hi, "direction_probability": pdir, "direction_label": direction})
    return pd.DataFrame(rows), pd.DataFrame(arr, columns=PREDICTORS)


def heat_file(segment_label, yvar):
    seg_name = {
        "S1": "S1(0.50,1.49]",
        "S2": "S2(1.49,2.58]",
        "S3": "S3(2.58,3.00]",
    }[segment_label]
    return TABLE_DIR / f"dq_surface_tables__{seg_name}__y{yvar}.csv"


def smooth_file(segment_label, yvar):
    seg_name = {
        "S1": "S1(0.50,1.49]",
        "S2": "S2(1.49,2.58]",
        "S3": "S3(2.58,3.00]",
    }[segment_label]
    return SMOOTH_DIR / f"dq_surface_tables__{seg_name}__y{yvar}__shape_data_smooth_grid.csv"


def table_edges(segment, yvar):
    tab = pd.read_csv(heat_file(segment, yvar))
    for c in ["x_bin", "y_bin", "x_lo", "x_hi", "y_lo", "y_hi"]:
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
    tab = tab.dropna(subset=["x_bin", "y_bin", "x_lo", "x_hi", "y_lo", "y_hi"])
    tab["x_bin"] = tab["x_bin"].astype(int)
    tab["y_bin"] = tab["y_bin"].astype(int)
    xmeta = tab.sort_values("x_bin").groupby("x_bin")[["x_lo", "x_hi"]].first()
    ymeta = tab.sort_values("y_bin").groupby("y_bin")[["y_lo", "y_hi"]].first()
    x_edges = np.r_[xmeta["x_lo"].to_numpy(), xmeta["x_hi"].to_numpy()[-1]]
    y_edges = np.r_[ymeta["y_lo"].to_numpy(), ymeta["y_hi"].to_numpy()[-1]]
    return x_edges, y_edges


def support_grid(df, segment, yvar):
    x0, y0 = table_edges(segment, yvar)
    # Keep the exact axis range of the main Fig. 4 tables, but aggregate cells
    # for observational support so independent-site coverage is interpretable.
    x_edges = np.linspace(x0.min(), x0.max(), 6)
    y_edges = np.linspace(y0.min(), y0.max(), 6)
    d = df[df["q_state"].eq(segment)].copy()
    d["xb"] = pd.cut(d["PET_p90_21"], x_edges, labels=False, include_lowest=True)
    d["yb"] = pd.cut(d[yvar], y_edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["xb", "yb"])
    d["xb"] = d["xb"].astype(int)
    d["yb"] = d["yb"].astype(int)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    site_count = np.zeros((ny, nx), dtype=float)
    event_count = np.zeros((ny, nx), dtype=float)
    for (yb, xb), g in d.groupby(["yb", "xb"], observed=True):
        if 0 <= yb < ny and 0 <= xb < nx:
            site_count[yb, xb] = g["SITE_ID"].nunique()
            event_count[yb, xb] = len(g)
    return x_edges, y_edges, site_count, event_count


def load_zero_contour(segment, yvar):
    path = smooth_file(segment, yvar)
    if not path.exists():
        return None
    d = pd.read_csv(path)
    for c in ["x", "y", "dq"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["x", "y", "dq"])
    x = np.sort(d["x"].unique())
    y = np.sort(d["y"].unique())
    z = d.pivot_table(index="y", columns="x", values="dq", aggfunc="first").reindex(index=y, columns=x).to_numpy(float)
    return x, y, z


def annotate_counts(ax, x_edges, y_edges, event_count, site_count):
    ny, nx = event_count.shape
    xcent = (x_edges[:-1] + x_edges[1:]) / 2
    ycent = (y_edges[:-1] + y_edges[1:]) / 2
    for iy in range(ny):
        for ix in range(nx):
            n = event_count[iy, ix]
            s = site_count[iy, ix]
            if n <= 0:
                continue
            if nx > 9 and (ix % 2) and s >= 3:
                continue
            color = "white" if s >= 5 else "black"
            ax.text(xcent[ix], ycent[iy], f"{int(n)}", ha="center", va="center", fontsize=4.6, color=color)


df = load_data()
loo, full = loo_effects(df)
boot_summary, boot_draws = bootstrap_core_effects(df)
site_summary = (
    df.groupby(["SITE_ID", "q_state"], observed=True)
    .agg(events=("dq_21", "size"), median_dq=("dq_21", "median"))
    .reset_index()
)
loo.to_csv(OUTDIR / "extended_data_fig5_savanna_leave_one_site.csv", index=False)
full.to_csv(OUTDIR / "extended_data_fig5_savanna_full_effects.csv", index=False)
boot_summary.to_csv(OUTDIR / "extended_data_fig5_savanna_bootstrap_core_effects.csv", index=False)
site_summary.to_csv(OUTDIR / "extended_data_fig5_savanna_site_state_counts.csv", index=False)

fig = plt.figure(figsize=(12.4, 11.2))
gs = fig.add_gridspec(3, 3, left=0.065, right=0.965, top=0.955, bottom=0.095, wspace=0.34, hspace=0.48)
axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]
letters = list("abcdefghi")

# a. Site-level event counts by q state.
ax = axes[0]
site_order = (
    df.groupby("SITE_ID", observed=True)["dq_21"]
    .size()
    .sort_values(ascending=False)
    .index.tolist()
)
bottom = np.zeros(len(site_order))
pivot_counts = site_summary.pivot_table(index="SITE_ID", columns="q_state", values="events", aggfunc="sum").reindex(index=site_order, columns=["S1", "S2", "S3"]).fillna(0)
site_median = df.groupby("SITE_ID", observed=True)["dq_21"].median().reindex(site_order)
for state in ["S1", "S2", "S3"]:
    vals = pivot_counts[state].to_numpy(float)
    ax.bar(np.arange(len(site_order)), vals, bottom=bottom, color=SEGMENT_COLORS[state], edgecolor="white", linewidth=0.45, label=SEGMENT_LABELS[state].replace("\n", " "))
    bottom += vals
for i, med in enumerate(site_median):
    ax.text(i, bottom[i] * 1.02 + 2, f"{med:.2f}", rotation=90, ha="center", va="bottom", fontsize=5.5, color="#333333")
ax.set_ylabel("Events")
ax.set_xticks(np.arange(len(site_order)))
ax.set_xticklabels(site_order, rotation=55, ha="right", fontsize=6.0)
ax.set_title("Site event support by initial state", fontsize=9, fontweight="bold")
ax.legend(frameon=False, fontsize=6.3, loc="upper right")
panel_label(ax, "a")

# b. Leave-one-site-out effects by state and predictor.
ax = axes[1]
row_labels = [f"{s} | {PRED_LABELS[p]}" for s in ["S1", "S2", "S3"] for p in PREDICTORS]
ypos = np.arange(len(row_labels))
for ri, label in enumerate(row_labels):
    state, pred_label = label.split(" | ")
    pred = {v: k for k, v in PRED_LABELS.items()}[pred_label]
    vals = loo[(loo["state"].eq(state)) & (loo["predictor"].eq(pred))]["beta"].dropna().to_numpy(float)
    ax.scatter(vals, np.full_like(vals, ri), s=13, color="#9a9a9a", alpha=0.62, zorder=2)
    fval = full[(full["state"].eq(state)) & (full["predictor"].eq(pred))]["beta"].iloc[0]
    ax.scatter([fval], [ri], s=45, marker="D", color="black", zorder=3)
ax.axvline(0, color="#666666", lw=0.8, ls=(0, (3, 2)))
ax.set_yticks(ypos)
ax.set_yticklabels(row_labels, fontsize=6.5)
ax.invert_yaxis()
ax.set_xlabel("Standardized effect on dq$_{21}$")
ax.set_title("Leave-one-site-out effects", fontsize=9, fontweight="bold")
ax.grid(axis="x", color="#e7e7e7", lw=0.7)
ax.set_xlim(-1.2, 1.2)
ax.legend(
    handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9a9a9a", alpha=0.7, markersize=4, label="one site removed"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="black", markersize=5, label="all sites"),
    ],
    frameon=False,
    fontsize=6.4,
    loc="lower right",
)
panel_label(ax, "b")

# c. Site-cluster bootstrap core effects.
ax = axes[2]
yy = np.arange(len(PREDICTORS))
for i, pred in enumerate(PREDICTORS):
    r = boot_summary[boot_summary["predictor"].eq(pred)].iloc[0]
    color = "#0b6b57" if pred == "drydown_rate_21" else ("#35aa88" if pred == "state_buffer_first" else "#b7cf3d")
    ax.errorbar(r["beta"], i, xerr=[[r["beta"] - r["lo"]], [r["hi"] - r["beta"]]], fmt="o", color=color, lw=1.5, capsize=3)
    ax.text(r["hi"] + 0.025, i, f"{r['direction_label']}={r['direction_probability']:.2f}", va="center", fontsize=6.5)
ax.axvline(0, color="#666666", lw=0.8, ls=(0, (3, 2)))
ax.set_yticks(yy)
ax.set_yticklabels([PRED_LABELS[p] for p in PREDICTORS])
ax.invert_yaxis()
ax.set_xlabel("Standardized effect on dq$_{21}$")
ax.set_title("Site-cluster bootstrap", fontsize=9, fontweight="bold")
ax.grid(axis="x", color="#e7e7e7", lw=0.7)
ax.set_xlim(-1.0, 1.0)
panel_label(ax, "c")

# d-i. Observational support grids.
support_cmap = LinearSegmentedColormap.from_list("support", ["#dcecc6", "#73c58d", "#006b5b"])
norm = Normalize(vmin=3, vmax=max(5, df["SITE_ID"].nunique()))

def draw_support(ax, segment, yvar, title=None, ylabel=None):
    x_edges, y_edges, sites, events = support_grid(df, segment, yvar)
    ax.set_facecolor("#eeeeeb")
    normal = np.ma.masked_where(sites < 3, sites)
    im = ax.pcolormesh(x_edges, y_edges, normal, cmap=support_cmap, norm=norm, shading="flat", edgecolors="white", linewidth=0.25)
    # Mark unsupported or weakly supported cells without giving them inferential weight.
    for iy in range(sites.shape[0]):
        for ix in range(sites.shape[1]):
            if sites[iy, ix] == 0:
                continue
            if sites[iy, ix] < 3:
                rect = Rectangle((x_edges[ix], y_edges[iy]), x_edges[ix + 1] - x_edges[ix], y_edges[iy + 1] - y_edges[iy], facecolor="#d9d9d6", edgecolor="white", lw=0.25, hatch="////", alpha=0.65)
                ax.add_patch(rect)
    annotate_counts(ax, x_edges, y_edges, events, sites)
    contour = load_zero_contour(segment, yvar)
    if contour is not None:
        x, y, z = contour
        if np.nanmin(z) < 0 < np.nanmax(z):
            X, Y = np.meshgrid(x, y)
            ax.contour(X, Y, z, levels=[0], colors="black", linewidths=1.0)
    if title:
        ax.set_title(title, fontsize=8.4, fontweight="bold", pad=4)
    ax.set_xlabel("PET (mm day$^{-1}$)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=6.7, length=2.8)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    if yvar == "drydown_rate_21":
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    return im

for col, (seg, _lo, _hi, title) in enumerate(SEGMENTS):
    ax = axes[3 + col]
    im = draw_support(
        ax,
        seg,
        "drydown_rate_21",
        title=title,
        ylabel="Drydown rate\n(m$^3$ m$^{-3}$ day$^{-1}$)" if col == 0 else None,
    )
    panel_label(ax, letters[3 + col])

for col, (seg, _lo, _hi, title) in enumerate(SEGMENTS):
    ax = axes[6 + col]
    im = draw_support(
        ax,
        seg,
        "state_buffer_first",
        title=None,
        ylabel="State buffer\n(m$^3$ m$^{-3}$)" if col == 0 else None,
    )
    panel_label(ax, letters[6 + col])

cax = fig.add_axes([0.972, 0.245, 0.012, 0.40])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Independent sites per cell", rotation=270, labelpad=13, fontsize=7.3)
cbar.ax.tick_params(labelsize=6.8)
fig.text(0.73, 0.035, "Numbers in cells indicate events; hatched cells have 1-2 supporting sites; black line is dq=0 contour.", ha="center", va="center", fontsize=7.0)

for ext in [".png", ".pdf", ".svg"]:
    fig.savefig(OUTBASE.with_suffix(ext), dpi=600 if ext == ".png" else None, bbox_inches="tight")

print(OUTBASE.with_suffix(".png"))
print(OUTBASE.with_suffix(".pdf"))
print(OUTBASE.with_suffix(".svg"))
