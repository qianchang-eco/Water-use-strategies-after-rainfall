import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import Delaunay, QhullError


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
DATA = ROOT / "results_paper" / "ismn_results_star_th0_04may_ecosystem.csv"
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTBASE = OUTDIR / "fig2_hydroclimatic_vegetation_backgrounds_unbounded_q"
SUBMISSION_PNG = ROOT / "提交" / "Figures" / "Fig. 2.png"
SUBMISSION_PNG.parent.mkdir(parents=True, exist_ok=True)
SOURCE_DIR = OUTDIR / "fig2_unbounded_q_source_data"
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

ECOS = ["forest", "savanna", "grassland"]
ECO_LABELS = {"forest": "Forest", "savanna": "Savanna", "grassland": "Grassland"}
LINE_ORDER = ["low", "median", "high"]
LINE_COLORS = {"low": "#b7cf3d", "median": "#32aa86", "high": "#075f55"}
LINE_LABELS = {"low": "Low LAI (P25)", "median": "Median LAI (P50)", "high": "High LAI (P75)"}
DRIVER_ORDER = ["AI60", "LAI", "AI60 x LAI"]
DRIVER_COLORS = {"AI60": "#b7cf3d", "LAI": "#35aa88", "AI60 x LAI": "#075f55"}
DRIVER_LABELS = {"AI60": r"$AI_{60}$", "LAI": "LAI", "AI60 x LAI": r"$AI_{60}\times LAI$"}
STATE_LABELS = ["R1\nlower tail", "R2\nlower-mid", "R3\nupper-mid", "R4\nupper tail"]

N_BOOT = 500
RNG = np.random.default_rng(20260626)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.85,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "figure.facecolor": "white",
    }
)


def panel_label(ax, letter):
    ax.text(
        -0.13,
        1.04,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
    )


def boxed_axes_no_grid(ax):
    ax.grid(False)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.85)
    ax.tick_params(top=False, right=False)


def load_data():
    usecols = ["ecosystem", "SITE_ID", "q_q", "AI_60d", "LAI"]
    df = pd.read_csv(DATA, usecols=usecols, low_memory=False)
    df["ecosystem"] = df["ecosystem"].astype(str).str.lower()
    df = df[df["ecosystem"].isin(ECOS)].copy()
    for col in ["q_q", "AI_60d", "LAI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ecosystem", "SITE_ID", "q_q", "AI_60d", "LAI"])
    # No upper q threshold is imposed in this figure. The log1p response keeps
    # large positive q estimates from dominating the regression numerically.
    df = df[(df["q_q"] >= 0.01) & (df["AI_60d"] > 0) & (df["LAI"] >= 0)].copy()
    return df


def robust_limits(sub, col, qlo=0.05, qhi=0.95):
    lo, hi = sub[col].quantile([qlo, qhi]).to_numpy(float)
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 0.1)
        lo, hi = lo - pad, hi + pad
    return lo, hi


def make_support(sub, ai_lim, lai_lim):
    support = sub[
        sub["AI_60d"].between(*ai_lim) & sub["LAI"].between(*lai_lim)
    ][["AI_60d", "LAI"]].drop_duplicates()
    if len(support) < 4:
        return None
    try:
        return Delaunay(support.to_numpy(float))
    except QhullError:
        return None


def in_support(points, hull, ai_lim, lai_lim):
    pts = np.asarray(points, dtype=float)
    inside_box = (
        (pts[:, 0] >= ai_lim[0])
        & (pts[:, 0] <= ai_lim[1])
        & (pts[:, 1] >= lai_lim[0])
        & (pts[:, 1] <= lai_lim[1])
    )
    if hull is None:
        return inside_box
    return inside_box & (hull.find_simplex(pts) >= 0)


def scale(v, lo, hi):
    return (np.asarray(v, dtype=float) - lo) / (hi - lo)


def design_matrix(ai, lai, ai_lim, lai_lim):
    x = np.clip(scale(ai, *ai_lim), -0.2, 1.2)
    z = np.clip(scale(lai, *lai_lim), -0.2, 1.2)
    return np.column_stack(
        [
            np.ones_like(x),
            x,
            x**2,
            z,
            z**2,
            x * z,
            x**2 * z,
            x * z**2,
        ]
    )


def fit_model(sub, ai_lim, lai_lim, site_boot_weights=None):
    work = sub[sub["AI_60d"].between(*ai_lim) & sub["LAI"].between(*lai_lim)].copy()
    if site_boot_weights is None:
        site_counts = work["SITE_ID"].map(work["SITE_ID"].value_counts()).to_numpy(float)
        weights = 1.0 / site_counts
    else:
        weights = work["SITE_ID"].map(site_boot_weights).fillna(0).to_numpy(float)
        site_counts = work["SITE_ID"].map(work["SITE_ID"].value_counts()).to_numpy(float)
        weights = weights / site_counts
    keep = weights > 0
    work = work.loc[keep]
    weights = weights[keep]

    y = np.log1p(work["q_q"].to_numpy(float))
    xmat = design_matrix(work["AI_60d"].to_numpy(), work["LAI"].to_numpy(), ai_lim, lai_lim)
    weights = weights / np.nanmean(weights)
    sw = np.sqrt(weights)
    lam = 1.5
    penalty = np.eye(xmat.shape[1])
    penalty[0, 0] = 0.0
    xtwx = (xmat * sw[:, None]).T @ (xmat * sw[:, None]) + lam * penalty
    xtwy = (xmat * sw[:, None]).T @ (y * sw)
    beta = np.linalg.solve(xtwx, xtwy)
    return beta


def predict(beta, ai, lai, ai_lim, lai_lim):
    eta = design_matrix(ai, lai, ai_lim, lai_lim) @ beta
    return np.maximum(np.expm1(eta), 0.01)


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2, side="left")])


def empirical_bin_points(sub, lai_quantile, ai_lim, seed, n_bins=5):
    """Site-level medians near a displayed LAI percentile, with cluster CIs."""
    lower = max(0.0, lai_quantile - 0.125)
    upper = min(1.0, lai_quantile + 0.125)
    lai_lo, lai_hi = sub["LAI"].quantile([lower, upper]).to_numpy(float)
    work = sub[sub["LAI"].between(lai_lo, lai_hi)
               & sub["AI_60d"].between(*ai_lim)].copy()
    work["ai_bin"] = pd.qcut(work["AI_60d"].rank(method="first"), n_bins,
                             labels=False)
    site_bin = (work.groupby(["SITE_ID", "ai_bin"], as_index=False)
                .agg(AI60=("AI_60d", "median"), q=("q_q", "median")))
    site_bin["SITE_ID"] = site_bin["SITE_ID"].astype(str)
    sites = site_bin["SITE_ID"].drop_duplicates().to_numpy()
    rows = []
    for bin_id in range(n_bins):
        g = site_bin[site_bin["ai_bin"].eq(bin_id)]
        rows.append({"ai_bin": bin_id, "AI60": float(g["AI60"].median()),
                     "q": float(g["q"].median()),
                     "n_sites": int(g["SITE_ID"].nunique())})
    points = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    draws = np.full((N_BOOT, n_bins), np.nan)
    for b in range(N_BOOT):
        sampled = rng.choice(sites, size=len(sites), replace=True)
        counts = pd.Series(sampled).value_counts()
        multiplicity = site_bin["SITE_ID"].map(counts).fillna(0).to_numpy(float)
        for bin_id in range(n_bins):
            mask = site_bin["ai_bin"].eq(bin_id).to_numpy()
            draws[b, bin_id] = weighted_median(site_bin.loc[mask, "q"], multiplicity[mask])
    points["ci_low"], points["ci_high"] = np.nanquantile(draws, [0.025, 0.975], axis=0)
    points["lai_quantile"] = lai_quantile
    points["lai_band_low"] = lower
    points["lai_band_high"] = upper
    return points, site_bin


def bootstrap_predictions(sub, ai_grid, lai_value, ai_lim, lai_lim, hull):
    points = np.column_stack([ai_grid, np.full_like(ai_grid, lai_value)])
    support_mask = in_support(points, hull, ai_lim, lai_lim)
    y = np.full_like(ai_grid, np.nan, dtype=float)
    lo = np.full_like(ai_grid, np.nan, dtype=float)
    hi = np.full_like(ai_grid, np.nan, dtype=float)
    if support_mask.sum() < 5:
        return y, lo, hi

    sites = sub["SITE_ID"].drop_duplicates().to_numpy()
    beta = fit_model(sub, ai_lim, lai_lim)
    y[support_mask] = predict(
        beta,
        ai_grid[support_mask],
        np.full(support_mask.sum(), lai_value),
        ai_lim,
        lai_lim,
    )

    boots = []
    for _ in range(N_BOOT):
        sampled = RNG.choice(sites, size=len(sites), replace=True)
        weights = pd.Series(sampled).value_counts()
        try:
            b = fit_model(sub, ai_lim, lai_lim, site_boot_weights=weights)
            boots.append(
                predict(
                    b,
                    ai_grid[support_mask],
                    np.full(support_mask.sum(), lai_value),
                    ai_lim,
                    lai_lim,
                )
            )
        except np.linalg.LinAlgError:
            continue
    if len(boots) >= 20:
        arr = np.vstack(boots)
        lo[support_mask] = np.nanpercentile(arr, 2.5, axis=0)
        hi[support_mask] = np.nanpercentile(arr, 97.5, axis=0)
    return y, lo, hi


def add_site_rug(ax, sub, ai_lim):
    site_ai = (
        sub[sub["AI_60d"].between(*ai_lim)]
        .groupby("SITE_ID", observed=True)["AI_60d"]
        .median()
        .to_numpy(float)
    )
    ax.plot(site_ai, np.full_like(site_ai, 0.045), "|", color="#1d1d1d", alpha=0.34, ms=6, mew=0.65)


def add_state_labels(ax, ai_lim):
    ax.axhline(1, color="#626262", lw=0.9, ls=(0, (4, 2)), zorder=0)


def weighted_r2(y, yhat, weights):
    weights = np.asarray(weights, dtype=float)
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ybar = np.average(y, weights=weights)
    ss_tot = np.sum(weights * (y - ybar) ** 2)
    ss_res = np.sum(weights * (y - yhat) ** 2)
    if ss_tot <= 0:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def fit_linear_weighted(xmat, y, weights, lam=0.03):
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.nanmean(weights)
    sw = np.sqrt(weights)
    penalty = np.eye(xmat.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve((xmat * sw[:, None]).T @ (xmat * sw[:, None]) + lam * penalty, (xmat * sw[:, None]).T @ (y * sw))


def contribution_by_state(sub):
    work = sub[["SITE_ID", "q_q", "AI_60d", "LAI"]].copy()
    work["q_state"] = pd.qcut(work["q_q"].rank(method="first"), 4, labels=False)
    rows = []
    for state in range(4):
        g = work[work["q_state"].eq(state)].copy()
        if len(g) < 30 or g["SITE_ID"].nunique() < 4:
            vals = np.array([np.nan, np.nan, np.nan])
        else:
            site_counts = g["SITE_ID"].map(g["SITE_ID"].value_counts()).to_numpy(float)
            weights = 1.0 / site_counts
            # Keep panels d-f identical to the original submitted figure.  The
            # original contribution calculation winsorised q at 3 even though
            # the revised response curves in panels a-c retain the full q range.
            y = np.log1p(g["q_q"].clip(0, 3.0).to_numpy(float))
            x_ai = g["AI_60d"].to_numpy(float)
            x_lai = g["LAI"].to_numpy(float)
            x_int = x_ai * x_lai
            xraw = np.column_stack([x_ai, x_lai, x_int])
            xstd = (xraw - xraw.mean(axis=0)) / np.where(xraw.std(axis=0, ddof=0) == 0, 1, xraw.std(axis=0, ddof=0))
            full = np.column_stack([np.ones(len(xstd)), xstd])
            beta_full = fit_linear_weighted(full, y, weights)
            r2_full = weighted_r2(y, full @ beta_full, weights)
            drops = []
            for j in range(3):
                cols = [k for k in range(3) if k != j]
                red = np.column_stack([np.ones(len(xstd)), xstd[:, cols]])
                beta_red = fit_linear_weighted(red, y, weights)
                r2_red = weighted_r2(y, red @ beta_red, weights)
                drops.append(max(r2_full - r2_red, 0.0))
            vals = np.asarray(drops, dtype=float)
            if np.nansum(vals) <= 0:
                vals = np.abs(beta_full[1:])
            vals = vals / np.nansum(vals) * 100 if np.nansum(vals) > 0 else np.array([np.nan, np.nan, np.nan])
        for driver, val in zip(DRIVER_ORDER, vals):
            rows.append({"state": state, "driver": driver, "contribution": val})
    return pd.DataFrame(rows)


df = load_data()
pd.DataFrame([
    {
        "ecosystem": eco,
        "events_q_ge_0.01": len(df[df["ecosystem"].eq(eco)]),
        "sites": df.loc[df["ecosystem"].eq(eco), "SITE_ID"].nunique(),
        "events_q_gt_3_retained": int((df.loc[df["ecosystem"].eq(eco), "q_q"] > 3).sum()),
        "fraction_q_gt_3_retained": float((df.loc[df["ecosystem"].eq(eco), "q_q"] > 3).mean()),
    }
    for eco in ECOS
]).to_csv(SOURCE_DIR / "fig2_unbounded_q_data_audit.csv", index=False)
empirical_point_tables = []
empirical_site_bin_tables = []
curve_tables = []

fig = plt.figure(figsize=(10.6, 6.05))
gs = fig.add_gridspec(
    2,
    3,
    left=0.075,
    right=0.99,
    top=0.88,
    bottom=0.12,
    wspace=0.23,
    hspace=0.43,
    height_ratios=[1.15, 1.0],
)
top_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

for i, (ax, eco) in enumerate(zip(top_axes, ECOS)):
    sub = df[df["ecosystem"].eq(eco)].copy()
    ai_lim = robust_limits(sub, "AI_60d", qhi=0.90 if sub["SITE_ID"].nunique() < 30 else 0.95)
    lai_lim = robust_limits(sub, "LAI")
    hull = make_support(sub, ai_lim, lai_lim)
    lai_levels = sub["LAI"].quantile([0.25, 0.50, 0.75]).to_dict()
    ai_grid = np.linspace(ai_lim[0], ai_lim[1], 180)

    for key, q in zip(LINE_ORDER, [0.25, 0.50, 0.75]):
        y, lo, hi = bootstrap_predictions(sub, ai_grid, lai_levels[q], ai_lim, lai_lim, hull)
        curve_tables.append(pd.DataFrame({
            "ecosystem": eco,
            "lai_level": key,
            "lai_value": float(lai_levels[q]),
            "AI60": ai_grid,
            "predicted_q": y,
            "ci_low": lo,
            "ci_high": hi,
        }))
        ax.fill_between(ai_grid, lo, hi, where=np.isfinite(lo) & np.isfinite(hi), color=LINE_COLORS[key], alpha=0.12, lw=0)
        ax.plot(ai_grid, y, color=LINE_COLORS[key], lw=2.2, solid_capstyle="round")
        points, site_bins = empirical_bin_points(
            sub, q, ai_lim, seed=20261116 + i * 100 + int(q * 100)
        )
        points["ecosystem"] = eco
        points["lai_level"] = key
        site_bins["ecosystem"] = eco
        site_bins["lai_level"] = key
        empirical_point_tables.append(points)
        empirical_site_bin_tables.append(site_bins)
        errors = np.vstack([
            points["q"].to_numpy(float) - points["ci_low"].to_numpy(float),
            points["ci_high"].to_numpy(float) - points["q"].to_numpy(float),
        ])
        ax.errorbar(
            points["AI60"], points["q"], yerr=errors, fmt="o",
            color=LINE_COLORS[key], ecolor=LINE_COLORS[key], ms=4.0,
            lw=0.9, capsize=2.0, markeredgecolor="white",
            markeredgewidth=0.5, zorder=5,
        )

    add_site_rug(ax, sub, ai_lim)
    add_state_labels(ax, ai_lim)

    ax.set_xlim(*ai_lim)
    ax.set_ylim(0, 10.05)
    ax.set_title(ECO_LABELS[eco], fontsize=11.2, fontweight="bold", pad=7)
    ax.set_xlabel(r"$AI_{60}$ (P/PET, preceding 60 d)", fontsize=9.5)
    if i == 0:
        ax.set_ylabel("Predicted q$_{original}$", fontsize=9.5)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(axis="both", labelsize=8.2, length=3)
    boxed_axes_no_grid(ax)
    if i == 1:
        ax.legend(
            handles=[Line2D([0], [0], color=LINE_COLORS[key], lw=2.4, label=LINE_LABELS[key]) for key in LINE_ORDER],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=1,
            frameon=False,
            fontsize=7.0,
            handlelength=2.2,
            borderaxespad=0.0,
        )
    panel_label(ax, "abc"[i])

pd.concat(empirical_point_tables, ignore_index=True).to_csv(
    SOURCE_DIR / "fig2_empirical_bin_points.csv", index=False
)
pd.concat(empirical_site_bin_tables, ignore_index=True).to_csv(
    SOURCE_DIR / "fig2_site_bin_source.csv", index=False
)
pd.concat(curve_tables, ignore_index=True).to_csv(
    SOURCE_DIR / "fig2_model_curves_with_site_bootstrap_ci.csv", index=False
)

for i, (ax, eco) in enumerate(zip(bottom_axes, ECOS)):
    sub = df[df["ecosystem"].eq(eco)].copy()
    contrib = contribution_by_state(sub)
    x = np.arange(4)
    bottom = np.zeros(4)
    dominant = (
        contrib.pivot(index="state", columns="driver", values="contribution")
        .reindex(index=range(4), columns=DRIVER_ORDER)
        .idxmax(axis=1)
    )
    for driver in DRIVER_ORDER:
        vals = (
            contrib[contrib["driver"].eq(driver)]
            .sort_values("state")["contribution"]
            .to_numpy(float)
        )
        bars = ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.72,
            color=DRIVER_COLORS[driver],
            edgecolor="white",
            linewidth=0.9,
            label=driver,
        )
        for xi, val, btm in zip(x, vals, bottom):
            if np.isfinite(val) and val >= 18:
                ax.text(
                    xi,
                    btm + val / 2,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="white" if driver == "AI60 x LAI" else "black",
                )
        bottom += np.nan_to_num(vals, nan=0.0)
    for xi, drv in dominant.items():
        if isinstance(drv, str):
            display_label = DRIVER_LABELS[drv]
            if drv == "AI60 x LAI":
                display_label = r"$AI_{60}$" + "\n" + r"$\times LAI$"
            ax.text(
                xi, 104, display_label,
                ha="center", va="bottom", fontsize=6.0, color="#333333"
            )
    ax.set_ylim(0, 116)
    ax.set_xlim(-0.55, 3.55)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS, fontsize=6.8)
    ax.set_ylabel("Relative contribution (%)" if i == 0 else "", fontsize=9.2)
    ax.set_yticks([0, 25, 50, 75, 100])
    if i != 0:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", labelsize=8.0, length=3)
    boxed_axes_no_grid(ax)
    panel_label(ax, "def"[i])

legend_handles = [
    Line2D([0], [0], color=LINE_COLORS[key], lw=2.4, label=LINE_LABELS[key])
    for key in LINE_ORDER
]
driver_handles = [
    Line2D([0], [0], color=DRIVER_COLORS[d], lw=6, label=DRIVER_LABELS[d])
    for d in DRIVER_ORDER
]
fig.legend(
    handles=driver_handles,
    loc="lower center",
    bbox_to_anchor=(0.52, 0.015),
    ncol=3,
    frameon=False,
    fontsize=7.4,
    handlelength=1.8,
    columnspacing=1.8,
)

fig.canvas.draw()
audit_dir = os.environ.get("NATURE_FIGURE_AUDIT_DIR")
if audit_dir:
    sys.path.insert(0, audit_dir)
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(SOURCE_DIR / "fig2_unbounded_q.alignment.json"),
        overlay_svg=str(SOURCE_DIR / "fig2_unbounded_q.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )

fig.savefig(OUTBASE.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUTBASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
fig.savefig(OUTBASE.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUTBASE.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(SUBMISSION_PNG, dpi=600, bbox_inches="tight")

print(OUTBASE.with_suffix(".png"))
print(OUTBASE.with_suffix(".pdf"))
print(OUTBASE.with_suffix(".svg"))
print(SUBMISSION_PNG)
