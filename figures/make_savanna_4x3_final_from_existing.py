import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
TABLE_DIR = ROOT / "rewet_savanna_process" / "results" / "phase_3x2_pet_tables"
SMOOTH_DIR = TABLE_DIR / "shape_data_smooth"
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTBASE = OUTDIR / "savanna_qstate_heatmap_smooth_4x3_final"
MAINBASE = OUTBASE
SUPPBASE = OUTDIR / "savanna_qstate_heatmap_raw_2x3_supplementary"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "figure.facecolor": "white",
    }
)

segments = [
    ("S1(0.50,1.49]", r"$q_{\mathrm{original}}$ (0.50-1.49)"),
    ("S2(1.49,2.58]", r"$q_{\mathrm{original}}$ (1.49-2.58)"),
    ("S3(2.58,3.00]", r"$q_{\mathrm{original}}$ (2.58-3.00)"),
]

row_specs = [
    ("heat", "drydown_rate_21", "Drydown rate\n(m$^3$ m$^{-3}$ day$^{-1}$)"),
    ("heat", "state_buffer_first", "Moisture buffering\n(m$^3$ m$^{-3}$)"),
    ("smooth", "drydown_rate_21", "Drydown rate\n(m$^3$ m$^{-3}$ day$^{-1}$)"),
    ("smooth", "state_buffer_first", "Moisture buffering\n(m$^3$ m$^{-3}$)"),
]


def heat_file(segment, yvar):
    return TABLE_DIR / f"dq_surface_tables__{segment}__y{yvar}.csv"


def smooth_file(segment, yvar):
    return SMOOTH_DIR / f"dq_surface_tables__{segment}__y{yvar}__shape_data_smooth_grid.csv"


def clean_table(path):
    df = pd.read_csv(path)
    keep = ["x_bin", "y_bin", "x_lo", "x_hi", "y_lo", "y_hi", "dq_median", "n"]
    df = df.dropna(subset=keep).copy()
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=keep)
    df["x_bin"] = df["x_bin"].astype(int)
    df["y_bin"] = df["y_bin"].astype(int)
    return df.sort_values(["y_bin", "x_bin"])


def load_heat(segment, yvar):
    df = clean_table(heat_file(segment, yvar))
    xmeta = df.sort_values("x_bin").groupby("x_bin")[["x_lo", "x_hi"]].first()
    ymeta = df.sort_values("y_bin").groupby("y_bin")[["y_lo", "y_hi"]].first()
    x_edges = np.r_[xmeta["x_lo"].to_numpy(), xmeta["x_hi"].to_numpy()[-1]]
    y_edges = np.r_[ymeta["y_lo"].to_numpy(), ymeta["y_hi"].to_numpy()[-1]]
    z = df.pivot_table(index="y_bin", columns="x_bin", values="dq_median", aggfunc="first").to_numpy(float)
    n = df.pivot_table(index="y_bin", columns="x_bin", values="n", aggfunc="first").to_numpy(float)
    xcent = (x_edges[:-1] + x_edges[1:]) / 2
    ycent = (y_edges[:-1] + y_edges[1:]) / 2
    return x_edges, y_edges, xcent, ycent, z, n


def load_smooth(segment, yvar):
    df = pd.read_csv(smooth_file(segment, yvar))
    for col in ["x", "y", "dq"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["x", "y", "dq"]).copy()
    x = np.sort(df["x"].unique())
    y = np.sort(df["y"].unique())
    z = df.pivot_table(index="y", columns="x", values="dq", aggfunc="first").reindex(index=y, columns=x).to_numpy(float)
    return trend_surface(x, y, z, segment, yvar)


def poly_terms(xs, ys, degree):
    terms = [np.ones_like(xs)]
    for total in range(1, degree + 1):
        for px in range(total, -1, -1):
            py = total - px
            terms.append((xs ** px) * (ys ** py))
    return np.vstack(terms).T


def trend_surface(x, y, z, segment, yvar):
    # Turn the lower two rows into a low-order response surface: this keeps the
    # expected regime-scale trend while deliberately suppressing local bin jumps.
    X, Y = np.meshgrid(x, y)
    mask = np.isfinite(z)
    x_mid, x_half = (x.min() + x.max()) / 2, (x.max() - x.min()) / 2
    y_mid, y_half = (y.min() + y.max()) / 2, (y.max() - y.min()) / 2
    xs = (X[mask] - x_mid) / x_half
    ys = (Y[mask] - y_mid) / y_half
    vals = np.clip(z[mask], -3, 3)

    degree = 2
    if segment.startswith("S3") or (segment.startswith("S2") and yvar == "state_buffer_first"):
        degree = 3

    A = poly_terms(xs, ys, degree)
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    Xs = (X - x_mid) / x_half
    Ys = (Y - y_mid) / y_half
    Zfit = poly_terms(Xs.ravel(), Ys.ravel(), degree).dot(coef).reshape(X.shape)
    return x, y, np.clip(Zfit, -3, 3)


def robust_norm(kind, segment, yvar):
    if kind == "smooth":
        return -3.0, 3.0
    if kind == "heat":
        *_unused, z, _n = load_heat(segment, yvar)
    else:
        *_unused, z = load_smooth(segment, yvar)
    vals = z[np.isfinite(z)]
    lo, hi = np.nanpercentile(vals, [1, 99])
    return float(lo), float(hi)


def annotate_heat(ax, xcent, ycent, z):
    ny, nx = z.shape
    # Preserve readability: annotate every cell for sparse S1, alternate cells for dense S2/S3.
    x_step = 1 if nx <= 6 else 2
    y_step = 1 if ny <= 6 else 2
    for iy in range(0, ny, y_step):
        for ix in range(0, nx, x_step):
            val = z[iy, ix]
            if not np.isfinite(val):
                continue
            color = "white" if abs(val) >= 1.1 else "black"
            txt = f"{val:.2f}" if abs(val) < 10 else f"{val:.1f}"
            ax.text(xcent[ix], ycent[iy], txt, ha="center", va="center", fontsize=6.8, color=color)


def style_axis(ax, yvar, show_ylabel, ylabel, title=None):
    if title:
        ax.set_title(title, fontsize=12.0, fontweight="bold", pad=8)
    ax.set_xlabel(r"$PET_{90}$ (mm day$^{-1}$)", fontsize=11.5, fontweight="bold", labelpad=7)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=11.5, fontweight="bold", labelpad=8)
    ax.tick_params(axis="both", labelsize=9.5, length=4.0, width=0.8)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    if yvar == "drydown_rate_21":
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))


def representative_ticks(values, max_ticks=5):
    idx = np.linspace(0, len(values) - 1, min(max_ticks, len(values))).round().astype(int)
    return np.unique(idx)


cmap = LinearSegmentedColormap.from_list(
    "green_teal_diverging_dq",
    ["#245764", "#5fa49a", "#f5f5ee", "#b4ca4a", "#c99f32"],
    N=256,
)
norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)


def draw_panel(fig, ax, kind, segment, yvar, ylabel, letter, col, title, annotate=False):
    if kind == "heat":
        x_edges, y_edges, xcent, ycent, z, n = load_heat(segment, yvar)
        ny, nx = z.shape
        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            extent=[-0.5, nx - 0.5, -0.5, ny - 0.5],
            rasterized=True,
        )
        if annotate:
            annotate_heat(ax, np.arange(nx), np.arange(ny), z)
    else:
        x, y, z = load_smooth(segment, yvar)
        X, Y = np.meshgrid(x, y)
        im = ax.imshow(
            z,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="bicubic",
            rasterized=True,
        )
        contour_levels = np.array([-2, -1, 0, 1, 2], dtype=float)
        contour_levels = contour_levels[(contour_levels > np.nanmin(z)) & (contour_levels < np.nanmax(z))]
        if contour_levels.size:
            cs = ax.contour(X, Y, z, levels=contour_levels, colors="black", linewidths=0.8, alpha=0.58)
            ax.clabel(cs, inline=True, fontsize=8.0, fmt="%g", colors="black")

    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", ha="left", va="bottom", color="black")
    panel_title = title if col in (0, 1, 2) else None
    style_axis(ax, yvar, col == 0, ylabel, panel_title)
    if kind == "heat":
        xt = representative_ticks(xcent, 5)
        yt = representative_ticks(ycent, 5)
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{xcent[i]:.2f}" for i in xt], fontsize=9.0)
        ax.set_yticks(yt)
        if yvar == "drydown_rate_21":
            ax.set_yticklabels([f"{ycent[i]:.3f}" for i in yt], fontsize=9.0)
        else:
            ax.set_yticklabels([f"{ycent[i]:.2f}" for i in yt], fontsize=9.0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.8%", pad=0.10)
    cb = fig.colorbar(im, cax=cax, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb.ax.tick_params(labelsize=8.5, length=3.0, width=0.7)
    cb.outline.set_linewidth(0.55)
    if col == 2:
        label = r"$\Delta q$" if kind == "heat" else r"Predicted $\Delta q$"
        cb.set_label(label, fontsize=9.5, rotation=270, labelpad=14)


def save_figure(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def calculate_surface_fit_statistics():
    """Quantify how much of the observed binned pattern is retained by each surface."""
    rows = []
    for segment, _title in segments:
        for _kind, yvar, _ylabel in row_specs[:2]:
            raw = clean_table(heat_file(segment, yvar))
            raw["x"] = (raw["x_lo"] + raw["x_hi"]) / 2.0
            raw["y"] = (raw["y_lo"] + raw["y_hi"]) / 2.0
            if raw.empty:
                continue

            x = raw["x"].to_numpy(float)
            y = raw["y"].to_numpy(float)
            observed = np.clip(raw["dq_median"].to_numpy(float), -3.0, 3.0)
            x_mid, x_half = (x.min() + x.max()) / 2.0, (x.max() - x.min()) / 2.0
            y_mid, y_half = (y.min() + y.max()) / 2.0, (y.max() - y.min()) / 2.0
            # Use the same polynomial order as the smoothed response-surface panels.
            degree = 2
            if segment.startswith("S3") or (segment.startswith("S2") and yvar == "ystate_buffer_first"):
                degree = 3

            design = poly_terms((x - x_mid) / x_half, (y - y_mid) / y_half, degree)
            coefficients, *_unused = np.linalg.lstsq(design, observed, rcond=None)
            predicted = design @ coefficients
            residual_ss = np.sum((observed - predicted) ** 2)
            total_ss = np.sum((observed - observed.mean()) ** 2)
            r2 = np.nan if total_ss == 0 else 1.0 - residual_ss / total_ss

            weights = raw["n"].to_numpy(float)
            weighted_mean = np.average(observed, weights=weights)
            weighted_total_ss = np.sum(weights * (observed - weighted_mean) ** 2)
            weighted_r2 = (
                np.nan
                if weighted_total_ss == 0
                else 1.0 - np.sum(weights * (observed - predicted) ** 2) / weighted_total_ss
            )
            rows.append(
                {
                    "qoriginal_range": segment,
                    "predictor_surface": yvar,
                    "polynomial_degree": degree,
                    "observed_bins": len(raw),
                    "fit_R2_unweighted": r2,
                    "fit_R2_event_weighted": weighted_r2,
                    "RMSE": np.sqrt(np.mean((observed - predicted) ** 2)),
                }
            )

    stats_path = OUTDIR / "savanna_qstate_heatmap_surface_fit_statistics.csv"
    stats = pd.DataFrame(rows)
    stats.to_csv(stats_path, index=False, float_format="%.6f")
    return stats


def title_with_r2(segment, yvar, stats):
    """Append the panel-specific response-surface fit R2 to the state title."""
    base = dict(segments)[segment]
    match = stats[
        (stats["qoriginal_range"] == segment)
        & (stats["predictor_surface"] == yvar)
    ]
    if match.empty:
        return base
    r2 = float(match.iloc[0]["fit_R2_unweighted"])
    return f"{base}, $R^2$ = {r2:.2f}"


# Calculate statistics before drawing so the same values are used in the panel titles.
surface_stats = calculate_surface_fit_statistics()


# Main-text figure: smoothed response surfaces only, arranged as 2 x 3.
fig_main, axes_main = plt.subplots(2, 3, figsize=(15.2, 8.8), squeeze=False)
plt.subplots_adjust(left=0.070, right=0.972, top=0.905, bottom=0.105, wspace=0.36, hspace=0.42)
smooth_specs = [
    ("smooth", "drydown_rate_21", row_specs[0][2]),
    ("smooth", "state_buffer_first", row_specs[1][2]),
]
for row, (kind, yvar, ylabel) in enumerate(smooth_specs):
    for col, (segment, title) in enumerate(segments):
        draw_panel(
            fig_main,
            axes_main[row, col],
            kind,
            segment,
            yvar,
            ylabel,
            "abcdef"[row * 3 + col],
            col,
            title_with_r2(segment, yvar, surface_stats),
        )
save_figure(fig_main, MAINBASE)

# Supplementary figure: observed binned surfaces, arranged as 2 x 3.
fig_supp, axes_supp = plt.subplots(2, 3, figsize=(15.2, 8.8), squeeze=False)
plt.subplots_adjust(left=0.070, right=0.972, top=0.905, bottom=0.105, wspace=0.36, hspace=0.42)
heat_specs = [
    ("heat", "drydown_rate_21", row_specs[0][2]),
    ("heat", "state_buffer_first", row_specs[1][2]),
]
for row, (kind, yvar, ylabel) in enumerate(heat_specs):
    for col, (segment, title) in enumerate(segments):
        draw_panel(fig_supp, axes_supp[row, col], kind, segment, yvar, ylabel, "abcdef"[row * 3 + col], col, title, annotate=True)
save_figure(fig_supp, SUPPBASE)

print(MAINBASE.with_suffix(".png"))
print(SUPPBASE.with_suffix(".png"))
print(OUTDIR / "savanna_qstate_heatmap_surface_fit_statistics.csv")
