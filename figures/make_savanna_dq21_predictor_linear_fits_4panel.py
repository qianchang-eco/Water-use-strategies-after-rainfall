import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
DATA = ROOT / "rewet_q_results_time_pet.csv"
OUTDIR = ROOT / "plots"
OUTBASE = OUTDIR / "savanna_dq21_predictor_linear_fits_4panel"

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

PANEL_META = [
    ("Drydown rate (m$^3$ m$^{-3}$ d$^{-1}$)", "drydown_rate", "Drydown rate\n(m$^3$ m$^{-3}$ d$^{-1}$)"),
    ("Moisture buffering (m$^3$ m$^{-3}$)", "state_buffer", "moisture buffering\n(m$^3$ m$^{-3}$)"),
    (r"$PET_{90}$", "PET90", r"$PET_{90}$"),
    ("q$_{original}$", "q_original", r"$q_{\mathrm{original}}$"),
]


def boxed_axes_no_grid(ax):
    ax.grid(False)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(top=False, right=False)


def panel_label(ax, letter):
    ax.text(-0.11, 1.03, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom", color="black")


def fmt_p(p):
    if not np.isfinite(p):
        return "p=n.a."
    if p == 0:
        return "p=0.0e+00"
    if p < 0.001:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def fit_linear_stats(x, y):
    """Fit OLS directly from the current data and return plotting statistics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.ptp(x) == 0:
        raise ValueError("Insufficient variation for linear regression.")
    xbar = x.mean()
    ybar = y.mean()
    sxx = np.sum((x - xbar) ** 2)
    slope = np.sum((x - xbar) * (y - ybar)) / sxx
    intercept = ybar - slope * xbar
    fitted = intercept + slope * x
    residuals = y - fitted
    sse = np.sum(residuals ** 2)
    sst = np.sum((y - ybar) ** 2)
    r = np.corrcoef(x, y)[0, 1]
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    se_slope = np.sqrt((sse / max(len(x) - 2, 1)) / sxx)
    t_stat = slope / se_slope if se_slope > 0 else np.inf
    p_value = 2.0 * student_t.sf(abs(t_stat), df=max(len(x) - 2, 1)) if np.isfinite(t_stat) else 0.0
    return {
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "R2": r2,
        "p_value": p_value,
        "n_fit": len(x),
    }


def regression_mean_ci(x, y, xline, slope, intercept):
    """95% confidence band for the fitted mean response, not a prediction interval."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = len(x)
    if n < 3:
        return np.full_like(xline, np.nan), np.full_like(xline, np.nan)
    xbar = np.mean(x)
    sxx = np.sum((x - xbar) ** 2)
    if sxx <= 0:
        return np.full_like(xline, np.nan), np.full_like(xline, np.nan)
    residuals = y - (intercept + slope * x)
    mse = np.sum(residuals ** 2) / max(n - 2, 1)
    se_mean = np.sqrt(mse * (1.0 / n + (xline - xbar) ** 2 / sxx))
    # With the very large event-level sample sizes here, t_0.975 is effectively 1.96.
    margin = 1.96 * se_mean
    yhat = intercept + slope * xline
    return yhat - margin, yhat + margin


def site_cluster_bootstrap_ci(x, y, sites, xline, slope, intercept, n_boot=1000, seed=20260722):
    """95% site-cluster bootstrap band centered on the full-data regression line."""
    work = pd.DataFrame({"x": x, "y": y, "site": sites}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 3:
        return np.full_like(xline, np.nan), np.full_like(xline, np.nan)
    work["site"] = work["site"].astype(str)
    site_codes, _ = pd.factorize(work["site"])
    n_sites = int(site_codes.max()) + 1
    if n_sites < 2:
        return regression_mean_ci(work["x"], work["y"], xline, 0.0, work["y"].mean())

    xvals = work["x"].to_numpy(float)
    yvals = work["y"].to_numpy(float)
    n_by_site = np.bincount(site_codes, minlength=n_sites).astype(float)
    sx = np.bincount(site_codes, weights=xvals, minlength=n_sites)
    sy = np.bincount(site_codes, weights=yvals, minlength=n_sites)
    sxx = np.bincount(site_codes, weights=xvals * xvals, minlength=n_sites)
    sxy = np.bincount(site_codes, weights=xvals * yvals, minlength=n_sites)

    rng = np.random.default_rng(seed)
    predictions = []
    for _ in range(n_boot):
        sampled = rng.integers(0, n_sites, size=n_sites)
        counts = np.bincount(sampled, minlength=n_sites).astype(float)
        n = np.sum(counts * n_by_site)
        sum_x = np.sum(counts * sx)
        sum_y = np.sum(counts * sy)
        sum_xx = np.sum(counts * sxx)
        sum_xy = np.sum(counts * sxy)
        denominator = n * sum_xx - sum_x * sum_x
        if denominator <= 0:
            continue
        boot_slope = (n * sum_xy - sum_x * sum_y) / denominator
        boot_intercept = (sum_y - boot_slope * sum_x) / n
        predictions.append(boot_intercept + boot_slope * xline)
    if len(predictions) < 50:
        return np.full_like(xline, np.nan), np.full_like(xline, np.nan)
    arr = np.vstack(predictions)
    full_prediction = intercept + slope * xline
    deviations = arr - full_prediction[None, :]
    # Use the site-bootstrap standard error around the full-data line. This
    # keeps the fitted line as the centre of the band and limits the influence
    # of a small number of extreme site-level bootstrap draws.
    half_width = 1.96 * np.nanstd(deviations, axis=0, ddof=1)
    return full_prediction - half_width, full_prediction + half_width


def read_savanna():
    cols = [
        "ecosystem", "SITE_ID", "q_original", "q_21", "q_21_start", "q_21_end",
        "q_21_theta_peak", "q_21_theta_jump", "pet_21",
    ]
    frames = []
    for chunk in pd.read_csv(DATA, usecols=cols, chunksize=450_000, low_memory=False, on_bad_lines="skip"):
        chunk["ecosystem"] = chunk["ecosystem"].astype(str).str.lower()
        chunk = chunk[chunk["ecosystem"].eq("savanna")].copy()
        if chunk.empty:
            continue
        for col in ["q_original", "q_21", "q_21_theta_peak", "q_21_theta_jump", "pet_21"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk["duration"] = (
            pd.to_datetime(chunk["q_21_end"], errors="coerce")
            - pd.to_datetime(chunk["q_21_start"], errors="coerce")
        ).dt.days.abs().clip(lower=1)
        chunk["drydown_rate"] = chunk["q_21_theta_jump"] / chunk["duration"]
        # The earlier working table defines state buffer as a theta-star-relative buffer.
        # The plotting source available here preserves its visual proxy as jump minus peak.
        chunk["state_buffer"] = chunk["q_21_theta_jump"] - chunk["q_21_theta_peak"]
        chunk["PET90"] = chunk["pet_21"]
        chunk["dq"] = (chunk["q_21"] - chunk["q_original"]).clip(-3, 3)
        frames.append(chunk[["SITE_ID", "q_original", "dq", "drydown_rate", "state_buffer", "PET90"]])
    if not frames:
        raise RuntimeError("No savanna rows found.")
    return pd.concat(frames, ignore_index=True)


df = read_savanna()

fig, axes = plt.subplots(2, 2, figsize=(8.7, 6.55), sharey=True)
plt.subplots_adjust(left=0.085, right=0.985, top=0.97, bottom=0.10, wspace=0.22, hspace=0.30)

rng = np.random.default_rng(20260629)
for i, (ax, (label, col, xlabel)) in enumerate(zip(axes.flat, PANEL_META)):
    sub = df[["SITE_ID", "dq", col]].replace([np.inf, -np.inf], np.nan).dropna()
    xmin = float(sub[col].quantile(0.003))
    xmax = float(sub[col].quantile(0.997))
    if col == "q_original":
        xmin, xmax = 1.0, 2.4
    fit_sub = sub[sub[col].between(xmin, xmax) & sub["dq"].between(-3, 3)].copy()
    st = fit_linear_stats(fit_sub[col], fit_sub["dq"])
    plot_sub = fit_sub
    if len(plot_sub) > 1200:
        plot_sub = plot_sub.sample(1200, random_state=20260629 + i)
    # Draw a faint background layer plus a clearer foreground sample.
    ax.scatter(plot_sub[col], plot_sub["dq"], s=5.5, color="#2aa7a0", alpha=0.20, linewidths=0, rasterized=True)
    if len(plot_sub) > 450:
        clear = plot_sub.sample(450, random_state=20260629 + i * 11)
    else:
        clear = plot_sub
    ax.scatter(clear[col], clear["dq"], s=6.5, color="#149a95", alpha=0.55, linewidths=0, rasterized=True)
    xline = np.linspace(xmin, xmax, 120)
    yline = st["intercept"] + st["slope"] * xline
    lo, hi = site_cluster_bootstrap_ci(
        sub[col].to_numpy(),
        sub["dq"].to_numpy(),
        sub["SITE_ID"].to_numpy(),
        xline,
        st["slope"],
        st["intercept"],
    )
    ax.fill_between(xline, lo, hi, color="#86c39b", alpha=0.45, linewidth=0, zorder=0.8)
    ax.plot(xline, yline, color="#1c7a4f", lw=1.7, zorder=2)
    ax.axhline(0, color="#666666", lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-3.55, 3.55)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(r"$\Delta q$" if i % 2 == 0 else "", fontsize=9)
    txt = f"slope={st['slope']:.3g}\nr={st['r']:.2f}, R$^2$={st['R2']:.2f}\n{fmt_p(st['p_value'])}"
    ax.text(
        0.04,
        0.94,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#d9ded9", alpha=0.82),
    )
    ax.tick_params(labelsize=7.5, length=3)
    boxed_axes_no_grid(ax)
    panel_label(ax, "abcd"[i])

for ext in [".png", ".pdf", ".svg"]:
    fig.savefig(OUTBASE.with_suffix(ext), dpi=600 if ext == ".png" else None, bbox_inches="tight")

print(OUTBASE.with_suffix(".png"))
