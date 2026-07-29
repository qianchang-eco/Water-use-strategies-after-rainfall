import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import kruskal


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
REWET = ROOT / "rewet_q_results_time_pet.csv"
QFIT = ROOT / "q_results_with_ecosystem_precip_swc_latlon_pet_ai.csv"
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTBASE = OUTDIR / "extended_data_fig4_robustness_event_definition"

ECOS = ["forest", "savanna", "grassland"]
ECO_LABELS = {"forest": "Forest", "savanna": "Savanna", "grassland": "Grassland"}
ECO_COLORS = {"forest": "#0b6b57", "savanna": "#26a69a", "grassland": "#a9c83f"}
PREDICTORS = ["drydown rate", "state buffer", "PET"]
PRED_COLORS = {"drydown rate": "#0b6b57", "state buffer": "#38aa87", "PET": "#b7cf3d"}
RNG = np.random.default_rng(20260626)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
    }
)


def panel_label(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold")


def boxed_axes_no_grid(ax):
    ax.grid(False)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(top=False, right=False)


def read_qfit_r2():
    q = pd.read_csv(QFIT, usecols=["file", "event_id", "r2"], low_memory=False)
    q["event_id"] = pd.to_numeric(q["event_id"], errors="coerce")
    q["r2"] = pd.to_numeric(q["r2"], errors="coerce")
    return q.dropna(subset=["file", "event_id"]).drop_duplicates(["file", "event_id"])


def read_rewet_long():
    usecols = [
        "file",
        "event_id",
        "ecosystem",
        "q_original",
        "q_7",
        "q_14",
        "q_21",
        "q_7_start",
        "q_7_end",
        "q_7_theta_peak",
        "q_7_theta_end",
        "q_7_theta_jump",
        "q_14_start",
        "q_14_end",
        "q_14_theta_peak",
        "q_14_theta_end",
        "q_14_theta_jump",
        "q_21_start",
        "q_21_end",
        "q_21_theta_peak",
        "q_21_theta_end",
        "q_21_theta_jump",
        "SITE_ID",
        "pet_7",
        "pet_14",
        "pet_21",
    ]
    df = pd.read_csv(REWET, usecols=usecols, low_memory=False)
    df["ecosystem"] = df["ecosystem"].astype(str).str.lower()
    df = df[df["ecosystem"].isin(ECOS)].copy()
    date_cols = {"q_7_start", "q_7_end", "q_14_start", "q_14_end", "q_21_start", "q_21_end"}
    for col in df.columns:
        if col not in ["file", "ecosystem", "SITE_ID"] and col not in date_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    r2 = read_qfit_r2()
    df = df.merge(r2, on=["file", "event_id"], how="left")
    rows = []
    for w in [7, 14, 21]:
        q = f"q_{w}"
        start = f"q_{w}_start"
        end = f"q_{w}_end"
        peak = f"q_{w}_theta_peak"
        theta_end = f"q_{w}_theta_end"
        jump = f"q_{w}_theta_jump"
        pet = f"pet_{w}"
        tmp = df[
            [
                "file",
                "event_id",
                "SITE_ID",
                "ecosystem",
                "q_original",
                q,
                start,
                end,
                peak,
                theta_end,
                jump,
                pet,
                "r2",
            ]
        ].copy()
        tmp.columns = [
            "file",
            "event_id",
            "SITE_ID",
            "ecosystem",
            "q_original",
            "q_prime",
            "start",
            "end",
            "theta_peak",
            "theta_end",
            "theta_jump",
            "PET",
            "r2",
        ]
        tmp["window"] = w
        rows.append(tmp)
    long = pd.concat(rows, ignore_index=True)
    long = long.dropna(subset=["q_original", "q_prime", "SITE_ID", "ecosystem", "theta_jump"])
    long["dq"] = long["q_prime"] - long["q_original"]
    long["start_date"] = pd.to_datetime(long["start"], errors="coerce")
    long["end_date"] = pd.to_datetime(long["end"], errors="coerce")
    long["duration"] = (long["end_date"] - long["start_date"]).dt.days.abs().clip(lower=1)
    long["drydown rate"] = (long["theta_jump"] / long["duration"]).replace([np.inf, -np.inf], np.nan)
    long["state buffer"] = long["theta_peak"].replace([np.inf, -np.inf], np.nan)
    long = long.dropna(subset=["dq"])
    return long


def scenario_table(long):
    q21 = long[long["window"].eq(21)]
    q7 = long[long["window"].eq(7)]
    q14 = long[long["window"].eq(14)]
    q = q21["theta_jump"].quantile([0.10, 0.25, 0.50]).to_dict()
    return pd.DataFrame(
        [
            ("Baseline", 21, q[0.25], 3, 0.80, False, "reference"),
            ("7d window", 7, q7["theta_jump"].quantile(0.25), 3, 0.80, False, "window"),
            ("14d window", 14, q14["theta_jump"].quantile(0.25), 3, 0.80, False, "window"),
            ("Loose pulse", 21, q[0.10], 3, 0.80, False, "wetting threshold"),
            ("Strict pulse", 21, q[0.50], 3, 0.80, False, "wetting threshold"),
            ("Short redry", 21, q[0.25], 2, 0.80, False, "redry duration"),
            ("Long redry", 21, q[0.25], 7, 0.80, False, "redry duration"),
            ("Loose q fit", 21, q[0.25], 3, 0.70, False, "q fit"),
            ("Strict q fit", 21, q[0.25], 3, 0.90, False, "q fit"),
            ("Exclude edge q", 21, q[0.25], 3, 0.80, True, "q boundary"),
        ],
        columns=["scenario", "window", "jump_min", "duration_min", "r2_min", "exclude_q_bounds", "changed_condition"],
    )


def apply_scenario(long, row):
    if not isinstance(row, pd.Series):
        row = pd.Series(row._asdict())
    d = long[long["window"].eq(int(row["window"]))].copy()
    jump_min = row["jump_min"]
    if np.isnan(jump_min):
        jump_min = d["theta_jump"].quantile(0.25)
    d = d[d["theta_jump"].ge(jump_min)]
    d = d[d["duration"].ge(row["duration_min"])]
    d = d[d["r2"].fillna(1).ge(row["r2_min"])]
    if row["exclude_q_bounds"]:
        d = d[d["q_original"].between(0.05, 2.95) & d["q_prime"].between(0.05, 2.95)]
    return d


def site_boot_median(d, n_boot=120):
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    med = float(d["dq"].median())
    sites = d["SITE_ID"].drop_duplicates().to_numpy()
    if len(sites) < 2:
        return med, np.nan, np.nan
    vals = []
    site_groups = {s: g["dq"].to_numpy() for s, g in d.groupby("SITE_ID", observed=True)}
    for _ in range(n_boot):
        sampled = RNG.choice(sites, size=len(sites), replace=True)
        vals.append(np.median(np.concatenate([site_groups[s] for s in sampled])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return med, lo, hi


def site_balanced_coef(d, predictors):
    use = d.dropna(subset=["dq"] + predictors + ["SITE_ID"]).copy()
    if len(use) < 50 or use["SITE_ID"].nunique() < 3:
        return np.full(len(predictors), np.nan)
    y = use["dq"].clip(-3, 3).to_numpy(float)
    y = (y - y.mean()) / (y.std(ddof=0) or 1)
    x = use[predictors].to_numpy(float)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0, ddof=0) == 0, 1, x.std(axis=0, ddof=0))
    x = np.column_stack([np.ones(len(x)), x])
    site_counts = use["SITE_ID"].map(use["SITE_ID"].value_counts()).to_numpy(float)
    w = 1 / site_counts
    w = w / w.mean()
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)[0]
    return beta[1:]


def thin_by_site(d, max_per_site=18):
    if len(d) == 0:
        return d
    pieces = []
    for _, g in d.groupby("SITE_ID", observed=True):
        if len(g) > max_per_site:
            pieces.append(g.sample(max_per_site, random_state=20260626))
        else:
            pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def coef_boot(d, predictors, n_boot=60):
    d = thin_by_site(d.dropna(subset=predictors + ["dq", "SITE_ID"]), max_per_site=18)
    base = site_balanced_coef(d, predictors)
    sites = d["SITE_ID"].drop_duplicates().to_numpy()
    if len(sites) < 3:
        return base, np.full_like(base, np.nan), np.full_like(base, np.nan)
    boots = []
    groups = {s: g for s, g in d.groupby("SITE_ID", observed=True)}
    for _ in range(n_boot):
        sampled = RNG.choice(sites, size=len(sites), replace=True)
        dd = pd.concat([groups[s] for s in sampled], ignore_index=True)
        boots.append(site_balanced_coef(dd, predictors))
    arr = np.vstack(boots)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
    return base, lo, hi


long = read_rewet_long()
scens = scenario_table(long)
scenario_data = {r.scenario: apply_scenario(long, r) for r in scens.itertuples(index=False)}

summary_rows = []
for scen, d in scenario_data.items():
    for eco in ECOS:
        de = d[d["ecosystem"].eq(eco)]
        med, lo, hi = site_boot_median(de)
        summary_rows.append(
            {
                "scenario": scen,
                "ecosystem": eco,
                "N": len(de),
                "negative_fraction": float((de["dq"] < 0).mean()) if len(de) else np.nan,
                "median_dq": med,
                "median_dq_lo": lo,
                "median_dq_hi": hi,
            }
        )
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTDIR / "extended_data_fig4_robustness_summary.csv", index=False)
scens.to_csv(OUTDIR / "extended_data_fig4_sensitivity_scenarios.csv", index=False)

fig = plt.figure(figsize=(11.2, 6.8))
gs = fig.add_gridspec(2, 3, left=0.06, right=0.982, top=0.93, bottom=0.11, wspace=0.38, hspace=0.42)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

# a. Scenario matrix with explicit values
ax = axes[0]
conditions = ["Window", "Pulse min", "Redry min", "q-fit R2", "q-boundary"]
base = scens.iloc[0]
ax.set_xlim(0, len(conditions))
ax.set_ylim(0, len(scens) + 1)
ax.invert_yaxis()
for j, cond in enumerate(conditions):
    ax.text(j + 0.5, 0.5, cond, ha="center", va="center", fontsize=6.7, fontweight="bold")
for i, r in scens.iterrows():
    y = i + 1.5
    vals = [
        f"{int(r['window'])}d",
        f">={r['jump_min']:.3f}",
        f">={int(r['duration_min'])}d",
        f">={r['r2_min']:.2f}",
        "edge q out" if r["exclude_q_bounds"] else "retain",
    ]
    base_vals = [
        f"{int(base['window'])}d",
        f">={base['jump_min']:.3f}",
        f">={int(base['duration_min'])}d",
        f">={base['r2_min']:.2f}",
        "edge q out" if base["exclude_q_bounds"] else "retain",
    ]
    ax.text(-0.08, y, r["scenario"], ha="right", va="center", fontsize=7.0)
    for j, val in enumerate(vals):
        changed = val != base_vals[j]
        ax.add_patch(
            plt.Rectangle(
                (j, y - 0.43),
                1,
                0.86,
                facecolor="#d9ece5" if changed else "#f7f7f2",
                edgecolor="white",
                lw=0.7,
            )
        )
        ax.text(j + 0.5, y, val, ha="center", va="center", fontsize=5.7, color="black")
for j in range(len(conditions) + 1):
    ax.axvline(j, color="white", lw=0.7)
for y in np.arange(1, len(scens) + 2):
    ax.axhline(y, color="white", lw=0.7)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Sensitivity scenarios", fontsize=9, fontweight="bold")
panel_label(ax, "a")

# b. Retained event count
ax = axes[1]
base_n = summary[summary["scenario"].eq("Baseline")].set_index("ecosystem")["N"]
x = np.arange(len(scens))
offsets = [-0.22, 0, 0.22]
for off, eco in zip(offsets, ECOS):
    vals = []
    for scen in scens["scenario"]:
        n = summary[(summary["scenario"].eq(scen)) & (summary["ecosystem"].eq(eco))]["N"].iloc[0]
        vals.append(100 * n / base_n[eco] if base_n[eco] else np.nan)
    ax.bar(x + off, vals, width=0.20, color=ECO_COLORS[eco], label=ECO_LABELS[eco], alpha=0.9)
ax.axhline(100, color="#666666", lw=0.8, ls=(0, (3, 2)))
ax.set_ylabel("Events retained (% of baseline)")
ax.set_xticks(x)
ax.set_xticklabels(scens["scenario"], rotation=45, ha="right")
ax.set_ylim(0, max(140, ax.get_ylim()[1]))
boxed_axes_no_grid(ax)
ax.legend(frameon=False, fontsize=6.8, ncol=3, loc="upper left")
panel_label(ax, "b")

# c. Median dq
ax = axes[2]
for off, eco in zip(offsets, ECOS):
    de = summary[summary["ecosystem"].eq(eco)].set_index("scenario").loc[scens["scenario"]]
    ax.errorbar(
        x + off,
        de["median_dq"],
        yerr=[de["median_dq"] - de["median_dq_lo"], de["median_dq_hi"] - de["median_dq"]],
        fmt="o",
        ms=3.8,
        lw=1.0,
        capsize=2,
        color=ECO_COLORS[eco],
        label=ECO_LABELS[eco],
    )
ax.axhline(0, color="#666666", lw=0.8, ls=(0, (3, 2)))
ax.set_ylabel("Median dq\n(site-cluster bootstrap 95% CI)")
ax.set_xticks(x)
ax.set_xticklabels(scens["scenario"], rotation=45, ha="right")
boxed_axes_no_grid(ax)
panel_label(ax, "c")

# d. Negative fraction and ecosystem effect size
ax = axes[3]
for eco in ECOS:
    vals = summary[summary["ecosystem"].eq(eco)].set_index("scenario").loc[scens["scenario"]]["negative_fraction"].to_numpy() * 100
    ax.plot(x, vals, "-o", color=ECO_COLORS[eco], ms=3.2, lw=1.5, label=ECO_LABELS[eco])
eps = []
for scen in scens["scenario"]:
    d = scenario_data[scen]
    groups = [d[d["ecosystem"].eq(eco)]["dq"].dropna().to_numpy() for eco in ECOS]
    groups = [g for g in groups if len(g) > 5]
    if len(groups) >= 2:
        H = kruskal(*groups).statistic
        n = sum(len(g) for g in groups)
        k = len(groups)
        eps.append(max((H - k + 1) / (n - k), 0))
    else:
        eps.append(np.nan)
ax2 = ax.twinx()
ax2.plot(x, eps, color="black", lw=1.5, marker="s", ms=3, label="Kruskal-Wallis epsilon2")
ax.set_ylabel("Negative dq events (%)")
ax2.tick_params(labelsize=6.5, pad=1)
emax = np.nanmax(eps) if np.isfinite(eps).any() else 0.01
ax2.set_ylim(0, max(emax * 1.25, 0.01))
ax2.set_ylabel("")
ax.text(0.98, 0.93, "black line: epsilon2", transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color="black")
ax.set_xticks(x)
ax.set_xticklabels(scens["scenario"], rotation=45, ha="right")
ax.set_ylim(0, 100)
boxed_axes_no_grid(ax)
ax2.grid(False)
ax2.tick_params(top=False)
for side in ["top", "right"]:
    ax2.spines[side].set_visible(True)
    ax2.spines[side].set_color("black")
    ax2.spines[side].set_linewidth(0.75)
panel_label(ax, "d")

# e. Savanna process controls
ax = axes[4]
forest_rows = []
ypos = np.arange(len(scens))
preds = ["drydown rate", "state buffer", "PET"]
for si, scen in enumerate(scens["scenario"]):
    d = scenario_data[scen]
    ds = d[d["ecosystem"].eq("savanna")].dropna(subset=preds)
    base, lo, hi = coef_boot(ds, preds, n_boot=50)
    for j, pred in enumerate(preds):
        y = si + (j - 1) * 0.22
        ax.errorbar(base[j], y, xerr=[[base[j] - lo[j]], [hi[j] - base[j]]], fmt="o", color=PRED_COLORS[pred], ms=3.4, lw=1.0, capsize=1.8)
        forest_rows.append({"scenario": scen, "predictor": pred, "beta": base[j], "lo": lo[j], "hi": hi[j]})
pd.DataFrame(forest_rows).to_csv(OUTDIR / "extended_data_fig4_savanna_process_controls.csv", index=False)
ax.axvline(0, color="#666666", lw=0.8, ls=(0, (3, 2)))
ax.set_yticks(ypos)
ax.set_yticklabels(scens["scenario"])
ax.invert_yaxis()
ax.set_xlabel("Standardized effect on dq")
ax.set_title("Savanna controls\n95% site-cluster CI", fontsize=9, fontweight="bold")
boxed_axes_no_grid(ax)
handles = [Line2D([0], [0], color=PRED_COLORS[p], marker="o", lw=1.5, label=p) for p in preds]
ax.legend(handles=handles, frameon=False, fontsize=6.8, loc="lower right")
panel_label(ax, "e")

# f. State-dependent process effects
ax = axes[5]
rows = []
for scen in scens["scenario"]:
    d = scenario_data[scen].dropna(subset=["dq"] + preds + ["q_original"]).copy()
    if len(d) < 100:
        continue
    d["state"] = pd.qcut(d["q_original"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    for state in ["Q1", "Q2", "Q3", "Q4"]:
        ds = d[d["state"].eq(state)]
        base, lo, hi = coef_boot(ds, preds, n_boot=35)
        for j, pred in enumerate(preds):
            rows.append({"scenario": scen, "state": state, "predictor": pred, "beta": base[j], "lo": lo[j], "hi": hi[j]})
heat = pd.DataFrame(rows)
heat.to_csv(OUTDIR / "extended_data_fig4_state_dependent_effects.csv", index=False)
row_order = [f"{q} | {p}" for q in ["Q1", "Q2", "Q3", "Q4"] for p in preds]
mat = np.full((len(row_order), len(scens)), np.nan)
sig = np.zeros_like(mat, dtype=bool)
for _, r in heat.iterrows():
    ri = row_order.index(f"{r['state']} | {r['predictor']}")
    ci = list(scens["scenario"]).index(r["scenario"])
    mat[ri, ci] = r["beta"]
    sig[ri, ci] = (r["lo"] > 0) or (r["hi"] < 0)
hm = ax.imshow(mat, cmap=mpl.colors.LinearSegmentedColormap.from_list("coef", ["#d8c91e", "#f7f7f2", "#006b5b"]), vmin=-0.35, vmax=0.35, aspect="auto")
for ri in range(mat.shape[0]):
    for ci in range(mat.shape[1]):
        if sig[ri, ci]:
            ax.plot(ci, ri, "o", color="black", ms=2.2)
ax.set_xticks(range(len(scens)))
ax.set_xticklabels(scens["scenario"], rotation=45, ha="right")
ax.set_yticks(range(len(row_order)))
ax.set_yticklabels(row_order, fontsize=6.2)
ax.set_title("State-dependent process effects", fontsize=9, fontweight="bold")
boxed_axes_no_grid(ax)
ax.text(
    0.98,
    0.98,
    "dots: bootstrap CI excludes 0",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=5.8,
    color="black",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
)
panel_label(ax, "f")
cax = fig.add_axes([0.988, 0.13, 0.010, 0.30])
cbar = fig.colorbar(hm, cax=cax)
cbar.set_label("Standardized effect", rotation=270, labelpad=11)
cbar.ax.tick_params(labelsize=6)

for ext in [".png", ".pdf", ".svg"]:
    fig.savefig(OUTBASE.with_suffix(ext), dpi=600 if ext == ".png" else None, bbox_inches="tight")

print(OUTBASE.with_suffix(".png"))
print(OUTBASE.with_suffix(".pdf"))
print(OUTBASE.with_suffix(".svg"))
