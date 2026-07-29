import os
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
OUT = ROOT / "supplementary_tables"
OUT.mkdir(parents=True, exist_ok=True)

QORIG = ROOT / "results_paper" / "ismn_results_star_th0_04may_ecosystem.csv"
REWET = ROOT / "rewet_q_results_time_pet.csv"
PROCESS = {
    "forest": ROOT / "rewet_forest_process" / "forest_rewet_merged_firstdrought_state.csv",
    "savanna": ROOT / "rewet_savanna_process" / "savanna_rewet_merged_firstdrought_state.csv",
    "grassland": ROOT / "rewet_grassland_process" / "grassland_rewet_merged_firstdrought_state.csv",
}
FIG2_COEF = ROOT / "plots" / "fig2_quantile_regression_coefficients.csv"
FIG2_CONTRIB_SCRIPT = Path(os.environ.get("WATER_USE_FIG2_SCRIPT", "figures/make_fig2_hydroclim_veg_backgrounds.py"))
FIG4_CONTROLS = ROOT / "plots" / "extended_data_fig4_savanna_process_controls.csv"
FIG4_STATE = ROOT / "plots" / "extended_data_fig4_state_dependent_effects.csv"

ECOS = ["forest", "savanna", "grassland"]
STATE_LABELS = ["Q1_low", "Q2_midlow", "Q3_midhigh", "Q4_high"]
PRED_ENV = ["AI_60d", "LAI", "AI60_x_LAI"]
PRED_PROCESS = ["drydown_rate", "state_buffer", "PET", "q_original", "q_original_x_drydown_rate"]


def sig3(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(f"{float(x):.3g}")
    except Exception:
        return x


def q25(x):
    return np.nanpercentile(x, 25) if len(x) else np.nan


def q75(x):
    return np.nanpercentile(x, 75) if len(x) else np.nan


def median_iqr(g, col):
    vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(float)
    return pd.Series(
        {
            f"{col}_median": sig3(np.nanmedian(vals)) if len(vals) else np.nan,
            f"{col}_q25": sig3(q25(vals)) if len(vals) else np.nan,
            f"{col}_q75": sig3(q75(vals)) if len(vals) else np.nan,
        }
    )


def load_rewet21():
    usecols = [
        "file", "event_id", "ecosystem", "SITE_ID", "q_original", "q_21",
        "q_21_start", "q_21_end", "q_21_theta_peak", "q_21_theta_end",
        "q_21_theta_jump", "pet_21"
    ]
    d = pd.read_csv(REWET, usecols=usecols, low_memory=False)
    d["ecosystem"] = d["ecosystem"].astype(str).str.lower()
    d = d[d["ecosystem"].isin(ECOS)].copy()
    for c in ["event_id", "q_original", "q_21", "q_21_theta_peak", "q_21_theta_end", "q_21_theta_jump", "pet_21"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["dq_21"] = d["q_21"] - d["q_original"]
    d["start_date"] = pd.to_datetime(d["q_21_start"], errors="coerce")
    d["end_date"] = pd.to_datetime(d["q_21_end"], errors="coerce")
    d["duration_21"] = (d["end_date"] - d["start_date"]).dt.days.abs().clip(lower=1)
    d["drydown_rate"] = d["q_21_theta_jump"] / d["duration_21"]
    # Use theta peak as a proxy where theta_end is missing; state-buffer-specific analyses use process tables below.
    d["state_buffer"] = d["q_21_theta_peak"] - d["q_21_theta_end"]
    d = d.dropna(subset=["q_original", "dq_21", "SITE_ID"])
    return d


def load_qorig_env():
    cols = ["SITE_ID", "ecosystem", "q_q", "AI_60d", "LAI"]
    q = pd.read_csv(QORIG, usecols=cols, low_memory=False)
    q["ecosystem"] = q["ecosystem"].astype(str).str.lower()
    q = q[q["ecosystem"].isin(ECOS)].copy()
    for c in ["q_q", "AI_60d", "LAI"]:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q.dropna(subset=["q_q", "AI_60d", "LAI"])
    return q


def add_q_state(df, col="q_original"):
    out = df.copy()
    out["q_state"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    for eco, idx in out.groupby("ecosystem", observed=True).groups.items():
        ranks = out.loc[idx, col].rank(method="first")
        out.loc[idx, "q_state"] = pd.qcut(ranks, 4, labels=STATE_LABELS).astype(str).to_numpy()
    return out


def build_s1():
    r = add_q_state(load_rewet21(), "q_original")
    q = load_qorig_env()
    q = q.rename(columns={"q_q": "q_original_env"})
    env = q.groupby(["ecosystem", "SITE_ID"], observed=True)[["AI_60d", "LAI"]].median().reset_index()
    r = r.merge(env, on=["ecosystem", "SITE_ID"], how="left")
    rows = []
    for (eco, state), g in r.groupby(["ecosystem", "q_state"], observed=True):
        row = {
            "ecosystem": eco,
            "q_state": state,
            "events": len(g),
            "independent_sites": g["SITE_ID"].nunique(),
        }
        for col in ["q_original", "dq_21", "drydown_rate", "state_buffer", "pet_21", "AI_60d", "LAI"]:
            row.update(median_iqr(g, col).to_dict())
        rows.append(row)
    s1 = pd.DataFrame(rows).sort_values(["ecosystem", "q_state"])
    return s1


def standardize(v):
    v = np.asarray(v, dtype=float)
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / (sd if sd else 1)


def weighted_ols(g, ycol, xcols):
    use = g.dropna(subset=[ycol] + xcols + ["SITE_ID"]).copy()
    if len(use) < len(xcols) + 5 or use["SITE_ID"].nunique() < 3:
        return None
    y = use[ycol].to_numpy(float)
    Xraw = use[xcols].to_numpy(float)
    X = np.column_stack([np.ones(len(use)), Xraw])
    counts = use["SITE_ID"].map(use["SITE_ID"].value_counts()).to_numpy(float)
    w = 1 / counts
    w = w / w.mean()
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    yhat = X @ beta
    resid = y - yhat
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = np.sum(w * resid ** 2) / dof
    cov = sigma2 * np.linalg.pinv((X * sw[:, None]).T @ (X * sw[:, None]))
    se = np.sqrt(np.diag(cov))
    tvals = beta / np.where(se == 0, np.nan, se)
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
    ybar = np.average(y, weights=w)
    r2 = 1 - np.sum(w * resid ** 2) / np.sum(w * (y - ybar) ** 2) if np.sum(w * (y - ybar) ** 2) else np.nan
    # standardized coefficients
    ys = standardize(y)
    Xs = np.column_stack([np.ones(len(use)), np.column_stack([standardize(use[c].to_numpy(float)) for c in xcols])])
    bstd = np.linalg.lstsq(Xs * sw[:, None], ys * sw, rcond=None)[0]
    return {
        "n": len(use), "sites": use["SITE_ID"].nunique(), "r2": r2,
        "beta": beta, "se": se, "p": pvals, "std_beta": bstd,
    }


def site_boot_direction(g, ycol, xcols, n_boot=200):
    use = g.dropna(subset=[ycol] + xcols + ["SITE_ID"]).copy()
    sites = use["SITE_ID"].drop_duplicates().to_numpy()
    if len(sites) < 3:
        return {c: np.nan for c in xcols}
    groups = {s: d for s, d in use.groupby("SITE_ID", observed=True)}
    dirs = {c: [] for c in xcols}
    for _ in range(n_boot):
        sampled = np.random.default_rng(20260626 + _).choice(sites, size=len(sites), replace=True)
        dd = pd.concat([groups[s] for s in sampled], ignore_index=True)
        fit = weighted_ols(dd, ycol, xcols)
        if fit is None:
            continue
        for i, c in enumerate(xcols):
            dirs[c].append(fit["std_beta"][i + 1])
    out = {}
    for c, vals in dirs.items():
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            out[c] = np.nan
        else:
            out[c] = sig3(max(np.mean(vals > 0), np.mean(vals < 0)))
    return out


def build_s2():
    rows = []
    # q_original background model
    q = load_qorig_env().rename(columns={"q_q": "q_original"})
    q = add_q_state(q, "q_original")
    q["AI60_x_LAI"] = q["AI_60d"] * q["LAI"]
    for eco, state in [(e, s) for e in ECOS for s in STATE_LABELS]:
        g = q[(q["ecosystem"].eq(eco)) & (q["q_state"].eq(state))]
        fit = weighted_ols(g, "q_original", ["AI_60d", "LAI", "AI60_x_LAI"])
        dirs = site_boot_direction(g, "q_original", ["AI_60d", "LAI", "AI60_x_LAI"], n_boot=80)
        if fit:
            for i, pred in enumerate(["AI_60d", "LAI", "AI60_x_LAI"]):
                rows.append({
                    "model_name": "antecedent_q_background",
                    "ecosystem": eco, "q_state": state,
                    "response_variable": "q_original",
                    "predictor": pred.replace("_x_", " x "),
                    "raw_coefficient": sig3(fit["beta"][i + 1]),
                    "standardized_coefficient": sig3(fit["std_beta"][i + 1]),
                    "standard_error": sig3(fit["se"][i + 1]),
                    "ci95_low": sig3(fit["beta"][i + 1] - 1.96 * fit["se"][i + 1]),
                    "ci95_high": sig3(fit["beta"][i + 1] + 1.96 * fit["se"][i + 1]),
                    "p_value": sig3(fit["p"][i + 1]),
                    "site_bootstrap_direction_probability": dirs[pred],
                    "n": fit["n"], "sites": fit["sites"],
                    "r2": sig3(fit["r2"]),
                    "cv_performance": np.nan,
                })
    # dq process model
    r = add_q_state(load_rewet21(), "q_original")
    r["q_original_x_drydown_rate"] = r["q_original"] * r["drydown_rate"]
    for eco, state in [(e, s) for e in ECOS for s in STATE_LABELS]:
        g = r[(r["ecosystem"].eq(eco)) & (r["q_state"].eq(state))]
        xcols = ["drydown_rate", "state_buffer", "pet_21", "q_original", "q_original_x_drydown_rate"]
        fit = weighted_ols(g, "dq_21", xcols)
        dirs = site_boot_direction(g, "dq_21", xcols, n_boot=80)
        if fit:
            for i, pred in enumerate(xcols):
                rows.append({
                    "model_name": "post_wetting_dq_process",
                    "ecosystem": eco, "q_state": state,
                    "response_variable": "dq_21",
                    "predictor": pred.replace("pet_21", "PET").replace("_x_", " x "),
                    "raw_coefficient": sig3(fit["beta"][i + 1]),
                    "standardized_coefficient": sig3(fit["std_beta"][i + 1]),
                    "standard_error": sig3(fit["se"][i + 1]),
                    "ci95_low": sig3(fit["beta"][i + 1] - 1.96 * fit["se"][i + 1]),
                    "ci95_high": sig3(fit["beta"][i + 1] + 1.96 * fit["se"][i + 1]),
                    "p_value": sig3(fit["p"][i + 1]),
                    "site_bootstrap_direction_probability": dirs[pred],
                    "n": fit["n"], "sites": fit["sites"],
                    "r2": sig3(fit["r2"]),
                    "cv_performance": np.nan,
                })
    return pd.DataFrame(rows)


def vif_table(g, vars_):
    rows = []
    use = g.dropna(subset=vars_).copy()
    for target in vars_:
        others = [v for v in vars_ if v != target]
        fit = weighted_ols(use.assign(SITE_ID="all"), target, others)
        r2 = fit["r2"] if fit else np.nan
        vif = 1 / (1 - r2) if pd.notna(r2) and r2 < 1 else np.inf
        rows.append({"variable": target, "VIF": sig3(vif)})
    return rows


def build_s3():
    q = load_qorig_env().rename(columns={"q_q": "q_original"})
    r = load_rewet21()
    env = q.groupby(["ecosystem", "SITE_ID"], observed=True)[["AI_60d", "LAI"]].median().reset_index()
    d = r.merge(env, on=["ecosystem", "SITE_ID"], how="left")
    d = d.rename(columns={"pet_21": "PET"})
    vars_ = ["AI_60d", "LAI", "PET", "drydown_rate", "state_buffer"]
    corr_rows = []
    for eco, g in d.groupby("ecosystem", observed=True):
        for i, a in enumerate(vars_):
            for b in vars_[i + 1:]:
                use = g[[a, b]].dropna()
                if len(use) > 3:
                    rho, p = stats.spearmanr(use[a], use[b])
                else:
                    rho, p = np.nan, np.nan
                corr_rows.append({"section": "Spearman correlation", "ecosystem": eco, "variable_1": a, "variable_2": b, "estimate": sig3(rho), "p_value": sig3(p)})
    vif_rows = []
    for eco, g in d.groupby("ecosystem", observed=True):
        for row in vif_table(g, vars_):
            vif_rows.append({"section": "VIF", "ecosystem": eco, "variable_1": row["variable"], "variable_2": "", "estimate": row["VIF"], "p_value": np.nan})
    return pd.DataFrame(corr_rows + vif_rows)


def build_s4():
    # Recompute contribution uncertainty from q_original background models.
    q = load_qorig_env().rename(columns={"q_q": "q_original"})
    q = add_q_state(q, "q_original")
    q["AI60_x_LAI"] = q["AI_60d"] * q["LAI"]
    xcols = ["AI_60d", "LAI", "AI60_x_LAI"]
    rows = []
    rng = np.random.default_rng(20260626)
    for eco in ECOS:
        for state in STATE_LABELS:
            g = q[(q["ecosystem"].eq(eco)) & (q["q_state"].eq(state))].dropna(subset=["q_original"] + xcols + ["SITE_ID"])
            sites = g["SITE_ID"].drop_duplicates().to_numpy()
            if len(g) < 30 or len(sites) < 3:
                continue
            def contrib(dd):
                full = weighted_ols(dd, "q_original", xcols)
                if full is None:
                    return np.full(3, np.nan), np.nan
                full_r2 = full["r2"]
                drops = []
                for x in xcols:
                    red = weighted_ols(dd, "q_original", [v for v in xcols if v != x])
                    drops.append(max(full_r2 - (red["r2"] if red else 0), 0))
                arr = np.asarray(drops, dtype=float)
                norm = arr / arr.sum() * 100 if arr.sum() > 0 else np.full(3, np.nan)
                return norm, full_r2
            base, r2 = contrib(g)
            boots = []
            groups = {s: dd for s, dd in g.groupby("SITE_ID", observed=True)}
            for _ in range(160):
                sampled = rng.choice(sites, size=len(sites), replace=True)
                dd = pd.concat([groups[s] for s in sampled], ignore_index=True)
                b, _ = contrib(dd)
                boots.append(b)
            boot = np.vstack(boots)
            valid = np.isfinite(boot).all(axis=1)
            boot_valid = boot[valid]
            if boot_valid.size == 0 or not np.all(np.isfinite(base)):
                continue
            max_idx = np.nanargmax(boot_valid, axis=1)
            classes = ["AI60-dominated", "LAI-dominated", "interaction-dominated"]
            for i, pred in enumerate(xcols):
                rows.append({
                    "ecosystem": eco, "q_state": state,
                    "model": "q_original ~ AI60 + LAI + AI60 x LAI",
                    "cv_r2_or_loss_reduction": np.nan,
                    "model_r2": sig3(r2),
                    "predictor": pred.replace("_x_", " x "),
                    "absolute_importance": sig3(base[i]),
                    "normalized_contribution_percent": sig3(base[i]),
                    "bootstrap_ci95_low": sig3(np.nanpercentile(boot_valid[:, i], 2.5)),
                    "bootstrap_ci95_high": sig3(np.nanpercentile(boot_valid[:, i], 97.5)),
                    "probability_largest_contributor": sig3(np.mean(max_idx == i)),
                    "dominance_class": classes[int(np.nanargmax(base))] if np.all(np.isfinite(base)) else "mixed",
                })
    return pd.DataFrame(rows)


tables = {
    "S1_sample_support": build_s1(),
    "S2_model_results": build_s2(),
    "S3_collinearity": build_s3(),
    "S4_fig2_contributions": build_s4(),
}

for name, df in tables.items():
    df = df.map(sig3)
    df.to_csv(OUT / f"{name}.csv", index=False)

notes = pd.DataFrame(
    [
        {"item": "Variables", "note": "dq_21 = q_21 - q_original; AI60 = precipitation/PET over the preceding 60 days; PET is PET during the 21-day post-wetting window; drydown_rate is theta_jump divided by redrying duration; state_buffer is derived from available post-wetting theta metrics in the event table."},
        {"item": "States", "note": "Q1-Q4 are ecosystem-specific quartiles of q_original unless otherwise noted."},
        {"item": "Standardization", "note": "Standardized coefficients use z-scored response and z-scored predictors within each fitted subset."},
        {"item": "Bootstrap", "note": "Direction probabilities and contribution intervals use site-cluster bootstrap resampling unless the available number of sites is insufficient."},
        {"item": "Precision", "note": "Numeric values are rounded to three significant digits."},
    ]
)
notes.to_csv(OUT / "Supplementary_Table_notes.csv", index=False)

manifest = {
    "tables": {name: str((OUT / f"{name}.csv")) for name in tables},
    "notes": str(OUT / "Supplementary_Table_notes.csv"),
}
(OUT / "supplementary_tables_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
