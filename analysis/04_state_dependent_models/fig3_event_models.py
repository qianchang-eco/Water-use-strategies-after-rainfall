#!/usr/bin/env python3
"""Build two non-destructive Fig. 3 revision candidates.

Candidate A preserves the manuscript's marginal Pearson-r design but rebuilds
it from unique paper-cohort events and adds site-cluster bootstrap 95% CIs.
Candidate B fits all three predictors simultaneously within each ecosystem x
antecedent-state stratum and reports fully standardized partial coefficients
with the same site-cluster uncertainty.

The script writes only to the requested output directory.  It never overwrites
the submitted Fig. 3 or the manuscript.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


KEY = ["file", "event_id", "window_days"]
ECOSYSTEMS = ["savanna", "forest", "grassland"]
ECOSYSTEM_LABELS = {"savanna": "Savanna", "forest": "Forest", "grassland": "Grassland"}
STATES = ["R1", "R2", "R3", "R4"]
STATE_LABELS = ["R1 low", "R2 midlow", "R3 midhigh", "R4 high"]
PREDICTORS = ["drydown_rate", "moisture_buffering", "PET90"]
PREDICTOR_LABELS = {
    "drydown_rate": "Drydown rate",
    "moisture_buffering": "Moisture buffering",
    "PET90": r"$PET_{90}$",
}
COLORS = ["#d9ee9b", "#88c76e", "#36aaa5", "#1f734e"]
SEED = 20261116




def require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def assign_equal_count_states(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["R_state"] = ""
    for ecosystem, indices in out.groupby("ecosystem", sort=False).groups.items():
        ordered = out.loc[indices].sort_values(["q_original", "file", "event_id"]).index.to_numpy()
        state_number = np.minimum(3, np.floor(np.arange(len(ordered)) * 4 / len(ordered)).astype(int))
        out.loc[ordered, "R_state"] = np.asarray(STATES, dtype=object)[state_number]
    if out["R_state"].eq("").any():
        raise AssertionError("Some events were not assigned to an antecedent state")
    return out


def attach_pet90(frame: pd.DataFrame, pet_dir: Path) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    result["PET90"] = np.nan
    missing_sites: list[str] = []
    incomplete_events = 0
    for site, group in result.groupby("SITE_ID", sort=False):
        path = pet_dir / f"{site}.csv"
        if not path.is_file():
            missing_sites.append(str(site))
            incomplete_events += len(group)
            continue
        daily = pd.read_csv(path, usecols=["date", "pet"])
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
        daily["pet"] = pd.to_numeric(daily["pet"], errors="coerce")
        daily = daily.dropna(subset=["date", "pet"])
        if daily.duplicated("date").any():
            raise ValueError(f"Duplicate daily PET dates for site {site}")
        series = daily.set_index("date")["pet"]
        for row in group.itertuples():
            if pd.isna(row.post_start) or pd.isna(row.post_end):
                incomplete_events += 1
                continue
            days = pd.date_range(row.post_start, row.post_end, freq="D")
            if len(days) == 0 or not days.isin(series.index).all():
                incomplete_events += 1
                continue
            result.loc[row.Index, "PET90"] = float(series.loc[days].quantile(0.90))
    audit = {
        "pet_missing_site_ids": missing_sites,
        "events_without_complete_daily_pet": int(incomplete_events),
        "events_with_complete_daily_pet": int(result["PET90"].notna().sum()),
    }
    return result, audit


def load_analysis_cohort(paper_cohort: Path, process_events: Path, pet_dir: Path) -> tuple[pd.DataFrame, dict]:
    paper = pd.read_csv(paper_cohort, low_memory=False)
    require_columns(
        paper,
        KEY + ["ecosystem", "q_original", "q_post", "delta_q", "SITE_ID"],
        "paper cohort",
    )
    paper = paper.loc[paper["window_days"].eq(21) & paper["ecosystem"].isin(ECOSYSTEMS)].copy()
    if paper.duplicated(KEY).any():
        raise ValueError("Paper cohort is not unique by file + event_id + window_days")

    usecols = KEY + [
        "SITE_ID", "q_original", "delta_q", "post_start", "post_end",
        "drydown_rate", "peak_minus_theta_star_proxy",
    ]
    process = pd.read_csv(process_events, usecols=usecols, low_memory=False)
    process = process.loc[process["window_days"].eq(21)].copy()
    if process.duplicated(KEY).any():
        raise ValueError("Process-event table is not unique by file + event_id + window_days")
    process["post_start"] = pd.to_datetime(process["post_start"], errors="coerce")
    process["post_end"] = pd.to_datetime(process["post_end"], errors="coerce")
    process = process.rename(columns={
        "SITE_ID": "SITE_ID_process",
        "q_original": "q_original_process",
        "delta_q": "delta_q_process",
        "peak_minus_theta_star_proxy": "moisture_buffering",
    })
    merged = paper.merge(process, on=KEY, how="left", validate="one_to_one", indicator=True)
    missing_process = int(merged["_merge"].ne("both").sum())
    merged = merged.drop(columns="_merge")
    site_disagreement = merged["SITE_ID_process"].notna() & merged["SITE_ID"].ne(merged["SITE_ID_process"])
    if site_disagreement.any():
        raise ValueError("SITE_ID disagreement between paper and process cohorts")
    comparable = merged[["q_original_process", "delta_q_process"]].notna().all(axis=1)
    q_disagreement = int((~np.isclose(
        merged.loc[comparable, "q_original"], merged.loc[comparable, "q_original_process"], atol=1e-10
    )).sum())
    dq_disagreement = int((~np.isclose(
        merged.loc[comparable, "delta_q"], merged.loc[comparable, "delta_q_process"], atol=1e-10
    )).sum())
    if q_disagreement:
        raise ValueError(f"Paper/process q_original disagreement: q={q_disagreement}")
    # The process inventory was assembled from a different historical post-q
    # export, so its stored delta_q is diagnostic only.  Preserve the exact
    # manuscript delta_q from the paper cohort and use the process inventory
    # solely for event-aligned hydrological predictors and post dates.

    merged = assign_equal_count_states(merged)
    merged, pet_audit = attach_pet90(merged, pet_dir)
    audit = {
        "paper_21d_events": int(len(paper)),
        "paper_21d_by_ecosystem": paper.groupby("ecosystem").size().astype(int).to_dict(),
        "paper_21d_sites_by_ecosystem": paper.groupby("ecosystem")["SITE_ID"].nunique().astype(int).to_dict(),
        "events_missing_process_match": missing_process,
        "diagnostic_process_delta_q_disagreements": dq_disagreement,
        "events_with_drydown_rate": int(merged["drydown_rate"].notna().sum()),
        "events_with_moisture_buffering": int(merged["moisture_buffering"].notna().sum()),
    } | pet_audit
    return merged, audit


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_correlation(group: pd.DataFrame, predictor: str, draws: int, seed: int) -> tuple[float, float, int]:
    groups = [g[[predictor, "delta_q"]].to_numpy(float) for _, g in group.groupby("SITE_ID", sort=True)]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        chosen = rng.integers(0, len(groups), size=len(groups))
        sampled = np.concatenate([groups[index] for index in chosen], axis=0)
        value = pearson_r(sampled[:, 0], sampled[:, 1])
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(200, draws // 2):
        return np.nan, np.nan, len(values)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), len(values)


def calculate_correlations(data: pd.DataFrame, draws: int) -> pd.DataFrame:
    rows: list[dict] = []
    for ei, ecosystem in enumerate(ECOSYSTEMS):
        for si, state in enumerate(STATES):
            base = data.loc[data["ecosystem"].eq(ecosystem) & data["R_state"].eq(state)]
            for pi, predictor in enumerate(PREDICTORS):
                frame = base[["SITE_ID", predictor, "delta_q"]].replace([np.inf, -np.inf], np.nan).dropna()
                n_events = len(frame)
                n_sites = frame["SITE_ID"].nunique()
                r = pearson_r(frame[predictor].to_numpy(float), frame["delta_q"].to_numpy(float))
                status = "estimated" if n_events >= 10 and n_sites >= 5 and np.isfinite(r) else "insufficient_support"
                low = high = np.nan
                valid = 0
                if status == "estimated":
                    low, high, valid = bootstrap_correlation(
                        frame, predictor, draws, SEED + 1000 * ei + 100 * si + pi
                    )
                    if not np.isfinite(low):
                        status = "bootstrap_unstable"
                rows.append({
                    "ecosystem": ecosystem,
                    "R_state": state,
                    "predictor": predictor,
                    "n_events": int(n_events),
                    "n_sites": int(n_sites),
                    "pearson_r": r,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_valid": int(valid),
                    "bootstrap_requested": int(draws),
                    "status": status,
                })
    result = pd.DataFrame(rows)
    if len(result) != 36 or result.duplicated(["ecosystem", "R_state", "predictor"]).any():
        raise AssertionError("Expected exactly 36 correlation results")
    return result


def standardized_beta(frame: pd.DataFrame) -> tuple[np.ndarray, float, int]:
    x = frame[PREDICTORS].to_numpy(float)
    y = frame["delta_q"].to_numpy(float)
    x_sd = x.std(axis=0, ddof=0)
    y_sd = y.std(ddof=0)
    if np.any(x_sd <= 0) or y_sd <= 0:
        return np.full(3, np.nan), np.nan, 0
    xz = (x - x.mean(axis=0)) / x_sd
    yz = (y - y.mean()) / y_sd
    design = np.column_stack([np.ones(len(xz)), xz])
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design))
    if rank < 4 or condition > 1e8:
        return np.full(3, np.nan), condition, rank
    beta = np.linalg.lstsq(design, yz, rcond=None)[0][1:]
    return beta, condition, rank


def bootstrap_partial(group: pd.DataFrame, draws: int, seed: int) -> tuple[np.ndarray, int]:
    groups = [g.copy() for _, g in group.groupby("SITE_ID", sort=True)]
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    for _ in range(draws):
        chosen = rng.integers(0, len(groups), size=len(groups))
        sampled = pd.concat([groups[index] for index in chosen], ignore_index=True)
        beta, _, rank = standardized_beta(sampled)
        if rank == 4 and np.isfinite(beta).all():
            values.append(beta)
    if len(values) < max(200, draws // 2):
        return np.full((2, 3), np.nan), len(values)
    array = np.vstack(values)
    return np.quantile(array, [0.025, 0.975], axis=0), len(values)


def calculate_partial_effects(data: pd.DataFrame, correlations: pd.DataFrame, draws: int) -> pd.DataFrame:
    rows: list[dict] = []
    for ei, ecosystem in enumerate(ECOSYSTEMS):
        for si, state in enumerate(STATES):
            frame = data.loc[
                data["ecosystem"].eq(ecosystem) & data["R_state"].eq(state),
                ["SITE_ID", "delta_q"] + PREDICTORS,
            ].replace([np.inf, -np.inf], np.nan).dropna()
            n_events = len(frame)
            n_sites = frame["SITE_ID"].nunique()
            beta, condition, rank = standardized_beta(frame) if n_events else (np.full(3, np.nan), np.nan, 0)
            status = (
                "estimated" if n_events >= 10 and n_sites >= 5 and rank == 4 and np.isfinite(beta).all()
                else "insufficient_support"
            )
            interval = np.full((2, 3), np.nan)
            valid = 0
            if status == "estimated":
                interval, valid = bootstrap_partial(frame, draws, SEED + 1000 * ei + 100 * si)
                if not np.isfinite(interval).all():
                    status = "bootstrap_unstable"
            for pi, predictor in enumerate(PREDICTORS):
                marginal = correlations.loc[
                    correlations["ecosystem"].eq(ecosystem)
                    & correlations["R_state"].eq(state)
                    & correlations["predictor"].eq(predictor),
                    "pearson_r",
                ].iloc[0]
                direction_match = bool(
                    status == "estimated" and np.sign(beta[pi]) == np.sign(float(marginal))
                )
                rows.append({
                    "ecosystem": ecosystem,
                    "R_state": state,
                    "predictor": predictor,
                    "n_events": int(n_events),
                    "n_sites": int(n_sites),
                    "std_partial_beta": float(beta[pi]) if np.isfinite(beta[pi]) else np.nan,
                    "ci_low": float(interval[0, pi]) if np.isfinite(interval[0, pi]) else np.nan,
                    "ci_high": float(interval[1, pi]) if np.isfinite(interval[1, pi]) else np.nan,
                    "marginal_pearson_r": float(marginal),
                    "direction_matches_main_r": direction_match,
                    "bootstrap_valid": int(valid),
                    "bootstrap_requested": int(draws),
                    "design_condition_number": condition,
                    "status": status,
                })
    result = pd.DataFrame(rows)
    if len(result) != 36 or result.duplicated(["ecosystem", "R_state", "predictor"]).any():
        raise AssertionError("Expected exactly 36 partial-effect results")
    return result


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
    })


def finish_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(top=False, right=False, labelsize=8.2, length=3)


def save_bundle(fig: plt.Figure, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")


def draw_marginal(correlations: pd.DataFrame, outbase: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(3, 3, figsize=(10.6, 7.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.885, bottom=0.105, wspace=0.17, hspace=0.34)
    for row, ecosystem in enumerate(ECOSYSTEMS):
        for col, predictor in enumerate(PREDICTORS):
            ax = axes[row, col]
            subset = (
                correlations.loc[
                    correlations["ecosystem"].eq(ecosystem)
                    & correlations["predictor"].eq(predictor)
                ].set_index("R_state").reindex(STATES)
            )
            for index, state in enumerate(STATES):
                record = subset.loc[state]
                if record["status"] != "estimated":
                    ax.text(index, 0, "n/a", ha="center", va="center", fontsize=7.2, color="#666666")
                    continue
                value = float(record["pearson_r"])
                low, high = float(record["ci_low"]), float(record["ci_high"])
                ax.bar(index, value, width=0.62, color=COLORS[index], edgecolor=COLORS[index], zorder=2)
                ax.errorbar(
                    index, value, yerr=[[value - low], [high - value]], fmt="none",
                    ecolor="#252525", elinewidth=0.9, capsize=2.8, capthick=0.9, zorder=4,
                )
                label_y = high + 0.045 if value >= 0 else low - 0.045
                ax.text(index, label_y, f"{value:+.2f}", ha="center",
                        va="bottom" if value >= 0 else "top", fontsize=7.2)
            ax.axhline(0, color="#666666", lw=0.8, ls=(0, (3, 2)), zorder=0)
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks(range(4), STATES)
            if row == 0:
                ax.set_title(PREDICTOR_LABELS[predictor], fontsize=10.0, fontweight="bold", pad=7)
            if col == 0:
                ax.set_ylabel(ECOSYSTEM_LABELS[ecosystem], fontsize=9.0)
            if col != 0:
                ax.tick_params(labelleft=False)
            ax.text(-0.13, 1.04, "abcdefghi"[3 * row + col], transform=ax.transAxes,
                    fontsize=11, fontweight="bold", ha="left", va="bottom")
            finish_axes(ax)
    handles = [Patch(facecolor=COLORS[i], edgecolor=COLORS[i], label=STATE_LABELS[i]) for i in range(4)]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.54, 0.96), columnspacing=1.5)
    fig.text(0.035, 0.50, r"Pearson correlation with $\Delta q_{21}$ ($r$)",
             rotation=90, va="center", ha="center", fontsize=10.0)
    save_bundle(fig, outbase)
    plt.close(fig)


def draw_partial(partial: pd.DataFrame, outbase: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(3, 3, figsize=(10.6, 7.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.87, bottom=0.095, wspace=0.17, hspace=0.34)
    for row, ecosystem in enumerate(ECOSYSTEMS):
        for col, predictor in enumerate(PREDICTORS):
            ax = axes[row, col]
            subset = (
                partial.loc[partial["ecosystem"].eq(ecosystem) & partial["predictor"].eq(predictor)]
                .set_index("R_state").reindex(STATES)
            )
            for index, state in enumerate(STATES):
                record = subset.loc[state]
                if record["status"] != "estimated":
                    ax.text(index, 0, "n/a", ha="center", va="center", fontsize=7.2, color="#666666")
                    continue
                value = float(record["std_partial_beta"])
                low, high = float(record["ci_low"]), float(record["ci_high"])
                ax.bar(
                    index, value, width=0.62, color=COLORS[index],
                    edgecolor=COLORS[index], zorder=2,
                )
                ax.errorbar(
                    index, value, yerr=[[value - low], [high - value]], fmt="none",
                    ecolor="#252525", elinewidth=0.9, capsize=2.8,
                    capthick=0.9, zorder=4,
                )
                if value >= 0:
                    label_y = min(high + 0.040, 0.92)
                    label_va = "bottom"
                elif low < -0.82:
                    # When a lower CI approaches the axis boundary, move the
                    # coefficient label above the upper whisker instead of
                    # clipping it or placing it on top of the lower cap.
                    label_y = min(high + 0.055, 0.92)
                    label_va = "bottom"
                else:
                    label_y = max(low - 0.040, -0.90)
                    label_va = "top"
                ax.text(
                    index, label_y, f"{value:+.2f}", ha="center",
                    va=label_va, fontsize=8.0,
                )
            ax.axhline(0, color="#666666", lw=0.8, ls=(0, (3, 2)), zorder=0)
            ax.set_xlim(-0.5, 3.5)
            # One Savanna interval reaches -1.025; the 0.05 outer margin keeps
            # the full CI visible while labelled ticks span exactly -1 to 1.
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks(np.arange(-1.0, 1.01, 0.25))
            ax.set_xticks(range(4), STATES)
            if row == 0:
                ax.set_title(PREDICTOR_LABELS[predictor], fontsize=11.0, fontweight="bold", pad=7)
            if col == 0:
                ax.set_ylabel(ECOSYSTEM_LABELS[ecosystem], fontsize=10.0)
            if col != 0:
                ax.tick_params(labelleft=False)
            ax.text(-0.13, 1.04, "abcdefghi"[3 * row + col], transform=ax.transAxes,
                    fontsize=12, fontweight="bold", ha="left", va="bottom")
            finish_axes(ax)
    state_handles = [Patch(facecolor=COLORS[i], edgecolor=COLORS[i], label=STATE_LABELS[i]) for i in range(4)]
    fig.legend(handles=state_handles, loc="upper center", ncol=4,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.54, 0.96), columnspacing=1.5)
    fig.text(0.032, 0.49, r"Standardized effect on $\Delta q_{21}$",
             rotation=90, va="center", ha="center", fontsize=11.0)
    save_bundle(fig, outbase)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-cohort", type=Path, required=True)
    parser.add_argument("--process-events", type=Path, required=True)
    parser.add_argument("--pet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=3000)
    args = parser.parse_args()

    data, cohort_audit = load_analysis_cohort(args.paper_cohort, args.process_events, args.pet_dir)
    correlations = calculate_correlations(data, args.bootstrap_draws)
    partial = calculate_partial_effects(data, correlations, args.bootstrap_draws)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "fig3_candidate_unique_event_cohort.csv", index=False)
    correlations.to_csv(args.output_dir / "fig3_candidate_marginal_correlations.csv", index=False)
    partial.to_csv(args.output_dir / "fig3_candidate_partial_coefficients.csv", index=False)
    support = (
        data.groupby(["ecosystem", "R_state"], observed=True)
        .agg(events=("delta_q", "size"), sites=("SITE_ID", "nunique"),
             events_with_pet90=("PET90", lambda x: int(x.notna().sum())))
        .reset_index()
    )
    support.to_csv(args.output_dir / "fig3_candidate_event_site_support.csv", index=False)

    marginal_base = args.output_dir / "Fig3_candidate_unique_events_site_cluster_CI"
    partial_base = args.output_dir / "ExtendedData_candidate_partial_coefficients"
    draw_marginal(correlations, marginal_base)
    draw_partial(partial, partial_base)

    manifest = {
        "purpose": "non-destructive Fig. 3 comparison candidates",
        "event_key": KEY,
        "paper_event_membership_preserved": True,
        "R_state_definition": "four equal-count strata assigned within ecosystem from paper-cohort q_original",
        "marginal_model": "separate Pearson correlation for each predictor",
        "partial_model": "fully standardized OLS: z(delta_q) ~ z(drydown_rate) + z(moisture_buffering) + z(PET90)",
        "bootstrap": {
            "unit": "SITE_ID",
            "draws": args.bootstrap_draws,
            "seed_base": SEED,
            "interval": "percentile 95%",
            "minimum_support": "10 events and 5 sites",
        },
        "cohort_audit": cohort_audit,
        "outputs": {
            "main_candidate": str(marginal_base.with_suffix(".png")),
            "extended_data_candidate": str(partial_base.with_suffix(".png")),
        },
    }
    (args.output_dir / "fig3_revision_candidates_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(cohort_audit, indent=2, ensure_ascii=False))
    print("\nSupport by ecosystem and state:\n", support.to_string(index=False))
    print("\nMarginal correlations:\n", correlations.to_string(index=False))
    print("\nPartial coefficients:\n", partial.to_string(index=False))
    print(marginal_base.with_suffix(".png"))
    print(partial_base.with_suffix(".png"))


if __name__ == "__main__":
    main()
