#!/usr/bin/env python3
"""Add site-cluster uncertainty to the exact point-q cohort used in the paper.

This script never refits q and never changes cohort membership. It attaches a
validated SITE_ID to the already deduplicated paper source table, reproduces
the published point summaries, and adds site-cluster bootstrap intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["file", "event_id"]
ECOSYSTEMS = ["forest", "grassland", "savanna"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-cohort", required=True, type=Path)
    parser.add_argument("--q-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261114)
    return parser.parse_args()


def attach_site_id(cohort: pd.DataFrame, q_table: Path) -> tuple[pd.DataFrame, dict]:
    metadata = pd.read_csv(q_table, usecols=KEY + ["SITE_ID"], low_memory=False)
    conflicts = metadata.groupby(KEY, dropna=False)["SITE_ID"].nunique(dropna=True)
    if (conflicts > 1).any():
        raise ValueError(f"SITE_ID conflicts for {int((conflicts > 1).sum())} event keys")
    metadata = metadata.dropna(subset=KEY + ["SITE_ID"]).drop_duplicates(KEY)
    joined = cohort.merge(metadata, on=KEY, how="left", validate="many_to_one")
    if joined["SITE_ID"].isna().any():
        raise ValueError(f"SITE_ID missing for {int(joined['SITE_ID'].isna().sum())} paper rows")
    return joined, {
        "paper_rows": int(len(cohort)),
        "paper_unique_event_windows": int(cohort[KEY + ["window_days"]].drop_duplicates().shape[0]),
        "site_id_conflicting_event_keys": 0,
        "site_id_missing_rows": 0,
    }


def cluster_bootstrap(group: pd.DataFrame, reps: int, seed: int) -> dict:
    sites = group["SITE_ID"].drop_duplicates().astype(str).to_numpy()
    by_site = {
        site: group.loc[group["SITE_ID"].astype(str).eq(site), "delta_q"].to_numpy(float)
        for site in sites
    }
    rng = np.random.default_rng(seed)
    medians = np.empty(reps, dtype=float)
    fractions = np.empty(reps, dtype=float)
    equal_site_medians = np.empty(reps, dtype=float)
    equal_site_fractions = np.empty(reps, dtype=float)
    spearman_draws = np.empty(reps, dtype=float)
    for replicate in range(reps):
        sampled = rng.choice(sites, size=len(sites), replace=True)
        values = np.concatenate([by_site[site] for site in sampled])
        site_medians = np.asarray([np.median(by_site[site]) for site in sampled])
        site_fractions = np.asarray([np.mean(by_site[site] < 0) for site in sampled])
        medians[replicate] = np.median(values)
        fractions[replicate] = np.mean(values < 0)
        equal_site_medians[replicate] = np.median(site_medians)
        equal_site_fractions[replicate] = np.mean(site_fractions)
        sampled_frame = pd.concat(
            [group.loc[group["SITE_ID"].astype(str).eq(site),
                       ["q_original", "q_post"]]
             for site in sampled],
            ignore_index=True,
        )
        spearman_draws[replicate] = sampled_frame["q_original"].corr(
            sampled_frame["q_post"], method="spearman")
    q_original = group["q_original"].to_numpy(float)
    q_post = group["q_post"].to_numpy(float)
    return {
        "n_events": int(len(group)),
        "n_sites": int(len(sites)),
        "median_delta_q": float(np.median(group["delta_q"])),
        "median_delta_q_cluster_low": float(np.quantile(medians, 0.025)),
        "median_delta_q_cluster_high": float(np.quantile(medians, 0.975)),
        "negative_fraction": float(np.mean(group["delta_q"] < 0)),
        "negative_fraction_cluster_low": float(np.quantile(fractions, 0.025)),
        "negative_fraction_cluster_high": float(np.quantile(fractions, 0.975)),
        "median_of_site_medians": float(np.median([
            np.median(values) for values in by_site.values()])),
        "equal_site_negative_fraction": float(np.mean([
            np.mean(values < 0) for values in by_site.values()])),
        "equal_site_median_cluster_low": float(np.quantile(equal_site_medians, 0.025)),
        "equal_site_median_cluster_high": float(np.quantile(equal_site_medians, 0.975)),
        "equal_site_negative_cluster_low": float(np.quantile(equal_site_fractions, 0.025)),
        "equal_site_negative_cluster_high": float(np.quantile(equal_site_fractions, 0.975)),
        "q_original_q_post_pearson": float(np.corrcoef(q_original, q_post)[0, 1]),
        "q_original_q_post_spearman": float(
            pd.Series(q_original).corr(pd.Series(q_post), method="spearman")),
        "q_original_q_post_spearman_cluster_low": float(
            np.nanquantile(spearman_draws, 0.025)),
        "q_original_q_post_spearman_cluster_high": float(
            np.nanquantile(spearman_draws, 0.975)),
        "q_original_delta_q_pearson": float(np.corrcoef(q_original, group["delta_q"])[0, 1]),
        "q_original_boundary_fraction": float(np.mean(
            np.isclose(q_original, 0.01, atol=1e-5) | np.isclose(q_original, 3.0, atol=1e-5))),
        "q_post_boundary_fraction": float(np.mean(
            np.isclose(q_post, 0.01, atol=1e-5) | np.isclose(q_post, 3.0, atol=1e-5))),
    }


def main() -> None:
    args = parse_args()
    cohort = pd.read_csv(args.paper_cohort, low_memory=False)
    required = set(KEY + ["ecosystem", "window_days", "q_original", "q_post", "delta_q"])
    missing = sorted(required - set(cohort))
    if missing:
        raise ValueError(f"paper cohort missing columns: {missing}")
    if cohort.duplicated(KEY + ["window_days"]).any():
        raise ValueError("paper cohort contains duplicate event-window keys")
    for column in ["event_id", "window_days", "q_original", "q_post", "delta_q"]:
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce")
    if cohort[list(required)].isna().any().any():
        raise ValueError("paper cohort contains missing required values")
    if not np.allclose(cohort["delta_q"], cohort["q_post"] - cohort["q_original"],
                       atol=1e-12, rtol=0):
        raise ValueError("stored delta_q disagrees with q_post - q_original")
    if not (cohort["q_original"].between(0.01, 3.0).all()
            and cohort["q_post"].between(0.01, 3.0).all()):
        raise ValueError("paper cohort contains q outside [0.01, 3.0]")
    cohort, audit = attach_site_id(cohort, args.q_table)

    rows = []
    for window in sorted(cohort["window_days"].unique()):
        for ecosystem in ECOSYSTEMS:
            group = cohort.loc[
                cohort["window_days"].eq(window) & cohort["ecosystem"].eq(ecosystem)
            ]
            if group.empty:
                continue
            rows.append({
                "window_days": int(window), "ecosystem": ecosystem,
                **cluster_bootstrap(
                    group, args.bootstrap_reps,
                    args.seed + int(window) * 100 + ECOSYSTEMS.index(ecosystem)),
            })
    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output_dir / "paper_cohort_with_site_id.csv.gz", index=False,
                  compression="gzip")
    summary.to_csv(args.output_dir / "paper_cohort_site_cluster_uncertainty.csv", index=False)
    uncertainty_columns = [
        "window_days", "ecosystem", "n_events", "n_sites", "median_delta_q",
        "median_delta_q_cluster_low", "median_delta_q_cluster_high",
        "negative_fraction", "negative_fraction_cluster_low",
        "negative_fraction_cluster_high",
    ]
    relationship_columns = [
        "window_days", "ecosystem", "n_events", "n_sites",
        "q_original_q_post_spearman",
        "q_original_q_post_spearman_cluster_low",
        "q_original_q_post_spearman_cluster_high",
    ]
    summary[uncertainty_columns].to_csv(
        args.output_dir / "paper_cohort_uncertainty_only.csv", index=False)
    summary[relationship_columns].to_csv(
        args.output_dir / "paper_cohort_q_relationship_only.csv", index=False)
    audit.update({
        "paper_cohort": str(args.paper_cohort.resolve()),
        "q_table": str(args.q_table.resolve()),
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "q_refitted": False,
        "cohort_membership_changed": False,
        "uncertainty_scope": "site-cluster sampling uncertainty around the published point-q cohort",
    })
    (args.output_dir / "paper_cohort_uncertainty_manifest.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")

    selected = summary.loc[summary["window_days"].eq(21)].copy()
    lines = [
        "# Paper-cohort uncertainty audit",
        "",
        "This analysis preserves the exact q values and event membership used in the manuscript.",
        "No q value was refitted and no event was added or removed.",
        "",
        "| Ecosystem | Events | Sites | Median Delta q | Site-cluster 95% CI | Delta q < 0 | Site-cluster 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.ecosystem} | {row.n_events:,} | {row.n_sites:,} | "
            f"{row.median_delta_q:.3f} | [{row.median_delta_q_cluster_low:.3f}, "
            f"{row.median_delta_q_cluster_high:.3f}] | {row.negative_fraction:.1%} | "
            f"[{row.negative_fraction_cluster_low:.1%}, {row.negative_fraction_cluster_high:.1%}] |"
        )
    lines += [
        "",
        "These intervals quantify monitoring-site sampling uncertainty. They do not quantify within-fit q uncertainty, because the historical q tables contain only the selected point estimate and not the alternative fits, residual draws or covariance information.",
        "",
        "## Separate q-original versus q-post relationship diagnostic",
        "",
        "Spearman correlation is a rank-persistence diagnostic, not an uncertainty estimate for Delta q.",
        "",
        "| Ecosystem | Spearman rho | Site-cluster 95% CI |",
        "|---|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.ecosystem} | {row.q_original_q_post_spearman:.3f} | "
            f"[{row.q_original_q_post_spearman_cluster_low:.3f}, "
            f"{row.q_original_q_post_spearman_cluster_high:.3f}] |"
        )
    (args.output_dir / "PAPER_COHORT_UNCERTAINTY_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
