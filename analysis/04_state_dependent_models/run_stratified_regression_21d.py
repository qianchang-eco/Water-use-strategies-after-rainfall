#!/usr/bin/env python3
"""Fit the 21-day process model separately within qoriginal quartiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


GROUPS = ["Q1_low", "Q2_midlow", "Q3_midhigh", "Q4_high"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-events", type=int, default=500)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    required = [
        "dq_21", "drydown_rate_21", "SWC_first", "theta_star_first",
        "PET_p90_21", "q_original",
    ]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["moisture_buffering"] = df["SWC_first"] - df["theta_star_first"]
    df = df.dropna(subset=required).copy()
    quantiles = df["q_original"].quantile([0, .25, .5, .75, 1]).values
    df["q_group"] = pd.cut(
        df["q_original"], bins=quantiles, include_lowest=True, labels=GROUPS,
    )

    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        sub = df[df["q_group"].eq(group)].copy()
        if len(sub) < args.min_events:
            print(f"Skipping {group}: {len(sub)} events")
            continue
        model = smf.ols(
            "dq_21 ~ drydown_rate_21 + moisture_buffering + PET_p90_21",
            data=sub,
        ).fit()
        (args.output_dir / f"summary_{group}.txt").write_text(
            model.summary().as_text(), encoding="utf-8"
        )
        for term in ["drydown_rate_21", "moisture_buffering", "PET_p90_21"]:
            rows.append({
                "q_group": group,
                "predictor": term,
                "coefficient": model.params[term],
                "standard_error": model.bse[term],
                "p_value": model.pvalues[term],
                "R2": model.rsquared,
                "n_events": int(model.nobs),
            })
    pd.DataFrame(rows).to_csv(args.output_dir / "stratified_coefficients.csv", index=False)


if __name__ == "__main__":
    main()
