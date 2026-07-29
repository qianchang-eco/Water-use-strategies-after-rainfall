#!/usr/bin/env python3
"""Fit the full 21-day post-rewetting process model for one ecosystem."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    for col in [
        "dq_21", "drydown_rate_21", "q_original", "SWC_first",
        "theta_star_first", "PET_p90_21",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["moisture_buffering"] = df["SWC_first"] - df["theta_star_first"]
    df = df.dropna(subset=[
        "dq_21", "drydown_rate_21", "moisture_buffering", "PET_p90_21", "q_original",
    ]).copy()

    df["qoriginal_c"] = df["q_original"] - df["q_original"].mean()
    df["drydown_rate_c"] = df["drydown_rate_21"] - df["drydown_rate_21"].mean()
    df["qoriginal_x_drydown"] = df["qoriginal_c"] * df["drydown_rate_c"]
    formula = (
        "dq_21 ~ drydown_rate_c + moisture_buffering + PET_p90_21 "
        "+ qoriginal_c + qoriginal_x_drydown"
    )
    model = smf.ols(formula, data=df).fit()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
    coefficients = pd.DataFrame({
        "term": model.params.index,
        "coefficient": model.params.values,
        "standard_error": model.bse.values,
        "p_value": model.pvalues.values,
    })
    coefficients["R2"] = model.rsquared
    coefficients["n_events"] = int(model.nobs)
    coefficients.to_csv(args.output_dir / "model_coefficients.csv", index=False)
    print(f"R2={model.rsquared:.3f}; events={int(model.nobs):,}")


if __name__ == "__main__":
    main()
