#!/usr/bin/env python3
"""Build one ecosystem's 21-day post-rewetting process table.

This replaces the former ecosystem-specific copies.  Run once per ecosystem,
for example: ``python build_rewet_process_table_21d.py --ecosystem savanna``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_pet_series(pet_dir: Path, site_id: str) -> pd.DataFrame | None:
    path = pet_dir / f"{site_id}.csv"
    if not path.exists():
        return None
    pet = pd.read_csv(path, usecols=["date", "pet"])
    pet["date"] = pd.to_datetime(pet["date"])
    pet["pet"] = numeric(pet["pet"])
    return pet.dropna()


def build_table(
    ecosystem: str,
    rewet_csv: Path,
    first_csv: Path,
    pet_dir: Path,
    output_csv: Path,
) -> None:
    rewet = pd.read_csv(rewet_csv)
    rewet = rewet[rewet["ecosystem"].eq(ecosystem)].copy()

    required = {
        "file", "event_id", "SITE_ID", "q_original", "q_21", "q_21_start", "q_21_end",
        "q_21_theta_peak", "q_21_theta_end", "q_21_theta_jump", "dpet_21",
    }
    missing = sorted(required - set(rewet.columns))
    if missing:
        raise ValueError(f"Missing columns in rewet table: {missing}")

    for col in ["q_21_start", "q_21_end"]:
        rewet[col] = pd.to_datetime(rewet[col])
    for col in [
        "q_original", "q_21", "q_21_theta_peak", "q_21_theta_end",
        "q_21_theta_jump", "dpet_21",
    ]:
        rewet[col] = numeric(rewet[col])

    rewet["duration_21"] = (rewet["q_21_end"] - rewet["q_21_start"]).dt.days
    rewet["dq_21"] = rewet["q_21"] - rewet["q_original"]
    rewet["drydown_rate_21"] = rewet["q_21_theta_jump"] / rewet["duration_21"]

    first = pd.read_csv(
        first_csv,
        usecols=["file", "event_id", "ecosystem", "theta_star", "SWC", "ai"],
    )
    first = first[first["ecosystem"].eq(ecosystem)].copy()
    key = ["file", "event_id"]
    values = ["theta_star", "SWC", "ai"]
    for column in values:
        conflicts = first.groupby(key, dropna=False)[column].nunique(dropna=True)
        if (conflicts > 1).any():
            raise ValueError(f"Conflicting {column} for {(conflicts > 1).sum()} antecedent events")
    first = first.rename(columns={
        "theta_star": "theta_star_first",
        "SWC": "SWC_first",
        "ai": "ai_first",
    }).drop_duplicates(key)
    for col in ["theta_star_first", "SWC_first", "ai_first"]:
        first[col] = numeric(first[col])
    before_join = len(rewet)
    rewet = rewet.merge(first.drop(columns=["ecosystem"]), on=key, how="left", validate="many_to_one")
    if len(rewet) != before_join:
        raise AssertionError("Antecedent-state join changed the number of rewetting rows")
    rewet["exposure_depth_21"] = np.maximum(
        0, rewet["theta_star_first"] - rewet["q_21_theta_end"]
    )

    p90, pmax = [], []
    for row in rewet.itertuples(index=False):
        pet = read_pet_series(pet_dir, str(row.SITE_ID))
        if pet is None:
            p90.append(np.nan)
            pmax.append(np.nan)
            continue
        window = pet[(pet["date"] >= row.q_21_start) & (pet["date"] <= row.q_21_end)]
        p90.append(window["pet"].quantile(0.9) if not window.empty else np.nan)
        pmax.append(window["pet"].max() if not window.empty else np.nan)
    rewet["PET_p90_21"] = p90
    rewet["PET_max_21"] = pmax

    keep = [
        "file", "event_id", "SITE_ID", "q_21_start", "q_21_end",
        "q_original", "q_21", "dq_21", "duration_21",
        "drydown_rate_21", "exposure_depth_21", "dpet_21", "PET_p90_21",
        "PET_max_21", "theta_star_first", "SWC_first", "ai_first",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rewet[keep].to_csv(output_csv, index=False)
    print(f"Saved {len(rewet):,} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecosystem", choices=["forest", "savanna", "grassland"], required=True)
    parser.add_argument("--rewet", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--pet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_table(args.ecosystem, args.rewet, args.first, args.pet_dir, args.output)


if __name__ == "__main__":
    main()
