"""Attach antecedent and post-rewetting PET without changing event cardinality.

The source q table may contain repeated sensor-depth metadata rows. These are
collapsed only after checking that SITE_ID and antecedent PET agree for each
``(file, event_id)``. Never join it to rewetting events by ``file`` alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["file", "event_id"]
VALID_ECOSYSTEMS = {"forest", "grassland", "savanna"}


def antecedent_pet_lookup(qdf: pd.DataFrame) -> pd.DataFrame:
    required = KEY + ["SITE_ID", "pet"]
    missing = sorted(set(required) - set(qdf.columns))
    if missing:
        raise ValueError(f"Missing q-table columns: {missing}")
    lookup = qdf[required].copy()
    if lookup[KEY].isna().any().any():
        raise ValueError("Null file/event_id in antecedent q table")
    for column in ["SITE_ID", "pet"]:
        conflicts = lookup.groupby(KEY, dropna=False)[column].nunique(dropna=True)
        if (conflicts > 1).any():
            raise ValueError(f"Conflicting {column} values for {(conflicts > 1).sum()} events")
    lookup = lookup.sort_values(["SITE_ID", "pet"], na_position="last", kind="stable").drop_duplicates(KEY)
    lookup["pet"] = pd.to_numeric(lookup["pet"], errors="coerce")
    # ISMN/forcing fill values are not plausible daily PET (mm d-1).
    lookup.loc[lookup["pet"].abs() > 1e6, "pet"] = np.nan
    return lookup


def attach_antecedent_pet(rewet: pd.DataFrame, qdf: pd.DataFrame) -> pd.DataFrame:
    if rewet[KEY].isna().any().any():
        raise ValueError("Null file/event_id in rewetting table")
    for column in ["SITE_ID", "pet"]:
        if column in rewet:
            raise ValueError(f"Rewetting input already contains {column}; refusing to overwrite")
    lookup = antecedent_pet_lookup(qdf)
    joined = rewet.merge(lookup, on=KEY, how="left", validate="many_to_one", sort=False)
    if len(joined) != len(rewet):
        raise AssertionError("Antecedent PET join changed the number of rewetting rows")
    return joined


def add_post_rewet_pet(rewet: pd.DataFrame, pet_dir: Path) -> pd.DataFrame:
    result = rewet.copy()
    cache: dict[str, pd.DataFrame | None] = {}

    def load_pet(site_id: str) -> pd.DataFrame | None:
        if site_id not in cache:
            path = pet_dir / f"{site_id}.csv"
            if not path.exists():
                cache[site_id] = None
            else:
                frame = pd.read_csv(path, usecols=["date", "pet"])
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                frame["pet"] = pd.to_numeric(frame["pet"], errors="coerce")
                frame.loc[frame["pet"].abs() > 1e6, "pet"] = np.nan
                cache[site_id] = frame.dropna(subset=["date"])
        return cache[site_id]

    for window in (7, 14, 21):
        start_col, end_col = f"q_{window}_start", f"q_{window}_end"
        result[start_col] = pd.to_datetime(result[start_col], errors="coerce")
        result[end_col] = pd.to_datetime(result[end_col], errors="coerce")
        values = []
        for site_id, start, end in result[["SITE_ID", start_col, end_col]].itertuples(index=False, name=None):
            if pd.isna(site_id) or pd.isna(start) or pd.isna(end):
                values.append(np.nan)
                continue
            pet = load_pet(str(site_id))
            if pet is None:
                values.append(np.nan)
                continue
            selected = pet.loc[pet["date"].between(start, end), "pet"]
            values.append(selected.mean() if not selected.empty else np.nan)
        result[f"pet_{window}"] = values
        result[f"dpet_{window}"] = result["pet"] - result[f"pet_{window}"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewet", type=Path, required=True)
    parser.add_argument("--q-table", type=Path, required=True)
    parser.add_argument("--pet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rewet = pd.read_csv(args.rewet, low_memory=False)
    rewet = rewet.loc[rewet["ecosystem"].isin(VALID_ECOSYSTEMS)].copy()
    qdf = pd.read_csv(args.q_table, usecols=KEY + ["SITE_ID", "pet"], low_memory=False)
    joined = attach_antecedent_pet(rewet, qdf)
    print(f"Event rows before/after metadata join: {len(rewet):,} / {len(joined):,}")
    result = add_post_rewet_pet(joined, args.pet_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()
