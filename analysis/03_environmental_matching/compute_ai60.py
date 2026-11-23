#!/usr/bin/env python3
"""Attach Methods-defined AI60 to unique antecedent events.

AI60 = total precipitation / total PET over the 60 calendar days immediately
preceding the initial drydown start; the start day itself is excluded.
Missing site-days are rejected rather than treated as zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def attach_ai60(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    site = "site_id" if "site_id" in result else "SITE_ID"
    start = "initial_start" if "initial_start" in result else "start"
    required_events = {site, start, "file", "event_id"}
    missing = required_events - set(result)
    if missing:
        raise ValueError(f"antecedent events missing columns: {sorted(missing)}")
    if result.duplicated(["file", "event_id"]).any():
        raise ValueError("antecedent input has duplicated (file,event_id) keys")
    result[start] = pd.to_datetime(result[start], errors="coerce").dt.normalize()
    if result[start].isna().any() or result[site].isna().any():
        raise ValueError("antecedent input has missing start date or site ID")

    required_daily = {"site_id", "date", "precip", "pet"}
    missing = required_daily - set(daily)
    if missing:
        raise ValueError(f"daily forcing missing columns: {sorted(missing)}")
    forcing = daily[list(required_daily)].copy()
    forcing["date"] = pd.to_datetime(forcing["date"], errors="coerce").dt.normalize()
    for field in ("precip", "pet"):
        forcing[field] = pd.to_numeric(forcing[field], errors="coerce")
    if forcing[["site_id", "date", "precip", "pet"]].isna().any().any():
        raise ValueError("daily forcing contains missing site/date/precip/PET")
    if forcing.duplicated(["site_id", "date"]).any():
        raise ValueError("daily forcing must be unique by site and date")
    if forcing[["precip", "pet"]].lt(0).any().any():
        raise ValueError("daily forcing contains negative precipitation or PET")
    by_site = {site_id: frame.set_index("date").sort_index()
               for site_id, frame in forcing.groupby("site_id", sort=False)}

    precipitation, pet_total = [], []
    for row in result.itertuples(index=False):
        site_id = getattr(row, site)
        t0 = getattr(row, start)
        expected = pd.date_range(t0 - pd.Timedelta(days=60), periods=60, freq="D")
        frame = by_site.get(site_id)
        if frame is None or not expected.isin(frame.index).all():
            raise ValueError(f"incomplete 60-day forcing for {row.file}/{row.event_id}")
        selected = frame.loc[expected]
        precipitation.append(float(selected["precip"].sum()))
        pet_total.append(float(selected["pet"].sum()))
    result["P60"] = precipitation
    result["PET60"] = pet_total
    if result["PET60"].le(0).any():
        raise ValueError("PET60 must be positive for an AI60 ratio")
    result["AI60"] = result["P60"] / result["PET60"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--daily-forcing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = attach_ai60(pd.read_csv(args.events), pd.read_csv(args.daily_forcing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out):,} unique antecedent events to {args.output}")


if __name__ == "__main__":
    main()
