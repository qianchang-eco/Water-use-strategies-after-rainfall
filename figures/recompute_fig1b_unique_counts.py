"""Recompute Fig. 1b occurrence counts at the unique-event grain.

An event is identified by (file, event_id).  The denominator is the number of
unique initial events in each ecosystem.  A detected event for a search window
has at least one non-missing post-rewetting q value for that event/window.
"""

import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
SOURCE = ROOT / "rewet_q_results_time_pet.csv"
OUTPUT = ROOT / "plots" / "rewet_window_ecosystem_unique_event_summary.csv"

ECOSYSTEMS = ("forest", "grassland", "savanna")
WINDOW_COLUMNS = {"7d": "q_7", "14d": "q_14", "21d": "q_21"}


def main() -> None:
    usecols = ["file", "event_id", "ecosystem", *WINDOW_COLUMNS.values()]
    event_flags: dict[tuple[str, str], dict[str, object]] = {}

    for chunk in pd.read_csv(SOURCE, usecols=usecols, chunksize=500_000, low_memory=False):
        chunk["ecosystem"] = chunk["ecosystem"].astype("string").str.lower()
        chunk = chunk[chunk["ecosystem"].isin(ECOSYSTEMS)].copy()
        if chunk.empty:
            continue

        for column in WINDOW_COLUMNS.values():
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

        # Collapse each chunk first; then OR detection flags across chunks.
        grouped = (
            chunk.assign(
                detected_7d=chunk["q_7"].notna(),
                detected_14d=chunk["q_14"].notna(),
                detected_21d=chunk["q_21"].notna(),
            )
            .groupby(["file", "event_id", "ecosystem"], dropna=False, sort=False)[
                ["detected_7d", "detected_14d", "detected_21d"]
            ]
            .any()
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            key = (str(row.file), str(row.event_id))
            current = event_flags.get(key)
            if current is None:
                event_flags[key] = {
                    "ecosystem": str(row.ecosystem),
                    "7d": bool(row.detected_7d),
                    "14d": bool(row.detected_14d),
                    "21d": bool(row.detected_21d),
                }
            else:
                if current["ecosystem"] != str(row.ecosystem):
                    raise ValueError(f"Conflicting ecosystem labels for event {key}")
                for window in WINDOW_COLUMNS:
                    current[window] = bool(current[window] or getattr(row, f"detected_{window}"))

    events = pd.DataFrame.from_records(
        [{"file": key[0], "event_id": key[1], **value} for key, value in event_flags.items()]
    )

    rows = []
    for ecosystem in ECOSYSTEMS:
        subset = events[events["ecosystem"].eq(ecosystem)]
        total = len(subset)
        for window in WINDOW_COLUMNS:
            detected = int(subset[window].sum())
            rows.append(
                {
                    "ecosystem": ecosystem,
                    "window": window,
                    "N_total_events": total,
                    "N_detected": detected,
                    "occurrence_frequency": detected / total if total else float("nan"),
                }
            )

    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(OUTPUT)


if __name__ == "__main__":
    main()
