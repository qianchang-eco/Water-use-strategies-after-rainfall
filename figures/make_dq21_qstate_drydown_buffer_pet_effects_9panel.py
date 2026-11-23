#!/usr/bin/env python3
"""Generate the manuscript Fig. 3 from unique events and a joint model.

The independent event key is ``file + event_id + window_days``. Within each
ecosystem x antecedent-state stratum, all three predictors enter the same
fully standardized OLS model::

    z(delta_q21) ~ z(drydown_rate) + z(moisture_buffering) + z(PET90)

Whiskers are percentile 95% confidence intervals from a site-cluster
bootstrap. A sampled site contributes all of its events, so repeated events
from a site are not treated as independent during uncertainty estimation.

The script writes an auditable analysis bundle under ``plots/fig3_final`` and
copies the resulting 600-dpi PNG to the manuscript Figures directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_STUDY_ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
DEFAULT_PAPER_COHORT = (
    DEFAULT_STUDY_ROOT / "q_uncertainty_paper_cohort" / "paper_cohort_with_site_id.csv.gz"
)
DEFAULT_PROCESS_EVENTS = (
    DEFAULT_STUDY_ROOT / "derived_extracted_cohort_20260930" / "extracted_paired_long.csv"
)
DEFAULT_PET_DIR = PROJECT_ROOT / "code" / "pet" / "csv"
DEFAULT_OUTPUT_DIR = DEFAULT_STUDY_ROOT / "plots" / "fig3_final"
DEFAULT_MANUSCRIPT_PNG = DEFAULT_STUDY_ROOT / "提交" / "Figures" / "Fig. 3.png"

ANALYSIS_SOURCE = (
    PROJECT_ROOT / "analysis" / "04_state_dependent_models" / "fig3_event_models.py"
)
OUTPUT_STEM = "Fig3_unique_events_joint_model_site_cluster_CI"


def load_analysis_module():
    """Load the tested event reconstruction, model and plotting functions."""
    spec = spec_from_file_location("fig3_unique_event_joint_model", ANALYSIS_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load analysis module: {ANALYSIS_SOURCE}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-cohort", type=Path, default=DEFAULT_PAPER_COHORT)
    parser.add_argument("--process-events", type=Path, default=DEFAULT_PROCESS_EVENTS)
    parser.add_argument("--pet-dir", type=Path, default=DEFAULT_PET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manuscript-png", type=Path, default=DEFAULT_MANUSCRIPT_PNG)
    parser.add_argument("--bootstrap-draws", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws < 1000:
        raise ValueError("Use at least 1,000 bootstrap draws for the manuscript figure")
    for path in (args.paper_cohort, args.process_events, args.pet_dir, ANALYSIS_SOURCE):
        if not path.exists():
            raise FileNotFoundError(path)

    analysis = load_analysis_module()
    data, cohort_audit = analysis.load_analysis_cohort(
        args.paper_cohort, args.process_events, args.pet_dir
    )
    if data.duplicated(analysis.KEY).any():
        raise AssertionError("Analysis cohort is not unique by file + event_id + window_days")

    marginal = analysis.calculate_correlations(data, args.bootstrap_draws)
    partial = analysis.calculate_partial_effects(data, marginal, args.bootstrap_draws)
    if not partial["status"].eq("estimated").all():
        failed = partial.loc[~partial["status"].eq("estimated")]
        raise RuntimeError(f"Some model strata failed estimation:\n{failed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outbase = args.output_dir / OUTPUT_STEM
    analysis.draw_partial(partial, outbase)

    data.to_csv(args.output_dir / "Fig3_unique_event_analysis_cohort.csv", index=False)
    partial.to_csv(args.output_dir / "Fig3_joint_model_coefficients.csv", index=False)
    marginal.to_csv(args.output_dir / "Fig3_marginal_correlations_for_direction_check.csv", index=False)
    support = (
        data.groupby(["ecosystem", "R_state"], observed=True)
        .agg(
            events=("delta_q", "size"),
            sites=("SITE_ID", "nunique"),
            complete_joint_model=("PET90", lambda x: int(x.notna().sum())),
        )
        .reset_index()
    )
    support.to_csv(args.output_dir / "Fig3_event_site_support.csv", index=False)

    args.manuscript_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(outbase.with_suffix(".png"), args.manuscript_png)

    manifest = {
        "figure": "Fig. 3",
        "event_key": analysis.KEY,
        "analysis_unit": "unique event",
        "model": "fully standardized OLS within ecosystem x R state: z(delta_q21) ~ z(drydown_rate) + z(moisture_buffering) + z(PET90)",
        "uncertainty": {
            "method": "site-cluster percentile bootstrap",
            "cluster": "SITE_ID",
            "draws": args.bootstrap_draws,
            "interval": "95%",
            "seed_base": analysis.SEED,
        },
        "R_state_definition": "four equal-count strata of q_original within ecosystem",
        "cohort_audit": cohort_audit,
        "support": support.to_dict(orient="records"),
        "formal_png": str(args.manuscript_png),
        "analysis_figure": str(outbase.with_suffix(".png")),
    }
    (args.output_dir / "Fig3_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(args.manuscript_png)
    print(support.to_string(index=False))


if __name__ == "__main__":
    main()
