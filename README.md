# Water-use strategies after rainfall

Code accompanying *Intermittent rainfall shifts ecosystems towards more aggressive water use, strongest in savannas*.

How does ecosystem water use change when rain interrupts a soil-moisture drydown? We compare two drydowns recorded by the same sensor, one before rewetting and one after it. The fitted exponent **q** describes how quickly water use declines as soil dries. We calculate **Δq = q<sub>post</sub> − q<sub>original</sub>** for each pair. A negative value indicates a shift towards more aggressive water use during the second drydown.

## Event detection and analysis

| Stage | What the code does |
| --- | --- |
| [Initial drydowns and q](analysis/01_initial_drydown_q/) | Detect sustained declines in daily soil moisture and fit the nonlinear drydown curve. The included fitting wrapper uses a fixed ET ceiling. The antecedent analysis also draws on the [upstream drydowns implementation](https://github.com/ecohydro/drydowns). |
| [Rewetting and redrying](analysis/02_rewetting_redrying/) | Locate rainfall-associated rewetting and subsequent drying, pair the two estimates of q, and summarize Δq. The 7-, 14- and 21-day windows specify the search for rewetting. |
| [Environmental matching](analysis/03_environmental_matching/) | Match sensor events with precipitation, potential evapotranspiration (PET), ecosystem metadata and antecedent wetness. |
| [Event-level associations](analysis/04_state_dependent_models/fig3_event_models.py) | Fit drydown rate, moisture buffering and PET₉₀ together within each ecosystem and antecedent-q quartile. Marginal Pearson correlations are also calculated for comparison. |
| [Ecosystem summaries and uncertainty](analysis/06_q_uncertainty/analyze_paper_cohort_uncertainty.py) | Summarize the paired-event cohort using the median Δq and fraction below zero. Resampling monitoring sites keeps events from the same site together. The intervals describe site-sampling uncertainty conditional on the fitted q values. |

## Manuscript figures

| Figure | Entry point | Inputs and output |
| --- | --- | --- |
| 1 | [Sequence frequencies and Δq](figures/make_fig1_unique_event_counts.py) | Run the [unique-event frequency summary](figures/recompute_fig1b_unique_counts.py) first. The figure also uses event tables and the schematic image. |
| 2 | [Antecedent wetness and vegetation](figures/make_fig2_hydroclim_veg_backgrounds.py) | Uses the antecedent drydown table to plot q across wetness and LAI gradients. |
| 3 | [State-dependent associations](figures/make_dq21_qstate_drydown_buffer_pet_effects_9panel.py) | Uses the paired-event cohort, process metrics and daily site PET. Writes standardized joint-model coefficients, site-bootstrap intervals, event and site counts, and a separate correlation table. |
| 4 | [Savanna response surfaces](figures/make_savanna_4x3_final_from_existing.py) | Renders previously calculated surface grids alongside summaries of observed binned medians. |

## Running the scripts

Install the Python dependencies from the repository root:

```sh
python -m pip install -r requirements.txt
```

This is a code-only repository. Daily soil-moisture records, environmental inputs and derived manuscript tables must be supplied locally. Readers working with prepared event tables can start directly with the ecosystem summaries or Fig. 3. For Fig. 3, a record is identified by sensor file, event ID and search window.

```sh
python analysis/06_q_uncertainty/analyze_paper_cohort_uncertainty.py --help
python figures/make_dq21_qstate_drydown_buffer_pet_effects_9panel.py --help
```

For Fig. 3, supply `--paper-cohort`, `--process-events` and `--pet-dir`. Use `--output-dir` and `--manuscript-png` to choose output locations. For figures that read `WATER_USE_DATA_ROOT`, set it to the study data directory before running. Earlier processing scripts use script-level input and output paths. Check those paths and the expected table columns before running each stage.
