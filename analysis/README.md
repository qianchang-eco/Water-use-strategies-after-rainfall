# Analysis workflow

Run the numbered stages in order. Each stage consumes the table produced by
the previous stage and adds one conceptual layer to the event record.

`01_initial_drydown_q` establishes the antecedent drydown and its fitted
strategy parameter. `02_rewetting_redrying` identifies the post-drought window
and computes the change in strategy. `03_environmental_matching` joins the
hydroclimatic and ecosystem covariates. `04_state_dependent_models` builds the
21 d process tables and estimates the state-dependent response models.

The original server workflow produced intermediate CSV files between stages.
Those files are intentionally excluded from this repository; use local paths
in `config/paths.env` when reproducing the analysis.
