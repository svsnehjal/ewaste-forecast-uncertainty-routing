# Independent experimental audit — status report

## Completed independent reruns
The current calibrated project code was executed independently for:
- `main`: fixed baselines and i.i.d. uncertainty sweep.
- `errmodels`: seven forecast-error structures.
- `sigmahat`: estimated-uncertainty pass-through and misestimation.

Outputs were written to `independent_rerun/`, separate from archived `results/`.

## Critical finding: archived tables did not reproduce
The independent reruns have matching row counts and column structures for the archived `main.csv`, `errmodels.csv`, `fixed_train_J.csv`, and `tuned_params_iid.csv`, but their numeric values differ. See `independent_comparison.csv`.

This is a substantive mismatch, not rounding noise. The current code uses the calibrated `params.yaml` configuration, including a CPCB-based scenario scale and an assumed network share of 0.0001. The archived tables represent a different code/configuration state and cannot be presented as independently reproduced by the current calibrated project.

## Stages not completed in this audit
- `pareto`: attempted, but did not finish within the available execution window.
- `ablation`, `heterogeneity`, and `scale`: not independently completed.
- `solvers`: not run. `pyvrp` and `ortools` are absent in this environment; see `dependency_audit.json`.
Archived outputs for these stages remain archived evidence, not new independent replications.

## Forecasting extension
The separately completed calibrated forecasting experiment uses independent training seeds 5000–5059, validation seeds 5500–5519, and test seeds 6000–6019. Its metrics are simulator-derived diagnostics, not real-world forecast accuracy. Vehicle-limited observations were excluded from supervised fitting (complete-case treatment), which can induce selection bias.

## Calibration caveat
The CPCB Tamil Nadu annual generation estimate is a scale anchor only. The network share 0.0001 is an explicit scenario assumption, not a measured share of statewide generation. No point-level operational collection logs, measured bin capacities, or actual route records are included.

## Publication decision
This package is a research-development and audit bundle, **not a fully independently reproduced submission package**. Do not claim all experiments were rerun, that external solvers were benchmarked in this audit, or that the simulator was validated against local operational data.

## Required next steps
1. Freeze one configuration and code version.
2. Rerun every stage to completion from a clean environment.
3. Regenerate every manuscript table and figure from those outputs.
4. Install and record exact PyVRP/OR-Tools versions before solver comparisons.
5. Publish the repository and attach an immutable release/commit identifier before stating a public code URL.
