# E-waste Forecast Uncertainty and Capacity-Constrained Routing

Simulation code and experiment outputs for the study *Impact of Forecast Uncertainty on Capacity-Constrained E-Waste Collection Routing: A Comparative Simulation Study*.

## Calibration limitation
The CPCB Tamil Nadu FY2023–24 statewide generation estimate (448,878 MT/year) is used only as an external scale anchor. The network share of 0.0001 (0.01%) is an explicit scenario assumption, not a measured local or statewide allocation. All point-level generation, bin and fleet values derived from it are scenario-calibrated, not local operational measurements.

## Solver limitation
PyVRP and OR-Tools comparisons have not been completed. The reported solver subset uses the in-house CW2OPT implementation only. Do not claim comparative superiority over PyVRP or OR-Tools.

## Forecasting alignment limitation
The earlier MA, RF, XGBoost and LSTM forecasting results were not retrained and evaluated under the same frozen configuration as the mechanism-level experiments. Keep those as a separate experiment set. Do not pool or directly compare them as if they shared one protocol. A unified result requires disjoint training/validation/test seeds and retraining under the frozen configuration.

## Reproducibility status
The frozen rerun covers main uncertainty, seven error models, sigma-hat, heterogeneity, scale, ablation and Pareto analyses. It also includes a limited in-house CW2OPT solver subset. External solver comparisons and frozen-configuration forecaster retraining remain outstanding. See `frozen_results/RUN_COMPLETION_STATUS.json`.

## Environment and execution
Install dependencies with `python -m pip install -r requirements.txt`.
Run simulation stages with `EW_OUT=./out python experiments.py main errmodels sigmahat heterogeneity scale ablation pareto`.
External solver comparisons require separately installing and version-pinning PyVRP and OR-Tools. Forecasters must be retrained with the same frozen parameters and disjoint seeds before being integrated into the manuscript's main result set.

## Data availability
No point-level operational collection dataset is included. Point-level demand, fill, capacity and route data are simulated.
