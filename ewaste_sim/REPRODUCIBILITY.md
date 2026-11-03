# Reproducibility and data availability

## Included
- Simulator and experiment source: `sim.py`, `experiments.py`, `analysis.py`.
- Human-readable physical-unit configuration snapshot: `params.yaml`.
- Frozen per-instance experiment outputs, summary tables, and status record: `frozen_results/`.
- Calibrated fixed-parameter main-run outputs: `calibrated_results/`.
- Seed assignments: `SEED_PROTOCOL.md`.
- Dependencies: `requirements.txt`.

## Important configuration detail
`params.yaml` is a human-readable configuration snapshot; the current Python implementation does **not** parse it at runtime. The operative defaults and calibration constants are defined in `sim.py`, while experiment seed ranges, grids, and policies are defined in `experiments.py` and `run_calibrated_main.py`. For exact reproduction, use those source files as the executable specification and compare them with `params.yaml`. Do not edit only the YAML and assume the simulation changes.

## Run
Create a Python environment compatible with the installed package versions, install `requirements.txt`, then run the desired stage from the project directory. For example, set `EW_OUT=./out` and run `python experiments.py main errmodels sigmahat heterogeneity scale ablation pareto`, followed by `python analysis.py` with the output paths configured as in the scripts. The frozen CSVs are included so the archived reported analyses can be inspected without rerunning all experiments.

The full `run_all.sh` also invokes solver and forecaster stages. Those stages are not all complete in the frozen evidence bundle; external PyVRP/OR-Tools comparisons and retraining MA/RF/XGBoost/LSTM under the frozen configuration remain outstanding.

## Calibration and interpretation
The CPCB Tamil Nadu FY2023–24 statewide estimate (448,878 MT/year) is used only as a scale anchor. The 0.0001 network share is an explicit scenario assumption, not measured local allocation. No point-level operational collection dataset is included; point-level demand, stock, fill, and routes are simulated.

## Outputs
The `frozen_results/` directory contains per-instance CSVs for the completed mechanism-level stages and derived summary tables. `RUN_COMPLETION_STATUS.json` is the authoritative completion/limitation record. The `calibrated_results/` directory contains the separate calibrated fixed-parameter main comparison. These result families must not be silently pooled.

## Public release status
This directory is a GitHub/Zenodo-ready release package. It has **not** been uploaded to GitHub or Zenodo, and therefore has no public repository URL or DOI yet. After creating the public repository, replace the repository/DOI placeholders in the manuscript with the actual persistent links.
