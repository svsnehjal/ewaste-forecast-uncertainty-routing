# E-waste collection under forecast uncertainty — simulation code

Reproduces every table and figure in the paper.

    pip install -r requirements.txt
    bash run_all.sh            # ~45 min on one CPU core; writes per-instance CSVs and figures to $EW_OUT (default ./out)

Files: `sim.py` (simulator, savings+2-opt, policies, error models, PyVRP/OR-Tools wrappers), `forecasters.py` (features, MA/RF/XGBoost/numpy-LSTM, closed-loop retraining),
`experiments.py` (all stages), `analysis.py` (tables, paired t / Wilcoxon, Holm, break-evens, figures), `params.yaml` (all parameters and seeds),
`load_real_logs.py` (template for plugging in real collection logs).

Seeds: tuning/training seeds 0-29; test seeds 1000-1029; large-instance seeds 2000+; solver seeds 3000+; forecaster training 5000-5059; forecaster held-out 6000-6019.
Common random numbers: generation, initial stock and forecast-error draws depend only on the instance seed, so policies are compared on identical realisations.
Repository URL / DOI: <insert after upload to GitHub or Zenodo>.


## Semi-real external calibration (added)

Run `real_data_calibration.py` to create:
- `results/external_reference_statistics.csv`
- `results/calibrated_network_scenarios.csv`
- `results/calibration_sources_and_method.json`

External references are kept separate by definition:
- CPCB/SPCB Tamil Nadu figures are statewide e-waste generation inventory estimates.
- TNPCB figures are industry-reported quantities collected/processed and materials recovered.
- MoEFCC/PIB figures are quantities processed by registered recyclers.

The network scenarios apply explicit illustrative shares (1e-8, 1e-7, 1e-6)
to the statewide annual estimate. These shares are modelling assumptions,
not measured proportions. The resulting point-level demand is synthetic.
The calibration does not claim access to daily campus, municipal or recycler
manifests, and does not independently validate routing performance.

Official references:
1. CPCB report in NGT proceedings:
   https://www.greentribunal.gov.in/sites/default/files/news_updates/Report%20on%20behalf%20of%20CPCB%20in%20EA%20NO%2004%20of%202024%20IN%20OA%20No.%20512%20of%202018%20%28Shailesh%20Singh%20Vs.%20State%20of%20U.P.%20%26%20Ors.%29.pdf
2. TNPCB Annual Report 2023-24:
   https://tnpcb.gov.in/PDF/Information/annualreports/English/AnnualRpt_Eng2024.pdf
3. MoEFCC parliamentary reply, 24 July 2025:
   https://www.pib.gov.in/PressReleasePage.aspx?PRID=2147876&lang=2&reg=48


## Calibrated main experiment (kg units)

The simulator demand parameters are now connected to the CPCB Tamil Nadu FY2023-24 statewide generation estimate (448,878 MT/year) using an explicitly assumed network share of 1e-4 (0.01%). This yields a synthetic 30-point network target of 44,887.8 kg/year or 122.9803 kg/day. The point-level median generation rate, bin capacity, and vehicle capacities are all scaled by the same kg-per-original-unit factor, preserving the original relative demand/capacity structure.

**This is a scenario calibration, not a measured local network.** The 1e-4 share is an assumption because the statewide source does not report the share attributable to this hypothetical 30-point network. The sensitivity generator now includes shares 1e-6, 1e-5, and 1e-4.

Run the calibrated fixed-parameter main comparison by opening/running `run_calibrated_main.py`. It evaluates 30 held-out seeds (1000-1029), the eight fixed-schedule baselines, and Naive/Buffered policies at forecast-error sigma 0, 0.3, 0.5, and 1.0. It does not tune theta/a/k; therefore do not describe these results as tuned-policy results. Outputs are saved under `calibrated_results/`: per-instance CSV, summary CSV, paired-test CSV with Holm correction, run configuration, and a figure.

The calibrated simulator mass-balance identity was checked for every evaluated run. Maximum absolute numerical error was approximately 3.1e-11 kg.

### Output provenance warning
The pre-existing files under `results/` were generated under the earlier uncalibrated parameterization unless their filenames explicitly say otherwise. For the calibrated fixed-parameter main comparison, use only files under `calibrated_results/`. The calibrated main run does not rerun the separate error-model, sigma-estimation, Pareto, ablation, large-scale solver, or ML-forecaster stages; those require separate calibrated-unit audits before being treated as calibrated evidence.
