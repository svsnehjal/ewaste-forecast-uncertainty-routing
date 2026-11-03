# Seed protocol and experiment provenance

All random seeds are explicit in `experiments.py`, `run_calibrated_main.py`, and `forecasters.py`. The simulator uses NumPy `default_rng` with independent streams derived from each instance seed (`[seed, 11]` for geometry/demand/initial stock and `[seed, 22]` for forecast error; the forecaster policy uses `[seed, 33]`).

| Experiment | Seed IDs | Role / qualification |
|---|---:|---|
| Main mechanism tuning/selection | 0–29 | Training/selection instances; not the held-out test set. |
| Main mechanism evaluation | 1000–1029 | 30 held-out test instances. |
| Error-model comparison | 1000–1029 | Same held-out instance seeds across policies and error models. |
| Sigma-estimation comparison | 1000–1029 | Same held-out instance seeds. |
| Heterogeneity | 1000–1029 | Same held-out instance seeds. |
| Scale: N=50 | 2000–2014 | 15 instances. |
| Scale: N=100 | 2000–2007 | 8 instances. |
| Scale: N=200 | 2000–2003 | 4 instances. |
| Depot/fleet sensitivity | 1000–1029 | 30 held-out seeds. |
| Solver experiment source protocol | 3000–3009 for N=30; 3000–3003 for N=100 | The completed frozen archive contains only the in-house CW2OPT subset; PyVRP and OR-Tools comparisons are not complete. |
| Earlier forecaster training | 5000–5059 | Separate prior forecasting pipeline; not retrained/aligned to the frozen mechanism configuration. |
| Earlier forecaster test | 6000–6019 | Separate prior forecasting pipeline; do not pool with the frozen mechanism test results. |
| RQ3 paired comparison | 1000–1029 | Paired by seed; 30 held-out instances. |

## Reproducibility cautions

1. The 0–29 / 1000–1029 split applies to the mechanism-level tuning and evaluation protocol.
2. The older forecaster pipeline uses a separate seed family and is not a completed frozen-configuration comparison.
3. Solver seed ranges describe the source protocol, not a completed external-solver benchmark.
4. The random streams are common across policy comparisons where the same instance seed is reused.
