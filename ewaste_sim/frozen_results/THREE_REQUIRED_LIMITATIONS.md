# Three required manuscript limitations

1. **CPCB scale anchor:** CPCB Tamil Nadu FY2023–24 statewide generation (448,878 MT/year) is only an external scale anchor. The 0.0001 (0.01%) network share is an explicit scenario assumption, not measured local data.
2. **External solvers:** PyVRP and OR-Tools comparisons have not been completed. Only the in-house CW2OPT subset is reported; no external solver comparison or superiority claim is supported.
3. **Forecaster alignment:** MA, RF, XGBoost and LSTM have not been retrained and reevaluated under the same frozen configuration as the mechanism-level reruns. Their earlier results must remain separate and must not be pooled or directly compared as one unified experiment until retrained with disjoint train/validation/test seeds.

The manuscript also states that these limitations restrict the scope of reproducibility claims.
