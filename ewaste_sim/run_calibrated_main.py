"""Reproducible main policy comparison under the CPCB-scaled kg scenario.
Uses 30 held-out test seeds; this is a fixed-parameter comparison, not a tuning run.
"""
import os
import numpy as np
import pandas as pd
from sim import Params, Instance, FixedPolicy, ForecastPolicy, ErrModel, run

OUT = os.environ.get('EW_OUT', './calibrated_results')
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, 'fig'), exist_ok=True)
TEST_SEEDS = list(range(1000, 1030))
SIGMAS = [0.0, 0.3, 0.5, 1.0]
P = Params()
rows = []

fixed = {
    'Fixed-6d-halves': lambda: FixedPolicy(2, 'halves'),
    'Fixed-3d-all': lambda: FixedPolicy(1, 'all'),
    'Fixed-6d-all': lambda: FixedPolicy(2, 'all'),
    'Fixed-9d-thirds': lambda: FixedPolicy(3, 'halves'),
    'Fixed-9d-all': lambda: FixedPolicy(3, 'all'),
    'Fixed-rate-tuned-0.6': lambda: FixedPolicy(rate_theta=0.6),
    'Fixed-rate-tuned-0.8': lambda: FixedPolicy(rate_theta=0.8),
    'Fixed-rate-tuned-1.0': lambda: FixedPolicy(rate_theta=1.0),
}

for seed in TEST_SEEDS:
    inst = Instance(seed, P)
    for name, factory in fixed.items():
        m = run(inst, factory(), P)
        m.update(seed=seed, policy=name, sigma=np.nan, scenario='CPCB-share-1e-4')
        rows.append(m)
    for sigma in SIGMAS:
        err = ErrModel('iid', sigma)
        for name, theta, a, k in [('Naive', 0.85, 0.0, 0.0), ('Buffered', 0.85, 0.3, 0.8)]:
            m = run(inst, ForecastPolicy(theta, a, k, err=err), P)
            m.update(seed=seed, policy=name, sigma=sigma, scenario='CPCB-share-1e-4')
            rows.append(m)
    if (seed - TEST_SEEDS[0] + 1) % 5 == 0:
        print(f'Completed {seed - TEST_SEEDS[0] + 1}/{len(TEST_SEEDS)} test instances', flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'calibrated_main_per_instance.csv'), index=False)

metrics = ['dist', 'stops', 'collected', 'lost', 'stock1', 'coll_pct', 'lost_pct', 'resid_pct', 'cpu', 'cost', 'gen']
summary = df.groupby(['policy', 'sigma'], dropna=False)[metrics].agg(['mean', 'std']).reset_index()
summary.to_csv(os.path.join(OUT, 'calibrated_main_summary.csv'), index=False)

# Paired test comparisons by identical random seed, with Holm correction.
from scipy import stats
comparisons = []
for sigma in SIGMAS:
    for metric in ['lost_pct', 'dist', 'cpu', 'coll_pct']:
        a = df[(df.policy == 'Buffered') & np.isclose(df.sigma, sigma)].set_index('seed')[metric]
        b = df[(df.policy == 'Naive') & np.isclose(df.sigma, sigma)].set_index('seed')[metric]
        d = (a - b).dropna().values
        t_p = float(stats.ttest_1samp(d, 0).pvalue)
        w_p = float(stats.wilcoxon(d).pvalue) if not np.allclose(d, 0) else 1.0
        comparisons.append(dict(sigma=sigma, metric=metric, A='Buffered', B='Naive',
                                mean_A=float(a.mean()), mean_B=float(b.mean()),
                                mean_paired_difference=float(d.mean()),
                                paired_t_p=t_p, wilcoxon_p=w_p, n=len(d)))

# Holm adjustment across this family of 16 paired tests, separately per test type.
def holm(vals):
    vals = np.asarray(vals, float); order = np.argsort(vals); out = np.empty(len(vals)); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(vals) - rank) * vals[idx])
        out[idx] = min(1.0, running)
    return out
for pcol in ['paired_t_p', 'wilcoxon_p']:
    adjusted = holm([r[pcol] for r in comparisons])
    for r, adj in zip(comparisons, adjusted):
        r[pcol + '_holm'] = float(adj)
pd.DataFrame(comparisons).to_csv(os.path.join(OUT, 'calibrated_paired_tests.csv'), index=False)

# Save the exact configuration used in this run.
config = pd.DataFrame([{
    'scenario': 'CPCB-share-1e-4', 'source_statewide_generation_MT_FY2023_24': 448878.0,
    'assumed_network_share': 1e-4, 'network_annual_kg': 44887.8,
    'network_daily_kg_target': 44887.8 / 365.0, 'N': P.N, 'B_kg': P.B,
    'K': P.K, 'vehicle_capacity_kg_each': P.Qs[0], 'median_point_rate_kg_day': P.med_rate,
    'lognormal_sigma': P.sig_ln, 'warmup_days': P.warm_days,
    'measurement_days': P.horizon_days, 'cycle_days': P.C,
    'test_seeds': '1000-1029', 'forecast_error_sigmas': ','.join(map(str, SIGMAS)),
    'tuning': 'No tuning in this fixed-parameter calibrated comparison'
}])
config.to_csv(os.path.join(OUT, 'calibrated_run_config.csv'), index=False)
print('Saved calibrated main experiment outputs to', OUT)
print('Maximum mass-balance absolute error:', float((df.stock0 + df.gen - df.collected - df.lost - df.stock1).abs().max()))
