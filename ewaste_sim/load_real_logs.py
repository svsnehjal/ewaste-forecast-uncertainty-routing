"""Template: calibrate the simulator from real collection logs (CSV: point_id,date,quantity_kg,bin_full,truck_full).
Estimates per-point mean rate (censoring-aware: bin_full visits give lower bounds) and dispersion; writes lam.csv for Instance(lam=...)."""
import pandas as pd, numpy as np, sys
df = pd.read_csv(sys.argv[1], parse_dates=['date']).sort_values(['point_id', 'date'])
df['gap'] = df.groupby('point_id').date.diff().dt.days
d = df.dropna(subset=['gap']); d = d[(d.bin_full == 0) & (d.truck_full == 0)]
lam = d.groupby('point_id').apply(lambda g: g.quantity_kg.sum() / g.gap.sum())
print('median rate', lam.median(), 'sigma_ln', np.log(lam).std()); lam.to_csv('lam.csv')
