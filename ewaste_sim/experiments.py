"""All experiments. Usage: python experiments.py <stage> ...   Outputs CSVs to ../out (per-instance rows)."""
import sys, time, itertools, pickle, os, numpy as np, pandas as pd
from dataclasses import replace
from sim import *
from forecasters import *

OUT = os.environ.get('EW_OUT', '/home/claude/out')
TRAIN = list(range(0, 30)); TEST = list(range(1000, 1030)); FC_TRAIN = list(range(5000, 5060)); FC_TEST = list(range(6000, 6020))
SIGS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]
CL_TUNE = 0.5     # penalty per lost unit used as the tuning objective
GRID = list(itertools.product([0.6, 0.7, 0.8, 0.9], [0.0, 0.3], [0.0, 0.4, 0.8]))   # theta, a, k

def J(M, cl=CL_TUNE): return M['cost'] + cl * M['lost']

def evalp(factory, seeds, P, tag, solver=None, record_fc=False, **extra):
    rows = []
    for s in seeds:
        inst = Instance(s, P); pol = factory(); M = run(inst, pol, P, solver=solver, record_fc=record_fc)
        M.update(seed=s, policy=pol.name if tag is None else tag, **extra); rows.append(M)
    return rows

def fixed_candidates():
    return {'Fixed-6d-halves-nominal0.6B': lambda: FixedPolicy(2, 'halves', nominal=0.6),
            'Fixed-6d-halves': lambda: FixedPolicy(2, 'halves'),
            'Fixed-3d-all': lambda: FixedPolicy(1, 'all'),
            'Fixed-6d-all': lambda: FixedPolicy(2, 'all'),
            'Fixed-9d-thirds': lambda: FixedPolicy(3, 'halves'),
            'Fixed-9d-all': lambda: FixedPolicy(3, 'all'),
            'Fixed-rate-tuned(0.6)': lambda: FixedPolicy(rate_theta=0.6),
            'Fixed-rate-tuned(0.8)': lambda: FixedPolicy(rate_theta=0.8),
            'Fixed-rate-tuned(1.0)': lambda: FixedPolicy(rate_theta=1.0)}

def tune(err_kind, sigma, P, seeds=TRAIN, sigma_mode='true', dispatch=None, forecaster=None):
    """Grid-search (theta,a,k) on training seeds by mean J. Returns best params and table."""
    Pm = replace(P, dispatch=dispatch) if dispatch else P
    best = None; tab = []
    for th, a, k in GRID:
        if sigma == 0 and (a, k) != (0.0, 0.0): continue
        rows = []
        for s in seeds:
            inst = Instance(s, Pm)
            pol = ForecastPolicy(th, a, k, err=ErrModel(err_kind, sigma) if forecaster is None else None, forecaster=forecaster, sigma_mode=sigma_mode)
            rows.append(J(run(inst, pol, Pm)))
        m = float(np.mean(rows)); tab.append((th, a, k, m))
        if best is None or m < best[3]: best = (th, a, k, m)
    return best, tab

def stage_main():
    P = Params(); allrows = []; t0 = time.time()
    # fixed baselines: train (selection) and test
    fx = fixed_candidates(); trainJ = {}
    for nm, f in fx.items():
        r = evalp(f, TRAIN, P, nm, sigma=np.nan, split='train', errkind='-'); trainJ[nm] = np.mean([J(m) for m in r]); allrows += r
        allrows += evalp(f, TEST, P, nm, sigma=np.nan, split='test', errkind='-')
    best_fixed = min([k for k in trainJ if 'nominal' not in k], key=lambda k: trainJ[k]); print('best fixed on train:', best_fixed, trainJ, flush=True)
    pd.DataFrame([dict(policy=k, trainJ=v) for k, v in trainJ.items()]).assign(best=lambda d: d.policy == best_fixed).to_csv(f'{OUT}/fixed_train_J.csv', index=False)
    tuned = []
    for sg in SIGS:
        er = ErrModel('iid', sg)
        allrows += evalp(lambda: ForecastPolicy(0.85, 0, 0, err=er), TEST, P, 'Naive', sigma=sg, split='test', errkind='iid')
        allrows += evalp(lambda: ForecastPolicy(0.85, 0.3, 0.8, err=er), TEST, P, 'Buffered', sigma=sg, split='test', errkind='iid')
        b, tab = tune('iid', sg, P); tuned.append(dict(sigma=sg, theta=b[0], a=b[1], k=b[2], trainJ=b[3]))
        allrows += evalp(lambda: ForecastPolicy(b[0], b[1], b[2], err=er), TEST, P, 'Tuned', sigma=sg, split='test', errkind='iid')
        allrows += evalp(lambda: ForecastPolicy(b[0], b[1], b[2], err=er), TRAIN, P, 'Tuned', sigma=sg, split='train', errkind='iid')
        print('sigma', sg, 'done', round(time.time() - t0), flush=True)
    pd.DataFrame(tuned).to_csv(f'{OUT}/tuned_params_iid.csv', index=False)
    pd.DataFrame(allrows).to_csv(f'{OUT}/main.csv', index=False)

def stage_errmodels():
    P = Params(); rows = []; t0 = time.time()
    for kind in ['iid', 'bias_under', 'bias_over', 'bias_point', 'ar1', 'xcorr', 't']:
        for sg in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            er = ErrModel(kind, sg)
            rows += evalp(lambda: ForecastPolicy(0.85, 0, 0, err=er), TEST, P, 'Naive', sigma=sg, errkind=kind)
            rows += evalp(lambda: ForecastPolicy(0.85, 0.3, 0.8, err=er), TEST, P, 'Buffered', sigma=sg, errkind=kind)
        print(kind, round(time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/errmodels.csv', index=False)

def stage_sigmahat():
    P = Params(); rows = []
    for kind in ['iid', 'bias_under', 'ar1']:
        for sg in [0.3, 0.5, 1.0]:
            er = ErrModel(kind, sg)
            for mode, mult in [('true', 1), ('est', 1), ('mult', 0.0), ('mult', 0.5), ('mult', 1.5), ('mult', 2.0)]:
                tag = f'Buffered[{mode}{"" if mode!="mult" else "x"+str(mult)}]'
                rows += evalp(lambda: ForecastPolicy(0.85, 0.3, 0.8, err=er, sigma_mode=mode, sig_mult=mult), TEST, P, tag, sigma=sg, errkind=kind, mode=mode, mult=mult)
    pd.DataFrame(rows).to_csv(f'{OUT}/sigmahat.csv', index=False)

def fc_one(kind):
    P = Params(); rows = []; D0 = collect_training(FC_TRAIN, P); Dt = collect_training(FC_TEST, P)
    obs = np.expm1(Dt['y']); m = Dt['q'] > (0.05 * P.B)
    f, D = train_closed_loop(kind, P, rounds=2, train_seeds=FC_TRAIN, D0=D0)
    if kind == 'MA': p = np.minimum(P.B, Dt['X'][:, 8] * Dt['X'][:, 0])
    else: p = f.predict_rows(Dt, B=P.B)
    lr = np.log((p[m] + 1) / (Dt['q'][m] + 1))
    acc = dict(model=kind, MAE_obs=np.mean(np.abs(p - obs)), RMSE_obs=np.sqrt(np.mean((p - obs) ** 2)), MAE_true=np.mean(np.abs(p - Dt['q'])),
               logRMSE_true=np.sqrt(np.mean(lr ** 2)), logbias_true=np.mean(lr), rows=len(D['y']) if kind != 'MA' else 0)
    for nm, (th, a, k) in {'Naive': (0.85, 0, 0), 'Buffered': (0.85, 0.3, 0.8)}.items():
        rows += evalp(lambda: ForecastPolicy(th, a, k, forecaster=f, sigma_mode='est'), TEST, P, f'{kind}-{nm}', record_fc=True, model=kind, variant=nm)
    best = None
    for th, a, k in GRID:
        jv = np.mean([J(run(Instance(s, P), ForecastPolicy(th, a, k, forecaster=f, sigma_mode='est'), P)) for s in TRAIN[:12]])
        if best is None or jv < best[3]: best = (th, a, k, jv)
    rows += evalp(lambda: ForecastPolicy(best[0], best[1], best[2], forecaster=f, sigma_mode='est'), TEST, P, f'{kind}-Tuned', record_fc=True, model=kind, variant='Tuned')
    pd.DataFrame(rows).to_csv(f'{OUT}/fc_rows_{kind}.csv', index=False); pd.DataFrame([acc]).to_csv(f'{OUT}/fc_acc_{kind}.csv', index=False)
    pd.DataFrame([dict(model=kind, theta=best[0], a=best[1], k=best[2])]).to_csv(f'{OUT}/fc_tuned_{kind}.csv', index=False)

def stage_fcMA(): fc_one('MA')
def stage_fcXGB(): fc_one('XGB')
def stage_fcRF(): fc_one('RF')
def stage_fcLSTM(): fc_one('LSTM')
def stage_fcmerge():
    ks = ['MA', 'XGB', 'RF', 'LSTM']; ks = [k for k in ks if os.path.exists(f'{OUT}/fc_rows_{k}.csv')]
    pd.concat([pd.read_csv(f'{OUT}/fc_rows_{k}.csv') for k in ks]).to_csv(f'{OUT}/forecasters.csv', index=False)
    pd.concat([pd.read_csv(f'{OUT}/fc_acc_{k}.csv') for k in ks]).to_csv(f'{OUT}/forecaster_accuracy.csv', index=False)
    pd.concat([pd.read_csv(f'{OUT}/fc_tuned_{k}.csv') for k in ks]).to_csv(f'{OUT}/forecaster_tuned.csv', index=False)

def stage_scale():
    rows = []; t0 = time.time()
    for N, reps in [(50, 15), (100, 8), (200, 4)]:
        P = make_params(N=N); seeds = list(range(2000, 2000 + reps)); er = ErrModel('iid', 0.5)
        fx = {'Fixed-6d-halves': lambda: FixedPolicy(2, 'halves'), 'Fixed-3d-all': lambda: FixedPolicy(1, 'all'), 'Fixed-rate-tuned(0.8)': lambda: FixedPolicy(rate_theta=0.8)}
        for nm, f in fx.items(): rows += evalp(f, seeds, P, nm, N=N, K=P.K, sigma=0.5, setting='scale')
        rows += evalp(lambda: ForecastPolicy(0.85, 0, 0, err=er), seeds, P, 'Naive', N=N, K=P.K, sigma=0.5, setting='scale')
        rows += evalp(lambda: ForecastPolicy(0.85, 0.3, 0.8, err=er), seeds, P, 'Buffered', N=N, K=P.K, sigma=0.5, setting='scale')
        rows += evalp(lambda: ForecastPolicy(0.7, 0.0, 0.0, err=er), seeds, P, 'Naive(theta=0.7)', N=N, K=P.K, sigma=0.5, setting='scale')
        print('scale', N, round(time.time() - t0), flush=True)
    # depots and heterogeneous fleets at N=30
    P0 = Params(); er = ErrModel('iid', 0.5)
    for dep in ['centre', 'corner', 'outside']:
        P = replace(P0, depot=dep)
        for nm, f in {'Fixed-6d-halves': lambda: FixedPolicy(2, 'halves'), 'Naive': lambda: ForecastPolicy(0.85, 0, 0, err=er), 'Buffered': lambda: ForecastPolicy(0.85, 0.3, 0.8, err=er)}.items():
            rows += evalp(f, TEST, P, nm, N=30, sigma=0.5, setting=f'depot-{dep}')
    for nm_f, Qs in {'4x300': (300.,) * 4, '2x200+2x400': (200., 200., 400., 400.), '1x100+1x200+2x450': (100., 200., 450., 450.)}.items():
        P = replace(P0, Qs=Qs, K=4)
        for nm, f in {'Fixed-6d-halves': lambda: FixedPolicy(2, 'halves'), 'Naive': lambda: ForecastPolicy(0.85, 0, 0, err=er), 'Buffered': lambda: ForecastPolicy(0.85, 0.3, 0.8, err=er)}.items():
            rows += evalp(f, TEST, P, nm, N=30, sigma=0.5, setting=f'fleet-{nm_f}')
    pd.DataFrame(rows).to_csv(f'{OUT}/scale.csv', index=False)

def stage_solvers():
    rows = []; er = ErrModel('iid', 0.5)
    for N, reps, tl in [(30, 10, 0.15), (100, 4, 0.4)]:
        P = replace(make_params(N=N), dispatch='trunc'); seeds = list(range(3000, 3000 + reps))
        for solver in ['cw2opt', 'pyvrp', 'ortools']:
            if solver == 'ortools' and N > 30: continue
            for nm, (th, a, k) in {'Naive': (0.85, 0, 0), 'Buffered': (0.85, 0.3, 0.8)}.items():
                t0 = time.time()
                for s in seeds:
                    inst = Instance(s, P); pol = ForecastPolicy(th, a, k, err=er); M = run(inst, pol, P, solver=solver, ctx_extra=dict(tl=tl))
                    M.update(seed=s, policy=nm, solver=solver, N=N, secs=time.time() - t0); rows.append(M)
                print('solver', N, solver, nm, round(time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/solvers.csv', index=False)

def stage_pareto():
    P = Params(); rows = []; er = ErrModel('iid', 0.5)
    for th, a, k in itertools.product([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95], [0.0, 0.3], [0.0, 0.4, 0.8, 1.2]):
        rows += evalp(lambda: ForecastPolicy(th, a, k, err=er), TEST, P, f'F(th={th},a={a},k={k})', family='forecast', theta=th, a=a, k=k, sigma=0.5)
    for nm, f in fixed_candidates().items(): rows += evalp(f, TEST, P, nm, family='fixed', sigma=np.nan)
    pd.DataFrame(rows).to_csv(f'{OUT}/pareto.csv', index=False)

def stage_ablation():
    rows = []; er = ErrModel('iid', 0.5)
    for Q, ratio in [(200., 0.98), (250., 1.22), (300., 1.47), (400., 1.96)]:
        for disp in ['drop', 'reinsert', 'trunc', 'unlimited']:
            P = replace(Params(), Qs=(Q,) * 4, dispatch=disp)
            for nm, f in {'Fixed-6d-halves-nominal0.6B': lambda: FixedPolicy(2, 'halves', nominal=0.6), 'Fixed-6d-halves': lambda: FixedPolicy(2, 'halves'),
                          'Naive': lambda: ForecastPolicy(0.85, 0, 0, err=er), 'Buffered': lambda: ForecastPolicy(0.85, 0.3, 0.8, err=er),
                          'Buffered-k0(threshold only)': lambda: ForecastPolicy(0.85, 0.3, 0.0, err=er), 'Buffered-a0(load only)': lambda: ForecastPolicy(0.85, 0.0, 0.8, err=er)}.items():
                rows += evalp(f, TEST, P, nm, Q=Q, ratio=ratio, dispatch=disp, sigma=0.5)
        print('ablation Q', Q, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/ablation.csv', index=False)

def stage_heterogeneity():
    rows = []; er = ErrModel('iid', 0.5)
    for sl in [0.25, 0.5, 1.0]:
        P = replace(Params(), sig_ln=sl)
        for nm, f in {'Fixed-6d-halves': lambda: FixedPolicy(2, 'halves'), 'Fixed-rate-tuned(0.8)': lambda: FixedPolicy(rate_theta=0.8), 'Naive': lambda: ForecastPolicy(0.85, 0, 0, err=er), 'Buffered': lambda: ForecastPolicy(0.85, 0.3, 0.8, err=er)}.items():
            rows += evalp(f, TEST, P, nm, sig_ln=sl, sigma=0.5)
    pd.DataFrame(rows).to_csv(f'{OUT}/heterogeneity.csv', index=False)

if __name__ == '__main__':
    for st in sys.argv[1:]:
        t0 = time.time(); globals()['stage_' + st](); print('STAGE', st, 'done in', round(time.time() - t0), 's', flush=True)
