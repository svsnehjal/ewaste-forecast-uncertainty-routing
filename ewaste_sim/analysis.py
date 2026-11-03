"""Turns per-instance CSVs into tables, tests, break-evens, figures -> out/results.json"""
import json, os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.environ.get('EW_OUT', '/home/claude/out'); FIG = f'{OUT}/fig'
CL = 0.5
plt.rcParams.update({'font.size': 8, 'figure.dpi': 200, 'axes.grid': True, 'grid.alpha': .3})
R = {}
def ci(x):
    x = np.asarray(x, float); return float(stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
def ms(x, nd=3): return f'{np.mean(x):.{nd}f}±{ci(x):.{nd}f}'
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
def addJ(d):
    d = d.copy(); d['J'] = d.cost + CL * d.lost; d['Jgen'] = d.J / d.gen; d['cgen'] = d.cost / d.gen; return d
def piv(d, pol, col, **flt):
    x = d[d.policy == pol]
    for k, v in flt.items(): x = x[np.isclose(x[k], v)] if isinstance(v, float) else x[x[k] == v]
    return x.sort_values('seed').set_index('seed')[col]
def paired(d, A, B, col, fa=None, fb=None):
    a = piv(d, A, col, **(fa or {})); b = piv(d, B, col, **(fb or {})); a, b = a.align(b, join='inner'); diff = (a - b).values
    t = stats.ttest_1samp(diff, 0); 
    w = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(diff).pvalue
    return dict(A=A, B=B, col=col, ma=float(a.mean()), mb=float(b.mean()), diff=float(diff.mean()), ci=ci(diff), pct=float(100 * (a.mean() / b.mean() - 1)) if b.mean() else np.nan, pt=float(t.pvalue), pw=float(w))
def crossing(x, y, ref):
    """first sigma at which y exceeds ref (linear interpolation); None if never, 0 if above at start"""
    x = np.asarray(x); y = np.asarray(y)
    if y[0] > ref: return 0.0
    for i in range(1, len(x)):
        if y[i] > ref >= y[i - 1]: return float(x[i - 1] + (ref - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
    return None

def main_tables():
    d = addJ(pd.read_csv(f'{OUT}/main.csv')); t = d[d.split == 'test']
    best = pd.read_csv(f'{OUT}/fixed_train_J.csv'); BF = best[best.best].policy.iloc[0]; FO = 'Fixed-6d-halves-nominal0.6B'
    R['best_fixed'] = BF
    rows = []
    for pol in [FO, 'Fixed-6d-halves', 'Fixed-3d-all', 'Fixed-6d-all', 'Fixed-9d-thirds', 'Fixed-9d-all', 'Fixed-rate-tuned(0.6)', 'Fixed-rate-tuned(0.8)', 'Fixed-rate-tuned(1.0)']:
        x = t[t.policy == pol]
        rows.append(dict(policy=pol, sigma='-', dist=ms(x.dist, 0), stops=f'{x.stops.mean():.0f}', coll=ms(x.coll_pct, 1), lost=ms(x.lost_pct, 1), resid=ms(x.resid_pct, 1), cpu=ms(x.cpu), fill=f'{x.fill_visit.mean():.0f}', J=f'{x.Jgen.mean():.4f}', n=len(x)))
    R['fixed_rows'] = rows
    rows = []
    for pol in ['Naive', 'Buffered', 'Tuned']:
        for s in [0, .1, .2, .3, .4, .5, .6, .75, 1.0]:
            x = t[(t.policy == pol) & np.isclose(t.sigma, s)]
            rows.append(dict(policy=pol, sigma=s, dist=ms(x.dist, 0), stops=f'{x.stops.mean():.0f}', coll=ms(x.coll_pct, 1), lost=ms(x.lost_pct, 1), resid=ms(x.resid_pct, 1), cpu=ms(x.cpu), fill=f'{x.fill_visit.mean():.0f}', J=f'{x.Jgen.mean():.4f}'))
    R['fc_rows'] = rows
    # identity check
    chk = (t.collected + t.lost + t.stock1 - t.stock0 - t.gen).abs().max(); R['identity_max_abs_err'] = float(chk)
    # fill-at-visit table
    R['fill_tab'] = {pol: {str(s): float(t[(t.policy == pol) & np.isclose(t.sigma, s)].fill_visit.mean()) for s in [0, .3, .5, 1.0]} for pol in ['Naive', 'Buffered', 'Tuned']}
    R['fill_tab']['Fixed-orig'] = float(t[t.policy == FO].fill_visit.mean()); R['fill_tab']['Fixed-best'] = float(t[t.policy == BF].fill_visit.mean())
    # RQ1 numbers
    nv0 = t[(t.policy == 'Naive') & np.isclose(t.sigma, 0)]; fo = t[t.policy == FO]; fb = t[t.policy == BF]
    R['rq1'] = {k: dict(orig=float(fo[k].mean()), best=float(fb[k].mean()), fc=float(nv0[k].mean())) for k in ['stops', 'dist', 'cpu', 'lost_pct', 'fill_visit']}
    R['rq1_tests'] = [paired(t, 'Naive', FO, c, fa=dict(sigma=0.0)) for c in ['stops', 'dist', 'cpu', 'lost_pct']] + [paired(t, 'Naive', BF, c, fa=dict(sigma=0.0)) for c in ['stops', 'dist', 'cpu', 'lost_pct']]
    # RQ3 comparison family
    fam = []
    for s in [0.3, 0.5, 1.0]:
        for A, B, fb_ in [('Buffered', 'Naive', dict(sigma=s)), ('Buffered', FO, None), ('Buffered', BF, None), ('Naive', FO, None), ('Naive', BF, None)]:
            for c in ['dist', 'cpu', 'lost_pct']:
                r = paired(t, A, B, c, fa=dict(sigma=s), fb=fb_); r['sigma'] = s; fam.append(r)
    for nm in ['pt', 'pw']:
        adj = holm([r[nm] for r in fam])
        for r, a in zip(fam, adj): r[nm + '_holm'] = float(a)
    R['rq3_family'] = fam
    # break-evens iid
    sg = [0, .1, .2, .3, .4, .5, .6, .75, 1.0]; be = {}
    for pol in ['Naive', 'Buffered', 'Tuned']:
        c = t[t.policy == pol].groupby('sigma')
        for refn, ref in [('orig', FO), ('best', BF)]:
            be[f'{pol}|loss|{refn}'] = crossing(sg, c.lost_pct.mean().values, fo.lost_pct.mean() if refn == 'orig' else fb.lost_pct.mean())
            be[f'{pol}|J|{refn}'] = crossing(sg, c.Jgen.mean().values, (fo if refn == 'orig' else fb).Jgen.mean())
            be[f'{pol}|cpu|{refn}'] = crossing(sg, c.cpu.mean().values, (fo if refn == 'orig' else fb).cpu.mean())
    R['breakeven_iid'] = be
    # train vs test for tuned
    tr = d[(d.split == 'train') & (d.policy == 'Tuned')]; te = t[t.policy == 'Tuned']
    R['tuned_train_test'] = [dict(sigma=s, train_J=float(tr[np.isclose(tr.sigma, s)].Jgen.mean()), test_J=float(te[np.isclose(te.sigma, s)].Jgen.mean()),
                                  train_loss=float(tr[np.isclose(tr.sigma, s)].lost_pct.mean()), test_loss=float(te[np.isclose(te.sigma, s)].lost_pct.mean())) for s in sg]
    R['tuned_params'] = pd.read_csv(f'{OUT}/tuned_params_iid.csv').to_dict('records')
    R['fixed_train_J'] = best.to_dict('records')
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for k, (col, yl) in enumerate([('lost_pct', 'Overflow loss (% of stock+generated)'), ('cpu', 'Cost per collected unit'), ('Jgen', f'Cost + {CL}·lost, per generated unit')]):
        for pol, m, ls in [('Naive', 'o', '-'), ('Buffered', 's', '--'), ('Tuned', '^', ':')]:
            g = t[t.policy == pol].groupby('sigma')[col]; mean = g.mean(); h = g.apply(ci)
            ax[k].errorbar(mean.index, mean.values, yerr=h.values, marker=m, ms=3, ls=ls, lw=1, capsize=1.5, label=pol)
        ax[k].axhline(fo[col].mean(), color='k', ls='-.', lw=.8, label='Fixed (paper baseline)'); ax[k].axhline(fb[col].mean(), color='r', ls='-.', lw=.8, label='Fixed (best tuned)')
        ax[k].set_xlabel('σ (i.i.d. lognormal)'); ax[k].set_ylabel(yl)
    ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(f'{FIG}/fig_main.png'); plt.close()

def err_tables():
    d = addJ(pd.read_csv(f'{OUT}/errmodels.csv')); m = addJ(pd.read_csv(f'{OUT}/main.csv')); m = m[m.split == 'test']
    BF = R['best_fixed']; FO = 'Fixed-6d-halves-nominal0.6B'
    fo = m[m.policy == FO]; fb = m[m.policy == BF]; sg = [0, .2, .4, .6, .8, 1.0]; rows = []
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for k, pol in enumerate(['Naive', 'Buffered']):
        for kind in ['iid', 'bias_under', 'bias_over', 'bias_point', 'ar1', 'xcorr', 't']:
            x = d[(d.policy == pol) & (d.errkind == kind)].groupby('sigma'); l = x.lost_pct.mean(); j = x.Jgen.mean()
            ax[k].plot(l.index, l.values, marker='o', ms=2.5, lw=1, label=kind)
            rows.append(dict(policy=pol, kind=kind, loss05=float(l.loc[0.4] * .5 + l.loc[0.6] * .5), loss10=float(l.loc[1.0]), loss0=float(l.loc[0.0]),
                             be_loss_orig=crossing(sg, l.values, fo.lost_pct.mean()), be_loss_best=crossing(sg, l.values, fb.lost_pct.mean()),
                             be_J_orig=crossing(sg, j.values, fo.Jgen.mean()), be_J_best=crossing(sg, j.values, fb.Jgen.mean())))
        ax[k].axhline(fo.lost_pct.mean(), color='k', ls='-.', lw=.8, label='Fixed (paper)'); ax[k].axhline(fb.lost_pct.mean(), color='r', ls='-.', lw=.8, label='Fixed (tuned)')
        ax[k].set_title(pol, fontsize=8); ax[k].set_xlabel('σ'); ax[k].set_ylabel('Overflow loss (%)')
    ax[0].legend(fontsize=5.5); plt.tight_layout(); plt.savefig(f'{FIG}/fig_errmodels.png'); plt.close()
    R['err_rows'] = rows
    # paired test iid vs each model at sigma .6 (buffered & naive loss)
    R['err_tests'] = []
    for pol in ['Naive', 'Buffered']:
        for kind in ['bias_under', 'bias_point', 'ar1', 'xcorr', 't']:
            r = paired(d, pol, pol, 'lost_pct', fa=dict(sigma=0.6, errkind=kind), fb=dict(sigma=0.6, errkind='iid')); r['kind'] = kind; r['policy'] = pol; R['err_tests'].append(r)
    for nm in ['pt', 'pw']:
        adj = holm([r[nm] for r in R['err_tests']])
        for r, a in zip(R['err_tests'], adj): r[nm + '_holm'] = float(a)

def sigmahat_table():
    d = addJ(pd.read_csv(f'{OUT}/sigmahat.csv')); rows = []
    for (kind, s), g in d.groupby(['errkind', 'sigma']):
        for pol, h in g.groupby('policy'):
            rows.append(dict(kind=kind, sigma=s, mode=pol, loss=float(h.lost_pct.mean()), cpu=float(h.cpu.mean()), Jgen=float(h.Jgen.mean()), sig_used=float(h.sig_used.mean()), dist=float(h.dist.mean())))
    R['sigmahat_rows'] = rows

def pareto():
    d = addJ(pd.read_csv(f'{OUT}/pareto.csv')); g = d.groupby('policy').agg(cgen=('cgen', 'mean'), loss=('lost_pct', 'mean'), family=('family', 'first'), cpu=('cpu', 'mean'),
                                                                          cost=('cost', 'mean'), lost=('lost', 'mean'), gen=('gen', 'mean')).reset_index()
    pts = g[['cgen', 'loss']].values; front = []
    for i, p in enumerate(pts):
        if not any((q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]) for q in pts): front.append(i)
    g['pareto'] = False; g.loc[front, 'pareto'] = True
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for fam, c, mk in [('forecast', 'tab:blue', 'o'), ('fixed', 'tab:red', 's')]:
        x = g[g.family == fam]; ax.scatter(x.loss, x.cgen, c=c, marker=mk, s=12, label=f'{fam} (σ=0.5)' if fam == 'forecast' else 'fixed schedules', alpha=.7)
    f = g[g.pareto].sort_values('loss'); ax.step(f.loss, f.cgen, where='post', color='k', lw=.8, label='Pareto front')
    ax.set_xlabel('Overflow loss (%)'); ax.set_ylabel('Cost per generated unit'); ax.set_xlim(0, 12); ax.set_ylim(0.12, 0.2); ax.legend(fontsize=6); plt.tight_layout(); plt.savefig(f'{FIG}/fig_pareto.png'); plt.close()
    R['pareto_front'] = f[['policy', 'cgen', 'loss', 'family']].to_dict('records')
    # penalty sweep: lower envelope of J(cl)=cgen+cl*lost/gen per family
    cls = [0, .1, .25, .5, 1, 2, 5]; rows = []
    for cl in cls:
        g['Jc'] = (g.cost + cl * g.lost) / g.gen
        fc = g[g.family == 'forecast'].sort_values('Jc').iloc[0]; fx = g[g.family == 'fixed'].sort_values('Jc').iloc[0]
        rows.append(dict(cl=cl, fc_best=fc.policy, fc_J=float(fc.Jc), fx_best=fx.policy, fx_J=float(fx.Jc), gain_pct=float(100 * (fc.Jc / fx.Jc - 1))))
    R['penalty_sweep'] = rows
    # break-even penalty: smallest cl at which best forecast config <= best fixed
    cgrid = np.linspace(0, 6, 601); wins = []
    for cl in cgrid:
        Jc = (g.cost + cl * g.lost) / g.gen
        if Jc[g.family == 'forecast'].min() < Jc[g.family == 'fixed'].min(): wins.append(float(cl))
    R['penalty_win_cl'] = [min(wins), max(wins)] if wins else None
    R['penalty_win_n'] = len(wins)

def ablation():
    d = addJ(pd.read_csv(f'{OUT}/ablation.csv')); rows = []
    for (Q, disp, pol), g in d.groupby(['Q', 'dispatch', 'policy']):
        rows.append(dict(Q=Q, ratio=float(g.ratio.iloc[0]), dispatch=disp, policy=pol, loss=float(g.lost_pct.mean()), loss_ci=ci(g.lost_pct), cpu=float(g.cpu.mean()), dist=float(g.dist.mean()),
                         drop_share=float(100 * g.dropped.sum() / max(g.sel.sum(), 1)), routes=float(g.routes.mean())))
    R['ablation_rows'] = rows
    tests = []
    for Q in [200., 250., 300., 400.]:
        for pol in ['Naive', 'Buffered', 'Fixed-6d-halves-nominal0.6B']:
            for disp in ['reinsert', 'trunc', 'unlimited']:
                r = paired(d, pol, pol, 'lost_pct', fa=dict(Q=Q, dispatch=disp), fb=dict(Q=Q, dispatch='drop')); r.update(Q=Q, pol=pol, disp=disp); tests.append(r)
    for nm in ['pt', 'pw']:
        adj = holm([r[nm] for r in tests])
        for r, a in zip(tests, adj): r[nm + '_holm'] = float(a)
    R['ablation_tests'] = tests
    # 1.22 anomaly: naive vs fixed at each Q under drop and unlimited
    an = []
    for Q in [200., 250., 300., 400.]:
        for disp in ['drop', 'unlimited']:
            for pol in ['Naive', 'Buffered']:
                r = paired(d, pol, 'Fixed-6d-halves-nominal0.6B', 'lost_pct', fa=dict(Q=Q, dispatch=disp), fb=dict(Q=Q, dispatch=disp)); r.update(Q=Q, disp=disp, pol=pol); an.append(r)
    adj = holm([r['pt'] for r in an]); adjw = holm([r['pw'] for r in an])
    for r, a, b in zip(an, adj, adjw): r['pt_holm'] = float(a); r['pw_holm'] = float(b)
    R['anomaly_rows'] = an
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for k, pol in enumerate(['Fixed-6d-halves-nominal0.6B', 'Naive', 'Buffered']):
        for disp, c in [('drop', 'tab:red'), ('reinsert', 'tab:orange'), ('trunc', 'tab:green'), ('unlimited', 'tab:blue')]:
            x = [r for r in rows if r['policy'] == pol and r['dispatch'] == disp]; x.sort(key=lambda r: r['ratio'])
            ax[k].plot([r['ratio'] for r in x], [r['loss'] for r in x], marker='o', ms=3, label=disp, color=c)
        ax[k].set_title(pol.replace('Fixed-6d-halves-nominal0.6B', 'Fixed (paper)'), fontsize=8); ax[k].set_xlabel('Fleet ratio')
    ax[0].set_ylabel('Overflow loss (%)'); ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(f'{FIG}/fig_ablation.png'); plt.close()

def scale_tables():
    d = addJ(pd.read_csv(f'{OUT}/scale.csv')); rows = []
    for (st, N, pol), g in d.groupby(['setting', 'N', 'policy'], sort=False):
        rows.append(dict(setting=st, N=int(N), policy=pol, n=len(g), loss=ms(g.lost_pct, 1), cpu=ms(g.cpu), dist=f'{g.dist.mean():.0f}', J=f'{g.Jgen.mean():.4f}'))
    R['scale_rows'] = rows
    if os.path.exists(f'{OUT}/solvers.csv'):
        s = addJ(pd.read_csv(f'{OUT}/solvers.csv')); R['solver_rows'] = [dict(N=int(N), solver=sv, policy=pol, n=len(g), dist=f'{g.dist.mean():.0f}', loss=ms(g.lost_pct, 1), cpu=ms(g.cpu), secs=f'{(g.secs.max()):.0f}') for (N, sv, pol), g in s.groupby(['N', 'solver', 'policy'], sort=False)]
        tt = []
        for N in s.N.unique():
            for pol in ['Naive', 'Buffered']:
                for sv in ['pyvrp', 'ortools']:
                    if ((s.N == N) & (s.solver == sv)).any():
                        r = paired(s, pol, pol, 'dist', fa=dict(N=N, solver=sv), fb=dict(N=N, solver='cw2opt')); r.update(N=int(N), solver=sv, pol=pol); tt.append(r)
        adj = holm([r['pt'] for r in tt]);
        for r, a in zip(tt, adj): r['pt_holm'] = float(a)
        R['solver_tests'] = tt
    h = addJ(pd.read_csv(f'{OUT}/heterogeneity.csv'))
    R['het_rows'] = [dict(sig_ln=sl, policy=pol, loss=ms(g.lost_pct, 1), cpu=ms(g.cpu), J=f'{g.Jgen.mean():.4f}') for (sl, pol), g in h.groupby(['sig_ln', 'policy'], sort=False)]

def fc_tables():
    if not os.path.exists(f'{OUT}/forecasters.csv'): return
    d = addJ(pd.read_csv(f'{OUT}/forecasters.csv')); acc = pd.read_csv(f'{OUT}/forecaster_accuracy.csv'); R['fc_acc'] = acc.round(3).to_dict('records')
    R['fc_tuned'] = pd.read_csv(f'{OUT}/forecaster_tuned.csv').to_dict('records')
    m = addJ(pd.read_csv(f'{OUT}/main.csv')); m = m[m.split == 'test']; BF = R['best_fixed']; FO = 'Fixed-6d-halves-nominal0.6B'; fb = m[m.policy == BF]; fo = m[m.policy == FO]
    rows = []
    for (mod, var), g in d.groupby(['model', 'variant'], sort=False):
        rows.append(dict(model=mod, variant=var, sigma_eff=ms(g.sigma_eff, 2), bias=f'{g.bias_eff.mean():.2f}', sig_hat=f'{g.sig_used.mean():.2f}', loss=ms(g.lost_pct, 1), cpu=ms(g.cpu), J=f'{g.Jgen.mean():.4f}', fill=f'{g.fill_visit.mean():.0f}',
                         sigma_eff_v=float(g.sigma_eff.mean()), loss_v=float(g.lost_pct.mean()), cpu_v=float(g.cpu.mean()), J_v=float(g.Jgen.mean())))
    R['fc_closed'] = rows
    tests = []
    for (mod, var), g in d.groupby(['model', 'variant'], sort=False):
        pol = f'{mod}-{var}'
        for ref in [FO, BF]:
            for c in ['cpu', 'lost_pct', 'Jgen']:
                a = g.sort_values('seed').set_index('seed')[c]; b = m[m.policy == ref].sort_values('seed').set_index('seed')[c]; a, b = a.align(b, join='inner'); diff = (a - b).values
                w = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(diff).pvalue
                tests.append(dict(policy=pol, ref='orig' if ref == FO else 'best', col=c, diff=float(diff.mean()), ci=ci(diff), pt=float(stats.ttest_1samp(diff, 0).pvalue), pw=float(w)))
    for nm in ['pt', 'pw']:
        adj = holm([r[nm] for r in tests])
        for r, a in zip(tests, adj): r[nm + '_holm'] = float(a)
    R['fc_tests'] = tests
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5)); n = m[m.policy == 'Naive'].groupby('sigma')
    for k, (col, yl, key) in enumerate([('lost_pct', 'Overflow loss (%)', 'loss_v'), ('cpu', 'Cost per collected unit', 'cpu_v'), ('Jgen', f'Cost+{CL}·lost per generated unit', 'J_v')]):
        mean = n[col].mean(); ax[k].plot(mean.index, mean.values, 'k-', lw=1, label='Naive, i.i.d. error model')
        bb = m[m.policy == 'Buffered'].groupby('sigma')[col].mean(); ax[k].plot(bb.index, bb.values, 'k--', lw=1, label='Buffered, i.i.d.')
        ax[k].axhline(fb[col].mean(), color='r', ls='-.', lw=.8, label='Fixed (best tuned)')
        for var, mk in [('Naive', 'o'), ('Buffered', 's'), ('Tuned', '^')]:
            for mod, c in [('MA', 'tab:gray'), ('XGB', 'tab:blue'), ('RF', 'tab:green'), ('LSTM', 'tab:purple')]:
                r = [x for x in rows if x['model'] == mod and x['variant'] == var]
                if r: ax[k].scatter(r[0]['sigma_eff_v'], r[0][key], marker=mk, c=c, s=18, label=f'{mod}' if var == 'Naive' else None)
        ax[k].set_xlabel('Measured σ of trained forecaster'); ax[k].set_ylabel(yl)
    ax[0].legend(fontsize=5); plt.tight_layout(); plt.savefig(f'{FIG}/fig_forecasters.png'); plt.close()

if __name__ == '__main__':
    main_tables(); err_tables(); sigmahat_table(); pareto(); ablation(); scale_tables(); fc_tables()
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1, default=lambda o: None if o is None else str(o))
    print('ok', list(R.keys()))
