"""Core simulator for capacity-constrained e-waste collection (see README.md).
Accounting identity (post warm-up window):  stock0 + generated = collected + lost + stock1.
"""
import numpy as np
from dataclasses import dataclass, field, replace

# Primary semi-real calibration scenario: 1e-4 of the CPCB Tamil Nadu
# FY2023-24 statewide inventory estimate. This share is an explicit scenario
# assumption, NOT an observed local-network share. All mass quantities are kg.
CALIBRATION_SHARE = 1e-4
TN_ANNUAL_GENERATION_MT = 448878.0
TN_NETWORK_DAILY_KG = TN_ANNUAL_GENERATION_MT * 1000.0 * CALIBRATION_SHARE / 365.0
N_REFERENCE = 30
SIGMA_REFERENCE = 0.5
BASE_MED_RATE = 8.0
BASE_B = 100.0
BASE_Q = 300.0
BASE_AGGREGATE_DAILY = N_REFERENCE * BASE_MED_RATE * np.exp(SIGMA_REFERENCE**2 / 2.0)
MASS_SCALE_KG_PER_OLD_UNIT = TN_NETWORK_DAILY_KG / BASE_AGGREGATE_DAILY
CAL_MED_RATE_KG = BASE_MED_RATE * MASS_SCALE_KG_PER_OLD_UNIT
CAL_BIN_CAPACITY_KG = BASE_B * MASS_SCALE_KG_PER_OLD_UNIT
CAL_VEHICLE_CAPACITY_KG = BASE_Q * MASS_SCALE_KG_PER_OLD_UNIT

@dataclass
class Params:
    N: int = 30
    area: float = 20.0          # km side of square region (scaled with sqrt(N/30) for large N)
    B: float = CAL_BIN_CAPACITY_KG
    K: int = 4
    Qs: tuple = (CAL_VEHICLE_CAPACITY_KG,) * 4    # kg; heterogeneous fleets allowed
    C: int = 3                  # cycle length (days)
    warm_days: int = 30
    horizon_days: int = 120
    detour: float = 1.3
    c_km: float = 0.9
    c_stop: float = 2.0
    med_rate: float = CAL_MED_RATE_KG
    sig_ln: float = 0.5
    gshape: float = 4.0
    seas_amp: float = 0.3
    init_frac: float = 0.5      # q_i(0) ~ U(0, init_frac*B)
    depot: str = 'centre'       # centre | corner | outside
    dispatch: str = 'drop'      # drop | reinsert | trunc | unlimited
    @property
    def ncyc_warm(self): return self.warm_days // self.C
    @property
    def ncyc(self): return (self.warm_days + self.horizon_days) // self.C

def make_params(N=30, Q=300.0, fleet_ratio=None, **kw):
    """Fleet scaled so that total capacity / mean demand per cycle = fleet_ratio (default 1.47 at N=30,K=4,Q=300)."""
    area = 20.0 * np.sqrt(N / 30.0)
    demand = N * 3 * CAL_MED_RATE_KG * np.exp(0.5 * 0.5 ** 2)
    if fleet_ratio is None:
        K = int(round(4 * N / 30.0)) if N != 30 else 4
        K = max(K, 4) if N == 30 else int(np.ceil(1.47 * demand / Q))
    else:
        K = int(round(fleet_ratio * demand / Q)) if False else 4
    # Q defaults to the calibrated physical capacity unless explicitly overridden.
    if Q == 300.0:
        Q = CAL_VEHICLE_CAPACITY_KG
    p = Params(N=N, area=area, K=K, Qs=(Q,) * K)
    return replace(p, **kw)

class Instance:
    def __init__(self, seed, P: Params):
        rng = np.random.default_rng([seed, 11])
        self.P = P; self.seed = seed
        self.xy = rng.uniform(0, P.area, (P.N, 2))
        a = P.area
        dep = {'centre': (a / 2, a / 2), 'corner': (0.0, 0.0), 'outside': (-a / 2, -a / 2)}[P.depot]
        pts = np.vstack([self.xy, np.array(dep)])
        diff = pts[:, None, :] - pts[None, :, :]
        self.D = P.detour * np.sqrt((diff ** 2).sum(-1))
        self.Dl = self.D.tolist()
        self.dep = P.N
        self.lam = P.med_rate * np.exp(P.sig_ln * rng.standard_normal(P.N))
        days = P.warm_days + P.horizon_days
        w = 1 + P.seas_amp * np.sin(2 * np.pi * np.arange(days) / 7.0)
        self.G = rng.gamma(P.gshape, self.lam[None, :] * w[:, None] / P.gshape)
        self.q0 = rng.uniform(0, P.init_frac * P.B, P.N)
        self.rng_err = np.random.default_rng([seed, 22])

# ---------------- routing -----------------
def clarke_wright(idx, loads, Q, D, dep):
    n = len(idx)
    if n == 0: return []
    if n == 1: return [[int(idx[0])]]
    idx = np.asarray(idx)
    d0 = D[dep, idx]; Ds = D[np.ix_(idx, idx)]
    a, b = np.triu_indices(n, 1)
    s = d0[a] + d0[b] - Ds[a, b]
    o = np.argsort(-s, kind='stable')
    rid = list(range(n)); routes = [[i] for i in range(n)]; rl = [float(x) for x in loads]
    for k in o:
        if s[k] <= 0: break
        i = a[k]; j = b[k]; ri = rid[i]; rj = rid[j]
        if ri == rj or rl[ri] + rl[rj] > Q: continue
        Ri, Rj = routes[ri], routes[rj]
        if Ri[-1] == i and Rj[0] == j: new = Ri + Rj
        elif Ri[0] == i and Rj[-1] == j: new = Rj + Ri
        elif Ri[0] == i and Rj[0] == j: new = Ri[::-1] + Rj
        elif Ri[-1] == i and Rj[-1] == j: new = Ri + Rj[::-1]
        else: continue
        routes[ri] = new; rl[ri] += rl[rj]; routes[rj] = None
        for x in Rj: rid[x] = ri
    return [[int(idx[x]) for x in r] for r in routes if r is not None]

def two_opt(route, Dl, dep):
    if len(route) < 3: return route
    path = [dep] + list(route) + [dep]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                a, b, c, d = path[i - 1], path[i], path[j], path[j + 1]
                if Dl[a][c] + Dl[b][d] < Dl[a][b] + Dl[c][d] - 1e-9:
                    path[i:j + 1] = path[i:j + 1][::-1]; improved = True
    return path[1:-1]

def route_len(r, Dl, dep):
    if not r: return 0.0
    L = Dl[dep][r[0]] + Dl[r[-1]][dep]
    for a, b in zip(r[:-1], r[1:]): L += Dl[a][b]
    return L

def solve_routes(sel, l, P, inst, solver='cw2opt', ctx=None):
    """sel: priority-ordered list of points; l: planned loads (array N). Returns list of routes dispatched."""
    Qmax = max(P.Qs); K = len(P.Qs)
    sel = list(sel)
    if not sel: return [], 0
    if P.dispatch == 'trunc':
        cum = 0.0; keep = []
        for i in sel:
            if cum + l[i] <= sum(P.Qs) * 0.95: keep.append(i); cum += l[i]
        extra = [i for i in sel if i not in set(keep)]
        sel = keep
    n_sel = len(sel)
    if solver == 'cw2opt':
        routes = clarke_wright(sel, [l[i] for i in sel], Qmax, inst.D, inst.dep)
        routes = [two_opt(r, inst.Dl, inst.dep) for r in routes]
    elif solver == 'pyvrp':
        routes = pyvrp_solve(sel, l, Qmax, K, inst, ctx)
    elif solver == 'ortools':
        routes = ortools_solve(sel, l, Qmax, K, inst, ctx)
    rl = [sum(l[i] for i in r) for r in routes]
    order = np.argsort(rl)[::-1]
    routes = [routes[o] for o in order]
    if P.dispatch == 'unlimited':
        return routes, 0
    kept = routes[:K]; dropped = routes[K:]
    ndrop = sum(len(r) for r in dropped)
    if dropped and P.dispatch in ('reinsert', 'trunc'):
        loads = [sum(l[i] for i in r) for r in kept]
        caps = sorted(P.Qs, reverse=True)
        pri = {p: k for k, p in enumerate(sel)}
        pts = sorted([p for r in dropped for p in r], key=lambda p: pri[p])
        for p in pts:
            best = None
            for ri, r in enumerate(kept):
                if loads[ri] + l[p] > caps[ri]: continue
                path = [inst.dep] + r + [inst.dep]
                for pos in range(len(path) - 1):
                    inc = inst.Dl[path[pos]][p] + inst.Dl[p][path[pos + 1]] - inst.Dl[path[pos]][path[pos + 1]]
                    if best is None or inc < best[0]: best = (inc, ri, pos)
            if best is not None:
                _, ri, pos = best; kept[ri].insert(pos, p); loads[ri] += l[p]; ndrop -= 1
    return kept, ndrop

# ---------------- optional external solvers -----------------
def pyvrp_solve(sel, l, Qmax, K, inst, ctx):
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    m = Model()
    locs = [m.add_location(x=0, y=0) for _ in range(len(sel) + 1)]
    m.add_depot(locs[0])
    for j, i in enumerate(sel): m.add_client(locs[j + 1], delivery=int(np.ceil(l[i])))
    m.add_vehicle_type(num_available=max(K, 1), capacity=int(Qmax), unit_distance_cost=1, unit_duration_cost=0)
    idx = [inst.dep] + list(sel)
    SC = 1000.0
    for a in range(len(locs)):
        for b in range(len(locs)):
            if a != b:
                m.add_edge(locs[a], locs[b], distance=int(round(inst.D[idx[a], idx[b]] * SC)))
    res = m.solve(stop=MaxRuntime((ctx or {}).get('tl', 0.15)), display=False, seed=(ctx or {}).get('seed', 0))
    routes = []
    for r in res.best.routes():
        routes.append([sel[v.idx] for v in r.schedule() if v.type.name == 'CLIENT'])
    return routes

def ortools_solve(sel, l, Qmax, K, inst, ctx):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    idx = [inst.dep] + list(sel); n = len(idx)
    mgr = pywrapcp.RoutingIndexManager(n, max(K, 1), 0)
    rt = pywrapcp.RoutingModel(mgr)
    Dm = (inst.D[np.ix_(idx, idx)] * 1000).astype(int)
    cb = rt.RegisterTransitCallback(lambda i, j: int(Dm[mgr.IndexToNode(i), mgr.IndexToNode(j)]))
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    dem = [0] + [int(np.ceil(l[i])) for i in sel]
    dc = rt.RegisterUnaryTransitCallback(lambda i: dem[mgr.IndexToNode(i)])
    rt.AddDimensionWithVehicleCapacity(dc, 0, [int(Qmax)] * max(K, 1), True, 'cap')
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.time_limit.FromMilliseconds(int(1000 * (ctx or {}).get('tl', 0.15)))
    sol = rt.SolveWithParameters(sp)
    routes = []
    if sol is None: return clarke_wright(sel, [l[i] for i in sel], Qmax, inst.D, inst.dep)
    for v in range(max(K, 1)):
        i = rt.Start(v); r = []
        while not rt.IsEnd(i):
            nd = mgr.IndexToNode(i)
            if nd: r.append(sel[nd - 1])
            i = sol.Value(rt.NextVar(i))
        if r: routes.append(r)
    return routes

# ---------------- history / logs -----------------
class History:
    """Operator-observable collection logs (censored)."""
    def __init__(self, N):
        self.N = N
        self.last_day = np.zeros(N)           # day of last visit (0 = start; initial 'virtual' visit)
        self.c = np.full((N, 3), np.nan)      # last 3 collected amounts (most recent first)
        self.iv = np.full((N, 3), np.nan)     # last 3 inter-visit intervals (days)
        self.full = np.zeros(N)               # last visit bin-full (censored at B)
        self.veh = np.zeros(N)                # last visit vehicle-limited (censored)
        self.nvis = np.zeros(N)
        self.tot = np.zeros(N)
    def record(self, i, day, take, bin_full, veh_lim):
        iv = day - self.last_day[i]
        self.c[i] = np.r_[take, self.c[i, :2]]
        self.iv[i] = np.r_[iv, self.iv[i, :2]]
        self.last_day[i] = day; self.full[i] = bin_full; self.veh[i] = veh_lim; self.nvis[i] += 1; self.tot[i] += take

# ---------------- policies -----------------
class FixedPolicy:
    """Periodic schedule. m_i = visit period in cycles per point; groups staggered by index offset."""
    def __init__(self, period=2, grouping='halves', nominal=None, rate_theta=None, name=None):
        self.period = period; self.grouping = grouping; self.nominal = nominal; self.rate_theta = rate_theta
        self.name = name or f'Fixed-{period*3}d-{grouping}'
    def setup(self, inst, P, seedstream=None):
        N = P.N
        if self.rate_theta is not None:   # rate-tuned: per-point period from known mean rate
            self.m = np.clip(np.floor(self.rate_theta * P.B / (P.C * inst.lam)), 1, 4).astype(int)
            self.off = np.arange(N) % self.m
            # balance offsets: order points by rate within each period class
            for mm in np.unique(self.m):
                ids = np.where(self.m == mm)[0]; self.off[ids] = np.arange(len(ids)) % mm
        else:
            m = self.period
            self.m = np.full(N, m)
            self.off = (np.arange(N) % m) if self.grouping == 'halves' else np.zeros(N, int)
        self.inst, self.P = inst, P
    def plan(self, c, t, q, hist, ctx):
        P, inst = self.P, self.inst
        sel = [i for i in range(P.N) if (c - self.off[i]) % self.m[i] == 0]
        l = np.zeros(P.N)
        for i in sel:
            l[i] = P.B * self.nominal if self.nominal else min(P.B, inst.lam[i] * P.C * self.m[i])
        return sel, l, None

class ErrModel:
    def __init__(self, kind='iid', sigma=0.3, rho_t=0.8, rho_c=0.5, nu=3.0, bias=np.log(0.8)):
        self.kind, self.sigma, self.rho_t, self.rho_c, self.nu, self.bias = kind, sigma, rho_t, rho_c, nu, bias
    def sample(self, rng, ncyc, N):
        s = self.sigma; k = self.kind
        if s == 0: return np.zeros((ncyc, N))
        if k in ('iid', 'bias_under', 'bias_over'):
            x = rng.standard_normal((ncyc, N)); L = s * x - s * s / 2
            if k == 'bias_under': L = L + np.log(0.8)
            if k == 'bias_over': L = L + np.log(1.25)
        elif k == 'bias_point':
            u = rng.standard_normal(N); z = rng.standard_normal((ncyc, N))
            L = s * (np.sqrt(0.5) * z + np.sqrt(0.5) * u[None, :]) - s * s / 2
        elif k == 'ar1':
            z = rng.standard_normal((ncyc, N)); x = np.zeros((ncyc, N)); x[0] = z[0]; r = self.rho_t
            for c in range(1, ncyc): x[c] = r * x[c - 1] + np.sqrt(1 - r * r) * z[c]
            L = s * x - s * s / 2
        elif k == 'xcorr':
            f = rng.standard_normal((ncyc, 1)); z = rng.standard_normal((ncyc, N))
            L = s * (np.sqrt(self.rho_c) * f + np.sqrt(1 - self.rho_c) * z) - s * s / 2
        elif k == 't':
            nu = self.nu; x = rng.standard_t(nu, (ncyc, N)) * np.sqrt((nu - 2) / nu)
            L = np.clip(s * x, -3, 3) - s * s / 2
        else: raise ValueError(k)
        return L

class SigmaEstimator:
    """Rolling RMS of log((qhat+1)/(collected+1)) over visited, non-vehicle-limited points (what an operator can observe)."""
    def __init__(self, window_cycles=5, prior=0.3, minn=8):
        self.w = window_cycles; self.prior = prior; self.minn = minn; self.buf = []
    def update(self, resid):
        self.buf.append(np.asarray(resid)); self.buf = self.buf[-self.w:]
    def value(self):
        r = np.concatenate(self.buf) if self.buf else np.array([])
        if len(r) < self.minn: return self.prior
        return float(np.clip(np.sqrt(np.mean(r ** 2)), 0.05, 1.5))

class ForecastPolicy:
    """Forecast-driven policy. q_hat from (a) oracle state x error model or (b) a trained forecaster.
    p_hat = q_hat + C*r (r = known mean rate for (a); history rate estimate for (b)).
    select p_hat >= (theta - a*min(sig,1))*B ; planned load l = min(B, q_hat*(1+k*sig))."""
    def __init__(self, theta=0.85, a=0.0, k=0.0, err=None, forecaster=None, sigma_mode='true', sig_mult=1.0,
                 solver='cw2opt', name=None):
        self.theta, self.a, self.k, self.err, self.fc = theta, a, k, err, forecaster
        self.sigma_mode, self.sig_mult, self.solver = sigma_mode, sig_mult, solver
        self.name = name or 'Forecast'
    def setup(self, inst, P, seedstream=None):
        self.inst, self.P = inst, P
        if self.err is not None:
            self.L = self.err.sample(inst.rng_err, P.ncyc, P.N); self.eps = np.exp(self.L)
        self.est = SigmaEstimator(); self.diag = []; self.last_qhat = None; self.sig_hist = []
    def sigma_used(self):
        if self.sigma_mode == 'true': return self.err.sigma if self.err is not None else self.est.value()
        if self.sigma_mode == 'mult': return self.err.sigma * self.sig_mult
        return self.est.value()          # 'est'
    def plan(self, c, t, q, hist, ctx):
        P, inst = self.P, self.inst
        if self.err is not None:
            qh = q * self.eps[c]; r = inst.lam
        else:
            qh, r = self.fc.predict(t, hist, P)
        self.last_qhat = qh; self.last_q = q.copy()
        sig = self.sigma_used(); self.sig_hist.append(sig)
        ph = qh + P.C * r
        thr = (self.theta - self.a * min(sig, 1.0)) * P.B
        sel = np.where(ph >= thr)[0]
        sel = sel[np.argsort(-ph[sel], kind='stable')]
        l = np.minimum(P.B, qh * (1 + self.k * sig))
        ctx['qhat'] = qh
        return list(sel), l, ph
    def observe(self, visited, takes, veh_lim):
        if self.last_qhat is None or len(visited) == 0: return
        ok = ~np.asarray(veh_lim, bool)
        if ok.any():
            v = np.asarray(visited)[ok]; tk = np.asarray(takes)[ok]
            self.est.update(np.log((self.last_qhat[v] + 1) / (tk + 1)))

# ---------------- run -----------------
def run(inst, policy, P=None, solver=None, record_fc=False, ctx_extra=None):
    P = P or inst.P
    policy.setup(inst, P)
    solver = solver or getattr(policy, 'solver', 'cw2opt')
    N, B = P.N, P.B
    q = inst.q0.copy(); hist = History(N)
    M = dict(dist=0.0, stops=0, collected=0.0, lost=0.0, gen=0.0, partial=0, fillsum=0.0, dropped=0, sel=0, routes=0)
    fc_pairs = []
    caps_sorted = sorted(P.Qs, reverse=True)
    for c in range(P.ncyc):
        t = c * P.C; measuring = c >= P.ncyc_warm
        if c == P.ncyc_warm: M['stock0'] = float(q.sum())
        ctx = dict(seed=inst.seed * 1000 + c, tl=(ctx_extra or {}).get('tl', 0.15))
        if hasattr(policy, 'fc') and policy.fc is not None and c < 2:     # bootstrap: visit all points in first two cycles
            sel, l, ph = list(range(N)), np.minimum(P.B, 0.6 * B * np.ones(N)), None
            ctx['qhat'] = None; policy.last_qhat = None
        else:
            sel, l, ph = policy.plan(c, t, q, hist, ctx)
        routes, ndrop = solve_routes(sel, l, P, inst, solver, ctx)
        visited, takes, vlim = [], [], []
        for r, cap in zip(routes, caps_sorted + [max(P.Qs)] * 1000):
            rem = cap
            if measuring:
                M['dist'] += route_len(r, inst.Dl, inst.dep); M['routes'] += 1
            for i in r:
                take = min(q[i], rem); vl = take < q[i] - 1e-9
                bf = q[i] >= B - 1e-9
                if measuring:
                    M['stops'] += 1; M['collected'] += take; M['fillsum'] += q[i] / B; M['partial'] += int(vl)
                if record_fc and measuring and ctx.get('qhat') is not None:
                    pass
                q[i] -= take; rem -= take
                hist.record(i, t, take, bf, vl); visited.append(i); takes.append(take); vlim.append(vl)
        if hasattr(policy, 'observe'): policy.observe(visited, takes, vlim)
        if measuring:
            M['dropped'] += ndrop; M['sel'] += len(sel)
        if record_fc and measuring and ctx.get('qhat') is not None:
            qt = policy.last_q; qh = ctx['qhat']; m = qt > 5
            fc_pairs.append((np.log((qh[m] + 1) / (qt[m] + 1)), qt[m], qh[m]))
        for d in range(P.C):
            new = q + inst.G[t + d]
            if measuring: M['lost'] += float(np.maximum(0, new - B).sum()); M['gen'] += float(inst.G[t + d].sum())
            q = np.minimum(new, B)
    M['stock1'] = float(q.sum())
    base = M['stock0'] + M['gen']
    assert abs(M['stock0'] + M['gen'] - M['collected'] - M['lost'] - M['stock1']) < 1e-6 * base, 'accounting identity violated'
    M['cost'] = P.c_km * M['dist'] + P.c_stop * M['stops']
    M['coll_pct'] = 100 * M['collected'] / base; M['lost_pct'] = 100 * M['lost'] / base; M['resid_pct'] = 100 * M['stock1'] / base
    M['cpu'] = M['cost'] / max(M['collected'], 1e-9)
    M['fill_visit'] = 100 * M['fillsum'] / max(M['stops'], 1)
    M['partial_pct'] = 100 * M['partial'] / max(M['stops'], 1)
    M['base'] = base
    if hasattr(policy, 'sig_hist') and policy.sig_hist: M['sig_used'] = float(np.mean(policy.sig_hist[P.ncyc_warm:]))
    if record_fc and fc_pairs:
        lr = np.concatenate([x[0] for x in fc_pairs]); M['sigma_eff'] = float(np.sqrt(np.mean(lr ** 2)))
        M['bias_eff'] = float(np.mean(lr)); M['sigma_sd'] = float(np.std(lr))
    return M
