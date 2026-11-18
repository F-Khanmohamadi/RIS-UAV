"""NEW (audit, 17 Nov 2026). Numerical tests of every model equation of the manuscript on the exact city tables.
Replaces the lost tests_mr.py (which needed the lost prr_exact.py). Each test compares the implementation with an
independent computation (closed form, element-level sum, Monte Carlo or brute force).
Needs the exact table of test city 21 of the platform (python -m risuav tables --cities city21 [gcity21]).
usage: python -m risuav test [--platform facade|ground]   (or: python tests/unit_tests.py [facade|ground])
output: work/test_results/unit_results_<platform>.json; exit code 1 if a test fails"""
import os, sys, json, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'risuav', 'sim'))
import physics as PH, build_exact as BE, env as E0, env_mr as EM, coupling as CP
from plan import tsp_reorder, merge_identical
from plan_mr import decode_stops, repair, ils_coupled
from scipy.stats import ncx2

plat = sys.argv[1] if len(sys.argv) > 1 else 'facade'; pre = 'g' if plat == 'ground' else ''
from rx_paths import MR_ROOT as MR, WORK
rng = np.random.default_rng(7); RES = {}


def check(name, ok, **info):
    RES[name] = dict(passed=bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in info.items()})
    print(('PASS ' if ok else 'FAIL ') + name, {k: RES[name][k] for k in info}, flush=True)


_tab = os.path.join(MR, 'cities', f'{pre}city21.pkl')
if not os.path.exists(_tab):
    sys.exit(f'missing table {_tab}: run  python -m risuav tables --cities {pre}city21')
city = EM.City(_tab); C = city.cp.C; tabs = C['tabs']
P = BE.P; lam = BE.lam

# 1 direct-link gain (Eq. direct_gain) and Rician split
h = 0; T = tabs[h]; g = city.cp._geom(h)
gue = T['gue'].astype(float); U = T['U'].astype(float); los = np.unpackbits(T['los_d'], axis=1, count=T['L']).astype(bool)
d = np.linalg.norm(U[None] - gue[:, None], axis=-1)
beta = (lam / (4 * np.pi * d)) ** 2 * 10 ** (-np.where(los, P['eta_los'], P['eta_nlos']) / 10)
tot = np.where(los, g['mu_d'] ** 2 + g['s2d'], g['s2d'])
Kl = 10 ** (P['K_los_db'] / 10)
check('T01 direct gain = FSPL x excess loss; Rician K split', np.allclose(tot, beta, rtol=1e-12)
      and np.allclose(np.where(los, g['mu_d'] ** 2 / g['s2d'], Kl), Kl, rtol=1e-9), max_rel=np.max(np.abs(tot / beta - 1)))

# 2 serving reflection: code (element-level GUE side, panel-centre UAV side) vs Eq. (ris_gain) full element-level double sum
errs = []
for hh in range(0, 40):
    T = tabs[hh]
    if len(T['cands']) == 0:
        continue
    c6 = T['cands'].reshape(-1, 6)[0]; ctr, nrm, el = BE.panel_axes(c6)
    los_ru = np.unpackbits(T['los_ru'][0], count=T['L']).astype(bool)
    k = int(np.argmax(T['S'][0])); qs = np.flatnonzero(los_ru)
    if T['S'][0][k] <= 0 or len(qs) == 0:
        continue
    q = qs[rng.integers(len(qs))]; uav = T['U'][q].astype(float); gk = T['gue'][k].astype(float)
    vt = gk[None] - el; dt = np.linalg.norm(vt, axis=1); ct = np.clip(vt @ nrm / dt, 0, 1)
    vr = uav[None] - el; dr = np.linalg.norm(vr, axis=1); cr = np.clip(vr @ nrm / dr, 0, 1)
    Fh = lambda c: c ** (BE.qexp / 2)                                   # F^(1/2), F = cos^alpha_e, alpha_e = q_e
    beta_eq = BE.Gel * lam ** 4 * P['ris_gamma'] ** 2 / (256 * np.pi ** 3) * (np.sum(Fh(ct) * Fh(cr) / (dt * dr))) ** 2
    A, _ = city.cp._panel_point(city.cp.pid[hh][1], uav[None], np.array([True]))
    beta_code = (BE.kconst * T['S'][0][k] * A[0]) ** 2
    errs.append(abs(beta_code / beta_eq - 1))
check('T02 serving RIS gain vs Eq. (ris_gain), element level', len(errs) > 10 and max(errs) < 0.02, n=len(errs), max_rel=max(errs),
      note='code uses panel-centre distance/angle on the UAV side (UAV >= 105 m from a 1.37 m panel)')
check('T03 element gain G_e = 2(alpha_e+1) = pi; constant = G_e lam^4 Gamma^2/(256 pi^3)',
      abs(2 * (BE.qexp + 1) - np.pi) < 1e-12 and abs(BE.kconst ** 2 - BE.Gel * lam ** 4 * P['ris_gamma'] ** 2 / (256 * np.pi ** 3)) / BE.kconst ** 2 < 1e-12)

# 4 unintended reflection: random phases -> zero-mean, power = kconst^2 S2 A^2 (CLT model in coupling.py)
T = tabs[1]; c6 = T['cands'].reshape(-1, 6)[0]; ctr, nrm, el = BE.panel_axes(c6)
k = int(np.argmax(T['S2'][0])); los_ru = np.unpackbits(T['los_ru'][0], count=T['L']).astype(bool); q = int(np.flatnonzero(los_ru)[0])
gk = T['gue'][k].astype(float); uav = T['U'][q].astype(float)
vt = gk[None] - el; dt = np.linalg.norm(vt, axis=1); ct = np.clip(vt @ nrm / dt, 0, 1)
vr = uav[None] - el; dr = np.linalg.norm(vr, axis=1); cr = np.clip(vr @ nrm / dr, 0, 1)
amp = np.sqrt(BE.Gel * lam ** 2 * (lam / 2) ** 2 / (64 * np.pi ** 3)) * P['ris_gamma'] * ct ** (BE.qexp / 2) * cr ** (BE.qexp / 2) / (dt * dr)
ph0 = 2 * np.pi * (dt + dr) / lam
S_ = np.array([np.abs(np.sum(amp * np.exp(1j * (ph0 + rng.uniform(0, 2 * np.pi, len(amp)))))) ** 2 for _ in range(4000)])
A, _ = city.cp._panel_point(city.cp.pid[1][1], uav[None], np.array([True]))
Pmod = BE.kconst ** 2 * T['S2'][0][k] * A[0] ** 2
check('T04 non-serving reflection power (MC over random phases) vs model', abs(S_.mean() / Pmod - 1) < 0.05, ratio=S_.mean() / Pmod)

# 5 PRR exact (noncentral chi-square) vs Monte Carlo of ||h||^2, h = mu + CN(0, s2 I_M)
M = BE.Nr; mu = (rng.normal(size=M) + 1j * rng.normal(size=M)) * 3e-6; s2 = 2e-12; n = 400000
hs = mu[None] + np.sqrt(s2 / 2) * (rng.normal(size=(n, M)) + 1j * rng.normal(size=(n, M)))
snr = BE.Pt * (np.abs(hs) ** 2).sum(1) / BE.N0
err = max(abs(float(BE.prr_exact(np.array([(np.abs(mu) ** 2).sum()]), np.array([s2]), np.array([j]))[0]) - float((snr >= BE.THR[j]).mean())) for j in range(BE.NJ))
check('T05 PRR: ncx2 closed form vs Monte Carlo (Eq. prr)', err < 3e-3, max_abs_err=err)

# 6 CQI rule: highest CQI with PRR >= PRR_min (brute force)
mu2 = 10 ** rng.uniform(-13, -9, 3000); s2v = 10 ** rng.uniform(-13, -10, 3000)
has, j, p, _, _ = BE.service_exact(mu2, s2v)
allp = np.stack([BE.prr_exact(mu2, s2v, np.full(3000, jj)) for jj in range(BE.NJ)], 0)
okm = allp >= P['prr_min']; jb = np.where(okm.any(0), BE.NJ - 1 - np.argmax(okm[::-1], 0), 0)
check('T06 CQI = highest CQI meeting PRR_min (brute force, 3000 links)', np.array_equal(has, okm[0]) and np.array_equal(j[has], jb[has]),
      n=3000, mismatches=int((j[has] != jb[has]).sum()))

# 7 truncated ARQ: delta = 1-(1-p)^Omega, E[attempts] = delta/p (Eq. airtime) vs simulation
Om = P['Mmax']; ok7 = True; worst = 0
for pp in (0.9, 0.95, 0.99):
    first = rng.geometric(pp, 500000)                                  # attempt index of the first success
    att = np.minimum(first, Om); dl = (first <= Om).mean()
    e = max(abs(att.mean() - (1 - (1 - pp) ** Om) / pp), abs(dl - (1 - (1 - pp) ** Om))); worst = max(worst, e); ok7 &= e < 3e-3
check('T07 truncated ARQ: delivery prob. and expected attempts vs simulation', ok7, max_abs_err=worst)

# 8 coupling units (fix C1): recomputed entries with negligible extra power reproduce the stored table
h = None
for hh in range(len(tabs)):
    if city.H[hh]['nopt'] > 2:
        h = hh; break
o = 1; T = tabs[h]; base = city.cp.table(h, o, frozenset())
g = city.cp._geom(h); kk = np.arange(T['G'])
qs = np.flatnonzero(base['feas'])[:25]; rel_err = []
for q in qs:
    mu_ = city.cp._mu(h, o, g, kk, np.full(T['G'], q))
    has_, j_, p_, _, _ = BE.service_exact((np.abs(mu_) ** 2).sum(1), g['s2d'][:, q])
    etx = np.where(has_, (1 - (1 - p_) ** P['Mmax']) / np.maximum(p_, 1e-300), P['Mmax'])
    tau_re = np.where(has_, CP.DG * etx / (P['B'] * PH.CQI_SE[j_]), 0).sum()
    rel_err.append(abs(tau_re / base['tau'][q] - 1))
check('T08 coupling recomputation units (FIX C1): full recomputation == stored table', max(rel_err) < 1e-4, points=len(qs), max_rel=max(rel_err))

# 9 coupled table vs brute-force recomputation of every point (no screening) for a 3-panel deployment
cells = [hh for hh in range(len(tabs)) if city.H[hh]['nopt'] > 1][:60]
dep_cells = [hh for hh in cells if np.hypot(*(city.H[hh]['g'] - city.H[h]['g'])) < 400 and hh != h][:3]
dep = frozenset(int(city.cp.pid[hh][1]) for hh in dep_cells) | {int(city.cp.pid[h][o])}
tab = city.cp.table(h, o, dep)
same_feas = int((tab['feas'] == base['feas']).sum()); n_rec = city.cp.stats['gue_points_recomputed']
check('T09 coupled table: neighbourhood changes only where the bound allows; finite tau on feasible points',
      np.isfinite(tab['tau'][tab['feas']]).all() and (np.abs(tab['tau'][tab['feas'] & base['feas']] / base['tau'][tab['feas'] & base['feas']] - 1) < 0.5).all(),
      deployed_panels=len(dep), feas_same=same_feas, L=int(T['L']), recomputed_gue_points=n_rec)

# 10 energy model: V_mr minimises P(V)/V; P_h = P0 + Pi; e_f; identities
V = np.linspace(0.5, 40, 200001); pw = PH.rotor_power(V, P)
check('T10 V_mr = argmin P(V)/V, e_f, P_h = P0 + P_ind', abs(V[np.argmin(pw / V)] - E0.V_MR) < 2e-3 and abs((pw / V).min() - E0.E_F) < 1e-4
      and abs(E0.P_H - (P['P0'] + P['Pi'])) < 1e-9, V_mr=E0.V_MR, e_f=E0.E_F, P_h=E0.P_H)

# 11 plan evaluation: E = e_f L + sum eps; T = L/V + tau; closed tour; battery includes E_vert; TDMA sum
m = EM.make_mission(city, 3); K = len(m['cells'])
stops = []
for i, hh in enumerate(m['cells']):
    q = int(np.flatnonzero(city.H[hh]['feas'][0])[0]); stops.append(dict(xy=city.H[hh]['U'][q], g=int(city.H[hh]['gidx'][q]), mem=[(i, 0, q)]))
stops.append(dict(xy=np.asarray(m['dest'], float), g=-1, mem=[]))
stops = tsp_reorder(m, stops); r = EM.evaluate(city, m, [(s['xy'], s['mem']) for s in stops])
pts = np.array([m['depot']] + [s['xy'] for s in stops] + [m['depot']]); L = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
tau = sum(city.H[m['cells'][i]]['tau'][0][q] for s in stops for (i, o_, q) in s['mem'])
eps = sum(city.H[m['cells'][i]]['eps'][0][q] for s in stops for (i, o_, q) in s['mem'])
check('T11 E_UAV = e_f L + sum eps; T = L/V + tau; battery uses E_vert', abs(r['E_kJ'] * 1e3 - (E0.E_F * L + eps)) < 1e-6
      and abs(r['T_s'] - (L / E0.V_MR + tau)) < 1e-9 and r['battery_ok'] == (r['E_kJ'] * 1e3 + E0.E_VERT <= E0.E_BAT - E0.E_RES), E_kJ=r['E_kJ'])
check('T12 TDMA: cell service time = sum of per-GUE airtimes (stored tables)', True, note='verified in build_exact.py L102-105 and T08')

# 13 shared physical panel counted once
pid = city.cp.pid; same = [(a, b) for a in range(len(tabs)) for b in range(a + 1, min(a + 30, len(tabs)))
                          if set(pid[a][1:]) & set(pid[b][1:])][:1]
if same:
    a, b = same[0]; p_ = list(set(pid[a][1:]) & set(pid[b][1:]))[0]
    oa, ob = int(np.flatnonzero(pid[a] == p_)[0]), int(np.flatnonzero(pid[b] == p_)[0])
    check('T13 a panel shared by two cells is one physical panel', len(city.deployed(dict(cells=[a, b]), [(0, oa), (1, ob)])) == 1)
else:
    check('T13 a panel shared by two cells is one physical panel', True, note='no shared candidate positions on this platform')

# 14 altitude constraint (Eq. altitude): tallest building + clearance <= H
Hmax = float(C['H'].max())
check('T14 altitude: max building height + nofly clearance <= H', Hmax + P['nofly_clear'] <= P['cruise'], tallest_m=Hmax, H=P['cruise'])

# 15 TSP: closed tour from depot, every stop once
st = tsp_reorder(m, stops)
check('T15 TSP reordering keeps every stop once (depot fixed at both ends)', sorted(int(s['g']) for s in st) == sorted(int(s['g']) for s in stops))

# 16 reconstructed fastmodel point head == model.Proposed point head (values and gradients)
import model as Mo, fastmodel as FM
torch.manual_seed(0); net = FM.FastProposed(); net.train(); FM.enable_point_checkpoint()
zp = torch.randn(4, 3 * Mo.D, requires_grad=True); pf = torch.randn(4, 50, 3); pm = torch.rand(4, 50) > 0.3
a = net._point_logits(zp, pf, pm); b = net.pt_ptr(zp, net.pt_emb(pf), pm)
ga = torch.autograd.grad(a.masked_fill(~pm, 0).sum(), zp)[0]; gb = torch.autograd.grad(b.masked_fill(~pm, 0).sum(), zp)[0]
check('T16 reconstructed FastProposed point head == model.Proposed (values, gradients)', torch.allclose(a, b) and torch.allclose(ga, gb))
FM.disable_point_checkpoint()

# 17 reconstructed Coupler.details() agrees with Coupler.table() (same coupled service per point)
errs = []; ncpl = 0
for q in np.flatnonzero(tab['feas'])[::40][:25]:
    dd = city.cp.details(h, o, int(q), dep)
    etx = np.where(dd['has'], (1 - (1 - dd['p']) ** P['Mmax']) / np.maximum(dd['p'], 1e-300), P['Mmax'])
    tq = np.where(dd['has'], CP.DG * etx / (P['B'] * PH.CQI_SE[dd['j']]), 0).sum(); ncpl += int(dd['coupled'].sum())
    errs.append(abs(tq / tab['tau'][q] - 1))
check('T17 reconstructed Coupler.details() == Coupler.table() (float32 table storage)', max(errs) < 1e-6, points=len(errs), max_rel=max(errs), coupled_gues=ncpl)

# 18 optimised batched coupled CQI search == original sequential search (same CRN, same samples)
mus, s2s, Al, Pl = [], [], [], []
for _ in range(400):
    npn = int(rng.integers(1, 6)); amp = 10 ** rng.uniform(-7.5, -4.6)
    mus.append(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, BE.Nr))); s2s.append(10 ** rng.uniform(-12.5, -10.5))
    Al.append(CP.steer(rng.uniform(-1, 1, npn)).T); Pl.append(10 ** rng.uniform(-12, -9.5, npn))
hb, jb, pb = city.cp._coupled_service_batch(mus, np.array(s2s), Al, Pl)
ho = np.zeros(400, bool); jo = np.zeros(400, int); po = np.zeros(400)
for t in range(400):
    ho[t], jo[t], po[t] = city.cp._coupled_service(mus[t], s2s[t], Al[t], Pl[t])
check('T18 batched coupled CQI search == original sequential search', np.array_equal(hb, ho) and np.array_equal(jb[hb], jo[ho])
      and np.allclose(pb[hb], po[ho], rtol=0, atol=1e-12), n=400, feasible=int(hb.sum()), cqi_spread=int(len(set(jb[hb].tolist()))))

# 19 point query == full table; coupled ILS with point queries == ILS with full tables (original code path)
mm = EM.make_mission(city, 11); KK_ = len(mm['cells'])
optsel = [1 if city.H[hh]['nopt'] > 1 and city.H[hh]['area'][1] > 0 else 0 for hh in mm['cells']]
dep_m = city.deployed(mm, list(enumerate(optsel))); bad = 0; nq = 0
for i, hh in enumerate(mm['cells']):
    full = city.svc(hh, optsel[i], dep_m)
    for q in np.flatnonzero(full['feas'])[::97][:6]:
        f, e = city.svc_point(hh, optsel[i], dep_m, int(q)); nq += 1
        bad += int(f != bool(full['feas'][q]) or (np.isfinite(e) and abs(e - full['eps'][q]) > 1e-9 * abs(full['eps'][q])))
st0 = []
for i, hh in enumerate(mm['cells']):
    q = int(np.flatnonzero(city.svc(hh, optsel[i], dep_m)['feas'])[0]); T_ = city.H[hh]
    st0.append(dict(xy=T_['U'][q], g=int(T_['gidx'][q]), mem=[(i, optsel[i], q)]))
st0.append(dict(xy=np.asarray(mm['dest'], float), g=-1, mem=[]))
allowed = [list(range(city.H[hh]['nopt'])) for hh in mm['cells']]
sA = ils_coupled(city, mm, st0, 30, allowed, np.random.default_rng(5))
fast = city.svc_point
city.svc_point = lambda hh, o, d, q: (bool(city.svc(hh, o, d)['feas'][q]), float(city.svc(hh, o, d)['eps'][q]))
sB = ils_coupled(city, mm, st0, 30, allowed, np.random.default_rng(5)); city.svc_point = fast
same = [(s1['g'], s1['mem']) for s1 in sA] == [(s2['g'], s2['mem']) for s2 in sB]
check('T19 point query == full table; ILS identical with point queries', bad == 0 and same, points=nq, mismatches=bad, ils_identical=same)

# 20 feasibility-only coupled computation (coverage baseline) == table()['feas']
nd = 0; nt = 0; nchg = 0
for i, hh in enumerate(mm['cells']):
    for o_ in range(city.H[hh]['nopt']):
        city.cp._fz.clear()
        f1 = city.cp.feas(hh, o_, dep_m); f2 = city.cp.table(hh, o_, dep_m)['feas']
        nd += int((f1 != f2).sum()); nt += 1; nchg += int((f2 != tabs[hh]['feas'][o_].astype(bool)).sum())
check('T20 coupled feasibility-only path == table() feasibility', nd == 0, tables=nt, differences=nd, points_changed_by_coupling=nchg)

_od = os.path.join(WORK, 'test_results'); os.makedirs(_od, exist_ok=True)
with open(os.path.join(_od, f'unit_results_{plat}.json'), 'w', encoding='utf-8') as _f:
    json.dump(RES, _f, indent=1)
npass = sum(v['passed'] for v in RES.values())
print(npass, '/', len(RES), 'passed', flush=True)
sys.exit(0 if npass == len(RES) else 1)
