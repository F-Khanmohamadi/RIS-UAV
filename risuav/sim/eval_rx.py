import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import param_override
import env_mr as EM, coupling as CP, build_exact as BE
from env_mr import City, make_mission, servable, evaluate, plan_assign
from batch import CityArrays
from model_rx import RxProposed, make_batch_rx
from plan_mr import decode_stops, to_eval, tsp_reorder, repair, ils_coupled

torch.set_num_threads(1)
scen, seed = sys.argv[1], int(sys.argv[2]); ckpt = sys.argv[3] if len(sys.argv) > 3 else 'best'
from rx_paths import MR_ROOT as R, RUNS_DIR, RESULTS_DIR
pre = 'g' if scen == 'ground' else ''
TAG = os.environ.get('RUN_TAG', 'proposed'); CDIR = os.environ.get('CITY_DIR', 'cities')
TEST = os.environ.get('TEST_CITIES', 'city21,city22').split(',')
k0, k1 = map(int, os.environ.get('K_RANGE', '0-59').split('-'))
N_ILS = int(os.environ.get('N_ILS', '100'))
ALL = ['proposed', 'none', 'random', 'coverage']
METHODS = os.environ.get('METHODS', ','.join(ALL)).split(',')
OUTD = os.path.join(RESULTS_DIR, os.environ.get('OUT_TAG', 'main')); os.makedirs(OUTD, exist_ok=True)
OUT = os.path.join(OUTD, f'{scen}_s{seed}_{ckpt}.jsonl')
MID = {m: i for i, m in enumerate(ALL)}


from rx_common import load_city as _load, DS, MAXS


def load_city(name):
    return _load(os.path.join(R, CDIR, f'{name}.pkl'))


arrs = {c: CityArrays(load_city(f'{pre}{c}')) for c in TEST}
pol = RxProposed(); pol.load_state_dict(torch.load(os.path.join(RUNS_DIR, f'{TAG}_{scen}_s{seed}', f'{ckpt}.pt'), map_location='cpu')); pol.eval()
M = BE.Nr


def comm_metrics(city, m, stops):
    dep = city.deployed(m, plan_assign(to_eval(stops)))
    P, D, EN, J, NC, PNS, SNRg, SNRs, nbh = [], [], [], [], 0, [], [], [], []
    for s in stops:
        for (i, o, q) in s['mem']:
            h = m['cells'][i]; d = city.cp.details(h, o, q, dep)
            p = np.where(d['has'], d['p'], 0.0); delta = 1 - (1 - p) ** BE.P['Mmax']
            P += p.tolist(); D += delta.tolist(); EN += np.where(p > 0, delta / np.maximum(p, 1e-300), BE.P['Mmax']).tolist()
            J += (d['j'] + 1).tolist(); NC += int(d['coupled'].sum()); PNS += d['Pns'].tolist()
            SNRg += (10 * np.log10(BE.Pt * (d['mu2'] + M * d['s2'] + M * d['Pns']) / BE.N0)).tolist()
            SNRs += (10 * np.log10(BE.Pt * (d['mu2'] + M * d['s2']) / BE.N0)).tolist()
            nbh.append(int(city.svc(h, o, dep)['feas'].sum()))
    P = np.array(P)
    return dict(prr_mean=float(P.mean()), prr_min=float(P.min()), delivery_mean=float(np.mean(D)), delivery_min=float(np.min(D)),
                attempts_mean=float(np.mean(EN)), cqi_hist=np.bincount(J, minlength=16)[1:].tolist(), gue_coupled=NC, n_gue=len(P),
                pns_max=float(max(PNS)) if PNS else 0.0, snr_db_mean_multi=float(np.mean(SNRg)),
                snr_db_mean_serving_only=float(np.mean(SNRs)), nbh_mean=float(np.mean(nbh)), nbh_min=int(min(nbh)))


def ris_metrics(city, m, stops):
    A = [(i, o) for s in stops for (i, o, q) in s['mem']]
    pids = [int(city.cp.pid[m['cells'][i]][o]) for i, o in A if o > 0]
    cnt = {p: pids.count(p) for p in set(pids)}; sizes = [len(s['mem']) for s in stops if s['mem']]
    return dict(n_panels=len(cnt), n_shared_panels=sum(1 for v in cnt.values() if v > 1), n_assisted_cells=len(pids),
                panels={str(p): [float(x) for x in city.cp.panels[p]] for p in sorted(cnt)},
                cells_per_stop=np.bincount(sizes).tolist(), frac_shared_stops=float(np.mean([s > 1 for s in sizes])))


def metrics(a, m, stops, t_inf, t_ils, rep):
    r = evaluate(a.city, m, to_eval(stops))
    r.update(comm_metrics(a.city, m, stops)); r.update(ris_metrics(a.city, m, stops))
    r['plan'] = [dict(xy=[float(s['xy'][0]), float(s['xy'][1])], mem=[[int(i), int(o), int(q)] for (i, o, q) in s['mem']]) for s in stops]
    r['shared_stops'] = int(sum(len(s['mem']) > 1 for s in stops if s['mem']))
    r['feasible'] = bool(r['prr_ok'] and r['served_once'] and r['battery_ok'] and r['deadline_ok'] and rep[1] == 0)
    r['t_inference_s'] = t_inf; r['t_ils_s'] = t_ils; r['t_total_s'] = t_inf + t_ils; r['repaired'] = rep[0]; r['unrepairable'] = rep[1]
    return r


def fixed_opts(a, m, kind, rng):
    out = []
    for h in m['cells']:
        ar = a.city.H[h]['area']; no = a.city.H[h]['nopt']
        if kind == 'none' or no == 1:
            out.append(0)
        elif kind == 'coverage_start':
            out.append(int(1 + np.argmax(ar[1:no])))
        else:
            ok = [o for o in range(1, no) if ar[o] > 0]
            out.append(int(rng.choice(ok)) if ok else 0)
    return out


def coverage_global(city, m):
    K = len(m['cells']); r = fixed_opts(type('A', (), {'city': city})(), m, 'coverage_start', None)
    Q = [len(city.H[h]['U']) for h in m['cells']]
    pid = lambda i, o: int(city.cp.pid[m['cells'][i]][o]) if o > 0 else -1
    dep = lambda rr: frozenset(pid(i, o) for i, o in enumerate(rr) if o > 0)

    rng_mask = [np.hypot(*(city.H[h]['U'] - city.H[h]['g']).T) <= EM.REF['R_hov'] for h in m['cells']]

    def cov(i, rr, d):
        return (city.cp.feas(m['cells'][i], rr[i], d) & rng_mask[i]).sum() / Q[i]
    d0 = dep(r); val = [cov(i, r, d0) for i in range(K)]
    for _ in range(2):
        improved = False
        for i in range(K):
            h = m['cells'][i]
            for o in range(city.H[h]['nopt']):
                if o == r[i] or (o > 0 and city.H[h]['area'][o] == 0):
                    continue
                r2 = list(r); r2[i] = o; d2 = dep(r2)
                changed = {pid(i, r[i]), pid(i, o)} - {-1}
                aff = [i] + [j for j in range(K) if j != i and any(city.cp.pair(p, m['cells'][j]) is not None for p in changed)]
                new = {j: cov(j, r2, d2) for j in aff}
                if sum(new[j] - val[j] for j in aff) > 1e-12:
                    r = r2; d0 = d2
                    for j in aff:
                        val[j] = new[j]
                    improved = True
        if not improved:
            break
    return r


def plan_method(a, m, bt, K, kind, rng):
    city = a.city; t = time.time()
    if kind == 'proposed':
        fo = None
        allowed = [list(range(city.H[h]['nopt'])) for h in m['cells']]
    else:
        opts = coverage_global(city, m) if kind == 'coverage' else fixed_opts(a, m, kind, rng)
        fo = torch.zeros(1, bt['Kp'], dtype=torch.long); fo[0, :K] = torch.tensor(opts)
        allowed = [[int(o)] for o in fo[0, :K]]
    with torch.no_grad():
        seq, _, _ = pol(bt, mode='greedy', force_opt=fo)
    s0 = decode_stops(city, m, seq.numpy()[0], bt['Kp']); s0, nrep, nbad = repair(city, m, s0)
    t_inf = time.time() - t; t = time.time()
    st = tsp_reorder(m, ils_coupled(city, m, s0, N_ILS, allowed, rng))
    return st, metrics(a, m, st, t_inf, time.time() - t, (nrep, nbad))


done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            d = json.loads(line); done.add((d['city'], d['k']))
        except Exception:
            pass
t0 = time.time()
for cname, a in arrs.items():
    for k in range(k0, k1 + 1):
        if (a.city.name, k) in done:
            continue
        m = make_mission(a.city, k)
        row = dict(city=a.city.name, k=k, K=len(m['cells']), cells=[int(h) for h in m['cells']],
                   depot=[float(x) for x in m['depot']], dest=[float(x) for x in m['dest']],
                   config=dict(N_ILS=N_ILS, D_scale=DS, max_sites=MAXS, city_dir=CDIR, ckpt=ckpt, run=TAG, override=param_override.OVR))
        if not servable(a.city, m, True):
            row['servable_ris'] = False
        else:
            nr = servable(a.city, m, False); row.update(servable_ris=True, servable_noris=nr)
            bt = make_batch_rx([(a, m)], D_bits=40e6 * DS); K = len(m['cells'])
            for kind in METHODS:
                if kind == 'none' and not nr:
                    row[kind] = None; continue
                rng = np.random.default_rng([seed, k, MID[kind], 1000 + (0 if cname == TEST[0] else 1)])
                _, row[kind] = plan_method(a, m, bt, K, kind, rng)
        with open(OUT, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(a.city.name, k, f'{time.time() - t0:.0f}s', flush=True)
print('done', OUT, flush=True)
