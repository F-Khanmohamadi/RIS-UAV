import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'sim'))
import env_mr as EM

CSV, PLANS, SEED, CITY, K, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6]
from rx_paths import MR_ROOT as MR, PRELIM_ROOT
PR = os.path.join(PRELIM_ROOT, 'cities')
os.makedirs(OUT, exist_ok=True); W = os.path.join(OUT, '_work'); os.makedirs(W, exist_ok=True)

# ---- 1. plans -> drawing format
R = {}
plans = [json.loads(l) for l in open(PLANS, encoding='utf-8')]
for plat, pre in (('facade', ''), ('ground', 'g')):
    city = EM.City(os.path.join(MR, 'cities', f'{pre}{CITY}.pkl'))
    for p in plans:
        if p['platform'] != plat or p['seed'] != SEED or p['city'] != f'{pre}{CITY}' or p['k'] != K or p['method'] not in ('none', 'random', 'coverage', 'proposed'):
            continue
        m = dict(cells=np.array(p['cells']))
        A = [(i, o) for s in p['plan'] for (i, o, q) in s['mem']]; dep = city.deployed(m, A)
        stops, assign, path = [], [], [p['depot']]
        for s in p['plan']:
            path.append(s['xy'])
            if s['mem']:
                stops.append(s['xy'])
                for (i, o, q) in s['mem']:
                    h = int(p['cells'][i])
                    assign.append(dict(cell=int(i), h=h, option=int(o), q=int(q), stop=len(stops) - 1,
                                       tau_s=float(city.svc(h, o, dep)['tau'][q])))
        path.append(p['depot'])
        R[f'{plat}/{p["method"]}'] = dict(stops=stops, path=path, assign=assign, E_kJ=p['E_kJ'], L_km=p['L_km'], source='saved plan')
        tau = sum(a['tau_s'] for a in R[f'{plat}/{p["method"]}']['assign'])
        assert abs(tau - p['tau_s']) < 1e-6 * max(1, tau), (plat, p['method'], tau, p['tau_s'])     # same service time as saved
need = [f'{p}/{m}' for p in ('facade', 'ground') for m in ('none', 'random', 'coverage', 'proposed')]
if any(k not in R for k in need):
    print(f'trajectory figures skipped: the results do not contain mission {CITY} k = {K} (seed {SEED}) for every platform and '
          f'method (missing: {", ".join(k for k in need if k not in R)})')
    sys.exit(0)
json.dump(R, open(os.path.join(W, 'plans_drawing.json'), 'w', encoding='utf-8'))
D = pd.read_csv(CSV); D = D[(D.seed == SEED) & (D.city == CITY) & (D.mission_id == K)]; D.to_csv(os.path.join(W, 'mission_rows.csv'), index=False)

# ---- 2. drawing (traj_3d/traj_draw.py reads its inputs from the environment)
os.environ.update(TRAJ_PLANS=os.path.join(W, 'plans_drawing.json'), TRAJ_ROWS=os.path.join(W, 'mission_rows.csv'),
                  TRAJ_TABLES=os.path.join(MR, 'cities'), TRAJ_PRELIM=PR, TRAJ_OUT=OUT, TRAJ_CITY=CITY, TRAJ_K=str(K))
sys.path.insert(0, os.path.join(HERE, 'traj_3d'))
import traj_draw as F
for plat in ('facade', 'ground'):
    F.fig_platform(plat)
# caption values (saved results of this mission)
cap = D[D.method.isin(['none', 'random', 'coverage', 'proposed'])][['platform', 'method', 'E_UAV_kJ', 'E_flight_kJ', 'E_hover_kJ', 'L_flight_km',
                                                                     'n_hovering_stops', 'n_shared_hovering_stops', 'T_hover_s', 'n_RIS_panels']]
cap.to_csv(os.path.join(OUT, 'trajectory_mission_values.csv'), index=False); print(cap.round(2).to_string())
