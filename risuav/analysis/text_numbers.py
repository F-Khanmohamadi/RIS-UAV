"""Numbers quoted in the text of Section V (Numerical Results), computed from the saved results, so that every value in
the text can be traced. Writes a Markdown report grouped by subsection.
Sources: per-mission CSVs and plans (export_rx.py), paired statistics, training summary and sensitivity CSVs
(make_main.py, make_sens.py), training logs, and - for the geometric quantities - the service tables of the test cities
(cell centres, RIS candidate ids). Without the tables those lines are reported as skipped.
usage: text_numbers.py <figures dir (with data/, main/, sensitivity/)> <results dir> <out.md>"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

FIG, RES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'sim'))
from rx_paths import MR_ROOT
PLAT = {'facade': 'Facade RIS', 'ground': 'Ground-mobile RIS'}
M4 = ['none', 'random', 'coverage', 'proposed']
L = []
def h(t): L.append(f'\n## {t}\n')
def p(t): L.append(f'- {t}')
pct = lambda a, b: 100 * (a - b) / b

P = pd.read_csv(os.path.join(FIG, 'data', 'per_mission_main.csv')); P = P[P.method.isin(M4)]
S = pd.read_csv(os.path.join(FIG, 'main', 'paired_statistics.csv'))
TR = pd.read_csv(os.path.join(FIG, 'main', 'training_per_seed.csv'))
rows = {pl: [] for pl in PLAT}
for pl in PLAT:
    for s in (0, 1, 2):
        f = os.path.join(RES, 'main', f'{pl}_s{s}_best.jsonl')
        if os.path.exists(f):
            rows[pl] += [dict(json.loads(l), seed=s) for l in open(f, encoding='utf-8')]

# ---------------------------------------------------------------- V-A
h('V-A Setup: effect of the reflections of non-serving panels (test plans of Random, Coverage-based and Proposed)')
for pl in PLAT:
    d = [r[m]['snr_db_mean_multi'] - r[m]['snr_db_mean_serving_only'] for r in rows[pl] for m in ('random', 'coverage', 'proposed') if r.get(m)]
    g = [r[m]['gue_coupled'] for r in rows[pl] for m in ('random', 'coverage', 'proposed') if r.get(m)]
    p(f'{PLAT[pl]}: mean SNR change {np.mean(d):.2e} dB, largest |change| {np.max(np.abs(d)):.4f} dB; '
      f'GUEs reached by non-serving reflections per mission: mean {np.mean(g):.1f}, max {max(g)}')

# ---------------------------------------------------------------- V-B
h('V-B Training convergence (mean over 3 seeds)')
for plat, x in TR.groupby('platform'):
    g = x.groupby('epoch').mean(numeric_only=True); s = g.loc[30:50]; gap = s.train_cost_kJ - s.val_E_kJ
    v = x.pivot(index='epoch', columns='seed', values='val_E_kJ'); q = x.pivot(index='epoch', columns='seed', values='val_feasible')
    b = x.loc[x.groupby('seed').val_E_kJ.idxmin()]
    p(f'{plat}: validation energy epoch 0 {g.val_E_kJ[0]:.1f} kJ (feasible {100 * g.val_feasible[0]:.1f}%), epoch 5 {g.val_E_kJ[5]:.1f} kJ; '
      f'epochs 30-50: {s.val_E_kJ.min():.1f}-{s.val_E_kJ.max():.1f} kJ; training cost - validation {gap.min():.1f}-{gap.max():.1f} kJ '
      f'(mean {gap.mean():.1f}), penalty mean {s.train_penalty_kJ.mean():.2f} kJ; entropy {g.entropy[1]:.2f} -> {g.entropy[50]:.2f} nats; '
      f'epoch 50 per seed {", ".join(f"{a:.1f}" for a in v.loc[50])} kJ; largest per-seed value after epoch 5 '
      f'{v.loc[6:].max().max():.1f} kJ, lowest per-seed feasibility after epoch 5 {100 * q.loc[6:].min().min():.1f}%; '
      f'best epochs {b.epoch.tolist()}, mean best validation {b.val_E_kJ.mean():.1f} kJ')

# ---------------------------------------------------------------- V-C
h('V-C Evaluation protocol')
for pl, name in PLAT.items():
    x = P[P.platform == name]
    p(f'{name}: {x.groupby("method").size().to_dict()} mission-seed pairs; feasible share {x.groupby("method").feasible.mean().to_dict()}')

# ---------------------------------------------------------------- V-D
h('V-D Trajectory mission (seed 0, city21, mission 6)')
try:
    import env_mr as EM
    have_tables = True
except Exception:
    have_tables = False
cities = {}
def city(pre, c):
    k = f'{pre}{c}'
    if k not in cities:
        f = os.path.join(MR_ROOT, 'cities', f'{k}.pkl')
        cities[k] = EM.City(f) if have_tables and os.path.exists(f) else None
    return cities[k]
for pl, pre in (('facade', ''), ('ground', 'g')):
    r = [r for r in rows[pl] if r['seed'] == 0 and r['city'] == f'{pre}city21' and r['k'] == 6]
    if not r:
        p(f'{PLAT[pl]}: mission not in the results'); continue
    r = r[0]; C = city(pre, 'city21')
    for m in M4:
        x = r[m]; cps = [len(s['mem']) for s in x['plan'] if s['mem']]
        extra = ''
        if C is not None:
            dh, dr, z = [], [], []
            for s in x['plan']:
                for (i, o, q) in s['mem']:
                    hh = int(r['cells'][i]); g = np.asarray(C.H[hh]['g'])[:2]; dh.append(np.hypot(*(np.asarray(s['xy']) - g)))
                    if o > 0:
                        pz = x['panels'][str(int(C.cp.pid[hh][o]))]; dr.append(np.hypot(pz[0] - g[0], pz[1] - g[1])); z.append(pz[2])
            extra = (f'; median cell-hover {np.median(dh):.0f} m' + (f', median RIS-cell {np.median(dr):.0f} m, mean panel height {np.mean(z):.1f} m' if dr else ''))
        p(f'{PLAT[pl]} {m}: E {x["E_kJ"]:.2f}, flight {x["E_fly_kJ"]:.2f}, hover {x["E_hov_kJ"]:.2f}, comm {x["E_comm_kJ"]:.3f} kJ; '
          f'L {x["L_km"]:.2f} km; stops {x["stops"]} ({x["shared_stops"]} shared, cells per stop max {max(cps)}); service time {x["tau_s"]:.1f} s; '
          f'R_ser {x["R_ser_Mbps"]:.1f} Mbit/s{extra}')
if not have_tables or city('', 'city21') is None:
    p('geometric values skipped (service tables of city21/gcity21 not built)')

# ---------------------------------------------------------------- V-E
h('V-E UAV energy (all test missions, 3 seeds)')
for name in PLAT.values():
    x = S[(S.platform == name) & (S.metric == 'E_UAV_kJ') & (S.method == 'proposed')].set_index('baseline')
    for b in ('none', 'random', 'coverage'):
        y = x.loc[b]
        p(f'{name}: proposed vs {b}: saving {y.gain_pct_of_baseline:.2f}% (95% CI {100 * y.ci95_low / y.mean_b:.2f}-{100 * y.ci95_high / y.mean_b:.2f}%), '
          f'win rate {100 * y.win_rate_mission_seed:.1f}% of mission-seed pairs ({100 * y.win_rate_seed_averaged:.1f}% seed-averaged), Holm p {y.wilcoxon_p_holm:.1e}')
    X = P[P.platform == name]; mE = X.groupby('method').E_UAV_kJ.mean()
    p(f'{name}: savings vs No RIS: random {-pct(mE.random, mE.none):.1f}%, coverage {-pct(mE.coverage, mE.none):.1f}%')
    t = X.groupby(['K_interval', 'method']).E_UAV_kJ.mean().unstack()
    p(f'{name}: saving of proposed vs No RIS per interval (%): ' + ', '.join(f'{k}: {100 * (t.none[k] - t.proposed[k]) / t.none[k]:.1f}' for k in t.index))
    sh = X.groupby(['K_interval', 'method']).frac_shared_hovering_stops.mean().unstack()
    p(f'{name}: share of shared hovering points of proposed per interval (%): ' + ', '.join(f'{k}: {100 * sh.proposed[k]:.0f}' for k in sh.index))
    g = X.groupby('method')[['frac_shared_hovering_stops', 'n_hovering_stops', 'L_flight_km', 'E_flight_kJ', 'E_hover_kJ', 'E_comm_kJ', 'E_UAV_kJ']].mean()
    p(f'{name}: shared hovering points (%): ' + ', '.join(f'{m} {100 * g.frac_shared_hovering_stops[m]:.0f}' for m in M4)
      + f'; hovering points of proposed vs No RIS {pct(g.n_hovering_stops.proposed, g.n_hovering_stops.none):.1f}%')
    p(f'{name}: flight share of E_UAV {100 * (g.E_flight_kJ / g.E_UAV_kJ).min():.1f}-{100 * (g.E_flight_kJ / g.E_UAV_kJ).max():.1f}%, '
      f'communication share max {100 * (g.E_comm_kJ / g.E_UAV_kJ).max():.2f}%; proposed vs No RIS: hovering {pct(g.E_hover_kJ.proposed, g.E_hover_kJ.none):.1f}%, '
      f'flight {pct(g.E_flight_kJ.proposed, g.E_flight_kJ.none):.1f}%, route {pct(g.L_flight_km.proposed, g.L_flight_km.none):.1f}%; '
      f'hovering part of the saving {100 * (g.E_hover_kJ.none - g.E_hover_kJ.proposed) / (g.E_UAV_kJ.none - g.E_UAV_kJ.proposed):.0f}%')
pp = P[P.method == 'proposed'].copy(); pp['cid'] = pp.city.astype(str).str.lstrip('g')
A = pp.groupby(['platform', 'cid', 'mission_id'])[['E_UAV_kJ', 'E_flight_kJ', 'E_hover_kJ', 'R_ser_Mbps']].mean()
F, G = A.loc['Facade RIS'], A.loc['Ground-mobile RIS']; J = F.join(G, lsuffix='_f', rsuffix='_g', how='inner')
for c in ('E_UAV_kJ', 'E_flight_kJ', 'E_hover_kJ', 'R_ser_Mbps'):
    p(f'platforms (proposed, {len(J)} paired seed-averaged missions), {c}: Facade {J[c + "_f"].mean():.2f}, Ground-mobile {J[c + "_g"].mean():.2f} '
      f'({pct(J[c + "_g"].mean(), J[c + "_f"].mean()):+.2f}%), Wilcoxon p {wilcoxon(J[c + "_f"], J[c + "_g"]).pvalue:.1e}')

# ---------------------------------------------------------------- V-F
h('V-F Communication performance')
for name in PLAT.values():
    for met in ('R_ser_Mbps', 'delay_per_cell_s'):
        x = S[(S.platform == name) & (S.metric == met) & (S.method == 'proposed')].set_index('baseline')
        p(f'{name} {met}: proposed vs ' + ', '.join(f'{b} {x.loc[b].gain_pct_of_baseline:+.1f}% (win {100 * x.loc[b].win_rate_mission_seed:.1f}%)'
                                                     for b in ('none', 'random', 'coverage')))
    X = P[P.platform == name]
    t = X.groupby(['K_interval', 'method'])[['R_ser_Mbps', 'delay_per_cell_s']].mean().unstack()
    k0, k1 = t.index[0], t.index[-1]
    p(f'{name}: No RIS from {k0} to {k1} cells: throughput {pct(t.R_ser_Mbps.none[k1], t.R_ser_Mbps.none[k0]):+.1f}%, '
      f'delay {pct(t.delay_per_cell_s.none[k1], t.delay_per_cell_s.none[k0]):+.1f}%; data per cell {(X.delivered_Mbit / X.n_congested_cells).mean():.0f} Mbit')
for pl, pre in (('facade', ''), ('ground', 'g')):
    if city(pre, 'city21') is None or city(pre, 'city22') is None:
        p(f'{PLAT[pl]}: facing angle and nearest-cell distance skipped (service tables of the test cities not built)'); continue
    cosv = {m: [] for m in ('coverage', 'proposed')}; snr = {m: [] for m in ('coverage', 'proposed')}; nn = {}
    for r in rows[pl]:
        C = city(pre, r['city'].lstrip('g'))
        for m in cosv:
            x = r[m]; snr[m].append(x['snr_db_mean_multi'])
            for s in x['plan']:
                for (i, o, q) in s['mem']:
                    if o > 0:
                        pz = np.array(x['panels'][str(int(C.cp.pid[int(r['cells'][i])][o]))]); v = np.array([s['xy'][0], s['xy'][1], EM.REF['cruise']]) - pz[:3]
                        cosv[m].append(max(0.0, float(v @ pz[3:6] / np.linalg.norm(v))))
        if pl == 'facade' and r['seed'] == 0:
            Gc = np.array([np.asarray(C.H[hh]['g'])[:2] for hh in r['cells']]); D = np.hypot(*(Gc[:, None] - Gc[None]).transpose(2, 0, 1))
            np.fill_diagonal(D, np.inf); K = len(r['cells'])
            b = next(f'{a}-{c}' for a, c in ((10, 13), (14, 17), (18, 21), (22, 25), (26, 30)) if a <= K <= c); nn.setdefault(b, []).append(D.min(1).mean())
    p(f'{PLAT[pl]}: median cosine between panel normal and direction to the UAV: proposed {np.median(cosv["proposed"]):.2f}, '
      f'coverage {np.median(cosv["coverage"]):.2f}; mean SNR proposed - coverage {np.mean(snr["proposed"]) - np.mean(snr["coverage"]):.2f} dB')
    if nn:
        p('mean distance to the nearest congested cell per interval (m): ' + ', '.join(f'{b}: {np.mean(v):.0f}' for b, v in sorted(nn.items(), key=lambda t: int(t[0].split("-")[0]))))

# ---------------------------------------------------------------- sensitivity
h('Parameter sensitivity (seed 0, 40 missions)')
for key in ('N', 'D'):
    f = os.path.join(FIG, 'sensitivity', f'sensitivity_{key}.csv')
    if not os.path.exists(f):
        p(f'{key}: no sensitivity results'); continue
    T = pd.read_csv(f); T = T[T.policy.str.startswith('fixed')]
    for name in PLAT.values():
        x = T[T.platform == name]; vals = sorted(x.value.unique())
        e = lambda m, v, c='E_UAV_kJ': float(x[(x.method == m) & (x.value == v)][c].iloc[0])
        sv = lambda b, v, c='saving_pct': float(x[(x.method == f'saving_vs_{b}') & (x.value == v)][c].iloc[0])
        p(f'{key}, {name}: saving vs No RIS (%) ' + ', '.join(f'{v:g}: {sv("none", v):.1f}' for v in vals)
          + '; vs Coverage (%) ' + ', '.join(f'{v:g}: {sv("coverage", v):.1f} (win {100 * sv("coverage", v, "win_rate"):.1f}%)' for v in vals))
        p(f'{key}, {name}: proposed hover {e("proposed", vals[0], "E_hover_kJ"):.2f} -> {e("proposed", vals[-1], "E_hover_kJ"):.2f} kJ, '
          f'flight {e("proposed", vals[0], "E_flight_kJ"):.2f} -> {e("proposed", vals[-1], "E_flight_kJ"):.2f} kJ; '
          f'No-RIS hover share {100 * e("none", vals[0], "E_hover_kJ") / e("none", vals[0]):.0f}% -> {100 * e("none", vals[-1], "E_hover_kJ") / e("none", vals[-1]):.0f}%; '
          f'R_ser proposed ' + ', '.join(f'{v:g}: {e("proposed", v, "R_ser"):.1f}' for v in vals))
        if key == 'D':
            v = vals[-1]
            p(f'D, {name}, D_k = {v:g}: hovering points No RIS {e("none", v, "stops"):.1f}, proposed {e("proposed", v, "stops"):.1f}; '
              f'flight part of the saving vs No RIS {e("none", v, "E_flight_kJ") - e("proposed", v, "E_flight_kJ"):.1f} of {e("none", v) - e("proposed", v):.1f} kJ')
        if key == 'N':
            p(f'N, {name}: proposed at 576 = {e("proposed", 576):.1f} kJ vs coverage at 1600 = {e("coverage", 1600):.1f} kJ')
    if key == 'N':
        Ns = sorted(T.value.unique()); p('far-field gain of successive N steps (dB): ' + ', '.join(f'{20 * np.log10(b / a):.1f}' for a, b in zip(Ns, Ns[1:])))
    d = [float(T[(T.method == 'proposed') & (T.value == v) & (T.platform == 'Facade RIS')].E_UAV_kJ.iloc[0]
               - T[(T.method == 'proposed') & (T.value == v) & (T.platform == 'Ground-mobile RIS')].E_UAV_kJ.iloc[0]) for v in sorted(T.value.unique())]
    p(f'{key}: proposed, Facade - Ground-mobile (kJ): ' + ', '.join(f'{a:+.2f}' for a in d))
    w = T[T.method == 'saving_vs_coverage']
    p(f'{key}: saving vs Coverage: CI lower bound > 0 at all points: {bool((w.ci95_low_kJ > 0).all())}; largest Wilcoxon p {w.wilcoxon_p.max():.1e}')

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('# Numbers quoted in Section V\n\nGenerated by `risuav/analysis/text_numbers.py` from the saved results.\n' + '\n'.join(L) + '\n')
print('written', OUT)
