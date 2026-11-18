"""Decoding to plans, the proposed ILS (Sec. IV-D), the ECTD ILS (ECTD Alg. 3, adapted) and TSP reordering."""
import numpy as np, elkai
from env import E_F, P_T, P_RX, E_DEC, E_BH, evaluate


def decode_stops(city, m, seq_row, Kp):
    """seq_row: (steps, 3) array of (node, option, local point); -1 rows are padding.
    Returns stops in decoded order; consecutive identical points form one group."""
    stops = []
    for node, o, p in seq_row:
        if node < 0:
            continue
        if node == Kp:
            stops.append(dict(xy=np.asarray(m['dest'], float), g=-1, mem=[]))
            continue
        T = city.H[m['cells'][node]]
        xy = T['U'][p]; g = int(T['gidx'][p])
        if stops and stops[-1]['g'] == g and stops[-1]['mem']:
            stops[-1]['mem'].append((int(node), int(o), int(p)))
        else:
            stops.append(dict(xy=xy, g=g, mem=[(int(node), int(o), int(p))]))
    return stops


def to_eval(stops):
    return [(s['xy'], s['mem']) for s in stops]


def merge_identical(stops):
    out, idx = [], {}
    for s in stops:
        if s['mem'] and s['g'] in idx:
            out[idx[s['g']]]['mem'].extend(s['mem'])
        else:
            if s['mem']:
                idx[s['g']] = len(out)
            out.append(dict(xy=s['xy'], g=s['g'], mem=list(s['mem'])))
    return out


def tsp_reorder(m, stops):
    stops = merge_identical(stops)
    X = np.vstack([m['depot']] + [s['xy'] for s in stops])
    n = len(X)
    if n <= 3:
        return stops
    Dm = np.rint(np.linalg.norm(X[:, None] - X[None], axis=-1) * 10).astype(int)
    t = elkai.solve_int_matrix(Dm.tolist(), runs=4)
    i = t.index(0); t = t[i:] + t[:i]
    return [stops[j - 1] for j in t[1:]]


def _locate(stops, i):
    for s, st in enumerate(stops):
        for j, (c, o, p) in enumerate(st['mem']):
            if c == i:
                return s, j
    raise KeyError(i)


def _pos(m, stops, s):
    return np.asarray(m['depot'], float) if s < 0 or s >= len(stops) else stops[s]['xy']


def ils_proposed(city, m, stops, n_iter, allowed, rng):
    """Relocation and merge moves on (RIS site, hovering point) of a random cell; accept if E_UAV decreases.
    allowed[i]: list of options cell i may use (all options for the proposed method, one for fixed placements)."""
    stops = [dict(xy=s['xy'], g=s['g'], mem=list(s['mem'])) for s in stops]
    K = len(m['cells'])
    for _ in range(n_iter):
        i = int(rng.integers(K)); T = city.H[m['cells'][i]]
        s, j = _locate(stops, i); _, oc, pc = stops[s]['mem'][j]
        alone = len(stops[s]['mem']) == 1
        prv, nxt, cur = _pos(m, stops, s - 1), _pos(m, stops, s + 1), stops[s]['xy']
        e_cur = T['eps'][oc, pc]
        best = (0.0, None)
        for o in allowed[i]:
            P = np.flatnonzero(T['feas'][o])
            if len(P) == 0:
                continue
            Q = T['U'][P]
            if alone:
                dL = np.linalg.norm(Q - prv, axis=1) + np.linalg.norm(Q - nxt, axis=1) \
                    - np.linalg.norm(cur - prv) - np.linalg.norm(cur - nxt)
            else:
                dL = np.linalg.norm(Q - cur, axis=1) + np.linalg.norm(Q - nxt, axis=1) - np.linalg.norm(cur - nxt)
            d = E_F * dL + T['eps'][o, P] - e_cur
            k = int(np.argmin(d))
            if d[k] < best[0] - 1e-6:
                best = (d[k], ('reloc', o, int(P[k])))
            # merge into an existing stop whose point lies in N_i(o)
            gl = {int(g): q for q, g in enumerate(T['gidx'])}
            for t, st in enumerate(stops):
                if t == s or not st['mem'] or st['g'] not in gl:
                    continue
                q = gl[st['g']]
                if not T['feas'][o, q]:
                    continue
                dLm = (np.linalg.norm(prv - nxt) - np.linalg.norm(cur - prv) - np.linalg.norm(cur - nxt)) if alone else 0.0
                dm = E_F * dLm + T['eps'][o, q] - e_cur
                if dm < best[0] - 1e-6:
                    best = (dm, ('merge', o, q, t))
        if best[1] is None:
            continue
        mv = best[1]
        if mv[0] == 'reloc':
            _, o, q = mv
            new = dict(xy=T['U'][q], g=int(T['gidx'][q]), mem=[(i, o, q)])
            if alone:
                stops[s] = new
            else:
                stops[s]['mem'].pop(j); stops.insert(s + 1, new)
        else:
            _, o, q, t = mv
            stops[t]['mem'].append((i, o, q))
            stops[s]['mem'].pop(j)
            if alone:
                stops.pop(s)
    return stops


def ils_ectd(city, m, stops, n_iter, rng, omega=0.8):
    """ECTD Alg. 3 adapted to the lattice: random cluster; relocate its CL to the minimiser of
    omega E_g,i + (1 - omega) E_comm,i over N_i(0) (ECTD's per-cluster subproblem); keep it if it is closer to
    the cluster than the nearest other CL (case 1), otherwise merge with that CL (case 2, only if that CL is
    feasible for the cluster). Moves are applied as in ECTD (no acceptance test)."""
    stops = [dict(xy=s['xy'], g=s['g'], mem=list(s['mem'])) for s in stops]
    K = len(m['cells'])
    for _ in range(n_iter):
        i = int(rng.integers(K)); T = city.H[m['cells'][i]]
        s, j = _locate(stops, i); alone = len(stops[s]['mem']) == 1
        P = np.flatnonzero(T['feas'][0])
        tau = T['tau'][0, P]
        loc = omega * P_T * tau + (1 - omega) * (P_RX * tau + E_DEC * T['txb'][0, P] + E_BH * T['dlv'][0, P])
        q = int(P[np.argmin(loc)]); qxy = T['U'][q]
        others = [t for t, st in enumerate(stops) if st['mem'] and t != s]
        gl = {int(g): k for k, g in enumerate(T['gidx'])}
        if others:
            dd = [np.linalg.norm(stops[t]['xy'] - T['g']) for t in others]
            t = others[int(np.argmin(dd))]
            if np.linalg.norm(qxy - T['g']) >= min(dd) and stops[t]['g'] in gl and T['feas'][0, gl[stops[t]['g']]]:
                stops[t]['mem'].append((i, 0, gl[stops[t]['g']])); stops[s]['mem'].pop(j)
                if alone:
                    stops.pop(s)
                continue
        new = dict(xy=qxy, g=int(T['gidx'][q]), mem=[(i, 0, q)])
        if alone:
            stops[s] = new
        else:
            stops[s]['mem'].pop(j); stops.insert(s + 1, new)
    return stops


def fallback_check(r):
    return r['battery_ok'] and r['deadline_ok']
