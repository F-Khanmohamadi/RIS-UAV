"""Revised planner components under the coupled multi-RIS service model.
decode_stops / merge_identical / tsp_reorder: unchanged from plan.py (route order is independent of the channel).
repair():      after decoding, every cell is checked under the plan's FULL deployment (later decoding steps may add panels
               that change earlier cells' neighborhoods); an infeasible cell is relocated to its cheapest feasible point
               (same RIS option; other options only if that option has no feasible point). Counted and reported.
ils_coupled(): the proposed ILS (relocation of (RIS site, hovering point) and merge into an existing stop) with exact
               coupled move evaluation: a move that changes the cell's panel changes the deployment, so the energy change of
               EVERY other cell (and its feasibility at its current point) is included in the acceptance test."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import decode_stops, to_eval, merge_identical, tsp_reorder, _pos
from env_mr import E_F


def _assign(stops):
    return {i: (o, p) for s in stops for (i, o, p) in s['mem']}


def _locate(stops, i):
    for s, st in enumerate(stops):
        for j, (c, o, p) in enumerate(st['mem']):
            if c == i:
                return s, j
    raise KeyError(i)


def _dep(city, m, A, override=None):
    pids = set()
    for c, (o, p) in A.items():
        if override is not None and c == override[0]:
            o = override[1]
        if o > 0:
            pids.add(int(city.cp.pid[m['cells'][c]][o]))
    return frozenset(pids)


def repair(city, m, stops):
    """Returns (stops, n_repaired, n_unrepairable)."""
    stops = [dict(xy=s['xy'], g=s['g'], mem=list(s['mem'])) for s in stops]
    A = _assign(stops); dep = _dep(city, m, A); nrep = nbad = 0
    for i, (o, p) in sorted(A.items()):
        h = m['cells'][i]; sv = city.svc(h, o, dep)
        if sv['feas'][p]:
            continue
        s, j = _locate(stops, i); alone = len(stops[s]['mem']) == 1
        prv, nxt, cur = _pos(m, stops, s - 1), _pos(m, stops, s + 1), stops[s]['xy']
        T = city.H[h]; best = None
        for oo in [o] + [x for x in range(T['nopt']) if x != o]:
            dep2 = _dep(city, m, A, (i, oo)); s2 = city.svc(h, oo, dep2)
            P = np.flatnonzero(s2['feas'])
            if len(P) == 0:
                continue
            if dep2 != dep:                                        # other cells must stay feasible
                if any(not city.svc_point(m['cells'][c], oc, dep2, pc)[0] for c, (oc, pc) in A.items() if c != i):   # OPTIMISED (audit): point query
                    continue
            Q = T['U'][P]
            dL = (np.linalg.norm(Q - prv, axis=1) + np.linalg.norm(Q - nxt, axis=1)) if alone else (np.linalg.norm(Q - cur, axis=1) + np.linalg.norm(Q - nxt, axis=1))
            k = int(np.argmin(E_F * dL + s2['eps'][P]))
            best = (oo, int(P[k])); break                           # keep the policy's option whenever possible
        if best is None:
            nbad += 1; continue
        oo, q = best; nrep += 1
        new = dict(xy=T['U'][q], g=int(T['gidx'][q]), mem=[(i, oo, q)])
        if alone:
            stops[s] = new
        else:
            stops[s]['mem'].pop(j); stops.insert(s + 1, new)
        A[i] = (oo, q); dep = _dep(city, m, A)
    return stops, nrep, nbad


def ils_coupled(city, m, stops, n_iter, allowed, rng, merge=True, stats=None):
    stops = [dict(xy=s['xy'], g=s['g'], mem=list(s['mem'])) for s in stops]
    K = len(m['cells'])
    for _ in range(n_iter):
        i = int(rng.integers(K)); h = m['cells'][i]; T = city.H[h]
        A = _assign(stops); dep = _dep(city, m, A)
        s, j = _locate(stops, i); _, oc, pc = stops[s]['mem'][j]
        alone = len(stops[s]['mem']) == 1
        prv, nxt, cur = _pos(m, stops, s - 1), _pos(m, stops, s + 1), stops[s]['xy']
        e_cur = city.svc_point(h, oc, dep, pc)[1]                                    # OPTIMISED (audit): point query
        best = (0.0, None)
        gl = None
        for o in allowed[i]:
            dep2 = _dep(city, m, A, (i, o))
            dother = 0.0
            if dep2 != dep:                                          # coupling: other cells' energy and feasibility
                okall = True
                for c, (o2, p2) in A.items():
                    if c == i:
                        continue
                    e_new = city.svc_point(m['cells'][c], o2, dep2, p2)[1]          # OPTIMISED (audit): point query,
                    if not np.isfinite(e_new):                                       # identical to svc(...)['eps'][p2]
                        okall = False; break
                    dother += e_new - city.svc_point(m['cells'][c], o2, dep, p2)[1]
                if not okall:
                    continue
            sv = city.svc(h, o, dep2); P = np.flatnonzero(sv['feas'])
            if len(P) == 0:
                continue
            Q = T['U'][P]
            if alone:
                dL = np.linalg.norm(Q - prv, axis=1) + np.linalg.norm(Q - nxt, axis=1) - np.linalg.norm(cur - prv) - np.linalg.norm(cur - nxt)
            else:
                dL = np.linalg.norm(Q - cur, axis=1) + np.linalg.norm(Q - nxt, axis=1) - np.linalg.norm(cur - nxt)
            d = E_F * dL + sv['eps'][P] - e_cur + dother
            k = int(np.argmin(d))
            if d[k] < best[0] - 1e-6:
                best = (d[k], ('reloc', o, int(P[k])))
            if merge:
                if gl is None:
                    gl = {int(g): q for q, g in enumerate(T['gidx'])}
                for t, st in enumerate(stops):
                    if t == s or not st['mem'] or st['g'] not in gl:
                        continue
                    q = gl[st['g']]
                    if not sv['feas'][q]:
                        continue
                    dLm = (np.linalg.norm(prv - nxt) - np.linalg.norm(cur - prv) - np.linalg.norm(cur - nxt)) if alone else 0.0
                    dm = E_F * dLm + sv['eps'][q] - e_cur + dother
                    if dm < best[0] - 1e-6:
                        best = (dm, ('merge', o, q, t))
        if best[1] is None:
            continue
        mv = best[1]
        if stats is not None:
            stats['accepted'] = stats.get('accepted', 0) + 1
        if mv[0] == 'reloc':
            _, o, q = mv
            new = dict(xy=T['U'][q], g=int(T['gidx'][q]), mem=[(i, o, q)])
            if alone:
                stops[s] = new
            else:
                stops[s]['mem'].pop(j); stops.insert(s + 1, new)
        else:
            _, o, q, t = mv
            stops[t]['mem'].append((i, o, q)); stops[s]['mem'].pop(j)
            if alone:
                stops.pop(s)
    return stops
