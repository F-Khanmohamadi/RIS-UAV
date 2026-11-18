"""Multi-RIS coupled service model (revised System Model, Sec. II-E/F/G) on the exact city tables.

Channel of GUE k (cell i) at hovering point q under placement r (set A(r) of distinct deployed panels):
    h_k = mu_k + w_k + sum_{p in A(r), p != serving(i)} a_p z_{k,p},
    mu_k = mu_D a(cos_D) + [aligned serving panel of cell i]      (deterministic; as in the original code),
    w_k ~ CN(0, s2_k I_M)                                         (direct-link scatter: Rician LoS / Rayleigh NLoS),
    z_{k,p} = sum_n c_{k,p,n} e^{j theta_{p,n}},  theta i.i.d. U[0, 2pi) per attempt  ->  CN(0, P_{k,p})  (N = 1024, CLT),
    P_{k,p}(q) = kconst^2 * S2_p[k] * A_p(q)^2,  S2_p[k] = sum_n cos^{q_e}(theta_t)/d_t^2 (LoS GUE-panel),
    A_p(q) = cos^{q_e/2}(theta_r)/d_r (LoS panel-UAV, UAV in front of the panel; far-field arrival per panel).
Non-serving reflections are zero-mean, so they only add covariance Delta = sum_p P_{k,p} a_p a_p^H. For every CQI,
    |PRR_coupled - PRR_serving-only| <= TV(CN(mu, s2 I), CN(mu, s2 I + Delta)) <= 0.5 * M * sum_p P_{k,p} / s2  =: B_k(q)
(Pinsker + log(1+x) >= x - x^2/2; derivation in the report). Entries with B_k(q) < min(TOL, margin) keep the exact
serving-only value (CQI choice and feasibility provably unchanged; airtime change <= TOL / p^2 relative); all other
GUE-points are recomputed with the exact conditional Monte Carlo PRR (prr_exact.py, fixed common random numbers).
A panel chosen by several cells is one physical panel (deduplicated by centre and normal, 1 cm): it is the serving
panel of each of those cells and is never counted twice."""
import sys, os, pickle, numpy as np
from collections import OrderedDict
from scipy.stats import ncx2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_exact as BE

M = BE.Nr; TOL = 1e-4; NS_SAMPLES = 128
# FIX C1 (audit, 17 Nov 2026): the released build_exact.py does not define P_DG. The stored exact tables are already in
# physical units (D_g = 40 Mbit applied by build_exact), so they are used with factor 1; entries recomputed below
# are per unit backlog and are scaled by D_g. Verified by tests/unit_tests.py::test_coupling_units.
BASE_SCALE = 1.0; DG = float(BE.P['D_g'])
_rng = np.random.default_rng(20261029)
_Z = (_rng.standard_normal((NS_SAMPLES, 64)) + 1j * _rng.standard_normal((NS_SAMPLES, 64))) / np.sqrt(2)   # common random numbers
ANT = np.arange(M)


def steer(c):
    return np.exp(1j * np.pi * np.asarray(c)[..., None] * ANT)


class Coupler:
    def __init__(self, C, cache_cells=256, cache_tables=4000):
        """C: raw exact-city dict (build_exact output)."""
        self.C = C; self.city = BE.MapLoS(C); self.tabs = C['tabs']
        self.pid = []; self.panels = []; key2pid = {}
        for T in self.tabs:
            ids = [-1]
            for c6 in T['cands'].reshape(-1, 6):
                key = tuple(np.round(c6, 2))
                if key not in key2pid:
                    key2pid[key] = len(self.panels); self.panels.append(np.asarray(c6, float))
                ids.append(key2pid[key])
            self.pid.append(np.array(ids))
        self._pair = {}; self._base = OrderedDict(); self._tab = OrderedDict(); self._pt = OrderedDict(); self._fz = OrderedDict()
        self.cache_cells = cache_cells; self.cache_tables = cache_tables
        self.stats = dict(pair_computed=0, tables_computed=0, gue_points_recomputed=0, gue_points_screened=0, mc_evals=0)

    # ------------------------------------------------------------------ geometry
    def _geom(self, h):
        """direct-link quantities of cell h over its lattice (LRU cached): mu_d, s2d, cos_d (G x L)."""
        if h in self._base:
            self._base.move_to_end(h); return self._base[h]
        T = self.tabs[h]; gue = T['gue'].astype(float); U = T['U'].astype(float)
        los = np.unpackbits(T['los_d'], axis=1, count=T['L']).astype(bool)
        vec = U[None] - gue[:, None]; d = np.linalg.norm(vec, axis=-1)
        gd = (BE.lam / (4 * np.pi * d)) ** 2 * np.where(los, BE.PH.db(-BE.P['eta_los']), BE.PH.db(-BE.P['eta_nlos']))
        g = dict(mu_d=np.where(los, np.sqrt(gd * BE.Kl / (BE.Kl + 1)), 0.0), s2d=np.where(los, gd / (BE.Kl + 1), gd),
                 cos_d=-vec[..., 0] / d)
        self._base[h] = g
        if len(self._base) > self.cache_cells:
            self._base.popitem(last=False)
        return g

    def _panel_point(self, pid, U, losmask):
        ctr, nrm = self.panels[pid][:3], self.panels[pid][3:]
        vu = U - ctr[None]; d2 = np.linalg.norm(vu, axis=1); cr = np.clip((vu @ nrm) / d2, 0, 1)
        A = np.where(losmask & (cr > 0), cr ** (BE.qexp / 2) / d2, 0.0)
        return A, -vu[:, 0] / d2

    def pair(self, pid, h):
        """(S2 over the GUEs of cell h, lattice LoS mask) for physical panel pid, or None if it cannot reach cell h."""
        k = (pid, h)
        if k in self._pair:
            return self._pair[k]
        T = self.tabs[h]; o = np.flatnonzero(self.pid[h] == pid)
        if len(o):                                                       # own candidate: stored in the table
            o = int(o[0]); S2 = T['S2'][o - 1].astype(float)
            los = np.unpackbits(T['los_ru'][o - 1], count=T['L']).astype(bool)
        else:
            gue = T['gue'].astype(float); U = T['U'].astype(float)
            ctr, nrm, el = BE.panel_axes(self.panels[pid])
            los_gr = self.city.los(gue, np.repeat(ctr[None], len(gue), 0))
            if not los_gr.any():
                self._pair[k] = None; return None
            vg = gue[:, None] - el[None]; rt = np.linalg.norm(vg, axis=-1); ct = np.clip((vg @ nrm) / rt, 0, 1)
            S2 = np.where(los_gr, (ct ** BE.qexp / rt ** 2).sum(1), 0.0)
            if not (S2 > 0).any():
                self._pair[k] = None; return None
            vu = U - ctr[None]; front = (vu @ nrm) > 0
            los = np.zeros(len(U), bool)
            if front.any():
                los[front] = self.city.los(np.repeat(ctr[None], int(front.sum()), 0), U[front])
            self.stats['pair_computed'] += 1
        r = None if not (S2 > 0).any() or not los.any() else (S2.astype(np.float32), np.packbits(los))
        self._pair[k] = r
        return r

    # ------------------------------------------------------------------ service evaluation
    def _mu(self, h, o, g, kk, qq):
        """deterministic channel vectors for GUE/point pairs (kk, qq): direct LoS + aligned serving panel."""
        mu = g['mu_d'][kk, qq][:, None] * steer(g['cos_d'][kk, qq])
        if o > 0:
            T = self.tabs[h]; U = T['U'].astype(float)[qq]
            los = np.unpackbits(T['los_ru'][o - 1], count=T['L']).astype(bool)[qq]
            A, cr = self._panel_point(self.pid[h][o], U, los)
            Ar = BE.kconst * T['S'][o - 1].astype(float)[kk] * A
            ar = steer(cr)
            ip = (mu * ar.conj()).sum(1)
            ph = np.where(np.abs(ip) > 0, ip / np.maximum(np.abs(ip), 1e-300), 1.0)
            mu = mu + (Ar * ph)[:, None] * ar
        return mu

    def _coupled_prr(self, mu, s2, Acols, Pv, j):
        """exact conditional-MC PRR at CQI index j (scalar) for one GUE-point."""
        z = _Z[:, :len(Pv)] * np.sqrt(Pv)[None]
        m = mu[None] + z @ Acols.T
        lam = 2 * (np.abs(m) ** 2).sum(1) / s2
        self.stats['mc_evals'] += len(lam)
        return float(ncx2.sf(BE.THR[j] * BE.N0 / BE.Pt / (s2 / 2), 2 * M, lam).mean())

    def _coupled_service(self, mu, s2, Acols, Pv):
        """CQI search with the coupled PRR. Returns (has, j, p)."""
        f = lambda j: self._coupled_prr(mu, s2, Acols, Pv, j)
        p0 = f(0)
        if p0 < BE.P['prr_min']:
            return False, 0, p0
        j, pj = 0, p0
        while j + 1 < BE.NJ:
            pn = f(j + 1)
            if pn < BE.P['prr_min']:
                break
            j, pj = j + 1, pn
        return True, j, pj

    def _coupled_service_batch(self, mus, s2s, Acols_l, Pv_l):
        """OPTIMISED (audit, 17 Nov 2026), numerically identical to calling _coupled_service() for every GUE-point:
        same common random numbers, same per-sample ncx2 survival function (scipy ncx2._sf = ncx2.sf without argument
        checking), same sample mean. The CQI search is a vectorised binary search; it returns the same CQI as the
        original sequential search because the sample-mean PRR is strictly decreasing in the CQI threshold
        (tests/unit_tests.py T18 compares both on random inputs)."""
        n = len(mus); lam = np.zeros((n, NS_SAMPLES)); s2 = np.asarray(s2s, float)
        for t in range(n):
            Pv = Pv_l[t]; z = _Z[:, :len(Pv)] * np.sqrt(Pv)[None]
            mm = mus[t][None] + z @ Acols_l[t].T
            lam[t] = 2 * (np.abs(mm) ** 2).sum(1) / s2[t]
        self.stats['mc_evals'] += n * NS_SAMPLES

        def prr(idx, j):
            x = (BE.THR[j] * BE.N0 / BE.Pt / (s2[idx] / 2))[:, None]
            return ncx2._sf(np.broadcast_to(x, (len(idx), NS_SAMPLES)), 2 * M, lam[idx]).mean(1)
        allk = np.arange(n); p0 = prr(allk, np.zeros(n, int))
        has = p0 >= BE.P['prr_min']; j = np.zeros(n, int); p = p0.copy()
        act = np.flatnonzero(has)
        if len(act):
            top = np.full(len(act), BE.NJ - 1); pt = prr(act, top); ok = pt >= BE.P['prr_min']
            j[act[ok]] = BE.NJ - 1; p[act[ok]] = pt[ok]
            act = act[~ok]; lo = np.zeros(len(act), int); hi = np.full(len(act), BE.NJ - 1)   # PRR(lo) ok, PRR(hi) fails
            plo = p0[act]
            while len(act) and (hi - lo > 1).any():
                w = hi - lo > 1; mid = (lo + hi) // 2
                pm = np.ones(len(act)); pm[w] = prr(act[w], mid[w]); okm = w & (pm >= BE.P['prr_min'])
                lo = np.where(okm, mid, lo); plo = np.where(okm, pm, plo); hi = np.where(w & ~okm, mid, hi)
            j[act] = lo; p[act] = plo
        return has, j, p

    def table(self, h, o, deployed):
        """Service table of cell h with option o over its whole lattice, given the set 'deployed' of physical panel ids
        (all panels of the plan, including the serving one). Returns dict(tau, txb, dlv, feas) (numpy, per point)."""
        T = self.tabs[h]; serv = self.pid[h][o] if o > 0 else -1
        rel = tuple(sorted(p for p in set(deployed) if p != serv and self.pair(p, h) is not None))
        base = dict(tau=T['tau'][o].astype(float) * BASE_SCALE, txb=T['txb'][o].astype(float) * BASE_SCALE,
                    dlv=T['deliv'][o].astype(float) * BASE_SCALE, feas=T['feas'][o].copy())
        if not rel:
            return base
        key = (h, o, rel)
        if key in self._tab:
            self._tab.move_to_end(key); return self._tab[key]
        g = self._geom(h); U = T['U'].astype(float); L = T['L']; G = T['G']
        Pk = np.zeros((G, L)); As, Ss = [], []
        for p in rel:
            S2, lb = self._pair[(p, h)]
            A, cr = self._panel_point(p, U, np.unpackbits(lb, count=L).astype(bool))
            Pp = BE.kconst ** 2 * S2.astype(float)[:, None] * (A ** 2)[None]
            Pk += Pp; As.append(steer(cr)); Ss.append(Pp)
        Bnd = 0.5 * M * Pk / g['s2d']
        thr = np.minimum(TOL, T['margin'][o].astype(float))[None]
        aff = (Bnd >= thr) & (Bnd > 0)
        self.stats['gue_points_screened'] += int(((Bnd > 0) & ~aff).sum())
        out = {k: v.copy() for k, v in base.items()}
        pts = np.flatnonzero(aff.any(0))
        if len(pts):                                                     # recompute the affected points from scratch
            # OPTIMISED (audit): all affected points in one vectorised call; identical arithmetic per GUE-point
            QQ = np.repeat(pts, G); KK = np.tile(np.arange(G), len(pts))
            mu = self._mu(h, o, g, KK, QQ); s2 = g['s2d'][KK, QQ]
            has, j, p, _, _ = BE.service_exact((np.abs(mu) ** 2).sum(1), s2)
            has = has.copy(); j = j.copy(); p = p.copy()
            idx = np.flatnonzero(aff[KK, QQ]); Al, Pl = [], []
            for t in idx:
                k, q = KK[t], QQ[t]
                sel = [i for i in range(len(rel)) if Ss[i][k, q] > 0]
                Al.append(np.stack([As[i][q] for i in sel], 1)); Pl.append(np.array([Ss[i][k, q] for i in sel]))
            if len(idx):
                has[idx], j[idx], p[idx] = self._coupled_service_batch([mu[t] for t in idx], s2[idx], Al, Pl)
                self.stats['gue_points_recomputed'] += len(idx)
            etx = np.where(has, (1 - (1 - p) ** BE.P['Mmax']) / np.maximum(p, 1e-300), BE.P['Mmax'])
            air = np.where(has, DG * etx / (BE.P['B'] * BE.PH.CQI_SE[j]), 0.0).reshape(len(pts), G)
            hasm = has.reshape(len(pts), G); ok = hasm.all(1)
            out['feas'][pts] = ok
            out['tau'][pts] = np.where(ok, air.sum(1), np.inf)
            out['txb'][pts] = np.where(has, DG * etx, 0.0).reshape(len(pts), G).sum(1)
            out['dlv'][pts] = np.where(has, DG * (1 - (1 - p) ** BE.P['Mmax']), 0.0).reshape(len(pts), G).sum(1)
        self.stats['tables_computed'] += 1
        self._tab[key] = out
        if len(self._tab) > self.cache_tables:
            self._tab.popitem(last=False)
        return out

    def feas(self, h, o, deployed):
        """OPTIMISED (audit, 17 Nov 2026): feasibility mask of table(h, o, deployed) only (used by the coverage baseline).
        A point is feasible iff every GUE meets PRR_min at the lowest CQI; the same quantities as in table() are computed
        (PRR at CQI 1 with the same common random numbers), without the CQI search. Test T20 checks equality."""
        T = self.tabs[h]; serv = self.pid[h][o] if o > 0 else -1
        rel = tuple(sorted(p for p in set(deployed) if p != serv and self.pair(p, h) is not None))
        if not rel:
            return T['feas'][o].astype(bool)
        key = (h, o, rel)
        if key in self._tab:
            return self._tab[key]['feas']
        if key in self._fz:
            return self._fz[key]
        g = self._geom(h); U = T['U'].astype(float); L = T['L']; G = T['G']
        Pk = np.zeros((G, L)); As, Ss = [], []
        for p in rel:
            S2, lb = self._pair[(p, h)]
            A, cr = self._panel_point(p, U, np.unpackbits(lb, count=L).astype(bool))
            Pp = BE.kconst ** 2 * S2.astype(float)[:, None] * (A ** 2)[None]
            Pk += Pp; As.append(steer(cr)); Ss.append(Pp)
        Bnd = 0.5 * M * Pk / g['s2d']
        thr = np.minimum(TOL, T['margin'][o].astype(float))[None]
        aff = (Bnd >= thr) & (Bnd > 0)
        out = T['feas'][o].astype(bool).copy(); pts = np.flatnonzero(aff.any(0))
        if len(pts):
            QQ = np.repeat(pts, G); KK = np.tile(np.arange(G), len(pts))
            mu = self._mu(h, o, g, KK, QQ); s2 = g['s2d'][KK, QQ]
            has = BE.prr_exact((np.abs(mu) ** 2).sum(1), s2, np.zeros(len(KK), int)) >= BE.P['prr_min']
            idx = np.flatnonzero(aff[KK, QQ])
            if len(idx):
                lam = np.zeros((len(idx), NS_SAMPLES))
                for n, t in enumerate(idx):
                    k, q = KK[t], QQ[t]
                    sel = [i for i in range(len(rel)) if Ss[i][k, q] > 0]
                    Acols = np.stack([As[i][q] for i in sel], 1); Pv = np.array([Ss[i][k, q] for i in sel])
                    mm = mu[t][None] + (_Z[:, :len(Pv)] * np.sqrt(Pv)[None]) @ Acols.T
                    lam[n] = 2 * (np.abs(mm) ** 2).sum(1) / s2[t]
                x = (BE.THR[0] * BE.N0 / BE.Pt / (s2[idx] / 2))[:, None]
                p0 = ncx2._sf(np.broadcast_to(x, lam.shape), 2 * M, lam).mean(1)
                has[idx] = p0 >= BE.P['prr_min']
            out[pts] = has.reshape(len(pts), G).all(1)
        self._fz[key] = out
        if len(self._fz) > 4000:
            self._fz.popitem(last=False)
        return out

    def point(self, h, o, deployed, q):
        """OPTIMISED (audit, 17 Nov 2026): table(h, o, deployed) restricted to one lattice point q, computed with the same
        arithmetic (tests/unit_tests.py T19 checks equality with table()). Used where only the current point of a cell
        is needed (coupled ILS and repair: other cells' energy and feasibility at their current points)."""
        T = self.tabs[h]; serv = self.pid[h][o] if o > 0 else -1
        rel = tuple(sorted(p for p in set(deployed) if p != serv and self.pair(p, h) is not None))
        if not rel:
            return dict(tau=float(T['tau'][o][q]) * BASE_SCALE, txb=float(T['txb'][o][q]) * BASE_SCALE,
                        dlv=float(T['deliv'][o][q]) * BASE_SCALE, feas=bool(T['feas'][o][q]))
        key = (h, o, rel)
        if key in self._tab:
            t = self._tab[key]; return dict(tau=float(t['tau'][q]), txb=float(t['txb'][q]), dlv=float(t['dlv'][q]), feas=bool(t['feas'][q]))
        pk = (h, o, rel, q)
        if pk in self._pt:
            return self._pt[pk]
        g = self._geom(h); L = T['L']; G = T['G']; Uq = T['U'].astype(float)[q:q + 1]
        Pk = np.zeros(G); As, Ss = [], []
        for p in rel:
            S2, lb = self._pair[(p, h)]
            A, cr = self._panel_point(p, Uq, np.unpackbits(lb, count=L).astype(bool)[q:q + 1])
            Pp = BE.kconst ** 2 * S2.astype(float) * (A[0] ** 2)
            Pk += Pp; As.append(steer(cr[0])); Ss.append(Pp)
        Bnd = 0.5 * M * Pk / g['s2d'][:, q]
        thr = min(TOL, float(T['margin'][o][q]))
        aff = (Bnd >= thr) & (Bnd > 0)
        if not aff.any():
            r = dict(tau=float(T['tau'][o][q]) * BASE_SCALE, txb=float(T['txb'][o][q]) * BASE_SCALE,
                     dlv=float(T['deliv'][o][q]) * BASE_SCALE, feas=bool(T['feas'][o][q]))
        else:
            KK = np.arange(G); QQ = np.full(G, q)
            mu = self._mu(h, o, g, KK, QQ); s2 = g['s2d'][KK, QQ]
            has, j, p_, _, _ = BE.service_exact((np.abs(mu) ** 2).sum(1), s2)
            has = has.copy(); j = j.copy(); p_ = p_.copy()
            idx = np.flatnonzero(aff); Al, Pl = [], []
            for k in idx:
                sel = [i for i in range(len(rel)) if Ss[i][k] > 0]
                Al.append(np.stack([As[i] for i in sel], 1)); Pl.append(np.array([Ss[i][k] for i in sel]))
            has[idx], j[idx], p_[idx] = self._coupled_service_batch([mu[k] for k in idx], s2[idx], Al, Pl)
            self.stats['gue_points_recomputed'] += len(idx)
            etx = np.where(has, (1 - (1 - p_) ** BE.P['Mmax']) / np.maximum(p_, 1e-300), BE.P['Mmax'])
            air = np.where(has, DG * etx / (BE.P['B'] * BE.PH.CQI_SE[j]), 0.0); ok = bool(has.all())
            r = dict(tau=float(air.sum()) if ok else np.inf, txb=float(np.where(has, DG * etx, 0.0).sum()),
                     dlv=float(np.where(has, DG * (1 - (1 - p_) ** BE.P['Mmax']), 0.0).sum()), feas=ok)
        self._pt[pk] = r
        if len(self._pt) > 200000:
            self._pt.popitem(last=False)
        return r

    # RECONSTRUCTED (audit, 17 Nov 2026): eval_mr.py calls Coupler.details(), which the released coupling.py does not
    # contain (finding C1b). Per-GUE quantities at one point, computed exactly as in table() (same screening bound,
    # same coupled CQI search), so that details() and table() always agree (tests/unit_tests.py T17).
    def details(self, h, o, q, deployed):
        T = self.tabs[h]; serv = self.pid[h][o] if o > 0 else -1; G = T['G']; L = T['L']
        rel = tuple(sorted(p for p in set(deployed) if p != serv and self.pair(p, h) is not None))
        g = self._geom(h); U = T['U'].astype(float); kk = np.arange(G); qq = np.full(G, q)
        mu = self._mu(h, o, g, kk, qq); s2 = g['s2d'][:, q]; mu2 = (np.abs(mu) ** 2).sum(1)
        has, j, p, _, _ = BE.service_exact(mu2, s2)
        has = has.copy(); j = j.copy(); p = p.copy(); Pns = np.zeros(G); coupled = np.zeros(G, bool)
        if rel:
            As, Ss = [], []
            for pp in rel:
                S2, lb = self._pair[(pp, h)]
                A, cr = self._panel_point(pp, U[q:q + 1], np.unpackbits(lb, count=L).astype(bool)[q:q + 1])
                Pp = BE.kconst ** 2 * S2.astype(float) * A[0] ** 2
                Pns += Pp; As.append(steer(cr[0])); Ss.append(Pp)
            Bnd = 0.5 * M * Pns / s2
            thr = np.minimum(TOL, T['margin'][o].astype(float)[q])
            aff = (Bnd >= thr) & (Bnd > 0)
            if aff.any():                                            # table() recomputes the whole point in this case
                ks = np.flatnonzero(aff); Al, Pl = [], []
                for k in ks:
                    sel = [i for i in range(len(rel)) if Ss[i][k] > 0]
                    Al.append(np.stack([As[i] for i in sel], 1)); Pl.append(np.array([Ss[i][k] for i in sel]))
                has[ks], j[ks], p[ks] = self._coupled_service_batch([mu[k] for k in ks], s2[ks], Al, Pl)
                coupled[ks] = True
        return dict(has=has, j=j, p=p, coupled=coupled, Pns=Pns, mu2=mu2, s2=s2)
