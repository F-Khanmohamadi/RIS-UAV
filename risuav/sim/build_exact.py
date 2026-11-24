
import sys, os, pickle, time, numpy as np
from scipy.stats import ncx2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics as PH

P = PH.REF; lam = PH.C0 / P['fc']; Nr = P['Nr']
Pt = PH.db(P['Ptx_dbm'] - 30); N0 = PH.db(-174 + 10 * np.log10(P['B']) + P['nf_db'] - 30)
nn = P['ris_n']; qexp = np.pi / 2 - 1.0; Gel = np.pi; dxdy = (lam / 2) ** 2
kconst = np.sqrt(Gel * dxdy * lam ** 2 / (64 * np.pi ** 3)) * P['ris_gamma']
off = (np.arange(nn) - (nn - 1) / 2) * lam / 2
THR = 2.0 ** PH.CQI_SE - 1.0; NJ = len(THR); Kl = PH.db(P['K_los_db'])


class MapLoS(PH.City):
    def __init__(self, C):
        self.H, self.p, self.w, self.lo = C['H'], C['p'], C['w'], C['lo']; self.n = self.H.shape[0]


def panel_axes(c6):
    ctr, nrm = np.asarray(c6[:3], float), np.asarray(c6[3:], float)
    u = np.array([-nrm[1], nrm[0], 0.0]); u /= np.linalg.norm(u); v = np.cross(nrm, u); v /= np.linalg.norm(v)
    el = (ctr[None, None] + off[:, None, None] * u + off[None, :, None] * v).reshape(-1, 3)
    return ctr, nrm, el


def prr_exact(mu2, s2, j):
    """PRR at CQI index j (array) for ||h||^2 with mean ||mu||^2 = mu2 and diffuse variance s2 per antenna."""
    thr = THR[j] * N0 / Pt
    return ncx2.sf(thr / (s2 / 2), 2 * Nr, 2 * mu2 / s2)


def service_exact(mu2, s2):
    """Exact CQI selection. Returns has, j (index), p_j, p_{j+1} (0 if j is the top CQI), p_1 (for infeasible GUEs)."""
    shp = mu2.shape; mu2 = mu2.ravel(); s2 = s2.ravel(); n = len(mu2)
    p1 = prr_exact(mu2, s2, np.zeros(n, int))
    has = p1 >= P['prr_min']
    lo = np.zeros(n, int); hi = np.full(n, NJ - 1)                      
    ptop = prr_exact(mu2, s2, hi); top = has & (ptop >= P['prr_min']); lo[top] = hi[top]
    act = has & ~top
    while act.any():                                                     
        mid = (lo + hi + 1) // 2
        pm = np.ones(n); pm[act] = prr_exact(mu2[act], s2[act], mid[act])
        ok = pm >= P['prr_min']
        lo = np.where(act & ok, mid, lo); hi = np.where(act & ~ok, mid - 1, hi)
        act = act & (lo < hi)
    j = lo
    pj = np.where(has, prr_exact(mu2, s2, j), 0.0)
    pj1 = np.where(has & (j < NJ - 1), prr_exact(mu2, s2, np.minimum(j + 1, NJ - 1)), 0.0)
    return has.reshape(shp), j.reshape(shp), pj.reshape(shp), pj1.reshape(shp), p1.reshape(shp)


def cell_tables(city, T, check_ft=None):
    gue = T['gue'].astype(np.float64); U = T['U'].astype(np.float64); G, L = len(gue), len(U)
    cands = T['cands'].reshape(-1, 6); R = len(cands)
    los_d = city.los(np.repeat(gue, L, 0), np.tile(U, (G, 1))).reshape(G, L)
    vec = U[None] - gue[:, None]; d = np.linalg.norm(vec, axis=-1)
    gd = (lam / (4 * np.pi * d)) ** 2 * np.where(los_d, PH.db(-P['eta_los']), PH.db(-P['eta_nlos']))
    mu_d = np.where(los_d, np.sqrt(gd * Kl / (Kl + 1)), 0.0); s2d = np.where(los_d, gd / (Kl + 1), gd)
    cos_d = -vec[..., 0] / d
    S = np.zeros((R, G)); S2 = np.zeros((R, G)); A_ru = np.zeros((R, L)); cos_r = np.zeros((R, L)); los_ru = np.zeros((R, L), bool)
    for r in range(R):
        ctr, nrm, el = panel_axes(cands[r])
        los_gr = city.los(gue, np.repeat(ctr[None], G, 0))
        vg = gue[:, None] - el[None]; rt = np.linalg.norm(vg, axis=-1); ct = np.clip((vg @ nrm) / rt, 0, 1)
        S[r] = np.where(los_gr, (ct ** (qexp / 2) / rt).sum(1), 0.0); S2[r] = np.where(los_gr, (ct ** qexp / rt ** 2).sum(1), 0.0)
        vu = U - ctr[None]; d2 = np.linalg.norm(vu, axis=1); cr = np.clip((vu @ nrm) / d2, 0, 1)
        los_ru[r] = city.los(np.repeat(ctr[None], L, 0), U) & (cr > 0)
        A_ru[r] = np.where(los_ru[r], cr ** (qexp / 2) / d2, 0.0); cos_r[r] = -vu[:, 0] / d2
    n_ant = np.arange(Nr)
    out = {k: np.zeros((R + 1, L), np.float32) for k in ('tau', 'deliv', 'cov', 'txb', 'margin')}
    out['tau'][:] = np.inf
    ft_out = None
    if check_ft is not None:
        ft_out = np.full((R + 1, L), np.inf, np.float32)
    for o in range(R + 1):
        if o == 0:
            mu2 = Nr * mu_d ** 2
        else:
            Ar = kconst * S[o - 1][:, None] * A_ru[o - 1][None, :]
            ph = np.pi * (cos_r[o - 1][None, :] - cos_d)
            corr = np.abs(np.exp(1j * ph[..., None] * n_ant).sum(-1))
            mu2 = Nr * mu_d ** 2 + Nr * Ar ** 2 + 2 * mu_d * Ar * corr
        if check_ft is not None:                                         # original interpolated computation
            mean = Pt * (mu2 + Nr * s2d) / N0; K = mu2 / np.maximum(Nr * s2d, 1e-300)
            prr = np.stack([check_ft.prr(K, THR[j] / mean) for j in range(NJ)], 0); ok = prr >= P['prr_min']; hs = ok.any(0)
            jj = np.where(hs, NJ - 1 - np.argmax(ok[::-1], axis=0), 0); pp = np.take_along_axis(prr, jj[None], 0)[0]
            etx = (1 - (1 - pp) ** P['Mmax']) / np.maximum(pp, 1e-9)
            ft_out[o] = np.where(hs, P['D_g'] * etx / (P['B'] * PH.CQI_SE[jj]), 0.0).sum(0)
        has, j, p, p_up, p1 = service_exact(mu2, s2d)
        se = PH.CQI_SE[j]
        etx = np.where(has, (1 - (1 - p) ** P['Mmax']) / np.maximum(p, 1e-300), P['Mmax'])
        air = np.where(has, P['D_g'] * etx / (P['B'] * se), 0.0)
        cellok = has.all(0)
        out['tau'][o] = np.where(cellok, air.sum(0), np.inf)
        out['deliv'][o] = np.where(has, P['D_g'] * (1 - (1 - p) ** P['Mmax']), 0.0).sum(0)
        out['txb'][o] = np.where(has, P['D_g'] * etx, 0.0).sum(0)
        out['cov'][o] = has.mean(0)
        mg = np.where(has, np.minimum(p - P['prr_min'], np.where(j < NJ - 1, P['prr_min'] - p_up, 1.0)), P['prr_min'] - p1)
        out['margin'][o] = mg.min(0)
    res = dict(gidx=T['gidx'], U=T['U'], gue=T['gue'], cands=T['cands'], centre=T['centre'], G=G, S=S.astype(np.float32),
               S2=S2.astype(np.float32), feas=out['cov'] >= 1.0, tau=out['tau'], deliv=out['deliv'], cov=out['cov'],
               txb=out['txb'], margin=out['margin'], los_d=np.packbits(los_d, axis=1), los_ru=np.packbits(los_ru, axis=1), L=L)
    return res, ft_out


def build(src, dst, check=False, ncheck=40):
    t0 = time.time(); C = pickle.load(open(src, 'rb')); city = MapLoS(C)
    ft = PH.FadingTables(Nr) if check else None
    tabs = []
    for h, T in enumerate(C['tabs']):
        if check and h >= ncheck:
            break
        res, ftv = cell_tables(city, T, ft)
        if check:
            a = np.asarray(T['tau'], np.float64); b = ftv.astype(np.float64); fin = np.isfinite(a) & (a > 0)
            # original table stores tau for every option; compare where the original cell is feasible
            feas_o = np.asarray(T['feas'], bool)
            rel = np.abs(b[feas_o] - a[feas_o]) / a[feas_o]
            tabs.append(dict(h=h, max_rel=float(rel.max()) if rel.size else 0.0,
                             feas_same=bool(((res['cov'] >= 1) == feas_o).mean() > 0) , n_feas_diff=int(((res['cov'] >= 1) != feas_o).sum())))
            continue
        tabs.append(res)
        if h % 100 == 0:
            print(f'  {dst.replace(chr(92), "/").split("/")[-1]}: cell {h}/{len(C["tabs"])}  {time.time() - t0:.0f} s', flush=True)
    if check:
        return tabs
    D = {k: C[k] for k in C if k != 'tabs'}; D['tabs'] = tabs; D['prr_method'] = 'exact ncx2 (scipy.stats.ncx2.sf), monotone CQI search'
    with open(dst + '.tmp', 'wb') as f:
        pickle.dump(D, f, protocol=4)
    import os; os.replace(dst + '.tmp', dst)
    print(f'{dst}: {len(tabs)} cells in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    if '--check-original' in sys.argv:
        r = build(sys.argv[1], None, check=True)
        print('reproduction of the original interpolated tables (first 40 cells):  max relative tau difference',
              max(x['max_rel'] for x in r))
        print('cells x options x points where exact feasibility differs from the original table:', sum(x['n_feas_diff'] for x in r))
    else:
        os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
        build(sys.argv[1], sys.argv[2])
