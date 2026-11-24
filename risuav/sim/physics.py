import numpy as np
from scipy.stats import ncx2
from scipy.ndimage import map_coordinates

C0 = 3e8
CQI_SE = np.array([0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063, 2.7305,
                   3.3223, 3.9023, 4.5234, 5.1152, 5.5547])

REF = dict(
    city_size=3000.0, margin=600.0, bld_alpha=0.5, bld_beta=300.0, bld_gamma=20.0,
    isd=500.0, hot_per_sector=4, hot_radius=40.0, hot_min_sep=80.0, hot_min_bts=75.0,
    G_mu=15, G_sd=5, G_lo=5, G_hi=30,
    z_gue=1.5,
    fc=3.5e9, ris_n=32, ris_h=15.0, ris_min_bld=8.0, ris_gamma=0.9, R_dep=100.0, n_ris=10, n_ris_eval=24,
    ris_mode='facade', ris_h_ground=5.0, ground_pitch=20.0, ground_tilts=(0.0, 45.0), n_ris_eval_ground=48, ris_offset=0.5,
    alts=(120.0,), cruise=120.0, d_sep=15.0, pitch=25.0, R_hov=450.0, nofly_clear=10.0, Nr=4,
    Ptx_dbm=23.0, B=20e6, nf_db=7.0, eta_los=1.6, eta_nlos=23.0, K_los_db=10.0,
    prr_min=0.9, Mmax=5, pkt_bits=8192, D_g=40e6, cov_min=1.0, p_rx_w=1.0,
    P0=79.86, Pi=88.63, Utip=120.0, v0=4.03, d0=0.6, rho=1.225, s=0.05, A=0.503, W=20.0, v_z=5.0,
)


def db(x): return 10.0 ** (np.asarray(x) / 10.0)


def rotor_power(V, P):
    V = np.asarray(V, float)
    return (P['P0'] * (1 + 3 * V ** 2 / P['Utip'] ** 2)
            + P['Pi'] * np.sqrt(np.sqrt(1 + V ** 4 / (4 * P['v0'] ** 4)) - V ** 2 / (2 * P['v0'] ** 2))
            + 0.5 * P['d0'] * P['rho'] * P['s'] * P['A'] * V ** 3)


def flight_consts(P):
    V = np.linspace(1, 35, 34001)
    pw = rotor_power(V, P)
    k = np.argmin(pw / V)
    Ph = float(rotor_power(0.0, P))
    return dict(V=float(V[k]), e_fly=float(pw[k] / V[k]), P_hover=Ph,
                e_up=Ph / P['v_z'] + P['W'], e_down=Ph / P['v_z'], t_vert=1.0 / P['v_z'])


class City:
    def __init__(self, P, seed):
        rng = np.random.default_rng(seed)
        self.P = P
        self.p = 1000.0 / np.sqrt(P['bld_beta'])
        self.w = 1000.0 * np.sqrt(P['bld_alpha'] / P['bld_beta'])
        self.lo = -P['margin']
        n = int((P['city_size'] + 2 * P['margin']) / self.p) + 2
        self.n = n
        self.H = rng.rayleigh(P['bld_gamma'], size=(n, n)).astype(np.float32)
        self.rng = rng

    def cell(self, x, y):
        ix = np.floor((x - self.lo) / self.p).astype(np.int64)
        iy = np.floor((y - self.lo) / self.p).astype(np.int64)
        return ix, iy

    def height_at(self, x, y):
        ix, iy = self.cell(x, y)
        ix = np.clip(ix, 0, self.n - 1); iy = np.clip(iy, 0, self.n - 1)
        inside = ((x - self.lo - ix * self.p) < self.w) & ((y - self.lo - iy * self.p) < self.w)
        return np.where(inside, self.H[ix, iy], 0.0)

    def los(self, A, B, step=4.0, chunk=3_000_000):
        A = np.atleast_2d(A).astype(np.float64); B = np.atleast_2d(B).astype(np.float64)
        N = len(A); out = np.ones(N, bool)
        dh = np.hypot(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])
        S = max(2, int(np.ceil(dh.max() / step)) + 1)
        t = (np.arange(1, S) / S)[None, :].astype(np.float32)
        rows = max(1, chunk // S)
        for a in range(0, N, rows):
            sl = slice(a, a + rows)
            x = (A[sl, 0:1] + t * (B[sl, 0:1] - A[sl, 0:1])).astype(np.float32)
            y = (A[sl, 1:2] + t * (B[sl, 1:2] - A[sl, 1:2])).astype(np.float32)
            z = (A[sl, 2:3] + t * (B[sl, 2:3] - A[sl, 2:3])).astype(np.float32)
            h = self.height_at(x, y)
            out[sl] = ~(z < h).any(1)
        return out


def hex_sites(P):
    isd = P['isd']; L = P['city_size']
    pts = []
    rows = int(L / (isd * np.sqrt(3) / 2)) + 2
    for r in range(rows):
        y = r * isd * np.sqrt(3) / 2
        off = (r % 2) * isd / 2
        for c in range(int(L / isd) + 2):
            x = c * isd + off
            if 0 <= x <= L and 0 <= y <= L:
                pts.append((x, y))
    return np.array(pts)


def drop_hotspots(city, P, rng):
    sites = hex_sites(P)
    n_target = 3 * P['hot_per_sector'] * len(sites)
    L = P['city_size']; hots = []
    tries = 0
    while len(hots) < n_target and tries < 200000:
        tries += 1
        q = rng.uniform(0, L, 2)
        if city.height_at(q[0:1], q[1:2])[0] > 0:
            continue
        if np.min(np.hypot(*(sites - q).T)) < P['hot_min_bts']:
            continue
        if hots and np.min(np.hypot(*(np.array(hots) - q).T)) < P['hot_min_sep']:
            continue
        hots.append(q)
    return sites, np.array(hots)


def drop_gues(city, c, P, rng):
    G = int(np.clip(round(rng.normal(P['G_mu'], P['G_sd'])), P['G_lo'], P['G_hi']))
    out = []
    while len(out) < G:
        r = P['hot_radius'] * np.sqrt(rng.random(4 * G)); a = rng.uniform(0, 2 * np.pi, 4 * G)
        x = c[0] + r * np.cos(a); y = c[1] + r * np.sin(a)
        ok = city.height_at(x, y) == 0
        out.extend(np.c_[x[ok], y[ok]].tolist())
    out = np.array(out[:G])
    return np.c_[out, np.full(G, P['z_gue'])]


def ris_candidates(city, c, gue, P):
    p, w = city.p, city.w
    ix0, iy0 = city.cell(np.array([c[0]]), np.array([c[1]]))
    rad = int(np.ceil(P['R_dep'] / p)) + 1
    cands = []
    for ix in range(ix0[0] - rad, ix0[0] + rad + 1):
        for iy in range(iy0[0] - rad, iy0[0] + rad + 1):
            if not (0 <= ix < city.n and 0 <= iy < city.n): continue
            hb = float(city.H[ix, iy])
            if hb < P['ris_min_bld']: continue
            x0 = city.lo + ix * p; y0 = city.lo + iy * p
            hs = [min(P['ris_h'], hb - 1.0)]
            if hb - 2.0 >= P['ris_h'] + P['d_sep']:
                hs.append(hb - 2.0)
            for h, (fx, fy, nx, ny, ux, uy) in [(h, f) for h in hs for f in [(x0 + w / 2, y0, 0, -1, 1, 0), (x0 + w / 2, y0 + w, 0, 1, 1, 0),
                                             (x0, y0 + w / 2, -1, 0, 0, 1), (x0 + w, y0 + w / 2, 1, 0, 0, 1)]]:
                ctr = np.array([fx + nx * P['ris_offset'], fy + ny * P['ris_offset'], h])
                if np.hypot(ctr[0] - c[0], ctr[1] - c[1]) > P['R_dep']: continue
                cands.append((ctr, np.array([nx, ny, 0.0]), np.array([ux, uy, 0.0])))
    return cands


def ground_candidates(city, c, P):
    g = np.arange(-P['R_dep'], P['R_dep'] + 1e-6, P['ground_pitch'])
    X, Y = np.meshgrid(c[0] + g, c[1] + g, indexing='ij'); X = X.ravel(); Y = Y.ravel()
    ok = (np.hypot(X - c[0], Y - c[1]) <= P['R_dep']) & (city.height_at(X, Y) == 0)
    pos = np.c_[X[ok], Y[ok], np.full(ok.sum(), P['ris_h_ground'])]
    cands = []
    for q in pos:
        for a in np.arange(8) * np.pi / 4:
            for t in np.radians(P['ground_tilts']):
                n = np.array([np.cos(t) * np.cos(a), np.cos(t) * np.sin(a), np.sin(t)])
                cands.append((q.copy(), n, np.array([-np.sin(a), np.cos(a), 0.0])))
    return cands


class FadingTables:

    def __init__(self, Nr, nK=81, nX=241):
        self.lK = np.linspace(-3, 4, nK); self.lX = np.linspace(-4, 1.2, nX)
        K = 10 ** self.lK[:, None]; X = 10 ** self.lX[None, :]
        dof = 2 * Nr
        lam = 2 * Nr * K
        thr = X * 2 * (Nr + Nr * K)
        self.T = ncx2.sf(thr, dof, lam).astype(np.float32)
        self.dlK = self.lK[1] - self.lK[0]; self.dlX = self.lX[1] - self.lX[0]

    def prr(self, K, x):
        iK = (np.log10(np.clip(K, 1e-3, 1e4)) - self.lK[0]) / self.dlK
        iX = (np.log10(np.clip(x, 1e-4, 10 ** 1.2)) - self.lX[0]) / self.dlX
        v = map_coordinates(self.T, [iK.ravel(), iX.ravel()], order=1, mode='nearest').reshape(np.shape(K))
        return np.where(x < 1e-4, 1.0, np.where(x > 10 ** 1.2, 0.0, v))


def global_lattice(P):
    pc = P['pitch']; lo = -P['margin'] + pc / 2; hi = P['city_size'] + P['margin']
    xs = np.arange(lo, hi, pc); nx = len(xs)
    return xs, nx


def hotspot_table(city, c, gue, cands, P, ft, xs, nx):
    lam = C0 / P['fc']; Nr = P['Nr']
    Pt = db(P['Ptx_dbm'] - 30); N0 = db(-174 + 10 * np.log10(P['B']) + P['nf_db'] - 30)
    i0 = np.searchsorted(xs, c[0] - P['R_hov']); i1 = np.searchsorted(xs, c[0] + P['R_hov'])
    j0 = np.searchsorted(xs, c[1] - P['R_hov']); j1 = np.searchsorted(xs, c[1] + P['R_hov'])
    II, JJ = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing='ij')
    X = xs[II.ravel()]; Y = xs[JJ.ravel()]
    keep = np.hypot(X - c[0], Y - c[1]) <= P['R_hov']
    II = II.ravel()[keep]; JJ = JJ.ravel()[keep]; X = X[keep]; Y = Y[keep]
    hb = city.height_at(X, Y)
    U, gidx = [], []
    for a, z in enumerate(P['alts']):
        ok = hb < z - P['nofly_clear']
        U.append(np.c_[X[ok], Y[ok], np.full(ok.sum(), z)])
        gidx.append((a * nx + II[ok]) * nx + JJ[ok])
    U = np.vstack(U); gidx = np.concatenate(gidx)
    L = len(U); G = len(gue)

    los_d = city.los(np.repeat(gue, L, 0), np.tile(U, (G, 1))).reshape(G, L)
    vec = U[None] - gue[:, None]
    d = np.linalg.norm(vec, axis=-1)
    fspl = (lam / (4 * np.pi * d)) ** 2
    gd = fspl * np.where(los_d, db(-P['eta_los']), db(-P['eta_nlos']))
    Kl = db(P['K_los_db'])
    mu_d_amp = np.where(los_d, np.sqrt(gd * Kl / (Kl + 1)), 0.0)
    sig2_d = np.where(los_d, gd / (Kl + 1), gd)
    cos_d = -vec[..., 0] / d
    R = len(cands)
    nn = P['ris_n']; q = np.pi / 2 - 1.0
    Gel = np.pi; dxdy = (lam / 2) ** 2
    kconst = np.sqrt(Gel * dxdy * lam ** 2 / (64 * np.pi ** 3)) * P['ris_gamma']
    off = (np.arange(nn) - (nn - 1) / 2) * lam / 2
    S = np.zeros((R, G)); A_ru = np.zeros((R, L)); cos_r = np.zeros((R, L))
    for r, (ctr, nrm, uax) in enumerate(cands):
        vax = np.cross(nrm, uax); vax = vax / np.linalg.norm(vax)
        el = ctr[None, None] + off[:, None, None] * uax[None, None] + off[None, :, None] * vax[None, None]
        el = el.reshape(-1, 3)
        los_gr = city.los(gue, np.repeat(ctr[None], G, 0))
        vg = gue[:, None] - el[None]
        rt = np.linalg.norm(vg, axis=-1)
        ct = np.clip((vg @ nrm) / rt, 0, 1)
        S[r] = np.where(los_gr, (ct ** (q / 2) / rt).sum(1), 0.0)
        vu = U - ctr[None]
        d2 = np.linalg.norm(vu, axis=1)
        cr = np.clip((vu @ nrm) / d2, 0, 1)
        los_ru = city.los(np.repeat(ctr[None], L, 0), U)
        A_ru[r] = np.where(los_ru & (cr > 0), cr ** (q / 2) / d2, 0.0)
        cos_r[r] = -vu[:, 0] / d2
    Mmax = P['Mmax']; Dg = P['D_g']; B = P['B']
    out_tau = np.full((R + 1, L), np.inf, np.float32)
    out_del = np.zeros((R + 1, L), np.float32)
    out_cov = np.zeros((R + 1, L), np.float32)
    out_air = np.zeros((R + 1, L), np.float32)
    out_txb = np.zeros((R + 1, L), np.float32)
    thr = (2.0 ** CQI_SE - 1.0)
    n_ant = np.arange(Nr)
    for o in range(R + 1):
        if o == 0:
            mu2 = Nr * mu_d_amp ** 2
        else:
            r = o - 1
            Ar = kconst * S[r][:, None] * A_ru[r][None, :]
            ph = np.pi * (cos_r[r][None, :] - cos_d)
            corr = np.abs(np.exp(1j * ph[..., None] * n_ant).sum(-1))
            mu2 = Nr * mu_d_amp ** 2 + Nr * Ar ** 2 + 2 * mu_d_amp * Ar * corr
        mean = Pt * (mu2 + Nr * sig2_d) / N0
        K = mu2 / np.maximum(Nr * sig2_d, 1e-300)
        prr = np.stack([ft.prr(K, thr[j] / mean) for j in range(len(CQI_SE))], 0)
        ok = prr >= P['prr_min']
        has = ok.any(0)
        j = np.where(has, len(CQI_SE) - 1 - np.argmax(ok[::-1], axis=0), 0)
        p = np.take_along_axis(prr, j[None], 0)[0]
        se = CQI_SE[j]
        etx = (1 - (1 - p) ** Mmax) / np.maximum(p, 1e-9)
        air = np.where(has, Dg * etx / (B * se), 0.0)
        dl = np.where(has, Dg * (1 - (1 - p) ** Mmax), 0.0)
        out_tau[o] = air.sum(0); out_del[o] = dl.sum(0); out_cov[o] = has.mean(0); out_air[o] = air.sum(0)
        out_txb[o] = np.where(has, Dg * etx, 0.0).sum(0)
    feas = out_cov >= P['cov_min']
    return dict(gidx=gidx.astype(np.int64), U=U.astype(np.float32), tau=out_tau, deliv=out_del, cov=out_cov, txb=out_txb,
                feas=feas, G=G, D=G * Dg, cands=np.array([np.r_[a, b] for a, b, _ in cands]).reshape(-1, 6),
                S=S.astype(np.float32), los_d_frac=float(los_d.mean()))


def build_city(seed, P, verbose=False):
    import time
    t = time.time()
    city = City(P, seed)
    rng = np.random.default_rng(seed + 1)
    sites, hots = drop_hotspots(city, P, rng)
    ft = FadingTables(P['Nr'])
    xs, nx = global_lattice(P)
    tabs = []
    for h, c in enumerate(hots):
        gue = drop_gues(city, c, P, rng)
        if P.get('ris_mode', 'facade') == 'ground':
            cands = ground_candidates(city, c, P)
            pos = np.unique(np.array([q for q, _, _ in cands]), axis=0)
            losp = {tuple(q): city.los(gue, np.repeat(q[None], len(gue), 0)) for q in pos}
            sc = np.array([int((losp[tuple(q)] & (((gue - q) @ n) > 0)).sum()) for q, n, _ in cands])
            keep = np.flatnonzero(sc >= max(1, 0.5 * sc.max()))
            feat = np.array([np.r_[cands[i][0][:2], 60 * cands[i][1]] for i in keep])
            sel = [int(np.argmax(sc[keep]))]
            dmin = np.linalg.norm(feat - feat[sel[0]], axis=1)
            while len(sel) < min(P['n_ris_eval_ground'], len(keep)):
                j = int(np.argmax(dmin)); sel.append(j); dmin = np.minimum(dmin, np.linalg.norm(feat - feat[j], axis=1))
            cands = [cands[keep[j]] for j in sel]
        else:
            cands = ris_candidates(city, c, gue, P)
            if cands:
                sc = []
                for ctr, nrm, _ in cands:
                    los = city.los(gue, np.repeat(ctr[None], len(gue), 0))
                    front = ((gue - ctr) @ nrm) > 0
                    sc.append(((los & front).sum(), -np.hypot(*(ctr[:2] - c))))
                order = sorted(range(len(cands)), key=lambda i: sc[i], reverse=True)
                kept = []
                for i in order:
                    if sc[i][0] <= 0: continue
                    q = cands[i][0]
                    if any(0 < np.linalg.norm(q - cands[j][0]) < P['d_sep'] for j in kept): continue
                    kept.append(i)
                    if len(kept) >= P['n_ris_eval']: break
                cands = [cands[i] for i in kept]
        T = hotspot_table(city, np.r_[c, 0.0], gue, cands, P, ft, xs, nx)
        if len(cands) > P['n_ris']:
            t0 = T['tau'][0]
            gain = [np.mean(np.clip((t0 - T['tau'][o]) / t0, 0, 1)) for o in range(1, len(cands) + 1)]
            keep = np.argsort(gain)[::-1][:P['n_ris']]
            sel = np.r_[0, keep + 1]
            for k in ('tau', 'deliv', 'cov', 'feas', 'txb'):
                T[k] = T[k][sel]
            T['cands'] = T['cands'][keep]; T['S'] = T['S'][keep]
        T['centre'] = c; T['gue'] = gue.astype(np.float32)
        tabs.append(T)
        if verbose and h % 50 == 0:
            print(f'  hotspot {h}/{len(hots)}  {time.time() - t:.0f}s', flush=True)
    return dict(seed=seed, P=P, sites=sites, hots=hots, tabs=tabs, xs=xs, nx=nx, H=city.H, p=city.p, w=city.w,
                lo=city.lo)
