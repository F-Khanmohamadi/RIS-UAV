import os, pickle, zlib, numpy as np
from physics import REF, flight_consts

FC = flight_consts(REF)
V_MR, E_F, P_H = FC['V'], FC['e_fly'], FC['P_hover']
P_T = 10 ** ((REF['Ptx_dbm'] - 30) / 10)
P_RX = 3.0 + REF['Nr'] * 0.5
E_DEC, E_BH, P_E = 0.8e-9, 0.25e-9, 7.8e-3
N_EL = REF['ris_n'] ** 2
H = REF['cruise']
E_VERT = H * (FC['e_up'] + FC['e_down'])
E_BAT, E_RES, T_MAX = 5.3e5, 0.1 * 5.3e5, 1800.0
D_G = 40e6
OMAX = 11
L_CITY = REF['city_size']


class City:

    def __init__(self, path):
        C = pickle.load(open(path, 'rb'))
        self.name = os.path.basename(path)[:-4]
        self.mode = C['P']['ris_mode']
        self.hots = C['hots']; self.sites = C['sites']
        self.H = []
        for T in C['tabs']:
            U = T['U'][:, :2].astype(np.float64)
            nopt = T['tau'].shape[0]
            feas = T['feas'].astype(bool)
            feas &= (np.hypot(*(U - T['centre']).T) <= REF['R_hov'])[None]
            tau = T['tau'].astype(np.float64) * D_G
            txb = T['txb'].astype(np.float64) * D_G
            dlv = T['deliv'].astype(np.float64) * D_G
            eps = (P_H + P_RX) * tau + E_DEC * txb + E_BH * dlv
            cands = T['cands'].reshape(-1, 6) if len(T['cands']) else np.zeros((0, 6))
            self.H.append(dict(U=U, gidx=T['gidx'], tau=tau, txb=txb, dlv=dlv, eps=eps, feas=feas, nopt=nopt,
                               cands=cands, g=np.asarray(T['centre'], float), G=int(T['G']),
                               area=feas.sum(1)))
        self.mode_flag = 1.0 if self.mode == 'facade' else 0.0


def seed_of(city_name):
    return zlib.crc32(city_name.lstrip('g').encode()) % 100000 * 1000


def make_mission(city, k, Kr=(10, 30), n_dist=(2, 3), sig=400.0):
    rng = np.random.default_rng(seed_of(city.name) + k)
    hots = city.hots; L = L_CITY
    K = int(rng.integers(Kr[0], Kr[1] + 1))
    nd = rng.integers(n_dist[0], n_dist[1] + 1)
    dc = rng.uniform(0.15 * L, 0.85 * L, (nd, 2))
    w = np.exp(-((hots[:, None] - dc[None]) ** 2).sum(-1) / (2 * sig ** 2)).sum(1) + 0.02
    sel = rng.choice(len(hots), K, replace=False, p=w / w.sum())
    side = rng.integers(4); t = rng.uniform(0.1, 0.9) * L
    depot = np.array([[t, 0.0], [L, t], [t, L], [0.0, t]][side])
    dest = rng.uniform(0.1 * L, 0.9 * L, 2)
    return dict(city=city.name, k=k, cells=sel, depot=depot, dest=dest)


def servable(city, m, allow_ris=True):
    for h in m['cells']:
        T = city.H[h]
        a = T['area'] if allow_ris else T['area'][:1]
        if a.max() == 0:
            return False
    return True


def evaluate(city, m, stops, order_xy=None):
    K = len(m['cells'])
    seen = np.zeros(K, int); tau = txb = dlv = eps = 0.0; ris_t = 0.0; ok = True
    pts = [m['depot']]
    for xy, cl in stops:
        pts.append(np.asarray(xy, float))
        for (i, o, p) in cl:
            T = city.H[m['cells'][i]]
            if not T['feas'][o, p]:
                ok = False
            seen[i] += 1
            tau += T['tau'][o, p]; txb += T['txb'][o, p]; dlv += T['dlv'][o, p]; eps += T['eps'][o, p]
            if o > 0:
                ris_t += T['tau'][o, p]
    pts.append(m['depot'])
    P = np.array(pts)
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    n_stops = len({tuple(np.round(xy, 3)) for xy, cl in stops if cl})
    E_fly = E_F * L; E_hov = P_H * tau; E_rx = P_RX * tau; E_bit = E_DEC * txb + E_BH * dlv
    E = E_fly + eps
    T_mis = L / V_MR + tau
    served_once = bool((seen == 1).all())
    return dict(E_kJ=E / 1e3, E_fly_kJ=E_fly / 1e3, E_hov_kJ=E_hov / 1e3, E_rx_kJ=E_rx / 1e3, E_bit_J=E_bit,
                L_km=L / 1e3, stops=n_stops, tau_s=tau, T_s=T_mis, E_gue_J=P_T * tau,
                R_ser_Mbps=dlv / max(tau, 1e-9) / 1e6, R_mis_Mbps=dlv / T_mis / 1e6, deliv_frac=dlv / (txb + 1e-9),
                E_ris_J=N_EL * P_E * ris_t, prr_ok=ok, served_once=served_once,
                battery_ok=bool(E + E_VERT <= E_BAT - E_RES), deadline_ok=bool(T_mis <= T_MAX),
                battery_slack_kJ=(E_BAT - E_RES - E - E_VERT) / 1e3, deadline_slack_s=T_MAX - T_mis)
