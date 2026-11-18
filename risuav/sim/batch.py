"""Padded per-city arrays and mission batches for the policies."""
import numpy as np, torch
from env import City, OMAX, L_CITY, D_G
from physics import REF

DMAX = REF['R_hov']          # service range d_max (m)
EPS_SCALE = 1e3              # J, normalisation of service energy features


class CityArrays:
    def __init__(self, city: City):
        self.city = city
        nh = len(city.H); Lmax = max(len(T['U']) for T in city.H); self.Lmax = Lmax
        self.rel = np.zeros((nh, Lmax, 2), np.float32)          # (q - g_i) / d_max
        self.xy = np.zeros((nh, Lmax, 2), np.float32)           # absolute point coordinates (m)
        self.pvalid = np.zeros((nh, Lmax), bool)
        self.eps = np.zeros((nh, OMAX, Lmax), np.float32)       # eps_i(q, r) (J)
        self.feas = np.zeros((nh, OMAX, Lmax), bool)            # q in N_i(r)
        self.ofeat = np.zeros((nh, OMAX, 7), np.float32)        # site features f_{i,r}
        self.omask = np.zeros((nh, OMAX), bool)                 # option exists and N_i(r) non-empty
        self.area = np.zeros((nh, OMAX), np.int32)
        self.nq = np.zeros(nh, np.int32)
        for h, T in enumerate(city.H):
            n = len(T['U']); no = T['nopt']; self.nq[h] = n
            self.rel[h, :n] = (T['U'] - T['g']) / DMAX; self.xy[h, :n] = T['U']; self.pvalid[h, :n] = True
            self.eps[h, :no, :n] = T['eps']; self.feas[h, :no, :n] = T['feas']
            self.area[h, :no] = T['area']; self.omask[h, :no] = T['area'] > 0
            for o in range(1, no):
                c = T['cands'][o - 1]
                self.ofeat[h, o, :3] = (c[:3] - np.r_[T['g'], 0.0]) / 100.0
                self.ofeat[h, o, 3:6] = c[3:6]
                self.ofeat[h, o, 6] = city.mode_flag
        self.cellf = np.zeros((nh, 3), np.float32)
        for h, T in enumerate(city.H):
            self.cellf[h] = [T['g'][0] / L_CITY, T['g'][1] / L_CITY, T['G'] * D_G / 1.2e9]


def make_batch(arrs, missions, Kpad=None):
    """missions: list of (CityArrays, mission dict). Node order: cells 0..K-1, destination at index Kpad."""
    B = len(missions); K = [len(m['cells']) for _, m in missions]; Kp = Kpad or max(K)
    Lmax = max(a.Lmax for a, _ in missions)
    nodef = np.zeros((B, Kp + 1, 3), np.float32); nmask = np.zeros((B, Kp + 1), bool)   # True = selectable
    ofeat = np.zeros((B, Kp, OMAX, 7), np.float32); omask = np.zeros((B, Kp, OMAX), bool)
    rel = np.zeros((B, Kp, Lmax, 2), np.float32); xy = np.zeros((B, Kp, Lmax, 2), np.float32)
    eps = np.zeros((B, Kp, OMAX, Lmax), np.float32); feas = np.zeros((B, Kp, OMAX, Lmax), bool)
    gxy = np.zeros((B, Kp, 2), np.float32); area = np.zeros((B, Kp, OMAX), np.float32)
    nq = np.zeros((B, Kp), np.int32)
    dep = np.zeros((B, 2), np.float32); dst = np.zeros((B, 2), np.float32)
    for b, (a, m) in enumerate(missions):
        hs = m['cells']; k = len(hs); L = a.Lmax
        nodef[b, :k] = a.cellf[hs]; nmask[b, :k] = True
        nodef[b, Kp] = [m['dest'][0] / L_CITY, m['dest'][1] / L_CITY, 0.0]; nmask[b, Kp] = True
        ofeat[b, :k] = a.ofeat[hs]; omask[b, :k] = a.omask[hs]
        rel[b, :k, :L] = a.rel[hs]; xy[b, :k, :L] = a.xy[hs]
        eps[b, :k, :, :L] = a.eps[hs]; feas[b, :k, :, :L] = a.feas[hs]
        gxy[b, :k] = [a.city.H[h]['g'] for h in hs]; area[b, :k] = a.area[hs]; nq[b, :k] = a.nq[hs]
        dep[b] = m['depot']; dst[b] = m['dest']
    t = lambda x: torch.from_numpy(x)
    return dict(nodef=t(nodef), nmask=t(nmask), ofeat=t(ofeat), omask=t(omask), rel=t(rel), xy=t(xy),
                eps=t(eps), feas=t(feas), gxy=t(gxy), area=t(area), dep=t(dep), dst=t(dst), K=K, Kp=Kp,
                missions=missions, nq=nq)
