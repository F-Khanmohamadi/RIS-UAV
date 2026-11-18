"""Revised environment: exact city tables + multi-RIS coupling (revised System Model). Mission generator, energy and
time model are those of the original preliminary environment (env.py), unchanged:
    E_UAV = e_f L + sum_i eps_i(q_i, r),  eps_i = (P_h + P_rx) tau_i + e_dec D~_i + e_bh Dbar_i,
    T = L / V_mr + tau_tot,  E_GUE = P_t tau_tot,  R_ser = Dbar_tot / tau_tot,  R_mis = Dbar_tot / T.
Constant take-off/landing energy E_vert is excluded from E_UAV (plan-independent) but included in the battery check.
Ground-vehicle travel energy of ground-mobile RISs is outside the UAV objective (deployment before departure)."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env as E0
from env import make_mission, servable, seed_of, V_MR, E_F, P_H, P_T, P_RX, E_DEC, E_BH, P_E, N_EL, E_VERT, E_BAT, E_RES, T_MAX, OMAX, L_CITY
from physics import REF
from coupling import Coupler
import build_exact as BE


class City:
    """Same interface as env.City (fields used by batch/plan code), on the exact tables, plus the coupling engine."""

    def __init__(self, path):
        C = pickle.load(open(path, 'rb'))
        self.name = os.path.basename(path)[:-4]   # (Windows compatibility: was path.split('/'))
        self.mode = C['P']['ris_mode']; self.hots = C['hots']; self.sites = C['sites']
        self.H = []
        for T in C['tabs']:
            U = T['U'][:, :2].astype(np.float64)
            feas = T['feas'].astype(bool) & (np.hypot(*(U - T['centre']).T) <= REF['R_hov'])[None]
            tau = T['tau'].astype(np.float64); txb = T['txb'].astype(np.float64); dlv = T['deliv'].astype(np.float64)
            eps = (P_H + P_RX) * tau + E_DEC * txb + E_BH * dlv
            cands = T['cands'].reshape(-1, 6) if len(T['cands']) else np.zeros((0, 6))
            self.H.append(dict(U=U, gidx=T['gidx'], tau=tau, txb=txb, dlv=dlv, eps=eps, feas=feas, nopt=tau.shape[0],
                               cands=cands, g=np.asarray(T['centre'], float), G=int(T['G']), area=feas.sum(1)))
        self.mode_flag = 1.0 if self.mode == 'facade' else 0.0
        self.cp = Coupler(C)

    # ---------------------------------------------------------------- coupled service of one cell
    def deployed(self, m, assign):
        """physical panel ids deployed by a plan; assign: iterable of (mission cell index, option)."""
        return frozenset(int(self.cp.pid[m['cells'][i]][o]) for i, o in assign if o > 0)

    def svc(self, h, o, dep):
        """coupled service table of cell h (hotspot index) with option o given the deployed panel set."""
        t = self.cp.table(h, o, dep)
        inr = self.H[h]['feas'][o] | ~self.H[h]['feas'][o]        # range mask applied below
        feas = t['feas'] & (np.hypot(*(self.H[h]['U'] - self.H[h]['g']).T) <= REF['R_hov'])
        eps = np.where(feas, (P_H + P_RX) * t['tau'] + E_DEC * t['txb'] + E_BH * t['dlv'], np.inf)
        return dict(tau=t['tau'], txb=t['txb'], dlv=t['dlv'], eps=eps, feas=feas)

    def svc_point(self, h, o, dep, q):
        """OPTIMISED (audit): (feasible, eps) of cell h, option o at lattice point q; identical to svc(h, o, dep)[.][q]."""
        t = self.cp.point(h, o, dep, q)
        feas = t['feas'] and bool(np.hypot(*(self.H[h]['U'][q] - self.H[h]['g'])) <= REF['R_hov'])
        eps = (P_H + P_RX) * t['tau'] + E_DEC * t['txb'] + E_BH * t['dlv'] if feas else np.inf
        return feas, eps


def plan_assign(stops):
    return [(i, o) for xy, cl in stops for (i, o, p) in cl]


def evaluate(city, m, stops):
    """stops: list of (xy, [(cell, option, point), ...]) in visiting order (destination with empty list).
    All service quantities are taken from the coupled tables under the plan's full deployment."""
    K = len(m['cells']); dep = city.deployed(m, plan_assign(stops))
    seen = np.zeros(K, int); tau = txb = dlv = 0.0; ris_t = 0.0; ok = True; pts = [m['depot']]
    for xy, cl in stops:
        pts.append(np.asarray(xy, float))
        for (i, o, p) in cl:
            s = city.svc(m['cells'][i], o, dep)
            if not s['feas'][p]:
                ok = False
            seen[i] += 1
            if np.isfinite(s['tau'][p]):
                tau += s['tau'][p]; txb += s['txb'][p]; dlv += s['dlv'][p]
                if o > 0:
                    ris_t += s['tau'][p]
    pts.append(m['depot'])
    Pm = np.array(pts); L = float(np.linalg.norm(np.diff(Pm, axis=0), axis=1).sum())
    n_stops = len({tuple(np.round(xy, 3)) for xy, cl in stops if cl})
    E_fly = E_F * L; E_hov = P_H * tau; E_rx = P_RX * tau; E_bit = E_DEC * txb + E_BH * dlv
    E = E_fly + E_hov + E_rx + E_bit; T_mis = L / V_MR + tau
    served_once = bool((seen == 1).all())
    return dict(E_kJ=E / 1e3, E_fly_kJ=E_fly / 1e3, E_hov_kJ=E_hov / 1e3, E_rx_kJ=E_rx / 1e3, E_bit_J=E_bit,
                E_comm_kJ=(E_rx + E_bit) / 1e3, L_km=L / 1e3, stops=n_stops, tau_s=tau, T_s=T_mis, E_gue_J=P_T * tau,
                R_ser_Mbps=dlv / max(tau, 1e-9) / 1e6, R_mis_Mbps=dlv / T_mis / 1e6, deliv_bits=dlv, txb_bits=txb,
                deliv_frac=dlv / (txb + 1e-9), E_ris_J=N_EL * P_E * ris_t, prr_ok=ok, served_once=served_once,
                battery_ok=bool(E + E_VERT <= E_BAT - E_RES), deadline_ok=bool(T_mis <= T_MAX),
                battery_slack_kJ=(E_BAT - E_RES - E - E_VERT) / 1e3, deadline_slack_s=T_MAX - T_mis,
                n_panels=len(dep), n_assist=sum(1 for i, o in plan_assign(stops) if o > 0))
