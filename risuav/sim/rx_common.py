"""NEW (audit, 17 Nov 2026). Loading of city tables for the new runs, with the two sensitivity transformations that
need no table rebuild:
  D_SCALE   : backlog D_k = 40 Mbit x D_SCALE. CQI choice does not depend on D_k, so airtime, transmitted and delivered bits
              scale exactly by D_SCALE (stored tables and the coupling engine's recomputed entries).
  MAX_SITES : keep the MAX_SITES best candidate sites of every cell (candidates are stored in decreasing order of their
              mean service-time gain), i.e. a smaller placement decision space."""
import os, numpy as np
import env_mr as EM, coupling as CP

DS = float(os.environ.get('D_SCALE', '1.0')); MAXS = int(os.environ.get('MAX_SITES', '10'))
if DS != 1.0:
    CP.DG = CP.DG * DS


def load_city(path):
    city = EM.City(path)
    if DS != 1.0:
        for T in city.cp.tabs:
            for k in ('tau', 'txb', 'deliv'):
                T[k] = T[k] * DS
        for H in city.H:
            H['tau'] = H['tau'] * DS; H['txb'] = H['txb'] * DS; H['dlv'] = H['dlv'] * DS
            H['eps'] = (EM.P_H + EM.P_RX) * H['tau'] + EM.E_DEC * H['txb'] + EM.E_BH * H['dlv']
    if MAXS < 10:
        for H in city.H:
            n = min(H['nopt'], MAXS + 1)
            for k in ('tau', 'txb', 'dlv', 'eps', 'feas', 'area'):
                H[k] = H[k][:n]
            H['cands'] = H['cands'][:n - 1]; H['nopt'] = n
    if os.environ.get('LEAN') == '1':                              # training on serving-only tables: drop unused arrays
        for H in city.H:
            for k in ('tau', 'txb', 'dlv'):
                H.pop(k, None)
        for T in city.cp.tabs:
            for k in ('cov', 'margin', 'S', 'S2', 'los_d', 'los_ru'):
                T.pop(k, None)
    return city
