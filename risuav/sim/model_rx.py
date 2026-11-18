"""NEW (audit implementation, 17 Nov 2026). Policy used for the new runs.

= model_mr.CoupledProposed (original, coupled masks) with one documented change:
  the cell feature follows the manuscript, Eq. (cell_feature): f_i = [g_i, K_i, sum_k D_k] (the original code used
  [g_i, sum_k D_k] only). Normalisation: g / L_city, K_i / 30, sum D_k / 1.2e9 bit (as in batch.py).
  The destination node is [q_D, 0, 0].
Everything else (encoder, context, three pointer heads, clipping 10, coupled masks) is unchanged."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from model_mr import CoupledProposed
from batch import make_batch
from env import L_CITY

K_SCALE = 30.0; D_SCALE = 1.2e9


class RxProposed(CoupledProposed):
    def __init__(self):
        super().__init__()
        self.enc = M.Encoder(4)


def make_batch_rx(missions, D_bits=40e6):
    """batch.make_batch + manuscript cell feature [g_x, g_y, K_i, sum D_k] (normalised)."""
    bt = make_batch(None, missions)
    nf = bt['nodef']; B, N1, _ = nf.shape; Kp = N1 - 1
    out = torch.zeros(B, N1, 4)
    out[:, :, :2] = nf[:, :, :2]
    for b, (a, m) in enumerate(missions):
        for i, h in enumerate(m['cells']):
            G = a.city.H[h]['G']
            out[b, i, 2] = G / K_SCALE
            out[b, i, 3] = G * D_bits / D_SCALE
    bt['nodef'] = out
    return bt
