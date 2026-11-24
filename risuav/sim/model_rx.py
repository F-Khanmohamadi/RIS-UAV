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
