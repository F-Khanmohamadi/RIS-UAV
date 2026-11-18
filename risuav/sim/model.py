"""Policies.
Proposed (Sec. IV-B of the manuscript): attention encoder (d_e = 128, N_L = 3 layers, N_H = 8 heads, BN, skip),
context [x_bar, x_first, x_last, xq(first point), xq(last point)] + one MHA glimpse, three pointer heads
(cell -> RIS site -> hovering point over the lattice points of Q_i, masked to N_i(r)), clipping 10.
ECTD (Iranmanesh et al., TVT 2026, Sec. IV-B): same encoder on 3D cluster centres, classification pointer over
clusters + destination with the same context, regression head [3 d_h context, FF(h_i)] -> MLP 64-32-3 giving the
relative deviation of the contact location. ADAPTATIONS (documented): the deviation is the mean of a Gaussian with
a learned std so that log p of Eq. (19)/(23) is defined; the z component is unused (fixed altitude); the sampled
location is snapped to the nearest feasible lattice point of N_i(0) (no RIS in ECTD).
"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from env import OMAX
from batch import DMAX, EPS_SCALE

D = 128; NH = 8; NL = 3; CLIP = 10.0


class EncLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(D, NH, batch_first=True)
        self.bn1 = nn.BatchNorm1d(D); self.ff = nn.Sequential(nn.Linear(D, 512), nn.ReLU(), nn.Linear(512, D))
        self.bn2 = nn.BatchNorm1d(D)

    def bn(self, bn, x):
        return bn(x.reshape(-1, D)).reshape(x.shape)

    def forward(self, x, kpm):
        h = self.mha(x, x, x, key_padding_mask=kpm, need_weights=False)[0]
        x = self.bn(self.bn1, x + h)
        return self.bn(self.bn2, x + self.ff(x))


class Encoder(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.inp = nn.Linear(fin, D); self.layers = nn.ModuleList([EncLayer() for _ in range(NL)])

    def forward(self, f, valid):
        x = self.inp(f); kpm = ~valid
        for l in self.layers:
            x = l(x, kpm)
        xb = (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        return x, xb


def ff(i, o=D):
    return nn.Sequential(nn.Linear(i, D), nn.ReLU(), nn.Linear(D, o))


class Pointer(nn.Module):
    def __init__(self, qin):
        super().__init__()
        self.q = nn.Linear(qin, D, bias=False); self.k = nn.Linear(D, D, bias=False)

    def forward(self, z, emb, mask):
        lg = CLIP * torch.tanh((self.k(emb) @ self.q(z).unsqueeze(-1)).squeeze(-1) / math.sqrt(D))
        return lg.masked_fill(~mask, -1e9)


class Base(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.enc = Encoder(fin)
        self.locemb = ff(2)
        self.ph = nn.Parameter(torch.randn(4, D) * 0.1)          # learned placeholders (tau = 1)
        self.ctx = nn.Linear(5 * D, D)
        self.glimpse = nn.MultiheadAttention(D, NH, batch_first=True)
        self.cell_ptr = Pointer(D)

    def context(self, x, xb, first, last, qf, ql, step0, nmask):
        B = x.shape[0]; ar = torch.arange(B)
        xf = torch.where(step0[:, None], self.ph[0].expand(B, D), x[ar, first])
        xl = torch.where(step0[:, None], self.ph[1].expand(B, D), x[ar, last])
        lf = torch.where(step0[:, None], self.ph[2].expand(B, D), self.locemb(qf))
        ll = torch.where(step0[:, None], self.ph[3].expand(B, D), self.locemb(ql))
        c = self.ctx(torch.cat([xb, xf, xl, lf, ll], -1)).unsqueeze(1)
        kpm = ~nmask
        allm = kpm.all(1)
        kpm = kpm.clone(); kpm[allm] = False
        return self.glimpse(c, x, x, key_padding_mask=kpm, need_weights=False)[0].squeeze(1)


class Proposed(Base):
    def __init__(self):
        super().__init__(3)
        self.site_emb = ff(7); self.norisk = nn.Parameter(torch.randn(D) * 0.1)
        self.pt_emb = ff(3)
        self.site_ptr = Pointer(2 * D); self.pt_ptr = Pointer(3 * D)

    def forward(self, bt, mode='sample', force_opt=None):
        """force_opt: None (joint) or LongTensor (B, Kp) of fixed options (baselines with fixed placement)."""
        nodef, nm0 = bt['nodef'], bt['nmask']; B, N1, _ = nodef.shape; Kp = N1 - 1; ar = torch.arange(B)
        x, xb = self.enc(nodef, nm0 | (torch.arange(N1) == Kp)[None])
        semb = self.site_emb(bt['ofeat'])                                   # (B,Kp,O,D)
        semb = torch.cat([self.norisk.expand(B, Kp, 1, D), semb[:, :, 1:]], 2)
        avail = nm0.clone(); first = torch.zeros(B, dtype=torch.long); last = first.clone()
        qf = torch.zeros(B, 2); ql = bt['dep'] / 3000.0; step0 = torch.ones(B, dtype=torch.bool)
        lp = torch.zeros(B); ent = torch.zeros(B)
        seq = []
        for m in range(N1):
            active = avail.any(1)
            if not active.any():
                break
            z = self.context(x, xb, first, last, qf, ql, step0, avail)
            lc = self.cell_ptr(z, x, avail)
            node = self._pick(lc, mode)
            is_dest = node == Kp; cell = node.clamp(max=Kp - 1)
            lpc = F.log_softmax(lc, -1)
            step_lp = lpc[ar, node]; step_ent = -(lpc.exp() * lpc).sum(-1)
            # site head
            om = bt['omask'][ar, cell]
            if force_opt is None:
                zs = torch.cat([z, x[ar, cell]], -1)
                ls = self.site_ptr(zs, semb[ar, cell], om | is_dest[:, None])
                opt = self._pick(ls, mode)
                lps = F.log_softmax(ls, -1)
                step_lp = step_lp + torch.where(is_dest, 0.0, lps[ar, opt])
                step_ent = step_ent + torch.where(is_dest, 0.0, -(lps.exp() * lps).sum(-1))
            else:
                opt = force_opt[ar, cell]
            # point head over the lattice points of Q_i, masked to N_i(r)
            pm = bt['feas'][ar, cell, opt]                                   # (B,L)
            pm = pm | is_dest[:, None]
            pf = torch.cat([bt['rel'][ar, cell], (bt['eps'][ar, cell, opt] / EPS_SCALE).unsqueeze(-1)], -1)
            zp = torch.cat([z, x[ar, cell], semb[ar, cell, opt]], -1)
            lq = self.pt_ptr(zp, self.pt_emb(pf), pm)
            pt = self._pick(lq, mode)
            lpq = F.log_softmax(lq, -1)
            step_lp = step_lp + torch.where(is_dest, 0.0, lpq[ar, pt])
            step_ent = step_ent + torch.where(is_dest, 0.0, -(lpq.exp() * lpq).sum(-1))
            lp = lp + torch.where(active, step_lp, 0.0); ent = ent + torch.where(active, step_ent, 0.0)
            pxy = torch.where(is_dest[:, None], bt['dst'], bt['xy'][ar, cell, pt])
            seq.append(torch.stack([node, opt, pt], 1) * active[:, None] - (~active)[:, None].long())
            avail = avail.clone(); avail[ar[active], node[active]] = False
            first = torch.where(step0 & active, node, first); last = torch.where(active, node, last)
            qf = torch.where((step0 & active)[:, None], pxy / 3000.0, qf); ql = torch.where(active[:, None], pxy / 3000.0, ql)
            step0 = step0 & ~active
        return torch.stack(seq, 1), lp, ent

    @staticmethod
    def _pick(lg, mode):
        if mode == 'greedy':
            return lg.argmax(-1)
        return torch.distributions.Categorical(logits=lg).sample()


class ECTD(Base):
    """ECTD policy (no RIS). Node features: 3D centre (x, y, z = 0) normalised."""

    def __init__(self):
        super().__init__(3)
        self.cl_emb = ff(3)                                    # ECTD location embedding g_i of the 3D CL
        self.locemb = None
        self.rc1 = nn.Linear(2 * D, D); self.rc2 = nn.Linear(2 * D, D)
        self.hff = ff(D)
        self.mlp = nn.Sequential(nn.Linear(4 * D, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
        self.logstd = nn.Parameter(torch.full((2,), math.log(0.25)))   # std in units of d_max (~110 m)

    def context(self, x, xb, first, last, gf, gl, step0, nmask):
        B = x.shape[0]; ar = torch.arange(B)
        xf = torch.where(step0[:, None], self.ph[0].expand(B, D), x[ar, first])
        xl = torch.where(step0[:, None], self.ph[1].expand(B, D), x[ar, last])
        lf = torch.where(step0[:, None], self.ph[2].expand(B, D), gf)
        ll = torch.where(step0[:, None], self.ph[3].expand(B, D), gl)
        c = self.ctx(torch.cat([xb, xf, xl, lf, ll], -1)).unsqueeze(1)
        kpm = ~nmask; allm = kpm.all(1); kpm = kpm.clone(); kpm[allm] = False
        return self.glimpse(c, x, x, key_padding_mask=kpm, need_weights=False)[0].squeeze(1)

    def forward(self, bt, mode='sample'):
        nodef, nm0 = bt['nodef'], bt['nmask']; B, N1, _ = nodef.shape; Kp = N1 - 1; ar = torch.arange(B)
        f3 = nodef.clone(); f3[..., 2] = 0.0                       # ECTD input: 3D coordinate of the centre
        x, xb = self.enc(f3, nm0 | (torch.arange(N1) == Kp)[None])
        avail = nm0.clone(); first = torch.zeros(B, dtype=torch.long); last = first.clone()
        gf = torch.zeros(B, D); gl = torch.zeros(B, D); step0 = torch.ones(B, dtype=torch.bool)
        lp = torch.zeros(B); ent = torch.zeros(B); seq = []
        std = self.logstd.exp()
        for m in range(N1):
            active = avail.any(1)
            if not active.any():
                break
            z = self.context(x, xb, first, last, gf, gl, step0, avail)
            lc = self.cell_ptr(z, x, avail)
            node = Proposed._pick(lc, mode); is_dest = node == Kp; cell = node.clamp(max=Kp - 1)
            lpc = F.log_softmax(lc, -1); step_lp = lpc[ar, node]; step_ent = -(lpc.exp() * lpc).sum(-1)
            # regression context RT (3 d_h) + FF(h_i) -> MLP 64-32-3 = relative deviation
            rt = torch.cat([xb, self.rc1(torch.cat([x[ar, first], x[ar, last]], -1)), self.rc2(torch.cat([gf, gl], -1))], -1)
            mu = torch.tanh(self.mlp(torch.cat([rt, self.hff(x[ar, cell])], -1)))[:, :2]
            dev = mu if mode == 'greedy' else mu + std * torch.randn_like(mu)
            dist = torch.distributions.Normal(mu, std)
            step_lp = step_lp + torch.where(is_dest, 0.0, dist.log_prob(dev.detach()).sum(-1))
            # snap to the nearest feasible lattice point of N_i(0)  (adaptation: discrete lattice, no RIS)
            tgt = bt['gxy'][ar, cell] + dev.detach() * DMAX
            d2 = ((bt['xy'][ar, cell] - tgt[:, None]) ** 2).sum(-1)
            fm = bt['feas'][ar, cell, 0]
            d2 = d2.masked_fill(~fm, float('inf'))
            pt = d2.argmin(-1)
            lp = lp + torch.where(active, step_lp, 0.0); ent = ent + torch.where(active, step_ent, 0.0)
            pxy = torch.where(is_dest[:, None], bt['dst'], bt['xy'][ar, cell, pt])
            opt = torch.zeros(B, dtype=torch.long)
            seq.append(torch.stack([node, opt, pt], 1) * active[:, None] - (~active)[:, None].long())
            g3 = torch.cat([pxy / 3000.0, torch.full((B, 1), 0.04)], -1)       # 3D CL (z = H / 3000)
            gemb = self.cl_emb(g3)
            avail = avail.clone(); avail[ar[active], node[active]] = False
            first = torch.where(step0 & active, node, first); last = torch.where(active, node, last)
            gf = torch.where((step0 & active)[:, None], gemb, gf); gl = torch.where(active[:, None], gemb, gl)
            step0 = step0 & ~active
        return torch.stack(seq, 1), lp, ent
