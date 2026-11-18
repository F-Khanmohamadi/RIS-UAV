"""Proposed policy with coupled multi-RIS masks (revised experiment). Architecture, parameters and decoding order are
those of the preliminary policy (model.Proposed, fast point head of prelim500/fastmodel.py, identical maths).
Change: at every decoding step the hovering-point mask N_i(r) and the service-energy feature eps_i(q, r) of the chosen
(cell, RIS option) are taken from the coupled service table given the panels deployed so far (all previously decoded
cells' panels + the serving panel just chosen; for fixed placements: the complete fixed deployment). Panels of cells
decoded later are not yet known; the decoded plan is therefore checked and, if needed, repaired under the full
deployment (plan_mr.repair). If the coupled mask of the chosen option is empty, the uncoupled mask is used and the cell
is repaired afterwards (counted)."""
import sys, os, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from fastmodel import FastProposed, disable_point_checkpoint
from batch import EPS_SCALE

disable_point_checkpoint()


class CoupledProposed(FastProposed):
    couple = True

    def _coupled_rows(self, bt, cell, opt, is_dest, active, deployed, pm, pe):
        Lmax = pm.shape[1]
        for b in range(pm.shape[0]):
            if not bool(active[b]) or bool(is_dest[b]):
                continue
            a, m = bt['missions'][b]; city = a.city
            h = m['cells'][int(cell[b])]; o = int(opt[b])
            dep = set(deployed[b])
            if o > 0:
                dep.add(int(city.cp.pid[h][o]))
            sv = city.svc(h, o, frozenset(dep))
            n = len(sv['feas'])
            f = torch.from_numpy(sv['feas'])
            if not bool(f.any()):
                self.n_empty += 1; continue
            pm[b, :n] = f; pm[b, n:] = False
            pe[b, :n] = torch.from_numpy(sv['eps'].astype(np.float32))
        return pm, pe

    def forward(self, bt, mode='sample', force_opt=None):
        nodef, nm0 = bt['nodef'], bt['nmask']; B, N1, _ = nodef.shape; Kp = N1 - 1; ar = torch.arange(B)
        x, xb = self.enc(nodef, nm0 | (torch.arange(N1) == Kp)[None])
        semb = self.site_emb(bt['ofeat'])
        semb = torch.cat([self.norisk.expand(B, Kp, 1, M.D), semb[:, :, 1:]], 2)
        avail = nm0.clone(); first = torch.zeros(B, dtype=torch.long); last = first.clone()
        qf = torch.zeros(B, 2); ql = bt['dep'] / 3000.0; step0 = torch.ones(B, dtype=torch.bool)
        lp = torch.zeros(B); ent = torch.zeros(B); seq = []
        self.n_empty = 0
        deployed = [set() for _ in range(B)]
        if force_opt is not None:                                     # fixed placement: whole deployment known
            for b, (a, m) in enumerate(bt['missions']):
                for i, h in enumerate(m['cells']):
                    o = int(force_opt[b, i])
                    if o > 0:
                        deployed[b].add(int(a.city.cp.pid[h][o]))
        for step in range(N1):
            active = avail.any(1)
            if not active.any():
                break
            z = self.context(x, xb, first, last, qf, ql, step0, avail)
            lc = self.cell_ptr(z, x, avail)
            node = self._pick(lc, mode)
            is_dest = node == Kp; cell = node.clamp(max=Kp - 1)
            lpc = F.log_softmax(lc, -1)
            step_lp = lpc[ar, node]; step_ent = -(lpc.exp() * lpc).sum(-1)
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
            pm = bt['feas'][ar, cell, opt].clone(); pe = bt['eps'][ar, cell, opt].clone()
            if self.couple:
                pm, pe = self._coupled_rows(bt, cell, opt, is_dest, active, deployed, pm, pe)
            pm = pm | is_dest[:, None]
            pf = torch.cat([bt['rel'][ar, cell], (pe / EPS_SCALE).unsqueeze(-1)], -1)
            zp = torch.cat([z, x[ar, cell], semb[ar, cell, opt]], -1)
            lq = self._point_logits(zp, pf, pm)
            pt = self._pick(lq, mode)
            lpq = F.log_softmax(lq, -1)
            step_lp = step_lp + torch.where(is_dest, 0.0, lpq[ar, pt])
            step_ent = step_ent + torch.where(is_dest, 0.0, -(lpq.exp() * lpq).sum(-1))
            lp = lp + torch.where(active, step_lp, 0.0); ent = ent + torch.where(active, step_ent, 0.0)
            pxy = torch.where(is_dest[:, None], bt['dst'], bt['xy'][ar, cell, pt])
            seq.append(torch.stack([node, opt, pt], 1) * active[:, None] - (~active)[:, None].long())
            if force_opt is None:
                for b in range(B):
                    if bool(active[b]) and not bool(is_dest[b]) and int(opt[b]) > 0:
                        a, m = bt['missions'][b]; deployed[b].add(int(a.city.cp.pid[m['cells'][int(cell[b])]][int(opt[b])]))
            avail = avail.clone(); avail[ar[active], node[active]] = False
            first = torch.where(step0 & active, node, first); last = torch.where(active, node, last)
            qf = torch.where((step0 & active)[:, None], pxy / 3000.0, qf); ql = torch.where(active[:, None], pxy / 3000.0, ql)
            step0 = step0 & ~active
        return torch.stack(seq, 1), lp, ent
