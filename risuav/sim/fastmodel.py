import torch
from torch.utils.checkpoint import checkpoint
import model as M

_CKPT = {'on': True}


def disable_point_checkpoint():
    _CKPT['on'] = False


def enable_point_checkpoint():
    _CKPT['on'] = True


class FastProposed(M.Proposed):
    def _point_logits(self, zp, pf, pm):
        if _CKPT['on'] and self.training and torch.is_grad_enabled():
            return checkpoint(lambda a, b, c: self.pt_ptr(a, self.pt_emb(b), c), zp, pf, pm, use_reentrant=False)
        return self.pt_ptr(zp, self.pt_emb(pf), pm)
