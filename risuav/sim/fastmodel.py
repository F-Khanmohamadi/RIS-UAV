"""RECONSTRUCTED (17 Nov 2026) -- not the original file.

The original `prelim500/fastmodel.py` is lost (see MISSING_FILES.md). `model_mr.py` imports from it only
`FastProposed` and `disable_point_checkpoint`, and uses two members of `FastProposed`: everything inherited from
`model.Proposed` and the method `_point_logits(zp, pf, pm)`. The docstring of `model_mr.py` states that the fast point
head has "identical maths" to `model.Proposed`. This reconstruction therefore computes exactly the point head of
`model.Proposed.forward` (L125-127): logits = pt_ptr(zp, pt_emb(pf), pm), optionally with activation checkpointing
(memory saving only; the point head has no BatchNorm, dropout or randomness, so values and gradients are identical).
State dicts are interchangeable with `model.Proposed` (no new parameters)."""
import torch
from torch.utils.checkpoint import checkpoint
import model as M

_CKPT = {'on': True}


def disable_point_checkpoint():
    """Turn activation checkpointing of the point head off (as done by model_mr at import)."""
    _CKPT['on'] = False


def enable_point_checkpoint():
    _CKPT['on'] = True


class FastProposed(M.Proposed):
    def _point_logits(self, zp, pf, pm):
        if _CKPT['on'] and self.training and torch.is_grad_enabled():
            return checkpoint(lambda a, b, c: self.pt_ptr(a, self.pt_emb(b), c), zp, pf, pm, use_reentrant=False)
        return self.pt_ptr(zp, self.pt_emb(pf), pm)
