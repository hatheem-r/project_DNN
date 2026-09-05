"""
Loss functions for the class-imbalance component (Piece 3).

THE PROBLEM
-----------
Measured in Step 1: only 4.14% of training tokens are offensive. A model that
answers "not offensive" to every token is ~96% accurate and useless. Plain
cross-entropy is dominated by the huge easy negative class, so the gradient
signal from the rare positives gets drowned out.

FOUR OPTIONS, COMPARED
----------------------
  cross_entropy  the control. What the baseline uses.
  weighted       multiply the offensive class's loss by a constant. Simple,
                 often competitive with anything fancier.
  focal          down-weight examples the model already gets right, so training
                 concentrates on the hard ones. (Lin et al., ICCV 2017)
  dice           optimise an F1-like overlap directly rather than per-token
                 likelihood. Built for imbalanced NLP. (Li et al., ACL 2020)

A WARNING SPECIFIC TO OUR SITUATION
-----------------------------------
After Piece 1 the model sits at precision 0.74 / recall 0.68 - close to
balanced. These losses all push toward recall. There is less headroom than
there was at Phase 1's 0.75/0.50, and pushing too hard trades precision for
recall at a NET LOSS in F1. Always read both columns, never F1 alone.

NOTE ON THE CRF
---------------
The CRF computes its own sequence likelihood and does not accept per-class
weights. Step 7b showed the CRF is now redundant (0.7064 without vs 0.7043
with, inside noise), so the loss comparison runs WITHOUT the CRF. That is a
documented, evidence-backed choice, not a convenience.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -100


def _flatten(logits: torch.Tensor, labels: torch.Tensor):
    """(B,T,C) and (B,T) -> (N,C) and (N,), padding removed."""
    C = logits.size(-1)
    logits = logits.reshape(-1, C)
    labels = labels.reshape(-1)
    keep = labels != IGNORE
    return logits[keep], labels[keep]


class TokenLoss(nn.Module):
    """One interface over the four options so the training loop never branches.

    kind:    cross_entropy | weighted | focal | dice
    weight:  class weights, used by 'weighted'
    gamma:   focusing strength for 'focal'. 0 = plain CE, 2 = the paper default
    alpha:   positive-class weight for 'focal'
    smooth:  smoothing constant for 'dice'
    """

    def __init__(self, kind: str = "cross_entropy",
                 weight: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, alpha: float = 0.75, smooth: float = 1.0):
        super().__init__()
        if kind not in ("cross_entropy", "weighted", "focal", "dice"):
            raise ValueError(f"unknown loss {kind!r}")
        self.kind = kind
        self.gamma, self.alpha, self.smooth = gamma, alpha, smooth
        self.register_buffer("weight", weight if weight is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        lg, lb = _flatten(logits, labels)
        if lg.numel() == 0:
            return logits.sum() * 0.0

        if self.kind == "cross_entropy":
            return F.cross_entropy(lg, lb)

        if self.kind == "weighted":
            w = self.weight if self.weight.numel() else None
            return F.cross_entropy(lg, lb, weight=w)

        if self.kind == "focal":
            # p_t is the probability the model gave to the CORRECT class.
            # (1 - p_t)^gamma shrinks the loss for examples it already gets
            # right, so the gradient concentrates on hard ones.
            logp = F.log_softmax(lg, dim=-1)
            logp_t = logp.gather(1, lb.unsqueeze(1)).squeeze(1)
            p_t = logp_t.exp()
            a = torch.where(lb == 1,
                            torch.full_like(p_t, self.alpha),
                            torch.full_like(p_t, 1.0 - self.alpha))
            return -(a * (1.0 - p_t).pow(self.gamma) * logp_t).mean()

        # dice: 1 - (2*overlap + s) / (total + s), on the positive class.
        # Optimises an F1-shaped objective directly instead of likelihood.
        p = F.softmax(lg, dim=-1)[:, 1]
        t = (lb == 1).float()
        inter = (p * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


def inverse_frequency_weights(df, device=None) -> torch.Tensor:
    """Class weights from TRAIN only. Never computed on validation or test."""
    pos = int(df["n_offensive_tokens"].sum())
    tot = int(df["n_tokens"].sum())
    neg = tot - pos
    w = torch.tensor([tot / (2.0 * neg), tot / (2.0 * pos)], dtype=torch.float)
    return w.to(device) if device is not None else w
